"""Join multi-pixel predictions to observations and score every readout (§26).

`eval_predict.py --pixel-csv` emits predictions only: an off-centre readout is scored
against a *different* station's record, which lives in that station's own zarr, so the
join cannot happen on the GPU pass.  This does the join and the metrics.

Every station x depth is scored twice over:

    own_centre   the station predicted from its OWN tile at pixel (112, 112)
                 -- the supervised readout, what every previous evaluation reports
    off_centre   the same station predicted from a NEIGHBOUR's tile at a pixel that
                 received no supervision

Outputs to eval_output/:
    {tag}_timeseries.parquet    long: one row per (tile, station, date, depth)
    {tag}_per_readout.csv       metrics per (tile, station, depth)
    {tag}_consistency.csv       tile-to-tile spread for stations with >= 2 estimates

Usage:
    python combine_network.py --pred eval_output/predictions_network_txson.parquet
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import SM_DEPTHS, ZARR_ROOT, _load_zarr_labels, _open_zarr
from eval_metrics import metrics_from_arrays

warnings.filterwarnings("ignore")

REPO       = Path(__file__).resolve().parent
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
LEVEL1_DIR = Path("/projects/prjs1968/raw_soil_moisture")
OUT_DIR    = REPO / "eval_output"

MIN_N = 30      # minimum paired days before a metric row is emitted


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def _obs_from_zarr(station: str, category: str) -> pd.DataFrame | None:
    """Observed SM from the token zarr — the exact ground truth training used.

    Goes through dataset._load_zarr_labels because `labels/qc` is longer than
    `labels/sm` for many stations (trim_pre2016.py trimmed sm/dates but not qc)
    and that function realigns them (dataset.py:184-186).
    """
    zg = _open_zarr(ZARR_ROOT / category / station, category)
    if zg is None:
        return None
    out = _load_zarr_labels(zg)
    if out is None:
        return None
    sm, depths, times, qc = out

    rows = []
    for d, depth in enumerate(depths):
        if depth not in SM_DEPTHS:
            continue
        keep = ~np.isnan(sm[d])
        if qc is not None:
            keep &= (qc[d] == 0)          # observed only, no gap-fill (dataset.py:961-965)
        if not keep.any():
            continue
        rows.append(pd.DataFrame({"date": times[keep],
                                  "depth": depth,
                                  "obs": sm[d][keep].astype(np.float32)}))
    return pd.concat(rows, ignore_index=True) if rows else None


def _obs_from_level1(station: str) -> pd.DataFrame | None:
    """Fallback for stations that never entered the token pipeline."""
    import xarray as xr

    stem = station[5:] if station.startswith("ISMN_") else station
    hits = sorted(LEVEL1_DIR.glob(f"{stem}_*.nc"))
    if not hits:
        return None
    ds = xr.open_dataset(hits[0])
    sm = ds["soil_moisture"].values
    qc = ds["soil_moisture_qc"].values
    depths = [str(d) for d in ds["depth"].values]
    times = pd.DatetimeIndex(ds["date_time"].values)

    rows = []
    for d, depth in enumerate(depths):
        if depth not in SM_DEPTHS:
            continue
        keep = ~np.isnan(sm[d]) & (qc[d] == 0)
        if not keep.any():
            continue
        rows.append(pd.DataFrame({"date": times[keep],
                                  "depth": depth,
                                  "obs": sm[d][keep].astype(np.float32)}))
    ds.close()
    return pd.concat(rows, ignore_index=True) if rows else None


def load_observations(stations: list[str], cat_map: dict[str, str]) -> pd.DataFrame:
    frames, missing = [], []
    for s in stations:
        obs = None
        if s in cat_map:
            obs = _obs_from_zarr(s, cat_map[s])
        if obs is None:
            obs = _obs_from_level1(s)
        if obs is None:
            missing.append(s)
            continue
        obs["station"] = s
        frames.append(obs)
    if missing:
        print(f"  WARNING no observations for {len(missing)} stations: {missing[:6]}")
    if not frames:
        raise SystemExit("no observations loaded")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions parquet from --pixel-csv")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default=None, help="output stem (default: from --pred)")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.pred).stem.replace("predictions_network_", "")

    pred = pd.read_parquet(args.pred)
    pred["date"] = pd.to_datetime(pred["date"])
    print(f"predictions : {len(pred):,} rows | "
          f"{pred.groupby(['tile','station']).ngroups} readouts | "
          f"{pred.station.nunique()} stations | {pred.tile.nunique()} tiles")

    splits = pd.read_csv(SPLITS_CSV)
    splits["key"] = splits.apply(
        lambda r: (f"ISMN_{r['network']}_{r['station_name']}"
                   if r["source_network"] == "ISMN"
                   else f"{r['source_network']}_{r['station_id']}"), axis=1)
    split_map = dict(zip(splits.key, splits.split))
    cat_map = {
        r.key: ("sm_and_flux" if (r.has_soil_moisture and r.has_flux)
                else "sm_only" if r.has_soil_moisture else "flux_only")
        for r in splits.itertuples()
    }

    print("loading observations …")
    obs = load_observations(sorted(pred.station.unique()), cat_map)
    print(f"observations: {len(obs):,} rows | {obs.station.nunique()} stations | "
          f"depths {sorted(obs.depth.unique())}")

    ts = pred.merge(obs, on=["station", "date", "depth"], how="left")
    ts["station_split"] = ts.station.map(split_map).fillna("not-in-splits")
    ts["tile_split"]    = ts.tile.map(split_map).fillna("not-in-splits")
    ts["arm"] = np.where(ts.is_centre, "own_centre", "off_centre")

    matched = ts.obs.notna()
    print(f"joined      : {int(matched.sum()):,} / {len(ts):,} rows have an observation "
          f"({100*matched.mean():.1f}%)")

    ts_path = out_dir / f"{tag}_timeseries.parquet"
    ts.to_parquet(ts_path, index=False, compression="snappy")
    print(f"wrote {ts_path}  ({ts_path.stat().st_size/1e6:.1f} MB)")

    # ── metrics per (tile, station, depth) ────────────────────────────────
    rows = []
    for (tile, station, depth), g in ts[matched].groupby(
            ["tile", "station", "depth"], observed=True):
        if len(g) < args.min_n:
            continue
        m = metrics_from_arrays(g.pred.to_numpy(np.float64), g.obs.to_numpy(np.float64))
        r0 = g.iloc[0]
        rows.append({"tile": tile, "station": station, "depth": depth,
                     "arm": r0.arm, "is_centre": bool(r0.is_centre),
                     "offset_px": int(r0.offset_px),
                     "row": int(r0.row), "col": int(r0.col),
                     "tile_split": r0.tile_split, "station_split": r0.station_split,
                     "obs_sd": float(g.obs.std()), "pred_sd": float(g.pred.std()),
                     **m})
    per = pd.DataFrame(rows).sort_values(["station", "depth", "offset_px"])
    per_path = out_dir / f"{tag}_per_readout.csv"
    per.to_csv(per_path, index=False)
    print(f"wrote {per_path}  ({len(per)} rows)")

    if per.empty:
        print("\nno readout reached --min-n; nothing to summarise")
        return

    # ── headline: own-centre vs off-centre, station-equal medians ─────────
    print(f"\n{'='*72}\nStation-equal medians by arm (depth 0-10 unless noted)\n{'='*72}")
    for depth in [d for d in SM_DEPTHS if d in set(per.depth)]:
        sub = per[per.depth == depth]
        print(f"\n{depth}   (n readouts: "
              f"{dict(sub.arm.value_counts())})")
        agg = sub.groupby("arm").agg(
            n_readouts=("station", "size"), n_stations=("station", "nunique"),
            ubRMSE=("ubRMSE", "median"), RMSE=("RMSE", "median"),
            bias=("bias", lambda s: float(np.sqrt(np.mean(s ** 2)))),
            R=("R", "median"), NSE_anom=("NSE_anom", "median"))
        agg = agg.rename(columns={"bias": "RMS_bias"})
        print(agg.round(4).to_string())

        # paired: only stations that have BOTH arms
        both = sub.pivot_table(index="station", columns="arm", values="ubRMSE",
                               aggfunc="median")
        if {"own_centre", "off_centre"} <= set(both.columns):
            both = both.dropna()
            if len(both):
                d = both.off_centre - both.own_centre
                print(f"  paired on {len(both)} stations: "
                      f"median ΔubRMSE(off − own) = {d.median():+.4f}  "
                      f"({100*(d > 0).mean():.0f}% of stations worse off-centre)")

    # ── tile-to-tile consistency for stations with >= 2 estimates ─────────
    cons = (per.groupby(["station", "depth"])
               .agg(n_estimates=("tile", "nunique"),
                    ubRMSE_min=("ubRMSE", "min"), ubRMSE_max=("ubRMSE", "max"),
                    ubRMSE_spread=("ubRMSE", lambda s: float(s.max() - s.min())),
                    R_spread=("R", lambda s: float(s.max() - s.min())),
                    max_offset_px=("offset_px", "max"),
                    station_split=("station_split", "first"))
               .reset_index())
    cons = cons[cons.n_estimates >= 2].sort_values("ubRMSE_spread", ascending=False)
    cons_path = out_dir / f"{tag}_consistency.csv"
    cons.to_csv(cons_path, index=False)
    print(f"\nwrote {cons_path}  ({len(cons)} stations with ≥2 tile estimates)")
    if len(cons):
        print(f"  median ubRMSE spread across tiles: "
              f"{cons.ubRMSE_spread.median():.4f}")


if __name__ == "__main__":
    main()
