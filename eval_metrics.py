"""
Per-station and summary metrics from the prediction tables (§22.5).

Reads eval_output/predictions_{split}.parquet, writes:
    per_station_{split}.csv    -- one row per station, metrics per depth + metadata
    metrics_summary.csv        -- 3 splits x 3 depths, station-equal AND sample-pooled

CPU only, seconds to run.  Re-runnable without a GPU.

Two conventions this module exists to keep straight (§22.5):

  * NSE and R2_pearson are NEVER merged as "R^2".  A pure level offset drives
    NSE negative while leaving Pearson untouched; the gap between them is the
    result, not a nuisance.
  * Every summary metric appears twice -- `_stn` (mean across stations) and
    `_pool` (over all samples).  Mixing them silently produced one wrong
    conclusion already (runbook §21.7).

Usage:
    python eval_metrics.py                       # all splits found
    python eval_metrics.py --splits val          # §22.6 hard gate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
OUT_DIR    = Path("/gpfs/work3/0/prjs1968/soilMoisture/eval_output")
SM_DEPTHS  = ["0-10", "10-30", "30-100"]

MIN_N        = 5    # matches evaluate_splits.py:86
MIN_N_UBRMSE = 2    # anomalies need at least two points

META_COLS = [
    "station_key", "latitude", "longitude", "IGBP", "igbp_macro",
    "koppen_geiger", "kg_macro", "elevation_band", "n_years",
    "start_date", "end_date", "split",
]

# §21.8 -- val ubRMSE at epoch 16.  The §22.6 gate reproduces these.
VAL_REFERENCE = {"0-10": 0.0539, "10-30": 0.0500, "30-100": 0.0552}


def _make_key(r) -> str:
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def metrics_from_arrays(p: np.ndarray, t: np.ndarray) -> dict:
    """All per-series metrics for one station x depth (or one pooled set).

    Note `bias` == mean(p) - mean(t) is exactly the per-station level offset;
    there is no separate `offset` column because they are the same quantity.
    RMS_offset in the summary is the RMS of this column across stations.
    """
    n = int(len(p))
    out = {"n": n, "RMSE": np.nan, "ubRMSE": np.nan, "MAE": np.nan, "bias": np.nan,
           "R": np.nan, "R2_pearson": np.nan, "NSE": np.nan, "NSE_anom": np.nan}
    if n == 0:
        return out

    err = p - t
    out["RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    out["MAE"]  = float(np.mean(np.abs(err)))
    out["bias"] = float(np.mean(err))

    if n >= MIN_N_UBRMSE:
        p_anom = p - p.mean()
        t_anom = t - t.mean()
        out["ubRMSE"] = float(np.sqrt(np.mean((p_anom - t_anom) ** 2)))

        sst = float(np.sum(t_anom ** 2))
        if sst > 0:
            # NSE = 1 - MSE/var(t);  NSE_anom = 1 - ubRMSE^2/var(t).
            # Their difference is exactly bias^2/var(t) -- the level penalty.
            out["NSE"]      = float(1.0 - np.sum(err ** 2) / sst)
            out["NSE_anom"] = float(1.0 - np.sum((p_anom - t_anom) ** 2) / sst)

    if n > 2 and np.std(p) > 0 and np.std(t) > 0:
        r = float(np.corrcoef(p, t)[0, 1])
        out["R"] = r
        out["R2_pearson"] = r ** 2

    return out


def per_station_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, depth)."""
    rows = []
    for (station, depth), g in df.groupby(["station_key", "depth"], observed=True):
        if len(g) < MIN_N:
            continue
        m = metrics_from_arrays(g["pred"].to_numpy(np.float64),
                                g["obs"].to_numpy(np.float64))
        rows.append({"station_key": station, "depth": depth, **m})
    return pd.DataFrame(rows)


def pooled_metrics(df: pd.DataFrame, depth: str) -> dict:
    """Sample-pooled metrics -- matches train.py:compute_metrics.

    Anomalies are removed per station, then ALL samples are pooled into one
    array before the RMS.  Stations with < 2 samples are excluded from ubRMSE
    but still count towards RMSE/MAE/bias, exactly as train.py:368 does.
    """
    g = df[df["depth"] == depth]
    if g.empty:
        return {}
    p = g["pred"].to_numpy(np.float64)
    t = g["obs"].to_numpy(np.float64)
    keys = g["station_key"].to_numpy()

    err = p - t
    out = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE":  float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n_obs": int(len(p)),
    }

    p_anom = np.empty_like(p)
    t_anom = np.empty_like(t)
    ub_mask = np.zeros(len(p), dtype=bool)
    for station in np.unique(keys):
        sel = keys == station
        if sel.sum() < MIN_N_UBRMSE:
            continue
        p_anom[sel] = p[sel] - p[sel].mean()
        t_anom[sel] = t[sel] - t[sel].mean()
        ub_mask[sel] = True

    if ub_mask.any():
        pa, ta = p_anom[ub_mask], t_anom[ub_mask]
        out["ubRMSE"] = float(np.sqrt(np.mean((pa - ta) ** 2)))
        sst = float(np.sum((t[ub_mask] - t[ub_mask].mean()) ** 2))
        if sst > 0:
            out["NSE"]      = float(1.0 - np.sum(err[ub_mask] ** 2) / sst)
            out["NSE_anom"] = float(1.0 - np.sum((pa - ta) ** 2) / sst)
    if len(p) > 2 and np.std(p) > 0 and np.std(t) > 0:
        r = float(np.corrcoef(p, t)[0, 1])
        out["R"] = r
        out["R2_pearson"] = r ** 2
    return out


def summarise(ps: pd.DataFrame, df: pd.DataFrame, split_name: str) -> list:
    """Station-equal and sample-pooled summary rows, one per depth."""
    rows = []
    for depth in SM_DEPTHS:
        sub = ps[ps["depth"] == depth]
        if sub.empty:
            continue
        row = {"split": split_name, "depth": depth,
               "n_stations": int(len(sub)),
               "n_obs": int(sub["n"].sum())}
        # station-equal
        for m in ["RMSE", "ubRMSE", "MAE", "bias", "R", "R2_pearson", "NSE", "NSE_anom"]:
            row[f"{m}_stn"] = float(np.nanmean(sub[m])) if sub[m].notna().any() else np.nan
        # Medians too: NSE is unbounded below, so a handful of low-variance
        # stations drag the mean arbitrarily far negative.  Report both and
        # quote the median when describing the typical station.
        for m in ["NSE", "NSE_anom", "R2_pearson", "ubRMSE"]:
            row[f"{m}_stn_median"] = (float(np.nanmedian(sub[m]))
                                      if sub[m].notna().any() else np.nan)
        # how many stations are worse than predicting their own mean
        row["frac_NSE_anom_lt0"] = (float((sub["NSE_anom"] < 0).mean())
                                    if sub["NSE_anom"].notna().any() else np.nan)
        # RMS of the per-station level offset (§20.1: do not read global bias as calibration)
        row["RMS_offset"] = float(np.sqrt(np.nanmean(sub["bias"] ** 2)))
        # sample-pooled
        for m, v in pooled_metrics(df, depth).items():
            if m == "n_obs":
                continue
            row[f"{m}_pool"] = v
        rows.append(row)
    return rows


def check_val_gate(summary: pd.DataFrame) -> bool:
    """§22.6 hard gate: reproduce val ubRMSE 0.0539 / 0.0500 / 0.0552."""
    val = summary[summary["split"] == "val"]
    if val.empty:
        print("\n[GATE] val split not present -- run "
              "`eval_predict.py --splits val` before trusting held-out numbers.")
        return False

    print("\n" + "=" * 66)
    print("§22.6 HARD GATE -- reproduce train.py val ubRMSE (epoch 16)")
    print("=" * 66)
    ok = True
    for _, r in val.iterrows():
        ref = VAL_REFERENCE.get(r["depth"])
        got = r.get("ubRMSE_pool", np.nan)
        if ref is None or np.isnan(got):
            continue
        delta = abs(got - ref)
        passed = delta < 0.0015          # 1.5 ppt -- covers fp/bf16 nondeterminism
        ok &= passed
        print(f"  {r['depth']:>7s}  expected {ref:.4f}  got {got:.4f}  "
              f"delta {delta:+.4f}  {'PASS' if passed else 'FAIL'}")
    if not ok:
        print("  ** GATE FAILED -- metric code disagrees with train.py:compute_metrics.")
        print("  ** Do not trust any OOS/OOT/OOST number until this passes.")
    return ok


def check_sanity(summary: pd.DataFrame) -> None:
    """§22.6 pre-registered checks. A FAIL is a finding, not a bug to hide."""
    print("\n" + "=" * 66)
    print("§22.6 PRE-REGISTERED SANITY CHECKS")
    print("=" * 66)

    def ub(split, depth):
        r = summary[(summary["split"] == split) & (summary["depth"] == depth)]
        return float(r["ubRMSE_stn"].iloc[0]) if len(r) else np.nan

    for depth in SM_DEPTHS:
        v, oos, oot, oost = (ub("val", depth), ub("oos", depth),
                             ub("oot", depth), ub("oost", depth))
        print(f"\n  {depth}:  val={v:.4f}  oos={oos:.4f}  oot={oot:.4f}  oost={oost:.4f}")
        if not np.isnan(v) and not np.isnan(oot):
            d = abs(oot - v) / v
            print(f"    1. OOT ~ val (same stations, novel year)   "
                  f"{'PASS' if d < 0.25 else 'FAIL'}  ({d*100:.1f}% apart)")
        if not np.isnan(v) and not np.isnan(oos):
            print(f"    2. OOS > val (novel stations)              "
                  f"{'PASS' if oos > v else 'FAIL'}")
        if not np.isnan(oos) and not np.isnan(oost):
            print(f"    3. OOST >= OOS (hardest condition)         "
                  f"{'PASS' if oost >= oos else 'FAIL'}")

    held = summary[summary["split"].isin(["oos", "oot", "oost"])]
    print(f"\n  4. metrics_summary has 9 held-out rows: "
          f"{'PASS' if len(held) == 9 else 'FAIL'}  (got {len(held)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default=str(OUT_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--splits",  nargs="+", default=None,
                   help="Default: every predictions_*.parquet found")
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A split may arrive as one file or as several station chunks
    # (predictions_oot_c0.parquet, ...).  Group every file by its split name.
    known = ["val", "oos", "oost", "oot"]          # longest-first below
    groups: dict[str, list[Path]] = {}
    for path in sorted(in_dir.glob("predictions_*.parquet")):
        stem = path.stem.replace("predictions_", "")
        # match the longest known split name that prefixes the stem, so
        # "oost_c1" resolves to oost and not oos
        split = next((s for s in sorted(known, key=len, reverse=True)
                      if stem == s or stem.startswith(s + "_")), stem)
        groups.setdefault(split, []).append(path)

    if args.splits:
        groups = {s: v for s, v in groups.items() if s in args.splits}
    if not groups:
        raise SystemExit(f"No prediction parquets in {in_dir} -- run eval_predict.py first")

    meta = pd.read_csv(SPLITS_CSV)
    meta["station_key"] = meta.apply(_make_key, axis=1)
    keep = [c for c in META_COLS if c in meta.columns]

    all_summary = []
    for split_name in sorted(groups, key=lambda s: {"val": 0, "oos": 1,
                                                    "oot": 2, "oost": 3}.get(s, 9)):
        chunk_paths = groups[split_name]
        df = pd.concat([pd.read_parquet(p_) for p_ in chunk_paths],
                       ignore_index=True)
        # a station must not appear in two chunks, or its metrics double-count
        df = df.drop_duplicates(subset=["station_key", "year", "doy", "depth"])
        print(f"\n{'='*66}")
        print(f"{split_name.upper()}: {len(df):,} rows | "
              f"{df['station_key'].nunique()} stations"
              + (f" | {len(chunk_paths)} chunks" if len(chunk_paths) > 1 else ""))

        ps = per_station_metrics(df)
        if ps.empty:
            print(f"  No station reached n >= {MIN_N} -- skipping")
            continue

        # wide per-station CSV: one row per station, metrics suffixed by depth
        wide = ps.pivot(index="station_key", columns="depth")
        wide.columns = [f"{m}_{str(d).replace('-', '_')}" for m, d in wide.columns]
        wide = wide.reset_index().merge(meta[keep], on="station_key", how="left")
        wide.insert(0, "eval_split", split_name)
        out_path = out_dir / f"per_station_{split_name}.csv"
        wide.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({len(wide)} stations)")

        rows = summarise(ps, df, split_name)
        all_summary.extend(rows)
        for r in rows:
            print(f"    {r['depth']:>7s}  "
                  f"ubRMSE {r['ubRMSE_stn']:.4f}/{r.get('ubRMSE_pool', np.nan):.4f} "
                  f"(stn/pool)  RMSE {r['RMSE_stn']:.4f}  R2p {r['R2_pearson_stn']:.3f}  "
                  f"NSE {r['NSE_stn']:+.3f}  NSEa {r['NSE_anom_stn']:+.3f}  "
                  f"RMSoff {r['RMS_offset']:.4f}  n={r['n_stations']}")

    summary = pd.DataFrame(all_summary)
    order = {"val": 0, "oos": 1, "oot": 2, "oost": 3}
    summary = summary.sort_values(
        ["split", "depth"],
        key=lambda s: s.map(order) if s.name == "split" else s.map(
            {d: i for i, d in enumerate(SM_DEPTHS)})
    ).reset_index(drop=True)

    summary_path = out_dir / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")

    gate_ok = check_val_gate(summary)
    check_sanity(summary)

    with open(out_dir / "gate.json", "w") as f:
        json.dump({"val_gate_passed": bool(gate_ok),
                   "reference": VAL_REFERENCE}, f, indent=2)


if __name__ == "__main__":
    main()
