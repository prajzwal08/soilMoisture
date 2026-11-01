"""TxSON map: does the model reproduce the spatial field, or flatten it? (§35.33)

The §35.33.6 finding is that the model predicts the station MEAN well only where it has
seen the station. TxSON is the place to look at that spatially: 40 stations in a 55 x 55 km
domain (§47 split: 8 train, 2 val, 30 oos; none in oot -- TxSON labels stop 2022-11-07).

Four panels over the same domain, one dot per station:

    (a) observed station-mean SM
    (b) predicted station-mean SM      -- SHARED colour scale with (a); that is the point
    (c) bias (predicted - observed)
    (d) ubRMSE

(a) and (b) share a scale deliberately. A model that had learned the field would reproduce
its range; one that has learned "TxSON is about this wet" produces a flat panel next to a
varied one, and no amount of per-panel autoscaling should be allowed to hide it. The
printed SD ratio under the title is the same statement as a number.

The six stations inside tile ISMN_TxSON_CR200-18 are ringed -- observed station-mean spread
there is 0.0601 (§29.1) against a predicted spread of 0.0113 at r = -0.175 (§26.11).

CPU only. Reads eval_output/{run}/predictions_{split}.parquet, or with --network-ts the own-tile
(centre) rows of combine_network.py's {tag}_timeseries.parquet -- the only source that also
holds the 8 TRAIN stations. Train stations are drawn as squares and left out of the
headline SD ratio: their own label was seen in training. Seconds to run.

Usage:
    python plot_txson_map.py --in-dir eval_output/pw_stage2a_L3 \
                             --out-dir figures/eval/pw_stage2a_L3
    python plot_txson_map.py --network-ts eval_output/<run>/txson_timeseries.parquet \
                             --out-dir figures/eval/<run>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")

# The six stations that fall inside tile ISMN_TxSON_CR200-18 (§29.1), centre first.
CR200_18_TILE = [
    "ISMN_TxSON_CR200-18", "ISMN_TxSON_CR200-25", "ISMN_TxSON_CR1000-2",
    "ISMN_TxSON_CR200-24", "ISMN_TxSON_CR200-15", "ISMN_TxSON_CR200-6",
]

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def _make_key(r) -> str:
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="eval_output/pw_stage2a_L3")
    ap.add_argument("--out-dir", default="figures/eval/pw_stage2a_L3")
    ap.add_argument("--splits",  nargs="+", default=["val", "oos", "oost"])
    ap.add_argument("--depth",   default="0-10")
    ap.add_argument("--network", default="TxSON")
    ap.add_argument("--network-ts", default=None,
                    help="combine_network.py timeseries parquet; uses its own-tile rows "
                         "instead of the per-split prediction parquets")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    meta = pd.read_csv(SPLITS_CSV)
    meta["station_key"] = meta.apply(_make_key, axis=1)
    cur_split = dict(zip(meta["station_key"], meta["split"]))

    frames = []
    if args.network_ts:
        ts = pd.read_parquet(args.network_ts)
        ts = ts[ts["is_centre"].astype(bool) & (ts["depth"] == args.depth)
                & ts["obs"].notna()
                & ts["station"].astype(str).str.contains(args.network, na=False)]
        if ts.empty:
            raise SystemExit(f"No own-tile {args.network} rows at {args.depth} "
                             f"in {args.network_ts}")
        d = pd.to_datetime(ts["date"])
        frames.append(pd.DataFrame({
            "station_key": ts["station"].astype(str).to_numpy(),
            "year": d.dt.year.to_numpy(), "doy": d.dt.dayofyear.to_numpy(),
            "obs": ts["obs"].to_numpy(), "pred": ts["pred"].to_numpy(),
            "eval_split": ts["station"].astype(str).map(cur_split).fillna("n/a").to_numpy()}))
        print(f"[network-ts] {len(frames[0]):,} rows | "
              f"{frames[0]['station_key'].nunique()} {args.network} stations")
    for split in ([] if args.network_ts else args.splits):
        p = in_dir / f"predictions_{split}.parquet"
        if not p.exists():
            print(f"[{split}] no parquet — skipping")
            continue
        df = pd.read_parquet(p)
        df = df[(df["depth"] == args.depth)
                & (df["station_key"].str.contains(args.network, na=False))]
        if df.empty:
            print(f"[{split}] no {args.network} stations at {args.depth}")
            continue
        df = df.assign(eval_split=split)
        frames.append(df)
        print(f"[{split}] {len(df):,} rows | {df['station_key'].nunique()} "
              f"{args.network} stations")

    if not frames:
        raise SystemExit(f"No {args.network} rows at {args.depth} in {in_dir}")

    df = pd.concat(frames, ignore_index=True)
    # A station appearing in two splits (oos continues into 2023 as oost) must not be
    # averaged twice with different weights — drop exact duplicate observations first.
    df = df.drop_duplicates(subset=["station_key", "year", "doy"])

    stn = (df.groupby("station_key")
             .agg(obs=("obs", "mean"), pred=("pred", "mean"),
                  n=("obs", "size"), split=("eval_split", "first"))
             .reset_index())
    stn["bias"] = stn["pred"] - stn["obs"]

    stn = stn.merge(meta[["station_key", "latitude", "longitude"]],
                    on="station_key", how="left")
    missing = stn["latitude"].isna().sum()
    if missing:
        print(f"  WARNING {missing} stations have no lat/lon and are dropped from the map")
        stn = stn.dropna(subset=["latitude", "longitude"])

    # ubRMSE per station: RMSE of the mean-removed series, computed on the daily rows.
    ub = []
    for k, g in df.groupby("station_key"):
        a = g["pred"].to_numpy(np.float64) - g["pred"].mean()
        b = g["obs"].to_numpy(np.float64) - g["obs"].mean()
        ub.append({"station_key": k, "ubRMSE": float(np.sqrt(np.mean((a - b) ** 2)))})
    stn = stn.merge(pd.DataFrame(ub), on="station_key", how="left")

    # Headline statistics on HELD-OUT stations only: a train station's mean was fitted.
    is_train = stn["split"] == "train"
    held = stn[~is_train]
    if is_train.any():
        print(f"  {int(is_train.sum())} train stations: drawn as squares, excluded from "
              f"the SD ratio and r")
    sd_obs, sd_pred = held["obs"].std(), held["pred"].std()
    r = held[["obs", "pred"]].corr().iloc[0, 1]
    print(f"\n{args.network} {args.depth}: {len(held)} held-out stations")
    print(f"  observed  station-mean SD  {sd_obs:.4f}   range "
          f"{held['obs'].min():.3f}–{held['obs'].max():.3f}")
    print(f"  predicted station-mean SD  {sd_pred:.4f}   range "
          f"{held['pred'].min():.3f}–{held['pred'].max():.3f}")
    print(f"  SD ratio (pred/obs)        {sd_pred / sd_obs:.3f}")
    print(f"  r(obs, pred) across stations  {r:+.3f}")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.dpi": 300, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), constrained_layout=True)

    # (a) and (b) SHARE a scale: the comparison is the figure.
    lo = float(min(stn["obs"].min(), stn["pred"].min()))
    hi = float(max(stn["obs"].max(), stn["pred"].max()))
    blim = float(np.abs(stn["bias"]).max())
    ulo, uhi = float(stn["ubRMSE"].min()), float(stn["ubRMSE"].max())

    panels = [
        ("(a) observed station-mean SM",  "obs",    "viridis",  lo,     hi,    None),
        ("(b) predicted station-mean SM", "pred",   "viridis",  lo,     hi,    None),
        ("(c) bias (pred − obs)",         "bias",   "RdBu_r",  -blim,   blim,  None),
        ("(d) ubRMSE",                    "ubRMSE", "magma_r",  ulo,    uhi,   None),
    ]

    tile_set = set(CR200_18_TILE)
    for ax, (title, col, cmap, vmin, vmax, _) in zip(axes.ravel(), panels):
        # Held-out stations as circles, train stations as squares, on one colour scale.
        for grp, mk in ((held, "o"), (stn[is_train], "s")):
            if grp.empty:
                continue
            sc = ax.scatter(grp["longitude"], grp["latitude"], c=grp[col],
                            cmap=cmap, vmin=vmin, vmax=vmax, s=64, marker=mk,
                            edgecolor="white", linewidth=0.6, zorder=3)
        # Ring the six CR200-18-tile stations — the within-tile claim of §26/§29.
        tile = stn[stn["station_key"].isin(tile_set)]
        if not tile.empty:
            ax.scatter(tile["longitude"], tile["latitude"], s=180, facecolors="none",
                       edgecolors=INK, linewidth=1.1, zorder=4)
        fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel("longitude");  ax.set_ylabel("latitude")
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        f"{args.network} {args.depth} cm, {len(held)} held-out stations — "
        f"predicted station-mean SD is {sd_pred / sd_obs:.2f}x the observed "
        f"({sd_pred:.4f} vs {sd_obs:.4f}), r = {r:+.2f}\n"
        f"(a) and (b) share a colour scale; ringed = the six inside tile CR200-18; "
        f"squares = train (not in the SD ratio)",
        fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{args.network.lower()}_map_pred_vs_obs.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")

    csv = out_dir / f"{args.network.lower()}_station_means.csv"
    stn.sort_values("obs").to_csv(csv, index=False)
    print(f"wrote {csv}")

    tile = stn[stn["station_key"].isin(tile_set)]
    if not tile.empty:
        print(f"\ntile CR200-18 ({len(tile)} of 6 stations present):")
        print(f"  observed  spread {tile['obs'].max() - tile['obs'].min():.4f}")
        print(f"  predicted spread {tile['pred'].max() - tile['pred'].min():.4f}")
        for _, r_ in tile.sort_values("obs").iterrows():
            print(f"    {r_['station_key']:<32s} obs {r_['obs']:.4f}  "
                  f"pred {r_['pred']:.4f}  bias {r_['bias']:+.4f}")


if __name__ == "__main__":
    main()
