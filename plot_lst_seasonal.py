"""
§29 — how heterogeneous is the tile, and does the pattern hold through the year?

Monthly climatology of Landsat ST over one tile at native 30 m.  All clear scenes are pooled by
calendar month, so each panel is the mean LST anomaly (tile mean removed per scene) for that
month across 2016-2022.

Three questions, three parts:
  1. the 12 monthly anomaly maps        -> is the same pattern there in every month?
  2. within-tile SD of anomaly by month -> HOW heterogeneous, in kelvin, and when
  3. pattern correlation vs the annual mean map -> is it the SAME pattern or a different one?

Part 3 is the one that matters: a tile can be heterogeneous every month yet incoherent, which
would mean the structure is weather, not landscape.

Usage:
  conda run -n terramind python plot_lst_seasonal.py --tile ISMN_TxSON_CR200-18
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from extract_lst_timeseries import station_table
from plot_lst import PALETTE, PATCH_PX, INK, MUTED, GRID, load_scene, mark, style

REPO = Path(__file__).resolve().parent
LST_ROOT = Path("/gpfs/scratch1/shared/pkhanal/lst/landsat_st/txson")
SERIES = REPO / "csvs" / "lst_station_timeseries.csv"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--root", type=Path, default=LST_ROOT)
    ap.add_argument("--min-scenes", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=REPO / "figures" / "tile_lst_seasonal")
    args = ap.parse_args()

    style()
    st = station_table()
    ser = pd.read_csv(SERIES)
    ser = ser[ser.tile == args.tile]
    sdf = ser.drop_duplicates("station")[
        ["station", "station_name", "tile_row", "tile_col"]].rename(
        columns={"tile_row": "row", "tile_col": "col"})
    sdf["short"] = sdf.station_name
    sdf = sdf.sort_values("station")

    tifs = sorted(p for p in glob.glob(str(args.root / args.tile / "*.tif"))
                  if not p.endswith(".tmp.tif"))
    print(f"{len(tifs)} scenes")

    # accumulate anomaly and absolute LST per calendar month
    acc = {m: None for m in range(1, 13)}
    cnt = {m: None for m in range(1, 13)}
    absacc = {m: None for m in range(1, 13)}
    nsc = {m: 0 for m in range(1, 13)}
    ext0 = None
    for t in tifs:
        L, A, C, ext, date, tm = load_scene(t, st, args.tile)
        if not np.isfinite(tm) or not date:
            continue
        m = int(date[5:7])
        if acc[m] is None:
            acc[m] = np.zeros_like(A); cnt[m] = np.zeros_like(A)
            absacc[m] = np.zeros_like(A); ext0 = ext
        if A.shape != acc[m].shape:
            continue
        acc[m] += np.nan_to_num(A); absacc[m] += np.nan_to_num(L)
        cnt[m] += C; nsc[m] += 1

    mean_a, mean_l = {}, {}
    for m in range(1, 13):
        if acc[m] is None or nsc[m] < args.min_scenes:
            mean_a[m] = mean_l[m] = None
            continue
        ok = cnt[m] >= max(2, args.min_scenes // 2)
        mean_a[m] = np.where(ok, acc[m] / np.maximum(cnt[m], 1), np.nan)
        mean_l[m] = np.where(ok, absacc[m] / np.maximum(cnt[m], 1), np.nan)

    valid = [m for m in range(1, 13) if mean_a[m] is not None]
    allv = np.concatenate([np.abs(mean_a[m][np.isfinite(mean_a[m])]) for m in valid])
    v = float(np.ceil(np.nanpercentile(allv, 98)))

    # annual mean pattern, for the coherence test
    stack = np.stack([mean_a[m] for m in valid])
    annual = np.nanmean(stack, axis=0)

    fig = plt.figure(figsize=(16.5, 15.5))
    gs = GridSpec(4, 6, figure=fig, hspace=.34, wspace=.16,
                  height_ratios=[1, 1, 1, .72])
    fig.suptitle(f"§29  Within-tile LST heterogeneity through the year — {args.tile}   "
                 f"(monthly mean anomaly, native 30 m, 2016–2022)",
                 x=.012, ha="left", fontsize=13, y=.975)

    im = None
    for k, m in enumerate(range(1, 13)):
        ax = fig.add_subplot(gs[k // 4, (k % 4) + (1 if k // 4 == 99 else 0)]) \
            if False else fig.add_subplot(gs[k // 4, (k % 4)])
        if mean_a[m] is None:
            ax.axis("off")
            ax.set_title(f"{MONTHS[m-1]} — no data", loc="left", fontsize=8.4)
            continue
        im = ax.imshow(mean_a[m], cmap="RdBu_r", vmin=-v, vmax=v,
                       interpolation="nearest", extent=ext0, origin="upper")
        mark(ax, sdf, labels=(k == 0))
        sd = float(np.nanstd(mean_a[m]))
        ax.set_title(f"{MONTHS[m-1]}   n={nsc[m]} scenes\n"
                     f"mean LST {np.nanmean(mean_l[m]):.0f} K · SD {sd:.2f} K",
                     loc="left", fontsize=8.4)

    # shared colourbar for all 12 (they ARE comparable — that's the point)
    cax = fig.add_subplot(gs[0:3, 4])
    cax.set_position([0.685, 0.40, 0.010, 0.42])
    plt.colorbar(im, cax=cax).set_label("monthly mean LST anomaly (K)", fontsize=9)

    # ---- annual mean pattern ----
    ax = fig.add_subplot(gs[0:2, 5])
    ax.set_position([0.75, 0.56, 0.20, 0.26])
    im2 = ax.imshow(annual, cmap="RdBu_r", vmin=-v, vmax=v,
                    interpolation="nearest", extent=ext0, origin="upper")
    mark(ax, sdf)
    ax.set_title("ANNUAL mean pattern\n(reference for coherence)", loc="left", fontsize=8.8)

    # ---- summary 1: how heterogeneous, by month ----
    ax1 = fig.add_subplot(gs[3, 0:2])
    sds = [float(np.nanstd(mean_a[m])) if mean_a[m] is not None else np.nan
           for m in range(1, 13)]
    p90 = [float(np.nanpercentile(mean_a[m], 95) - np.nanpercentile(mean_a[m], 5))
           if mean_a[m] is not None else np.nan for m in range(1, 13)]
    ax1.bar(range(1, 13), sds, color=PALETTE[0], width=.62, label="spatial SD")
    ax1.plot(range(1, 13), p90, "o-", color=PALETTE[1], ms=5, lw=1.6,
             label="p95 − p5 spread")
    ax1.axhline(2.13, color=MUTED, ls="--", lw=1.2)
    ax1.annotate("median per-pixel ST_QA 2.13 K (single-date noise floor)",
                 (0.6, 2.13), xytext=(0, 5), textcoords="offset points",
                 fontsize=7.0, color=MUTED)
    ax1.set_xticks(range(1, 13)); ax1.set_xticklabels(MONTHS, fontsize=7.6)
    ax1.set_ylabel("K")
    ax1.set_title("HOW heterogeneous (K)", loc="left", fontsize=9.0)
    ax1.legend(fontsize=7.6, ncol=2)

    # ---- summary 2: is it the SAME pattern each month? ----
    ax2 = fig.add_subplot(gs[3, 2:4])
    cors = []
    for m in range(1, 13):
        if mean_a[m] is None:
            cors.append(np.nan); continue
        a, b = mean_a[m].ravel(), annual.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        cors.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    ax2.bar(range(1, 13), cors, color=PALETTE[2], width=.62)
    ax2.axhline(0, color=MUTED, lw=1.0)
    ax2.set_ylim(-0.2, 1.05)
    ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(MONTHS, fontsize=7.6)
    ax2.set_ylabel("spatial r vs annual mean")
    ax2.set_title("SAME pattern each month?", loc="left", fontsize=9.0)

    # ---- summary 3: per-station seasonal cycle of anomaly ----
    ax3 = fig.add_subplot(gs[3, 4:])
    s = ser.copy(); s["month"] = pd.to_datetime(s.date).dt.month
    for k, (name, g) in enumerate(s.groupby("station_name")):
        mm = g.groupby("month").lst_anom_k.mean()
        ax3.plot(mm.index, mm.values, "o-", ms=4, lw=1.4,
                 color=PALETTE[k % len(PALETTE)], label=name)
    ax3.axhline(0, color=MUTED, lw=1.0)
    ax3.set_xticks(range(1, 13)); ax3.set_xticklabels(MONTHS, fontsize=7.6)
    ax3.set_ylabel("station LST anomaly (K)")
    ax3.set_title("Per-station anomaly by month", loc="left", fontsize=9.0)
    ax3.legend(fontsize=7.0, ncol=2)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.tile}.png"
    fig.savefig(out, dpi=175, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    print("\nmonth  n_scenes  spatialSD(K)  r_vs_annual")
    for m in range(1, 13):
        print(f"  {MONTHS[m-1]}    {nsc[m]:3d}       "
              f"{sds[m-1]:8.2f}      {cors[m-1]:+.3f}"
              if mean_a[m] is not None else f"  {MONTHS[m-1]}    {nsc[m]:3d}          --")
    print(f"\nannual-mean pattern spatial SD = {np.nanstd(annual):.2f} K")
    print(f"mean month-to-annual coherence  = {np.nanmean(cors):+.3f}")


if __name__ == "__main__":
    main()
