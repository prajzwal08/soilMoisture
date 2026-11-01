#!/usr/bin/env python
"""Per-station r(DTR, same-day surface SM) -- and whether any station really is better.

One r per station, over that station's own usable DTR days.  Then the question the
boxplot is for: is the spread across stations REAL, or is it what you would get from
pure sampling noise at n = 10-30 days?

THE NULL IS THE POINT.  At n = 15 the sampling SD of r under no relationship is about
0.27, so a station showing r = -0.5 is entirely unremarkable on its own.  Every panel
therefore carries a shuffled control: SM permuted within station, r recomputed, repeated
N_SHUFFLE times.  If the observed box is not visibly shifted from the shuffled box, the
"stations differ" reading is unsupported no matter how wide the spread looks.

Breakdowns are by the macro classes already in station_splits.csv -- Koppen climate,
IGBP land cover, elevation band -- plus r against n, which exposes the usual artefact
where the most extreme correlations all come from the fewest observations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import ROOT  # noqa: E402

N_SHUFFLE = 200
ACC, NUL = "#2E7D5B", "#9a9a9a"


def rv(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def box(ax, data, labels, colors, widths=.6):
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=widths,
                    showfliers=False, medianprops=dict(color="#111", lw=1.8),
                    whiskerprops=dict(color="#666"), capprops=dict(color="#666"))
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c); p.set_alpha(.45); p.set_edgecolor("#555")
    return bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "csvs" / "ecostress_dtr_vs_sm_all.csv"))
    ap.add_argument("--splits", default=str(ROOT / "csvs" / "station_splits.csv"))
    ap.add_argument("--min-days", type=int, default=8)
    ap.add_argument("--dtr-min", type=float, default=None,
                    help="drop days with DTR below this (use 0 to drop negative DTR)")
    ap.add_argument("--dtr-max", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    d = pd.read_csv(args.csv).dropna(subset=["sm_surface", "dtr_k"])
    if args.dtr_min is not None or args.dtr_max is not None:
        lo = -np.inf if args.dtr_min is None else args.dtr_min
        hi = np.inf if args.dtr_max is None else args.dtr_max
        n0, s0 = len(d), d.station_id.nunique()
        d = d[(d.dtr_k >= lo) & (d.dtr_k <= hi)]
        print(f"DTR filter [{lo}, {hi}] K: {n0:,} -> {len(d):,} station-days, "
              f"{s0} -> {d.station_id.nunique()} stations")
    meta = (pd.read_csv(args.splits)
            .drop_duplicates("station_id")
            .set_index("station_id")[["kg_macro", "igbp_macro", "elevation_band",
                                      "network", "latitude"]])
    rng = np.random.default_rng(args.seed)

    rows, null = [], []
    for sid, g in d.groupby("station_id"):
        if len(g) < args.min_days:
            continue
        x, y = g["sm_surface"].to_numpy(), g["dtr_k"].to_numpy()
        r = rv(x, y)
        if not np.isfinite(r):
            continue
        rows.append({"station_id": sid, "r": r, "n": len(g),
                     "dtr_mean": float(np.mean(y)), "sm_mean": float(np.mean(x)),
                     "sm_range": float(np.ptp(x))})
        null.extend(rv(rng.permutation(x), y) for _ in range(N_SHUFFLE))

    t = pd.DataFrame(rows).join(meta, on="station_id")
    null = np.array([v for v in null if np.isfinite(v)])
    t.to_csv(ROOT / "csvs" / f"ecostress_dtr_sm_per_station_r{args.tag}.csv", index=False)

    print(f"stations with >= {args.min_days} DTR days : {len(t)}")
    print(f"  observed r : median {t.r.median():+.3f}  mean {t.r.mean():+.3f}  "
          f"SD {t.r.std():.3f}  {100*(t.r<0).mean():.0f}% negative")
    print(f"  shuffled r : median {np.median(null):+.3f}  mean {null.mean():+.3f}  "
          f"SD {null.std():.3f}  {100*(null<0).mean():.0f}% negative")
    print(f"  p05/p95    observed {np.percentile(t.r,5):+.3f}/{np.percentile(t.r,95):+.3f}"
          f"   shuffled {np.percentile(null,5):+.3f}/{np.percentile(null,95):+.3f}")
    print(f"\n  median n per station = {t.n.median():.0f}  "
          f"(sampling SD of r at that n ~ {1/np.sqrt(t.n.median()-3):.2f})")

    for key in ("kg_macro", "igbp_macro", "elevation_band"):
        print(f"\n  by {key}:")
        for k, g in t.groupby(key):
            if len(g) >= 8:
                print(f"    {str(k):<16} n={len(g):4d}  median r {g.r.median():+.3f}"
                      f"   IQR {g.r.quantile(.25):+.3f}..{g.r.quantile(.75):+.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

    ax = axes[0, 0]
    box(ax, [t.r.values, null], ["observed\n(one r per station)",
                                 f"shuffled control\n({N_SHUFFLE}x per station)"],
        [ACC, NUL])
    ax.scatter(np.random.normal(1, .055, len(t)), t.r, s=9, alpha=.35, lw=0, color=ACC)
    ax.axhline(0, color="#bbb", lw=.9, ls="--")
    ax.set_ylabel("r(DTR, same-day 0–10 cm SM)")
    ax.set_title(f"A  Per-station r vs its own null\n"
                 f"{len(t)} stations, median n = {t.n.median():.0f} days", fontsize=10.5)

    ax = axes[0, 1]
    ax.scatter(t.n, t.r, s=16, alpha=.5, lw=0, color=ACC)
    nn = np.arange(max(4, t.n.min()), t.n.max() + 1)
    ax.plot(nn, 1.96 / np.sqrt(nn - 3), "--", color="#999", lw=1.2,
            label="±1.96/√(n−3): the null band")
    ax.plot(nn, -1.96 / np.sqrt(nn - 3), "--", color="#999", lw=1.2)
    ax.axhline(0, color="#bbb", lw=.9)
    ax.set_xscale("log"); ax.set_xlabel("DTR days at that station (n)")
    ax.set_ylabel("r"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("B  Are the strong stations just the thin ones?", fontsize=10.5)

    for ax, key, ttl in ((axes[1, 0], "kg_macro", "C  by Köppen macro class"),
                         (axes[1, 1], "igbp_macro", "D  by IGBP macro class")):
        grp = [(k, g.r.values) for k, g in t.groupby(key) if len(g) >= 8]
        grp.sort(key=lambda kv: np.median(kv[1]))
        if not grp:
            ax.set_visible(False)
            continue
        box(ax, [v for _, v in grp],
            [f"{k}\n(n={len(v)})" for k, v in grp], [ACC] * len(grp), widths=.55)
        ax.axhline(np.median(null), color=NUL, lw=1.4, ls="--",
                   label="shuffled median")
        ax.axhline(0, color="#ddd", lw=.8)
        ax.set_ylabel("r"); ax.set_title(ttl, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)

    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Does DTR track soil moisture better at some stations than others?",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    p = ROOT / "fig" / "dtr_txson" / f"dtr_sm_per_station_r{args.tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
