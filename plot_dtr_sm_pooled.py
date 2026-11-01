#!/usr/bin/env python
"""Pooled DTR vs same-day soil moisture -- every station, every DTR day, one plot.

One point per (station, day) on which a usable DTR pair and an OBSERVED (qc==0) soil
moisture value both exist.  All stations combined, no centring, no per-station split:
the literal "soil moisture that day against DTR that day".

Drawn as a density rather than a scatter.  At n = 11k the scatter is a solid blob and
hides exactly the thing being asked about, so the marks are hexbin counts on a log scale
(single-hue sequential = magnitude) with a binned median and interquartile band over the
top -- the median line, not the cloud, is what carries the relationship.

Soil moisture here is 0-10 cm ONLY -- a FIXED depth bin, and the layer thermal physics
actually senses. The depth-averaged alternative is deliberately not drawn: its depth set
is whatever passes qc==0 that day, so the quantity is not constant between stations or
between dates, and its pooled r is weaker anyway (-0.076 vs -0.107).
.

The 0-40 K guide marks the physically plausible DTR band.  11.8% of these days are
NEGATIVE DTR, which is not all bad retrieval: the ISS precesses, so a "day" overpass in
the 6-19 h band can land near dawn, before peak heating (§36.24 keeps solar phase as a
covariate for this and it is NOT conditioned on here).  r is reported both over
everything and over the plausible band, so the effect of those days is visible.
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


def rval(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, 0
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def binned(x, y, nbins=14):
    """Median and IQR of y in equal-count bins of x."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    cx, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b) if b != edges[-1] else (x >= a) & (x <= b)
        if s.sum() < 25:
            continue
        cx.append(np.median(x[s])); med.append(np.median(y[s]))
        q1.append(np.percentile(y[s], 25)); q3.append(np.percentile(y[s], 75))
    return map(np.asarray, (cx, med, q1, q3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "csvs" / "ecostress_dtr_vs_sm_all.csv"))
    ap.add_argument("--network", default="", help="substring match on the station folder")
    ap.add_argument("--dt-lo", type=float, default=None)
    ap.add_argument("--dt-hi", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(ROOT / "fig" / "dtr_txson" / "dtr_sm_pooled_surface.png"))
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    if args.network:
        n0 = len(d)
        d = d[d["folder"].str.contains(args.network, case=False, na=False)]
        print(f"network {args.network}: {n0:,} -> {len(d):,} station-days")
    if args.dt_lo is not None or args.dt_hi is not None:
        if "dt_hours" not in d.columns:
            raise SystemExit("dt_hours is not in the CSV -- rerun plot_dtr_vs_sm.py first")
        lo = -np.inf if args.dt_lo is None else args.dt_lo
        hi = np.inf if args.dt_hi is None else args.dt_hi
        n0 = len(d)
        d = d[(d.dt_hours >= lo) & (d.dt_hours < hi)]
        print(f"dt band [{lo}, {hi}) h : {n0:,} -> {len(d):,} station-days")
    print(f"{len(d):,} station-days over {d['station_id'].nunique()} stations")

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    col, xlab, cmap = "sm_surface", "0-10 cm soil moisture (m³/m³)", "Greens"

    keep = [col, "dtr_k"] + (["dt_hours"] if "dt_hours" in d.columns else [])
    s = d[keep].dropna(subset=[col, "dtr_k"])
    x, y = s[col].to_numpy(), s["dtr_k"].to_numpy()
    # MARK CHOICE FOLLOWS n.  A hexbin needs enough points per cell to read as a density;
    # at a few hundred it is a sparse grid of near-identical pale cells and hides both the
    # individual days and the spread. Below 1500 points the raw marks are drawn instead,
    # split by dt band because §36.24's two bands are different geometries (6-9 h samples
    # day_tst ~14.1 h, >=15 h samples ~9.7 h) and not two draws from one population.
    if len(s) < 1500:
        short = s["dt_hours"] < 12 if "dt_hours" in s else np.ones(len(s), bool)
        ax.scatter(x[short], y[short], s=17, alpha=.55, lw=0, color="#C1502E",
                   label=f"dt 6–9 h  (n = {int(short.sum()):,})")
        ax.scatter(x[~short], y[~short], s=17, alpha=.55, lw=0, color="#3B6EA5",
                   marker="s", label=f"dt ≥15 h  (n = {int((~short).sum()):,})")
        hb = None
    else:
        hb = ax.hexbin(x, y, gridsize=58, bins="log", cmap=cmap, mincnt=1, linewidths=0)

    cx, med, q1, q3 = binned(x, y)
    ax.fill_between(cx, q1, q3, color="#00000018", lw=0, zorder=3,
                    label="interquartile range")
    ax.plot(cx, med, color="#111111", lw=2.4, zorder=4,
            label="median DTR in equal-count SM bins")
    ax.plot(cx, med, "o", color="#111111", ms=5, zorder=5)

    r_all, n_all = rval(x, y)
    band = (y >= 0) & (y <= 40)
    r_bd, n_bd = rval(x[band], y[band])

    ax.axhline(0, color="#999", lw=.9, ls="--", zorder=2)
    ax.axhline(40, color="#999", lw=.9, ls="--", zorder=2)
    ax.set_xlabel(xlab, fontsize=10.5)
    ax.set_ylabel("DTR = day LST - night LST  (K)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    if hb is not None:
        cb = fig.colorbar(hb, ax=ax, pad=.02, fraction=.046)
        cb.set_label("station-days per cell", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)

    print(f"{col}: r_all {r_all:+.3f} (n={n_all})   r_0-40K {r_bd:+.3f} (n={n_bd})")
    print("   binned medians:",
          ", ".join(f"{a:.2f}->{b:.1f}K" for a, b in zip(cx, med)))

    fig.suptitle(f"ECOSTRESS DTR vs same-day SURFACE (0-10 cm) soil moisture{args.label}\n"
                 f"all {d['station_id'].nunique()} stations pooled, {len(s):,} station-days"
                 f"     r = {r_all:+.3f}"
                 f"     within the plausible 0-40 K band: r = {r_bd:+.3f} (n = {n_bd:,})",
                 fontsize=11.5, y=1.015)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
