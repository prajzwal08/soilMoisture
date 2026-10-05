#!/usr/bin/env python
"""§67 input-ablation figure: how much worse the frozen baseline gets when one input is shuffled.

Reads the summary CSV written by compare_ablation.py (one row per pass x soil layer, paired
per-station medians with 95 % bootstrap CI) and draws, in the paper style (plot_style_bw):
  (a) heatmap pass x layer of the relative ubRMSE change (%)   — the dynamics
  (b) heatmap pass x layer of the relative RMSE change (%)     — dynamics + level (statics show here)
Cells whose 95 % CI excludes 0 are marked with an asterisk. Diverging colours centred on 0
(orange = worse, blue = better), so "no effect" reads as white.
Also writes the table as .md next to the figure.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import plot_style_bw

DEPTHS = ["0-10", "10-30", "30-100"]
# pass order + paper labels (Q1-Q9 in runbook §67)
PASSES = [
    ("era5_cross_station",   "ERA5 (other station)"),
    ("era5_within_station",  "ERA5 (same station, other time)"),
    ("sat_cross_station",    "All satellite"),
    ("s2_cross_station",     "S2 160 m"),
    ("s1_cross_station",     "S1 160 m"),
    ("s1_within_station",    "S1 160 m (same station, other time)"),
    ("fine_cross_station",   "20 m imagery"),
    ("dem_cross_station",    "DEM"),
    ("lulc_cross_station",   "Land cover"),
    ("soil_cross_station",   "Soil"),
    ("sif_cross_station",    "SIF"),
    ("twsa_cross_station",   "TWSA"),
]


def _pass_key(stem: str) -> str:
    """predictions_oos_era5_cross_station_s0 -> era5_cross_station"""
    s = stem.split("predictions_oos_", 1)[-1]
    return s.rsplit("_s", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="compare_ablation.py --csv output")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    plt.rcParams.update(plot_style_bw.RC_PAPER)
    s = pd.read_csv(a.summary)
    s["key"] = s["ablation"].map(_pass_key)
    order = [(k, lab) for k, lab in PASSES if k in set(s["key"])]
    extra = sorted(set(s["key"]) - {k for k, _ in order})
    order += [(k, k) for k in extra]

    # orange (worse) - white - blue (better), from palette H
    cmap = LinearSegmentedColormap.from_list("worse_better", [plot_style_bw.BLUE, "white", plot_style_bw.RED])
    panels = [("d_ubRMSE_pct", "d_ubRMSE_lo", "d_ubRMSE_hi", "(a) ubRMSE change (%)"),
              ("d_RMSE_pct", "d_RMSE_lo", "d_RMSE_hi", "(b) RMSE change (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(order) + 1.6), constrained_layout=True,
                             sharey=True)
    for ax, (col, lo, hi, title) in zip(axes, panels):
        m = np.full((len(order), len(DEPTHS)), np.nan)
        sig = np.zeros_like(m, dtype=bool)
        for i, (k, _) in enumerate(order):
            for jx, d in enumerate(DEPTHS):
                r = s[(s["key"] == k) & (s["depth"] == d)]
                if len(r):
                    m[i, jx] = r[col].iloc[0]
                    sig[i, jx] = (r[lo].iloc[0] > 0) or (r[hi].iloc[0] < 0)
        vmax = max(np.nanmax(np.abs(m)), 1.0)
        im = ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        for i in range(m.shape[0]):
            for jx in range(m.shape[1]):
                if np.isfinite(m[i, jx]):
                    v = m[i, jx]
                    ax.text(jx, i, f"{v:+.1f}{'*' if sig[i, jx] else ''}", ha="center", va="center",
                            color="white" if abs(v) > 0.6 * vmax else "black")
        ax.set_xticks(range(len(DEPTHS)), [f"{d} cm" for d in DEPTHS])
        ax.xaxis.tick_top()
        ax.set_title(title, loc="left", pad=28)
        fig.colorbar(im, ax=ax, shrink=0.9, label="% change vs. baseline")
    axes[0].set_yticks(range(len(order)), [lab for _, lab in order])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"input_ablation_heatmap.{ext}", dpi=600)
    keep = ["key", "depth", "n", "ubRMSE_base", "d_ubRMSE", "d_ubRMSE_lo", "d_ubRMSE_hi", "d_ubRMSE_pct",
            "d_RMSE_pct", "d_absbias", "d_r", "frac_worse"]
    tab = s[[c for c in keep if c in s]].round(4)
    try:
        md = tab.to_markdown(index=False)
    except ImportError:
        md = tab.to_string(index=False)
    (out / "input_ablation_table.md").write_text(md + "\n")
    print(f"wrote {out}/input_ablation_heatmap.{{png,pdf}} and input_ablation_table.md")


if __name__ == "__main__":
    main()
