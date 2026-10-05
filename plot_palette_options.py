#!/usr/bin/env python
"""§66 palette chooser: the 0-10 cm ubRMSE-by-land-cover panel drawn in candidate scientific palettes.

Reads the backing CSV that plot_eval_ecosystem.py already wrote (no metrics recomputed) and draws one
panel per palette, same data, same paper fonts (Times-like serif, bold), solid fills (no transparency
wash-out). All candidates pass the colour-blind check (dataviz validate_palette.js, worst CVD dE >= 8.9)
(the plain matplotlib blue/green/red FAILS, dE 3.9, and is not offered).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import plot_style_bw

SRC = Path("figures/eval/baseline_selected_20261005_paper/box_ubrmse_by_igbp_macro.csv")
OUT = Path("figures/eval/baseline_selected_20261005_paper/palette_options")
SPLITS = ["oos", "oot", "oost"]
PALETTES = [  # (name, OOS, OOT, OOST) — all pass the colour-blind check (worst CVD dE >= 8.9)
    ("A  Nature (NPG): navy / green / red",          "#3C5488", "#00A087", "#E64B35"),
    ("B  Paul Tol bright: blue / green / rose",      "#4477AA", "#228833", "#EE6677"),
    ("C  Paul Tol high-contrast: blue / gold / rose", "#004488", "#DDAA33", "#BB5566"),
    ("D  ColorBrewer Dark2: teal / orange / purple", "#1B9E77", "#D95F02", "#7570B3"),
    ("E  navy / teal / gold",                        "#0F4C81", "#5AA9A6", "#E1B12C"),
    ("F  navy / grey / brick",                       "#2E5A87", "#A3A3A3", "#C8553D"),
]


def main():
    plt.rcParams.update(plot_style_bw.RC_PAPER)
    d = pd.read_csv(SRC)
    d = d[d["depth"] == "0-10"]
    order = d.groupby("class")["station_key"].nunique().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(len(PALETTES), 1, figsize=(8.0, 2.6 * len(PALETTES)),
                             sharex=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    width = 0.84 / len(SPLITS)
    for ax, (name, *cols) in zip(axes, PALETTES):
        col = dict(zip(SPLITS, cols))
        for k, s in enumerate(SPLITS):
            off = (k - 1) * width
            data = [d[(d["split"] == s) & (d["class"] == c)]["value"].to_numpy() for c in order]
            pos = [i + off for i in range(len(order))]
            for v, x in zip(data, pos):
                ax.scatter(x + rng.uniform(-width * 0.2, width * 0.2, len(v)), v, s=6,
                           color="black", alpha=0.35, lw=0, zorder=2)   # behind the boxes
            bp = ax.boxplot(data, positions=pos, widths=width * 0.66, showfliers=False,
                            patch_artist=True, zorder=3, medianprops=dict(color="black", lw=1.6),
                            boxprops=dict(lw=1.0), whiskerprops=dict(lw=1.0), capprops=dict(lw=1.0))
            for p in bp["boxes"]:
                p.set_facecolor(col[s]); p.set_edgecolor("black"); p.set_alpha(1.0)
        ax.set_title(name, loc="left")
        ax.set_ylabel("ubRMSE (m$^3$/m$^3$)")
        ax.set_ylim(0, 0.11)
        ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
        ax.legend(handles=[Patch(fc=col[s], ec="black", label=s.upper()) for s in SPLITS],
                  frameon=False, loc="upper right", ncol=3)
    axes[-1].set_xticks(range(len(order)), order)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"palette_options.{ext}", dpi=300)
    print(f"wrote {OUT}/palette_options.png")


if __name__ == "__main__":
    main()
