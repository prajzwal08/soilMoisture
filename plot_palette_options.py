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
PALETTES = [  # (name, OOS, OOT, OOST) — all pass the colour-blind check; counts are drawn in black
    ("C  Paul Tol high-contrast: blue / gold / rose (current)", "#004488", "#DDAA33", "#BB5566"),
    ("G  navy / teal / brick",                        "#1F4E79", "#2A9D8F", "#C0392B"),
    ("H  dark blue / light blue / orange",            "#08519C", "#6BAED6", "#E6550D"),
    ("I  Paul Tol muted: indigo / green / rose",      "#332288", "#117733", "#CC6677"),
    ("J  dark blue / rose / light blue",              "#004488", "#BB5566", "#6699CC"),
    ("A  Nature (NPG): navy / green / red",           "#3C5488", "#00A087", "#E64B35"),
    ("F  navy / grey / brick",                        "#2E5A87", "#7F7F7F", "#C8553D"),
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
            for v, x in zip(data, pos):            # station counts in plain black, as in the paper style
                ax.annotate(f"{len(v)}", xy=(x, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                            fontsize=10, color="black")
        ax.set_title(name, loc="left")
        ax.set_ylabel("ubRMSE (m$^3$/m$^3$)")
        ax.set_ylim(0, 0.11)
        ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
        ax.legend(handles=[Patch(fc=col[s], ec="black", label=s.upper()) for s in SPLITS],
                  frameon=False, loc="upper right", ncol=3)
    axes[-1].set_xticks(range(len(order)), order)
    axes[-1].tick_params(axis="x", pad=18)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"palette_options.{ext}", dpi=300)
    print(f"wrote {OUT}/palette_options.png")


if __name__ == "__main__":
    main()
