"""
Per-station ubRMSE distributions by depth and split (§22, Phase 3).

Reads eval_output/per_station_{split}.csv -- the wide per-station tables written
by eval_metrics.py -- so no parquet engine is needed and it runs in seconds on a
login node.  One box = the distribution of per-station ubRMSE for one
(split, depth) cell; one dot = one station.

Figure (PNG + PDF, dpi 300, house style §13.3):
    box_ubrmse_by_depth   depth bins on x, splits side by side within each bin

Usage:
    python plot_eval_boxplot.py [--in-dir eval_output] [--out-dir figures/eval]
                                [--splits oos oot oost] [--metric ubRMSE]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

try:
    import scienceplots        # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})
# the login nodes have texlive but not sfmath.sty, so render maths with mathtext
plt.rcParams["text.usetex"] = False

# §13.3 house style -- kept identical to plot_eval_scatter.py
SM_DEPTHS    = ["0-10", "10-30", "30-100"]
DEPTH_COLS   = {"0-10": "0_10", "10-30": "10_30", "30-100": "30_100"}
DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
SPLIT_COLORS = {"oos": "#1a6faf", "oot": "#e8851a", "oost": "#9b59b6", "val": "#7f8c8d"}
SPLIT_LABELS = {"oos": "OOS (novel stations, 2016-2022)",
                "oot": "OOT (seen stations, 2023)",
                "oost": "OOST (novel stations, 2023)",
                "val": "val (internal)"}
METRIC_LABELS = {"ubRMSE": "per-station ubRMSE (m$^3$/m$^3$)",
                 "RMSE":   "per-station RMSE (m$^3$/m$^3$)",
                 "MAE":    "per-station MAE (m$^3$/m$^3$)",
                 "bias":   "per-station bias (m$^3$/m$^3$)"}


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir/name}.png")


def load_long(in_dir: Path, splits: list, metric: str) -> pd.DataFrame:
    """Wide per-station tables -> long frame [split, station_key, depth, value, n]."""
    rows = []
    for split in splits:
        p = in_dir / f"per_station_{split}.csv"
        if not p.exists():
            print(f"  (skipping {split} -- 1{p} not found)")
            continue
        df = pd.read_csv(p)
        for depth, suf in DEPTH_COLS.items():
            col = f"{metric}_{suf}"
            if col not in df:
                continue
            g = df[["station_key", col, f"n_{suf}"]].rename(
                columns={col: "value", f"n_{suf}": "n_obs"})
            g = g.dropna(subset=["value"])
            g["split"] = split
            g["depth"] = depth
            rows.append(g)
    if not rows:
        raise SystemExit(f"No per-station CSVs with metric '{metric}' in {in_dir}")
    return pd.concat(rows, ignore_index=True)


def fig_box_by_depth(long: pd.DataFrame, metric: str, out_dir: Path,
                     splits: list, show_points: bool = True):
    splits = [s for s in splits if s in long["split"].unique()]
    depths = [d for d in SM_DEPTHS if d in long["depth"].unique()]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    rng   = np.random.default_rng(0)
    span  = 0.88                                   # width of a depth group
    width = span / max(len(splits), 1)

    for k, split in enumerate(splits):
        offset = (k - (len(splits) - 1) / 2) * width
        data, pos = [], []
        for i, depth in enumerate(depths):
            v = long[(long["split"] == split) &
                     (long["depth"] == depth)]["value"].to_numpy()
            data.append(v)
            pos.append(i + offset)
            if show_points and len(v):
                x = pos[-1] + rng.uniform(-width * 0.22, width * 0.22, len(v))
                ax.scatter(x, v, s=5, alpha=0.35, c=SPLIT_COLORS[split],
                           edgecolors="none", zorder=2)

        bp = ax.boxplot(data, positions=pos, widths=width * 0.62,
                        showfliers=False, patch_artist=True, zorder=3,
                        medianprops=dict(color="k", lw=1.2),
                        boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7),
                        capprops=dict(lw=0.7))
        for patch in bp["boxes"]:
            patch.set_facecolor(SPLIT_COLORS[split])
            patch.set_alpha(0.55)
            patch.set_edgecolor("k")

        # median on a clean row at the top, station count below the axis
        for v, x in zip(data, pos):
            if not len(v):
                continue
            ax.annotate(f"{np.median(v):.3f}", xy=(x, 0.985),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=5.5,
                        color=SPLIT_COLORS[split])
            ax.annotate(f"n={len(v)}", xy=(x, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top", fontsize=5.5, color="grey")

    ax.annotate("median", xy=(0, 0.985), xycoords=("axes fraction", "axes fraction"),
                xytext=(-4, 0), textcoords="offset points",
                ha="right", va="top", fontsize=5.5, color="grey")
    for i in range(len(depths) - 1):                # separate the depth groups
        ax.axvline(i + 0.5, color="grey", lw=0.5, ls=":", zorder=1)

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([DEPTH_LABELS[d] for d in depths])
    for tick, d in zip(ax.get_xticklabels(), depths):
        tick.set_color(DEPTH_COLORS[d])
    ax.set_xlim(-0.55, len(depths) - 0.45)
    ax.set_ylim(0, float(long["value"].max()) * 1.16)   # headroom for the median row
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(fc=SPLIT_COLORS[s], ec="k", lw=0.5, alpha=0.55,
                             label=SPLIT_LABELS.get(s, s.upper()))
                       for s in splits],
              fontsize=6, frameon=False, loc="upper left",
              bbox_to_anchor=(0.005, 0.955), ncol=1)
    ax.set_title(f"Per-station {metric} by depth and held-out split "
                 "(one dot = one station)", fontsize=9)
    save(fig, out_dir, f"box_{metric.lower()}_by_depth")


def print_table(long: pd.DataFrame, metric: str, splits: list):
    print(f"\n{metric}: median [IQR] across stations")
    print(f"{'split':6s} {'depth':8s} {'n':>5s} {'median':>8s} {'q1':>8s} "
          f"{'q3':>8s} {'mean':>8s}")
    for split in splits:
        for depth in SM_DEPTHS:
            v = long[(long["split"] == split) &
                     (long["depth"] == depth)]["value"].to_numpy()
            if not len(v):
                continue
            print(f"{split:6s} {depth:8s} {len(v):5d} {np.median(v):8.4f} "
                  f"{np.percentile(v, 25):8.4f} {np.percentile(v, 75):8.4f} "
                  f"{v.mean():8.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default="eval_output")
    p.add_argument("--out-dir", default="figures/eval")
    p.add_argument("--splits",  nargs="+", default=["oos", "oot", "oost"],
                   help="splits to plot, in order (add 'val' for the internal split)")
    p.add_argument("--metric",  default="ubRMSE",
                   choices=["ubRMSE", "RMSE", "MAE", "bias"])
    p.add_argument("--no-points", action="store_true",
                   help="boxes only, no per-station dots")
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    long = load_long(in_dir, args.splits, args.metric)
    print(f"Loaded {len(long)} station-depth cells from "
          f"{long['split'].nunique()} splits")

    fig_box_by_depth(long, args.metric, out_dir, args.splits,
                     show_points=not args.no_points)
    print_table(long, args.metric, args.splits)
    long.to_csv(out_dir / f"box_{args.metric.lower()}_by_depth.csv", index=False)
    print(f"\nFigure + backing data in {out_dir}/")


if __name__ == "__main__":
    main()
