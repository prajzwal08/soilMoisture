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
SPLIT_HATCH  = {}          # filled by plot_style_bw.apply under --style bw
BW           = False
FS           = 1.0         # annotation font-size multiplier (paper style raises it)
DPI          = 300
PAPER        = False       # paper style: no in-figure titles/descriptions (the caption carries them)
CS           = 1.0         # extra multiplier for count / median annotations (paper style)
BOX_ALPHA    = 0.55        # box fill opacity (bw + paper: 1.0, solid)
XROT         = None        # category tick rotation override (paper style: 90)
SPLIT_LABELS = {"oos": "OOS (novel stations, 2016-2022)",
                "oot": "OOT (seen stations, 2023-2025)",
                "oost": "OOST (novel stations, 2023-2025)",
                "val": "val (internal)"}
METRIC_LABELS = {"ubRMSE": "per-station ubRMSE (m$^3$/m$^3$)",
                 "RMSE":   "per-station RMSE (m$^3$/m$^3$)",
                 "MAE":    "per-station MAE (m$^3$/m$^3$)",
                 "bias":   "per-station bias (m$^3$/m$^3$)",
                 "R2_pearson": "per-station r$^2$",
                 "R":      "per-station r",
                 "NSE":    "per-station NSE",
                 "NSE_anom": "per-station NSE (offset removed)"}


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
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

    whisk_lo, whisk_hi = [], []                    # drawn whisker ends, for the y limits
    med_fs = 5.5 * FS * CS
    if PAPER:
        med_fs = max(med_fs, 10.0)
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
            patch.set_hatch(SPLIT_HATCH.get(split, ""))
            patch.set_alpha(BOX_ALPHA)
            patch.set_edgecolor("k")
        for w in bp["whiskers"]:
            y = np.asarray(w.get_ydata(), dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                whisk_lo.append(y.min())
                whisk_hi.append(y.max())

        # median on a clean row at the top (paper: just ABOVE the axes, so it can never
        # touch a y tick label), station count below the axis
        for v, x in zip(data, pos):
            if not len(v):
                continue
            if PAPER:
                ax.annotate(f"{np.median(v):.3f}", xy=(x, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=med_fs,
                            color="black", annotation_clip=False, clip_on=False)
            else:
                ax.annotate(f"{np.median(v):.3f}", xy=(x, 0.985),
                            xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=5.5 * FS * CS,
                            color="black" if BW else SPLIT_COLORS[split])
            ax.annotate(f"{len(v)}" if PAPER else f"n={len(v)}", xy=(x, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -14 * CS), textcoords="offset points",
                        ha="center", va="top", fontsize=med_fs if PAPER else 5.5 * FS * CS,
                        color="black" if PAPER else "grey")

    if not PAPER:                                   # paper: numbers only, no "median" word
        ax.annotate("median", xy=(0, 0.985), xycoords=("axes fraction", "axes fraction"),
                    xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="top", fontsize=5.5 * FS * CS, color="grey")
    for i in range(len(depths) - 1):                # separate the depth groups
        ax.axvline(i + 0.5, color="grey", lw=0.5, ls=":", zorder=1)

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([DEPTH_LABELS[d] for d in depths])
    for tick, d in zip(ax.get_xticklabels(), depths):
        tick.set_color(DEPTH_COLORS[d])
    ax.set_xlim(-0.55, len(depths) - 0.45)
    # y limits from the data, not from 0 / the max: negative bias must show, and one
    # outlier station must not squash the boxes (dots beyond the 99.5th pct are clipped)
    vals = long.loc[long["split"].isin(splits) & long["depth"].isin(depths), "value"]
    vals = vals.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    lo = min(float(np.percentile(vals, 0.5)), 0.0, min(whisk_lo, default=0.0))
    hi = max(float(np.quantile(vals, 0.995)), max(whisk_hi, default=0.0))
    rng_y = max(hi - lo, 1e-6)
    y0 = lo - 0.04 * rng_y if lo < 0 else 0.0
    # colour/bw: the median row sits INSIDE the axes, so it needs ~16 % headroom;
    # paper: the median row sits above the axes, so only a little air above the whiskers
    y1 = hi + (0.06 if PAPER else 0.16) * rng_y
    ax.set_ylim(y0, y1)
    if lo < 0:                                      # bias: zero reference line
        ax.axhline(0, color="k", lw=0.8, zorder=1.5)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    handles = [Patch(fc=SPLIT_COLORS[s], ec="k", lw=0.5, alpha=BOX_ALPHA,
                     hatch=SPLIT_HATCH.get(s, ""),
                     label=SPLIT_LABELS.get(s, s.upper()))
               for s in splits]
    if PAPER:
        # legend OUTSIDE, above the median row: the pad (in legend-font units) clears
        # the 3 pt offset + one line of median text + 3 pt of air
        leg_fs = 6 * FS
        pad = (3 + 1.25 * med_fs + 3) / leg_fs
        ax.legend(handles=handles, fontsize=leg_fs, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  borderaxespad=pad, ncol=min(len(splits), 3))
    else:
        ax.legend(handles=handles, fontsize=6 * FS, frameon=False, loc="upper left",
                  bbox_to_anchor=(0.005, 0.955), ncol=1)
    if not PAPER:
        ax.set_title(f"Per-station {metric} by depth and held-out split "
                     "(one dot = one station)", fontsize=9 * FS)
    save(fig, out_dir, f"box_{metric.lower()}_by_depth")


# ── raincloud (§74, 2026-10-13) ─────────────────────────────────────────────────
# A box hides WHERE the stations sit (bunched near the ideal with a tail, or evenly
# spread). Raincloud = half-violin (density) left of centre + narrow box + one dot per
# station right of centre, with a dashed line at the metric's ideal value. Skill scores
# that run to -inf (NSE) get a fixed axis floor; stations below it are counted ("↓n")
# at the bottom edge instead of squashing the plot.
IDEAL  = {"R2_pearson": 1.0, "R": 1.0, "NSE": 1.0, "NSE_anom": 1.0,
          "ubRMSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "bias": 0.0}
YFLOOR = {"NSE": -1.0, "NSE_anom": -1.0}          # axis floor; values below are counted
YLIM   = {"R2_pearson": (0.0, 1.0), "R": (-0.2, 1.0),
          "NSE": (-1.0, 1.0), "NSE_anom": (-1.0, 1.0)}


def fig_raincloud_by_depth(long: pd.DataFrame, metric: str, out_dir: Path, splits: list):
    splits = [s for s in splits if s in long["split"].unique()]
    depths = [d for d in SM_DEPTHS if d in long["depth"].unique()]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    rng   = np.random.default_rng(0)
    span  = 0.88
    width = span / max(len(splits), 1)
    floor = YFLOOR.get(metric)
    med_fs = max(5.5 * FS * CS, 10.0) if PAPER else 5.5 * FS * CS

    if metric in YLIM:
        y0, y1 = YLIM[metric]
    else:                                           # error metrics: 0 .. 99.5th pct
        v_all = long.loc[long["split"].isin(splits) & long["depth"].isin(depths), "value"]
        v_all = v_all.to_numpy(dtype=float)
        v_all = v_all[np.isfinite(v_all)]
        y0 = min(0.0, float(np.percentile(v_all, 0.5)))
        y1 = float(np.quantile(v_all, 0.995)) * 1.05
    pad_lo = 0.06 * (y1 - y0) if floor is not None else 0.0   # room for the ↓n marker

    for k, split in enumerate(splits):
        offset = (k - (len(splits) - 1) / 2) * width
        col = SPLIT_COLORS[split]
        for i, depth in enumerate(depths):
            v = long[(long["split"] == split) & (long["depth"] == depth)]["value"]
            v = v.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            x = i + offset
            shown = v[v >= floor] if floor is not None else v
            n_below = len(v) - len(shown)

            # half violin on the left (density of the stations that are on the axis)
            if len(shown) >= 3 and np.ptp(shown) > 0:
                vp = ax.violinplot([shown], positions=[x], widths=width * 0.9,
                                   showmeans=False, showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    verts = body.get_paths()[0].vertices
                    verts[:, 0] = np.minimum(verts[:, 0], x)
                    body.set_facecolor(col)
                    body.set_edgecolor("k")
                    body.set_linewidth(0.4)
                    body.set_alpha(0.45 if not PAPER else 0.6)
                    body.set_zorder(2)
            # narrow box at the centre, from ALL stations (quantiles are not clipped)
            bp = ax.boxplot([v], positions=[x], widths=width * 0.12, showfliers=False,
                            patch_artist=True, zorder=4,
                            medianprops=dict(color="k", lw=1.2),
                            boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                            capprops=dict(lw=0))
            for patch in bp["boxes"]:
                patch.set_facecolor("white")
                patch.set_edgecolor("k")
            # dots on the right
            xs = x + width * 0.10 + rng.uniform(0, width * 0.28, len(shown))
            ax.scatter(xs, shown, s=5, alpha=0.5, c=col, edgecolors="none", zorder=3)
            if n_below:
                ax.annotate(f"↓{n_below}", xy=(x, y0 - pad_lo * 0.5),
                            ha="center", va="center", fontsize=med_fs * 0.85,
                            color="black", annotation_clip=False)
            ax.annotate(f"{np.median(v):.2f}" if metric in YLIM else f"{np.median(v):.3f}",
                        xy=(x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=med_fs, color="black",
                        annotation_clip=False)
            ax.annotate(f"{len(v)}", xy=(x, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -14 * CS), textcoords="offset points", ha="center",
                        va="top", fontsize=med_fs, color="black", annotation_clip=False)

    for i in range(len(depths) - 1):
        ax.axvline(i + 0.5, color="grey", lw=0.5, ls=":", zorder=1)
    ax.axhline(IDEAL.get(metric, 0.0), color="k", lw=0.8, ls="--", zorder=1.5)
    if floor is not None:
        ax.axhline(floor, color="grey", lw=0.6, zorder=1.5)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([DEPTH_LABELS[d] for d in depths])
    for tick, d in zip(ax.get_xticklabels(), depths):
        tick.set_color(DEPTH_COLORS[d])
    ax.set_xlim(-0.55, len(depths) - 0.45)
    ax.set_ylim(y0 - pad_lo, y1 + 0.03 * (y1 - y0))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    handles = [Patch(fc=SPLIT_COLORS[s], ec="k", lw=0.5, alpha=0.6,
                     label=SPLIT_LABELS.get(s, s.upper())) for s in splits]
    leg_fs = 6 * FS
    pad = (3 + 1.25 * med_fs + 3) / leg_fs
    ax.legend(handles=handles, fontsize=leg_fs, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=pad, ncol=min(len(splits), 3))
    if not PAPER:
        ax.set_title(f"Per-station {metric} by depth (density | box | one dot = one station)",
                     fontsize=9 * FS)
    save(fig, out_dir, f"raincloud_{metric.lower()}_by_depth")


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
                   choices=["ubRMSE", "RMSE", "MAE", "bias", "R2_pearson", "R", "NSE", "NSE_anom"])
    p.add_argument("--kind", choices=["box", "raincloud"], default="box",
                   help="raincloud (§74) = half violin + narrow box + station dots, ideal line")
    p.add_argument("--no-points", action="store_true",
                   help="boxes only, no per-station dots")
    p.add_argument("--style", choices=["color", "bw", "paper"], default="color",
                   help="bw = black-and-white; paper = blue/green/red, Times (plot_style_bw.py)")
    args = p.parse_args()
    if args.style != "color":
        import plot_style_bw
        plot_style_bw.apply(globals(), args.style)

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    long = load_long(in_dir, args.splits, args.metric)
    print(f"Loaded {len(long)} station-depth cells from "
          f"{long['split'].nunique()} splits")

    if args.kind == "raincloud":
        fig_raincloud_by_depth(long, args.metric, out_dir, args.splits)
    else:
        fig_box_by_depth(long, args.metric, out_dir, args.splits,
                         show_points=not args.no_points)
    print_table(long, args.metric, args.splits)
    long.to_csv(out_dir / f"{args.kind}_{args.metric.lower()}_by_depth.csv", index=False)
    print(f"\nFigure + backing data in {out_dir}/")


if __name__ == "__main__":
    main()
