"""Station inventory and ubRMSE by ecosystem / climate class (§22.11).

Two questions, two figures:

    stations_by_{class}   how many stations of each class are in train vs each
                          evaluation split, as counts and as within-split
                          fractions.  The fraction panel is the important one:
                          it shows whether the held-out pools are composed like
                          the training pool, which conditions every §22 result.

    box_ubrmse_by_{class} per-station ubRMSE by class, one panel per depth,
                          boxes grouped by split.

Both read CSVs only -- csvs/station_splits.csv for the inventory (which is the
only place train stations appear at all) and eval_output/per_station_{split}.csv
for the metrics -- so no parquet engine and no GPU is needed.

Classes with fewer than --min-stations members are dropped from the boxplot and
LOGGED; a silently truncated panel reads as "we covered everything".

Usage:
    python plot_eval_ecosystem.py [--by igbp_macro] [--out-dir figures/eval]
    python plot_eval_ecosystem.py --by kg_macro
    python plot_eval_ecosystem.py --by IGBP --min-stations 8
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
plt.rcParams["text.usetex"] = False     # sfmath.sty is absent on the login nodes

# §13.3 house style
SM_DEPTHS    = ["0-10", "10-30", "30-100"]
DEPTH_COLS   = {"0-10": "0_10", "10-30": "10_30", "30-100": "30_100"}
DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
SPLIT_COLORS = {"train": "#34495e", "val": "#7f8c8d",
                "oos": "#1a6faf", "oot": "#e8851a", "oost": "#9b59b6"}
EVAL_SPLITS  = ["oos", "oot", "oost"]
SPLITS_CSV   = Path("csvs/station_splits.csv")


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir/name}.png")


def load_inventory(by: str) -> pd.DataFrame:
    """Long frame [split, class, n].

    OOT and OOST are not values of the `split` column -- they are 2023 windows
    over stations flagged oot_eligible / oost_eligible -- so they are counted
    from those flags, not from `split`.
    """
    d = pd.read_csv(SPLITS_CSV)
    if by not in d.columns:
        raise SystemExit(f"'{by}' not in {SPLITS_CSV}; have: {sorted(d.columns)}")
    d[by] = d[by].fillna("unknown")

    rows = []
    for split in ["train", "val", "oos"]:
        g = d[d["split"] == split]
        rows += [{"split": split, "class": k, "n": v}
                 for k, v in g[by].value_counts().items()]
    for split, flag in (("oot", "oot_eligible"), ("oost", "oost_eligible")):
        if flag not in d.columns:
            print(f"  ({flag} missing -- {split} omitted from the inventory)")
            continue
        g = d[d[flag].astype(str).str.lower().isin(["true", "1", "yes"])]
        rows += [{"split": split, "class": k, "n": v}
                 for k, v in g[by].value_counts().items()]
    return pd.DataFrame(rows)


def load_metrics(in_dir: Path, by: str, metric: str) -> pd.DataFrame:
    """Long frame [split, station_key, class, depth, value]."""
    rows = []
    for split in EVAL_SPLITS:
        p = in_dir / f"per_station_{split}.csv"
        if not p.exists():
            print(f"  ({p.name} not found -- {split} skipped)")
            continue
        d = pd.read_csv(p)
        if by not in d.columns:
            raise SystemExit(f"'{by}' not in {p}; have: {sorted(d.columns)}")
        d[by] = d[by].fillna("unknown")
        for depth, suf in DEPTH_COLS.items():
            col = f"{metric}_{suf}"
            if col not in d:
                continue
            g = d[["station_key", by, col]].rename(columns={by: "class",
                                                           col: "value"})
            g = g.dropna(subset=["value"])
            g["split"], g["depth"] = split, depth
            rows.append(g)
    if not rows:
        raise SystemExit(f"no per-station CSVs with '{metric}' in {in_dir}")
    return pd.concat(rows, ignore_index=True)


# ── 1. inventory ──────────────────────────────────────────────────────────────

def fig_inventory(inv: pd.DataFrame, by: str, out_dir: Path):
    splits = [s for s in ["train", "val", "oos", "oot", "oost"]
              if s in inv["split"].unique()]
    order  = (inv[inv["split"] == "train"].set_index("class")["n"]
              .sort_values(ascending=False).index.tolist())
    order += [c for c in inv["class"].unique() if c not in order]

    wide  = inv.pivot(index="class", columns="split", values="n").reindex(order)
    wide  = wide.reindex(columns=splits).fillna(0)
    frac  = wide / wide.sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    width = 0.8 / len(splits)

    for k, split in enumerate(splits):
        offset = (k - (len(splits) - 1) / 2) * width
        x = np.arange(len(order)) + offset
        axes[0].bar(x, wide[split], width=width * 0.9,
                    color=SPLIT_COLORS[split], label=f"{split} (n={int(wide[split].sum())})",
                    edgecolor="k", lw=0.4)
        for xi, v in zip(x, wide[split]):
            if v:
                axes[0].annotate(f"{int(v)}", (xi, v), ha="center", va="bottom",
                                 fontsize=5, rotation=90, xytext=(0, 1),
                                 textcoords="offset points")
        axes[1].bar(x, frac[split] * 100, width=width * 0.9,
                    color=SPLIT_COLORS[split], edgecolor="k", lw=0.4)

    for ax, ylab, title in ((axes[0], "stations", "counts"),
                            (axes[1], "share of split (\\%)"
                             if plt.rcParams["text.usetex"] else "share of split (%)",
                             "composition")):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6, frameon=False)

    fig.suptitle(f"Station inventory by {by} -- train vs held-out splits", y=1.04)
    save(fig, out_dir, f"stations_by_{by}")
    return wide


# ── 2. ubRMSE by class ────────────────────────────────────────────────────────

def fig_box_by_class(long: pd.DataFrame, by: str, metric: str, out_dir: Path,
                     min_stations: int):
    keep = (long.groupby("class")["station_key"].nunique()
            .pipe(lambda s: s[s >= min_stations]).index.tolist())
    dropped = sorted(set(long["class"]) - set(keep))
    if dropped:
        counts = long[long["class"].isin(dropped)].groupby("class")["station_key"].nunique()
        print(f"  dropped (< {min_stations} stations): "
              + ", ".join(f"{c} (n={counts[c]})" for c in dropped))
    long = long[long["class"].isin(keep)]
    order = (long.groupby("class")["station_key"].nunique()
             .sort_values(ascending=False).index.tolist())
    splits = [s for s in EVAL_SPLITS if s in long["split"].unique()]

    fig, axes = plt.subplots(len(SM_DEPTHS), 1, figsize=(8.0, 8.4),
                             sharex=True, constrained_layout=True)
    rng   = np.random.default_rng(0)
    span  = 0.84
    width = span / max(len(splits), 1)

    for ax, depth in zip(axes, SM_DEPTHS):
        for k, split in enumerate(splits):
            offset = (k - (len(splits) - 1) / 2) * width
            data, pos = [], []
            for i, cls in enumerate(order):
                v = long[(long["split"] == split) & (long["depth"] == depth) &
                         (long["class"] == cls)]["value"].to_numpy()
                data.append(v)
                pos.append(i + offset)
                if len(v):
                    x = pos[-1] + rng.uniform(-width * 0.2, width * 0.2, len(v))
                    ax.scatter(x, v, s=5, alpha=0.4, c=SPLIT_COLORS[split],
                               edgecolors="none", zorder=2)
                ax.annotate(f"{len(v)}", xy=(pos[-1], 0.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, -9), textcoords="offset points",
                            ha="center", va="top", fontsize=5,
                            color=SPLIT_COLORS[split])
            bp = ax.boxplot([d if len(d) else [np.nan] for d in data],
                            positions=pos, widths=width * 0.62, showfliers=False,
                            patch_artist=True, zorder=3,
                            medianprops=dict(color="k", lw=1.1),
                            boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                            capprops=dict(lw=0.6))
            for patch in bp["boxes"]:
                patch.set_facecolor(SPLIT_COLORS[split])
                patch.set_alpha(0.55)
                patch.set_edgecolor("k")

        ax.set_ylabel(f"{DEPTH_LABELS[depth]}\n{metric} (m$^3$/m$^3$)",
                      color=DEPTH_COLORS[depth])
        ax.set_ylim(0, float(long["value"].quantile(0.99)) * 1.1)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].legend(handles=[Patch(fc=SPLIT_COLORS[s], ec="k", lw=0.5, alpha=0.55,
                                  label=s.upper()) for s in splits],
                   fontsize=6, frameon=False, loc="upper right", ncol=len(splits))
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels(order, rotation=15, ha="right")
    axes[-1].set_xlabel(f"{by}   (small numbers = stations per box)")
    fig.suptitle(f"Per-station {metric} by {by} and split", y=1.02)
    save(fig, out_dir, f"box_{metric.lower()}_by_{by}")


def print_table(long: pd.DataFrame, by: str, metric: str):
    print(f"\n{metric} median [IQR] by {by}")
    print(f"{'class':16s} {'split':5s} {'depth':7s} {'n':>4s} {'median':>8s} "
          f"{'q1':>8s} {'q3':>8s}")
    for cls, g0 in long.groupby("class"):
        for split in EVAL_SPLITS:
            for depth in SM_DEPTHS:
                v = g0[(g0["split"] == split) &
                       (g0["depth"] == depth)]["value"].to_numpy()
                if len(v) < 3:
                    continue
                print(f"{cls:16s} {split:5s} {depth:7s} {len(v):4d} "
                      f"{np.median(v):8.4f} {np.percentile(v, 25):8.4f} "
                      f"{np.percentile(v, 75):8.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default="eval_output")
    p.add_argument("--out-dir", default="figures/eval")
    p.add_argument("--by",      default="igbp_macro",
                   help="igbp_macro | IGBP | kg_macro | koppen_geiger | "
                        "elevation_band")
    p.add_argument("--metric",  default="ubRMSE",
                   choices=["ubRMSE", "RMSE", "MAE", "bias"])
    p.add_argument("--min-stations", type=int, default=5)
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    inv = load_inventory(args.by)
    wide = fig_inventory(inv, args.by, out_dir)
    print(f"\nstations per {args.by} and split\n{wide.astype(int)}")

    long = load_metrics(in_dir, args.by, args.metric)
    fig_box_by_class(long, args.by, args.metric, out_dir, args.min_stations)
    print_table(long, args.by, args.metric)

    long.to_csv(out_dir / f"box_{args.metric.lower()}_by_{args.by}.csv", index=False)
    wide.to_csv(out_dir / f"stations_by_{args.by}.csv")
    print(f"\nFigures + backing data in {out_dir}/")


if __name__ == "__main__":
    main()
