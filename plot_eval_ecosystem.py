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
SPLIT_HATCH  = {}          # filled by plot_style_bw.apply under --style bw
BW           = False
FS           = 1.0         # annotation font-size multiplier (paper style raises it)
DPI          = 300
PAPER        = False       # paper style: no in-figure titles/descriptions (the caption carries them)
CS           = 1.0         # extra multiplier for count / median annotations (paper style)
BOX_ALPHA    = 0.55        # box fill opacity (bw + paper: 1.0, solid)
XROT         = None        # category tick rotation override (paper style: 90)
EVAL_SPLITS  = ["oos", "oot", "oost"]
INVENTORY_SPLITS = ["train", "val", "oos", "oot", "oost"]   # --no-val drops "val"
SPLITS_CSV   = Path("csvs/station_splits.csv")


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir/name}.png")


def load_inventory(by: str, in_dir: Path = None) -> pd.DataFrame:
    """Long frame [split, class, n] -- SOIL-MOISTURE stations only.

    station_splits.csv also lists flux_only stations, which carry no SM target and are
    never evaluated (§47.7), so rows are kept only when has_soil_moisture is true AND
    splits_config.category_of(row) is in SM_CATEGORIES.

    Held-out splits (oos / oot / oost) are counted from <in_dir>/per_station_{split}.csv
    when it exists -- the stations actually evaluated -- and otherwise fall back to the
    splits CSV.  OOT and OOST are not values of the `split` column -- they are 2023
    windows over stations flagged oot_eligible / oost_eligible -- so the fallback counts
    them from those flags.
    """
    from splits_config import SM_CATEGORIES, category_of

    d = pd.read_csv(SPLITS_CSV)
    if by not in d.columns:
        raise SystemExit(f"'{by}' not in {SPLITS_CSV}; have: {sorted(d.columns)}")
    n_all = len(d)
    has_sm = d["has_soil_moisture"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    d = d[has_sm]
    d = d[d.apply(category_of, axis=1).isin(SM_CATEGORIES)].copy()
    print(f"  inventory: {len(d)} soil-moisture stations of {n_all} rows in {SPLITS_CSV}")
    d[by] = d[by].fillna("unknown")

    def _evaluated(split):
        if in_dir is None:
            return None
        p = Path(in_dir) / f"per_station_{split}.csv"
        if not p.exists():
            return None
        e = pd.read_csv(p)
        if by not in e.columns:
            print(f"  ('{by}' not in {p.name} -- {split} counted from {SPLITS_CSV})")
            return None
        e = e.drop_duplicates("station_key") if "station_key" in e.columns else e
        print(f"  inventory: {split} counted from {p} ({len(e)} evaluated stations)")
        return e[by].fillna("unknown")

    rows = []
    for split in ["train", "val", "oos"]:
        cls = _evaluated(split) if split == "oos" else None
        if cls is None:
            cls = d.loc[d["split"] == split, by]
        rows += [{"split": split, "class": k, "n": v}
                 for k, v in cls.value_counts().items()]
    for split, flag in (("oot", "oot_eligible"), ("oost", "oost_eligible")):
        cls = _evaluated(split)
        if cls is None:
            if flag not in d.columns:
                print(f"  ({flag} missing -- {split} omitted from the inventory)")
                continue
            cls = d.loc[d[flag].astype(str).str.strip().str.lower()
                        .isin(["true", "1", "yes"]), by]
        rows += [{"split": split, "class": k, "n": v}
                 for k, v in cls.value_counts().items()]
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


def _rotation(labels, default):
    """Category tick rotation: the colour style keeps `default`; the paper style lays short
    label sets flat and turns crowded ones to 90 degrees.  "Crowded" is about the total
    printed length, not the class count: seven 3-letter IGBP codes stay flat."""
    if not PAPER:
        return default
    lens = [max((len(p) for p in str(s).split("\n")), default=0) for s in labels]
    crowded = (len(labels) > 5 and sum(lens) > 45) or max(lens, default=0) > 14   # <= 5 classes stay flat
    return 90 if crowded else 0


# ordinal / conventional class orders; anything not listed follows in count order
CLASS_ORDER = {
    "elevation_band": ["Low", "Mid", "High"],
    "kg_macro":       ["A", "B", "C", "D", "E"],
}
KG_GLOSS = {"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Continental", "E": "Polar"}
MIN_SLOT = 3                # no box, dots or count for a (class, split, depth) slot below this
TOP_N_NETWORKS = 12         # inventory --by network: top-N by train count + "Other"


def _order_classes(by: str, count_order: list) -> list:
    """`count_order` (descending count) re-sorted into the ordinal order for `by`, if any."""
    fixed = CLASS_ORDER.get(by)
    if not fixed:
        return list(count_order)
    head = [c for c in fixed if c in count_order]
    return head + [c for c in count_order if c not in head]


def _class_labels(by: str, order: list, default_rot):
    """Tick labels + rotation.  Paper style glosses the Koppen letters (C -> C Temperate)."""
    labels = [str(c) for c in order]
    if PAPER and by == "kg_macro":
        labels = [f"{c} {KG_GLOSS[c]}" if c in KG_GLOSS else str(c) for c in order]
    rot = _rotation(labels, default_rot)
    if PAPER and by == "kg_macro" and rot == 0:          # stacked when flat: letter over gloss
        labels = [s.replace(" ", "\n", 1) for s in labels]
    return labels, rot


def _count_fs(n_classes: int) -> float:
    """Count-annotation font size: the colour style keeps 5*FS*CS; paper keeps it >= 10 pt
    when there is room (<= 8 classes)."""
    fs = 5 * FS * CS
    if PAPER and n_classes <= 8:
        fs = max(fs, 10.0)
    return fs


# ── 1. inventory ──────────────────────────────────────────────────────────────

def fig_inventory(inv: pd.DataFrame, by: str, out_dir: Path):
    splits = [s for s in INVENTORY_SPLITS
              if s in inv["split"].unique()]
    if by == "network":                 # legibility: top-N networks by train count + "Other"
        train_n = (inv[inv["split"] == "train"].set_index("class")["n"]
                   .sort_values(ascending=False))
        top = train_n.index[:TOP_N_NETWORKS].tolist()
        if inv["class"].nunique() > len(top):
            inv = inv.assign(cls=np.where(inv["class"].isin(top), inv["class"], "Other"))
            inv = (inv.groupby(["split", "cls"], as_index=False)["n"].sum()
                   .rename(columns={"cls": "class"}))
    order  = (inv[inv["split"] == "train"].set_index("class")["n"]
              .sort_values(ascending=False).index.tolist())
    order += [c for c in inv["class"].unique() if c not in order]
    if "Other" in order and by == "network":          # the bucket goes last
        order = [c for c in order if c != "Other"] + ["Other"]
    order = _order_classes(by, order)

    wide  = inv.pivot(index="class", columns="split", values="n").reindex(order)
    wide  = wide.reindex(columns=splits).fillna(0)
    frac  = wide / wide.sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    width = 0.8 / len(splits)
    # paper: per-bar counts only while they stay readable (<= 6 classes)
    show_counts = (not PAPER) or len(order) <= 6
    cfs = _count_fs(len(order))

    for k, split in enumerate(splits):
        offset = (k - (len(splits) - 1) / 2) * width
        x = np.arange(len(order)) + offset
        axes[0].bar(x, wide[split], width=width * 0.9,
                    color=SPLIT_COLORS[split],
                    label=f"{split.upper()} (n={int(wide[split].sum())})",
                    edgecolor="k", lw=0.4, hatch=SPLIT_HATCH.get(split, ""))
        if show_counts:
            for xi, v in zip(x, wide[split]):
                if v:
                    axes[0].annotate(f"{int(v)}", (xi, v), ha="center", va="bottom",
                                     fontsize=cfs, rotation=90, xytext=(0, 1),
                                     textcoords="offset points")
        axes[1].bar(x, frac[split] * 100, width=width * 0.9,
                    color=SPLIT_COLORS[split], edgecolor="k", lw=0.4,
                    hatch=SPLIT_HATCH.get(split, ""))

    labels, rot = _class_labels(by, order, 20)
    for ax, ylab, title, letter in ((axes[0], "stations", "counts", "(a)"),
                                    (axes[1], "share of split (\\%)"
                                     if plt.rcParams["text.usetex"] else "share of split (%)",
                                     "composition", "(b)")):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=rot, ha="center" if rot in (0, 90) else "right",
                           fontsize=7 * FS)
        ax.set_ylabel(ylab)
        if PAPER:                       # panel letter instead of a title, just above the corner
            ax.text(0.0, 1.01, letter, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8 * FS, fontweight="bold")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)   # headroom: counts clear the top spine
        else:
            ax.set_title(title, fontsize=8 * FS)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6 * FS, frameon=False)

    if not PAPER:
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
    splits = [s for s in EVAL_SPLITS if s in long["split"].unique()]

    # station count per (class, split, depth) slot; a slot below MIN_SLOT gets no box,
    # no dots and no count, and a class with no drawable slot at all is dropped
    slot_n = long.groupby(["class", "split", "depth"])["station_key"].nunique()
    drawable = slot_n[slot_n >= MIN_SLOT].reset_index()["class"].unique().tolist()
    empty = sorted(set(keep) - set(drawable))
    if empty:
        print(f"  dropped (no split/depth slot with >= {MIN_SLOT} stations): "
              + ", ".join(map(str, empty)))
    long = long[long["class"].isin(drawable)]
    order = (long.groupby("class")["station_key"].nunique()
             .sort_values(ascending=False).index.tolist())
    order = _order_classes(by, order)
    n_cls = len(order)

    many = n_cls > 8                    # crowded: rotate the counts, widen the figure
    fig_w = max(8.0, 0.55 * n_cls * max(len(splits), 1)) if many else 8.0
    fig, axes = plt.subplots(len(SM_DEPTHS), 1, figsize=(fig_w, 8.4),
                             sharex=True, constrained_layout=True)
    rng   = np.random.default_rng(0)
    span  = 0.84
    width = span / max(len(splits), 1)
    cfs   = _count_fs(n_cls)
    count_rot = 90 if many else 0
    max_digits = 1

    for ax, depth in zip(axes, SM_DEPTHS):
        w_lo, w_hi = [], []
        for k, split in enumerate(splits):
            offset = (k - (len(splits) - 1) / 2) * width
            data, pos = [], []
            for i, cls in enumerate(order):
                v = long[(long["split"] == split) & (long["depth"] == depth) &
                         (long["class"] == cls)]["value"].to_numpy()
                if len(v) < MIN_SLOT:           # too thin to summarise: draw nothing
                    continue
                x0 = i + offset
                data.append(v)
                pos.append(x0)
                x = x0 + rng.uniform(-width * 0.2, width * 0.2, len(v))
                ax.scatter(x, v, s=5, alpha=0.4, c=SPLIT_COLORS[split],
                           edgecolors="none", zorder=2)
                max_digits = max(max_digits, len(str(len(v))))
                ax.annotate(f"{len(v)}", xy=(x0, 0.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, -9 * CS), textcoords="offset points",
                            ha="center", va="top", fontsize=cfs, rotation=count_rot,
                            color="black" if (BW or PAPER) else SPLIT_COLORS[split])
            if not data:
                continue
            bp = ax.boxplot(data, positions=pos, widths=width * 0.62, showfliers=False,
                            patch_artist=True, zorder=3,
                            medianprops=dict(color="k", lw=1.1),
                            boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                            capprops=dict(lw=0.6))
            for patch in bp["boxes"]:
                patch.set_facecolor(SPLIT_COLORS[split])
                patch.set_hatch(SPLIT_HATCH.get(split, ""))
                patch.set_alpha(BOX_ALPHA)
                patch.set_edgecolor("k")
            for w in bp["whiskers"]:
                y = np.asarray(w.get_ydata(), dtype=float)
                y = y[np.isfinite(y)]
                if y.size:
                    w_lo.append(y.min())
                    w_hi.append(y.max())

        ax.set_ylabel(f"{DEPTH_LABELS[depth]}\n{metric} (m$^3$/m$^3$)",
                      color=DEPTH_COLORS[depth])
        # limits from the drawn whiskers (+8 %), so no whisker is cut; beyond-whisker
        # dots may be clipped.  Negative values (bias) keep a zero line.
        if w_hi:
            lo, hi = min(min(w_lo), 0.0), max(w_hi)
            if PAPER:                   # also keep the plotted dots in view (99.5th pct), not only whiskers
                vals = long[long["depth"] == depth]["value"].to_numpy(float)
                if vals.size:
                    hi = max(hi, float(np.nanpercentile(vals, 99.5)))
                    lo = min(lo, float(np.nanpercentile(vals, 0.5)), 0.0)
            pad = 0.08 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - pad if lo < 0 else 0.0, hi + pad)
            if lo < 0:
                ax.axhline(0, color="k", lw=0.8, zorder=1.5)
        ax.set_xlim(-0.6, n_cls - 0.4)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)

    leg_kw = (dict(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=_count_fs(len(order)))
              if PAPER else dict(loc="upper right", fontsize=6 * FS))   # paper: legend above, never on data
    axes[0].legend(handles=[Patch(fc=SPLIT_COLORS[s], ec="k", lw=0.5, alpha=BOX_ALPHA,
                                  hatch=SPLIT_HATCH.get(s, ""), label=s.upper()) for s in splits],
                   frameon=False, ncol=len(splits), **leg_kw)
    axes[-1].set_xticks(range(n_cls))
    labels, rot = _class_labels(by, order, 15)
    axes[-1].set_xticklabels(labels, rotation=rot, ha="center" if rot in (0, 90) else "right")
    # clear the per-box station counts drawn just under the axis
    count_h = (0.62 * cfs * max_digits) if count_rot else (1.2 * cfs)
    if PAPER:
        axes[-1].tick_params(axis="x", pad=9 * CS + count_h + 3)
    elif many:
        axes[-1].tick_params(axis="x", pad=9 * CS + count_h + 2)
    if not PAPER:                       # paper style: no description / title (the caption carries it)
        axes[-1].set_xlabel(f"{by}   (small numbers = stations per box, "
                            f"boxes need >= {MIN_SLOT})")
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
    p.add_argument("--no-val", action="store_true",
                   help="leave val out of the inventory figure (held-out splits only, + train)")
    p.add_argument("--style", choices=["color", "bw", "paper"], default="color",
                   help="bw = black-and-white; paper = blue/green/red, Times (plot_style_bw.py)")
    args = p.parse_args()
    if args.style != "color":
        import plot_style_bw
        plot_style_bw.apply(globals(), args.style)

    if args.no_val:
        INVENTORY_SPLITS.remove("val")
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    inv = load_inventory(args.by, in_dir)
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
