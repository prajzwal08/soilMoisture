"""
Publication-quality breakdown figures by IGBP and climate zone.

Figures saved to meeting_output/breakdown/:
  1. igbp_violin.png      — surface ubRMSE by IGBP macro (OOS)
  2. climate_violin.png   — surface ubRMSE by Koppen macro (OOS)
  3. oos_oot_oost_bar.png — grouped bar: OOS vs OOT vs OOST by IGBP macro
  4. depth_bar.png        — depth-wise ubRMSE across all 3 splits
  5. metrics_heatmap.png  — heatmap: split × depth × metric

Requires:
    meeting_output/per_station_oos.csv
    meeting_output/per_station_oot.csv
    meeting_output/per_station_oost.csv

Usage:
    python plot_breakdown_meeting.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9})

MET_DIR = Path("meeting_output")
OUT_DIR = Path("meeting_output/breakdown")

SPLIT_COLORS = {"oos": "#1a6faf", "oot": "#e8851a", "oost": "#9b59b6"}
SPLIT_LABELS = {"oos": "OOS", "oot": "OOT", "oost": "OOST"}
DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_TAGS   = {"0-10": "0_10",   "10-30": "10_30",  "30-100": "30_100"}

IGBP_ORDER    = ["Forest", "Grass-Crop", "Shrub-Savanna", "Other"]
CLIMATE_ORDER = ["A", "B", "C", "D", "E"]
CLIMATE_NAMES = {"A": "Tropical (A)", "B": "Arid (B)",
                 "C": "Temperate (C)", "D": "Continental (D)", "E": "Polar (E)"}


def load_dfs():
    dfs = {}
    for split in ["oos", "oot", "oost"]:
        path = MET_DIR / f"per_station_{split}.csv"
        if path.exists():
            dfs[split] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found — skipping {split}")
    return dfs


# ── Figure 1: IGBP violin (OOS) ──────────────────────────────────────────────

def plot_igbp_violin(df_oos, out_dir):
    col = "ubRMSE_0_10"
    df  = df_oos.dropna(subset=[col, "igbp_macro"])

    classes  = [c for c in IGBP_ORDER if c in df["igbp_macro"].values]
    data     = [df[df["igbp_macro"] == c][col].values for c in classes]
    medians  = [np.median(d) for d in data]
    order    = sorted(range(len(classes)), key=lambda i: medians[i])
    classes  = [classes[i] for i in order]
    data     = [data[i]    for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    vp = ax.violinplot(data, positions=range(len(classes)),
                       showmedians=True, showextrema=False, widths=0.6)

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(classes)))
    for i, (pc, med) in enumerate(zip(vp["bodies"], data)):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor("none")
        # scatter jitter
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(med))
        ax.scatter(np.full(len(med), i) + jitter, med,
                   s=6, color="k", alpha=0.4, linewidths=0, zorder=4)
        # N annotation above scatter cloud
        ax.text(i, float(np.nanmax(med)) + 0.005,
                f"n={len(med)}", ha="center", va="bottom", fontsize=7)

    vp["cmedians"].set_color("k")
    vp["cmedians"].set_linewidth(1.8)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8.5)
    ax.set_ylabel("ubRMSE  (m³/m³)", fontsize=9)
    ax.set_title("Surface SM ubRMSE by land-cover class  (OOS stations)", fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "igbp_violin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Climate violin (OOS) ───────────────────────────────────────────

def plot_climate_violin(df_oos, out_dir):
    col = "ubRMSE_0_10"
    df  = df_oos.dropna(subset=[col, "kg_macro"])

    classes = [c for c in CLIMATE_ORDER if c in df["kg_macro"].values]
    data    = [df[df["kg_macro"] == c][col].values for c in classes]
    labels  = [CLIMATE_NAMES.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    vp = ax.violinplot(data, positions=range(len(classes)),
                       showmedians=True, showextrema=False, widths=0.6)

    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    for i, (pc, med) in enumerate(zip(vp["bodies"], data)):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.75)
        pc.set_edgecolor("none")
        jitter = np.random.default_rng(i + 10).uniform(-0.18, 0.18, len(med))
        ax.scatter(np.full(len(med), i) + jitter, med,
                   s=6, color="k", alpha=0.4, linewidths=0, zorder=4)
        ax.text(i, max(med) + 0.004,
                f"n={len(med)}", ha="center", va="bottom", fontsize=7)

    vp["cmedians"].set_color("k")
    vp["cmedians"].set_linewidth(1.8)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ubRMSE  (m³/m³)", fontsize=9)
    ax.set_title("Surface SM ubRMSE by climate zone  (OOS stations)", fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "climate_violin.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Grouped bar — OOS vs OOT vs OOST by IGBP macro ─────────────────

def plot_oos_oot_oost_bar(dfs, out_dir):
    col = "ubRMSE_0_10"
    # Collect mean ubRMSE per split × IGBP macro
    igbp_classes = [c for c in IGBP_ORDER
                    if any(c in dfs[s]["igbp_macro"].values
                           for s in dfs if col in dfs[s].columns)]

    splits = [s for s in ["oos", "oot", "oost"] if s in dfs]
    n_splits = len(splits)
    n_classes = len(igbp_classes)
    x = np.arange(n_classes)
    width = 0.75 / n_splits

    fig, ax = plt.subplots(figsize=(7, 4))
    for si, split in enumerate(splits):
        df  = dfs[split].dropna(subset=[col, "igbp_macro"])
        means = []
        sems  = []
        for cls in igbp_classes:
            vals = df[df["igbp_macro"] == cls][col].values
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            sems.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
        offset = (si - n_splits / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width * 0.9,
                      color=SPLIT_COLORS[split],
                      label=SPLIT_LABELS[split],
                      alpha=0.85, zorder=3)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="k", capsize=2.5, linewidth=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(igbp_classes, fontsize=9)
    ax.set_ylabel("Mean surface ubRMSE  (m³/m³)", fontsize=9)
    ax.set_title("Generalisation across land-cover classes: OOS vs OOT vs OOST", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out_dir / "oos_oot_oost_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: Depth-wise ubRMSE bar across all splits ────────────────────────

def plot_depth_bar(dfs, out_dir):
    depths = ["0-10", "10-30", "30-100"]
    labels = ["0–10 cm", "10–30 cm", "30–100 cm"]
    splits = [s for s in ["oos", "oot", "oost"] if s in dfs]
    n_splits = len(splits)
    x = np.arange(len(depths))
    width = 0.72 / n_splits

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for si, split in enumerate(splits):
        df = dfs[split]
        means, sems = [], []
        for depth in depths:
            tag = DEPTH_TAGS[depth]
            col = f"ubRMSE_{tag}"
            if col in df.columns:
                vals = df[col].dropna().values
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
            else:
                means.append(np.nan)
                sems.append(np.nan)
        offset = (si - n_splits / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9,
               color=SPLIT_COLORS[split],
               label=SPLIT_LABELS[split],
               alpha=0.85, zorder=3)
        ax.errorbar(x + offset, means, yerr=sems,
                    fmt="none", color="k", capsize=2.5, linewidth=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean ubRMSE  (m³/m³)", fontsize=9)
    ax.set_title("Depth-stratified ubRMSE: OOS / OOT / OOST", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out_dir / "depth_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 5: Metrics heatmap (split × depth, ubRMSE) ────────────────────────

def plot_metrics_heatmap(dfs, out_dir):
    summary_path = MET_DIR / "metrics_summary.csv"
    if not summary_path.exists():
        print("metrics_summary.csv not found — skipping heatmap")
        return
    sm = pd.read_csv(summary_path)

    metrics = ["ubRMSE", "R", "bias"]
    splits  = ["oos", "oot", "oost"]
    depths  = ["0-10", "10-30", "30-100"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3.2))
    for ax, metric in zip(axes, metrics):
        mat = np.full((len(splits), len(depths)), np.nan)
        for si, split in enumerate(splits):
            for di, depth in enumerate(depths):
                row = sm[(sm["split"] == split) & (sm["depth"] == depth)]
                if len(row) and metric in row.columns:
                    mat[si, di] = float(row[metric].values[0])

        cmap = "RdYlGn_r" if metric == "ubRMSE" else ("RdYlGn" if metric == "R" else "RdBu_r")
        vmin = np.nanmin(mat) - 0.002
        vmax = np.nanmax(mat) + 0.002
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

        for si in range(len(splits)):
            for di in range(len(depths)):
                v = mat[si, di]
                if not np.isnan(v):
                    ax.text(di, si, f"{v:.3f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if metric != "bias" else "k")

        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d} cm" for d in depths], fontsize=8)
        ax.set_yticks(range(len(splits)))
        ax.set_yticklabels([SPLIT_LABELS[s] for s in splits], fontsize=8)
        ax.set_title(metric, fontsize=9, fontweight="bold")

    fig.suptitle("Model metrics: OOS / OOT / OOST × depth", fontsize=10, y=1.01)
    fig.tight_layout()
    path = out_dir / "metrics_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    dfs = load_dfs()
    if not dfs:
        print("No per_station CSV files found. Run evaluate_splits.py first.")
        return

    oos = dfs.get("oos")
    if oos is not None:
        plot_igbp_violin(oos, OUT_DIR)
        plot_climate_violin(oos, OUT_DIR)

    plot_oos_oot_oost_bar(dfs, OUT_DIR)
    plot_depth_bar(dfs, OUT_DIR)
    plot_metrics_heatmap(dfs, OUT_DIR)
    print(f"\nAll breakdown figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
