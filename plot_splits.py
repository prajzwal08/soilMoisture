"""
Visualise the OOS / OOT / OOST / ablation evaluation splits.

Plots saved to plots/splits/ as SVG (publication quality).

Run with:
  /home/khanalp/miniforge3/envs/soilmoisture/bin/python plot_splits.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

SPLITS_CSV = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_splits.csv")
OUT_DIR    = Path("/home/khanalp/code/PhD/soilMoisture/plots/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour / marker scheme ──────────────────────────────────────────────────
SPLIT_STYLE = {
    "train":       dict(color="#4878D0", marker="o", s=8,  alpha=0.55, zorder=3, label="Train"),
    "val":         dict(color="#6ACC65", marker="o", s=12, alpha=0.80, zorder=4, label="Validation"),
    "oos":         dict(color="#EE854A", marker="o", s=14, alpha=0.85, zorder=5, label="OOS"),
}
SPECIAL_STYLE = {
    "joint_eval":      dict(color="#D65F5F", marker="*", s=70,  alpha=1.0, zorder=7, label="Joint-eval (flux+SM)"),
    "flux_only_eval":  dict(color="#956CB4", marker="D", s=30,  alpha=1.0, zorder=6, label="Flux-only eval"),
}

KG_COLORS = {
    "A": "#E41A1C", "B": "#FF7F00", "C": "#4DAF4A",
    "D": "#377EB8", "E": "#984EA3",
}
KG_LABELS = {
    "A": "Tropical", "B": "Arid", "C": "Temperate",
    "D": "Continental", "E": "Polar",
}
IGBP_ORDER = ["Forest", "Grass-Crop", "Shrub-Savanna", "Other"]

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(SPLITS_CSV)

# ── helper ───────────────────────────────────────────────────────────────────
def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  saved → {path.name}")
    plt.close(fig)


# ============================================================
# 1. Global map
# ============================================================
def plot_global_map():
    world = gpd.read_file(
        "/home/khanalp/miniforge3/envs/soilmoisture/lib/python3.11/site-packages/"
        "pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    world.plot(ax=ax, color="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.3)

    # base splits: train → val → oos (layered so OOS is visible on top)
    for split_name, style in SPLIT_STYLE.items():
        sub = df[df["split"] == split_name]
        # exclude special sites from base layer
        if split_name == "oos":
            sub = sub[~sub["joint_eval"] & ~sub["flux_only_eval"]]
        ax.scatter(sub["longitude"], sub["latitude"], **style)

    # special sites on top
    for col, style in SPECIAL_STYLE.items():
        sub = df[df[col]]
        ax.scatter(sub["longitude"], sub["latitude"], **style)

    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title("Global station distribution by evaluation split", fontsize=11)

    handles = [mpatches.Patch(color=v["color"], label=v["label"])
               for v in {**SPLIT_STYLE, **SPECIAL_STYLE}.values()]
    ax.legend(handles=handles, loc="lower left", fontsize=7, ncol=2,
              framealpha=0.9, edgecolor="#AAAAAA")

    # counts annotation
    counts = (f"Train {(df.split=='train').sum()} | "
              f"Val {(df.split=='val').sum()} | "
              f"OOS {(df.split=='oos').sum()}")
    ax.text(0.5, -0.06, counts, transform=ax.transAxes,
            ha="center", fontsize=8, color="#444444")
    savefig(fig, "01_global_map.svg")


# ============================================================
# 2. IGBP × KG heatmap (counts, train vs OOS side by side)
# ============================================================
def plot_igbp_kg_heatmap():
    kg_order = ["A", "B", "C", "D", "E"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, split_name, title in zip(
        axes,
        ["train", "oos"],
        ["Training set", "OOS set"],
    ):
        sub = df[df["split"] == split_name]
        pivot = (sub.groupby(["igbp_macro", "kg_macro"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=IGBP_ORDER, columns=kg_order, fill_value=0))

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(kg_order)))
        ax.set_xticklabels([f"{k}\n({KG_LABELS[k]})" for k in kg_order], fontsize=7)
        ax.set_yticks(range(len(IGBP_ORDER)))
        ax.set_yticklabels(IGBP_ORDER, fontsize=8)
        ax.set_title(f"{title}  (n = {len(sub)})", fontsize=9)
        ax.set_xlabel("Köppen-Geiger macro-zone", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("IGBP macro-group", fontsize=8)

        # annotate cells
        for i in range(len(IGBP_ORDER)):
            for j in range(len(kg_order)):
                val = pivot.values[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color="black" if val < pivot.values.max()*0.6 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Station count")

    fig.suptitle("Station distribution across IGBP ecosystem × Köppen-Geiger climate cells",
                 fontsize=10, y=1.02)
    savefig(fig, "02_igbp_kg_heatmap.svg")


# ============================================================
# 3. IGBP distribution — train / val / OOS normalised bar
# ============================================================
def plot_igbp_distribution():
    igbp_full = sorted(df["IGBP"].value_counts().index.tolist())
    splits    = ["train", "val", "oos"]
    split_labels = {"train": "Train", "val": "Val", "oos": "OOS"}
    colors    = ["#4878D0", "#6ACC65", "#EE854A"]

    counts = {s: df[df["split"]==s]["IGBP"].value_counts().reindex(igbp_full, fill_value=0)
              for s in splits}
    fracs  = {s: counts[s] / counts[s].sum() for s in splits}

    x = np.arange(len(igbp_full))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (s, c) in enumerate(zip(splits, colors)):
        ax.bar(x + (i-1)*w, fracs[s].values * 100, w, label=split_labels[s],
               color=c, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x); ax.set_xticklabels(igbp_full, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of split (%)")
    ax.set_title("IGBP land cover distribution across splits")
    ax.legend(fontsize=8); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "03_igbp_distribution.svg")


# ============================================================
# 4. Köppen-Geiger macro-zone distribution
# ============================================================
def plot_kg_distribution():
    kg_order = ["A", "B", "C", "D", "E"]
    splits   = ["train", "val", "oos"]
    split_labels = {"train": "Train", "val": "Val", "oos": "OOS"}
    colors   = ["#4878D0", "#6ACC65", "#EE854A"]

    fracs = {}
    for s in splits:
        sub = df[df["split"]==s]["kg_macro"].value_counts()
        sub = sub.reindex(kg_order, fill_value=0)
        fracs[s] = sub / sub.sum() * 100

    x = np.arange(len(kg_order)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (s, c) in enumerate(zip(splits, colors)):
        ax.bar(x + (i-1)*w, fracs[s].values, w, label=split_labels[s],
               color=c, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} – {KG_LABELS[k]}" for k in kg_order], fontsize=8)
    ax.set_ylabel("Fraction of split (%)")
    ax.set_title("Köppen-Geiger macro-zone distribution across splits")
    ax.legend(fontsize=8); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, "04_kg_distribution.svg")


# ============================================================
# 5. Record length histogram
# ============================================================
def plot_record_length():
    splits = ["train", "val", "oos"]
    split_labels = {"train": "Train", "val": "Val", "oos": "OOS"}
    colors = ["#4878D0", "#6ACC65", "#EE854A"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 13.5, 0.5)
    for s, c in zip(splits, colors):
        sub = df[df["split"]==s]["n_years"].dropna()
        ax.hist(sub, bins=bins, alpha=0.55, color=c, edgecolor="white",
                linewidth=0.4, label=f"{split_labels[s]} (n={len(sub)}, μ={sub.mean():.1f}yr)")

    ax.axvline(3, color="#888888", linestyle="--", linewidth=0.8, label="3-year minimum")
    ax.set_xlabel("Record length (years)"); ax.set_ylabel("Number of stations")
    ax.set_title("Record length distribution by split")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    savefig(fig, "05_record_length.svg")


# ============================================================
# 6. Temporal coverage strip + OOT window
# ============================================================
def plot_temporal_coverage():
    OOT_START = 2023

    sub = df[df["split"].isin(["train", "oos"])].copy()
    sub["start_yr"] = sub["start_date"].astype(str).str[:4].astype(float)
    sub["end_yr"]   = sub["end_date"].astype(str).str[:4].astype(float) + \
                      sub["end_date"].astype(str).str[4:6].astype(float)/12
    sub = sub.sort_values(["split", "start_yr"]).reset_index(drop=True)

    color_map = {"train": "#4878D0", "oos": "#EE854A"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, row in sub.iterrows():
        c = color_map[row["split"]]
        ax.barh(i, row["end_yr"] - row["start_yr"], left=row["start_yr"],
                height=0.8, color=c, alpha=0.5, linewidth=0)

    # OOT window shading
    ax.axvspan(OOT_START, 2026, alpha=0.12, color="#D65F5F", label="OOT window (2023→)")
    ax.axvline(OOT_START, color="#D65F5F", linewidth=1.0, linestyle="--")

    ax.set_xlabel("Year"); ax.set_ylabel("Station index (sorted by start year)")
    ax.set_title("Temporal coverage of training and OOS stations")
    ax.set_xlim(2013, 2026)

    handles = [mpatches.Patch(color="#4878D0", alpha=0.6, label=f"Train (n={(sub.split=='train').sum()})"),
               mpatches.Patch(color="#EE854A", alpha=0.6, label=f"OOS (n={(sub.split=='oos').sum()})"),
               mpatches.Patch(color="#D65F5F", alpha=0.2, label="OOT test window (2023→)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    savefig(fig, "06_temporal_coverage.svg")


# ============================================================
# 7. Network × split stacked bar
# ============================================================
def plot_network_composition():
    networks = ["ISMN", "ICOS", "AmeriFlux"]
    split_order  = ["train", "val", "oos"]
    split_colors = {"train": "#4878D0", "val": "#6ACC65", "oos": "#EE854A"}
    split_labels = {"train": "Train", "val": "Val", "oos": "OOS"}

    fig, ax = plt.subplots(figsize=(7, 4))
    bottoms = np.zeros(len(networks))
    for s in split_order:
        counts = [len(df[(df["source_network"]==n) & (df["split"]==s)]) for n in networks]
        ax.bar(networks, counts, bottom=bottoms, color=split_colors[s],
               label=split_labels[s], alpha=0.85, edgecolor="white", linewidth=0.5)
        for i, (c, b) in enumerate(zip(counts, bottoms)):
            if c > 0:
                ax.text(i, b + c/2, str(c), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += np.array(counts, dtype=float)

    ax.set_ylabel("Number of stations"); ax.set_title("Split composition by network")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    savefig(fig, "07_network_composition.svg")


# ============================================================
# 8. Elevation distribution
# ============================================================
def plot_elevation_distribution():
    splits = ["train", "val", "oos"]
    split_labels = {"train": "Train", "val": "Val", "oos": "OOS"}
    colors = ["#4878D0", "#6ACC65", "#EE854A"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 4200, 200)
    for s, c in zip(splits, colors):
        sub = df[df["split"]==s]["elevation_m"].dropna()
        ax.hist(sub, bins=bins, alpha=0.55, color=c, edgecolor="white",
                linewidth=0.4, label=f"{split_labels[s]} (μ={sub.mean():.0f} m)")

    ax.axvline(500,  color="#888888", linestyle=":", linewidth=0.8)
    ax.axvline(1500, color="#888888", linestyle=":", linewidth=0.8)
    ax.text(250,  ax.get_ylim()[1]*0.95, "Low",  ha="center", fontsize=7, color="#888888")
    ax.text(1000, ax.get_ylim()[1]*0.95, "Mid",  ha="center", fontsize=7, color="#888888")
    ax.text(2500, ax.get_ylim()[1]*0.95, "High", ha="center", fontsize=7, color="#888888")
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel("Number of stations")
    ax.set_title("Elevation distribution by split")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    savefig(fig, "08_elevation_distribution.svg")


# ============================================================
# 9. Ablation subset overview — miniature vs full
# ============================================================
def plot_ablation_overview():
    kg_order = ["A", "B", "C", "D", "E"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, mask, title in zip(
        axes,
        [df["split"] == "train",
         (df["split"] == "train") & df["ablation_train"]],
        [f"Full training set  (n = {(df['split']=='train').sum()})",
         f"Ablation subset  (n = {df['ablation_train'].sum()}, ~20%)"],
    ):
        pivot = (df[mask].groupby(["igbp_macro", "kg_macro"])
                         .size()
                         .unstack(fill_value=0)
                         .reindex(index=IGBP_ORDER, columns=kg_order, fill_value=0))

        im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(range(len(kg_order)))
        ax.set_xticklabels([f"{k}\n({KG_LABELS[k]})" for k in kg_order], fontsize=7)
        ax.set_yticks(range(len(IGBP_ORDER)))
        ax.set_yticklabels(IGBP_ORDER, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Köppen-Geiger macro-zone", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("IGBP macro-group", fontsize=8)

        for i in range(len(IGBP_ORDER)):
            for j in range(len(kg_order)):
                val = pivot.values[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=7,
                            color="black" if val < pivot.values.max()*0.6 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Station count")

    fig.suptitle("Full training set vs ablation subset — environmental cell distribution",
                 fontsize=10, y=1.02)
    savefig(fig, "09_ablation_subset.svg")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("Generating split visualisations...")
    plot_global_map()
    plot_igbp_kg_heatmap()
    plot_igbp_distribution()
    plot_kg_distribution()
    plot_record_length()
    plot_temporal_coverage()
    plot_network_composition()
    plot_elevation_distribution()
    plot_ablation_overview()
    print(f"\nAll plots saved to {OUT_DIR}")
