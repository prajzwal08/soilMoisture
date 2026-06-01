"""
plot_image_stats.py
===================
Exploratory plots for per-station per-year S2L2A and S1RTC image counts.

Outputs (fig/image_stats/):
  01_yearly_totals.png          — total tiles per year, S2 vs S1
  02_yearly_avg_per_station.png — avg tiles/station/year, S2 vs S1
  03_station_count_per_year.png — how many stations have data each year
  04_distribution_s2.png        — distribution of S2 tiles/station-year
  05_distribution_s1.png        — distribution of S1 tiles/station-year
  06_distribution_combined.png  — S2 and S1 side-by-side
  07_per_station_total.png      — total tiles per station (sorted), S2 and S1
  08_heatmap_s2.png             — station × year heatmap (S2, top-50 stations)
  09_heatmap_s1.png             — station × year heatmap (S1, top-50 stations)
  10_split_breakdown.png        — tiles per split (train/val/oos) per year
  11_zero_tile_years.png        — stations with zero-tile years per modality
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

OUT_DIR = Path("fig/image_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG = pd.read_csv("csvs/images_per_station_year.csv", dtype={"year": str})
SPL = pd.read_csv("csvs/station_splits.csv")

# Build folder → split mapping
def _folder(row):
    sn, nw = row["source_network"], row["network"]
    return f"{sn}_{row['station_id']}" if sn == nw else f"{sn}_{nw}_{row['station_id']}"

SPL["folder"] = SPL.apply(_folder, axis=1)
split_map = dict(zip(SPL["folder"], SPL["split"]))
IMG["split"] = IMG["station"].map(split_map).fillna("unknown")

YEARS = sorted(IMG["year"].unique())
COLORS = {"S2L2A": "#2166ac", "S1RTC": "#d6604d"}

# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


# ── 01: yearly totals ─────────────────────────────────────────────────────────
print("01 yearly_totals")
s2_yr = IMG.groupby("year")["s2"].sum()
s1_yr = IMG.groupby("year")["s1"].sum()

x = np.arange(len(YEARS))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, [s2_yr.get(y, 0) for y in YEARS], w, label="S2L2A", color=COLORS["S2L2A"])
ax.bar(x + w/2, [s1_yr.get(y, 0) for y in YEARS], w, label="S1RTC", color=COLORS["S1RTC"])
ax.set_xticks(x); ax.set_xticklabels(YEARS, rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xlabel("Year"); ax.set_ylabel("Total tiles")
ax.set_title("Total satellite tiles per year (1,011 stations)")
ax.legend(); fig.tight_layout()
savefig("01_yearly_totals.png")


# ── 02: avg per station per year ──────────────────────────────────────────────
print("02 yearly_avg_per_station")
s2_avg = IMG[IMG["s2"] > 0].groupby("year")["s2"].mean()
s1_avg = IMG[IMG["s1"] > 0].groupby("year")["s1"].mean()
s2_med = IMG[IMG["s2"] > 0].groupby("year")["s2"].median()
s1_med = IMG[IMG["s1"] > 0].groupby("year")["s1"].median()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(YEARS, [s2_avg.get(y, np.nan) for y in YEARS], "o-", color=COLORS["S2L2A"], label="S2 mean")
ax.plot(YEARS, [s2_med.get(y, np.nan) for y in YEARS], "s--", color=COLORS["S2L2A"], alpha=0.6, label="S2 median")
ax.plot(YEARS, [s1_avg.get(y, np.nan) for y in YEARS], "o-", color=COLORS["S1RTC"], label="S1 mean")
ax.plot(YEARS, [s1_med.get(y, np.nan) for y in YEARS], "s--", color=COLORS["S1RTC"], alpha=0.6, label="S1 median")
ax.set_xticks(range(len(YEARS))); ax.set_xticklabels(YEARS, rotation=45)
ax.set_xlabel("Year"); ax.set_ylabel("Tiles / station")
ax.set_title("Average & median tiles per station-year (stations with >0 tiles)")
ax.legend(); fig.tight_layout()
savefig("02_yearly_avg_per_station.png")


# ── 03: station count per year ────────────────────────────────────────────────
print("03 station_count_per_year")
s2_n = IMG[IMG["s2"] > 0].groupby("year")["station"].nunique()
s1_n = IMG[IMG["s1"] > 0].groupby("year")["station"].nunique()

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, [s2_n.get(y, 0) for y in YEARS], w, label="S2L2A", color=COLORS["S2L2A"])
ax.bar(x + w/2, [s1_n.get(y, 0) for y in YEARS], w, label="S1RTC", color=COLORS["S1RTC"])
ax.set_xticks(x); ax.set_xticklabels(YEARS, rotation=45)
ax.set_xlabel("Year"); ax.set_ylabel("Number of stations")
ax.set_title("Stations with at least 1 tile per year")
ax.legend(); fig.tight_layout()
savefig("03_station_count_per_year.png")


# ── 04: distribution S2 ───────────────────────────────────────────────────────
print("04 distribution_s2")
s2_vals = IMG[IMG["s2"] > 0]["s2"]
bins = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500, 1000]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(s2_vals, bins=50, color=COLORS["S2L2A"], edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("S2L2A tiles / station-year")
axes[0].set_ylabel("Count (station-years)")
axes[0].set_title("S2L2A distribution (station-years with >0 tiles)")
axes[0].axvline(s2_vals.median(), color="k", ls="--", lw=1, label=f"Median {s2_vals.median():.0f}")
axes[0].axvline(s2_vals.mean(),   color="k", ls=":",  lw=1, label=f"Mean {s2_vals.mean():.0f}")
axes[0].legend(fontsize=8)

# CDF
sorted_v = np.sort(s2_vals)
axes[1].plot(sorted_v, np.arange(1, len(sorted_v)+1) / len(sorted_v), color=COLORS["S2L2A"])
axes[1].set_xlabel("S2L2A tiles / station-year"); axes[1].set_ylabel("Cumulative fraction")
axes[1].set_title("CDF — S2L2A tiles per station-year")
axes[1].axhline(0.25, color="grey", ls=":", lw=0.8)
axes[1].axhline(0.50, color="grey", ls=":", lw=0.8)
axes[1].axhline(0.75, color="grey", ls=":", lw=0.8)
fig.tight_layout(); savefig("04_distribution_s2.png")


# ── 05: distribution S1 ───────────────────────────────────────────────────────
print("05 distribution_s1")
s1_vals = IMG[IMG["s1"] > 0]["s1"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(s1_vals, bins=50, color=COLORS["S1RTC"], edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("S1RTC tiles / station-year")
axes[0].set_ylabel("Count (station-years)")
axes[0].set_title("S1RTC distribution (station-years with >0 tiles)")
axes[0].axvline(s1_vals.median(), color="k", ls="--", lw=1, label=f"Median {s1_vals.median():.0f}")
axes[0].axvline(s1_vals.mean(),   color="k", ls=":",  lw=1, label=f"Mean {s1_vals.mean():.0f}")
axes[0].legend(fontsize=8)

sorted_v = np.sort(s1_vals)
axes[1].plot(sorted_v, np.arange(1, len(sorted_v)+1) / len(sorted_v), color=COLORS["S1RTC"])
axes[1].set_xlabel("S1RTC tiles / station-year"); axes[1].set_ylabel("Cumulative fraction")
axes[1].set_title("CDF — S1RTC tiles per station-year")
for q in [0.25, 0.5, 0.75]:
    axes[1].axhline(q, color="grey", ls=":", lw=0.8)
fig.tight_layout(); savefig("05_distribution_s1.png")


# ── 06: combined distribution ─────────────────────────────────────────────────
print("06 distribution_combined")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, col, label, color in [
    (axes[0], "s2", "S2L2A", COLORS["S2L2A"]),
    (axes[0], "s1", "S1RTC", COLORS["S1RTC"]),
]:
    vals = IMG[IMG[col] > 0][col]
    ax.hist(vals, bins=60, alpha=0.6, label=label, color=color, edgecolor="none")
axes[0].set_xlabel("Tiles / station-year"); axes[0].set_ylabel("Count")
axes[0].set_title("S2L2A vs S1RTC — tile count distribution")
axes[0].legend()

for col, label, color in [("s2", "S2L2A", COLORS["S2L2A"]), ("s1", "S1RTC", COLORS["S1RTC"])]:
    vals = np.sort(IMG[IMG[col] > 0][col])
    axes[1].plot(vals, np.arange(1, len(vals)+1) / len(vals), label=label, color=color)
axes[1].set_xlabel("Tiles / station-year"); axes[1].set_ylabel("Cumulative fraction")
axes[1].set_title("CDF comparison — S2L2A vs S1RTC")
for q in [0.25, 0.5, 0.75]:
    axes[1].axhline(q, color="grey", ls=":", lw=0.8)
axes[1].legend()
fig.tight_layout(); savefig("06_distribution_combined.png")


# ── 07: total per station (sorted) ────────────────────────────────────────────
print("07 per_station_total")
s2_tot = IMG.groupby("station")["s2"].sum().sort_values(ascending=False)
s1_tot = IMG.groupby("station")["s1"].sum().sort_values(ascending=False)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
axes[0].bar(range(len(s2_tot)), s2_tot.values, color=COLORS["S2L2A"], width=1.0)
axes[0].set_ylabel("Total S2L2A tiles"); axes[0].set_xlabel("Stations (sorted)")
axes[0].set_title("Total S2L2A tiles per station (all years)")
axes[0].axhline(s2_tot.median(), color="k", ls="--", lw=1, label=f"Median {s2_tot.median():.0f}")
axes[0].legend()

axes[1].bar(range(len(s1_tot)), s1_tot.values, color=COLORS["S1RTC"], width=1.0)
axes[1].set_ylabel("Total S1RTC tiles"); axes[1].set_xlabel("Stations (sorted)")
axes[1].set_title("Total S1RTC tiles per station (all years)")
axes[1].axhline(s1_tot.median(), color="k", ls="--", lw=1, label=f"Median {s1_tot.median():.0f}")
axes[1].legend()
fig.tight_layout(); savefig("07_per_station_total.png")


# ── 08 & 09: heatmaps (top 60 stations by total tiles) ───────────────────────
for col, label, color, num in [
    ("s2", "S2L2A", COLORS["S2L2A"], "08"),
    ("s1", "S1RTC", COLORS["S1RTC"], "09"),
]:
    print(f"{num} heatmap_{col}")
    top_stations = IMG.groupby("station")[col].sum().nlargest(60).index
    pivot = IMG[IMG["station"].isin(top_stations)].pivot_table(
        index="station", columns="year", values=col, aggfunc="sum", fill_value=0
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_title(f"{label} tiles per station per year (top 60 stations by total)")
    plt.colorbar(im, ax=ax, label="Tiles")
    fig.tight_layout(); savefig(f"{num}_heatmap_{col}.png")


# ── 10: split breakdown per year ──────────────────────────────────────────────
print("10 split_breakdown")
split_colors = {"train": "#1b7837", "val": "#762a83", "oos": "#e08214", "unknown": "grey"}
splits_order = ["train", "val", "oos"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, label in [(axes[0], "s2", "S2L2A"), (axes[1], "s1", "S1RTC")]:
    bottom = np.zeros(len(YEARS))
    for sp in splits_order:
        sub = IMG[IMG["split"] == sp].groupby("year")[col].sum()
        vals = np.array([sub.get(y, 0) for y in YEARS], dtype=float)
        ax.bar(YEARS, vals, bottom=bottom, label=sp, color=split_colors[sp])
        bottom += vals
    ax.set_xlabel("Year"); ax.set_ylabel("Total tiles")
    ax.set_title(f"{label} tiles per year by split")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend()
fig.tight_layout(); savefig("10_split_breakdown.png")


# ── 11: zero-tile years per station ──────────────────────────────────────────
print("11 zero_tile_years")
zero_s2 = IMG[IMG["s2"] == 0].groupby("station")["year"].count()
zero_s1 = IMG[IMG["s1"] == 0].groupby("station")["year"].count()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, zero, col, label, color in [
    (axes[0], zero_s2, "s2", "S2L2A", COLORS["S2L2A"]),
    (axes[1], zero_s1, "s1", "S1RTC", COLORS["S1RTC"]),
]:
    counts = zero.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=color)
    ax.set_xlabel("Number of years with 0 tiles")
    ax.set_ylabel("Number of stations")
    ax.set_title(f"{label} — stations by number of zero-tile years\n({len(zero)} stations affected)")
fig.tight_layout(); savefig("11_zero_tile_years.png")


print(f"\nAll plots saved to {OUT_DIR}/")
