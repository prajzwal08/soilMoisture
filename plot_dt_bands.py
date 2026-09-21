#!/usr/bin/env python
"""Publication figures: does restricting ECOSTRESS pairs to one dt band bias the sample?

Three figures, vector PDF + 600 dpi PNG:

  fig1_dt_spectrum   the bimodal separation spectrum -- why no window exists between the
                     two orbital configurations
  fig2_band_bias     (a) station retention vs latitude, (b) Koppen composition,
                     (c) pairs-per-station ECDF -- each band against what was available
  fig3_global_maps   global station maps: (a) pre-dawn share per station, (b) which band
                     each station belongs to

The maps are full-extent equirectangular (Plate Carree).  Coastlines come from Natural
Earth 110m, fetched once into data/naturalearth/ -- there is no cartopy in either conda
env, and geopandas + pyogrio read the shapefile directly.

Inputs (written by count_pairs_dt12.py):
    csvs/ecostress_dt_bands_by_station.csv
    csvs/ecostress_clearfirst_dt.csv
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

ROOT = "/gpfs/work3/0/prjs1968/soilMoisture"
BANDS = f"{ROOT}/csvs/ecostress_dt_bands_by_station.csv"
DTCSV = f"{ROOT}/csvs/ecostress_clearfirst_dt.csv"
OUTDIR = f"{ROOT}/fig/ecostress_dt_bands"
NEDIR = f"{ROOT}/data/naturalearth"
NE_URL = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"

# validated categorical palette (dataviz six-checks pass, light surface); assigned to the
# ENTITY, so adding a band never repaints the others
C_EARLY = "#B8651A"   # dt 6-9 h    evening night half
C_LATE = "#3167C9"    # dt 12-24 h  pre-dawn night half
C_ALL = "#4a4742"     # the available inventory -- neutral, it is the reference not a series
C_NEUTRAL = "#c9c5bc"

EARLY, LATE, ALLB = "n_6_9", "n_12_24", "n_0_26"

LAND_FILL = "#eceae5"
LAND_EDGE = "#b9b5ac"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef3fb", "#7aa3e0", "#3167C9", "#16305f"])


def ensure_land():
    """Natural Earth 110m land, fetched once.  Returns a GeoDataFrame or None."""
    import geopandas as gpd
    shp = f"{NEDIR}/ne_110m_land.shp"
    if not os.path.exists(shp):
        import io
        import urllib.request
        import zipfile
        os.makedirs(NEDIR, exist_ok=True)
        print(f"fetching {NE_URL}", flush=True)
        with urllib.request.urlopen(NE_URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(NEDIR)
    return gpd.read_file(shp)


def setup():
    """Journal defaults: serif, 7-8 pt, hairline rules, no chartjunk."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def finish(ax, minor_x=True, minor_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="out", top=False, right=False)


def panel_label(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUTDIR}/{stem}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


# ------------------------------------------------------------------
def fig_spectrum(dt, stem):
    """The separation spectrum.  One panel, single-column width."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bins = np.arange(0, 24.25, 0.5)
    h, edges = np.histogram(dt["dt_hours"], bins=bins)
    for left, v in zip(edges[:-1], h):
        if v == 0:
            continue
        c = C_EARLY if 6 <= left < 9 else (C_LATE if 12 <= left < 24 else C_NEUTRAL)
        ax.bar(left + 0.25, v / 1000.0, width=0.44, color=c, lw=0, align="center")

    ax.axvspan(9, 12, color="0.5", alpha=0.10, lw=0, zorder=0)
    ax.annotate("no pairs\n(9–12 h)", xy=(10.5, 13.0), fontsize=6.5, ha="center",
                va="top", color="0.35", style="italic")

    ax.set_xlabel("Night $-$ day separation, $\\Delta t$ (h)")
    ax.set_ylabel(r"Clear pairs ($\times 10^3$)")
    ax.set_xlim(2, 23)
    ax.set_ylim(0, 21)  # headroom so the legend never sits on the 16-17 h bar
    ax.xaxis.set_major_locator(MultipleLocator(4))
    finish(ax)

    n_e = int(((dt.dt_hours >= 6) & (dt.dt_hours < 9)).sum())
    n_l = int(((dt.dt_hours >= 12) & (dt.dt_hours < 24)).sum())
    ax.legend(handles=[
        Line2D([], [], color=C_EARLY, lw=4, label=f"6–9 h, evening ($n$={n_e:,})"),
        Line2D([], [], color=C_LATE, lw=4, label=f"12–24 h, pre-dawn ($n$={n_l:,})")],
        loc="upper left", bbox_to_anchor=(-0.02, 1.03), handlelength=1.1,
        handletextpad=0.5, labelspacing=0.2)
    save(fig, stem)


def fig_bias(b, stem):
    """Three panels, double-column width."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    avail = b[b[ALLB] > 0]

    # ---- (a) station retention vs latitude -------------------------------
    ax = axes[0]
    edges = np.arange(25, 57.5, 5.0)
    ctr = edges[:-1] + 2.5
    n_av, _ = np.histogram(avail["lat"].abs(), bins=edges)
    for col, c, lab, mk in ((EARLY, C_EARLY, "6–9 h", "^"), (LATE, C_LATE, "12–24 h", "s")):
        n_b, _ = np.histogram(b.loc[b[col] > 0, "lat"].abs(), bins=edges)
        keep = n_av > 0
        frac = np.where(keep, n_b / np.maximum(n_av, 1) * 100.0, np.nan)
        ax.plot(ctr[keep], frac[keep], color=c, marker=mk, ms=3.4, mew=0, label=lab)
    ax.axhline(100, color="0.6", lw=0.6, ls=(0, (3, 2)))
    ax.text(26, 101.5, "all available", fontsize=6.5, color="0.4", style="italic")
    ax.set_xlabel("Station latitude, $|\\phi|$ (deg)")
    ax.set_ylabel("Stations retained (%)")
    ax.set_ylim(0, 112)
    ax.set_xlim(24, 56)
    ax.legend(loc="lower left", handlelength=1.4, handletextpad=0.5)
    panel_label(ax, "(a)")
    finish(ax)

    # ---- (b) climate composition -----------------------------------------
    ax = axes[1]
    cls = [k for k in sorted(avail["kg_macro"].dropna().unique())
           if (avail["kg_macro"] == k).sum() >= 5]
    x = np.arange(len(cls))
    w = 0.26
    for i, (d, c, lab) in enumerate((
            (avail, C_ALL, "Available"),
            (b[b[EARLY] > 0], C_EARLY, "6–9 h"),
            (b[b[LATE] > 0], C_LATE, "12–24 h"))):
        share = [100.0 * (d["kg_macro"] == k).mean() for k in cls]
        ax.bar(x + (i - 1) * w, share, w * 0.86, color=c, lw=0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(cls)
    ax.set_xlabel("Köppen macro-class")
    ax.set_ylabel("Share of stations (%)")
    ax.set_ylim(0, 46)
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.5, ncol=1)
    panel_label(ax, "(b)")
    finish(ax, minor_x=False)

    # ---- (c) pairs-per-station ECDF --------------------------------------
    ax = axes[2]
    for col, c, lab in ((ALLB, C_ALL, "Available"), (EARLY, C_EARLY, "6–9 h"),
                        (LATE, C_LATE, "12–24 h")):
        v = np.sort(b.loc[b[col] > 0, col].to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100.0, where="post", color=c,
                label=f"{lab} ($n$={len(v)})")
    ax.axvline(20, color="0.6", lw=0.6, ls=(0, (3, 2)))
    ax.annotate("20-pair floor", xy=(20, 2), xytext=(23, 2), fontsize=6.5,
                color="0.35", style="italic", va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("Clear pairs per station")
    ax.set_ylabel("Cumulative stations (%)")
    ax.set_xlim(1, 400)
    ax.set_ylim(0, 103)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), handlelength=1.4,
              handletextpad=0.5, labelspacing=0.25)
    panel_label(ax, "(c)")
    finish(ax, minor_x=False)

    fig.subplots_adjust(wspace=0.42)
    save(fig, stem)


def fig_maps(b, stem):
    """Global station maps, equirectangular, full extent."""
    land = ensure_land()
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.5))

    def base(ax):
        if land is not None:
            land.plot(ax=ax, facecolor=LAND_FILL, edgecolor=LAND_EDGE, lw=0.3, zorder=0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 84)
        ax.set_aspect("equal")  # true Plate Carree, not a stretched scatter
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-60, 61, 30))
        ax.set_xticklabels([f"{abs(v)}°{'' if v == 0 else ('E' if v > 0 else 'W')}"
                            for v in range(-180, 181, 60)])
        ax.set_yticklabels([f"{abs(v)}°{'' if v == 0 else ('N' if v > 0 else 'S')}"
                            for v in range(-60, 61, 30)])
        for lat in (52, -52):
            ax.axhline(lat, color="#8a8681", lw=0.55, ls=(0, (3.5, 2.5)), zorder=1)
        ax.tick_params(which="both", direction="out", top=False, right=False, length=2.5)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_color("#8a8681")

    # ---- (a) pre-dawn share -------------------------------------------
    ax = axes[0]
    base(ax)
    m = b[b[ALLB] > 0].copy()
    m["frac"] = m[LATE] / m[ALLB]
    m = m.sort_values("frac")  # pale underneath, so dark points are not hidden
    sc = ax.scatter(m["lon"], m["lat"], c=m["frac"], cmap=SEQ, vmin=0, vmax=1, s=6.5,
                    lw=0.15, edgecolor="white", zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.021, aspect=18)
    cb.set_label("Pre-dawn share of clear pairs, $\\Delta t \\geq 12$ h", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#8a8681")
    ax.text(-178, 53.5, "ECOSTRESS ±52° coverage limit", fontsize=6, color="#6b6862",
            style="italic", va="bottom", zorder=4)
    panel_label(ax, "(a)", dx=-0.055, dy=1.01)

    # ---- (b) band membership ------------------------------------------
    ax = axes[1]
    base(ax)
    both = b[(b[EARLY] > 0) & (b[LATE] > 0)]
    only_e = b[(b[EARLY] > 0) & (b[LATE] == 0)]
    only_l = b[(b[EARLY] == 0) & (b[LATE] > 0)]
    ax.scatter(both["lon"], both["lat"], s=5, color="#9c9890", lw=0, zorder=2,
               label=f"Both bands ($n$={len(both)})")
    ax.scatter(only_e["lon"], only_e["lat"], s=13, color=C_EARLY, marker="^", lw=0.2,
               edgecolor="white", zorder=4, label=f"Only 6–9 h ($n$={len(only_e)})")
    ax.scatter(only_l["lon"], only_l["lat"], s=11, color=C_LATE, marker="s", lw=0.2,
               edgecolor="white", zorder=3, label=f"Only 12–24 h ($n$={len(only_l)})")
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.02), ncol=1, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3, borderpad=0.35, frameon=True,
              facecolor="white", edgecolor="#b9b5ac", framealpha=0.92, fontsize=6.5)
    panel_label(ax, "(b)", dx=-0.055, dy=1.01)

    fig.subplots_adjust(hspace=0.16)
    save(fig, stem)


def table(b, dt):
    """The numbers behind the figures -- a chart is never the only record."""
    avail = b[b[ALLB] > 0]
    print("\n" + "=" * 72)
    print(f"{'band':<13}{'stations':>9}{'pairs':>9}{'images':>9}{'>=20 pairs':>12}"
          f"{'med pairs':>11}")
    print("=" * 72)
    for lab, col in (("available", ALLB), ("6-9 h", EARLY), ("12-24 h", LATE),
                     ("15-19 h", "n_15_19"), ("0-12 h", "n_0_12")):
        d = b[b[col] > 0]
        print(f"{lab:<13}{len(d):>9,}{int(b[col].sum()):>9,}{2 * int(b[col].sum()):>9,}"
              f"{int((b[col] >= 20).sum()):>12,}{int(d[col].median()):>11,}")
    print("=" * 72)

    print("\nBIAS vs available (percentage points; + = over-represented)")
    kl = [k for k in sorted(avail["kg_macro"].dropna().unique())
          if (avail["kg_macro"] == k).sum() >= 5]
    lat_bins = [(25, 35), (35, 40), (40, 45), (45, 60)]
    print(f"{'band':<11}" + "".join(f"{f'|lat| {a}-{c}':>12}" for a, c in lat_bins)
          + "".join(f"{'K:' + k:>7}" for k in kl))
    for lab, col in (("6-9 h", EARLY), ("12-24 h", LATE), ("0-12 h", "n_0_12")):
        d = b[b[col] > 0]
        cells = ""
        for a, c in lat_bins:
            sd = 100.0 * ((d["lat"].abs() >= a) & (d["lat"].abs() < c)).mean()
            sa = 100.0 * ((avail["lat"].abs() >= a) & (avail["lat"].abs() < c)).mean()
            cells += f"{sd - sa:>+12.1f}"
        for k in kl:
            cells += f"{100.0 * (d['kg_macro'] == k).mean() - 100.0 * (avail['kg_macro'] == k).mean():>+7.1f}"
        print(f"{lab:<11}" + cells)

    # retention at the extremes, the claim panel (a) makes
    for col, lab in ((EARLY, "6-9 h"), (LATE, "12-24 h")):
        hi_b = int(((b[col] > 0) & (b["lat"].abs() >= 45)).sum())
        hi_a = int((avail["lat"].abs() >= 45).sum())
        print(f"{lab}: retains {hi_b}/{hi_a} = {100.0 * hi_b / hi_a:.0f}% of "
              f"stations at |lat| >= 45")
    gap = int(((dt.dt_hours >= 9) & (dt.dt_hours < 12)).sum())
    print(f"pairs in the 9-12 h gap: {gap:,} of {len(dt):,} "
          f"({100.0 * gap / len(dt):.2f}%)")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    setup()
    b = pd.read_csv(BANDS)
    dt = pd.read_csv(DTCSV)
    print(f"{len(b):,} stations, {len(dt):,} unrestricted clear-first pairs")

    for stale in ("dt_histogram.png", "dt_band_maps.png", "dt_band_bias.png"):
        p = f"{OUTDIR}/{stale}"
        if os.path.exists(p):
            os.remove(p)
            print(f"removed superseded {stale}")

    fig_spectrum(dt, "fig1_dt_spectrum")
    fig_bias(b, "fig2_band_bias")
    fig_maps(b, "fig3_global_maps")
    table(b, dt)
    print(f"\nwrote fig1_dt_spectrum, fig2_band_bias, fig3_global_maps "
          f"(.pdf + .png) to {OUTDIR}")


if __name__ == "__main__":
    main()
