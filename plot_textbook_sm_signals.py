"""Textbook schematic: how S2 bands / indices and S1 VV / VH relate to soil moisture (idealised curves, NOT data).

    (a) spectral signatures 400-2500 nm: dry soil, wet soil, green vegetation, water-stressed vegetation,
        with all 13 S2 bands and the water absorption bands
    (b) the role of the optical indices (NDVI, NDRE, NDMI, MSI, STR, OPTRAM)
    (c) C-band scattering over bare soil, sparse canopy and dense forest: how VV, VH and VH/VV relate to soil
        moisture (directly, through vegetation, or not at all) and how deep 5.6 cm waves get

    python plot_textbook_sm_signals.py --out figures/textbook/sm_signals
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style_bw
from plot_style_bw import BLUE, GREEN, RED

WL = np.linspace(400, 2500, 2101)
# all 13 Sentinel-2 MSI bands: centre, width (nm), label row (0 top, 1 below) so neighbours do not collide
S2 = {"B1": (443, 21, 0), "B2": (492, 66, 1), "B3": (560, 36, 0), "B4": (665, 31, 1), "B5": (704, 15, 0),
      "B6": (741, 15, 1), "B7": (783, 20, 0), "B8": (833, 106, 1), "B8A": (865, 21, 0), "B9": (945, 20, 1),
      "B10": (1374, 31, 0), "B11": (1614, 91, 0), "B12": (2202, 175, 0)}
WATER = [(1350, 1460), (1800, 1960)]                      # atmospheric/liquid water absorption


def dip(c, w, d):
    return d * np.exp(-0.5 * ((WL - c) / w) ** 2)


def soil(wet: bool):
    base = 0.08 + 0.27 * (1 - np.exp(-(WL - 400) / 700))       # rises to ~0.33 in the SWIR
    r = base - dip(1450, 35, 0.04) - dip(1940, 45, 0.05) - dip(2200, 30, 0.01)
    if wet:                                                    # darker everywhere, most in the SWIR
        r = r * (0.62 - 0.14 * (WL - 400) / 2100) - dip(1450, 60, 0.03) - dip(1940, 80, 0.05)
    return r


def veg(stressed: bool):
    sig = lambda x: 1 / (1 + np.exp(-x))
    vis = 0.04 + dip(550, 35, 0.065)
    edge = sig((WL - 715) / 14)
    plateau = 0.48 - 0.20 * sig((WL - 1330) / 50) - 0.13 * np.clip((WL - 1650) / 850, 0, 1)
    if stressed:                                           # less leaf water: NIR lower, SWIR brighter, red up
        plateau = plateau - 0.06 + 0.11 * sig((WL - 1330) / 50)
        vis = vis + dip(670, 30, 0.03)
    return (vis * (1 - edge) + plateau * edge - dip(970, 25, 0.025) - dip(1200, 35, 0.05)
            - dip(1450, 50, 0.15) - dip(1940, 60, 0.18))


def panel_a(ax):
    for lo, hi in WATER:
        ax.axvspan(lo, hi, color="0.88", lw=0, zorder=0)
    for name, (c, w, row) in S2.items():
        ax.axvspan(c - w / 2, c + w / 2, color=GREEN, alpha=0.30, lw=0, zorder=1)
        ax.text(c, 0.588 - 0.03 * row, name, ha="center", va="bottom", fontsize=10.5)
    ax.plot(WL, soil(False), color=RED, lw=2.2, label="Dry soil")
    ax.plot(WL, soil(True), color=RED, lw=2.2, ls="--", label="Wet soil")
    ax.plot(WL, veg(False), color=BLUE, lw=2.2, label="Green vegetation")
    ax.plot(WL, veg(True), color=BLUE, lw=2.2, ls="--", label="Water-stressed vegetation")
    i = int(1100 - 400)
    ax.annotate("", xy=(1100, soil(True)[i] + 0.006), xytext=(1100, soil(False)[i] - 0.006),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
    ax.text(1115, 0.195, "wetting darkens soil\n(all bands, most in SWIR)", fontsize=11, va="center")
    ax.text(415, 0.42, "Visible (B2-B4): chlorophyll\nabsorbs red, so leaves are dark", fontsize=10.5, va="top")
    ax.text(985, 0.545, "NIR (B8, B8A): scattered by\nleaf structure; NDVI = NIR vs red\n= green biomass", fontsize=10.5, va="top")
    ax.text(1480, 0.53, "SWIR (B11, B12): absorbed by water\nin soil and leaves; NDMI = NIR vs SWIR\n= leaf water", fontsize=10.5, va="top")
    ax.text(2040, 0.125, "B1, B9, B10: atmosphere\n(aerosol, water vapour, cirrus)", fontsize=10.5, va="top")
    ax.text(1405, 0.02, "water", ha="center", fontsize=10, color="0.35")
    ax.text(1880, 0.02, "water", ha="center", fontsize=10, color="0.35")
    ax.set_xlim(400, 2500); ax.set_ylim(0, 0.62)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=True, framealpha=1, edgecolor="black")




INDICES = [
    ("NDVI = (B8 - B4) / (B8 + B4)",
     "green biomass and cover: how much soil is hidden by plants. Follows soil moisture with a lag of weeks; "
     "saturates over dense canopy."),
    ("NDRE = (B8A - B5) / (B8A + B5)",
     "chlorophyll from the red edge; keeps working where NDVI saturates."),
    ("NDMI = (B8 - B11) / (B8 + B11)",
     "water held in the leaves (canopy water). Drops within days when plants run short of water."),
    ("MSI = B11 / B8",
     "moisture stress index, the inverse view of NDMI: higher = drier canopy."),
    ("STR = (1 - B12)^2 / (2 B12)",
     "SWIR transformed reflectance: rises as the visible soil gets wetter. The only direct optical soil "
     "signal, and only where bare soil is seen."),
    ("NDVI-STR trapezoid (OPTRAM)",
     "STR read against NDVI separates soil wetness from vegetation cover."),
]


def panel_indices(ax):
    ax.axis("off")
    for k, (name, role) in enumerate(INDICES):
        col, row = k % 2, k // 2
        x, y = 0.0 + 0.5 * col, 0.97 - 0.34 * row
        ax.text(x, y, name, fontsize=11.5, va="top", transform=ax.transAxes)
        ax.text(x + 0.015, y - 0.12, textwrap.fill(role, 78), fontsize=10.5, va="top", transform=ax.transAxes)


# ── radar scattering schematic ─────────────────────────────────────────────────────────────────
GROUND = 1.5


def _arrow(ax, p0, p1, color, lw=2.2, ls="-"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls, mutation_scale=16))


def _scene_base(ax, title):
    ax.set_xlim(0, 10); ax.set_ylim(-8.2, 10); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 10, GROUND, color=RED, alpha=0.22, lw=0))
    ax.add_patch(plt.Rectangle((0, GROUND - 0.35), 10, 0.35, color=RED, alpha=0.55, lw=0))
    ax.plot([0, 10], [GROUND, GROUND], color="black", lw=1.2)
    ax.text(9.9, 0.55, "soil: C-band enters\nonly the top ~1-5 cm", fontsize=9.5, ha="right", va="center")
    ax.add_patch(plt.Rectangle((0.35, 8.9), 0.7, 0.45, color="0.3"))
    ax.add_patch(plt.Rectangle((0.0, 9.0), 0.35, 0.25, color=GREEN, ec="black", lw=0.6))
    ax.add_patch(plt.Rectangle((1.05, 9.0), 0.35, 0.25, color=GREEN, ec="black", lw=0.6))
    ax.text(1.6, 9.15, "S1, C-band, 5.6 cm", fontsize=10, va="center")
    ax.text(0.0, -0.4, title, fontsize=12, va="top")


def _bullets(ax, lines, y0=-1.5):
    ax.text(0.0, y0, "\n".join(textwrap.fill(s, 46, initial_indent="- ", subsequent_indent="  ") for s in lines),
            fontsize=10, va="top", linespacing=1.25)


def scene_bare(ax):
    _scene_base(ax, "Bare soil")
    _arrow(ax, (0.9, 8.7), (5.2, GROUND + 0.05), "black")
    _arrow(ax, (5.2, GROUND + 0.05), (9.3, 7.0), "0.55", ls="--")
    _arrow(ax, (5.4, GROUND + 0.05), (1.6, 8.2), RED, lw=2.6)
    _arrow(ax, (5.6, GROUND + 0.05), (2.6, 7.0), BLUE, lw=1.0)
    ax.text(1.85, 8.25, "VV", color=RED, fontsize=12, va="center")
    ax.text(4.0, 5.2, "VH (weak)", color=BLUE, fontsize=11)
    ax.text(9.0, 4.0, "specular,\nlost", fontsize=10, color="0.4", ha="center")
    _bullets(ax, [
        "Surface scattering dominates.",
        "VV rises ~0.25 dB per 1 % soil moisture: water raises the soil dielectric constant (dry ~4, wet ~25).",
        "VH is low: a bare surface barely depolarises the wave, so VH/VV is low.",
        "Roughness (ploughing, crust) also changes VV.",
        "Direct link to soil moisture, but only the top few cm, not the root zone.",
    ])


def scene_crop(ax):
    _scene_base(ax, "Grass / crop (sparse canopy)")
    for x in np.arange(2.6, 9.8, 0.75):
        ax.plot([x, x], [GROUND, 4.6], color=BLUE, lw=1.6)
        for h, s in [(2.6, 1), (3.3, -1), (4.0, 1)]:
            ax.plot([x, x + 0.35 * s], [h, h + 0.35], color=BLUE, lw=1.4)
    _arrow(ax, (0.9, 8.7), (6.2, GROUND + 0.05), "black")
    _arrow(ax, (6.4, GROUND + 0.05), (3.0, 8.4), RED, lw=1.8)
    _arrow(ax, (3.4, 4.3), (0.4, 7.4), BLUE, lw=2.4)
    ax.text(3.6, 8.2, "VV: soil + canopy", color=RED, fontsize=11)
    ax.text(0.1, 5.4, "VH: canopy\nvolume", color=BLUE, fontsize=11, va="top")
    _bullets(ax, [
        "The wave crosses the canopy twice and is attenuated, but part of it still reaches the soil.",
        "VV = soil (direct soil moisture) + canopy; VH = mostly volume scattering in leaves and stems.",
        "Stem-ground double bounce brightens VV over wet ground.",
        "VH/VV grows with biomass: used to remove the vegetation part from VV (water cloud model).",
        "Wetter plants raise the canopy dielectric: VH follows soil moisture indirectly, through plant water.",
    ])


def scene_forest(ax):
    _scene_base(ax, "Dense forest")
    from matplotlib.patches import Ellipse
    for x in (2.6, 5.2, 7.8):
        ax.add_patch(plt.Rectangle((x - 0.18, GROUND), 0.36, 3.0, color="0.35", lw=0))
    for x, y in ((2.6, 5.6), (5.2, 6.0), (7.8, 5.6), (3.9, 6.6), (6.5, 6.6)):
        ax.add_patch(Ellipse((x, y), 3.0, 2.8, color=GREEN, ec=BLUE, lw=1.4, alpha=0.95))
    _arrow(ax, (0.9, 8.7), (3.9, 7.4), "black")
    for p1, c in [((1.5, 8.2), RED), ((2.4, 8.3), BLUE), ((1.2, 7.7), BLUE)]:
        _arrow(ax, (3.9, 7.4), p1, c, lw=1.6)
    ax.text(5.2, 3.0, "ground not\nreached", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.8))
    ax.text(4.2, 8.4, "VV + VH from the crown", fontsize=10.5)
    _bullets(ax, [
        "5.6 cm waves are scattered by leaves and twigs in the crown; almost nothing reaches the soil.",
        "VV and VH both come from the canopy and saturate; VH/VV is high.",
        "Direct sensitivity to soil moisture is ~0; only an indirect link through canopy water.",
        "Seeing forest soil needs longer waves: L-band (24 cm) or P-band (70 cm).",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/textbook/sm_signals")
    args = ap.parse_args()
    plot_style_bw.apply(globals(), "paper")
    fig = plt.figure(figsize=(16, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.42, 1.2])
    ax_a = fig.add_subplot(gs[0, :])
    ax_i = fig.add_subplot(gs[1, :])
    axs = [fig.add_subplot(gs[2, k]) for k in range(3)]
    panel_a(ax_a); panel_indices(ax_i)
    for ax, f in zip(axs, (scene_bare, scene_crop, scene_forest)):
        f(ax)
    for ax, lab, x in [(ax_a, "(a)", -0.05), (ax_i, "(b)", -0.05), (axs[0], "(c)", -0.02)]:
        ax.text(x, 1.02, lab, transform=ax.transAxes, fontsize=13)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
