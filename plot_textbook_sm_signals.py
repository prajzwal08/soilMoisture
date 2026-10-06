"""Textbook schematic: how S2 bands / indices and S1 VV / VH relate to soil moisture (idealised curves, NOT data).

    (a) spectral signatures 400-2500 nm: dry soil, wet soil, green vegetation, water-stressed vegetation,
        with all 13 S2 bands and the water absorption bands
    (b) VV / VH vs soil moisture   (c) VV / VH vs vegetation water content
    (d) idealised response after a rain event: SM, VV, SWIR darkening (same day), NDMI (days), NDVI (weeks)

    python plot_textbook_sm_signals.py --out figures/textbook/sm_signals
"""
from __future__ import annotations

import argparse
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


def panel_b(ax):
    """VV and VH against soil moisture (bare / sparse vegetation)."""
    sm = np.linspace(0.05, 0.45, 100)
    ax.plot(sm, -20 + 25 * (sm - 0.05), color=RED, lw=2.2, label="VV")
    ax.plot(sm, -27 + 15 * (sm - 0.05), color=BLUE, lw=2.2, label="VH")
    ax.text(0.06, -9.5, "VV: surface scattering\nsoil moisture, roughness\n(~0.25 dB per 1 % vol.)", fontsize=10.5, va="top")
    ax.text(0.20, -27.2, "VH: weak, depolarised\nless sensitive to soil", fontsize=10.5, va="top")
    ax.set_xlim(0.05, 0.45); ax.set_ylim(-30, -6)
    ax.set_xlabel("Soil moisture (m3/m3)"); ax.set_ylabel("C-band backscatter (dB)")
    ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="black")


def panel_c(ax):
    """VV and VH against vegetation water content (fixed soil moisture)."""
    v = np.linspace(0, 5, 100)
    ax.plot(v, -12 + 2.5 * (1 - np.exp(-v / 1.5)), color=RED, lw=2.2, label="VV")
    ax.plot(v, -26 + 12 * (1 - np.exp(-v / 2.0)), color=BLUE, lw=2.2, label="VH")
    ax.text(0.15, -7.0, "VV: soil signal increasingly\nhidden (attenuated) by canopy", fontsize=10.5, va="top")
    ax.text(1.6, -23.0, "VH: volume scattering\nin leaves and stems\n= canopy biomass, structure", fontsize=10.5, va="top")
    ax.text(4.95, -12.6, "VH \u2212 VV rises:\nvegetation index", fontsize=10.5, ha="right", va="top")
    ax.set_xlim(0, 5); ax.set_ylim(-30, -6)
    ax.set_xlabel("Vegetation water content (kg/m2)"); ax.set_ylabel("C-band backscatter (dB)")
    ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="black")


def panel_d(ax):
    t = np.linspace(-5, 40, 901)
    pos = np.clip(t, 0, None)

    def pulse(tau_rise, tau_fall):
        r = (1 - np.exp(-pos / tau_rise)) * np.exp(-pos / tau_fall)
        return r / r.max()

    sm = np.where(t >= 0, np.exp(-pos / 7), 0.0)
    curves = [("Soil moisture", sm, "black", "-"),
              ("S1 VV", 0.85 * sm, RED, "-"),
              ("SWIR darkening", 0.75 * np.where(t >= 0, np.exp(-pos / 4), 0.0), RED, "--"),
              ("NDMI (leaf water)", 0.6 * pulse(2.0, 9.0), GREEN, "-"),
              ("NDVI (green biomass)", 0.45 * pulse(9.0, 25.0), BLUE, "-")]
    ax.bar([0], [1.08], width=0.8, color="0.6", zorder=1)
    ax.text(0.6, 1.08, "rain", fontsize=11, va="center")
    for lab, y, c, ls in curves:
        ax.plot(t, y, color=c, lw=2.2, ls=ls, label=lab, zorder=3)
    for x, y, lab in [(1.0, 1.0, "same day"), (3.6, 0.63, "days"), (12.0, 0.48, "weeks")]:
        ax.text(x + 0.6, y, lab, fontsize=11, va="bottom")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(-5, 40); ax.set_ylim(-0.02, 1.15)
    ax.set_xlabel("Days after rain"); ax.set_ylabel("Response (normalised)")
    ax.legend(loc="upper right", frameon=True, framealpha=1, edgecolor="black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/textbook/sm_signals")
    args = ap.parse_args()
    plot_style_bw.apply(globals(), "paper")
    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])
    ax_a = fig.add_subplot(gs[0, :])
    axs = [fig.add_subplot(gs[1, k]) for k in range(3)]
    for ax, f, lab in [(ax_a, panel_a, "(a)"), (axs[0], panel_b, "(b)"), (axs[1], panel_c, "(c)"),
                       (axs[2], panel_d, "(d)")]:
        f(ax)
        ax.text(-0.05 if ax is ax_a else -0.17, 1.02, lab, transform=ax.transAxes, fontsize=13)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
