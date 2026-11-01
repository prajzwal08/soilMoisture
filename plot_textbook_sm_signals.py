"""Textbook schematic: spectral signatures of dry / wet soil and green / water-stressed vegetation with all 13
Sentinel-2 bands and the H2O absorption bands (idealised curves, NOT data). Journal double-column width (7.2 in).
Annotations reviewed by an optical remote-sensing agent 2026-10-06 (chlorophyll absorbs blue + red, H2O bands,
atmospheric bands B1/B9/B10).

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

TXT = 7.5                                                   # body text (pt) at 7.2 in width
WL = np.linspace(400, 2500, 2101)
# all 13 Sentinel-2A MSI bands: centre, width (nm), label row (0 top) so neighbours do not collide
S2 = {"B1": (443, 21, 0), "B2": (492, 66, 1), "B3": (560, 36, 0), "B4": (665, 31, 1), "B5": (704, 15, 0),
      "B6": (741, 15, 1), "B7": (783, 20, 0), "B8": (833, 106, 1), "B8A": (865, 21, 2), "B9": (945, 20, 0),
      "B10": (1374, 31, 0), "B11": (1614, 91, 0), "B12": (2202, 175, 0)}
H2O = [(1350, 1460), (1800, 1960)]
WHITE_BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none")


def dip(c, w, d):
    return d * np.exp(-0.5 * ((WL - c) / w) ** 2)


def soil(wet: bool):
    base = 0.08 + 0.27 * (1 - np.exp(-(WL - 400) / 700))
    r = base - dip(1450, 35, 0.04) - dip(1940, 45, 0.05) - dip(2200, 30, 0.01)
    if wet:                                                # darker in every band, most in the SWIR
        r = r * (0.62 - 0.14 * (WL - 400) / 2100) - dip(1450, 60, 0.03) - dip(1940, 80, 0.05)
    return r


def veg(stressed: bool):
    sig = lambda x: 1 / (1 + np.exp(-x))
    vis = 0.04 + dip(550, 35, 0.065)
    edge = sig((WL - 715) / 14)
    plateau = 0.48 - 0.20 * sig((WL - 1330) / 50) - 0.13 * np.clip((WL - 1650) / 850, 0, 1)
    if stressed:                                           # less leaf water: NIR lower, SWIR higher, red higher
        plateau = plateau - 0.06 + 0.11 * sig((WL - 1330) / 50)
        vis = vis + dip(670, 30, 0.03)
    return (vis * (1 - edge) + plateau * edge - dip(970, 25, 0.025) - dip(1200, 35, 0.05)
            - dip(1450, 50, 0.15) - dip(1940, 60, 0.18))


# ── (a) spectra ────────────────────────────────────────────────────────────────────────────────
def panel_a(ax):
    for lo, hi in H2O:
        ax.axvspan(lo, hi, color="0.88", lw=0, zorder=0)
        ax.text((lo + hi) / 2, 0.012, "H$_2$O\nabsorption", ha="center", fontsize=6.5, color="0.3")
    for name, (c, w, row) in S2.items():
        ax.axvspan(c - w / 2, c + w / 2, color=GREEN, alpha=0.30, lw=0, zorder=1)
        ax.text(c, 0.593 - 0.028 * row, name, ha="center", va="bottom", fontsize=6.5)
    ax.plot(WL, soil(False), color=RED, lw=1.5, label="Dry soil")
    ax.plot(WL, soil(True), color=RED, lw=1.5, ls="--", label="Wet soil")
    ax.plot(WL, veg(False), color=BLUE, lw=1.5, label="Green vegetation")
    ax.plot(WL, veg(True), color=BLUE, lw=1.5, ls="--", label="Water-stressed vegetation")
    i = 1100 - 400
    ax.annotate("", xy=(1100, soil(True)[i] + 0.006), xytext=(1100, soil(False)[i] - 0.006),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", mutation_scale=9))
    ax.text(770, 0.068, "Wetting darkens soil in all bands;\nthe decrease is largest in the SWIR",
            fontsize=TXT - 0.5, va="center")
    ax.text(410, 0.355, "Visible (B2–B4):\nchlorophyll absorbs\nblue and red", fontsize=TXT - 0.5, va="top")
    ax.text(985, 0.535, "NIR (B8, B8A): scattering in leaf mesophyll", fontsize=TXT - 0.5, va="center",
            bbox=WHITE_BOX)
    ax.text(1480, 0.50, "SWIR (B11, B12): liquid-water absorption\nin leaves and soil", fontsize=TXT - 0.5,
            va="center", bbox=WHITE_BOX)
    ax.text(2030, 0.125, "B1 aerosol, B9 water vapour,\nB10 cirrus: atmospheric\ncorrection only",
            fontsize=TXT - 1, va="top")
    ax.set_xlim(400, 2500); ax.set_ylim(0, 0.63)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/textbook/sm_signals")
    args = ap.parse_args()
    plot_style_bw.apply(globals(), "paper")
    plt.rcParams.update({"font.size": TXT, "axes.labelsize": TXT + 0.5, "xtick.labelsize": TXT - 0.5,
                         "ytick.labelsize": TXT - 0.5, "legend.fontsize": TXT - 0.5, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    panel_a(ax)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
