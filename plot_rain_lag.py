"""§72 rain-event lag ladder figure from analyse_rain_lag.py output.

Left column = water in the soil (rain, SM 0-10 cm, S1 VV, S2 SWIR B11);
right column = water in / amount of the plant (Red B04, NIR B08, NDMI, NDVI).
Lines = land-cover macro (median over stations, 95 % CI band); grey = placebo (dry-spell days, all stations).

    python plot_rain_lag.py --in-dir eval_output/rain_lag --out figures/rain_lag/lag_ladder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import plot_style_bw
from plot_style_bw import BLUE, GREEN, RED

MACRO_COLOR = {"Grass-Crop": BLUE, "Shrub-Savanna": GREEN, "Forest": RED}
PANELS = [  # (var, y label, scale)
    ("rain", "Rain (mm/day)", 1.0),
    ("SM",   "SM 0-10 cm (m3/m3)", 1.0),
    ("VV",   "S1 VV (dB)", 1.0),
    ("B11",  "SWIR B11 (refl. x100)", 100.0),
    ("B04",  "Red B04 (refl. x100)", 100.0),
    ("B08",  "NIR B08 (refl. x100)", 100.0),
    ("NDMI", "NDMI (x100)", 100.0),
    ("NDVI", "NDVI (x100)", 100.0),
]
ORDER = [0, 4, 1, 5, 2, 6, 3, 7]          # fill column-wise: left soil, right plant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="eval_output/rain_lag")
    ap.add_argument("--out", default="figures/rain_lag/lag_ladder")
    args = ap.parse_args()
    plot_style_bw.apply(globals(), "paper")
    comp = pd.read_csv(Path(args.in_dir) / "composite.csv")
    lags = pd.read_csv(Path(args.in_dir) / "lags.csv")
    st = pd.read_csv(Path(args.in_dir) / "stations.csv")
    n_st = st.groupby("macro").size().to_dict()

    fig, axs = plt.subplots(4, 2, figsize=(13, 12.5), sharex=True, constrained_layout=True)
    flat = axs.ravel()
    for k, pi in enumerate(ORDER):
        var, lab, sc = PANELS[pi]
        ax = flat[k]
        ax.axvline(0, color="black", lw=1.0, zorder=1)
        if var != "rain":
            ax.axhline(0, color="0.5", lw=0.8, zorder=1)
        p = comp[(comp["var"] == var) & (comp["kind"] == "placebo") & (comp["macro"] == "All")].sort_values("lag")
        if len(p):
            ax.fill_between(p["lag"], p["lo"] * sc, p["hi"] * sc, color="0.8", lw=0, zorder=2)
            ax.plot(p["lag"], p["median"] * sc, color="0.45", lw=1.4, zorder=3)
        for m, c in MACRO_COLOR.items():
            e = comp[(comp["var"] == var) & (comp["kind"] == "event") & (comp["macro"] == m)].sort_values("lag")
            if e.empty:
                continue
            ax.fill_between(e["lag"], e["lo"] * sc, e["hi"] * sc, color=c, alpha=0.18, lw=0, zorder=4)
            ax.plot(e["lag"], e["median"] * sc, color=c, lw=1.8, zorder=5,
                    marker=None if var in ("rain", "SM") else "o", ms=3.5)
            pk = lags[(lags["var"] == var) & (lags["macro"] == m)]
            pk = pk[(pk["peak_lo"] > 0) | (pk["peak_hi"] < 0)]           # only a significant peak
            if var != "rain" and len(pk):
                ax.plot(pk["peak_lag_d"], pk["peak"] * sc, "D", mfc=c, mec="black", ms=7, zorder=6)
        ax.set_ylabel(lab)
        ax.text(-0.14, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes, fontsize=13)
    for ax in axs[-1]:
        ax.set_xlabel("Days since rain event")
        ax.set_xlim(-15, 40)
    h = [plt.Line2D([], [], color=c, lw=2, label=f"{m} ({n_st.get(m, 0)})") for m, c in MACRO_COLOR.items()]
    h += [plt.Line2D([], [], color="0.45", lw=1.4, label="Placebo (dry days)"),
          plt.Line2D([], [], ls="none", marker="D", mfc="white", mec="black", ms=7, label="Peak response")]
    fig.legend(handles=h, loc="outside upper center", ncol=5, frameon=False)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
