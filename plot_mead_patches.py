# ============================================================
# Mead: which 160 m TerraMind patches the §69.5 cosines compare (runbook §69.5)
# ============================================================
# One clear S2 scene of Ne1's 2.24 km tile (true colour + NDVI) with the 14 x 14 L12 token
# grid (160 m patches) and the three patches plot_mead_embed.py uses, all in row 7:
#   Ne1 patch (7,7), Ne2 patch (7,10), reference (7,4) = 3 patches (480 m) west of Ne1.
# Towers at their exact pixel (tower_px from plot_mead_s2.py).
#
# Usage:
#   python plot_mead_patches.py [--date 2019-07-22]
# ============================================================

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.patches import Rectangle

import plot_style_bw
from plot_mead_pred import FIG_DIR
from plot_mead_s2 import RAW, B, MARK, tower_px

PATCH   = 16                                        # 224 px / 14 tokens = 16 px = 160 m
BOXES   = {"Ne1 patch (7,7)": ((7, 7), plot_style_bw.BLUE),
           "Ne2 patch (7,10)": ((7, 10), plot_style_bw.GREEN),
           "reference (7,4)": ((7, 4), plot_style_bw.RED)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2019-07-22")
    d = pd.Timestamp(ap.parse_args().date)

    root = zarr.open_group(str(RAW), mode="r")
    g = root["s2"]
    dates = pd.to_datetime([x.decode() if isinstance(x, bytes) else str(x) for x in g["dates"][:]],
                           format="%Y%m%d")
    i = int(np.where(dates == d)[0][0])
    a = np.asarray(g["data"][i]).astype(np.float32)
    rgb = np.clip(np.stack([a[B["B04"]], a[B["B03"]], a[B["B02"]]], -1) / 3000.0, 0, 1) ** (1 / 1.4)
    nir, red = a[B["B08"]], a[B["B04"]]
    ndvi = (nir - red) / np.where(nir + red > 0, nir + red, np.nan)
    px = tower_px(dict(root.attrs))
    print(f"{d:%Y-%m-%d}: towers (row, col) {px}")

    plot_style_bw.apply(globals(), "paper")
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    for k, (x, img) in enumerate(zip(ax, (rgb, ndvi))):
        im = x.imshow(img, interpolation="nearest", **({} if k == 0 else
                      dict(cmap="RdYlGn", vmin=0, vmax=0.9)))
        for e in np.arange(PATCH, 224, PATCH) - 0.5:
            x.axhline(e, color="white", lw=0.4, alpha=0.6)
            x.axvline(e, color="white", lw=0.4, alpha=0.6)
        for lab, ((r, c), col) in BOXES.items():
            x.add_patch(Rectangle((c * PATCH - 0.5, r * PATCH - 0.5), PATCH, PATCH,
                                  fill=False, ec=col, lw=2.2, label=lab))
        for s, (r, c) in px.items():
            mk, col = MARK[s]
            x.plot(c, r, mk, ms=8, color=col, mec="black", mew=0.8, label=f"{s} tower")
        x.set_xticks(np.arange(0, 224, PATCH) + PATCH / 2 - 0.5)
        x.set_xticklabels(range(14), fontsize="x-small")
        x.set_yticks(np.arange(0, 224, PATCH) + PATCH / 2 - 0.5)
        x.set_yticklabels(range(14), fontsize="x-small")
        x.set_xlabel("patch column (160 m)")
        x.set_ylabel("patch row (160 m)")
        x.text(0.02, 0.98, f"({'ab'[k]}) {d:%d %b %Y}, {'true colour' if k == 0 else 'NDVI'}",
               transform=x.transAxes, va="top", fontsize="small",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
        if k == 1:
            fig.colorbar(im, ax=x, fraction=0.046, pad=0.02, label="NDVI")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_patches_{d:%Y%m%d}.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_patches_{d:%Y%m%d}.png")


if __name__ == "__main__":
    main()
