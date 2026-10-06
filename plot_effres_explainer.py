"""Teaching figures for the §70 effective-resolution explainer (text/effres_explainer.tex).

    fig1  one real tile (CR200-18, 0-10 cm): the average map, one day's map, that day's anomaly
    fig2  toy fields: a smooth field and a rough field, and their variograms
    fig3  toy 1-D Nyquist sketch: what a 160 m grid can and cannot draw

    python plot_effres_explainer.py --in-dir eval_output/effres_txson --out-dir figures/effres/explainer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import plot_style_bw
from plot_style_bw import BLUE, GREEN, RED

TILE = "ISMN_TxSON_CR200-18"
MAP_PX, PX_M = 112, 20
EXT_KM = [0, MAP_PX * PX_M / 1000, 0, MAP_PX * PX_M / 1000]


def variogram(f: np.ndarray, rng, n=60000, edges=np.geomspace(20, 1600, 15)):
    r1, c1, r2, c2 = (rng.integers(0, f.shape[0], n) for _ in range(4))
    k = (r1 != r2) | (c1 != c2)
    d = np.hypot(r1[k] - r2[k], c1[k] - c2[k]) * PX_M
    g = 0.5 * (f[r1[k], c1[k]] - f[r2[k], c2[k]]) ** 2
    b = np.digitize(d, edges)
    mid = np.array([d[b == i].mean() for i in range(1, len(edges)) if (b == i).any()])
    gam = np.array([g[b == i].mean() for i in range(1, len(edges)) if (b == i).any()])
    return mid, gam / f.var()


def km_axes(ax):
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance (km)")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])


def fig1(d: Path, out: Path):
    idx = pd.read_parquet(d / "index_L0.parquet")
    ii = np.sort(idx.loc[idx["tile"] == TILE, "idx"].to_numpy())
    maps = np.load(d / "maps_L0.npy", mmap_mode="r")
    x = np.asarray(maps[ii, 0], np.float32)                        # (T, 112, 112), 0-10 cm
    mean = x.mean(0)
    anom = x - mean
    t = int(np.argmax(anom.std(axis=(1, 2))))                       # the day with most spatial contrast
    px = pd.read_csv(d / "readouts_pixels.csv")
    st = px[px["tile"] == TILE]
    sx = (st["mcol"].to_numpy() + 0.5) * PX_M / 1000
    sy = (MAP_PX - st["mrow"].to_numpy() - 0.5) * PX_M / 1000

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    lo, hi = np.percentile(np.r_[mean.ravel(), x[t].ravel()], [1, 99])
    for ax, f, lab in [(axs[0], mean, "(a)"), (axs[1], x[t], "(b)")]:
        im = ax.imshow(f, extent=EXT_KM, cmap="Blues", vmin=lo, vmax=hi)
        ax.plot(sx, sy, "o", mfc=RED, mec="black", ms=8)
        km_axes(ax); ax.text(-0.18, 1.02, lab, transform=ax.transAxes, fontsize=13)
    fig.colorbar(im, ax=axs[:2], shrink=0.85, label="Soil moisture (m3/m3)")
    v = np.percentile(np.abs(anom[t]), 99)
    im = axs[2].imshow(anom[t], extent=EXT_KM, cmap="RdBu", vmin=-v, vmax=v)
    axs[2].plot(sx, sy, "o", mfc=RED, mec="black", ms=8)
    km_axes(axs[2]); axs[2].text(-0.18, 1.02, "(c)", transform=axs[2].transAxes, fontsize=13)
    fig.colorbar(im, ax=axs[2], shrink=0.85, label="Anomaly (m3/m3)")
    for s in ("png", "pdf"):
        fig.savefig(out / f"fig1_tile.{s}", dpi=300)
    plt.close(fig)
    print(f"fig1: day index {t}, anomaly spatial SD {anom[t].std():.4f}, mean-map SD {mean.std():.4f}")


def fig2(out: Path):
    rng = np.random.default_rng(1)
    base = gaussian_filter(rng.standard_normal((MAP_PX, MAP_PX)), 300 / PX_M / 2, mode="wrap")
    smooth = base / base.std()
    rough = smooth + 0.8 * rng.standard_normal((MAP_PX, MAP_PX))
    rough /= rough.std()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, f, lab in [(axs[0], smooth, "(a)"), (axs[1], rough, "(b)")]:
        ax.imshow(f, extent=EXT_KM, cmap="RdBu", vmin=-2.5, vmax=2.5)
        km_axes(ax); ax.text(-0.18, 1.02, lab, transform=ax.transAxes, fontsize=13)
    ax = axs[2]
    for f, c, lab in [(smooth, BLUE, "Smooth field (a)"), (rough, RED, "Rough field (b)")]:
        h, g = variogram(f, rng)
        ax.semilogx(h, g, "o-", color=c, label=lab, ms=5)
    ax.axhline(1, color="0.45", ls=":", lw=1.2)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Separation (m)"); ax.set_ylabel("Semivariance / variance")
    ax.legend(loc="upper left", frameon=True, framealpha=1)
    ax.text(-0.15, 1.02, "(c)", transform=ax.transAxes, fontsize=13)
    for s in ("png", "pdf"):
        fig.savefig(out / f"fig2_toy_variogram.{s}", dpi=300)
    plt.close(fig)


def fig3(out: Path):
    x = np.linspace(0, 1600, 3201)
    cell = 160
    centres = np.arange(cell / 2, 1600, cell)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, constrained_layout=True)
    for ax, wl, lab in [(axs[0], 320, "(a)"), (axs[1], 100, "(b)")]:
        y = np.sin(2 * np.pi * x / wl)
        ax.plot(x, y, color="0.45", lw=1.2)
        means = [y[(x >= c - cell / 2) & (x < c + cell / 2)].mean() for c in centres]
        for c, m in zip(centres, means):
            ax.plot([c - cell / 2, c + cell / 2], [m, m], color=BLUE, lw=3)
        for e in np.arange(0, 1601, cell):
            ax.axvline(e, color="black", lw=0.6, ls="--")
        ax.set_ylim(-1.3, 1.3); ax.set_yticks([-1, 0, 1])
        ax.set_ylabel("Wetness")
        ax.text(-0.08, 1.02, lab, transform=ax.transAxes, fontsize=13)
    axs[1].set_xlabel("Distance along the ground (m)")
    h = [plt.Line2D([], [], color="0.45", lw=1.2, label="True wave"),
         plt.Line2D([], [], color=BLUE, lw=3, label="What one 160 m cell can hold (its average)"),
         plt.Line2D([], [], color="black", lw=0.6, ls="--", label="160 m cell edges")]
    fig.legend(handles=h, loc="outside upper center", ncol=3, frameon=False)
    for s in ("png", "pdf"):
        fig.savefig(out / f"fig3_nyquist.{s}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="eval_output/effres_txson")
    ap.add_argument("--out-dir", default="figures/effres/explainer")
    args = ap.parse_args()
    plot_style_bw.apply(globals(), "paper")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig1(Path(args.in_dir), out)
    fig2(out)
    fig3(out)


if __name__ == "__main__":
    main()
