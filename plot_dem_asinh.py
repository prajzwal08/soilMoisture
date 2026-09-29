"""
plot_dem_asinh.py — how the fine CNN's DEM channel is encoded (asinh relative relief)
======================================================================================
Paper figure. Two real 2.24 km tiles from the raw imagery store, on the 20 m grid the fine CNN
reads (10 m GLO-30 -> masked 2x2 mean, as dataset.build_fine does):
  Hupsel      ISMN_TWENTE_Hupsel   Twente, Netherlands  (flat, ~7 m p5-p95 relief)
  Ngari SQ19  ISMN_NGARI_SQ19      western Tibet        (~190 m p5-p95 relief, 4,600 m a.s.l.)

  (a)    encoding curves vs relief, symlog x (linear inside +/-1 m, log beyond)
  (b-g)  2 tiles x {raw elevation, global z-score, asinh relative relief}

The encoding:  asinh((elev - tile mean) / s) / k,  s = 1 m (linear-to-log knee), k = 4 (output
~ +/-2, the magnitude of the other normalised channels). Global z-score = fine_stats.json's
(elev - 670.665) / 951.272.

Two envs (zarr lives in terramind, scienceplots in soilmoisture):
  python plot_dem_asinh.py --extract   # terramind   -> figures/dem_asinh/dem_asinh_tiles.npz
  python plot_dem_asinh.py --plot      # soilmoisture -> figures/dem_asinh/dem_asinh_encoding.{png,pdf}
Figure: PNG + PDF, dpi 300, house style §13.3.
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "figures" / "dem_asinh"
NPZ = OUT / "dem_asinh_tiles.npz"
RAW_ROOTS = [Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr"), Path("/projects/prjs1968/satellite_zarr")]
TILES = [("ISMN_TWENTE_Hupsel", "Hupsel", "Twente, Netherlands"),
         ("ISMN_NGARI_SQ19", "Ngari SQ19", "western Tibet")]
DEM_MEAN, DEM_STD = 670.665, 951.272          # csvs/fine_stats.json
S_M, K = 1.0, 4.0                             # asinh knee (m) and output scale
PIX_KM = 0.02                                 # 20 m grid


def asinh_enc(dh):
    return np.arcsinh(dh / S_M) / K


def extract():
    import zarr
    OUT.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for st, *_ in TILES:
        root = next(r for r in RAW_ROOTS if (r / f"{st}.zarr").exists())
        x = np.asarray(zarr.open_group(str(root / f"{st}.zarr"), mode="r")["dem/data"][0], np.float64)
        m = np.isfinite(x) & (x > -1000)
        s = np.where(m, x, 0).reshape(112, 2, 112, 2).sum((1, 3))
        c = m.reshape(112, 2, 112, 2).sum((1, 3))
        arrs[st] = np.where(c > 0, s / np.maximum(c, 1), np.nan).astype(np.float32)
        e = arrs[st]
        print(f"{st}: {np.nanmin(e):.1f}-{np.nanmax(e):.1f} m, mean {np.nanmean(e):.1f}, "
              f"nodata {np.isnan(e).mean():.3%}  (from {root})")
    np.savez_compressed(NPZ, **arrs)
    print(f"-> {NPZ}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "nature"])
    except ImportError:
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})
    plt.rcParams["text.usetex"] = False
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

    tiles = np.load(NPZ)
    fig = plt.figure(figsize=(7.2, 6.9))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.78, 1, 1], hspace=0.42, wspace=0.34)

    # ── (a) encoding curves ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    dh = np.concatenate([-np.logspace(3.2, -2, 400), [0.0], np.logspace(-2, 3.2, 400)])
    ax.plot(dh, asinh_enc(dh), color="#1a6faf", lw=1.6,
            label=r"asinh relative relief: $\mathrm{asinh}(\Delta h / 1\,\mathrm{m})\,/\,4$")
    ax.plot(dh, dh / DEM_STD, color="#e8851a", lw=1.3, ls="--",
            label=r"global z-score (within-tile part): $\Delta h\,/\,951\,\mathrm{m}$")
    refs = [(-2, "polder\nhollow"), (0.5, "ditch"), (2, "dike"), (10, "ridge"), (100, "hills"), (1000, "alpine")]
    for h, lab in refs:
        y = asinh_enc(h)
        ax.plot(h, y, "o", ms=3.6, color="#1a6faf", mec="white", mew=0.6, zorder=3)
        ax.annotate(f"{lab}\n{h:g} m → {y:.2f}", (h, y), xytext=(0, 7 if h > 0 else -18),
                    textcoords="offset points", ha="center", fontsize=6.2, color="0.25", linespacing=0.95)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.set_xlim(-1600, 1600)
    ax.set_ylim(-2.3, 2.6)
    ax.axhline(0, color="0.75", lw=0.5, zorder=0)
    ax.axvspan(-1, 1, color="0.93", zorder=0, lw=0)
    ax.text(0, -2.05, "linear\n|Δh| < 1 m", ha="center", fontsize=6, color="0.4", linespacing=0.95)
    ax.set_xlabel(r"$\Delta h$, height above tile mean (m, symmetric-log axis)")
    ax.set_ylabel("DEM channel value")
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    ax.set_title("(a) Encoding of relative relief", loc="left", fontweight="bold")

    # ── (b-g) tiles ────────────────────────────────────────────────────────────
    ext = [0, 112 * PIX_KM, 112 * PIX_KM, 0]
    cols = [("Raw elevation (m)", None),
            ("Global z-score", TwoSlopeNorm(vcenter=0, vmin=-5, vmax=5)),
            ("asinh relative relief", TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2))]
    letters = iter("bcdefg")
    for r, (st, name, where) in enumerate(TILES):
        e = tiles[st]
        mean = float(np.nanmean(e))
        fields = [e, (e - DEM_MEAN) / DEM_STD, asinh_enc(e - mean)]
        for c, ((title, norm), f) in enumerate(zip(cols, fields)):
            ax = fig.add_subplot(gs[r + 1, c])
            if norm is None:
                im = ax.imshow(f, cmap="cividis", extent=ext, interpolation="nearest")
            else:
                im = ax.imshow(f, cmap="RdBu_r", norm=norm, extent=ext, interpolation="nearest")
            ax.plot(56.5 * PIX_KM, 56.5 * PIX_KM, marker="s", ms=3.2, mfc="none", mec="white", mew=0.8)
            sd = float(np.nanstd(f))
            ax.set_title(f"({next(letters)}) {name}: {title}", loc="left", fontweight="bold")
            ax.text(0.03, 0.04, f"within-tile sd {sd:.3g}", transform=ax.transAxes, fontsize=6,
                    color="white", bbox=dict(fc="black", alpha=0.45, lw=0, pad=1.2))
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
            ax.tick_params(length=2)
            if r == 1:
                ax.set_xlabel("km")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{where}\nkm")
            else:
                ax.set_yticklabels([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6, length=2)
            cb.outline.set_linewidth(0.4)

    for ext_ in ("png", "pdf"):
        fig.savefig(OUT / f"dem_asinh_encoding.{ext_}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT / 'dem_asinh_encoding'}.png/.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--extract", action="store_true")
    g.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    extract() if a.extract else plot()
