"""
Publication-quality architecture flowcharts for the soil moisture model.

Emits two standalone figures (§30.7):
  figures/architecture_current.{png,pdf}   — the shipped model (cls_depth_star_reg)
  figures/architecture_proposed.{png,pdf}  — the §30 per-location processor

Boxes are annotated with tensor shapes so the resolution argument can be read
straight off the figure: the current design's only spatial carrier is a 14x14
token grid, and 100% of the temporal signal reaches the decoder through one
spatially constant vector.

Usage:
  python plot_architecture.py [--dpi 400] [--out figures]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# STYLE
# ============================================================

# Muted, print-safe palette. Hue encodes role, not magnitude.
C_INPUT   = "#dbe5f1"   # raw inputs
C_INPUT_E = "#5b7fa6"
C_PROC    = "#e4dcef"   # attention / learned blocks
C_PROC_E  = "#7a6796"
C_OUT     = "#d9e8db"   # outputs
C_OUT_E   = "#4f7d59"
C_BAD     = "#f6dcdc"   # the defect
C_BAD_E   = "#a94d4d"
C_NEW     = "#fdeacd"   # new in the proposed design
C_NEW_E   = "#bf8b32"
C_TXT     = "#1a1a1a"
C_MUTE    = "#5a5a5a"

FS_BOX, FS_SHAPE, FS_NOTE, FS_HEAD = 8.2, 7.0, 7.4, 9.6


def _setup():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.usetex": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def box(ax, x, y, w, h, label, shape=None, fc=C_INPUT, ec=C_INPUT_E,
        fs=FS_BOX, bold=False, lw=1.0, ls="-"):
    """Rounded box centred at (x, y). `shape` is printed under the label in mono."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.1",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    if shape:
        ax.text(x, y + h * 0.17, label, ha="center", va="center", fontsize=fs,
                color=C_TXT, zorder=3, fontweight="bold" if bold else "normal")
        ax.text(x, y - h * 0.24, shape, ha="center", va="center",
                fontsize=FS_SHAPE, color=C_MUTE, family="monospace", zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=C_TXT,
                zorder=3, fontweight="bold" if bold else "normal")
    return x, y, w, h


def arrow(ax, x0, y0, x1, y1, color=C_MUTE, lw=1.15, style="-|>", ls="-",
          rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
        color=color, linewidth=lw, linestyle=ls, zorder=1,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def note(ax, x, y, txt, color=C_BAD_E, fs=FS_NOTE, ha="left", style="italic",
         weight="normal"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=color,
            style=style, fontweight=weight, zorder=4)


def header(ax, x, y, txt, color=C_TXT):
    ax.text(x, y, txt, ha="left", va="center", fontsize=FS_HEAD,
            color=color, fontweight="bold", zorder=4)


def blank_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# ============================================================
# FIGURE 1 — CURRENT
# ============================================================

def figure_current():
    fig, ax = blank_ax((7.6, 9.2))

    # ---- inputs -------------------------------------------------------
    header(ax, 3, 98.0, "Inputs")
    inputs = [
        ("DEM",       "(4, 768)"),
        ("LULC",      "(4, 768)"),
        ("Soil",      "(4, 768)"),
        ("anchor L12", "(196, 768)"),
        ("S2 hist",   "(60, 4, 768)"),
        ("S1 hist",   "(40, 4, 768)"),
        ("ERA5",      "(365, 19)"),
        ("SIF",       "(50, 1)"),
        ("TWSA",      "(12, 1)"),
    ]
    n = len(inputs)
    w, gap = 9.4, 1.2
    x0 = (100 - (n * w + (n - 1) * gap)) / 2 + w / 2
    xs = [x0 + i * (w + gap) for i in range(n)]
    for (lab, shp), x in zip(inputs, xs):
        pooled = lab in ("DEM", "LULC", "Soil", "S2 hist", "S1 hist")
        box(ax, x, 92.0, w, 7.0, lab, shp,
            fc=C_BAD if pooled else C_INPUT,
            ec=C_BAD_E if pooled else C_INPUT_E)
    for x in xs:
        arrow(ax, x, 88.4, x, 85.5)

    # ---- concat + transformer ----------------------------------------
    box(ax, 50, 82.9, 92, 4.6, "concatenate  →  one flat sequence",
        "~1035 tokens x 768        spatial_start = 12", fc=C_PROC, ec=C_PROC_E)
    arrow(ax, 50, 80.5, 50, 77.7)
    box(ax, 50, 74.9, 92, 5.2, "6 x Transformer encoder layer",
        "full bidirectional self-attention over ALL 1035 tokens",
        fc=C_PROC, ec=C_PROC_E, bold=True)

    # ---- the two readouts ---------------------------------------------
    arrow(ax, 34, 72.2, 27, 67.7)
    arrow(ax, 66, 72.2, 73, 67.7)
    box(ax, 27, 63.7, 40, 7.4, "bottleneck", "ctx[:, 12:208] → (768, 14, 14)",
        fc=C_PROC, ec=C_PROC_E)
    box(ax, 73, 63.7, 44, 7.4, "context  = masked mean of every valid\nnon-CLS, non-spatial, non-pad token",
        "(768,)   model.py:736-743", fc=C_BAD, ec=C_BAD_E)
    note(ax, 73, 57.4, "SPATIALLY CONSTANT — the ENTIRE time series", ha="center", weight="bold")
    note(ax, 73, 54.9, "(ERA5 365 d, S2/S1 history, SIF, TWSA)", ha="center", weight="bold")
    note(ax, 73, 52.4, "reaches the decoder through this one vector", ha="center", weight="bold")

    # skips
    box(ax, 9.0, 51.0, 14, 6.4, "skips\nL3 / L6 / L9", "3 x (196, 768)",
        fc=C_INPUT, ec=C_INPUT_E, fs=7.6)
    arrow(ax, 9.0, 47.6, 9.0, 42.3)

    # ---- decoder -------------------------------------------------------
    arrow(ax, 27, 59.9, 27, 42.3)
    arrow(ax, 73, 50.2, 73, 42.3, ls=(0, (3, 2)))
    ax.add_patch(FancyBboxPatch((3, 27.0), 94, 15.0,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_BAD, edgecolor=C_BAD_E,
                                linewidth=1.3, zorder=2))
    ax.text(50, 39.3, "UNetDecoder", ha="center", va="center", fontsize=FS_HEAD,
            fontweight="bold", color=C_TXT, zorder=3)
    ax.text(50, 35.8, "14 → 28 → 56 → 112 → 224     bilinear upsampling",
            ha="center", va="center", fontsize=FS_BOX, color=C_TXT, zorder=3)
    ax.text(50, 32.6, "skips resized 14×14 → stage size (F.interpolate) · FiLM(context) applied "
                      "uniformly at every pixel · BatchNorm2d",
            ha="center", va="center", fontsize=FS_SHAPE, color=C_MUTE, zorder=3)
    ax.text(50, 29.5, "no input finer than 14×14 ever reaches it  →  effective resolution ≈ 160 m, not 10 m",
            ha="center", va="center", fontsize=FS_SHAPE, color=C_BAD_E,
            style="italic", zorder=3)

    # ---- head + loss ----------------------------------------------------
    arrow(ax, 50, 26.7, 50, 23.0)
    box(ax, 50, 20.2, 44, 5.0, "1 × 1 conv head", "(B, 3, 224, 224)",
        fc=C_OUT, ec=C_OUT_E)
    arrow(ax, 50, 17.5, 50, 13.8)
    box(ax, 50, 11.0, 62, 4.8, "masked Huber loss  @  pixel (112, 112)",
        "1 of 50,176 pixels supervised — the other 50,175 are unconstrained",
        fc=C_BAD, ec=C_BAD_E)

    # ---- footnote legend -------------------------------------------------
    note(ax, 50, 3.4,
         "red = pyramid-pooled to 4 nested windows  ·  97–98% of within-tile position "
         "information destroyed (§27a.2)", ha="center")

    fig.tight_layout(pad=0.4)
    return fig


# ============================================================
# FIGURE 2 — PROPOSED
# ============================================================

def figure_proposed():
    fig, ax = blank_ax((7.6, 9.2))

    # ---- inputs : block-1 group (left) and weather group (right) ---------
    b1 = [("anchor L12", "(196, 768)", C_INPUT, C_INPUT_E),
          ("dem_tok", "(196, 768)", C_NEW, C_NEW_E),
          ("lulc_tok", "(196, 768)", C_NEW, C_NEW_E),
          ("terrain stem\nstride-16", "→ (768,14,14)", C_NEW, C_NEW_E),
          ("soil + pooled\npyramid", "tile context", C_INPUT, C_INPUT_E)]
    w, gap = 11.4, 1.5
    xs = [8.7 + i * (w + gap) for i in range(len(b1))]
    for (lab, shp, fc, ec), x in zip(b1, xs):
        box(ax, x, 92.0, w, 7.4, lab, shp, fc=fc, ec=ec, fs=7.0)
        arrow(ax, x, 88.2, x, 85.6)
    box(ax, 84, 92.0, 30, 7.4, "ERA5 (365,19) · SIF (50,1)\nTWSA (12,1)",
        "9 km — constant over the tile", fc=C_INPUT, ec=C_INPUT_E, fs=7.0)
    arrow(ax, 84, 88.2, 84, 85.6)

    # ---- block 1 / block 2 encoders --------------------------------------
    box(ax, 34.5, 82.4, 63, 5.6, "BLOCK 1 · Context encoder — 6 × self-attention",
        "runs ONCE per sample, over the whole tile", fc=C_PROC, ec=C_PROC_E, bold=True, fs=8.0)
    box(ax, 84, 82.4, 30, 5.6, "BLOCK 2 · Weather encoder",
        "runs ONCE — shared, never replicated", fc=C_PROC, ec=C_PROC_E, bold=True, fs=8.0)

    arrow(ax, 24, 79.5, 18, 75.6)
    arrow(ax, 46, 79.5, 52, 75.6)
    arrow(ax, 84, 79.5, 84, 75.6)

    box(ax, 18, 72.6, 28, 5.8, "S   one context vector\nPER LOCATION", "(B, 196, 768)",
        fc=C_OUT, ec=C_OUT_E, fs=7.6)
    box(ax, 52, 72.6, 16, 5.8, "g   tile summary", "(B, 768)", fc=C_PROC, ec=C_PROC_E, fs=7.6)
    box(ax, 84, 72.6, 26, 5.8, "W   weather sequence", "(B, T, 768)",
        fc=C_PROC, ec=C_PROC_E, fs=7.6)

    # ---- three clean verticals into the processor -------------------------
    arrow(ax, 18, 69.4, 18, 64.2)
    arrow(ax, 52, 69.4, 52, 64.2)
    arrow(ax, 84, 69.4, 84, 64.2)
    note(ax, 20.0, 66.6, "varies with k", color=C_OUT_E, fs=6.8)
    note(ax, 54.0, 66.6, "constant", color=C_MUTE, fs=6.8)
    note(ax, 86.0, 66.6, "cross-attn", color=C_MUTE, fs=6.8)

    # ---- BLOCK 3 : processor ----------------------------------------------
    ax.add_patch(FancyBboxPatch((3, 47.5), 94, 16.4,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_NEW, edgecolor=C_NEW_E,
                                linewidth=1.3, zorder=2))
    ax.text(50, 61.2, "BLOCK 3 · Processor      per location k  ·  weights SHARED across all k",
            ha="center", va="center", fontsize=8.6, color=C_TXT,
            fontweight="bold", zorder=3)
    ax.text(50, 57.6, "seq$_k$  =  [  S[:, k, :]   |   g   |   s2_hist[:, :, k, :]   |   "
                      "s1_hist[:, :, k, :]  ]      ≈ 102 tokens",
            ha="center", va="center", fontsize=7.8, color=C_TXT, zorder=3)
    ax.text(50, 55.0, "varies            constant        varies                     varies",
            ha="center", va="center", fontsize=6.6, color=C_MUTE,
            family="monospace", zorder=3)
    ax.text(50, 52.1, "L × (  self-attention over seq$_k$   →   cross-attention into W  )",
            ha="center", va="center", fontsize=8.4, color=C_TXT,
            fontweight="bold", zorder=3)
    ax.text(50, 49.2, "→  h$_k$  (B, 768)          train: 1 location   ·   infer: all 196",
            ha="center", va="center", fontsize=FS_BOX, color=C_TXT, zorder=3)

    # ---- BLOCK 4 : per-pixel head -------------------------------------------
    arrow(ax, 50, 47.2, 50, 42.3)
    ax.add_patch(FancyBboxPatch((3, 26.0), 94, 16.0,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_OUT, edgecolor=C_OUT_E,
                                linewidth=1.3, zorder=2))
    ax.text(50, 39.4, "BLOCK 4 · Per-pixel head      no upsampling anywhere — nearest GATHER, "
                      "an index op",
            ha="center", va="center", fontsize=8.6, color=C_TXT,
            fontweight="bold", zorder=3)
    ax.text(50, 35.4, "input(i, j)  =  [   S[:, i//16, j//16, :]   |   raster_stack[:, :, i, j]   ]",
            ha="center", va="center", fontsize=8.2, color=C_TXT, zorder=3)
    ax.text(50, 32.4, "the token covering (i,j)                measured 10 m pixels:  "
                      "S1 VV/VH · S2 · DEM · soil · LULC",
            ha="center", va="center", fontsize=FS_SHAPE, color=C_MUTE, zorder=3)
    ax.text(50, 28.6, "shared MLP / 1×1 conv        train: 1 pixel   ·   infer: 50,176",
            ha="center", va="center", fontsize=FS_BOX, color=C_TXT, zorder=3)

    arrow(ax, 50, 25.7, 50, 21.8)
    box(ax, 50, 18.4, 66, 5.4, "(B, 3, 224, 224)",
        "every value = measured token  +  measured 10 m pixels",
        fc=C_OUT, ec=C_OUT_E, bold=True)
    arrow(ax, 50, 15.5, 50, 12.2)
    box(ax, 50, 9.0, 66, 5.0, "masked Huber loss  @  per-sample pixel index",
        "translation crop breaks the always-(112,112) position leak",
        fc=C_NEW, ec=C_NEW_E)

    # ---- footnote legend ----------------------------------------------------
    note(ax, 50, 2.4,
         "orange = new  ·  slope · curvature · TPI · TWI · HAND enter at the CONTEXT ENCODER, "
         "not at the output", color=C_NEW_E, ha="center")

    fig.tight_layout(pad=0.4)
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()

    _setup()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, fn in [("architecture_current", figure_current),
                     ("architecture_proposed", figure_proposed)]:
        fig = fn()
        for ext in ("png", "pdf"):
            p = args.out / f"{name}.{ext}"
            fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
            print(f"wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
