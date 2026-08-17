"""
Publication-quality architecture flowcharts for the soil moisture model.

Emits two standalone figures:
  figures/architecture_current.{png,pdf}   — the shipped model (cls_depth_star_reg)
  figures/architecture_proposed.{png,pdf}  — the per-location processor

Kept in step with text/architecture_per_location.txt. Boxes are annotated with
tensor shapes so the resolution argument can be read straight off the figure: the
current design's only spatial carrier is a 14x14 token grid, and every bit of the
time-series signal reaches the decoder through one spatially constant vector.

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
    fig, ax = blank_ax((7.9, 10.2))

    # ---- inputs : block-1 group (left) and weather group (right) ---------
    b1 = [("anchor L12", "(196, 768)", C_INPUT, C_INPUT_E),
          ("terrain_lo  stem\nTWI · HAND · mask", "(3,28,28) → 196", C_NEW, C_NEW_E),
          ("dem_pyr\nlulc_pyr", "(4, 768) each", C_INPUT, C_INPUT_E),
          ("soil", "(21, 74, 74)", C_INPUT, C_INPUT_E)]
    w, gap = 14.0, 1.6
    xs = [12.0 + i * (w + gap) for i in range(len(b1))]
    for (lab, shp, fc, ec), x in zip(b1, xs):
        box(ax, x, 94.5, w, 7.2, lab, shp, fc=fc, ec=ec, fs=6.9)
        arrow(ax, x, 90.9, x, 88.4)
    box(ax, 86, 94.5, 26, 7.2, "ERA5 · SIF · TWSA", "(365,19)   9 km",
        fc=C_INPUT, ec=C_INPUT_E, fs=6.9)
    arrow(ax, 86, 90.9, 86, 88.4)

    # ---- block 1 / block 2 encoders --------------------------------------
    box(ax, 38, 85.4, 60, 5.4, "BLOCK 1 · Context encoder — 6 × self-attention",
        "runs ONCE per sample, over the whole tile", fc=C_PROC, ec=C_PROC_E,
        bold=True, fs=8.0)
    box(ax, 86, 85.4, 26, 5.4, "BLOCK 2 · Weather encoder",
        "runs ONCE, shared", fc=C_PROC, ec=C_PROC_E, bold=True, fs=7.6)

    arrow(ax, 30, 82.6, 30, 79.3)
    arrow(ax, 56, 82.6, 62, 79.3)
    arrow(ax, 86, 82.6, 86, 79.3)

    box(ax, 30, 76.4, 26, 5.6, "S   one context vector\nPER LOCATION",
        "(B, 196, 768)", fc=C_OUT, ec=C_OUT_E, fs=7.5)
    box(ax, 62, 76.4, 15, 5.6, "g   tile summary", "(B, 768)",
        fc=C_PROC, ec=C_PROC_E, fs=7.5)
    box(ax, 86, 76.4, 24, 5.6, "W   weather sequence", "(B, T, 768)",
        fc=C_PROC, ec=C_PROC_E, fs=7.5)

    # ---- LST auxiliary : a side branch off S, gradient only ---------------
    arrow(ax, 20.5, 73.6, 12.5, 70.4, rad=0.18, color=C_NEW_E)
    box(ax, 11.5, 66.4, 17, 6.6, "LST head\nLinear 768→1", "(14,14) · 196 targets",
        fc=C_NEW, ec=C_NEW_E, fs=6.9)
    note(ax, 7.5, 58.0, "side branch —\nonly makes gradient",
         color=C_NEW_E, fs=6.3, ha="left")
    # long dashed run down the far left, then right into the loss box
    arrow(ax, 4.0, 63.0, 4.0, 9.4, color=C_NEW_E, ls=(0, (2.5, 2)), lw=1.0,
          style="-")
    arrow(ax, 4.0, 9.4, 8.6, 9.4, color=C_NEW_E, ls=(0, (2.5, 2)), lw=1.0)

    # ---- three verticals into the processor -------------------------------
    arrow(ax, 34, 73.6, 34, 68.2)
    arrow(ax, 62, 73.6, 62, 68.2)
    arrow(ax, 86, 73.6, 86, 68.2)
    note(ax, 35.6, 71.0, "varies with k", color=C_OUT_E, fs=6.6)
    note(ax, 63.6, 71.0, "constant", color=C_MUTE, fs=6.6)
    note(ax, 87.6, 71.0, "cross-attn", color=C_MUTE, fs=6.6)

    # ---- BLOCK 3 : processor ----------------------------------------------
    ax.add_patch(FancyBboxPatch((23, 51.0), 74, 16.2,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_NEW, edgecolor=C_NEW_E,
                                linewidth=1.3, zorder=2))
    ax.text(60, 64.6, "BLOCK 3 · Processor    per location k  ·  weights SHARED",
            ha="center", va="center", fontsize=8.4, color=C_TXT,
            fontweight="bold", zorder=3)
    ax.text(60, 61.2, "seq$_k$ = [  S[:, k, :]  |  g  |  s2_tok$_k$(60)  |  "
                      "s1_tok$_k$(40)  ]   ≈ 102 tokens",
            ha="center", va="center", fontsize=7.4, color=C_TXT, zorder=3)
    ax.text(60, 58.8, "varies      constant     varies           varies",
            ha="center", va="center", fontsize=6.4, color=C_MUTE,
            family="monospace", zorder=3)
    ax.text(60, 55.8, "L × ( self-attention over seq$_k$  →  cross-attention into W )",
            ha="center", va="center", fontsize=8.0, color=C_TXT,
            fontweight="bold", zorder=3)
    ax.text(60, 52.8, "→ h$_k$ (B, 768) @ 160 m     train: 1 location  ·  infer: all 196",
            ha="center", va="center", fontsize=7.6, color=C_TXT, zorder=3)

    # ---- BLOCK 4 : per-cell head -------------------------------------------
    arrow(ax, 60, 50.7, 60, 45.8)
    ax.add_patch(FancyBboxPatch((23, 28.5), 74, 17.0,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_OUT, edgecolor=C_OUT_E,
                                linewidth=1.3, zorder=2))
    ax.text(60, 42.9, "BLOCK 4 · Per-cell head    nearest GATHER, no upsampling",
            ha="center", va="center", fontsize=8.4, color=C_TXT,
            fontweight="bold", zorder=3)
    ax.text(60, 39.2, "out(i, j) = MLP( [  h[:, i//5, j//5, :]  |  fine[:, i, j]  ] )",
            ha="center", va="center", fontsize=8.0, color=C_TXT, zorder=3)
    ax.text(60, 36.4, "the 160 m token covering (i,j)      C = 17 MEASURED channels",
            ha="center", va="center", fontsize=6.8, color=C_MUTE, zorder=3)
    ax.text(60, 33.9, "TWI · HAND 30 m · S1 vv/vh + median 10 m · S2 10/20 m · "
                      "soil 30 m · LULC · LST",
            ha="center", va="center", fontsize=6.2, color=C_MUTE, zorder=3)
    ax.text(60, 30.8, "shared MLP / 1×1 conv      train: 1 cell  ·  infer: 4,900",
            ha="center", va="center", fontsize=7.6, color=C_TXT, zorder=3)
    note(ax, 14, 26.4, "elevation EXCLUDED — token already carries it (0.05)",
         color=C_MUTE, fs=6.3)

    arrow(ax, 60, 28.2, 60, 24.0)
    box(ax, 60, 20.6, 58, 5.4, "(B, 3, 70, 70)   @ 32 m",
        "70 = 2240/32 exactly · 5×5 cells per token", fc=C_OUT, ec=C_OUT_E,
        bold=True, fs=8.0)
    arrow(ax, 60, 17.7, 60, 12.6)

    # ---- combined loss ------------------------------------------------------
    box(ax, 52, 9.4, 84, 5.6, "L  =  masked Huber @ pixel_idx   +   λ · LST anomaly loss",
        "1 label vs 196 tokens · λ balanced by gradient, not pixel count",
        fc=C_NEW, ec=C_NEW_E, fs=8.0)

    # ---- footnote legend ----------------------------------------------------
    note(ax, 50, 2.6,
         "orange = new  ·  TWI and HAND enter at the CONTEXT ENCODER, not at the output — "
         "only there can terrain change the response to weather",
         color=C_NEW_E, ha="center", fs=6.8)

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
