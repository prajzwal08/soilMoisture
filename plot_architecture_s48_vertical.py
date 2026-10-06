"""Vertical-input variant of the baseline architecture flowchart (plot_architecture_s48.py is
kept unchanged beside it).

Every input is one row of a single left-hand column, top to bottom:
    S2 L2A | S1 RTC (asc / desc) | DEM | LULC | ERA5-Land (365 d x 18) | SIF | TWSA | soil
each row = name, image, and its tokens at their real dates on ONE shared 365-day axis
[D-364, D] (dataset._window_ints), with the age index 0..364 the model embeds (rel_pos_emb /
rel_pos_emb_hist). Rows feed their encoders (frozen TerraMind / driver MLPs / soil CNN), which
merge into one token sequence -> transformer -> U-Net decoder -> SM maps. The fine 20 m path
runs along the bottom as before.

Reads the same assets.npz as plot_architecture_s48.py:
    python plot_architecture_s48.py --prep                 # once, zarr -> assets.npz
    python plot_architecture_s48_vertical.py               # render only
"""
from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import plot_style_bw
from plot_architecture_s48 import (DBLUE, DEPTHS, FROZEN_EDGE, FROZEN_FILL, FS_B, FS_T, LBLUE,
                                   ORANGE, OUT, box, image, label, path, stack, vbox)

W, H = 194, 128                   # canvas units; 1 unit = 0.1 inch
TX0, TX1 = 24.0, 70.0             # the shared 365-day axis
LAB_X = 12.2                      # row names, right-aligned
THUMB_X, TH = 13.0, 8.0           # thumbnail column
NX = 71.5                         # count column
EX0, EX1 = 80.0, 93.0             # encoders
SQ0, SQ1 = 98.0, 112.0            # token sequence
TR0, TR1 = 116.0, 130.0           # transformer
HH = 1.4                          # half tick height


def tx(days_before):
    return TX1 - np.asarray(days_before, float) * (TX1 - TX0) / 364


def inset(fig, x0, y0, w, h):
    a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.9)
    a.set_zorder(4)
    return a


def ticks(ax, days, y, color, hh=HH, lw=1.0):
    for t in np.atleast_1d(days):
        ax.add_line(Line2D([tx(t)] * 2, [y - hh, y + hh], color=color, lw=lw, zorder=3))


def row_name(ax, y, *lines):
    step = 1.7
    y0 = y + (len(lines) - 1) * step / 2
    for k, s in enumerate(lines):
        ax.text(LAB_X, y0 - k * step, s, ha="right", va="center",
                fontsize=FS_T if k == 0 else FS_B - 1, zorder=6)


def count(ax, y, s):
    ax.text(NX, y, s, ha="left", va="center", fontsize=FS_B, zorder=6)


def static_note(ax, y, s):
    ax.text((TX0 + TX1) / 2, y, s, ha="center", va="center", fontsize=FS_B, zorder=6,
            color=FROZEN_EDGE)


def render() -> None:
    plot_style_bw.apply(globals(), "paper")
    a = np.load(OUT / "assets.npz", allow_pickle=True)
    D = datetime.strptime(str(a["day"]), "%Y-%m-%d")
    fig = plt.figure(figsize=(W / 10, H / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    Y = {"s2": 113.0, "s1": 102.5, "dem": 92.0, "lulc": 82.0, "era5": 68.0,
         "sif": 56.0, "twsa": 46.0, "soil": 36.0}
    lulc = a["lulc"]
    cm = ListedColormap(list(a["lulc_colours"]))
    lnorm = BoundaryNorm(np.arange(-.5, 11), 11)

    # ── header over the column ─────────────────────────────────────────────────────────────
    ax.text(TX0, 119.5, "365-day window ending on D", ha="left", va="center", fontsize=FS_T,
            zorder=6)
    ax.plot(TX0 + 29, 119.5, "v", ms=4.5, color=ORANGE, zorder=6)
    ax.text(TX0 + 30, 119.5, "anchor", ha="left", va="center", fontsize=FS_B, zorder=6)
    ax.text(NX, 119.5, "n", ha="left", va="center", fontsize=FS_T, zorder=6)
    ax.add_patch(Rectangle((TX0 - 2.0, 31.0), 2.0, 86.0, fc=FROZEN_FILL, ec="none",
                           hatch="////", lw=0, zorder=1))          # before D-364: never loaded
    ax.text(TX0 - 1.0, 74.0, "not loaded", rotation=90, ha="center", va="center",
            fontsize=FS_B - 2, zorder=6, bbox=dict(fc="white", ec="none", pad=0.4))
    ax.add_line(Line2D([TX1] * 2, [29.5, 117.0], color="black", lw=1.6, zorder=5))

    # ── S2 ─────────────────────────────────────────────────────────────────────────────────
    y = Y["s2"]
    row_name(ax, y, "S2 L2A", "≤ 60 scenes")
    stack(ax, a["s2_rgb"], THUMB_X, y - 4, 7.0, off=0.6)
    ticks(ax, a["h_s2_age"], y, FROZEN_EDGE)
    count(ax, y, f"{len(a['h_s2_age'])}")

    # ── S1: asc above, desc below ──────────────────────────────────────────────────────────
    y = Y["s1"]
    row_name(ax, y, "S1 RTC", "≤ 40 passes")
    stack(ax, a["s1_vv"], THUMB_X, y - 4, 7.0, off=0.6, cmap="gray", vmin=0, vmax=1)
    asc = ~a["h_s1_desc"]
    ticks(ax, a["h_s1_age"][asc], y + 2.0, FROZEN_EDGE, hh=1.2)
    ticks(ax, a["h_s1_age"][~asc], y - 2.0, FROZEN_EDGE, hh=1.2)
    count(ax, y + 2.0, f"{int(asc.sum())} asc")
    count(ax, y - 2.0, f"{int((~asc).sum())} desc")

    yanc = {"s2": Y["s2"] + HH, "s1_asc": Y["s1"] + 3.2,
            "s1_desc": Y["s1"] - 0.8}[str(a["anchor_orbit"])]
    ax.plot(tx(a["anchor_age"]), yanc + 0.5, "v", ms=4.5, color=ORANGE, zorder=6)

    # ── statics ────────────────────────────────────────────────────────────────────────────
    y = Y["dem"]
    row_name(ax, y, "DEM", "30 m")
    image(ax, a["dem_hs"], THUMB_X, y - 4, TH, TH, cmap="gray")
    static_note(ax, y, "static · one map")
    count(ax, y, "1")
    y = Y["lulc"]
    row_name(ax, y, "LULC", f"{int(a['lulc_year'])}")
    image(ax, lulc, THUMB_X, y - 4, TH, TH, cmap=cm, norm=lnorm)
    static_note(ax, y, "static · map of the year before D")
    count(ax, y, "1")

    # ── ERA5-Land: 18 columns x 365 days, each column z-scored over the window ─────────────
    y = Y["era5"]
    for k, s in enumerate(("ERA5-Land", "5 vars × daily", "min, mean, max", "+ 3 daily sums")):
        ax.text(LAB_X, y + 2.6 - k * 1.7, s, ha="right", va="center",
                fontsize=FS_T if k == 0 else FS_B - 1, zorder=6)
    e5 = np.full((365, 18), np.nan)
    e5[364 - a["era5_age"]] = a["era5"]
    z = (e5 - np.nanmean(e5, 0)) / (np.nanstd(e5, 0) + 1e-6)
    ax.imshow(z.T, extent=(tx(364.5), tx(-0.5), y - 6, y + 6), aspect="auto", cmap="Blues",
              vmin=-2, vmax=2, interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((tx(364.5), y - 6), tx(-0.5) - tx(364.5), 12, fill=False,
                           ec="black", lw=0.7, zorder=4))
    for k, g in enumerate(("T2m", "Td2m", "u10", "v10", "psurf", "P · SW↓ · LW↓")):
        yc = y + 6 - (k + 0.5) * 2.0                              # 3 rows of 12/18 each
        ax.text(TX0 - 2.6, yc, g, ha="right", va="center", fontsize=FS_B - 1, zorder=6)
    count(ax, y + 0.9, f"{len(a['era5_age'])}")
    count(ax, y - 0.9, "× 18")

    # ── SIF / TWSA ─────────────────────────────────────────────────────────────────────────
    y = Y["sif"]
    row_name(ax, y, "SIF", "≤ 50")
    t = inset(fig, THUMB_X, y - 4, TH, TH)
    t.plot(-a["sif_age"], a["sif"], "o", ms=1.6, color=DBLUE)
    t.set_xlim(-365, 8)
    ticks(ax, a["sif_age"], y, DBLUE)
    count(ax, y, f"{len(a['sif_age'])}")
    y = Y["twsa"]
    row_name(ax, y, "TWSA", "≤ 12, monthly")
    t = inset(fig, THUMB_X, y - 4, TH, TH)
    t.step(-a["twsa_age"], a["twsa"], where="post", color=DBLUE, lw=1.0)
    t.set_xlim(-365, 0)
    ticks(ax, a["twsa_age"], y, DBLUE, lw=1.6)
    count(ax, y, f"{len(a['twsa_age'])}")

    y = Y["soil"]
    row_name(ax, y, "Soil", "21 ch")
    image(ax, a["soc"], THUMB_X, y - 4, TH, TH, cmap="YlOrBr")
    static_note(ax, y, "static · one stack")
    count(ax, y, "1")

    # ── shared axis: date and the age index the model embeds ───────────────────────────────
    path(ax, [(TX0, 29.5), (TX1, 29.5)], head=False, lw=1.0)
    for db in (364, 270, 180, 90, 0):
        x = float(tx(db))
        ax.add_line(Line2D([x, x], [29.0, 29.5], color="black", lw=1.0, zorder=3))
        dlab = "D" if db == 0 else f"D−{db}"
        ax.text(x, 28.2, dlab, ha="center", va="center", fontsize=FS_B, zorder=6)
        ax.text(x, 26.6, f"{364 - db}", ha="center", va="center", fontsize=FS_B, zorder=6)
    ax.text(LAB_X, 28.2, "date", ha="right", va="center", fontsize=FS_B, zorder=6)
    ax.text(LAB_X, 26.6, "age token", ha="right", va="center", fontsize=FS_B, zorder=6)
    ax.text(TX1 + 1.2, 28.2, f"= {D:%Y-%m-%d}", ha="left", va="center", fontsize=FS_B,
            zorder=6)

    # ── encoders ───────────────────────────────────────────────────────────────────────────
    for yy in (Y["s2"], Y["s1"], Y["dem"], Y["lulc"]):
        path(ax, [(77.0, yy), (EX0, yy)])
    box(ax, EX0, 78, EX1, 117, None, kind="frozen")
    for k, s in enumerate(("TerraMind", "ViT-B", "frozen · offline", "layer 12")):
        ax.text((EX0 + EX1) / 2, 114.6 - k * 1.85, s, ha="center", va="center",
                fontsize=FS_T if k < 2 else FS_B, zorder=3)
    gx, gy, gs = EX0 + 1.5, 93.0, 10.0                        # anchor tokens, 14x14 PCA
    image(ax, np.kron(a["tok_rgb"], np.ones((8, 8, 1))), gx, gy, gs, gs)
    g = gs / 14
    for k in range(1, 14):
        ax.add_line(Line2D([gx + k * g] * 2, [gy, gy + gs], color="white", lw=0.25, zorder=5))
        ax.add_line(Line2D([gx, gx + gs], [gy + k * g] * 2, color="white", lw=0.25, zorder=5))
    for k, s in enumerate(("196 × 768 · 160 m", "pyramid × 4", "anchor 196")):
        ax.text((EX0 + EX1) / 2, 91.1 - k * 1.85, s, ha="center", va="center",
                fontsize=FS_B - 1, zorder=3)
    box(ax, EX0, 74.3, EX1, 77.3, ["coarse drop 0.3"], kind="plain", dashed=True, fs=FS_B - 1)

    for yy in (Y["era5"], Y["sif"], Y["twsa"]):
        path(ax, [(77.0, yy), (EX0, yy)])
    box(ax, EX0, 45, EX1, 73, ["Driver MLPs", "1 → 256 → 768", "ERA5 18 → 768",
                               "SIF 1 → 768", "TWSA 1 → 768"])
    box(ax, EX0, 40.8, EX1, 43.8, ["ERA5 drop 0.5"], kind="plain", dashed=True, fs=FS_B - 1)
    path(ax, [(77.0, Y["soil"]), (EX0, Y["soil"])])
    box(ax, EX0, 32.5, EX1, 39.5, ["Soil CNN", "21 ch → 4"])

    # ── one token sequence ─────────────────────────────────────────────────────────────────
    path(ax, [(EX1, 97), (SQ0, 97)])
    path(ax, [(EX1, 59), (SQ0, 59)])
    path(ax, [(EX1, 36), (95.5, 36), (95.5, 54), (SQ0, 54)])
    box(ax, SQ0, 52, SQ1, 100, None)
    xc = (SQ0 + SQ1) / 2
    label(ax, xc, 97.6, "Token sequence", fs=FS_T)
    seq = [("Depth CLS", 3, "train"), ("DEM · LULC", 8, "frozen"), ("Soil", 4, "train"),
           ("Anchor", 196, "frozen"), ("S2 history", 240, "frozen"), ("S1 history", 160, "frozen"),
           ("ERA5", 365, "train"), ("SIF", 50, "train"), ("TWSA", 12, "train")]
    yy = 94.6
    for name, n, kind in seq:
        ax.add_patch(Rectangle((SQ0 + 1.0, yy - 0.8), 1.6, 1.6,
                               fc=FROZEN_FILL if kind == "frozen" else LBLUE,
                               ec="black", lw=0.6, zorder=3))
        ax.text(SQ0 + 3.2, yy, name, ha="left", va="center", fontsize=FS_B, zorder=3)
        ax.text(SQ1 - 0.7, yy, f"{n}", ha="right", va="center", fontsize=FS_B, zorder=3)
        yy -= 2.55
    ax.add_line(Line2D([SQ0 + 1, SQ1 - 0.7], [yy + 1.1] * 2, color="black", lw=0.8, zorder=3))
    ax.text(SQ1 - 0.7, yy - 0.4, "1038", ha="right", va="center", fontsize=FS_B, zorder=3)
    ax.text(SQ0 + 3.2, yy - 0.4, "× 768", ha="left", va="center", fontsize=FS_B, zorder=3)
    for k, s in enumerate(("every token +", "age 0–364", "+ DOY · modality", "+ position")):
        label(ax, xc, 63.0 - k * 1.85, s)

    path(ax, [(SQ1, 76), (TR0, 76)])
    box(ax, TR0, 52, TR1, 100, ["Transformer", "3 layers", "d = 768", "12 heads", "full self-",
                                "attention", "pre-norm"], title_n=1)

    # ── fine 20 m path along the bottom ────────────────────────────────────────────────────
    fx, fy, fs = [3.0, 12.0, 21.0, 30.0], 8.0, 8.0
    image(ax, a["s2_rgb"][-1][::2, ::2], fx[0], fy, fs, fs)
    image(ax, a["s1_vv"][-1][::2, ::2], fx[1], fy, fs, fs, cmap="gray", vmin=0, vmax=1)
    image(ax, a["dem_hs"][::2, ::2], fx[2], fy, fs, fs, cmap="gray")
    image(ax, lulc, fx[3], fy, fs, fs, cmap=cm, norm=lnorm)
    for x, s_ in zip(fx, ["S2 on D", "S1 on D", "DEM 20 m", "LULC 10 m"]):
        label(ax, x + fs / 2, 17.6, s_)
    for x in fx:
        path(ax, [(x + fs / 2, fy), (x + fs / 2, 5.0)], head=False)
    path(ax, [(fx[0] + fs / 2, 5.0), (48.5, 5.0), (48.5, 7.0)])
    box(ax, 42, 7, 55, 19, ["Stems 3×3", "S2 12 → 16", "S1 5 → 8", "DEM 2 → 4", "LULC 8 → 4"])
    box(ax, 42, 20.5, 55, 23.5, ["modality drop 0.2"], kind="plain", dashed=True, fs=FS_B - 1)
    path(ax, [(55, 13), (59, 13)])
    vbox(ax, 59, 6, 64, 20, "E1 · 32 @ 112")
    path(ax, [(64, 13), (67, 13)])
    vbox(ax, 67, 7, 72, 19, "E2 · 64 @ 56")
    path(ax, [(72, 13), (75, 13)])
    vbox(ax, 75, 8.5, 80, 17.5, "E3 · 128 @ 28")
    label(ax, 69.5, 22.0, "GroupNorm CNN")

    # ── decoder ────────────────────────────────────────────────────────────────────────────
    vbox(ax, 136, 9.5, 141, 16.5, "512 @ 14")
    path(ax, [(141, 13), (143, 13)])
    vbox(ax, 143, 8.5, 149, 17.5, "256 @ 28")
    path(ax, [(149, 13), (151, 13)])
    vbox(ax, 151, 7, 158, 19, "128 @ 56")
    path(ax, [(158, 13), (160, 13)])
    vbox(ax, 160, 6, 167, 20, "z · 64 @ 112")
    ax.text(121.5, 22.0, "U-Net decoder", ha="left", va="center", fontsize=FS_B, zorder=6)
    ax.text(121.5, 20.3, "bilinear ×2", ha="left", va="center", fontsize=FS_B, zorder=6)
    path(ax, [(77.5, 8.5), (77.5, 5.0), (146, 5.0), (146, 8.5)])     # skips, nested
    path(ax, [(69.5, 7), (69.5, 3.0), (154.5, 3.0), (154.5, 7)])
    path(ax, [(61.5, 6), (61.5, 1.0), (163.5, 1.0), (163.5, 6)])
    label(ax, 105, 6.4, "skips (FiLM by context)")

    # transformer outputs
    path(ax, [(120, 52), (120, 13), (136, 13)])
    label(ax, 119, 40.0, "bottleneck", ha="right")
    label(ax, 119, 38.2, "768 @ 14×14", ha="right")
    path(ax, [(125, 52), (125, 26), (163.5, 26)], head=False)
    for xd, ytop in ((146, 17.5), (154.5, 19), (163.5, 20)):
        path(ax, [(xd, 26), (xd, ytop)])
    label(ax, 145, 27.4, "context 768 (FiLM)")
    path(ax, [(TR1, 76), (133, 76), (133, 30), (170, 30), (170, 17), (172, 17)])
    label(ax, 152, 31.4, "depth CLS 3 × 768")

    # ── heads, maps, loss ──────────────────────────────────────────────────────────────────
    path(ax, [(167, 13), (172, 13)])
    box(ax, 172, 7, 186, 19, ["3 depth", "heads", "FiLM(z, CLS)", "+ 1×1 conv"], title_n=2)
    sm = a["sm"]
    vmin, vmax = np.percentile(sm, 1), np.percentile(sm, 99)
    path(ax, [(177, 19), (177, 33)])
    for d, (ym, dep) in enumerate(zip((57, 45, 33), DEPTHS)):
        image(ax, sm[d], 172, ym, 10, 10, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.add_patch(plt.Circle((172 + 10 * 56.5 / 112, ym + 10 * (1 - 56.5 / 112)), 0.55,
                                fill=False, ec=ORANGE, lw=1.6, zorder=6))
        ax.text(172.4, ym + 9.4, dep, ha="left", va="top", fontsize=FS_B - 1, zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.6))
    cax = fig.add_axes([183.2 / W, 33 / H, 1.0 / W, 34 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), "YlGnBu"), cax=cax)
    cb.set_ticks(np.round(np.linspace(vmin, vmax, 3), 2))
    cb.ax.tick_params(labelsize=FS_B, length=2, pad=1)
    cb.set_label("m³/m³", fontsize=FS_B, labelpad=1)
    path(ax, [(177, 67), (177, 71)])
    box(ax, 170, 71, 186, 82, ["Huber loss", "pixel (56, 56)", "vs ISMN", "training only"],
        kind="loss", dashed=True, title_n=1)

    # ── legend row ─────────────────────────────────────────────────────────────────────────
    lx, ly = 2, H - 4.4
    for kind, txt, dashed in (("frozen", "frozen (TerraMind, offline)", False),
                              ("train", "trainable (26.7 M parameters)", False),
                              ("plain", "training only", True),
                              ("loss", "supervision", True)):
        box(ax, lx, ly, lx + 3.4, ly + 2.4, None, kind=kind, dashed=dashed)
        ax.text(lx + 4.2, ly + 1.2, txt, ha="left", va="center", fontsize=FS_B)
        lx += 4.2 + 0.62 * len(txt) + 4
    ax.text(W - 2, ly + 1.2, f"TxSON CR200-18 · day D = {a['day']} · 2.24 km tile",
            ha="right", va="center", fontsize=FS_B)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"architecture_s48_vertical.{ext}", dpi=600)
    print(f"wrote {OUT}/architecture_s48_vertical.{{png,pdf}}")


if __name__ == "__main__":
    render()
