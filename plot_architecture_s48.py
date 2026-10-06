"""Publication flowchart of the baseline SM model (nolst_L3_wu200_20261005), left to right, with
real imagery from one tile (TxSON CR200-18) on one day D.

Three lanes, orthogonal arrows only:
    A  drivers    ERA5-Land / SIF / TWSA / soil  -> MLPs + soil CNN          -> token sequence
    B  satellite  S2 / S1 / DEM / LULC history   -> frozen TerraMind (L12)   -> token sequence
       token sequence (1038) -> 3-layer transformer -> bottleneck 14x14 / context / depth CLS
    C  fine 20 m  latest S2 / S1 / DEM / LULC    -> stems -> CNN E1-E3 -> skips into the U-Net
       decoder 14 -> 28 -> 56 -> 112 -> 3 depth heads -> SM maps -> Huber at pixel (56,56)

Every number on the figure was traced from model.py / dataset.py / train.py and the run's
CONFIG line (logs/lst_tmean_diff/train_nolst_L3_wu200_27575211.out:288).

    python plot_architecture_s48.py --prep      # reads the zarr stores -> assets.npz
    python plot_architecture_s48.py --render    # draws from assets.npz only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

import plot_style_bw

REPO   = Path(__file__).resolve().parent
OUT    = REPO / "figures" / "architecture_s48"
TILE   = "ISMN_TxSON_CR200-18"
MAPS   = REPO / "eval_output" / "effres_txson_smoke"        # baseline best.pt, CR200-18 only
DEPTHS = ["0–10 cm", "10–30 cm", "30–100 cm"]

# palette H (feedback_paper_figure_style)
DBLUE, LBLUE, ORANGE = "#08519C", "#6BAED6", "#E6550D"
TRAIN_FILL  = "#D6E7F4"           # opaque tint of LBLUE
FROZEN_FILL, FROZEN_EDGE = "#D9D9D9", "#4D4D4D"
LOSS_FILL   = "#FBDCCB"           # opaque tint of ORANGE
INPUT_EDGE, INPUT_FILL = "#238B45", "#EEF7F0"   # input frames: green edge, faint green tint
MASK        = "#7F7F7F"           # cloud-mask arrows (dashed): a mask, not data
LINE        = "black"
FS_T, FS_B  = 11, 10              # box titles / box body
W, H        = 178, 114            # canvas units; 1 unit = 0.1 inch


# ══ prep ══════════════════════════════════════════════════════════════════════════════════
def _ymd(i) -> datetime:
    return datetime.strptime(str(int(i)), "%Y%m%d")


def prep(day: str) -> None:
    import pandas as pd
    from plot_tile_context import LULC_COLOURS, hillshade, open_raw, open_tokens, s2_rgb, soil_layer
    from visualize_embeddings import _dstr, cloud_frac_by_date

    D = datetime.strptime(day, "%Y-%m-%d")
    Dint = int(D.strftime("%Y%m%d"))
    raw, tok = open_raw(TILE), open_tokens(TILE)
    a = {"day": day}

    # S2: the 3 most recent clear scenes on or before D (the last one = the fine path's scene)
    frac = cloud_frac_by_date(tok)
    rdates = list(_dstr(raw["s2/dates"][:]))
    clear = sorted(d for d in rdates if d.isdigit() and int(d) <= Dint and frac.get(d, 1) < .02)
    s2d = clear[-3:]
    a["s2_rgb"] = np.stack([s2_rgb(raw, rdates.index(d)) for d in s2d])
    a["s2_dates"] = np.array(s2d)

    # anchor tokens = L12 of the latest clear S2 scene -> PCA to RGB on the 14x14 grid
    tdates = list(_dstr(tok["s2/dates"][:]))
    l12 = np.asarray(tok["s2/l12"][tdates.index(s2d[-1])], np.float32)       # (196, 768)
    x = l12 - l12.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pc = x @ vt[:3].T
    lo, hi = np.percentile(pc, 2, axis=0), np.percentile(pc, 98, axis=0)
    a["tok_rgb"] = np.clip((pc - lo) / (hi - lo), 0, 1).reshape(14, 14, 3)

    # S1: 3 most recent passes (either orbit) on or before D, VV dB
    s1 = []
    for orb in ("s1_asc", "s1_desc"):
        if f"{orb}/dates" not in raw:
            continue
        for j, d in enumerate(_dstr(raw[f"{orb}/dates"][:])):
            if d[:8].isdigit() and int(d[:8]) <= Dint:
                s1.append((d[:8], orb, j))
    s1 = sorted(s1)[-3:]
    vv = []
    for _, orb, j in s1:
        v = np.clip(np.asarray(raw[f"{orb}/data"][j, 0], np.float32), -20, 0)
        lo_, hi_ = np.nanpercentile(v, (2, 98))
        vv.append(np.clip((v - lo_) / (hi_ - lo_), 0, 1))
    a["s1_vv"] = np.nan_to_num(np.stack(vv))
    a["s1_dates"] = np.array([d for d, _, _ in s1])

    dem = np.asarray(raw["dem/data"][0], np.float32)
    a["dem"], a["dem_hs"] = dem, hillshade(dem)
    years = list(raw["lulc/years"][:])
    prev = [y for y in years if y < D.year]                  # dataset.py: latest year before D's
    li = years.index(prev[-1] if prev else years[0])
    a["lulc"] = np.asarray(raw["lulc/data"][li], np.uint8)
    a["lulc_year"] = years[li]
    a["lulc_colours"] = np.array([LULC_COLOURS.get(i, "#999999") for i in range(11)])
    a["soc"] = soil_layer(TILE, 3)

    # the 365-day window the transformer sees, selected exactly as dataset.py does it:
    # _window_ints -> _candidates (nvalid > 0) -> most recent MAX_S2 / MAX_S1; select_anchor rule
    from dataset import (HIST_ORBITS, ANCHOR_ORBIT_ID, MAX_S1, MAX_S2, N_TOKENS, _candidates,
                         _cm_token_mask, _window_ints)
    cache = {}
    cm_idx = {d: i for i, d in enumerate(_dstr(tok["cm/dates"][:]))} if "cm/masks" in tok else {}
    cm_arr = tok["cm/masks"][:] if cm_idx else None
    for orb in HIST_ORBITS:
        if f"{orb}/dates" not in tok:
            continue
        dd = list(_dstr(tok[f"{orb}/dates"][:]))
        if orb == "s2":
            cm = np.full((len(dd), 224, 224), 255, np.uint8)
            for i, d in enumerate(dd):
                if d in cm_idx:
                    cm[i] = cm_arr[cm_idx[d], :224, :224]
            tm = _cm_token_mask(cm)
        elif f"{orb}/token_mask" in tok:
            tm = np.asarray(tok[f"{orb}/token_mask"][:]).astype(bool)
        else:
            tm = np.zeros((len(dd), 14, 14), bool)
        cache[f"{orb}_nvalid"] = tm.reshape(len(dd), -1).sum(1)
        cache[f"{orb}_date_ints"] = np.array([int(d[:8]) for d in dd], np.int32)
    yr, doy = D.year, D.timetuple().tm_yday
    w0, w1 = _window_ints(yr, doy)
    a["win"] = np.array([w0, w1])
    s2h = sorted(_candidates(cache, ("s2",), w0, w1))[-MAX_S2:]
    s1h = sorted(_candidates(cache, ("s1_asc", "s1_desc"), w0, w1))[-MAX_S1:]
    a["h_s2_age"] = np.array([(D - _ymd(c[0])).days for c in s2h])
    a["h_s1_age"] = np.array([(D - _ymd(c[0])).days for c in s1h])
    a["h_s1_desc"] = np.array([c[1] == "s1_desc" for c in s1h])
    cands = _candidates(cache, HIST_ORBITS, w0, w1)
    clear = [c for c in cands if c[3] == N_TOKENS]
    anc = max(clear or cands, key=lambda c: (c[0], -ANCHOR_ORBIT_ID[c[1]]))
    a["anchor_age"], a["anchor_orbit"] = (D - _ymd(anc[0])).days, anc[1]

    # cloud-mask panel: every S2 scene goes through SEnSeIv2 -> shown as a stack of the 3 most
    # recent scenes in the window with 5-70 % cloud/shadow (so the grey is visible), each with
    # its 14x14 token mask exactly as dataset._cm_token_mask computes it
    if cm_idx:
        picks = []
        for d in rdates:
            if d.isdigit() and w0 <= int(d) <= w1 and d in cm_idx:
                cmk = cm_arr[cm_idx[d], :224, :224]
                f = np.isin(cmk, [3, 4, 5]).mean()                # thin / thick cloud, shadow
                if 0.05 <= f <= 0.7:
                    picks.append((d, cmk))
        picks = sorted(picks)[-3:]
        if picks:
            a["cm_dates"] = np.array([d for d, _ in picks])
            a["cm_rgbs"] = np.stack([s2_rgb(raw, rdates.index(d)) for d, _ in picks])
            a["cm_toks"] = _cm_token_mask(np.stack([c for _, c in picks]))

    # drivers in the same window [D-364, D]
    def window(vals, dints):
        dints = np.asarray(dints)
        m = (dints >= w0) & (dints <= w1)
        return np.asarray(vals)[m], np.array([(D - _ymd(t)).days for t in dints[m]])
    era5 = np.asarray(tok["era5/values18"][:], np.float32)
    v, age = window(era5, tok["era5/date_ints"][:])
    a["era5"], a["era5_age"] = v, age
    # context around the window (vertical figure): everything on [D-455, D+120], window or not.
    # age = days before D, negative = after D. Out-of-window and capped values are drawn grey.
    c0, c1 = int((D - timedelta(days=455)).strftime("%Y%m%d")), \
             int((D + timedelta(days=120)).strftime("%Y%m%d"))
    a["ctx"] = np.array([455, -120])
    cs2 = _candidates(cache, ("s2",), c0, c1)
    cs1 = _candidates(cache, ("s1_asc", "s1_desc"), c0, c1)
    a["c_s2_age"] = np.array([(D - _ymd(c[0])).days for c in cs2])
    a["c_s1_age"] = np.array([(D - _ymd(c[0])).days for c in cs1])
    a["c_s1_desc"] = np.array([c[1] == "s1_desc" for c in cs1])
    ei = np.asarray(tok["era5/date_ints"][:])
    m = (ei >= c0) & (ei <= c1)
    a["c_era5"] = era5[m]
    a["c_era5_age"] = np.array([(D - _ymd(t)).days for t in ei[m]])

    def ctx_mid(dints, doys):                  # the date the dataset stamps (year + doy)
        dints, doys = np.asarray(dints), np.asarray(doys)
        dd = np.array([datetime(int(i) // 10000, 1, 1) + timedelta(days=int(j) - 1)
                       for i, j in zip(dints, doys)])
        return np.array([(D - t).days for t in dd])
    for key in ("sif", "twsa"):
        age = ctx_mid(tok[f"{key}/date_ints"][:], tok[f"{key}/doys"][:])
        a[f"c_{key}_age"] = age[(age <= 455) & (age >= -120)]

    # SIF / TWSA through the dataset's own slicers: last MAX_SIF values; TWSA month usable only
    # from stamp + 45 d, placed at its mid-month doy. age = days before D = 364 - rel_pos.
    from dataset import _load_zarr_sif, _load_zarr_twsa, load_sif_rolling, load_twsa_rolling
    for key, loader, slicer in (("sif", _load_zarr_sif, load_sif_rolling),
                                ("twsa", _load_zarr_twsa, load_twsa_rolling)):
        vals, _, rel, valid = slicer(loader(tok), yr, doy)
        a[key] = vals[valid, 0].numpy()
        a[f"{key}_age"] = 364 - rel[valid].numpy()

    # predicted SM maps on D (plain pass of the baseline, effres smoke)
    idx = pd.read_parquet(MAPS / "index_L0.parquet")
    idx = idx[idx["tile"] == TILE]
    doy = D.timetuple().tm_yday
    k = (idx["year"] == D.year) & (idx["doy"] == doy)
    if not k.any():
        raise SystemExit(f"{day} not in {MAPS}/index_L0.parquet; pick another --date")
    i = int(idx.loc[k, "idx"].iloc[0])
    a["sm"] = np.asarray(np.load(MAPS / "maps_L0.npy", mmap_mode="r")[i], np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "assets.npz", **a)
    print(f"S2 {s2d}  S1 {list(a['s1_dates'])}  LULC {a['lulc_year']}  "
          f"ERA5 {len(a['era5'])} d  SIF {len(a['sif'])}  TWSA {len(a['twsa'])}  "
          f"SM range {a['sm'].min():.3f}-{a['sm'].max():.3f}")
    print(f"window {w0}-{w1}  history S2 {len(s2h)}  S1 {len(s1h)} "
          f"(desc {int(a['h_s1_desc'].sum())})  anchor {anc[1]} {anc[0]} "
          f"(age {a['anchor_age']} d before D)")


# ══ drawing helpers ═══════════════════════════════════════════════════════════════════════
def box(ax, x0, y0, x1, y1, lines, kind="train", dashed=False, title_n=1, fs=None):
    fc, ec = {"train": (TRAIN_FILL, DBLUE), "frozen": (FROZEN_FILL, FROZEN_EDGE),
              "loss": (LOSS_FILL, ORANGE), "plain": ("white", "black"),
              "emph": (LBLUE, DBLUE)}[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec,
                                lw=1.4, ls=(0, (4, 2)) if dashed else "-", zorder=2))
    if lines:
        n = len(lines)
        step = 1.85
        ytop = (y0 + y1) / 2 + (n - 1) * step / 2
        for k, s in enumerate(lines):
            ax.text((x0 + x1) / 2, ytop - k * step, s, ha="center", va="center",
                    fontsize=fs or (FS_T if k < title_n else FS_B), zorder=3)


def vbox(ax, x0, y0, x1, y1, text, kind="train"):
    """Narrow block (CNN / decoder stage) with its label rotated along the height."""
    box(ax, x0, y0, x1, y1, None, kind)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, rotation=90, ha="center", va="center",
            fontsize=FS_B, zorder=3, linespacing=1.0)


def path(ax, pts, head=True, lw=1.4, color=LINE, dashed=False):
    """Orthogonal polyline; arrowhead on the last segment. Refuses diagonals.
    dashed=True: the whole line dashed, with a short solid stub carrying a solid head."""
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        assert xa == xb or ya == yb, f"diagonal segment {(xa, ya)} -> {(xb, yb)}"
    xs, ys = zip(*pts)
    if dashed:
        ax.add_line(Line2D(xs, ys, lw=lw, color=color, ls=(0, (3, 2)), zorder=1))
        if head:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            n = max(abs(x1 - x0), abs(y1 - y0))
            stub = (x1 - 0.5 * (x1 - x0) / n, y1 - 0.5 * (y1 - y0) / n)
            ax.annotate("", xy=pts[-1], xytext=stub, zorder=1,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                        lw=lw, color=color, shrinkA=0, shrinkB=0))
        return
    if head:
        ax.add_line(Line2D(xs[:-1], ys[:-1], lw=lw, color=color, zorder=1,
                           solid_capstyle="butt"))
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                    lw=lw, color=color, shrinkA=0, shrinkB=0))
    else:
        ax.add_line(Line2D(xs, ys, lw=lw, color=color, zorder=1))


def image(ax, img, x0, y0, w, h, **kw):
    ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), zorder=4, interpolation="nearest", **kw)
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="black", lw=0.9, zorder=5))


def stack(ax, imgs, x0, y0, s, off=0.8, **kw):
    """Oldest at the back (upper right), latest at the front (lower left)."""
    n = len(imgs)
    for k, im in enumerate(imgs):
        o = (n - 1 - k) * off
        image(ax, im, x0 + o, y0 + o, s, s, **kw)


def inset(fig, x0, y0, w, h):
    a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.9)
    a.set_zorder(4)
    return a


def label(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    ax.text(x, y, s, va="center", fontsize=kw.pop("fs", FS_B), zorder=6, **kw)


def open_gaps(fig, ax, s0=86.35, d0=7.0, s1=58.75, d1=4.0, s2=27.6, d2=7.0):
    """Open blank bands between the three lanes after everything is drawn.

    Three sections, per the user: upper = the 365-day timeline band (above s0, up d0 + d1),
    middle = drivers + satellite + transformer, lower = fine 20 m lane + decoder (below s2).
    Everything below s2 (fine 20 m lane, CNN, decoder, heads) moves down by d2. On the left
    (x <= 86, up to the token-sequence box) everything above s1 (driver lane) moves up by d1,
    and so does everything above y = 79 (timeline band, legend). The token sequence, the
    transformer and the SM maps keep their shape; lines and boxes that cross a gap stretch.
    """
    def Y(x, y):
        if y < s2:                                   # lower: fine 20 m lane + decoder
            return y - d2
        if y > s0:                                   # upper: the 365-day timeline band
            return y + d1 + d0
        if y > s1 and (x <= 86.01 or y > 79):
            return y + d1
        return y

    for ln in ax.lines:
        xs, ys = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        ln.set_ydata([Y(x, y) for x, y in zip(xs, ys)])
    for p in ax.patches:
        if isinstance(p, matplotlib.patches.Ellipse):
            cx, cy = p.center
            p.center = (cx, Y(cx, cy))
        elif hasattr(p, "get_bounds") or isinstance(p, (Rectangle, FancyBboxPatch)):
            x0, y0, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            xc = x0 + w / 2
            y0n, y1n = Y(xc, y0), Y(xc, y0 + h)
            p.set_bounds(x0, y0n, w, y1n - y0n)
    for t in ax.texts:
        if isinstance(t, matplotlib.text.Annotation):
            (x, y), (xa, ya) = t.xy, t.xyann
            t.xy, t.xyann = (x, Y(x, y)), (xa, Y(xa, ya))
        else:
            x, y = t.get_position()
            t.set_position((x, Y(x, y)))
    for im in ax.images:
        x0, x1, y0, y1 = im.get_extent()
        xc = (x0 + x1) / 2
        im.set_extent((x0, x1, Y(xc, y0), Y(xc, y1)))
    Hn = H + d0 + d1 + d2
    others = [a_ for a_ in fig.axes if a_ is not ax]
    boxes = []
    for a_ in others:                          # read positions in the OLD figure fractions
        b = a_.get_position()
        x0, x1, y0, y1 = b.x0 * W, b.x1 * W, b.y0 * H, b.y1 * H
        xc = (x0 + x1) / 2
        boxes.append((x0, x1, Y(xc, y0), Y(xc, y1)))
    fig.set_size_inches(W / 10, Hn / 10)
    ax.set_ylim(-d2, H + d1 + d0)
    for a_, (x0, x1, y0, y1) in zip(others, boxes):
        a_.set_position([x0 / W, (y0 + d2) / Hn, (x1 - x0) / W, (y1 - y0) / Hn])


# ══ render ════════════════════════════════════════════════════════════════════════════════
def render() -> None:
    plot_style_bw.apply(globals(), "paper")
    a = np.load(OUT / "assets.npz", allow_pickle=True)
    fig = plt.figure(figsize=(W / 10, H / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # ── the 365-day window: every token at its real date ───────────────────────────────────
    # Sliding window: the axis runs past both ends (D-455 ... D+120). dataset.py keeps only
    # [D-364, D] (_window_ints), framed white; everything the model is NOT given is grey
    # (outside the window, SIF over its cap of 50, TWSA not yet released). Each token is tagged
    # with age = rel_pos, 0 = D-364 ... 364 = D (rel_pos_emb / rel_pos_emb_hist).
    X0, X1, CB, CA = 22.0, 104.0, 455, -120
    def tx(days_before):
        return X0 + (CB - np.asarray(days_before, float)) * (X1 - X0) / (CB - CA)
    D = datetime.strptime(str(a["day"]), "%Y-%m-%d")
    GREY, WASH = "#B0B0B0", "#EDEDED"
    rows_t = [("S2 history", 101.0), ("S1 ascending", 99.1), ("S1 descending", 97.2),
              ("ERA5-Land", 95.3), ("SIF", 93.4), ("TWSA", 91.5)]
    yr = dict(rows_t)
    hh = 0.75                                                  # half row height
    for name, y in rows_t:
        ax.text(16.6, y, name, ha="right", va="center", fontsize=FS_B, zorder=6)
    wa, wb = float(tx(364.5)), float(tx(-0.5))
    ax.add_patch(Rectangle((X0, 90.4), wa - X0, 11.6, fc=WASH, ec="none", zorder=0))
    ax.add_patch(Rectangle((wb, 90.4), X1 - wb, 11.6, fc=WASH, ec="none", zorder=0))
    ax.add_patch(Rectangle((wa, 90.4), wb - wa, 11.6, fill=False, ec="black", lw=1.4,
                           zorder=5))
    def ticks(days, y, color, lw=1.0):
        for t in np.atleast_1d(days):
            ax.add_line(Line2D([tx(t)] * 2, [y - hh, y + hh], color=color, lw=lw, zorder=3))
    def series(all_age, given_age, y, color, lw=1.0):
        given = set(np.atleast_1d(given_age).tolist())
        ticks([t for t in np.atleast_1d(all_age) if t not in given], y, GREY, lw)
        ticks(list(given), y, color, lw)
    asc, casc = ~a["h_s1_desc"], ~a["c_s1_desc"]
    series(a["c_s2_age"], a["h_s2_age"], yr["S2 history"], FROZEN_EDGE)
    series(a["c_s1_age"][casc], a["h_s1_age"][asc], yr["S1 ascending"], FROZEN_EDGE)
    series(a["c_s1_age"][~casc], a["h_s1_age"][~asc], yr["S1 descending"], FROZEN_EDGE)
    anc_row = {"s2": "S2 history", "s1_asc": "S1 ascending",
               "s1_desc": "S1 descending"}[str(a["anchor_orbit"])]
    ax.plot(tx(a["anchor_age"]), yr[anc_row] + hh + 0.45, "v", ms=4.5, color=ORANGE, zorder=7,
            clip_on=False)
    t2m = np.full(CB - CA + 1, np.nan)
    t2m[CB - a["c_era5_age"]] = a["c_era5"][:, 0]
    ax.imshow(t2m[None], extent=(tx(CB + 0.5), tx(CA - 0.5), yr["ERA5-Land"] - hh,
                                 yr["ERA5-Land"] + hh), aspect="auto", cmap="Blues",
              interpolation="nearest", zorder=2)
    for x0, x1 in ((X0, wa), (wb, X1)):                       # outside the window: washed out
        ax.add_patch(Rectangle((x0, yr["ERA5-Land"] - hh), x1 - x0, 2 * hh, fc="white",
                               alpha=0.7, ec="none", zorder=2.5))
    series(a["c_sif_age"], a["sif_age"], yr["SIF"], DBLUE)
    series(a["c_twsa_age"], a["twsa_age"], yr["TWSA"], DBLUE, lw=1.6)
    for name, n in (("S2 history", len(a["h_s2_age"])), ("S1 ascending", int(asc.sum())),
                    ("S1 descending", int((~asc).sum())), ("ERA5-Land", len(a["era5_age"])),
                    ("SIF", len(a["sif_age"])), ("TWSA", len(a["twsa_age"]))):
        ax.text(X1 + 1.6, yr[name], f"{n}", ha="left", va="center", fontsize=FS_B, zorder=6)
    # brackets: the window for D, and (dashed) the window for D+30 -- it slides
    for d_end, yb, dashed in ((0, 103.0, False), (-30, 106.4, True)):
        xa, xb = float(tx(d_end + 364.5)), float(tx(d_end - 0.5))
        ax.add_line(Line2D([xa, xa, xb, xb], [yb - 0.7, yb, yb, yb - 0.7], color="black",
                           lw=1.1 if dashed else 1.4, ls=(0, (4, 2)) if dashed else "-",
                           zorder=6))
    ax.text(tx(182), 104.2, "window for day D: the last 365 days", ha="center", va="center",
            fontsize=FS_T, zorder=6)
    ax.text(tx(152), 107.6, "slides per day", ha="center",
            va="center", fontsize=FS_B, zorder=6)
    for x, s in (((X0 + wa) / 2, "dropped"), ((wb + X1) / 2, "future")):
        ax.text(x, 104.2, s, ha="center", va="center", fontsize=FS_B, color=FROZEN_EDGE,
                zorder=6)
    ax.text(X1 + 1.6, 104.2, "tokens", ha="left", va="center", fontsize=FS_B, zorder=6)
    path(ax, [(X0, 90.2), (X1, 90.2)], head=False, lw=1.0)
    for db, dlab in ((364, "D−364"), (182, "D−182"), (0, "D")):
        x = float(tx(db))
        ax.add_line(Line2D([x, x], [89.7, 90.2], color="black", lw=1.0, zorder=3))
        if db in (364, 0):
            dlab += f" · {(D - timedelta(days=db)):%Y-%m-%d}"
        ax.text(x, 88.9, dlab, ha="center", va="center", fontsize=FS_B, zorder=6)
        ax.text(x, 87.3, f"{364 - db}", ha="center", va="center", fontsize=FS_B, zorder=6)
    ax.text(16.6, 88.9, "date", ha="right", va="center", fontsize=FS_B, zorder=6)
    ax.text(16.6, 87.3, "age token", ha="right", va="center", fontsize=FS_B, zorder=6)
    # key, right of the band
    kx = 116.0
    ax.plot(kx + 0.6, 101.0, "v", ms=4.5, color=ORANGE, zorder=6)
    ax.text(kx + 2.0, 101.0, "anchor = latest clear scene", ha="left", va="center",
            fontsize=FS_B, zorder=6)
    ax.add_line(Line2D([kx + 0.6] * 2, [97.9, 99.5], color=FROZEN_EDGE, lw=1.2, zorder=6))
    ax.text(kx + 2.0, 98.7, "given to the model", ha="left", va="center", fontsize=FS_B,
            zorder=6)
    ax.add_line(Line2D([kx + 0.6] * 2, [95.4, 97.0], color=GREY, lw=1.2, zorder=6))
    for k, s in enumerate(("not given: outside the window,", "over the cap (SIF last 50),",
                           "not yet released (TWSA +45 d)")):
        ax.text(kx + 2.0, 96.2 - k * 1.7, s, ha="left", va="center", fontsize=FS_B, zorder=6)
    # every token enters the sequence carrying its age
    path(ax, [(93, 86.4), (93, 78)])
    label(ax, 94.5, 83.6, "each token + age embedding", ha="left")
    label(ax, 94.5, 81.8, "(rel_pos 0–364) + DOY", ha="left")

    # ── lane A: drivers ────────────────────────────────────────────────────────────────────
    pw, px = 8.2, [2.0, 11.6, 21.2, 30.8]                    # lanes B and C
    # ERA5-Land: the 18 columns (dataset.ERA5_VARS) as 8 sparklines (mean line in a min-max
    # band = 3 columns, or a daily sum = 1); the daily statistic is written under each name
    e5, age = a["era5"], -a["era5_age"]
    rows = [("T2m", (0, 1, 2)), ("Td2m", (3, 4, 5)), ("u10", (6, 7, 8)), ("v10", (9, 10, 11)),
            ("psurf", (12, 13, 14)), ("precip", 15), ("SW↓", 16), ("LW↓", 17)]
    for k, (name, col) in enumerate(rows):
        y0 = 84 - (k + 1) * 2.5 + 0.2
        t = inset(fig, 7.6, y0, 12.6, 2.3)
        if isinstance(col, tuple):
            t.fill_between(age, e5[:, col[1]], e5[:, col[2]], color=LBLUE, lw=0)
            t.plot(age, e5[:, col[0]], color="#3776AB", lw=0.7)
        elif name == "precip":
            t.bar(age, e5[:, col], width=1.0, color=DBLUE, lw=0)
        else:
            t.plot(age, e5[:, col], color="#3776AB", lw=0.7)
        t.set_xlim(-365, 0)
        ax.text(7.2, y0 + 1.6, name, ha="right", va="center", fontsize=FS_B, zorder=6)
        ax.text(7.2, y0 + 0.45, "min·mean·max" if isinstance(col, tuple) else "daily sum",
                ha="right", va="center", fontsize=7, color=FROZEN_EDGE, zorder=6)
    label(ax, 13.4, 85.0, "ERA5-Land · 365 d × 18")
    # SIF: the last 50 values span only ~7 weeks, so zoom to them (on a 365-d axis they are a
    # smudge at the right edge); TWSA: monthly points joined, last 45 d shaded = not released
    # SIF / TWSA / soil: one wide frame each (label left, plot right), stacked in a column,
    # each with ONE straight arrow to its own encoder at the same height
    o = np.argsort(-a["sif_age"])
    t = inset(fig, 29.4, 79.6, 12.6, 5.6)
    t.plot(-a["sif_age"][o], a["sif"][o], "-o", ms=1.6, lw=0.7, color="#3776AB", mfc=DBLUE,
           mec="none")
    t.set_xlim(-a["sif_age"].max() - 2, 1)
    label(ax, 22.6, 83.4, "SIF", ha="left")
    label(ax, 22.6, 81.6, "last 50", ha="left", fs=FS_B - 2, color=FROZEN_EDGE)
    o = np.argsort(-a["twsa_age"])
    t = inset(fig, 29.4, 71.8, 12.6, 5.6)
    t.axvspan(-45, 0, color="#EDEDED", lw=0)
    t.plot(-a["twsa_age"][o], a["twsa"][o], "-o", ms=2.6, lw=1.0, color="#3776AB", mfc=DBLUE,
           mec="none")
    t.set_xlim(-365, 0)
    label(ax, 22.6, 75.6, "TWSA", ha="left")
    label(ax, 22.6, 73.8, "monthly", ha="left", fs=FS_B - 2, color=FROZEN_EDGE)
    image(ax, a["soc"], 33.7, 63.8, 5.6, 5.6, cmap="YlOrBr")
    label(ax, 22.6, 67.8, "Soil", ha="left")
    label(ax, 22.6, 66.0, "21 layers", ha="left", fs=FS_B - 2, color=FROZEN_EDGE)
    # one encoder per driver (model.py era5_mlp / twsa_mlp / soil_encoder / sif_mlp: separate
    # weights); the outputs join on a collector into the sequence
    drv = [("ERA5 MLP 18 → 768", 59.0), ("Soil CNN → 4 tokens", 65.0),
           ("TWSA MLP 1 → 768", 72.8), ("SIF MLP 1 → 768", 80.7)]
    for txt, y0 in drv:
        box(ax, 45, y0, 58, y0 + 3.2, [txt], fs=FS_B - 1)
        path(ax, [(58, y0 + 1.6), (59.6, y0 + 1.6)], head=False)
    path(ax, [(59.6, 60.6), (59.6, 82.3)], head=False)
    path(ax, [(13.0, 63.0), (13.0, 60.6), (45, 60.6)])                        # ERA5
    path(ax, [(42.6, 66.6), (45, 66.6)])                                      # soil
    path(ax, [(42.6, 74.4), (45, 74.4)])                                      # TWSA
    path(ax, [(42.6, 82.3), (45, 82.3)])                                      # SIF

    # ── lane B: satellite tokens ───────────────────────────────────────────────────────────
    yb0, s = 45.0, 6.6
    lulc = a["lulc"]
    cm = ListedColormap(list(a["lulc_colours"]))
    stack(ax, a["s2_rgb"], px[0], yb0, s)
    stack(ax, a["s1_vv"], px[1], yb0, s, cmap="gray", vmin=0, vmax=1)
    image(ax, a["dem_hs"], px[2], yb0, pw, pw, cmap="gray")
    image(ax, lulc, px[3], yb0, pw, pw, cmap=cm, norm=BoundaryNorm(np.arange(-.5, 11), 11))
    for x, s_ in zip(px, ["S2 L2A ≤ 60", "S1 RTC ≤ 40", "DEM", "LULC"]):
        label(ax, x + pw / 2, 54.8, s_, fs=FS_B - 1)
    bus_b = 42.0
    for x in px:
        path(ax, [(x + pw / 2, yb0), (x + pw / 2, bus_b)], head=False)
    path(ax, [(px[0] + pw / 2, bus_b), (51, bus_b), (51, 45)])
    box(ax, 44, 45, 58, 57.5, ["TerraMind", "ViT-B", "frozen · offline", "layer 12"],
        kind="frozen", title_n=2)
    path(ax, [(58, 51.25), (62, 51.25)])
    image(ax, np.kron(a["tok_rgb"], np.ones((8, 8, 1))), 62, 45, 12.5, 12.5)
    g = 12.5 / 14
    for k in range(1, 14):
        ax.add_line(Line2D([62 + k * g] * 2, [45, 57.5], color="white", lw=0.25, zorder=5))
        ax.add_line(Line2D([62, 74.5], [45 + k * g] * 2, color="white", lw=0.25, zorder=5))
    for w in (7, 5, 3, 1):                                   # nested windows 14/10/6/2 tokens
        ax.add_patch(Rectangle((62 + (7 - w) * g, 45 + (7 - w) * g), 2 * w * g, 2 * w * g,
                               fill=False, ec="white", lw=1.1, zorder=6))
    label(ax, 68.25, 58.4, "196 × 768 · 160 m")
    path(ax, [(74.5, 51.25), (86, 51.25)])
    label(ax, 80.25, 53.2, "pyramid ×4")
    label(ax, 79.6, 49.3, "anchor 196")
    box(ax, 62, 38.3, 76, 42.8, ["satellite modality", "dropout 0.3"], kind="plain",
        dashed=True, fs=FS_B)

    # ── token sequence + transformer ───────────────────────────────────────────────────────
    path(ax, [(59.6, 71), (86, 71)])
    box(ax, 61.5, 72.4, 84, 77.2, ["ERA5 token dropout 0.15", "ERA5 modality dropout 0.5"],
        kind="plain", dashed=True, fs=FS_B)
    box(ax, 86, 42, 100, 78, None)
    seq = [("Depth CLS", 3, "train"), ("DEM · LULC", 8, "frozen"), ("Soil", 4, "train"),
           ("Anchor", 196, "frozen"), ("S2 history", 240, "frozen"), ("S1 history", 160, "frozen"),
           ("ERA5", 365, "train"), ("SIF", 50, "train"), ("TWSA", 12, "train")]
    label(ax, 93, 76.2, "Token sequence", fs=FS_T)
    # slots actually filled for this tile/day; the rest are key-padded (model.py
    # _build_sequence -> src_key_padding_mask): empty history / SIF / TWSA slots, missing
    # ERA5 days, and at training time whatever the dropouts withhold. 4 pyramid rows per scene.
    filled = {"S2 history": 4 * len(a["h_s2_age"]), "S1 history": 4 * len(a["h_s1_age"]),
              "ERA5": len(a["era5_age"]), "SIF": len(a["sif_age"]), "TWSA": len(a["twsa_age"])}
    # one illustrative training draw of ERA5 token dropout: 15 % of the filled days
    # (dataset.py: torch.rand(365) < 0.15 -> doy 0 -> key-padded)
    era5_drop = np.flatnonzero(np.random.default_rng(0).random(filled["ERA5"]) < 0.15)
    EMPTY = "#D9D9D9"
    def slots(x0, x1, yb, frac):
        xm = x0 + (x1 - x0) * frac
        ax.add_patch(Rectangle((x0, yb), xm - x0, 0.4, fc=DBLUE, ec="none", zorder=3))
        if frac < 1:
            ax.add_patch(Rectangle((xm, yb), x1 - xm, 0.4, fc=EMPTY, ec="none", zorder=3))
    y = 73.6
    for name, n, kind in seq:
        fc = FROZEN_FILL if kind == "frozen" else LBLUE
        ax.add_patch(Rectangle((87.0, y - 0.8), 1.6, 1.6, fc=fc, ec="black", lw=0.6, zorder=3))
        ax.text(89.2, y, name, ha="left", va="center", fontsize=FS_B, zorder=3)
        ax.text(99.3, y, f"{n}", ha="right", va="center", fontsize=FS_B, zorder=3)
        slots(89.2, 99.3, y - 1.35, filled.get(name, n) / n)
        if name == "ERA5":                    # randomly dropped days, in place, widened to show
            for i in era5_drop:
                xd = 89.2 + (99.3 - 89.2) * i / n
                ax.add_patch(Rectangle((xd - 0.03, y - 1.55), 0.08, 0.8, fc=ORANGE,
                                       ec="none", zorder=3.2))
        y -= 2.55
    ax.add_line(Line2D([87, 99.3], [y + 1.1] * 2, color="black", lw=0.8, zorder=3))
    ax.text(99.3, y - 0.4, "1038", ha="right", va="center", fontsize=FS_B, zorder=3)
    ax.text(89.2, y - 0.4, "× 768", ha="left", va="center", fontsize=FS_B, zorder=3)
    # no key (user): orange ticks = random ERA5 token dropout, grey bar ends = no data, both key-padded
    label(ax, 93, 44.9, "+ age 0–364 · DOY ·", fs=FS_B)
    label(ax, 93, 43.2, "modality · position", fs=FS_B)
    path(ax, [(100, 60), (104, 60)])
    # no causal mask: every token attends to every other (past/future, all inputs) except
    # key-padded ones (model.py forward: src_key_padding_mask only)
    # one pre-norm encoder layer drawn as coloured blocks (nn.TransformerEncoderLayer,
    # norm_first=True, FFN 768 -> 3072 -> 768), repeated x3, then a final LayerNorm
    box(ax, 104, 42, 118, 78, None)
    ax.add_patch(FancyBboxPatch((104, 42), 14, 36, fill=False, ec=DBLUE, lw=2.6, zorder=2.1,
                                boxstyle="round,pad=0,rounding_size=0.8"))   # thick border
    label(ax, 111, 76.3, "Transformer", fs=FS_T)
    label(ax, 111, 74.5, "d = 768 · pre-norm", fs=FS_B - 1)
    ax.add_patch(FancyBboxPatch((105, 47.6), 12, 25.4, fill=False, ec=DBLUE, lw=1.0,
                                ls=(0, (3, 2)), boxstyle="round,pad=0,rounding_size=0.6",
                                zorder=2.2))
    label(ax, 116.6, 72.0, "× 3", ha="right", fs=FS_B)
    def blk(y0, y1, lines, fc, tc="black"):
        ax.add_patch(FancyBboxPatch((106, y0), 10, y1 - y0, fc=fc, ec=DBLUE, lw=0.9,
                                    boxstyle="round,pad=0,rounding_size=0.5", zorder=2.3))
        n = len(lines)
        for k, s in enumerate(lines):
            ax.text(111, (y0 + y1) / 2 + (n - 1) * 0.85 - k * 1.7, s, ha="center",
                    va="center", fontsize=FS_B - 1, color=tc, zorder=2.4)
    blk(68.4, 70.6, ["LayerNorm"], "white")
    blk(61.6, 67.6, ["self-attention", "12 heads", "bidirectional"], DBLUE, tc="white")
    label(ax, 111, 60.5, "⊕ residual", fs=FS_B - 1)
    blk(57.0, 59.2, ["LayerNorm"], "white")
    blk(51.0, 56.2, ["feed-forward", "768→3072→768"], LBLUE)
    label(ax, 111, 49.6, "⊕ residual", fs=FS_B - 1)
    blk(43.6, 45.8, ["final LayerNorm"], "white")
    label(ax, 111, 46.7, "key-padding mask", fs=FS_B - 2, color=FROZEN_EDGE)

    # ── lane C: fine 20 m path ─────────────────────────────────────────────────────────────
    yc0 = 14.0
    s2f = a["s2_rgb"][-1][::2, ::2]
    image(ax, s2f, px[0], yc0, pw, pw)
    image(ax, a["s1_vv"][-1][::2, ::2], px[1], yc0, pw, pw, cmap="gray", vmin=0, vmax=1)
    # what the DEM stem actually sees (dataset.fine_dem_static): 2x2 mean to 20 m, then
    # asinh((h - tile mean) / 1 m) / 4, clipped +/-2.5; nodata (<= -1000 m or non-finite) = 0
    dem = np.asarray(a["dem"], np.float64)
    v = np.isfinite(dem) & (dem > -1000)
    s_ = np.where(v, dem, 0).reshape(112, 2, 112, 2).sum((1, 3))
    n_ = v.reshape(112, 2, 112, 2).sum((1, 3))
    p_ = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
    rel = np.clip(np.arcsinh((p_ - np.nanmean(p_)) / 1.0) / 4.0, -2.5, 2.5)
    lim = float(np.nanpercentile(np.abs(rel), 98))
    image(ax, np.nan_to_num(rel), px[2], yc0, pw, pw, cmap="RdBu_r", vmin=-lim, vmax=lim)
    image(ax, lulc, px[3], yc0, pw, pw, cmap=cm, norm=BoundaryNorm(np.arange(-.5, 11), 11))
    # name + what dataset.py / model.py do to it before the stem (fine_*_scene, _lulc_20m)
    # NOT "on D": dataset.build_fine takes the most recent S2 / S1 scene on or before D (its
    # age is an input channel); LULC = latest year before D's year. Ages from the shown scenes.
    Dd = datetime.strptime(str(a["day"]), "%Y-%m-%d")
    age2 = (Dd - _ymd(a["s2_dates"][-1])).days
    age1 = (Dd - _ymd(a["s1_dates"][-1])).days
    for x, s_, n_ in zip(px, ["S2 · latest", "S1 · latest", "DEM · static",
                              f"LULC · {int(a['lulc_year'])}"],
                         ["masked 2×2 pool", "linear pool → dB", "asinh(Δh/1 m)/4",
                          "8-d embedding"]):
        label(ax, x + pw / 2, 25.4, s_)
        label(ax, x + pw / 2, 23.7, n_, fs=FS_B - 2, color=FROZEN_EDGE)
    label(ax, 23.0, 27.3, "fine 20 m path: most recent scene on or before D", fs=FS_B - 1)
    # recency badges on the images, same orange as the "latest clear scene" anchor marker
    for x, ag in ((px[0], age2), (px[1], age1)):
        ax.text(x + 0.35, yc0 + pw - 0.35, f"D − {ag} d", ha="left", va="top", fontsize=FS_B - 1,
                color="white", zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc=ORANGE,
                                                   ec="none"))

    # ── cloud mask: SEnSeIv2 classes -> token mask (history, anchor, pyramid) + pixel mask ──
    # dataset.py CM_BAD_CLASSES = thin / thick cloud, shadow, nodata; a 160 m token is valid
    # at <= 1 % bad pixels (_cm_token_mask); fine S2 pools clear pixels only (fine_s2_scene).
    # scene -> [SEnSeIv2 on the arrow] -> token mask; the model is named on the arrow, no box
    # every S2 scene -> SEnSeIv2 -> its token mask: a stack in, the same stack masked out.
    # The mask (grey rejected tokens + white 14x14 grid) is baked into each RGB so the stacked
    # layers occlude each other correctly.
    xs2 = px[0] + pw / 2
    cs, off, cy0 = 6.4, 0.8, 29.4
    xa, xb = xs2 - (cs + 2 * off) / 2, xs2 - (cs + 2 * off) / 2 + cs + 2 * off + 10.0
    # mask branch = dashed grey (MASK): it leaves the S2 L2A frame on its own line, left of
    # the S2 -> TerraMind bus, so it does not read as S2 data flowing down to the fine lane
    path(ax, [(2.6, 44.4), (2.6, cy0 + cs)], color=MASK, dashed=True)   # S2 L2A -> SEnSeIv2
    # pixel mask leaves the SEnSeIv2 OUTPUT stack (not its input) and goes to the fine S2
    path(ax, [(xb + 1.0, cy0), (xb + 1.0, 28.3), (xs2, 28.3), (xs2, 26.3)], color=MASK,
         dashed=True)
    label(ax, (xs2 + xb + 1.0) / 2 + 1.0, 29.3, "pixel mask", color=MASK)
    xm0, xm1 = xa + cs + 2 * off, xb
    path(ax, [(xm0, 33.5), (xm1, 33.5)], color=MASK, dashed=True)
    label(ax, (xm0 + xm1) / 2, 35.0, "SEnSeIv2", fs=FS_B)
    label(ax, (xm0 + xm1) / 2, 32.0, "SegFormer-B2", fs=FS_B - 1)
    if "cm_rgbs" in a:
        rgbs = np.asarray(a["cm_rgbs"], np.float32)
        if rgbs.max() > 1.0:                               # stored as uint8 0-255
            rgbs = rgbs / 255.0
        masked = []
        for rgb, tok in zip(rgbs, a["cm_toks"]):
            rej = np.kron(~tok, np.ones((16, 16), bool))
            m = rgb.copy()
            m[rej] = 0.1 * m[rej] + 0.9 * np.array([0.62, 0.62, 0.62])
            m[::16, :] = 1.0
            m[:, ::16] = 1.0
            masked.append(m)
        stack(ax, list(rgbs), xa, cy0, cs, off=off)
        stack(ax, masked, xb, cy0, cs, off=off)
        kept = " · ".join(str(int(t.sum())) for t in a["cm_toks"])
        label(ax, xb + cs / 2 + off, 38.4, f"tokens kept {kept} /196", fs=FS_B - 1)
    # the mask acts INSIDE the pyramid pooling (dataset._cpu_pyramid_pool: masked mean over
    # valid tokens per window) and in scene / anchor selection -> arrow onto the pyramid step
    path(ax, [(xb + cs + 2 * off, 33.5), (84.8, 33.5), (84.8, 51.25)], color=MASK, dashed=True)
    label(ax, 61, 36.7, "token mask: ≤ 1 % cloudy px per 160 m token", color=MASK)
    label(ax, 61, 35.0, "masked mean in pyramid ×4 · picks the clear anchor", color=MASK)
    # one 3x3 stem per input (model.py stem_s2 / stem_s1 / stem_dem / stem_lulc), then concat
    bus_c = 5.5
    for x, nm, ch in zip(px, ["S2", "S1", "DEM", "LULC"],
                         ["12 → 16", "5 → 8", "2 → 4", "emb 8 → 4"]):
        xm = x + pw / 2
        path(ax, [(xm, yc0), (xm, 11.8)])
        box(ax, x, 7.4, x + pw, 11.8, [f"{nm} stem", ch], title_n=1, fs=FS_B)
        path(ax, [(xm, 7.4), (xm, bus_c)], head=False)
    path(ax, [(px[0] + pw / 2, bus_c), (51, bus_c), (51, 13.0)])
    box(ax, 44, 13.0, 58, 21.0, ["concat", "32 @ 112", "(3×3 stems)"])
    box(ax, 44, 21.8, 58, 26.2, ["fine modality", "dropout 0.2 (S2/S1)"], kind="plain",
        dashed=True, fs=FS_B)
    path(ax, [(58, 20), (62, 20)])
    vbox(ax, 62, 12.5, 68, 26.5, "E1 · 32 @ 112")
    path(ax, [(68, 20), (71, 20)])
    vbox(ax, 71, 14.5, 77, 25.5, "E2 · 64 @ 56")
    path(ax, [(77, 20), (80, 20)])
    vbox(ax, 80, 15.3, 86, 24.7, "E3 · 128 @ 28")
    label(ax, 74, 26.7, "GroupNorm CNN")

    # ── decoder ────────────────────────────────────────────────────────────────────────────
    vbox(ax, 122, 16.5, 127, 23.5, "512 @ 14")
    path(ax, [(127, 20), (129, 20)])
    vbox(ax, 129, 15.5, 135, 24.5, "256 @ 28")
    path(ax, [(135, 20), (137, 20)])
    vbox(ax, 137, 14, 144, 26, "128 @ 56")
    path(ax, [(144, 20), (146, 20)])
    vbox(ax, 146, 12.5, 153, 26.5, "z · 64 @ 112")
    label(ax, 114.5, 25.6, "U-Net decoder")
    label(ax, 114.5, 23.8, "bilinear ×2")
    # skips, nested so they never cross
    path(ax, [(83, 15.3), (83, 8.0), (132, 8.0), (132, 15.5)])
    path(ax, [(74, 14.5), (74, 5.5), (140.5, 5.5), (140.5, 14)])
    path(ax, [(65, 12.5), (65, 3.0), (149.5, 3.0), (149.5, 12.5)])
    label(ax, 107, 6.8, "skips (FiLM by context)")
    # transformer outputs
    path(ax, [(107, 42), (107, 20), (122, 20)])
    label(ax, 106, 38.0, "bottleneck", ha="right")
    label(ax, 106, 36.2, "768 @ 14×14", ha="right")
    path(ax, [(114, 42), (114, 31.5), (149.5, 31.5)], head=False)
    for xd, ytop in ((132, 24.5), (140.5, 26), (149.5, 26.5)):
        path(ax, [(xd, 31.5), (xd, ytop)])
    label(ax, 131.5, 33.0, "context 768 (FiLM)")
    path(ax, [(118, 66), (121, 66), (121, 35.0), (156, 35.0), (156, 24), (158, 24)])
    label(ax, 138, 36.6, "depth CLS 3 × 768")

    # ── heads, maps, loss ──────────────────────────────────────────────────────────────────
    path(ax, [(153, 18), (158, 18)])
    box(ax, 158, 14, 172, 26, ["3 depth", "heads", "FiLM(z, CLS)", "+ 1×1 conv"], title_n=2)
    sm = a["sm"]
    vmin, vmax = np.percentile(sm, 1), np.percentile(sm, 99)
    path(ax, [(163, 26), (163, 30)])
    ys = [54, 42, 30]
    for d, (ym, dep) in enumerate(zip(ys, DEPTHS)):
        image(ax, sm[d], 158, ym, 10, 10, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.add_patch(plt.Circle((158 + 10 * 56.5 / 112, ym + 10 * (1 - 56.5 / 112)), 0.55,
                                fill=False, ec=ORANGE, lw=1.6, zorder=6))
        ax.text(158.4, ym + 9.4, dep, ha="left", va="top", fontsize=FS_B - 1, zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.6))
    cax = fig.add_axes([169.2 / W, 30 / H, 1.0 / W, 34 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), "YlGnBu"), cax=cax)
    cb.set_ticks(np.round(np.linspace(vmin, vmax, 3), 2))
    cb.ax.tick_params(labelsize=FS_B, length=2, pad=1)
    cb.set_label("m³/m³", fontsize=FS_B, labelpad=1)
    path(ax, [(163, 64), (163, 67)])
    box(ax, 156, 67, 172, 78, ["Huber loss", "pixel (56, 56)", "vs ISMN", "training only"],
        kind="loss", dashed=True, title_n=1)

    # ── input frames: every model input sits inside a thick black frame ──────────────────
    def frame(x0, y0, x1, y1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, fc=INPUT_FILL, ec=INPUT_EDGE,
                                    lw=2.4, boxstyle="round,pad=0,rounding_size=1.0",
                                    zorder=0.5))
    # one frame per input, label included
    frame(0.8, 63.0, 21.3, 86.0)         # ERA5-Land
    frame(22.0, 78.8, 42.6, 86.0)        # SIF
    frame(22.0, 70.9, 42.6, 78.1)        # TWSA
    frame(22.0, 63.0, 42.6, 70.2)        # soil
    for x in px:                         # S2 L2A / S1 RTC / DEM / LULC, then the fine four
        frame(x - 0.4, 44.4, x + pw + 0.4, 55.7)
        frame(x - 0.4, 13.2, x + pw + 0.4, 26.3)

    # ── legend row ─────────────────────────────────────────────────────────────────────────
    lx, ly = 2, 110.0
    for kind, txt, dashed in (("input", "input", False),
                              ("frozen", "frozen (TerraMind, SEnSeIv2; offline)", False),
                              ("train", "trainable (26.7 M parameters)", False),
                              ("plain", "training only", True),
                              ("loss", "supervision", True),
                              ("mask", "cloud mask (not data)", False)):
        if kind == "mask":
            path(ax, [(lx, ly + 1.2), (lx + 3.4, ly + 1.2)], color=MASK, dashed=True)
        elif kind == "input":
            frame(lx, ly, lx + 3.4, ly + 2.4)
        else:
            box(ax, lx, ly, lx + 3.4, ly + 2.4, None, kind=kind, dashed=dashed)
        ax.text(lx + 4.2, ly + 1.2, txt, ha="left", va="center", fontsize=FS_B)
        lx += 4.2 + 0.62 * len(txt) + 4
    ax.text(W - 2, ly + 1.2, f"TxSON CR200-18 · day D = {a['day']} · 2.24 km tile",
            ha="right", va="center", fontsize=FS_B)

    open_gaps(fig, ax)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"architecture_s48.{ext}", dpi=600)
    print(f"wrote {OUT}/architecture_s48.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--date", default="2019-09-28", help="day D (day after a clear S2 scene)")
    args = ap.parse_args()
    if args.prep:
        prep(args.date)
    if args.render:
        render()


if __name__ == "__main__":
    main()
