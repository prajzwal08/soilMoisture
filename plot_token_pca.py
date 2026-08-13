"""§27a — What do the TerraMind tokens actually look like inside one tile, and what does
pooling do to them?

One tile, all stations that fall inside it, every layer, every season, raw vs pooled.

The question this answers is runbook §27.7: not "are the embeddings diverse" (§15's Tier-0
diagnostic settled that) but "is the diversity of the right kind and magnitude to cover soil
moisture". Six TxSON stations sit in ISMN_TxSON_CR200-18's tile, in six DISTINCT tokens
(105, 62, 100, 20, 44, 172), spanning 0.137-0.287 m3/m3 of observed mean SM. If those six
token vectors render the same colour, the answer is in the picture.

Figures
    {tile}_{s2,s1_asc,s1_desc}.png
        rows  : raw scene, L3, L6, L9, L12
        cols  : 4 seasons RAW (14x14 tokens, 160 m) | the same 4 seasons AFTER POOLING
        cells : PCA->RGB of the token grid, station positions marked with observed mean SM
    {tile}_static.png
        rows  : DEM, LULC   (L12-only in the token store)
        cols  : raw raster | per-token L2 norm | PCA->RGB | pooled | pooled + 50% dropout

What is actually pooled (runbook §27.8 — the earlier blanket claim was wrong):
    anchor acquisition L12      -> all 196 tokens, UNPOOLED          (model.py:497-505)
    anchor acquisition L3/L6/L9 -> all 196 tokens, decoder skips     (model.py:519-530)
    S2/S1 history               -> 4 nested scales, finest 320 m     (model.py:446-475)
    DEM and LULC                -> 4 nested scales, finest 320 m     (dataset.py:190-220)
So the pooled half is what the model gets from the HISTORY and from DEM/LULC — not what it
gets full stop. DEM/LULC matter most here: they are the static terrain and land-cover
signals that would drive persistent within-tile wetness, and they are the ones pooled away.

Rendering rules that make the panels comparable (visualize_embeddings.pca_rgb does none of
these — it refits PCA per panel with an arbitrary SVD sign, so its colours mean nothing
across panels):
    * ONE PCA basis per (modality, layer), fit on all seasons at once
    * deterministic component sign, so reruns are identical
    * each season centred by its own valid-token mean -> we look at WITHIN-tile structure
    * one colour scale across the whole row, raw and pooled alike

Usage
    python plot_token_pca.py --tile ISMN_TxSON_CR200-18 --years 2019 2020
    python plot_token_pca.py --tile ISMN_TxSON_CR200-18 --modality static
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from dataset import _load_zarr_labels
from plot_tile_context import (
    LULC_COLOURS,
    LULC_NAMES,
    PATCH_PX,
    RES_M,
    STATION_COLOURS,
    TOKEN_PX,
    hillshade,
    open_raw,
    open_tokens,
    s2_rgb,
    seasonal_s2,
)
from visualize_embeddings import _dstr

REPO     = Path(__file__).resolve().parent
READOUTS = REPO / "csvs" / "txson_readouts.csv"
SPLITS   = REPO / "csvs" / "station_splits.csv"
OUT_DIR  = REPO / "figures" / "token_pca"

GRID    = 14                      # 14x14 tokens
LAYERS  = ("l3", "l6", "l9", "l12")
N_DRAWS = 8                       # dropout draws for the static figure
SEED    = 0

# Every number that appears on a panel, defined once and printed on the figure itself.
_LEGEND_RAW = (
    "HOW TO READ THIS FIGURE\n"
    "colour   PC1→red, PC2→green, PC3→blue of the token vectors, each stretched 2–98% "
    "ACROSS THE WHOLE ROW so raw and pooled panels are directly comparable. "
    "Same colour = same embedding.  Grey = masked / cloudy token.\n"
    "pca a/b/c   fraction of WITHIN-TILE variance held by PC1 / PC2 / PC3 "
    "(denominator = all 768 singular values). Their sum is how much of the real variation "
    "the colours actually show; a large first value means one dominant direction, i.e. low "
    "effective spatial rank.\n"
    "variance kept   replace every token by the pooled vector of the smallest nested window "
    "containing it, then 1 − ‖X−X̂‖²/‖X‖² on tile-mean-removed tokens.  I.e. of everything "
    "the tokens know about WHERE INSIDE THE TILE something is, the fraction those 4 pooled "
    "vectors can reproduce.\n"
    "white squares   the 4 nested pooling windows of dataset.py:190-220 — "
    "2×2 = 320 m, 6×6 = 960 m, 10×10 = 1.6 km, 14×14 = 2.24 km (the whole tile).\n"
    "station label   short name and observed mean 0–10 cm soil moisture in m³/m³ "
    "(QC-clean days, whole record).  One token = 160 m.\n"
    "sink tok.   tokens whose L2 norm exceeds 5x the panel median — ViT massive-activation / "
    "attention-sink registers, not landscape (§27a.3: 97.7% of stations have one; median "
    "13x the median norm, holding a median 66.7% of the tile's DEM variance).  They are "
    "EXCLUDED FROM THE PCA FIT so the colours show terrain instead of the artefact, but "
    "they are still drawn."
)

import textwrap as _tw


def add_legend(fig, width_in: float, fontsize: float = 6.6) -> float:
    """Wrap the legend to the figure width, draw it, and return the inches it needs.

    Reserving a FRACTION of the canvas does not work: the legend has a fixed physical
    height, so on a short figure it eats the panels. Reserve inches instead.
    """
    ncols = max(60, int(width_in * 100 / max(fontsize, 1e-3) * 0.62))
    text = "\n".join(
        "\n".join(_tw.wrap(l, ncols, subsequent_indent="      ")) if l else ""
        for l in _LEGEND_RAW.split("\n"))
    n_lines = text.count("\n") + 1
    need_in = 0.16 + n_lines * fontsize * 1.55 / 72.0
    fig.text(0.006, 0.006, text, fontsize=fontsize, va="bottom", ha="left",
             linespacing=1.45,
             bbox=dict(fc="#f6f6f6", ec="#bbbbbb", lw=.6, pad=4))
    return need_in

# Nested pooling windows, replicated from dataset._cpu_pyramid_pool:212.
# widths = [max(1, G*(i+1)//8) for i in range(4)] -> [1,3,5,7]; window = [7-w, 7+w)
WIDTHS  = [max(1, GRID * (i + 1) // 8) for i in range(4)]
assert WIDTHS == [1, 3, 5, 7], WIDTHS


# ── pooling ──────────────────────────────────────────────────────────────────

def pyramid_pool(tok: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(196, D) + (196,) bool -> (4, D) masked means over nested centred windows.

    Numpy replica of dataset._cpu_pyramid_pool (which is torch and expects a batch).
    Verified against it in verify_pooling() below.
    """
    d = tok.shape[-1]
    g = tok.astype(np.float32).reshape(GRID, GRID, d)
    v = valid.astype(np.float32).reshape(GRID, GRID, 1)
    half = GRID // 2
    out = []
    for w in WIDTHS:
        rs, re = half - w, half + w
        rg, rv = g[rs:re, rs:re, :], v[rs:re, rs:re, :]
        out.append((rg * rv).sum((0, 1)) / np.clip(rv.sum((0, 1)), 1, None))
    return np.stack(out)                                          # (4, D)


def window_of_token() -> np.ndarray:
    """(14,14) int — index of the SMALLEST nested window containing each token."""
    half = GRID // 2
    win = np.full((GRID, GRID), len(WIDTHS) - 1, int)
    for i in reversed(range(len(WIDTHS))):        # smallest last so it wins
        w = WIDTHS[i]
        win[half - w:half + w, half - w:half + w] = i
    return win


WIN_OF_TOKEN = window_of_token()


def pooled_reconstruction(tok: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(196, D) -> (196, D): every token replaced by its smallest window's pooled vector.

    This is what the 4 pooled vectors can say about the tile. The gap to `tok` is the
    information the pooling throws away.
    """
    pooled = pyramid_pool(tok, valid)                             # (4, D)
    return pooled[WIN_OF_TOKEN.reshape(-1)]                       # (196, D)


def retained_fraction(tok: np.ndarray, valid: np.ndarray) -> float:
    """Fraction of within-tile variance surviving the 4-vector pyramid."""
    x = tok.astype(np.float32)
    x = x - x[valid].mean(0)
    xhat = pooled_reconstruction(tok, valid)
    xhat = xhat - xhat[valid].mean(0)
    num = float(((x[valid] - xhat[valid]) ** 2).sum())
    den = float((x[valid] ** 2).sum())
    return float("nan") if den <= 0 else max(0.0, 1.0 - num / den)


# ── PCA shared across a row ──────────────────────────────────────────────────

SINK_RATIO = 5.0        # ‖e‖ above this multiple of the panel median = sink token


def sink_tokens(tok: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(196,) bool — tokens whose norm is a large multiple of the panel median.

    §27a.3: 97.7% of stations have a DEM token at >3x the median norm, median ratio 13x,
    holding a median 66.7% of that tile's within-tile variance. The spike sits in a handful
    of coordinates (CR200-18 DEM: min = -1671 against a mean of +1.76), i.e. the massive
    activations / attention-sink registers ViTs are known to develop. They are not terrain.
    Left in, they dominate PC1 and the picture renders the artefact instead of the
    landscape.
    """
    nn = np.linalg.norm(tok.astype(np.float32), axis=-1)
    med = float(np.median(nn[valid])) if valid.any() else float(np.median(nn))
    return (nn > SINK_RATIO * med) if med > 0 else np.zeros_like(nn, bool)


class Basis:
    """One PCA basis for a whole (modality, layer) row, with a fixed sign convention.

    Sink tokens are excluded from the FIT (they would eat PC1) but are still projected and
    drawn, so nothing is hidden from the viewer.
    """

    def __init__(self, panels: list[tuple[np.ndarray, np.ndarray]]):
        parts, self.means = [], []
        self.n_sink = 0
        for tok, valid in panels:
            x = tok.astype(np.float32)
            fit = valid & ~sink_tokens(tok, valid)
            self.n_sink += int((valid & sink_tokens(tok, valid)).sum())
            if fit.sum() < 8:
                fit = valid
            mu = x[fit].mean(0) if fit.any() else x.mean(0)
            self.means.append(mu)
            parts.append(x[fit] - mu)
        big = np.concatenate(parts, 0) if parts else np.zeros((1, 1), np.float32)
        _, s, vt = np.linalg.svd(big, full_matrices=False)
        comps = vt[:3]
        sign = np.sign(comps.sum(1))
        sign[sign == 0] = 1.0
        self.comps = comps * sign[:, None]              # deterministic across runs
        tot = float((s ** 2).sum())
        self.var_ratio = (s ** 2)[:3] / tot if tot > 0 else np.zeros(3)
        self.lo = self.hi = None                        # set by calibrate()

    def project(self, tok: np.ndarray, panel: int) -> np.ndarray:
        return (tok.astype(np.float32) - self.means[panel]) @ self.comps.T   # (N, 3)

    def calibrate(self, projections: list[np.ndarray]) -> None:
        """One colour scale for the entire row — raw and pooled together."""
        allp = np.concatenate([p.reshape(-1, 3) for p in projections], 0)
        self.lo = np.percentile(allp, 2, axis=0)
        self.hi = np.percentile(allp, 98, axis=0)

    @property
    def sink_note(self) -> str:
        return f"  [{self.n_sink} sink tok. excl. from fit]" if self.n_sink else ""

    def rgb(self, proj: np.ndarray) -> np.ndarray:
        out = (proj - self.lo) / np.maximum(self.hi - self.lo, 1e-8)
        return np.clip(out, 0, 1).reshape(GRID, GRID, 3)



# ── data loading ─────────────────────────────────────────────────────────────

def station_table(tile: str) -> pd.DataFrame:
    """Stations inside this tile: pixel, token, observed mean surface SM."""
    st = pd.read_csv(READOUTS)
    st = st[st.tile == tile].sort_values("offset_px").reset_index(drop=True)
    if st.empty:
        raise SystemExit(f"no readouts for {tile} in {READOUTS}")
    st["tok_r"] = st.row // TOKEN_PX
    st["tok_c"] = st.col // TOKEN_PX
    st["tok_i"] = st.tok_r * GRID + st.tok_c
    st["short"] = st.station.str.replace("ISMN_TxSON_", "", regex=False)

    means = []
    for s in st.station:
        try:
            lab = _load_zarr_labels(open_tokens(s))
        except Exception:
            lab = None
        if lab is None:
            means.append(np.nan)
            continue
        sm, _depths, _times, qc = lab
        v = sm[0].astype(np.float64)
        if qc is not None:
            v = np.where(qc[0] == 0, v, np.nan)
        means.append(float(np.nanmean(v)))
    st["sm_mean"] = means
    return st


def s2_token_mask(tok, date: str) -> np.ndarray:
    """(196,) bool — replicates the cloud rule at dataset.py:298-303."""
    try:
        cm_dates = list(_dstr(tok["cm/dates"][:]))
        j = cm_dates.index(date)
    except Exception:
        return np.ones(GRID * GRID, bool)
    cm = tok["cm/masks"][j][:PATCH_PX, :PATCH_PX]
    bad = np.isin(cm.reshape(GRID, TOKEN_PX, GRID, TOKEN_PX), [3, 4, 5, 255]).mean((1, 3))
    return (bad <= 0.01).reshape(-1)


def load_seasons(tok, raw, lat: float, kg: str, years: tuple[int, int],
                 modality: str) -> list[dict]:
    """One entry per season: date, label, raw image, per-layer tokens, valid mask."""
    seasons = seasonal_s2(tok, lat, kg, years[0], years[1])
    if not seasons:
        raise SystemExit("no S2 acquisitions in the requested year range")

    tdates = list(_dstr(tok[f"{modality}/dates"][:]))
    rdates = list(_dstr(raw[f"{modality}/dates"][:]))
    s2_rdates = list(_dstr(raw["s2/dates"][:]))
    out = []
    for slot in sorted(seasons):
        s2_date, frac, label = seasons[slot]
        if modality == "s2":
            date = s2_date
        else:                                   # nearest S1 acquisition to that S2 date
            if not tdates:
                continue
            date = min(tdates, key=lambda d: abs(int(d) - int(s2_date)))
        if date not in tdates:
            continue
        ti = tdates.index(date)

        if modality == "s2":
            ri = s2_rdates.index(date) if date in s2_rdates else None
            img = s2_rgb(raw, ri) if ri is not None else None
            valid = s2_token_mask(tok, date)
        else:
            ri = rdates.index(date) if date in rdates else None
            img = None
            if ri is not None:
                vv = np.asarray(raw[f"{modality}/data"][ri][0], np.float32)
                p2, p98 = np.nanpercentile(vv, (2, 98))
                img = np.clip((vv - p2) / max(p98 - p2, 1e-6), 0, 1)
            valid = (np.asarray(tok[f"{modality}/token_mask"][ti], bool).reshape(-1)
                     if f"{modality}/token_mask" in tok else np.ones(GRID * GRID, bool))

        if not valid.any():
            valid = np.ones(GRID * GRID, bool)
        out.append(dict(
            slot=slot, label=label, date=date, cloud=frac, img=img, valid=valid,
            layers={l: np.asarray(tok[f"{modality}/{l}"][ti], np.float32) for l in LAYERS},
        ))
    return out


# ── drawing ──────────────────────────────────────────────────────────────────

def mark_tokens(ax, st: pd.DataFrame, labels: bool = True) -> None:
    """Station markers in TOKEN coordinates on a 14x14 panel."""
    for k, r in enumerate(st.itertuples()):
        c = STATION_COLOURS[k % len(STATION_COLOURS)]
        ax.plot(r.tok_c, r.tok_r, marker="o", ms=6, mfc=c, mec="white", mew=1.2, zorder=6)
        if labels:
            txt = f"{r.short}\n{r.sm_mean:.3f}" if np.isfinite(r.sm_mean) else r.short
            ax.annotate(txt, (r.tok_c, r.tok_r), textcoords="offset points",
                        xytext=(6, 4), fontsize=5.4, color="white", zorder=7,
                        bbox=dict(fc="black", ec="none", alpha=.6, pad=.8))
    ax.set_xticks([]); ax.set_yticks([])


def draw_windows(ax, scale: float = 1.0, lw: float = 1.0) -> None:
    """The four nested pooling windows, largest to smallest."""
    half = GRID // 2
    for w in sorted(WIDTHS, reverse=True):
        ax.add_patch(Rectangle(((half - w) * scale - .5 * scale, (half - w) * scale - .5 * scale),
                               2 * w * scale, 2 * w * scale,
                               fill=False, ec="white", lw=lw, alpha=.85, zorder=5))


def modality_figure(tile: str, modality: str, tok, raw, st: pd.DataFrame,
                    lat: float, kg: str, years: tuple[int, int], out: Path) -> None:
    seasons = load_seasons(tok, raw, lat, kg, years, modality)
    ns = len(seasons)
    nrow, ncol = 1 + len(LAYERS), 2 * ns

    w_in, h_panels = 2.15 * ncol, 2.45 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(w_in, h_panels + 2.0), squeeze=False)
    fig.suptitle(
        f"{tile} — {modality.upper()} TerraMind tokens, {years[0]}–{years[1]}\n"
        f"left: RAW 14×14 grid (one token = 160 m)   |   "
        f"right: AFTER `_cpu_pyramid_pool` — the 4 nested windows the model receives "
        f"for the HISTORY (the anchor date is NOT pooled)",
        fontsize=11)

    # Row 0 — the scene itself, left half plain, right half with the pooling windows drawn
    for k, se in enumerate(seasons):
        for half_i, ax in ((0, axes[0][k]), (1, axes[0][ns + k])):
            if se["img"] is None:
                ax.text(.5, .5, "raw n/a", ha="center", va="center", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.imshow(se["img"], cmap=None if se["img"].ndim == 3 else "gray")
                for g in range(TOKEN_PX, PATCH_PX, TOKEN_PX):
                    ax.axhline(g - .5, color="white", lw=.25, alpha=.28)
                    ax.axvline(g - .5, color="white", lw=.25, alpha=.28)
                for j, r in enumerate(st.itertuples()):
                    ax.plot(r.col, r.row, marker="o", ms=5,
                            mfc=STATION_COLOURS[j % len(STATION_COLOURS)],
                            mec="white", mew=1.1, zorder=6)
                if half_i:
                    draw_windows(ax, scale=TOKEN_PX, lw=1.3)
                ax.set_xlim(-.5, PATCH_PX - .5); ax.set_ylim(PATCH_PX - .5, -.5)
                ax.set_xticks([]); ax.set_yticks([])
            if half_i == 0:
                ax.set_title(f"{se['label']}  {se['date']}\ncloud {100*se['cloud']:.1f}%",
                             fontsize=7.6)
            else:
                ax.set_title(f"{se['label']}  — pooling windows", fontsize=7.6)
    axes[0][0].set_ylabel("raw scene", rotation=0, ha="right", labelpad=34, fontsize=8)

    # One shared PCA basis per layer, then raw and pooled under that same basis
    for li, layer in enumerate(LAYERS):
        ax_row = axes[1 + li]
        panels = [(se["layers"][layer], se["valid"]) for se in seasons]
        basis = Basis(panels)

        raw_proj, pooled_proj = [], []
        for pi, (t, v) in enumerate(panels):
            raw_proj.append(basis.project(t, pi))
            pooled_proj.append(basis.project(pooled_reconstruction(t, v), pi))
        basis.calibrate(raw_proj + pooled_proj)

        for k, se in enumerate(seasons):
            v2 = se["valid"].reshape(GRID, GRID)
            for j, (proj, ax) in enumerate(((raw_proj[k], ax_row[k]),
                                            (pooled_proj[k], ax_row[ns + k]))):
                rgb = basis.rgb(proj)
                rgb[~v2] = 0.5                                # padded/cloudy -> grey
                ax.imshow(rgb, interpolation="nearest")
                if j:
                    draw_windows(ax, scale=1.0, lw=1.0)
                mark_tokens(ax, st, labels=(li == 0))
            keep = retained_fraction(se["layers"][layer], se["valid"])
            ax_row[ns + k].set_xlabel(f"variance kept {100*keep:.0f}%", fontsize=5.6)

        ax_row[0].set_ylabel(
            f"{layer.upper()}\npca {basis.var_ratio[0]:.2f}/{basis.var_ratio[1]:.2f}/"
            f"{basis.var_ratio[2]:.2f}{basis.sink_note}",
            rotation=0, ha="right", labelpad=34, fontsize=8)

    need = add_legend(fig, w_in)
    figh = h_panels + 2.0
    fig.tight_layout(rect=[0, need / figh, 1, 1 - 0.62 / figh], h_pad=1.4)
    fig.subplots_adjust(hspace=0.22, wspace=0.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=190)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png').name}  ({ns} seasons)")


def static_figure(tile: str, tok, raw, st: pd.DataFrame, out: Path) -> None:
    """DEM and LULC — L12 only, and the pair that pooling actually damages."""
    rng = np.random.default_rng(SEED)
    cols = ["raw raster", "PCA→RGB (raw)", "pooled (4 windows)",
            f"pooled + 50% dropout (mean of {N_DRAWS})"]
    rows = []
    for mod in ("dem", "lulc"):
        if mod not in tok:
            continue
        t = np.asarray(tok[mod][:], np.float32)
        mk = f"{mod}_token_mask"
        v = (np.asarray(tok[mk][:], bool).reshape(-1) if mk in tok
             else np.ones(GRID * GRID, bool))
        if not v.any():
            v = np.ones(GRID * GRID, bool)
        rows.append((mod, t, v))
    if not rows:
        print("  no dem/lulc arrays — skipping static figure")
        return

    w_in, h_panels = 2.75 * len(cols), 3.15 * len(rows)
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(w_in, h_panels + 2.6), squeeze=False)
    fig.suptitle(
        f"{tile} — static modalities (L12 only). DEM and LULC are the terrain and "
        f"land-cover signals\nthat would drive persistent within-tile wetness — and they "
        f"ARE pooled to 4 nested windows (dataset.py:190-220).", fontsize=11)

    for ri, (mod, t, v) in enumerate(rows):
        ax = axes[ri]
        v2 = v.reshape(GRID, GRID)

        # raw raster
        if mod == "dem":
            dem = np.asarray(raw["dem/data"][0], np.float32)
            ax[0].imshow(hillshade(dem), cmap="gray", alpha=.85)
            ax[0].imshow(dem, cmap="terrain", alpha=.55)
            ax[0].set_title(f"DEM  {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m "
                            f"(sd {np.nanstd(dem):.1f})", fontsize=7.6)
        else:
            years = [int(y) for y in raw["lulc/years"][:]]
            lulc = np.asarray(raw["lulc/data"][len(years) - 1], np.uint8)
            present = sorted(int(x) for x in np.unique(lulc))
            cmap = ListedColormap([LULC_COLOURS.get(x, "#999999") for x in present])
            ax[0].imshow(np.searchsorted(present, lulc), cmap=cmap,
                         norm=BoundaryNorm(np.arange(-.5, len(present)), cmap.N))
            frac = sorted(((100 * float((lulc == x).mean()), x) for x in present),
                          reverse=True)
            ax[0].set_title("LULC  " + "  ".join(
                f"{LULC_NAMES.get(x, x)} {f:.0f}%" for f, x in frac[:3]), fontsize=7.0)
        for g in range(TOKEN_PX, PATCH_PX, TOKEN_PX):
            ax[0].axhline(g - .5, color="white", lw=.25, alpha=.3)
            ax[0].axvline(g - .5, color="white", lw=.25, alpha=.3)
        for j, r in enumerate(st.itertuples()):
            ax[0].plot(r.col, r.row, marker="o", ms=5,
                       mfc=STATION_COLOURS[j % len(STATION_COLOURS)],
                       mec="white", mew=1.1, zorder=6)
        ax[0].set_xlim(-.5, PATCH_PX - .5); ax[0].set_ylim(PATCH_PX - .5, -.5)
        ax[0].set_xticks([]); ax[0].set_yticks([])

        basis = Basis([(t, v)])
        proj_raw = basis.project(t, 0)
        proj_pool = basis.project(pooled_reconstruction(t, v), 0)
        draws = []
        for _ in range(N_DRAWS):
            vd = v & (rng.random(GRID * GRID) >= 0.5)
            if not vd.any():
                vd = v
            draws.append(basis.project(
                pyramid_pool(t, vd)[WIN_OF_TOKEN.reshape(-1)], 0))
        proj_drop = np.mean(draws, axis=0)
        basis.calibrate([proj_raw, proj_pool, proj_drop])

        for ci, proj in ((1, proj_raw), (2, proj_pool), (3, proj_drop)):
            rgb = basis.rgb(proj)
            rgb[~v2] = 0.5
            ax[ci].imshow(rgb, interpolation="nearest")
            if ci > 1:
                draw_windows(ax[ci], scale=1.0, lw=1.0)
            mark_tokens(ax[ci], st, labels=(ri == 0 and ci == 1))
        keep = retained_fraction(t, v)
        ax[2].set_xlabel(f"variance kept {100*keep:.0f}%", fontsize=7.0)
        # dropout spread, expressed in the same units as the colour scale so it is readable
        rng_col = float(np.percentile(proj_raw, 98) - np.percentile(proj_raw, 2))
        spread = float(np.std(draws, axis=0).mean())
        ax[3].set_xlabel(f"draw spread {100*spread/max(rng_col, 1e-8):.0f}% "
                         f"of colour range", fontsize=7.0)
        ax[0].set_ylabel(
            f"{mod.upper()}\npca {basis.var_ratio[0]:.2f}/{basis.var_ratio[1]:.2f}/"
            f"{basis.var_ratio[2]:.2f}{basis.sink_note}",
            rotation=0, ha="right", labelpad=30, fontsize=8.5)

        # A single token with many times the median norm is an outlier big enough to
        # drive every variance statistic on this panel, including `variance kept`.
        # Report its share and re-run the statistic without it, so the headline number
        # is known to be robust rather than assumed to be.
        nn = np.linalg.norm(t, axis=-1)
        top = np.argsort(nn)[::-1][:3]
        xc = t.astype(np.float32) - t[v].mean(0)
        share = float((xc[top[0]] ** 2).sum() / (xc[v] ** 2).sum())
        v_drop = v.copy()
        v_drop[top[0]] = False
        keep_drop = retained_fraction(t, v_drop)
        print(f"  {mod}: top-norm tokens " + ", ".join(
            f"(r{i // GRID},c{i % GRID})={nn[i]:.1f}" for i in top)
            + f"  median={np.median(nn):.1f}\n"
            f"        top token holds {100*share:.1f}% of within-tile variance; "
            f"variance kept = {100*keep:.1f}% with it, {100*keep_drop:.1f}% without it")

    for ci, c in enumerate(cols):
        if ci >= 1:
            axes[0][ci].set_title(c, fontsize=8.0)

    need = add_legend(fig, w_in)
    figh = h_panels + 2.6
    fig.tight_layout(rect=[0, need / figh, 1, 1 - 0.85 / figh], h_pad=3.0)
    fig.subplots_adjust(hspace=0.30, wspace=0.10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=190)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.png').name}")


# ── verification ─────────────────────────────────────────────────────────────

def verify_pooling() -> None:
    """pyramid_pool must equal dataset._cpu_pyramid_pool, and the widest window the mean."""
    import torch

    from dataset import _cpu_pyramid_pool
    rng = np.random.default_rng(1)
    tok = rng.standard_normal((GRID * GRID, 32)).astype(np.float32)
    val = rng.random(GRID * GRID) > 0.25
    mine = pyramid_pool(tok, val)
    theirs = _cpu_pyramid_pool(torch.from_numpy(tok)[None],
                               torch.from_numpy(val.reshape(1, GRID, GRID)))[0].numpy()
    assert np.allclose(mine, theirs, atol=1e-5), np.abs(mine - theirs).max()
    assert np.allclose(mine[3], tok[val].mean(0), atol=1e-5)      # 14x14 window == mean
    assert WIN_OF_TOKEN[7, 7] == 0 and WIN_OF_TOKEN[0, 0] == 3
    print("verify: pooling matches dataset._cpu_pyramid_pool; widest window == plain mean")


def verify_tokens(st: pd.DataFrame) -> None:
    """Token index must be (row//16)*14 + (col//16) and the centre must be token 105."""
    calc = (st.row // TOKEN_PX) * GRID + (st.col // TOKEN_PX)
    assert (calc == st.tok_i).all()
    centre = st[st.is_centre]
    assert (centre.row == 112).all() and (centre.col == 112).all(), "centre not at (112,112)"
    assert (centre.tok_i == 105).all(), "centre token is not 105"
    print(f"verify: {len(st)} stations, {st.tok_i.nunique()} distinct tokens, "
          f"centre at token 105")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--years", type=int, nargs=2, default=(2019, 2020))
    ap.add_argument("--modality", default="all",
                    choices=["all", "s2", "s1_asc", "s1_desc", "static"])
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    verify_pooling()

    tok = open_tokens(args.tile)
    raw = open_raw(args.tile)
    st = station_table(args.tile)
    verify_tokens(st)
    print(st[["short", "row", "col", "tok_r", "tok_c", "tok_i", "sm_mean"]]
          .to_string(index=False))

    meta = pd.read_csv(SPLITS)
    row = meta[meta.apply(
        lambda r: f"ISMN_{r['network']}_{r['station_name']}" == args.tile
        if r["source_network"] == "ISMN"
        else f"{r['source_network']}_{r['station_id']}" == args.tile, axis=1)]
    if row.empty:
        raise SystemExit(f"{args.tile} not found in {SPLITS}")
    lat = float(row.iloc[0]["latitude"])
    kg = str(row.iloc[0].get("kg_macro", "C"))
    print(f"tile lat={lat:.4f} koppen_macro={kg} years={args.years[0]}-{args.years[1]}")

    outdir = Path(args.out)
    todo = (["s2", "s1_asc", "s1_desc", "static"] if args.modality == "all"
            else [args.modality])
    for mod in todo:
        if mod == "static":
            static_figure(args.tile, tok, raw, st, outdir / f"{args.tile}_static")
            continue
        if f"{mod}/dates" not in tok:
            print(f"  {mod}: absent, skipping")
            continue
        modality_figure(args.tile, mod, tok, raw, st, lat, kg,
                        tuple(args.years), outdir / f"{args.tile}_{mod}")


if __name__ == "__main__":
    main()
