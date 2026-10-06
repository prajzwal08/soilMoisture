"""Publication figure: what the frozen TerraMind encoder gives the soil-moisture model.

Three panels, paper style (plot_style_bw "paper": palette H, Times bold, no titles):
    (a) one tile (TxSON CR200-18): S2 true colour for 4 seasons, and below each the L12
        token map (14 x 14 tokens, 160 m), PCA -> RGB with ONE basis for all four seasons
    (b) the same tile, S1 ascending pass nearest each S2 date: VV and its L12 token map
    (c) the same tile, static modalities: DEM hillshade | DEM tokens | LULC map | LULC tokens
    The station UMAP (--umap) is still fitted on request but NOT drawn (user: later).

Token-map rules come from plot_token_pca.py (§27a), reused, not re-implemented:
    one PCA basis per row, deterministic sign, each panel centred by its own valid-token
    mean (within-tile structure), sink tokens (> 5x median norm, ViT registers) excluded
    from the fit but still drawn, one 2-98 % colour stretch per row, masked tokens grey.
UMAP settings and station features come from probe_token_sm_structure.py (§27b):
    StandardScaler -> UMAP(n_neighbors 15, min_dist 0.1, random_state 0), centre token 105,
    mean over <= 24 acquisitions inside the station's label window, sm_only, finite only.

    python plot_terramind_value.py --prep      # zarr stores + UMAP -> assets.npz (CPU, Pool 64)
    python plot_terramind_value.py --render    # draws from assets.npz only
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

import plot_style_bw

REPO  = Path(__file__).resolve().parent
OUT   = REPO / "figures" / "terramind_value"
TILE  = "ISMN_TxSON_CR200-18"
YEARS = (2019, 2020)
GRID, PATCH = 14, 224

DBLUE, LBLUE, ORANGE = "#08519C", "#6BAED6", "#E6550D"
# Koppen macro classes: only palette-H hues for the three populous classes (B, C, D);
# A and E hold a handful of stations and get neutral greys.
KG = [("A", "Tropical", "#4D4D4D"), ("B", "Arid", ORANGE), ("C", "Temperate", LBLUE),
      ("D", "Cold", DBLUE), ("E", "Polar", "#A6A6A6")]


# ══ prep ══════════════════════════════════════════════════════════════════════════════════
def prep(workers: int, do_umap: bool = False) -> None:
    import pandas as pd

    from plot_tile_context import LULC_COLOURS, hillshade, open_raw, open_tokens
    from plot_token_pca import Basis, load_seasons, station_table
    import probe_token_sm_structure as P

    a = {}
    tok, raw = open_tokens(TILE), open_raw(TILE)
    meta = pd.read_csv(P.SPLITS)
    row = meta[(meta.source_network == "ISMN") & (meta.network == "TxSON")
               & (meta.station_name == TILE.replace("ISMN_TxSON_", ""))].iloc[0]

    # (a) S2 seasons, L12 tokens, one basis for all four
    seasons = load_seasons(tok, raw, float(row.latitude), str(row.kg_macro), YEARS, "s2")
    panels = [(s["layers"]["l12"], s["valid"]) for s in seasons]
    basis = Basis(panels)
    projs = [basis.project(t, i) for i, (t, _) in enumerate(panels)]
    basis.calibrate(projs)
    rgbs = []
    for p, (_, v) in zip(projs, panels):
        rgb = basis.rgb(p)
        rgb[~v.reshape(GRID, GRID)] = 0.6
        rgbs.append(rgb)
    a["s2_img"] = np.stack([s["img"] for s in seasons])
    a["s2_tok"] = np.stack(rgbs)
    a["s2_label"] = np.array([s["label"] for s in seasons])
    a["s2_date"] = np.array([s["date"] for s in seasons])
    a["s2_var"] = basis.var_ratio
    print("seasons", list(zip(a["s2_label"], a["s2_date"])), "pca", basis.var_ratio)

    # (b) S1 ascending, the pass nearest each S2 date (load_seasons' rule), VV + L12 tokens,
    # one basis for all four. Ascending: CR200-18 has 27 asc vs 3 desc passes in 2018-19.
    s1 = load_seasons(tok, raw, float(row.latitude), str(row.kg_macro), YEARS, "s1_asc")
    panels = [(s["layers"]["l12"], s["valid"]) for s in s1]
    basis = Basis(panels)
    projs = [basis.project(t, i) for i, (t, _) in enumerate(panels)]
    basis.calibrate(projs)
    rgbs = []
    for p, (_, v) in zip(projs, panels):
        rgb = basis.rgb(p)
        rgb[~v.reshape(GRID, GRID)] = 0.6
        rgbs.append(rgb)
    a["s1_img"] = np.stack([s["img"] for s in s1])
    a["s1_tok"] = np.stack(rgbs)
    a["s1_date"] = np.array([s["date"] for s in s1])
    print("s1_asc", list(a["s1_date"]), "pca", basis.var_ratio)

    # (c) DEM and LULC tokens, one basis each (single panel)
    for mod in ("dem", "lulc"):
        t = np.asarray(tok[mod][:], np.float32)
        mk = f"{mod}_token_mask"
        v = np.asarray(tok[mk][:], bool).reshape(-1) if mk in tok else np.ones(GRID * GRID, bool)
        if not v.any():
            v = np.ones(GRID * GRID, bool)
        b = Basis([(t, v)])
        p = b.project(t, 0)
        b.calibrate([p])
        rgb = b.rgb(p)
        rgb[~v.reshape(GRID, GRID)] = 0.6
        a[f"{mod}_tok"] = rgb
        print(mod, "pca", b.var_ratio, "sinks", b.n_sink)
    dem = np.asarray(raw["dem/data"][0], np.float32)
    a["dem"], a["dem_hs"] = dem, hillshade(dem)
    years = [int(y) for y in raw["lulc/years"][:]]
    a["lulc"] = np.asarray(raw["lulc/data"][len(years) - 1], np.uint8)
    a["lulc_year"] = years[-1]
    a["lulc_colours"] = np.array([LULC_COLOURS.get(i, "#999999") for i in range(10)])

    st = station_table(TILE)
    a["st_rc"] = st[["row", "col"]].to_numpy()

    # UMAP of every sm_only station's centre token, s2/l12 -- only with --umap (not drawn now)
    if not do_umap:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "assets.npz", **a)
        print(f"wrote {OUT / 'assets.npz'} (no UMAP)")
        return
    disk = P.disk_index(("sm_only",))
    meta = meta.dropna(subset=["source_network", "network", "station_id"]).copy()
    meta["dir"] = [P.station_dir(r) for _, r in meta.iterrows()]
    meta = meta[meta["dir"].isin(disk)].drop_duplicates("dir").set_index("dir")
    names = list(meta.index)
    with Pool(workers) as pool:
        recs = [r for r in pool.map(P.load_station, [(n, disk[n], 24) for n in names],
                                    chunksize=2) if r is not None]
    have = [r for r in recs if "s2/l12" in r["X"] and "0-10" in r["y"]]
    X = np.stack([r["X"]["s2/l12"] for r in have]).astype(np.float64)
    kg = meta.loc[[r["station"] for r in have], "kg_macro"].fillna("?").to_numpy()
    sm = np.array([r["y"]["0-10"] for r in have])       # station mean 0-10 cm SM, qc==0 days
    fin = np.isfinite(X).all(1)
    fin &= np.isfinite(sm)
    X, kg, sm = X[fin], kg[fin], sm[fin]
    from sklearn.preprocessing import StandardScaler
    import umap
    U = umap.UMAP(n_components=2, random_state=0, n_neighbors=15,
                  min_dist=0.1).fit_transform(StandardScaler().fit_transform(X))
    a["umap"], a["umap_kg"], a["umap_sm"] = U.astype(np.float32), kg.astype(str), sm
    print(f"UMAP: {len(U)} stations (dropped {int((~fin).sum())} non-finite)",
          {c: int((kg == c).sum()) for c in sorted(set(kg))})

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "assets.npz", **a)
    print(f"wrote {OUT / 'assets.npz'}")


# ══ render ════════════════════════════════════════════════════════════════════════════════
def _img(ax, img, **kw):
    ax.imshow(img, extent=(-.5, PATCH - .5, PATCH - .5, -.5), interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)


def _stations(ax, rc):
    for r, c in rc:
        ax.plot(c, r, "o", ms=5.5, mfc="white", mec="black", mew=1.3, zorder=6)


def _scalebar(ax):
    """1 km bar, bottom left (10 m px)."""
    ax.plot([12, 112], [208, 208], color="white", lw=4.0, solid_capstyle="butt", zorder=7)
    ax.plot([12, 112], [208, 208], color="black", lw=2.0, solid_capstyle="butt", zorder=8)
    ax.text(62, 197, "1 km", ha="center", va="bottom", fontsize=11, color="black",
            zorder=8, bbox=dict(fc="white", ec="none", pad=0.6))


def render() -> None:
    plot_style_bw.apply(globals(), "paper")
    a = np.load(OUT / "assets.npz", allow_pickle=True)
    # rows sized in inches: square panels are ~1.68 in wide at this width, so give each image
    # row exactly that, plus explicit spacer rows for titles, x-labels and the (c) legend
    # UMAP (assets umap / umap_sm, --umap prep) is NOT drawn: user, 2026-10-06, "leave for later"
    #     S2    S2 tok gap  S1    S1 tok gap   static
    hr = [1.68, 1.68, 0.40, 1.68, 1.68, 0.40, 1.68]
    fig = plt.figure(figsize=(7.4, sum(hr) + 1.0))
    gs = fig.add_gridspec(7, 4, height_ratios=hr, hspace=0.05, wspace=0.06, left=0.09,
                          right=0.99, top=1 - 0.55 / (sum(hr) + 1.0), bottom=0.45 / (sum(hr) + 1.0))

    # (a) S2 seasons + L12 token maps
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        _img(ax, a["s2_img"][j])
        _stations(ax, a["st_rc"])
        d = str(a["s2_date"][j])
        ax.set_title(f"{str(a['s2_label'][j]).split()[0]}\n{d[:4]}-{d[4:6]}-{d[6:]}",
                     fontsize=11, pad=3)
        if j == 0:
            ax.set_ylabel("S2 RGB")
            _scalebar(ax)
        ax = fig.add_subplot(gs[1, j])
        _img(ax, a["s2_tok"][j])
        _stations(ax, a["st_rc"])
        if j == 0:
            ax.set_ylabel("L12 tokens")
    pos = fig.axes[0].get_position()
    fig.text(0.012, pos.y1 + 0.022, "(a)", fontsize=13, va="bottom")

    # (b) S1 (ascending) nearest each S2 date: VV + L12 tokens, one basis for all four
    first_b = len(fig.axes)
    for j in range(4):
        ax = fig.add_subplot(gs[3, j])
        _img(ax, a["s1_img"][j], cmap="gray", vmin=0, vmax=1)
        _stations(ax, a["st_rc"])
        d = str(a["s1_date"][j])
        ax.set_title(f"{d[:4]}-{d[4:6]}-{d[6:]}", fontsize=11, pad=3)
        if j == 0:
            ax.set_ylabel("S1 VV")
        ax = fig.add_subplot(gs[4, j])
        _img(ax, a["s1_tok"][j])
        _stations(ax, a["st_rc"])
        if j == 0:
            ax.set_ylabel("L12 tokens")
    pos = fig.axes[first_b].get_position()
    fig.text(0.012, pos.y1 + 0.012, "(b)", fontsize=13, va="bottom")

    # (c) DEM / LULC
    ax = fig.add_subplot(gs[6, 0])
    _img(ax, a["dem_hs"], cmap="gray", alpha=0.9)
    ax.imshow(a["dem"], extent=(-.5, PATCH - .5, PATCH - .5, -.5), cmap="terrain", alpha=0.45)
    _stations(ax, a["st_rc"])
    ax.set_xlabel("DEM")
    ax = fig.add_subplot(gs[6, 1])
    _img(ax, a["dem_tok"]); _stations(ax, a["st_rc"])
    ax.set_xlabel("DEM tokens")
    lulc = a["lulc"]
    present = sorted(int(x) for x in np.unique(lulc))
    cm = ListedColormap([str(a["lulc_colours"][x]) for x in present])
    ax = fig.add_subplot(gs[6, 2])
    _img(ax, np.searchsorted(present, lulc), cmap=cm,
         norm=BoundaryNorm(np.arange(-.5, len(present)), cm.N))
    _stations(ax, a["st_rc"])
    ax.set_xlabel(f"LULC {int(a['lulc_year'])}")
    ax = fig.add_subplot(gs[6, 3])
    _img(ax, a["lulc_tok"]); _stations(ax, a["st_rc"])
    ax.set_xlabel("LULC tokens")
    pos = fig.axes[-4].get_position()
    fig.text(0.012, pos.y1 + 0.006, "(c)", fontsize=13, va="bottom")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"terramind_value.{ext}", dpi=600)
    print(f"wrote {OUT}/terramind_value.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--umap", action="store_true", help="also fit the station UMAP (not drawn)")
    args = ap.parse_args()
    if args.prep:
        prep(args.workers, args.umap)
    if args.render:
        render()


if __name__ == "__main__":
    main()
