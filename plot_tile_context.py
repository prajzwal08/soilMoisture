"""One tile, several stations: landscape context + seasonal imagery + modelled vs observed.

§26 showed that inside a 2.24 km tile the model paints ~one soil-moisture series at every
station.  This figure puts the evidence on one page, before any statistics:

  row 1  what the surface looks like through the year   (S2 true colour, one per season)
  row 2  what the static controls say                   (DEM, LULC, soil, S1 VV)
  row 3  what was measured vs what the model predicts    (six series, shared axes)
  row 4  the station layout coloured by observed and by predicted mean level

All panels in rows 1-2 carry the station markers and the 14x14 token grid, so the 160 m
sampling of the TerraMind embedding can be read against the landscape.

Usage
-----
    python plot_tile_context.py --tile ISMN_TxSON_CR200-18
    python plot_tile_context.py --tile ISMN_TxSON_CR200-3 --years 2018 2020
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec

from visualize_embeddings import cloud_frac_by_date, season_slot, _dstr

warnings.filterwarnings("ignore")

REPO      = Path(__file__).resolve().parent
SAT_ZARR  = Path("/projects/prjs1968/satellite_zarr")
TOK_ZARR  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS    = REPO / "csvs" / "station_splits.csv"

PATCH_PX, TOKEN_PX, RES_M = 224, 16, 10
STATION_COLOURS = ["#111111", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                   "#46f0f0", "#f032e6"]

# The stored values are TerraMind indices, NOT raw ESRI classes -- the download step
# remaps them (download_s1_lulc_mpc.py:50-54).  Getting this wrong reads Texas rangeland
# as flooded vegetation, so it is asserted against that source, not assumed.
LULC_NAMES = {0: "NoData", 1: "Water", 2: "Trees", 3: "Flooded veg.", 4: "Crops",
              5: "Built area", 6: "Bare ground", 7: "Snow/Ice", 8: "Clouds",
              9: "Rangeland"}
LULC_COLOURS = {0: "#ffffff", 1: "#1f6cb0", 2: "#1a7a3a", 3: "#4c9fd4", 4: "#e8c34a",
                5: "#c43c3c", 6: "#cfc6b8", 7: "#f0f0f0", 8: "#dddddd", 9: "#a8bf6a"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def open_raw(tile: str):
    """satellite_zarr has NO .zmetadata -> open_consolidated raises KeyError."""
    return zarr.open_group(str(SAT_ZARR / f"{tile}.zarr"), mode="r")


def open_tokens(tile: str, category: str = "sm_only"):
    """The token store DOES use .zmetadata -> plain open_group returns empty."""
    return zarr.open_consolidated(str(TOK_ZARR / category / tile))


def s2_rgb(raw, idx: int) -> np.ndarray:
    """True colour, 2-98% stretch per channel.  Band order asserted, not assumed."""
    bands = list(raw["s2"].attrs["bands"])
    want = ["B04", "B03", "B02"]
    chan = [bands.index(b) for b in want]
    img = raw["s2/data"][idx].astype(np.float32)
    out = np.zeros((PATCH_PX, PATCH_PX, 3), np.uint8)
    for k, c in enumerate(chan):
        p2, p98 = np.percentile(img[c], (2, 98))
        out[..., k] = np.clip((img[c] - p2) / max(p98 - p2, 1e-6) * 255, 0, 255)
    return out


def hillshade(dem: np.ndarray, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(dem, RES_M)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az_r, alt_r = np.radians(360.0 - az + 90.0), np.radians(alt)
    hs = (np.sin(alt_r) * np.cos(slope)
          + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0, 1)


def seasonal_s2(tok, lat: float, kg: str, year_lo: int, year_hi: int) -> dict:
    """Least-cloudy S2 acquisition per season, restricted to [year_lo, year_hi].

    visualize_embeddings.choose_seasonal_s2 searches the whole record, which would put
    2016 imagery next to a 2019-2020 time series.  Same logic, windowed.
    """
    dates = _dstr(tok["s2/dates"][:])
    frac = cloud_frac_by_date(tok)
    chosen: dict[int, tuple] = {}
    for d in dates:
        if len(d) != 8 or not d.isdigit():
            continue
        if not (year_lo <= int(d[:4]) <= year_hi):
            continue
        f = frac.get(d)
        if f is None:
            continue
        slot, label = season_slot(int(d[4:6]), lat, kg)
        if slot not in chosen or f < chosen[slot][1]:
            chosen[slot] = (str(d), float(f), label)
    return chosen


def soil_layer(tile: str, channel: int = 3) -> np.ndarray | None:
    """Soil is 74x74 @30 m = 2.22 km vs the tile's 2.24 km -- a 20 m mismatch, both
    station-centred.  Nearest-neighbour to 224x224 for display only."""
    try:
        s = open_tokens(tile)["soil"][channel].astype(np.float32)
    except Exception:
        return None
    idx = np.clip((np.arange(PATCH_PX) * s.shape[0] / PATCH_PX).astype(int),
                  0, s.shape[0] - 1)
    return s[np.ix_(idx, idx)]


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def mark_stations(ax, st: pd.DataFrame, grid: bool = True, labels: bool = True):
    if grid:
        for k in range(TOKEN_PX, PATCH_PX, TOKEN_PX):
            ax.axhline(k - .5, color="white", lw=.3, alpha=.30)
            ax.axvline(k - .5, color="white", lw=.3, alpha=.30)
    for k, r in enumerate(st.itertuples()):
        c = STATION_COLOURS[k % len(STATION_COLOURS)]
        ax.plot(r.col, r.row, marker="o", ms=7, mfc=c, mec="white", mew=1.3, zorder=6)
        if labels:
            ax.annotate(r.short, (r.col, r.row), textcoords="offset points",
                        xytext=(8, 5), fontsize=6.6, color="white", zorder=7,
                        path_effects=None,
                        bbox=dict(fc="black", ec="none", alpha=.55, pad=.9))
    ax.set_xlim(-.5, PATCH_PX - .5)
    ax.set_ylim(PATCH_PX - .5, -.5)
    ax.set_xticks([]); ax.set_yticks([])


def scale_bar(ax, km: float = 0.5):
    w = km * 1000 / RES_M
    ax.add_patch(mpatches.Rectangle((8, PATCH_PX - 16), w, 3.2, fc="white", ec="black",
                                    lw=.4, zorder=8))
    ax.annotate(f"{int(km*1000)} m", (8 + w / 2, PATCH_PX - 18), ha="center",
                fontsize=6.2, color="white", zorder=8,
                bbox=dict(fc="black", ec="none", alpha=.5, pad=.7))


# Soil channel layout, download_soil_openlandmap.py:12-22 -- 3 depths x 7 properties
SOIL_PROPS = [("clay", "wt%"), ("sand", "wt%"), ("silt", "wt%"), ("soc", "g/kg"),
              ("socd", "mg/cm3"), ("bd", "g/cm3"), ("ph", "pH")]
SOIL_DEPTHS = ["0-30", "30-60", "60-100"]


def soil_at_pixel(soil: np.ndarray, row: int, col: int) -> dict:
    """Soil is 74x74 over 2.22 km; the tile is 224x224 over 2.24 km.  Both are
    station-centred, so a proportional index is correct to within the 20 m mismatch."""
    n = soil.shape[-1]
    sr = int(np.clip(row * n / PATCH_PX, 0, n - 1))
    sc = int(np.clip(col * n / PATCH_PX, 0, n - 1))
    out = {}
    for d, depth in enumerate(SOIL_DEPTHS):
        for p, (prop, _u) in enumerate(SOIL_PROPS):
            out[f"{prop} {depth}"] = float(soil[d * 7 + p, sr, sc])
    return out


def variable_table(ax, tbl: pd.DataFrame, fmt: dict, extra: pd.DataFrame,
                   rules: list[int]):
    """Rows = variables, columns = stations, shaded row-wise min->max.

    Shading is per row because the variables have incomparable units; the point is
    which station is high or low for each variable, never the absolute value.
    """
    nr, nc = tbl.shape
    norm = np.full((nr, nc), np.nan)
    for i in range(nr):
        v = pd.to_numeric(tbl.iloc[i], errors="coerce").to_numpy(float)
        if np.isfinite(v).sum() > 1 and np.nanmax(v) > np.nanmin(v):
            norm[i] = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
    ax.imshow(np.ma.masked_invalid(norm), cmap="YlGnBu", vmin=0, vmax=1,
              aspect="auto", alpha=.50)

    for i in range(nr):
        for j in range(nc):
            v = tbl.iat[i, j]
            s = (fmt.get(tbl.index[i], "{:.2f}").format(v)
                 if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
                 else str(v))
            ax.text(j, i, s, ha="center", va="center", fontsize=6.4)
    # right-hand block: spread and correlation with the two levels
    for k, col in enumerate(extra.columns):
        for i in range(nr):
            v = extra.iloc[i, k]
            s = "" if not np.isfinite(v) else (f"{v:+.2f}" if "r(" in col else f"{v:.3g}")
            ax.text(nc + .35 + k, i, s, ha="center", va="center", fontsize=6.4,
                    color="#b2182b" if ("r(" in col and np.isfinite(v) and abs(v) > .6)
                    else "#333333",
                    weight="bold" if ("r(" in col and np.isfinite(v) and abs(v) > .6)
                    else "normal")
    ax.axvline(nc - .5 + .35 / 2, color="#333", lw=.9)

    ax.set_xticks(list(range(nc)) + [nc + .35 + k for k in range(len(extra.columns))])
    ax.set_xticklabels(list(tbl.columns) + list(extra.columns), fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(nr))
    ax.set_yticklabels(tbl.index, fontsize=6.4, family="monospace")
    ax.set_xlim(-.5, nc + .35 + len(extra.columns) - .5)
    ax.set_ylim(nr - .5, -.5)
    for y in rules:
        ax.axhline(y - .5, color="#333", lw=.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def layout_panel(ax, st: pd.DataFrame, values: pd.Series, title: str, vmin, vmax):
    ax.add_patch(mpatches.Rectangle((0, 0), PATCH_PX, PATCH_PX, fc="#f2f4f7",
                                    ec="#888", lw=.7))
    sc = None
    for k, r in enumerate(st.itertuples()):
        v = values.get(r.station, np.nan)
        sc = ax.scatter(r.col, r.row, c=[v], cmap="YlGnBu", vmin=vmin, vmax=vmax,
                        s=190, ec="black", lw=.8, zorder=5)
        ax.annotate(f"{v:.3f}" if np.isfinite(v) else "–", (r.col, r.row),
                    ha="center", va="center", fontsize=5.8, zorder=6,
                    color="white" if v > (vmin + vmax) / 2 else "black")
    ax.set_xlim(-.5, PATCH_PX - .5); ax.set_ylim(PATCH_PX - .5, -.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, loc="left", weight="bold")
    return sc


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--ts", default="eval_output/txson_timeseries.parquet")
    ap.add_argument("--readouts", default="csvs/txson_readouts.csv")
    ap.add_argument("--years", type=int, nargs=2, default=[2019, 2020])
    ap.add_argument("--soil-channel", type=int, default=3,
                    help="0..20; 3 = SOC 0-30 cm")
    ap.add_argument("--out-dir", default=str(REPO / "figures" / "tile_context"))
    ap.add_argument("--dpi", type=int, default=190)
    args = ap.parse_args()

    tile = args.tile
    st = pd.read_csv(args.readouts)
    st = st[st.tile == tile].sort_values("offset_px").reset_index(drop=True)
    if st.empty:
        raise SystemExit(f"no readouts for {tile}")
    st["short"] = st.station.str.replace(r"^ISMN_[^_]+_", "", regex=True)
    assert int(st.loc[st.is_centre, "row"].iloc[0]) == PATCH_PX // 2, \
        "centre station is not at pixel 112 — readout table is wrong"
    print(f"{tile}: {len(st)} stations")

    splits = pd.read_csv(SPLITS)
    srow = splits[splits.apply(
        lambda r: (f"ISMN_{r['network']}_{r['station_name']}" if r.source_network == "ISMN"
                   else f"{r['source_network']}_{r['station_id']}") == tile, axis=1)].iloc[0]
    lat, kg = float(srow.latitude), str(srow.kg_macro)

    raw = open_raw(tile)
    tok = open_tokens(tile)
    seasons = seasonal_s2(tok, lat, kg, args.years[0], args.years[1])
    print("  seasons chosen: "
          + ", ".join(f"{lab} {d} (cloud {100*f:.1f}%)"
                      for d, f, lab in (seasons[s] for s in sorted(seasons))))

    # S2 index in the RAW store, matched by DATE (raw and token stores need not share order)
    raw_dates = _dstr(raw["s2/dates"][:])
    raw_idx = {d: i for i, d in enumerate(raw_dates)}

    # ── time series ───────────────────────────────────────────────────────
    ts = pd.read_parquet(args.ts)
    ts = ts[(ts.tile == tile) & (ts.depth == "0-10")].copy()
    assert ts.tile.nunique() == 1, "all six series must come from ONE tile's map"
    ts["date"] = pd.to_datetime(ts.date)
    lo, hi = args.years
    tsw = ts[(ts.date.dt.year >= lo) & (ts.date.dt.year <= hi)]

    P = tsw.pivot_table(index="date", columns="station", values="pred")
    O = tsw.pivot_table(index="date", columns="station", values="obs")
    common = P.dropna().index.intersection(O.dropna().index)
    sp_p = float(P.loc[common].std(axis=1).mean())
    sp_o = float(O.loc[common].std(axis=1).mean())
    lvl_p = ts.groupby("station").pred.mean()
    lvl_o = ts.dropna(subset=["obs"]).groupby("station").obs.mean()
    r_lvl = float(np.corrcoef(lvl_p.reindex(st.station), lvl_o.reindex(st.station))[0, 1])
    print(f"  across-station spread: predicted {sp_p:.4f}  observed {sp_o:.4f} "
          f"({100*sp_p/sp_o:.0f}%)   r(level) = {r_lvl:+.3f}")

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16.5, 22.5))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1, 1, .92, .80, 1.95],
                  hspace=.20, wspace=.10, left=.035, right=.985, top=.948, bottom=.030)

    # Row 1 — seasonal S2
    for k, slot in enumerate(sorted(seasons)):
        date, frac, label = seasons[slot]
        ax = fig.add_subplot(gs[0, k])
        ridx = raw_idx.get(date)
        if ridx is None:
            ax.set_axis_off(); ax.set_title(f"{label}: no raw scene", fontsize=8); continue
        ax.imshow(s2_rgb(raw, ridx))
        mark_stations(ax, st)
        scale_bar(ax)
        flag = "" if frac < .02 else "  ⚠ cloudy"
        ax.set_title(f"{label}  ·  {date[:4]}-{date[4:6]}-{date[6:]}\n"
                     f"cloud {100*frac:.1f}%{flag}", fontsize=8.2, loc="left")

    # Row 2 — static controls
    dem = raw["dem/data"][0].astype(np.float32)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(hillshade(dem), cmap="gray", alpha=.55)
    im = ax.imshow(dem, cmap="terrain", alpha=.65)
    mark_stations(ax, st, labels=False)
    ax.set_title(f"DEM  {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m "
                 f"(sd {np.nanstd(dem):.1f})", fontsize=8.2, loc="left")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02).ax.tick_params(labelsize=6)

    # LULC: use the year matching the plotted acquisitions where possible
    years = list(raw["lulc/years"][:])
    yr_want = max(lo, min(hi, years[-1]))
    li = years.index(yr_want) if yr_want in years else len(years) - 1
    lulc = raw["lulc/data"][li].astype(np.uint8)
    present = sorted(int(v) for v in np.unique(lulc))
    pct = {v: 100 * float((lulc == v).mean()) for v in present}
    cmap = ListedColormap([LULC_COLOURS.get(v, "#999999") for v in present])
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.searchsorted(present, lulc), cmap=cmap,
              norm=BoundaryNorm(np.arange(-.5, len(present)), cmap.N))
    mark_stations(ax, st, labels=False)
    ax.set_title(f"Land cover {years[li]}  ·  {len(present)} classes "
                 f"(TerraMind indices)", fontsize=8.2, loc="left")
    ax.legend(handles=[mpatches.Patch(fc=LULC_COLOURS.get(v, "#999999"), ec="none",
                                      label=f"{v} {LULC_NAMES.get(v, '?')}  {pct[v]:.1f}%")
                       for v in sorted(present, key=lambda v: -pct[v])],
              fontsize=5.9, loc="lower left", framealpha=.85, handlelength=1.1)
    # which class each station sits in -- the cleanest ground contrast on this tile
    st_lulc = {r.short: int(lulc[r.row, r.col]) for r in st.itertuples()}
    print(f"  station land cover: "
          + ", ".join(f"{k}={LULC_NAMES.get(v,'?')}" for k, v in st_lulc.items()))

    soc = soil_layer(tile, args.soil_channel)
    ax = fig.add_subplot(gs[1, 2])
    if soc is not None:
        im = ax.imshow(soc, cmap="YlOrBr")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.02).ax.tick_params(labelsize=6)
        ax.set_title(f"Soil ch{args.soil_channel} (SOC 0–30 cm)  "
                     f"{np.nanmin(soc):.0f}–{np.nanmax(soc):.0f}\n"
                     f"74×74 @30 m resampled — 2.22 vs 2.24 km", fontsize=8.2, loc="left")
    mark_stations(ax, st, labels=False)

    ax = fig.add_subplot(gs[1, 3])
    s1d = _dstr(raw["s1_asc/dates"][:])
    j = 0
    if seasons:
        target = pd.Timestamp(str(seasons[sorted(seasons)[0]][0]))
        j = int(np.argmin(np.abs(pd.to_datetime(list(s1d), format="%Y%m%d") - target)))
    vv = np.clip(raw["s1_asc/data"][j, 0].astype(np.float32), -20, 0)
    im = ax.imshow(vv, cmap="gray")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02).ax.tick_params(labelsize=6)
    mark_stations(ax, st, labels=False)
    ax.set_title(f"S1 VV (dB)  {s1d[j]}", fontsize=8.2, loc="left")

    # Row 3 — observed vs predicted
    ylo = min(O.min().min(), P.min().min()) - .02
    yhi = max(O.max().max(), P.max().max()) + .02
    for col, (D, name, note) in enumerate([
            (O, "OBSERVED", f"across-station spread {sp_o:.4f}"),
            (P, "PREDICTED (all six read from this ONE tile's map)",
             f"across-station spread {sp_p:.4f}  =  {100*sp_p/sp_o:.0f}% of observed")]):
        ax = fig.add_subplot(gs[2, col * 2:col * 2 + 2])
        for k, r in enumerate(st.itertuples()):
            if r.station not in D:
                continue
            ax.plot(D.index, D[r.station], lw=.85,
                    color=STATION_COLOURS[k % len(STATION_COLOURS)], label=r.short)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, lw=.3, alpha=.35)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("SM (m³/m³)", fontsize=8)
        ax.set_title(f"{name}\n{note}", fontsize=9, loc="left", weight="bold")
        if col == 0:
            ax.legend(fontsize=6.8, ncol=3, loc="upper right", frameon=False)

    # Row 4 — layout coloured by level, plus the summary
    vmin = min(lvl_o.min(), lvl_p.min()); vmax = max(lvl_o.max(), lvl_p.max())
    ax = fig.add_subplot(gs[3, 0])
    layout_panel(ax, st, lvl_o, "OBSERVED mean level", vmin, vmax)
    ax = fig.add_subplot(gs[3, 1])
    sc = layout_panel(ax, st, lvl_p, "PREDICTED mean level", vmin, vmax)
    cax = ax.inset_axes([1.05, 0.0, .035, 1.0])
    fig.colorbar(sc, cax=cax).ax.tick_params(labelsize=6.5)
    cax.set_title("m³/m³", fontsize=6, pad=3)

    ax = fig.add_subplot(gs[3, 2:])
    ax.set_axis_off()
    lulc_list = ", ".join(f"{v}={LULC_NAMES.get(v, '?')}" for v in present)
    ax.text(0, 1, (
        f"THE TILE IS NOT UNIFORM\n"
        f"  elevation     {np.nanmin(dem):.1f} – {np.nanmax(dem):.1f} m  "
        f"(sd {np.nanstd(dem):.1f} m)\n"
        f"  land cover    {len(present)} classes: {lulc_list}\n"
        f"  station sep.  {st.dist_m.max():.0f} m across the tile\n\n"
        f"THE OBSERVATIONS ARE NOT UNIFORM\n"
        f"  mean level    {lvl_o.min():.3f} – {lvl_o.max():.3f}   "
        f"(range {lvl_o.max()-lvl_o.min():.3f} m³/m³)\n"
        f"  across-station spread   {sp_o:.4f}\n\n"
        f"THE PREDICTION IS\n"
        f"  mean level    {lvl_p.min():.3f} – {lvl_p.max():.3f}   "
        f"(range {lvl_p.max()-lvl_p.min():.3f} m³/m³)\n"
        f"  across-station spread   {sp_p:.4f}  "
        f"= {100*sp_p/sp_o:.0f}% of observed\n"
        f"  r(predicted level, observed level) = {r_lvl:+.3f}\n\n"
        f"Every station is `val`: no gradients, but best.pt was selected on val loss.\n"
        f"Grid overlay = 14×14 TerraMind tokens (160 m). S2 panels are stretched\n"
        f"independently, so colours are not comparable between them."),
        transform=ax.transAxes, va="top", fontsize=8.4, family="monospace")

    # ── Row 5 — every value, per station ──────────────────────────────────
    soil_arr = tok["soil"][:]
    recs = {}
    for r in st.itertuples():
        d = {"latitude": r.lat, "longitude": r.lon,
             "pixel row": r.row, "pixel col": r.col,
             "dist from centre (m)": r.dist_m,
             "token index": (r.row // TOKEN_PX) * 14 + (r.col // TOKEN_PX),
             "elevation (m)": float(dem[r.row, r.col]),
             "land cover": LULC_NAMES.get(int(lulc[r.row, r.col]), "?")}
        d.update(soil_at_pixel(soil_arr, r.row, r.col))
        d["OBSERVED mean SM"] = float(lvl_o.get(r.station, np.nan))
        d["PREDICTED mean SM"] = float(lvl_p.get(r.station, np.nan))
        d["obs - pred"] = d["OBSERVED mean SM"] - d["PREDICTED mean SM"]
        recs[r.short] = d
    tbl = pd.DataFrame(recs)

    obs_v = tbl.loc["OBSERVED mean SM"].astype(float).to_numpy()
    pred_v = tbl.loc["PREDICTED mean SM"].astype(float).to_numpy()
    ext = []
    for name in tbl.index:
        v = pd.to_numeric(tbl.loc[name], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        rng = (np.nanmax(v) - np.nanmin(v)) if ok.sum() > 1 else np.nan
        def _r(y):
            if ok.sum() < 3 or np.nanstd(v[ok]) == 0 or np.nanstd(y[ok]) == 0:
                return np.nan
            return float(np.corrcoef(v[ok], y[ok])[0, 1])
        ext.append({"range": rng, "r(obs)": _r(obs_v), "r(pred)": _r(pred_v)})
    ext = pd.DataFrame(ext, index=tbl.index)

    fmt = {"latitude": "{:.5f}", "longitude": "{:.5f}", "pixel row": "{:.0f}",
           "pixel col": "{:.0f}", "dist from centre (m)": "{:.0f}",
           "token index": "{:.0f}", "elevation (m)": "{:.1f}",
           "OBSERVED mean SM": "{:.4f}", "PREDICTED mean SM": "{:.4f}",
           "obs - pred": "{:+.4f}"}
    rules = [8, 15, 22, 29]          # after the location block and each soil depth
    ax = fig.add_subplot(gs[4, :])
    variable_table(ax, tbl, fmt, ext, rules)
    ax.set_title(
        "Every input value at the six station pixels.  Shading is per row (min → max across "
        "the six), because the units are not comparable.\n"
        "Soil: OpenLandMap 30 m, 3 depths × 7 properties — clay/sand/silt wt%, soc g/kg, "
        "socd mg/cm³, bd g/cm³, ph — stored in native integer scaling.\n"
        "r(obs) / r(pred) = correlation of that variable with the observed / predicted mean "
        "level across the six stations (n=6; |r|>0.6 in red).",
        fontsize=8.2, loc="left", pad=26)

    fig.suptitle(
        f"{tile.replace('ISMN_','')} — one 2.24 km tile, {len(st)} soil-moisture stations, "
        f"{lo}–{hi}",
        x=.035, ha="left", fontsize=14, weight="bold", y=.982)

    out = Path(args.out_dir) / tile
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=args.dpi, bbox_inches="tight")
        print(f"  wrote {out}.{ext}")


if __name__ == "__main__":
    main()
