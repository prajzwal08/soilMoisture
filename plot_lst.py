"""
§29 figures — Landsat ST over the CR200-18 tile at native 30 m resolution, and the full record.

  figures/tile_lst/{tile}.png         maps: absolute + anomaly at native resolution,
                                      station overlays, mean-anomaly map, nodata mask, scatter
  figures/lst_timeseries/{tile}.png   whole 2016-2022 record: observed SM, LST anomaly,
                                      tile-mean absolute LST, per-station scatter

Native resolution is preserved by rendering each 30 m grid with interpolation="nearest" and an
extent that maps it onto the 224 px (10 m) tile axis via UTM bounds — so the blocky 30 m pixels
stay blocky while the station markers, which are in 10 m tile pixels, land in the right place.

No dual-axis panels: soil moisture and LST anomaly are stacked panels sharing x, never two
y-scales on one plot.

Usage:
  conda run -n terramind python plot_lst.py --tile ISMN_TxSON_CR200-18
"""

import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.gridspec import GridSpec
from rasterio.transform import rowcol

from extract_lst_timeseries import station_table, tile_bounds_utm, landsat_clear

REPO = Path(__file__).resolve().parent
LST_ROOT = Path("/gpfs/scratch1/shared/pkhanal/lst/landsat_st/txson")
SERIES = REPO / "csvs" / "lst_station_timeseries.csv"
OBS_PQ = REPO / "eval_output" / "txson_timeseries.parquet"

# validated categorical palette (dataviz skill, light mode: all checks PASS)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
PATCH_PX, TILE_RES_M = 224, 10

INK, MUTED, GRID = "#1a1a19", "#5c5b54", "#dcdbd4"


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8.5, "axes.titlesize": 8.8, "legend.frameon": False,
    })


# ------------------------------------------------------------------

def load_scene(tif, st, tile, max_st_qa=3.0):
    """Return (lst, anomaly, clear, extent_in_tile_px, date, tile_mean)."""
    with rasterio.open(tif) as src:
        a = src.read(); T = src.transform; tags = src.tags()
    lst, stqa, qap = a[0], a[1], a[2]
    clear = (landsat_clear(qap) & np.isfinite(lst) & (lst > 250) & (lst < 350)
             & np.isfinite(stqa) & (stqa <= max_st_qa))
    w, s, e, n = tile_bounds_utm(st, tile)
    r0, c0 = rowcol(T, w, n, op=math.floor)
    r1, c1 = rowcol(T, e, s, op=math.ceil)
    H, W = lst.shape
    r0, c0, r1, c1 = max(0, r0), max(0, c0), min(H, r1), min(W, c1)
    sl = (slice(r0, r1), slice(c0, c1))
    L, C = lst[sl], clear[sl]
    tm = float(np.nanmean(L[C])) if C.any() else np.nan
    # UTM bounds of the cropped block -> tile pixel coords (10 m)
    wx, ny_ = T * (c0, r0)
    ex, sy = T * (c1, r1)
    ext = ((wx - w) / TILE_RES_M - .5, (ex - w) / TILE_RES_M - .5,
           (n - sy) / TILE_RES_M - .5, (n - ny_) / TILE_RES_M - .5)
    return (np.where(C, L, np.nan), np.where(C, L - tm, np.nan), C, ext,
            tags.get("datetime_utc", "")[:10], tm)


def mark(ax, sdf, labels=True):
    for k, (_, r) in enumerate(sdf.iterrows()):
        ax.plot(r.col, r.row, "o", ms=7, mfc=PALETTE[k % len(PALETTE)],
                mec="white", mew=1.4, zorder=6)
        if labels:
            ax.annotate(r.short, (r.col, r.row), textcoords="offset points",
                        xytext=(8, 5), fontsize=6.4, color="white", zorder=7,
                        bbox=dict(fc="black", ec="none", alpha=.6, pad=1.0))
    ax.set_xlim(-.5, PATCH_PX - .5); ax.set_ylim(PATCH_PX - .5, -.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ------------------------------------------------------------------

def fig_maps(tile, st, ser, obs, tifs, out):
    style()
    # station markers must be in 10 m TILE pixels (0-224), not 30 m LST raster indices
    sdf = ser.drop_duplicates("station")[
        ["station", "station_name", "tile_row", "tile_col"]].copy()
    sdf = sdf.rename(columns={"tile_row": "row", "tile_col": "col"})
    sdf["short"] = sdf.station_name
    sdf = sdf.sort_values("station")
    assert ((sdf.row.between(0, PATCH_PX)) & (sdf.col.between(0, PATCH_PX))).all(), \
        "station markers outside the 224 px tile grid — wrong coordinate column?"

    # pick 2 wettest + 2 driest dates among fully-clear scenes
    dd = (ser.groupby("date").agg(fc=("tile_frac_clear", "first")).reset_index()
          .merge(obs.groupby("date").obs.mean().rename("sm").reset_index(), on="date"))
    # NB: tile_frac_clear can never reach 1.0 — 17.8% of this tile has no ST retrieval at all
    # (permanent product nodata, see the clear-count panel).  So gate on the achievable
    # maximum, not on an absolute fraction.
    dd = dd[dd.fc >= 0.99 * dd.fc.max()].sort_values("sm")
    picks = list(dd.date.head(2)) + list(dd.date.tail(2))
    if len(picks) < 4:
        dd2 = (ser.groupby("date").agg(fc=("tile_frac_clear", "first")).reset_index()
               .merge(obs.groupby("date").obs.mean().rename("sm").reset_index(), on="date")
               .sort_values("sm"))
        picks = list(dd2.date.head(2)) + list(dd2.date.tail(2))
    by_date = {Path(t).name[:8]: t for t in tifs}
    chosen = [(d, by_date[d.replace("-", "")]) for d in picks if d.replace("-", "") in by_date]

    fig = plt.figure(figsize=(15.5, 15.0))
    gs = GridSpec(4, 4, figure=fig, hspace=.30, wspace=.14,
                  height_ratios=[1, 1, 1, .42])
    fig.suptitle(f"§29  Landsat 8/9 surface temperature at native 30 m — {tile}",
                 x=.012, ha="left", fontsize=13, y=.985)

    scenes = [load_scene(t, st, tile) for _, t in chosen]

    for j, ((d, _), (L, A, C, ext, _, tm)) in enumerate(zip(chosen, scenes)):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(L, cmap="inferno", interpolation="nearest", extent=ext, origin="upper")
        mark(ax, sdf, labels=(j == 0))
        sm = obs[obs.date == d].obs.mean()
        ax.set_title(f"{d}   tile mean {tm:.1f} K\nmean observed SM {sm:.3f} m³/m³",
                     loc="left", fontsize=8.2)
        plt.colorbar(im, ax=ax, fraction=.046, pad=.02, label="LST (K)")

    # robust limits: the raw max is set by a handful of edge pixels bordering the nodata
    # hole, which would wash the whole tile out to pale blue.
    v = max(3.0, float(np.ceil(np.nanpercentile(
        np.concatenate([np.abs(A[np.isfinite(A)]) for _, A, *_ in scenes]), 98))))
    for j, ((d, _), (L, A, C, ext, _, _)) in enumerate(zip(chosen, scenes)):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(A, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                       extent=ext, origin="upper")
        mark(ax, sdf, labels=False)
        ax.set_title(f"{d}  anomaly (tile mean removed)", loc="left", fontsize=8.2)
        if j == 3:
            plt.colorbar(im, ax=ax, fraction=.046, pad=.02, label="LST anomaly (K)")

    # mean anomaly over the whole record
    acc, cnt, ext0 = None, None, None
    for t in tifs:
        L, A, C, ext, _, tm = load_scene(t, st, tile)
        if not np.isfinite(tm):
            continue
        if acc is None:
            acc, cnt, ext0 = np.zeros_like(A), np.zeros_like(A), ext
        if A.shape != acc.shape:
            continue
        acc += np.nan_to_num(A); cnt += C
    mean_a = np.where(cnt > 20, acc / np.maximum(cnt, 1), np.nan)

    ax = fig.add_subplot(gs[2, 0])
    vv = float(np.nanpercentile(np.abs(mean_a), 98))
    im = ax.imshow(mean_a, cmap="RdBu_r", vmin=-vv, vmax=vv, interpolation="nearest",
                   extent=ext0, origin="upper")
    mark(ax, sdf)
    ax.set_title(f"MEAN anomaly over {int(np.nanmax(cnt))} clear scenes\n"
                 "(the persistent thermal pattern)", loc="left", fontsize=8.2)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.02).set_label(
        "mean LST anomaly (K)", fontsize=7.5)

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(cnt, cmap="Greys_r", interpolation="nearest", extent=ext0, origin="upper")
    mark(ax, sdf)
    frac = 100 * float((cnt == 0).mean())
    ax.set_title(f"clear-scene count per pixel\n{frac:.1f}% NEVER retrieved "
                 f"(black) — CR200-6 sits here", loc="left", fontsize=8.2)

    # observed vs predicted mean SM as layout squares
    lvl = (ser.merge(obs[["station", "date", "obs", "pred"]], on=["station", "date"])
           .groupby("station_name")
           .agg(sm=("obs", "mean"), pred=("pred", "mean"), la=("lst_anom_k", "mean"))
           .reset_index())
    lvl = lvl.merge(sdf, on="station_name")

    for k, (col, ttl, cm) in enumerate([("sm", "OBSERVED mean SM", "YlGnBu"),
                                        ("pred", "MODEL predicted mean SM", "YlGnBu")]):
        ax = fig.add_subplot(gs[2, 2 + k])
        sc = ax.scatter(lvl.col, lvl.row, c=lvl[col], s=330, cmap=cm,
                        edgecolor="black", linewidth=.9, zorder=5)
        for _, r in lvl.iterrows():
            ax.annotate(f"{r.station_name}\n{r[col]:.3f}", (r.col, r.row),
                        textcoords="offset points", xytext=(10, -4), fontsize=6.3)
        ax.set_xlim(-.5, PATCH_PX - .5); ax.set_ylim(PATCH_PX - .5, -.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{ttl}\nspread {lvl[col].max()-lvl[col].min():.4f} m³/m³",
                     loc="left", fontsize=8.2)
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.02)

    # the claim
    ax = fig.add_subplot(gs[3, :2])
    ax.grid(True)
    for k, (_, r) in enumerate(lvl.sort_values("station").iterrows()):
        ax.scatter(r.la, r.sm, s=150, color=PALETTE[k % len(PALETTE)],
                   edgecolor="white", linewidth=1.4, zorder=5)
        ax.annotate(r.station_name, (r.la, r.sm), textcoords="offset points",
                    xytext=(9, 3), fontsize=7.4)
    b, a0 = np.polyfit(lvl.la, lvl.sm, 1)
    xs = np.linspace(lvl.la.min() - .3, lvl.la.max() + .3, 20)
    ax.plot(xs, b * xs + a0, color=MUTED, lw=1.6, ls="--", zorder=2)
    rr = np.corrcoef(lvl.la, lvl.sm)[0, 1]
    ax.set_xlabel("station mean LST anomaly (K)   →  warmer")
    ax.set_ylabel("observed mean SM (m³/m³)")
    ax.set_title(f"THE CLAIM — n = {len(lvl)} stations,  r = {rr:+.3f}"
                 f"   (§29 predicted NEGATIVE)", loc="left", fontsize=9.2)

    ax = fig.add_subplot(gs[3, 2:]); ax.axis("off")
    txt = (
        "WITHIN-TILE HETEROGENEITY, THREE WAYS\n"
        f"  observed SM spread      {lvl.sm.max()-lvl.sm.min():.4f} m³/m³\n"
        f"  model predicted spread  {lvl.pred.max()-lvl.pred.min():.4f} m³/m³  "
        f"({100*(lvl.pred.max()-lvl.pred.min())/(lvl.sm.max()-lvl.sm.min()):.0f}% of observed)\n"
        f"  LST anomaly spread      {lvl.la.max()-lvl.la.min():.2f} K\n\n"
        "IS THE THERMAL PATTERN REAL?\n"
        f"  station means separated by {lvl.la.max()-lvl.la.min():.2f} K,\n"
        f"  standard error ~0.07 K  →  yes, ~27x\n"
        f"  median per-pixel ST_QA 2.13 K (single-date noise)\n\n"
        "DOES IT TRACK SOIL MOISTURE?  NO.\n"
        f"  station level     r = {rr:+.3f}  (n=5, p=0.59 — not significant)\n"
        "  pooled            r = +0.167  (n=546) but CONFOUNDED\n"
        "  within-station    r = -0.077  (p=0.07) once station\n"
        "                    identity is removed\n"
        "  per-station r: -0.39 -0.09 +0.06 -0.12 +0.32 (3/5 neg)\n\n"
        "  The pooled positive is Simpson's paradox: persistently\n"
        "  warm pixels happen to sit on persistently wetter soil.\n"
        "  Remove that and the temporal coupling is ~zero.\n\n"
        "CAVEAT: 18.2% of the tile has NO ST retrieval, and the\n"
        "  only station in it is CR200-6 — the WETTEST (0.287).\n"
        "  The test is missing its most informative station."
    )
    ax.text(0, 1, txt, va="top", ha="left", family="monospace", fontsize=7.8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=175, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return lvl


def fig_series(tile, ser, obs, out):
    style()
    stns = sorted(ser.station_name.unique())
    cols = {s: PALETTE[k % len(PALETTE)] for k, s in enumerate(stns)}

    o = obs.copy(); o["dt"] = pd.to_datetime(o.date)
    s = ser.copy(); s["dt"] = pd.to_datetime(s.date)

    fig = plt.figure(figsize=(15.5, 13.0))
    gs = GridSpec(4, len(stns), figure=fig, hspace=.42, wspace=.26,
                  height_ratios=[1.15, 1.0, .78, .95])
    fig.suptitle(f"§29  Full record 2016–2022 — {tile}", x=.012, ha="left",
                 fontsize=13, y=.985)

    # 1. observed SM (all stations, one axis)
    ax1 = fig.add_subplot(gs[0, :])
    for st_ in stns:
        d = o[o.station_name == st_].sort_values("dt")
        ax1.plot(d.dt, d.obs, lw=1.0, color=cols[st_], label=st_, alpha=.9)
        if len(d):
            ax1.annotate(st_, (d.dt.iloc[-1], d.obs.iloc[-1]), xytext=(6, 0),
                         textcoords="offset points", fontsize=7.2, color=cols[st_],
                         va="center")
    ax1.set_ylabel("observed SM (m³/m³)")
    ax1.set_title("Observed soil moisture, 0–10 cm — the six stations are 405–936 m apart",
                  loc="left")
    ax1.legend(ncol=len(stns), fontsize=7.4, loc="upper left", bbox_to_anchor=(0, -.12))

    # 2. LST anomaly (same x, separate panel — never a twin axis)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    for st_ in stns:
        d = s[s.station_name == st_].sort_values("dt")
        ax2.plot(d.dt, d.lst_anom_k, "o", ms=3.4, color=cols[st_], alpha=.85, label=st_)
    ax2.axhline(0, color=MUTED, lw=1.0)
    ax2.set_ylabel("LST anomaly (K)")
    ax2.set_title("Landsat ST anomaly at each station pixel (tile mean removed) — "
                  f"{s.date.nunique()} clear dates of 246 scenes", loc="left")

    # 3. tile-mean absolute LST — the seasonal cycle the anomaly removes
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    tm = s.drop_duplicates("date").sort_values("dt")
    ax3.plot(tm.dt, tm.tile_mean_k, "-o", ms=3.0, lw=.8, color=INK, alpha=.75)
    ax3.set_ylabel("tile mean LST (K)")
    ax3.set_xlabel("")
    ax3.set_title("Tile-mean absolute LST — plausibility check: Texas range 281–332 K, "
                  "correct seasonal phase", loc="left")

    # 4. per-station scatter, small multiples (no dual axis anywhere)
    j = s.merge(o[["station", "date", "obs"]], on=["station", "date"])
    j["sm_anom"] = j.obs - j.groupby("date").obs.transform("mean")
    for k, st_ in enumerate(stns):
        ax = fig.add_subplot(gs[3, k])
        d = j[j.station_name == st_]
        ax.scatter(d.lst_anom_k, d.sm_anom, s=13, color=cols[st_], alpha=.6,
                   edgecolor="none")
        if len(d) > 3:
            r = np.corrcoef(d.lst_anom_k, d.sm_anom)[0, 1]
            b, a0 = np.polyfit(d.lst_anom_k, d.sm_anom, 1)
            xs = np.linspace(d.lst_anom_k.min(), d.lst_anom_k.max(), 10)
            ax.plot(xs, b * xs + a0, color=MUTED, lw=1.3, ls="--")
            ax.set_title(f"{st_}\nr = {r:+.2f}  n = {len(d)}", loc="left", fontsize=8.0)
        ax.axhline(0, color=GRID, lw=.8); ax.axvline(0, color=GRID, lw=.8)
        if k == 0:
            ax.set_ylabel("SM anomaly (m³/m³)")
        ax.set_xlabel("LST anom (K)", fontsize=7.6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=175, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--root", type=Path, default=LST_ROOT)
    args = ap.parse_args()

    st = station_table()
    ser = pd.read_csv(SERIES)
    ser = ser[ser.tile == args.tile]
    obs = pd.read_parquet(OBS_PQ)
    obs = obs[(obs.tile == args.tile) & (obs.depth == "0-10")].dropna(subset=["obs"]).copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")
    obs["station_name"] = obs.station.str.replace("ISMN_TxSON_", "", regex=False)
    obs = obs[obs.station.isin(ser.station.unique())]

    tifs = sorted(p for p in glob.glob(str(args.root / args.tile / "*.tif"))
                  if not p.endswith(".tmp.tif"))
    print(f"{len(tifs)} scenes, {len(ser)} station-date records, {len(obs)} obs")

    fig_maps(args.tile, st, ser, obs, tifs, REPO / "figures" / "tile_lst" / f"{args.tile}.png")
    fig_series(args.tile, ser, obs, REPO / "figures" / "lst_timeseries" / f"{args.tile}.png")


if __name__ == "__main__":
    main()
