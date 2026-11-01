"""TxSON: the derived 30 m terrain next to the soil moisture, on one page.

The §32 terrain build produced HAND and TWI at 30 m for every region. TxSON is region 2
(40 stations in 55 x 55 km, 27 CR200 + 6 CR1000), which makes it the densest place to
look at whether the terrain we derived has anything to do with the water we measure.

  row 1  the two derived fields over the station domain, on hillshade
         HAND — height above nearest drainage, the stable one (dHAND survives a 0.2 m
                DEM perturbation at r = 0.99)
         TWI  — ln(a/tan b), the unstable one (dTWI retains r = 0.20 under the same
                perturbation, §32.9.4), shown so the difference is visible not just
                asserted
  row 2  the same domain with stations coloured by soil moisture, and the scatter that
         asks the actual question: does station-mean SM track HAND or TWI?

The scatter is the point of the figure. It is NOT the sufficiency gate — that regresses
dSM on dHAND between colocated pairs under station fixed effects (§32.8), and a
cross-station scatter confounds terrain with soil, land cover and sensor. It is the
look-before-the-statistics version, and if there is nothing here it is worth knowing
before building the gate.

Usage
-----
    python plot_txson_terrain_sm.py
    python plot_txson_terrain_sm.py --region-id 2 --depth 0-10 --source glo30
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

REPO         = Path(__file__).resolve().parent
TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
TS_PARQUET   = REPO / "eval_output" / "txson_timeseries.parquet"
READOUT_CSV  = REPO / "eval_output" / "txson_per_readout.csv"
OUT_DIR      = REPO / "figures"

BANDS = ["twi", "hand", "acc_cells_mfd", "slope_rad", "valid"]


def hillshade(dem: np.ndarray, res: float, az: float = 315.0, alt: float = 45.0):
    """Same convention as plot_tile_context.py:84, so the two figures read alike."""
    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az_r, alt_r = np.radians(360.0 - az + 90.0), np.radians(alt)
    hs = (np.sin(alt_r) * np.cos(slope)
          + np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0, 1)


def load_terrain(rid: int, source: str):
    p = TERRAIN_ROOT / f"region_{rid:04d}" / "terrain_30m.tif"
    dem_p = TERRAIN_ROOT / f"region_{rid:04d}" / f"dem_{source}_30m.tif"
    with rasterio.open(p) as src:
        arr = src.read()
        tr = src.transform
    with rasterio.open(dem_p) as src:
        dem = src.read(1)
    return {b: arr[i] for i, b in enumerate(BANDS)}, dem, tr


def station_table(rid: int) -> pd.DataFrame:
    s = pd.read_csv(REPO / "csvs" / "station_dem_region.csv")
    return s[s["region_id"] == rid].reset_index(drop=True)


def station_sm(depth: str) -> pd.DataFrame:
    """
    Station-mean OBSERVED soil moisture at the supervised centre pixel, with the
    model's prediction kept alongside.

    Observed, not predicted, is the one to regress terrain against. §31's diagnosis
    measured r(predicted level, observed level) = -0.175 across ALL stations; at TxSON
    it comes out +0.43, but either way the model's between-station level is its own
    object and not evidence about hydrology. Measured here: against PREDICTED SM the
    apparent terrain relationship is r = +0.23 (HAND) and -0.21 (TWI), against OBSERVED
    SM it is +0.08 and -0.00. The model manufactures a terrain signal that the
    observations do not contain.
    """
    ts = pd.read_parquet(TS_PARQUET)
    ts = ts[(ts["depth"] == depth) & ts["is_centre"]]
    sm = (ts.groupby("station")
            .agg(sm_mean=("obs", "mean"), sm_sd=("obs", "std"),
                 pred_mean=("pred", "mean"), n=("obs", "size"))
            .reset_index())
    ro = pd.read_csv(READOUT_CSV)
    ro = ro[(ro["depth"] == depth) & ro["is_centre"]]
    return sm.merge(ro[["station", "R", "ubRMSE", "station_split"]],
                    on="station", how="left")


def to_px(x, y, tr):
    return (x - tr.c) / tr.a, (tr.f - y) / (-tr.e)


def main() -> None:
    ap = argparse.ArgumentParser(description="TxSON terrain vs soil moisture")
    ap.add_argument("--region-id", type=int, default=2)
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--source", choices=["glo30", "fabdem"], default="glo30")
    ap.add_argument("--margin-km", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    T, dem, tr = load_terrain(args.region_id, args.source)
    res = float(tr.a)
    st = station_table(args.region_id)
    sm = station_sm(args.depth)
    st = st.merge(sm, left_on="station_id", right_on="station", how="left")

    # crop to the stations plus a margin: the region is 55 km, the network is smaller
    m = args.margin_km * 1000.0
    cx, cy = st["laea_x"].to_numpy(), st["laea_y"].to_numpy()
    c0, r0 = to_px(cx.min() - m, cy.max() + m, tr)
    c1, r1 = to_px(cx.max() + m, cy.min() - m, tr)
    c0, r0 = max(int(c0), 0), max(int(r0), 0)
    c1 = min(int(c1) + 1, T["hand"].shape[1])
    r1 = min(int(r1) + 1, T["hand"].shape[0])
    sl = (slice(r0, r1), slice(c0, c1))
    hand, twi, dem_c = T["hand"][sl], T["twi"][sl], dem[sl]
    hs = hillshade(dem_c, res)
    ext = [0, (c1 - c0) * res / 1000.0, 0, (r1 - r0) * res / 1000.0]   # km

    sx = ((cx - tr.c) / res - c0) * res / 1000.0
    sy = ext[3] - ((tr.f - cy) / res - r0) * res / 1000.0

    # terrain at each station cell
    for name, fld in (("hand_st", T["hand"]), ("twi_st", T["twi"])):
        st[name] = [fld[int((tr.f - y) / res), int((x - tr.c) / res)]
                    for x, y in zip(cx, cy)]

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.22, wspace=0.22,
                  height_ratios=[1.0, 0.92])

    def basemap(ax, field, cmap, label, vmin=None, vmax=None, log=False):
        ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1.4, origin="upper")
        f = np.log1p(np.clip(field, 0, None)) if log else field
        im = ax.imshow(f, extent=ext, cmap=cmap, alpha=0.72, origin="upper",
                       vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label, fontsize=7.5)
        ax.set_xlabel("km"); ax.set_ylabel("km")
        return im

    # ── row 1: the two derived fields ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    basemap(ax, hand, "YlGnBu_r", "HAND  log1p(m)", log=True)
    ax.scatter(sx, sy, s=16, c="crimson", edgecolor="w", linewidth=0.5, zorder=5)
    ax.set_title(f"HAND — stable under DEM error (r$_{{\\Delta}}$ = 0.99)")

    ax = fig.add_subplot(gs[0, 1])
    basemap(ax, twi, "Blues", "TWI  ln(a/tan$\\beta$)",
            vmin=np.nanpercentile(twi, 2), vmax=np.nanpercentile(twi, 98))
    ax.scatter(sx, sy, s=16, c="crimson", edgecolor="w", linewidth=0.5, zorder=5)
    ax.set_title("TWI — unstable under DEM error (r$_{\\Delta}$ = 0.20)")

    # ── row 1 col 3: stations coloured by soil moisture ──────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1.4, origin="upper")
    ok = st["sm_mean"].notna()
    sc = ax.scatter(sx[ok.to_numpy()], sy[ok.to_numpy()], s=52,
                    c=st.loc[ok, "sm_mean"], cmap="RdYlBu", edgecolor="k",
                    linewidth=0.4, zorder=5)
    ax.scatter(sx[~ok.to_numpy()], sy[~ok.to_numpy()], s=26, facecolor="none",
               edgecolor="0.35", linewidth=0.6, zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(f"mean OBSERVED SM {args.depth} cm  (m$^3$/m$^3$)", fontsize=7.5)
    ax.set_xlabel("km"); ax.set_ylabel("km")
    ax.set_title(f"soil moisture — {int(ok.sum())} of {len(st)} stations evaluated")

    # ── row 2: does SM track either field? ───────────────────────────────────
    def scatter(ax, xcol, xlabel, colour):
        d = st.dropna(subset=[xcol, "sm_mean"])
        ax.scatter(d[xcol], d["sm_mean"], s=40, c=colour, edgecolor="k", linewidth=0.4)
        if len(d) > 3:
            r = float(np.corrcoef(d[xcol], d["sm_mean"])[0, 1])
            k = np.polyfit(d[xcol], d["sm_mean"], 1)
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 20)
            ax.plot(xs, np.polyval(k, xs), color="0.25", lw=1.2, ls="--")
            ax.set_title(f"{xlabel}   r = {r:+.3f}   n = {len(d)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"mean observed SM {args.depth} (m$^3$/m$^3$)")
        ax.grid(alpha=0.25)

    scatter(fig.add_subplot(gs[1, 0]), "hand_st", "HAND at station (m)", "#2c7fb8")
    scatter(fig.add_subplot(gs[1, 1]), "twi_st", "TWI at station", "#41ab5d")

    # spread of the two fields inside the network — is there anything to regress on?
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    d = st.dropna(subset=["hand_st", "twi_st"])
    lv = st.dropna(subset=["sm_mean", "pred_mean"])
    lvl_r = float(np.corrcoef(lv["sm_mean"], lv["pred_mean"])[0, 1]) if len(lv) > 3 else float("nan")
    lines = [
        f"TxSON, region {args.region_id}   {len(st)} stations   "
        f"{(c1-c0)*res/1000:.0f} x {(r1-r0)*res/1000:.0f} km",
        f"DEM source for hillshade: {args.source}",
        "",
        "spread ACROSS stations (what a cross-station regression sees)",
        f"   HAND   {d['hand_st'].min():6.1f} .. {d['hand_st'].max():6.1f} m"
        f"    sd {d['hand_st'].std():5.2f}",
        f"   TWI    {d['twi_st'].min():6.2f} .. {d['twi_st'].max():6.2f} "
        f"     sd {d['twi_st'].std():5.2f}",
        "",
        "spread WITHIN the mapped domain (what the model tile sees)",
        f"   HAND   sd {np.nanstd(hand):5.2f} m",
        f"   TWI    sd {np.nanstd(twi):5.2f}",
        "",
        f"r(observed level, PREDICTED level) = {lvl_r:+.3f}",
        "  (§31 measured -0.175 across ALL stations; at TxSON",
        "   it is positive. Either way the model's level is",
        "   its own object, so terrain is plotted against",
        "   OBSERVED SM, not predicted.)",
        "",
        "reminder: this scatter is NOT the sufficiency gate.",
        "It confounds terrain with soil, land cover and sensor.",
        "§32.8 regresses dSM on dHAND between colocated pairs",
        "under station fixed effects. This is the look before",
        "the statistics.",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=7.6,
            family="monospace", transform=ax.transAxes)

    fig.suptitle(
        f"TxSON — derived 30 m terrain and soil moisture   "
        f"(§32 terrain build, region {args.region_id})", fontsize=11, y=0.98)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (OUT_DIR / f"txson_terrain_sm_{args.source}")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")

    for col, lab in (("hand_st", "HAND"), ("twi_st", "TWI ")):
        for ycol, ylab in (("sm_mean", "observed "), ("pred_mean", "predicted")):
            d = st.dropna(subset=[col, ycol])
            if len(d) > 3:
                r = float(np.corrcoef(d[col], d[ycol])[0, 1])
                print(f"{ylab} SM vs {lab}   r = {r:+.4f}  n={len(d)}")
    lv = st.dropna(subset=["sm_mean", "pred_mean"])
    print(f"r(observed level, predicted level) = "
          f"{np.corrcoef(lv['sm_mean'], lv['pred_mean'])[0,1]:+.4f}  n={len(lv)}")


if __name__ == "__main__":
    main()
