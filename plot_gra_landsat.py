#!/usr/bin/env python
"""Step 4: Landsat surface temperature beside the ECOSTRESS panels, per cluster.

Landsat C2 L2 ST at 30 m over the same 2.24 km window, chosen the same way -- one clear
scene per growing-season month, years pooled.

THERE IS NO LANDSAT DTR.  Collection-2 L2 ST is daytime only (29.2), so this figure
carries the absolute field and its anomaly and nothing else.  That is stated in the
caption rather than left as an empty panel.

Resolution is NOT harmonised with the ECOSTRESS figure.  S2 is 10 m over 224 px,
ECOSTRESS 70 m over 32, Landsat 30 m over ~75 -- all covering the same 2240 m.  Each is
drawn on its own grid; resampling them onto a common one would make the comparison an
interpolation artefact.

Env: terramind (rasterio, matplotlib).  The MPC stack is NOT importable here, so the
QA bit logic is duplicated from download_landsat_st_mpc.py:landsat_clear rather than
imported -- keep the two in step.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot_gra_thermal import (REPO, OUT_DIR, s2_rgb_clear, member_labels, _sm_at,
                              _timeseries, load_bundle, pick_summer_dates, _as_str)

LST_ROOT = Path("/gpfs/work3/0/prjs1968/data/landsat_st")
FNAME_RE = re.compile(r"^(\d{8})_(LC0[89])_(\d{6})\.tif$")

log = logging.getLogger("gra_landsat")


def landsat_clear(qa_dn: np.ndarray) -> np.ndarray:
    """Duplicated from download_landsat_st_mpc.py:landsat_clear (29.5 tier 2).

    Rejects fill(0) dilated-cloud(1) cirrus(2) cloud(3) shadow(4) snow(5) water(7),
    requires Clear(6) and cloud/shadow/cirrus confidence <= low.
    """
    q = np.nan_to_num(qa_dn, nan=1.0).astype(np.uint16)
    single = ~np.any([((q >> b) & 1).astype(bool) for b in (0, 1, 2, 3, 4, 5, 7)], axis=0)
    conf = (((q >> 8) & 3) <= 1) & (((q >> 10) & 3) <= 1) & (((q >> 14) & 3) <= 1)
    return single & conf & (((q >> 6) & 1) == 1)


def scan_scenes(cid: str, months, min_clear: float, anchor: dict | None = None):
    """-> [(month, date, path, lst masked, transform, frac_clear)], one scene per month.

    `anchor` maps month -> the ECOSTRESS date chosen for that month.  When present the
    Landsat scene NEAREST that date wins among those clearing min_clear, so the two
    figures show the same months and comparable years.  Without it, the clearest wins.
    """
    d = LST_ROOT / cid
    if not d.exists():
        return []
    best = {}
    for f in sorted(d.glob("*.tif")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        date, mon = m.group(1), int(m.group(1)[4:6])
        if mon not in months:
            continue
        try:
            with rasterio.open(f) as src:
                lst = src.read(1).astype(np.float32)     # band 1: Kelvin
                qa = src.read(3)                         # band 3: raw QA_PIXEL DN
                T = src.transform
        except Exception as exc:                          # noqa: BLE001
            log.warning("  unreadable %s: %s", f.name, exc)
            continue
        clear = landsat_clear(qa) & np.isfinite(lst)
        fr = float(clear.mean())
        if fr < min_clear:
            continue
        iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        if anchor and mon in anchor:
            key = -abs((pd.Timestamp(iso) - pd.Timestamp(anchor[mon])).days)
        else:
            key = fr
        if mon not in best or key > best[mon][0]:
            best[mon] = (key, (mon, iso, f, np.where(clear, lst, np.nan), T, fr))
    return [best[m][1] for m in sorted(best)]


def draw(cl, mem, args) -> bool:
    cid = cl.cluster_id
    # the ECOSTRESS dates for this cluster, so the Landsat scenes can be matched to them
    anchor = {}
    zb, _ = load_bundle(cl.rep_folder)
    if zb is not None:
        anchor = {m: _as_str(zb["day_utc"][i])[:10]
                  for m, i in pick_summer_dates(zb, args.months, 0.75)}
    rows = scan_scenes(cid, args.months, args.min_clear, anchor or None)
    if not rows:
        log.warning("%s: no Landsat scene clears %.0f%% in months %s -- skipped",
                    cid, 100 * args.min_clear, args.months)
        return False
    log.info("%s: %d/%d month(s) %s", cid, len(rows), len(args.months),
             [f"{m:02d}:{dt} ({100*fr:.0f}% clear)" for m, dt, _, _, _, fr in rows])

    inside = mem[mem.in_tile == 1].reset_index(drop=True)
    labels = {r.station_id: member_labels(r.folder, r.category) for _, r in inside.iterrows()}

    stack = np.stack([r[3] for r in rows])
    f = stack[np.isfinite(stack)]
    v0, v1 = (float(np.percentile(f, 2)), float(np.percentile(f, 98))) if f.size else (270., 330.)
    ano = np.stack([a - np.nanmean(a) for a in stack])
    am = float(np.nanmax(np.abs([np.nanpercentile(ano, 2), np.nanpercentile(ano, 98)]))) \
        if np.isfinite(ano).any() else 1.0

    n = len(rows)
    fig = plt.figure(figsize=(10.5, 3.1 * n + 5.0))
    gs = GridSpec(n + 2, 3, figure=fig, height_ratios=[1] * n + [0.10, 1.25],
                  hspace=0.34, wspace=0.07)
    ims = [None] * 3

    for r, (mon, dt, fpath, lst, T, fr) in enumerate(rows):
        # RGB from the S2 store, chosen clear and near this Landsat date
        ax = fig.add_subplot(gs[r, 0])
        got = s2_rgb_clear(cl.rep_folder, cl.rep_category, dt,
                           args.max_s2_days, args.max_cloud)
        if got is None:
            ax.text(.5, .5, "no clear S2", ha="center", va="center",
                    transform=ax.transAxes, color="#666")
            ax.set_facecolor("#f0f0f0")
        else:
            rgb, sd, off, cfr = got
            ax.imshow(rgb)
            ax.text(.03, .04, f"S2 {sd}  ({off:+d} d)", transform=ax.transAxes,
                    fontsize=7.2, color="w", va="bottom",
                    bbox=dict(fc="black", alpha=.55, pad=1.6, lw=0))
            _mark_s2(ax, inside, labels, dt, args.depth)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{dt}\n{100*fr:.0f}% clear", fontsize=8.5, rotation=0,
                      ha="right", va="center", labelpad=8)
        if r == 0:
            ax.set_title("Sentinel-2 RGB", fontsize=10, pad=6)

        for c, (arr, cmap, lim, title) in enumerate(
                [(lst, "magma", (v0, v1), "Landsat ST (K)"),
                 (ano[r], "RdBu_r", (-am, am), "Landsat ST anomaly (K)")], start=1):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#e8e8e8")
            ims[c] = ax.imshow(arr, cmap=cmap, vmin=lim[0], vmax=lim[1],
                               interpolation="nearest")
            _mark_ls(ax, inside, labels, dt, args.depth, T)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10, pad=6)
            g = arr[np.isfinite(arr)]
            if g.size and c == 1:
                ax.text(.03, .04, f"mean {g.mean():.1f}", transform=ax.transAxes,
                        fontsize=7.2, color="w", va="bottom",
                        bbox=dict(fc="black", alpha=.55, pad=1.6, lw=0))
            ax.text(.97, .04, f"{int(np.isfinite(arr).sum())}/{arr.size} px",
                    transform=ax.transAxes, fontsize=7, color="w", ha="right",
                    va="bottom", bbox=dict(fc="black", alpha=.45, pad=1.4, lw=0))

    for c in (1, 2):
        cax = fig.add_subplot(gs[n, c])
        fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=7.5, length=2)

    # the observed soil-moisture record, drawn by the same function the ECOSTRESS figure
    # uses, so the two are directly comparable panel for panel
    axt = fig.add_subplot(gs[n + 1, :])
    _timeseries(axt, inside, labels, args.depth, [r[1] for r in rows])

    fig.suptitle(
        f"{cid}   rep {cl.rep_station}   {cl.n_stations} station(s), "
        f"extent {cl.extent_km:.2f} km   Landsat C2 L2 ST at 30 m, 2.24 km window\n"
        f"DAYTIME ONLY -- Collection-2 L2 ST has no night overpass, so there is no "
        f"Landsat DTR (29.2).   Markers show {args.depth} cm observed SM.",
        fontsize=10.5, y=0.995)

    out = OUT_DIR / f"landsat_{cid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("  wrote %s", out)

    flat = ano.reshape(n, -1)
    m = np.isfinite(flat).all(0)
    if n > 1 and m.sum() > 30:
        C = np.corrcoef(flat[:, m])
        iu = np.triu_indices(n, 1)
        log.info("  ST-anomaly spatial r, single date vs single date (%d common px): "
                 "mean %+.3f  min %+.3f  max %+.3f  (attenuated -- not 29's +0.967, "
                 "which is date vs annual MEAN)",
                 int(m.sum()), C[iu].mean(), C[iu].min(), C[iu].max())
    return True


def _mark_s2(ax, inside, labels, date, depth):
    for _, m in inside.iterrows():
        v, q = _sm_at(labels, m.station_id, depth, date)
        ax.plot(m.col, m.row, marker="o" if m.is_rep else "s",
                ms=6.5 if m.is_rep else 5.0, mfc="none" if q == 1 else "#00E5FF",
                mec="white", mew=1.3, zorder=5)
        ax.annotate(f"{v:.3f}" if np.isfinite(v) else "n/a", (m.col, m.row),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.4,
                    color="w", zorder=6, bbox=dict(fc="black", alpha=.55, pad=1.0, lw=0))


def _mark_ls(ax, inside, labels, date, depth, T):
    """Landsat pixel from the GeoTIFF's OWN affine -- stackstac snaps bounds outward, so
    the nominal grid from aoi.json is a pixel short (extract_lst_timeseries.py:10-11)."""
    for _, m in inside.iterrows():
        rr, cc = rowcol(T, float(m.utm_x), float(m.utm_y))
        v, q = _sm_at(labels, m.station_id, depth, date)
        ax.plot(cc, rr, marker="o" if m.is_rep else "s",
                ms=6.5 if m.is_rep else 5.0, mfc="none" if q == 1 else "#00E5FF",
                mec="black", mew=1.3, zorder=5)
        ax.annotate(f"{v:.3f}" if np.isfinite(v) else "n/a", (cc, rr),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.4,
                    color="w", zorder=6, bbox=dict(fc="black", alpha=.55, pad=1.0, lw=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=str(REPO / "csvs" / "gra_thermal_clusters.csv"))
    ap.add_argument("--members",  default=str(REPO / "csvs" / "gra_thermal_members.csv"))
    ap.add_argument("--cluster", default="")
    ap.add_argument("--months", default="5,6,7,8,9")
    ap.add_argument("--min-clear", type=float, default=0.60)
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--max-s2-days", type=int, default=45,
                    help="wider than the ECOSTRESS figure: S2 was S2A-only "
                         "until mid-2017 and many Landsat scenes predate S2B")
    ap.add_argument("--max-cloud", type=float, default=0.05)
    ap.add_argument("--min-members", type=int, default=2)
    args = ap.parse_args()
    args.months = [int(x) for x in args.months.split(",") if x.strip()]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C = pd.read_csv(args.clusters)
    M = pd.read_csv(args.members)
    C["rep_category"] = C.cluster_id.map(M[M.is_rep == 1].set_index("cluster_id")["category"])
    if args.cluster:
        C = C[C.cluster_id == args.cluster]
    C = C[C.n_stations >= args.min_members]
    log.info("rendering %d cluster(s), months %s, clear >= %.0f%%",
             len(C), args.months, 100 * args.min_clear)

    ok = 0
    for _, cl in C.iterrows():
        ok += bool(draw(cl, M[M.cluster_id == cl.cluster_id], args))
    log.info("")
    log.info("%d/%d Landsat figures written to %s", ok, len(C), OUT_DIR)


if __name__ == "__main__":
    main()
