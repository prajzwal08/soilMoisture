#!/usr/bin/env python
"""One figure per cluster: S2 RGB, ECOSTRESS day/night/DTR, Landsat ST, and the observed
soil-moisture record -- all in the same image, one row per growing-season month.

    RGB | Day LST | Night LST | DTR | DTR anom | Landsat ST | Landsat ST anom
    ------------------------------------------------------------------------
    [ observed soil moisture, one line per member, whole record ]

CLEAR IMAGES ONLY.  A month is rendered only if it has an ECOSTRESS pair with essentially
every pixel valid AND a Landsat scene essentially fully clear.  Partial scenes are
dropped rather than drawn with grey holes: a pattern read off a half-missing map is not
comparable to one read off a full map, and the missing part is not missing at random --
29 measured 18.2% of the TxSON tile with no ST retrieval at all, and that region contains
the wettest station.

Colour scales are shared DOWN each column so rows are comparable, which is the whole
point: the question is whether the pattern is the same every month.

THREE GRIDS, NOT RESAMPLED.  S2 is 224 px at 10 m, ECOSTRESS 32 px at 70 m, Landsat
~75 px at 30 m, all covering the same 2240 m box.  The ECOSTRESS window is the
station-centred box snapped onto the ECOSTRESS grid (read_ecostress_lst.py:136-138), so
it is co-registered to the S2 window to within half a 70 m pixel. Good enough to see what
is where; not good enough to difference.

Env: terramind.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rasterio.transform import rowcol

from plot_gra_thermal import (REPO, OUT_DIR, N_ECO_PX, load_bundle, member_labels,
                              s2_rgb_clear, _as_str, _sm_at, _timeseries)
from plot_gra_landsat import scan_scenes

log = logging.getLogger("gra_combined")


def eco_clear_months(z, months, min_valid):
    """-> {month: index} for pairs with essentially every pixel valid."""
    ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] >= min_valid * N_ECO_PX)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return {}
    mon = np.array([int(_as_str(z["day_utc"][i])[5:7]) for i in idx])
    out = {}
    for m in months:
        sel = idx[mon == m]
        if sel.size:
            out[m] = int(sel[np.argmax(z["n_valid_px"][sel])])
    return out


def draw(cl, mem, args) -> bool:
    cid = cl.cluster_id
    z, _ = load_bundle(cl.rep_folder)
    if z is None:
        log.warning("%s: no ECOSTRESS bundle -- skipped", cid)
        return False

    # "Clear" has to mean AS CLEAR AS THIS TILE EVER GETS, not an absolute fraction.
    # 29 measured 18.2% of the TxSON tile with no ST retrieval at ALL, so its Landsat
    # clear fraction ceilings at ~82% permanently -- an absolute 99% cut discards the two
    # six-station clusters, which are the only tiles in the archive with >= 4 probes.
    # Same for ECOSTRESS: a tile can have a standing swath or retrieval gap.
    eco_ceiling = float((z["n_valid_px"][z["grid_aligned"] == 1] / N_ECO_PX).max()) \
        if (z["grid_aligned"] == 1).any() else 0.0
    ls_all = scan_scenes(cid, args.months, 0.0, None)
    ls_ceiling = max((r[5] for r in ls_all), default=0.0)

    eco_thr = min(args.min_valid, max(0.0, eco_ceiling - args.margin))
    ls_thr = min(args.min_clear, max(0.0, ls_ceiling - args.margin))
    log.info("%s: ceilings  ECOSTRESS %.2f -> thr %.2f   Landsat %.2f -> thr %.2f",
             cid, eco_ceiling, eco_thr, ls_ceiling, ls_thr)

    eco = eco_clear_months(z, args.months, eco_thr)
    anchor = {m: _as_str(z["day_utc"][i])[:10] for m, i in eco.items()}
    ls = {r[0]: r for r in scan_scenes(cid, args.months, ls_thr, anchor or None)}

    keep = [m for m in args.months if m in eco and m in ls]
    if not keep:
        log.warning("%s: no month is clear on BOTH sensors "
                    "(ECOSTRESS %s, Landsat %s) -- skipped",
                    cid, sorted(eco), sorted(ls))
        return False
    dropped = [m for m in args.months if m not in keep]
    log.info("%s: %d/%d month(s) clear on both: %s", cid, len(keep), len(args.months),
             [f"{m:02d} eco {anchor[m]} / ls {ls[m][1]}" for m in keep])
    if dropped:
        log.info("       dropped %s (eco-only %s, landsat-only %s, neither %s)",
                 dropped, [m for m in dropped if m in eco],
                 [m for m in dropped if m in ls],
                 [m for m in dropped if m not in eco and m not in ls])

    idx = [eco[m] for m in keep]
    val = z["valid"][idx].astype(bool)
    day = np.where(val, z["day_lst_k"][idx], np.nan)
    nig = np.where(val, z["night_lst_k"][idx], np.nan)
    dtr = z["dtr_k"][idx]
    dano = np.stack([d - np.nanmean(d) for d in dtr])
    lst = np.stack([ls[m][3] for m in keep])
    lano = np.stack([a - np.nanmean(a) for a in lst])

    inside = mem[mem.in_tile == 1].reset_index(drop=True)
    labels = {r.station_id: member_labels(r.folder, r.category) for _, r in inside.iterrows()}

    def lim(a, p=2):
        f = a[np.isfinite(a)]
        return (float(np.percentile(f, p)), float(np.percentile(f, 100 - p))) if f.size else (0., 1.)

    def sym(a):
        v = float(np.nanmax(np.abs([np.nanpercentile(a, 2), np.nanpercentile(a, 98)]))) \
            if np.isfinite(a).any() else 1.0
        return (-v, v)

    cols = [("Sentinel-2 RGB", None, None, None),
            ("ECO Day LST (K)",   "magma",   lim(day),  day),
            ("ECO Night LST (K)", "magma",   lim(nig),  nig),
            ("ECO DTR (K)",       "viridis", lim(dtr),  dtr),
            ("ECO DTR anom (K)",  "RdBu_r",  sym(dano), dano),
            ("Landsat ST (K)",    "magma",   lim(lst),  lst),
            ("Landsat ST anom (K)", "RdBu_r", sym(lano), lano)]

    n = len(keep)
    fig = plt.figure(figsize=(19.0, 2.75 * n + 5.4))
    gs = GridSpec(n + 2, 7, figure=fig, height_ratios=[1] * n + [0.09, 1.20],
                  hspace=0.30, wspace=0.06)
    ims = [None] * 7

    for r, m in enumerate(keep):
        i = eco[m]
        du, nu = _as_str(z["day_utc"][i])[:10], _as_str(z["night_utc"][i])[:10]
        lsd, lsT, lsfr = ls[m][1], ls[m][4], ls[m][5]

        ax = fig.add_subplot(gs[r, 0])
        got = s2_rgb_clear(cl.rep_folder, cl.rep_category, du,
                           args.max_s2_days, args.max_cloud)
        if got is None:
            ax.text(.5, .5, "no clear S2", ha="center", va="center",
                    transform=ax.transAxes, color="#666")
            ax.set_facecolor("#f0f0f0")
        else:
            rgb, sd, off, cfr = got
            ax.imshow(rgb)
            ax.text(.03, .04, f"S2 {sd} ({off:+d} d)", transform=ax.transAxes,
                    fontsize=6.8, color="w", va="bottom",
                    bbox=dict(fc="black", alpha=.55, pad=1.4, lw=0))
            _mark(ax, inside, labels, du, args.depth, "s2", None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"ECO {du}\nnight {nu}  dt {z['dt_hours'][i]:.1f} h\n"
                      f"Landsat {lsd}", fontsize=8.0, rotation=0, ha="right",
                      va="center", labelpad=8)
        if r == 0:
            ax.set_title(cols[0][0], fontsize=9.5, pad=6)

        for c in range(1, 7):
            title, cmap, (v0, v1), arr3 = cols[c]
            arr = arr3[r]
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#e8e8e8")
            ims[c] = ax.imshow(arr, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            _mark(ax, inside, labels, du, args.depth,
                  "ls" if c >= 5 else "eco", lsT if c >= 5 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9.5, pad=6)
            f = arr[np.isfinite(arr)]
            if f.size and c in (1, 2, 3, 5):
                ax.text(.03, .04, f"mean {f.mean():.1f}", transform=ax.transAxes,
                        fontsize=6.8, color="w", va="bottom",
                        bbox=dict(fc="black", alpha=.55, pad=1.4, lw=0))

    for c in range(1, 7):
        cax = fig.add_subplot(gs[n, c])
        fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=7, length=2)

    axt = fig.add_subplot(gs[n + 1, :])
    _timeseries(axt, inside, labels, args.depth, [anchor[m] for m in keep])

    miss = cl.n_stations - cl.members_inside_tile
    fig.suptitle(
        f"{cid}   rep {cl.rep_station}   {cl.n_stations} station(s), extent "
        f"{cl.extent_km:.2f} km"
        + (f"   |   {miss} member(s) outside the window" if miss else "") + "\n"
        f"CLEAREST SCENES ONLY (ECOSTRESS >= {100*eco_thr:.0f}% valid px, Landsat >= "
        f"{100*ls_thr:.0f}% clear -- each tile's own ceiling; TxSON's Landsat ceiling is "
        f"~82% because 18.2% of that tile has no ST retrieval at all, 29)   |   "
        f"2.24 km box on three grids: S2 10 m, ECOSTRESS 70 m, Landsat 30 m -- "
        f"co-registered to ~35 m, NOT resampled   |   "
        f"colour scales shared DOWN each column",
        fontsize=10.5, y=0.995)

    out = OUT_DIR / f"combined_{cid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("  wrote %s", out)

    for nm, A in (("ECO DTR", dano), ("Landsat ST", lano)):
        flat = A.reshape(n, -1)
        mm = np.isfinite(flat).all(0)
        if n > 1 and mm.sum() > 30:
            C = np.corrcoef(flat[:, mm])
            iu = np.triu_indices(n, 1)
            log.info("  %-11s anomaly spatial r across months (%d px): mean %+.3f "
                     "[%+.3f, %+.3f]", nm, int(mm.sum()), C[iu].mean(),
                     C[iu].min(), C[iu].max())
    return True


def _mark(ax, inside, labels, date, depth, grid, T):
    for _, m in inside.iterrows():
        if grid == "s2":
            y, x, edge = m.row, m.col, "white"
        elif grid == "eco":
            y, x, edge = m.eco_row, m.eco_col, "black"
        else:
            y, x = rowcol(T, float(m.utm_x), float(m.utm_y))
            edge = "black"
        v, q = _sm_at(labels, m.station_id, depth, date)
        ax.plot(x, y, marker="o" if m.is_rep else "s", ms=6.0 if m.is_rep else 4.6,
                mfc="none" if q == 1 else "#00E5FF", mec=edge, mew=1.2, zorder=5)
        ax.annotate(f"{v:.3f}" if np.isfinite(v) else "n/a", (x, y),
                    textcoords="offset points", xytext=(5, 3), fontsize=6.0,
                    color="w", zorder=6, bbox=dict(fc="black", alpha=.55, pad=.9, lw=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=str(REPO / "csvs" / "gra_thermal_clusters.csv"))
    ap.add_argument("--members",  default=str(REPO / "csvs" / "gra_thermal_members.csv"))
    ap.add_argument("--cluster", default="")
    ap.add_argument("--months", default="5,6,7,8,9")
    ap.add_argument("--min-valid", type=float, default=0.99,
                    help="ECOSTRESS valid-pixel fraction; 0.99 = essentially full scene")
    ap.add_argument("--min-clear", type=float, default=0.99,
                    help="Landsat clear fraction; 0.99 = essentially fully clear")
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--max-s2-days", type=int, default=45)
    ap.add_argument("--max-cloud", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="how far below a tile's own clear-fraction ceiling "
                         "a scene may fall and still count as clear")
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
    log.info("rendering %d cluster(s), months %s, ECO valid >= %.0f%%, Landsat clear >= %.0f%%",
             len(C), args.months, 100 * args.min_valid, 100 * args.min_clear)

    ok = 0
    for _, cl in C.iterrows():
        ok += bool(draw(cl, M[M.cluster_id == cl.cluster_id], args))
    log.info("")
    log.info("%d/%d combined figures written to %s", ok, len(C), OUT_DIR)


if __name__ == "__main__":
    main()
