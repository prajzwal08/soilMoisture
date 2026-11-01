#!/usr/bin/env python
"""Step 2 of the grassland thermal-figure plan: one map figure per cluster.

Generalises plot_dtr_txson.py, which drew exactly these panels but for a single station
and a single window.  Three things change:

  1. ONE FIGURE PER CLUSTER, drawn in the representative station's window, with every
     member marked.  Run per-station over TxSON and you get ~20 near-identical,
     heavily-overlapping figures.

  2. DATES ARE ONE PER SUMMER MONTH (Jun-Sep), years pooled, each the best-covered pair
     in its month.  Pooling years is the test, not a compromise: if the pattern looks the
     same in Jul 2019, Aug 2021 and Sep 2020, that is 29's r = +0.967 made visual.

  3. THE S2 SCENE IS CHOSEN BY CLOUD MASK, not by mean blue reflectance.  The mask is
     the 7-class SEnSeIv2 output in the token store; "cloudy" is classes 3,4,5 only --
     using `cm != 0` counts water and snow and marks every lakeside or winter station
     permanently cloudy, which would bite on the SNOTEL and iRON clusters here.

Below the maps, the observed soil moisture of every member over the whole record, with
the chosen dates marked -- because the map grid is uninterpretable without knowing
whether the probes were actually spread in wetness on those dates.

Env: terramind.  Nothing runs on the login node -- submit it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path("/gpfs/work3/0/prjs1968/soilMoisture")
sys.path.insert(0, str(REPO))
import dataset as _ds                                           # noqa: E402
_ds.ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")
from dataset import SM_DEPTHS, _load_zarr_labels, _open_zarr     # noqa: E402

DATA_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
SAT_ZARR    = Path("/projects/prjs1968/satellite_zarr")
TOKEN_ROOT  = Path("/projects/prjs1968/zarr_tokens")
OUT_DIR     = REPO / "fig" / "gra_thermal"

S2_RGB_IDX  = (3, 2, 1)          # B04, B03, B02 in the stored band order
S2_OFFSET   = 1000.0             # ESA baseline-04.00: 1000 DN == 0.0 reflectance
S2_SCALE    = 10000.0
CLOUD_CLS   = (3, 4, 5)          # thin cloud, thick cloud, shadow
N_ECO_PX    = 1024               # 32 x 32

log = logging.getLogger("gra_thermal")


# ---------------------------------------------------------------- data loading

def load_bundle(folder: str):
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        hits = sorted((DATA_ROOT / cat / folder / "ECOSTRESS").glob(f"{folder}_dtr_*.npz"))
        if hits:
            return np.load(hits[-1], allow_pickle=False), hits[-1]
    return None, None


def _as_str(x) -> str:
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def pick_summer_dates(z, months, min_valid_frac):
    """One pair per month: the best-covered aligned pair clearing the valid-px floor.

    Returns [(month, index)] ordered by month.  Years will differ between entries --
    that is intended, and the full date is printed on every row.
    """
    ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] >= min_valid_frac * N_ECO_PX)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return []
    days = np.array([_as_str(z["day_utc"][i])[:10] for i in idx])
    mon = np.array([int(d[5:7]) for d in days])
    out = []
    for m in months:
        sel = idx[mon == m]
        if sel.size == 0:
            continue
        out.append((m, int(sel[np.argmax(z["n_valid_px"][sel])])))
    return out


def cloud_fraction(folder: str, cat: str):
    """-> dict {YYYYMMDD: cloud_frac} from the token store, or {} if unavailable.

    Classes 3,4,5 only.  cm/dates is NOT guaranteed positionally aligned to s2/dates,
    so the caller joins on the date string; cm/dates is <U8 while s2/dates is |S8.
    """
    import zarr
    p = TOKEN_ROOT / cat / folder
    if not (p / ".complete").exists():
        return {}
    try:
        g = zarr.open_consolidated(str(p), mode="r")
    except KeyError:
        g = zarr.open_group(str(p), mode="r")
    if "cm" not in g:
        return {}
    try:
        masks = g["cm"]["masks"][:]                 # one 128-deep chunk; read once
        dates = [_as_str(d) for d in g["cm"]["dates"][:]]
    except Exception as exc:                        # noqa: BLE001
        log.warning("    cloud mask unreadable for %s: %s", folder, exc)
        return {}
    frac = np.isin(masks, CLOUD_CLS).reshape(len(dates), -1).mean(1)
    return dict(zip(dates, frac.astype(float)))


def s2_rgb_clear(folder: str, cat: str, want: str, max_days: int, max_cloud: float):
    """-> (rgb[224,224,3], date, offset_days, cloud_frac) or None.

    NEAREST CLEAR, not nearest.  Falls back to least-cloudy in the window, then to
    nearest at all, and reports which rule fired via the returned cloud fraction.
    """
    import zarr
    p = SAT_ZARR / f"{folder}.zarr"
    if not p.exists():
        return None
    g = zarr.open_group(str(p), mode="r")           # raw store has no .zmetadata
    if "s2" not in g:
        return None
    dates = np.array([_as_str(d) for d in g["s2"]["dates"][:]])
    wd = pd.Timestamp(str(want))
    off = np.array([(pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}") - wd).days for d in dates])
    cand = np.where(np.abs(off) <= max_days)[0]
    if cand.size == 0:
        return None

    cf = cloud_fraction(folder, cat)
    cfrac = np.array([cf.get(dates[i], np.nan) for i in cand])
    clear = cand[np.nan_to_num(cfrac, nan=1.0) < max_cloud]
    if clear.size:
        i = int(clear[np.argmin(np.abs(off[clear]))])
    elif np.isfinite(cfrac).any():
        i = int(cand[int(np.nanargmin(cfrac))])
    else:
        i = int(cand[int(np.argmin(np.abs(off[cand])))])

    cube = np.asarray(g["s2"]["data"][i], dtype=np.float32)      # [12,224,224]
    rgb = np.stack([cube[b] for b in S2_RGB_IDX], -1)
    rgb = np.where(rgb == 0, np.nan, (rgb - S2_OFFSET) / S2_SCALE)   # DN 0 = nodata
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(rgb, nan=0.0), dates[i], int(off[i]), float(cf.get(dates[i], np.nan))


def member_labels(folder: str, cat: str):
    """-> {depth: DataFrame(date, sm, qc)} for every depth present, gap-fill KEPT.

    qc 0 = observed, 1 = gap-filled (month-day climatology), 2 = still missing.  The
    fill is climatological, so a gap-filled value is not an observation and must not be
    drawn as one.
    """
    zg = _open_zarr(TOKEN_ROOT / cat / folder, cat)
    out = _load_zarr_labels(zg) if zg is not None else None
    if out is None:
        return {}
    sm, depths, times, qc = out
    t = pd.to_datetime(times).normalize()
    res = {}
    for d, dep in enumerate(depths):
        dep = _as_str(dep)
        q = qc[d] if qc is not None else np.zeros(sm.shape[1], np.uint8)
        res[dep] = pd.DataFrame({"date": t, "sm": sm[d].astype(np.float32),
                                 "qc": np.asarray(q).astype(np.int16)})
    return res


# ---------------------------------------------------------------- the figure

def draw_cluster(cl, mem, args) -> bool:
    cid, folder, rep = cl.cluster_id, cl.rep_folder, cl.rep_station
    z, path = load_bundle(folder)
    if z is None:
        log.warning("%s: no ECOSTRESS bundle for rep %s -- skipped", cid, folder)
        return False

    rows = pick_summer_dates(z, args.months, args.min_valid_frac)
    if not rows:
        log.warning("%s: no %s pair clears %.0f%% valid -- skipped",
                    cid, "/".join(map(str, args.months)), 100 * args.min_valid_frac)
        return False
    idx = [i for _, i in rows]
    got = {m for m, _ in rows}
    missing = [m for m in args.months if m not in got]
    log.info("%s: rep %s, %d/%d month(s) %s", cid, rep, len(rows), len(args.months),
             [f"{m:02d}:{_as_str(z['day_utc'][i])[:10]}" for m, i in rows])
    if missing:
        # Distinguish "cloudy" from "never observed": if the best pair in a month has a
        # valid fraction of 0, every overpass that month was off-swath and no threshold
        # recovers it.  Swath loss is all-or-nothing.
        mon_all = np.array([int(_as_str(d)[5:7]) for d in z["day_utc"]])
        vf = z["n_valid_px"] / N_ECO_PX
        note = []
        for m in missing:
            s_m = (mon_all == m) & (z["grid_aligned"] == 1)
            if not s_m.any():
                note.append(f"{m:02d}:no aligned pair")
            else:
                b = float(vf[s_m].max())
                note.append(f"{m:02d}:best {b:.2f}" + (" OFF-SWATH" if b == 0 else ""))
        log.info("       months not rendered: %s", "  ".join(note))

    day = np.where(z["valid"][idx].astype(bool), z["day_lst_k"][idx], np.nan)
    nig = np.where(z["valid"][idx].astype(bool), z["night_lst_k"][idx], np.nan)
    dtr = z["dtr_k"][idx]

    inside = mem[mem.in_tile == 1].reset_index(drop=True)
    labels = {r.station_id: member_labels(r.folder, r.category) for _, r in inside.iterrows()}

    def lim(a, p=2):
        f = a[np.isfinite(a)]
        return (float(np.percentile(f, p)), float(np.percentile(f, 100 - p))) if f.size else (0.0, 1.0)

    # Each date minus its OWN scene mean.  The raw columns share one absolute scale so
    # rows stay comparable in level, but scene means differ by >10 K between months, which
    # renders the cooler row nearly flat.  The anomaly column is where the spatial pattern
    # is actually legible -- and comparing it across rows is the G0 question.
    ano = np.stack([d - np.nanmean(d) for d in dtr])
    am = float(np.nanmax(np.abs([np.nanpercentile(ano, 2), np.nanpercentile(ano, 98)])))

    cols = [("Sentinel-2 RGB", None, None),
            ("Day LST (K)", "magma", lim(day)),
            ("Night LST (K)", "magma", lim(nig)),
            ("DTR = day - night (K)", "viridis", lim(dtr)),
            ("DTR anomaly (K)", "RdBu_r", (-am, am))]
    arrs = [None, day, nig, dtr, ano]

    n = len(rows)
    fig = plt.figure(figsize=(16.0, 3.0 * n + 5.2))
    gs = GridSpec(n + 2, 5, figure=fig,
                  height_ratios=[1] * n + [0.10, 1.25],
                  hspace=0.34, wspace=0.07)
    ims = [None] * 5

    for r, (month, i) in enumerate(rows):
        du = _as_str(z["day_utc"][i])[:10]
        nu = _as_str(z["night_utc"][i])[:10]
        dt_h = float(z["dt_hours"][i])

        # --- RGB
        ax = fig.add_subplot(gs[r, 0])
        got = s2_rgb_clear(folder, cl.rep_category, du, args.max_s2_days, args.max_cloud)
        if got is None:
            ax.text(.5, .5, f"no S2 within {args.max_s2_days} d", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#666")
            ax.set_facecolor("#f0f0f0")
            ax.set_xlim(0, 224); ax.set_ylim(224, 0)
        else:
            rgb, sd, off, cfr = got
            ax.imshow(rgb)
            tag = f"S2 {sd}  ({off:+d} d)"
            tag += f"  cloud {cfr:.0%}" if np.isfinite(cfr) else "  cloud n/a"
            ax.text(.03, .04, tag, transform=ax.transAxes, fontsize=7.2, color="w",
                    va="bottom", bbox=dict(fc="black", alpha=.55, pad=1.6, lw=0))
        _mark(ax, inside, labels, du, args.depth, scale=1.0, rep=rep)
        ax.set_ylabel(f"{du}\nnight {nu}\ndt {dt_h:.1f} h", fontsize=8.5,
                      rotation=0, ha="right", va="center", labelpad=8)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[0][0], fontsize=10, pad=6)

        # --- thermal columns
        for c in (1, 2, 3, 4):
            title, cmap, (v0, v1) = cols[c]
            arr = arrs[c][r]
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#e8e8e8")
            ims[c] = ax.imshow(arr, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            _mark(ax, inside, labels, du, args.depth, scale=None, rep=rep)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10, pad=6)
            f = arr[np.isfinite(arr)]
            if f.size and c != 4:
                ax.text(.03, .04, f"mean {f.mean():.1f}", transform=ax.transAxes,
                        fontsize=7.2, color="w", va="bottom",
                        bbox=dict(fc="black", alpha=.55, pad=1.6, lw=0))
            ax.text(.97, .04, f"{int(np.isfinite(arr).sum())}/1024 px",
                    transform=ax.transAxes, fontsize=7, color="w", ha="right",
                    va="bottom", bbox=dict(fc="black", alpha=.45, pad=1.4, lw=0))

    for c in (1, 2, 3, 4):
        cax = fig.add_subplot(gs[n, c])
        fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=7.5, length=2)

    # --- soil moisture time series, full width
    axt = fig.add_subplot(gs[n + 1, :])
    _timeseries(axt, inside, labels, args.depth,
                [_as_str(z["day_utc"][i])[:10] for _, i in rows])

    lat, lon = float(z["latitude"]), float(z["longitude"])
    miss = cl.n_stations - cl.members_inside_tile
    extra = f"   |   {miss} member(s) outside this window" if miss else ""
    fig.suptitle(
        f"{cid}   rep {rep}   ({lat:.4f}, {lon:.4f})   "
        f"{cl.n_stations} station(s), extent {cl.extent_km:.2f} km{extra}\n"
        f"2.24 km x 2.24 km   ECOSTRESS L2T LSTE v002 at 70 m   "
        f"colour scales SHARED DOWN EACH COLUMN, so rows are comparable   "
        f"grey = no valid LST",
        fontsize=10.5, y=0.997)

    out = OUT_DIR / f"ecostress_{cid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("  wrote %s", out)

    # The number the eye cannot give: is the DTR pattern the SAME every date?
    ano = np.stack([d - np.nanmean(d) for d in dtr])
    flat = ano.reshape(len(idx), -1)
    m = np.isfinite(flat).all(0)
    if len(idx) > 1 and m.sum() > 30:
        C = np.corrcoef(flat[:, m])
        iu = np.triu_indices(len(idx), 1)
        log.info("  DTR-anomaly spatial r, SINGLE DATE vs SINGLE DATE (%d common px): "
                 "mean %+.3f  min %+.3f  max %+.3f  over %d month-pair(s)",
                 int(m.sum()), C[iu].mean(), C[iu].min(), C[iu].max(), len(iu[0]))
        log.info("       NOT comparable to 29's +0.967 (date vs annual MEAN) or G0's "
                 "+0.661 (split-half): averaging suppresses per-date noise in those and "
                 "not in this one, so a low value here is attenuation, not evidence of a "
                 "dynamic field. Use split-half to make the claim.")
    return True


def _sm_at(labels, sid, depth, date):
    """-> (value, qc) or (nan, -1).  qc 1 means climatological fill, not an observation."""
    d = labels.get(sid, {}).get(depth)
    if d is None:
        return np.nan, -1
    hit = d[d.date == pd.Timestamp(date)]
    if hit.empty:
        return np.nan, -1
    return float(hit.sm.iloc[0]), int(hit.qc.iloc[0])


def _mark(ax, inside, labels, date, depth, scale, rep):
    """scale=1.0 -> 224 px S2 grid (row/col); scale=None -> 32 px ECOSTRESS grid."""
    for _, m in inside.iterrows():
        y, x = (m.row, m.col) if scale else (m.eco_row, m.eco_col)
        v, q = _sm_at(labels, m.station_id, depth, date)
        edge = "white" if scale else "black"
        ax.plot(x, y, marker="o" if m.is_rep else "s", ms=6.5 if m.is_rep else 5.0,
                mfc="none" if q == 1 else "#00E5FF", mec=edge, mew=1.3, zorder=5)
        txt = f"{v:.3f}" if np.isfinite(v) else "n/a"
        if q == 1:
            txt += "*"
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=6.4, color="w", zorder=6,
                    bbox=dict(fc="black", alpha=.55, pad=1.0, lw=0))


def _timeseries(ax, inside, labels, depth, dates):
    cmap = plt.get_cmap("tab10")
    any_line = False
    for k, (_, m) in enumerate(inside.iterrows()):
        d = labels.get(m.station_id, {}).get(depth)
        if d is None or d.empty:
            continue
        any_line = True
        col = cmap(k % 10)
        obs = d[d.qc == 0]
        fil = d[d.qc == 1]
        ax.plot(obs.date, obs.sm, lw=0.9, color=col, label=str(m.station_id))
        if not fil.empty:
            ax.plot(fil.date, fil.sm, lw=0.7, color=col, alpha=0.28, ls=":")
    for dt in dates:
        ax.axvline(pd.Timestamp(dt), color="#444", lw=0.9, ls="--", alpha=.75, zorder=1)
        ax.annotate(dt, (pd.Timestamp(dt), 0.97), xycoords=("data", "axes fraction"),
                    fontsize=6.4, rotation=90, va="top", ha="right", color="#333",
                    bbox=dict(fc="white", alpha=.7, pad=0.8, lw=0))
    ax.set_ylabel(f"{depth} cm SM\n(m3/m3)", fontsize=9)
    ax.set_xlabel("date", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.25, lw=.5)
    if any_line:
        ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper left")
    else:
        ax.text(.5, .5, "no soil-moisture record loaded", ha="center", va="center",
                transform=ax.transAxes, color="#888")
    ax.set_title("observed soil moisture (solid = observed, dotted = gap-filled "
                 "climatology; * on a marker above means the value is gap-filled)",
                 fontsize=8.5, pad=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=str(REPO / "csvs" / "gra_thermal_clusters.csv"))
    ap.add_argument("--members",  default=str(REPO / "csvs" / "gra_thermal_members.csv"))
    ap.add_argument("--cluster", default="", help="render only this cluster_id")
    ap.add_argument("--months", default="5,6,7,8,9",
                    help="growing season. JJAS alone yields only 2 columns at "
                         "most TxSON clusters -- not because of cloud but because "
                         "whole months are off-swath (best valid fraction 0.00)")
    ap.add_argument("--min-valid-frac", type=float, default=0.75)
    ap.add_argument("--depth", default=SM_DEPTHS[0])
    ap.add_argument("--max-s2-days", type=int, default=16)
    ap.add_argument("--max-cloud", type=float, default=0.05)
    ap.add_argument("--min-members", type=int, default=2,
                    help="skip clusters smaller than this")
    args = ap.parse_args()
    args.months = [int(x) for x in args.months.split(",") if x.strip()]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C = pd.read_csv(args.clusters)
    M = pd.read_csv(args.members)
    # the representative's own category, needed to reach its token store
    reps = M[M.is_rep == 1].set_index("cluster_id")["category"]
    C["rep_category"] = C.cluster_id.map(reps)

    if args.cluster:
        C = C[C.cluster_id == args.cluster]
    C = C[C.n_stations >= args.min_members]
    log.info("rendering %d cluster(s), months %s, valid >= %.0f%%",
             len(C), args.months, 100 * args.min_valid_frac)

    ok = 0
    for _, cl in C.iterrows():
        ok += bool(draw_cluster(cl, M[M.cluster_id == cl.cluster_id], args))
    log.info("")
    log.info("%d/%d cluster figures written to %s", ok, len(C), OUT_DIR)


if __name__ == "__main__":
    main()
