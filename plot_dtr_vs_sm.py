#!/usr/bin/env python
"""§36.21(iii) -- DTR against observed soil moisture.

Three panels, and the ORDER is the point:

  A  WITHIN-STATION   both variables centred on each station's own mean.  This is the
                      only panel that asks the physical question -- when THIS place gets
                      wetter, does its diurnal temperature range shrink?
  B  BETWEEN-STATION  one point per station, its mean DTR against its mean SM.
  C  POOLED           every station-date together, which mixes A and B.

§29.13 ran the pooled version for daytime LST, got +0.167, and the within-station answer
was -0.077 -- a Simpson's paradox, the pooled number carried by station identity rather
than by moisture.  So within-station is reported first and the pooled panel is drawn last,
labelled as the artefact-prone one.

LABELS come from /projects/prjs1968/zarr_tokens, NOT dataset.ZARR_ROOT -- that path points
at scratch, which has been purged (the category dirs survive, the stations do not).  qc==0
only, so gap-filled days never enter.  "avg soil moisture" is the mean across the depth
bins present for that station-date; the 0-10 cm surface series is scored too and printed,
because thermal only senses the surface.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")

import dataset as _ds                       # noqa: E402
_ds.ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")     # the purge fix, see docstring
from dataset import SM_DEPTHS, _load_zarr_labels, _open_zarr  # noqa: E402
from census_ecostress import ROOT, setup_logging             # noqa: E402

ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")
SURFACE = "0-10"


def obs_for(station: str, category: str):
    """-> DataFrame[date, depth, obs], observed days only (qc == 0)."""
    zg = _open_zarr(ZARR_ROOT / category / station, category)
    if zg is None:
        return None
    out = _load_zarr_labels(zg)
    if out is None:
        return None
    sm, depths, times, qc = out
    rows = []
    for d, depth in enumerate(depths):
        depth = depth.decode() if isinstance(depth, bytes) else str(depth)
        if depth not in SM_DEPTHS:
            continue
        keep = ~np.isnan(sm[d])
        if qc is not None:
            keep &= (qc[d] == 0)
        if not keep.any():
            continue
        rows.append(pd.DataFrame({"date": pd.to_datetime(times[keep]).normalize(),
                                  "depth": depth, "obs": sm[d][keep].astype(np.float32)}))
    return pd.concat(rows, ignore_index=True) if rows else None


def one_station(job):
    path, sid, folder, cat, min_usable = job
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:                                      # noqa: BLE001
        return None
    ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] > 0)
    if ok.sum() < min_usable:
        return None

    dtr = z["dtr_k"][ok].reshape(int(ok.sum()), -1)
    val = z["valid"][ok].reshape(int(ok.sum()), -1).astype(bool)
    with np.errstate(invalid="ignore"):
        scene = np.array([d[v].mean() if v.any() else np.nan for d, v in zip(dtr, val)])
        # day and night separately, on the SAME valid pixels, so DTR == day - night
        # exactly and the three can be compared without a masking difference.
        dayl = z["day_lst_k"][ok].reshape(int(ok.sum()), -1)
        nigl = z["night_lst_k"][ok].reshape(int(ok.sum()), -1)
        s_day = np.array([a[v].mean() if v.any() else np.nan for a, v in zip(dayl, val)])
        s_nig = np.array([a[v].mean() if v.any() else np.nan for a, v in zip(nigl, val)])
    dates = pd.to_datetime([str(x.decode() if isinstance(x, bytes) else x)[:10]
                            for x in z["day_utc"][ok]]).normalize()
    df = pd.DataFrame({"date": dates, "dtr_k": scene,
                       "day_lst_k": s_day, "night_lst_k": s_nig,
                       "n_valid_px": z["n_valid_px"][ok]})
    # Carry the 36.24 phase covariates through, so the dt band and the solar hour can be
    # conditioned on downstream instead of being averaged over.
    for c in ("dt_hours", "day_tst", "night_tst", "well_phased"):
        if c in z.files:
            v = z[c][ok]
            df[c] = v.astype(float) if v.dtype.kind in "fiub" else np.nan
    df = df.dropna(subset=["dtr_k"])

    o = obs_for(folder, cat)
    if o is None or df.empty:
        return None
    avg = o.groupby("date", as_index=False)["obs"].mean().rename(columns={"obs": "sm_avg"})
    sur = (o[o["depth"] == SURFACE].groupby("date", as_index=False)["obs"].mean()
           .rename(columns={"obs": "sm_surface"}))
    j = df.merge(avg, on="date", how="inner").merge(sur, on="date", how="left")
    if j.empty:
        return None
    j["station_id"], j["folder"] = sid, folder
    return j


def rval(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=str(ROOT / "csvs" / "ecostress_dtr_bundles.all.csv"))
    ap.add_argument("--min-dates", type=int, default=1, help="paired days per station")
    ap.add_argument("--min-usable", type=int, default=1,
                    help="usable DTR pairs a bundle needs before it is opened at all")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out-tag", default="dtr_vs_sm")
    args = ap.parse_args()

    setup_logging(f"plot_{args.out_tag}")
    log = logging.getLogger("dtrsm")

    b = pd.read_csv(args.bundles)
    b = b[b["n_pairs_usable"] >= args.min_usable]
    log.info("bundles           : %d stations", len(b))

    jobs = [(r["path"], r["station_id"], r["folder"], r["category"], args.min_usable)
            for _, r in b.iterrows()]
    with Pool(min(args.workers, max(len(jobs), 1))) as pool:
        parts = [p for p in pool.map(one_station, jobs, chunksize=4) if p is not None]
    if not parts:
        log.error("no station produced a DTR/SM join -- nothing to plot")
        return
    d = pd.concat(parts, ignore_index=True)
    log.info("joined            : %d station-days over %d stations",
             len(d), d["station_id"].nunique())

    n = d.groupby("station_id")["dtr_k"].transform("size")
    d = d[n >= args.min_dates].copy()
    log.info("after >= %d paired days: %d rows over %d stations",
             args.min_dates, len(d), d["station_id"].nunique())
    if d.empty:
        log.error("nothing left after the minimum-days filter")
        return

    d.to_csv(ROOT / "csvs" / f"ecostress_{args.out_tag}.csv", index=False)

    for col, lbl in (("sm_avg", "depth-averaged SM"), ("sm_surface", "0-10 cm SM")):
        d[f"dtr_anom_{col}"] = d["dtr_k"] - d.groupby("station_id")["dtr_k"].transform("mean")
        d[f"sm_anom_{col}"] = d[col] - d.groupby("station_id")[col].transform("mean")

    res = {}
    for col, lbl in (("sm_avg", "depth-averaged SM"), ("sm_surface", "0-10 cm SM")):
        rw, nw = rval(d[f"sm_anom_{col}"].to_numpy(), d[f"dtr_anom_{col}"].to_numpy())
        g = d.groupby("station_id").agg(dtr=("dtr_k", "mean"), sm=(col, "mean")).dropna()
        rb, nb = rval(g["sm"].to_numpy(), g["dtr"].to_numpy())
        rp, np_ = rval(d[col].to_numpy(), d["dtr_k"].to_numpy())
        res[col] = (rw, nw, rb, nb, rp, np_)
        log.info("")
        log.info("=== %s ===", lbl)
        log.info("  WITHIN-station  r = %+.3f  (n = %d station-days)", rw, nw)
        log.info("  BETWEEN-station r = %+.3f  (n = %d stations)", rb, nb)
        log.info("  POOLED          r = %+.3f  (n = %d)  <- the 29.13 artefact position",
                 rp, np_)

    # per-station within r, so the sign test is available rather than one pooled number
    per = []
    for sid, g in d.groupby("station_id"):
        r, nn = rval(g["sm_avg"].to_numpy(), g["dtr_k"].to_numpy())
        if np.isfinite(r):
            per.append({"station_id": sid, "r": r, "n": nn})
    per = pd.DataFrame(per)
    if len(per):
        frac_neg = float((per["r"] < 0).mean())
        log.info("")
        log.info("per-station within r (depth-averaged SM): median %+.3f, "
                 "%.1f%% negative, n = %d stations",
                 per["r"].median(), 100 * frac_neg, len(per))
        log.info("  (thermal inertia predicts NEGATIVE: wetter -> smaller DTR. "
                 "A coin flip is 50%%.)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    C = "#2E7D5B"

    ax = axes[0]
    ax.scatter(d["sm_anom_sm_avg"], d["dtr_anom_sm_avg"], s=5, alpha=.18, lw=0, color=C)
    rw, nw = res["sm_avg"][0], res["sm_avg"][1]
    ax.set_title(f"A  WITHIN-station (the physical question)\nr = {rw:+.3f}   n = {nw:,}",
                 fontsize=10.5)
    ax.set_xlabel("SM anomaly from station mean (m³/m³)")
    ax.set_ylabel("DTR anomaly from station mean (K)")
    ax.axhline(0, lw=.7, color="#bbb"); ax.axvline(0, lw=.7, color="#bbb")

    ax = axes[1]
    g = d.groupby("station_id").agg(dtr=("dtr_k", "mean"), sm=("sm_avg", "mean"),
                                    n=("dtr_k", "size")).dropna()
    ax.scatter(g["sm"], g["dtr"], s=np.clip(g["n"], 8, 90), alpha=.6, lw=0, color="#3B6EA5")
    rb, nb = res["sm_avg"][2], res["sm_avg"][3]
    ax.set_title(f"B  BETWEEN-station (one point per station)\nr = {rb:+.3f}   n = {nb}",
                 fontsize=10.5)
    ax.set_xlabel("station mean SM (m³/m³)"); ax.set_ylabel("station mean DTR (K)")

    ax = axes[2]
    ax.scatter(d["sm_avg"], d["dtr_k"], s=5, alpha=.15, lw=0, color="#C1502E")
    rp, np_ = res["sm_avg"][4], res["sm_avg"][5]
    ax.set_title(f"C  POOLED — mixes A and B (29.13's artefact)\nr = {rp:+.3f}   n = {np_:,}",
                 fontsize=10.5)
    ax.set_xlabel("SM, depth-averaged (m³/m³)"); ax.set_ylabel("DTR (K)")

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9)

    fig.suptitle("ECOSTRESS DTR against observed soil moisture — "
                 f"{d['station_id'].nunique()} stations, {len(d):,} station-days "
                 "(qc==0, observed only)\n"
                 "thermal inertia predicts a NEGATIVE slope: wetter ground swings less",
                 fontsize=11.5, y=1.045)
    fig.tight_layout()
    p = ROOT / "fig" / "dtr_txson" / f"{args.out_tag}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    log.info("wrote %s", p)


if __name__ == "__main__":
    main()
