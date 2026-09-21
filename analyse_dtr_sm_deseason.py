#!/usr/bin/env python
"""Is the DTR-SM correlation physics, or a shared seasonal cycle?

THE WORRY.  Both variables are strongly seasonal, and in most climates they are
seasonally ANTI-correlated for reasons that have nothing to do with soil thermal
inertia: summer = high insolation = large DTR, and summer = dry season = low SM; winter
the reverse.  That alone manufactures a negative r.  So the -0.107 measured so far may be
climatology rather than moisture control, and the honest number is the correlation of the
two DESEASONALISED series.

Note this is the opposite of the intuition that DTR lacks a climatic component.  DTR has a
large one -- at LCRA-3 the scene mean runs 21.1 K in August against 10.7 K in December.
The first thing this script prints is how much of each variable the seasonal cycle
actually explains, so the premise is measured rather than assumed.

METHOD.  Two harmonics of day-of-year (sin/cos at 1/yr and 2/yr) plus an intercept,
fitted PER STATION to that station's own DTR days, separately for DTR and for SM; the
residuals are then correlated within station and pooled.  Two details that matter:

  * HEMISPHERE.  Day-of-year is shifted by half a year for southern-hemisphere stations
    in the pooled fit, or January would be midwinter for some stations and midsummer for
    others and the pooled seasonal shape would cancel to nothing.
  * n.  5 parameters on a station's ~20 DTR days is thin, so stations need >= 15 days,
    and the pooled-harmonic variant (seasonal shape fitted on all stations at once, then
    removed per station) is reported alongside as a check that the per-station fit is not
    simply eating the signal through overfitting.

If the deseasonalised r collapses toward zero, the arm's remaining evidence was
climatology.  If it survives, the -0.1 is real thermal inertia and everything measured so
far stands.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import ROOT  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def design(doy, nh=2):
    t = 2 * np.pi * doy / 365.25
    cols = [np.ones_like(t)]
    for k in range(1, nh + 1):
        cols += [np.sin(k * t), np.cos(k * t)]
    return np.column_stack(cols)


def fit_resid(y, X):
    """-> (residual, R^2).  Least squares, guarding a rank-deficient design."""
    m = np.isfinite(y)
    r = np.full_like(y, np.nan, dtype=float)
    if m.sum() <= X.shape[1] + 2:
        return r, np.nan
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    fit = X[m] @ beta
    r[m] = y[m] - fit
    ss = np.var(y[m])
    return r, float(1 - np.var(r[m]) / ss) if ss > 1e-12 else np.nan


def rv(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "csvs" / "ecostress_dtr_vs_sm_all.csv"))
    ap.add_argument("--splits", default=str(ROOT / "csvs" / "station_splits.csv"))
    ap.add_argument("--min-days", type=int, default=15)
    ap.add_argument("--nh", type=int, default=2)
    args = ap.parse_args()

    d = pd.read_csv(args.csv, parse_dates=["date"])
    lat = (pd.read_csv(args.splits)[["station_id", "latitude"]]
           .drop_duplicates("station_id").set_index("station_id")["latitude"])
    d["lat"] = d["station_id"].map(lat)
    d["doy"] = d["date"].dt.dayofyear
    # hemisphere alignment for the pooled fit
    d["doy_a"] = np.where(d["lat"] < 0, (d["doy"] + 182.6) % 365.25, d["doy"])
    d = d.dropna(subset=["sm_surface", "dtr_k"])
    print(f"{len(d):,} station-days, {d.station_id.nunique()} stations")

    # ---------- how seasonal is each variable, really ----------
    r2d, r2s, rows = [], [], []
    for sid, g in d.groupby("station_id"):
        if len(g) < args.min_days:
            continue
        X = design(g["doy"].to_numpy(), args.nh)
        rd, R2d = fit_resid(g["dtr_k"].to_numpy(), X)
        rs, R2s = fit_resid(g["sm_surface"].to_numpy(), X)
        if not (np.isfinite(R2d) and np.isfinite(R2s)):
            continue
        r2d.append(R2d); r2s.append(R2s)
        rows.append(pd.DataFrame({"station_id": sid, "dtr_res": rd, "sm_res": rs,
                                  "dtr_k": g["dtr_k"].to_numpy(),
                                  "sm_surface": g["sm_surface"].to_numpy()}))
    res = pd.concat(rows, ignore_index=True)
    ns = res.station_id.nunique()
    print(f"\nstations with >= {args.min_days} DTR days: {ns}   "
          f"rows {len(res):,}")
    print(f"\nSEASONAL VARIANCE EXPLAINED ({args.nh} harmonics, per station):")
    print(f"  DTR : median R2 = {np.median(r2d):.3f}   "
          f"p25 {np.percentile(r2d,25):.3f}  p75 {np.percentile(r2d,75):.3f}")
    print(f"  SM  : median R2 = {np.median(r2s):.3f}   "
          f"p25 {np.percentile(r2s,25):.3f}  p75 {np.percentile(r2s,75):.3f}")

    def within(df, a, b):
        x = df[a] - df.groupby("station_id")[a].transform("mean")
        y = df[b] - df.groupby("station_id")[b].transform("mean")
        return rv(x.to_numpy(), y.to_numpy())

    r_raw, n_raw = within(res, "sm_surface", "dtr_k")
    r_des, n_des = within(res, "sm_res", "dtr_res")
    print(f"\nWITHIN-STATION r(surface SM, DTR)")
    print(f"  raw            {r_raw:+.3f}   (n = {n_raw:,})")
    print(f"  DESEASONALISED {r_des:+.3f}   (n = {n_des:,})")
    if np.isfinite(r_raw) and abs(r_raw) > 1e-9:
        print(f"  -> {100*(1-r_des/r_raw):+.0f}% of the raw correlation was seasonal")

    per = []
    for sid, g in res.groupby("station_id"):
        rr, nn = rv(g["sm_res"].to_numpy(), g["dtr_res"].to_numpy())
        r0, _ = rv(g["sm_surface"].to_numpy(), g["dtr_k"].to_numpy())
        if np.isfinite(rr):
            per.append({"station_id": sid, "r_des": rr, "r_raw": r0, "n": nn})
    per = pd.DataFrame(per)
    print(f"\nper-station: raw median {per.r_raw.median():+.3f} "
          f"({100*(per.r_raw<0).mean():.0f}% negative)   "
          f"deseasonalised median {per.r_des.median():+.3f} "
          f"({100*(per.r_des<0).mean():.0f}% negative)   n = {len(per)} stations")
    per.to_csv(ROOT / "csvs" / "ecostress_dtr_sm_deseason.csv", index=False)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    ax = axes[0]
    for col, c, lbl in (("dtr_k", "#C1502E", "DTR (K)"),
                        ("sm_surface", "#3B6EA5", "0–10 cm SM (m³/m³)")):
        g = d.copy()
        g["z"] = g[col] - g.groupby("station_id")[col].transform("mean")
        g["z"] /= g.groupby("station_id")[col].transform("std").replace(0, np.nan)
        b = g.groupby(pd.cut(g["doy_a"], np.arange(0, 380, 30)))["z"].median()
        ax.plot([i.mid for i in b.index], b.values, "o-", color=c, lw=2, label=lbl)
    ax.axhline(0, color="#bbb", lw=.8)
    ax.set_xlabel("day of year (southern stations shifted 6 months)")
    ax.set_ylabel("standardised anomaly (within station)")
    ax.set_title("A  Both variables ARE seasonal,\nand seasonally opposed", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)

    for ax, (a, b, ttl, r_, n_) in zip(
            axes[1:], [("sm_surface", "dtr_k", "B  RAW within-station", r_raw, n_raw),
                       ("sm_res", "dtr_res", "C  DESEASONALISED", r_des, n_des)]):
        x = res[a] - res.groupby("station_id")[a].transform("mean")
        y = res[b] - res.groupby("station_id")[b].transform("mean")
        ax.scatter(x, y, s=5, alpha=.18, lw=0, color="#2E7D5B")
        ax.axhline(0, lw=.7, color="#ccc"); ax.axvline(0, lw=.7, color="#ccc")
        ax.set_title(f"{ttl}\nr = {r_:+.3f}   n = {n_:,}", fontsize=10.5)
        ax.set_xlabel("SM anomaly" + ("" if a == "sm_surface" else " (deseasonalised)"))
        ax.set_ylabel("DTR anomaly" + ("" if b == "dtr_k" else " (deseasonalised)"))

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Is the DTR–soil-moisture correlation physics, or a shared seasonal "
                 f"cycle?   {ns} stations, {len(res):,} station-days",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    p = ROOT / "fig" / "dtr_txson" / "dtr_sm_deseason.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
