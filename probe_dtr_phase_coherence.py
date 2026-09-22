#!/usr/bin/env python
"""Is ECOSTRESS DTR's low spatial coherence real, or an artefact of variable overpass time?

The figures measure a single-date-vs-single-date anomaly correlation of ~+0.30 for DTR
against ~+0.75 for Landsat ST over the same windows, months and stations.  Before that is
called a result, two confounds have to be excluded, because ECOSTRESS rides the ISS and
Landsat is sun-synchronous at ~10:30:

  PHASE   if the pattern's SHAPE changes with solar time -- a 9 h field is shadow-driven,
          a 13 h field albedo- and moisture-driven -- then two passes at different times
          decorrelate for reasons that have nothing to do with soil moisture.
          TEST: stratify r by |dtst|, the solar-time gap between the two dates.
          r falling with |dtst| => phase is driving it.  Flat => the low r is real.

  NOISE   DTR is a DIFFERENCE of two retrievals, so its noise is ~sqrt(2) that of a single
          scene, and correlation is attenuated by noise.
          TEST: compute the same statistic for day LST and night LST separately, on the
          SAME pairs.  If day-alone is as coherent as Landsat and DTR is much lower, the
          subtraction is what costs the coherence, not the sampling time.

Level is NOT the issue and is not tested: a 2.24 km tile is imaged in one instant, so time
of day shifts the whole scene uniformly and the spatial centring cancels it exactly.
Amplitude is not the issue either -- Pearson r is scale-invariant.  Only SHAPE is at stake.

Env: terramind.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from plot_gra_thermal import load_bundle, _as_str, N_ECO_PX

log = logging.getLogger("dtr_phase")
MIN_VALID = 0.75


def spatial_r(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two spatial anomaly maps over their common finite pixels."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C = pd.read_csv("csvs/gra_thermal_clusters.csv")
    C = C[C.n_stations >= 2]

    rows = []
    for _, cl in C.iterrows():
        z, _ = load_bundle(cl.rep_folder)
        if z is None:
            continue
        vf = z["n_valid_px"] / N_ECO_PX
        keep = np.where((z["grid_aligned"] == 1) & (vf >= MIN_VALID))[0]
        if keep.size < 3:
            continue
        val = z["valid"].astype(bool)
        day = np.where(val, z["day_lst_k"], np.nan)
        nig = np.where(val, z["night_lst_k"], np.nan)
        dtr = np.where(val, z["dtr_k"], np.nan)
        tst_d = np.asarray(z["day_tst"], float)
        tst_n = np.asarray(z["night_tst"], float)
        dth = np.asarray(z["dt_hours"], float)
        dates = np.array([_as_str(d)[:10] for d in z["day_utc"]])

        for i, j in combinations(keep, 2):
            # solar-time gap, wrapped: tst is a clock, 23.5 and 0.5 are one hour apart
            dd = abs(tst_d[i] - tst_d[j]); dd = min(dd, 24 - dd)
            dn = abs(tst_n[i] - tst_n[j]); dn = min(dn, 24 - dn)
            rows.append(dict(
                cluster=cl.cluster_id, d1=dates[i], d2=dates[j],
                dtst_day=dd, dtst_night=dn,
                ddt_hours=abs(dth[i] - dth[j]),
                r_dtr=spatial_r(dtr[i], dtr[j]),
                r_day=spatial_r(day[i], day[j]),
                r_night=spatial_r(nig[i], nig[j])))

    T = pd.DataFrame(rows).dropna(subset=["r_dtr", "r_day", "r_night"])
    T.to_csv("csvs/dtr_phase_coherence.csv", index=False)
    log.info("%d date-pairs over %d clusters  (valid >= %.0f%%)",
             len(T), T.cluster.nunique(), 100 * MIN_VALID)
    if T.empty:
        return

    log.info("")
    log.info("=== NOISE TEST: same pairs, three fields ===")
    for k, nm in (("r_day", "day LST"), ("r_night", "night LST"), ("r_dtr", "DTR")):
        log.info("  %-10s mean %+.3f   median %+.3f   %.0f%% positive",
                 nm, T[k].mean(), T[k].median(), 100 * (T[k] > 0).mean())
    log.info("  [Landsat ST over the same windows/months measured +0.61 to +0.97]")

    log.info("")
    log.info("=== PHASE TEST: does coherence fall as the two passes' solar times diverge? ===")
    bands = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 24.0)]
    log.info("  %-14s %5s %10s %10s %10s", "|dtst_day| h", "n", "r_day", "r_night", "r_dtr")
    for lo, hi in bands:
        s = T[(T.dtst_day >= lo) & (T.dtst_day < hi)]
        if len(s) < 5:
            continue
        log.info("  %-14s %5d %10.3f %10.3f %10.3f", f"{lo:g}-{hi:g}", len(s),
                 s.r_day.mean(), s.r_night.mean(), s.r_dtr.mean())
    for k in ("dtst_day", "dtst_night", "ddt_hours"):
        for rk in ("r_day", "r_dtr"):
            if T[k].std() > 1e-9:
                log.info("  corr(%s, %s) = %+.3f", k, rk,
                         float(np.corrcoef(T[k], T[rk])[0, 1]))

    log.info("")
    log.info("=== per cluster ===")
    g = T.groupby("cluster").agg(n=("r_dtr", "size"), r_day=("r_day", "mean"),
                                 r_night=("r_night", "mean"), r_dtr=("r_dtr", "mean"),
                                 dtst=("dtst_day", "mean"))
    for c, r in g.iterrows():
        log.info("  %-26s n=%3d  day %+.3f  night %+.3f  DTR %+.3f  mean|dtst| %.2f h",
                 c, r.n, r.r_day, r.r_night, r.r_dtr, r.dtst)
    log.info("")
    log.info("wrote csvs/dtr_phase_coherence.csv")


if __name__ == "__main__":
    main()
