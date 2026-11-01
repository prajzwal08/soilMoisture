#!/usr/bin/env python
"""What does an ST_QA gate actually cost, scene-level vs pixel-level, across all 993 stations?

SCENE-LEVEL means: drop a scene when the median ST_QA over its CLEAR pixels exceeds T.  That
number is already stored per scene as `median_st_qa` (download_landsat_st30.py computes it as
nanmedian(st_qa[clear & finite])), so the whole curve is a covariate read -- no pixel data.

PIXEL-LEVEL means: keep individual clear pixels with st_qa <= T, so partial scenes survive.

Step 0 measured within-tile ST_QA spread at 0.362 K against 1.365 K between scenes (ratio 0.266),
which predicts the two gates land in nearly the same place.  That prediction is tested here
rather than assumed, because if it is wrong the whole "ST_QA is a scene gate wearing a
pixel-level costume" argument collapses.

THREE WAYS TO COUNT THE LOSS, and they are not the same number:
  * SCENES retained -- what a scene gate is naturally expressed in
  * CLEAR PIXELS retained -- the actual supervision volume, since a scene with more clear
    pixels carries more gradient.  This is the number that matters for the head.
  * STATIONS still usable -- a gate that leaves a station with almost no scenes has removed
    that station from the dense-supervision task entirely, which no pixel count reveals.

OUTPUT  csvs/landsat_stqa_yield.csv   per-threshold, and per-threshold x kg_macro
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, SPLITS, qa_decode

OUT = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_stqa_yield.csv")
THRESH = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, np.inf]
MIN_SCENES = 20          # below this a station is effectively out of the dense task


def one(path_str: str) -> dict | None:
    z = np.load(path_str, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    med = np.asarray(z["median_st_qa"], dtype="float64")     # per scene, over clear pixels
    lst, stq = z["lst30"], z["st_qa30"]
    clear, _ = qa_decode(z["qa_pixel30"].astype("float64"))
    ok = clear & np.isfinite(lst) & np.isfinite(stq)
    n_clear_per_scene = ok.reshape(ok.shape[0], -1).sum(axis=1).astype("int64")

    r = {"station_id": meta["station_id"], "n_scenes": int(len(med)),
         "n_clear_px": int(n_clear_per_scene.sum()),
         "median_st_qa_station": float(np.nanmedian(med))}
    for t in THRESH:
        k = f"{t:g}"
        keep_scene = np.isfinite(med) & (med <= t)
        r[f"sc_scenes_{k}"] = int(keep_scene.sum())
        r[f"sc_px_{k}"] = int(n_clear_per_scene[keep_scene].sum())
        # pixel gate: every clear pixel under T, wherever it sits
        r[f"px_px_{k}"] = int((ok & (stq <= t)).sum()) if np.isfinite(t) else int(ok.sum())
    return r


def main():
    paths = sorted(str(p) for p in DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    print(f"{len(paths)} bundles")
    with Pool(64) as pool:
        df = pd.DataFrame([x for x in pool.map(one, paths, chunksize=4) if x])

    sp = pd.read_csv(SPLITS)[["station_id", "kg_macro"]]
    df["station_id"] = df.station_id.astype(str)
    sp["station_id"] = sp.station_id.astype(str)
    df = df.merge(sp, on="station_id", how="left")
    df.to_csv(OUT, index=False)

    tot_sc, tot_px = df.n_scenes.sum(), df.n_clear_px.sum()
    print("\n" + "=" * 92)
    print(f"ST_QA GATE YIELD -- {len(df)} stations, {tot_sc:,} scenes, {tot_px:,} clear pixels")
    print("=" * 92)
    print("\nSCENE-LEVEL gate (drop a scene when its median clear-pixel ST_QA exceeds T)\n")
    print(f"{'T (K)':>7}{'scenes kept':>14}{'%':>8}{'clear px kept':>16}{'%':>8}"
          f"{'stations <20 sc':>17}")
    for t in THRESH:
        k = f"{t:g}"
        s, p = df[f"sc_scenes_{k}"].sum(), df[f"sc_px_{k}"].sum()
        lost = int((df[f"sc_scenes_{k}"] < MIN_SCENES).sum())
        lab = "none" if not np.isfinite(t) else k
        print(f"{lab:>7}{s:>14,}{100*s/tot_sc:>8.1f}{p:>16,}{100*p/tot_px:>8.1f}{lost:>17}")

    print("\nPIXEL-LEVEL gate, for comparison (keep any clear pixel with st_qa <= T)\n")
    print(f"{'T (K)':>7}{'clear px kept':>16}{'%':>8}{'vs scene gate':>16}")
    for t in THRESH:
        k = f"{t:g}"
        p_px, p_sc = df[f"px_px_{k}"].sum(), df[f"sc_px_{k}"].sum()
        lab = "none" if not np.isfinite(t) else k
        d = 100 * (p_px - p_sc) / tot_px
        print(f"{lab:>7}{p_px:>16,}{100*p_px/tot_px:>8.1f}{d:>+15.1f}pp")

    print("\nCLEAR-PIXEL RETENTION BY KOPPEN MACRO-CLASS, scene gate (climate bias)\n")
    g = df.groupby("kg_macro")
    cols = [f"sc_px_{t:g}" for t in (2.0, 3.0, 4.0, 5.0)]
    tab = g[cols].sum().div(g["n_clear_px"].sum(), axis=0)
    tab.insert(0, "n_stations", g.size())
    print(tab.round(4).to_string())
    print(f"\nwrote {OUT}")
    print("=" * 92)


if __name__ == "__main__":
    main()
