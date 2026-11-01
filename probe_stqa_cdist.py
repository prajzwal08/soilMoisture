#!/usr/bin/env python
"""Does cloud adjacency inflate ST_QA?  Recompute the scene median under CDIST filters.

`median_st_qa` as shipped is taken over pixels the reference decoder calls clear -- so fill,
dilated cloud, cirrus, cloud, shadow and snow are already gone -- but with NO cloud-distance
filter.  QA_PIXEL's dilated-cloud bit reaches only 3 px (90 m), so the cloud-adjacent halo
survives into the median.

That matters because the ST retrieval's atmospheric correction is known to degrade near cloud,
which is why USGS ships ST_CDIST at all.  If adjacency inflates ST_QA, then a CDIST filter
partly SUBSTITUTES for an ST_QA gate and applying both is double-charging.  If it does not,
the two are independent knobs and ST_QA's spread is purely atmospheric.

Three things measured, pooled over every station:
  1. the per-scene median ST_QA recomputed under each CDIST threshold -- the histogram
  2. pixel-level ST_QA as a FUNCTION of distance to cloud, which is the mechanism itself
  3. what the CDIST filter alone costs, in clear pixels and in scenes that lose their median

OUTPUT  csvs/landsat_stqa_cdist.csv, fig/landsat_st30_check/stqa_vs_cdist.png
"""
from __future__ import annotations

import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, SPLITS, qa_decode

warnings.filterwarnings("ignore", category=RuntimeWarning)

FIG = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/landsat_st30_check")
OUT = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_stqa_cdist.csv")

# km.  0.09 = the 3-px dilated-cloud buffer QA_PIXEL already applies, i.e. "no extra reach".
CD = [0.0, 0.09, 0.30, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 2.00]
CD_EDGES = np.array([0, .09, .3, .5, 1, 2, 5, 10, 20, 1e4])   # for the mechanism curve
QA_EDGES = np.arange(0, 12.01, 0.25)
MIN_SCENES = 20


def one(path_str: str):
    z = np.load(path_str, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    lst, stq, cd = z["lst30"], z["st_qa30"], z["cdist30"]
    clear, _ = qa_decode(z["qa_pixel30"].astype("float64"))
    base = clear & np.isfinite(lst) & np.isfinite(stq) & np.isfinite(cd)
    n = lst.shape[0]

    r = {"station_id": meta["station_id"], "n_scenes": int(n),
         "n_clear_px": int(base.sum())}
    meds = {}
    for t in CD:
        m = base & (cd > t) if t > 0 else base
        flat = m.reshape(n, -1)
        cnt = flat.sum(axis=1)
        med = np.full(n, np.nan)
        for i in np.nonzero(cnt > 0)[0]:
            med[i] = np.median(stq[i][m[i]])
        meds[t] = med
        r[f"px_{t:g}"] = int(m.sum())
        r[f"scenes_with_median_{t:g}"] = int(np.isfinite(med).sum())
        r[f"med_{t:g}"] = float(np.nanmedian(med)) if np.isfinite(med).any() else np.nan

    # mechanism: ST_QA binned by distance to cloud, pooled
    h2 = np.histogram2d(np.clip(cd[base], 0, 9e3), np.clip(stq[base], 0, 11.99),
                        bins=[CD_EDGES, QA_EDGES])[0]
    return r, {t: meds[t][np.isfinite(meds[t])] for t in CD}, h2


def main():
    paths = sorted(str(p) for p in DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    print(f"{len(paths)} bundles")
    with Pool(48) as pool:
        out = pool.map(one, paths, chunksize=2)

    df = pd.DataFrame([o[0] for o in out])
    pooled = {t: np.concatenate([o[1][t] for o in out if o[1][t].size]) for t in CD}
    H = np.sum([o[2] for o in out], axis=0)

    sp = pd.read_csv(SPLITS)[["station_id", "kg_macro"]]
    df["station_id"] = df.station_id.astype(str)
    sp["station_id"] = sp.station_id.astype(str)
    df = df.merge(sp, on="station_id", how="left")
    df.to_csv(OUT, index=False)

    tot_px = df.n_clear_px.sum()
    print("\n" + "=" * 90)
    print(f"ST_QA UNDER A CDIST FILTER -- {len(df)} stations, {tot_px:,} clear pixels")
    print("=" * 90)
    print(f"\n{'CDIST >':>9}{'clear px kept':>16}{'%':>8}{'scenes w/ median':>19}"
          f"{'median ST_QA':>15}{'shift':>9}")
    base_med = float(np.median(pooled[0.0]))
    for t in CD:
        k = f"{t:g}"
        px, sc = df[f"px_{k}"].sum(), df[f"scenes_with_median_{k}"].sum()
        m = float(np.median(pooled[t]))
        lab = "none" if t == 0 else f"{t:g} km"
        print(f"{lab:>9}{px:>16,}{100*px/tot_px:>8.1f}{sc:>19,}{m:>15.3f}{m-base_med:>+9.3f}")

    print("\nMECHANISM -- pixel-level ST_QA as a function of distance to cloud\n")
    print(f"{'CDIST bin (km)':>18}{'n pixels':>16}{'%':>7}{'median ST_QA':>15}")
    centres = (QA_EDGES[:-1] + QA_EDGES[1:]) / 2
    tot = H.sum()
    for i in range(len(CD_EDGES) - 1):
        row = H[i]
        if row.sum() == 0:
            continue
        c = np.cumsum(row) / row.sum()
        med = centres[np.searchsorted(c, 0.5)]
        lo, hi = CD_EDGES[i], CD_EDGES[i + 1]
        lab = f"{lo:g}-{hi:g}" if hi < 1e3 else f">{lo:g}"
        print(f"{lab:>18}{int(row.sum()):>16,}{100*row.sum()/tot:>7.1f}{med:>15.3f}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for t in CD:
        lab = "no CDIST filter" if t == 0 else f"CDIST > {t:g} km"
        ax.hist(pooled[t], bins=np.arange(0, 10.05, 0.1), histtype="step", lw=1.5,
                density=True, label=f"{lab}  (med {np.median(pooled[t]):.2f} K)")
    ax.set_xlabel("per-scene median ST_QA (K)"); ax.set_ylabel("density")
    ax.set_title("Scene-median ST_QA under a CDIST filter", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[0, 1]
    for i in range(len(CD_EDGES) - 1):
        row = H[i]
        if row.sum() < 1e4:
            continue
        lo, hi = CD_EDGES[i], CD_EDGES[i + 1]
        lab = f"{lo:g}-{hi:g} km" if hi < 1e3 else f">{lo:g} km"
        ax.plot(centres, row / row.sum(), lw=1.3, label=lab)
    ax.set_xlabel("pixel ST_QA (K)"); ax.set_ylabel("density")
    ax.set_title("THE MECHANISM: pixel ST_QA by distance to cloud", fontsize=10)
    ax.legend(fontsize=7, title="CDIST", title_fontsize=7); ax.grid(alpha=.3); ax.set_xlim(0, 10)

    ax = axes[1, 0]
    meds_by_cd, xs, ns = [], [], []
    for i in range(len(CD_EDGES) - 1):
        row = H[i]
        if row.sum() == 0:
            continue
        c = np.cumsum(row) / row.sum()
        meds_by_cd.append(centres[np.searchsorted(c, 0.5)])
        xs.append(0.5 * (CD_EDGES[i] + min(CD_EDGES[i + 1], 20)))
        ns.append(row.sum())
    ax.semilogx(xs, meds_by_cd, "o-", lw=1.6)
    ax.set_xlabel("distance to nearest cloud (km, bin centre)")
    ax.set_ylabel("median ST_QA (K)")
    ax.set_title("Does ST_QA fall with distance from cloud?", fontsize=10)
    ax.grid(alpha=.3, which="both")

    ax = axes[1, 1]
    px = [100 * df[f"px_{t:g}"].sum() / tot_px for t in CD]
    lost = [int((df[f"scenes_with_median_{t:g}"] < MIN_SCENES).sum()) for t in CD]
    ax.plot(CD, px, "o-", color="tab:blue", label="clear pixels kept (%)")
    ax.set_xlabel("CDIST threshold (km)"); ax.set_ylabel("clear pixels kept (%)", color="tab:blue")
    ax.grid(alpha=.3)
    a2 = ax.twinx()
    a2.plot(CD, lost, "s--", color="tab:red", label="stations < 20 scenes")
    a2.set_ylabel("stations dropped below 20 scenes", color="tab:red")
    ax.set_title("What the CDIST filter alone costs", fontsize=10)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stqa_vs_cdist.png", dpi=120)
    print(f"\nwrote {OUT}\nwrote {FIG/'stqa_vs_cdist.png'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
