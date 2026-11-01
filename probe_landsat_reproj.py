#!/usr/bin/env python
"""What does the cross-UTM-zone warp actually cost?

30% of scenes arrive in a neighbouring UTM zone and are warped 30 m -> 30 m onto the station
grid.  plot_landsat_st30_check.py compared the median map built from reprojected scenes against
the one built from in-zone scenes and found r = 0.63..0.94 with a 0.04..2.4 K offset -- but that
comparison is CONFOUNDED: the two populations are different scenes on different dates, and water
has far lower seasonal amplitude than land, so a seasonal imbalance alone produces a difference
map shaped like the land-cover map.  That test cannot distinguish a warp artefact from a season
artefact.

TEST A -- ROUND TRIP.  Confound-free.  Take the in-zone median map, warp it to the neighbouring
zone the reprojected scenes actually came from, warp it back, and compare with the original.
No season, no pass-time, no different scenes.  This DOUBLE-warps, so it is an upper bound on the
single warp production performs.

TEST B -- SAME-DATE PAIRS.  Empirical.  Dates carrying both an in-zone and an out-of-zone scene
(WRS sidelap) image the same ground the same day.  Season is removed; ~99 min of solar time is
not, because adjacent paths are adjacent orbits, so the residual is reported against the
measured day_tst gap rather than pretended away.

OUTPUT  csvs/landsat_reproj_probe.csv
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, qa_decode

OUT = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_reproj_probe.csv")
RES, N = 30.0, 76


def round_trip(a: np.ndarray, epsg_a: int, epsg_b: int, west: float, north: float):
    """a (EPSG_a grid) -> EPSG_b -> back. Nearest, exactly as production warps."""
    ta = from_origin(west, north, RES, RES)
    # a generous destination in B, so nothing falls off the edge
    with rasterio.Env():
        from rasterio.warp import calculate_default_transform
        tb, wb, hb = calculate_default_transform(
            f"EPSG:{epsg_a}", f"EPSG:{epsg_b}", N, N,
            west, north - N * RES, west + N * RES, north, resolution=RES)
        mid = np.full((hb, wb), np.nan)
        reproject(a, mid, src_transform=ta, src_crs=f"EPSG:{epsg_a}",
                  dst_transform=tb, dst_crs=f"EPSG:{epsg_b}",
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
        back = np.full((N, N), np.nan)
        reproject(mid, back, src_transform=tb, src_crs=f"EPSG:{epsg_b}",
                  dst_transform=ta, dst_crs=f"EPSG:{epsg_a}",
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    return back


def _stats(x, y):
    g = np.isfinite(x) & np.isfinite(y)
    if g.sum() < 40:
        return {}
    d = y[g] - x[g]
    a, b = x[g] - x[g].mean(), y[g] - y[g].mean()
    return {"n_px": int(g.sum()), "bias_k": round(float(d.mean()), 4),
            "rmse_k": round(float(np.sqrt((d ** 2).mean())), 4),
            "p95_abs_k": round(float(np.percentile(np.abs(d), 95)), 4),
            "r": round(float(np.corrcoef(a, b)[0, 1]), 5)}


def station(path: Path) -> dict | None:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    epsg_a = int(meta["epsg"])
    west, south, east, north = meta["bounds"]
    lst, repro = z["lst30"], z["reprojected"].astype(bool)
    clear, _ = qa_decode(z["qa_pixel30"].astype("float64"))
    ok = clear & np.isfinite(lst)
    r = {"station": meta["station_id"], "epsg": epsg_a,
         "n": int(lst.shape[0]), "n_repro": int(repro.sum())}
    if not repro.any() or not (~repro).any():
        r["note"] = "all reprojected" if repro.all() else "none reprojected"
        return r

    zones = Counter(int(e) for e, m in zip(z["native_epsg"], repro) if m and int(e) > 0)
    if not zones:
        return r
    epsg_b = zones.most_common(1)[0][0]
    r["epsg_b"] = epsg_b
    r["n_from_b"] = zones[epsg_b]

    # ---- TEST A: round trip
    med_in = np.nanmedian(np.where(ok[~repro], lst[~repro], np.nan), axis=0)
    if np.isfinite(med_in).sum() > 40:
        rt = round_trip(med_in, epsg_a, epsg_b, west, north)
        for k, v in _stats(med_in, rt).items():
            r[f"rt_{k}"] = v

    # ---- TEST B: same-date pairs
    dates = z["dates"]
    tst = z["day_tst"] if "day_tst" in z else np.full(len(dates), np.nan)
    di, dr = {}, {}
    for i, (d, m) in enumerate(zip(dates, repro)):
        (dr if m else di).setdefault(str(d), []).append(i)
    common = sorted(set(di) & set(dr))
    diffs, rs, gaps = [], [], []
    for d in common:
        i, j = di[d][0], dr[d][0]
        g = ok[i] & ok[j]
        if g.sum() < 200:
            continue
        x, y = lst[i][g], lst[j][g]
        diffs.append(float((y - x).mean()))
        if x.std() > 0 and y.std() > 0:
            rs.append(float(np.corrcoef(x, y)[0, 1]))
        gaps.append(abs(float(tst[j] - tst[i])) if np.isfinite(tst[i]) else np.nan)
    r["n_samedate_pairs"] = len(diffs)
    if diffs:
        r["sd_bias_k"] = round(float(np.mean(diffs)), 4)
        r["sd_absbias_k"] = round(float(np.mean(np.abs(diffs))), 4)
        r["sd_r"] = round(float(np.mean(rs)), 5) if rs else None
        r["sd_tst_gap_h"] = round(float(np.nanmean(gaps)), 3)
    return r


def main():
    rows = [x for x in (station(p) for p in
                        sorted(DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))) if x]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("=" * 96)
    print("CROSS-ZONE WARP -- WHAT IT ACTUALLY COSTS")
    print("=" * 96)
    print("\nTEST A -- round trip through the neighbouring zone (confound-free, DOUBLE warp,")
    print("          so an upper bound on the single warp production does):\n")
    print(f"{'station':<22}{'A':>7}{'B':>7}{'r':>9}{'bias K':>9}{'RMSE K':>9}{'p95|d| K':>10}")
    for _, x in df.iterrows():
        if pd.notna(x.get("rt_r")):
            print(f"{x.station:<22}{int(x.epsg):>7}{int(x.epsg_b):>7}{x.rt_r:>9.5f}"
                  f"{x.rt_bias_k:>9.4f}{x.rt_rmse_k:>9.4f}{x.rt_p95_abs_k:>10.4f}")
    print("\nTEST B -- same-date in-zone vs out-of-zone scenes (season removed, ~solar-time gap")
    print("          remains, because adjacent paths are adjacent orbits):\n")
    print(f"{'station':<22}{'pairs':>7}{'mean r':>9}{'bias K':>9}{'|bias| K':>10}{'tst gap h':>11}")
    any_pairs = False
    for _, x in df.iterrows():
        npair = x.get("n_samedate_pairs")
        if pd.isna(npair) or int(npair) == 0:
            continue
        any_pairs = True
        print(f"{x.station:<22}{int(npair):>7}"
              f"{(x.sd_r if pd.notna(x.get('sd_r')) else float('nan')):>9.5f}"
              f"{x.sd_bias_k:>9.4f}{x.sd_absbias_k:>10.4f}{x.sd_tst_gap_h:>11.3f}")
    if not any_pairs:
        print("  (none -- no date carries both an in-zone and an out-of-zone scene at these")
        print("   stations, so Test A is the only evidence.  Adjacent WRS paths are imaged")
        print("   7-9 days apart, and same-day two-path coverage needs the high-latitude")
        print("   orbit convergence.)")
    print(f"\nwrote {OUT}")
    print("=" * 96)


if __name__ == "__main__":
    main()
