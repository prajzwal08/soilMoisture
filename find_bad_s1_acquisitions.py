"""
Find the exact (station, orbit, date) acquisitions whose S1 TerraMind tokens
are fully NaN, so they can be re-generated at the source by re-running
precompute_terramind.py.

Only checks s1_asc/l3 and s1_desc/l3 (one layer is enough -- L3/L6/L9/L12 go
NaN together for the same acquisition, per the full zarr scan in
logs/scan_nan_23706813.out).

Usage:
    conda run -n terramind python find_bad_s1_acquisitions.py
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

SPLITS_CSV = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv"
ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
CATEGORY_FILTER = ["sm_only"]


def _category(r):
    if r["has_soil_moisture"] and r["has_flux"]:
        return "sm_and_flux"
    if r["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def _check_station(row: dict) -> list:
    cat, dir_name = row["cat"], row["dir_name"]
    path = ZARR_ROOT / cat / dir_name
    if not (path / ".complete").exists():
        return []
    try:
        zg = zarr.open_consolidated(str(path), mode="r")
    except Exception:
        try:
            zg = zarr.open_group(str(path), mode="r")
        except Exception:
            return []

    results = []
    for orbit_key in ("s1_asc", "s1_desc"):
        arr_key = f"{orbit_key}/l3"
        dates_key = f"{orbit_key}/dates"
        if arr_key not in zg or dates_key not in zg:
            continue
        dates = [str(d) for d in zg[dates_key][:]]
        data = np.asarray(zg[arr_key][:])
        # fully-NaN acquisition: every value along (196,768) is NaN
        nan_mask = np.isnan(data).reshape(data.shape[0], -1).all(axis=1)
        bad_idx = np.where(nan_mask)[0]
        for i in bad_idx:
            results.append((dir_name, orbit_key, dates[i]))
    return results


def main():
    df = pd.read_csv(SPLITS_CSV)
    df["cat"]      = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)
    df = df[df["cat"].isin(CATEGORY_FILTER)]

    rows = df[["cat", "dir_name"]].to_dict("records")
    with Pool(32) as pool:
        results = pool.map(_check_station, rows)

    all_bad = [r for station_results in results for r in station_results]

    print(f"=== Bad S1 acquisitions: {len(df)} stations checked ===")
    print(f"Total bad acquisitions: {len(all_bad)}")
    by_station = {}
    for dir_name, orbit_key, date in all_bad:
        by_station.setdefault(dir_name, {}).setdefault(orbit_key, []).append(date)

    for dir_name, orbits in sorted(by_station.items()):
        for orbit_key, dates in orbits.items():
            print(f"  {dir_name} {orbit_key}: {len(dates)} dates -> {sorted(dates)}")


if __name__ == "__main__":
    main()
