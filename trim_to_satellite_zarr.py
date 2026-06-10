"""
trim_to_satellite_zarr.py

For 5 legacy AmeriFlux-Canada stations (CA-Cbo, CA-DB2, CA-DBB, CA-DSM, CA-Mer),
the token zarr's s2 group has more dates than the raw `satellite_zarr` store
(tokens were computed earlier from now-deleted scratch TIFs; satellite_zarr is a
smaller, later raw-pixel re-acquisition). Per project decision, trim s2 tokens
(dates, l3, l6, l9, l12) down to the date set present in satellite_zarr/s2, then
trim cm (dates, masks) down to the intersection with the new s2 date set.

s1_asc/s1_desc are already aligned with satellite_zarr everywhere (no trim).

Stations with no s2/raw mismatch are skipped automatically.

Archived extras: /projects/prjs1968/data/excluded_stations/_satzarr_trim/{station}_satzarr_trim.npz

Usage:
    conda run --no-capture-output -n sensei python trim_to_satellite_zarr.py [--execute] [--workers N]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ZARR_ROOT   = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ROOT    = Path("/projects/prjs1968/satellite_zarr")
SPLITS_CSV  = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ARCHIVE_DIR = Path("/projects/prjs1968/data/excluded_stations/_satzarr_trim")

S2_ARRAYS = ["dates", "l3", "l6", "l9", "l12"]
CM_ARRAYS = ["dates", "masks"]


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


def _decode_dates(arr):
    return np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                      for x in arr], dtype="U8")


def _take(arr, mask, axis):
    return np.take(arr, np.nonzero(mask)[0], axis=axis)


def _trim_group(zg, group, keep_mask, arrays, archive, execute):
    n_drop = int((~keep_mask).sum())
    if n_drop == 0:
        return 0
    for name in arrays:
        arr_path = f"{group}/{name}"
        if arr_path not in zg:
            continue
        arr = zg[arr_path][:]
        dtype = zg[arr_path].dtype
        compressor = zg[arr_path].compressor

        kept = _take(arr, keep_mask, 0)
        removed = _take(arr, ~keep_mask, 0)

        archive[arr_path.replace("/", "_")] = removed

        if execute:
            zg.array(arr_path, kept, chunks=kept.shape,
                     dtype=dtype, compressor=compressor, overwrite=True)
    return n_drop


def process(args):
    station, cat, execute = args
    token_path = ZARR_ROOT / cat / station
    raw_path = SAT_ROOT / f"{station}.zarr"

    try:
        if not raw_path.exists():
            return f"SKIP   {station}: no satellite_zarr"

        store = zarr.DirectoryStore(str(token_path))
        zg = zarr.open_group(store=store, mode="a" if execute else "r")
        zg2 = zarr.open_group(str(raw_path), mode="r")

        if "s2" not in zg or "s2" not in zg2 or "dates" not in zg2["s2"].array_keys():
            return f"SKIP   {station}: no s2 in token or raw"

        token_s2_dates = _decode_dates(zg["s2/dates"][:])
        raw_s2_dates = set(_decode_dates(zg2["s2/dates"][:]))

        keep_mask_s2 = np.isin(token_s2_dates, list(raw_s2_dates))
        n_drop_s2 = int((~keep_mask_s2).sum())
        if n_drop_s2 == 0:
            return f"SKIP   {station}: s2 already aligned with satellite_zarr"

        archive = {}
        summary = []

        n_total_s2 = len(token_s2_dates)
        _trim_group(zg, "s2", keep_mask_s2, S2_ARRAYS, archive, execute)
        summary.append(f"s2 {n_total_s2}->{n_total_s2 - n_drop_s2}")

        new_s2_dates = token_s2_dates[keep_mask_s2]

        if "cm" in zg and "dates" in zg["cm"].array_keys():
            cm_dates = _decode_dates(zg["cm/dates"][:])
            keep_mask_cm = np.isin(cm_dates, new_s2_dates)
            n_drop_cm = int((~keep_mask_cm).sum())
            if n_drop_cm > 0:
                n_total_cm = len(cm_dates)
                _trim_group(zg, "cm", keep_mask_cm, CM_ARRAYS, archive, execute)
                summary.append(f"cm {n_total_cm}->{n_total_cm - n_drop_cm}")

        if not execute:
            return f"DRY    {station}: " + ", ".join(summary)

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARCHIVE_DIR / f"{station}_satzarr_trim.npz", **archive)
        zarr.consolidate_metadata(store)

        return f"OK     {station}: " + ", ".join(summary)
    except Exception as exc:
        import traceback
        return f"ERR    {station}: {exc}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(SPLITS_CSV)
    df["cat"] = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)

    jobs = [(r["dir_name"], r["cat"], args.execute) for _, r in df.iterrows()]
    print(f"{len(jobs)} stations")

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process, jobs)):
            if result.startswith(("OK", "DRY", "ERR")):
                print(f"[{i+1:4d}/{len(jobs)}] {result}", flush=True)


if __name__ == "__main__":
    main()
