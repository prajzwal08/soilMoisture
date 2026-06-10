"""
trim_pre2016.py

No Sentinel-1/2 imagery exists before 2016-01-01 anywhere in the dataset (the
earliest s2/s1_asc/s1_desc date across all 993 stations is 2016-01-01), so
pre-2016 entries in era5/sif/twsa/labels can never produce a training sample
(dataset.py only generates samples for years where era5 AND s2 both have
coverage). This script trims those dead pre-2016 entries out of the production
zarr stores and archives the removed slices (not deleted).

Trims, per station, any of:
  era5/date_ints, era5/values, era5/doys
  sif/date_ints, sif/values, sif/doys
  twsa/date_ints, twsa/lwe, twsa/doys
  labels/dates, labels/sm                  (trim axis=1 for labels/sm)
  labels/dates_flux, labels/le, labels/le_qc

Archived extras: /projects/prjs1968/data/excluded_stations/_pre2016/{station}_pre2016.npz

Usage:
    conda run --no-capture-output -n sensei python trim_pre2016.py [--execute] [--workers N]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ZARR_ROOT   = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS_CSV  = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
ARCHIVE_DIR = Path("/projects/prjs1968/data/excluded_stations/_pre2016")
CUTOFF = 20160101

# (date_key, is_date_str, [(array_path, axis), ...])  -- date_key is always
# included as one of the arrays (axis=0) so it gets trimmed/archived too.
GROUPS = [
    ("era5/date_ints",      False, [("era5/date_ints", 0), ("era5/values", 0), ("era5/doys", 0)]),
    ("sif/date_ints",       False, [("sif/date_ints", 0), ("sif/values", 0), ("sif/doys", 0)]),
    ("twsa/date_ints",      False, [("twsa/date_ints", 0), ("twsa/lwe", 0), ("twsa/doys", 0)]),
    ("labels/dates",        True,  [("labels/dates", 0), ("labels/sm", 1)]),
    ("labels/dates_flux",   True,  [("labels/dates_flux", 0), ("labels/le", 0), ("labels/le_qc", 0)]),
]


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


def process(args):
    station, cat, execute = args
    path = ZARR_ROOT / cat / station
    try:
        store = zarr.DirectoryStore(str(path))
        zg = zarr.open_group(store=store, mode="a" if execute else "r")

        archive = {}
        summary = []

        for date_key, is_date_str, arrays in GROUPS:
            if date_key not in zg:
                continue

            raw_dates = zg[date_key][:]
            if is_date_str:
                date_strs = _decode_dates(raw_dates)
                date_ints = np.array([int(d[:8]) for d in date_strs])
            else:
                date_ints = raw_dates.astype(np.int64)

            keep_mask = date_ints >= CUTOFF
            n_pre = int((~keep_mask).sum())
            if n_pre == 0:
                continue

            n_total = len(date_ints)
            group_name = date_key.split("/")[0] + ("_flux" if "flux" in date_key else "")
            summary.append(f"{group_name} {n_total}->{n_total - n_pre}")

            if not execute:
                continue

            for arr_path, axis in arrays:
                if arr_path not in zg:
                    continue
                arr = zg[arr_path][:]
                dtype = zg[arr_path].dtype
                compressor = zg[arr_path].compressor

                kept = _take(arr, keep_mask, axis)
                removed = _take(arr, ~keep_mask, axis)

                archive_key = arr_path.replace("/", "_")
                archive[archive_key] = removed

                zg.array(arr_path, kept, chunks=kept.shape,
                         dtype=dtype, compressor=compressor, overwrite=True)

        if not summary:
            return f"SKIP   {station}: no pre-2016 entries"

        if not execute:
            return f"DRY    {station}: " + ", ".join(summary)

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARCHIVE_DIR / f"{station}_pre2016.npz", **archive)
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
            print(f"[{i+1:4d}/{len(jobs)}] {result}", flush=True)


if __name__ == "__main__":
    main()
