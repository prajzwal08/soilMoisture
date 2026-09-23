#!/usr/bin/env python
"""
splice_era5_radiation.py
========================
Join `ssrd_sum` / `strd_sum` onto the existing 19-column ERA5 array and write the
result as a NEW array beside it.  §43.12.

    era5/values     (N,19)  UNTOUCHED  -- t2m,d2m,skt,u10,v10,sp x{mean,min,max} + tp_sum
    era5/values18   (N,18)  NEW        -- skt dropped, ssrd_sum + strd_sum appended
    era5/vars18     (18,)   NEW        -- the names, so nothing depends on column memory

WRITE BESIDE, NEVER OVER.  `/projects/prjs1968/zarr_tokens` is 1.4 TB and is the
ONLY copy of the drivers -- the scratch mirror is purged and the source ERA5Land
NetCDFs for active stations no longer exist (only 35 survive, all under
excluded_stations/).  `era5/values` is (N,19) float32, ~277 KB per station and
under 300 MB across all 993, so there is no reason to destroy it.  Rollback is
then free and no prior run becomes unreproducible.

Modelled on `trim_pre2016.py`, which is this repo's working precedent for
array-level in-place zarr patching: open the group `mode="a"`, rewrite one array
with `zg.array(..., overwrite=True)` preserving the existing dtype and
compressor, then `zarr.consolidate_metadata(store)`.  Do NOT use
`create_token_zarr.py` for this: `convert_station` opens the store `mode="w"`
(:314), which destroys everything not re-written.

COLUMN MAP.  Old indices 6,7,8 are `skt_{mean,min,max}` and are dropped; the
other 16 carry over IN ORDER, then the two new sums are appended:

    values18[:, 0:16] == values[:, [0,1,2,3,4,5, 9,10,11,12,13,14,15,16,17,18]]
    values18[:, 16]   == ssrd_sum
    values18[:, 17]   == strd_sum

GAPS ARE REPORTED, NOT FILLED.  A day present in `era5/date_ints` with no
radiation row becomes NaN, and the datasets z-score without any NaN handling --
the model would train on `nan`.  So by default a station with ANY gap is refused
and reported; `--max-gap-days N` opts into a bounded number after you have seen
the dry-run distribution.

STALENESS HAZARD.  Readers check only that `.complete` EXISTS
(`dataset.py:186-187`, `dataset_unet.py:134-135`) -- never a timestamp or hash.
A half-spliced store reads as valid.  Hence the dry-run default, and
`verify_era5_18.py` afterwards.

Usage
-----
    python splice_era5_radiation.py                    # dry run, all 993
    python splice_era5_radiation.py --execute --workers 64
    sbatch slurm/splice_era5_radiation.sh

Env: `terramind` (needs zarr 2.x).  The download half runs in `soilmoisture`.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

REPO_ROOT   = Path("/gpfs/work3/0/prjs1968/soilMoisture")
DATA_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
ZARR_ROOT   = Path("/projects/prjs1968/zarr_tokens")
STATION_CSV = REPO_ROOT / "csvs" / "station_splits.csv"
REPORT_CSV  = REPO_ROOT / "csvs" / "era5_splice_report.csv"

# The 19 columns as `create_token_zarr.py:47` wrote them.
OLD_VARS = [
    "t2m_mean", "t2m_min", "t2m_max",
    "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max",
    "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max",
    "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]
DROP     = ["skt_mean", "skt_min", "skt_max"]
NEW_VARS = ["ssrd_sum", "strd_sum"]

KEEP_IDX = [i for i, v in enumerate(OLD_VARS) if v not in DROP]
NEW_ORDER = [OLD_VARS[i] for i in KEEP_IDX] + NEW_VARS
assert KEEP_IDX == [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
assert len(NEW_ORDER) == 18


def station_rows() -> pd.DataFrame:
    """Folder name + category dir for the 993 authoritative stations.

    Deliberately duplicated from `download_era5_radiation.py` rather than
    imported: that module pulls in `earthengine-api`, which does not exist in the
    `terramind` env this script runs under.
    """
    df = pd.read_csv(STATION_CSV)

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"

    def _cat(r):
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux", "False")).lower() == "true"
        return "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")

    df["folder"] = df.apply(_folder, axis=1)
    df["cat"]    = df.apply(_cat, axis=1)
    return df[["folder", "cat"]].reset_index(drop=True)


def _read_radiation(era5_dir: Path) -> pd.DataFrame | None:
    """Concatenate rad_{year}.nc into one date-indexed frame of the two sums."""
    files = sorted(era5_dir.glob("rad_????.nc"))
    if not files:
        return None
    frames = []
    for f in files:
        with xr.open_dataset(f) as ds:
            frames.append(ds[NEW_VARS].to_dataframe())
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # date_ints in the zarr are YYYYMMDD integers
    df.index = pd.DatetimeIndex(df.index).strftime("%Y%m%d").astype(np.int64)
    return df


def process(args) -> dict:
    # zarr_root travels in the tuple rather than a module global so the worker
    # behaves identically under fork and spawn.
    folder, cat, execute, max_gap, zarr_root = args
    out = {"folder": folder, "status": "error", "n_days": 0, "n_gap": 0,
           "n_extra": 0, "msg": ""}

    zpath = Path(zarr_root) / cat / folder
    try:
        if not (zpath / ".complete").exists():
            out["status"] = "skip:no-complete"
            return out

        # PRE-FLIGHT WRITABILITY.  zarr_tokens is deliberately chmod'd read-only
        # (dr-xr-x---) as the data-safety lock on the only copy of the drivers.
        # Without this check zarr swallows the PermissionError and then fails
        # READING BACK the array it could not create, surfacing a baffling
        # `KeyError: 'era5/values18/.zarray'` for every station.  Say what is
        # actually wrong instead.
        if execute and not os.access(zpath / "era5", os.W_OK):
            out["status"] = "skip:read-only"
            out["msg"] = (f"{zpath}/era5 is not writable -- the store is chmod'd "
                          f"read-only on purpose. Unlock deliberately, then re-lock.")
            return out

        store = zarr.DirectoryStore(str(zpath))
        zg = zarr.open_group(store=store, mode="a" if execute else "r")

        if "era5/values" not in zg or "era5/date_ints" not in zg:
            out["status"] = "skip:no-era5"
            return out

        values = zg["era5/values"][:]
        dates  = zg["era5/date_ints"][:].astype(np.int64)
        n = values.shape[0]
        out["n_days"] = int(n)

        if values.shape[1] != len(OLD_VARS):
            out["status"] = "skip:unexpected-width"
            out["msg"] = f"era5/values has {values.shape[1]} cols, expected {len(OLD_VARS)}"
            return out

        rad = _read_radiation(DATA_ROOT / cat / folder / "ERA5Land")
        if rad is None:
            out["status"] = "skip:no-rad-files"
            return out

        aligned = rad.reindex(dates)
        n_gap   = int(aligned[NEW_VARS].isna().any(axis=1).sum())
        n_extra = int(len(rad) - len(rad.index.intersection(pd.Index(dates))))
        out["n_gap"], out["n_extra"] = n_gap, n_extra

        if n_gap > max_gap:
            out["status"] = "gap"
            out["msg"] = (f"{n_gap} of {n} days have no radiation "
                          f"(limit {max_gap}); refusing to write NaN")
            return out

        new = np.concatenate(
            [values[:, KEEP_IDX], aligned[NEW_VARS].to_numpy(dtype=np.float32)],
            axis=1,
        ).astype(np.float32)

        if not execute:
            out["status"] = "dry"
            return out

        src = zg["era5/values"]
        zg.array("era5/values18", new, chunks=new.shape,
                 dtype=src.dtype, compressor=src.compressor, overwrite=True)
        zg.array("era5/vars18", np.array(NEW_ORDER, dtype="<U12"), overwrite=True)
        zarr.consolidate_metadata(store)

        out["status"] = "ok"

    except Exception as exc:
        out["msg"] = str(exc)[:300]

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="actually write; without it this is a dry run")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--stations", type=str, default=None,
                    help="comma-separated folder names, for a smoke run")
    ap.add_argument("--max-gap-days", type=int, default=0,
                    help="tolerated days with no radiation (default 0 = refuse any)")
    ap.add_argument("--zarr-root", type=Path, default=ZARR_ROOT)
    args = ap.parse_args()

    rows = station_rows()
    if args.stations:
        wanted = {s.strip() for s in args.stations.split(",") if s.strip()}
        rows = rows[rows["folder"].isin(wanted)]
    print(f"stations: {len(rows)}   zarr_root: {args.zarr_root}   "
          f"mode: {'EXECUTE' if args.execute else 'DRY RUN'}")

    tasks = [(r.folder, r.cat, args.execute, args.max_gap_days, args.zarr_root)
             for r in rows.itertuples()]
    with Pool(args.workers) as pool:
        results = pool.map(process, tasks)

    rep = pd.DataFrame(results)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(REPORT_CSV, index=False)

    print("\nstatus counts:")
    for k, v in rep["status"].value_counts().items():
        print(f"  {k:24s} {v}")

    gaps = rep[rep.n_gap > 0]
    if len(gaps):
        print(f"\nstations with radiation gaps: {len(gaps)}")
        print(f"  gap days: min {gaps.n_gap.min()}  median "
              f"{int(gaps.n_gap.median())}  max {gaps.n_gap.max()}")
        print(gaps.nlargest(10, "n_gap")[["folder", "n_days", "n_gap"]].to_string(index=False))

    bad = rep[rep.status == "error"]
    if len(bad):
        print(f"\nerrors: {len(bad)}")
        print(bad[["folder", "msg"]].head(10).to_string(index=False))

    print(f"\nreport -> {REPORT_CSV}")
    return 0 if len(bad) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
