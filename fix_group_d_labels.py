"""
Fix Group D: write SM labels from level1_organised into existing zarr stores.

Group D = 4 sm_and_flux ICOS stations that have .complete but no labels/sm:
  ICOS_BE-Vie, ICOS_FI-Sii, ICOS_FR-Tou, ICOS_IT-Tor

Does NOT overwrite any existing zarr data — only adds/overwrites labels/*.
Rewrites .zmetadata (consolidated) and touches .complete again at the end.
"""

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from pathlib import Path

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
LEVEL1_DIR = Path("/projects/prjs1968/level1_organised")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

GROUP_D = [
    ("ICOS_BE-Vie", "sm_and_flux"),
    ("ICOS_FI-Sii", "sm_and_flux"),
    ("ICOS_FR-Tou", "sm_and_flux"),
    ("ICOS_IT-Tor", "sm_and_flux"),
]


def write_labels_to_zarr(zarr_dir: Path, dir_name: str, category: str,
                         start_date: str, end_date: str) -> str:
    nc = LEVEL1_DIR / category / f"{dir_name}.nc"
    if not nc.exists():
        return f"MISS  {dir_name}: {nc} not found"

    ds = xr.open_dataset(nc)
    time_coord = "date_time" if "date_time" in ds else "time"
    times_all = pd.DatetimeIndex(ds[time_coord].values)

    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)
    mask    = (times_all >= t_start) & (times_all <= t_end)

    if mask.sum() == 0:
        ds.close()
        return f"EMPTY {dir_name}: no data in [{start_date}, {end_date}]"

    sm     = ds["soil_moisture"].values.astype(np.float32)[:, mask]
    depths = np.array([str(d) for d in ds["depth"].values], dtype="U20")
    qc     = np.zeros_like(sm, dtype=np.uint8)
    if "soil_moisture_qc" in ds:
        qc = ds["soil_moisture_qc"].values.astype(np.uint8)[:, mask]
    elif "quality_flag" in ds:
        qc = ds["quality_flag"].values.astype(np.uint8)[:, mask]
    ds.close()

    times = times_all[mask]
    dates = np.array([t.strftime("%Y%m%d") for t in times], dtype="U8")

    store = zarr.DirectoryStore(str(zarr_dir))
    root  = zarr.open_group(store=store, mode="a")   # append — don't wipe existing data
    root.array("labels/sm",     sm,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/qc",     qc,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/depths", depths, overwrite=True)
    root.array("labels/dates",  dates,  overwrite=True)
    zarr.consolidate_metadata(store)
    return f"OK    {dir_name}: {sm.shape[1]} days, depths={list(depths)}"


def main():
    df = pd.read_csv(SPLITS_CSV)
    df = df.dropna(subset=["source_network", "network", "station_id"])

    for dir_name, category in GROUP_D:
        row = df[df.apply(
            lambda r: (
                (f"ISMN_{r['network']}_{r['station_name']}" if str(r["source_network"]) == "ISMN"
                 else f"{r['source_network']}_{r['station_id']}") == dir_name
            ), axis=1
        )]
        if row.empty:
            print(f"WARN  {dir_name}: not found in station_splits.csv")
            continue

        r = row.iloc[0]
        start_date = str(r["start_date"])
        end_date   = str(r["end_date"])
        zarr_dir   = ZARR_ROOT / category / dir_name

        if not zarr_dir.exists():
            print(f"WARN  {dir_name}: zarr dir missing at {zarr_dir}")
            continue

        result = write_labels_to_zarr(zarr_dir, dir_name, category, start_date, end_date)
        print(result)


if __name__ == "__main__":
    main()
