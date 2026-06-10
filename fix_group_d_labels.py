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

    # Normalise to (depth, date_time) — sm_only is (depth, date_time),
    # sm_and_flux is (date_time, depth)
    sm_da = ds["soil_moisture"]
    if sm_da.dims[0] == time_coord:
        sm_da = sm_da.transpose("depth", time_coord)
    sm     = sm_da.values.astype(np.float32)[:, mask]
    depths = np.array([str(d) for d in ds["depth"].values], dtype="U20")
    qc     = np.zeros_like(sm, dtype=np.uint8)
    for qc_var in ("soil_moisture_qc", "quality_flag"):
        if qc_var in ds:
            qc_da = ds[qc_var]
            if qc_da.dims[0] == time_coord:
                qc_da = qc_da.transpose("depth", time_coord)
            qc = qc_da.values.astype(np.uint8)[:, mask]
            break
    ds.close()

    times = times_all[mask]
    dates = np.array([t.strftime("%Y%m%d") for t in times], dtype="U8")

    # Flux labels — sm_and_flux stations have LE_F_MDS in the same level1 file
    le = le_qc = dates_flux = None
    ds2 = xr.open_dataset(nc)
    if "LE_F_MDS" in ds2:
        tc2    = "date_time" if "date_time" in ds2 else "time"
        t2_all = pd.DatetimeIndex(ds2[tc2].values)
        m2     = (t2_all >= pd.Timestamp(start_date)) & (t2_all <= pd.Timestamp(end_date))
        le          = ds2["LE_F_MDS"].values.astype(np.float32)[m2]
        le_qc_raw   = ds2.get("LE_F_MDS_QC", None)
        le_qc       = (le_qc_raw.values.astype(np.uint8)[m2]
                       if le_qc_raw is not None else np.zeros_like(le, dtype=np.uint8))
        dates_flux  = np.array([t.strftime("%Y%m%d") for t in t2_all[m2]], dtype="U8")
    ds2.close()

    store = zarr.DirectoryStore(str(zarr_dir))
    root  = zarr.open_group(store=store, mode="a")   # append — don't wipe existing data
    root.array("labels/sm",     sm,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/qc",     qc,     compressor=COMPRESSOR, overwrite=True)
    root.array("labels/depths", depths, overwrite=True)
    root.array("labels/dates",  dates,  overwrite=True)
    if le is not None:
        root.array("labels/le",         le,         compressor=COMPRESSOR, overwrite=True)
        root.array("labels/le_qc",      le_qc,      compressor=COMPRESSOR, overwrite=True)
        root.array("labels/dates_flux", dates_flux, overwrite=True)
    zarr.consolidate_metadata(store)
    flux_note = f" + flux={len(le)} days" if le is not None else ""
    return f"OK    {dir_name}: sm={sm.shape[1]} days, depths={list(depths)}{flux_note}"


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
