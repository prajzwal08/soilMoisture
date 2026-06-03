"""
Download ERA5-Land daily statistics for ISMN stations that have satellite data.

Two datasets are used per station×year, results merged into one meteo_{YYYY}.nc:

  Group A — derived-era5-land-daily-statistics
    Variables : 2m_temperature, 2m_dewpoint_temperature, skin_temperature,
                10m_u_component_of_wind, 10m_v_component_of_wind, surface_pressure
    Statistics: daily_mean, daily_minimum, daily_maximum  (3 CDS requests)

  Group B — reanalysis-era5-land-timeseries  (hourly, processed locally)
    Variables : surface_solar_radiation_downwards, surface_thermal_radiation_downwards,
                total_precipitation
    Values are already de-accumulated — no post-processing needed.
    Aggregate: SSRD/STRD → daily mean/min/max   |   TP → daily sum
    (1 CDS request, then aggregate locally)

Final output per station per year:
  /home/khanalp/data/satellite/{network}_{station}/ERA5Land/meteo_{YYYY}.nc
  Variables: t2m/d2m/skt/u10/v10/sp × {mean,min,max}  +  ssrd/strd × {mean,min,max}  +  tp_sum
  Shape: (time=365or366, latitude=3, longitude=3) at 0.1° resolution

Usage:
  python download_era5land.py
  nohup python download_era5land.py > /tmp/download_era5land.log 2>&1 &
"""

import logging
import os
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

# ============================================================
# CONFIGURATION
# ============================================================

STATION_CSV   = Path("/home/khanalp/data/soilmoisture/level1/station_metadata.csv")
SPLITS_CSV    = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
SATELLITE_DIR = Path("/gpfs/scratch1/shared/pkhanal/era5_staging")
LOG_FILE      = Path("/gpfs/work3/0/prjs1968/soilMoisture/logs/era5land_download.log")

# Concurrent CDS requests (CDS allows ~20 per user; 10 is conservative)
N_WORKERS = 10

DATASET_A = "derived-era5-land-daily-statistics"
VARS_A = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "skin_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
STATS_A = ["daily_mean", "daily_minimum", "daily_maximum"]

DATASET_B = "reanalysis-era5-land-timeseries"
VARS_B = [
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
]

LOG_COLS = ["station_id", "year", "status", "error_msg", "timestamp"]

# Short ERA5 variable names as they appear in downloaded NetCDF files
_VARNAME_A = {
    "2m_temperature":            "t2m",
    "2m_dewpoint_temperature":   "d2m",
    "skin_temperature":          "skt",
    "10m_u_component_of_wind":   "u10",
    "10m_v_component_of_wind":   "v10",
    "surface_pressure":          "sp",
}

# ============================================================
# LOGGING
# ============================================================

def setup_logging(staging_dir: Path = None):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if staging_dir:
        staging_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )


# ============================================================
# STATION SELECTION
# ============================================================

def load_stations() -> pd.DataFrame:
    """
    Return station rows with station_id, latitude, longitude, start_date, end_date.

    Tries STATION_CSV first; falls back to SPLITS_CSV (station_splits.csv) when
    STATION_CSV no longer exists.
    """
    log = logging.getLogger(__name__)
    if STATION_CSV.exists():
        df = pd.read_csv(STATION_CSV)
        df = df[df["status"] == "saved"].copy()
        df["station_id"] = df["network"] + "_" + df["station"]
        existing = {p.name for p in SATELLITE_DIR.iterdir() if p.is_dir()}
        df = df[df["station_id"].isin(existing)].reset_index(drop=True)
        log.info(f"Stations with satellite data: {len(df)}")
    else:
        log.warning(f"{STATION_CSV} not found — loading from {SPLITS_CSV}")
        df = pd.read_csv(SPLITS_CSV)
        df["station_id"] = df["network"] + "_" + df["station_name"]
        df = df[["station_id", "latitude", "longitude", "start_date", "end_date"]].copy()
        log.info(f"Stations from splits CSV: {len(df)}")
    return df


# ============================================================
# JOB LIST
# ============================================================

def build_job_list(df: pd.DataFrame, staging_dir: Path = None) -> list[dict]:
    """
    One job = one (station, year). Both Group A and B are handled inside the job.
    Skip if output meteo_{year}.nc already exists.
    """
    base = staging_dir or SATELLITE_DIR
    jobs = []
    for _, row in df.iterrows():
        station_id = row["station_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        try:
            start_year = int(str(row["start_date"])[:4])
            end_year   = int(str(row["end_date"])[:4])
        except (ValueError, TypeError):
            logging.getLogger(__name__).warning(
                f"  {station_id}: invalid date range, skipping"
            )
            continue

        era5_dir = base / station_id / "ERA5Land"
        era5_dir.mkdir(parents=True, exist_ok=True)

        for year in range(start_year, end_year + 1):
            out = era5_dir / f"meteo_{year}.nc"
            if not out.exists():
                jobs.append({
                    "station_id": station_id,
                    "lat": lat, "lon": lon,
                    "year": year,
                    "output_path": out,
                })

    return jobs


# ============================================================
# CDS HELPERS
# ============================================================

def _area(lat: float, lon: float) -> list:
    """3×3 pixel patch at 0.1° resolution: [N, W, S, E]."""
    return [lat + 0.15, lon - 0.15, lat - 0.15, lon + 0.15]


def _download_to_tmp(client: cdsapi.Client, dataset: str, request: dict) -> str:
    """Download to a temp file; if CDS returns a zip, extract the first .nc inside."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp = f.name
    client.retrieve(dataset, request).download(tmp)
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise RuntimeError(f"zip from CDS contains no .nc file: {zf.namelist()}")
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f2:
                extracted = f2.name
            with zf.open(nc_names[0]) as src, open(extracted, "wb") as dst:
                dst.write(src.read())
        os.unlink(tmp)
        return extracted
    return tmp


def _group_a_all_nan(ds_parts: list) -> bool:
    """Return True if every Group A variable across all stat parts is entirely NaN."""
    import numpy as np
    for ds in ds_parts:
        for v in ds.data_vars:
            if not np.isnan(ds[v].values).all():
                return False
    return True


def _download_group_a_hourly(client: cdsapi.Client, area: list, year: int,
                              tmps: list) -> list:
    """
    Fallback for Group A: download reanalysis-era5-land (hourly) and aggregate
    to daily mean/min/max ourselves.  Returns a list with one xr.Dataset
    (same structure as ds_parts from the primary path).
    """
    log = logging.getLogger(__name__)
    log.warning(f"    Group A fallback: using reanalysis-era5-land hourly for {year}")
    monthly = []
    for month in range(1, 13):
        req = {
            "variable": VARS_A,
            "year":  str(year),
            "month": f"{month:02d}",
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  area,
            "data_format":     "netcdf",
            "download_format": "unarchived",
        }
        tmp = _download_to_tmp(client, "reanalysis-era5-land", req)
        tmps.append(tmp)
        monthly.append(xr.open_dataset(tmp))

    ds_hourly = xr.concat(monthly, dim="time")
    for ds_m in monthly:
        ds_m.close()

    parts = {}
    for long_name, short in _VARNAME_A.items():
        v = short if short in ds_hourly else long_name
        daily = ds_hourly[v].resample(time="1D")
        parts[f"{short}_mean"] = daily.mean()
        parts[f"{short}_min"]  = daily.min()
        parts[f"{short}_max"]  = daily.max()

    ds_hourly.close()
    return [xr.Dataset(parts)]


# ============================================================
# DOWNLOAD + PROCESS ONE STATION-YEAR
# ============================================================

def process_station_year(job: dict) -> dict:
    """
    1. Download Group A (3 CDS requests: mean/min/max).
    2. Download Group B (1 CDS request: hourly SSRD/STRD/TP).
    3. Aggregate Group B to daily mean/min/max (SSRD,STRD) and daily sum (TP).
    4. Merge all into one xr.Dataset and save as meteo_{year}.nc.
    """
    station_id = job["station_id"]
    lat, lon   = job["lat"], job["lon"]
    year       = job["year"]
    out        = Path(job["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    log    = logging.getLogger(__name__)
    client = cdsapi.Client(quiet=True, verify=False)
    area   = _area(lat, lon)
    tmps   = []   # temp files to clean up

    result = {
        "station_id": station_id,
        "year": year,
        "status": "error",
        "error_msg": "",
        "timestamp": "",
    }

    try:
        # ---- Group A: one request per (stat, month) to stay within CDS size limits
        ds_parts = []
        suffix_map = {"daily_mean": "mean", "daily_minimum": "min", "daily_maximum": "max"}
        for stat in STATS_A:
            suffix = suffix_map[stat]
            monthly = []
            for month in range(1, 13):
                req = {
                    "variable": VARS_A,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day":   [f"{d:02d}" for d in range(1, 32)],
                    "daily_statistic": stat,
                    "time_zone": "utc+00:00",
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                }
                tmp = _download_to_tmp(client, DATASET_A, req)
                tmps.append(tmp)
                monthly.append(xr.open_dataset(tmp))

            # Concatenate 12 monthly chunks into one annual dataset
            ds_a = xr.concat(monthly, dim="time")
            for ds_m in monthly:
                ds_m.close()

            # Rename variables: t2m → t2m_mean, etc.
            rename = {short: f"{short}_{suffix}"
                      for short in _VARNAME_A.values() if short in ds_a}
            ds_a = ds_a.rename(rename)

            keep = [v for v in ds_a.data_vars if any(v.endswith(f"_{s}") for s in ("mean", "min", "max"))]
            ds_parts.append(ds_a[keep])

        # If derived-daily-statistics returned all-NaN, fall back to hourly reanalysis
        if _group_a_all_nan(ds_parts):
            for ds in ds_parts:
                try: ds.close()
                except Exception: pass
            ds_parts = _download_group_a_hourly(client, area, year, tmps)

        # ---- Group B: one request per month (hourly SSRD / STRD / TP) --------
        monthly_b = []
        for month in range(1, 13):
            req_b = {
                "variable": VARS_B,
                "year": str(year),
                "month": f"{month:02d}",
                "day":   [f"{d:02d}" for d in range(1, 32)],
                "time":  [f"{h:02d}:00" for h in range(24)],
                "area":  area,
                "data_format":     "netcdf",
                "download_format": "unarchived",
            }
            tmp_b = _download_to_tmp(client, DATASET_B, req_b)
            tmps.append(tmp_b)
            monthly_b.append(xr.open_dataset(tmp_b))

        ds_b = xr.concat(monthly_b, dim="time")
        for ds_m in monthly_b:
            ds_m.close()

        # Identify actual variable names (short names in downloaded file)
        ssrd_v = "ssrd" if "ssrd" in ds_b else "surface_solar_radiation_downwards"
        strd_v = "strd" if "strd" in ds_b else "surface_thermal_radiation_downwards"
        tp_v   = "tp"   if "tp"   in ds_b else "total_precipitation"

        # Daily aggregation (values already de-accumulated)
        ds_b_daily = xr.Dataset({
            "ssrd_mean": ds_b[ssrd_v].resample(time="1D").mean(),
            "ssrd_min":  ds_b[ssrd_v].resample(time="1D").min(),
            "ssrd_max":  ds_b[ssrd_v].resample(time="1D").max(),
            "strd_mean": ds_b[strd_v].resample(time="1D").mean(),
            "strd_min":  ds_b[strd_v].resample(time="1D").min(),
            "strd_max":  ds_b[strd_v].resample(time="1D").max(),
            "tp_sum":    ds_b[tp_v].resample(time="1D").sum(),
        })

        # ---- Merge everything into one dataset ------------------------------
        ds_out = xr.merge([*ds_parts, ds_b_daily])

        # Add variable metadata
        units_a = {"t2m": "K", "d2m": "K", "skt": "K",
                   "u10": "m s**-1", "v10": "m s**-1", "sp": "Pa"}
        for short, unit in units_a.items():
            for suf in ("mean", "min", "max"):
                v = f"{short}_{suf}"
                if v in ds_out:
                    ds_out[v].attrs["units"] = unit

        for v in ("ssrd_mean", "ssrd_min", "ssrd_max",
                  "strd_mean", "strd_min", "strd_max"):
            if v in ds_out:
                ds_out[v].attrs["units"] = "J m**-2"
        if "tp_sum" in ds_out:
            ds_out["tp_sum"].attrs["units"] = "m"

        ds_out.attrs["station_id"] = station_id
        ds_out.attrs["created"]    = datetime.now(timezone.utc).isoformat()

        ds_out.to_netcdf(str(out))
        log.debug(f"    Saved {out.name}")
        result["status"] = "done"

        # Only delete temp files after successful save
        for ds in ds_parts:
            try: ds.close()
            except Exception: pass
        if 'ds_b' in locals():
            try: ds_b.close()
            except Exception: pass
        for tmp in tmps:
            try: os.unlink(tmp)
            except FileNotFoundError: pass

    except Exception as exc:
        result["error_msg"] = str(exc)
        log.error(f"  {station_id} {year} FAILED: {exc}")
        if out.exists():
            out.unlink()
        log.info(f"  Temp files preserved in: {[t for t in tmps]}")

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


# ============================================================
# CHECKPOINT LOG
# ============================================================

def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype=str)
    return pd.DataFrame(columns=LOG_COLS)


def append_log(log_df: pd.DataFrame, row: dict, lock: threading.Lock):
    with lock:
        updated = pd.concat(
            [log_df, pd.DataFrame([row])], ignore_index=True
        )
        updated.to_csv(LOG_FILE, index=False)


# ============================================================
# CONSOLIDATION (for --fix-consolidated-dir)
# ============================================================

def consolidate_station(station_id: str, gpfs_dir: Path, staging_dir: Path = None):
    """
    Concatenate per-year meteo_{YYYY}.nc files from staging_dir into one
    consolidated meteo_{start}_{end}.nc, then overwrite the existing file
    in gpfs_dir/  (which is the path dataset.py reads at training time).
    """
    log = logging.getLogger(__name__)
    era5_scratch = (staging_dir or SATELLITE_DIR) / station_id / "ERA5Land"
    per_year = sorted(era5_scratch.glob("meteo_????.nc"))
    if not per_year:
        log.warning(f"  {station_id}: no per-year files found in {era5_scratch}")
        return

    datasets = [xr.open_dataset(p) for p in per_year]
    ds_concat = xr.concat(datasets, dim="time")
    for ds in datasets:
        ds.close()

    times = ds_concat["time"].values
    start = str(pd.Timestamp(times[0]).date()).replace("-", "")
    end   = str(pd.Timestamp(times[-1]).date()).replace("-", "")

    # Remove any existing consolidated files in the target directory
    for old in sorted(gpfs_dir.glob("meteo_*_*.nc")):
        old.unlink()
        log.info(f"  Removed old consolidated file: {old.name}")

    out = gpfs_dir / f"meteo_{start}_{end}.nc"
    tmp = out.with_suffix(".tmp.nc")
    ds_concat.to_netcdf(str(tmp))
    tmp.rename(out)
    log.info(f"  Consolidated → {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download ERA5-Land for ISMN stations")
    parser.add_argument(
        "--stations", default=None,
        help="Comma-separated station_ids to process (e.g. SCAN_Combate,SNOTEL_PortGraham). "
             "Default: all stations in station_metadata.csv."
    )
    parser.add_argument(
        "--fix-consolidated-dir", default=None,
        help="Root of the gpfs data dir (e.g. /gpfs/work3/0/prjs1968/data). "
             "If set, after downloading per-year files, concatenate them and overwrite "
             "the consolidated meteo_{start}_{end}.nc in the correct sm_only/ subdir."
    )
    parser.add_argument(
        "--staging-dir", default=None,
        help="Directory for staging per-year downloads. Defaults to SATELLITE_DIR "
             "(/home/khanalp/data/satellite). Override when that path is unavailable."
    )
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir) if args.staging_dir else SATELLITE_DIR
    setup_logging(staging_dir)
    log = logging.getLogger(__name__)

    df = load_stations()

    if args.stations:
        keep = set(s.strip() for s in args.stations.split(","))
        df = df[df["station_id"].isin(keep)].reset_index(drop=True)
        log.info(f"Filtered to {len(df)} station(s): {list(df['station_id'])}")
        if df.empty:
            log.error("No matching stations found. Check --stations values.")
            return

    # For --fix-consolidated-dir: force re-download of all years (ignore existing per-year files)
    jobs = build_job_list(df, staging_dir=staging_dir)
    if args.fix_consolidated_dir:
        extra = []
        for _, row in df.iterrows():
            station_id = row["station_id"]
            lat, lon   = float(row["latitude"]), float(row["longitude"])
            start_year = int(str(row["start_date"])[:4])
            end_year   = int(str(row["end_date"])[:4])
            era5_dir   = staging_dir / station_id / "ERA5Land"
            for year in range(start_year, end_year + 1):
                out = era5_dir / f"meteo_{year}.nc"
                already_queued = any(
                    j["station_id"] == station_id and j["year"] == year for j in jobs
                )
                if not already_queued:
                    if out.exists():
                        out.unlink()
                    extra.append({"station_id": station_id, "lat": lat, "lon": lon,
                                  "year": year, "output_path": out})
        jobs.extend(extra)

    log_df = load_log()
    lock   = threading.Lock()
    done   = [0]

    log.info(f"Jobs: {len(jobs)} station-years  (workers={N_WORKERS})")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_job = {pool.submit(process_station_year, j): j for j in jobs}

        for fut in as_completed(fut_to_job):
            j = fut_to_job[fut]
            try:
                row = fut.result()
            except Exception as exc:
                log.error(f"  Worker error {j['station_id']} {j['year']}: {exc}")
                continue

            done[0] += 1
            log.info(
                f"[{done[0]:4d}/{len(jobs)}] {row['station_id']} "
                f"{row['year']}  status={row['status']}"
            )
            append_log(log_df, row, lock)

    # Consolidate per-year files into the gpfs training path if requested
    if args.fix_consolidated_dir:
        gpfs_root = Path(args.fix_consolidated_dir)
        for station_id in df["station_id"]:
            # Find the category directory (sm_only / sm_and_flux)
            for cat in ("sm_only", "sm_and_flux"):
                gpfs_era5 = gpfs_root / cat / f"ISMN_{station_id}" / "ERA5Land"
                if gpfs_era5.exists():
                    consolidate_station(station_id, gpfs_era5, staging_dir=staging_dir)
                    break
            else:
                log.warning(f"  {station_id}: no ERA5Land dir found under {gpfs_root}")

    log.info("Done.")


if __name__ == "__main__":
    main()
