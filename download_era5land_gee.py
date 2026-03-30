"""
Download ERA5-Land daily meteorological statistics for ISMN stations
using Google Earth Engine (collection: ECMWF/ERA5_LAND/HOURLY).

Data are fetched at hourly resolution for a single point (station lat/lon)
and aggregated to daily statistics locally.

Variables (19 total):
  t2m, d2m, skt, u10, v10, sp  →  daily mean, min, max  (18 vars)
  tp                            →  daily sum              ( 1 var)

Output per station per year:
  /home/khanalp/data/satellite/{network}_{station}/ERA5Land/meteo_{YYYY}.nc
  Shape: (time=365 or 366,)

Usage:
  python download_era5land_gee.py
  nohup conda run -n geo python download_era5land_gee.py \
      > /tmp/download_era5land_gee.log 2>&1 &

Prerequisites:
  pip install earthengine-api
  earthengine authenticate        # one-time browser login
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ee
import json
import pandas as pd
import xarray as xr
from google.oauth2.credentials import Credentials

GEE_PROJECT    = "1066500857818"
_CREDENTIALS_FILE = Path.home() / ".config/earthengine/credentials"

def _gee_credentials() -> Credentials:
    with open(_CREDENTIALS_FILE) as f:
        d = json.load(f)
    return Credentials(
        token=None,
        refresh_token=d["refresh_token"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )

# ============================================================
# CONFIGURATION
# ============================================================

STATION_CSV   = Path("/home/khanalp/data/soilmoisture/level1/station_metadata.csv")
SATELLITE_DIR = Path("/home/khanalp/data/satellite")
LOG_FILE      = SATELLITE_DIR / "era5land_gee_log.csv"

N_WORKERS = 10   # concurrent GEE getRegion calls

GEE_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"

GEE_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "skin_temperature",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_pressure",
    "total_precipitation_hourly",
]

# Short output names (same order as GEE_BANDS)
SHORT_NAMES = ["t2m", "d2m", "skt", "u10", "v10", "sp", "tp"]

# Variables that get mean+min+max (all except tp)
MEAN_MIN_MAX_VARS = ["t2m", "d2m", "skt", "u10", "v10", "sp"]

GEE_SCALE = 11132   # metres ≈ 0.1°

LOG_COLS = ["station_id", "year", "status", "error_msg", "timestamp"]


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(SATELLITE_DIR / "era5land_gee_download.log"),
        ],
    )


# ============================================================
# STATION SELECTION
# ============================================================

def load_stations() -> pd.DataFrame:
    """
    Return rows from station_metadata.csv whose {network}_{station} folder
    already exists in SATELLITE_DIR.
    """
    df = pd.read_csv(STATION_CSV)
    df = df[df["status"] == "saved"].copy()
    df["station_id"] = df["network"] + "_" + df["station"]

    existing = {p.name for p in SATELLITE_DIR.iterdir() if p.is_dir()}
    df = df[df["station_id"].isin(existing)].reset_index(drop=True)

    logging.getLogger(__name__).info(f"Stations with satellite data: {len(df)}")
    return df


# ============================================================
# JOB LIST
# ============================================================

def build_job_list(df: pd.DataFrame) -> list[dict]:
    """One job = one (station, year). Skip if output meteo_{year}.nc already exists."""
    jobs = []
    log = logging.getLogger(__name__)
    for _, row in df.iterrows():
        station_id = row["station_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        try:
            start_year = int(str(row["start_date"])[:4])
            end_year   = int(str(row["end_date"])[:4])
        except (ValueError, TypeError):
            log.warning(f"  {station_id}: invalid date range, skipping")
            continue

        era5_dir = SATELLITE_DIR / station_id / "ERA5Land"
        for year in range(start_year, end_year + 1):
            out = era5_dir / f"meteo_{year}.nc"
            if not out.exists():
                jobs.append({
                    "station_id": station_id,
                    "lat": lat,
                    "lon": lon,
                    "year": year,
                    "output_path": out,
                })
    return jobs


# ============================================================
# GEE FETCH
# ============================================================

def _fetch_month_gee(lat: float, lon: float, year: int, month: int) -> pd.DataFrame:
    """
    Fetch one month of hourly ERA5-Land data at (lat, lon) via GEE getRegion.

    Returns a DataFrame with columns: [time, t2m, d2m, skt, u10, v10, sp, tp]
    where `time` is a UTC-aware datetime index.
    """
    start = f"{year}-{month:02d}-01"
    # End is first day of next month
    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"

    geometry = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterDate(start, end)
        .select(GEE_BANDS)
    )

    # getRegion returns: [[id, lon, lat, time_ms, band0, band1, ...], ...]
    # First row is the header.
    raw = collection.getRegion(geometry, GEE_SCALE).getInfo()

    if len(raw) <= 1:
        # No data returned (e.g. collection gap); return empty DataFrame
        return pd.DataFrame(columns=["time"] + SHORT_NAMES)

    header = raw[0]  # ['id', 'longitude', 'latitude', 'time', band0, ...]
    rows   = raw[1:]

    df = pd.DataFrame(rows, columns=header)

    # `time` column is milliseconds since epoch (UTC)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.rename(columns=dict(zip(GEE_BANDS, SHORT_NAMES)))
    df = df[["time"] + SHORT_NAMES].copy()

    # Cast band columns to float (GEE may return None for missing pixels)
    for col in SHORT_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# DOWNLOAD + PROCESS ONE STATION-YEAR
# ============================================================

def process_station_year(job: dict) -> dict:
    """
    1. Fetch 12 monthly DataFrames from GEE (hourly data at station point).
    2. Aggregate to daily: mean/min/max for t2m/d2m/skt/u10/v10/sp; sum for tp.
    3. Save as xr.Dataset with shape (time=365/366,).
    """
    station_id = job["station_id"]
    lat, lon   = job["lat"], job["lon"]
    year       = job["year"]
    out        = Path(job["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(__name__)

    result = {
        "station_id": station_id,
        "year":       year,
        "status":     "error",
        "error_msg":  "",
        "timestamp":  "",
    }

    try:
        monthly = []
        for month in range(1, 13):
            df_m = _fetch_month_gee(lat, lon, year, month)
            monthly.append(df_m)

        df = pd.concat(monthly, ignore_index=True)
        df = df.dropna(subset=["time"])
        df = df.set_index("time").sort_index()

        # Daily aggregation
        daily_parts = []

        for var in MEAN_MIN_MAX_VARS:
            grp = df[var].resample("1D")
            daily_parts.append(grp.mean().rename(f"{var}_mean"))
            daily_parts.append(grp.min().rename(f"{var}_min"))
            daily_parts.append(grp.max().rename(f"{var}_max"))

        # tp: sum only
        daily_parts.append(df["tp"].resample("1D").sum().rename("tp_sum"))

        daily = pd.concat(daily_parts, axis=1)

        # Keep only days belonging to the requested year
        daily = daily[daily.index.year == year]

        # Convert to xr.Dataset — index becomes the 'time' dimension
        # Strip timezone (xarray doesn't serialise tz-aware timestamps to NetCDF)
        daily.index = daily.index.tz_localize(None)
        ds = xr.Dataset.from_dataframe(daily)

        # Units metadata
        units = {
            "t2m": "K", "d2m": "K", "skt": "K",
            "u10": "m s**-1", "v10": "m s**-1", "sp": "Pa",
        }
        for short, unit in units.items():
            for suf in ("mean", "min", "max"):
                v = f"{short}_{suf}"
                if v in ds:
                    ds[v].attrs["units"] = unit
        if "tp_sum" in ds:
            ds["tp_sum"].attrs["units"] = "m"

        ds.attrs["station_id"] = station_id
        ds.attrs["latitude"]   = lat
        ds.attrs["longitude"]  = lon
        ds.attrs["created"]    = datetime.now(timezone.utc).isoformat()
        ds.attrs["source"]     = GEE_COLLECTION

        ds.to_netcdf(str(out))
        log.debug(f"    Saved {out.name}")
        result["status"] = "done"

    except Exception as exc:
        result["error_msg"] = str(exc)
        log.error(f"  {station_id} {year} FAILED: {exc}")
        if out.exists():
            out.unlink()

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
# MAIN
# ============================================================

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    ee.Initialize(credentials=_gee_credentials(), project=GEE_PROJECT)

    df     = load_stations()
    jobs   = build_job_list(df)
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

    log.info("Done.")


if __name__ == "__main__":
    main()
