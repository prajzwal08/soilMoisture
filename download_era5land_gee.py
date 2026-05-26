"""
Download ERA5-Land daily meteorological statistics and GRACE/GRACE-FO TWSA
for all stations using Google Earth Engine.

ERA5-Land (ECMWF/ERA5_LAND/HOURLY):
  Variables (19 total):
    t2m, d2m, skt, u10, v10, sp  →  daily mean, min, max  (18 vars)
    tp                            →  daily sum              ( 1 var)
  Output: {station}/ERA5Land/meteo_{YYYY}.nc  shape (time=365/366,)

TWSA — GRACE/GRACE-FO mascons (NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI):
  Variables (2 total):
    lwe          → liquid water equivalent thickness (cm EWT)
    lwe_uncertainty → 1-sigma uncertainty (cm EWT)
  Coverage: April 2002 – September 2024 (monthly; gap Aug 2017–May 2018)
  Output: {station}/TWSA/twsa_{YYYY}.nc  shape (time=≤12,)

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

DATA_ROOT     = Path("/gpfs/work3/0/prjs1968/data")
STATION_CSV   = DATA_ROOT / "station_splits.csv"
LOG_DIR       = DATA_ROOT / "logs"
LOG_FILE      = LOG_DIR / "era5land_gee_log.csv"
TWSA_LOG_FILE = LOG_DIR / "twsa_gee_log.csv"

TEST_MODE     = False
TEST_STATION  = "ISMN_TWENTE_Hupsel"

N_WORKERS = 6    # concurrent GEE getRegion calls (reduced from 16 to avoid 429 rate limits)

# ── ERA5-Land ─────────────────────────────────────────────────────────────────
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

SHORT_NAMES       = ["t2m", "d2m", "skt", "u10", "v10", "sp", "tp"]
MEAN_MIN_MAX_VARS = ["t2m", "d2m", "skt", "u10", "v10", "sp"]
GEE_SCALE         = 11132   # metres ≈ 0.1°

# ── GRACE/GRACE-FO TWSA ───────────────────────────────────────────────────────
GRACE_COLLECTION  = "NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI"
GRACE_BANDS       = ["lwe_thickness", "uncertainty"]
GRACE_SHORT_NAMES = ["lwe", "lwe_uncertainty"]
GRACE_SCALE       = 55000   # metres ≈ 0.5°
GRACE_START_YEAR  = 2002    # GRACE begins April 2002
GRACE_END_YEAR    = 2024    # GEE collection ends September 2024

# Mid-month DoY for TWSA positional encoding (architecture.md)
GRACE_MID_DOY = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]

LOG_COLS      = ["station_id", "year", "status", "error_msg", "timestamp"]
TWSA_LOG_COLS = ["station_id", "year", "n_months", "status", "error_msg", "timestamp"]


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "era5land_gee_download.log"),
        ],
    )


# ============================================================
# STATION SELECTION
# ============================================================

def load_stations() -> pd.DataFrame:
    """Return all stations from station_splits.csv."""
    df = pd.read_csv(STATION_CSV)
    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"
    df["station_id"] = df.apply(_folder, axis=1)
    def _dir(r):
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux", "False")).lower() == "true"
        cat = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
        return DATA_ROOT / cat / r["station_id"]
    df["station_dir"] = df.apply(_dir, axis=1)
    df = df.reset_index(drop=True)
    logging.getLogger(__name__).info(f"Stations: {len(df)}")
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

        era5_dir = Path(row["station_dir"]) / "ERA5Land"
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
# TWSA — BUILD JOB LIST
# ============================================================

def build_twsa_job_list(df: pd.DataFrame) -> list[dict]:
    """One job = one (station, year). Skip if output twsa_{year}.nc exists."""
    jobs = []
    log  = logging.getLogger(__name__)
    for _, row in df.iterrows():
        station_id = row["station_id"]
        lat  = float(row["latitude"])
        lon  = float(row["longitude"])
        try:
            start_year = max(GRACE_START_YEAR, int(str(row["start_date"])[:4]))
            end_year   = min(GRACE_END_YEAR,   int(str(row["end_date"])[:4]))
        except (ValueError, TypeError):
            log.warning(f"  {station_id}: invalid date range, skipping TWSA")
            continue
        twsa_dir = Path(row["station_dir"]) / "TWSA"
        for year in range(start_year, end_year + 1):
            out = twsa_dir / f"twsa_{year}.nc"
            if not out.exists():
                jobs.append({
                    "station_id":   station_id,
                    "lat":          lat,
                    "lon":          lon,
                    "year":         year,
                    "output_path":  out,
                })
    return jobs


# ============================================================
# TWSA — PROCESS ONE STATION-YEAR
# ============================================================

def process_twsa_station_year(job: dict) -> dict:
    """
    Extract monthly GRACE/GRACE-FO mascon values at station point for one year.
    Saves xr.Dataset with shape (time=N_months,) — N_months ≤ 12 due to gaps.
    """
    station_id = job["station_id"]
    lat, lon   = job["lat"], job["lon"]
    year       = job["year"]
    out        = Path(job["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    log    = logging.getLogger(__name__)
    result = {
        "station_id": station_id,
        "year":       year,
        "n_months":   0,
        "status":     "error",
        "error_msg":  "",
        "timestamp":  "",
    }

    try:
        geometry = ee.Geometry.Point([lon, lat])
        start    = f"{year}-01-01"
        end      = f"{year + 1}-01-01"

        collection = (
            ee.ImageCollection(GRACE_COLLECTION)
            .filterDate(start, end)
            .select(GRACE_BANDS)
        )

        # getRegion returns [[header], [row0], [row1], ...]
        raw = collection.getRegion(geometry, GRACE_SCALE).getInfo()

        if len(raw) <= 1:
            # No GRACE data for this year (gap period or outside coverage)
            result["status"]   = "no_data"
            result["n_months"] = 0
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            return result

        header = raw[0]
        rows   = raw[1:]
        df     = pd.DataFrame(rows, columns=header)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.rename(columns=dict(zip(GRACE_BANDS, GRACE_SHORT_NAMES)))
        df = df[["time"] + GRACE_SHORT_NAMES].copy()
        for col in GRACE_SHORT_NAMES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=GRACE_SHORT_NAMES)
        df = df[df["time"].dt.year == year]

        if df.empty:
            result["status"]    = "no_data"
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            return result

        # Assign mid-month DoY for positional encoding
        df["doy"] = df["time"].dt.month.map(
            lambda m: GRACE_MID_DOY[m - 1]
        ).astype("int32")

        df.index = df["time"].dt.tz_localize(None)
        df = df.drop(columns=["time"])

        ds = xr.Dataset.from_dataframe(df)
        ds["lwe"].attrs["units"]             = "cm"
        ds["lwe_uncertainty"].attrs["units"] = "cm"
        ds["doy"].attrs["long_name"]         = "mid_month_day_of_year"
        ds.attrs["station_id"]  = station_id
        ds.attrs["latitude"]    = lat
        ds.attrs["longitude"]   = lon
        ds.attrs["source"]      = GRACE_COLLECTION
        ds.attrs["created"]     = datetime.now(timezone.utc).isoformat()

        ds.to_netcdf(str(out))
        result["status"]   = "done"
        result["n_months"] = len(df)
        log.debug(f"    TWSA {station_id} {year}: {len(df)} months")

    except Exception as exc:
        result["error_msg"] = str(exc)
        log.error(f"  TWSA {station_id} {year} FAILED: {exc}")
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

def _run_jobs(jobs, process_fn, log_df, lock, label):
    log  = logging.getLogger(__name__)
    done = [0]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_job = {pool.submit(process_fn, j): j for j in jobs}
        for fut in as_completed(fut_to_job):
            j = fut_to_job[fut]
            try:
                row = fut.result()
            except Exception as exc:
                log.error(f"  Worker error {j['station_id']} {j['year']}: {exc}")
                continue
            done[0] += 1
            log.info(
                f"[{label}] [{done[0]:4d}/{len(jobs)}] {row['station_id']} "
                f"{row['year']}  status={row['status']}"
            )
            append_log(log_df, row, lock)


def main():
    setup_logging()
    log = logging.getLogger(__name__)

    ee.Initialize(credentials=_gee_credentials(), project=GEE_PROJECT)

    df   = load_stations()
    if TEST_MODE:
        df = df[df["station_id"] == TEST_STATION].reset_index(drop=True)
        log.info(f"TEST_MODE: running on {len(df)} station(s): {list(df['station_id'])}")
    lock = threading.Lock()

    # ── ERA5-Land ─────────────────────────────────────────────────────────────
    era5_jobs = build_job_list(df)
    log.info(f"ERA5-Land jobs: {len(era5_jobs)} station-years  (workers={N_WORKERS})")
    era5_log = load_log()
    _run_jobs(era5_jobs, process_station_year, era5_log, lock, "ERA5")

    # ── TWSA ──────────────────────────────────────────────────────────────────
    twsa_jobs = build_twsa_job_list(df)
    log.info(f"TWSA jobs: {len(twsa_jobs)} station-years  (workers={N_WORKERS})")
    twsa_log_df = pd.read_csv(TWSA_LOG_FILE, dtype=str) \
        if TWSA_LOG_FILE.exists() else pd.DataFrame(columns=TWSA_LOG_COLS)

    def _append_twsa(log_df, row, lock):
        with lock:
            updated = pd.concat(
                [log_df, pd.DataFrame([row])], ignore_index=True
            )
            updated.to_csv(TWSA_LOG_FILE, index=False)

    twsa_lock = threading.Lock()
    done = [0]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_job = {pool.submit(process_twsa_station_year, j): j for j in twsa_jobs}
        for fut in as_completed(fut_to_job):
            j = fut_to_job[fut]
            try:
                row = fut.result()
            except Exception as exc:
                log.error(f"  TWSA worker error {j['station_id']} {j['year']}: {exc}")
                continue
            done[0] += 1
            log.info(
                f"[TWSA] [{done[0]:4d}/{len(twsa_jobs)}] {row['station_id']} "
                f"{row['year']}  status={row['status']}  months={row.get('n_months',0)}"
            )
            _append_twsa(twsa_log_df, row, twsa_lock)

    log.info("Done.")


if __name__ == "__main__":
    main()
