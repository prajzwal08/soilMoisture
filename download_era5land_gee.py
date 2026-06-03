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

import csv
import logging
import time
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

N_WORKERS       = 6    # concurrent GEE getRegion calls (reduced from 16 to avoid 429 rate limits)
GEE_RETRY_WAITS = [2, 4, 8]   # seconds between GEE retry attempts (exponential backoff)

# Errors that will never succeed on retry — raise immediately
_GEE_NO_RETRY = (
    "403",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",   # quota exceeded — retrying now won't help
    "Invalid credentials",
    "not found",
)

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

# ── ERA5 reanalysis (fallback for ocean-masked ERA5-Land pixels) ──────────────
# Strategy order for coastal/island stations:
#   1. ERA5-Land point          — default, 0.1° land-only
#   2. ERA5-Land buffered       — 25 km buffer, mean of nearby valid land pixels
#   3. ERA5 reanalysis (0.25°)  — global coverage, no land mask
#   4. Exclude                  — all strategies failed; station flagged for removal
ERA5_REANALYSIS_COLLECTION = "ECMWF/ERA5/HOURLY"
ERA5_REANALYSIS_TP_BAND    = "total_precipitation"   # ERA5 uses different band name
ERA5_BUFFER_M              = 25000   # 25 km buffer radius for strategy 2

STRATEGY_POINT     = "era5land_point"
STRATEGY_BUFFER    = "era5land_buffer_25km"
STRATEGY_ERA5      = "era5_reanalysis_0.25deg"
STRATEGY_EXCLUDE   = "exclude"

# ── GRACE/GRACE-FO TWSA ───────────────────────────────────────────────────────
GRACE_COLLECTION  = "NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI"
GRACE_BANDS       = ["lwe_thickness", "uncertainty"]
GRACE_SHORT_NAMES = ["lwe", "lwe_uncertainty"]
GRACE_SCALE       = 55000   # metres ≈ 0.5°
GRACE_START_YEAR  = 2002    # GRACE begins April 2002
GRACE_END_YEAR    = 2024    # GEE collection ends September 2024

# Mid-month DoY for TWSA positional encoding (architecture.md)
GRACE_MID_DOY = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]

LOG_COLS      = ["station_id", "year", "status", "strategy", "error_msg", "timestamp"]
TWSA_LOG_COLS = ["station_id", "year", "n_months", "status", "error_msg", "timestamp"]


# ============================================================
# VALIDATION
# ============================================================

def _validate_coords(lat: float, lon: float, station_id: str) -> None:
    import math
    if math.isnan(lat) or math.isnan(lon):
        raise ValueError(f"{station_id}: NaN coordinates (lat={lat}, lon={lon})")
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"{station_id}: out-of-bounds coordinates (lat={lat}, lon={lon})")


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
                    "strategy": None,   # resolved lazily in process_station_year
                })
    return jobs


# ============================================================
# GEE FETCH
# ============================================================

def _getregion_to_df(raw: list, band_names: list, short_names: list) -> pd.DataFrame:
    """Parse GEE getRegion response into a DataFrame with UTC time index."""
    if len(raw) <= 1:
        return pd.DataFrame(columns=["time"] + short_names)
    header = raw[0]
    df = pd.DataFrame(raw[1:], columns=header)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.rename(columns=dict(zip(band_names, short_names)))
    df = df[["time"] + short_names].copy()
    for col in short_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _all_nan(df: pd.DataFrame) -> bool:
    """True if every non-time column is entirely NaN."""
    cols = [c for c in df.columns if c != "time"]
    return df[cols].isna().all().all() if cols and len(df) > 0 else True


def _gee_getregion_with_retry(collection, geometry, scale: int) -> list:
    """Call getRegion with exponential-backoff retry."""
    last_exc = None
    for attempt, wait in enumerate(GEE_RETRY_WAITS + [None]):
        try:
            return collection.getRegion(geometry, scale).getInfo()
        except Exception as exc:
            last_exc = exc
            if any(p in str(exc) for p in _GEE_NO_RETRY):
                raise
            if wait is None:
                raise last_exc
            logging.getLogger(__name__).warning(
                f"  GEE retry {attempt + 1} ({exc}), waiting {wait}s…"
            )
            time.sleep(wait)


def detect_strategy(lat: float, lon: float) -> str:
    """
    Probe one ERA5-Land image to decide which fetch strategy to use for this station.

    Strategy order:
      1. era5land_point       — default; point query on ERA5-Land 0.1°
      2. era5land_buffer_25km — buffer 25 km, mean of nearby valid land pixels
      3. era5_reanalysis      — full ERA5 0.25°, no land mask
      4. exclude              — all strategies returned NaN; station must be excluded
    """
    log = logging.getLogger(__name__)

    # Use one month of data as probe
    col_land = (ee.ImageCollection(GEE_COLLECTION)
                .filterDate("2020-06-01", "2020-07-01")
                .select(GEE_BANDS))

    # Strategy 1: point
    raw = _gee_getregion_with_retry(col_land, ee.Geometry.Point([lon, lat]), GEE_SCALE)
    df  = _getregion_to_df(raw, GEE_BANDS, SHORT_NAMES)
    if not _all_nan(df):
        log.info(f"  Strategy for ({lat:.4f},{lon:.4f}): {STRATEGY_POINT}")
        return STRATEGY_POINT

    # Strategy 2: buffer
    geom_buf = ee.Geometry.Point([lon, lat]).buffer(ERA5_BUFFER_M)
    raw = _gee_getregion_with_retry(col_land, geom_buf, GEE_SCALE)
    df  = _getregion_to_df(raw, GEE_BANDS, SHORT_NAMES)
    if not _all_nan(df):
        log.info(f"  Strategy for ({lat:.4f},{lon:.4f}): {STRATEGY_BUFFER}")
        return STRATEGY_BUFFER

    # Strategy 3: ERA5 reanalysis (different band name for tp)
    bands_era5 = [b if b != "total_precipitation_hourly" else ERA5_REANALYSIS_TP_BAND
                  for b in GEE_BANDS]
    col_era5 = (ee.ImageCollection(ERA5_REANALYSIS_COLLECTION)
                .filterDate("2020-06-01", "2020-07-01")
                .select(bands_era5))
    raw = _gee_getregion_with_retry(col_era5, ee.Geometry.Point([lon, lat]), 27830)
    df  = _getregion_to_df(raw, bands_era5, SHORT_NAMES)
    if not _all_nan(df):
        log.info(f"  Strategy for ({lat:.4f},{lon:.4f}): {STRATEGY_ERA5}")
        return STRATEGY_ERA5

    log.warning(f"  Strategy for ({lat:.4f},{lon:.4f}): {STRATEGY_EXCLUDE} — all strategies returned NaN")
    return STRATEGY_EXCLUDE


def _fetch_month_gee(lat: float, lon: float, year: int, month: int,
                     strategy: str = STRATEGY_POINT) -> pd.DataFrame:
    """
    Fetch one month of hourly ERA5 data at (lat, lon) using the given strategy.

    Strategies:
      STRATEGY_POINT    — ERA5-Land, point query (default)
      STRATEGY_BUFFER   — ERA5-Land, 25 km buffer + mean of valid pixels
      STRATEGY_ERA5     — ERA5 reanalysis 0.25°, point query
      STRATEGY_EXCLUDE  — returns empty DataFrame (station should be excluded)
    """
    if strategy == STRATEGY_EXCLUDE:
        return pd.DataFrame(columns=["time"] + SHORT_NAMES)

    start = f"{year}-{month:02d}-01"
    end   = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"

    if strategy == STRATEGY_ERA5:
        bands = [b if b != "total_precipitation_hourly" else ERA5_REANALYSIS_TP_BAND
                 for b in GEE_BANDS]
        collection = (ee.ImageCollection(ERA5_REANALYSIS_COLLECTION)
                      .filterDate(start, end)
                      .select(bands))
        geometry = ee.Geometry.Point([lon, lat])
        scale    = 27830   # ≈ 0.25°
    else:
        bands      = GEE_BANDS
        collection = (ee.ImageCollection(GEE_COLLECTION)
                      .filterDate(start, end)
                      .select(bands))
        if strategy == STRATEGY_BUFFER:
            geometry = ee.Geometry.Point([lon, lat]).buffer(ERA5_BUFFER_M)
        else:
            geometry = ee.Geometry.Point([lon, lat])
        scale = GEE_SCALE

    raw = _gee_getregion_with_retry(collection, geometry, scale)
    return _getregion_to_df(raw, bands, SHORT_NAMES)


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
        "strategy":   STRATEGY_POINT,
        "timestamp":  "",
    }

    try:
        _validate_coords(lat, lon, station_id)

        # Detect fetch strategy once per station-year (uses first year only to avoid
        # repeated probing; strategy is stable across years for a given location)
        strategy = job.get("strategy")
        if strategy is None:
            strategy = detect_strategy(lat, lon)
        result["strategy"] = strategy

        if strategy == STRATEGY_EXCLUDE:
            result["status"]    = "exclude"
            result["error_msg"] = "All ERA5 strategies returned NaN — station at ocean pixel"
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            return result

        monthly = []
        for month in range(1, 13):
            df_m = _fetch_month_gee(lat, lon, year, month, strategy=strategy)
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

        # Variable metadata (units + long_name per CF conventions)
        var_meta = {
            "t2m": ("K",        "2 metre temperature"),
            "d2m": ("K",        "2 metre dewpoint temperature"),
            "skt": ("K",        "skin temperature"),
            "u10": ("m s**-1",  "10 metre U wind component"),
            "v10": ("m s**-1",  "10 metre V wind component"),
            "sp":  ("Pa",       "surface pressure"),
        }
        stat_labels = {"mean": "daily mean", "min": "daily minimum", "max": "daily maximum"}
        for short, (unit, desc) in var_meta.items():
            for suf, stat in stat_labels.items():
                v = f"{short}_{suf}"
                if v in ds:
                    ds[v].attrs["units"]     = unit
                    ds[v].attrs["long_name"] = f"{desc} {stat}"
        if "tp_sum" in ds:
            ds["tp_sum"].attrs["units"]     = "m"
            ds["tp_sum"].attrs["long_name"] = "total precipitation daily sum"

        ds.attrs["station_id"] = station_id
        ds.attrs["latitude"]   = lat
        ds.attrs["longitude"]  = lon
        ds.attrs["created"]    = datetime.now(timezone.utc).isoformat()
        ds.attrs["source"]     = GEE_COLLECTION

        tmp = out.with_suffix(".tmp")
        ds.to_netcdf(str(tmp))
        ds.close()
        tmp.rename(out)   # atomic on POSIX — file is complete or absent
        log.debug(f"    Saved {out.name}")
        result["status"] = "done"

    except Exception as exc:
        result["error_msg"] = str(exc)
        log.error(f"  {station_id} {year} FAILED: {exc}")
        for p in (out.with_suffix(".tmp"), out):
            if p.exists():
                p.unlink()

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
        _validate_coords(lat, lon, station_id)
        geometry = ee.Geometry.Point([lon, lat])
        start    = f"{year}-01-01"
        end      = f"{year + 1}-01-01"

        collection = (
            ee.ImageCollection(GRACE_COLLECTION)
            .filterDate(start, end)
            .select(GRACE_BANDS)
        )

        # getRegion returns [[header], [row0], [row1], ...]
        last_exc = None
        for attempt, wait in enumerate(GEE_RETRY_WAITS + [None]):
            try:
                raw = collection.getRegion(geometry, GRACE_SCALE).getInfo()
                break
            except Exception as exc:
                last_exc = exc
                if any(p in str(exc) for p in _GEE_NO_RETRY):
                    raise  # non-recoverable — don't waste retries
                if wait is None:
                    raise last_exc
                logging.getLogger(__name__).warning(
                    f"  GEE TWSA getInfo failed attempt {attempt + 1} ({exc}), retrying in {wait}s…"
                )
                time.sleep(wait)

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
        df.index.name = "time"
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

        tmp = out.with_suffix(".tmp")
        ds.to_netcdf(str(tmp))
        ds.close()
        tmp.rename(out)   # atomic on POSIX — file is complete or absent
        result["status"]   = "done"
        result["n_months"] = len(df)
        log.debug(f"    TWSA {station_id} {year}: {len(df)} months")

    except Exception as exc:
        result["error_msg"] = str(exc)
        log.error(f"  TWSA {station_id} {year} FAILED: {exc}")
        for p in (out.with_suffix(".tmp"), out):
            if p.exists():
                p.unlink()

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


# ============================================================
# CHECKPOINT LOG
# ============================================================

def _append_csv_row(filepath: Path, row: dict, cols: list, lock: threading.Lock):
    """Append one row to a CSV log file in a thread-safe way. Never overwrites existing rows."""
    with lock:
        write_header = not filepath.exists()
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def _run_jobs(jobs, process_fn, log_file, cols, lock, label):
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
            extra = f"  months={row['n_months']}" if "n_months" in row else ""
            log.info(
                f"[{label}] [{done[0]:4d}/{len(jobs)}] {row['station_id']} "
                f"{row['year']}  status={row['status']}{extra}"
            )
            _append_csv_row(log_file, row, cols, lock)


def consolidate_era5(station_dir: Path):
    """
    Merge per-year meteo_{YYYY}.nc files in station_dir/ERA5Land/ into one
    consolidated meteo_{start}_{end}.nc and delete the per-year files.
    Overwrites any existing consolidated file.
    """
    log = logging.getLogger(__name__)
    era5_dir   = station_dir / "ERA5Land"
    per_year   = sorted(era5_dir.glob("meteo_????.nc"))
    if not per_year:
        log.warning(f"  consolidate: no per-year files found in {era5_dir}")
        return

    datasets = [xr.open_dataset(p) for p in per_year]
    ds_out   = xr.concat(datasets, dim="time")
    for ds in datasets:
        ds.close()

    import pandas as _pd
    times = ds_out["time"].values
    start = str(_pd.Timestamp(times[0]).date()).replace("-", "")
    end   = str(_pd.Timestamp(times[-1]).date()).replace("-", "")

    # Remove any old consolidated files
    for old in era5_dir.glob("meteo_*_*.nc"):
        old.unlink()
        log.info(f"  Removed old consolidated file: {old.name}")

    out = era5_dir / f"meteo_{start}_{end}.nc"
    tmp = out.with_suffix(".tmp.nc")
    ds_out.to_netcdf(str(tmp))
    tmp.rename(out)
    log.info(f"  Consolidated {len(per_year)} years → {out.name}")

    # Clean up per-year files
    for p in per_year:
        p.unlink()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download ERA5-Land + TWSA via GEE")
    parser.add_argument(
        "--stations", default=None,
        help="Comma-separated station_ids to process (e.g. ISMN_SCAN_Combate). "
             "Default: all stations in station_splits.csv."
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="After downloading per-year files, merge them into one consolidated "
             "meteo_{start}_{end}.nc per station (overwrites existing consolidated file)."
    )
    parser.add_argument(
        "--era5-only", action="store_true",
        help="Skip TWSA download (ERA5-Land only)."
    )
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    ee.Initialize(credentials=_gee_credentials(), project=GEE_PROJECT)

    df = load_stations()
    if TEST_MODE:
        df = df[df["station_id"] == TEST_STATION].reset_index(drop=True)
        log.info(f"TEST_MODE: running on {len(df)} station(s): {list(df['station_id'])}")
    elif args.stations:
        keep = set(s.strip() for s in args.stations.split(","))
        df   = df[df["station_id"].isin(keep)].reset_index(drop=True)
        log.info(f"Filtered to {len(df)} station(s): {list(df['station_id'])}")
        if df.empty:
            log.error("No matching stations found. Check --stations values.")
            return

    lock = threading.Lock()

    # ── ERA5-Land ─────────────────────────────────────────────────────────────
    era5_jobs = build_job_list(df)
    log.info(f"ERA5-Land jobs: {len(era5_jobs)} station-years  (workers={N_WORKERS})")
    _run_jobs(era5_jobs, process_station_year, LOG_FILE, LOG_COLS, lock, "ERA5")

    # Report any stations that need exclusion (all strategies failed)
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
        to_exclude = (log_df[log_df["status"] == "exclude"]["station_id"]
                      .unique().tolist())
        if to_exclude:
            log.warning(
                f"\n{'='*60}\n"
                f"STATIONS TO EXCLUDE (all ERA5 strategies returned NaN):\n"
                + "\n".join(f"  {s}" for s in to_exclude) +
                f"\nAdd these to csvs/excluded_stations.csv and remove from station_splits.csv\n"
                f"{'='*60}"
            )

    if args.consolidate:
        log.info("Consolidating per-year ERA5 files...")
        for _, row in df.iterrows():
            consolidate_era5(Path(row["station_dir"]))

    # ── TWSA ──────────────────────────────────────────────────────────────────
    if not args.era5_only:
        twsa_jobs = build_twsa_job_list(df)
        log.info(f"TWSA jobs: {len(twsa_jobs)} station-years  (workers={N_WORKERS})")
        twsa_lock = threading.Lock()
        _run_jobs(twsa_jobs, process_twsa_station_year, TWSA_LOG_FILE, TWSA_LOG_COLS, twsa_lock, "TWSA")

    log.info("Done.")


if __name__ == "__main__":
    main()
