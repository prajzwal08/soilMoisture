"""
Download TROPOMI SIF (Solar-Induced Fluorescence) for all stations
from the S5P-PAL STAC API (collection: L2B_SIF___).

Approach: day-centric — for each calendar day, fetch the L2B daily file
once and extract observations near ALL stations simultaneously. This is
much more efficient than per-station downloads since each global file
serves all 1,048 stations.

Source: S5P-PAL portal  https://data-portal.s5p-pal.com
Collection: L2B_SIF___  (daily, cloud-free valid retrievals only)
Variables extracted:
  sif              → SIF_743  (mW/m²/sr/nm, 743–758 nm window)
  sif_uncertainty  → SIF_743_uncertainty

Output per station per year:
  {station}/SIF/sif_{YYYY}.nc
  Variables: sif, sif_uncertainty, shape (time=N_valid_days,)

Coverage: April 2018 – present (~1-4 week latency)
Auth: None required — publicly accessible

Usage:
  python download_sif_tropomi.py
  nohup python download_sif_tropomi.py > /tmp/download_sif.log 2>&1 &
"""

import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import pystac_client

# ============================================================
# CONFIGURATION
# ============================================================

STATION_CSV   = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_splits.csv")
SATELLITE_DIR = Path("/home/khanalp/data/satellite")
LOG_FILE      = SATELLITE_DIR / "sif_download_log.csv"

STAC_URL       = "https://data-portal.s5p-pal.com/api/s5p-l2"
COLLECTION     = "L2B_SIF___"
SIF_START_DATE = date(2018, 4, 30)   # first available L2B file

SEARCH_RADIUS_DEG = 0.05   # ~5.5 km buffer around station for STAC bbox
EXTRACT_RADIUS_M  = 5000   # 5 km radius for pixel matching

N_WORKERS = 4   # concurrent day downloads

LOG_COLS = ["date", "n_stations_found", "status", "error_msg", "timestamp"]

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
            logging.FileHandler(SATELLITE_DIR / "sif_download.log"),
        ],
    )

# ============================================================
# STATION LOADING
# ============================================================

def load_stations() -> pd.DataFrame:
    df = pd.read_csv(STATION_CSV)
    df["station_id"] = df["network"] + "_" + df["station_id"]
    # Only keep stations with records overlapping SIF coverage (post-2018)
    df["end_year"] = df["end_date"].astype(str).str[:4].astype(int, errors="ignore")
    df = df[df["end_year"] >= 2018].reset_index(drop=True)
    logging.getLogger(__name__).info(f"Stations with SIF coverage: {len(df)}")
    return df

# ============================================================
# DATE RANGE UTILITIES
# ============================================================

def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_date(val) -> str:
    s = str(val).split(".")[0].strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]

# ============================================================
# STAC SEARCH
# ============================================================

def get_l2b_url(catalog, day: date):
    """Return download URL for L2B file for the given day, or None."""
    day_str  = day.strftime("%Y-%m-%d")
    next_str = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    items = list(catalog.search(
        collections=[COLLECTION],
        datetime=f"{day_str}T00:00:00Z/{next_str}T00:00:00Z",
        max_items=1,
    ).items())
    if not items:
        return None
    assets = items[0].assets
    if "product" in assets:
        return assets["product"].href
    return None

# ============================================================
# EXTRACTION
# ============================================================

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorised haversine distance (km) from (lat1, lon1) to arrays."""
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def extract_stations_from_file(nc_path: Path, stations_df: pd.DataFrame, day: date) -> dict:
    """
    Open one L2B daily file and extract SIF observations within EXTRACT_RADIUS_M
    of each station. Returns dict: station_id → {sif, sif_uncertainty} or None.
    """
    results = {}
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")

        # Variable names in L2B files
        sif_var = None
        unc_var = None
        for candidate in ["SIF_743", "sif_743", "SIF", "sif"]:
            if candidate in ds:
                sif_var = candidate
                break
        for candidate in ["SIF_743_uncertainty", "sif_743_uncertainty", "SIF_uncertainty"]:
            if candidate in ds:
                unc_var = candidate
                break

        if sif_var is None:
            logging.getLogger(__name__).warning(f"  No SIF variable found in {nc_path.name}. Variables: {list(ds.data_vars)}")
            ds.close()
            return results

        # Flatten lat/lon/sif arrays
        lat_arr = ds["latitude"].values.ravel().astype(np.float32)
        lon_arr = ds["longitude"].values.ravel().astype(np.float32)
        sif_arr = ds[sif_var].values.ravel().astype(np.float32)
        unc_arr = ds[unc_var].values.ravel().astype(np.float32) if unc_var else np.full_like(sif_arr, np.nan)
        ds.close()

        # Valid pixels only (filter NaN and fill values)
        valid = np.isfinite(sif_arr) & np.isfinite(lat_arr) & np.isfinite(lon_arr)
        if not valid.any():
            return results

        lat_v = lat_arr[valid]
        lon_v = lon_arr[valid]
        sif_v = sif_arr[valid]
        unc_v = unc_arr[valid]

        radius_km = EXTRACT_RADIUS_M / 1000.0

        for _, row in stations_df.iterrows():
            sid  = row["station_id"]
            slat = float(row["latitude"])
            slon = float(row["longitude"])

            dist = haversine_km(slat, slon, lat_v, lon_v)
            nearby = dist <= radius_km

            if not nearby.any():
                continue

            results[sid] = {
                "sif":             float(np.nanmean(sif_v[nearby])),
                "sif_uncertainty": float(np.nanmean(unc_v[nearby])),
                "date":            day,
            }

    except Exception as exc:
        logging.getLogger(__name__).warning(f"  Failed to extract from {nc_path.name}: {exc}")

    return results

# ============================================================
# PER-DAY PROCESSING
# ============================================================

def process_day(day: date, catalog, stations_df: pd.DataFrame) -> dict:
    """Download one L2B file and extract SIF for all stations."""
    log = logging.getLogger(__name__)
    result = {
        "date":             day.isoformat(),
        "n_stations_found": 0,
        "status":           "no_data",
        "error_msg":        "",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    url = get_l2b_url(catalog, day)
    if url is None:
        return result

    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Stream download to temp file
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

        # Extract all stations
        obs = extract_stations_from_file(tmp_path, stations_df, day)
        result["n_stations_found"] = len(obs)
        result["status"] = "done"

        # Write each observation to station's buffer (in-memory accumulation)
        # Caller collects these into per-station lists
        result["observations"] = obs

    except Exception as exc:
        result["error_msg"] = str(exc)
        result["status"]    = "error"
        log.error(f"  Day {day}: {exc}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return result

# ============================================================
# SAVE PER-STATION ANNUAL FILE
# ============================================================

def save_station_year(station_id: str, year: int, obs_list: list):
    """
    obs_list: list of dicts with keys date, sif, sif_uncertainty
    Save to {station}/SIF/sif_{year}.nc
    """
    if not obs_list:
        return
    out_dir = SATELLITE_DIR / station_id / "SIF"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sif_{year}.nc"
    if out_path.exists():
        return

    times = pd.DatetimeIndex([o["date"] for o in obs_list])
    sif   = np.array([o["sif"] for o in obs_list], dtype=np.float32)
    unc   = np.array([o["sif_uncertainty"] for o in obs_list], dtype=np.float32)

    ds = xr.Dataset(
        {
            "sif":             ("time", sif,  {"units": "mW m-2 sr-1 nm-1", "long_name": "SIF_743"}),
            "sif_uncertainty": ("time", unc,  {"units": "mW m-2 sr-1 nm-1", "long_name": "SIF_743_uncertainty"}),
        },
        coords={"time": times},
    )
    ds.attrs["station_id"] = station_id
    ds.attrs["source"]     = f"{STAC_URL} / {COLLECTION}"
    ds.attrs["created"]    = datetime.now(timezone.utc).isoformat()
    ds.to_netcdf(str(out_path))

# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    df = load_stations()

    # Build the overall date range (SIF start → max station end date)
    max_end = date.today() - timedelta(days=30)  # ~1-month latency
    start_d = SIF_START_DATE
    end_d   = max_end

    # Per-station, per-year observation accumulator
    # station_id → year → list of obs dicts
    station_obs: dict[str, dict[int, list]] = {
        sid: {} for sid in df["station_id"]
    }

    # Filter to stations still needing data
    def needs_year(sid, year):
        return not (SATELLITE_DIR / sid / "SIF" / f"sif_{year}.nc").exists()

    # STAC catalog (one client — not thread-safe, so each worker opens its own)
    log.info(f"Processing SIF from {start_d} to {end_d}")

    all_days = list(date_range(start_d, end_d))
    log.info(f"Total days: {len(all_days)}")

    done = [0]

    def _process_one_day(day):
        catalog = pystac_client.Client.open(STAC_URL)
        # Only process if at least one station needs this year
        year = day.year
        active = df[df["station_id"].apply(lambda s: needs_year(s, year))]
        if active.empty:
            return {"date": day.isoformat(), "status": "skip",
                    "n_stations_found": 0, "error_msg": "", "timestamp": ""}
        return process_day(day, catalog, active)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_day = {pool.submit(_process_one_day, d): d for d in all_days}

        for fut in as_completed(fut_to_day):
            day = fut_to_day[fut]
            try:
                result = fut.result()
            except Exception as exc:
                log.error(f"  Worker error {day}: {exc}")
                continue

            done[0] += 1
            if done[0] % 100 == 0:
                log.info(f"  [{done[0]}/{len(all_days)}] processed up to {day}")

            obs = result.get("observations", {})
            for sid, o in obs.items():
                yr = o["date"].year
                if yr not in station_obs[sid]:
                    station_obs[sid][yr] = []
                station_obs[sid][yr].append(o)

    # Write per-station per-year files
    log.info("Writing per-station annual SIF files...")
    for sid, years in station_obs.items():
        for year, obs_list in years.items():
            if obs_list:
                save_station_year(sid, year, sorted(obs_list, key=lambda x: x["date"]))

    log.info("Done.")


if __name__ == "__main__":
    main()
