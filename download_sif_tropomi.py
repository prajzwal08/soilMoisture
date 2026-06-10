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
  sif              → SIF_743       (mW/m²/sr/nm, 743–758 nm window)
  sif_uncertainty  → SIF_ERROR_743 (uncertainty)

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
import time
from collections import defaultdict
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

DATA_ROOT     = Path("/gpfs/work3/0/prjs1968/data")
STATION_CSV   = DATA_ROOT / "station_splits.csv"
LOG_DIR       = DATA_ROOT / "logs"

TEST_MODE     = False
TEST_STATION  = "ISMN_TWENTE_Hupsel"
TEST_DAYS     = 10   # limit days processed in test mode (SIF files are ~390 MB each)

STAC_URL       = "https://data-portal.s5p-pal.com/api/s5p-l2"
COLLECTION     = "L2B_SIF___"
SIF_START_DATE = date(2018, 4, 30)   # first available L2B file

SEARCH_RADIUS_DEG = 0.05   # ~5.5 km buffer around station for STAC bbox
EXTRACT_RADIUS_M  = 5000   # 5 km radius for pixel matching

N_WORKERS    = 12   # concurrent day downloads; 16 CPUs allocated so 12 is safe
LATENCY_DAYS = 30   # estimated days of latency before files are available

LOG_COLS = ["date", "n_stations_found", "status", "error_msg", "timestamp"]

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
            logging.FileHandler(LOG_DIR / "sif_download.log"),
        ],
    )

# ============================================================
# STATION LOADING
# ============================================================

REQUIRED_COLS = {"source_network", "network", "station_id", "latitude", "longitude",
                 "end_date", "has_soil_moisture", "has_flux"}

def load_stations() -> pd.DataFrame:
    df = pd.read_csv(STATION_CSV)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"station_splits.csv missing required columns: {sorted(missing)}")
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
    # Only keep stations with records overlapping SIF coverage (post-2018)
    df["end_year"] = pd.to_datetime(df["end_date"].astype(str), format="%Y%m%d", errors="coerce").dt.year
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

    L2B files use group structure: variables are under /PRODUCT group.
    """
    results = {}
    try:
        sif_var = "SIF_743"
        unc_var = "SIF_ERROR_743"

        # L2B files store variables in the PRODUCT group
        with xr.open_dataset(nc_path, engine="netcdf4", group="PRODUCT") as ds:
            if sif_var not in ds:
                logging.getLogger(__name__).warning(
                    f"  No SIF_743 in {nc_path.name}/PRODUCT. Variables: {list(ds.data_vars)}"
                )
                return results

            # Flatten lat/lon/sif arrays
            lat_arr = ds["latitude"].values.ravel().astype(np.float32)
            lon_arr = ds["longitude"].values.ravel().astype(np.float32)
            sif_arr = ds[sif_var].values.ravel().astype(np.float32)
            unc_arr = ds[unc_var].values.ravel().astype(np.float32) if unc_var in ds else np.full_like(sif_arr, np.nan)

        # Valid pixels only (filter NaN and fill values)
        valid = np.isfinite(sif_arr) & np.isfinite(lat_arr) & np.isfinite(lon_arr)
        if not valid.any():
            return results

        lat_v = lat_arr[valid]
        lon_v = lon_arr[valid]
        sif_v = sif_arr[valid]
        unc_v = unc_arr[valid]

        radius_km  = EXTRACT_RADIUS_M / 1000.0
        margin_deg = EXTRACT_RADIUS_M / 111_000.0  # conservative degree buffer (~0.045°)

        for row in stations_df.itertuples(index=False):
            sid  = row.station_id
            slat = float(row.latitude)
            slon = float(row.longitude)

            # Cheap bbox pre-filter before haversine (cuts haversine array size ~1000x)
            bbox = (
                (lat_v >= slat - margin_deg) & (lat_v <= slat + margin_deg) &
                (lon_v >= slon - margin_deg) & (lon_v <= slon + margin_deg)
            )
            if not bbox.any():
                continue

            dist   = haversine_km(slat, slon, lat_v[bbox], lon_v[bbox])
            nearby = dist <= radius_km

            if not nearby.any():
                continue

            sif_bbox = sif_v[bbox]
            unc_bbox = unc_v[bbox]
            results[sid] = {
                "sif":             float(np.nanmean(sif_bbox[nearby])),
                "sif_uncertainty": float(np.nanmean(unc_bbox[nearby])),
                "date":            day,
            }

    except Exception as exc:
        logging.getLogger(__name__).warning(f"  Failed to extract from {nc_path.name}: {exc}")

    return results

# ============================================================
# DOWNLOAD WITH RETRY
# ============================================================

def _download(url: str, dest: Path, max_retries: int = 3, backoff_s: float = 15.0):
    """Stream-download url → dest with exponential backoff on transient errors.

    Fails immediately on 4xx (client errors); retries on 5xx and network errors.
    """
    log = logging.getLogger(__name__)
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            return
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise  # 4xx — client error, retrying won't help
            if attempt == max_retries - 1:
                raise
            wait = backoff_s * (2 ** attempt)
            log.warning(f"  Download attempt {attempt + 1} failed ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = backoff_s * (2 ** attempt)
            log.warning(f"  Download attempt {attempt + 1} failed ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)

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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        _download(url, tmp_path)

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
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return result

# ============================================================
# SAVE PER-STATION ANNUAL FILE
# ============================================================

def save_station_year(station_dir: Path, station_id: str, year: int, obs_list: list):
    """
    obs_list: list of dicts with keys date, sif, sif_uncertainty
    Save to {station_dir}/SIF/sif_{year}.nc
    """
    log = logging.getLogger(__name__)
    if not obs_list:
        return
    out_dir = station_dir / "SIF"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sif_{year}.nc"
    if out_path.exists():
        log.debug(f"  Skipping {out_path} (already exists)")
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
    tmp_out = out_dir / f"sif_{year}.tmp.nc"
    ds.to_netcdf(str(tmp_out))
    ds.close()
    tmp_out.rename(out_path)

# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download TROPOMI SIF")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override DATA_ROOT (default: /gpfs/work3/0/prjs1968/data).")
    parser.add_argument("--stations-file", type=Path, default=None,
                        help="File with one station_id per line to restrict processing.")
    args = parser.parse_args()

    if args.data_root is not None:
        global DATA_ROOT, LOG_DIR
        DATA_ROOT = args.data_root
        LOG_DIR   = DATA_ROOT / "logs"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging()
    log = logging.getLogger(__name__)

    df = load_stations()
    if TEST_MODE:
        df = df[df["station_id"] == TEST_STATION].reset_index(drop=True)
        log.info(f"TEST_MODE: running on {len(df)} station(s), {TEST_DAYS} days only")
    elif args.stations_file:
        keep = {l.strip() for l in args.stations_file.read_text().splitlines() if l.strip()}
        df   = df[df["station_id"].isin(keep)].reset_index(drop=True)
        log.info(f"Filtered to {len(df)} station(s) from {args.stations_file}")

    # Build the overall date range (SIF start → max station end date)
    max_end = date.today() - timedelta(days=LATENCY_DAYS)
    start_d = SIF_START_DATE
    end_d   = max_end

    # Per-station, per-year observation accumulator
    # station_id → year → list of obs dicts
    station_obs: dict[str, dict[int, list]] = {
        sid: {} for sid in df["station_id"]
    }
    station_dirs: dict[str, Path] = {
        row["station_id"]: Path(row["station_dir"]) for _, row in df.iterrows()
    }

    # STAC catalog (one client — not thread-safe, so each worker opens its own)
    log.info(f"Processing SIF from {start_d} to {end_d}")

    all_days = list(date_range(start_d, end_d))
    if TEST_MODE:
        all_days = all_days[:TEST_DAYS]
    log.info(f"Total days: {len(all_days)}")

    # Pre-compute which (station, year) pairs still need data (avoids per-day filesystem hits)
    all_years = {d.year for d in all_days}
    needed: set[tuple[str, int]] = {
        (sid, yr)
        for sid in df["station_id"]
        for yr in all_years
        if not (station_dirs[sid] / "SIF" / f"sif_{yr}.nc").exists()
    }
    log.info(f"Station-year pairs still needing data: {len(needed)}")

    # Pre-filter active stations per year (avoid repeated DataFrame.apply inside each worker)
    year_to_active: dict[int, pd.DataFrame] = {
        yr: df[df["station_id"].apply(lambda s: (s, yr) in needed)].reset_index(drop=True)
        for yr in all_years
    }

    # Track pending days per year so we can flush as soon as a full year is processed
    year_pending: dict[int, set] = defaultdict(set)
    for d in all_days:
        year_pending[d.year].add(d)

    done = [0]

    def _process_one_day(day):
        catalog = pystac_client.Client.open(STAC_URL)
        active = year_to_active.get(day.year, pd.DataFrame())
        if active.empty:
            log.debug(f"  Day {day}: all stations already have sif_{day.year}.nc, skipping")
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
                year_pending[day.year].discard(day)
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

            # Flush completed years to disk immediately to avoid all-or-nothing loss
            year_pending[day.year].discard(day)
            if not year_pending[day.year]:
                yr = day.year
                log.info(f"  Year {yr} complete — flushing to disk...")
                for sid in df["station_id"]:
                    obs_list = station_obs[sid].pop(yr, [])
                    if obs_list:
                        save_station_year(
                            station_dirs[sid], sid, yr,
                            sorted(obs_list, key=lambda x: x["date"]),
                        )
                log.info(f"  Year {yr} flushed.")

    log.info("Done.")


if __name__ == "__main__":
    main()
