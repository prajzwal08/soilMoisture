"""
Download S1 RTC and LULC for all stations via Microsoft Planetary Computer (MPC).

Products per station:
  {station}/S1RTC/YYYYMMDD_ASC.tif      float32, dB [-50,+10], bands: VV VH
  {station}/S1RTC/YYYYMMDD_DESC.tif     float32, dB [-50,+10], bands: VV VH
  {station}/LULC/YYYY.tif               uint8, class 0-9, nearest 10m
  {station}/metadata.json

Sources:
  S1RTC  → Microsoft Planetary Computer (sentinel-1-rtc)
  LULC   → Microsoft Planetary Computer (io-lulc-annual-v02)

Note: S2L2A and DEM are handled by download_s2_dem_cdse.py (CDSE/OpenEO).

Usage:
  python download_s1_lulc_mpc.py
  nohup python download_s1_lulc_mpc.py > /tmp/download_s1_lulc.log 2>&1 &
"""

import csv
import json
import logging
import os
os.environ.pop("PROJ_DATA", None)   # prevent stale conda PROJ_DATA from conflicting with ~/.local pyproj
import rasterio
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  registers .rio accessor
import stackstac
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
STATION_CSV  = DATA_ROOT / "station_splits.csv"
LOG_DIR      = DATA_ROOT / "logs"
LOG_FILE     = LOG_DIR / "download_s1_lulc_log.csv"

# Raw S1 RTC and LULC tiles go to scratch (~3 TB total).
SCRATCH_DIR  = Path("/gpfs/scratch1/shared/pkhanal/satellite")

TEST_MODE    = False
TEST_STATION = "ISMN_TWENTE_Hupsel"

PIXEL_SIZE   = 224      # pixels per side (TerraMind native size, UNetMobV2 compatible)
RES_M        = 10       # metres per pixel

# Parallelism
# N_WORKERS: concurrent stations. 4–6 is a safe default (network + MPC rate limits).
# Modalities within each station always run in parallel (5 threads per station).
N_WORKERS    = 8   # concurrent stations (each uses 2 threads internally for S1+LULC)

GLOBAL_START     = "2016-01-01"
LULC_START_YEAR  = 2017
LULC_END_YEAR    = 2024

MAX_RETRIES  = 3
RETRY_WAITS  = [5, 15, 45]   # seconds between attempts

MPC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_ASSETS = ["vv", "vh"]

LOG_COLS = [
    "station_id","network","lat","lon",
    "s1rtc_status","s1rtc_n",
    "lulc_status",
    "error_msg","timestamp",
]

# ============================================================
# UTILITIES
# ============================================================

def parse_date(val) -> str:
    """
    Return ISO date string (YYYY-MM-DD) from either:
      - ISO string already: '2021-04-29' or '2021-04-29T...'
      - YYYYMMDD integer or float: 20210429 / 20210429.0
    """
    s = str(val).split(".")[0].strip()  # drop decimal part
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _write_datetime_tag(path: Path, iso_dt: str) -> None:
    """Write acquisition UTC datetime into GeoTIFF TIFFTAG_DATETIME tag."""
    dt_tag = iso_dt[:19].replace("T", " ").replace("-", ":")
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(TIFFTAG_DATETIME=dt_tag, datetime_utc=iso_dt)


def _station_folder(row) -> str:
    if row.source_network != row.network:
        return f"{row.source_network}_{row.network}_{row.station_id}"
    return f"{row.network}_{row.station_id}"


def _station_dir(row) -> Path:
    has_sm = str(getattr(row, "has_soil_moisture", "False")).lower() == "true"
    has_fl = str(getattr(row, "has_flux", "False")).lower() == "true"
    cat = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
    return DATA_ROOT / cat / _station_folder(row)


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "download_s1_lulc.log"),
        ],
    )


def get_utm_epsg(lat: float, lon: float) -> int:
    """Return EPSG code for the UTM zone covering (lat, lon)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def station_grid(lat: float, lon: float):
    """
    Return (epsg, bounds, bbox_latlon) for the station patch.
      bounds      – (xmin, ymin, xmax, ymax) in UTM metres
      bbox_latlon – [lon_min, lat_min, lon_max, lat_max] for STAC search
    """
    epsg = get_utm_epsg(lat, lon)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    half = (PIXEL_SIZE * RES_M) / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)
    corners = [inv.transform(x, y) for x, y in
               [(bounds[0], bounds[1]), (bounds[2], bounds[3])]]
    bbox = [corners[0][0], corners[0][1], corners[1][0], corners[1][1]]
    return epsg, bounds, bbox


def linear_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert linear SAR backscatter (σ₀) → dB, clip to [-50, +10]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(arr > 0, 10.0 * np.log10(arr), np.nan)
    return np.clip(db, -50.0, 10.0).astype(np.float32)


def orbit_suffix(item) -> str:
    """Return 'ASC' or 'DESC' from STAC item orbit state."""
    state = str(item.properties.get("sat:orbit_state", "")).lower()
    return "ASC" if "asc" in state else "DESC"


_NO_RETRY_PHRASES = (
    "does not exist",      # S3 key missing — retrying won't help
    "Access Denied",       # requester-pays / auth — retrying won't help
    "403",                 # HTTP Forbidden
    "NoSuchKey",           # S3 explicit error code
    "out of bounds",       # empty stackstac array — no data at this location
)

def with_retry(func, *args, **kwargs):
    """Call func(*args, **kwargs) up to MAX_RETRIES times with backoff.
    Errors matching _NO_RETRY_PHRASES are raised immediately without retrying."""
    last_exc = None
    for attempt, wait in enumerate(RETRY_WAITS + [None], start=1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(p in msg for p in _NO_RETRY_PHRASES):
                raise  # non-recoverable — don't waste time retrying
            if attempt > MAX_RETRIES:
                break
            logging.warning(f"  Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)
    raise last_exc


def center_crop(da: xr.DataArray, size: int = PIXEL_SIZE) -> xr.DataArray:
    """
    Crop the last two (y, x) dims to exactly `size × size`.
    stackstac may snap bounds outward by ≤1 pixel; this corrects that.
    """
    h, w = da.shape[-2], da.shape[-1]
    if h == size and w == size:
        return da
    if h < size or w < size:
        raise ValueError(f"Array too small to crop: {h}×{w} < {size}×{size}")
    sh = (h - size) // 2
    sw = (w - size) // 2
    return da[..., sh : sh + size, sw : sw + size]


def save_geotiff(da: xr.DataArray, path: Path, dtype: str, epsg: int):
    """
    Write a (band, y, x) DataArray to a compressed GeoTIFF.
    Sets the CRS from epsg before writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    da = da.rio.write_crs(f"EPSG:{epsg}")
    da.rio.to_raster(str(path), dtype=dtype, compress="deflate", tiled=True)


def load_s1rtc_done() -> set:
    """Return station_ids where S1RTC is already complete (done or no_data)."""
    if not LOG_FILE.exists():
        return set()
    df = pd.read_csv(LOG_FILE, dtype=str).reindex(columns=LOG_COLS, fill_value="")
    df = df.drop_duplicates(subset="station_id", keep="last")
    return set(df.loc[df["s1rtc_status"].isin(["done", "no_data"]), "station_id"])


def check_lulc_complete(station_dir: Path, start_year: int, end_year: int) -> bool:
    """Return True if every expected LULC year file exists on disk."""
    if start_year > end_year:
        return True
    return all(
        (station_dir / "LULC" / f"{y}.tif").exists()
        for y in range(start_year, end_year + 1)
    )


def append_checkpoint_row(row: dict, lock: threading.Lock):
    """Append one row to the checkpoint CSV without rewriting the whole file."""
    with lock:
        write_header = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

# ============================================================
# DOWNLOAD: SENTINEL-1 RTC  (MPC)
# ============================================================

def download_s1rtc(
    station_dir: Path, catalog_mpc,
    epsg: int, bounds: tuple, bbox: list,
    start: str, end: str, meta: dict,
) -> int:
    out_dir = station_dir / "S1RTC"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(catalog_mpc.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=f"{start}/{end}",
    ).items())
    # Keep IW mode only (client-side safeguard)
    items = [i for i in items if i.properties.get("s1:instrument_mode", "IW") == "IW"]
    # Keep only dual-pol scenes (both VV and VH present)
    items = [i for i in items if all(a in i.assets for a in S1_ASSETS)]

    if not items:
        return 0

    n = 0
    for item in items:
        date_str = item.datetime.strftime("%Y%m%d")
        suffix   = orbit_suffix(item)
        fpath    = out_dir / f"{date_str}_{suffix}.tif"
        if fpath.exists():
            n += 1
            continue

        def _load(it=item):
            # Re-sign immediately before download so SAS token is fresh
            planetary_computer.sign_inplace(it)
            da = stackstac.stack(
                [it], assets=S1_ASSETS,
                epsg=epsg, resolution=RES_M, bounds=bounds,
                rescale=False,
            ).squeeze("time")
            return da.compute()

        try:
            da = with_retry(_load)
        except IndexError:
            # stackstac returned an empty array — item footprint does not
            # intersect station bounds after reprojection; treat as no data
            logging.debug(f"    S1RTC {fpath.name}: no spatial overlap — skipping")
            continue

        da = da.astype("float32")
        da = center_crop(da)
        da.values = linear_to_db(da.values)
        da = da.assign_coords(band=["VV", "VH"])
        tmp = fpath.with_suffix(".tmp")
        try:
            save_geotiff(da, tmp, "float32", epsg)
            iso_dt = item.datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
            _write_datetime_tag(tmp, iso_dt)
            tmp.rename(fpath)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        finally:
            da.close()

        meta["S1RTC"][f"{date_str}_{suffix}"] = {
            "datetime_utc": iso_dt,
            "platform": item.properties.get("platform", ""),
            "orbit":    item.properties.get("sat:orbit_state", ""),
        }
        n += 1
        logging.debug(f"    S1RTC {fpath.name}")

    return n

# ============================================================
# DOWNLOAD: ESRI LULC  (MPC)
# ============================================================

def download_lulc(
    station_dir: Path, catalog_mpc,
    epsg: int, bounds: tuple, bbox: list,
    start_year: int, end_year: int,
    meta: dict,
) -> bool:
    out_dir = station_dir / "LULC"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0

    for year in range(start_year, end_year + 1):
        fpath = out_dir / f"{year}.tif"
        if fpath.exists():
            n += 1
            continue

        items = list(catalog_mpc.search(
            collections=["io-lulc-annual-v02"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
        ).items())

        if not items:
            logging.debug(f"    LULC {year}: no items found")
            continue

        def _load(its=items):
            da = stackstac.stack(
                its,
                epsg=epsg, resolution=RES_M, bounds=bounds,
                resampling=Resampling.nearest,
                rescale=False,
            )
            return stackstac.mosaic(da).compute()

        try:
            da = with_retry(_load)
        except IndexError:
            logging.debug(f"    LULC {year}: no spatial overlap — skipping")
            continue
        da = da.squeeze(drop=True)
        if da.ndim == 2:
            da = da.expand_dims("band")
        da = center_crop(da)
        da.values = np.clip(da.values, 0, 9).astype(np.uint8)
        tmp = fpath.with_suffix(".tmp")
        try:
            save_geotiff(da, tmp, "uint8", epsg)
            tmp.rename(fpath)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        finally:
            da.close()

        meta["LULC"][str(year)] = {"source": "io-lulc-annual-v02"}
        n += 1
        logging.debug(f"    LULC {year}.tif")

    return n > 0

# ============================================================
# PER-STATION ORCHESTRATION
# ============================================================

def process_station(row: pd.Series, catalog_mpc, skip_s1rtc: bool = False) -> dict:
    station_id  = _station_folder(row)
    lat, lon    = float(row.latitude), float(row.longitude)

    # Clamp time range to [GLOBAL_START, station end_date]
    start = max(GLOBAL_START, parse_date(row.start_date)) if pd.notna(row.start_date) else GLOBAL_START
    end   = parse_date(row.end_date) if pd.notna(row.end_date) else datetime.now().strftime("%Y-%m-%d")
    start_year = max(LULC_START_YEAR, int(start[:4]))
    end_year   = min(LULC_END_YEAR,   int(end[:4]))

    epsg, bounds, bbox = station_grid(lat, lon)
    station_dir = SCRATCH_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "station":      row.station_id,
        "network":      row.network,
        "latitude":     lat,
        "longitude":    lon,
        "epsg":         epsg,
        "patch_size_px": PIXEL_SIZE,
        "pixel_size_m":  RES_M,
        "bounds_utm":    list(bounds),
        "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "S1RTC": {}, "LULC": {},
    }

    log      = dict.fromkeys(LOG_COLS, "")
    log.update(station_id=station_id, network=row.network, lat=lat, lon=lon)
    err_lock = threading.Lock()   # guards log["error_msg"] concatenation

    def _run(label, func, *args):
        key_status = f"{label}_status"
        key_n      = f"{label}_n"
        try:
            result = func(*args)
            if isinstance(result, int):
                log[key_n]      = result
                log[key_status] = "done" if result > 0 else "no_data"
            else:
                log[key_status] = "done" if result else "no_data"
        except Exception as exc:
            log[key_status] = "error"
            with err_lock:
                log["error_msg"] = str(log["error_msg"]) + f"{label.upper()}:{exc}; "
            logging.error(f"  {station_id} {label.upper()} failed: {exc}")

    # S1 RTC + LULC — run sequentially to avoid nested thread pool anti-pattern
    if skip_s1rtc:
        log["s1rtc_status"] = "done"   # completed in a prior run
        log["s1rtc_n"]      = ""
    else:
        _run("s1rtc", download_s1rtc, station_dir, catalog_mpc, epsg, bounds, bbox, start, end, meta)
    _run("lulc", download_lulc, station_dir, catalog_mpc, epsg, bounds, bbox,
         start_year, end_year, meta)

    # Write metadata.json
    try:
        with open(station_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        logging.warning(f"  {station_id}: metadata.json write failed: {exc}")

    log["timestamp"] = datetime.now(timezone.utc).isoformat()
    return log

# ============================================================
# 2025 LULC PROXY
# ============================================================

def apply_lulc_2025_proxy(pilot: pd.DataFrame):
    """Copy LULC/2024.tif → LULC/2025.tif for stations whose data extends into 2025.
    MPC does not yet carry 2025 ESRI Annual LULC; 2024 is used as a proxy."""
    log = logging.getLogger(__name__)
    n_copied = n_already = n_missing_src = 0
    for _, row in pilot.iterrows():
        end = parse_date(row.end_date) if pd.notna(row.end_date) else datetime.now().strftime("%Y-%m-%d")
        if int(end[:4]) < 2025:
            continue
        sid      = _station_folder(row)
        lulc_dir = SCRATCH_DIR / sid / "LULC"
        src      = lulc_dir / "2024.tif"
        dst      = lulc_dir / "2025.tif"
        if dst.exists():
            n_already += 1
            continue
        if not src.exists():
            log.warning(f"  {sid}: LULC/2024.tif missing — cannot create 2025 proxy")
            n_missing_src += 1
            continue
        shutil.copy2(str(src), str(dst))
        with rasterio.open(str(dst), "r+") as ds:
            ds.update_tags(TIFFTAG_DATETIME="2025:01:01 00:00:00", datetime_utc="2025-01-01T00:00:00Z",
                           lulc_proxy="2024")
        n_copied += 1
    log.info(
        f"LULC 2025 proxy: copied={n_copied}  already_present={n_already}  "
        f"missing_2024_src={n_missing_src}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    pilot = pd.read_csv(STATION_CSV).reset_index(drop=True)
    if TEST_MODE:
        pilot = pilot[pilot.apply(lambda r: _station_folder(r) == TEST_STATION, axis=1)].reset_index(drop=True)
        log.info(f"TEST_MODE: running on {len(pilot)} station(s): {list(pilot.apply(_station_folder, axis=1))}")
    log.info(f"Stations: {len(pilot)} (all from station_splits.csv)")

    # S1RTC completion is checkpointed in the log CSV.
    # LULC completion is verified on disk (per-year files) — the log is not
    # trustworthy for LULC because a partial run sets lulc_status="done".
    s1rtc_done = load_s1rtc_done()
    log.info(f"S1RTC already done: {len(s1rtc_done)} stations")

    todo = []
    for i, row in pilot.iterrows():
        sid   = _station_folder(row)
        start = max(GLOBAL_START, parse_date(row.start_date)) if pd.notna(row.start_date) else GLOBAL_START
        end   = parse_date(row.end_date) if pd.notna(row.end_date) else datetime.now().strftime("%Y-%m-%d")
        sy    = max(LULC_START_YEAR, int(start[:4]))
        ey    = min(LULC_END_YEAR,   int(end[:4]))
        if sid in s1rtc_done and check_lulc_complete(SCRATCH_DIR / sid, sy, ey):
            continue
        todo.append((i, row))
    log.info(f"Stations to process: {len(todo)}  (workers={N_WORKERS})")

    ckpt_lock = threading.Lock()
    completed = [0]

    def _process_one(row):
        sid     = _station_folder(row)
        cat_mpc = pystac_client.Client.open(MPC_URL, modifier=planetary_computer.sign_inplace)
        return process_station(row, cat_mpc, skip_s1rtc=(sid in s1rtc_done))

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_row = {pool.submit(_process_one, row): (i, row) for i, row in todo}

        for fut in as_completed(fut_to_row):
            i, row = fut_to_row[fut]
            station_id = _station_folder(row)
            try:
                row_dict = fut.result()
            except Exception as exc:
                log.error(f"  {station_id} unexpected worker error: {exc}")
                continue

            completed[0] += 1
            log.info(
                f"[{completed[0]:3d}/{len(todo)}] Done  {station_id}  "
                f"S1RTC={row_dict.get('s1rtc_n', '?')} LULC={row_dict['lulc_status']}"
            )
            append_checkpoint_row(row_dict, ckpt_lock)

    # Copy 2024 LULC → 2025 for stations with 2025 records (MPC has no 2025 yet)
    apply_lulc_2025_proxy(pilot)

    log.info("Done.")


if __name__ == "__main__":
    main()
