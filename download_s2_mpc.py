"""
Download Sentinel-2 L2A patches for all stations via Microsoft Planetary Computer (MPC).

Products per station:
  {station}/S2L2A/YYYYMMDD.tif    int16, DN [0,10000], 12 bands (B01-B12, no B10)
  {station}/metadata.json

Source: MPC  sentinel-2-l2a collection
        Direct COG streaming via stackstac — no batch job queue.

Bands (12):
  B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
  (B10 excluded — not present in L2A)

Patch: 224×224 pixels @ 10 m = 2.24 km × 2.24 km centred on station.
       All bands resampled to 10 m via bilinear interpolation.

Usage:
  python download_s2_mpc.py
  nohup python download_s2_mpc.py > /tmp/download_s2_mpc.log 2>&1 &
"""

import csv
import json
import logging
import os
os.environ.pop("PROJ_DATA", None)
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rioxarray  # noqa: F401  registers .rio accessor
import stackstac
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT    = Path(os.getenv("SOIL_DATA_ROOT",   "/gpfs/work3/0/prjs1968/data"))
STATION_CSV  = DATA_ROOT / "station_splits.csv"
LOG_DIR      = DATA_ROOT / "logs"
LOG_FILE     = LOG_DIR / "download_s2_mpc_log.csv"

SCRATCH_DIR  = Path(os.getenv("SOIL_SCRATCH_DIR", "/gpfs/scratch1/shared/pkhanal/satellite"))

PIXEL_SIZE   = 224
RES_M        = 10
MAX_CLOUD    = 75        # % scene-level cloud cover
GLOBAL_START = "2016-01-01"

N_WORKERS    = 12        # concurrent stations (MPC direct streaming — scales well)
MAX_RETRIES  = 3
RETRY_WAITS  = [5, 15, 45]

MPC_URL  = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]

TEST_MODE    = False
TEST_STATION = "ISMN_TWENTE_Hupsel"

LOG_COLS      = ["station_id","lat","lon","n_scenes","dem_status","status","error_msg","timestamp"]
REQUIRED_COLS = {"latitude","longitude","network","station_id","source_network","start_date","end_date"}

# ============================================================
# HELPERS
# ============================================================

def _station_folder(row) -> str:
    src, net = row.source_network, row.network
    if pd.notna(src) and src != net:
        return f"{src}_{net}_{row.station_id}"
    return f"{net}_{row.station_id}"


def parse_date(val) -> date:
    try:
        return pd.Timestamp(str(val)).date()
    except Exception:
        raise ValueError(f"Cannot parse date: {val!r}")


def get_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def station_grid(lat: float, lon: float):
    epsg = get_utm_epsg(lat, lon)
    fwd  = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv  = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    half   = (PIXEL_SIZE * RES_M) / 2
    bounds = (cx - half, cy - half, cx + half, cy + half)
    corners = [inv.transform(cx - half, cy - half), inv.transform(cx + half, cy - half),
               inv.transform(cx - half, cy + half), inv.transform(cx + half, cy + half)]
    bbox = [min(c[0] for c in corners), min(c[1] for c in corners),
            max(c[0] for c in corners), max(c[1] for c in corners)]
    return epsg, bounds, bbox


def center_crop(da: xr.DataArray, size: int = PIXEL_SIZE) -> xr.DataArray:
    h, w = da.shape[-2], da.shape[-1]
    if h < size or w < size:
        raise ValueError(f"Raster shape {h}×{w} is smaller than crop size {size}")
    sh = (h - size) // 2
    sw = (w - size) // 2
    return da[..., sh : sh + size, sw : sw + size]


_NO_RETRY_HTTP = frozenset({401, 403, 404})

def with_retry(fn, max_retries=MAX_RETRIES, waits=RETRY_WAITS):
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in _NO_RETRY_HTTP:
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt])
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt])


def save_geotiff(da: xr.DataArray, path: Path, epsg: int, iso_dt: str, dtype: str = "int16"):
    path.parent.mkdir(parents=True, exist_ok=True)
    da = da.rio.write_crs(f"EPSG:{epsg}")
    da.rio.to_raster(str(path), dtype=dtype, compress="deflate", tiled=True)
    dt_tag = iso_dt[:19].replace("T", " ").replace("-", ":")
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(TIFFTAG_DATETIME=dt_tag, datetime_utc=iso_dt)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype=str)
    return pd.DataFrame(columns=LOG_COLS)


def append_checkpoint_row(row_dict: dict):
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# SETUP
# ============================================================

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "download_s2_mpc.log"),
        ],
    )


# ============================================================
# DOWNLOAD ONE STATION
# ============================================================

def download_s2(station_dir: Path, catalog, epsg: int, bounds: tuple,
                bbox: list, start: str, end: str, meta: dict) -> tuple[int, int]:
    out_dir = station_dir / "S2L2A"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).items())

    if not items:
        return 0, 0

    n_new = 0
    n_present = 0
    for item in items:
        date_str = item.datetime.strftime("%Y%m%d")
        fpath    = out_dir / f"{date_str}.tif"
        if fpath.exists():
            n_present += 1
            continue

        iso_dt = item.datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

        def _load(it=item):
            planetary_computer.sign_inplace(it)
            da = stackstac.stack(
                [it], assets=S2_BANDS,
                epsg=epsg, resolution=RES_M, bounds=bounds,
                rescale=False,
                resampling=Resampling.bilinear,
            ).squeeze("time")
            return da.compute()

        try:
            da = with_retry(_load)
        except Exception as exc:
            logging.warning(f"    S2 {date_str}: {exc}")
            continue

        da = center_crop(da).astype("int16")
        save_geotiff(da, fpath, epsg, iso_dt)

        meta["S2L2A"][date_str] = {
            "datetime_utc":  iso_dt,
            "cloud_cover":   item.properties.get("eo:cloud_cover", ""),
            "platform":      item.properties.get("platform", ""),
        }
        n_new += 1
        logging.debug(f"    S2L2A {fpath.name}")

    return n_new, n_present


# ============================================================
# DOWNLOAD DEM  (Copernicus GLO-30 via MPC, static — once per station)
# ============================================================

def download_dem(station_dir: Path, catalog, epsg: int, bounds: tuple,
                 bbox: list, meta: dict) -> bool:
    out_path = station_dir / "DEM" / "dem.tif"
    if out_path.exists():
        return True

    items = list(catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
    ).items())

    if not items:
        return False

    def _load(it=items[0]):
        planetary_computer.sign_inplace(it)
        da = stackstac.stack(
            [it], assets=["data"],
            epsg=epsg, resolution=RES_M, bounds=bounds,
            rescale=False,
            resampling=Resampling.bilinear,
        ).squeeze("time").squeeze("band")
        return da.compute()

    try:
        da = with_retry(_load)
    except Exception as exc:
        logging.warning(f"    DEM failed: {exc}")
        return False

    da = center_crop(da).astype("float32")
    # Add band dim back for save_geotiff (expects band × y × x)
    da = da.expand_dims("band")
    save_geotiff(da, out_path, epsg, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 dtype="float32")
    meta["DEM"] = {
        "source":        "cop-dem-glo-30",
        "resolution_m":  RES_M,
        "epsg":          epsg,
        "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    logging.debug(f"    DEM saved → {out_path}")
    return True


# ============================================================
# PER-STATION ORCHESTRATION
# ============================================================

def process_station(row: pd.Series, catalog) -> dict:
    for col in ("latitude", "longitude", "network", "station_id"):
        if pd.isna(row[col]):
            raise ValueError(f"Missing required field: {col}")
    station_id = _station_folder(row)
    lat, lon   = float(row.latitude), float(row.longitude)

    _global_start = date.fromisoformat(GLOBAL_START)
    start_dt = max(_global_start, parse_date(row.start_date)) if pd.notna(row.start_date) else _global_start
    end_dt   = parse_date(row.end_date) if pd.notna(row.end_date) else datetime.now(timezone.utc).date()
    start    = start_dt.isoformat()
    end      = end_dt.isoformat()

    epsg, bounds, bbox = station_grid(lat, lon)
    station_dir = SCRATCH_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "station":       row.station_id,
        "network":       row.network,
        "latitude":      lat,
        "longitude":     lon,
        "epsg":          epsg,
        "patch_size_px": PIXEL_SIZE,
        "pixel_size_m":  RES_M,
        "bounds_utm":    list(bounds),
        "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "S2L2A": {},
        "DEM":   {},
    }

    log = dict.fromkeys(LOG_COLS, "")
    log.update(station_id=station_id, lat=lat, lon=lon)

    try:
        n_new, n_present = download_s2(station_dir, catalog, epsg, bounds, bbox, start, end, meta)
        log["n_scenes"] = n_new + n_present
        log["status"]   = "done" if (n_new + n_present) > 0 else "no_data"
        if n_present:
            logging.debug(f"  {station_id}: {n_present} pre-existing, {n_new} newly downloaded")
    except Exception as exc:
        log["status"]    = "error"
        log["error_msg"] = str(exc)
        logging.error(f"  {station_id} S2 FAILED: {exc}")

    try:
        ok = download_dem(station_dir, catalog, epsg, bounds, bbox, meta)
        log["dem_status"] = "done" if ok else "no_data"
    except Exception as exc:
        log["dem_status"] = "error"
        prefix = f"{log['error_msg']}; " if log["error_msg"] else ""
        log["error_msg"] = f"{prefix}DEM:{exc}"
        logging.error(f"  {station_id} DEM FAILED: {exc}")

    log["timestamp"] = datetime.now(timezone.utc).isoformat()

    _tmp = station_dir / "metadata.json.tmp"
    with open(_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    _tmp.rename(station_dir / "metadata.json")

    return log


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    pilot = pd.read_csv(STATION_CSV).reset_index(drop=True)
    missing_cols = REQUIRED_COLS - set(pilot.columns)
    if missing_cols:
        raise ValueError(f"Station CSV missing required columns: {missing_cols}")
    if TEST_MODE:
        pilot = pilot[pilot.apply(lambda r: _station_folder(r) == TEST_STATION, axis=1)].reset_index(drop=True)
        log.info(f"TEST_MODE: {list(pilot.apply(_station_folder, axis=1))}")
    log.info(f"Stations: {len(pilot)}")

    checkpoint = load_checkpoint()
    done_ids   = set(checkpoint.loc[
        checkpoint["status"].isin(["done", "no_data"]), "station_id"
    ]) if len(checkpoint) else set()
    log.info(f"Already completed: {len(done_ids)} stations")

    todo = [(i, row) for i, row in pilot.iterrows()
            if _station_folder(row) not in done_ids]
    log.info(f"Stations to download: {len(todo)}  (workers={N_WORKERS})")

    ckpt_lock = threading.Lock()
    completed = [0]

    def _process_one(row):
        cat = pystac_client.Client.open(MPC_URL, modifier=planetary_computer.sign_inplace)
        return process_station(row, cat)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_row = {pool.submit(_process_one, row): (i, row) for i, row in todo}

        for fut in as_completed(fut_to_row):
            i, row = fut_to_row[fut]
            station_id = _station_folder(row)
            try:
                row_dict = fut.result()
            except Exception as exc:
                log.error(f"  {station_id} unexpected error: {exc}")
                continue

            completed[0] += 1
            log.info(
                f"[{completed[0]:3d}/{len(todo)}] Done  {station_id}  "
                f"S2={row_dict.get('n_scenes','?')} scenes  "
                f"DEM={row_dict.get('dem_status','?')}  "
                f"status={row_dict.get('status')}"
            )

            with ckpt_lock:
                append_checkpoint_row(row_dict)

    log.info("Done.")


if __name__ == "__main__":
    main()
