"""
Download OpenLandMap-soildb soil property patches for all stations.

Uses HTTP range requests on Cloud Optimized GeoTIFFs (COGs) hosted at
s3.opengeohub.org — no full file download needed (~14 GB each).
Reads only the 74×74 px window centred on each station.

Source: OpenLandMap-soildb (Hengl et al., 2026 ESSD)
        https://doi.org/10.5281/zenodo.15470431
        COG server: https://s3.opengeohub.org/global-soil/

Variables (21 channels = 3 depths × 7 properties, 2020-2022 composite):
  Depth layers: 0–30 cm (ch 0–6), 30–60 cm (ch 7–13), 60–100 cm (ch 14–20)
  Per depth: clay, sand, silt, soc, socd, bd, ph

  ch 0 / 7 / 14   clay   clay content       wt%
  ch 1 / 8 / 15   sand   sand content       wt%
  ch 2 / 9 / 16   silt   silt content       wt%
  ch 3 / 10 / 17  soc    SOC content        g/kg
  ch 4 / 11 / 18  socd   SOC density        mg/cm³
  ch 5 / 12 / 19  bd     bulk density       g/cm³
  ch 6 / 13 / 20  ph     pH in H₂O          –

Patch: 74×74 pixels @ 30 m = 2.22 km × 2.22 km centred on station
       Physical extent is consistent across all latitudes via UTM projection.

Output per station:
  {station}/soil/soil_patch.tif   float32, 21 bands, 74×74 px

Usage:
  python download_soil_openlandmap.py
  nohup python download_soil_openlandmap.py > /tmp/download_soil.log 2>&1 &
"""

import logging
import os
os.environ.pop("PROJ_DATA", None)   # prevent stale conda PROJ_DATA from conflicting with ~/.local pyproj
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
import rasterio.crs
from pyproj import Transformer

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT     = Path("/gpfs/work3/0/prjs1968/data")
STATION_CSV   = DATA_ROOT / "station_splits.csv"
LOG_DIR       = DATA_ROOT / "logs"

TEST_MODE     = False
TEST_STATION  = "ISMN_TWENTE_Hupsel"

PATCH_PX  = 74   # pixels per side
RES_M     = 30   # target physical resolution (m) — gives 74×30 = 2220 m patch

N_WORKERS = 16      # concurrent station downloads (public S3 COGs — no rate limit)

# COG URLs: 3 depths × 7 variables = 21 channels, 2020-2022 composite
# Source: https://raw.githubusercontent.com/openlandmap/soildb/main/tables/OpenLandMap_soildb_COGS.csv
_BASE_V523 = "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/"
_BASE_V204 = "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/"

SOIL_LAYERS = [
    # ── 0–30 cm (channels 0–6) ───────────────────────────────────────────────
    {"name": "clay_0_30",  "depth": "0-30cm",  "url": _BASE_V523 + "clay.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",  "units": "wt%"},
    {"name": "sand_0_30",  "depth": "0-30cm",  "url": _BASE_V523 + "sand.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",  "units": "wt%"},
    {"name": "silt_0_30",  "depth": "0-30cm",  "url": _BASE_V523 + "silt.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",  "units": "wt%"},
    {"name": "soc_0_30",   "depth": "0-30cm",  "url": _BASE_V204 + "oc_iso.10694.1995.wpml_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",        "units": "g/kg"},
    {"name": "socd_0_30",  "depth": "0-30cm",  "url": _BASE_V204 + "oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",      "units": "mg/cm3"},
    {"name": "bd_0_30",    "depth": "0-30cm",  "url": _BASE_V204 + "bd.core_iso.11272.2017.g.cm3_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",  "units": "g/cm3"},
    {"name": "ph_0_30",    "depth": "0-30cm",  "url": _BASE_V204 + "ph.h2o_iso.10390.2021.index_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",   "units": "pH"},
    # ── 30–60 cm (channels 7–13) ─────────────────────────────────────────────
    {"name": "clay_30_60", "depth": "30-60cm", "url": _BASE_V523 + "clay.tot_iso.11277.2020.wpct_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250523.tif", "units": "wt%"},
    {"name": "sand_30_60", "depth": "30-60cm", "url": _BASE_V523 + "sand.tot_iso.11277.2020.wpct_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250523.tif", "units": "wt%"},
    {"name": "silt_30_60", "depth": "30-60cm", "url": _BASE_V523 + "silt.tot_iso.11277.2020.wpct_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250523.tif", "units": "wt%"},
    {"name": "soc_30_60",  "depth": "30-60cm", "url": _BASE_V204 + "oc_iso.10694.1995.wpml_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250204.tif",       "units": "g/kg"},
    {"name": "socd_30_60", "depth": "30-60cm", "url": _BASE_V204 + "oc_iso.10694.1995.mg.cm3_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250204.tif",     "units": "mg/cm3"},
    {"name": "bd_30_60",   "depth": "30-60cm", "url": _BASE_V204 + "bd.core_iso.11272.2017.g.cm3_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250204.tif", "units": "g/cm3"},
    {"name": "ph_30_60",   "depth": "30-60cm", "url": _BASE_V204 + "ph.h2o_iso.10390.2021.index_m_30m_b30cm..60cm_20200101_20221231_g_epsg.4326_v20250204.tif",  "units": "pH"},
    # ── 60–100 cm (channels 14–20) ───────────────────────────────────────────
    {"name": "clay_60_100","depth": "60-100cm","url": _BASE_V523 + "clay.tot_iso.11277.2020.wpct_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250523.tif","units": "wt%"},
    {"name": "sand_60_100","depth": "60-100cm","url": _BASE_V523 + "sand.tot_iso.11277.2020.wpct_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250523.tif","units": "wt%"},
    {"name": "silt_60_100","depth": "60-100cm","url": _BASE_V523 + "silt.tot_iso.11277.2020.wpct_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250523.tif","units": "wt%"},
    {"name": "soc_60_100", "depth": "60-100cm","url": _BASE_V204 + "oc_iso.10694.1995.wpml_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250204.tif",      "units": "g/kg"},
    {"name": "socd_60_100","depth": "60-100cm","url": _BASE_V204 + "oc_iso.10694.1995.mg.cm3_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250204.tif",    "units": "mg/cm3"},
    {"name": "bd_60_100",  "depth": "60-100cm","url": _BASE_V204 + "bd.core_iso.11272.2017.g.cm3_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250204.tif","units": "g/cm3"},
    {"name": "ph_60_100",  "depth": "60-100cm","url": _BASE_V204 + "ph.h2o_iso.10390.2021.index_m_30m_b60cm..100cm_20200101_20221231_g_epsg.4326_v20250204.tif", "units": "pH"},
]

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
            logging.FileHandler(LOG_DIR / "soil_download.log"),
        ],
    )

# ============================================================
# STATION LOADING
# ============================================================

def load_stations() -> pd.DataFrame:
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
    return df.reset_index(drop=True)

# ============================================================
# PATCH EXTRACTION
# ============================================================

def _utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def station_bbox_wgs84(lat: float, lon: float) -> tuple:
    """
    Return (west, south, east, north) in WGS84 degrees for a physically
    consistent PATCH_PX × RES_M patch centred on (lat, lon).

    Projects to UTM → creates ±half_m bbox → unprojects corners back to WGS84.
    This ensures equal physical size (~2220 m × 2220 m) at all latitudes,
    avoiding the east-west shrinkage of a degree-based bbox near the poles.
    """
    epsg = _utm_epsg(lat, lon)
    fwd  = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv  = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    cx, cy   = fwd.transform(lon, lat)
    half_m   = (PATCH_PX / 2) * RES_M          # = 37 × 30 = 1110 m

    w_lon, s_lat = inv.transform(cx - half_m, cy - half_m)
    e_lon, n_lat = inv.transform(cx + half_m, cy + half_m)
    return (w_lon, s_lat, e_lon, n_lat)


def read_patch(url: str, lat: float, lon: float) -> tuple[np.ndarray | None, object]:
    """
    Open COG via HTTP range request and read 74×74 px window.
    Returns (float32 array (PATCH_PX, PATCH_PX), crs) or (None, None) on failure.
    """
    west, south, east, north = station_bbox_wgs84(lat, lon)
    try:
        with rasterio.open(url) as src:
            window = window_from_bounds(west, south, east, north, src.transform)
            data   = src.read(1, window=window, out_shape=(PATCH_PX, PATCH_PX),
                              resampling=rasterio.enums.Resampling.bilinear)
            nodata = src.nodata
            crs    = src.crs
        arr = data.astype(np.float32)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, crs
    except Exception as exc:
        logging.getLogger(__name__).warning(f"  Read failed ({url.split('/')[-1]}): {exc}")
        return None, None


def process_station(row: pd.Series) -> str:
    """Download and save 7-band soil patch for one station. Returns status."""
    station_id = row["station_id"]
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    log = logging.getLogger(__name__)

    out_dir  = Path(row["station_dir"]) / "soil"
    out_path = out_dir / "soil_patch.tif"

    if out_path.exists():
        return "skip"

    bands = []
    src_crs = None
    for layer in SOIL_LAYERS:
        arr, crs = read_patch(layer["url"], lat, lon)
        if arr is None:
            log.warning(f"  {station_id}: failed to read {layer['name']}, filling NaN")
            arr = np.full((PATCH_PX, PATCH_PX), np.nan, dtype=np.float32)
        if crs is not None and src_crs is None:
            src_crs = crs
        bands.append(arr)

    stack = np.stack(bands, axis=0)   # (21, 74, 74)

    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = station_bbox_wgs84(lat, lon)
    transform = from_bounds(west, south, east, north, PATCH_PX, PATCH_PX)

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=PATCH_PX, width=PATCH_PX,
        count=len(SOIL_LAYERS),
        dtype="float32",
        crs=src_crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(stack)
        for i, layer in enumerate(SOIL_LAYERS, start=1):
            dst.update_tags(i, name=layer["name"], units=layer["units"], depth=layer["depth"])

    log.debug(f"  {station_id}: soil_patch.tif saved")
    return "done"

# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    df = load_stations()
    if TEST_MODE:
        df = df[df["station_id"] == TEST_STATION].reset_index(drop=True)
        log.info(f"TEST_MODE: running on {len(df)} station(s): {list(df['station_id'])}")

    # Skip stations that already have the patch
    todo = df[~df["station_dir"].apply(
        lambda d: (Path(d) / "soil" / "soil_patch.tif").exists()
    )].reset_index(drop=True)

    log.info(f"Stations: {len(df)} total, {len(todo)} to download, "
             f"{len(df) - len(todo)} already done")

    done = [0]

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        fut_to_row = {pool.submit(process_station, row): row
                      for _, row in todo.iterrows()}

        for fut in as_completed(fut_to_row):
            row = fut_to_row[fut]
            try:
                status = fut.result()
            except Exception as exc:
                log.error(f"  {row['station_id']} worker error: {exc}")
                continue

            if status != "skip":
                done[0] += 1
                log.info(f"  [{done[0]:4d}/{len(todo)}] {row['station_id']}  {status}")

    log.info("Done.")


if __name__ == "__main__":
    main()
