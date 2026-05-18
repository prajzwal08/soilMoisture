"""
Download OpenLandMap-soildb soil property patches for all stations.

Uses HTTP range requests on Cloud Optimized GeoTIFFs (COGs) hosted at
s3.opengeohub.org — no full file download needed (~14 GB each).
Reads only the 74×74 px window centred on each station.

Source: OpenLandMap-soildb (Hengl et al., 2026 ESSD)
        https://doi.org/10.5281/zenodo.15470431
        COG server: https://s3.opengeohub.org/global-soil/

Variables (7 channels, 0–30 cm, 2020-2022 composite):
  0  clay      clay content       (wt%, uint8 → divide by 2 for %)
  1  sand      sand content       (wt%, uint8 → divide by 2 for %)
  2  silt      silt content       (wt%, uint8 → divide by 2 for %)
  3  soc       SOC content        (g/kg, uint16 → multiply by scaler)
  4  socd      SOC density        (mg/cm³, uint16 → multiply by scaler)
  5  bd        bulk density       (g/cm³, uint8 → divide by 100)
  6  ph        pH in H₂O          (pH*10, uint8 → divide by 10)

Patch: 74×74 pixels @ ~27.5 m = 2.035 km × 2.035 km centred on station

Output per station:
  {station}/soil/soil_patch.tif   float32, 7 bands, 74×74 px

Usage:
  python download_soil_openlandmap.py
  nohup python download_soil_openlandmap.py > /tmp/download_soil.log 2>&1 &
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
import rasterio.crs

# ============================================================
# CONFIGURATION
# ============================================================

STATION_CSV   = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_splits.csv")
SATELLITE_DIR = Path("/home/khanalp/data/satellite")

PATCH_PX  = 74      # pixels per side
RES_DEG   = 0.00025 # ~27.5 m per pixel in degrees

N_WORKERS = 8       # concurrent station downloads

# COG URLs for 7 soil variables (0–30 cm mean, 2020–2022 composite)
# Source: https://raw.githubusercontent.com/openlandmap/soildb/main/tables/OpenLandMap_soildb_COGS.csv
SOIL_LAYERS = [
    {
        "name":   "clay",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/"
                  "clay.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
        "units":  "wt%",
        "scaler": 1.0,   # uint8, values already in %
    },
    {
        "name":   "sand",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/"
                  "sand.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
        "units":  "wt%",
        "scaler": 1.0,
    },
    {
        "name":   "silt",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/"
                  "silt.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
        "units":  "wt%",
        "scaler": 1.0,
    },
    {
        "name":   "soc",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/"
                  "oc_iso.10694.1995.wpml_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",
        "units":  "g/kg",
        "scaler": 1.0,
    },
    {
        "name":   "socd",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/"
                  "oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",
        "units":  "mg/cm3",
        "scaler": 1.0,
    },
    {
        "name":   "bd",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/"
                  "bd.core_iso.11272.2017.g.cm3_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",
        "units":  "g/cm3",
        "scaler": 1.0,
    },
    {
        "name":   "ph",
        "url":    "https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/"
                  "ph.h2o_iso.10390.2021.index_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif",
        "units":  "pH",
        "scaler": 1.0,
    },
]

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
            logging.FileHandler(SATELLITE_DIR / "soil_download.log"),
        ],
    )

# ============================================================
# STATION LOADING
# ============================================================

def load_stations() -> pd.DataFrame:
    df = pd.read_csv(STATION_CSV)
    df["station_id"] = df["network"] + "_" + df["station_id"]
    return df.reset_index(drop=True)

# ============================================================
# PATCH EXTRACTION
# ============================================================

def station_bbox(lat: float, lon: float) -> tuple:
    """Return (west, south, east, north) bounding box for 74×74 patch."""
    half = (PATCH_PX / 2) * RES_DEG
    return (lon - half, lat - half, lon + half, lat + half)


def read_patch(url: str, lat: float, lon: float) -> np.ndarray | None:
    """
    Open COG via HTTP range request and read 74×74 px window.
    Returns float32 array (PATCH_PX, PATCH_PX) or None on failure.
    """
    west, south, east, north = station_bbox(lat, lon)
    try:
        with rasterio.open(url) as src:
            window = window_from_bounds(west, south, east, north, src.transform)
            data   = src.read(1, window=window, out_shape=(PATCH_PX, PATCH_PX),
                              resampling=rasterio.enums.Resampling.bilinear)
            nodata = src.nodata
        arr = data.astype(np.float32)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr
    except Exception as exc:
        logging.getLogger(__name__).warning(f"  Read failed ({url.split('/')[-1]}): {exc}")
        return None


def process_station(row: pd.Series) -> str:
    """Download and save 7-band soil patch for one station. Returns status."""
    station_id = row["station_id"]
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    log = logging.getLogger(__name__)

    out_dir  = SATELLITE_DIR / station_id / "soil"
    out_path = out_dir / "soil_patch.tif"

    if out_path.exists():
        return "skip"

    bands = []
    for layer in SOIL_LAYERS:
        arr = read_patch(layer["url"], lat, lon)
        if arr is None:
            log.warning(f"  {station_id}: failed to read {layer['name']}, filling NaN")
            arr = np.full((PATCH_PX, PATCH_PX), np.nan, dtype=np.float32)
        if layer["scaler"] != 1.0:
            arr = arr * layer["scaler"]
        bands.append(arr)

    stack = np.stack(bands, axis=0)   # (7, 74, 74)

    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = station_bbox(lat, lon)
    transform = from_bounds(west, south, east, north, PATCH_PX, PATCH_PX)

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=PATCH_PX, width=PATCH_PX,
        count=len(SOIL_LAYERS),
        dtype="float32",
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(stack)
        for i, layer in enumerate(SOIL_LAYERS, start=1):
            dst.update_tags(i, name=layer["name"], units=layer["units"])

    log.debug(f"  {station_id}: soil_patch.tif saved")
    return "done"

# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    log = logging.getLogger(__name__)

    df = load_stations()

    # Skip stations that already have the patch
    todo = df[df["station_id"].apply(
        lambda s: not (SATELLITE_DIR / s / "soil" / "soil_patch.tif").exists()
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
