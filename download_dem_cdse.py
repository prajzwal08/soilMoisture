"""
Download Copernicus DEM for all ISMN stations via OpenEO / CDSE.

Products per station:
  {station}/DEM/dem.tif                float32, metres, bilinear 10m

Source: Copernicus Data Space Ecosystem (CDSE) via OpenEO Python client.
  DEM     → COPERNICUS_30  + resample_spatial(10m, bilinear)

Note: S2L2A is handled by download_s2_mpc.py (MPC).
      S1RTC and LULC are handled by download_s1_lulc_mpc.py (MPC).

Authentication:
  First run triggers OIDC device-code flow in the browser.
  Credentials are cached locally (~/.local/share/openeo-python-client/).
  Subsequent runs authenticate silently from the cache.

Usage:
  python download_dem_cdse.py
  nohup python download_dem_cdse.py > /tmp/download_dem_cdse.log 2>&1 &
"""

import json
import logging
import os
os.environ.pop("PROJ_DATA", None)   # prevent stale conda PROJ_DATA from conflicting with ~/.local pyproj
import re
from datetime import datetime, timezone
from pathlib import Path

import openeo
import pandas as pd
import rasterio
from dotenv import load_dotenv
from openeo.extra.job_management import (
    CsvJobDatabase,
    MultiBackendJobManager,
)
from pyproj import Transformer

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
STATION_CSV  = DATA_ROOT / "station_splits.csv"
LOG_DIR      = DATA_ROOT / "logs"
JOB_DB_FILE  = LOG_DIR / "openeo_jobs_v2.csv"
LOG_FILE     = LOG_DIR / "download_dem_cdse.log"

# DEM tiles go to scratch. Logs and permanent outputs stay in DATA_ROOT.
SCRATCH_DIR  = Path("/gpfs/scratch1/shared/pkhanal/satellite")

TEST_MODE    = False
TEST_STATION = "ISMN_TWENTE_Hupsel"

PIXEL_SIZE        = 224   # pixels per side stored on disk (TerraMind native size, UNetMobV2 compatible)
RES_M             = 10    # metres per pixel
GLOBAL_START      = "2016-01-01"

CDSE_URL     = "openeo.dataspace.copernicus.eu"
PARALLEL_JOBS = 10      # max simultaneous jobs on CDSE

MODALITIES = ["DEM"]

# ============================================================
# SPATIAL UTILITIES  (same logic as download_satellite.py)
# ============================================================

def get_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def station_grid(lat: float, lon: float):
    """
    Return:
      epsg        – UTM EPSG code
      utm_bounds  – (xmin, ymin, xmax, ymax) in UTM metres (exact 264×264 box)
      latlon_bbox – {"west":…, "south":…, "east":…, "north":…} for OpenEO
                    (slightly expanded to guarantee full coverage after reprojection)
    """
    epsg = get_utm_epsg(lat, lon)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    half = (PIXEL_SIZE * RES_M) / 2                       # = 1320 m
    utm_bounds = (cx - half, cy - half, cx + half, cy + half)

    # Expand by 2 pixels on each side for load_collection (WGS84);
    # exact UTM clip is applied server-side via filter_bbox.
    margin = 2 * RES_M
    corners = [
        inv.transform(utm_bounds[0] - margin, utm_bounds[1] - margin),
        inv.transform(utm_bounds[2] + margin, utm_bounds[3] + margin),
    ]
    latlon_bbox = {
        "west":  corners[0][0],
        "south": corners[0][1],
        "east":  corners[1][0],
        "north": corners[1][1],
    }
    return epsg, utm_bounds, latlon_bbox


def center_crop_tif(path: Path, size: int = PIXEL_SIZE) -> None:
    """Crop a GeoTIFF to size×size pixels centred on the image (atomic write)."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        if h == size and w == size:
            return
        if h < size or w < size:
            raise ValueError(f"Image too small to crop ({h}×{w} < {size}×{size}): {path}")
        row_off = (h - size) // 2
        col_off = (w - size) // 2
        window = rasterio.windows.Window(col_off, row_off, size, size)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
    # tiled=False prevents a crash when source block size (e.g. 256) > target size (224)
    profile.update(height=size, width=size, transform=transform, tiled=False)
    tmp = path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def parse_date(val) -> str:
    """YYYYMMDD float/int or ISO string → 'YYYY-MM-DD'."""
    s = str(val).split(".")[0].strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


# ============================================================
# FILENAME UTILITIES
# ============================================================

def cdse_filename_to_date(name: str) -> str:
    """
    Extract YYYYMMDD from a CDSE output filename.

    CDSE names output files like:
      openEO_2021-06-15Z.tif
      openEO_2021-06-15T10:23:11Z.tif
      2021-06-15.tif
    Returns '20210615'.
    """
    # Find first ISO date pattern in the filename
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    # Fallback: strip extension and non-digits
    return re.sub(r"\D", "", Path(name).stem)[:8]


# ============================================================
# PROCESS GRAPH BUILDER
# ============================================================

def build_cube(connection: openeo.Connection, row: pd.Series):
    """
    Build and return an OpenEO DataCube for the given (station, modality) row.
    Does NOT create/start a batch job — just defines the process graph.
    """
    lat      = float(row["latitude"])
    lon      = float(row["longitude"])
    modality = row["modality"]
    start    = str(row["start_date"])
    end      = str(row["end_date"])

    epsg, utm_bounds, latlon_bbox = station_grid(lat, lon)
    xmin, ymin, xmax, ymax = utm_bounds

    if modality == "DEM":
        cube = connection.load_collection(
            "COPERNICUS_30",
            spatial_extent=latlon_bbox,
            bands=["DEM"],
        )
        cube = cube.resample_spatial(
            resolution=RES_M,
            projection=f"EPSG:{epsg}",
            method="bilinear",
        )
        # DEM: collapse time dimension (take max to handle any duplicates)
        cube = cube.max_time()

    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Clip to exact 264×264 UTM bbox
    cube = cube.filter_bbox(
        west=xmin, south=ymin, east=xmax, north=ymax,
        crs=f"EPSG:{epsg}",
    )

    return cube


# ============================================================
# JOB MANAGER  (subclass to override on_job_done / on_job_error)
# ============================================================

def _write_datetime_tag(path: Path, iso_dt: str) -> None:
    """
    Write the full UTC acquisition datetime into the GeoTIFF TIFFTAG_DATETIME tag.
    iso_dt: ISO 8601 string e.g. '2021-07-04T10:23:11Z'
    Tag format required by TIFF spec: 'YYYY:MM:DD HH:MM:SS'
    """
    try:
        dt = iso_dt.replace("T", " ").replace("Z", "").split(".")[0]  # → '2021-07-04 10:23:11'
        dt_tag = dt[:10].replace("-", ":") + dt[10:]                  # → '2021:07:04 10:23:11'
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(TIFFTAG_DATETIME=dt_tag, datetime_utc=iso_dt)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not write datetime tag to {path}: {e}")


def _get_scratch_dir(row: pd.Series, station_id: str) -> Path:
    if "scratch_dir" in row.index and pd.notna(row["scratch_dir"]):
        return Path(row["scratch_dir"])
    return SCRATCH_DIR / station_id


class SatelliteJobManager(MultiBackendJobManager):
    """
    Extends MultiBackendJobManager to:
    - Download results into the station/modality folder
    - Rename CDSE output files to YYYYMMDD.tif / YYYYMMDD_ASC.tif convention
    - Write/update metadata.json per station
    - Auto-reconnect when CDSE token expires (403 TokenInvalid)
    """

    def __init__(self, *args, username: str = None, password: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cdse_username = username
        self._cdse_password = password

    def _connection_factory(self) -> openeo.Connection:
        """Create a fresh authenticated CDSE connection."""
        log = logging.getLogger(__name__)
        log.info("Authenticating with CDSE…")
        conn = openeo.connect(CDSE_URL).authenticate_oidc_resource_owner_password_credentials(
            client_id="cdse-public",
            username=self._cdse_username,
            password=self._cdse_password,
            client_secret="",
        )
        log.info("  ✓ Authenticated with CDSE")
        return conn

    def _track_statuses(self, job_db, stats=None):
        import collections
        if stats is None:
            stats = collections.defaultdict(int)
        errors_before = stats.get("job tracking error", 0)
        result = super()._track_statuses(job_db, stats)
        if stats.get("job tracking error", 0) > errors_before:
            # Token likely expired — clear cached connection so factory is called next cycle
            log = logging.getLogger(__name__)
            log.info("Tracking errors detected — clearing cached CDSE connection for re-auth next cycle")
            self._connections.pop("cdse", None)
        return result

    def on_job_done(self, job, row):
        station_id = row["station_id"]
        modality   = row["modality"]
        log        = logging.getLogger(__name__)

        out_dir = _get_scratch_dir(row, station_id) / modality
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = job.get_results()
            # Build asset-name → full UTC datetime map from STAC metadata
            stac_meta = results.get_metadata()
            datetime_map = {}
            for feature in stac_meta.get("features", []):
                dt = feature.get("properties", {}).get("datetime", "")
                for aname in feature.get("assets", {}):
                    if dt:
                        datetime_map[aname] = dt

            n_saved = 0
            for asset in results.get_assets():
                if not asset.name.lower().endswith(".tif"):
                    continue

                dest = out_dir / "dem.tif"

                # Skip if already a valid, correctly-sized file
                if dest.exists():
                    try:
                        with rasterio.open(dest) as src:
                            if src.height == PIXEL_SIZE and src.width == PIXEL_SIZE:
                                n_saved += 1
                                continue
                    except Exception:
                        pass
                    dest.unlink(missing_ok=True)  # corrupt or wrong size — redownload

                asset.download(str(dest))
                center_crop_tif(dest)
                # DEM is a static composite (max_time strips STAC datetime); no tag to write
                n_saved += 1

            log.info(f"  ✓ {station_id}/{modality}  {n_saved} files → {out_dir}")
            self._update_metadata(station_id, modality, job, n_saved, row)

        except Exception as exc:
            log.error(f"  ✗ {station_id}/{modality} download failed: {exc}")

    def on_job_error(self, job, row):
        station_id = row["station_id"]
        modality   = row["modality"]
        log        = logging.getLogger(__name__)
        log.error(f"  ✗ {station_id}/{modality} job error: {job.job_id}")
        try:
            for entry in job.logs():
                if entry.get("level") in ("error", "warning"):
                    log.error(f"    [{entry['level']}] {entry.get('message','')}")
        except Exception as e:
            log.debug(f"Could not retrieve logs for job {job.job_id}: {e}")

    def _update_metadata(self, station_id: str, modality: str,
                         job, n_files: int, row: pd.Series):
        """Write/update metadata.json for the station."""
        meta_path = _get_scratch_dir(row, station_id) / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            epsg, bounds, _ = station_grid(lat, lon)
            meta = {
                "station":       row["station_id"],
                "network":       row["network"],
                "latitude":      lat,
                "longitude":     lon,
                "epsg":          epsg,
                "patch_size_px": PIXEL_SIZE,
                "pixel_size_m":  RES_M,
                "bounds_utm":    list(bounds),
                "download_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "DEM": {},
            }
        meta[modality] = meta.get(modality, {})
        meta[modality]["job_id"]  = job.job_id
        meta[modality]["n_files"] = n_files
        meta[modality]["source"]  = "CDSE/OpenEO"
        meta_path.write_text(json.dumps(meta, indent=2))


# ============================================================
# BUILD JOBS DATAFRAME
# ============================================================

def _station_folder(st) -> str:
    if st.source_network != st.network:
        return f"{st.source_network}_{st.network}_{st.station_id}"
    return f"{st.network}_{st.station_id}"


def _station_dir(st) -> Path:
    has_sm = str(getattr(st, "has_soil_moisture", "False")).lower() == "true"
    has_fl = str(getattr(st, "has_flux", "False")).lower() == "true"
    cat = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
    return DATA_ROOT / cat / _station_folder(st)


def build_jobs_df(stations: pd.DataFrame) -> pd.DataFrame:
    """One row per station (single DEM modality)."""
    rows = []
    for _, st in stations.iterrows():
        station_id  = _station_folder(st)
        station_dir = str(_station_dir(st))
        start = max(GLOBAL_START, parse_date(st.start_date)) \
                if pd.notna(st.start_date) else GLOBAL_START
        end   = parse_date(st.end_date) \
                if pd.notna(st.end_date) \
                else datetime.now().strftime("%Y-%m-%d")

        for modality in MODALITIES:
            rows.append({
                "station_id":      station_id,
                "station_dir":     station_dir,
                "scratch_dir":     str(SCRATCH_DIR / station_id),
                "network":         st.network,
                "station":         st.station_id,
                "latitude":        float(st.latitude),
                "longitude":       float(st.longitude),
                "start_date":      start,
                "end_date":        end,
                "modality":        modality,
                "backend_name":    "cdse",
            })

    return pd.DataFrame(rows)


# ============================================================
# START-JOB CALLBACK  (called by MultiBackendJobManager per row)
# ============================================================

def start_job(row: pd.Series, connection: openeo.Connection, **kwargs):
    """
    Build the process graph for this (station, modality) and create a batch job.
    Returns a BatchJob (not yet started — the manager starts it).
    """
    station_id = row["station_id"]
    modality   = row["modality"]
    cube = build_cube(connection, row)
    job  = cube.create_job(
        out_format="GTiff",
        title=f"{station_id}_{modality}",
        description=f"264×264 px @ 10m | {row['start_date']} → {row['end_date']}",
    )
    logging.getLogger(__name__).info(
        f"  Created job {job.job_id}  [{station_id}  {modality}]"
    )
    return job


# ============================================================
# MAIN
# ============================================================

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )
    log = logging.getLogger(__name__)

    # ── Load stations ────────────────────────────────────────
    pilot = pd.read_csv(STATION_CSV).reset_index(drop=True)
    if TEST_MODE:
        pilot = pilot[pilot.apply(lambda r: _station_folder(r) == TEST_STATION, axis=1)].reset_index(drop=True)
        if len(pilot) == 0:
            log.error(f"TEST_MODE: no station matching '{TEST_STATION}' — check TEST_STATION")
            return
        log.info(f"TEST_MODE: running on {len(pilot)} station(s): {list(pilot.apply(_station_folder, axis=1))}")
    log.info(f"Stations: {len(pilot)} (all from station_splits.csv)")

    # ── Build jobs DataFrame ─────────────────────────────────
    jobs_df = build_jobs_df(pilot)
    log.info(f"Total jobs: {len(jobs_df)} "
             f"({len(pilot)} stations × {len(MODALITIES)} modalities)")

    # ── Authenticate with CDSE ───────────────────────────────
    load_dotenv(Path(__file__).parent / ".env")
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set CDSE_USERNAME and CDSE_PASSWORD in "
            f"{Path(__file__).parent / '.env'}"
        )
    # ── Set up job manager (credentials stored for auto-reconnect) ──
    manager = SatelliteJobManager(
        poll_sleep=60,
        root_dir=str(LOG_DIR / "_job_meta"),
        username=username,
        password=password,
    )
    log.info(f"Connecting to CDSE as {username}…")
    manager.add_backend(
        "cdse",
        connection=manager._connection_factory,
        parallel_jobs=PARALLEL_JOBS,
    )

    # ── Job database (CSV) — resumable ───────────────────────
    # If the file already exists, the manager skips finished jobs automatically.
    job_db = CsvJobDatabase(JOB_DB_FILE)
    if not job_db.exists():
        job_db.initialize_from_df(jobs_df)
        log.info(f"Job database created: {JOB_DB_FILE}")
    else:
        log.info(f"Resuming from existing job database: {JOB_DB_FILE}")

    # ── Run ──────────────────────────────────────────────────
    log.info(f"Starting job manager  (parallel_jobs={PARALLEL_JOBS}, "
             f"poll_sleep=60s)…")
    stats = manager.run_jobs(start_job=start_job, job_db=job_db)

    log.info("Job manager finished.")
    log.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
