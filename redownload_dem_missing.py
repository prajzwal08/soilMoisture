"""
redownload_dem_missing.py
=========================
Download DEM for stations that have no dem.tif in scratch (original download
failures or tiles deleted by the bad-tile filter).

Detects missing stations dynamically from csvs/station_splits.csv — no
hardcoded list required.

Usage:
    python redownload_dem_missing.py
"""

import logging
import math
import os
os.environ.pop("PROJ_DATA", None)
from pathlib import Path

import openeo
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from dotenv import load_dotenv
from openeo.extra.job_management import CsvJobDatabase, MultiBackendJobManager
from pyproj import Transformer

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
LOG_DIR     = DATA_ROOT / "logs"
JOB_DB_FILE = LOG_DIR / "openeo_jobs_dem_missing.csv"
LOG_FILE    = LOG_DIR / "redownload_dem_missing.log"

PIXEL_SIZE    = 224
RES_M         = 10
CDSE_URL      = "openeo.dataspace.copernicus.eu"
PARALLEL_JOBS = 10
DEM_TILE_BUFFER = 0.15


# ── spatial utilities (identical to redownload_dem_fix.py) ───────────────────

def get_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def station_grid(lat: float, lon: float):
    epsg = get_utm_epsg(lat, lon)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    half = (PIXEL_SIZE * RES_M) / 2
    utm_bounds = (cx - half, cy - half, cx + half, cy + half)

    margin = 2 * RES_M
    corners = [
        inv.transform(utm_bounds[0] - margin, utm_bounds[1] - margin),
        inv.transform(utm_bounds[2] + margin, utm_bounds[3] + margin),
    ]
    latlon_bbox = {
        "west":  corners[0][0], "south": corners[0][1],
        "east":  corners[1][0], "north": corners[1][1],
    }
    if math.floor(latlon_bbox["north"]) > math.floor(latlon_bbox["south"]):
        latlon_bbox["south"] -= DEM_TILE_BUFFER
        latlon_bbox["north"] += DEM_TILE_BUFFER
    if math.floor(latlon_bbox["east"]) > math.floor(latlon_bbox["west"]):
        latlon_bbox["west"]  -= DEM_TILE_BUFFER
        latlon_bbox["east"]  += DEM_TILE_BUFFER
    return epsg, utm_bounds, latlon_bbox


def center_crop_tif(path: Path, size: int = PIXEL_SIZE) -> None:
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        if h == size and w == size:
            return
        if h < size or w < size:
            raise ValueError(f"Image too small to crop ({h}×{w}): {path}")
        row_off = (h - size) // 2
        col_off = (w - size) // 2
        window = rasterio.windows.Window(col_off, row_off, size, size)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
    profile.update(height=size, width=size, transform=transform, tiled=False)
    tmp = path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── job manager ───────────────────────────────────────────────────────────────

class MissingDEMJobManager(MultiBackendJobManager):

    def __init__(self, *args, username=None, password=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cdse_username = username
        self._cdse_password = password

    def _connection_factory(self):
        log = logging.getLogger(__name__)
        log.info("Authenticating with CDSE…")
        conn = openeo.connect(CDSE_URL).authenticate_oidc_resource_owner_password_credentials(
            client_id="cdse-public",
            username=self._cdse_username,
            password=self._cdse_password,
            client_secret="",
        )
        log.info("  ✓ Authenticated")
        return conn

    def _track_statuses(self, job_db, stats=None):
        import collections
        if stats is None:
            stats = collections.defaultdict(int)
        errors_before = stats.get("job tracking error", 0)
        result = super()._track_statuses(job_db, stats)
        if stats.get("job tracking error", 0) > errors_before:
            self._connections.pop("cdse", None)
        return result

    def on_job_done(self, job, row):
        station_id = row["station_id"]
        log = logging.getLogger(__name__)
        out_dir = SCRATCH_DIR / station_id / "DEM"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "dem.tif"
        try:
            results = job.get_results()
            for asset in results.get_assets():
                if not asset.name.lower().endswith(".tif"):
                    continue
                dest.unlink(missing_ok=True)
                asset.download(str(dest))
                center_crop_tif(dest)
                with rasterio.open(dest) as src:
                    nan_frac = np.isnan(src.read(1).astype(np.float32)).mean()
                if nan_frac > 0:
                    log.warning(f"  ! {station_id}: NaN present (frac={nan_frac:.3f})")
                else:
                    log.info(f"  ✓ {station_id}: DEM saved → {dest}")
        except Exception as exc:
            log.error(f"  ✗ {station_id} download failed: {exc}")

    def on_job_error(self, job, row):
        log = logging.getLogger(__name__)
        log.error(f"  ✗ {row['station_id']} job error: {job.job_id}")
        try:
            for entry in job.logs():
                if entry.get("level") in ("error", "warning"):
                    log.error(f"    [{entry['level']}] {entry.get('message','')}")
        except Exception:
            pass


def start_job(row: pd.Series, connection: openeo.Connection, **kwargs):
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    epsg, utm_bounds, latlon_bbox = station_grid(lat, lon)
    xmin, ymin, xmax, ymax = utm_bounds

    cube = connection.load_collection(
        "COPERNICUS_30",
        spatial_extent=latlon_bbox,
        bands=["DEM"],
    )
    cube = cube.resample_spatial(resolution=RES_M, projection=f"EPSG:{epsg}", method="bilinear")
    cube = cube.max_time()
    cube = cube.filter_bbox(west=xmin, south=ymin, east=xmax, north=ymax, crs=f"EPSG:{epsg}")

    job = cube.create_job(out_format="GTiff", title=f"{row['station_id']}_DEM_missing")
    logging.getLogger(__name__).info(f"  Created job {job.job_id}  [{row['station_id']}]")
    return job


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
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

    stations = pd.read_csv(Path("csvs/station_splits.csv")).reset_index(drop=True)

    def station_folder(st):
        if st.source_network != st.network:
            return f"{st.source_network}_{st.network}_{st.station_id}"
        return f"{st.network}_{st.station_id}"

    stations["station_folder"] = stations.apply(station_folder, axis=1)

    # Find stations with no dem.tif
    missing = stations[
        ~stations["station_folder"].apply(
            lambda f: (SCRATCH_DIR / f / "DEM" / "dem.tif").exists()
        )
    ].reset_index(drop=True)

    log.info(f"Stations missing dem.tif: {len(missing)}")
    for _, st in missing.iterrows():
        log.info(f"  {st['station_folder']}")

    if len(missing) == 0:
        log.info("Nothing to do.")
        return

    rows = [
        {
            "station_id":   st["station_folder"],
            "network":      st["network"],
            "latitude":     float(st["latitude"]),
            "longitude":    float(st["longitude"]),
            "backend_name": "cdse",
        }
        for _, st in missing.iterrows()
    ]
    jobs_df = pd.DataFrame(rows)

    load_dotenv(Path(__file__).parent / ".env")
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD in .env")

    manager = MissingDEMJobManager(
        poll_sleep=60,
        root_dir=str(LOG_DIR / "_job_meta_dem_missing"),
        username=username,
        password=password,
    )
    manager.add_backend("cdse", connection=manager._connection_factory,
                        parallel_jobs=PARALLEL_JOBS)

    job_db = CsvJobDatabase(JOB_DB_FILE)
    if not job_db.exists():
        job_db.initialize_from_df(jobs_df)
        log.info(f"Job DB: {JOB_DB_FILE}")
    else:
        log.info(f"Resuming from: {JOB_DB_FILE}")

    manager.run_jobs(start_job=start_job, job_db=job_db)


if __name__ == "__main__":
    main()
