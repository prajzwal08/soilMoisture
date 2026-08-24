"""
Fetch the wide DEM for each processing region: GLO-30 from AWS public COGs,
reprojected to the region's Lambert Azimuthal Equal Area grid at exactly 30 m (§32.3).

Why not the existing downloader: GLO-30 is published as public COGs on
copernicus-dem-30m.s3.amazonaws.com with no auth and no request signing, so
download_dem_cdse.py's 993-job openEO MultiBackendJobManager is unnecessary here.
That file is left untouched and keeps serving the existing 10 m station tiles.

Why regions and not station windows: flow accumulation is non-local. `a` must be
integrated over tens of km and HAND's D8 trace must reach a stream that lies
outside the 2.24 km tile in 25 of 30 sampled cases (§31.4). Regions also put the
75 colocated pairs inside one continuous flow field, so the sufficiency gate's
DTWI carries no differential boundary error.

Two GLO-30 grid traps, both handled here:
  1. Longitude is decimated poleward (measured: 3600 px wide at S34, 2400 at N52,
     1800 at N60, 1200 at N70), so a region can span source tiles of different
     pixel width. Tiles are mosaicked per resolution group before warping.
  2. Ocean tiles do not exist — they 404 rather than returning zeros. And the COGs
     carry nodata=None, so sea-level 0.0 is indistinguishable from no data by
     value alone. Absent tiles therefore become NaN in the output and the valid
     mask comes from tile coverage, not from the pixel values.

Elevations are written as float32 with NaN outside coverage. Nothing downstream
consumes these elevations directly — only TWI and HAND derived from them
(build_twi_hand.py) — so this is an intermediate, not a model input.

Output per region:
  {TERRAIN_ROOT}/region_{id:04d}/dem_glo30_30m.tif
      float32, 1 band, LAEA @ 30 m, NaN nodata, deflate, tiled
      tags: laea_proj4, buffer_km, tiles_used, tiles_missing, nan_frac, seam_lat_band

Usage:
  conda activate soilmoisture
  python download_wide_dem.py                       # all 353 regions, resume-safe
  python download_wide_dem.py --region-id 0         # one region
  python download_wide_dem.py --max-stations 12     # pilot: regions covering N stations
  python download_wide_dem.py --overwrite
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.pop("PROJ_DATA", None)                          # stale conda PROJ_DATA vs pyproj
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")  # no bucket listing per open
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds

REPO         = Path(__file__).resolve().parent
REGION_CSV   = REPO / "csvs" / "dem_regions.csv"
STATION_CSV  = REPO / "csvs" / "station_dem_region.csv"
TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
LOG_DIR      = Path("/gpfs/work3/0/prjs1968/data/logs")

N_WORKERS    = 8       # concurrent regions; GDAL releases the GIL during warp
_MAX_RETRIES = 3
_RETRY_BASE  = 2       # 2 s, 4 s before attempts 2 and 3

S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# GLO-30 longitude-decimation boundaries (§32.3). A region spanning one mosaics
# source grids of different pixel width, so a 1-px bilinear seam can appear on
# that latitude line; recorded in the output tags rather than silently ignored.
GLO30_LAT_BANDS = (50.0, 60.0, 70.0, 80.0)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOG_DIR / "wide_dem_download.log")],
    )


def tile_url(tile: str) -> str:
    """tile is e.g. 'N52E006'; COG names embed the SW corner twice."""
    name = f"Copernicus_DSM_COG_10_{tile[:3]}_00_{tile[3:]}_00_DEM"
    return f"{S3_BASE}/{name}/{name}.tif"


def open_tile(tile: str, log: logging.Logger):
    """
    Open a GLO-30 COG, retrying transient failures. Returns the dataset, or None
    if the tile does not exist (404 = ocean, which is not an error) or if it stays
    unreachable. A 404 is distinguished from a transient failure so that missing
    ocean does not burn three retries each.
    """
    url = tile_url(tile)
    last = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return rasterio.open(url)
        except rasterio.errors.RasterioIOError as exc:
            if "404" in str(exc):
                return None                      # ocean / outside the land mask
            last = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BASE ** attempt)
        except Exception as exc:
            last = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BASE ** attempt)
    log.warning(f"  tile {tile}: unreachable after {_MAX_RETRIES} attempts: {last}")
    return "FAILED"


def process_region(row: pd.Series, overwrite: bool) -> str:
    """Mosaic and warp one region's DEM. Returns a status string."""
    rid = int(row["region_id"])
    log = logging.getLogger(__name__)

    out_dir  = TERRAIN_ROOT / f"region_{rid:04d}"
    out_path = out_dir / "dem_glo30_30m.tif"
    if out_path.exists() and not overwrite:
        return "skip"

    res       = float(row["res_m"])
    width_px  = int(row["width_px"])
    height_px = int(row["height_px"])
    proj4     = str(row["laea_proj4"])
    dst_crs   = CRS.from_proj4(proj4)
    # north-up: origin is the top-left corner, y decreasing
    dst_transform = Affine(res, 0.0, float(row["x_min"]),
                           0.0, -res, float(row["y_max"]))

    dst = np.full((height_px, width_px), np.nan, dtype=np.float32)

    tiles = [t for t in str(row["glo30_tiles"]).split(";") if t]
    used, missing, failed = [], [], []

    # Group sources by pixel size: rasterio.merge cannot mosaic grids of
    # different resolution, and poleward decimation guarantees that happens
    # wherever a region crosses 50/60/70/80 degrees.
    groups: dict[tuple, list] = {}
    try:
        for t in tiles:
            ds = open_tile(t, log)
            if ds is None:
                missing.append(t)
                continue
            if isinstance(ds, str):          # "FAILED" sentinel — transient, retry the region
                failed.append(t)
                continue
            key = (round(ds.res[0], 9), round(ds.res[1], 9))
            groups.setdefault(key, []).append(ds)
            used.append(t)

        if failed:
            return f"fail:tiles_unreachable:{','.join(failed)}"
        if not used:
            return "fail:no_land_tiles"

        for key, dss in groups.items():
            # nodata=nan is load-bearing: the COGs carry nodata=None, so without
            # it merge fills cells no tile covers with 0.0 — indistinguishable
            # from sea level, and bilinear would then warp a fake coastal plain
            # into the region.
            mosaic, src_transform = rio_merge(dss, nodata=np.nan,
                                              resampling=Resampling.bilinear)
            src = mosaic[0].astype(np.float32)
            src_crs = dss[0].crs

            # Warp only into the destination window this group can cover, so a
            # region with many source tiles does not pay for a full-grid warp per
            # group. Bounds are transformed densely (21 points per edge) because
            # the geographic rectangle maps to a curved shape in LAEA.
            w, s, e, n = rasterio.transform.array_bounds(
                src.shape[0], src.shape[1], src_transform)
            dw, ds_, de, dn = transform_bounds(src_crs, dst_crs, w, s, e, n, densify_pts=21)

            # Integer window by explicit floor/ceil, not Window.round_*():
            # rasterio's round_lengths() rounds to nearest, which can clip the
            # far edge by a cell and leave a one-pixel NaN stripe between two
            # source groups — a fake ditch running through the flow field.
            # One extra cell of pad gives bilinear its neighbours at the seam.
            x0 = float(row["x_min"])
            y1 = float(row["y_max"])
            c0 = int(np.floor((dw - x0) / res)) - 1
            c1 = int(np.ceil((de - x0) / res)) + 1
            r0 = int(np.floor((y1 - dn) / res)) - 1
            r1 = int(np.ceil((y1 - ds_) / res)) + 1
            c0, r0 = max(c0, 0), max(r0, 0)
            c1, r1 = min(c1, width_px), min(r1, height_px)
            if c1 <= c0 or r1 <= r0:
                continue
            win = Window(c0, r0, c1 - c0, r1 - r0)

            sub = np.full((int(win.height), int(win.width)), np.nan, dtype=np.float32)
            reproject(
                source=src,
                destination=sub,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=rasterio.windows.transform(win, dst_transform),
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                num_threads=2,
            )
            view = dst[int(win.row_off):int(win.row_off + win.height),
                       int(win.col_off):int(win.col_off + win.width)]
            np.copyto(view, sub, where=np.isnan(view) & ~np.isnan(sub))
    finally:
        for g in groups.values():
            for ds in g:
                ds.close()

    nan_frac = float(np.isnan(dst).mean())
    seam = len(groups) > 1

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(
            tmp, "w", driver="GTiff",
            height=height_px, width=width_px, count=1, dtype="float32",
            crs=dst_crs, transform=dst_transform, nodata=np.nan,
            compress="deflate", predictor=3, tiled=True,
            blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
        ) as dstds:
            dstds.write(dst, 1)
            dstds.update_tags(
                source="Copernicus GLO-30 DSM (AWS public COGs)",
                laea_proj4=proj4,
                res_m=f"{res:g}",
                buffer_km=str(row["buffer_km"]),
                linkage_km=str(row["linkage_km"]),
                n_stations=str(int(row["n_stations"])),
                tiles_used=";".join(sorted(used)),
                tiles_missing=";".join(sorted(missing)),
                nan_frac=f"{nan_frac:.6f}",
                seam_lat_band=str(seam),
                note="DSM: canopy is in the surface. Breach, do not fill (§32.4).",
            )
        tmp.rename(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    msg = f"done nan={nan_frac:.1%} tiles={len(used)}"
    if missing:
        msg += f" missing={len(missing)}"
    if seam:
        msg += f" SEAM({len(groups)} src grids)"
    log.info(f"  region {rid:04d}: {msg}")
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch wide DEM per region (GLO-30 -> LAEA 30 m)")
    ap.add_argument("--region-csv", type=Path, default=REGION_CSV)
    ap.add_argument("--terrain-root", type=Path, default=None)
    ap.add_argument("--region-id", type=int, action="append", default=None,
                    help="Process only these region ids (repeatable).")
    ap.add_argument("--max-stations", type=int, default=None,
                    help="Pilot mode: smallest set of regions covering N stations.")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.terrain_root is not None:
        global TERRAIN_ROOT
        TERRAIN_ROOT = args.terrain_root

    setup_logging()
    log = logging.getLogger(__name__)

    reg = pd.read_csv(args.region_csv)
    if args.region_id:
        reg = reg[reg["region_id"].isin(args.region_id)]
    elif args.max_stations:
        reg = reg.sort_values("n_stations")
        keep = reg["n_stations"].cumsum() <= args.max_stations
        reg = reg[keep]

    # largest first: the long pole starts immediately instead of last
    reg = reg.sort_values("n_cells", ascending=False).reset_index(drop=True)

    log.info(f"terrain root {TERRAIN_ROOT}")
    log.info(f"{len(reg)} regions, {int(reg['n_stations'].sum())} stations, "
             f"{reg['n_cells'].sum()/1e9:.2f}e9 cells, {reg['dem_gb'].sum():.1f} GB, "
             f"workers={args.workers}")

    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_region, row, args.overwrite): row
                for _, row in reg.iterrows()}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            rid = int(row["region_id"])
            try:
                status = fut.result()
            except Exception as exc:
                failed += 1
                log.error(f"  region {rid:04d} worker error: {type(exc).__name__}: {exc}")
                continue
            if status == "skip":
                skipped += 1
            elif status == "done":
                done += 1
            else:
                failed += 1
                log.error(f"  region {rid:04d}: {status}")
            if i % 25 == 0:
                log.info(f"[{i}/{len(reg)}] done={done} skip={skipped} fail={failed}")

    log.info(f"Finished. {done} written, {skipped} already present, {failed} failed.")


if __name__ == "__main__":
    main()
