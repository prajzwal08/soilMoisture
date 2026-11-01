"""
landsat_target.py — the 22x22 @ 100 m Landsat ST target grid, defined once
===========================================================================

§46 supervises the decoder with Landsat surface temperature at 100 m. Two pieces of code
need the same grid and they must not derive it separately:

  * `compute_lst_stats.py`, which fits `sigma_ST` before training
  * §46's dataset loader, which produces the per-sample target

If they disagree, `sigma_ST` is in different units from the residual it divides, and nothing
raises -- the thermal loss is simply mis-scaled.

THE GRID IS FORCED, NOT CHOSEN. The loss compares the pooled prediction against the warped
target, so the target footprint must equal the prediction footprint exactly:

    prediction   112x112 @ 20 m           = 2240 m, the model tile
    F.avg_pool2d(kernel_size=5, stride=5) consumes pixels 0..109 and emits 22x22
    => footprint  2200 m starting at the tile's WEST/NORTH edge, NOT centred on the station

    west  = cx - 1120        east  = cx - 1120 + 2200 = cx + 1080
    north = cy + 1120        south = cy + 1120 - 2200 = cy - 1080

The last 40 m of the tile (2 pixels of 20 m) is unsupervised on each axis. That falls out of
112 not dividing by 5; it is recorded here so nobody "fixes" it into a centred grid and
silently shifts every thermal cell by one 20 m pixel.

SOURCE SIDE. The st30 bundles are 76x76 @ 30 m (= 2280 m) on the Landsat grid, whose origin
is snapped to (15, 15) mod 30 (`download_landsat_st_mpc.py:70`). 2200 / 30 = 73.33 source
pixels, so the resampling is genuinely area-weighted: `Resampling.average`, never a strided
pool (§46 :8217).
"""

from __future__ import annotations

import math

import numpy as np

# These three are DUPLICATED from `download_landsat_st_mpc.py:69-70,97` rather than imported.
# That module imports `planetary_computer`, which exists only in the `soilmoisture` env; this
# one has to be importable from `terramind`, where training and this job run. Copying three
# lines is the lesser evil against making the training env depend on the download stack.
# If the Landsat grid ever moves, it moves in both places.
LS_RES_M       = 30
LS_GRID_OFFSET = 15      # verified: proj:transform origin = (15, 15) mod 30


def snap(v: float, res: int = LS_RES_M, off: int = LS_GRID_OFFSET, up: bool = False) -> float:
    """Snap a UTM coordinate onto the Landsat pixel grid (origin = off mod res)."""
    k = math.ceil((v - off) / res) if up else math.floor((v - off) / res)
    return off + res * k

# The model tile, from download_s2_mpc.py:57-58 / download_s1_lulc_mpc.py:82-83
TILE_PX, TILE_RES_M = 224, 10
TILE_M = TILE_PX * TILE_RES_M          # 2240
HALF_TILE_M = TILE_M / 2.0             # 1120

# The pooled target, from §46 :8208 -- avg_pool2d(k=5, s=5) on the 112x112 @ 20 m map
OUT_N, OUT_RES_M = 22, 100
OUT_M = OUT_N * OUT_RES_M              # 2200

# The st30 bundle grid, from download_landsat_st30.py:105
SRC_N = 76


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def target_transform(cx: float, cy: float):
    """Affine for the 22x22 @ 100 m target, anchored to the MODEL TILE's west/north edge.

    Returns (transform, bounds) with bounds as (west, south, east, north).
    """
    from rasterio.transform import from_origin
    west, north = cx - HALF_TILE_M, cy + HALF_TILE_M
    return (from_origin(west, north, OUT_RES_M, OUT_RES_M),
            (west, north - OUT_M, west + OUT_M, north))


def source_transform(cx: float, cy: float):
    """Affine for the 76x76 @ 30 m st30 cube, reproducing `download_landsat_st_mpc.station_grid`.

    The Landsat grid snaps its origin to (15, 15) mod 30; the model tile does not snap at all.
    The two grids are therefore offset by up to 15 m, which is precisely why the target is
    produced by a warp and not by slicing.
    """
    from rasterio.transform import from_origin
    half = SRC_N * LS_RES_M / 2.0
    west = snap(cx - half)
    south = snap(cy - half)
    north = south + SRC_N * LS_RES_M
    return from_origin(west, north, LS_RES_M, LS_RES_M)


def to_target(lst30: np.ndarray, cx: float, cy: float, epsg: int) -> np.ndarray:
    """Warp one or more 76x76 @ 30 m Kelvin fields onto the 22x22 @ 100 m target grid.

    `lst30` is (76, 76) or (N, 76, 76), float32 Kelvin with NaN where there is no retrieval.
    NaN is carried through as NaN: `Resampling.average` over a window that is entirely NaN
    yields NaN, and a partially valid window yields the average of what is there. That is the
    behaviour we want -- §37 measured LST absent over ~45% of pixels QC calls clear, so a
    target cell built from half a window is still a real measurement of that cell.
    """
    from rasterio.warp import reproject, Resampling

    single = lst30.ndim == 2
    cube = lst30[None] if single else lst30
    src_t = source_transform(cx, cy)
    dst_t, _ = target_transform(cx, cy)
    crs = f"EPSG:{epsg}"

    out = np.full((cube.shape[0], OUT_N, OUT_N), np.nan, dtype=np.float32)
    for i, plane in enumerate(cube):
        src = np.ascontiguousarray(plane, dtype=np.float32)
        reproject(
            source=src, destination=out[i],
            src_transform=src_t, src_crs=crs,
            dst_transform=dst_t, dst_crs=crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out[0] if single else out


def centre(field: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the scene's own mean over valid cells -> (centred field, the mean removed).

    This is the split §46 relies on: the centred field is the PATTERN term the thermal head
    is trained on, and the mean is the LEVEL term, reported but not trained (`alpha = 0`).
    """
    m = np.nanmean(field)
    if not np.isfinite(m):
        return np.full_like(field, np.nan), np.nan
    return field - m, float(m)
