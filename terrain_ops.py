"""
Terrain derivation primitives for §32.4 — TWI and HAND from a conditioned DEM.

Kept separate from build_twi_hand.py so the Tier-1 synthetic tests (§32.5) exercise
exactly the functions the region driver calls, rather than a reimplementation of them.

The four choices here that are not the obvious ones, each with its reason:

1. BREACH, do not fill. Filling raises a pit until it spills over its rim; breaching
   carves down through the obstruction. Our obstructions are canopy — GLO-30 is a DSM,
   TanDEM-X X-band scatters near canopy top — so a 20 m tree line across a valley is a
   20 m dam. Filling floods the valley into a fake lake and routes flow around the
   hill; breaching cuts a notch, which is what the real stream does under the canopy.

2. MFD for accumulation, D8 for the HAND trace. D8 sends all water to the steepest of
   eight neighbours and stripes smooth hillslopes; MFD (Quinn 1991, slope^p, p=1.1)
   splits among all downslope neighbours and is more physical for `a`. But under MFD
   there is no single downstream path and HAND is *defined* by following one, so a
   second D8-only field exists purely for tracing.

3. Slope from the RAW DEM. Conditioning deliberately flattens things, so taking slope
   off the conditioned surface puts artificial zeros exactly in the valleys where TWI
   matters most. Route on conditioned, measure slope on raw.

4. `a` is area per unit contour width, so a = A_cells * cellsize, in METRES not m^2.
   Ridge 30 m, valley bottom thousands — that spread is §31.4's dln(a) ~ 2.8.

pyflwdir gives D8 (its from_dem does its own internal filling, which is why the
conditioned DEM goes in). MFD accumulation comes from WhiteboxTools FD8FlowAccumulation
because pyflwdir is a single-flow-direction library. The two codebases share no
lineage, which is what makes the Tier-3 cross-check meaningful.

Environment: terramind (pyflwdir needs numba, which soilmoisture does not have).
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

# Whitebox refuses NaN in places and its nodata handling is inconsistent across
# tools, so a sentinel is used for every file that crosses the process boundary.
WBT_NODATA = -32768.0

TAN_SLOPE_FLOOR = 1e-3     # §32.4: floor tan(beta), and report the floored fraction
MFD_EXPONENT    = 1.1      # Quinn 1991 slope^p
STREAM_HA       = 10.0     # 10 ha = 111 cells at 30 m; calibrated against MERIT hnd~0
# Max least-cost breach search distance, in cells. §32.4 says breach aggressively,
# but 'aggressive' means willing to cut, not willing to cut far: the obstructions are
# canopy, and a tree line across a valley is 1-3 cells wide at 30 m. The search is
# roughly O(dist^2) per pit, and on coastal regions GLO-30's flat sea surface presents
# huge numbers of pits — dist=100 (a 3 km breach path) had not finished conditioning a
# single 742x742 region after 3 minutes. 20 cells = 600 m is far beyond any canopy dam.
BREACH_DIST_CELLS = 20

# WhiteboxTools refuses a GeoTIFF with no geokeys ("does not contain geokeys"), so
# every file handed to it carries a CRS even when the grid is synthetic. Equal-area
# with metre units, matching what the real regions use.
SYNTHETIC_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

_wbt = None


def wbt():
    """One WhiteboxTools instance per process, quiet."""
    global _wbt
    if _wbt is None:
        import whitebox
        _wbt = whitebox.WhiteboxTools()
        _wbt.verbose = False
    return _wbt


def run_wbt(tool: str, out_path: Path, timeout_s: float = 3600.0,
            attempts: int = 3, **kw) -> None:
    """
    Run a WhiteboxTools tool by invoking its binary directly, and insist it succeeded.

    The whitebox python wrapper is bypassed deliberately. Its run_tool():
      - returns 0 unconditionally, so a Rust panic ('The TIFF file does not contain
        geokeys') is reported as success and the failure surfaces much later as a
        confusing 'No such file' from rasterio — or worse, as a stale file from a
        previous run being read as the current result;
      - discards the tool's own stdout entirely when verbose is False, which is where
        the panic message goes, so there is nothing to diagnose from;
      - chdir()s the whole process into the executable's directory for the duration
        of the call, which is not safe alongside anything else using relative paths.

    Calling the binary through subprocess.run gives the real exit code, the real
    error text, and a real wait for the process to finish writing.
    """
    w = wbt()
    exe = Path(w.exe_path) / w.exe_name
    args = [str(exe), f"--run={tool}"]
    for k, v in kw.items():
        if v is None or v is False:
            continue
        args.append(f"--{k}" if v is True else f"--{k}={v}")
    args += ["-v=false", "--compress_rasters=False"]

    # The binary panics intermittently on identical input (observed on
    # FillDepressions: 'Error unwrapping output', rc=101, ~1 call in 6), so a bounded
    # retry sits under the hard failure rather than in place of it.
    last = ""
    for attempt in range(1, attempts + 1):
        out_path.unlink(missing_ok=True)
        proc = subprocess.run(args, cwd=str(w.exe_path), capture_output=True,
                              text=True, timeout=timeout_s)
        if proc.returncode == 0 and out_path.exists():
            try:
                with rasterio.open(out_path) as src:
                    _ = src.width
                if attempt > 1:
                    logging.getLogger(__name__).warning(
                        f"  WhiteboxTools {tool} succeeded on attempt {attempt}")
                return
            except Exception as exc:
                last = f"wrote an unreadable raster: {exc}"
        else:
            last = (f"rc={proc.returncode}, output "
                    f"{'written' if out_path.exists() else 'MISSING'}: "
                    + ((proc.stdout or "")[-600:] + (proc.stderr or "")[-600:]).strip())

    raise RuntimeError(f"WhiteboxTools {tool} failed after {attempts} attempts — {last}")


def interior_sinks(dem: np.ndarray, split: bool = False):
    """
    Count interior cells with no strictly-lower 8-neighbour, computed directly rather
    than asked of pyflwdir — pyflwdir.from_dem does its own internal filling, so its
    pit list cannot tell you whether the DEM you handed it had depressions (§32.4's
    API caveat).

    With split=True, returns (pits, flats) separately. The distinction matters on
    real terrain: a PIT has every neighbour strictly higher and is a genuine
    depression that conditioning must remove, while a FLAT merely has an equal
    neighbour and is what a lake, a sea surface, or a plateau looks like in a DSM.
    Conditioning is not obliged to eliminate flats, so counting them as failures
    makes the Tier-2 check fire on correct output.

    Boundary cells are excluded: on any tilted surface they legitimately drain off
    the grid.
    """
    z = np.where(np.isfinite(dem), dem, np.inf).astype(np.float64)
    lower = np.zeros(z.shape, dtype=bool)
    equal = np.zeros(z.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = np.roll(np.roll(z, dy, axis=0), dx, axis=1)
            lower |= nb < z
            equal |= nb == z
    finite = np.isfinite(dem)[1:-1, 1:-1]
    no_lower = finite & ~lower[1:-1, 1:-1]
    if not split:
        return int(no_lower.sum())
    flats = no_lower & equal[1:-1, 1:-1]
    return int((no_lower & ~flats).sum()), int(flats.sum())


# ─────────────────────────────────────────────────────────────────────────────
# raster I/O across the WhiteboxTools process boundary
# ─────────────────────────────────────────────────────────────────────────────

def write_wbt_tif(path: Path, arr: np.ndarray, res: float, crs=None,
                  origin: tuple[float, float] = (0.0, 0.0)) -> None:
    """Write a float32 GeoTIFF with a sentinel nodata that whitebox understands."""
    a = np.where(np.isfinite(arr), arr, WBT_NODATA).astype(np.float32)
    transform = Affine(res, 0.0, origin[0], 0.0, -res, origin[1])
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                       count=1, dtype="float32",
                       crs=crs if crs is not None else SYNTHETIC_CRS,
                       transform=transform,
                       nodata=WBT_NODATA, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER") as dst:
        dst.write(a, 1)


def read_wbt_tif(path: Path) -> np.ndarray:
    """Read a whitebox output back, sentinel -> NaN."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        a[a == nd] = np.nan
    a[a <= WBT_NODATA + 1.0] = np.nan     # some tools re-emit the sentinel unregistered
    return a


# ─────────────────────────────────────────────────────────────────────────────
# conditioning
# ─────────────────────────────────────────────────────────────────────────────

def condition_dem(dem: np.ndarray, res: float, workdir: Path,
                  breach_dist: int = BREACH_DIST_CELLS,
                  crs=None, origin=(0.0, 0.0),
                  return_stats: bool = False):
    """
    Breach depressions by least cost; fill residuals only if any actually remain.

    BreachDepressionsLeastCost with fill=True already fills what it cannot breach
    within `dist`, so a second unconditional FillDepressions pass is usually a no-op.
    It is made conditional on a measured sink count for two reasons. It is a real
    verification — interior_sinks() counts cells with no lower neighbour directly,
    so the no-sinks guarantee is checked rather than assumed. And FillDepressions
    panics intermittently ('Error unwrapping output', rc=101, roughly one call in
    six on identical input), so keeping it off the common path removes a flake from
    all 353 regions instead of retrying it on all of them.

    With return_stats, also returns dict(sinks_raw, sinks_after_breach, sinks_final,
    filled, carved_max_m, raised_max_m, touched_frac) for the region log.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    f_in  = workdir / "dem_raw.tif"
    f_br  = workdir / "dem_breached.tif"
    f_fill = workdir / "dem_filled.tif"

    write_wbt_tif(f_in, dem, res, crs=crs, origin=origin)
    run_wbt("BreachDepressionsLeastCost", f_br,
            dem=str(f_in), output=str(f_br), dist=breach_dist, min_dist=True, fill=True)
    out = read_wbt_tif(f_br)

    # only genuine PITS force the fill pass; flats (sea surface, lakes, plateaus)
    # are not something conditioning is obliged to remove
    sinks_raw, flats_raw = interior_sinks(dem, split=True)
    sinks_br, flats_br = interior_sinks(out, split=True)
    filled = False
    if sinks_br > 0:
        run_wbt("FillDepressions", f_fill,
                dem=str(f_br), output=str(f_fill), fix_flats=True)
        out = read_wbt_tif(f_fill)
        filled = True

    # conditioning must not invent data where there was none
    out[~np.isfinite(dem)] = np.nan

    if not return_stats:
        return out

    diff = dem - out
    pits_final, flats_final = interior_sinks(out, split=True)
    stats = {
        "sinks_raw": sinks_raw,
        "flats_raw": flats_raw,
        "sinks_after_breach": sinks_br,
        "sinks_final": pits_final,
        "flats_final": flats_final,
        "filled": filled,
        "carved_max_m": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
        "raised_max_m": float(np.nanmax(-diff)) if np.isfinite(diff).any() else 0.0,
        "touched_frac": float(np.nanmean(np.abs(diff) > 0.01)),
    }
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# slope
# ─────────────────────────────────────────────────────────────────────────────

def horn_slope(dem: np.ndarray, res: float) -> np.ndarray:
    """
    Horn (1981) 3x3 slope in radians: beta = atan(sqrt((dz/dx)^2 + (dz/dy)^2)),
    with 8*res denominators. Edges are replicated so the result has no border of
    spurious zeros; the region buffer means station tiles never sit on an edge.
    """
    z = np.pad(dem.astype(np.float64), 1, mode="edge")
    # a b c / d e f / g h i, rows increasing southward
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f    = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2],  z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * res)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * res)
    beta = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    return np.where(np.isfinite(dem), beta, np.nan).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# flow accumulation
# ─────────────────────────────────────────────────────────────────────────────

def flow_accum_mfd(dem_cond: np.ndarray, res: float, workdir: Path,
                   exponent: float = MFD_EXPONENT, crs=None, origin=(0.0, 0.0)
                   ) -> np.ndarray:
    """
    MFD (FD8) accumulation in CELLS, from WhiteboxTools.

    out_type='cells' rather than 'specific contributing area': the cell count is the
    quantity the mass-conservation assertion is stated in, and converting afterwards
    keeps one convention. A factor of res^2 is a constant 6.8 inside a log — harmless
    after standardisation, poisonous when mixed between regions or against MERIT.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    f_in  = workdir / "dem_for_fd8.tif"
    f_out = workdir / "acc_fd8.tif"
    write_wbt_tif(f_in, dem_cond, res, crs=crs, origin=origin)
    # note: FD8FlowAccumulation's input flag is --dem, D8FlowAccumulation's is --input
    run_wbt("FD8FlowAccumulation", f_out,
            dem=str(f_in), output=str(f_out), out_type="cells", exponent=exponent)
    return read_wbt_tif(f_out)


def flow_accum_d8_wbt(dem_cond: np.ndarray, res: float, workdir: Path,
                      crs=None, origin=(0.0, 0.0)) -> np.ndarray:
    """D8 accumulation in cells from WhiteboxTools — the Tier-3 second opinion."""
    workdir.mkdir(parents=True, exist_ok=True)
    f_in  = workdir / "dem_for_d8.tif"
    f_out = workdir / "acc_d8_wbt.tif"
    write_wbt_tif(f_in, dem_cond, res, crs=crs, origin=origin)
    run_wbt("D8FlowAccumulation", f_out,
            input=str(f_in), output=str(f_out), out_type="cells")
    return read_wbt_tif(f_out)


def d8_network(dem_cond: np.ndarray, res: float):
    """
    pyflwdir D8 network on the conditioned DEM — used for the HAND trace and for the
    exact mass-conservation test (D8 has single-path flow, so outlet sums are exact).
    """
    import pyflwdir
    z = np.where(np.isfinite(dem_cond), dem_cond, -9999.0).astype(np.float32)
    return pyflwdir.from_dem(z, nodata=-9999.0,
                             transform=Affine(res, 0.0, 0.0, 0.0, -res, 0.0),
                             latlon=False)


# ─────────────────────────────────────────────────────────────────────────────
# TWI / HAND
# ─────────────────────────────────────────────────────────────────────────────

def sca_from_cells(acc_cells: np.ndarray, res: float) -> np.ndarray:
    """
    Specific catchment area: a = A / contour width = (cells * res^2) / res = cells * res.
    Units of LENGTH, metres. Ridge 30 m, valley bottom thousands.
    """
    return (acc_cells * res).astype(np.float32)


def twi_from(acc_cells: np.ndarray, beta: np.ndarray, res: float,
             floor: float = TAN_SLOPE_FLOOR) -> tuple[np.ndarray, float]:
    """
    TWI = ln(a / tan beta), with tan beta floored. Returns (twi, floored_fraction) —
    a large floored fraction means TWI is degenerate there, not merely clipped, which
    is a different statement from 'it was clipped' and has to be reported as such.
    """
    a = sca_from_cells(acc_cells, res)
    tanb = np.tan(beta)
    valid = np.isfinite(a) & np.isfinite(tanb)
    floored = float(np.mean((tanb < floor)[valid])) if valid.any() else float("nan")
    twi = np.log(a / np.clip(tanb, floor, None))
    return np.where(valid, twi, np.nan).astype(np.float32), floored


def stream_mask(acc_cells: np.ndarray, res: float, stream_ha: float = STREAM_HA
                ) -> np.ndarray:
    """Cells whose contributing area exceeds the threshold. 10 ha = 111 cells at 30 m."""
    thresh_cells = stream_ha * 1e4 / (res * res)
    return (np.nan_to_num(acc_cells, nan=0.0) > thresh_cells)


def hand_from(flw, dem: np.ndarray, streams: np.ndarray) -> np.ndarray:
    """
    HAND = elevation - elevation of the first stream cell on the D8 trace, zero on
    streams.

    Pass the CONDITIONED DEM here, not the raw one. §32.4 asks for both
    `elevtn=dem_raw` and 'HAND >= 0 everywhere', and those two requirements are
    mutually inconsistent: breaching carves a notch through an obstruction, so on
    the raw surface a cell behind that obstruction sits BELOW the raw elevation of
    the stream cell its flow path reaches, and HAND comes out negative there by
    construction. Measured on the first two real regions: carving reached 65 m and
    152 m, and raw-surface HAND reached -31.5 m and -84.9 m over 0.7% and 1.3% of
    cells. That is not a conditioning bug, it is the definition colliding with
    itself.

    The conditioned surface is the one flow was routed on, so HAND measured on it is
    >= 0 by construction and non-increasing downstream, which is what makes the
    Tier-2 assertions meaningful rather than vacuous. Slope still comes from the raw
    DEM (§32.4 point 5), because there the raw surface is the honest one:
    conditioning deliberately flattens valleys.
    """
    z = np.where(np.isfinite(dem), dem, -9999.0).astype(np.float32)
    hand = flw.hand(drain=streams, elevtn=z).astype(np.float32)
    hand[~np.isfinite(dem)] = np.nan
    return hand


# ─────────────────────────────────────────────────────────────────────────────
# exact checks used by both the tests and the region driver
# ─────────────────────────────────────────────────────────────────────────────

def mass_conservation_d8(flw, n_valid: int) -> tuple[float, float]:
    """
    The only exact test in the pipeline: under D8 every cell's water leaves through
    exactly one pit, so accumulation summed over all pits equals the valid cell
    count, to the integer. Returns (outlet_sum, n_valid).
    """
    acc = flw.upstream_area(unit="cell")
    flat = acc.ravel()
    return float(flat[flw.idxs_pit].sum()), float(n_valid)


def scratch_dir(prefix: str = "terrain_") -> Path:
    """
    Whitebox works on files, so every region needs a private scratch directory.
    Node-local storage if SLURM gave us one — a region can move several GB through
    here and shared scratch is slower and noisier.
    """
    base = os.environ.get("TMPDIR") or "/tmp"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
