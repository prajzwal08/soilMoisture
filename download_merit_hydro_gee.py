"""
Download MERIT Hydro (MERIT/Hydro/v1_0_1) windows for every station — the reference
side of §32.6's MERIT gate.

MERIT Hydro is the necessary condition on our own terrain, not a nice-to-have:
no station's TWI/HAND may be consumed by the sufficiency gate or by dataset.py
until its upslope area has been checked against MERIT's. MERIT Hydro is 90 m
(3 arcsec) and carries its own errors, so the gate is on the MAGNITUDE of upslope
area — the non-local quantity the region build exists to capture — not on sub-90 m
structure.

Bands fetched (all 5 needed downstream, none redundant):
  upa   upstream drainage area          km^2   <- the gate's reference quantity
  upg   upstream pixel count            cells  <- fetched so the km^2 assertion is
                                                  reproducible offline, not just
                                                  in the session that measured it
  hnd   height above nearest drainage   m      <- calibrates our stream threshold
  elv   adjusted elevation              m
  dir   flow direction                  code

  upa IS IN km^2, NOT m^2 — measured, not assumed: upa/upg matches the 3-arcsec
  cell area in km^2 to four decimals across 30-52 N, and the ratio tracks cell
  area with latitude, which is what makes it decisive. That is a factor of 1e6
  inside a log if taken as m^2. This script re-derives the ratio per station and
  writes it to the log so the unit can never be silently wrong again.

Window: 25 km per side, snapped to the MERIT 1/1200-degree grid so the returned
array aligns exactly with the source cells and the GeoTIFF transform is exact
rather than inferred. ~271 x 436 cells at 52 N = 118k, under the sampleRectangle
request cap; a larger window would need getDownloadURL/computePixels instead.

The comparison this feeds is over the 2.24 km footprint, never at the station
point: four probe points placed on named large rivers returned upg of 1-8, i.e.
hillslope cells, because at 90 m a couple of hundred metres of georeferencing
error drops upa by orders of magnitude (§32.6).

Auth: earthengine authenticate --auth_mode=notebook. The default auth mode shells
out to gcloud, which is not installed on the Snellius login node.

Output per station:
  {station_dir}/MERIT/merit_hydro_25km.tif   float32, 5 bands, EPSG:4326
  logs/merit_hydro_log.csv                   per-station shape, unit ratio, coverage

Usage:
  conda activate soilmoisture
  python download_merit_hydro_gee.py
  python download_merit_hydro_gee.py --station ISMN_TWENTE_Hupsel --verify-geo
"""

import argparse
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.pop("PROJ_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from google.oauth2.credentials import Credentials

GEE_PROJECT       = "1066500857818"
_CREDENTIALS_FILE = Path.home() / ".config/earthengine/credentials"

REPO         = Path(__file__).resolve().parent
STATION_CSV  = REPO / "csvs" / "station_splits.csv"
DATA_ROOT    = Path("/gpfs/work3/0/prjs1968/data")
LOG_DIR      = DATA_ROOT / "logs"
LOG_CSV      = LOG_DIR / "merit_hydro_log.csv"

COLLECTION   = "MERIT/Hydro/v1_0_1"
BANDS        = ["upa", "upg", "hnd", "elv", "dir"]
BAND_UNITS   = {"upa": "km2", "upg": "cells", "hnd": "m", "elv": "m", "dir": "code"}
MERIT_RES_DEG = 1.0 / 1200.0        # 3 arcsec
WINDOW_M      = 25_000.0            # per side
NODATA        = -9999.0

N_WORKERS       = 8                 # remote API — not Pool(64)
GEE_RETRY_WAITS = [2, 4, 8]

# Errors that will never succeed on retry
_GEE_NO_RETRY = ("403", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
                 "Invalid credentials", "not found",
                 "incompatible with band")   # a type error, not a transient one

R_EARTH_KM = 6371.0072              # authalic radius, for the cell-area assertion

_ee_lock = threading.Lock()
_ee_ready = False
_log_lock = threading.Lock()

# Filled from the asset's own crs_transform at init, never assumed. Measured
# 2026-08-24: origin (-180.000416666667, 84.999583333333), scale 1/1200 — so cell
# CENTRES lie on exact multiples of 1/1200 and the edges are half a cell off from
# them. Snapping a window to multiples of 1/1200 (the obvious thing) therefore
# lands on centres, putting the GeoTIFF transform half a cell (46 m in latitude)
# out and biasing every footprint comparison in §32.6 by the same amount.
_GRID: dict[str, float] = {}


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(LOG_DIR / "merit_hydro_gee.log")],
    )


def _gee_credentials() -> Credentials:
    with open(_CREDENTIALS_FILE) as f:
        d = json.load(f)
    return Credentials(
        token=None,
        refresh_token=d["refresh_token"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )


def ee_init():
    """Initialise Earth Engine once, lazily, under a lock (threads share it)."""
    global _ee_ready
    import ee
    log = logging.getLogger(__name__)
    with _ee_lock:
        if not _ee_ready:
            ee.Initialize(credentials=_gee_credentials(), project=GEE_PROJECT)
            p = ee.Image(COLLECTION).select("upa").projection().getInfo()
            t = p["transform"]
            _GRID.update(x0=float(t[2]), y0=float(t[5]),
                         sx=float(t[0]), sy=float(abs(t[4])))
            if abs(_GRID["sx"] - MERIT_RES_DEG) > 1e-12:
                raise RuntimeError(f"MERIT scale {_GRID['sx']} != expected {MERIT_RES_DEG}")
            _ee_ready = True
            log.info(f"EE initialised (project {GEE_PROJECT}); grid origin "
                     f"({_GRID['x0']:.12f}, {_GRID['y0']:.12f}) scale {_GRID['sx']:.12f}")
    return ee


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
    return df[["station_id", "latitude", "longitude", "station_dir"]].reset_index(drop=True)


def snapped_bounds(lat: float, lon: float) -> tuple[float, float, float, float, int, int]:
    """
    A 25 km window in metres, converted to degrees at this latitude, then snapped
    outward onto MERIT's own cell EDGES, taken from the asset's crs_transform.

    Snapping matters twice over. sampleRectangle returns whole source cells
    covering the region, so an unsnapped rectangle makes the returned array size
    depend on sub-cell placement and the GeoTIFF transform becomes a guess. And
    the grid is centre-registered on multiples of 1/1200, so snapping to those
    multiples snaps to centres — half a cell wrong. Edges are x0 + i*sx.
    """
    if not _GRID:
        raise RuntimeError("call ee_init() before snapped_bounds()")
    x0, y0, k = _GRID["x0"], _GRID["y0"], _GRID["sx"]

    half = WINDOW_M / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))

    i_w = math.floor((lon - dlon - x0) / k)
    i_e = math.ceil((lon + dlon - x0) / k)
    j_n = math.floor((y0 - (lat + dlat)) / k)
    j_s = math.ceil((y0 - (lat - dlat)) / k)

    west  = x0 + i_w * k
    east  = x0 + i_e * k
    north = y0 - j_n * k
    south = y0 - j_s * k
    return west, south, east, north, i_e - i_w, j_s - j_n


def merit_cell_area_km2(lat: float) -> float:
    """Analytic 3-arcsec cell area at the row centre — the upa unit assertion."""
    d = math.radians(MERIT_RES_DEG)
    return (R_EARTH_KM * d) * (R_EARTH_KM * d * math.cos(math.radians(lat)))


def _with_retry(fn, what: str, log: logging.Logger):
    """Call a GEE-backed function with backoff; give up immediately on hard errors."""
    last = None
    for i, wait in enumerate([0] + GEE_RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            return fn()
        except Exception as exc:
            last = exc
            msg = str(exc)
            if any(p in msg for p in _GEE_NO_RETRY):
                log.error(f"  {what}: unrecoverable: {msg[:160]}")
                raise
            if i < len(GEE_RETRY_WAITS):
                log.warning(f"  {what}: attempt {i+1} failed ({msg[:110]}) — retrying")
    raise last


def sample_rect(ee, image, west, south, east, north, bands, default) -> dict:
    # toFloat() before the default: `dir` is an integer band, and EE rejects a
    # defaultValue outside the band's type range rather than promoting it.
    rect = ee.Geometry.Rectangle([west, south, east, north], "EPSG:4326", False)
    return image.select(bands).toFloat().sampleRectangle(
        region=rect, defaultValue=default).getInfo()["properties"]


def process_station(row: pd.Series, overwrite: bool, verify_geo: bool) -> dict:
    """Fetch one station's MERIT window. Returns a log record."""
    sid = row["station_id"]
    lat, lon = float(row["latitude"]), float(row["longitude"])
    log = logging.getLogger(__name__)

    out_dir  = Path(row["station_dir"]) / "MERIT"
    out_path = out_dir / "merit_hydro_25km.tif"
    if out_path.exists() and not overwrite:
        return {"station_id": sid, "status": "skip"}

    ee = ee_init()
    west, south, east, north, n_lon, n_lat = snapped_bounds(lat, lon)
    img = ee.Image(COLLECTION)

    props = _with_retry(
        lambda: sample_rect(ee, img, west, south, east, north, BANDS, NODATA),
        f"{sid} sampleRectangle", log)

    arrs = []
    for b in BANDS:
        a = np.asarray(props[b], dtype=np.float32)   # rows = north->south
        arrs.append(a)
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        return {"station_id": sid, "status": f"fail:band_shape_mismatch:{shapes}"}
    h, w = arrs[0].shape

    # Expected from the snapped grid. A one-cell difference means the returned
    # window is offset from the grid we assumed, which would put the transform
    # half a cell out — treat it as a failure rather than writing a wrong geo.
    if abs(h - n_lat) > 1 or abs(w - n_lon) > 1:
        return {"station_id": sid,
                "status": f"fail:shape {h}x{w} vs expected {n_lat}x{n_lon}"}

    stack = np.stack(arrs, axis=0)
    stack[stack == NODATA] = np.nan

    # Bounds implied by the ACTUAL returned size, anchored at the snapped NW
    # corner: if EE trimmed a row at the antimeridian or a tile edge, the
    # transform still describes the array that was returned.
    east_a  = west + w * MERIT_RES_DEG
    south_a = north - h * MERIT_RES_DEG
    transform = from_bounds(west, south_a, east_a, north, w, h)

    geo_err_m = float("nan")
    if verify_geo:
        ll = _with_retry(
            lambda: sample_rect(ee, ee.Image.pixelLonLat().reproject(
                img.select("upa").projection()),
                west, south, east, north, ["longitude", "latitude"], NODATA),
            f"{sid} pixelLonLat", log)
        plon = np.asarray(ll["longitude"], dtype=np.float64)
        plat = np.asarray(ll["latitude"], dtype=np.float64)
        if plon.shape == (h, w):
            exp_lon = west + (np.arange(w) + 0.5) * MERIT_RES_DEG
            exp_lat = north - (np.arange(h) + 0.5) * MERIT_RES_DEG
            dx = np.abs(plon[0, :] - exp_lon).max() * 111_320.0 * math.cos(math.radians(lat))
            dy = np.abs(plat[:, 0] - exp_lat).max() * 111_320.0
            geo_err_m = float(max(dx, dy))

    # ── the unit assertion, per station ───────────────────────────────────────
    upa, upg = stack[BANDS.index("upa")], stack[BANDS.index("upg")]
    ok = np.isfinite(upa) & np.isfinite(upg) & (upg > 0)
    ratio = float(np.median(upa[ok] / upg[ok])) if ok.any() else float("nan")
    expect = merit_cell_area_km2(lat)
    unit_ratio = ratio / expect if expect > 0 else float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(
            tmp, "w", driver="GTiff", height=h, width=w, count=len(BANDS),
            dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
            compress="deflate", predictor=3, tiled=True,
            blockxsize=256, blockysize=256,
        ) as dst:
            dst.write(stack)
            for i, b in enumerate(BANDS, start=1):
                dst.set_band_description(i, b)
                dst.update_tags(i, name=b, units=BAND_UNITS[b])
            dst.update_tags(
                source=COLLECTION, res_deg=f"{MERIT_RES_DEG:.10f}",
                window_m=f"{WINDOW_M:g}", station=sid,
                station_lat=f"{lat:.6f}", station_lon=f"{lon:.6f}",
                upa_units="km2",
                upa_per_upg_km2=f"{ratio:.8f}",
                cell_area_km2_analytic=f"{expect:.8f}",
                unit_ratio_measured_over_analytic=f"{unit_ratio:.5f}",
                geo_err_m=f"{geo_err_m:.3f}",
                note="Compare over the footprint, never at the station point (§32.6).",
            )
        tmp.rename(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return {
        "station_id": sid, "status": "done",
        "height": h, "width": w,
        "west": west, "south": south_a, "east": east_a, "north": north,
        "nan_frac": float(np.isnan(stack).mean()),
        "upa_max_km2": float(np.nanmax(upa)) if np.isfinite(upa).any() else np.nan,
        "upg_max": float(np.nanmax(upg)) if np.isfinite(upg).any() else np.nan,
        "upa_per_upg_km2": ratio,
        "cell_area_km2_analytic": expect,
        "unit_ratio": unit_ratio,
        "hnd_zero_frac": float(np.nanmean(stack[BANDS.index("hnd")] <= 0.0)),
        "geo_err_m": geo_err_m,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download MERIT Hydro windows per station")
    ap.add_argument("--station", action="append", default=None,
                    help="Process only these station_ids (repeatable).")
    ap.add_argument("--max-stations", type=int, default=None)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-geo", action="store_true",
                    help="Second call per station sampling pixelLonLat, to prove the "
                         "transform. Doubles the request count; default is the first "
                         "--n-verify stations only.")
    ap.add_argument("--n-verify", type=int, default=20,
                    help="Verify the geo transform for this many stations (0 = none).")
    ap.add_argument("--out-csv", type=Path, default=LOG_CSV)
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    st = load_stations()
    if args.station:
        st = st[st["station_id"].isin(args.station)].reset_index(drop=True)
    if args.max_stations:
        st = st.head(args.max_stations).reset_index(drop=True)

    log.info(f"{len(st)} stations, window {WINDOW_M/1000:g} km, bands {BANDS}, "
             f"workers={args.workers}")

    records, n_done, n_fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for i, (_, row) in enumerate(st.iterrows()):
            vg = args.verify_geo or i < args.n_verify
            futs[pool.submit(process_station, row, args.overwrite, vg)] = row

        for k, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                n_fail += 1
                log.error(f"  {row['station_id']}: {type(exc).__name__}: {str(exc)[:160]}")
                records.append({"station_id": row["station_id"],
                                "status": f"error:{type(exc).__name__}"})
                continue
            records.append(rec)
            if rec["status"] == "done":
                n_done += 1
                if abs(rec["unit_ratio"] - 1.0) > 0.02:
                    log.warning(f"  {rec['station_id']}: upa/upg = "
                                f"{rec['upa_per_upg_km2']:.6f} km2/cell vs analytic "
                                f"{rec['cell_area_km2_analytic']:.6f} "
                                f"(ratio {rec['unit_ratio']:.3f}) — upa unit suspect")
            elif rec["status"].startswith("fail"):
                n_fail += 1
                log.error(f"  {rec['station_id']}: {rec['status']}")
            if k % 25 == 0:
                log.info(f"[{k}/{len(st)}] done={n_done} fail={n_fail}")

    df = pd.DataFrame(records)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_csv.exists() and not args.overwrite:
        old = pd.read_csv(args.out_csv)
        df = pd.concat([old[~old["station_id"].isin(df["station_id"])], df],
                       ignore_index=True)
    df.to_csv(args.out_csv, index=False)

    ok = df[df["status"] == "done"] if "status" in df else df
    if len(ok):
        log.info(f"unit_ratio (upa/upg over analytic cell area): "
                 f"median {ok['unit_ratio'].median():.5f}  "
                 f"min {ok['unit_ratio'].min():.5f}  max {ok['unit_ratio'].max():.5f}  "
                 f"-> upa is in km2" if abs(ok['unit_ratio'].median() - 1) < 0.02
                 else f"unit_ratio median {ok['unit_ratio'].median():.5f} — NOT km2, STOP")
        if "geo_err_m" in ok:
            g = ok["geo_err_m"].dropna()
            if len(g):
                log.info(f"geo transform error: max {g.max():.2f} m over {len(g)} verified")
    log.info(f"Finished. {n_done} written, {n_fail} failed. Log: {args.out_csv}")


if __name__ == "__main__":
    main()
