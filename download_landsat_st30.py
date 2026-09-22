#!/usr/bin/env python
"""The 993-station Landsat C2 L2 Surface Temperature pull -- native 30 m, RAW, no QC baked in.

§41 made Landsat ST the dense spatial supervision target for the second head.  This is §41.7
step 1, amended: the target is stored at the product's native 30 m, NOT pooled to 22x22 @ 100 m.

WHY RAW 30 m, AND WHY THAT MAKES EVERYTHING SIMPLER.

Pooling is irreversible with respect to the pixel mask -- you cannot recover "mean of pixels
with ST_QA <= 2" from "mean of pixels with ST_QA <= 3".  An earlier draft of this pull therefore
had to fix the QC thresholds before running, carry a second "loose" pooling as insurance, and
justify a cell valid-fraction cut.  Storing the bands raw at 30 m deletes that whole problem:
nothing is masked, nothing is averaged, and EVERY QC threshold becomes a training-time decision
that can be swept without touching the network.  qa_pixel is kept as raw uint16 DN for the same
reason it always was (§29.4) -- it is a bitfield, and a decoded boolean throws the bits away.

What is measured per scene (clear fraction, median ST_QA, tile-min CDIST, ...) is RECORDED in
the checkpoint and the bundle as covariates.  None of it is applied.

Consumers pool to whatever grid the model wants.  §33.12(d)'s 22x22 @ 100 m is an exact
partition of the 2200 m patch grid and remains the expected default; 56x56 @ 40 m and 14x14 @
160 m are the other exact options.  Note that 30 m is NOT reachable exactly from the model's
112x112 @ 20 m feature map (ratio 1.5), so some pooling happens regardless -- it just happens
downstream, where it can be changed.

THREE THINGS THIS FILE GETS RIGHT, each of which is a way to be silently wrong.

1.  A SCENE'S CRS IS NOT THE STATION'S CRS.  Landsat delivers each scene in the UTM zone of the
    SCENE CENTRE, which differs from the station's own zone for any station near a zone edge.
    download_landsat_st_mpc.assert_grid_invariants() raised SystemExit on that and killed 8 of
    51 stations in job 27015709 -- 430 wrong-zone scenes at one station, 282 at another, 254 at
    a third -- and MPC additionally returned a malformed 'EPSG:3264'.  A station needs ONE grid,
    so those scenes are reprojected 30 m -> 30 m, nearest, and the fact is RECORDED per scene
    (native_epsg, reprojected) rather than being fatal.  It is not rare: 180 of 272 scenes at
    CentraliaLake.

2.  SOME MPC ASSETS ARE GENUINELY UNREADABLE, SERVER-SIDE.  §41.6 recorded "one corrupt asset
    per cluster" and was right.  The signature is a tile-level read failure --
    "TIFFFillTile: got 0 bytes, expected 101395" -> "not recognized as being in a supported file
    format" -- on ONE scene while other scenes sharing the same SAS signature succeed.  36
    distinct scenes across job 27015709.  Retrying cannot fix it, so it is counted, logged and
    skipped, never allowed to fail a station.  (An earlier draft of this docstring blamed
    expired SAS tokens on the strength of a grep that had actually matched the log's own
    progress counters, "[401/528]".  It was wrong.)

3.  A SILENT AUTH FAILURE WRITES A BUNDLE OF NaN AND LOOKS LIKE SUCCESS.  No station bundle is
    written until at least one scene returns a tile mean in [220, 340] K.

OUTPUT  {DATA_ROOT}/{cat}/{folder}/LANDSAT_ST/{folder}_st30_{start}_{end}.npz
        csvs/landsat_st30_log.{tag}.csv     one row per STATION, per array task
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
os.environ.pop("PROJ_DATA", None)
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import requests
import stackstac
from pyproj import Transformer
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_landsat_st_mpc import (  # noqa: E402
    LS_GRID_OFFSET, LS_RES_M, MPC_URL, bbox_wgs84, snap,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================

REPO       = Path(__file__).resolve().parent
SPLITS     = REPO / "csvs" / "station_splits.csv"
DATA_ROOT  = Path(os.getenv("SOIL_DATA_ROOT", "/gpfs/work3/0/prjs1968/data"))
LOG_DIR    = REPO / "logs"
CKPT_BASE  = REPO / "csvs" / "landsat_st30_log.csv"

COLLECTION = "landsat-c2-l2"
ASSETS     = ["lwir11", "qa", "qa_pixel", "cdist", "emis"]
PLATFORMS  = ("landsat-8", "landsat-9")
TIER       = "T1"
MAX_CLOUD  = 80                    # scene-level, deliberately loose (§29.5 tier 1)

# The station window: 76 x 76 px @ 30 m = 2280 m, snapped onto the Landsat lattice
# (origin = 15,15 mod 30) so an in-zone scene is read with NO resampling at all.
# Slightly larger than the 2240 m S2 tile, which costs nothing and keeps the 2200 m / 22 px
# @ 100 m partition of §33.12(d) fully inside it if a consumer wants that grid.
GRID_N = 76

# Fallback scale/offset/nodata.  The item's raster:bands is authoritative and used when present;
# these exist only so a missing extension degrades loudly rather than silently.  (The ECOSTRESS
# v002 burn: the documented uint16 x 0.02 was the SWATH spec, the tiled product was float32 K.)
FALLBACK = {
    "lwir11":   {"scale": 0.00341802, "offset": 149.0, "nodata": 0},
    "qa":       {"scale": 0.01,       "offset": 0.0,   "nodata": -9999},   # K
    "qa_pixel": {"scale": 1.0,        "offset": 0.0,   "nodata": 1},       # raw DN bitfield
    "cdist":    {"scale": 0.01,       "offset": 0.0,   "nodata": -9999},   # km
    "emis":     {"scale": 0.0001,     "offset": 0.0,   "nodata": -9999},
}

# Physical plausibility, used ONLY for the per-scene statistics and the tripwire -- never to
# mask the stored arrays.  §29.5's 250/350 K guard is a consumer-side decision now.
LST_LO, LST_HI = 250.0, 350.0
K_LO, K_HI     = 220.0, 340.0      # Kelvin tripwire

MAX_RETRIES = 5
RETRY_WAITS = [2, 5, 15, 30, 60]
_NO_RETRY_HTTP = frozenset({404})

LOG_COLS = ["station_id", "folder", "category", "n_scenes_found", "n_scenes_kept",
            "n_error", "n_unreadable", "n_reprojected", "start", "end", "status", "error_msg",
            "mean_clear_frac", "clear_ceiling", "median_st_qa", "mb", "bundle", "timestamp"]

# GDAL's signature for a genuinely unreadable server-side asset (point 2 in the module docs).
_CORRUPT_MARKS = ("not recognized as being in a supported file format",
                  "TIFFReadEncodedTile", "TIFFFillTile", "IReadBlock failed")


# ============================================================
# GEOMETRY
# ============================================================

def station_grid30(row) -> dict:
    """A FIXED 76x76 @ 30 m station-centred window on the Landsat lattice.

    The east/north edges are derived by adding exactly GRID_N*30 to the snapped west/south
    rather than snapping independently -- snapping both ends is what made job 27015709's
    rasters ragged (76x76, 76x77, 77x76 and 77x77 all appear on disk), and a ragged shape
    cannot be stacked into one array.
    """
    lat, lon = float(row.latitude), float(row.longitude)
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = fwd.transform(lon, lat)

    half  = GRID_N * LS_RES_M / 2.0
    west  = snap(cx - half)
    south = snap(cy - half)
    return {
        "epsg": epsg,
        "bounds": (west, south, west + GRID_N * LS_RES_M, south + GRID_N * LS_RES_M),
        "res_m": LS_RES_M, "grid_offset": LS_GRID_OFFSET,
        "cx": cx, "cy": cy, "lat": lat, "lon": lon,
        # where the station actually sits in the array, since the snap shifts it up to 30 m
        "centre_px": ((cx - west) / LS_RES_M, (south + GRID_N * LS_RES_M - cy) / LS_RES_M),
    }


# ============================================================
# SEARCH
# ============================================================

def search_scenes(catalog, bbox, start, end) -> list:
    items = list(catalog.search(
        collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).items())
    kept = []
    for it in items:
        p = it.properties
        if p.get("platform") not in PLATFORMS:
            continue
        if p.get("landsat:collection_category") != TIER:
            continue
        if not all(a in it.assets for a in ASSETS):    # L7 exposes `lwir`, not `lwir11`
            continue
        kept.append(it)
    kept.sort(key=lambda i: i.properties["datetime"])
    return kept


def native_epsg_of(item) -> int | None:
    """The scene's own CRS, tolerating MPC's occasional malformed proj:code ('EPSG:3264')."""
    p = item.properties
    code = p.get("proj:code")
    if isinstance(code, str) and code.upper().startswith("EPSG:"):
        try:
            v = int(code.split(":", 1)[1])
            if 1024 <= v <= 32767 or v >= 100000:
                return v
        except ValueError:
            pass
    v = p.get("proj:epsg")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ============================================================
# RETRY
# ============================================================

def _status_of(exc):
    r = getattr(exc, "response", None)
    return getattr(r, "status_code", None) if r is not None else None


def is_corrupt_asset(exc) -> bool:
    s = str(exc)
    return any(m in s for m in _CORRUPT_MARKS)


def with_retry(fn, max_retries=MAX_RETRIES, waits=RETRY_WAITS):
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as exc:
            if _status_of(exc) in _NO_RETRY_HTTP or attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt] * (1.0 + random.random()))
        except Exception as exc:
            # A server-side unreadable tile is an ANSWER, not a transient failure.  Retrying it
            # four more times buys nothing and costs 110 s per scene (module docs, point 2).
            if is_corrupt_asset(exc) or attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt] * (1.0 + random.random()))


# ============================================================
# QA  (computed for STATISTICS ONLY -- never applied to the stored arrays)
# ============================================================

def qa_decode(qa_dn: np.ndarray):
    """(clear, water) from Landsat C2 QA_PIXEL, for the per-scene covariates. See §29.5 tier 2.

    Rejects fill(0) dilated-cloud(1) cirrus(2) cloud(3) shadow(4) snow(5); requires
    cloud/shadow/cirrus confidence <= low.  Bit 6 ("Clear") is NOT required -- it is defined as
    cloud==0 AND dilated==0, already implied, and Step 0 measured it firing on 62.5% of pixels
    against a derived clear of 48%.  Bit 7 (Water) is returned separately, never folded into the
    rejection: it fires on flooded fields and saturated bare soil, and Step 0 measured it at only
    0.52% of pixels, so nothing is gained by dropping it and a static hole is avoided.
    """
    q = np.nan_to_num(qa_dn, nan=1.0).astype(np.uint16)
    single = ~(((q >> 0) & 1) | ((q >> 1) & 1) | ((q >> 2) & 1)
               | ((q >> 3) & 1) | ((q >> 4) & 1) | ((q >> 5) & 1)).astype(bool)
    conf = (((q >> 8) & 3) <= 1) & (((q >> 10) & 3) <= 1) & (((q >> 14) & 3) <= 1)
    return single & conf, ((q >> 7) & 1).astype(bool)


def band(da, item, name: str) -> np.ndarray:
    """Apply the asset's own scale/offset/nodata, taken from the item, not from memory."""
    rb = (item.assets[name].extra_fields.get("raster:bands") or [{}])[0]
    fb = FALLBACK[name]
    scale  = rb.get("scale",  fb["scale"])
    offset = rb.get("offset", fb["offset"])
    nodata = rb.get("nodata", fb["nodata"])
    v = da.sel(band=name).values.astype("float64")
    v = np.where(np.isnan(v) | (v == nodata), np.nan, v)
    return v if name == "qa_pixel" else v * scale + offset


# ============================================================
# ONE SCENE
# ============================================================

def solar_time(iso_utc: str, lon: float) -> float:
    """Local solar time in hours, WRAPPED AT 24 -- §36's well_phased flag was wrong precisely
    because it was not.  Phase is a covariate here, never a filter (§41.3)."""
    t = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return (t.hour + t.minute / 60.0 + t.second / 3600.0 + lon / 15.0) % 24.0


def process_scene(item, grid: dict) -> dict:
    p, iso = item.properties, item.properties["datetime"]

    def _load():
        planetary_computer.sign_inplace(item)      # SAS tokens expire -- re-sign every attempt
        return stackstac.stack(
            [item], assets=ASSETS,
            epsg=grid["epsg"], resolution=grid["res_m"], bounds=grid["bounds"],
            # snap_bounds=True (the default) rounds bounds outward to multiples of the
            # resolution ANCHORED AT ZERO.  The Landsat lattice is offset by 15 m, so a request
            # for exactly 76*30 m renders as 77x77 -- which is why job 27015709's rasters came
            # out ragged (76x76, 76x77, 77x76, 77x77 all on disk).  False keeps the bounds we
            # asked for, which are already ON the Landsat lattice, so an in-zone scene is read
            # with no resampling and the shape is exactly 76x76 every time.
            snap_bounds=False,
            rescale=False, resampling=Resampling.nearest,
            dtype="float64", fill_value=np.nan,
        ).squeeze("time").compute()

    da = with_retry(_load)

    lst   = band(da, item, "lwir11")
    st_qa = band(da, item, "qa")
    qa_dn = band(da, item, "qa_pixel")
    cdist = band(da, item, "cdist")
    emis  = band(da, item, "emis")

    if lst.shape != (GRID_N, GRID_N):
        raise ValueError(f"grid is {lst.shape}, expected ({GRID_N}, {GRID_N})")

    clear, water = qa_decode(qa_dn)
    finite = np.isfinite(lst)
    ok = clear & finite

    rec = {
        "date": iso[:10].replace("-", ""),
        "item_id": item.id,
        "platform": p.get("platform", ""),
        "wrs_path": int(p.get("landsat:wrs_path", -1)),
        "wrs_row": int(p.get("landsat:wrs_row", -1)),
        "eo_cloud_cover": float(p.get("eo:cloud_cover", np.nan)),
        "sun_elevation": float(p.get("view:sun_elevation", np.nan)),
        "sun_azimuth": float(p.get("view:sun_azimuth", np.nan)),
        "day_tst": solar_time(iso, grid["lon"]),
        "native_epsg": native_epsg_of(item) or -1,
        # --- covariates: MEASURED, NOT APPLIED ---
        "frac_onswath": float(finite.mean()),
        "clear_frac": float(ok.mean()),
        "water_frac": float(water.mean()),
        "frac_in_range": float(((lst > LST_LO) & (lst < LST_HI))[ok].mean()) if ok.any() else 0.0,
        "median_st_qa": float(np.nanmedian(st_qa[ok])) if ok.any() and np.isfinite(st_qa[ok]).any() else np.nan,
        "tile_min_cdist": float(np.nanmin(cdist[ok])) if ok.any() and np.isfinite(cdist[ok]).any() else np.nan,
        "tile_mean_k": float(np.nanmean(lst[ok])) if ok.any() else np.nan,
    }
    rec["reprojected"] = int(rec["native_epsg"] != grid["epsg"])
    rec["empty"] = int(not finite.any())
    if rec["empty"]:
        return rec

    # RAW. No mask applied, no averaging. qa_pixel stays a uint16 bitfield.
    rec["_lst"]      = lst.astype("float32")
    rec["_st_qa"]    = st_qa.astype("float32")
    rec["_cdist"]    = cdist.astype("float32")
    rec["_qa_pixel"] = np.nan_to_num(qa_dn, nan=1.0).astype("uint16")
    rec["_emis"]     = emis.astype("float32")
    return rec


# ============================================================
# ONE STATION
# ============================================================

def station_folder(row) -> str:
    src, net = row.source_network, row.network
    return f"{src}_{net}_{row.station_id}" if (pd.notna(src) and src != net) \
        else f"{net}_{row.station_id}"


def category_of(row) -> str:
    sm = str(row.has_soil_moisture).lower() == "true"
    fl = str(row.has_flux).lower() == "true"
    if sm and fl:
        return "sm_and_flux"
    return "sm_only" if sm else "flux_only"


def _d(v) -> str:
    s = str(v).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s


def process_station(row, catalog, workers: int, overwrite: bool) -> dict:
    sid, folder, cat = str(row.station_id), station_folder(row), category_of(row)
    start, end = _d(row.start_date), _d(row.end_date)
    rec = {"station_id": sid, "folder": folder, "category": cat, "start": start, "end": end,
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    out_dir = DATA_ROOT / cat / folder / "LANDSAT_ST"
    bundle = out_dir / f"{folder}_st30_{start.replace('-','')}_{end.replace('-','')}.npz"
    rec["bundle"] = str(bundle)
    if bundle.exists() and not overwrite:
        rec["status"] = "skip_exists"
        return rec

    grid = station_grid30(row)
    items = search_scenes(catalog, bbox_wgs84(grid), start, end)
    rec["n_scenes_found"] = len(items)
    if not items:
        rec["status"] = "no_scenes"
        return rec

    n_repro = sum(1 for it in items if (native_epsg_of(it) or grid["epsg"]) != grid["epsg"])
    rec["n_reprojected"] = n_repro
    if n_repro:
        logging.info("%s: %d/%d scenes in another UTM zone -- reprojected 30->30, not dropped",
                     sid, n_repro, len(items))

    out, n_err, n_corrupt, errs = [], 0, 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_scene, it, grid): it for it in items}
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as exc:
                if is_corrupt_asset(exc):
                    n_corrupt += 1                      # server-side, expected, not a failure
                else:
                    n_err += 1
                    if len(errs) < 3:
                        errs.append(f"{futs[fut].id}: {str(exc)[:110]}")
    rec["n_error"], rec["n_unreadable"] = n_err, n_corrupt
    if errs:
        rec["error_msg"] = " | ".join(errs)[:300]

    keep = sorted((r for r in out if not r.get("empty", 1) and "_lst" in r),
                  key=lambda r: (r["date"], r["item_id"]))
    rec["n_scenes_kept"] = len(keep)
    if not keep:
        rec["status"] = "no_usable_scenes"
        return rec

    means = np.array([r["tile_mean_k"] for r in keep], dtype="float64")
    if not np.any((means > K_LO) & (means < K_HI)):
        rec["status"] = "tripwire"
        rec["error_msg"] = (f"no tile mean in [{K_LO},{K_HI}] K over {len(keep)} scenes "
                            f"-- this is what a silent auth failure looks like")
        logging.error("%s TRIPWIRE: %s", sid, rec["error_msg"])
        return rec

    st = lambda k: np.stack([r[k] for r in keep])        # noqa: E731
    payload = {
        "lst30":      st("_lst"),          # float32 Kelvin, NaN only where no retrieval
        "st_qa30":    st("_st_qa"),        # float32 K
        "cdist30":    st("_cdist"),        # float32 km
        "qa_pixel30": st("_qa_pixel"),     # uint16 RAW DN -- it is a bitfield
        # emis is ASTER-GED derived and near-static; one median map instead of N copies
        "emis30":     np.nanmedian(np.stack([r["_emis"] for r in keep]), axis=0).astype("float32"),
        "dates":      np.array([r["date"] for r in keep]),
        "item_ids":   np.array([r["item_id"] for r in keep]),
        "platform":   np.array([r["platform"] for r in keep]),
        "wrs_path":   np.array([r["wrs_path"] for r in keep], dtype="int16"),
        "wrs_row":    np.array([r["wrs_row"] for r in keep], dtype="int16"),
        "native_epsg": np.array([r["native_epsg"] for r in keep], dtype="int32"),
        "reprojected": np.array([r["reprojected"] for r in keep], dtype="uint8"),
    }
    for k in ("eo_cloud_cover", "sun_elevation", "sun_azimuth", "day_tst", "frac_onswath",
              "clear_frac", "water_frac", "frac_in_range", "median_st_qa", "tile_min_cdist",
              "tile_mean_k"):
        payload[k] = np.array([r.get(k, np.nan) for r in keep], dtype="float32")

    payload["meta"] = np.array([json.dumps({
        "station_id": sid, "folder": folder, "category": cat,
        "epsg": grid["epsg"], "centre_utm": [grid["cx"], grid["cy"]],
        "centre_px": list(grid["centre_px"]), "bounds": list(grid["bounds"]),
        "lat": grid["lat"], "lon": grid["lon"],
        "grid": f"{GRID_N}x{GRID_N} @ {LS_RES_M} m = {GRID_N*LS_RES_M} m, station-centred, "
                f"Landsat lattice (origin = {LS_GRID_OFFSET},{LS_GRID_OFFSET} mod {LS_RES_M})",
        "target": "ABSOLUTE Landsat ST in Kelvin at NATIVE 30 m. No centring, no ERA5 residual, "
                  "NO QC APPLIED -- every threshold is a consumer-side decision.",
        "bands": {"lst30": "float32 K", "st_qa30": "float32 K uncertainty",
                  "cdist30": "float32 km to nearest cloud",
                  "qa_pixel30": "uint16 RAW DN bitfield (C2 QA_PIXEL)",
                  "emis30": "float32, station-median, static"},
        "qc_note": "qa_decode() in download_landsat_st30.py is the reference QA_PIXEL decoder; "
                   "Step 0 (csvs/landsat_qc_yield_summary.txt) measured the yield curves. "
                   "TIRS acquires at ~100 m, so 30 m carries no finer thermal information -- "
                   "pool to an exact partition (22x22 @ 100 m, 56x56 @ 40 m) before use.",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })])

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = bundle.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.rename(bundle)

    cf = np.array([r["clear_frac"] for r in keep])
    mq = np.array([r["median_st_qa"] for r in keep], dtype="float64")
    rec.update(status="done",
               mb=round(bundle.stat().st_size / 1e6, 2),
               mean_clear_frac=round(float(cf.mean()), 5),
               clear_ceiling=round(float(cf.max()), 5),
               median_st_qa=round(float(np.nanmedian(mq)), 4) if np.isfinite(mq).any() else None)
    return rec


# ============================================================
# CHECKPOINT
# ============================================================

def ckpt_path(tag: str) -> Path:
    return CKPT_BASE.with_suffix(f".{tag}.csv") if tag else CKPT_BASE


def load_done() -> set:
    """Union of EVERY shard, so resume works no matter how the array was sliced.

    Only done/skip_exists/no_scenes count.  `error`, `tripwire` and `no_usable_scenes` do NOT,
    so a plain resubmit retries exactly the failures -- the §37.8 lesson, where read_ok=0 rows
    counted as done and resubmits retried nothing.
    """
    done = set()
    for f in CKPT_BASE.parent.glob(f"{CKPT_BASE.stem}*.csv"):
        try:
            d = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        if "status" in d and "station_id" in d:
            done |= set(d.loc[d.status.isin(["done", "skip_exists", "no_scenes"]), "station_id"])
    return done


def append_row(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=-1,
                    help="this task's shard; work splits by md5(station_id), NOT by position")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--station", default="")
    ap.add_argument("--stations", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.shard_tag or (f"s{args.shard:03d}_{args.nshards:03d}" if args.shard >= 0 else "")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOG_DIR / f"landsat_st30{('.'+tag) if tag else ''}.log")])

    df = pd.read_csv(SPLITS).reset_index(drop=True)   # pandas, never awk: quoted commas
    if args.station:
        df = df[df.station_id.astype(str) == args.station]
    elif args.stations:
        want = {s.strip() for s in args.stations.split(",") if s.strip()}
        df = df[df.station_id.astype(str).isin(want)]

    # SHARD BEFORE THE CHECKPOINT, AND BY HASH, NOT BY POSITION.  Positional slicing of a
    # checkpoint-filtered list is the bug read_ecostress_lst.py:579 records: every task reads the
    # checkpoint at its own start time, so the slices stop tiling the work.  On array 26800268
    # that silently left 7,785 of 119,566 reads unattempted.
    if args.shard >= 0:
        if not 0 <= args.shard < args.nshards:
            raise SystemExit(f"--shard {args.shard} out of range for --nshards {args.nshards}")
        keep = df.station_id.astype(str).map(
            lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % args.nshards == args.shard)
        df = df[keep]
        logging.info("shard %d/%d: %d stations", args.shard, args.nshards, len(df))

    if not args.no_checkpoint:
        done = load_done()
        before = len(df)
        df = df[~df.station_id.astype(str).isin(done)]
        logging.info("checkpoint: %d of %d already done", before - len(df), before)

    df = df.reset_index(drop=True)
    logging.info("%d stations to process", len(df))
    if df.empty:
        logging.info("nothing to do")
        return

    catalog = pystac_client.Client.open(MPC_URL, modifier=planetary_computer.sign_inplace)
    ck = ckpt_path(tag)
    t0, n_ok, n_bad, mb = time.time(), 0, 0, 0.0

    for i, row in enumerate(df.itertuples(index=False), 1):
        try:
            rec = process_station(row, catalog, args.workers, args.overwrite)
        except Exception as exc:
            rec = {"station_id": str(row.station_id), "status": "error",
                   "error_msg": str(exc)[:300],
                   "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            logging.exception("%s FAILED", row.station_id)
        if rec.get("status") == "done":
            n_ok += 1
            mb += rec.get("mb", 0.0)
        elif rec.get("status") not in ("skip_exists", "no_scenes"):
            n_bad += 1
        if not args.no_checkpoint:
            append_row(ck, rec)
        logging.info("[%d/%d] %-26s %-16s found=%-4s kept=%-4s err=%-3s corrupt=%-3s repro=%-4s %sMB",
                     i, len(df), rec.get("station_id"), rec.get("status"),
                     rec.get("n_scenes_found", "-"), rec.get("n_scenes_kept", "-"),
                     rec.get("n_error", "-"), rec.get("n_unreadable", "-"),
                     rec.get("n_reprojected", "-"), rec.get("mb", "-"))

    logging.info("finished: %d ok, %d problem, %.1f GB, in %.1f min",
                 n_ok, n_bad, mb / 1000.0, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
