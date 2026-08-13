"""
Download Landsat 8/9 Collection-2 Level-2 Surface Temperature over the TxSON domain via MPC.

§29 Phase A.  One AOI over the whole TxSON network (or a single tile window with --extent tile),
NOT 40 station patches — the tiles overlap heavily (§26.2: 131 km² in 17 islands).

Output per scene: 3-band float32 GeoTIFF
    band 1  lst_kelvin      lwir11 * 0.00341802 + 149.0     (nodata DN 0   -> NaN)
    band 2  st_qa_kelvin    qa     * 0.01                   (nodata DN -9999 -> NaN)
    band 3  qa_pixel_dn     raw uint16 DN, stored as float32 (exact: DN < 2**24)

qa_pixel is kept as raw DN on purpose so the cloud threshold can be revisited without
re-downloading.  See §29.5 for the bit definitions and the three-tier filtering scheme.

Grid: verified 2026-08-13 that every L8/9 scene over TxSON is EPSG:32614 with proj:transform
origin ≡ (15, 15) mod 30.  The 10 m TxSON tile grid is NOT on that grid, so the AOI is snapped
explicitly — otherwise stackstac silently resamples onto a 15 m-shifted grid.

Assets on MPC are named lwir11 / qa / qa_pixel (NOT ST_B10).  Landsat-7 exposes `lwir` instead
and has SLC-off wedges, so it is excluded.

Usage:
  python download_landsat_st_mpc.py --smoke                    # 1 scene, tile window
  python download_landsat_st_mpc.py --extent tile              # all scenes, 76x76 window
  python download_landsat_st_mpc.py --extent aoi --workers 12  # all scenes, full 1162x1168 AOI
"""

import argparse
import csv
import json
import logging
import math
import os
os.environ.pop("PROJ_DATA", None)
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
import rioxarray  # noqa: F401  registers .rio accessor
import stackstac
import xarray as xr
from rasterio.enums import Resampling

# ============================================================
# CONFIGURATION
# ============================================================

REPO       = Path(__file__).resolve().parent
GRID_JSON  = REPO / "csvs" / "txson_mosaic_grid.json"
READOUTS   = REPO / "csvs" / "txson_readouts.csv"
LOG_FILE   = REPO / "csvs" / "landsat_st_download_log.csv"
LOG_DIR    = REPO / "logs"

OUT_ROOT   = Path("/gpfs/scratch1/shared/pkhanal/lst/landsat_st/txson")

MPC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
ASSETS     = ["lwir11", "qa_pixel", "qa"]

LS_RES_M        = 30
LS_GRID_OFFSET  = 15         # verified: proj:transform origin ≡ (15,15) mod 30
TILE_PX         = 224        # TxSON tile is 224 px @ 10 m
TILE_RES_M      = 10

ST_SCALE, ST_OFFSET = 0.00341802, 149.0
ST_FILL             = 0
QA_SCALE            = 0.01
QA_FILL             = -9999

GLOBAL_START = "2016-01-01"
GLOBAL_END   = "2022-11-07"
MAX_CLOUD    = 80            # scene-level; deliberately loose (see §29.5 tier 1)
PLATFORMS    = ("landsat-8", "landsat-9")
TIER         = "T1"

MAX_RETRIES = 3
RETRY_WAITS = [5, 15, 45]

LOG_COLS = ["item_id", "date", "platform", "wrs_path", "wrs_row", "eo_cloud_cover",
            "n_clear_px", "frac_clear", "mean_lst_k", "extent", "status", "error_msg", "timestamp"]

_LOG_LOCK_MSG = "checkpoint"

# ============================================================
# GEOMETRY
# ============================================================

def snap(v: float, res: int = LS_RES_M, off: int = LS_GRID_OFFSET, up: bool = False) -> float:
    """Snap a UTM coordinate onto the Landsat pixel grid (origin ≡ off mod res)."""
    k = math.ceil((v - off) / res) if up else math.floor((v - off) / res)
    return off + res * k


def aoi_grid(grid_json: Path = GRID_JSON) -> dict:
    """Full TxSON domain, snapped outward onto the Landsat 30 m grid."""
    g = json.loads(grid_json.read_text())
    o = g["origin_utm"]
    west  = snap(float(o["west"]))
    south = snap(float(o["south"]))
    east  = snap(float(o["east"]),  up=True)
    north = snap(float(o["north"]), up=True)
    return {
        "epsg": int(g["epsg"]),
        "bounds": (west, south, east, north),
        "nx": int(round((east - west) / LS_RES_M)),
        "ny": int(round((north - south) / LS_RES_M)),
        "res_m": LS_RES_M,
        "grid_offset": LS_GRID_OFFSET,
    }


def tile_grid(tile: str, readouts: Path = READOUTS) -> dict:
    """The 2.24 km window of one station tile, snapped outward onto the Landsat 30 m grid.

    Tile UTM bounds are reconstructed from the tile's own centre station: the centre station
    sits at pixel (112,112) of a 224 px @ 10 m station-centred grid (download_s2_mpc.py:98-109),
    so west = x_centre - 112*10 and north = y_centre + 112*10.
    """
    from pyproj import Transformer

    df = pd.read_csv(readouts)
    sub = df[(df.tile == tile) & (df.is_centre)]
    if sub.empty:
        raise SystemExit(f"No centre readout row for tile {tile!r} in {readouts}")
    row  = sub.iloc[0]
    epsg = int(row.tile_epsg)
    fwd  = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = fwd.transform(float(row.lon), float(row.lat))
    half = TILE_PX * TILE_RES_M / 2
    west  = snap(cx - half)
    south = snap(cy - half)
    east  = snap(cx + half, up=True)
    north = snap(cy + half, up=True)
    return {
        "epsg": epsg,
        "bounds": (west, south, east, north),
        "nx": int(round((east - west) / LS_RES_M)),
        "ny": int(round((north - south) / LS_RES_M)),
        "res_m": LS_RES_M,
        "grid_offset": LS_GRID_OFFSET,
        "tile": tile,
    }


def bbox_wgs84(grid: dict, pad_m: float = 0.0) -> list:
    """WGS84 bbox enclosing a UTM grid, for the STAC search."""
    from pyproj import Transformer

    inv = Transformer.from_crs(f"EPSG:{grid['epsg']}", "EPSG:4326", always_xy=True)
    w, s, e, n = grid["bounds"]
    w, s, e, n = w - pad_m, s - pad_m, e + pad_m, n + pad_m
    corners = [inv.transform(w, s), inv.transform(e, s), inv.transform(w, n), inv.transform(e, n)]
    return [min(c[0] for c in corners), min(c[1] for c in corners),
            max(c[0] for c in corners), max(c[1] for c in corners)]


# ============================================================
# QA
# ============================================================

def landsat_clear(qa_dn: np.ndarray) -> np.ndarray:
    """Boolean clear-sky mask from Landsat C2 QA_PIXEL. See §29.5 tier 2.

    Rejects: fill(0) dilated-cloud(1) cirrus(2) cloud(3) shadow(4) snow(5) water(7),
    requires Clear(6) set and cloud/shadow/cirrus confidence <= low.
    """
    q = np.nan_to_num(qa_dn, nan=1.0).astype(np.uint16)
    single = ~np.any([((q >> b) & 1).astype(bool) for b in (0, 1, 2, 3, 4, 5, 7)], axis=0)
    conf = (((q >> 8) & 3) <= 1) & (((q >> 10) & 3) <= 1) & (((q >> 14) & 3) <= 1)
    return single & conf & (((q >> 6) & 1) == 1)


# ============================================================
# RETRY / CHECKPOINT  (reused from download_s2_mpc.py:123-166)
# ============================================================

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
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt])


def load_checkpoint() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, dtype=str)
    return pd.DataFrame(columns=LOG_COLS)


def append_checkpoint_row(row: dict):
    write_header = not LOG_FILE.exists()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def setup_logging(name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"{name}.log")],
    )


# ============================================================
# SEARCH
# ============================================================

def search_scenes(catalog, bbox, start, end, cloud_max, platforms, tier) -> list:
    items = list(catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_max}},
    ).items())

    kept = []
    for it in items:
        p = it.properties
        if p.get("platform") not in platforms:
            continue
        if tier and p.get("landsat:collection_category") != tier:
            continue
        if "lwir11" not in it.assets:          # Landsat-7 exposes `lwir`
            logging.warning("%s has no lwir11 asset — skipped", it.id)
            continue
        kept.append(it)

    kept.sort(key=lambda i: i.properties["datetime"])
    return kept


def assert_grid_invariants(items, epsg: int):
    """Every retained scene must be on the CRS and pixel grid the AOI snap assumes."""
    bad_crs, bad_grid = [], []
    for it in items:
        code = it.properties.get("proj:code") or f"EPSG:{it.properties.get('proj:epsg')}"
        if code != f"EPSG:{epsg}":
            bad_crs.append((it.id, code))
        t = it.properties.get("proj:transform")
        if t is not None and (t[2] % LS_RES_M, t[5] % LS_RES_M) != (LS_GRID_OFFSET, LS_GRID_OFFSET):
            bad_grid.append((it.id, t[2] % LS_RES_M, t[5] % LS_RES_M))
    if bad_crs:
        raise SystemExit(f"{len(bad_crs)} scenes not in EPSG:{epsg}, e.g. {bad_crs[:3]}")
    if bad_grid:
        raise SystemExit(f"{len(bad_grid)} scenes off the (15,15) mod 30 grid, e.g. {bad_grid[:3]}")
    logging.info("grid invariants OK on %d scenes (EPSG:%d, origin ≡ (15,15) mod 30)",
                 len(items), epsg)


# ============================================================
# DOWNLOAD ONE SCENE
# ============================================================

def scene_filename(item) -> str:
    p = item.properties
    d = p["datetime"][:10].replace("-", "")
    plat = {"landsat-8": "LC08", "landsat-9": "LC09"}.get(p["platform"], p["platform"])
    return f"{d}_{plat}_{_pr(p)}.tif"


def _pr(p) -> str:
    """path/row as a zero-padded 6-char string. MPC returns these as str, not int."""
    return f"{int(p['landsat:wrs_path']):03d}{int(p['landsat:wrs_row']):03d}"


def download_scene(item, grid: dict, out_dir: Path, overwrite: bool = False) -> dict:
    fname = scene_filename(item)
    fpath = out_dir / fname
    p = item.properties
    rec = {
        "item_id": item.id,
        "date": p["datetime"][:10],
        "platform": p["platform"],
        "wrs_path": p.get("landsat:wrs_path"),
        "wrs_row": p.get("landsat:wrs_row"),
        "eo_cloud_cover": p.get("eo:cloud_cover"),
        "extent": grid.get("tile", "aoi"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if fpath.exists() and not overwrite:
        rec.update(status="skip_exists")
        return rec

    def _load():
        planetary_computer.sign_inplace(item)      # SAS tokens expire — re-sign inside the retry
        da = stackstac.stack(
            [item], assets=ASSETS,
            epsg=grid["epsg"], resolution=grid["res_m"], bounds=grid["bounds"],
            rescale=False, resampling=Resampling.nearest,
            dtype="float64", fill_value=np.nan,
        ).squeeze("time")
        return da.compute()

    da = with_retry(_load)

    st_dn = da.sel(band="lwir11").values
    qa_dn = da.sel(band="qa").values
    qap   = da.sel(band="qa_pixel").values

    lst_k = np.where(np.isnan(st_dn) | (st_dn == ST_FILL), np.nan, st_dn * ST_SCALE + ST_OFFSET)
    st_qa = np.where(np.isnan(qa_dn) | (qa_dn == QA_FILL), np.nan, qa_dn * QA_SCALE)
    qap_f = np.nan_to_num(qap, nan=1.0)            # DN 1 == fill bit set

    clear = landsat_clear(qap_f) & np.isfinite(lst_k)
    n_clear = int(clear.sum())
    rec["n_clear_px"] = n_clear
    rec["frac_clear"] = round(n_clear / clear.size, 5)
    rec["mean_lst_k"] = round(float(np.nanmean(lst_k[clear])), 3) if n_clear else None

    if not np.isfinite(lst_k).any():
        rec.update(status="no_data")
        return rec

    stack = np.stack([lst_k, st_qa, qap_f]).astype("float32")
    out = xr.DataArray(
        stack,
        dims=("band", "y", "x"),
        coords={"band": ["lst_kelvin", "st_qa_kelvin", "qa_pixel_dn"],
                "y": da.y.values, "x": da.x.values},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_suffix(".tmp.tif")
    out = out.rio.write_crs(f"EPSG:{grid['epsg']}")
    out.rio.to_raster(str(tmp), dtype="float32", compress="deflate", tiled=True)
    iso = p["datetime"]
    with rasterio.open(tmp, "r+") as dst:
        dst.descriptions = ("lst_kelvin", "st_qa_kelvin", "qa_pixel_dn")
        dst.update_tags(
            TIFFTAG_DATETIME=iso[:19].replace("T", " ").replace("-", ":"),
            datetime_utc=iso, item_id=item.id, platform=p["platform"],
            wrs_path=str(p.get("landsat:wrs_path")), wrs_row=str(p.get("landsat:wrs_row")),
            eo_cloud_cover=str(p.get("eo:cloud_cover")),
            st_scale=str(ST_SCALE), st_offset=str(ST_OFFSET), st_qa_scale=str(QA_SCALE),
            note="band3 qa_pixel is RAW DN stored as float32 (exact, DN < 2**24)",
        )
    tmp.rename(fpath)
    rec.update(status="done")
    return rec


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extent", choices=["tile", "aoi"], default="aoi",
                    help="tile = one 2.24 km window (fast); aoi = whole TxSON domain")
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--start", default=GLOBAL_START)
    ap.add_argument("--end", default=GLOBAL_END)
    ap.add_argument("--cloud-max", type=float, default=MAX_CLOUD)
    ap.add_argument("--platforms", nargs="+", default=list(PLATFORMS))
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 scene, tile extent, no checkpoint write — prints the read back")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setup_logging("download_landsat_st_mpc")

    if args.smoke:
        args.extent, args.limit = "tile", 1

    grid = tile_grid(args.tile) if args.extent == "tile" else aoi_grid()
    sub  = args.tile if args.extent == "tile" else "aoi"
    out_dir = args.out_root / sub
    bbox = bbox_wgs84(grid)

    logging.info("extent=%s  EPSG:%d  bounds=%s  %d x %d px @ %d m",
                 args.extent, grid["epsg"], grid["bounds"], grid["nx"], grid["ny"], grid["res_m"])
    logging.info("search bbox (WGS84) = %s", [round(v, 4) for v in bbox])

    catalog = pystac_client.Client.open(MPC_URL, modifier=planetary_computer.sign_inplace)
    items = search_scenes(catalog, bbox, args.start, args.end,
                          args.cloud_max, set(args.platforms), args.tier)
    logging.info("%d scenes after platform/tier/asset filter", len(items))
    if not items:
        raise SystemExit("no scenes found")

    assert_grid_invariants(items, grid["epsg"])

    byplat = pd.Series([i.properties["platform"] for i in items]).value_counts().to_dict()
    bypr = pd.Series([_pr(i.properties) for i in items]).value_counts().to_dict()
    logging.info("platforms=%s  path/row=%s", byplat, bypr)

    if args.limit:
        items = items[: args.limit]

    done = set()
    if not args.smoke:
        ck = load_checkpoint()
        if not ck.empty:
            # key on (item_id, extent): the same scene can be fetched for a tile window
            # AND for the full AOI, and those are different files.
            m = ck.status.isin(["done", "no_data"]) & (ck.extent == sub)
            done = set(ck.loc[m, "item_id"])
            logging.info("checkpoint: %d scenes already done for extent=%s", len(done), sub)
    todo = [i for i in items if i.id not in done]
    logging.info("%d scenes to download -> %s", len(todo), out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aoi.json").write_text(json.dumps(
        {**grid, "bounds": list(grid["bounds"]),
         "collection": COLLECTION, "assets": ASSETS,
         "st_scale": ST_SCALE, "st_offset": ST_OFFSET, "st_qa_scale": QA_SCALE,
         "bands": ["lst_kelvin", "st_qa_kelvin", "qa_pixel_dn"],
         "nx_ny_note": "NOMINAL. stackstac snaps bounds outward, so the written rasters are "
                       "typically 1 px larger. ALWAYS take the affine transform and shape from "
                       "the GeoTIFF itself when indexing station pixels — never from nx/ny here.",
         "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))

    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_scene, it, grid, out_dir, args.overwrite): it for it in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            try:
                rec = fut.result()
                n_ok += 1
            except Exception as exc:
                rec = {"item_id": it.id, "date": it.properties["datetime"][:10],
                       "platform": it.properties.get("platform"),
                       "extent": sub, "status": "error", "error_msg": str(exc)[:300],
                       "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                n_err += 1
                logging.error("%s FAILED: %s", it.id, exc)
            if not args.smoke:
                append_checkpoint_row(rec)
            if rec.get("status") == "done":
                logging.info("[%d/%d] %s  clear=%.1f%%  mean_LST=%s K",
                             k, len(todo), rec["date"], 100 * (rec.get("frac_clear") or 0),
                             rec.get("mean_lst_k"))

    dt = time.time() - t0
    logging.info("finished: %d ok, %d errors in %.1f min", n_ok, n_err, dt / 60)

    if args.smoke and todo:
        f = sorted(out_dir.glob("*.tif"))
        if f:
            with rasterio.open(f[-1]) as src:
                a = src.read()
            lst, qa, qap = a[0], a[1], a[2]
            clear = landsat_clear(qap)
            print("\n" + "=" * 68)
            print(f"SMOKE  {f[-1].name}   shape={lst.shape}  crs={src.crs}")
            print(f"  LST      min/med/max = {np.nanmin(lst):.2f} / "
                  f"{np.nanmedian(lst):.2f} / {np.nanmax(lst):.2f} K")
            print(f"  ST_QA    median      = {np.nanmedian(qa):.2f} K")
            print(f"  qa_pixel unique      = {np.unique(qap.astype(np.uint16))[:8]}")
            print(f"  clear                = {clear.sum()}/{clear.size} "
                  f"({100*clear.mean():.1f}%)")
            if clear.any():
                print(f"  LST(clear) mean      = {np.nanmean(lst[clear]):.2f} K   "
                      f"sd = {np.nanstd(lst[clear]):.2f} K")
            print("=" * 68)


if __name__ == "__main__":
    main()
