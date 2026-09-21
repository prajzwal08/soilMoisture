#!/usr/bin/env python
"""§37 TIER 2 -- read ECOSTRESS LST over the station window for every pair in the band.

No LST pixel has ever been read in this project.  census_ecostress.py and
qc_wellphased_pairs.py open _QC, _cloud and _water only; the single mention of "LST" in
the Python is probe_granule's grid check, which reads the transform and never a pixel.
This script is the missing Tier 2.

WHAT IT READS.  Four layers per granule-half, in ONE pass over ONE window:

    _QC      anchors CRS/transform; bits 15&14 accuracy, bits 1&0 mandatory QA
    _cloud   per-pixel masking -- a granule-level clear_frac cannot mask pixels
    _water   keeps the mask IDENTICAL to §36.23, so clear_frac reconciles exactly
    _LST     the measurement

Four opens per half, eight per pair.  Open count is the entire cost model (~2.1 s of LP
DAAC redirect latency each, ~90% of wall), which is why _view_zenith (§36.0b: all-NaN on
2 of 3 probed granules) and _LST_err are not read.

WHAT IT WRITES.  Day and night are the stored primitives; DTR is DERIVED later (D7).
Storing DTR alone would discard the absolute temperature level, make the per-pixel
validity intersection irrecoverable, and foreclose the noise-floor test.

    csvs/ecostress_lst_reads.{tag}.s{N}.csv   the index + summary stats, and the resume
                                              checkpoint
    {STAGING}/lst_reads.{tag}.s{N}.bin        fixed-size raw records, appended:
                                                  lst_k  float32[32,32]   Kelvin
                                                  keep   uint8  [32,32]
                                              offset in the CSV; 5120 B per read

LST SCALING -- MEASURED 2026-09-21, and it contradicts §29.6 and §36.15c.  Both say "LST is
uint16, scale 0.02 K, fill 0 -- the only factor safe to hardcode."  That is the spec for
**ECO2LSTE v001, the SWATH product**.  We read **ECO_L2T_LSTE v002, the TILED product**
(C2076090826-LPCLOUD), and its _LST.tif is delivered as:

    dtype=float32   nodata=nan   scales=(1.0,)   offsets=(0.0,)   units=Kelvin

i.e. already in Kelvin, with NaN fill and no scaling.  Applying the documented 0.02 gives
~5.9 K instead of ~295 K, and casting the float array into a uint16 buffer raises
"invalid value encountered in cast" as the NaNs go through.  The smoke test's Kelvin
tripwire caught exactly this on its first run.  The same NASA catalogue page cannot
describe the tiled layers at all -- it states that cloud, water, view_zenith and height
"are not documented in this product specification" -- which is the tell that it is
documenting a different product.

So the type is taken FROM THE COG and both paths are supported: float -> Kelvin directly,
integer -> multiply by the file's own scale (falling back to 0.02 only if the file
declares none).  Values are stored as float32 Kelvin.  float16 is NOT used: it has ~0.25 K
spacing near 300 K, coarser than the sensor noise.

Resume-safe: the checkpoint is EVERY shard file, so restarting with a different shard
count never re-reads.  Sharded on a stable MD5 of the task key, never on position in the
todo list -- position sharding left 7,785 of 119,566 reads unattempted on array 26800268
because each task reads the checkpoint at its own start time and slices a different
partition.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import (  # noqa: E402
    ROOT, STATION_CSV, N_PX_EXPECTED, WINDOW_FRAC_MIN, CLEAR_FRAC_MIN, TILE_M,
    append_rows, configure_gdal, decode_qc, layer_pool, layer_url, setup_logging,
)

# Fallback scale for the INTEGER path only.  ECO_L2T_LSTE v002 is float32 Kelvin and
# never uses it; it exists so a uint16 granule (v001-style) would still decode.
LST_SCALE_FALLBACK = 0.02
LST_K_LO, LST_K_HI = 220.0, 340.0     # the tripwire band, not a filter. 220 not 240: a
                                      # Michigan night in December legitimately reads 231 K.

SIDE = int(round(TILE_M / 70.0))       # 32
REC_BYTES = N_PX_EXPECTED * 4 + N_PX_EXPECTED   # float32 grid + uint8 mask = 5120

STAGING = Path("/gpfs/work3/0/prjs1968/data/_ecostress_staging")

READ_COLS = [
    "station_id", "granule_ur", "half", "lat", "lon",
    "read_ok", "n_px", "window_frac", "clipped",
    "clear_frac", "usable_frac", "lst_nan_frac", "valid_frac", "frac_mand00",
    "frac_mand01", "frac_lstacc_ge2",
    "frac_water", "frac_cloud", "passed_qc",
    "lst_mean_k", "lst_sd_k", "lst_min_k", "lst_max_k", "n_valid_px",
    "lst_scale_file", "lst_nodata_file", "lst_dtype_file", "win_row_off", "win_col_off",
    "blob_offset", "error",
]

_BLOB_LOCK = threading.Lock()


# ------------------------------------------------------------------
def read_station_lst(ur: str, lon: float, lat: float, read_water: bool = True):
    """Windowed read of QC + cloud + water + LST over the 2.24 km station box.

    Returns (summary_dict, lst_k float32[SIDE,SIDE] Kelvin, keep bool[SIDE,SIDE]).

    Mirrors census_ecostress.read_station_window exactly for everything it shares -- same
    anchor layer, same window construction, same keep mask, same N_PX_EXPECTED denominator
    -- so clear_frac is directly comparable to the §36.23 CSVs.  It differs in retaining
    the arrays rather than only their summary, and in embedding a clipped read into a
    fixed SIDE x SIDE canvas so every blob record is the same size.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window, from_bounds

    out = {k: np.nan for k in ("window_frac", "clear_frac", "usable_frac", "lst_nan_frac",
                               "valid_frac", "frac_mand00",
                               "frac_mand01", "frac_lstacc_ge2", "frac_water",
                               "frac_cloud", "lst_mean_k", "lst_sd_k", "lst_min_k",
                               "lst_max_k", "lst_scale_file", "lst_nodata_file")}
    out.update({"read_ok": 0, "n_px": 0, "clipped": 0, "passed_qc": 0, "n_valid_px": 0,
                "win_row_off": -1, "win_col_off": -1, "blob_offset": -1, "error": ""})
    out["lst_dtype_file"] = ""
    lst_canvas = np.full((SIDE, SIDE), np.nan, dtype=np.float32)
    keep_canvas = np.zeros((SIDE, SIDE), dtype=bool)

    try:
        with rasterio.open(f"/vsicurl/{layer_url(ur, 'QC')}") as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
            half = TILE_M / 2.0
            win = from_bounds(cx - half, cy - half, cx + half, cy + half,
                              src.transform).round_offsets().round_lengths()
            # Clamp EXPLICITLY rather than letting rasterio do it silently (H5), so the
            # offset of the valid part inside the canvas is known instead of inferred.
            sub = win.intersection(Window(0, 0, src.width, src.height))
            qc = src.read(1, window=sub)
            r0 = int(round(sub.row_off - win.row_off))
            c0 = int(round(sub.col_off - win.col_off))
            out["win_row_off"], out["win_col_off"] = int(win.row_off), int(win.col_off)

        if qc.size == 0:
            out["error"] = "empty window -- station outside the tile footprint"
            return out, lst_canvas, keep_canvas

        n = int(qc.size)
        out["n_px"] = n
        out["window_frac"] = n / float(N_PX_EXPECTED)
        if n < N_PX_EXPECTED:
            out["clipped"] = 1
            if out["window_frac"] < WINDOW_FRAC_MIN:
                out["error"] = (f"window clipped to {n}/{N_PX_EXPECTED} px "
                                f"({out['window_frac']:.2f}) -- station too near a tile edge")
                return out, lst_canvas, keep_canvas

        layers = ["cloud", "LST"] + (["water"] if read_water else [])
        bands, nodata, scales = {}, {}, {}

        def _read_layer(lyr):
            with rasterio.open(f"/vsicurl/{layer_url(ur, lyr)}") as s:
                # H4 / §36.15: fill and scale FROM THE COG, never assumed.
                return (lyr, s.read(1, window=sub), s.nodata,
                        (s.scales[0] if s.scales else 1.0))

        for lyr, arr, nd, sc in layer_pool().map(_read_layer, layers):
            bands[lyr], nodata[lyr], scales[lyr] = arr, nd, sc

        mand, dataq, lst_acc, keep_qc = decode_qc(qc)
        cloud = bands["cloud"]
        cloud_fill = 255 if nodata["cloud"] is None else int(nodata["cloud"])
        cloud_valid = cloud != cloud_fill

        if read_water:
            water = bands["water"]
            water_fill = 255 if nodata["water"] is None else int(nodata["water"])
            water_valid = water != water_fill
            keep = keep_qc & (cloud == 0) & (water == 0) & cloud_valid & water_valid
        else:
            # frac_water for this granule is already on disk from §36.23; excluding it
            # here changes the mask, so clear_frac will NOT reconcile in this mode.
            water = np.zeros_like(cloud)
            water_valid = np.ones_like(cloud, dtype=bool)
            keep = keep_qc & (cloud == 0) & cloud_valid

        # --- LST: the type decides the decode, not the documentation (see module docstring)
        lst_raw = bands["LST"]
        sc = float(scales["LST"] or 1.0)
        nd = nodata["LST"]
        out["lst_scale_file"] = sc
        out["lst_nodata_file"] = np.nan if nd is None else float(nd)
        out["lst_dtype_file"] = str(lst_raw.dtype)

        if lst_raw.dtype.kind == "f":
            # ECO_L2T_LSTE v002: float32 Kelvin, NaN fill, scale 1.0.
            lst_k = lst_raw.astype(np.float32) * np.float32(sc)
            lst_valid = np.isfinite(lst_raw)
            if nd is not None and np.isfinite(nd):
                lst_valid &= (lst_raw != nd)
        else:
            # integer path (v001-style uint16 DN). Use the file's scale; fall back to the
            # documented 0.02 only when the file declares none.
            eff = sc if sc not in (0.0, 1.0) else LST_SCALE_FALLBACK
            fill = 0 if nd is None else nd
            lst_valid = (lst_raw != fill)
            lst_k = lst_raw.astype(np.float32) * np.float32(eff)

        # TWO masks, deliberately.  `clear` is the §36.23 mask EXACTLY -- QC + cloud +
        # water, and nothing else -- so clear_frac reconciles bit-for-bit against
        # csvs/ecostress_wp_reads.wp*.csv.  `keep` additionally requires an LST retrieval
        # to exist, which is what DTR can actually use.
        #
        # These differ more than expected: ECOSTRESS returns NaN LST over windows that QC,
        # cloud and water all call clear, so folding LST validity into clear_frac (the
        # first version of this code) turned a 1.0 into a 0.0 and broke reconciliation.
        # The gap is measured per read as lst_nan_frac rather than assumed away.
        clear = keep
        keep = clear & lst_valid

        denom = float(N_PX_EXPECTED)
        out.update({
            "read_ok": 1,
            "clear_frac": float(clear.sum()) / denom,
            "usable_frac": float(keep.sum()) / denom,
            "lst_nan_frac": float((clear & ~lst_valid).sum()) / denom,
            "valid_frac": float((mand <= 1).sum()) / denom,
            "frac_mand00": float((mand == 0).sum()) / denom,
            "frac_mand01": float((mand == 1).sum()) / denom,
            "frac_lstacc_ge2": float((lst_acc >= 2).sum()) / denom,
            "frac_water": float((water == 1).sum()) / denom,
            "frac_cloud": float((cloud == 1).sum()) / denom,
        })
        out["passed_qc"] = int(out["clear_frac"] >= CLEAR_FRAC_MIN)

        lst_canvas[r0:r0 + lst_k.shape[0], c0:c0 + lst_k.shape[1]] = lst_k
        keep_canvas[r0:r0 + keep.shape[0], c0:c0 + keep.shape[1]] = keep

        if keep.any():
            k = lst_k[keep].astype(np.float64)
            out.update({"n_valid_px": int(keep.sum()), "lst_mean_k": float(k.mean()),
                        "lst_sd_k": float(k.std()), "lst_min_k": float(k.min()),
                        "lst_max_k": float(k.max())})
    except Exception as exc:                                        # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out, lst_canvas, keep_canvas


# ------------------------------------------------------------------
def build_tasks(args, log):
    """The task list: one read per (station, granule), from ALREADY-VALIDATED pairs."""
    pairs = pd.read_csv(args.pairs)
    n_all = len(pairs)
    # The census appends per station; a station reprocessed across a resume lands twice.
    pairs = pairs.drop_duplicates(subset=["station_id", "day_ur", "night_ur"])
    n_dedup = len(pairs)

    sel = ((pairs["quality"] == 1) & (pairs["both_read"] == 1)
           & (pairs["dt_hours"] >= args.dt_lo) & (pairs["dt_hours"] < args.dt_hi))
    pairs = pairs[sel].copy()

    log.info("pairs file        : %s", args.pairs)
    log.info("pairs total       : %d  -> %d after duplicate drop", n_all, n_dedup)
    log.info("band              : %g <= dt_hours < %g, quality==1, both_read==1",
             args.dt_lo, args.dt_hi)
    log.info("pairs selected    : %d over %d stations",
             len(pairs), pairs["station_id"].nunique())

    st = pd.read_csv(STATION_CSV)
    coord = st.set_index("station_id")[["latitude", "longitude"]].to_dict("index")
    missing = sorted(set(pairs["station_id"]) - set(coord))
    if missing:
        log.warning("%d pair stations absent from station_splits.csv: %s",
                    len(missing), missing[:5])
        pairs = pairs[~pairs["station_id"].isin(missing)]

    # NOT deduped across stations: the read is a 2.24 km window centred on THAT station,
    # so the same granule at two stations is two different windows.
    tasks: dict[tuple[str, str], dict] = {}
    for half, col in (("day", "day_ur"), ("night", "night_ur")):
        for sid, ur in zip(pairs["station_id"], pairs[col]):
            key = (sid, ur)
            if key not in tasks:
                c = coord[sid]
                tasks[key] = {"station_id": sid, "granule_ur": ur, "half": half,
                              "lat": float(c["latitude"]), "lon": float(c["longitude"])}
    log.info("read tasks        : %d (station, granule) pairs", len(tasks))
    return tasks, pairs


def preflight(log):
    """The silent-401 detector.  COOKIE_JAR is $TMPDIR/edl_cookies.txt evaluated at
    import; if TMPDIR is missing or unwritable, every /vsicurl open 401s, the bare except
    turns it into read_ok=0, and the run finishes CLEANLY with zero data -- which looks
    exactly like total cloud."""
    from census_ecostress import COOKIE_JAR
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    if not tmp.is_dir() or not os.access(tmp, os.W_OK):
        log.error("FATAL: TMPDIR=%s is not a writable directory. Every /vsicurl open "
                  "would 401 and the run would look like total cloud.", tmp)
        sys.exit(1)
    log.info("TMPDIR            : %s (writable)", tmp)
    log.info("cookie jar        : %s", COOKIE_JAR)
    netrc = Path.home() / ".netrc"
    if not netrc.is_file():
        log.error("FATAL: no ~/.netrc -- EDL auth would fail silently.")
        sys.exit(1)
    if "urs.earthdata" not in netrc.read_text():
        log.error("FATAL: ~/.netrc has no urs.earthdata entry.")
        sys.exit(1)
    log.info("netrc             : ok (mode %o)", netrc.stat().st_mode & 0o777)


# ------------------------------------------------------------------
def run_reads(todo, args, log, reads_csv, blob_path, read_water=True):
    """Execute the reads, appending index rows and fixed-size blob records together.

    The CSV row is written only AFTER its bytes are on the blob, so a row always points at
    a record that exists.  The reverse order would leave a resume trusting an offset into
    nothing.
    """
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    buf, t0, n_ok, n_err = [], time.time(), 0, 0
    armed = False          # the Kelvin tripwire: no shard is written until one read looks

    blob = open(blob_path, "ab")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def work(t):
                res, lst, keep = read_station_lst(t["granule_ur"], t["lon"], t["lat"],
                                                  read_water=read_water)
                return t, res, lst, keep

            for i, (t, res, lst, keep) in enumerate(pool.map(work, todo), 1):
                if res["read_ok"]:
                    with _BLOB_LOCK:
                        res["blob_offset"] = blob.tell()
                        blob.write(lst.astype(np.float32).tobytes())
                        blob.write(keep.astype(np.uint8).tobytes())
                    n_ok += 1
                    m = res["lst_mean_k"]
                    if np.isfinite(m) and LST_K_LO <= m <= LST_K_HI:
                        armed = True
                    elif np.isfinite(m):
                        log.warning("LST mean %.1f K outside [%g, %g] for %s @ %s",
                                    m, LST_K_LO, LST_K_HI, t["granule_ur"], t["station_id"])
                else:
                    n_err += 1
                buf.append({**t, **{k: v for k, v in res.items() if k in READ_COLS}})

                if len(buf) >= 500:
                    if not armed:
                        log.error("FATAL: %d reads and not one plausible Kelvin value "
                                  "(%g-%g K). This is what a silent 401 looks like. "
                                  "Refusing to write the shard.", i, LST_K_LO, LST_K_HI)
                        sys.exit(2)
                    blob.flush()
                    os.fsync(blob.fileno())
                    append_rows(reads_csv, READ_COLS, buf)
                    buf = []
                if i % 2000 == 0 or i == len(todo):
                    el = time.time() - t0
                    log.info("%6d/%d  ok=%d err=%d  %.2f reads/s  eta %.0f min",
                             i, len(todo), n_ok, n_err, i / max(el, 1e-9),
                             (len(todo) - i) / max(i / max(el, 1e-9), 1e-9) / 60.0)
        if buf:
            if not armed and n_ok:
                log.error("FATAL: no plausible Kelvin value in any read. Refusing to write.")
                sys.exit(2)
            blob.flush()
            os.fsync(blob.fileno())
            append_rows(reads_csv, READ_COLS, buf)
    finally:
        blob.close()
    el = time.time() - t0
    return {"n": len(todo), "ok": n_ok, "err": n_err, "sec": el,
            "rate": len(todo) / max(el, 1e-9)}


def smoke(tasks, args, log):
    """§37.6 -- measure before committing 315k opens.

    Four questions: what worker count is best for a FOUR-open read (§36.23.5 measured the
    curve for a three-open one), whether _water earns its open, what bytes/read actually
    is, and whether the numbers are right at all.
    """
    import itertools
    configs = list(itertools.product(args.smoke_workers, (True, False)))
    # EVERY CONFIG GETS ITS OWN DISJOINT GRANULES.  The first version reused the same 40
    # tasks for all six configs and measured 1.76 reads/s cold then ~20 reads/s for every
    # repeat -- that was GDAL's VSI cache and the OS page cache, not concurrency. A sweep
    # that re-reads its own warm data measures nothing.
    need = args.smoke_n * len(configs)
    # RANDOM, not sorted.  sorted(tasks) is ordered by station_id, so the first N tasks
    # come from two or three alphabetically-first stations -- fine for a latency number,
    # useless for anything about data availability, which varies by geography and season.
    keys = sorted(tasks)
    rng = np.random.default_rng(args.smoke_seed)
    rng.shuffle(keys)
    if len(keys) < need:
        log.warning("only %d tasks for %d configs x %d", len(keys), len(configs),
                    args.smoke_n)
    rows = []
    STAGING.mkdir(parents=True, exist_ok=True)

    for ci, (workers, water) in enumerate(configs):
        sub = [tasks[k] for k in keys[ci * args.smoke_n:(ci + 1) * args.smoke_n]]
        if not sub:
            continue
        tag = f"w{workers}_{'water' if water else 'nowater'}"
        csv_p = ROOT / "csvs" / f"ecostress_lst_smoke.{tag}.csv"
        blob_p = STAGING / f"smoke.{tag}.bin"
        for p in (csv_p, blob_p):
            if p.exists():
                p.unlink()
        a = argparse.Namespace(**vars(args))
        a.workers = workers
        log.info("--- smoke: workers=%d water=%s n=%d ---", workers, water, len(sub))
        st = run_reads(sub, a, log, csv_p, blob_p, read_water=water)
        st.update({"workers": workers, "water": water,
                   "opens": len(sub) * (4 if water else 3),
                   "blob_bytes": blob_p.stat().st_size if blob_p.exists() else 0})
        st["opens_per_s"] = st["opens"] / max(st["sec"], 1e-9)
        rows.append(st)

    df = pd.DataFrame(rows)
    log.info("\n=== SMOKE TABLE ===\n%s", df.to_string(index=False))

    # question 4: does clear_frac reconcile against §36.23?
    ref = []
    for f in sorted(ROOT.glob("csvs/ecostress_wp_reads.wp*.csv")):
        ref.append(pd.read_csv(f, usecols=["station_id", "granule_ur", "clear_frac"]))
    if ref:
        ref = pd.concat(ref).drop_duplicates(subset=["station_id", "granule_ur"],
                                             keep="last")
        # only the _water arms carry the §36.23-identical mask
        mine = pd.concat([pd.read_csv(p) for p in
                          sorted(ROOT.glob("csvs/ecostress_lst_smoke.w*_water.csv"))])
        j = mine.merge(ref, on=["station_id", "granule_ur"], suffixes=("_new", "_sep"))
        j = j[j["read_ok"] == 1]
        if len(j):
            d = (j["clear_frac_new"] - j["clear_frac_sep"]).abs()
            log.info("clear_frac reconciliation vs §36.23: n=%d  max|diff|=%.6g  "
                     "n_exact=%d/%d", len(j), d.max(), int((d < 1e-9).sum()), len(j))
            if d.max() > 1e-9:
                log.warning("MISMATCH -- the window or the mask has moved:\n%s",
                            j.loc[d.sort_values(ascending=False).index[:5],
                                  ["station_id", "granule_ur", "clear_frac_new",
                                   "clear_frac_sep", "usable_frac", "lst_nan_frac"]]
                            .to_string(index=False))
            # the LST-availability gap, measured rather than assumed
            log.info("LST availability on CLEAR pixels: mean usable_frac=%.4f vs "
                     "clear_frac=%.4f; mean lst_nan_frac=%.4f; %d of %d reads are clear "
                     "but have NO LST at all",
                     j["usable_frac"].mean(), j["clear_frac_new"].mean(),
                     j["lst_nan_frac"].mean(),
                     int(((j["clear_frac_new"] > 0) & (j["usable_frac"] == 0)).sum()),
                     len(j))
        else:
            log.warning("no overlap with the §36.23 reads -- reconciliation not possible")

    df.to_csv(ROOT / "csvs" / "ecostress_lst_smoke.csv", index=False)
    log.info("wrote csvs/ecostress_lst_smoke.csv")
    log.info("bytes/read (blob is 3072 B by construction; watch opens_per_s and sec)")


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "csvs" / "ecostress_wp_pairs.wp.csv"))
    ap.add_argument("--out-tag", default="dtr")
    ap.add_argument("--dt-lo", type=float, default=6.0)
    ap.add_argument("--dt-hi", type=float, default=19.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-water", action="store_true",
                    help="skip _water (3 opens). clear_frac then does NOT reconcile.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=50)
    ap.add_argument("--smoke-workers", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--smoke-seed", type=int, default=20260921)
    args = ap.parse_args()

    setup_logging(f"ecostress_lst_{args.out_tag}_s{args.shard}")
    log = logging.getLogger("lst")
    preflight(log)
    configure_gdal()
    layer_pool(args.workers)        # size the singleton BEFORE the first read

    tasks, _ = build_tasks(args, log)

    if args.smoke:
        smoke(tasks, args, log)
        return

    reads_csv = (ROOT / "csvs" / f"ecostress_lst_reads.{args.out_tag}.csv"
                 if args.nshards <= 1 else
                 ROOT / "csvs" / f"ecostress_lst_reads.{args.out_tag}.s{args.shard}.csv")
    blob = STAGING / (f"lst_reads.{args.out_tag}.bin" if args.nshards <= 1
                      else f"lst_reads.{args.out_tag}.s{args.shard}.bin")

    # The checkpoint is EVERY shard file, so a restart with a different shard count never
    # re-reads what is already measured.
    done: set[tuple[str, str]] = set()
    files = sorted(ROOT.glob(f"csvs/ecostress_lst_reads.{args.out_tag}*.csv"))
    for f in files:
        prev = pd.read_csv(f, usecols=["station_id", "granule_ur"])
        done |= set(zip(prev["station_id"], prev["granule_ur"]))
    log.info("checkpoint        : %d reads on disk across %d file(s)", len(done), len(files))

    if args.nshards > 1:
        todo = [t for k, t in tasks.items()
                if k not in done
                and int(hashlib.md5(f"{k[0]}|{k[1]}".encode()).hexdigest(), 16)
                % args.nshards == args.shard]
        log.info("shard             : %d of %d (stable hash) -> %d reads",
                 args.shard, args.nshards, len(todo))
    else:
        todo = [t for k, t in tasks.items() if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        log.info("nothing to do")
        return

    log.info("workers=%d  layers=%d  blob=%s", args.workers,
             3 if args.no_water else 4, blob)
    st = run_reads(todo, args, log, reads_csv, blob, read_water=not args.no_water)
    log.info("DONE  %d reads  ok=%d err=%d  %.1f min  %.2f reads/s",
             st["n"], st["ok"], st["err"], st["sec"] / 60.0, st["rate"])


if __name__ == "__main__":
    main()
