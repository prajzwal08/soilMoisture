"""
prepare_fine_cache.py — precompute the fine CNN's per-scene inputs into the §48 cache
======================================================================================
2026-09-29. Training was data-bound: build_fine read and pooled a raw S2 and a raw S1 scene
from GPFS for EVERY sample (~0.6 s of the ~1 s per sample; GPFS small random reads cap a node
at ~40 samples/s). Everything build_fine does except picking the scene and the day-D channels
(age, orbit) depends on the scene alone, so it is done here once per scene, with the SAME
functions (dataset.fine_s2_scene / fine_s1_scene / fine_dem_static / lulc_remap), into

  CACHE_ROOT_GPFS/{cat}/{station}/
    fine_s2.npy        (K, 11, 112, 112) f16   one row per S2 candidate (raw row with a cloud mask)
    fine_s1_asc.npy    (Na, 3, 112, 112) f16   one row per raw S1 ascending pass
    fine_s1_desc.npy   (Nd, 3, 112, 112) f16
    fine_meta.npz      s2_cand (K,3) [date, raw_row, tok_row], s1_*_dates, dem (2,112,112) f16,
                       lulc_years, lulc (Y,224,224) u1, fine_stats_sha   <- written LAST

build_fine then only looks rows up (verify_fine_cache.py proves the output is bit-identical).
All years, all stations, so eval uses it too. Resume-safe: a station with fine_meta.npz is
skipped unless --force. Reads the raw store and the cache; writes only the files above.

Usage: python prepare_fine_cache.py [--stations A B ...] [--limit N] [--workers 64] [--force]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
if os.environ.get("S48_CACHE_ROOT"):
    sys.exit("S48_CACHE_ROOT is set — this script writes the GPFS cache only; unset it.")
import dataset as D  # noqa: E402

BLOCK = 32   # raw rows read per zarr call


def _save(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(path)


def one(task):
    cat, st, force = task
    out = D.CACHE_ROOT_GPFS / cat / st
    if (out / "fine_meta.npz").exists() and not force:
        return st, "skip_done", {}
    t0 = time.time()
    cache = D._load_station_cache(out, fine=False)   # never read our own previous output
    raw = D._open_raw(st)
    if cache is None or raw is None:
        return st, "no_cache" if cache is None else "no_raw", {}
    fs = D._load_fine_stats()
    rg = raw["zg"]
    rep = {}

    # S2: one row per raw row whose date has a token row (hence a cloud mask), as build_fine
    cand = []
    if len(raw["s2"]) and "s2_date_ints" in cache and "s2_cm" in cache:
        tok_idx = {int(d): i for i, d in enumerate(cache["s2_date_ints"])}
        cand = [(int(d), ri, tok_idx[int(d)]) for ri, d in enumerate(raw["s2"]) if int(d) in tok_idx]
    s2 = np.zeros((len(cand), 11, 112, 112), np.float16)
    rows = [c[1] for c in cand]
    for b in range(0, len(rows), BLOCK):
        lo, hi = min(rows[b:b + BLOCK]), max(rows[b:b + BLOCK])
        blk = np.asarray(rg["s2/data"][lo:hi + 1])
        for k in range(b, min(b + BLOCK, len(rows))):
            _, ri, ti = cand[k]
            s2[k] = D.fine_s2_scene(blk[ri - lo], cache["s2_cm"][ti], fs).astype(np.float16)
    rep["s2"] = len(cand)

    s1 = {}
    for key in ("s1_asc", "s1_desc"):
        n = len(raw[key])
        a = np.zeros((n, 3, 112, 112), np.float16)
        for b in range(0, n, BLOCK):
            blk = np.asarray(rg[f"{key}/data"][b:b + BLOCK])
            for j in range(len(blk)):
                a[b + j] = D.fine_s1_scene(blk[j], fs).astype(np.float16)
        s1[key] = a
        rep[key] = n

    dem = (D.fine_dem_static(rg["dem/data"][0], fs) if raw["has_dem"]
           else np.zeros((2, 112, 112), np.float32)).astype(np.float16)
    ly = raw["lulc_years"]
    lulc = (np.stack([D.lulc_remap(rg["lulc/data"][i]) for i in range(len(ly))]) if len(ly)
            else np.zeros((0, 224, 224), np.uint8))

    if len(cand):
        _save(out / "fine_s2.npy", s2)
    for key, a in s1.items():
        if len(a):
            _save(out / f"fine_{key}.npy", a)
    tmp = out / "fine_meta.tmp.npz"
    np.savez(tmp, s2_cand=np.array(cand, dtype=np.int64).reshape(-1, 3),
             s1_asc_dates=np.asarray(raw["s1_asc"], np.int64),
             s1_desc_dates=np.asarray(raw["s1_desc"], np.int64),
             dem=dem, lulc_years=np.asarray(ly, np.int32), lulc=lulc,
             fine_stats_sha=np.array(D._fine_stats_sha()))
    tmp.rename(out / "fine_meta.npz")
    rep["sec"] = round(time.time() - t0, 1)
    return st, "ok", rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tasks = []
    for cat_dir in sorted(p for p in D.CACHE_ROOT_GPFS.iterdir() if p.is_dir()):
        for st_dir in sorted(p for p in cat_dir.iterdir() if (p / "pyr.npz").exists()):
            if a.stations and st_dir.name not in a.stations:
                continue
            tasks.append((cat_dir.name, st_dir.name, a.force))
    if a.limit:
        tasks = tasks[: a.limit]
    print(f"prepare_fine_cache: {len(tasks)} stations, {a.workers} workers", flush=True)
    counts, t0 = {}, time.time()
    with Pool(a.workers) as p:
        for i, (st, status, rep) in enumerate(p.imap_unordered(one, tasks), 1):
            counts[status] = counts.get(status, 0) + 1
            if status != "skip_done":
                print(f"  [{i}/{len(tasks)}] {st:40s} {status} {rep}", flush=True)
    print(f"done in {time.time() - t0:.0f} s: {counts}")
    return 0 if set(counts) <= {"ok", "skip_done"} else 1


if __name__ == "__main__":
    sys.exit(main())
