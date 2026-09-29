"""
stage_shm.py — copy the training slice of the §48 cache into /dev/shm (2026-09-29)
===================================================================================
GPFS small random reads capped the loader at ~40 samples/s per node. Training reads the
anchor L12 rows and the precomputed fine rows per sample, so those go into RAM:

  train + val stations only (OOS is eval-only), scenes dated <= --cut (default 2022-12-31:
  the last training day; every window only looks back from D), from CACHE_ROOT_GPFS to --dest:

    pyr.npz             per-orbit arrays sliced to kept scenes; dem/lulc pyramids as-is
    {orbit}_l12.npy     kept rows
    fine_s2.npy         kept S2 candidates, tok_row remapped to the sliced token index
    fine_s1_{orbit}.npy kept passes
    fine_meta.npz       sliced / remapped to match
  s2_cm.npy is NOT staged: with the precomputed fine rows build_fine never reads it.

pyr.npz is written LAST per station (the loader's completion marker). Measured 2026-09-29:
~186 GB for 662 stations (/dev/shm is 378 GB). Exit non-zero if any station fails.

Usage: python stage_shm.py --dest /dev/shm/s48_$SLURM_JOB_ID [--cut 20221231] [--workers 32]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.environ.pop("S48_CACHE_ROOT", None)          # source is always the GPFS cache
import dataset as D  # noqa: E402
from splits_config import SM_CATEGORIES, category_of, station_dir_name  # noqa: E402

ORBITS = ("s2", "s1_asc", "s1_desc")


def one(args):
    cat, st, dest, cut = args
    src, dst = D.CACHE_ROOT_GPFS / cat / st, Path(dest) / cat / st
    if not (src / "pyr.npz").exists():
        return st, "no_cache", 0
    if not (src / "fine_meta.npz").exists():
        return st, "no_fine", 0
    dst.mkdir(parents=True, exist_ok=True)
    nbytes = 0
    with np.load(src / "pyr.npz") as z:
        pyr = {k: z[k] for k in z.files}
    tok_map = {}
    for o in ORBITS:
        if f"{o}_date_ints" not in pyr:
            continue
        keep = pyr[f"{o}_date_ints"] <= cut
        tok_map[o] = np.where(keep, np.cumsum(keep) - 1, -1)
        for k in (f"{o}_pyr", f"{o}_nvalid", f"{o}_date_ints"):
            pyr[k] = pyr[k][keep]
        f = src / f"{o}_l12.npy"
        if f.exists():
            a = np.load(f, mmap_mode="r")[np.where(keep)[0]]
            np.save(dst / f"{o}_l12.npy", a)
            nbytes += a.nbytes
    with np.load(src / "fine_meta.npz") as z:
        meta = {k: z[k] for k in z.files}
    cand = meta["s2_cand"]
    ks = np.where(cand[:, 0] <= cut)[0] if len(cand) else np.zeros(0, int)
    new_cand = cand[ks].copy()
    if len(new_cand):
        new_cand[:, 2] = tok_map["s2"][new_cand[:, 2]]
        if (new_cand[:, 2] < 0).any():
            return st, "tok_remap_error", 0
        a = np.load(src / "fine_s2.npy", mmap_mode="r")[ks]
        np.save(dst / "fine_s2.npy", a)
        nbytes += a.nbytes
    meta["s2_cand"] = new_cand
    for o in ("s1_asc", "s1_desc"):
        d = meta[f"{o}_dates"]
        keep = np.where(d <= cut)[0]
        meta[f"{o}_dates"] = d[keep]
        if len(keep):
            a = np.load(src / f"fine_{o}.npy", mmap_mode="r")[keep]
            np.save(dst / f"fine_{o}.npy", a)
            nbytes += a.nbytes
    np.savez(dst / "fine_meta.npz", **meta)
    np.savez(dst / "pyr.npz", **pyr)                 # LAST: completion marker
    return st, "ok", nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--cut", type=int, default=20221231)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    df = df[df["cat"].isin(SM_CATEGORIES) & df["split"].isin(a.splits)]
    tasks, seen = [], set()
    for _, r in df.iterrows():
        st = station_dir_name(r)
        if st not in seen:
            seen.add(st)
            tasks.append((r["cat"], st, a.dest, a.cut))
    if a.limit:
        tasks = tasks[: a.limit]
    t0 = time.time()
    print(f"stage_shm: {len(tasks)} stations ({'+'.join(a.splits)}), scenes <= {a.cut} -> {a.dest}",
          flush=True)
    counts, total, bad = {}, 0, []
    with Pool(a.workers) as p:
        for st, status, nb in p.imap_unordered(one, tasks):
            counts[status] = counts.get(status, 0) + 1
            total += nb
            if status != "ok":
                bad.append((st, status))
    print(f"stage_shm: {counts}  {total / 1e9:.1f} GB in {time.time() - t0:.0f} s")
    for b in bad[:20]:
        print("   FAILED", *b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
