"""
verify_fine_cache.py — prove the fast data path changes no input value (2026-09-29)
====================================================================================
  1  PRECOMPUTE: for --stations random stations x --dates random (year, doy) in 2016-2025,
     build_fine(raw, cache WITHOUT fine_*)  ==  build_fine(raw, cache WITH fine_*)
     (fine fp16 tensor bit-equal, lulc equal, info equal). The first reads raw zarr as training
     did before; the second only looks rows up.
  2  STAGING: stage --stage-stations train stations with stage_shm.one into a temp dir, then for
     random dates in 2016-2022 compare, staged vs GPFS cache: build_fine, select_anchor,
     load_history (s2 and s1) — every tensor bit-equal.
Exit non-zero on any mismatch. Read-only apart from the temp dir it deletes.
"""
import argparse
import random
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import dataset as D  # noqa: E402
import stage_shm  # noqa: E402
from splits_config import SM_CATEGORIES, category_of, station_dir_name  # noqa: E402


def _eq(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _dates(rng, n, y0, y1):
    return [(rng.randint(y0, y1), rng.randint(1, 365)) for _ in range(n)]


def part1(args):
    cat, st, n, seed = args
    d = D.CACHE_ROOT_GPFS / cat / st
    raw = D._open_raw(st)
    c_raw, c_pre = D._load_station_cache(d, fine=False), D._load_station_cache(d, fine=True)
    if raw is None or c_raw is None or "fine_s2" not in c_pre:
        return st, 0, ["missing raw/cache/fine"]
    fs, rng, bad = D._load_fine_stats(), random.Random(seed), []
    dates = _dates(rng, n, 2016, 2025)
    for y, doy in dates:
        a, b = D.build_fine(raw, c_raw, y, doy, fs), D.build_fine(raw, c_pre, y, doy, fs)
        if not _eq(list(a), list(b)):
            bad.append(f"{y}-{doy:03d}")
    return st, len(dates), bad


def part2(args):
    cat, st, tmp, n, seed = args
    stage_shm.one((cat, st, tmp, 20221231))
    raw = D._open_raw(st)
    g = D._load_station_cache(D.CACHE_ROOT_GPFS / cat / st)
    s = D._load_station_cache(Path(tmp) / cat / st)
    if s is None or g is None:
        return st, 0, ["stage failed"]
    fs, rng, bad = D._load_fine_stats(), random.Random(seed), []
    dates = _dates(rng, n, 2016, 2022)
    for y, doy in dates:
        checks = {
            "fine": (D.build_fine(raw, g, y, doy, fs), D.build_fine(raw, s, y, doy, fs)),
            "anchor": (D.select_anchor(g, y, doy), D.select_anchor(s, y, doy)),
            "hist_s2": (D.load_history(g, ("s2",), y, doy, D.MAX_S2),
                        D.load_history(s, ("s2",), y, doy, D.MAX_S2)),
            "hist_s1": (D.load_history(g, ("s1_asc", "s1_desc"), y, doy, D.MAX_S1),
                        D.load_history(s, ("s1_asc", "s1_desc"), y, doy, D.MAX_S1)),
        }
        for k, (x, z) in checks.items():
            if not _eq(list(x), list(z)):
                bad.append(f"{k}@{y}-{doy:03d}")
    return st, len(dates), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, default=40)
    ap.add_argument("--dates", type=int, default=25)
    ap.add_argument("--stage-stations", type=int, default=10)
    a = ap.parse_args()
    rng = random.Random(0)
    all_st = [(c.name, s.name) for c in sorted(D.CACHE_ROOT_GPFS.iterdir()) if c.is_dir()
              for s in sorted(c.iterdir()) if (s / "fine_meta.npz").exists()]
    pick = rng.sample(all_st, min(a.stations, len(all_st)))
    with Pool(32) as p:
        r1 = p.map(part1, [(c, s, a.dates, i) for i, (c, s) in enumerate(pick)])
    n1 = sum(r[1] for r in r1)
    b1 = [(r[0], r[2]) for r in r1 if r[2]]
    print(f"1 PRECOMPUTE: {len(pick)} stations, {n1} samples, {len(b1)} stations with mismatches")
    for st, b in b1[:10]:
        print("   ", st, b[:5])

    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    tr = df[df["cat"].isin(SM_CATEGORIES) & (df["split"] == "train")]
    trs = sorted({(r["cat"], station_dir_name(r)) for _, r in tr.iterrows()
                  if (D.CACHE_ROOT_GPFS / r["cat"] / station_dir_name(r) / "fine_meta.npz").exists()})
    tmp = tempfile.mkdtemp(prefix="s48stage_verify_", dir="/dev/shm")
    try:
        sp = rng.sample(trs, min(a.stage_stations, len(trs)))
        with Pool(min(10, len(sp))) as p:
            r2 = p.map(part2, [(c, s, tmp, a.dates, 100 + i) for i, (c, s) in enumerate(sp)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n2 = sum(r[1] for r in r2)
    b2 = [(r[0], r[2]) for r in r2 if r[2]]
    print(f"2 STAGING: {len(sp)} stations, {n2} dates x 4 checks, {len(b2)} stations with mismatches")
    for st, b in b2[:10]:
        print("   ", st, b[:5])
    ok = not b1 and not b2 and n1 > 0 and n2 > 0
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
