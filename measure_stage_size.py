"""
measure_stage_size.py — how big is the /dev/shm training set? (2026-09-29, read-only)
======================================================================================
Training (train + val stations, years 2016-2022) is data-bound on GPFS small random reads.
The fix under consideration stages the per-sample data into /dev/shm (378 GB per node).
This measures, for train+val stations and scenes dated <= 2022-12-31 (the last training
day; the 365-day window only looks back), the bytes of:

  anchor L12   {s2,s1_asc,s1_desc}_l12 rows              196 x 768 fp16    = 301,056 B / scene
  fine S2      10 bands + valid @ 112 (cm folded in)       11 x 112^2 fp16  = 275,968 B / scene
  fine S1      VV, VH, valid @ 112, per orbit               3 x 112^2 fp16  =  75,264 B / scene
  pyr.npz      already loaded into RAM at init (reported for completeness)

and, for comparison, the same with no year cut and with OOS stations included.
Output: printed table + csvs/stage_size.csv (one row per station).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from splits_config import SM_CATEGORIES, category_of, station_dir_name  # noqa: E402

CACHE = Path("/gpfs/scratch1/shared/pkhanal/s48cache")
RAW = Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr")
CUT = 20221231
B_L12, B_FS2, B_FS1 = 196 * 768 * 2, 11 * 112 * 112 * 2, 3 * 112 * 112 * 2


def _dates(arr):
    return np.array([int((bytes(x).decode() if isinstance(x, (bytes, np.bytes_)) else str(x))[:8])
                     for x in arr], dtype=np.int64)


def one(args):
    st, cat, split = args
    import zarr
    rec = dict(station=st, split=split)
    p = CACHE / cat / st / "pyr.npz"
    if not p.exists():
        return {**rec, "missing": True}
    with np.load(p) as z:
        di = {o: np.asarray(z[f"{o}_date_ints"]) for o in ("s2", "s1_asc", "s1_desc")
              if f"{o}_date_ints" in z.files}
    rec["pyr_bytes"] = p.stat().st_size
    for o, d in di.items():
        rec[f"l12_{o}_all"], rec[f"l12_{o}_cut"] = len(d), int((d <= CUT).sum())
    try:
        g = zarr.open_group(str(RAW / f"{st}.zarr"), mode="r")
        for o in ("s2", "s1_asc", "s1_desc"):
            if f"{o}/dates" in g:
                d = _dates(g[f"{o}/dates"][:])
                rec[f"raw_{o}_all"], rec[f"raw_{o}_cut"] = len(d), int((d <= CUT).sum())
    except Exception as e:  # noqa: BLE001
        rec["raw_error"] = repr(e)[:100]
    return rec


def main():
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    df = df[df["cat"].isin(SM_CATEGORIES) & df["split"].isin(["train", "val", "oos"])]
    tasks, seen = [], set()
    for _, r in df.iterrows():
        st = station_dir_name(r)
        if st not in seen:
            seen.add(st)
            tasks.append((st, r["cat"], r["split"]))
    with Pool(64) as p:
        res = pd.DataFrame(p.map(one, tasks))
    for col in ("missing", "raw_error"):
        if col not in res:
            res[col] = 0
    res = res.fillna(0)
    res.to_csv(REPO / "csvs" / "stage_size.csv", index=False)

    def gb(sub, suffix):
        l12 = sum(sub.get(f"l12_{o}_{suffix}", 0).sum() for o in ("s2", "s1_asc", "s1_desc")) * B_L12
        fs2 = sub.get(f"raw_s2_{suffix}", 0).sum() * B_FS2
        fs1 = (sub.get(f"raw_s1_asc_{suffix}", 0).sum() + sub.get(f"raw_s1_desc_{suffix}", 0).sum()) * B_FS1
        return l12 / 1e9, fs2 / 1e9, fs1 / 1e9

    print(f"stations: {res.split.value_counts().to_dict()}   missing cache: "
          f"{int(res['missing'].astype(bool).sum())}   raw errors: "
          f"{int((res['raw_error'] != 0).sum())}")
    print(f"\n{'set':<34s} {'anchor L12':>11s} {'fine S2':>9s} {'fine S1':>9s} {'TOTAL':>9s}  (GB)")
    for name, sub, sfx in (("train+val, <= 2022  (proposed)", res[res.split.isin(["train", "val"])], "cut"),
                           ("train+val, all years", res[res.split.isin(["train", "val"])], "all"),
                           ("train+val+oos, all years", res, "all")):
        a, s2, s1 = gb(sub, sfx)
        print(f"{name:<34s} {a:11.1f} {s2:9.1f} {s1:9.1f} {a + s2 + s1:9.1f}")
    print(f"\npyr.npz (RAM at init, train+val): "
          f"{res[res.split.isin(['train', 'val'])].pyr_bytes.sum() / 1e9:.1f} GB")
    print("/dev/shm per GPU node: 378 GB; node RAM 755 GB")


if __name__ == "__main__":
    main()
