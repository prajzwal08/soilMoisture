"""
preflight_s48.py — fail-loud pre-flight for the §48 training inputs (review B5)
================================================================================
slurm/train.sh's store check covers only the scratch token zarr. The §48 dataset also reads
  CACHE_ROOT/{cat}/{station}/   pyr.npz, s2_l12.npy, s2_cm.npy, s1_{asc,desc}_l12.npy
  RAW_ROOT/{station}.zarr       raw S2/S1/DEM for the fine CNN (build_fine)
and a missing one surfaces only as a per-station skip AFTER every rank has built its
datasets (§51.1 then raises). This checks every train/val/oos station in seconds.

Also prints stored TWSA stamps for a few stations: review A1 assumes GRACE months are
stamped on the 1st (time_start); dataset.py now admits a month only from stamp + 45 d.

Exit non-zero on any missing input. Usage: python preflight_s48.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from dataset import CACHE_ROOT, RAW_ROOT, ZARR_ROOT  # noqa: E402
from splits_config import SM_CATEGORIES, category_of, station_dir_name  # noqa: E402

CACHE_FILES = ("pyr.npz", "s2_l12.npy", "s2_cm.npy")


def main() -> int:
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    df = df[df["cat"].isin(SM_CATEGORIES) & df["split"].isin(["train", "val", "oos"])]
    bad = []
    seen = set()
    for _, r in df.iterrows():
        st, cat = station_dir_name(r), r["cat"]
        if st in seen:
            continue
        seen.add(st)
        c = CACHE_ROOT / cat / st
        miss = [f for f in CACHE_FILES if not (c / f).is_file() or (c / f).stat().st_size == 0]
        if miss:
            bad.append((st, r["split"], f"s48cache missing {miss}"))
        if not (RAW_ROOT / f"{st}.zarr" / "s2" / "dates" / ".zarray").is_file():
            bad.append((st, r["split"], "raw satellite_zarr s2/dates missing"))
    print(f"preflight_s48: {len(seen)} train/val/oos stations checked; {len(bad)} problems")
    for b in bad[:40]:
        print("   ", *b)
    by = pd.Series([b[1] for b in bad]).value_counts().to_dict() if bad else {}
    print(f"    by split: {by}")

    import zarr
    for st, cat in list(zip(df.apply(station_dir_name, axis=1), df["cat"]))[:3]:
        try:
            d = zarr.open_group(str(ZARR_ROOT / cat / st), mode="r")["twsa/date_ints"][:]
            days = sorted(set(int(x) % 100 for x in d))
            print(f"    TWSA stamps {st}: first {list(d[:3])}  day-of-month values {days[:5]}")
        except Exception as e:  # noqa: BLE001
            print(f"    TWSA stamps {st}: unreadable ({e!r})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
