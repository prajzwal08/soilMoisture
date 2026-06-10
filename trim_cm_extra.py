"""
trim_cm_extra.py

For stations where cm/dates has more entries than s2/dates (cm computed on a
broader date set than the final S2 token set), trim cm/masks + cm/dates down
to exactly the dates present in s2/dates, and archive the removed
(date, mask) pairs.

Every station in csvs/audit_zarr_complete.csv with n_cm_dates > n_s2_dates has
cm_coverage_pct == 100% (i.e. cm/dates is a strict superset of s2/dates) —
the extra entries are unused by dataset.py (which looks up cm by s2 date).
AmeriFlux_CA-Cbo is the opposite case (cm subset of s2) and is excluded here.

Archived extras: /projects/prjs1968/data/excluded_stations/_cm_extra/{station}_cm_extra.npz
  (dates: U8 array, masks: uint8 array [n_extra, H, W])

Usage:
    conda run --no-capture-output -n sensei python trim_cm_extra.py [--execute] [--workers N]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import zarr

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
ARCHIVE_DIR = Path("/projects/prjs1968/data/excluded_stations/_cm_extra")
COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
T_CM = 128


def _decode(arr):
    return np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                      for x in arr], dtype="U8")


def process(args):
    station, cat, execute = args
    token_dir = ZARR_ROOT / cat / station
    try:
        store = zarr.DirectoryStore(str(token_dir))
        zg = zarr.open_group(store=store, mode="a" if execute else "r")

        cm_dates = _decode(zg["cm/dates"][:])
        s2_dates = set(_decode(zg["s2/dates"][:]))

        keep_mask = np.array([d in s2_dates for d in cm_dates])
        n_extra = (~keep_mask).sum()
        if n_extra == 0:
            return f"SKIP   {station}: no extra cm dates"

        if not execute:
            return f"DRY    {station}: n_cm={len(cm_dates)} n_s2={len(s2_dates)} n_extra={n_extra}"

        cm_masks = zg["cm/masks"][:]

        # Archive extras
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ARCHIVE_DIR / f"{station}_cm_extra.npz",
            dates=cm_dates[~keep_mask],
            masks=cm_masks[~keep_mask],
        )

        # Trim and overwrite
        kept_dates = cm_dates[keep_mask]
        kept_masks = cm_masks[keep_mask]
        N = len(kept_dates)
        H, W = kept_masks.shape[1], kept_masks.shape[2]
        zg.array("cm/masks", kept_masks, chunks=(min(T_CM, N), H, W),
                 dtype=np.uint8, compressor=COMPRESSOR, overwrite=True)
        zg.array("cm/dates", kept_dates, chunks=(N,), overwrite=True)
        zarr.consolidate_metadata(store)

        return f"OK     {station}: removed {n_extra} extra cm dates ({len(cm_dates)} -> {N})"
    except Exception as exc:
        import traceback
        return f"ERR    {station}: {exc}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    audit = pd.read_csv("csvs/audit_zarr_complete.csv")
    targets = audit[
        (audit["station"] != "AmeriFlux_CA-Cbo")
        & (audit["n_cm_dates"] > audit["n_s2_dates"])
    ]
    print(f"{len(targets)} stations with extra cm dates")

    jobs = [(r["station"], r["category"], args.execute) for _, r in targets.iterrows()]

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process, jobs)):
            print(f"[{i+1:4d}/{len(jobs)}] {result}", flush=True)


if __name__ == "__main__":
    main()
