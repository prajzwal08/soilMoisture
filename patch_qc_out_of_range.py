"""
Patch labels/qc in zarr stores for sm_only stations where observed SM (qc==0)
is outside [0, 1]. Sets those days to qc=3 (out-of-range, excluded from training).

QC flag scheme:
  0 = observed            (used for training)
  1 = gap-filled          (excluded)
  2 = still missing       (excluded)
  3 = out-of-range [0,1]  (excluded — added by this script)

Only modifies the 25 stations flagged in csvs/audit_sm_range.csv.
Run with: conda run -n terramind python patch_qc_out_of_range.py
"""

import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr/sm_only")
AUDIT_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_sm_range.csv")
N_WORKERS = 64


def patch_station(station: str) -> dict:
    station_dir = ZARR_ROOT / station
    try:
        zg = zarr.open_group(str(station_dir), mode="r+")

        sm     = zg["labels/sm"][:]          # (n_depths, n_days)
        qc     = zg["labels/qc"][:]          # (n_depths, n_days) uint8
        depths = list(zg["labels/depths"][:])

        # align qc to sm if trim_pre2016 trimmed sm but not qc
        qc_offset = 0
        if qc.shape[1] != sm.shape[1]:
            qc_offset = qc.shape[1] - sm.shape[1]
            qc_aligned = qc[:, qc_offset:]
        else:
            qc_aligned = qc

        total_patched = 0
        depth_summary = []

        for i, depth in enumerate(depths):
            sm_i  = sm[i].astype(float)
            qc_i  = qc_aligned[i].copy()

            # days that are observed (qc==0) AND out of range
            bad_mask = (qc_i == 0) & ((sm_i < 0.0) | (sm_i > 1.0))
            n_bad = int(bad_mask.sum())

            if n_bad > 0:
                qc_i[bad_mask] = 3
                # write back — accounting for offset in original qc array
                full_qc_row = qc[i].copy()
                full_qc_row[qc_offset:][bad_mask] = 3
                zg["labels/qc"][i] = full_qc_row
                total_patched += n_bad
                depth_summary.append(f"{depth}:{n_bad}")

        return {
            "station": station,
            "status": "patched" if total_patched > 0 else "clean",
            "n_patched": total_patched,
            "depths": ", ".join(depth_summary),
        }

    except Exception as e:
        return {"station": station, "status": "ERROR", "n_patched": 0, "depths": str(e)}


if __name__ == "__main__":
    audit = pd.read_csv(AUDIT_CSV)
    stations = list(audit["station"].unique())
    print(f"Patching {len(stations)} stations (QC=0 & SM∉[0,1] → QC=3)...\n")

    with Pool(N_WORKERS) as pool:
        results = pool.map(patch_station, stations)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    total = df["n_patched"].sum()
    errors = df[df["status"] == "ERROR"]
    print(f"\nTotal days patched to QC=3: {total}")
    if len(errors):
        print(f"Errors ({len(errors)} stations):")
        print(errors[["station", "depths"]].to_string(index=False))
    else:
        print("No errors.")

    # Save patch log
    out = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/patch_qc_log.csv")
    df.to_csv(out, index=False)
    print(f"\nLog saved to {out}")
