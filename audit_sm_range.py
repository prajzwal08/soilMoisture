"""
Audit all sm_only zarr stations for soil moisture values outside [0, 1].
Reports station name, depth, min, max, and count of out-of-range observed (qc==0) values.
"""
import zarr
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import pandas as pd

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr/sm_only")
N_WORKERS = 64


def audit_station(station_dir: Path):
    try:
        zg = zarr.open_group(str(station_dir), mode="r")
        if "labels" not in zg:
            return None

        sm    = zg["labels/sm"][:]
        qc    = zg["labels/qc"][:] if "labels/qc" in zg else None
        depths = list(zg["labels/depths"][:])

        # align qc if trimmed (trim_pre2016 may have trimmed sm but not qc)
        if qc is not None and qc.shape[1] != sm.shape[1]:
            qc = qc[:, -sm.shape[1]:]

        results = []
        for i, depth in enumerate(depths):
            sm_i = sm[i].astype(float)
            if qc is not None:
                sm_i[qc[i] != 0] = np.nan  # observed only

            valid = sm_i[~np.isnan(sm_i)]
            if len(valid) == 0:
                continue

            out_of_range = valid[(valid < 0.0) | (valid > 1.0)]
            if len(out_of_range) > 0:
                results.append({
                    "station": station_dir.name,
                    "depth": depth,
                    "sm_min": float(np.nanmin(sm_i)),
                    "sm_max": float(np.nanmax(sm_i)),
                    "n_total_obs": len(valid),
                    "n_out_of_range": len(out_of_range),
                    "pct_out_of_range": round(100 * len(out_of_range) / len(valid), 1),
                })
        return results if results else None

    except Exception as e:
        return [{"station": station_dir.name, "depth": "ERROR", "sm_min": None,
                 "sm_max": None, "n_total_obs": None, "n_out_of_range": None,
                 "pct_out_of_range": None, "error": str(e)}]


if __name__ == "__main__":
    stations = sorted(ZARR_ROOT.iterdir())
    print(f"Auditing {len(stations)} sm_only stations with {N_WORKERS} workers...")

    with Pool(N_WORKERS) as pool:
        results = pool.map(audit_station, stations)

    rows = [r for res in results if res for r in res]

    if not rows:
        print("All stations have SM values within [0, 1]. No issues found.")
    else:
        df = pd.DataFrame(rows).sort_values("sm_max", ascending=False)
        print(f"\nFound {len(df)} depth-station combinations with out-of-range SM:\n")
        print(df.to_string(index=False))
        out = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/audit_sm_range.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")
