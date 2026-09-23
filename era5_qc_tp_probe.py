#!/usr/bin/env python
"""§45 probe: how negative is tp_sum, really?

era5_qc_all_stations.py flagged 966 of 990 stations for `tp_sum < -1e-9`. A count
cannot distinguish an IFS numerical artefact from a defect, so measure the
MAGNITUDE and compare it to the physical scale of the variable.
"""
from __future__ import annotations
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, zarr

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from era5_qc_all_stations import station_rows, ZARR_ROOT, IDX


def probe(task):
    folder, cat = task[0], task[1]
    try:
        p = ZARR_ROOT / cat / folder
        try:
            zg = zarr.open_consolidated(str(p), mode="r")
        except KeyError:
            zg = zarr.open_group(str(p), mode="r")
        tp = np.asarray(zg["era5/values"][:, IDX["tp_sum"]], dtype=np.float64)
        neg = tp[tp < 0]
        return {"station": folder, "n": tp.size, "tp_min": float(tp.min()),
                "tp_max": float(tp.max()), "tp_mean_pos": float(tp[tp > 0].mean()) if (tp > 0).any() else np.nan,
                "n_neg": int(neg.size),
                "n_neg_1e6": int((tp < -1e-6).sum()),
                "n_neg_1e4": int((tp < -1e-4).sum()),
                "n_neg_1e3": int((tp < -1e-3).sum()),
                "worst_neg": float(neg.min()) if neg.size else 0.0}
    except Exception as e:
        return {"station": folder, "n": 0, "tp_min": np.nan, "worst_neg": np.nan}


if __name__ == "__main__":
    rows = station_rows()
    with Pool(64) as pool:
        res = pool.map(probe, list(rows.itertuples(index=False, name=None)), chunksize=1)
    d = pd.DataFrame(res)
    print(f"{len(d)} stations\n")
    print("=== how negative does tp_sum get? (units: m of water per day) ===")
    print(f"  most negative value anywhere : {d['worst_neg'].min():.3e} m")
    print(f"  median station's worst_neg   : {d['worst_neg'].median():.3e} m")
    print(f"  typical POSITIVE daily tp    : {d['tp_mean_pos'].median():.3e} m")
    print(f"  ratio |worst neg| / typical  : {abs(d['worst_neg'].min()) / d['tp_mean_pos'].median():.2e}")
    print()
    print("=== station counts by how strict the threshold is ===")
    for col, lab in [("n_neg", "tp < 0"), ("n_neg_1e6", "tp < -1e-6 m (1 micron)"),
                     ("n_neg_1e4", "tp < -1e-4 m (0.1 mm)"), ("n_neg_1e3", "tp < -1e-3 m (1 mm)")]:
        print(f"  {lab:28s} {int((d[col] > 0).sum()):4d} stations, "
              f"{int(d[col].sum()):8d} station-days")
    print()
    print("=== 10 most negative stations ===")
    print(d.nsmallest(10, "worst_neg")[["station", "n", "worst_neg", "tp_min", "tp_max"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
