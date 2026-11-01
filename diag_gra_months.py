#!/usr/bin/env python
"""Which summer months actually survive consolidation, per multi-station GRA cluster.

pick_summer_dates() kept only 2 of 4 months for TxSON_6st_01 even at a 75% valid floor,
so the binding constraint is not the floor. This prints the month x validity table the
choice should be made from.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from plot_gra_thermal import load_bundle, _as_str

C = pd.read_csv("csvs/gra_thermal_clusters.csv")
C = C[C.n_stations >= 2]
print(f"{len(C)} multi-station GRA clusters\n")
for _, cl in C.iterrows():
    z, p = load_bundle(cl.rep_folder)
    if z is None:
        print(f"{cl.cluster_id}: NO BUNDLE")
        continue
    days = np.array([_as_str(d)[:10] for d in z["day_utc"]])
    mon = np.array([int(d[5:7]) for d in days])
    al = z["grid_aligned"] == 1
    vf = z["n_valid_px"] / 1024.0
    print(f"{cl.cluster_id}  rep {cl.rep_station}  {len(days)} pairs, {int(al.sum())} aligned")
    line = [f"{m:02d}:n={int(((mon == m) & al).sum())},best={vf[(mon == m) & al].max():.2f}"
            for m in range(1, 13) if ((mon == m) & al).any()]
    print("   " + "  ".join(line))
    jjas = (mon >= 6) & (mon <= 9) & al
    for thr in (0.90, 0.75, 0.50, 0.25, 0.01):
        print(f"     thr {thr:.2f} -> JJAS months {sorted(set(mon[jjas & (vf >= thr)].tolist()))}")
    warm = (mon >= 4) & (mon <= 10) & al
    print(f"     Apr-Oct at thr 0.75 -> {sorted(set(mon[warm & (vf >= 0.75)].tolist()))}")
    print()
