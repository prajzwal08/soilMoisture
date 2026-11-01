#!/usr/bin/env python
"""Do the ECOSTRESS 32x32 window and the S2 224x224 window cover the same ground?

plot_dtr_txson.py asserts they are "co-located to within about half an LST pixel" because
they are cut from different grids -- S2 from bounds_utm in the station's own UTM zone,
ECOSTRESS from the MGRS grid it happens to land on.  That assertion has never been
checked, and the figures place station markers with eco_row = row * 10/70, which assumes
the two windows share an ORIGIN.  If they do not, every marker is off by up to one 70 m
pixel and the visual comparison is misaligned.

This measures the actual offset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from plot_gra_thermal import load_bundle, SAT_ZARR

C = pd.read_csv("csvs/gra_thermal_clusters.csv")
C = C[C.n_stations >= 2]

print("=== what the ECOSTRESS bundle actually stores ===")
z, p = load_bundle(C.iloc[0].rep_folder)
print(f"{p.name}")
for k in sorted(z.files):
    a = z[k]
    print(f"  {k:16s} shape={str(a.shape):16s} dtype={a.dtype}"
          + (f"  value={a.item()}" if a.ndim == 0 and a.size == 1 else ""))

print()
print("=== per cluster: is the station at the CENTRE of the ECOSTRESS window? ===")
print("If the window were cut station-centred like S2, the station would sit at the exact")
print("centre of the 32x32 block, i.e. 1120 m from each edge.  A block snapped onto the")
print("MGRS grid instead puts it anywhere within the central pixel.")
print()
rows = []
for _, cl in C.iterrows():
    z, _ = load_bundle(cl.rep_folder)
    if z is None:
        continue
    lat, lon = float(z["latitude"]), float(z["longitude"])
    att = json.loads((SAT_ZARR / f"{cl.rep_folder}.zarr" / ".zattrs").read_text())
    epsg = int(att["epsg"])
    w, s, e, n = [float(v) for v in att["bounds_utm"]]
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = fwd.transform(lon, lat)
    # S2 window: is the station at its centre?
    dx_s2, dy_s2 = x - (w + e) / 2, y - (s + n) / 2
    rows.append(dict(cluster=cl.cluster_id, folder=cl.rep_folder, epsg=epsg,
                     s2_off_x=dx_s2, s2_off_y=dy_s2,
                     eco_lat=lat, eco_lon=lon,
                     eco_side=int(z["side_px"]) if "side_px" in z.files else -1,
                     eco_px_m=float(z["pixel_size_m"]) if "pixel_size_m" in z.files else -1,
                     day_tile=str(z["day_tile"]) if "day_tile" in z.files else "?"))
T = pd.DataFrame(rows)
for _, r in T.iterrows():
    print(f"  {r.cluster:26s} EPSG:{r.epsg}  station offset from S2 window centre = "
          f"({r.s2_off_x:+7.1f}, {r.s2_off_y:+7.1f}) m   eco {r.eco_side}px @ {r.eco_px_m} m")

print()
print("=== the lat/lon each store thinks the station is at ===")
sp = pd.read_csv("csvs/station_splits.csv")
M = pd.read_csv("csvs/gra_thermal_members.csv")
for _, r in T.iterrows():
    mm = M[(M.cluster_id == r.cluster) & (M.is_rep == 1)].iloc[0]
    att = json.loads((SAT_ZARR / f"{r.folder}.zarr" / ".zattrs").read_text())
    d_lat = r.eco_lat - float(att.get("latitude", np.nan))
    d_lon = r.eco_lon - float(att.get("longitude", np.nan))
    m_per_deg = 111320.0
    print(f"  {r.cluster:26s} ECOSTRESS vs S2 zarr station lat/lon differ by "
          f"({d_lat*m_per_deg:+.1f}, {d_lon*m_per_deg*np.cos(np.deg2rad(r.eco_lat)):+.1f}) m")
