#!/usr/bin/env python
"""Does the CDIST filter inherit the climate bias that disqualified the ST_QA gate?

The ST_QA gate was rejected partly because `ST_QA <= 3 K` retained 0.725 of arid (B) supervision
against 0.243 of tropical (A) -- a 3x skew hitting the two climates already least represented.
CDIST predicts ST_QA almost deterministically, so it could inherit exactly that skew: if cloud
proximity and high ST_QA both merely track a humid atmosphere, then filtering on either one is
filtering on climate.  Reads csvs/landsat_stqa_cdist.csv, nothing else.
"""
import pandas as pd

D = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_stqa_cdist.csv"
d = pd.read_csv(D)
g = d.groupby("kg_macro")
cols = [c for c in ("px_0.09", "px_0.3", "px_0.5", "px_1", "px_2") if c in d.columns]
ret = g[cols].sum().div(g["n_clear_px"].sum(), axis=0)
ret.insert(0, "n_stations", g.size())

print("CLEAR-PIXEL RETENTION BY KOPPEN MACRO-CLASS, CDIST filter\n")
print(ret.round(4).to_string())

mcols = [c for c in ("med_0", "med_1") if c in d.columns]
if len(mcols) == 2:
    m = g[mcols].median()
    m["shift_K"] = m["med_1"] - m["med_0"]
    print("\nresulting median ST_QA per class, no filter -> CDIST > 1 km:\n")
    print(m.round(3).to_string())

if "px_1" in ret.columns:
    s = ret["px_1"]
    print(f"\nCDIST > 1 km spread : {s.min():.3f} ({s.idxmin()}) to {s.max():.3f} ({s.idxmax()})"
          f"   ratio {s.max()/s.min():.2f}x")
if "px_0.5" in ret.columns:
    s = ret["px_0.5"]
    print(f"CDIST > 0.5 km spread: {s.min():.3f} ({s.idxmin()}) to {s.max():.3f} ({s.idxmax()})"
          f"   ratio {s.max()/s.min():.2f}x")
print("\nReference: the ST_QA <= 3 K gate spanned 0.243 (A) to 0.725 (B) -- a 2.98x skew.")
