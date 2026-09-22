#!/usr/bin/env python
"""CDIST 0.5 km or 1 km?  The trade-off, resolved per Koppen macro-class.

The ST_QA gate was disqualified on climate skew: `ST_QA <= 3 K` retained 0.725 of arid (B)
supervision against 0.243 of tropical (A), a 2.98x spread hitting the two classes already least
represented.  CDIST predicts ST_QA almost deterministically, so it can inherit that skew, and the
only question left is where to put the threshold.

Two things this plots that the aggregate numbers hide:

  * SKEW IS NOT LINEAR IN THE THRESHOLD.  Retention curves for the five classes run together at
    low thresholds and fan out as the filter bites.  Where they start to separate IS the answer,
    and it is not visible in any single-threshold table.

  * A AND E ARE 14 OF 993 STATIONS.  A large RELATIVE loss there is a small ABSOLUTE loss of
    supervision, but it is exactly the loss that costs generalisation across climate.  Both views
    are plotted, because arguing from either alone gives a different answer.

OUTPUT  fig/landsat_st30_check/cdist_climate.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_stqa_cdist.csv")
FIG = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/landsat_st30_check")
NAMES = {"A": "A tropical", "B": "B arid", "C": "C temperate",
         "D": "D continental", "E": "E polar"}
MIN_SCENES = 20

d = pd.read_csv(SRC)
thr = sorted(float(c[3:]) for c in d.columns if c.startswith("px_"))
g = d.groupby("kg_macro")

ret = pd.DataFrame({f"{t:g}": g[f"px_{t:g}"].sum() / g["n_clear_px"].sum() for t in thr})
absol = pd.DataFrame({f"{t:g}": g[f"px_{t:g}"].sum() for t in thr})
meds = pd.DataFrame({f"{t:g}": g[f"med_{t:g}"].median() for t in thr})
drop = pd.DataFrame({f"{t:g}": g.apply(
    lambda x, t=t: int((x[f"scenes_with_median_{t:g}"] < MIN_SCENES).sum())) for t in thr})
n_st = g.size()

print("CLEAR-PIXEL RETENTION BY KOPPEN MACRO-CLASS\n")
print(ret.round(4).to_string())
print("\nSKEW (max/min across classes) AND OVERALL RETENTION\n")
print(f"{'CDIST >':>9}{'overall':>10}{'skew':>8}{'worst class':>14}"
      f"{'stations lost':>15}{'median ST_QA':>14}")
tot = d.n_clear_px.sum()
for t in thr:
    k = f"{t:g}"
    s = ret[k]
    overall = d[f"px_{k}"].sum() / tot
    lost = int((d[f"scenes_with_median_{k}"] < MIN_SCENES).sum())
    med = float(d[f"med_{k}"].median())
    lab = "none" if t == 0 else f"{t:g} km"
    print(f"{lab:>9}{overall:>10.3f}{s.max()/s.min():>8.2f}"
          f"{s.idxmin()+' '+format(s.min(),'.3f'):>14}{lost:>15}{med:>14.3f}")

# ---------------- figure ----------------
x = np.array(thr)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
cols = {"A": "tab:red", "B": "tab:orange", "C": "tab:green",
        "D": "tab:blue", "E": "tab:purple"}

def mark(ax):
    for v, c in ((0.5, "k"), (1.0, "k")):
        ax.axvline(v, color=c, ls=":", lw=1.1, alpha=.6)
    ax.grid(alpha=.3)

ax = axes[0, 0]
for cl in ret.index:
    ax.plot(x, ret.loc[cl].values, "o-", ms=3.5, color=cols.get(cl),
            label=f"{NAMES.get(cl, cl)}  (n={n_st[cl]})")
ax.plot(x, [d[f"px_{t:g}"].sum() / tot for t in thr], "k--", lw=2, label="all 993")
ax.set_xlabel("CDIST threshold (km)"); ax.set_ylabel("clear pixels retained")
ax.set_title("Retention by climate — where do the classes fan out?", fontsize=10)
ax.legend(fontsize=7); mark(ax)

ax = axes[0, 1]
skew = (ret.max() / ret.min()).values
ax.plot(x, skew, "o-", color="tab:red", lw=2)
ax.axhline(2.98, color="grey", ls="--", lw=1.2)
ax.text(x[-1], 2.98, "  ST_QA ≤ 3 K (2.98x, rejected)", va="center", ha="right",
        fontsize=8, color="grey")
for v in (0.5, 1.0):
    i = int(np.argmin(np.abs(x - v)))
    ax.annotate(f"{skew[i]:.2f}x", (x[i], skew[i]), textcoords="offset points",
                xytext=(6, 8), fontsize=9, fontweight="bold")
ax.set_xlabel("CDIST threshold (km)"); ax.set_ylabel("skew, max/min across classes")
ax.set_title("THE DECISION: climate skew vs threshold", fontsize=10)
mark(ax)

ax = axes[1, 0]
for cl in meds.index:
    ax.plot(x, meds.loc[cl].values, "o-", ms=3.5, color=cols.get(cl), label=NAMES.get(cl, cl))
ax.set_xlabel("CDIST threshold (km)"); ax.set_ylabel("median ST_QA (K)")
ax.set_title("Purity bought, by climate", fontsize=10)
ax.legend(fontsize=7); mark(ax)

ax = axes[1, 1]
bot = np.zeros(len(x))
for cl in absol.index:
    v = absol.loc[cl].values / 1e6
    ax.bar(x, v, bottom=bot, width=0.055, color=cols.get(cl), label=NAMES.get(cl, cl))
    bot += v
ax.set_xlabel("CDIST threshold (km)"); ax.set_ylabel("clear pixels retained (millions)")
ax.set_title("ABSOLUTE supervision — A and E are 14 of 993 stations", fontsize=10)
ax.legend(fontsize=7); mark(ax)

fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "cdist_climate.png", dpi=120)
print(f"\nwrote {FIG/'cdist_climate.png'}")
