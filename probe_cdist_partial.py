#!/usr/bin/env python
"""Does a CDIST filter mask whole scenes, or carve partial tiles?

ST_CDIST is a PER-PIXEL raster: for each 30 m pixel it holds the distance from THAT pixel to the
nearest cloud.  So `cdist > T` is tested independently at all 5,776 pixels, and a cloud sitting
~T away from the tile produces a PARTIAL mask -- near side fails, far side passes.

That matters for a dense spatial target.  A partial mask is spatially systematic: supervision
density would depend on which side of the tile the weather was on, and a 22x22 head can fit that
as a positional artefact.  An all-or-nothing scene mask cannot.

I previously inferred "nearly all-or-nothing" from 74.9% of pixels surviving against 73.7% of
scenes retaining at least one pixel.  That is suggestive but not proof -- clearer scenes have
both more clear pixels AND greater cloud distance, which would inflate the pixel figure
independently.  So this measures the per-scene passing fraction directly.

If the distribution piles up at 0 and 1, the filter is a scene gate.  If it is spread across the
middle, partial masking is real and the spatial bias needs handling.

OUTPUT  csvs/landsat_cdist_partial.csv, fig/landsat_st30_check/cdist_partial.png
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, qa_decode

FIG = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/landsat_st30_check")
OUT = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_cdist_partial.csv")
T = [0.5, 1.0]
BINS = np.linspace(0, 1, 51)


def one(p):
    z = np.load(p, allow_pickle=False)
    lst, cd = z["lst30"], z["cdist30"]
    clear, _ = qa_decode(z["qa_pixel30"].astype("float64"))
    base = clear & np.isfinite(lst) & np.isfinite(cd)
    n = lst.shape[0]
    nb = base.reshape(n, -1).sum(axis=1)
    out = {}
    for t in T:
        np_ = (base & (cd > t)).reshape(n, -1).sum(axis=1)
        live = nb > 0
        frac = np.divide(np_, nb, out=np.zeros(n), where=live)[live]
        out[t] = np.histogram(frac, bins=BINS)[0]
        # a scene is "partial" if it keeps some but not nearly all of its clear pixels
        out[f"part_{t}"] = int(((frac > 0.02) & (frac < 0.98)).sum())
        out[f"full_{t}"] = int((frac >= 0.98).sum())
        out[f"none_{t}"] = int((frac <= 0.02).sum())
    out["n_live"] = int((nb > 0).sum())
    return out


def main():
    paths = sorted(str(p) for p in DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    with Pool(48) as pool:
        res = pool.map(one, paths, chunksize=2)

    H = {t: np.sum([r[t] for r in res], axis=0) for t in T}
    tot = sum(r["n_live"] for r in res)
    rows = []
    print("=" * 84)
    print(f"IS THE CDIST FILTER A SCENE GATE OR A PARTIAL MASK?  {tot:,} scenes with clear pixels")
    print("=" * 84)
    print(f"\n{'CDIST >':>9}{'fully kept':>14}{'%':>8}{'PARTIAL':>12}{'%':>8}"
          f"{'fully cut':>12}{'%':>8}")
    for t in T:
        f = sum(r[f"full_{t}"] for r in res)
        p = sum(r[f"part_{t}"] for r in res)
        n = sum(r[f"none_{t}"] for r in res)
        rows.append({"threshold_km": t, "full": f, "partial": p, "none": n, "n_scenes": tot})
        print(f"{t:>7g} km{f:>14,}{100*f/tot:>8.1f}{p:>12,}{100*p/tot:>8.1f}{n:>12,}{100*n/tot:>8.1f}")
    pd.DataFrame(rows).to_csv(OUT, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    c = (BINS[:-1] + BINS[1:]) / 2
    for ax, t in zip(axes, T):
        h = H[t] / H[t].sum()
        ax.bar(c, h, width=0.018, color="tab:blue")
        f = 100 * sum(r[f"full_{t}"] for r in res) / tot
        p = 100 * sum(r[f"part_{t}"] for r in res) / tot
        n = 100 * sum(r[f"none_{t}"] for r in res) / tot
        ax.set_title(f"CDIST > {t:g} km\nfully kept {f:.1f}%   PARTIAL {p:.1f}%   fully cut {n:.1f}%",
                     fontsize=10)
        ax.set_xlabel("fraction of a scene's clear pixels that survive")
        ax.set_ylabel("share of scenes"); ax.grid(alpha=.3)
    fig.suptitle("A scene gate piles up at 0 and 1.  Mass in the middle is partial masking.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "cdist_partial.png", dpi=120)
    print(f"\nwrote {OUT}\nwrote {FIG/'cdist_partial.png'}")
    print("=" * 84)


if __name__ == "__main__":
    main()
