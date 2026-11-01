#!/usr/bin/env python
"""What the surviving Landsat ST archive looks like: where, when, which climate, which land use.

Counts scenes that survive the decided QC -- no ST_QA gate, CDIST > 1 km -- which is NOT a
covariate read: `tile_min_cdist` in the bundle is the MINIMUM over clear pixels, so it answers
"are ALL pixels far from cloud", while a scene is usable if ANY pixel is.  So the cdist raster
has to be opened.

A scene counts as usable if at least one clear pixel passes, which is the definition behind the
133,211 figure.  Per-station-per-year counts come out of the same pass.

OUTPUT  csvs/landsat_inventory.csv          per station
        csvs/landsat_inventory_year.csv     per station x year
        fig/landsat_st30_check/inventory.png
"""
from __future__ import annotations

import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, SPLITS, qa_decode

warnings.filterwarnings("ignore", category=RuntimeWarning)
FIG = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/landsat_st30_check")
CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs")
CDIST_MIN_KM = 1.0


def one(p):
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    lst, cd = z["lst30"], z["cdist30"]
    clear, _ = qa_decode(z["qa_pixel30"].astype("float64"))
    base = clear & np.isfinite(lst) & np.isfinite(cd)
    n = lst.shape[0]
    keep = (base & (cd > CDIST_MIN_KM)).reshape(n, -1).sum(axis=1)
    years = np.array([int(str(d)[:4]) for d in z["dates"]])
    usable = keep > 0
    return {"station_id": meta["station_id"],
            "n_downloaded": int(n),
            "n_any_clear": int((base.reshape(n, -1).sum(axis=1) > 0).sum()),
            "n_usable": int(usable.sum()),
            "px_usable": int(keep.sum()),
            "years": years[usable]}


def main():
    paths = sorted(str(p) for p in DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    print(f"{len(paths)} bundles")
    with Pool(48) as pool:
        res = pool.map(one, paths, chunksize=2)

    sp = pd.read_csv(SPLITS)[["station_id", "latitude", "longitude", "kg_macro",
                              "igbp_macro", "IGBP", "network"]]
    sp["station_id"] = sp.station_id.astype(str)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "years"} for r in res])
    df["station_id"] = df.station_id.astype(str)
    df = df.merge(sp, on="station_id", how="left")
    df.to_csv(CSV / "landsat_inventory.csv", index=False)

    yr = pd.DataFrame([{"station_id": r["station_id"], "year": y, "n": c}
                       for r in res
                       for y, c in zip(*np.unique(r["years"], return_counts=True))])
    yr["station_id"] = yr.station_id.astype(str)
    yr = yr.merge(sp, on="station_id", how="left")
    yr.to_csv(CSV / "landsat_inventory_year.csv", index=False)

    tot = df.n_usable.sum()
    print("\n" + "=" * 86)
    print(f"SURVIVING ARCHIVE  (no ST_QA gate, CDIST > {CDIST_MIN_KM:g} km)")
    print("=" * 86)
    print(f"  stations              : {len(df)}")
    print(f"  scenes downloaded     : {df.n_downloaded.sum():,}")
    print(f"  scenes with any clear : {df.n_any_clear.sum():,}")
    print(f"  scenes USABLE         : {tot:,}")
    print(f"  clear px usable       : {df.px_usable.sum():,}")
    s = df.n_usable
    print(f"  per station           : min {s.min()}  p10 {s.quantile(.1):.0f}  "
          f"median {s.median():.0f}  p90 {s.quantile(.9):.0f}  max {s.max()}")
    for lo in (0, 10, 20, 50):
        print(f"  stations with <{lo:>3} scenes : {int((s < lo).sum()) if lo else 0}")

    print("\nBY KOPPEN MACRO-CLASS\n")
    g = df.groupby("kg_macro").agg(stations=("station_id", "count"),
                                   scenes=("n_usable", "sum"),
                                   median_per_station=("n_usable", "median"))
    g["pct_of_scenes"] = (100 * g.scenes / tot).round(1)
    print(g.to_string())

    print("\nBY IGBP MACRO-CLASS\n")
    g2 = df.groupby("igbp_macro").agg(stations=("station_id", "count"),
                                      scenes=("n_usable", "sum"),
                                      median_per_station=("n_usable", "median"))
    g2["pct_of_scenes"] = (100 * g2.scenes / tot).round(1)
    print(g2.to_string())

    print("\nBY IGBP CLASS\n")
    g3 = df.groupby("IGBP").agg(stations=("station_id", "count"),
                                scenes=("n_usable", "sum")).sort_values("scenes",
                                                                        ascending=False)
    g3["pct"] = (100 * g3.scenes / tot).round(1)
    print(g3.to_string())

    print("\nBY YEAR\n")
    y = yr.groupby("year").agg(scenes=("n", "sum"), stations=("station_id", "nunique"))
    print(y.to_string())

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, :2])
    sc = ax.scatter(df.longitude, df.latitude, c=df.n_usable, s=16, cmap="viridis",
                    vmin=0, vmax=float(df.n_usable.quantile(.98)), edgecolors="none")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Usable scenes per station  (CDIST > {CDIST_MIN_KM:g} km, "
                 f"{len(df)} stations, {tot:,} scenes)", fontsize=11)
    ax.grid(alpha=.25)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="usable scenes")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df.n_usable, bins=40, color="tab:blue")
    ax.axvline(df.n_usable.median(), color="k", ls="--", lw=1.2,
               label=f"median {df.n_usable.median():.0f}")
    ax.axvline(20, color="tab:red", ls=":", lw=1.4, label="20-scene floor")
    ax.set_xlabel("usable scenes per station"); ax.set_ylabel("stations")
    ax.set_title("Distribution across stations", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 0])
    piv = yr.pivot_table(index="year", columns="kg_macro", values="n", aggfunc="sum").fillna(0)
    piv.plot(kind="bar", stacked=True, ax=ax, width=.85, legend=True)
    ax.set_xlabel("year"); ax.set_ylabel("usable scenes")
    ax.set_title("Scenes per year, by climate", fontsize=10)
    ax.legend(fontsize=7, title="Köppen", title_fontsize=7); ax.grid(alpha=.3, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    gg = df.groupby("kg_macro").n_usable.sum().sort_values()
    ax.barh(gg.index, gg.values, color="tab:orange")
    for i, (k, v) in enumerate(gg.items()):
        ax.text(v, i, f" {v:,} ({100*v/tot:.0f}%)  n={int((df.kg_macro==k).sum())}",
                va="center", fontsize=7)
    ax.set_xlabel("usable scenes"); ax.set_title("By Köppen macro-class", fontsize=10)
    ax.grid(alpha=.3, axis="x"); ax.set_xlim(0, gg.max() * 1.45)

    ax = fig.add_subplot(gs[1, 2])
    gg = df.groupby("igbp_macro").n_usable.sum().sort_values()
    ax.barh(gg.index, gg.values, color="tab:green")
    for i, (k, v) in enumerate(gg.items()):
        ax.text(v, i, f" {v:,} ({100*v/tot:.0f}%)  n={int((df.igbp_macro==k).sum())}",
                va="center", fontsize=7)
    ax.set_xlabel("usable scenes"); ax.set_title("By IGBP land-cover macro-class", fontsize=10)
    ax.grid(alpha=.3, axis="x"); ax.set_xlim(0, gg.max() * 1.5)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "inventory.png", dpi=120)
    print(f"\nwrote {FIG/'inventory.png'}")
    print("=" * 86)


if __name__ == "__main__":
    main()
