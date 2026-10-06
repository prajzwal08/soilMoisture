#!/usr/bin/env python
"""Paper figure: the station data set and how it is split.

  (a) global map of the soil-moisture stations, coloured by split (Train / Validation / OOS)
  (b) how the splits are defined: stations x time, cut at 2023-01-01 (Train, OOT, OOS, OOST)
  (c) record length per station and depth (years of observed days, 2016-2025)
  (d) number of observed station-days (data points) per evaluation set and depth

Sources
  csvs/station_splits.csv                      station split, coordinates
  token zarr labels (combine_network._obs_from_zarr, QC = 0 observed days only)   record length,
                                               and the Train count (no eval file exists for train)
  eval_output/<run>/per_station_{val,oos,oot,oost}.csv   Val / OOS / OOT / OOST counts, i.e. the
                                               exact days behind every reported metric
  The label-based counts for Val/OOS/OOT/OOST are printed next to the eval ones as a check.

Paper style (plot_style_bw.RC_PAPER). CPU; label scan uses Pool(64).
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style_bw
from eval_metrics import _make_key

RUN    = "baseline_selected_20261005"
SHP    = ("/gpfs/home5/pkhanal/miniforge3/envs/terramind/lib/python3.11/site-packages/"
          "pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp")
DEPTHS = ["0-10", "10-30", "30-100"]
CUT    = pd.Timestamp("2023-01-01")
T0, T1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31")
BLUE, LBLUE, ORANGE = plot_style_bw.BLUE, plot_style_bw.GREEN, plot_style_bw.RED
GREY = "0.45"
SPLIT_STYLE = {"train": (GREY, "Train"), "val": (LBLUE, "Validation"), "oos": (BLUE, "OOS")}


def _counts(args):
    """Observed days per depth, before / after the cut, for one station."""
    from combine_network import _obs_from_zarr
    key, cat = args
    try:
        o = _obs_from_zarr(key, cat)
    except Exception:
        o = None
    out = {"station_key": key}
    for d in DEPTHS:
        q = o[(o.depth == d) & (o.date >= T0) & (o.date <= T1)] if o is not None else None
        out[f"pre_{d}"] = int((q.date < CUT).sum()) if q is not None else 0
        out[f"post_{d}"] = int((q.date >= CUT).sum()) if q is not None else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"eval_output/{RUN}")
    ap.add_argument("--out-dir", default=f"figures/eval/{RUN}_paper")
    a = ap.parse_args()
    plt.rcParams.update(plot_style_bw.RC_PAPER)

    sp = pd.read_csv("csvs/station_splits.csv")          # pandas: quoted commas in AmeriFlux names
    sp["key"] = sp.apply(_make_key, axis=1)
    sp = sp[sp.split.isin(["train", "val", "oos"]) & sp.has_soil_moisture.astype(bool)].copy()
    sp["cat"] = np.where(sp.has_flux.astype(bool), "sm_and_flux", "sm_only")
    print(sp.split.value_counts().to_string())

    with Pool(64) as pool:
        cnt = pd.DataFrame(pool.map(_counts, list(zip(sp.key, sp.cat)), chunksize=4))
    sp = sp.merge(cnt, left_on="key", right_on="station_key", how="left")

    ev = {s: pd.read_csv(Path(a.eval_dir) / f"per_station_{s}.csv") for s in ("val", "oos", "oot", "oost")}

    # ── data points per set x depth ────────────────────────────────────────
    tr = sp[sp.split == "train"]
    pts = {"Train": [int(tr[f"pre_{d}"].sum()) for d in DEPTHS]}
    nst = {"Train": int((tr[[f"pre_{d}" for d in DEPTHS]].sum(axis=1) > 0).sum())}
    for s, lab in [("val", "Validation"), ("oos", "OOS"), ("oot", "OOT"), ("oost", "OOST")]:
        pts[lab] = [int(ev[s][f"n_{d.replace('-', '_')}"].fillna(0).sum()) for d in DEPTHS]
        nst[lab] = len(ev[s])
    # label-based check for the eval sets
    chk = {"Validation": sp[sp.split == "val"], "OOS": sp[sp.split == "oos"]}
    print("\nobserved days (eval file vs labels), per depth:")
    for lab, g in chk.items():
        print(f"  {lab:10s} eval {pts[lab]}  labels {[int(g[f'pre_{d}'].sum()) for d in DEPTHS]}")
    for lab, s in [("OOT", ["train", "val"]), ("OOST", ["oos"])]:
        g = sp[sp.split.isin(s)]
        print(f"  {lab:10s} eval {pts[lab]}  labels(all post-cut) {[int(g[f'post_{d}'].sum()) for d in DEPTHS]}")
    print("stations:", nst)
    miss = sorted(set(sp[sp.split == "oos"].key) - set(ev["oos"].station_key))
    print(f"OOS stations in the split but not evaluated: {len(miss)} {miss}")

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.5, 12.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.22, wspace=0.30,
                          left=0.06, right=0.985, top=0.97, bottom=0.07)

    # (a) map
    ax = fig.add_subplot(gs[0, :])
    world = gpd.read_file(SHP)
    world.plot(ax=ax, color="0.92", edgecolor="0.6", linewidth=0.4)
    for s in ["train", "val", "oos"]:
        c, lab = SPLIT_STYLE[s]
        g = sp[sp.split == s]
        ax.scatter(g.longitude, g.latitude, s=22, c=c, edgecolors="black", linewidths=0.35,
                   zorder={"train": 3, "val": 4, "oos": 5}[s], label=f"{lab} ({len(g)} stations)")
    ax.set_xlim(-170, 180); ax.set_ylim(-46, 78); ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.legend(loc="lower left", frameon=True, framealpha=1.0, edgecolor="black", markerscale=1.8)
    ax.text(0.0, 1.01, f"(a)   {len(sp)} soil-moisture stations", transform=ax.transAxes,
            ha="left", va="bottom")

    # (b) split definition
    ax = fig.add_subplot(gs[1, 0])
    rows = [("Train\nstations", "train", "Training", GREY, "OOT", LBLUE),
            ("Validation\nstations", "val", "Validation", LBLUE, "OOT", LBLUE),
            ("OOS\nstations", "oos", "OOS", BLUE, "OOST", ORANGE)]
    for i, (ylab, s, l1, c1, l2, c2) in enumerate(rows):
        y = len(rows) - 1 - i
        n = (sp.split == s).sum()
        ax.add_patch(mpatches.Rectangle((2016, y - 0.35), 7, 0.7, fc=c1, ec="black", lw=0.8))
        ax.add_patch(mpatches.Rectangle((2023, y - 0.35), 3, 0.7, fc=c2, ec="black", lw=0.8,
                                        hatch="//" if l2 == "OOT" else None))
        ax.text(2019.5, y, f"{l1}\n{n} stations", ha="center", va="center",
                color="white" if c1 in (GREY, BLUE) else "black")
        n2 = (ev["oot"].split == s).sum() if l2 == "OOT" else len(ev["oost"])
        ax.text(2024.5, y, f"{l2}\n{n2} stations", ha="center", va="center", color="black",
                bbox=dict(fc="white", ec="none", pad=1.0))
    ax.axvline(2023, color="black", lw=1.5, ls="--")
    ax.text(2023, len(rows) - 0.42, "cut: 2023-01-01", ha="center", va="bottom",
            bbox=dict(fc="white", ec="black", lw=0.8, pad=2.0))
    ax.set_xlim(2015.6, 2026.4); ax.set_ylim(-0.6, len(rows) - 0.4 + 0.55)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xticks([2016, 2019, 2023, 2026])
    ax.set_xlabel("Year")
    ax.text(0.0, 1.01, "(b)", transform=ax.transAxes, ha="left", va="bottom")

    # (c) record length
    ax = fig.add_subplot(gs[1, 1])
    dcols = [BLUE, LBLUE, ORANGE]
    bins = np.arange(0, 10.5, 0.5)
    for d, c in zip(DEPTHS, dcols):
        yrs_d = (sp[f"pre_{d}"] + sp[f"post_{d}"]) / 365.25
        yrs_d = yrs_d[yrs_d > 0]
        ax.hist(yrs_d, bins=bins, histtype="step", lw=2.2, color=c,
                label=f"{d} cm ({len(yrs_d)} stations, median {yrs_d.median():.1f} years)")
    ax.set_xlabel("Observed record per station (years, 2016–2025)")
    ax.set_ylabel("Stations")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.text(0.0, 1.01, "(c)", transform=ax.transAxes, ha="left", va="bottom")

    # (d) data points
    ax = fig.add_subplot(gs[1, 2])
    sets = ["Train", "Validation", "OOS", "OOT", "OOST"]
    w = 0.27
    for di, (d, c) in enumerate(zip(DEPTHS, dcols)):
        x = np.arange(len(sets)) + (di - 1) * w
        v = np.array([pts[s][di] for s in sets]) / 1000
        ax.bar(x, v, w * 0.92, color=c, edgecolor="black", lw=0.8, label=f"{d} cm")
    for i, s in enumerate(sets):
        ax.text(i, max(pts[s]) / 1000 * 1.04, f"{nst[s]}\nstations", ha="center", va="bottom")
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylabel("Observed station-days (thousands)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(max(v) for v in pts.values()) / 1000 * 1.25)
    ax.text(0.0, 1.01, "(d)", transform=ax.transAxes, ha="left", va="bottom")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"data_overview.{ext}", dpi=600)
    pd.DataFrame(pts, index=DEPTHS).T.assign(stations=pd.Series(nst)).to_csv(out / "data_overview_points.csv")
    sp[["key", "split", "latitude", "longitude"] + [f"{p}_{d}" for p in ("pre", "post") for d in DEPTHS]] \
        .to_csv(out / "data_overview_stations.csv", index=False)
    print(f"wrote {out}/data_overview.{{png,pdf}}, data_overview_points.csv, data_overview_stations.csv")


if __name__ == "__main__":
    main()
