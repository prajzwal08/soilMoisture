#!/usr/bin/env python
"""Step 2c: DTR against soil moisture, every station in a tile pooled.

Each member contributes its own 70 m ECOSTRESS pixel -- not the tile mean -- against its
own observed soil moisture that day.  All usable pairs, not just the months the map
figure draws.

TWO THINGS THAT WOULD OTHERWISE MANUFACTURE A RESULT:

1. "A day with a soil-moisture observation" means qc == 0.  The record is daily and
   gap-filled, so a value always exists; where qc == 1 it is a month-day climatology.
   Correlating DTR against climatology would produce a seasonal relationship out of
   nothing, and the fraction dropped is reported.

2. The pooled scatter is never shown alone.  29.13 measured pooled r = +0.167 against
   within-station r = -0.077 ON THE SAME 546 RECORDS -- Simpson's paradox, and the sign
   reverses under exactly the grouping the thesis cares about.  So:

     within-station   each station de-meaned   does a probe's DTR fall as IT gets wetter?
     between-station  each date de-meaned      on one date, is the wetter probe the
                                               lower-DTR one?   <- the SPATIAL question
     pooled           raw                      shown for completeness only

Phase is a COVARIATE, not a filter -- the well_phased flag had a wrap-at-24 bug -- so
points are coloured by day solar time rather than screened on it.

Env: terramind.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_gra_thermal import REPO, OUT_DIR, load_bundle, member_labels, _as_str

log = logging.getLogger("gra_dtr_sm")
RNG = np.random.default_rng(0)


def boot_r(x, y, groups, n=2000):
    """Cluster-bootstrapped CI on Pearson r, resampling whole groups."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    uq = np.unique(groups)
    if len(uq) < 2 or x.size < 6:
        return np.nan, np.nan
    out = []
    for _ in range(n):
        pick = RNG.choice(uq, size=len(uq), replace=True)
        m = np.concatenate([np.where(groups == g)[0] for g in pick])
        if m.size < 4:
            continue
        xs, ys = x[m], y[m]
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            continue
        out.append(np.corrcoef(xs, ys)[0, 1])
    if len(out) < 50:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def collect(cl, mem, depth):
    """-> long DataFrame(station, date, dtr, sm, tst) over every usable pair."""
    z, _ = load_bundle(cl.rep_folder)
    if z is None:
        return pd.DataFrame(), 0
    inside = mem[mem.in_tile == 1].reset_index(drop=True)
    dtr = z["dtr_k"]
    val = z["valid"].astype(bool)
    days = np.array([_as_str(d)[:10] for d in z["day_utc"]])
    aligned = z["grid_aligned"] == 1
    tst = np.asarray(z["day_tst"], dtype=float) if "day_tst" in z.files else \
        np.full(len(days), np.nan)

    rows, dropped = [], 0
    for _, m in inside.iterrows():
        lab = member_labels(m.folder, m.category).get(depth)
        if lab is None or lab.empty:
            continue
        sm_all = lab.set_index("date")
        r, c = int(m.eco_row), int(m.eco_col)
        if not (0 <= r < dtr.shape[1] and 0 <= c < dtr.shape[2]):
            continue
        for p in range(len(days)):
            if not aligned[p] or not val[p, r, c]:
                continue
            v = float(dtr[p, r, c])
            if not np.isfinite(v):
                continue
            key = pd.Timestamp(str(days[p]))
            if key not in sm_all.index:
                continue
            row = sm_all.loc[key]
            if int(row.qc) != 0:            # gap-filled -> not an observation
                dropped += 1
                continue
            if not np.isfinite(row.sm):
                continue
            rows.append(dict(station=m.station_id, date=key, dtr=v,
                             sm=float(row.sm), tst=float(tst[p])))
    return pd.DataFrame(rows), dropped


def draw(cl, mem, args) -> bool:
    cid = cl.cluster_id
    D, dropped = collect(cl, mem, args.depth)
    if D.empty or D.station.nunique() < 1:
        log.warning("%s: no (DTR, observed SM) point survives -- skipped", cid)
        return False

    within = D.copy()
    within["x"] = within.groupby("station")["sm"].transform(lambda s: s - s.mean())
    within["y"] = within.groupby("station")["dtr"].transform(lambda s: s - s.mean())

    nper = D.groupby("date")["station"].transform("nunique")
    btw = D[nper >= 2].copy()
    if not btw.empty:
        btw["x"] = btw.groupby("date")["sm"].transform(lambda s: s - s.mean())
        btw["y"] = btw.groupby("date")["dtr"].transform(lambda s: s - s.mean())

    panels = [
        ("within-station\n(each station de-meaned)", within, "x", "y", "station",
         "does a probe's DTR fall as IT gets wetter?"),
        ("between-station\n(each date de-meaned)", btw, "x", "y", "date",
         "on one date, is the wetter probe the lower-DTR one?  <- SPATIAL"),
        ("pooled (raw)", D, "sm", "dtr", "station",
         "shown only alongside the other two -- 29.13 saw the sign reverse"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.9))
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.055, right=0.90, wspace=0.26)
    for ax, (title, T, xc, yc, gc, sub) in zip(axes, panels):
        if T is None or T.empty or len(T) < 4:
            ax.text(.5, .5, "not enough points", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(title, fontsize=10)
            continue
        s = ax.scatter(T[xc], T[yc], c=T["tst"], cmap="twilight", s=16,
                       alpha=.8, lw=.3, edgecolor="k", vmin=9, vmax=15)
        x, y = T[xc].to_numpy(float), T[yc].to_numpy(float)
        r = np.corrcoef(x, y)[0, 1] if x.std() > 1e-12 and y.std() > 1e-12 else np.nan
        lo, hi = boot_r(x, y, T[gc].to_numpy())
        if np.isfinite(r):
            k = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 20)
            ax.plot(xx, np.polyval(k, xx), color="#B03A2E", lw=1.4)
        ci = f"  [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "  CI n/a"
        ax.set_title(f"{title}\nr = {r:+.3f}{ci}   n = {len(T)}   "
                     f"{T[gc].nunique()} {gc}s", fontsize=9.5)
        ax.set_xlabel(("SM anomaly" if xc == "x" else "SM") + " (m3/m3)", fontsize=9)
        ax.set_ylabel(("DTR anomaly" if yc == "y" else "DTR") + " (K)", fontsize=9)
        ax.grid(alpha=.25, lw=.5)
        ax.tick_params(labelsize=8)
        ax.text(.02, .02, sub, transform=ax.transAxes, fontsize=7.4, color="#555",
                va="bottom", style="italic")
    cb = fig.colorbar(s, cax=fig.add_axes([0.915, 0.12, 0.012, 0.58]))
    cb.set_label("day solar time (h)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f"{cid}   {cl.n_stations} station(s), extent {cl.extent_km:.2f} km   "
        f"{args.depth} cm, observed only (qc 0; {dropped} gap-filled point(s) dropped)\n"
        f"each member uses ITS OWN 70 m ECOSTRESS pixel   |   "
        f"at this n this is a look, not a test -- 38.10 answered it over the whole archive",
        fontsize=10.5, y=0.985)

    out = OUT_DIR / f"dtr_vs_sm_{cid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    log.info("%s: n=%d over %d station(s), %d date(s); %d gap-filled dropped -> %s",
             cid, len(D), D.station.nunique(), D.date.nunique(), dropped, out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=str(REPO / "csvs" / "gra_thermal_clusters.csv"))
    ap.add_argument("--members",  default=str(REPO / "csvs" / "gra_thermal_members.csv"))
    ap.add_argument("--cluster", default="")
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--min-members", type=int, default=2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C = pd.read_csv(args.clusters)
    M = pd.read_csv(args.members)
    if args.cluster:
        C = C[C.cluster_id == args.cluster]
    C = C[C.n_stations >= args.min_members]
    log.info("rendering %d cluster(s), depth %s", len(C), args.depth)

    ok = 0
    for _, cl in C.iterrows():
        ok += bool(draw(cl, M[M.cluster_id == cl.cluster_id], args))
    log.info("")
    log.info("%d/%d DTR-vs-SM figures written to %s", ok, len(C), OUT_DIR)


if __name__ == "__main__":
    main()
