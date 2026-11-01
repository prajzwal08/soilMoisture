#!/usr/bin/env python
"""The six TxSON stations on the CR200-18 tile: DTR and surface SM on every DTR day.

These are the six §36.21(iii) names -- CR200-18 at centre, CR200-25 at 405 m, CR1000-2 at
684 m, CR200-24 at 865 m, CR200-15 at 925 m, CR200-6 at 936 m -- i.e. six in-situ probes
inside one 2.24 km ECOSTRESS window, the same set §29.13 used for daytime LST.

Three columns per station:
  1  DTR on each usable pair date, split by dt band (the two bands are different
     geometries, not two samples of one: 6-9 h has day_tst ~14.1 h and night ~20.5 h,
     >=15 h has day ~9.7 h and night ~3.2 h -- neither is max minus min)
  2  the full observed (qc==0) 0-10 cm record as a faint line, with the DTR days marked,
     so it is visible whether the DTR days sample wet and dry periods or only one
  3  DTR against SM on those days, with r

NO DUAL AXES.  DTR in K and SM in m3/m3 get their own panels; overlaying them on twin
y-scales would let the visual slope be set by the scaling choice.

n IS 8-16 DAYS PER STATION.  §29.10's warning applies directly -- a single per-station r
at this n has a very wide interval and none of them should be quoted on its own. The
pooled row at the bottom is the only number here with any power, and even it is six
stations inside one tile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
import dataset as _ds                                          # noqa: E402
_ds.ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")
from dataset import SM_DEPTHS, _load_zarr_labels, _open_zarr    # noqa: E402
from census_ecostress import ROOT                               # noqa: E402

ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")
SURFACE = "0-10"

SIX = [("CR200-18", "centre"), ("CR200-25", "405 m"), ("CR1000-2", "684 m"),
       ("CR200-24", "865 m"), ("CR200-15", "925 m"), ("CR200-6", "936 m")]

C_SHORT, C_LONG = "#C1502E", "#3B6EA5"      # dt 6-9 h, dt >=15 h


def surface_series(folder, cat):
    zg = _open_zarr(ZARR_ROOT / cat / folder, cat)
    out = _load_zarr_labels(zg) if zg is not None else None
    if out is None:
        return None
    sm, depths, times, qc = out
    for d, dep in enumerate(depths):
        dep = dep.decode() if isinstance(dep, bytes) else str(dep)
        if dep != SURFACE:
            continue
        keep = ~np.isnan(sm[d])
        if qc is not None:
            keep &= (qc[d] == 0)
        if not keep.any():
            return None
        return pd.DataFrame({"date": pd.to_datetime(times[keep]).normalize(),
                             "sm": sm[d][keep].astype(np.float32)})
    return None


def rv(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=str(ROOT / "csvs" / "ecostress_dtr_bundles.TxSON.csv"))
    ap.add_argument("--out", default=str(ROOT / "fig" / "dtr_txson" / "txson_six_dtr_sm.png"))
    args = ap.parse_args()

    b = pd.read_csv(args.bundles).set_index("station_id")
    n = len(SIX)
    fig, axes = plt.subplots(n, 3, figsize=(15.5, 2.35 * n),
                             gridspec_kw={"width_ratios": [1.5, 1.5, 1.0]})
    allx, ally = [], []

    for r, (sid, dist) in enumerate(SIX):
        if sid not in b.index:
            for c in range(3):
                axes[r, c].text(.5, .5, f"{sid}: no bundle", ha="center",
                                transform=axes[r, c].transAxes)
            continue
        row = b.loc[sid]
        z = np.load(row["path"], allow_pickle=False)
        ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] > 0)
        nd = int(ok.sum())

        dtr = z["dtr_k"][ok].reshape(nd, -1)
        val = z["valid"][ok].reshape(nd, -1).astype(bool)
        with np.errstate(invalid="ignore"):
            scene = np.array([d[v].mean() if v.any() else np.nan
                              for d, v in zip(dtr, val)])
        dates = pd.to_datetime([str(x.decode() if isinstance(x, bytes) else x)[:10]
                                for x in z["day_utc"][ok]]).normalize()
        dth = z["dt_hours"][ok].astype(float) if "dt_hours" in z.files \
            else np.full(nd, np.nan)
        dd = pd.DataFrame({"date": dates, "dtr": scene, "dt": dth}).dropna(subset=["dtr"])

        sm = surface_series(row["folder"], row["category"])
        j = dd.merge(sm, on="date", how="inner") if sm is not None else dd.assign(sm=np.nan)
        short = j["dt"] < 12

        ax = axes[r, 0]
        ax.vlines(j.date[short], 0, j.dtr[short], color=C_SHORT, lw=1, alpha=.55)
        ax.vlines(j.date[~short], 0, j.dtr[~short], color=C_LONG, lw=1, alpha=.55)
        ax.plot(j.date[short], j.dtr[short], "o", ms=5, color=C_SHORT,
                label="dt 6–9 h" if r == 0 else None)
        ax.plot(j.date[~short], j.dtr[~short], "s", ms=5, color=C_LONG,
                label="dt ≥15 h" if r == 0 else None)
        ax.axhline(0, color="#bbb", lw=.8)
        ax.set_ylabel("DTR (K)", fontsize=8.5)
        if r == 0:
            ax.set_title("DTR on each usable pair date", fontsize=10)
            ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[r, 1]
        if sm is not None:
            ax.plot(sm.date, sm.sm, "-", color="#999", lw=.7, alpha=.8)
        ax.plot(j.date[short], j.sm[short], "o", ms=5, color=C_SHORT)
        ax.plot(j.date[~short], j.sm[~short], "s", ms=5, color=C_LONG)
        ax.set_ylabel("0–10 cm SM\n(m³/m³)", fontsize=8.5)
        if r == 0:
            ax.set_title("observed surface SM (qc==0); marks = the DTR days",
                         fontsize=10)

        ax = axes[r, 2]
        ax.plot(j.sm[short], j.dtr[short], "o", ms=6, color=C_SHORT)
        ax.plot(j.sm[~short], j.dtr[~short], "s", ms=6, color=C_LONG)
        rr, nn = rv(j.sm.to_numpy(), j.dtr.to_numpy())
        ax.set_title(f"r = {rr:+.2f}   n = {nn}", fontsize=9.5)
        ax.set_ylabel("DTR (K)", fontsize=8.5)
        if r == 0:
            ax.text(.5, 1.28, "DTR vs SM on those days", ha="center", fontsize=10,
                    transform=ax.transAxes)
        allx.append(j.sm.to_numpy()); ally.append(j.dtr.to_numpy())

        axes[r, 0].text(-0.28, .5, f"{sid}\n{dist}", transform=axes[r, 0].transAxes,
                        fontsize=10, ha="center", va="center", weight="bold")
        for c in range(3):
            axes[r, c].spines[["top", "right"]].set_visible(False)
            axes[r, c].tick_params(labelsize=8)
        if r < n - 1:
            axes[r, 0].set_xticklabels([]); axes[r, 1].set_xticklabels([])

    axes[-1, 0].set_xlabel("date", fontsize=9)
    axes[-1, 1].set_xlabel("date", fontsize=9)
    axes[-1, 2].set_xlabel("0–10 cm SM (m³/m³)", fontsize=9)

    X, Y = np.concatenate(allx), np.concatenate(ally)
    rp, npd = rv(X, Y)
    print(f"pooled over the six stations: r = {rp:+.3f}  n = {npd}")
    fig.suptitle("Six TxSON stations inside one ECOSTRESS window — "
                 "DTR and surface soil moisture on every DTR day\n"
                 f"pooled over the six: r = {rp:+.3f} (n = {npd}).  "
                 "8–16 days per station: per-station r values have very wide intervals "
                 "(§29.10) and none should be read on its own.",
                 fontsize=11.5, y=1.005)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
