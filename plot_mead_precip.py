# ============================================================
# Mead precipitation: ERA5-Land (model forcing) vs tower gauges (runbook §69.4)
# ============================================================
# Ne1 and Ne2 share one ERA5-Land cell (values18 identical), but their tower gauges
# differ.  Suspicion: the irrigated gauges catch pivot water.  Per year:
#   (a) cumulative P: ERA5, Ne1/Ne2/Ne3 gauges; logged irrigation dates on Ne1/Ne2
#   (b) daily P: ERA5 bars, gauges as markers
# Table: annual and Jun-Sep totals, logged irrigation, gauges on logged irrigation days.
#
# Usage:
#   python plot_mead_precip.py [--years 2019 ...]
# ============================================================

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr

import plot_style_bw

ROOT    = Path("/gpfs/work3/0/prjs1968/data_mead")
ERA5_Z  = Path("/gpfs/scratch1/shared/pkhanal/zarr/flux_only/AmeriFlux_US-Ne1/era5")
FIG_DIR = Path("/gpfs/work3/0/prjs1968/soilMoisture/figures/mead")
SITES   = ["US-Ne1", "US-Ne2", "US-Ne3"]
COL     = {"ERA5": "black", "US-Ne1": plot_style_bw.BLUE, "US-Ne2": plot_style_bw.GREEN,
           "US-Ne3": plot_style_bw.RED}
LABEL   = {"ERA5": "ERA5-Land (model forcing)", "US-Ne1": "Ne1 gauge (irrigated)",
           "US-Ne2": "Ne2 gauge (irrigated)", "US-Ne3": "Ne3 gauge (rainfed)"}


def era5_tp():
    g = zarr.open_group(str(ERA5_Z), mode="r")
    names = [str(v) for v in g["vars18"][:]]
    k = names.index("tp_sum")
    d = pd.to_datetime([str(x) for x in g["date_ints"][:]], format="%Y%m%d")
    tp = pd.Series(np.asarray(g["values18"][:, k], dtype=float), index=d).sort_index()
    yearly = tp.groupby(tp.index.year).sum().median()
    if yearly < 5:                                   # metres -> mm
        tp = tp * 1000.0
    print(f"ERA5 tp_sum column {k}, median annual {tp.groupby(tp.index.year).sum().median():.0f} mm")
    return tp


def load():
    p = {"ERA5": era5_tp()}
    irr = {}
    for s in SITES:
        ds = xr.open_dataset(ROOT / "observed" / f"AmeriFlux_{s}_observed.nc")
        p[s] = pd.Series(ds["precip_mm"].values, index=pd.to_datetime(ds["date_time"].values))
        f = ROOT / "bif" / f"{s}_irrigation_events.csv"
        e = (pd.read_csv(f, parse_dates=["date_start"]) if f.exists() and f.stat().st_size > 1
             else pd.DataFrame(columns=["date_start", "amount_in", "cancelled"]))
        if "cancelled" in e:
            e = e[~e["cancelled"].fillna(False).astype(bool)]
        irr[s] = e
    return p, irr


def table(p, irr, years):
    print("\nyear  | annual mm: ERA5   Ne1   Ne2   Ne3 | Jun-Sep mm: ERA5   Ne1   Ne2   Ne3 | "
          "logged irr mm Ne1 Ne2 | on logged irr days (mean mm): own gauge vs Ne3")
    for y in years:
        a = {k: v.loc[str(y)].sum() for k, v in p.items()}
        js = {k: v.loc[f"{y}-06-01":f"{y}-09-30"].sum() for k, v in p.items()}
        il = {s: irr[s][irr[s]["date_start"].dt.year == y]["amount_in"].sum() * 25.4
              for s in ("US-Ne1", "US-Ne2")}
        on = []
        for s in ("US-Ne1", "US-Ne2"):
            d = irr[s][irr[s]["date_start"].dt.year == y]["date_start"].dt.normalize()
            if len(d):
                on.append(f"{s[-3:]} {p[s].reindex(d).mean():5.1f} vs {p['US-Ne3'].reindex(d).mean():4.1f}")
        print(f"{y}  | {a['ERA5']:9.0f} {a['US-Ne1']:5.0f} {a['US-Ne2']:5.0f} {a['US-Ne3']:5.0f} | "
              f"{js['ERA5']:10.0f} {js['US-Ne1']:5.0f} {js['US-Ne2']:5.0f} {js['US-Ne3']:5.0f} | "
              f"{il['US-Ne1']:13.0f} {il['US-Ne2']:3.0f} | " + "  ".join(on))


def figure(p, irr, y):
    plot_style_bw.apply(globals(), "paper")
    lo, hi = f"{y}-01-01", f"{y}-12-31"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    for k in ["ERA5"] + SITES:
        s = p[k].loc[lo:hi]
        c = s.fillna(0).cumsum()
        a1.plot(c.index, c.values, color=COL[k], lw=1.8, label=LABEL[k])
        if k in ("US-Ne1", "US-Ne2") and len(irr[k]):
            d = irr[k]["date_start"].dt.normalize()
            d = d[(d >= lo) & (d <= hi)]
            a1.plot(d, c.reindex(d).values, "s", ms=6, color=COL[k], mec="black", mew=0.8,
                    label=f"Logged irrigation {k[-3:]}")
    a1.set_ylabel("Cumulative P (mm)")
    a1.text(0.01, 0.95, "(a)", transform=a1.transAxes, va="top")
    e = p["ERA5"].loc[lo:hi]
    a2.bar(e.index, e.values, width=1.0, color="0.6", label="ERA5-Land")
    mk = {"US-Ne1": "o", "US-Ne2": "^", "US-Ne3": "x"}
    for s in SITES:
        g = p[s].loc[lo:hi]
        g = g[g > 0.2]
        a2.plot(g.index, g.values, mk[s], ms=4, color=COL[s], label=LABEL[s])
    a2.set_ylabel("Daily P (mm d$^{-1}$)")
    a2.text(0.01, 0.95, "(b)", transform=a2.transAxes, va="top")
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    fig.align_ylabels([a1, a2])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_precip_{y}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_precip_{y}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019])
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p, irr = load()
    table(p, irr, range(2017, 2025))
    for y in args.years:
        figure(p, irr, y)


if __name__ == "__main__":
    main()
