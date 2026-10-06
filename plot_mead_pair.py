# ============================================================
# Ne1 vs Ne2 side by side, per year (runbook §69)
# ============================================================
# 2019 is the natural experiment: Ne2 irrigated (5 logged events),
# Ne1 not (no log, no rain-free wetting).  2020 has both irrigated.
# Per year column:
#   (a) tower rain (black, Ne1 gauge) + irrigation events per site
#   (b-d) SM at 0-10 / 10-30 / 30-100 cm, Ne1 vs Ne2
#   (e) difference Ne1 - Ne2 per depth (frozen days shaded)
#
# Usage:
#   python plot_mead_pair.py [--years 2019 2020] [--a US-Ne1 --b US-Ne2]
# ============================================================

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import plot_style_bw

ROOT    = Path("/gpfs/work3/0/prjs1968/data_mead")
FIG_DIR = Path("/gpfs/work3/0/prjs1968/soilMoisture/figures/mead")
BINS    = ["0-10", "10-30", "30-100"]


def load(site):
    ds = xr.open_dataset(ROOT / "observed" / f"AmeriFlux_{site}_observed.nc")
    t = pd.to_datetime(ds["date_time"].values)
    sm = pd.DataFrame(ds["soil_moisture"].values, index=t,
                      columns=[str(d) for d in ds["depth"].values])
    rain = pd.Series(ds["precip_mm"].values, index=t)
    ts = pd.Series(ds["ts_degc"].values, index=t) if "ts_degc" in ds else None
    f = ROOT / "bif" / f"{site}_irrigation_events.csv"
    irr = (pd.read_csv(f, parse_dates=["date_start"]) if f.exists() and f.stat().st_size > 1
           else pd.DataFrame(columns=["date_start", "amount_in", "cancelled", "not_over_tower"]))
    return sm, rain, ts, irr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020])
    ap.add_argument("--a", default="US-Ne1")
    ap.add_argument("--b", default="US-Ne2")
    args = ap.parse_args()

    plot_style_bw.apply(globals(), "paper")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    A, B = args.a, args.b
    col = {A: plot_style_bw.BLUE, B: plot_style_bw.RED}
    dcol = {"0-10": plot_style_bw.BLUE, "10-30": plot_style_bw.GREEN, "30-100": plot_style_bw.RED}
    data = {s: load(s) for s in (A, B)}
    name = {s: s.replace("US-", "") for s in (A, B)}

    ny = len(args.years)
    fig, axes = plt.subplots(5, ny, figsize=(6.5 * ny, 12), sharex="col",
                             gridspec_kw={"height_ratios": [0.7, 1, 1, 1, 1]}, squeeze=False)

    # common y-limits from the data across all years shown
    sm_all = pd.concat([data[s][0].loc[f"{y}"] for s in (A, B) for y in args.years]).stack()
    y_lo, y_hi = np.nanpercentile(sm_all, 0.5) - 0.02, np.nanpercentile(sm_all, 99.5) + 0.02
    diffs = pd.concat([(data[A][0][b] - data[B][0][b]).loc[f"{y}"]
                       for b in BINS for y in args.years])
    d_lim = np.nanpercentile(np.abs(diffs), 99.5) + 0.01

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for j, y in enumerate(args.years):
        lo, hi = f"{y}-01-01", f"{y}-12-31"
        ax = axes[0, j]
        rain = data[A][1].loc[lo:hi]
        ax.bar(rain.index, rain.values, width=1.0, color="black", label="Rain")
        tot = {}
        for k, s in enumerate((A, B)):
            e = data[s][3]
            e = e[(e["date_start"] >= lo) & (e["date_start"] <= hi)]
            if "cancelled" in e:
                e = e[~e["cancelled"].fillna(False).astype(bool)]
            tot[s] = e["amount_in"].sum() * 25.4
            off = pd.Timedelta(days=-1.5 if k == 0 else 1.5)
            ax.bar(e["date_start"] + off, e["amount_in"].fillna(1.0) * 25.4, width=3.0,
                   color=col[s], edgecolor="black", lw=0.8, label=f"Irrigation {name[s]}")
        ax.set_ylim(0, max(60, rain.max() * 1.1))
        ax.set_ylabel("mm d$^{-1}$")
        ax.text(0.01, 0.95, f"({next(letters)}) {y}: {name[A]} {tot[A]:.0f} mm, "
                f"{name[B]} {tot[B]:.0f} mm irrigation", transform=ax.transAxes, va="top")

        for i, b in enumerate(BINS, start=1):
            ax = axes[i, j]
            ts = data[A][2]
            if ts is not None:
                fz = ts.loc[lo:hi] <= 0
                for d in fz.index[fz.values]:
                    ax.axvspan(d, d + pd.Timedelta("1D"), color="0.88", lw=0, zorder=0)
            for s in (A, B):
                v = data[s][0][b].loc[lo:hi]
                ax.plot(v.index, v.values, color=col[s], lw=1.6, label=name[s])
            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(f"SM {b} cm (m$^3$ m$^{{-3}}$)")
            ax.text(0.01, 0.04, f"({next(letters)})", transform=ax.transAxes, va="bottom")

        ax = axes[4, j]
        ax.axhline(0, color="black", lw=1.0)
        for b in BINS:
            d = (data[A][0][b] - data[B][0][b]).loc[lo:hi]
            ax.plot(d.index, d.values, color=dcol[b], lw=1.6, label=f"{b} cm")
        ax.set_ylim(-d_lim, d_lim)
        ax.set_ylabel(f"{name[A]} $-$ {name[B]} (m$^3$ m$^{{-3}}$)")
        ax.text(0.01, 0.04, f"({next(letters)})", transform=ax.transAxes, va="bottom")

    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    axes[1, 0].legend(loc="upper right", ncol=2, frameon=True, framealpha=1.0)
    axes[4, 0].legend(loc="upper right", ncol=3, frameon=True, framealpha=1.0)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    tag = "_".join(str(y) for y in args.years)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_{name[A]}_vs_{name[B]}_{tag}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_{name[A]}_vs_{name[B]}_{tag}.png")

    # numbers behind the figure: Jun-Sep mean difference per depth
    for y in args.years:
        row = []
        for b in BINS:
            d = (data[A][0][b] - data[B][0][b]).loc[f"{y}-06-01":f"{y}-09-30"]
            row.append(f"{b}: {d.mean():+.3f}")
        print(f"  {y} Jun-Sep mean {name[A]}-{name[B]}  " + "  ".join(row))


if __name__ == "__main__":
    main()
