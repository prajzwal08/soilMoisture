# ============================================================
# Mead irrigation contrast, observations only (runbook §69 step 2)
# ============================================================
# How different are the irrigated Mead sites (Ne1, Ne2) from the
# rainfed one (Ne3)?  Reads the observed-only daily files written by
# preprocessing_mead_base.py (no gap-fill) and the BIF irrigation log.
#
#   T1  monthly mean difference (Ne1-Ne3, Ne2-Ne3) per depth bin,
#       2017-2024 (our imagery period) and 2001-2016 (context);
#       only days where both sites are observed
#   T2  per year, Jun-Sep: mean SM per site/bin, tower rain, irrigation
#       total and "no irrigation over towers" flags
#   T3  wetting on rain-free days (May-Sep): count of days with
#       d(theta 0-10) > WET_JUMP while the site gauge and the day before
#       read < DRY_MM
#   T4  Jun-Sep mean LE per site per year (irrigated ET should be higher)
#   ref §66 OOS station-mean ubRMSE (.052/.048/.047) as the yardstick
#
# One sample figure (paper style) for --year: rain + three depth panels.
#
# Usage:
#   python analyse_mead_irrigation.py [--year 2022]
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
SITES   = {"US-Ne1": "Ne1 irrigated maize",
           "US-Ne2": "Ne2 irrigated maize/soy",
           "US-Ne3": "Ne3 rainfed maize/soy"}
RAINFED = "US-Ne3"
BINS    = ["0-10", "10-30", "30-100"]
UBRMSE_OOS = {"0-10": 0.052, "10-30": 0.048, "30-100": 0.047}   # §66
WET_JUMP = 0.01     # m3/m3 per day
DRY_MM   = 0.5      # mm/day
SITE_COLORS = {"US-Ne1": plot_style_bw.BLUE, "US-Ne2": plot_style_bw.GREEN,
               "US-Ne3": plot_style_bw.RED}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def load():
    sm, rain, le, irr = {}, {}, {}, {}
    for s in SITES:
        ds = xr.open_dataset(ROOT / "observed" / f"AmeriFlux_{s}_observed.nc")
        t = pd.to_datetime(ds["date_time"].values)
        sm[s] = pd.DataFrame(ds["soil_moisture"].values, index=t,
                             columns=[str(d) for d in ds["depth"].values])
        rain[s] = pd.Series(ds["precip_mm"].values, index=t) if "precip_mm" in ds else None
        le[s] = pd.Series(ds["le_wm2"].values, index=t) if "le_wm2" in ds else None
        f = ROOT / "bif" / f"{s}_irrigation_events.csv"
        irr[s] = (pd.read_csv(f, parse_dates=["date_start", "date_end"])
                  if f.exists() and f.stat().st_size > 1 else pd.DataFrame())
    return sm, rain, le, irr


def t1_monthly(sm, lo, hi):
    out = []
    for s in ["US-Ne1", "US-Ne2"]:
        for b in BINS:
            if b not in sm[s] or b not in sm[RAINFED]:
                continue
            d = (sm[s][b] - sm[RAINFED][b]).loc[lo:hi].dropna()
            m = d.groupby(d.index.month).mean()
            row = {"pair": f"{s[-3:]}-Ne3", "bin": b, "n_days": len(d)}
            row.update({f"{k:02d}": round(v, 3) for k, v in m.items()})
            out.append(row)
    return pd.DataFrame(out)


def jjas(x):
    return x[(x.index.month >= 6) & (x.index.month <= 9)]


def t2_yearly(sm, rain, irr):
    rows = []
    for y in range(2001, 2026):
        r = {"year": y}
        for s in SITES:
            for b in BINS:
                if b in sm[s]:
                    v = jjas(sm[s][b].loc[str(y)])
                    r[f"{s[-3:]}_{b}"] = round(v.mean(), 3) if v.notna().sum() > 60 else np.nan
        p = rain[RAINFED]
        r["rain_JJAS_mm"] = round(jjas(p.loc[str(y)]).sum(), 0) if p is not None else np.nan
        for s in ["US-Ne1", "US-Ne2"]:
            e = irr[s]
            if len(e):
                e = e[e["date_start"].dt.year == y]
                r[f"{s[-3:]}_irr_in"] = round(e["amount_in"].sum(), 2)
                r[f"{s[-3:]}_n_irr"] = len(e)
                r[f"{s[-3:]}_notower"] = int(e["not_over_tower"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def t3_dry_day_wetting(sm, rain):
    rows = []
    for s in SITES:
        th = sm[s]["0-10"]
        p = rain[s] if rain[s] is not None else rain[RAINFED]
        dth = th.diff()
        dry = (p < DRY_MM) & (p.shift(1) < DRY_MM)
        season = (th.index.month >= 5) & (th.index.month <= 9)
        ok = dth.notna() & p.notna() & p.shift(1).notna() & season
        jump = (dth > WET_JUMP) & dry & ok
        g = pd.DataFrame({"jump": jump, "dry": dry & ok}).groupby(th.index.year).sum()
        for y, r in g.iterrows():
            rows.append({"site": s[-3:], "year": y, "dry_days": int(r["dry"]),
                         "wet_jumps_on_dry_days": int(r["jump"])})
    return (pd.DataFrame(rows)
              .pivot(index="year", columns="site", values="wet_jumps_on_dry_days"))


def t4_le(le):
    rows = []
    for y in range(2001, 2026):
        r = {"year": y}
        for s in SITES:
            if le[s] is not None:
                v = jjas(le[s].loc[str(y)])
                r[s[-3:]] = round(v.mean(), 1) if v.notna().sum() > 60 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def sample_figure(sm, rain, irr, year):
    """One row per site: SM at three depths (left axis) and, hanging from the
    top (right axis, inverted), the site's tower rain (black) and irrigation
    events (filled = over the towers, hollow = log says not over the towers)."""
    plot_style_bw.apply(globals(), "paper")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lo, hi = f"{year}-01-01", f"{year}-12-31"
    depth_col = {"0-10": plot_style_bw.BLUE, "10-30": plot_style_bw.GREEN,
                 "30-100": plot_style_bw.RED}

    sm_all = pd.concat([sm[s].loc[lo:hi] for s in SITES]).stack()
    y_lo, y_hi = np.nanpercentile(sm_all, 0.5) - 0.02, np.nanpercentile(sm_all, 99.5) + 0.12
    p_max = max(rain[s].loc[lo:hi].max() for s in SITES if rain[s] is not None)
    i_max = max([irr[s]["amount_in"].max() * 25.4 for s in SITES if len(irr[s])] + [0])
    w_max = 2.5 * max(p_max, i_max)                     # bars use the top ~40 %

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, (s, name), lab in zip(axes, SITES.items(), ["(a)", "(b)", "(c)"]):
        for b in BINS:
            if b in sm[s]:
                v = sm[s][b].loc[lo:hi]
                ax.plot(v.index, v.values, color=depth_col[b], lw=1.6, label=f"{b} cm")
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("SM (m$^3$ m$^{-3}$)")
        ax.text(0.01, 0.04, f"{lab} {name}", transform=ax.transAxes, va="bottom")

        ax2 = ax.twinx()
        if rain[s] is not None:
            p = rain[s].loc[lo:hi]
            ax2.bar(p.index, p.values, width=1.0, color="black", label="Rain")
        e = irr[s]
        if len(e):
            e = e[(e["date_start"] >= lo) & (e["date_start"] <= hi)]
            for _, r in e.iterrows():
                mm = r["amount_in"] * 25.4 if pd.notna(r["amount_in"]) else 0.3 * w_max
                hollow = bool(r["not_over_tower"])
                ax2.bar(r["date_start"], mm, width=3.0, color="white" if hollow else "0.55",
                        edgecolor="black", lw=1.0,
                        label="Irrigation (not over towers)" if hollow else "Irrigation")
        ax2.set_ylim(w_max, 0)
        ax2.set_ylabel("Rain, irrigation (mm d$^{-1}$)")

    hs, ls = map(list, axes[0].get_legend_handles_labels())
    for ax2 in [a for a in fig.axes if a not in axes]:
        for h, l in zip(*ax2.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    axes[0].legend(hs, ls, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_sample_{year}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_sample_{year}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()
    sm, rain, le, irr = load()

    print("T1 monthly mean difference vs rainfed Ne3 (m3/m3), observed days only")
    print("   2017-2024:")
    print(t1_monthly(sm, "2017-01-01", "2024-12-31").to_string(index=False))
    print("   2001-2016:")
    print(t1_monthly(sm, "2001-01-01", "2016-12-31").to_string(index=False))
    print(f"   yardstick: §66 OOS ubRMSE {UBRMSE_OOS}\n")

    print("T2 Jun-Sep per year: mean SM per site/bin, Ne3 tower rain, irrigation log")
    print(t2_yearly(sm, rain, irr).to_string(index=False), "\n")

    print(f"T3 May-Sep wetting jumps on rain-free days (d theta_0-10 > {WET_JUMP}, "
          f"rain < {DRY_MM} mm today and yesterday)")
    print(t3_dry_day_wetting(sm, rain).to_string(), "\n")

    print("T4 Jun-Sep mean LE (W m-2)")
    print(t4_le(le).to_string(index=False), "\n")

    sample_figure(sm, rain, irr, args.year)


if __name__ == "__main__":
    main()
