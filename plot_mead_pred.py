# ============================================================
# Mead Ne1 / Ne2: model time series vs measurements (runbook §69.3)
# ============================================================
# Observed = data_mead/observed (BASE, observed only, no gap-fill).
# Predicted = predict_mead_pixels.py (frozen §66 model, every day):
#   own-tile centre pixel  (solid, Python blue, on top of the black dots)
#   neighbour-tile pixel   (dashed, light blue: the same station read 546 m off-centre)
# One row per (station, depth).  Two figures: the full period and a zoom.
# Prints RMSE / ubRMSE / bias / R per station-depth (observed days only) and the
# observed-vs-predicted Ne1 - Ne2 difference (anomaly correlation, Jun-Sep 2019).
#
# Usage:
#   python plot_mead_pred.py [--zoom 2019 2020]
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
PRED    = ROOT / "pred" / "mead_pixels_2017-2024.parquet"
FIG_DIR = Path("/gpfs/work3/0/prjs1968/soilMoisture/figures/mead")
ST      = ["AmeriFlux_US-Ne1", "AmeriFlux_US-Ne2"]
BINS    = ["0-10", "10-30", "30-100"]
NAME    = {"AmeriFlux_US-Ne1": "Ne1 irrigated maize", "AmeriFlux_US-Ne2": "Ne2 irrigated maize/soy"}
PRED_COLOR, OFF_COLOR = "#3776AB", plot_style_bw.GREEN


def observed(st):
    ds = xr.open_dataset(ROOT / "observed" / f"{st}_observed.nc")
    t = pd.to_datetime(ds["date_time"].values)
    sm = pd.DataFrame(ds["soil_moisture"].values, index=t,
                      columns=[str(d) for d in ds["depth"].values])
    ts = pd.Series(ds["ts_degc"].values, index=t) if "ts_degc" in ds else None
    return sm, ts


def predicted(df, st, own: bool):
    """own=True: st at its own tile centre; False: st read from the other tile."""
    sel = df[(df["station"] == st) & ((df["tile"] == st) if own else (df["tile"] != st))]
    return sel.pivot_table(index="date", columns="depth", values="pred")


def metrics(o, p):
    j = pd.concat([o, p], axis=1, keys=["o", "p"]).dropna()
    if len(j) < 30:
        return dict(n=len(j))
    e = j["p"] - j["o"]
    return dict(n=len(j), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                ubrmse=np.sqrt(((e - e.mean()) ** 2).mean()), r=j["o"].corr(j["p"]))


def figure(obs, own, off, lo, hi, tag):
    plot_style_bw.apply(globals(), "paper")
    fig, axes = plt.subplots(6, 1, figsize=(12 if tag == "full" else 9, 14), sharex=True)
    vals = pd.concat([obs[s][0].loc[lo:hi] for s in ST] + [own[s].loc[lo:hi] for s in ST]).stack()
    y_lo, y_hi = np.nanpercentile(vals, 0.5) - 0.02, np.nanpercentile(vals, 99.5) + 0.03
    letters = iter("abcdef")
    for i, (s, b) in enumerate([(s, b) for s in ST for b in BINS]):
        ax = axes[i]
        ts = obs[s][1]
        if ts is not None:                               # frozen soil: TDR misses ice
            fz = ts.loc[lo:hi] <= 0
            for d in fz.index[fz.values]:
                ax.axvspan(d, d + pd.Timedelta("1D"), color="0.90", lw=0, zorder=0)
        o = obs[s][0][b].loc[lo:hi]
        ax.plot(o.index, o.values, "o", ms=1.6 if tag == "full" else 2.4, color="black",
                label="Observed", zorder=2)
        p2 = off[s][b].loc[lo:hi]
        ax.plot(p2.index, p2.values, "--", color=OFF_COLOR, lw=1.2,
                label="Model, read 546 m off-centre (neighbour tile)", zorder=3)
        p = own[s][b].loc[lo:hi]
        ax.plot(p.index, p.values, "-", color=PRED_COLOR, lw=1.4,
                label="Model, own tile centre", zorder=4)
        m = metrics(obs[s][0][b].loc[lo:hi], own[s][b].loc[lo:hi])
        txt = (f"({next(letters)}) {NAME[s]}, {b} cm" +
               (f"   ubRMSE {m['ubrmse']:.3f}  bias {m['bias']:+.3f}  R {m['r']:.2f}"
                if "r" in m else ""))
        ax.text(0.01, 0.95, txt, transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5), zorder=5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("SM (m$^3$ m$^{-3}$)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    fig.align_ylabels(axes)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_pred_vs_obs_{tag}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_pred_vs_obs_{tag}.png")


ZARR = Path("/gpfs/scratch1/shared/pkhanal/zarr/flux_only")
CM_BAD = [3, 4, 5, 255]          # dataset.py CM_BAD_CLASSES (SEnSeIv2 classes, not SCL)
CLEAR_WIN = 32                   # +-32 px at 10 m = 640 m box around the station


def acquisitions(st):
    """S1 ascending pass dates, and S2 dates with the clear fraction of the 640 m box
    around the station from the stored cloud mask (NaN = no mask -> unusable)."""
    import zarr
    g = zarr.open_group(str(ZARR / st), mode="r")
    to_dt = lambda a: pd.to_datetime([str(x)[:8] for x in np.asarray(a)], format="%Y%m%d")
    s1 = to_dt(g["s1_asc/dates"][:]) if "s1_asc" in g else pd.DatetimeIndex([])
    s2 = to_dt(g["s2/dates"][:])
    cmd = to_dt(g["cm/dates"][:])
    cm = g["cm/masks"]
    c, w = 112, CLEAR_WIN
    row = {d: i for i, d in enumerate(cmd)}
    clear = []
    for d in s2:
        i = row.get(d)
        if i is None:
            clear.append(np.nan)
            continue
        box = np.asarray(cm[i, c - w:c + w, c - w:c + w])
        clear.append(1.0 - np.isin(box, CM_BAD).mean())
    return s1, pd.Series(clear, index=s2)


def diff_figure(obs, own, df, lo, hi, tag, acq=None, precip=None, same_tile=False):
    """Ne1 - Ne2 per depth: observed vs model at each station's own tile centre.
    acq: optional (s1_dates, s2_clear_series, {site: irrigation DataFrame}) -> top strip.
    precip: optional {"ERA5": s, "US-Ne1": s, "US-Ne2": s} daily mm -> bottom panel (needs acq).
    same_tile: model Ne1 - Ne2 read inside ONE forward pass (Ne1 tile solid, Ne2 tile dashed)
    instead of two tile centres."""
    plot_style_bw.apply(globals(), "paper")
    a, b = ST
    within = (df[df["tile"] == a].pivot_table(index=["date", "depth"], columns="station",
                                              values="pred"))
    within = (within[a] - within[b]).unstack("depth")
    within2 = (df[df["tile"] == b].pivot_table(index=["date", "depth"], columns="station",
                                               values="pred"))
    within2 = (within2[a] - within2[b]).unstack("depth")
    if acq is None:
        fig, axes = plt.subplots(3, 1, figsize=(12 if tag == "full" else 9, 8), sharex=True)
    else:
        nr = 4 if precip is None else 5
        fig, ax_all = plt.subplots(nr, 1, figsize=(9, 9.6 if precip is None else 11.4), sharex=True,
                                   gridspec_kw={"height_ratios": [0.55, 1, 1, 1, 0.8][:nr]})
        axes = ax_all[1:4]
        _acq_strip(ax_all[0], acq, lo, hi)
    series = {}
    for dep in BINS:
        series[dep] = ((obs[a][0][dep] - obs[b][0][dep]).loc[lo:hi],
                       (within[dep] if same_tile else own[a][dep] - own[b][dep]).loc[lo:hi],
                       within2[dep].loc[lo:hi])
    lim = np.nanpercentile(np.abs(pd.concat([s for t in series.values() for s in t[:2]])), 99.5) + 0.01
    for ax, dep, lab in zip(axes, BINS, "abc" if acq is None else "bcd"):
        ts = obs[a][1]
        if ts is not None:
            fz = ts.loc[lo:hi] <= 0
            for d in fz.index[fz.values]:
                ax.axvspan(d, d + pd.Timedelta("1D"), color="0.90", lw=0, zorder=0)
        if precip is not None:                           # ERA5, Ne1 and Ne2 gauges all < 1 mm
            days = pd.date_range(lo, hi)
            dry = np.ones(len(days), bool)
            for k in ("ERA5", "US-Ne1", "US-Ne2"):
                dry &= precip[k].reindex(days).fillna(np.inf).values < 1.0
            for d in days[dry]:
                ax.axvspan(d, d + pd.Timedelta("1D"), color="0.85", lw=0, zorder=0)
        o, p, w = series[dep]
        ax.axhline(0, color="black", lw=1.0, zorder=1)
        o_d = o.asfreq("D")                              # NaN gaps break the line
        ax.plot(o_d.index, o_d.values, "-", color="black", lw=1.4,
                label="Observed Ne1 $-$ Ne2", zorder=2)
        ax.plot(p.index, p.values, "-", color=PRED_COLOR, lw=1.4,
                label="Model Ne1 $-$ Ne2" + (", Ne1 tile" if same_tile else ""), zorder=4)
        j = pd.concat([o, p], axis=1, keys=["o", "p"]).dropna()
        txt = (f"({lab}) {dep} cm   mean observed {j['o'].mean():+.3f}  "
               f"model {j['p'].mean():+.3f}   R {j['o'].corr(j['p']):+.2f}")
        if same_tile:
            ax.plot(w.index, w.values, "--", color=plot_style_bw.GREEN, lw=1.4,
                    label="Model Ne1 $-$ Ne2, Ne2 tile", zorder=3)
            j2 = pd.concat([o, w], axis=1, keys=["o", "w"]).dropna()
            txt = (f"({lab}) {dep} cm   mean obs {j['o'].mean():+.3f}   model Ne1 tile "
                   f"{j['p'].mean():+.3f} (R {j['o'].corr(j['p']):+.2f})   Ne2 tile "
                   f"{j2['w'].mean():+.3f} (R {j2['o'].corr(j2['w']):+.2f})")
        ax.text(0.01, 0.95, txt,
                transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5), zorder=5)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Ne1 $-$ Ne2 (m$^3$ m$^{-3}$)")
        if acq is not None:                              # clear S2 scenes as faint guides
            s2c = acq[1].loc[lo:hi]
            for d in s2c.index[s2c.values >= 0.99]:
                ax.axvline(d, color=plot_style_bw.RED, lw=0.6, alpha=0.5, zorder=1)
    if precip is not None:                               # ERA5 bars, tower gauges as markers
        ax = ax_all[4]
        e = precip["ERA5"].loc[lo:hi]
        ax.bar(e.index, e.values, width=1.0, color="0.6", label="ERA5-Land P", zorder=2)
        for s, mk, c in (("US-Ne1", "o", plot_style_bw.BLUE), ("US-Ne2", "^", plot_style_bw.GREEN)):
            g = precip[s].loc[lo:hi]
            g = g[g >= 0.2]
            ax.plot(g.index, g.values, mk, ms=4.5, color=c, label=f"{s[-3:]} gauge", zorder=3)
        ax.text(0.01, 0.95, "(e) precipitation", transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5), zorder=5)
        ax.set_ylabel("P (mm d$^{-1}$)")
        ax.set_xlim(pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta("1D"))
    top = axes[0] if acq is None else fig.axes[0]
    h, l = axes[0].get_legend_handles_labels()
    if acq is not None:
        h2, l2 = fig.axes[0].get_legend_handles_labels()
        h, l = h + h2, l + l2
        if precip is not None:
            h3, l3 = ax_all[4].get_legend_handles_labels()
            h, l = h + h3, l + l3
            from matplotlib.patches import Patch
            h.append(Patch(color="0.85"))
            l.append("No P (ERA5, Ne1, Ne2 < 1 mm)")
    top.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.02),
               ncol=2 if acq is None else 3, frameon=False)
    fig.align_ylabels(fig.axes)
    fig.tight_layout()
    suffix = ("" if acq is None else "_acq") + ("_sametile" if same_tile else "")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_diff_Ne1_Ne2_{tag}{suffix}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_diff_Ne1_Ne2_{tag}{suffix}.png")


def _acq_strip(ax, acq, lo, hi):
    """Rows: Ne1/Ne2 irrigation (bottom), S2 scenes by clear fraction, S1 ascending (top)."""
    s1, s2c, irr = acq
    s1 = s1[(s1 >= lo) & (s1 <= hi)]
    s2c = s2c.loc[lo:hi]
    ax.plot(s1, np.full(len(s1), 3), "|", ms=12, mew=1.6, color="black", label="S1 pass")
    ok = s2c.values >= 0.99
    part = (s2c.values > 0.5) & ~ok
    bad = ~(ok | part)                                   # cloudy or no cloud mask
    ax.plot(s2c.index[ok], np.full(ok.sum(), 2), "^", ms=6, color=plot_style_bw.RED,
            label="S2 clear (>=99 %)")
    ax.plot(s2c.index[part], np.full(part.sum(), 2), "^", ms=6, mfc="white",
            mec=plot_style_bw.RED, label="S2 partly cloudy")
    ax.plot(s2c.index[bad], np.full(bad.sum(), 2), "x", ms=5, color="0.5",
            label="S2 cloudy / no mask")
    for k, (site, e) in enumerate(irr.items()):
        if len(e):
            e = e[(e["date_start"] >= lo) & (e["date_start"] <= hi)]
            if "cancelled" in e:
                e = e[~e["cancelled"].fillna(False).astype(bool)]
            y = 1 if site.endswith("Ne1") else 0
            ax.plot(e["date_start"], np.full(len(e), y), "s", ms=6,
                    color=plot_style_bw.BLUE if y == 1 else plot_style_bw.GREEN,
                    label=f"Irrigation {site[-3:]}")
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Irr. Ne2", "Irr. Ne1", "S2", "S1"])
    ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, nargs="+", default=[2019, 2020])
    ap.add_argument("--diff-only", action="store_true", help="only the Ne1 - Ne2 figures")
    ap.add_argument("--per-year", action="store_true", help="Ne1 - Ne2 figure per year 2017-2024")
    ap.add_argument("--acq-year", type=int, default=None,
                    help="one Ne1 - Ne2 figure for this year with S1/S2/irrigation dates on top")
    ap.add_argument("--acq-window", nargs=2, default=None, metavar=("START", "END"),
                    help="as --acq-year over START..END, plus an ERA5 + tower-gauge precipitation panel")
    ap.add_argument("--same-tile", action="store_true",
                    help="with --acq-window: model Ne1 - Ne2 from one forward pass (both tiles shown)")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PRED)
    df["date"] = pd.to_datetime(df["date"])
    obs = {s: observed(s) for s in ST}
    own = {s: predicted(df, s, True) for s in ST}
    off = {s: predicted(df, s, False) for s in ST}

    print("Own-tile centre vs observed, 2017-2024 (observed days only)")
    for s in ST:
        for b in BINS:
            m = metrics(obs[s][0][b].loc["2017":"2024"], own[s][b])
            print(f"  {s[-3:]} {b:>6}: n {m['n']:>5}  RMSE {m.get('rmse', np.nan):.3f}  "
                  f"ubRMSE {m.get('ubrmse', np.nan):.3f}  bias {m.get('bias', np.nan):+.3f}  "
                  f"R {m.get('r', np.nan):.3f}")

    print("\nNe1 - Ne2 difference, observed vs predicted (own-tile centres)")
    for b in BINS:
        do = (obs[ST[0]][0][b] - obs[ST[1]][0][b]).loc["2017":"2024"]
        dp = own[ST[0]][b] - own[ST[1]][b]
        j = pd.concat([do, dp], axis=1, keys=["o", "p"]).dropna()
        jj = j[(j.index.year == 2019) & (j.index.month >= 6) & (j.index.month <= 9)]
        print(f"  {b:>6}: all-days corr {j['o'].corr(j['p']):+.3f}  "
              f"mean obs {j['o'].mean():+.3f} pred {j['p'].mean():+.3f} | "
              f"JJAS 2019 obs {jj['o'].mean():+.3f} pred {jj['p'].mean():+.3f}")

    lo, hi = f"{min(args.zoom)}-01-01", f"{max(args.zoom)}-12-31"
    zt = f"{min(args.zoom)}_{max(args.zoom)}"
    if not args.diff_only:
        figure(obs, own, off, "2017-01-01", "2024-12-31", "full")
        figure(obs, own, off, lo, hi, zt)
    if args.acq_year or args.acq_window:                # S1/S2/irrigation strip on top
        s1, s2c = acquisitions(ST[0])
        irr = {}
        for s in ST:
            f = ROOT / "bif" / f"{s.replace('AmeriFlux_', '')}_irrigation_events.csv"
            irr[s] = pd.read_csv(f, parse_dates=["date_start"]) if f.exists() else pd.DataFrame()
        if args.acq_window:
            from plot_mead_precip import load as load_precip
            lo, hi = args.acq_window
            diff_figure(obs, own, df, lo, hi, f"{lo.replace('-', '')}_{hi.replace('-', '')}",
                        acq=(s1, s2c, irr), precip=load_precip()[0], same_tile=args.same_tile)
            return
        y = args.acq_year
        print(f"{y}: S1 passes {int(((s1 >= f'{y}-01-01') & (s1 <= f'{y}-12-31')).sum())}, "
              f"S2 scenes {len(s2c.loc[str(y)])} "
              f"(clear {int((s2c.loc[str(y)] >= 0.99).sum())}, "
              f"no mask {int(s2c.loc[str(y)].isna().sum())})")
        diff_figure(obs, own, df, f"{y}-01-01", f"{y}-12-31", str(y), acq=(s1, s2c, irr))
        return
    if args.per_year:                                   # one Ne1 - Ne2 figure per year
        for y in range(2017, 2025):
            diff_figure(obs, own, df, f"{y}-01-01", f"{y}-12-31", str(y))
    else:
        diff_figure(obs, own, df, "2017-01-01", "2024-12-31", "full")
        diff_figure(obs, own, df, lo, hi, zt)


if __name__ == "__main__":
    main()
