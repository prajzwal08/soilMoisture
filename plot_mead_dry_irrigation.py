# ============================================================
# Mead: rain-free irrigation events, observed vs model (runbook §69.5)
# ============================================================
# ERA5 forcing is the same for Ne1 and Ne2 and has no irrigation in it, so on a
# rain-free day any predicted wetting can only come from imagery (mainly S1).
#
# Rain-free day: ERA5 tp < 1 mm AND Ne3 gauge (rainfed, no pivot) < 1 mm AND the
#   Ne1/Ne2 gauges < 1 mm, except that a site's own gauge may be wet inside its logged
#   irrigation interval (date_start .. date_end + 1): that reading is the pivot.
# Event: a logged, non-cancelled irrigation at Ne1 or Ne2.  d0 = the day of the largest
#   gauge reading inside the interval (pivot over the tower), else date_start.
#   Kept only if every day in [d0 - 3, d0 + 3] is rain-free.
# Per event: observed and predicted dSM (d0-1 -> d0+1, d0-1 -> d0+3) at 0-10/10-30 cm
#   for the irrigated site, the other irrigated site, and Ne3 (observed only);
#   S1 passes in [d0, d0+3] (none = the model cannot see it).
#
# Usage:
#   python plot_mead_dry_irrigation.py [--year 2018]
#   python plot_mead_dry_irrigation.py --pair [--year 2018]   (Ne2 logged irrigation, ERA5/Ne1/Ne2 dry; Ne2 vs Ne1)
#   python plot_mead_dry_irrigation.py --pair --year 2019 --window 2019-08-01 2019-08-31
# ============================================================

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style_bw
from plot_mead_precip import era5_tp, load as load_precip
from plot_mead_pred import ROOT, FIG_DIR, PRED, PRED_COLOR, observed, predicted, acquisitions

DRY_MM = 1.0
HALF   = 3          # rain-free half window around d0
PAD    = 10         # figure half window
IRR    = ["US-Ne1", "US-Ne2"]
BINS   = ["0-10", "10-30"]


def intervals(irr, year):
    """Logged, non-cancelled events of one year -> list of (start, end+1)."""
    e = irr[(irr["date_start"].dt.year == year)].copy()
    e["date_end"] = pd.to_datetime(e["date_end"]).fillna(e["date_start"])
    return [(r.date_start.normalize(), r.date_end.normalize() + pd.Timedelta("1D"), r.amount_in)
            for r in e.itertuples()]


def rain_free(p, irr, year):
    """Boolean daily series: True where nobody received rain (pivot water at its own gauge allowed)."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    ok = (p["ERA5"].reindex(days).fillna(np.inf) < DRY_MM) & \
         (p["US-Ne3"].reindex(days).fillna(np.inf) < DRY_MM)
    for s in IRR:
        g = p[s].reindex(days)
        pivot = pd.Series(False, index=days)
        for a, b, _ in intervals(irr[s], year):
            pivot |= (days >= a) & (days <= b)
        ok &= (g.fillna(np.inf) < DRY_MM) | pivot
    return ok


def delta(series, d0, a, b):
    x = series.reindex([d0 + pd.Timedelta(f"{a}D"), d0 + pd.Timedelta(f"{b}D")])
    return x.iloc[1] - x.iloc[0]


def pair(p, irr, y, irr_site="US-Ne2", ctl="US-Ne1", window=None):
    """Ne1/Ne2 share weather.  Days inside a logged irr_site interval on which ERA5, the Ne1
    and the Ne2 gauge all read < DRY_MM: irrigation is logged but nothing reached a gauge.
    Per interval: observed and predicted SM change from start-1 to end+1, irrigated vs control.
    window=(lo, hi): one figure column over that period instead of one per interval;
    there every S1/S2 pass in the window gets a star."""
    df = pd.read_parquet(PRED)
    df["date"] = pd.to_datetime(df["date"])
    obs = {s: observed(f"AmeriFlux_{s}")[0] for s in (irr_site, ctl)}
    pred = {s: predicted(df, f"AmeriFlux_{s}", True) for s in (irr_site, ctl)}
    s1, s2c = acquisitions(f"AmeriFlux_{irr_site}")
    s2 = s2c.index
    ctl_iv = intervals(irr[ctl], y)
    g = {k: p[k] for k in ("ERA5", irr_site, ctl)}
    a_, c_ = irr_site[-3:], ctl[-3:]

    rows, events = [], []
    print(f"\n{y}: logged {a_} irrigation, days with ERA5, {a_} and {c_} gauges all < {DRY_MM} mm")
    for a, b, amt in intervals(irr[irr_site], y):
        days = pd.date_range(a, b - pd.Timedelta("1D"))            # logged start .. end
        dry = [d for d in days if all(g[k].get(d, np.inf) < DRY_MM for k in g)]
        ov = [f"{x:%m-%d}..{(z - pd.Timedelta('1D')):%m-%d}" for x, z, _ in ctl_iv
              if x <= b and z >= a - pd.Timedelta("1D")]
        lo, hi = a - pd.Timedelta("1D"), b
        r = dict(start=a.date(), end=(b - pd.Timedelta("1D")).date(),
                 amount_mm=round(amt * 25.4, 1), n_days=len(days), n_all_dry=len(dry),
                 all_dry_days=",".join(f"{d:%m-%d}" for d in dry),
                 gauge_max={k[-4:] if k != "ERA5" else k: round(float(g[k].loc[a:b].max()), 1)
                            for k in g},
                 ctl_irrigated=",".join(ov),
                 s1=",".join(f"{d:%m-%d}" for d in s1[(s1 >= a) & (s1 < b)]),
                 s2=",".join(f"{d:%m-%d}({s2c[d]:.0%} clear)" for d in s2[(s2 >= a) & (s2 < b)]))
        for bn in BINS:
            for s in (irr_site, ctl):
                r[f"obs{s[-3:]}_{bn}"] = delta(obs[s][bn], lo, 0, (hi - lo).days)
                r[f"pred{s[-3:]}_{bn}"] = delta(pred[s][bn], lo, 0, (hi - lo).days)
        rows.append(r)
        events.append((a, b - pd.Timedelta("1D"), dry))
        print(f"  {a:%m-%d}..{b - pd.Timedelta('1D'):%m-%d} {r['amount_mm']:5.1f} mm: "
              f"{len(dry)}/{len(days)} all-dry days [{r['all_dry_days']}]  "
              f"max gauge {r['gauge_max']}  {c_} irrigated: {r['ctl_irrigated'] or 'no'}  "
              f"S1: {r['s1'] or 'none'}  S2: {r['s2'] or 'none'}")
        for bn in BINS:
            o = r[f"obs{a_}_{bn}"] - r[f"obs{c_}_{bn}"]
            pr = r[f"pred{a_}_{bn}"] - r[f"pred{c_}_{bn}"]
            print(f"      {bn:>5} dSM start-1 -> end+1: obs {a_} {r[f'obs{a_}_{bn}']:+.3f} "
                  f"{c_} {r[f'obs{c_}_{bn}']:+.3f} ({a_}-{c_} {o:+.3f}) | model {a_} "
                  f"{r[f'pred{a_}_{bn}']:+.3f} {c_} {r[f'pred{c_}_{bn}']:+.3f} ({a_}-{c_} {pr:+.3f})")
    out = ROOT / "pred" / f"pair_irrigation_{a_}_{y}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"table: {out}")
    if not events and window is None:
        return
    if window is None:                                   # one column per logged interval
        cols = [dict(lo=a - pd.Timedelta(f"{PAD}D"), hi=e + pd.Timedelta(f"{PAD}D"), ivs=[(a, e)],
                     dry=dry, star=[(a, e)], text=f", {a_} irrigation {a:%d %b}-{e:%d %b}")
                for a, e, dry in events]
        tag = str(y)
    else:                                                # one column over the window
        lo, hi = (pd.Timestamp(w) for w in window)
        iv = [(a, e, d) for a, e, d in events if a <= hi and e >= lo]
        cols = [dict(lo=lo, hi=hi, ivs=[(a, e) for a, e, _ in iv], dry=[d for *_, ds in iv for d in ds],
                     star=[(lo, hi)], text=f", {lo:%d %b}-{hi:%d %b %Y}")]
        tag = f"{lo:%Y%m%d}_{hi:%Y%m%d}"

    plot_style_bw.apply(globals(), "paper")
    n = len(cols)
    fig, axes = plt.subplots(3, n, figsize=(max(3.6 * n, 10), 8.5), sharex="col",
                             squeeze=False, gridspec_kw={"height_ratios": [1, 1, 0.8]})
    col = {irr_site: plot_style_bw.BLUE, ctl: plot_style_bw.GREEN}
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for j, c in enumerate(cols):
        lo, hi = c["lo"], c["hi"]
        for ax in axes[:, j]:
            for a, e in c["ivs"]:
                ax.axvspan(a, e + pd.Timedelta("1D"), color="0.92", lw=0, zorder=0)
            for d in c["dry"]:
                ax.axvspan(d, d + pd.Timedelta("1D"), color="#FDD0A2", alpha=0.6,
                           lw=0, zorder=0)
        for ax in axes[:, j]:                            # passes inside the starred span(s)
            x = ax.get_xaxis_transform()
            in1 = np.zeros(len(s1), bool)
            in2 = np.zeros(len(s2), bool)
            for a, e in c["star"]:
                in1 |= (s1 >= a) & (s1 < e + pd.Timedelta("1D"))
                in2 |= (s2 >= a) & (s2 < e + pd.Timedelta("1D"))
            p1, p2 = s1[in1], s2[in2]
            ax.plot(p1 + pd.Timedelta("12h"), np.full(len(p1), 0.80), "*", ms=11, color="black",
                    transform=x, zorder=6)
            ax.plot(p2 + pd.Timedelta("12h"), np.full(len(p2), 0.80), "*", ms=11,
                    color=plot_style_bw.RED, mec="black", mew=0.4, transform=x, zorder=6)
        ax = axes[2, j]
        er = p["ERA5"].loc[lo:hi]
        ax.bar(er.index, er.values, width=1.0, color="0.6", label="ERA5-Land")
        for s, mk in ((ctl, "o"), (irr_site, "^")):
            gg = p[s].loc[lo:hi]
            gg = gg[gg >= 0.2]
            ax.plot(gg.index, gg.values, mk, ms=5, color=col[s], label=f"{s[-3:]} gauge")
        ax.text(0.03, 0.95, f"({chr(ord('a') + 3 * j + 2)}) precipitation", transform=ax.transAxes,
                va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
        if j == 0:
            ax.set_ylabel("P (mm d$^{-1}$)")
        for i, bn in enumerate(BINS):
            ax = axes[i, j]
            for s in (irr_site, ctl):
                o = obs[s][bn].loc[lo:hi]
                ax.plot(o.index, o.values, "o", ms=2.8, color=col[s], label=f"Observed {s[-3:]}")
                pp = pred[s][bn].loc[lo:hi]
                ax.plot(pp.index, pp.values, "-" if s == irr_site else "--", color=col[s],
                        lw=1.5, label=f"Model {s[-3:]}")
            ax.text(0.03, 0.95, f"({chr(ord('a') + 3 * j + i)}) {bn} cm"
                    + (c["text"] if i == 0 else ""),
                    transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
            if j == 0:
                ax.set_ylabel("SM (m$^3$ m$^{-3}$)")
        axes[2, j].tick_params(axis="x", rotation=45)
        axes[2, j].set_xlim(lo, hi + pd.Timedelta("1D"))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="o", ls="", ms=4, color=col[irr_site], label=f"Observed {a_}"),
         Line2D([], [], ls="-", lw=1.5, color=col[irr_site], label=f"Model {a_}"),
         Line2D([], [], marker="o", ls="", ms=4, color=col[ctl], label=f"Observed {c_}"),
         Line2D([], [], ls="--", lw=1.5, color=col[ctl], label=f"Model {c_}"),
         Line2D([], [], marker="*", ls="", ms=10, color="black", label="S1 pass"),
         Line2D([], [], marker="*", ls="", ms=10, color=plot_style_bw.RED, mec="black",
                mew=0.4, label="S2 pass"),
         Patch(color="0.6", label="ERA5-Land P"),
         Line2D([], [], marker="^", ls="", ms=5, color=col[irr_site], label=f"{a_} gauge"),
         Line2D([], [], marker="o", ls="", ms=5, color=col[ctl], label=f"{c_} gauge"),
         Patch(color="0.92", label=f"{a_} irrigation logged"),
         Patch(color="#FDD0A2", alpha=0.6, label="ERA5, Ne1, Ne2 all dry")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_pair_irrigation_{a_}_{tag}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_pair_irrigation_{a_}_{tag}.png  (grey = logged interval, "
          f"peach = ERA5/Ne1/Ne2 all dry, black * = S1, orange * = S2 inside the interval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--pair", action="store_true",
                    help="Ne2 logged irrigation on days ERA5, Ne1 and Ne2 gauges all dry; Ne2 vs Ne1")
    ap.add_argument("--window", nargs=2, default=None, metavar=("START", "END"),
                    help="with --pair: one figure over this period, e.g. 2019-08-01 2019-08-31")
    args = ap.parse_args()
    y = args.year
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    p, irr = load_precip()
    if args.pair:
        pair(p, irr, y, window=args.window)
        return
    dry = rain_free(p, irr, y)
    print(f"{y}: {int(dry.sum())} rain-free days of {len(dry)} "
          f"(ERA5, Ne3 < {DRY_MM} mm; Ne1/Ne2 < {DRY_MM} mm outside own logged irrigation)")

    df = pd.read_parquet(PRED)
    df["date"] = pd.to_datetime(df["date"])
    obs = {s: observed(f"AmeriFlux_{s}")[0] for s in IRR + ["US-Ne3"]}
    pred = {s: predicted(df, f"AmeriFlux_{s}", True) for s in IRR}
    s1 = {s: acquisitions(f"AmeriFlux_{s}")[0] for s in IRR}

    rows, kept = [], []
    for s in IRR:
        for a, b, amt in intervals(irr[s], y):
            g = p[s].loc[a:b]
            d0 = g.idxmax() if len(g) and g.max() >= DRY_MM else a
            win = pd.date_range(d0 - pd.Timedelta(f"{HALF}D"), d0 + pd.Timedelta(f"{HALF}D"))
            ok = bool(dry.reindex(win).fillna(False).all())
            wet = [d.strftime("%m-%d") for d in win if not dry.get(d, False)]
            s1_in = s1[s][(s1[s] >= d0) & (s1[s] <= d0 + pd.Timedelta(f"{HALF}D"))]
            other = [o for o in IRR if o != s][0]
            r = dict(site=s[-3:], start=a.date(), d0=d0.date(), amount_mm=round(amt * 25.4, 1),
                     gauge_d0=round(float(p[s].get(d0, np.nan)), 1),
                     era5_d0=round(float(p["ERA5"].get(d0, np.nan)), 1),
                     ne3_d0=round(float(p["US-Ne3"].get(d0, np.nan)), 1),
                     rain_free=ok, wet_days=",".join(wet),
                     s1_d0_d3=",".join(d.strftime("%m-%d") for d in s1_in))
            for bn in BINS:
                for (k0, k1) in [(-1, 1), (-1, 3)]:
                    t = f"{bn}_{k0}_{k1}"
                    r[f"obs_{t}"] = delta(obs[s][bn], d0, k0, k1)
                    r[f"pred_{t}"] = delta(pred[s][bn], d0, k0, k1)
                    r[f"obs{other[-3:]}_{t}"] = delta(obs[other][bn], d0, k0, k1)
                    r[f"predNe{other[-1]}_{t}"] = delta(pred[other][bn], d0, k0, k1)
                    r[f"obsNe3_{t}"] = delta(obs["US-Ne3"][bn], d0, k0, k1)
            rows.append(r)
            if ok:
                kept.append((s, d0))
    tab = pd.DataFrame(rows).sort_values("d0")
    out = ROOT / "pred" / f"dry_irrigation_events_{y}.csv"
    tab.to_csv(out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(tab[["site", "start", "d0", "amount_mm", "gauge_d0", "era5_d0", "ne3_d0",
               "rain_free", "wet_days", "s1_d0_d3"]].to_string(index=False))
    print("\ndSM d0-1 -> d0+1 | d0-1 -> d0+3   (irrigated site obs / pred, Ne3 obs)")
    for _, r in tab.iterrows():
        for bn in BINS:
            print(f"  {r['site']} {r['d0']} {'KEPT' if r['rain_free'] else 'rain'} {bn:>5}: "
                  f"obs {r[f'obs_{bn}_-1_1']:+.3f} pred {r[f'pred_{bn}_-1_1']:+.3f} "
                  f"Ne3 {r[f'obsNe3_{bn}_-1_1']:+.3f} | "
                  f"obs {r[f'obs_{bn}_-1_3']:+.3f} pred {r[f'pred_{bn}_-1_3']:+.3f} "
                  f"Ne3 {r[f'obsNe3_{bn}_-1_3']:+.3f}")
    print(f"table: {out}")
    if not kept:
        print("no rain-free event, no figure")
        return

    plot_style_bw.apply(globals(), "paper")
    n = len(kept)
    fig, axes = plt.subplots(3, n, figsize=(3.6 * n, 8), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [0.8, 1, 1]})
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for j, (s, d0) in enumerate(kept):
        lo, hi = d0 - pd.Timedelta(f"{PAD}D"), d0 + pd.Timedelta(f"{PAD}D")
        ax = axes[0, j]
        e = p["ERA5"].loc[lo:hi]
        ax.bar(e.index, e.values, width=1.0, color="0.6", label="ERA5-Land")
        for k, mk, c in [("US-Ne1", "o", plot_style_bw.BLUE), ("US-Ne2", "^", plot_style_bw.GREEN),
                         ("US-Ne3", "x", plot_style_bw.RED)]:
            g = p[k].loc[lo:hi]
            g = g[g >= 0.2]
            ax.plot(g.index, g.values, mk, ms=5, color=c, label=f"{k[-3:]} gauge")
        for d in s1[s][(s1[s] >= lo) & (s1[s] <= hi)]:
            for a_ in axes[:, j]:
                a_.axvline(d, color="0.75", lw=0.8, ls=":", zorder=0)
        ax.text(0.03, 0.95, f"({next(letters)}) {s[-3:]} pivot {d0:%d %b %Y}",
                transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
        if j == 0:
            ax.set_ylabel("P (mm d$^{-1}$)")
        for i, bn in enumerate(BINS, start=1):
            ax = axes[i, j]
            ax.axvspan(d0 - pd.Timedelta(f"{HALF}D"), d0 + pd.Timedelta(f"{HALF}D"),
                       color="0.93", lw=0, zorder=0)
            o = obs[s][bn].loc[lo:hi]
            ax.plot(o.index, o.values, "o", ms=3, color="black", label=f"Observed {s[-3:]}")
            o3 = obs["US-Ne3"][bn].loc[lo:hi]
            ax.plot(o3.index, o3.values, "s", ms=2.5, mfc="white", mec="0.45",
                    label="Observed Ne3 (rainfed)")
            pp = pred[s][bn].loc[lo:hi]
            ax.plot(pp.index, pp.values, "-", color=PRED_COLOR, lw=1.6, label=f"Model {s[-3:]}")
            ax.axvline(d0, color=plot_style_bw.RED, lw=1.0)
            ax.text(0.03, 0.95, f"({next(letters)}) {bn} cm", transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
            if j == 0:
                ax.set_ylabel("SM (m$^3$ m$^{-3}$)")
        axes[2, j].tick_params(axis="x", rotation=45)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False)
    axes[1, 0].legend(loc="lower left", bbox_to_anchor=(0, -0.02), frameon=False, fontsize="small")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_dry_irrigation_{y}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_dry_irrigation_{y}.png  (dotted = S1 pass, red = d0, "
          f"grey band = rain-free window)")


if __name__ == "__main__":
    main()
