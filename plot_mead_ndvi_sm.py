# ============================================================
# Mead: NDVI difference vs soil-moisture difference, Ne1 - Ne2 (runbook §69.5)
# ============================================================
# NDVI from the raw S2 store the model reads, both towers read in Ne1's tile
# (5 x 5 px = 50 m box), CLEAR scenes only: cloud-mask clear fraction >= 0.99 in the
# 640 m box AND no snow (mean B02 at both towers < SNOW_DN; the cloud mask calls snow clear).
# SM difference on the scene date: observed, model (own tile centres), model (same tile).
# Figures: mead_ndvi_sm_<year>       time series (dNDVI; dSM 0-10, 10-30)
#          mead_ndvi_sm_scatter_<year>  dNDVI vs dSM, Spearman, Jun-Sep highlighted
#
# Usage:
#   python plot_mead_ndvi_sm.py [--year 2019]
# ============================================================

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import spearmanr

import plot_style_bw
from plot_mead_pred import ROOT, FIG_DIR, PRED, PRED_COLOR, observed, predicted, acquisitions
from plot_mead_s2 import RAW, B, BOX, tower_px

ST      = {"Ne1": "AmeriFlux_US-Ne1", "Ne2": "AmeriFlux_US-Ne2"}
BINS    = ["0-10", "10-30"]
SNOW_DN = 1500          # B02 reflectance 0.15: bare soil / canopy are well below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    y = args.year
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(RAW), mode="r")
    g = root["s2"]
    dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d) for d in g["dates"][:]],
                           format="%Y%m%d")
    _, clear = acquisitions(ST["Ne1"])
    clear = clear[~clear.index.duplicated()]
    px = tower_px(dict(root.attrs))
    rows = []
    for i in np.where(dates.year == y)[0]:
        a = np.asarray(g["data"][i]).astype(np.float32)
        r = dict(date=dates[i], clear=clear.get(dates[i], np.nan))
        for s, (rr, cc) in px.items():
            ri, ci = int(round(rr)), int(round(cc))
            box = (slice(ri - BOX, ri + BOX + 1), slice(ci - BOX, ci + BOX + 1))
            nir, red = a[B["B08"]][box], a[B["B04"]][box]
            r[f"ndvi_{s[-3:]}"] = float(np.mean((nir - red) / np.maximum(nir + red, 1)))
            r[f"blue_{s[-3:]}"] = float(a[B["B02"]][box].mean())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("date")
    t["snow"] = (t[["blue_Ne1", "blue_Ne2"]] >= SNOW_DN).any(axis=1)
    t["use"] = (t["clear"] >= 0.99) & ~t["snow"]
    print(f"{y}: {len(t)} S2 scenes; cloud-clear {int((t['clear'] >= 0.99).sum())}, "
          f"snow {int(t['snow'].sum())} ({', '.join(f'{d:%m-%d}' for d in t.index[t['snow']])}), "
          f"used {int(t['use'].sum())}")
    t = t[t["use"]].copy()
    t["dndvi"] = t["ndvi_Ne1"] - t["ndvi_Ne2"]

    df = pd.read_parquet(PRED)
    df["date"] = pd.to_datetime(df["date"])
    obs = {k: observed(v)[0] for k, v in ST.items()}
    own = {k: predicted(df, v, True) for k, v in ST.items()}
    w = df[df["tile"] == ST["Ne1"]].pivot_table(index=["date", "depth"], columns="station", values="pred")
    same = (w[ST["Ne1"]] - w[ST["Ne2"]]).unstack("depth")
    for b in BINS:
        t[f"obs_{b}"] = (obs["Ne1"][b] - obs["Ne2"][b]).reindex(t.index)
        t[f"own_{b}"] = (own["Ne1"][b] - own["Ne2"][b]).reindex(t.index)
        t[f"same_{b}"] = same[b].reindex(t.index)
    out = ROOT / "pred" / f"ndvi_sm_diff_{y}.csv"
    t.to_csv(out)
    js = (t.index.month >= 6) & (t.index.month <= 9)
    print("\nSpearman of dNDVI (Ne1 - Ne2) with dSM (Ne1 - Ne2), clear scenes:  all / Jun-Sep / Oct-May")
    for b in BINS:
        for lab in ("obs", "own", "same"):
            o = []
            for sel in (np.ones(len(t), bool), js, ~js):
                x, z = t.loc[sel, "dndvi"].values, t.loc[sel, f"{lab}_{b}"].values
                ok = np.isfinite(x) & np.isfinite(z)
                o.append(f"{spearmanr(x[ok], z[ok])[0]:+.2f} (n {ok.sum():2d})" if ok.sum() >= 5 else "  n/a")
            print(f"  {b:>5} {lab:>4}: " + "   ".join(o))
    print(f"\ndNDVI Jun-Sep mean {t.loc[js, 'dndvi'].mean():+.3f} (|max| {t.loc[js, 'dndvi'].abs().max():.3f}), "
          f"Oct-May mean {t.loc[~js, 'dndvi'].mean():+.3f} (|max| {t.loc[~js, 'dndvi'].abs().max():.3f})")
    print(f"table: {out}")

    plot_style_bw.apply(globals(), "paper")
    lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    irr = pd.read_csv(ROOT / "bif" / "US-Ne2_irrigation_events.csv", parse_dates=["date_start", "date_end"])
    irr = irr[(irr["date_start"].dt.year == y) & ~irr["cancelled"].fillna(False).astype(bool)]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].plot(t.index, t["dndvi"], "-o", ms=4, lw=0.8, color=plot_style_bw.RED, label="NDVI Ne1 $-$ Ne2")
    ax[0].set_ylabel("$\\Delta$NDVI")
    ax[0].text(0.01, 0.95, "(a) NDVI Ne1 $-$ Ne2, clear scenes", transform=ax[0].transAxes, va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
    for k, b in enumerate(BINS, start=1):
        a = ax[k]
        a.axhline(0, color="black", lw=0.8)
        o = (obs["Ne1"][b] - obs["Ne2"][b]).loc[lo:hi].asfreq("D")
        a.plot(o.index, o.values, "-", color="black", lw=1.0, label="SM Ne1 $-$ Ne2 observed")
        m = (own["Ne1"][b] - own["Ne2"][b]).loc[lo:hi]
        a.plot(m.index, m.values, "-", color=PRED_COLOR, lw=1.0, label="SM Ne1 $-$ Ne2 model")
        a.plot(t.index, t[f"obs_{b}"], "o", ms=4, color="black", mfc="white", label="observed on scene date")
        a.set_ylabel("$\\Delta$SM (m$^3$ m$^{-3}$)")
        a.text(0.01, 0.95, f"({'bc'[k - 1]}) {b} cm", transform=a.transAxes, va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
    for a in ax:
        for rr in irr.itertuples():
            a.axvspan(rr.date_start, rr.date_end + pd.Timedelta("1D"), color="0.88", lw=0, zorder=0)
    ax[2].set_xlim(lo, hi + pd.Timedelta("1D"))
    h, l = ax[0].get_legend_handles_labels()
    h2, l2 = ax[1].get_legend_handles_labels()
    h, l = h + h2 + [plt.Rectangle((0, 0), 1, 1, color="0.88")], l + l2 + ["Ne2 irrigation logged"]
    ax[0].legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    fig.align_ylabels(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_ndvi_sm_{y}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_ndvi_sm_{y}.png")

    fig, ax = plt.subplots(2, 2, figsize=(8, 7.5), sharex=True)
    for j, b in enumerate(BINS):
        for k, (lab, name) in enumerate((("obs", "observed"), ("own", "model"))):
            a = ax[k, j]
            a.axhline(0, color="0.6", lw=0.6)
            a.axvline(0, color="0.6", lw=0.6)
            for sel, mk, c, ll in ((js, "o", plot_style_bw.BLUE, "Jun-Sep"),
                                   (~js, "s", plot_style_bw.RED, "Oct-May")):
                a.plot(t.loc[sel, "dndvi"], t.loc[sel, f"{lab}_{b}"], mk, ms=5, color=c, mec="black",
                       mew=0.4, label=ll)
            x, z = t["dndvi"].values, t[f"{lab}_{b}"].values
            ok = np.isfinite(x) & np.isfinite(z)
            rho = spearmanr(x[ok], z[ok])[0]
            a.text(0.03, 0.97, f"({'abcd'[2 * k + j]}) {b} cm, {name}\nSpearman {rho:+.2f} (n {ok.sum()})",
                   transform=a.transAxes, va="top", fontsize="small",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
            if j == 0:
                a.set_ylabel("$\\Delta$SM Ne1 $-$ Ne2 (m$^3$ m$^{-3}$)")
            if k == 1:
                a.set_xlabel("$\\Delta$NDVI Ne1 $-$ Ne2")
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_ndvi_sm_scatter_{y}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_ndvi_sm_scatter_{y}.png")


if __name__ == "__main__":
    main()
