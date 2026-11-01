"""Does HAND modulate soil moisture DYNAMICS rather than level? (§32.10 follow-up)

§32.10's gate tested station-mean ΔSM against ΔHAND and found nothing: r = -0.102,
p = 0.48, with no wet/dry contrast and a slope that weakens when restricted to
high-contrast pairs. But it only ever asked about the LEVEL.

A site near drainage need not sit wetter on average. Lateral subsurface convergence
delivers water slowly and continuously, so its signature can be in the TIME BEHAVIOUR:
a low-HAND site drains more slowly, holds a longer memory, and recedes toward a higher
residual, while its neighbour on the shoulder dries out fast after every storm. Two
stations can share a mean and differ entirely in how they get there.

That is a genuinely different hypothesis from the one §32.10 rejected, it is consistent
with every measurement so far, and it is the last cheap test before terrain is pruned.

Metrics per station, all computed on the pair's COMMON observed dates so weather is
held fixed (two stations measured in different years differ because the years differed):

  tau           e-folding drydown time, median over drydown segments. Fitted as
                ln(SM - SM_floor) ~ -t/tau on each monotone recession of >= 5 days.
                THE headline: convergence should lengthen it.
  ac1           lag-1 autocorrelation of the deviation from a 30-day rolling mean —
                short-term memory, with the seasonal cycle removed so it is not just
                measuring summer.
  wet_response  mean positive daily increment: how hard the site responds to rain.
  recession_min the residual the site recedes toward, as a fraction of its own range.
  sd_t          temporal sd on the common dates.

PREDICTION IF HAND MATTERS DYNAMICALLY
  Δtau  vs ΔHAND : NEGATIVE  (higher above drainage -> drains faster)
  Δac1  vs ΔHAND : NEGATIVE  (higher above drainage -> less memory)
  Δrecession_min vs ΔHAND : NEGATIVE (higher above drainage -> recedes drier)
A null on all three, having already failed on level, retires terrain for this dataset.

Usage
    conda activate terramind
    python probe_drydown_dynamics.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from math import erfc, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from gate_sm_vs_terrain import MIN_COMMON, station_series

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parent
OUT_DIR = REPO / "figures"

MIN_SEG   = 5      # a recession must last this many days to be fitted
TAU_LO, TAU_HI = 0.5, 200.0    # reject degenerate fits rather than let them dominate


def contiguous_runs(s: pd.Series):
    """Split a gappy daily series into runs of consecutive calendar days."""
    if len(s) < 2:
        return []
    gap = s.index.to_series().diff().dt.days.fillna(1).to_numpy()
    cut = np.where(gap != 1)[0]
    return [r for r in np.split(np.arange(len(s)), cut) if len(r) >= MIN_SEG]


def drydown_tau(s: pd.Series) -> tuple[float, int, float]:
    """
    Median e-folding time over monotone recessions, the number of segments used, and
    the median residual level those recessions head toward (as a fraction of range).

    Segments are runs of consecutive falling days, which on a daily series IS a
    drydown — no rainfall product is needed to define one, and using one would import
    ERA5's grid resolution into a per-location question.
    """
    taus, floors = [], []
    rng = float(s.max() - s.min())
    for run in contiguous_runs(s):
        v = s.iloc[run].to_numpy()
        d = np.diff(v)
        falling = d < 0
        i = 0
        while i < len(falling):
            if not falling[i]:
                i += 1
                continue
            j = i
            while j < len(falling) and falling[j]:
                j += 1
            seg = v[i:j + 1]
            if len(seg) >= MIN_SEG:
                floor = seg.min() - 0.002
                y = seg - floor
                if np.all(y > 0):
                    t = np.arange(len(seg), dtype=float)
                    b = np.polyfit(t, np.log(y), 1)[0]
                    if b < 0:
                        tau = -1.0 / b
                        if TAU_LO < tau < TAU_HI:
                            taus.append(tau)
                            if rng > 1e-6:
                                floors.append((seg.min() - s.min()) / rng)
            i = j + 1
    return (float(np.median(taus)) if taus else np.nan,
            len(taus),
            float(np.median(floors)) if floors else np.nan)


def metrics(s: pd.Series) -> dict:
    tau, n_seg, floor = drydown_tau(s)
    # short-term memory with the seasonal cycle taken out
    anom = s - s.rolling(31, center=True, min_periods=10).mean()
    ac = []
    for run in contiguous_runs(anom.dropna()):
        v = anom.dropna().iloc[run].to_numpy()
        if len(v) > 10 and np.std(v) > 1e-8:
            ac.append(float(np.corrcoef(v[:-1], v[1:])[0, 1]))
    inc = np.diff(s.to_numpy())
    return {"tau": tau, "n_drydowns": n_seg, "recession_min": floor,
            "ac1": float(np.median(ac)) if ac else np.nan,
            "wet_response": float(np.mean(inc[inc > 0.005])) if (inc > 0.005).any() else np.nan,
            "sd_t": float(s.std()), "mean": float(s.mean())}


def corr_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5:
        return {"r": np.nan, "p": np.nan, "n": n}
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    t = abs(r) * np.sqrt(n - 2) / np.sqrt(max(1 - r * r, 1e-12))
    return {"r": r, "p": float(erfc(t / sqrt(2))), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description="HAND vs soil moisture dynamics")
    ap.add_argument("--depth", default="0-10")
    args = ap.parse_args()

    st = pd.read_csv(REPO / "csvs" / "gate_station_table.csv").set_index("station_id")
    pairs = pd.read_csv(REPO / "csvs" / "colocated_pairs.csv")

    rows, cache = [], {}
    for _, p in pairs.iterrows():
        a, b = p["station_a"], p["station_b"]
        if a not in st.index or b not in st.index:
            continue
        for s in (a, b):
            if s not in cache:
                cache[s] = station_series(s)
        sa, sb = cache[a].get(args.depth), cache[b].get(args.depth)
        if sa is None or sb is None:
            continue
        common = sa.index.intersection(sb.index)
        if len(common) < MIN_COMMON:
            continue
        ma, mb = metrics(sa.loc[common]), metrics(sb.loc[common])
        rec = {"station_a": a, "station_b": b, "sep_m": p["sep_m"],
               "n_common": len(common),
               "d_hand": st.loc[a, "hand"] - st.loc[b, "hand"],
               "kg_macro": st.loc[a, "kg_macro"]}
        for k in ma:
            rec[f"d_{k}"] = ma[k] - mb[k]
            rec[f"a_{k}"], rec[f"b_{k}"] = ma[k], mb[k]
        rows.append(rec)

    d = pd.DataFrame(rows)
    print(f"{len(d)} pairs with >= {MIN_COMMON} common observed days at {args.depth} cm")
    print(f"drydown segments per station: median "
          f"{np.nanmedian(pd.concat([d.a_n_drydowns, d.b_n_drydowns])):.0f}")
    print(f"tau across stations: median "
          f"{np.nanmedian(pd.concat([d.a_tau, d.b_tau])):.1f} d   "
          f"IQR {np.nanpercentile(pd.concat([d.a_tau, d.b_tau]), 25):.1f}"
          f"-{np.nanpercentile(pd.concat([d.a_tau, d.b_tau]), 75):.1f} d")

    res = {}
    print("\n  DYNAMICS GATE — Δmetric vs ΔHAND   (prediction: all NEGATIVE)")
    tests = [("d_tau", "Δ drydown tau (days)"),
             ("d_ac1", "Δ lag-1 memory"),
             ("d_recession_min", "Δ recession floor"),
             ("d_wet_response", "Δ wetting response"),
             ("d_sd_t", "Δ temporal sd"),
             ("d_mean", "Δ mean level (§32.10's test, for reference)")]
    for col, lab in tests:
        r = corr_p(d["d_hand"], d[col])
        res[col] = r
        star = "  <<<" if (np.isfinite(r["p"]) and r["p"] < 0.05) else ""
        print(f"    {lab:<44} r = {r['r']:+.3f}  p = {r['p']:.3f}  n = {r['n']}{star}")

    print("\n  Δtau vs ΔHAND by Koppen macro-climate")
    for kg, g in d.groupby("kg_macro"):
        if g["d_tau"].notna().sum() >= 5:
            r = corr_p(g["d_hand"], g["d_tau"])
            res[f"tau_kg_{kg}"] = r
            print(f"    kg {kg:<41} r = {r['r']:+.3f}  p = {r['p']:.3f}  n = {r['n']}")

    print("\n  restricted to pairs with real HAND contrast (Δtau)")
    for thr in (0.0, 2.0, 5.0):
        g = d[d.d_hand.abs() >= thr]
        r = corr_p(g["d_hand"], g["d_tau"])
        res[f"tau_thr{thr:g}"] = r
        print(f"    |ΔHAND| >= {thr:4.1f} m{'':<30} r = {r['r']:+.3f}  "
              f"p = {r['p']:.3f}  n = {r['n']}")

    d.to_csv(REPO / "csvs" / "drydown_pair_metrics.csv", index=False)
    (REPO / "csvs" / "drydown_results.json").write_text(json.dumps(res, indent=2, default=float))

    # ── figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5})
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)
    panels = [("d_tau", "Δ drydown $\\tau$ (days)", "A  drydown speed"),
              ("d_ac1", "Δ lag-1 memory", "B  short-term memory"),
              ("d_recession_min", "Δ recession floor (frac of range)", "C  residual wetness"),
              ("d_wet_response", "Δ wetting response (m$^3$/m$^3$)", "D  storm response")]
    for k, (col, ylab, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        v = d.dropna(subset=["d_hand", col])
        ax.scatter(v["d_hand"], v[col], s=36, c="#2c7fb8", edgecolor="k", linewidth=0.35)
        if len(v) > 3:
            xs = np.linspace(v["d_hand"].min(), v["d_hand"].max(), 20)
            ax.plot(xs, np.polyval(np.polyfit(v["d_hand"], v[col], 1), xs),
                    color="0.25", ls="--", lw=1.3)
            r = corr_p(v["d_hand"], v[col])
            ax.set_title(f"{title}   r = {r['r']:+.3f}, p = {r['p']:.3f}, n = {r['n']}")
        else:
            ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.axvline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel("$\\Delta$HAND (m)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)

    fig.suptitle("Does HAND modulate soil moisture DYNAMICS? — colocated pairs, "
                 "common dates   (prediction: all slopes negative)", fontsize=11, y=0.975)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "drydown_vs_hand"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"\nwrote {out}.png / .pdf, csvs/drydown_pair_metrics.csv")


if __name__ == "__main__":
    main()
