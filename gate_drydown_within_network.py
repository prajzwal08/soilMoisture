"""THE decisive test: does HAND control drydown time? One pre-specified regression.

Everything before this searched. §32.10's level gate ran 6 metrics x 3 strata; the
dynamics probe ran 6 metrics x 3 thresholds and its best cell (Δtau vs ΔHAND = -0.398
at |ΔHAND| >= 5 m) failed a permutation test over that grid at corrected p = 0.196 —
not because the effect misbehaved, but because 49 pairs cannot outrun an 18-cell search.

Two facts from that probe justify one more test rather than none:
  - tau is a REAL station property, not a fitting artefact: tau(first half of record)
    vs tau(second half) correlates at r = +0.663, p < 0.001, over 97 station-halves.
  - the ΔHAND relationship replicated across those two independent time halves
    (-0.388 and -0.378) and strengthened monotonically with terrain contrast, which is
    the opposite of how the level effect behaved.

So the limitation was power, and power is available: pairing was bought to control
climate, protocol and sensor type, and NETWORK FIXED EFFECTS buy the same control over
890 stations instead of 49 pairs.

PRE-SPECIFIED, AND THIS IS THE WHOLE TEST:

    primary:   tau(0-10 cm) ~ HAND, within-network demeaned, all qualifying stations
    direction: NEGATIVE (higher above drainage -> faster drying)
    PASS:      r < 0 and p < 0.05
    ONE test. No thresholds, no strata, no metric selection. Anything else reported
    below is explicitly labelled secondary and carries no weight in the verdict.

At n ~ 800 this detects |r| = 0.07. If it comes back null, terrain is done for this
dataset at this depth, and §32.10's pruning stands without an asterisk.

Usage
    conda activate terramind
    python gate_drydown_within_network.py --workers 64
"""
from __future__ import annotations

import argparse
import json
import warnings
from math import erfc, sqrt
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gate_sm_vs_terrain import station_series
from probe_drydown_dynamics import metrics

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parent
OUT_DIR = REPO / "figures"
MIN_NET = 3       # a network must have this many stations to have a usable mean
MIN_DRYDOWNS = 10  # a station needs this many recessions for tau to mean anything


def one_station(sid: str) -> dict | None:
    ser = station_series(sid)
    s = ser.get("0-10")
    if s is None:
        return None
    m = metrics(s)
    m["station_id"] = sid
    return m


def demean_within(df: pd.DataFrame, col: str, by: str) -> pd.Series:
    return df[col] - df.groupby(by)[col].transform("mean")


def corr_p(x, y, dof_penalty: int = 0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return {"r": np.nan, "p": np.nan, "n": n}
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    # degrees of freedom lose one per network mean that was removed
    dof = max(n - 2 - dof_penalty, 1)
    t = abs(r) * np.sqrt(dof) / np.sqrt(max(1 - r * r, 1e-12))
    return {"r": r, "p": float(erfc(t / sqrt(2))), "n": n, "dof": dof}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    st = pd.read_csv(REPO / "csvs" / "gate_station_table.csv")
    st = st[np.isfinite(st["hand"])].reset_index(drop=True)
    print(f"computing drydown metrics for {len(st)} stations with Pool({args.workers})",
          flush=True)
    with Pool(args.workers) as pool:
        recs = [r for r in pool.imap_unordered(one_station, st["station_id"].tolist())
                if r is not None]
    dyn = pd.DataFrame(recs)
    df = st.merge(dyn, on="station_id", how="inner")

    df = df[(df["n_drydowns"] >= MIN_DRYDOWNS) & np.isfinite(df["tau"])]
    counts = df["network"].value_counts()
    df = df[df["network"].isin(counts[counts >= MIN_NET].index)].reset_index(drop=True)
    n_net = df["network"].nunique()
    print(f"{len(df)} stations across {n_net} networks with >= {MIN_NET} stations each")

    df["hand_w"] = demean_within(df, "hand", "network")
    df["tau_w"] = demean_within(df, "tau", "network")

    # ── THE test ─────────────────────────────────────────────────────────────
    primary = corr_p(df["hand_w"], df["tau_w"], dof_penalty=n_net)
    passed = bool(primary["r"] < 0 and primary["p"] < 0.05)
    print("\nPRIMARY (pre-specified, the whole test):")
    print(f"   tau ~ HAND, within-network demeaned   r = {primary['r']:+.4f}   "
          f"p = {primary['p']:.4f}   n = {primary['n']}   dof = {primary['dof']}")
    print(f"   -> {'PASS' if passed else 'FAIL'}  "
          f"(pass required r < 0 and p < 0.05)")

    res = {"primary": primary, "primary_pass": passed,
           "n_stations": int(len(df)), "n_networks": int(n_net),
           "detectable_r_at_n": float(1.96 / np.sqrt(max(len(df) - n_net, 2)))}
    print(f"   detectable |r| at this n: {res['detectable_r_at_n']:.3f}")

    # ── secondary, explicitly carrying no weight in the verdict ──────────────
    print("\nSECONDARY (no weight in the verdict, reported for completeness):")
    for col, lab in (("ac1", "lag-1 memory"), ("recession_min", "recession floor"),
                     ("mean", "mean level"), ("sd_t", "temporal sd")):
        if col in df:
            df[f"{col}_w"] = demean_within(df, col, "network")
            r = corr_p(df["hand_w"], df[f"{col}_w"], dof_penalty=n_net)
            res[f"secondary_{col}"] = r
            print(f"   {lab:<22} vs HAND   r = {r['r']:+.4f}  p = {r['p']:.4f}  n = {r['n']}")
    for kg, g in df.groupby("kg_macro"):
        if len(g) >= 30:
            r = corr_p(g["hand_w"], g["tau_w"], dof_penalty=g["network"].nunique())
            res[f"secondary_kg_{kg}"] = r
            print(f"   tau ~ HAND, Koppen {str(kg):<4}      r = {r['r']:+.4f}  "
                  f"p = {r['p']:.4f}  n = {r['n']}")

    (REPO / "csvs" / "drydown_within_network.json").write_text(
        json.dumps(res, indent=2, default=float))
    df.to_csv(REPO / "csvs" / "drydown_within_network.csv", index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.scatter(df["hand_w"], df["tau_w"], s=14, c="#2c7fb8", alpha=0.6,
               edgecolor="none")
    xs = np.linspace(df["hand_w"].quantile(0.01), df["hand_w"].quantile(0.99), 20)
    k = np.polyfit(df["hand_w"], df["tau_w"], 1)
    ax.plot(xs, np.polyval(k, xs), color="0.2", ls="--", lw=1.6)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("HAND, within-network anomaly (m)")
    ax.set_ylabel("drydown $\\tau$, within-network anomaly (days)")
    ax.set_title(f"PRIMARY  r = {primary['r']:+.4f}, p = {primary['p']:.4f}, "
                 f"n = {primary['n']}  [{'PASS' if passed else 'FAIL'}]")
    ax.grid(alpha=0.25)

    ax = axes[1]
    # binned means: a linear r can miss a monotone but non-linear control
    q = pd.qcut(df["hand_w"], 10, duplicates="drop")
    g = df.groupby(q)["tau_w"].agg(["mean", "sem", "size"])
    ctr = [iv.mid for iv in g.index]
    ax.errorbar(ctr, g["mean"], yerr=g["sem"], fmt="o-", color="#b2182b",
                capsize=3, lw=1.3)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("HAND within-network anomaly, decile centre (m)")
    ax.set_ylabel("mean $\\tau$ anomaly (days)")
    ax.set_title("binned by decile — would show a non-linear control")
    ax.grid(alpha=0.25)

    fig.suptitle("Decisive test: does HAND control drydown time? "
                 "One pre-specified regression, network fixed effects", fontsize=11)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "drydown_within_network"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"\nwrote {out}.png, csvs/drydown_within_network.json")


if __name__ == "__main__":
    main()
