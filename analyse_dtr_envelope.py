#!/usr/bin/env python
"""Is the DTR-vs-SM cloud a WEDGE, and if so is the wedge real?

The pooled scatter looks trapezoidal -- wide DTR spread on the dry side, narrowing
toward the wet side.  That shape matters, because a one-sided constraint ("wet soil
CANNOT swing much; dry soil MAY or may not, depending on cloud, canopy and insolation")
produces exactly a weak linear r alongside a strong physical relationship.  It is the
same structure as the Ts-VI trapezoid the TVDI literature is built on.

But the visual wedge is not evidence.  The x axis is in SM units and most station-days
sit at low SM, so the drawn cloud is wider on the left for no reason but sample count.

THE FAIR TEST: EQUAL-COUNT SM BINS.  Each bin holds the same n by construction, so a
narrowing of its QUANTILES cannot be a sampling artefact.  Reported per bin: n, p05,
p25, p50, p75, p90, p95, and the IQR.  Then a slope is fitted at each quantile across
bins -- if the upper quantiles fall much faster than the median, the relationship is an
upper bound and a Pearson r understates it.
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
from census_ecostress import ROOT  # noqa: E402

QS = [5, 25, 50, 75, 90, 95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "csvs" / "ecostress_dtr_vs_sm_all.csv"))
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--band", action="store_true",
                    help="restrict to the plausible 0-40 K DTR band")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)[["sm_surface", "dtr_k"]].dropna()
    if args.band:
        d = d[(d.dtr_k >= 0) & (d.dtr_k <= 40)]
    x, y = d.sm_surface.to_numpy(), d.dtr_k.to_numpy()
    print(f"n = {len(d):,}  (band={args.band})")

    edges = np.unique(np.quantile(x, np.linspace(0, 1, args.nbins + 1)))
    rows = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        s = (x >= a) & (x < b) if i < len(edges) - 2 else (x >= a) & (x <= b)
        if s.sum() < 50:
            continue
        r = {"sm_lo": a, "sm_hi": b, "sm_mid": float(np.median(x[s])), "n": int(s.sum())}
        for q in QS:
            r[f"p{q}"] = float(np.percentile(y[s], q))
        r["iqr"] = r["p75"] - r["p25"]
        r["p90_p10"] = float(np.percentile(y[s], 90) - np.percentile(y[s], 10))
        rows.append(r)
    t = pd.DataFrame(rows)

    print(f"\n{'sm_mid':>7} {'n':>6} " + " ".join(f"{'p'+str(q):>6}" for q in QS)
          + f" {'IQR':>6}")
    for _, r in t.iterrows():
        print(f"{r.sm_mid:7.3f} {int(r.n):6d} "
              + " ".join(f"{r['p'+str(q)]:6.1f}" for q in QS) + f" {r.iqr:6.1f}")

    print("\nslope of each quantile against SM  (K per 0.1 m3/m3), OLS across bins:")
    for q in QS + ["iqr"]:
        col = f"p{q}" if q != "iqr" else "iqr"
        sl = np.polyfit(t.sm_mid, t[col], 1)[0] / 10.0
        print(f"  {col:>5}  {sl:+7.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    ax = axes[0]
    ax.fill_between(t.sm_mid, t["p5"], t["p95"], color="#2E7D5B", alpha=.14, lw=0,
                    label="p05–p95")
    ax.fill_between(t.sm_mid, t["p25"], t["p75"], color="#2E7D5B", alpha=.30, lw=0,
                    label="p25–p75 (IQR)")
    for q, ls in ((95, ":"), (90, "--"), (50, "-"), (25, "--"), (5, ":")):
        ax.plot(t.sm_mid, t[f"p{q}"], ls, color="#14503A", lw=1.9 if q == 50 else 1.3,
                label=f"p{q}" if q in (95, 50, 5) else None)
    ax.set_xlabel("0–10 cm soil moisture (m³/m³)")
    ax.set_ylabel("DTR (K)")
    ax.set_title(f"Quantiles of DTR in EQUAL-COUNT SM bins\n"
                 f"n = {int(t.n.iloc[0]):,} per bin — narrowing here cannot be a "
                 f"sampling artefact", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)

    ax = axes[1]
    ax.plot(t.sm_mid, t.iqr, "o-", color="#C1502E", lw=2, label="IQR (p75−p25)")
    ax.plot(t.sm_mid, t.p90_p10, "s--", color="#3B6EA5", lw=1.6, label="p90−p10")
    ax.set_xlabel("0–10 cm soil moisture (m³/m³)")
    ax.set_ylabel("spread of DTR (K)")
    ax.set_title("Does the spread itself shrink with moisture?", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Is the DTR–SM cloud a one-sided constraint (a wedge) or just noise?",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = ROOT / "fig" / "dtr_txson" / ("dtr_sm_envelope"
                                        + ("_band" if args.band else "") + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    t.to_csv(ROOT / "csvs" / "ecostress_dtr_sm_envelope.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
