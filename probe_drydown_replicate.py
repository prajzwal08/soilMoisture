"""Is the Δtau–ΔHAND result real, or is it the best cell of a grid I searched?

probe_drydown_dynamics.py found Δtau vs ΔHAND = -0.398, p = 0.046 at |ΔHAND| >= 5 m,
strengthening monotonically with terrain contrast (-0.173 -> -0.255 -> -0.398) where
§32.10's level test weakened (-0.102 -> -0.065 -> +0.002). That contrast is what makes
it interesting. Three things make it untrustworthy as it stands:

  1. roughly 20 tests were run; one p = 0.046 among them is what chance looks like
  2. the 5 m threshold was chosen AFTER seeing the data
  3. tau has median 3.2 d with IQR 2.9-3.6 d over ~108 segments per station — a
     suspiciously narrow spread for a hydrological time constant, consistent with the
     metric partly measuring sensor noise rather than recession

Two pre-registered tests, with criteria fixed before running:

  SPLIT-HALF REPLICATION
    Split each pair's common dates in two by date. Compute tau independently in each
    half and correlate Δtau against ΔHAND separately in each.
    PASS = both halves negative at |ΔHAND| >= 5 m, AND tau itself reproduces between
    halves (corr(tau_first, tau_second) > 0 across stations). A real recession
    property survives resampling in time; a fitting artefact does not.

  PERMUTATION TEST OVER THE WHOLE SEARCH
    Shuffle ΔHAND across pairs 10,000 times and rebuild the ENTIRE grid of tests
    (6 metrics x 3 thresholds), recording the most extreme |r| found each time. This
    is the null distribution of "best result obtainable by searching this grid".
    PASS = the observed best |r| exceeds the 95th percentile of that distribution.
    This is the honest correction for having looked in twenty places.

Sign convention throughout: NEGATIVE is the prediction. Higher above drainage means
faster drying and shorter memory.

Usage
    conda activate terramind
    python probe_drydown_replicate.py --n-perm 10000
"""
from __future__ import annotations

import argparse
import json
import warnings
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from gate_sm_vs_terrain import MIN_COMMON, station_series
from probe_drydown_dynamics import corr_p, metrics

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
METRICS = ["tau", "ac1", "recession_min", "wet_response", "sd_t", "mean"]
THRESHOLDS = [0.0, 2.0, 5.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
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
        rec = {"station_a": a, "station_b": b,
               "d_hand": st.loc[a, "hand"] - st.loc[b, "hand"]}
        # full-record metrics, for the permutation grid
        ma, mb = metrics(sa.loc[common]), metrics(sb.loc[common])
        for k in METRICS:
            rec[f"d_{k}"] = ma[k] - mb[k]
        # split-half by date
        mid = len(common) // 2
        for half, sl in (("h1", common[:mid]), ("h2", common[mid:])):
            if len(sl) < MIN_COMMON // 2:
                continue
            m1, m2 = metrics(sa.loc[sl]), metrics(sb.loc[sl])
            rec[f"d_tau_{half}"] = m1["tau"] - m2["tau"]
            rec[f"a_tau_{half}"], rec[f"b_tau_{half}"] = m1["tau"], m2["tau"]
        rows.append(rec)

    d = pd.DataFrame(rows)
    print(f"{len(d)} pairs at {args.depth} cm\n")

    res = {}

    # ── 1. split-half replication ────────────────────────────────────────────
    print("SPLIT-HALF REPLICATION  (pass = both halves negative at |dHAND| >= 5 m,")
    print("                         and tau itself reproduces between halves)")
    hi = d[d.d_hand.abs() >= 5.0]
    for half in ("h1", "h2"):
        c = f"d_tau_{half}"
        if c in d:
            r = corr_p(hi["d_hand"], hi[c])
            res[f"splithalf_{half}"] = r
            print(f"   half {half}: dtau vs dHAND (|dHAND|>=5m)   "
                  f"r = {r['r']:+.3f}  p = {r['p']:.3f}  n = {r['n']}")
    # does tau reproduce at all between halves?
    t1 = pd.concat([d.get("a_tau_h1"), d.get("b_tau_h1")], ignore_index=True)
    t2 = pd.concat([d.get("a_tau_h2"), d.get("b_tau_h2")], ignore_index=True)
    rr = corr_p(t1, t2)
    res["tau_self_reproducibility"] = rr
    print(f"   tau(first half) vs tau(second half), per station: "
          f"r = {rr['r']:+.3f}  p = {rr['p']:.3f}  n = {rr['n']}")
    both_neg = all(res.get(f"splithalf_{h}", {}).get("r", 1) < 0 for h in ("h1", "h2"))
    reproducible = rr["r"] > 0
    print(f"   -> both halves negative: {both_neg};  tau reproducible: {reproducible}")
    print(f"   -> SPLIT-HALF {'PASS' if (both_neg and reproducible) else 'FAIL'}")
    res["splithalf_pass"] = bool(both_neg and reproducible)

    # ── 2. permutation over the whole search grid ────────────────────────────
    print(f"\nPERMUTATION TEST over {len(METRICS)}x{len(THRESHOLDS)} = "
          f"{len(METRICS)*len(THRESHOLDS)} tests, {args.n_perm} shuffles")

    def best_abs_r(hand: np.ndarray) -> float:
        best = 0.0
        for m in METRICS:
            y = d[f"d_{m}"].to_numpy(float)
            for thr in THRESHOLDS:
                sel = np.abs(hand) >= thr
                x, yy = hand[sel], y[sel]
                ok = np.isfinite(x) & np.isfinite(yy)
                if ok.sum() < 8:
                    continue
                r = abs(float(np.corrcoef(x[ok], yy[ok])[0, 1]))
                best = max(best, r)
        return best

    hand = d["d_hand"].to_numpy(float)
    observed = best_abs_r(hand)
    rng = np.random.default_rng(args.seed)
    null = np.array([best_abs_r(rng.permutation(hand)) for _ in range(args.n_perm)])
    p_corrected = float((null >= observed).mean())
    res["permutation"] = {"observed_best_abs_r": observed,
                          "null_p95": float(np.percentile(null, 95)),
                          "null_median": float(np.median(null)),
                          "p_corrected": p_corrected,
                          "n_perm": args.n_perm}
    print(f"   observed best |r| over the grid : {observed:.3f}")
    print(f"   null median / 95th percentile   : {np.median(null):.3f} / "
          f"{np.percentile(null, 95):.3f}")
    print(f"   corrected p                     : {p_corrected:.4f}")
    print(f"   -> PERMUTATION {'PASS' if p_corrected < 0.05 else 'FAIL'}")
    res["permutation_pass"] = bool(p_corrected < 0.05)

    verdict = "SURVIVES" if (res["splithalf_pass"] and res["permutation_pass"]) else "DOES NOT SURVIVE"
    print(f"\nVERDICT: the dtau-dHAND result {verdict} pre-registered checking.")
    res["verdict"] = verdict

    (REPO / "csvs" / "drydown_replication.json").write_text(
        json.dumps(res, indent=2, default=float))
    d.to_csv(REPO / "csvs" / "drydown_replication_pairs.csv", index=False)
    print(f"wrote csvs/drydown_replication.json")


if __name__ == "__main__":
    main()
