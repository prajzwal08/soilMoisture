"""
measure_noise_floor.py — the irreducible noise floor of a 160 m prediction (§35.30)
===================================================================================

§35.29 identified station pairs closer than 160 m: both land in the SAME TerraMind
token, so a patchwise model cannot distinguish them by construction — it emits one
number for both. Whatever they disagree by is therefore a FLOOR that no model at
this resolution can beat, however good it is.

That number is the missing anchor for the §35.10 gate. Session 33 picked unmeasured
numbers three separate times (§35.25's "0.04% invisible", §35.26's residual-stream
argument, §35.27's fatal-QC severity) and measuring changed the answer every time.
Choosing gate thresholds by intuition would be the fourth. So: measure first, then
express every threshold relative to what came out.

TWO QUANTITIES, and the second is the one that matters:

  raw RMSD       sqrt(mean((sm_A - sm_B)^2)) over days both report qc == 0.
                 Includes any constant offset between the two sensors — calibration,
                 installation depth, soil contact. Real, but not what §35.10 scores.

  anomaly RMSD   each series de-meaned over the co-observed days FIRST. §35.10's
                 primary endpoint is a WITHIN-STATION criterion — de-mean prediction
                 and observation by station, then score the residuals — so the floor
                 has to be computed the same way or it is not comparable to it.

Separation is reported per pair and never averaged blind: a 6 m pair (the same
physical site ingested twice, §35.29) and a 152 m pair (genuinely different
instruments at ARM SGP) are not the same measurement.

Usage:  sbatch slurm/noise_floor.sh
Output: csvs/noise_floor.json + a printed table
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import (ZARR_ROOT, SM_DEPTHS, QC_OBSERVED,          # noqa: E402
                     _open_zarr, _load_zarr_labels)
from update_splits_tile_pairs import haversine_m                  # noqa: E402

SPLITS   = Path("csvs/station_splits.csv")
OUT_JSON = Path("csvs/noise_floor.json")
PATCH_M  = 160.0
MIN_DAYS = 30          # a pair with fewer co-observed days says nothing


def _cat(r):
    sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
    fl = str(r.get("has_flux", "False")).lower() == "true"
    return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def _load(row):
    """(dates, sm, qc, depths) for one station, or None."""
    zg = _open_zarr(Path(_dir_name(row)), _cat(row))
    if zg is None:
        return None
    try:
        out = _load_zarr_labels(zg, strict=True)
    except Exception as e:
        print(f"    [skip] {_dir_name(row)}: {e}")
        return None
    if out is None:
        return None
    sm, depths, times, qc = out
    if qc is None:
        return None
    return times, sm, qc, depths


def main() -> int:
    df = pd.read_csv(SPLITS)
    sm_rows = df[df["has_soil_moisture"].astype(str).str.lower() == "true"]
    if "same_patch_pair" not in df.columns:
        print("FATAL: station_splits.csv has no same_patch_pair column — run "
              "update_splits_tile_pairs.py --apply first (§35.29).", file=sys.stderr)
        return 2
    cand = sm_rows[sm_rows["same_patch_pair"].astype(str).str.lower() == "true"]
    print(f"same-patch stations flagged: {len(cand)}\n")

    recs = cand.to_dict("records")
    cache, pairs = {}, []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            d = haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            if d >= PATCH_M:
                continue
            for st in (a, b):
                k = st["station_id"]
                if k not in cache:
                    cache[k] = _load(st)
            A, B = cache[a["station_id"]], cache[b["station_id"]]
            if A is None or B is None:
                print(f"  [skip] {a['station_id']} <-> {b['station_id']}: labels unavailable")
                continue

            tA, smA, qcA, dpA = A
            tB, smB, qcB, dpB = B
            common = tA.intersection(tB)
            if len(common) < MIN_DAYS:
                continue
            iA = tA.get_indexer(common)
            iB = tB.get_indexer(common)

            for depth in SM_DEPTHS:
                if depth not in dpA or depth not in dpB:
                    continue
                da, db = dpA.index(depth), dpB.index(depth)
                va, vb = smA[da, iA], smB[db, iB]
                ok = ((qcA[da, iA] == QC_OBSERVED) & (qcB[db, iB] == QC_OBSERVED)
                      & np.isfinite(va) & np.isfinite(vb))
                n = int(ok.sum())
                if n < MIN_DAYS:
                    continue
                x, y = va[ok].astype(np.float64), vb[ok].astype(np.float64)
                raw  = float(np.sqrt(np.mean((x - y) ** 2)))
                anom = float(np.sqrt(np.mean(((x - x.mean()) - (y - y.mean())) ** 2)))
                pairs.append({
                    "a": a["station_id"], "b": b["station_id"], "depth": depth,
                    "sep_m": round(d, 1), "n_days": n,
                    "rmsd": raw, "anom_rmsd": anom,
                    "bias": float(x.mean() - y.mean()),
                    "networks": f"{a['network']}|{b['network']}",
                })

    if not pairs:
        print("FATAL: no usable pair survived — nothing to anchor the gate to.",
              file=sys.stderr)
        return 2

    print("=" * 104)
    print(f"{'depth':<9}{'A':<20}{'B':<20}{'sep_m':>7}{'n':>7}{'RMSD':>9}"
          f"{'anomRMSD':>10}{'bias':>9}  networks")
    print("-" * 104)
    for p in sorted(pairs, key=lambda q: (q["depth"], q["sep_m"])):
        print(f"{p['depth']:<9}{p['a'][:19]:<20}{p['b'][:19]:<20}{p['sep_m']:>7.1f}"
              f"{p['n_days']:>7}{p['rmsd']:>9.4f}{p['anom_rmsd']:>10.4f}"
              f"{p['bias']:>9.4f}  {p['networks']}")

    summary = {}
    print("\n" + "=" * 70)
    print("NOISE FLOOR  (median over pairs; anomRMSD is the §35.10-comparable one)")
    print("=" * 70)
    print(f"{'depth':<10}{'pairs':>7}{'days':>9}{'RMSD':>10}{'anomRMSD':>11}")
    for depth in SM_DEPTHS:
        sel = [p for p in pairs if p["depth"] == depth]
        if not sel:
            print(f"{depth:<10}{'--':>7}   no pair reports this depth")
            continue
        f_raw  = float(np.median([p["rmsd"] for p in sel]))
        f_anom = float(np.median([p["anom_rmsd"] for p in sel]))
        summary[depth] = {"n_pairs": len(sel),
                          "n_days_total": int(sum(p["n_days"] for p in sel)),
                          "rmsd_median": f_raw, "anom_rmsd_median": f_anom}
        print(f"{depth:<10}{len(sel):>7}{sum(p['n_days'] for p in sel):>9}"
              f"{f_raw:>10.4f}{f_anom:>11.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"patch_m": PATCH_M, "min_days": MIN_DAYS,
                   "per_depth": summary, "pairs": pairs}, f, indent=2)
    print(f"\nwrote {OUT_JSON}")
    print("\nHOW TO USE IT: F = anom_rmsd_median per depth is the best any 160 m model")
    print("can do. §35.30 expresses the patch_map_sd gate as a fraction of F, and any")
    print("reported ubRMSE at or below F is at the resolution limit, not skill.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
