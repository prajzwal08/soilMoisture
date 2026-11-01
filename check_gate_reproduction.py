"""§24.13 gate — did eval_predict_unet.py reproduce the August §24.11 run exactly?

compare_ablation.py answers a weaker question: it reports medians, and those shift whenever the
station set changes. `ablation_oos` now flags 49 stations rather than the 50 of 2026-08-10
(ISMN_USCRN_Cape-Charles-5-ENE was unflagged), so n fell 36/28/21 -> 35/27/20 and EVERY median
moved -- including `ubRMSE base`, which is computed from a parquet nobody touched.

This compares the PREDICTIONS themselves, row by row, on the rows the two runs share. That is
invariant to the station set: if the restored arm is faithful, pred_new == pred_aug wherever both
exist, whatever the medians do.

    python check_gate_reproduction.py NEW.parquet AUG.parquet
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

KEYS = ["station_key", "year", "doy", "depth"]


def main() -> int:
    new_p, aug_p = sys.argv[1], sys.argv[2]
    new = pd.read_parquet(new_p)
    aug = pd.read_parquet(aug_p)

    print(f"new : {new_p}\n      {len(new):,} rows, {new.station_key.nunique()} stations")
    print(f"aug : {aug_p}\n      {len(aug):,} rows, {aug.station_key.nunique()} stations\n")

    only_new = set(new.station_key) - set(aug.station_key)
    only_aug = set(aug.station_key) - set(new.station_key)
    if only_new:
        print(f"stations only in NEW ({len(only_new)}): {sorted(only_new)}")
    if only_aug:
        print(f"stations only in AUG ({len(only_aug)}): {sorted(only_aug)}")

    m = new.merge(aug[KEYS + ["pred"]], on=KEYS, suffixes=("", "_aug"))
    print(f"\npaired on {len(m):,} rows / {m.station_key.nunique()} stations")
    if m.empty:
        print("NO OVERLAP — the two runs share no rows.")
        return 1

    d = (m["pred"] - m["pred_aug"]).to_numpy()
    fin = np.isfinite(d)
    d = d[fin]
    print(f"  finite diffs      : {fin.sum():,} / {len(fin):,}")
    print(f"  max |pred_new - pred_aug| : {np.abs(d).max():.3e}")
    print(f"  mean|pred_new - pred_aug| : {np.abs(d).mean():.3e}")
    print(f"  rows differing > 1e-6     : {(np.abs(d) > 1e-6).sum():,}")
    print(f"  rows differing > 1e-3     : {(np.abs(d) > 1e-3).sum():,}")

    print("\n  per depth:")
    for dep, g in m.groupby("depth", observed=True):
        dd = np.abs(g["pred"] - g["pred_aug"]).to_numpy()
        dd = dd[np.isfinite(dd)]
        if dd.size:
            print(f"    {dep:7s} n={len(g):7,}  max={dd.max():.3e}  mean={dd.mean():.3e}")

    tol = 1e-5
    ok = np.abs(d).max() <= tol
    print(f"\n  VERDICT: {'REPRODUCED' if ok else 'DIFFERS'} (tolerance {tol:.0e} on max|Δpred|)")
    if not ok:
        worst = m.assign(ad=np.abs(m["pred"] - m["pred_aug"])).nlargest(5, "ad")
        print("\n  worst rows:")
        print(worst[KEYS + ["pred", "pred_aug", "ad"]].to_string(index=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
