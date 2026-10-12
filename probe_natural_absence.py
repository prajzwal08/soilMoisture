"""§73.7 natural-absence probe: how often is each Shapley player's WHOLE group already missing
in the training data, before any dropout?

q feeds the dropout rule of thumb (runbook §73.7): effective absence = 1 - (1-q)(1-p), target 0.5
-> p = (0.5 - q) / (1 - q), clipped to [0.3, 0.5]. Whole-group absence only; fewer tokens is not
absence. Also split by month and |latitude| band, to see whether natural gaps are random.

The dataset is built with training=False: no dropout and no 15 % ERA5 day mask, so every absence
counted here is in the data itself. Statics (DEM, LULC, soil) are present by construction.

Usage: python probe_natural_absence.py --n 5000 --out csvs/natural_absence_<jobid>.csv
"""
import argparse
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dataset import SoilMoistureDataset
from splits_config import station_dir_name
from train import CONFIG

PLAYERS = ("era5", "sat", "s2", "s1", "fine_s2", "fine_s1", "sif", "twsa")
LAT_BANDS = [(0, 23.5, "tropics"), (23.5, 45, "subtropic-temperate"), (45, 60, "temperate-boreal"),
             (60, 91, "high-latitude")]


def absences(s) -> dict:
    s2_none = not bool(s["s2_valid"].any())
    s1_none = not bool(s["s1_valid"].any())
    return {
        "era5":    not bool((s["era5_doys"] > 0).any()),
        "sat":     s2_none and s1_none and not bool(s["anchor_found"]),
        "s2":      s2_none,
        "s1":      s1_none,
        "fine_s2": not bool(s["fine"][10].any()),     # fine s2_valid channel (model.py FINE_S2)
        "fine_s1": not bool(s["fine"][14].any()),     # fine s1_valid channel (model.py FINE_S1)
        "sif":     not bool(s["sif_valid"].any()),
        "twsa":    not bool(s["twsa_valid"].any()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = SoilMoistureDataset(splits_csv=CONFIG["splits_csv"], era5_stats_path=CONFIG["era5_stats"],
                             years=CONFIG["years"], category_filter=CONFIG["category_filter"],
                             split_filter=["train"], training=False)
    splits = pd.read_csv(CONFIG["splits_csv"])
    lat = {station_dir_name(r): float(r["latitude"]) for _, r in splits.iterrows()}

    random.seed(args.seed)
    idx = random.sample(range(len(ds)), min(args.n, len(ds)))
    rows = []
    for k, i in enumerate(idx):
        meta = ds.samples[i]
        a = absences(ds[i])
        d = date(meta["year"], 1, 1) + timedelta(days=int(meta["doy"]) - 1)
        rows.append({"station_key": meta["station_key"], "year": meta["year"], "doy": meta["doy"],
                     "month": d.month, "abs_lat": abs(lat.get(meta["station_key"], np.nan)), **a})
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(idx)} samples", flush=True)
    df = pd.DataFrame(rows)
    df["lat_band"] = pd.cut(df["abs_lat"], [b[0] for b in LAT_BANDS] + [91],
                            labels=[b[2] for b in LAT_BANDS], right=False)
    df.to_csv(args.out, index=False)

    n_st = df["station_key"].nunique()
    print(f"\n=== natural whole-group absence q over {len(df)} train samples, {n_st} stations "
          f"(of {len(ds)} samples) ===")
    for g in PLAYERS:
        q = df[g].mean()
        p_rule = min(0.5, max(0.3, (0.5 - q) / (1 - q))) if q < 1 else float("nan")
        print(f"  {g:8s} q = {q:.4f}   ({int(df[g].sum())} absent)   rule p = {p_rule:.2f}")
    print("\n--- q by month ---")
    print(df.groupby("month")[list(PLAYERS)].mean().round(3).to_string())
    print("\n--- q by |latitude| band (n samples) ---")
    t = df.groupby("lat_band", observed=True)[list(PLAYERS)].mean().round(3)
    t["n"] = df.groupby("lat_band", observed=True).size()
    print(t.to_string())
    print(f"\nper-sample CSV: {args.out}")


if __name__ == "__main__":
    main()
