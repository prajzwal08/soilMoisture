#!/usr/bin/env python
"""Paired per-station comparison of an ablation run against the baseline (§24).

Every ablation parquet covers a subset of the baseline's stations/rows, so the
comparison is done on the inner join: same station, same day, same depth. Metrics
are per-station and then aggregated station-equal (median), because the station
pool is small (36) and observation-weighting hides the tail.

Usage:
  python compare_ablation.py eval_output/predictions_oos_sat_within_station_s0.parquet [...]
  python compare_ablation.py --base eval_output/predictions_oos.parquet ABL.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["station_key", "year", "doy", "depth"]
DEPTHS = ["0-10", "10-30", "30-100"]


def per_station(df, pred_col):
    """ubRMSE / bias / r / NSE_anom per (station, depth). Anomalies are within-station."""
    out = []
    for (st, dep), g in df.groupby(["station_key", "depth"]):
        if len(g) < 30:
            continue
        p, o = g[pred_col].to_numpy(float), g["obs"].to_numpy(float)
        pa, oa = p - p.mean(), o - o.mean()
        r = np.corrcoef(p, o)[0, 1] if p.std() > 0 and o.std() > 0 else np.nan
        nse = 1 - np.sum((pa - oa) ** 2) / np.sum(oa ** 2) if oa.std() > 0 else np.nan
        out.append(dict(station_key=st, depth=dep, n=len(g),
                        ubRMSE=np.sqrt(np.mean((pa - oa) ** 2)),
                        bias=p.mean() - o.mean(), r=r, NSE_anom=nse))
    return pd.DataFrame(out)


def compare(base_path, abl_path):
    base = pd.read_parquet(base_path)
    abl = pd.read_parquet(abl_path)
    m = base.merge(abl[KEYS + ["pred"]], on=KEYS, suffixes=("", "_abl"))
    if m.empty:
        raise SystemExit(f"no overlapping rows between {base_path} and {abl_path}")

    b = per_station(m, "pred").set_index(["station_key", "depth"])
    a = per_station(m, "pred_abl").set_index(["station_key", "depth"])
    j = b.join(a, rsuffix="_abl", how="inner").reset_index()

    print(f"\n=== {Path(abl_path).name}")
    print(f"    paired on {len(m):,} rows / {m.station_key.nunique()} stations "
          f"(ablation file had {len(abl):,} rows)")
    hdr = f"{'depth':>7} {'n':>4} | {'ubRMSE base':>11} {'abl':>7} {'delta':>8} | " \
          f"{'r base':>7} {'abl':>6} | {'NSE base':>8} {'abl':>7}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for dep in DEPTHS:
        d = j[j.depth == dep]
        if d.empty:
            continue
        med = d.median(numeric_only=True)
        # paired delta: median of per-station differences, not difference of medians
        dub = float((d.ubRMSE_abl - d.ubRMSE).median())
        print(f"{dep:>7} {len(d):>4} | {med.ubRMSE:>11.4f} {med.ubRMSE_abl:>7.4f} "
              f"{dub:>+8.4f} | {med.r:>7.2f} {med.r_abl:>6.2f} | "
              f"{med.NSE_anom:>8.3f} {med.NSE_anom_abl:>7.3f}")
        rows.append(dict(ablation=Path(abl_path).stem, depth=dep, n=len(d),
                         ubRMSE_base=med.ubRMSE, ubRMSE_abl=med.ubRMSE_abl,
                         d_ubRMSE=dub, r_base=med.r, r_abl=med.r_abl,
                         NSE_base=med.NSE_anom, NSE_abl=med.NSE_anom_abl,
                         frac_worse=float((d.ubRMSE_abl > d.ubRMSE).mean())))
    print("    fraction of stations made worse: " +
          ", ".join(f"{r['depth']} {r['frac_worse']:.0%}" for r in rows))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ablation", nargs="+")
    ap.add_argument("--base", default="eval_output/predictions_oos.parquet")
    ap.add_argument("--csv", default=None, help="write the summary table here")
    args = ap.parse_args()

    summary = pd.concat([compare(args.base, p) for p in args.ablation], ignore_index=True)
    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
