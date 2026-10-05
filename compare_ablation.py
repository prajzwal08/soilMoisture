#!/usr/bin/env python
"""Paired per-station comparison of an ablation run against the baseline (§24, §67).

Every ablation parquet covers a subset of the baseline's stations/rows, so the
comparison is done on the inner join: same station, same day, same depth. Metrics
are per-station and then aggregated station-equal (median of per-station deltas),
because observation-weighting hides the tail.

§67 additions: RMSE and |bias| next to ubRMSE (ubRMSE removes each station's mean, so
the statics' effect on the station LEVEL only shows in bias / RMSE); 95 % bootstrap CI
over stations for every delta; relative change in %; predictions clipped to the
physical range when EVAL_CLIP is set (same rule as eval_metrics.clip_pred).

Usage:
  python compare_ablation.py eval_output/predictions_oos_sat_within_station_s0.parquet [...]
  python compare_ablation.py --base eval_output/predictions_oos.parquet ABL.parquet --csv out.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval_metrics import clip_pred

KEYS = ["station_key", "year", "doy", "depth"]
DEPTHS = ["0-10", "10-30", "30-100"]
N_BOOT = 2000


def per_station(df, pred_col):
    """ubRMSE / RMSE / bias / r / NSE_anom per (station, depth). Anomalies are within-station."""
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
                        RMSE=np.sqrt(np.mean((p - o) ** 2)),
                        bias=p.mean() - o.mean(), r=r, NSE_anom=nse))
    return pd.DataFrame(out)


def boot_median_ci(x, seed=0):
    """Median of per-station paired deltas and its 95 % bootstrap CI over stations."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, (N_BOOT, x.size))], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def compare(base_path, abl_path):
    base = clip_pred(pd.read_parquet(base_path))
    abl = clip_pred(pd.read_parquet(abl_path))
    m = base.merge(abl[KEYS + ["pred"]], on=KEYS, suffixes=("", "_abl"))
    if m.empty:
        raise SystemExit(f"no overlapping rows between {base_path} and {abl_path}")

    b = per_station(m, "pred").set_index(["station_key", "depth"])
    a = per_station(m, "pred_abl").set_index(["station_key", "depth"])
    j = b.join(a, rsuffix="_abl", how="inner").reset_index()

    print(f"\n=== {Path(abl_path).name}")
    print(f"    paired on {len(m):,} rows / {m.station_key.nunique()} stations "
          f"(ablation file had {len(abl):,} rows)")
    hdr = (f"{'depth':>7} {'n':>4} | {'ubRMSE base':>11} {'abl':>7} {'delta [95% CI]':>26} {'%':>6} | "
           f"{'dRMSE':>8} {'d|bias|':>8} {'dr':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for dep in DEPTHS:
        d = j[j.depth == dep]
        if d.empty:
            continue
        med = d.median(numeric_only=True)
        # paired deltas: median of per-station differences, not difference of medians
        dub, dub_lo, dub_hi = boot_median_ci(d.ubRMSE_abl - d.ubRMSE)
        drm, drm_lo, drm_hi = boot_median_ci(d.RMSE_abl - d.RMSE)
        dab, dab_lo, dab_hi = boot_median_ci(d.bias_abl.abs() - d.bias.abs())
        dr, dr_lo, dr_hi = boot_median_ci(d.r_abl - d.r)
        pct = 100 * dub / med.ubRMSE if med.ubRMSE else np.nan
        print(f"{dep:>7} {len(d):>4} | {med.ubRMSE:>11.4f} {med.ubRMSE_abl:>7.4f} "
              f"{dub:>+8.4f} [{dub_lo:+.4f},{dub_hi:+.4f}] {pct:>+6.1f} | "
              f"{drm:>+8.4f} {dab:>+8.4f} {dr:>+7.3f}")
        rows.append(dict(ablation=Path(abl_path).stem, depth=dep, n=len(d),
                         ubRMSE_base=med.ubRMSE, ubRMSE_abl=med.ubRMSE_abl,
                         d_ubRMSE=dub, d_ubRMSE_lo=dub_lo, d_ubRMSE_hi=dub_hi, d_ubRMSE_pct=pct,
                         RMSE_base=med.RMSE, d_RMSE=drm, d_RMSE_lo=drm_lo, d_RMSE_hi=drm_hi,
                         d_RMSE_pct=100 * drm / med.RMSE if med.RMSE else np.nan,
                         absbias_base=float(d.bias.abs().median()),
                         d_absbias=dab, d_absbias_lo=dab_lo, d_absbias_hi=dab_hi,
                         r_base=med.r, r_abl=med.r_abl, d_r=dr, d_r_lo=dr_lo, d_r_hi=dr_hi,
                         NSE_base=med.NSE_anom, NSE_abl=med.NSE_anom_abl,
                         frac_worse=float((d.ubRMSE_abl > d.ubRMSE).mean())))
    print("    fraction of stations made worse (ubRMSE): " +
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
