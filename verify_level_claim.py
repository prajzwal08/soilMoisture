"""§22.10 -- verification of the absolute-level claim, station-equal (report grade).

The claim under test: the model reproduces the absolute soil-moisture level of a
station it saw in training and cannot infer it for a novel station, so on novel
stations the dominant error term is a constant per-station offset.

Three independent calculations, all from eval_output/predictions_{split}.parquet:

  A  error decomposition        MSE = bias^2 + ubRMSE^2 per station-depth, then
                                aggregated station-equal (each station one vote)
                                and observation-weighted.  §22.9: the two differ,
                                so both are reported rather than one silently.
  B  skill against climatology  does the predicted station mean beat a constant?
                                The constant is the mean of the VAL station means
                                -- i.e. what a model that knows nothing about the
                                station would answer.  Skill = 1 - RMS_model/RMS_null.
                                Level skill ~ 0 with high daily r is the claim.
  C  out-of-sample offset       fit the offset on the first half of each record,
                                apply it to the second half.  Non-circular: an
                                offset that is sampling noise cannot transfer.
                                The noise floor uses n_eff from the lag-1
                                autocorrelation of the residual, since daily soil
                                moisture is far from independent.

CPU only, ~1 min.  Needs pyarrow -> run under the terramind env.

    python verify_level_claim.py [--in-dir eval_output] [--out-dir eval_output]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SM_DEPTHS = ["0-10", "10-30", "30-100"]
SPLITS    = ["val", "oos", "oot", "oost"]
MIN_N     = 100      # days required for a station-depth to enter any statistic
MIN_N_AB  = 200      # days required for the split-half test (100 per half)
N_BOOT    = 2000


def station_depth_table(in_dir: Path) -> pd.DataFrame:
    """One row per (split, station, depth) with everything the three tests need."""
    rows = []
    for split in SPLITS:
        path = in_dir / f"predictions_{split}.parquet"
        if not path.exists():
            print(f"  (no {path.name} -- skipping)")
            continue
        df = pd.read_parquet(path)
        for (station, depth), g in df.groupby(["station_key", "depth"],
                                              observed=True):
            g = g.sort_values("date")
            p = g["pred"].to_numpy(np.float64)
            o = g["obs"].to_numpy(np.float64)
            if len(p) < MIN_N:
                continue
            bias = float(p.mean() - o.mean())
            mse  = float(np.mean((p - o) ** 2))
            rec = dict(split=split, station=station, depth=depth, n=len(p),
                       mean_obs=float(o.mean()), mean_pred=float(p.mean()),
                       bias=bias, mse=mse,
                       ubRMSE=float(np.sqrt(max(mse - bias ** 2, 0.0))),
                       r=float(np.corrcoef(p, o)[0, 1])
                       if o.std() > 0 and p.std() > 0 else np.nan)

            # C -- offset estimated on the first half, scored on the second
            if len(p) >= MIN_N_AB:
                h   = len(p) // 2
                off = float(p[:h].mean() - o[:h].mean())
                res = p[h:] - o[h:]
                e   = res - res.mean()
                rho = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if len(e) > 2 else 0.0
                n_eff = len(e) * (1 - rho) / (1 + rho) if rho < 1 else 1.0
                rec.update(off_h1=off,
                           mse_h2_raw=float(np.mean(res ** 2)),
                           mse_h2_corr=float(np.mean((res - off) ** 2)),
                           mse_h2_oracle=float(np.mean(e ** 2)),
                           noise=float(e.std() / max(np.sqrt(max(n_eff, 1.0)), 1.0)),
                           n_eff=float(n_eff))
            rows.append(rec)
    return pd.DataFrame(rows)


def boot_ci(x, y, stat, rng, n_boot=N_BOOT):
    """Percentile CI over stations -- the sampling unit is the station, not the day."""
    if len(x) < 5:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i], y[i]) for i in idx])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="eval_output")
    ap.add_argument("--out-dir", default="eval_output")
    args = ap.parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    s = station_depth_table(in_dir)
    s.to_csv(Path(out_dir) / "level_claim_station_depth.csv", index=False)
    rng = np.random.default_rng(0)

    # climatological constant: the mean of the val station means, per depth
    const = {d: float(s[(s.split == "val") & (s.depth == d)]["mean_obs"].mean())
             for d in SM_DEPTHS}
    print("\nClimatological constant (mean of val station means): "
          + "  ".join(f"{d} {const[d]:.3f}" for d in SM_DEPTHS))

    out = []
    print("\n=== A. error decomposition, station-equal (obs-weighted in brackets) ===")
    print(f"{'split':5s} {'depth':7s} {'stn':>4s} {'med|bias|':>9s} {'med ubRMSE':>11s} "
          f"{'bias^2/MSE':>11s} {'[weighted]':>11s}")
    for split in SPLITS:
        for d in SM_DEPTHS:
            g = s[(s.split == split) & (s.depth == d)]
            if g.empty:
                continue
            share_eq = float(np.mean(g.bias ** 2 / g.mse))
            share_w  = float((g.bias ** 2 * g.n).sum() / (g.mse * g.n).sum())
            print(f"{split:5s} {d:7s} {len(g):4d} {g.bias.abs().median():9.4f} "
                  f"{g.ubRMSE.median():11.4f} {share_eq:11.1%} {share_w:11.1%}")
            out.append(dict(split=split, depth=d, n_stations=len(g),
                            median_abs_bias=g.bias.abs().median(),
                            median_ubRMSE=g.ubRMSE.median(),
                            level_share_station_equal=share_eq,
                            level_share_weighted=share_w))

    print("\n=== B. station level vs climatology (skill = 1 - RMS_model/RMS_null) ===")
    print(f"{'split':5s} {'depth':7s} {'RMS_model':>9s} {'RMS_null':>9s} {'skill':>7s} "
          f"{'skill 95% CI':>18s} {'r(mean)':>8s} {'daily r':>8s}")
    for split in SPLITS:
        for d in SM_DEPTHS:
            g = s[(s.split == split) & (s.depth == d)]
            if len(g) < 5:
                continue
            mo, mp = g.mean_obs.to_numpy(), g.mean_pred.to_numpy()
            rms_model = float(np.sqrt(np.mean((mp - mo) ** 2)))
            rms_null  = float(np.sqrt(np.mean((const[d] - mo) ** 2)))
            skill = 1 - rms_model / rms_null
            lo, hi = boot_ci(mo, mp, lambda a, b: 1 - np.sqrt(np.mean((b - a) ** 2)) /
                             np.sqrt(np.mean((const[d] - a) ** 2)), rng)
            r_mean = float(np.corrcoef(mo, mp)[0, 1])
            print(f"{split:5s} {d:7s} {rms_model:9.4f} {rms_null:9.4f} {skill:+7.1%} "
                  f"  [{lo:+.2f}, {hi:+.2f}]  {r_mean:+8.2f} {g.r.median():8.2f}")
            for rec in out:
                if rec["split"] == split and rec["depth"] == d:
                    rec.update(rms_station_mean_model=rms_model,
                               rms_station_mean_null=rms_null, level_skill=skill,
                               level_skill_lo=lo, level_skill_hi=hi,
                               r_station_mean=r_mean, median_daily_r=g.r.median())

    print("\n=== C. offset fitted on 1st half, applied to 2nd half ===")
    print(f"{'split':5s} {'depth':7s} {'stn':>4s} {'MSE removed':>12s} "
          f"{'oracle ceiling':>15s} {'med|off|':>9s} {'noise floor':>12s} "
          f"{'|off|>2*noise':>14s}")
    c = s.dropna(subset=["off_h1"])
    for split in SPLITS:
        for d in SM_DEPTHS:
            g = c[(c.split == split) & (c.depth == d)]
            if g.empty:
                continue
            red  = 1 - g.mse_h2_corr.sum() / g.mse_h2_raw.sum()
            ceil = 1 - g.mse_h2_oracle.sum() / g.mse_h2_raw.sum()
            frac = float((g.off_h1.abs() > 2 * g.noise).mean())
            print(f"{split:5s} {d:7s} {len(g):4d} {red:12.1%} {ceil:15.1%} "
                  f"{g.off_h1.abs().median():9.4f} {g.noise.median():12.4f} "
                  f"{frac:14.0%}")
            for rec in out:
                if rec["split"] == split and rec["depth"] == d:
                    rec.update(oos_offset_mse_removed=red,
                               oracle_offset_mse_removed=ceil,
                               frac_offset_above_noise=frac)

    path = Path(out_dir) / "level_claim_summary.csv"
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"\n→ {path}")
    print(f"→ {Path(out_dir) / 'level_claim_station_depth.csv'}")


if __name__ == "__main__":
    main()
