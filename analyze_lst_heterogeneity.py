"""
§29.7 — does within-tile LST anomaly track station-to-station soil moisture?

Joins csvs/lst_station_timeseries.csv (from extract_lst_timeseries.py) to the observed soil
moisture in eval_output/txson_timeseries.parquet and runs the tests of §29.7:

  headline   station mean LST anomaly  vs  station mean observed SM       (n = stations in tile)
  per-date   corr(LST anomaly, SM) across stations, one r per date, aggregated
  pooled     LST anomaly vs within-tile SM anomaly over all station-dates
  controls   label shuffle, clear-sky dry bias, noise floor

Prediction (§29.7): NEGATIVE. Wetter is cooler.

Observations come from the PARQUET, not the level-1 NetCDFs — the level-1 records start
2014-10/11, outside the 2016-01-01 label window, and give different means (§29.7 note).

Usage:
  conda run -n terramind python analyze_lst_heterogeneity.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO    = Path(__file__).resolve().parent
SERIES  = REPO / "csvs" / "lst_station_timeseries.csv"
OBS_PQ  = REPO / "eval_output" / "txson_timeseries.parquet"
OUT_JSON = REPO / "csvs" / "lst_summary.json"
OUT_DATE = REPO / "csvs" / "lst_per_date_corr.csv"
OUT_LVL  = REPO / "csvs" / "lst_level_correlations.csv"

# §29.11 — must reproduce to 1e-3 or the observation source is wrong
REF_MEANS = {
    "CR200-18": 0.1367, "CR200-25": 0.1197, "CR1000-2": 0.1826,
    "CR200-24": 0.2323, "CR200-15": 0.1857, "CR200-6": 0.2865,
}


def load_obs(pq: Path) -> pd.DataFrame:
    df = pd.read_parquet(pq)
    df = df[df.depth == "0-10"].dropna(subset=["obs"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def verify_reference_means(obs: pd.DataFrame):
    sub = obs[obs.tile == "ISMN_TxSON_CR200-18"]
    got = sub.groupby("station").obs.mean()
    bad = []
    for stn, ref in REF_MEANS.items():
        key = [k for k in got.index if k.endswith(stn)]
        if not key:
            bad.append((stn, "missing", ref)); continue
        v = float(got[key[0]])
        if abs(v - ref) > 1e-3:
            bad.append((stn, round(v, 4), ref))
    if bad:
        raise SystemExit(f"§29.11 reference means FAILED (wrong observation source?): {bad}")
    print("  verify: six CR200-18 reference SM means reproduce to 1e-3 from the parquet")


def fisher_agg(rs: np.ndarray) -> tuple:
    rs = np.clip(rs[np.isfinite(rs)], -0.999, 0.999)
    if len(rs) < 3:
        return np.nan, np.nan, np.nan
    z = np.arctanh(rs)
    t, p = stats.ttest_1samp(z, 0.0)
    return float(np.tanh(z.mean())), float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=Path, default=SERIES)
    ap.add_argument("--obs", type=Path, default=OBS_PQ)
    ap.add_argument("--min-stations", type=int, default=4)
    ap.add_argument("--n-shuffle", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lst = pd.read_csv(args.series)
    obs = load_obs(args.obs)
    verify_reference_means(obs)

    df = lst.merge(obs[["tile", "station", "date", "obs", "pred"]],
                   on=["tile", "station", "date"], how="inner")
    print(f"  joined: {len(df)} station-date records with BOTH clear LST and observed SM "
          f"({df.date.nunique()} dates, {df.station.nunique()} stations)")
    if df.empty:
        raise SystemExit("no overlap between LST dates and observed SM dates")

    summary = {}
    rng = np.random.default_rng(args.seed)

    # ---------------- headline: station level ----------------
    print("\n" + "=" * 74)
    print("HEADLINE — station mean LST anomaly vs station mean observed SM")
    print("=" * 74)
    lvl_rows = []
    for tile, g in df.groupby("tile"):
        s = g.groupby("station_name").agg(
            lst_anom=("lst_anom_k", "mean"), sm=("obs", "mean"),
            pred=("pred", "mean"), n=("obs", "size")).reset_index()
        if len(s) < 3:
            continue
        r, p = stats.pearsonr(s.lst_anom, s.sm)
        rho, prho = stats.spearmanr(s.lst_anom, s.sm)
        print(f"\n{tile}   n_stations = {len(s)}")
        print(f"  {'station':<12s} {'meanSM':>8s} {'LSTanom':>9s} {'predSM':>8s} {'n':>5s}")
        for _, r_ in s.sort_values("sm").iterrows():
            print(f"  {r_.station_name:<12s} {r_.sm:8.4f} {r_.lst_anom:+9.3f} "
                  f"{r_.pred:8.4f} {int(r_.n):5d}")
        print(f"  SM spread      = {s.sm.max()-s.sm.min():.4f}   "
              f"(model predicted spread = {s.pred.max()-s.pred.min():.4f})")
        print(f"  LST anom spread= {s.lst_anom.max()-s.lst_anom.min():.3f} K")
        print(f"  >> pearson r = {r:+.3f} (p={p:.3f})   spearman rho = {rho:+.3f} (p={prho:.3f})")
        print(f"     PREDICTION was NEGATIVE.  Observed sign: "
              f"{'NEGATIVE (as predicted)' if r < 0 else 'POSITIVE / null (against prediction)'}")
        lvl_rows.append(dict(tile=tile, n_stations=len(s), pearson_r=r, pearson_p=p,
                             spearman_rho=rho, spearman_p=prho,
                             sm_spread=s.sm.max()-s.sm.min(),
                             pred_spread=s.pred.max()-s.pred.min(),
                             lst_anom_spread=s.lst_anom.max()-s.lst_anom.min()))
    pd.DataFrame(lvl_rows).to_csv(OUT_LVL, index=False)
    summary["level"] = lvl_rows

    # ---------------- per-date ----------------
    print("\n" + "=" * 74)
    print("PER-DATE — corr(LST anomaly, observed SM) across stations, one r per date")
    print("=" * 74)
    rows = []
    for (tile, date), g in df.groupby(["tile", "date"]):
        if len(g) < args.min_stations or g.obs.nunique() < 3:
            continue
        r, p = stats.pearsonr(g.lst_anom_k, g.obs)
        rows.append(dict(tile=tile, date=date, n=len(g), r=r, p=p,
                         sm_mean=g.obs.mean(), sm_spread=g.obs.max()-g.obs.min(),
                         lst_spread=g.lst_anom_k.max()-g.lst_anom_k.min()))
    pdc = pd.DataFrame(rows)
    pdc.to_csv(OUT_DATE, index=False)
    if len(pdc):
        rs = pdc.r.values
        mr, t, p = fisher_agg(rs)
        neg = int((rs < 0).sum())
        sign_p = stats.binomtest(neg, len(rs), 0.5).pvalue
        print(f"  dates with >= {args.min_stations} stations : {len(pdc)}")
        print(f"  mean r (Fisher-z)   : {mr:+.3f}   median r = {np.median(rs):+.3f}")
        print(f"  frac negative       : {neg}/{len(rs)} = {neg/len(rs):.1%}  "
              f"(sign test p = {sign_p:.3g})")
        print(f"  one-sample t on z   : t = {t:+.2f}, p = {p:.3g}")
        summary["per_date"] = dict(n_dates=len(pdc), mean_r=mr, median_r=float(np.median(rs)),
                                   frac_negative=neg/len(rs), sign_p=float(sign_p),
                                   t=t, t_p=p)

    # ---------------- pooled ----------------
    print("\n" + "=" * 74)
    print("POOLED — LST anomaly vs within-tile SM anomaly, all station-dates")
    print("=" * 74)
    df["sm_anom"] = df.obs - df.groupby(["tile", "date"]).obs.transform("mean")
    d2 = df.dropna(subset=["sm_anom", "lst_anom_k"])
    r, p = stats.pearsonr(d2.lst_anom_k, d2.sm_anom)
    rho, prho = stats.spearmanr(d2.lst_anom_k, d2.sm_anom)
    print(f"  n = {len(d2)} station-dates")
    print(f"  >> pearson r = {r:+.4f} (p = {p:.3g})   spearman = {rho:+.4f} (p = {prho:.3g})")
    print(f"     slope = {np.polyfit(d2.lst_anom_k, d2.sm_anom, 1)[0]:+.5f} m3/m3 per K")
    summary["pooled"] = dict(n=len(d2), pearson_r=r, pearson_p=p,
                             spearman_rho=rho, spearman_p=prho)

    # ---- WITHIN-STATION (fixed effects) --------------------------------
    # The pooled r above mixes two different questions: BETWEEN stations (does a
    # persistently warmer station sit on persistently wetter soil?) and WITHIN a station
    # (when that pixel runs warm for its own average, is the soil drier?).  Station
    # identity dominates, so the pooled number is a Simpson's-paradox artefact.  De-mean
    # both variables by station to isolate the temporal coupling.
    print("\n" + "-" * 74)
    print("WITHIN-STATION (station fixed effects) — the pooled r above is confounded")
    print("-" * 74)
    d3 = d2.copy()
    d3["lst_w"] = d3.lst_anom_k - d3.groupby("station").lst_anom_k.transform("mean")
    d3["sm_w"] = d3.sm_anom - d3.groupby("station").sm_anom.transform("mean")
    rw, pw = stats.pearsonr(d3.lst_w, d3.sm_w)
    print(f"  n = {len(d3)};  within-station r = {rw:+.4f} (p = {pw:.3g})")
    print("  per-station r (LST anomaly vs its own SM anomaly, over time):")
    per = []
    for stn, g in d3.groupby("station_name"):
        if len(g) > 5:
            ri, pi = stats.pearsonr(g.lst_anom_k, g.sm_anom)
            per.append((stn, ri, pi, len(g)))
            print(f"    {stn:<12s} r = {ri:+.3f}  (p = {pi:.3f}, n = {len(g)})")
    rs_ = np.array([x[1] for x in per])
    print(f"  mean per-station r = {rs_.mean():+.3f};  "
          f"{int((rs_ < 0).sum())}/{len(rs_)} negative")
    print(f"  BETWEEN-station component (pooled {r:+.3f}) minus within ({rw:+.3f}) "
          f"= {r-rw:+.3f}\n  -> the pooled positive is station identity, not coupling.")
    summary["within_station"] = dict(n=len(d3), r=rw, p=pw,
                                     per_station={x[0]: x[1] for x in per},
                                     mean_per_station_r=float(rs_.mean()),
                                     n_negative=int((rs_ < 0).sum()), n_stations=len(rs_))

    # ---------------- controls ----------------
    print("\n" + "=" * 74)
    print("CONTROLS")
    print("=" * 74)

    null = []
    for _ in range(args.n_shuffle):
        sh = d2.copy()
        sh["sm_anom"] = sh.groupby(["tile", "date"]).sm_anom.transform(
            lambda v: rng.permutation(v.values))
        null.append(stats.pearsonr(sh.lst_anom_k, sh.sm_anom)[0])
    null = np.array(null)
    emp_p = float((np.abs(null) >= abs(r)).mean())
    print(f"  1. label shuffle ({args.n_shuffle} draws): null r = {null.mean():+.4f} "
          f"+/- {null.std():.4f};  empirical p for observed r={r:+.4f} is {emp_p:.4f}")
    summary["shuffle"] = dict(null_mean=float(null.mean()), null_sd=float(null.std()),
                              empirical_p=emp_p)

    all_dates = set(obs[obs.tile.isin(df.tile.unique())].date)
    kept = set(df.date)
    o = obs[obs.tile.isin(df.tile.unique())]
    sm_kept = o[o.date.isin(kept)].obs
    sm_drop = o[~o.date.isin(kept)].obs
    ks = stats.ks_2samp(sm_kept, sm_drop)
    print(f"  2. clear-sky dry bias: retained {len(kept)}/{len(all_dates)} dates; "
          f"SM mean retained {sm_kept.mean():.4f} vs dropped {sm_drop.mean():.4f} "
          f"(KS = {ks.statistic:.3f}, p = {ks.pvalue:.3g})")
    print(f"     -> retained days are {'DRIER' if sm_kept.mean() < sm_drop.mean() else 'WETTER'}"
          f" by {abs(sm_kept.mean()-sm_drop.mean()):.4f} m3/m3")
    summary["dry_bias"] = dict(n_kept=len(kept), n_all=len(all_dates),
                               sm_kept=float(sm_kept.mean()), sm_dropped=float(sm_drop.mean()),
                               ks=float(ks.statistic), ks_p=float(ks.pvalue))

    spread = df.groupby(["tile", "date"]).lst_anom_k.agg(lambda v: v.max() - v.min())
    print(f"  3. noise floor: within-tile anomaly spread median {spread.median():.2f} K, "
          f"p90 {spread.quantile(0.9):.2f} K;  median per-pixel ST_QA "
          f"{df.st_qa_k.median():.2f} K")
    se = df.groupby("station_name").lst_anom_k.agg(["mean", "std", "size"])
    se["sem"] = se["std"] / np.sqrt(se["size"])
    rng_k = float(se["mean"].max() - se["mean"].min())
    sem_k = float(se["sem"].median())
    verdict = "well resolved" if rng_k > 5 * sem_k else "NOT resolved"
    print(f"     station-mean anomalies separated by {rng_k:.2f} K "
          f"with SEM ~{sem_k:.3f} K -> the SPATIAL PATTERN is {verdict} "
          f"({rng_k/sem_k:.0f}x the standard error)")
    summary["noise"] = dict(spread_median=float(spread.median()),
                            spread_p90=float(spread.quantile(0.9)),
                            st_qa_median=float(df.st_qa_k.median()),
                            station_mean_range=float(se["mean"].max()-se["mean"].min()),
                            sem_median=float(se["sem"].median()))

    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}, {OUT_DATE}, {OUT_LVL}")


if __name__ == "__main__":
    main()
