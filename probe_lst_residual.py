#!/usr/bin/env python
"""Does observed LST − T2m carry SM information the trained model is MISSING? (2026-10-05)

On Landsat scene days, correlate the model's SM error (obs − pred) with the deseasonalised
LST tile − T2m anomaly. Reference on the SAME rows: r(SM anomaly, LST anomaly) — the raw signal.

  r(error, LST anom) ~ 0       the model already has what LST knows (ERA5 rain/radiation gave it)
  r(error, LST anom) < 0       LST holds information the model lacks -> an input is worth building
  ratio r_error / r_SM         share of the raw signal the model has NOT captured

Inputs: eval_predict.py val parquet (one row per station, date, depth) and the full-probe
scenes.csv. Anomalies use the same rules as plot_lst_level_deseason_all.py (imported):
SM vs a 31-day DOY climatology of the full daily record, LST − T2m vs a per-station harmonic fit.
Join check: obs in the parquet must equal the scene-day SM in scenes.csv (printed).
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_lst_level_deseason_all import (INK, MIN_SCENES, OTHER, doy_of, harmonic_anomaly,
                                         r_of, sm_climatology, style)

DEPTHS = ["0-10", "10-30", "30-100"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions_val.parquet from eval_predict.py")
    ap.add_argument("--scenes", default="csvs/probe_lst_level_pattern_full/scenes.csv")
    ap.add_argument("--out-dir", default="figures/probe_lst_residual")
    ap.add_argument("--csv-dir", default="csvs/probe_lst_residual")
    ap.add_argument("--label", default="", help="model name for titles")
    a = ap.parse_args()
    out, csvd = Path(a.out_dir), Path(a.csv_dir)
    out.mkdir(parents=True, exist_ok=True)
    csvd.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(a.pred)
    p = p[p["depth"].isin(DEPTHS)]
    p = p.pivot_table(index=["station_key", "year", "doy"], columns="depth",
                      values=["pred", "obs"]).reset_index()
    p.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in p.columns]

    s = pd.read_csv(a.scenes)
    s = s[s["station"].isin(p["station_key"].unique())].copy()
    t = pd.to_datetime(s["date"].astype(str), format="%Y%m%d")
    s["year"], s["doy_raw"] = t.dt.year, t.dt.dayofyear
    m = s.merge(p, left_on=["station", "year", "doy_raw"], right_on=["station_key", "year", "doy"],
                how="inner")
    print(f"val stations in parquet: {p['station_key'].nunique()} | with Landsat scenes: "
          f"{s['station'].nunique()} | joined scene-days: {len(m)} ({m['station'].nunique()} stations)")
    chk = m[["sm_0-10", "obs_0-10"]].dropna()
    print(f"JOIN CHECK (scene SM vs parquet obs, 0-10): n={len(chk)}  max|diff|="
          f"{(chk['sm_0-10'] - chk['obs_0-10']).abs().max():.2e}  (must be ~0)")

    # SM climatology from the full daily record, per station (same rule as the C1 figures)
    from splits_config import category_of, station_dir_name
    sp = pd.read_csv("csvs/station_splits.csv")
    sp["dir"] = sp.apply(station_dir_name, axis=1)
    cats = sp.set_index("dir").apply(category_of, axis=1)
    sts = sorted(m["station"].unique())
    with Pool(16) as pool:
        res = pool.map(sm_climatology, [(st, cats.get(st)) for st in sts], chunksize=1)
    clim = {st: c for st, msg, c in res if msg == "ok"}
    print("SM climatology:", pd.Series([msg.split(":")[0] for _, msg, _ in res]).value_counts().to_dict())

    rows, parts = [], []
    for st, g in m.groupby("station"):
        if st not in clim:
            continue
        g = g.copy()
        dfold = doy_of(g["date"].to_numpy())
        g["lst_anom"] = harmonic_anomaly(dfold.astype(float), g["dT_mean"].to_numpy(float))
        g["sm_anom"] = g["obs_0-10"] - clim[st][dfold - 1]
        for d in DEPTHS:
            g[f"err_{d}"] = g[f"obs_{d}"] - g[f"pred_{d}"]
        parts.append(g)
        row = {"station": st}
        r, n = r_of(g["lst_anom"], g["sm_anom"])
        row["r_sm_anom"], row["n"] = (r if n >= MIN_SCENES else np.nan), n
        for d in DEPTHS:
            r, n = r_of(g["lst_anom"], g[f"err_{d}"])
            row[f"r_err_{d}"], row[f"n_err_{d}"] = (r if n >= MIN_SCENES else np.nan), n
        rows.append(row)
    d = pd.concat(parts, ignore_index=True)
    per = pd.DataFrame(rows)
    per.to_csv(csvd / "per_station_r.csv", index=False)
    d.to_csv(csvd / "scene_rows.csv", index=False)

    summ = []
    for col, name in [("sm_anom", "SM anomaly 0-10 (reference)")] + \
                     [(f"err_{x}", f"model error {x} (obs − pred)") for x in DEPTHS]:
        pr, pn = r_of(d["lst_anom"], d[col])
        key = "r_sm_anom" if col == "sm_anom" else f"r_{col}"
        ps = per[key].dropna()
        summ.append(dict(target=name, pooled_r=round(pr, 3), n=pn, station_median_r=round(ps.median(), 3),
                         q25=round(ps.quantile(.25), 3), q75=round(ps.quantile(.75), 3),
                         n_stations=len(ps), frac_neg=round((ps < 0).mean(), 2)))
    summ = pd.DataFrame(summ)
    summ.to_csv(csvd / "summary.csv", index=False)
    print("\n" + summ.to_string(index=False))
    ref = summ.iloc[0]["station_median_r"]
    e10 = summ.iloc[1]["station_median_r"]
    print(f"\nRATIO error/SM (0-10, station medians): {e10:+.3f} / {ref:+.3f} = "
          f"{(e10 / ref if ref else np.nan):.2f}  -> share of the LST signal the model has NOT captured")

    # ---- figure ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, col, lab, tag in ((axes[0], "sm_anom", "Observed SM 0-10 anomaly (m³/m³)", "a"),
                              (axes[1], "err_0-10", "Model error 0-10, obs − pred (m³/m³)", "b")):
        style(ax)
        ax.scatter(d[col], d["lst_anom"], s=7, color=OTHER if col == "err_0-10" else "#2a78d6",
                   alpha=0.35, linewidths=0, rasterized=True)
        mm = d[col].notna() & d["lst_anom"].notna()
        k, b = np.polyfit(d.loc[mm, col], d.loc[mm, "lst_anom"], 1)
        xx = np.linspace(d.loc[mm, col].quantile(.01), d.loc[mm, col].quantile(.99), 2)
        ax.plot(xx, k * xx + b, color=INK, lw=1.5)
        r, n = r_of(d["lst_anom"], d[col])
        key = "r_sm_anom" if col == "sm_anom" else "r_err_0-10"
        ax.set_title(f"({tag}) pooled r = {r:+.2f} (n = {n})\nper-station median r = "
                     f"{per[key].median():+.2f} ({per[key].notna().sum()} stations)",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel(lab, fontsize=9, color=INK)
        ax.set_ylabel("LST tile − T2m mean, anomaly (K)", fontsize=9, color=INK)
        lo, hi = d["lst_anom"].quantile([.005, .995])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax = axes[2]
    style(ax, zero_x=False)
    rng = np.random.default_rng(0)
    for i, (key, col) in enumerate((("r_sm_anom", "#2a78d6"), ("r_err_0-10", OTHER))):
        v = per[key].dropna()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=col, alpha=0.8,
                   edgecolors="white", linewidths=0.4)
        ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=INK, lw=2)
        ax.annotate(f"{v.median():+.2f}", (i + 0.3, v.median()), va="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["SM anomaly", "Model error"])
    ax.set_xlim(-0.6, 1.8)
    ax.set_ylabel("Per-station r with LST − T2m anomaly", fontsize=9, color=INK)
    ax.set_title(f"(c) One dot per val station (≥ {MIN_SCENES} scenes)", fontsize=9, color=INK, loc="left")
    fig.suptitle(f"Val stations, Landsat scene days 2016-2022: does LST − T2m explain what the model "
                 f"gets wrong?{'  [' + a.label + ']' if a.label else ''}", fontsize=10, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"R1_lst_vs_model_error_0-10.{ext}", dpi=300)
    print(f"wrote {out}/R1_lst_vs_model_error_0-10.{{png,pdf}} and {csvd}/")


if __name__ == "__main__":
    main()
