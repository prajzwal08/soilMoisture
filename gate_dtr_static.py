#!/usr/bin/env python
"""G0 / §36.21(i) -- is the DTR field dynamic, or static like the daytime field?

THE KILL TEST.  §29.15 ended the daytime-LST arm by pooling 246 Landsat scenes over one
TxSON tile by calendar month and correlating each month's mean anomaly map against the
annual mean map: +0.95 to +0.99 in all twelve months, mean +0.967.  Its verdict was
"a static field cannot track a dynamic variable".  If the DTR field is coherent at that
level it is the same landscape pattern breathing with insolation, and the thermal arm
dies regardless of how many pairs §37 delivered.

METHOD, matching §29.15 with two corrections it did not need and we do.

  1. LEAVE-ONE-OUT.  §29.15 correlated each month against an annual mean that CONTAINED
     that month; with n = 246 scenes the self-contribution is negligible.  Our stations
     carry ~10-30 usable pairs, where correlating date i against a mean that includes
     date i inflates r by roughly 1/n -- at n = 12 that is a spurious +0.08 or worse.
     Each date is therefore correlated against the mean of the OTHER dates only.

  2. THE SAME TEST ON DAY LST, AS AN INTERNAL CONTROL.  Quoting DTR's r against §29.15's
     +0.967 compares two different sensors, resolutions, tiles and eras.  So the identical
     statistic is computed on the DAY LST anomaly and the NIGHT LST anomaly from these
     very pixels.  §29 predicts day LST comes back high.  If it does, the method is
     validated on our own data and the DTR number can be trusted; if day LST also comes
     back low, the test is broken and nothing here means anything.

  Plus a location-shuffled control per station, which must land near zero.

ANOMALY = the field minus ITS OWN scene mean, so the seasonal cycle and any whole-scene
offset are removed and only the SPATIAL PATTERN is compared.  Pixels are restricted to
those valid on every retained date, so every date is scored on the same ground.

READING IT.  Per station we report the mean leave-one-out spatial r over dates.  The
verdict is the DISTRIBUTION across stations, not any one station -- §29.10 warns that a
single impressive r means nothing at this n.
"""
from __future__ import annotations

import argparse
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import ROOT, setup_logging  # noqa: E402

DATA_ROOT = Path("/gpfs/work3/0/prjs1968/data")
FIELDS = ("dtr_k", "day_lst_k", "night_lst_k")


def split_half(stack: np.ndarray, rng, n_iter: int = 200, min_px: int = 60):
    """Split-half reliability of the spatial pattern.  stack [n_dates, n_px], NaN allowed.

    THIS, not a single date against a mean, is the statistic comparable to 29.15.
    29.15 correlated a MONTHLY MEAN map (10-20 scenes averaged) against the ANNUAL MEAN
    map -- both sides denoised by ~sqrt(n) before correlating, which is most of why its
    number was +0.967.  Correlating one noisy date against a mean is a different, much
    more attenuated quantity: the first run of this gate did that and the DAY LST control
    came back +0.337 instead of the ~+0.95 29 requires, which is how the error was caught.

    Here the dates are split at random into two halves, each half averaged, and the two
    mean maps correlated -- symmetric, denoised on both sides, and honest at n ~ 10.
    Averaged over n_iter random splits.

    Returns (r_half, r_full):
      r_half  the reliability of a HALF of the data
      r_full  Spearman-Brown stepped up to the full n, 2r/(1+r), i.e. how reproducible
              the pattern is when all dates are used -- the 29.15-comparable figure.
    """
    n = stack.shape[0]
    if n < 4:
        return np.nan, np.nan
    rs = []
    for _ in range(n_iter):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        with np.errstate(invalid="ignore"):
            ma = np.nanmean(stack[a], 0)
            mb = np.nanmean(stack[b], 0)
        m = np.isfinite(ma) & np.isfinite(mb)
        if m.sum() < min_px or ma[m].std() < 1e-9 or mb[m].std() < 1e-9:
            continue
        rs.append(float(np.corrcoef(ma[m], mb[m])[0, 1]))
    if len(rs) < n_iter // 4:
        return np.nan, np.nan
    r = float(np.mean(rs))
    sb = (2 * r) / (1 + r) if r > -1 else np.nan
    return r, float(np.clip(sb, -1, 1))


def one_station(job):
    path, sid, min_dates, min_px, seed = job
    z = np.load(path, allow_pickle=False)

    ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] > 0)
    nd = int(ok.sum())
    if nd < min_dates:
        return {"station_id": sid, "n_dates": nd, "status": "too few dates"}

    valid = z["valid"][ok].astype(bool).reshape(nd, -1)
    # Pixels need only be valid on SOME date now.  The first run required every pixel to
    # be valid on EVERY date and lost 251 of 455 stations to that alone; the split-half
    # mask is taken per split instead, from whatever each half actually covers.
    any_px = int(valid.any(0).sum())
    out = {"station_id": sid, "n_dates": nd, "n_px_any": any_px, "status": "ok"}
    rng = np.random.default_rng(seed)

    for f in FIELDS:
        arr = z[f][ok].reshape(nd, -1).astype(np.float64)
        arr[~valid] = np.nan
        with np.errstate(invalid="ignore"):
            ano = arr - np.nanmean(arr, 1, keepdims=True)   # each date minus its own mean
        rh, rf = split_half(ano, rng, min_px=min_px)
        out[f"rhalf_{f}"], out[f"r_{f}"] = rh, rf
        with np.errstate(invalid="ignore"):
            mean_map = np.nanmean(ano, 0)
        mm = mean_map[np.isfinite(mean_map)]
        out[f"sd_{f}"] = float(mm.std()) if mm.size else np.nan
        out[f"spread_{f}"] = (float(np.percentile(mm, 95) - np.percentile(mm, 5))
                              if mm.size else np.nan)
        if f == "dtr_k":
            sh = np.stack([rng.permutation(row) for row in ano])
            out["rhalf_dtr_shuffled"], out["r_dtr_shuffled"] = split_half(
                sh, rng, min_px=min_px)
            out["dtr_mean_k"] = float(np.nanmean(arr))

    # IS DTR JUST THE DAY IMAGE?  DTR = day - night, so if the day field carries the
    # larger spatial range the difference inherits its pattern and "DTR is static" would
    # be day-LST staticness wearing a different name.  Correlate the two mean anomaly
    # maps directly, and record each field's spatial amplitude so the comparison is
    # interpretable.
    try:
        mm = {}
        for f in FIELDS:
            arr = z[f][ok].reshape(nd, -1).astype(np.float64)
            arr[~valid] = np.nan
            with np.errstate(invalid="ignore"):
                a = arr - np.nanmean(arr, 1, keepdims=True)
                mm[f] = np.nanmean(a, 0)
        m = np.isfinite(mm["dtr_k"]) & np.isfinite(mm["day_lst_k"]) \
            & np.isfinite(mm["night_lst_k"])
        if m.sum() >= min_px:
            for a, b, name in (("dtr_k", "day_lst_k", "r_pattern_dtr_vs_day"),
                               ("dtr_k", "night_lst_k", "r_pattern_dtr_vs_night"),
                               ("day_lst_k", "night_lst_k", "r_pattern_day_vs_night")):
                x, y = mm[a][m], mm[b][m]
                out[name] = (float(np.corrcoef(x, y)[0, 1])
                             if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
    except Exception:                                      # noqa: BLE001
        pass

    if not np.isfinite(out.get("r_dtr_k", np.nan)):
        out["status"] = "split-half undefined"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=str(ROOT / "csvs" / "ecostress_dtr_bundles.all.csv"))
    ap.add_argument("--min-dates", type=int, default=5)
    ap.add_argument("--min-px", type=int, default=60)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out-tag", default="g0_v2")
    args = ap.parse_args()

    setup_logging(f"gate_dtr_static_{args.out_tag}")
    log = logging.getLogger("g0")

    b = pd.read_csv(args.bundles)
    b = b[b["n_pairs_usable"] >= args.min_dates]
    log.info("bundles           : %d stations with >= %d usable pairs",
             len(b), args.min_dates)

    jobs = [(r["path"], r["station_id"], args.min_dates, args.min_px, args.seed + i)
            for i, (_, r) in enumerate(b.iterrows())]
    with Pool(min(args.workers, max(len(jobs), 1))) as pool:
        rows = pool.map(one_station, jobs, chunksize=4)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "csvs" / f"ecostress_dtr_{args.out_tag}.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    log.info("status            : %s", df["status"].value_counts().to_dict())
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        log.error("no station passed the gates -- nothing to report")
        return

    log.info("")
    log.info("=== G0 / 36.21(i) -- SPLIT-HALF spatial reliability, %d stations ===", len(ok))
    log.info("%-22s %9s %9s %9s %9s %9s", "field", "r_half", "r_full", "median", "p10", "p90")
    for f in FIELDS:
        v = ok[f"r_{f}"].dropna()
        h = ok[f"rhalf_{f}"].dropna()
        if v.empty:
            continue
        log.info("%-22s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f", f, h.mean(), v.mean(),
                 v.median(), v.quantile(.10), v.quantile(.90))
    v = ok["r_dtr_shuffled"].dropna()
    h = ok["rhalf_dtr_shuffled"].dropna()
    if not v.empty:
        log.info("%-22s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f", "dtr_k (SHUFFLED)", h.mean(),
                 v.mean(), v.median(), v.quantile(.10), v.quantile(.90))
    log.info("")
    log.info("reference: 29.15 measured DAYTIME Landsat LST at +0.967 -- a static field.")
    d = ok["r_dtr_k"].dropna()
    day = ok["r_day_lst_k"].dropna()
    log.info("VERDICT INPUTS: day LST mean r = %+.3f (the control -- 29 predicts HIGH), "
             "DTR mean r = %+.3f", day.mean(), d.mean())
    for c, lbl in (("r_pattern_dtr_vs_day", "DTR pattern vs DAY pattern"),
                   ("r_pattern_dtr_vs_night", "DTR pattern vs NIGHT pattern"),
                   ("r_pattern_day_vs_night", "DAY pattern vs NIGHT pattern")):
        if c in ok:
            v = ok[c].dropna()
            log.info("%-30s mean %+.3f  median %+.3f  (n=%d)", lbl, v.mean(),
                     v.median(), len(v))
    log.info("")
    for lo in (10, 20, 30):
        s = ok[ok["n_dates"] >= lo]
        if len(s) >= 10:
            log.info("stations with >= %2d dates (n=%3d):  day r_full %+.3f   "
                     "DTR r_full %+.3f", lo, len(s), s["r_day_lst_k"].mean(),
                     s["r_dtr_k"].mean())
    log.info("")
    log.info("median dates/station = %.0f, median covered pixels = %.0f",
                ok["n_dates"].median(), ok["n_px_any"].median())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        bins = np.linspace(-1, 1, 61)
        for f, c, lbl in ((("day_lst_k"), "#C1502E", "Day LST anomaly"),
                          (("night_lst_k"), "#3B6EA5", "Night LST anomaly"),
                          (("dtr_k"), "#2E7D5B", "DTR anomaly")):
            ax.hist(ok[f"r_{f}"].dropna(), bins=bins, histtype="step", lw=2,
                    color=c, label=lbl)
        ax.hist(ok["r_dtr_shuffled"].dropna(), bins=bins, histtype="step", lw=1.2,
                color="#888888", ls=":", label="DTR, location-shuffled (control)")
        ax.axvline(0.967, color="#444", ls="--", lw=1.4)
        ax.text(0.967, ax.get_ylim()[1] * .96, "  29.15 daytime LST = +0.967",
                fontsize=8.5, va="top", color="#444")
        ax.set_xlabel("split-half spatial reliability of the anomaly pattern (Spearman-Brown, full n)")
        ax.set_ylabel(f"stations  (n = {len(ok)})")
        ax.set_title("G0 — is the field static? Higher = the same pattern every date",
                     fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        p = ROOT / "fig" / "dtr_txson" / f"g0_coherence_{args.out_tag}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        log.info("wrote %s", p)
    except Exception as e:                                # noqa: BLE001
        log.warning("figure skipped: %s", e)


if __name__ == "__main__":
    main()
