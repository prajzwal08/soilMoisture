"""§72 rain-event lag ladder: how SM, S1 backscatter and S2 bands / indices respond to a rain event.

Superposed-epoch composite at the station pixel (centre 3 x 3 px = 30 m of the raw 10 m store):
    1. per station: daily SM 0-10 (qc == 0), ERA5-Land rain (mm/day, UTC days), S2 B04/B08/B11/B12
       reflectance + NDVI/NDMI on clear scenes (SEnSeIv2 + snow test), S1 VV/VH in dB per orbit
    2. remove the season: subtract each variable's day-of-year climatology (+-15 d window; S1 per orbit)
    3. events: day with rain >= 10 mm after 7 days with < 2 mm in total; window -15..+40 d
    4. event-relative anomaly: value at lag minus the mean of that event's pre-event values (lags -15..-1)
    5. placebo: for each event a random dry-spell day (rain < 1 mm, previous 7 d < 2 mm) in the same
       calendar month of the same station -> what happens WITHOUT rain (drydown, chance)
    6. composite: per station per lag bin mean, then median over stations per land-cover macro
       (igbp_macro), 95 % CI by bootstrap over stations

Writes <out>/rows.parquet (station-level bin means), composite.csv, lags.csv, stations.csv.

    python analyse_rain_lag.py --smoke 20 --out eval_output/rain_lag_smoke
    python analyse_rain_lag.py --out eval_output/rain_lag
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

import dataset as _ds
_ds.ZARR_ROOT = Path("/gpfs/work3/0/prjs1968/zarr_tokens")   # permanent store (as analyse_s1_spatial_cv.py)
from dataset import _load_zarr_era5, _load_zarr_labels, _open_zarr, CM_BAD_CLASSES  # noqa: E402
ZARR_ROOT = _ds.ZARR_ROOT
from splits_config import category_of, station_dir_name

REPO   = Path(__file__).resolve().parent
SPLITS = REPO / "csvs" / "station_splits.csv"
RAW    = Path("/gpfs/work3/0/prjs1968/satellite_zarr")
SM_CATEGORIES = ["sm_only", "sm_and_flux"]
MACROS = ["Grass-Crop", "Shrub-Savanna", "Forest"]

C, H = 112, 1                                   # station pixel, 3 x 3 box
BOX = (slice(C - H, C + H + 1), slice(C - H, C + H + 1))
S2B = {"B03": 2, "B04": 3, "B08": 7, "B11": 10, "B12": 11}
S2_OFF = 1000.0
RAIN_MM, DRY_DAYS, DRY_MM, PLACEBO_RAIN = 10.0, 7, 2.0, 1.0
LAG_MIN, LAG_MAX = -15, 40
CLIM_HALF = 15
NBOOT = 500
VARS = ["rain", "SM", "VV", "VH", "B04", "B08", "B11", "B12", "NDVI", "NDMI"]


def _dates(arr) -> pd.DatetimeIndex:
    return pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d) for d in arr[:]],
                          format="%Y%m%d")


# ── per-station series ─────────────────────────────────────────────────────────────────────
def load_rain(zg) -> pd.Series | None:
    out = _load_zarr_era5(zg)
    if out is None:
        return None
    vals, dints, _ = out
    names = [str(v) for v in zg["era5/vars18"][:]]
    tp = np.clip(np.asarray(vals[:, names.index("tp_sum")], float), 0, None) * 1000.0
    return pd.Series(tp, index=_dates(dints)).sort_index()


def load_sm(zg) -> pd.Series | None:
    out = _load_zarr_labels(zg)
    if out is None:
        return None
    sm, depths, times, qc = out
    depths = [d.decode() if isinstance(d, bytes) else str(d) for d in depths]
    if "0-10" not in depths:
        return None
    i = depths.index("0-10")
    k = ~np.isnan(sm[i])
    if qc is not None:
        k &= qc[i] == 0
    return pd.Series(sm[i][k].astype(float), index=pd.to_datetime(times[k])).sort_index()


def load_s2(rg, zg) -> pd.DataFrame | None:
    if "s2" not in rg or rg["s2/data"].shape[0] == 0 or "cm" not in zg:
        return None
    d = _dates(rg["s2/dates"])
    x = np.asarray(rg["s2/data"][:, :, BOX[0], BOX[1]], np.float32)          # (N,12,3,3) DN
    cmd = _dates(zg["cm/dates"])
    cm = np.asarray(zg["cm/masks"][:, BOX[0], BOX[1]])
    cm_by_date = {dt: cm[j] for j, dt in enumerate(cmd)}
    rows = []
    for j, dt in enumerate(d):
        m = cm_by_date.get(dt)
        if m is None or np.isin(m, CM_BAD_CLASSES).any():
            continue
        dn = x[j]
        if (dn[[S2B[b] for b in S2B]] == 0).any():                           # nodata
            continue
        r = {b: float((dn[k].mean() - S2_OFF) / 1e4) for b, k in S2B.items()}
        if not all(0.0 < r[b] < 1.0 for b in r):
            continue
        ndsi = (r["B03"] - r["B11"]) / (r["B03"] + r["B11"])
        if ndsi > 0.4 and r["B03"] > 0.1:                                    # snow
            continue
        r["NDVI"] = (r["B08"] - r["B04"]) / (r["B08"] + r["B04"])
        r["NDMI"] = (r["B08"] - r["B11"]) / (r["B08"] + r["B11"])
        r["date"] = dt
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows).groupby("date").mean()
    return df.drop(columns="B03")


def load_s1(rg, orbit) -> pd.DataFrame | None:
    if orbit not in rg or rg[f"{orbit}/data"].shape[0] < 20:
        return None
    x = np.asarray(rg[f"{orbit}/data"][:, :, BOX[0], BOX[1]], np.float32)     # (N,2,3,3) dB
    x[x == 0] = np.nan
    lin = 10 ** (x / 10)
    db = 10 * np.log10(np.nanmean(lin, axis=(2, 3)))                         # mean in power
    df = pd.DataFrame(db, columns=["VV", "VH"], index=_dates(rg[f"{orbit}/dates"])).dropna()
    return df.groupby(level=0).mean()


def deseason(s: pd.Series) -> pd.Series:
    """Subtract the day-of-year climatology (circular +-CLIM_HALF days, this station's own values)."""
    doy = s.index.dayofyear.to_numpy()
    v = s.to_numpy()
    clim = np.full(367, np.nan)
    for t in range(1, 367):
        dd = np.abs(doy - t)
        w = np.minimum(dd, 366 - dd) <= CLIM_HALF
        if w.sum() >= 3:
            clim[t] = v[w].mean()
    return pd.Series(v - clim[doy], index=s.index)


# ── events ─────────────────────────────────────────────────────────────────────────────────
def find_events(rain: pd.Series, rng) -> tuple[list, list]:
    r = rain.asfreq("D")
    prev = r.shift(1).rolling(DRY_DAYS, min_periods=DRY_DAYS).sum()
    ev = r.index[(r >= RAIN_MM) & (prev < DRY_MM)]
    dry = r.index[(r < PLACEBO_RAIN) & (prev < DRY_MM)]
    ev = [e for e in ev if e - pd.Timedelta(days=-LAG_MIN) >= r.index[0]
          and e + pd.Timedelta(days=LAG_MAX) <= r.index[-1]]
    plac = []
    for e in ev:
        c = dry[(dry.month == e.month) & (np.abs((dry - e).days) > LAG_MAX)]
        if len(c):
            plac.append(c[rng.integers(len(c))])
    return ev, plac


def epoch_rows(s: pd.Series, events, kind, var, min_pre, raw=False) -> list[dict]:
    """Event-relative anomalies of series s (index = dates) around each event."""
    out = []
    if s is None or s.empty:
        return out
    idx = s.index.values
    vals = s.to_numpy()
    for e in events:
        lag = ((idx - np.datetime64(e)) / np.timedelta64(1, "D")).astype(int)
        w = (lag >= LAG_MIN) & (lag <= LAG_MAX) & np.isfinite(vals)
        if not w.any():
            continue
        pre = w & (lag < 0)
        if raw:
            base = 0.0
        else:
            if pre.sum() < min_pre or not (w & (lag >= 0)).any():
                continue
            base = vals[pre].mean()
        for lg, v in zip(lag[w], vals[w]):
            out.append(dict(kind=kind, var=var, event=str(e.date()), lag=int(lg), value=float(v - base)))
    return out


def station_job(job):
    key, cat, macro, seed = job
    rng = np.random.default_rng(seed)
    try:
        zg = _open_zarr(ZARR_ROOT / cat / key, cat)
        rp = RAW / f"{key}.zarr"
        if zg is None or not rp.exists():
            return None
        rg = zarr.open_group(str(rp), mode="r")
        rain, sm = load_rain(zg), load_sm(zg)
        if rain is None or sm is None or len(sm) < 365:
            return None
        rain = rain[(rain.index >= sm.index[0]) & (rain.index <= sm.index[-1])]
        events, plac = find_events(rain, rng)
        if not events:
            return None
        series = [("rain", rain, 0, True), ("SM", deseason(sm), 5, False)]
        s2 = load_s2(rg, zg)
        if s2 is not None and len(s2) >= 20:
            series += [(v, deseason(s2[v]), 1, False) for v in ["B04", "B08", "B11", "B12", "NDVI", "NDMI"]]
        for orbit in ("s1_asc", "s1_desc"):                      # each orbit its own climatology
            s1 = load_s1(rg, orbit)
            if s1 is not None:
                series += [(v, deseason(s1[v]), 1, False) for v in ("VV", "VH")]
        rows = []
        for kind, evs in (("event", events), ("placebo", plac)):
            for var, s, min_pre, raw in series:
                rows += epoch_rows(s, evs, kind, var, min_pre, raw=raw)
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df["station"], df["macro"] = key, macro
        meta = dict(station=key, macro=macro, n_events=len(events), n_placebo=len(plac),
                    sm_days=len(sm))
        return df, meta
    except Exception as ex:                                                  # one bad store must not kill the run
        print(f"  {key}: {type(ex).__name__}: {ex}", flush=True)
        return None


# ── composites ─────────────────────────────────────────────────────────────────────────────
def composite(st: pd.DataFrame, rng) -> pd.DataFrame:
    out = []
    groups = [(m, st[st["macro"] == m]) for m in MACROS] + [("All", st)]
    for macro, g0 in groups:
        for (kind, var, b), g in g0.groupby(["kind", "var", "bin"]):
            v = g["value"].to_numpy()
            if len(v) < 3:
                continue
            boots = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(NBOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append(dict(macro=macro, kind=kind, var=var, lag=int(b), median=float(np.median(v)),
                            lo=float(lo), hi=float(hi), n_stations=len(v)))
    return pd.DataFrame(out)


def lag_metrics(comp: pd.DataFrame) -> pd.DataFrame:
    out = []
    ev = comp[(comp["kind"] == "event") & (comp["lag"] >= 0) & (comp["var"] != "rain")]
    for (macro, var), g in ev.groupby(["macro", "var"]):
        g = g.sort_values("lag")
        p = g.loc[g["median"].abs().idxmax()]
        half = g[(g["lag"] > p["lag"]) & (g["median"].abs() <= 0.5 * abs(p["median"]))]
        out.append(dict(macro=macro, var=var, peak_lag_d=int(p["lag"]), peak=float(p["median"]),
                        peak_lo=float(p["lo"]), peak_hi=float(p["hi"]),
                        half_recovery_d=int(half["lag"].iloc[0]) if len(half) else None))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_output/rain_lag")
    ap.add_argument("--smoke", type=int, default=0, help="N stations (stratified by igbp_macro)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(SPLITS)
    sp["cat"] = sp.apply(category_of, axis=1)
    sp = sp[sp["cat"].isin(SM_CATEGORIES)].copy()
    sp["key"] = sp.apply(station_dir_name, axis=1)
    sp = sp[sp["igbp_macro"].isin(MACROS)]
    if args.smoke:
        per = int(np.ceil(args.smoke / len(MACROS)))
        sp = pd.concat([g.sample(min(per, len(g)), random_state=args.seed)
                        for _, g in sp.groupby("igbp_macro")])
    jobs = [(r.key, r.cat, r.igbp_macro, args.seed + i) for i, r in enumerate(sp.itertuples())]
    print(f"stations: {len(jobs)}  ({sp['igbp_macro'].value_counts().to_dict()})", flush=True)

    with Pool(min(args.workers, len(jobs))) as pool:
        res = [r for r in pool.imap_unordered(station_job, jobs, chunksize=1) if r is not None]
    print(f"stations with events: {len(res)}", flush=True)
    rows = pd.concat([r[0] for r in res], ignore_index=True)
    meta = pd.DataFrame([r[1] for r in res])
    meta.to_csv(out / "stations.csv", index=False)

    rows["bin"] = [b for b in lag_bin_all(rows)]
    st = rows.groupby(["station", "macro", "kind", "var", "bin"], as_index=False)["value"].mean()
    st.to_parquet(out / "rows.parquet", index=False)
    comp = composite(st, np.random.default_rng(args.seed))
    comp.to_csv(out / "composite.csv", index=False)
    lm = lag_metrics(comp)
    lm.to_csv(out / "lags.csv", index=False)

    print("\nevents per station: median", int(meta["n_events"].median()),
          " total", int(meta["n_events"].sum()), " by macro", meta.groupby("macro")["n_events"].sum().to_dict())
    print("\nsanity -- raw value ranges (event rows):")
    print(rows[rows["kind"] == "event"].groupby("var")["value"].describe()[["count", "mean", "min", "max"]]
          .to_string())
    print("\nlag metrics (All):\n" + lm[lm["macro"] == "All"].to_string(index=False))
    print("\nlag metrics by land cover:\n" + lm[lm["macro"] != "All"].to_string(index=False))


def lag_bin_all(rows: pd.DataFrame) -> np.ndarray:
    b = rows["lag"].to_numpy().copy()
    sat = ~rows["var"].isin(["rain", "SM"]).to_numpy()
    b[sat] = 3 * np.round(b[sat] / 3).astype(int)
    return b


if __name__ == "__main__":
    main()
