#!/usr/bin/env python
"""Does S1 resolve the six TxSON probes the way soil moisture does, where DTR could not?

THE TEST DTR FAILED (§38.10).  Six in-situ probes sit inside one 2.24 km window --
CR200-18 centre, CR200-25 405 m, CR1000-2 684 m, CR200-24 865 m, CR200-15 925 m, CR200-6
936 m.  Their soil moisture differs by a factor of 2.4.  DTR did not separate them.

TWO CORRECTIONS TO HOW §38.10 MEASURED IT, both of which make the test FAIRER TO DTR:

  1. §38.10 compared SCENE MEANS over windows that overlap almost entirely, so they were
     near-identical by construction and the 3.2% CV was partly guaranteed.  Here every
     signal is read AT EACH STATION'S OWN PIXEL -- each window is centred on its own
     station, so that is index (16,16) of the 32x32 LST grid and (112,112) of the 224x224
     S1 grid.  Footprints are matched at ~70 m: 1 LST pixel against a 7x7 S1 block.
  2. CV (SD/mean) is undefined for d_VV, which is a zero-mean anomaly by construction.
     So the quantity actually reported is the one CV was standing in for: ON EACH DATE,
     ACROSS THE SIX STATIONS, does the signal rank them the way SM ranks them?

SIGNS DIFFER AND BOTH ARE PREDICTED.  Wetter soil raises the dielectric constant and so
RAISES backscatter: r(d_VV, SM) should be POSITIVE.  Wetter soil damps the diurnal swing:
r(DTR, SM) should be NEGATIVE.

d_VV IS THE §33.12 DOUBLE-CENTRED ANOMALY (text/s1processing.md:562) -- each cell minus
its own temporal norm minus the whole-tile level that day.  The static landscape pattern
is exactly what the temporal norm absorbs, which is the structural reason to expect this
to behave unlike DTR, whose pattern IS the static one (§38.6, r = +0.816 against day LST).
Double-centring is done in dB, which is linear and is what §33.12 line 728 relies on.

§29.10 GOVERNS THE READING: n = 6 per date gives roughly a +/-0.7 CI on a single-date r.
No single date may be quoted.  Power comes from aggregating dates plus the sign test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
import dataset as _ds                                          # noqa: E402
_ds.ZARR_ROOT = Path("/projects/prjs1968/zarr_tokens")
from dataset import _load_zarr_labels, _open_zarr               # noqa: E402
from census_ecostress import ROOT                               # noqa: E402

ZARR_TOK = Path("/projects/prjs1968/zarr_tokens")
ZARR_SAT = Path("/projects/prjs1968/satellite_zarr")
SIX = [("CR200-18", "centre"), ("CR200-25", "405 m"), ("CR1000-2", "684 m"),
       ("CR200-24", "865 m"), ("CR200-15", "925 m"), ("CR200-6", "936 m")]
S1_HALF = 3          # 7x7 at 10 m ~ 70 m, matching one ECOSTRESS pixel


def s1_station_series(folder, orbit="s1_asc"):
    """-> DataFrame[date, d_VV, d_CR, c_VV] at the station's own pixel."""
    import zarr
    p = ZARR_SAT / f"{folder}.zarr"
    if not p.exists():
        return None
    g = zarr.open(str(p), mode="r")
    if orbit not in g or g[orbit]["data"].shape[0] < 20:
        return None
    x = np.asarray(g[orbit]["data"][:], dtype=np.float32)      # [K,2,224,224] dB
    dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d)
                            for d in g[orbit]["dates"][:]]).normalize()
    x[x == 0] = np.nan                                          # fill_value 0 -> missing

    out = {}
    for bi, band in enumerate(("VV", "VH")):
        b = x[:, bi]                                            # [K,224,224] dB
        # DOUBLE-CENTRING: cell minus its own temporal norm, minus the tile level today.
        c = np.nanmean(b, axis=0, keepdims=True)                # [1,H,W] the cell norm
        w = np.nanmean(b, axis=(1, 2), keepdims=True)           # [K,1,1] tile level today
        grand = np.nanmean(b)
        out[band] = b - c - w + grand
        if band == "VV":
            out["c_VV_map"] = c[0]

    h = S1_HALF
    sl = (slice(112 - h, 112 + h + 1), slice(112 - h, 112 + h + 1))
    d_vv = np.nanmean(out["VV"][:, sl[0], sl[1]], axis=(1, 2))
    d_vh = np.nanmean(out["VH"][:, sl[0], sl[1]], axis=(1, 2))
    return pd.DataFrame({"date": dates, "d_VV": d_vv, "d_CR": d_vh - d_vv,
                         "c_VV": np.nanmean(out["c_VV_map"][sl[0], sl[1]])}).dropna(
        subset=["d_VV"])


def sm_series(folder, cat):
    zg = _open_zarr(ZARR_TOK / cat / folder, cat)
    out = _load_zarr_labels(zg) if zg is not None else None
    if out is None:
        return None
    sm, depths, times, qc = out
    for i, dep in enumerate(depths):
        dep = dep.decode() if isinstance(dep, bytes) else str(dep)
        if dep != "0-10":
            continue
        k = ~np.isnan(sm[i])
        if qc is not None:
            k &= (qc[i] == 0)
        return pd.DataFrame({"date": pd.to_datetime(times[k]).normalize(),
                             "sm": sm[i][k].astype(np.float32)})
    return None


def dtr_station_pixel(path):
    """DTR at the station's OWN pixel (16,16), not the scene mean."""
    z = np.load(path, allow_pickle=False)
    ok = (z["grid_aligned"] == 1) & (z["n_valid_px"] > 0)
    nd = int(ok.sum())
    dtr = z["dtr_k"][ok]
    val = z["valid"][ok].astype(bool)
    c = dtr[:, 16, 16].astype(float)
    c[~val[:, 16, 16]] = np.nan
    dates = pd.to_datetime([str(x.decode() if isinstance(x, bytes) else x)[:10]
                            for x in z["day_utc"][ok]]).normalize()
    return pd.DataFrame({"date": dates, "dtr": c}).dropna()


def per_date_r(wide_sig, wide_sm):
    """r across stations on each date -> (mean r, frac positive, n dates)."""
    rs = []
    for dt in wide_sig.index.intersection(wide_sm.index):
        a, b = wide_sig.loc[dt].to_numpy(float), wide_sm.loc[dt].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 4 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
            continue
        rs.append(np.corrcoef(a[m], b[m])[0, 1])
    rs = np.array(rs)
    if rs.size == 0:
        return np.nan, np.nan, 0
    return float(rs.mean()), float((rs > 0).mean()), int(rs.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=str(ROOT / "csvs" / "ecostress_dtr_bundles.TxSON.csv"))
    args = ap.parse_args()

    b = pd.read_csv(args.bundles).set_index("station_id")
    S1, SM, DTR = {}, {}, {}
    for sid, _ in SIX:
        row = b.loc[sid]
        s1 = s1_station_series(row["folder"])
        sm = sm_series(row["folder"], row["category"])
        dt = dtr_station_pixel(row["path"])
        print(f"{sid:10s}  S1 {0 if s1 is None else len(s1):4d} dates   "
              f"SM {0 if sm is None else len(sm):5d} days   DTR {len(dt):3d} days")
        if s1 is not None:
            S1[sid] = s1.set_index("date")["d_VV"]
        if sm is not None:
            SM[sid] = sm.set_index("date")["sm"]
        DTR[sid] = dt.set_index("date")["dtr"]

    s1w = pd.DataFrame(S1); smw = pd.DataFrame(SM); dtw = pd.DataFrame(DTR)
    cols = [s for s, _ in SIX if s in s1w.columns]
    s1w, dtw = s1w[cols], dtw[[c for c in cols if c in dtw.columns]]
    smw = smw[cols]

    print(f"\ndates with all six: S1 {int(s1w.notna().all(1).sum())}   "
          f"DTR {int(dtw.notna().all(1).sum())}")

    print("\n--- BETWEEN-STATION SPREAD at each station's OWN pixel ---")
    for name, w, unit in (("d_VV (S1)", s1w, "dB"), ("DTR", dtw, "K"),
                          ("SM (in-situ)", smw.reindex(s1w.index), "m3/m3")):
        f = w.dropna()
        if len(f) == 0:
            continue
        bs = f.std(axis=1).mean()
        ts = f.mean(axis=1).std()
        print(f"  {name:14s} between-station SD {bs:8.4f} {unit:6s}"
              f"   temporal SD of the tile mean {ts:8.4f}"
              f"   between/temporal {bs/max(ts,1e-9):5.2f}")

    print("\n--- 0. THE BASIC CHECK: does d_VV track its OWN point's wetness in time? ---")
    print("    (per station, over its own dates; physics predicts POSITIVE)")
    tmp = []
    for sid in cols:
        j = pd.concat([s1w[sid].rename("d"), smw[sid].rename("sm")], axis=1).dropna()
        if len(j) < 10:
            continue
        a = j["d"].to_numpy(); b = (j["sm"] - j["sm"].mean()).to_numpy()
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else np.nan
        tmp.append(r)
        print(f"    {sid:10s} r = {r:+.3f}   n = {len(j)}")
    if tmp:
        print(f"    median {np.nanmedian(tmp):+.3f}   "
              f"{100*np.mean(np.array(tmp) > 0):.0f}% positive of {len(tmp)}")

    print("\n--- DOES IT RANK THE SIX THE WAY SM DOES?  (per date, n=6) ---")
    print("    BOTH SIDES AS ANOMALIES FROM EACH STATION'S OWN NORM.  d_VV is")
    print("    double-centred, so every cell's temporal mean is ZERO by construction;")
    print("    correlating it against absolute SM compares a deviation to a level and")
    print("    is guaranteed to return nothing.  SM is therefore centred per station.")
    sm_anom = smw - smw.reindex(s1w.index).mean()
    r1, f1, n1 = per_date_r(s1w, sm_anom)
    r1r, f1r, n1r = per_date_r(s1w, smw)
    r2, f2, n2 = per_date_r(dtw, sm_anom.reindex(dtw.index))
    print(f"\n  d_VV vs SM ANOMALY   mean r {r1:+.3f}   {100*f1:5.1f}% of dates positive"
          f"   n = {n1:4d}   (predicts POSITIVE)")
    print(f"  d_VV vs raw SM       mean r {r1r:+.3f}   {100*f1r:5.1f}% positive"
          f"   n = {n1r:4d}   (the broken comparison, kept as a control)")
    print(f"  DTR  vs SM ANOMALY   mean r {r2:+.3f}   {100*(1-f2):5.1f}% of dates negative"
          f"   n = {n2:4d}   (predicts NEGATIVE)")
    print("\n  §29.10: a single date's r carries ~±0.7 CI at n=6. Read the sign test,")
    print("  not the mean, and note a coin flip is 50%.")

    print("\n--- station means, for reference ---")
    t = pd.DataFrame({"sm": smw.reindex(s1w.index).mean(),
                      "d_VV": s1w.mean(), "dtr": dtw.mean()})
    print(t.round(3).to_string())
    t.to_csv(ROOT / "csvs" / "txson_six_s1_vs_sm.csv")


if __name__ == "__main__":
    main()
