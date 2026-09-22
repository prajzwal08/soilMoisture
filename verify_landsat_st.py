#!/usr/bin/env python
"""Read every Landsat ST bundle back and check it is what the pull claims it is.

A download that reports success is not evidence.  §37.8's cookie-jar contention produced 31,329
rows saying read_ok=0 that were still counted as done; §41.6's "corrupt asset" was only ever
visible in the bytes.  So this opens the arrays.

The scale checks matter most.  The bundle applies each asset's scale/offset from the item's
raster:bands, and the ECOSTRESS v002 burn was exactly a wrong scale that produced numbers which
looked like data: the documented uint16 x 0.02 was the SWATH spec while the tiled product was
float32 Kelvin.  Emissivity in [0.7, 1.0] and cloud distance in kilometres (not metres) are the
two cheapest ways to catch that class of error, so both are asserted, not just reported.

OUTPUT  csvs/landsat_st30_verify.csv     one row per bundle
Exit 1 if any station fails, so the job turns red rather than printing into a log nobody reads.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_landsat_st30 import qa_decode  # noqa: E402  the reference decoder

REPO      = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("SOIL_DATA_ROOT", "/gpfs/work3/0/prjs1968/data"))
OUT_CSV   = REPO / "csvs" / "landsat_st30_verify.csv"
CKPT_GLOB = "landsat_st30_log*.csv"

GRID_N     = 76
K_LO, K_HI = 220.0, 360.0   # 360 not 340: bare desert soil is genuinely this hot -- Landsat
                            # measured 354.5 K at Stovepipe Wells (Death Valley) and 345 K at
                            # Yuma.  A 340 K ceiling flags real deserts as broken data.
LST_SATURATED = 372.9999    # = DN 65535, the uint16 ceiling, exactly.  NOT a temperature:
                            # three SNOTEL stations hit 373.000 to four decimals.  Consumers
                            # must drop DN 65535 alongside DN 0.
EMIS_LO, EMIS_HI = 0.70, 1.00      # ASTER GED emissivity; a scale error lands far outside
CDIST_MAX_KM     = 250.0           # ST_CDIST is documented 0-24000 DN x 0.01 = 0-240 km, so
                                   # the guard only has to exclude metres (~24000)

PER_SCENE = ["lst30", "st_qa30", "cdist30", "qa_pixel30", "dates", "item_ids", "platform",
             "wrs_path", "wrs_row", "native_epsg", "reprojected"]
CUBES     = ["lst30", "st_qa30", "cdist30", "qa_pixel30"]



def _finish(r: dict, fails: list, path: Path) -> dict:
    r["mb"] = round(path.stat().st_size / 1e6, 2)
    r["fail"] = " ; ".join(fails)
    r["ok"] = int(not fails)
    return r

def check(path_str: str) -> dict:
    path = Path(path_str)
    r = {"bundle": path.name, "station_dir": path.parent.parent.name, "ok": 0, "fail": ""}
    fails = []
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:
        r["fail"] = f"unreadable: {str(exc)[:130]}"
        return r

    missing = [k for k in PER_SCENE + ["emis30", "meta"] if k not in z]
    if missing:
        r["fail"] = f"missing arrays: {missing}"
        return r

    n = z["lst30"].shape[0]
    r["n_scenes"] = int(n)
    if n == 0:
        r["fail"] = "zero scenes"
        return r

    bad_n = [k for k in PER_SCENE if z[k].shape[0] != n]
    if bad_n:
        fails.append(f"length mismatch on {bad_n}")

    for k in CUBES:
        if z[k].shape[1:] != (GRID_N, GRID_N):
            fails.append(f"{k} grid is {z[k].shape[1:]}, expected ({GRID_N}, {GRID_N})")
    if z["emis30"].shape != (GRID_N, GRID_N):
        fails.append(f"emis30 is {z['emis30'].shape}, expected ({GRID_N}, {GRID_N})")

    d = z["dates"]
    if not np.all(d[:-1] <= d[1:]):
        fails.append("dates not sorted")
    r["n_dup_dates"] = int(len(d) - len(set(d.tolist())))   # legal: WRS sidelap, L8+L9 same day

    # ---- lst30
    # The arrays are stored RAW, so cloud tops, shadow and diverged retrievals are all still
    # in there -- that is the point.  The physical range therefore only binds over pixels the
    # reference decoder calls clear; asserting it over every pixel would just be re-testing
    # that we did not apply QC.  The raw range is reported, the CLEAR range is enforced.
    lst = z["lst30"]
    fin = np.isfinite(lst)
    r["lst_valid_frac"] = round(float(fin.mean()), 5)
    if not fin.any():
        fails.append("lst30 entirely NaN")
        return _finish(r, fails, path)

    v = lst[fin]
    r["lst_raw_min"], r["lst_raw_max"] = round(float(v.min()), 2), round(float(v.max()), 2)
    r["lst_median"] = round(float(np.median(v)), 3)
    r["n_allnan_scenes"] = int((~fin.reshape(n, -1).any(axis=1)).sum())

    clear, _water = qa_decode(z["qa_pixel30"].astype("float64"))
    ok = clear & fin
    r["clear_frac_decoded"] = round(float(ok.mean()), 5)
    if not ok.any():
        fails.append("no clear pixel in the whole bundle")
    else:
        c = lst[ok]
        # DN 65535 -> 372.99994 K exactly.  That is the uint16 ceiling, a saturation sentinel,
        # not a temperature; it is counted and excluded before the range test rather than
        # dragging the ceiling up to 373 K and blinding the test to real divergence.
        sat = c >= LST_SATURATED
        r["n_clear_saturated"] = int(sat.sum())
        r["frac_clear_saturated"] = round(float(sat.mean()), 8)
        c = c[~sat]
        if c.size == 0:
            fails.append("every clear pixel is saturated (DN 65535)")
            return _finish(r, fails, path)

        r["lst_clear_min"], r["lst_clear_max"] = round(float(c.min()), 2), round(float(c.max()), 2)
        r["lst_clear_median"] = round(float(np.median(c)), 3)
        n_out = int(((c < K_LO) | (c > K_HI)).sum())
        r["n_clear_out_of_range"] = n_out
        r["frac_clear_out_of_range"] = round(n_out / c.size, 7)
        # a handful of diverged retrievals that QA_PIXEL misses is expected; a systematic
        # failure is not.  1 in 10,000 clear pixels is the line.
        if n_out / c.size > 1e-4:
            fails.append(f"{n_out} clear pixels ({100*n_out/c.size:.3f}%) outside "
                         f"[{K_LO},{K_HI}] K: {c.min():.1f}..{c.max():.1f}")
        # saturation should be vanishingly rare; if it is not, the station is in trouble
        if r["frac_clear_saturated"] > 1e-3:
            fails.append(f"{r['n_clear_saturated']} clear pixels "
                         f"({100*r['frac_clear_saturated']:.3f}%) saturated at DN 65535")

    # ---- st_qa30
    q = z["st_qa30"]
    qf = np.isfinite(q)
    if qf.any():
        r["stqa_median"] = round(float(np.median(q[qf])), 4)
        if q[qf].min() < 0:
            fails.append(f"negative ST_QA: {q[qf].min():.3f}")
        if r["stqa_median"] > 20:
            fails.append(f"ST_QA median {r['stqa_median']} K implausible -- scale error?")
    # the two no-retrieval masks were measured identical (Step 0, agreement 1.00000)
    r["nodata_agree"] = round(float((fin == qf).mean()), 5)

    # ---- cdist30: kilometres, not metres
    c = z["cdist30"]
    cf = np.isfinite(c)
    if cf.any():
        r["cdist_median"] = round(float(np.median(c[cf])), 4)
        r["cdist_max"] = round(float(c[cf].max()), 3)
        if r["cdist_max"] > CDIST_MAX_KM:
            fails.append(f"cdist max {r['cdist_max']} -- metres, not km? scale error")

    # ---- qa_pixel30 must still be a raw bitfield
    if z["qa_pixel30"].dtype != np.uint16:
        fails.append(f"qa_pixel30 dtype is {z['qa_pixel30'].dtype}, expected uint16")
    else:
        qp = z["qa_pixel30"]
        r["qa_n_unique"] = int(np.unique(qp).size)
        if r["qa_n_unique"] < 2:
            fails.append(f"qa_pixel30 has {r['qa_n_unique']} unique value(s) -- not a bitfield")

    # ---- emis30: ASTER GED, a scale error lands far outside [0.7, 1.0]
    e = z["emis30"]
    ef = np.isfinite(e)
    if not ef.any():
        fails.append("emis30 entirely NaN")
    else:
        r["emis_median"] = round(float(np.median(e[ef])), 5)
        if not (EMIS_LO <= r["emis_median"] <= EMIS_HI):
            fails.append(f"emis median {r['emis_median']} outside [{EMIS_LO},{EMIS_HI}] "
                         f"-- scale error")

    r["n_reprojected"] = int(z["reprojected"].sum())
    r["n_native_epsg"] = int(len(set(z["native_epsg"].tolist())))

    if "tile_mean_k" in z:
        tm = z["tile_mean_k"]
        if not np.any((tm > K_LO) & (tm < K_HI)):
            fails.append("no scene with a plausible tile mean -- the tripwire should have fired")
    if "clear_frac" in z:
        r["mean_clear_frac"] = round(float(np.nanmean(z["clear_frac"])), 5)

    return _finish(r, fails, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    bundles = sorted(str(p) for p in args.data_root.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    logging.info("%d bundles under %s", len(bundles), args.data_root)
    if not bundles:
        raise SystemExit("no bundles -- nothing to verify")

    with Pool(args.workers) as pool:
        df = pd.DataFrame(pool.map(check, bundles, chunksize=4))
    df.to_csv(OUT_CSV, index=False)

    ck = []
    for f in (REPO / "csvs").glob(CKPT_GLOB):
        try:
            ck.append(pd.read_csv(f))
        except Exception:
            pass
    n_done = int((pd.concat(ck, ignore_index=True).status == "done").sum()) if ck else 0

    n_ok = int(df.ok.sum())
    print("=" * 76)
    print("LANDSAT ST30 -- BUNDLE VERIFICATION")
    print("=" * 76)
    print(f"  bundles on disk      : {len(df)}")
    print(f"  passed               : {n_ok}")
    print(f"  FAILED               : {len(df) - n_ok}")
    print(f"  checkpoint says done : {n_done}")
    if n_done and n_done != len(df):
        print(f"  !! checkpoint/disk mismatch of {abs(n_done - len(df))}")
    for col, lab, fmt in (("n_scenes", "scenes per station", "{:.0f}"),
                          ("mb", "MB per station", "{:.1f}")):
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                print(f"  {lab:<20} : min {fmt.format(s.min())}  median {fmt.format(s.median())}"
                      f"  max {fmt.format(s.max())}  total {fmt.format(s.sum())}")
    for col, lab in (("lst_median", "LST median, raw (K)"),
                     ("lst_clear_median", "LST median, clear (K)"),
                     ("lst_raw_min", "LST raw min (K)"), ("lst_clear_min", "LST clear min (K)"),
                     ("frac_clear_out_of_range", "clear px out of range"),
                     ("frac_clear_saturated", "clear px saturated"),
                     ("stqa_median", "ST_QA median (K)"),
                     ("cdist_median", "CDIST median (km)"), ("emis_median", "emis median"),
                     ("lst_valid_frac", "LST valid frac"), ("nodata_agree", "nodata agreement"),
                     ("mean_clear_frac", "clear frac")):
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                print(f"  {lab:<20} : {s.median():.4f}")
    if "n_reprojected" in df:
        print(f"  stations with reprojected scenes : "
              f"{int((df.n_reprojected > 0).sum())} of {len(df)}  "
              f"({int(df.n_reprojected.sum())} scenes)")
    print(f"  wrote {OUT_CSV}")
    bad = df[df.ok == 0]
    if len(bad):
        print("-" * 76)
        print("FAILURES:")
        for _, b in bad.head(40).iterrows():
            print(f"  {str(b.station_dir):<32} {str(b.fail)[:110]}")
    print("=" * 76)
    sys.exit(0 if n_ok == len(df) else 1)


if __name__ == "__main__":
    main()
