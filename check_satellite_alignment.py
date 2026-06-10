"""
For every station, compare raw satellite_zarr date counts against token-zarr
date counts for s2, s1_asc, s1_desc, and cm-vs-s2, to see how widespread the
AmeriFlux_CA-Cbo-style mismatch (token has MORE dates than raw satellite_zarr)
is across the dataset.

Usage:
    conda run -n sensei python check_satellite_alignment.py [--workers 32]
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import zarr

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ROOT  = Path("/projects/prjs1968/satellite_zarr")
SPLITS_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")


def _category(r):
    if r["has_soil_moisture"] and r["has_flux"]:
        return "sm_and_flux"
    if r["has_soil_moisture"]:
        return "sm_only"
    return "flux_only"


def _dir_name(r):
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def _dates(zg, key):
    if key not in zg:
        return None
    if not hasattr(zg[key], "array_keys"):
        return None
    if "dates" not in zg[key].array_keys():
        return None
    return set(d.decode() if isinstance(d, bytes) else str(d) for d in zg[f"{key}/dates"][:])


def process(args):
    cat, station = args
    out = {"station": station, "category": cat, "has_raw": False}

    token_path = ZARR_ROOT / cat / station
    try:
        zg = zarr.open_consolidated(str(token_path), mode="r")
    except Exception:
        try:
            zg = zarr.open_group(str(token_path), mode="r")
        except Exception as e:
            out["error"] = f"token open failed: {e}"
            return out

    token = {m: _dates(zg, m) for m in ["s2", "s1_asc", "s1_desc", "cm"]}

    raw_path = SAT_ROOT / f"{station}.zarr"
    if not raw_path.exists():
        out["has_raw"] = False
    else:
        out["has_raw"] = True
        try:
            zg2 = zarr.open_group(str(raw_path), mode="r")
        except Exception as e:
            out["error"] = f"raw open failed: {e}"
            return out
        raw = {m: _dates(zg2, m) for m in ["s2", "s1_asc", "s1_desc"]}

        for m in ["s2", "s1_asc", "s1_desc"]:
            t, r = token[m], raw[m]
            out[f"n_{m}_token"] = len(t) if t is not None else None
            out[f"n_{m}_raw"] = len(r) if r is not None else None
            if t is not None and r is not None:
                out[f"{m}_token_minus_raw"] = len(t - r)
                out[f"{m}_raw_minus_token"] = len(r - t)

    # cm vs s2 (token-only check)
    s2t, cmt = token["s2"], token["cm"]
    out["n_s2_token"] = len(s2t) if s2t is not None else None
    out["n_cm_token"] = len(cmt) if cmt is not None else None
    if s2t is not None and cmt is not None:
        out["s2_minus_cm"] = len(s2t - cmt)
        out["cm_minus_s2"] = len(cmt - s2t)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(SPLITS_CSV)
    df["cat"] = df.apply(_category, axis=1)
    df["dir_name"] = df.apply(_dir_name, axis=1)
    jobs = list(zip(df["cat"], df["dir_name"]))

    with Pool(args.workers) as pool:
        results = pool.map(process, jobs)

    out_df = pd.DataFrame(results)
    out_df.to_csv("csvs/satellite_alignment.csv", index=False)

    print(f"Total stations: {len(out_df)}")
    print(f"has_raw satellite_zarr: {out_df['has_raw'].sum()}/{len(out_df)}")

    for m in ["s2", "s1_asc", "s1_desc"]:
        col_tmr = f"{m}_token_minus_raw"
        col_rmt = f"{m}_raw_minus_token"
        if col_tmr not in out_df:
            continue
        n_mismatch_tmr = (out_df[col_tmr].fillna(0) > 0).sum()
        n_mismatch_rmt = (out_df[col_rmt].fillna(0) > 0).sum()
        print(f"\n{m}: token has extra dates not in raw: {n_mismatch_tmr} stations")
        print(f"{m}: raw has extra dates not in token: {n_mismatch_rmt} stations")
        if n_mismatch_tmr:
            sub = out_df[out_df[col_tmr].fillna(0) > 0][["station", "category", f"n_{m}_token", f"n_{m}_raw", col_tmr]]
            print(sub.to_string(index=False))

    # cm vs s2
    n_s2_minus_cm = (out_df["s2_minus_cm"].fillna(0) > 0).sum()
    print(f"\ns2 dates without cm: {n_s2_minus_cm} stations")
    if n_s2_minus_cm:
        sub = out_df[out_df["s2_minus_cm"].fillna(0) > 0][["station", "category", "n_s2_token", "n_cm_token", "s2_minus_cm"]]
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
