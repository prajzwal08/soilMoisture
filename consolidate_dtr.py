#!/usr/bin/env python
"""§37.9 -- consolidate the §37 LST blobs into one DTR bundle per station.

INPUT   csvs/ecostress_lst_reads.dtr.s{0..7}.csv   the index (and the resume checkpoint)
        {STAGING}/lst_reads.dtr.s{0..7}.bin        fixed 5120 B records
        csvs/ecostress_wp_pairs.wp.csv             the day/night pairing
        csvs/station_splits.csv                    category + folder name

OUTPUT  {DATA_ROOT}/{cat}/{folder}/ECOSTRESS/{folder}_dtr_{start}_{end}.npz

THREE THINGS THIS FILE GETS RIGHT, each of which is a way to be silently wrong.

1.  read_ok == 1 ONLY.  The shard CSVs carry 31,329 stale read_ok=0 rows from array
    26984229, whose 39.9% failure rate was cookie-jar contention (§37.8).  Those rows
    have blob_offset = -1 and were superseded by array 26987690.  There is exactly one
    read_ok=1 row per (station, granule), so filtering is also the dedupe.

2.  DTR IS ONLY DEFINED ON A SHARED GRID.  Each half's window is cut from its own
    granule, in its own MGRS tile's CRS.  MEASURED over the 39,223 pairs in the band:
    36,547 (93.2%) share both the tile and the window offsets, 2,676 (6.8%) sit on
    DIFFERENT MGRS tiles.  Subtracting those per pixel silently compares different
    ground.  They are kept -- both halves, with their tile ids -- but dtr is NaN and
    grid_aligned = 0, so a reprojection can be added later without re-reading anything.
    Among USABLE pairs the loss is 69 of 11,676 (0.6%): 11,607 over 594 stations.

3.  VALID IS AN INTERSECTION, TWICE OVER.  A pixel counts only where BOTH halves pass
    QC/cloud/water AND both carry a finite LST.  The second half of that is not
    redundant: §37.7 measured LST absent over ~45% of pixels the QC calls clear, because
    an L2T granule is a whole MGRS tile and the ISS swath covers only part of it.  A
    keep-mask intersection alone would count off-swath area as good.

DTR SIGN: day minus night, so positive.  Day and night are stored as the primitives and
DTR is derived here rather than at read time, so the absolute level survives.
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

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import ROOT, STATION_CSV, N_PX_EXPECTED, setup_logging  # noqa: E402

DATA_ROOT = Path(os.getenv("SOIL_DATA_ROOT", "/gpfs/work3/0/prjs1968/data"))
STAGING   = Path("/gpfs/work3/0/prjs1968/data/_ecostress_staging")
SIDE      = 32
REC_BYTES = N_PX_EXPECTED * 4 + N_PX_EXPECTED      # 5120

# Covariates carried through from the pairs file.  dt_hours and the two solar times are
# the §36.24 phase covariates -- phase is a COVARIATE, not a filter (§36.23).
PAIR_COVARS = ["day_utc", "night_utc", "dt_hours", "day_tst", "night_tst",
               "day_elev", "night_elev", "elev_drop", "well_phased",
               "day_clear", "night_clear"]

_MM: dict[int, np.memmap] = {}


def _mm(shard: int) -> np.memmap:
    """One memmap per shard per process.  Opened lazily so Pool workers do not inherit
    file handles through fork."""
    if shard not in _MM:
        _MM[shard] = np.memmap(STAGING / f"lst_reads.dtr.s{shard}.bin",
                               dtype=np.uint8, mode="r")
    return _MM[shard]


def read_record(shard: int, off: int):
    """-> (lst_k float32[32,32] Kelvin, keep bool[32,32])."""
    rec = _mm(shard)[off:off + REC_BYTES]
    lst = rec[:N_PX_EXPECTED * 4].view(np.float32).reshape(SIDE, SIDE)
    keep = rec[N_PX_EXPECTED * 4:].view(np.uint8).reshape(SIDE, SIDE).astype(bool)
    return lst, keep


def station_folder(row) -> str:
    src, net = row["source_network"], row["network"]
    if pd.notna(src) and src != net:
        return f"{src}_{net}_{row['station_id']}"
    return f"{net}_{row['station_id']}"


def category(row) -> str:
    sm, fl = bool(row["has_soil_moisture"]), bool(row["has_flux"])
    return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")


def tile_of(ur: str) -> str:
    p = ur.split("_")
    return p[5] if len(p) >= 7 else ""


def build_index(tag: str, log):
    """(station_id, granule_ur) -> (shard, blob_offset, win_row_off, win_col_off, usable).

    Only read_ok == 1 rows.  The shard number comes from the FILENAME, because the blob
    offset is an offset into that shard's blob and means nothing in another one.
    """
    idx = {}
    files = sorted(ROOT.glob(f"csvs/ecostress_lst_reads.{tag}.s*.csv"))
    if not files:
        log.error("FATAL: no shard CSVs for tag %r", tag)
        sys.exit(1)
    n_stale = 0
    for f in files:
        shard = int(f.stem.split(".s")[-1])
        df = pd.read_csv(f, usecols=["station_id", "granule_ur", "read_ok", "usable_frac",
                                     "win_row_off", "win_col_off", "blob_offset"])
        n_stale += int((df["read_ok"] != 1).sum())
        df = df[df["read_ok"] == 1]
        for sid, ur, uf, ro, co, off in zip(df["station_id"], df["granule_ur"],
                                            df["usable_frac"], df["win_row_off"],
                                            df["win_col_off"], df["blob_offset"]):
            idx[(sid, ur)] = (shard, int(off), int(ro), int(co), float(uf))
    log.info("index             : %d successful reads across %d shard(s) "
             "(%d stale read_ok=0 rows ignored)", len(idx), len(files), n_stale)
    dup = len(idx)
    if dup != sum(1 for _ in idx):
        log.warning("duplicate keys in the index -- this should be impossible")
    return idx


def consolidate_one(job):
    """One station -> one .npz.  Returns a log row."""
    sid, folder, cat, lat, lon, pairs, idx = job
    out_dir = DATA_ROOT / cat / folder / "ECOSTRESS"
    n = len(pairs)

    day_lst = np.full((n, SIDE, SIDE), np.nan, np.float32)
    nig_lst = np.full((n, SIDE, SIDE), np.nan, np.float32)
    dtr     = np.full((n, SIDE, SIDE), np.nan, np.float32)
    valid   = np.zeros((n, SIDE, SIDE), np.uint8)
    aligned = np.zeros(n, np.uint8)

    for i, p in enumerate(pairs.itertuples(index=False)):
        dk, nk = (sid, p.day_ur), (sid, p.night_ur)
        ds, do, dro, dco, _ = idx[dk]
        ns, no, nro, nco, _ = idx[nk]
        dl, dkeep = read_record(ds, do)
        nl, nkeep = read_record(ns, no)
        day_lst[i], nig_lst[i] = dl, nl

        # THE ALIGNMENT GATE.  Same MGRS tile AND same window offsets, or the two 32x32
        # grids are not the same ground and the difference is meaningless.
        if tile_of(p.day_ur) == tile_of(p.night_ur) and (dro, dco) == (nro, nco):
            aligned[i] = 1
            v = dkeep & nkeep & np.isfinite(dl) & np.isfinite(nl)
            valid[i] = v.astype(np.uint8)
            d = dl - nl
            dtr[i] = np.where(v, d, np.nan)

    n_valid = valid.reshape(n, -1).sum(1).astype(np.int32)

    out = {
        "day_lst_k": day_lst, "night_lst_k": nig_lst, "dtr_k": dtr,
        "valid": valid, "n_valid_px": n_valid, "grid_aligned": aligned,
        "day_ur": pairs["day_ur"].to_numpy().astype("S"),
        "night_ur": pairs["night_ur"].to_numpy().astype("S"),
        "day_tile": np.array([tile_of(u) for u in pairs["day_ur"]], dtype="S5"),
        "night_tile": np.array([tile_of(u) for u in pairs["night_ur"]], dtype="S5"),
        "station_id": sid, "latitude": lat, "longitude": lon,
        "side_px": SIDE, "pixel_size_m": 70.0, "tile_m": SIDE * 70.0,
        "dtr_definition": "day_lst_k - night_lst_k on valid pixels only",
    }
    for c in PAIR_COVARS:
        v = pairs[c].to_numpy()
        out[c] = v.astype("S") if v.dtype == object else v

    d0 = str(pairs["day_utc"].min())[:10].replace("-", "")
    d1 = str(pairs["day_utc"].max())[:10].replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{folder}_dtr_{d0}_{d1}.npz"
    np.savez_compressed(path, **out)

    return {"station_id": sid, "folder": folder, "category": cat, "n_pairs": n,
            "n_aligned": int(aligned.sum()),
            "n_pairs_usable": int((n_valid > 0).sum()),
            "median_valid_px": float(np.median(n_valid[n_valid > 0]))
                               if (n_valid > 0).any() else 0.0,
            "path": str(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dtr")
    ap.add_argument("--pairs", default=str(ROOT / "csvs" / "ecostress_wp_pairs.wp.csv"))
    ap.add_argument("--dt-lo", type=float, default=6.0)
    ap.add_argument("--dt-hi", type=float, default=19.0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--network", default="", help="restrict to one ISMN network, e.g. TxSON")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setup_logging(f"consolidate_dtr_{args.network or 'all'}")
    log = logging.getLogger("dtr")

    idx = build_index(args.tag, log)

    stations = pd.read_csv(STATION_CSV)
    if args.network:
        stations = stations[stations["network"] == args.network]
        log.info("network filter    : %s -> %d stations", args.network, len(stations))
    meta = {}
    for _, r in stations.iterrows():
        meta[r["station_id"]] = (station_folder(r), category(r),
                                 float(r["latitude"]), float(r["longitude"]))

    pairs = pd.read_csv(args.pairs)
    n0 = len(pairs)
    pairs = pairs[(pairs["dt_hours"] >= args.dt_lo) & (pairs["dt_hours"] < args.dt_hi)
                  & (pairs["quality"] == 1) & (pairs["both_read"] == 1)]
    log.info("pairs             : %d -> %d in band [%g, %g) h, quality==1, both_read==1",
             n0, len(pairs), args.dt_lo, args.dt_hi)

    # Every half must be in the index.  After §37.8 all 78,446 reads succeeded, so a
    # miss here means the blobs and the CSVs have drifted apart -- stop, do not skip.
    have = pairs.apply(lambda r: (r["station_id"], r["day_ur"]) in idx
                       and (r["station_id"], r["night_ur"]) in idx, axis=1)
    if not have.all():
        log.error("FATAL: %d pairs reference a read that is not in the index. "
                  "Re-run read_ecostress_lst.py --retry-failed before consolidating.",
                  int((~have).sum()))
        sys.exit(1)

    jobs = []
    for sid, grp in pairs.groupby("station_id", sort=True):
        if sid not in meta:
            if not args.network:      # under a network filter this is the filter working
                log.warning("station %s is in the pairs file but not in station_splits.csv"
                            " -- skipped", sid)
            continue
        folder, cat, lat, lon = meta[sid]
        if not args.overwrite:
            existing = list((DATA_ROOT / cat / folder / "ECOSTRESS")
                            .glob(f"{folder}_dtr_*.npz"))
            if existing:
                continue
        jobs.append((sid, folder, cat, lat, lon,
                     grp.sort_values("day_utc").reset_index(drop=True), idx))

    log.info("stations to write : %d (workers=%d)", len(jobs), args.workers)
    if not jobs:
        log.info("nothing to do")
        return

    with Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(consolidate_one, jobs, chunksize=1)

    df = pd.DataFrame(rows)
    out_csv = ROOT / "csvs" / f"ecostress_dtr_bundles.{args.network or 'all'}.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)
    log.info("SUMMARY  stations=%d  pairs=%d  aligned=%d (%.1f%%)  "
             "pairs with >=1 valid px=%d (%.1f%%)  stations with >=1 usable pair=%d",
             len(df), int(df.n_pairs.sum()), int(df.n_aligned.sum()),
             100 * df.n_aligned.sum() / max(df.n_pairs.sum(), 1),
             int(df.n_pairs_usable.sum()),
             100 * df.n_pairs_usable.sum() / max(df.n_pairs.sum(), 1),
             int((df.n_pairs_usable > 0).sum()))


if __name__ == "__main__":
    main()
