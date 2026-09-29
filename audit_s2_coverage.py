"""
audit_s2_coverage.py — did the S2 download silently skip scenes? (§48 finding, Session 44)
==========================================================================================
download_s2_mpc.py skips a scene that fails after retries and still marks the station
"done". On 2026-05-20 an expired Planetary Computer SAS token (se=11:05:26Z) produced
853,552 HTTP 403s, and 8 train stations ended up with no pre-2023 S2 at all. This audit
asks the same question of every station, per year:

    catalogue  unique acquisition dates MPC returns (sentinel-2-l2a, same bbox, cloud < 75)
    obtained   unique dates in satellite_zarr/{station}.zarr/s2/dates
               + dates the cloud filter deleted (text/cloudy_tile_manifest_delete_log.csv)

obtained / catalogue per year should be ~1 (the download fetched everything the catalogue
had, and only the cloud filter removed some). A year well below 1 is a silent download loss.
Read-only: nothing is downloaded or written except the CSV.

OUTPUT  csvs/s2_coverage_audit.csv   one row per (station, year)
Usage:  sbatch slurm/audit_s2_coverage.sh [--stations A B ...] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import json

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from splits_config import ALL_CATEGORIES, category_of, station_dir_name  # noqa: E402

RAW_ROOT   = Path("/projects/prjs1968/satellite_zarr")
STORE_DATES = REPO / "csvs" / "_s2_store_dates.json"   # written by --dump-store (terramind env)
DELETE_LOG = REPO / "text" / "cloudy_tile_manifest_delete_log.csv"
BACKFILL_MANIFEST = REPO / "text" / "s2_backfill_manifest"     # §50 per-station keep/reject
OUT        = REPO / "csvs" / "s2_coverage_audit.csv"
YEARS      = list(range(2016, 2026))
MAX_CLOUD  = 75
_DELETED: dict[str, set] = {}


def _catalogue_dates(lat, lon):
    import planetary_computer
    import pystac_client
    from download_s2_mpc import MPC_URL, station_grid
    _, _, bbox = station_grid(lat, lon)
    cat = pystac_client.Client.open(MPC_URL, modifier=planetary_computer.sign_inplace)
    for attempt in range(4):
        try:
            items = cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                               datetime="2016-01-01/2025-12-31",
                               query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items()
            return {int(i.datetime.strftime("%Y%m%d")) for i in items}
        except Exception:                               # noqa: BLE001
            time.sleep(10 * (attempt + 1))
    return None


def one(task):
    name = task["name"]
    rows = []
    cat = _catalogue_dates(task["lat"], task["lon"])
    store = task["store"]
    deleted = _DELETED.get(name, set())
    # The download window is [max(start, 2016-01-01), end] from the station CSV; scenes
    # outside it were never requested, so they are clipped on BOTH sides, at date level.
    lo, hi = max(task["start"], 20160101), task["end"]
    clip = lambda s: {d for d in s if lo <= d <= hi} if s is not None else None  # noqa: E731
    cat, store, deleted = clip(cat), clip(store), clip(deleted)
    for y in YEARS:
        in_y = lambda s: {d for d in s if d // 10000 == y}  # noqa: E731
        c = len(in_y(cat)) if cat is not None else np.nan
        s = len(in_y(store)) if store is not None else np.nan
        dl = len(in_y(deleted))
        got = (len(in_y(store) | in_y(deleted)) if store is not None else np.nan)
        rows.append(dict(station=name, split=task["split"], year=y, start=task["start"], end=task["end"],
                         catalogue=c, store=s, cloud_deleted=dl, obtained=got,
                         ratio=(got / c) if (c and c == c and got == got) else np.nan))
    return rows


def dump_store(store_json=STORE_DATES):
    """terramind env: the soilmoisture env has pystac but no zarr, so the store dates are
    read in a separate step and handed over as JSON."""
    import zarr
    out = {}
    for p in sorted(RAW_ROOT.glob("*.zarr")):
        try:
            rg = zarr.open_group(str(p), mode="r")
            out[p.stem] = ([int(bytes(d).decode()[:8]) for d in rg["s2/dates"][:]]
                           if "s2/dates" in rg else [])
        except Exception:                               # noqa: BLE001
            out[p.stem] = None
    store_json.write_text(json.dumps(out))
    print(f"store dates for {len(out)} stations -> {store_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-store", action="store_true")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--store-json", type=Path, default=STORE_DATES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.dump_store:
        return dump_store(args.store_json)
    store_dates = json.loads(args.store_json.read_text())

    dl = pd.read_csv(DELETE_LOG, usecols=["station", "date", "s2_deleted"])
    dl = dl[dl["s2_deleted"].astype(str) == "True"]
    for st, g in dl.groupby("station"):
        _DELETED[st] = set(g["date"].astype(int))
    # §50 backfill: scenes the backfill cloud filter rejected were obtained, then filtered
    for p in BACKFILL_MANIFEST.glob("*.csv"):
        m = pd.read_csv(p, usecols=["station", "date", "verdict"])
        for st, g in m[m["verdict"] == "reject"].groupby("station"):
            _DELETED.setdefault(st, set()).update(g["date"].astype(int))

    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    tasks, seen = [], set()
    for _, r in df[df["cat"].isin(ALL_CATEGORIES)].iterrows():
        n = station_dir_name(r)
        if n in seen or (args.stations and n not in args.stations):
            continue
        seen.add(n)
        tasks.append(dict(name=n, split=r["split"], start=int(r["start_date"]), end=int(r["end_date"]),
                          lat=float(r["latitude"]), lon=float(r["longitude"]),
                          store=(set(store_dates[n]) if store_dates.get(n) is not None
                                 else None)))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"stations: {len(tasks)}", flush=True)

    out = []
    with Pool(args.workers) as p:
        for i, rows in enumerate(p.imap_unordered(one, tasks), 1):
            out.extend(rows)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    res = pd.DataFrame(out)
    # only years the station's own record covers count as "should have been downloaded"
    res["in_window"] = ((res["year"] >= (res["start"] // 10000).clip(lower=2016))
                        & (res["year"] <= res["end"] // 10000))
    res.to_csv(args.out if not args.stations and not args.limit else args.out.with_suffix(".smoke.csv"),
               index=False)

    w = res[res["in_window"] & (res["catalogue"] > 0)]
    print(f"\ncatalogue failures: {int(res.groupby('station')['catalogue'].apply(lambda s: s.isna().all()).sum())} stations")
    print("\nobtained/catalogue, station-years in window — distribution:")
    print(w["ratio"].describe(percentiles=[.01, .05, .1, .25, .5]).to_string())
    for thr in (0.1, 0.5, 0.8):
        bad = w[w["ratio"] < thr]
        print(f"  ratio < {thr}: {len(bad)} station-years at {bad['station'].nunique()} stations "
              f"(split {bad.drop_duplicates('station')['split'].value_counts().to_dict()})")
    print("\nby year, median ratio:")
    print(w.groupby("year")["ratio"].median().round(3).to_string())
    worst = (w.assign(lost=w["catalogue"] - w["obtained"])
               .groupby(["station", "split"]).agg(lost=("lost", "sum"), cat=("catalogue", "sum"))
               .assign(frac_lost=lambda d: d["lost"] / d["cat"]).sort_values("frac_lost", ascending=False))
    print("\nworst 40 stations by fraction of catalogue scenes never obtained:")
    print(worst.head(40).round(3).to_string())


if __name__ == "__main__":
    main()
