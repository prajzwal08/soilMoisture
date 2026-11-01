"""
backfill_s2_download.py — §50 phases 1-3 (download, harmonise by baseline, cloud filter)
=========================================================================================
Works ONLY from csvs/s2_backfill_targets.csv (backfill_catalogue.py --report), one catalogue
item per (station, date), and writes ONLY to the backfill staging area:

    /gpfs/scratch1/shared/pkhanal/s2_backfill/{station}/S2L2A/{date}.tif      phase 1-2
    /gpfs/scratch1/shared/pkhanal/s2_backfill_cm/{cat}/{station}/CloudMask/   phase 3 (sensei,
                                                   cloud_masking_inference.py, run by the slurm)
    csvs/s2_backfill_ledger/{station}.csv          one row per target scene, rewritten per run
    text/s2_backfill_manifest.csv                  phase 3 filter verdicts — a NEW file; the
                                                   original cloud-filter CSVs are never touched

--download   Same grid, crop, bands and resampling as download_s2_mpc.py (imported, not copied).
             Differences, each a fix for the §50 failure:
               * the item is re-fetched by id and freshly signed on EVERY attempt, and the
                 planetary_computer token cache is cleared after a failure — the 2026-05-20 loss
                 was an expired SAS token that retries could not refresh;
               * NaN -> 0 before the int16 cast (download_s2_mpc.py:233 cast NaN straight to
                 int16, which is how -32768 / -31768 got into the existing store);
               * every failure is a ledger row; a station with any failure is PARTIAL.
             Resume: a scene with an OK ledger row and its TIF is skipped; failures are retried.
--harmonise  +1000 DN to non-zero pixels iff the item's s2:processing_baseline < 04.00 — the
             convention harmonize_s2_pre2022.py intended, decided by baseline, not by date
             (pre-cut scenes at 04.00/05.10 exist and already carry the offset).
--filter     harmonised TIF + its SEnSeIv2 mask -> keep/reject with filter_cloudy_tiles.py's own
             patch_validity() and TILE_REJECT_THRESH (imported). Rejected TIFs are MOVED to
             s2_backfill/{station}/rejected/, not deleted.

Usage: sbatch slurm/backfill_s2.sh [--stations A B ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from splits_config import ALL_CATEGORIES, category_of, station_dir_name  # noqa: E402

STAGE      = Path("/gpfs/scratch1/shared/pkhanal/s2_backfill")
STAGE_CM   = Path("/gpfs/scratch1/shared/pkhanal/s2_backfill_cm")
TARGETS    = REPO / "csvs" / "s2_backfill_targets.csv"
LEDGER_DIR = REPO / "csvs" / "s2_backfill_ledger"
MANIFEST   = REPO / "text" / "s2_backfill_manifest.csv"          # smoke run (legacy)
MANIFEST_DIR = REPO / "text" / "s2_backfill_manifest"          # one CSV per station
ATTEMPTS   = 4
WAITS      = [5, 20, 60]


def _station_meta():
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    out = {}
    for _, r in df[df["cat"].isin(ALL_CATEGORIES)].iterrows():
        out.setdefault(station_dir_name(r), dict(cat=r["cat"], lat=float(r["latitude"]),
                                                 lon=float(r["longitude"])))
    return out


def _targets(only):
    t = pd.read_csv(TARGETS, dtype={"baseline": str})
    return t[t.station.isin(only)] if only else t


# ── phase 1: download ────────────────────────────────────────────────────────

def _fetch_one(item_id, bounds, epsg):
    import planetary_computer
    import pystac_client
    import stackstac
    from rasterio.enums import Resampling
    from download_s2_mpc import MPC_URL, RES_M, S2_BANDS
    client = pystac_client.Client.open(MPC_URL)
    items = list(client.search(collections=["sentinel-2-l2a"], ids=[item_id]).items())
    if not items:
        raise LookupError(f"item {item_id} not in catalogue")
    it = planetary_computer.sign(items[0])                   # fresh signature, this attempt
    da = stackstac.stack([it], assets=S2_BANDS, epsg=epsg, resolution=RES_M, bounds=bounds,
                         rescale=False, resampling=Resampling.bilinear).squeeze("time")
    return da.compute(), it


def _download_station(args):
    station, rows, meta = args
    import planetary_computer
    from download_s2_mpc import center_crop, save_geotiff, station_grid
    epsg, bounds, _ = station_grid(meta["lat"], meta["lon"])
    out_dir = STAGE / station / "S2L2A"
    out_dir.mkdir(parents=True, exist_ok=True)
    (STAGE_CM / meta["cat"] / station).mkdir(parents=True, exist_ok=True)   # cloud-mask target
    led_p = LEDGER_DIR / f"{station}.csv"
    old = pd.read_csv(led_p, dtype={"baseline": str}) if led_p.exists() else pd.DataFrame()
    ok_before = set(old[old.status == "ok"].date) if len(old) else set()
    recs = []
    for r in rows.itertuples():
        f = out_dir / f"{int(r.date)}.tif"
        if int(r.date) in ok_before and f.exists():
            recs.append(old[old.date == r.date].iloc[-1].to_dict())
            continue
        rec = dict(station=station, date=int(r.date), item_id=r.item_id, baseline=r.baseline,
                   needs_offset=bool(r.needs_offset), status="failed", error="", attempts=0,
                   nodata_frac=np.nan, harmonised=False)
        for a in range(ATTEMPTS):
            rec["attempts"] = a + 1
            try:
                da, it = _fetch_one(r.item_id, bounds, epsg)
                da = center_crop(da)
                v = da.values
                rec["nodata_frac"] = float(np.isnan(v).all(axis=0).mean())
                da = da.fillna(0).clip(-32768, 32767).astype("int16")
                save_geotiff(da, f, epsg, it.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"))
                rec.update(status="ok", error="")
                break
            except Exception as e:                                        # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                try:
                    planetary_computer.sas.TOKEN_CACHE.clear()            # force a new token
                except Exception:                                         # noqa: BLE001
                    pass
                if a < ATTEMPTS - 1:
                    time.sleep(WAITS[min(a, len(WAITS) - 1)])
        recs.append(rec)
    led = pd.DataFrame(recs)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    led.to_csv(led_p, index=False)
    n_ok = int((led.status == "ok").sum())
    return station, len(led), n_ok, len(led) - n_ok


def run_download(only, workers):
    t, meta = _targets(only), _station_meta()
    tasks = [(s, g, meta[s]) for s, g in t.groupby("station")]
    print(f"download: {len(t):,} scenes at {len(tasks)} stations", flush=True)
    tot = [0, 0, 0]
    with Pool(workers) as p:
        for i, (s, n, ok, bad) in enumerate(p.imap_unordered(_download_station, tasks), 1):
            tot[0] += n; tot[1] += ok; tot[2] += bad                      # noqa: E702
            print(f"  [{i}/{len(tasks)}] {s:40s} requested={n:5d} ok={ok:5d} failed={bad:4d}"
                  f"{'  PARTIAL' if bad else ''}", flush=True)
    print(f"download total: requested {tot[0]:,}  ok {tot[1]:,}  FAILED {tot[2]:,}")


# ── phase 2: harmonise by baseline ───────────────────────────────────────────

def _harmonise_station(station):
    import rasterio
    led_p = LEDGER_DIR / f"{station}.csv"
    led = pd.read_csv(led_p, dtype={"baseline": str})
    n = 0
    for i, r in led.iterrows():
        if r.status != "ok" or not bool(r.needs_offset) or bool(r.harmonised):
            continue
        p = STAGE / station / "S2L2A" / f"{int(r.date)}.tif"
        tmp = p.with_suffix(".tmp")
        with rasterio.open(p) as src:
            arr, prof = src.read().astype(np.int32), src.profile.copy()
            tags = src.tags()
        if (arr < 0).any():
            raise ValueError(f"{p}: negative DN before harmonisation — NaN leaked through")
        arr[arr != 0] += 1000
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(arr.astype(np.int16))
            dst.update_tags(**tags)
        tmp.replace(p)
        led.loc[i, "harmonised"] = True                  # ledger is the idempotency record
        n += 1
    led.to_csv(led_p, index=False)
    return station, n


def run_harmonise(only, workers):
    stations = sorted(_targets(only).station.unique())
    with Pool(workers) as p:
        res = p.map(_harmonise_station, stations)
    print(f"harmonise: +1000 applied to {sum(n for _, n in res):,} scenes (pb < 04.00 only)")


# ── phase 3b: cloud filter on the backfill set ───────────────────────────────

def _filter_station(args):
    station, cat = args
    import rasterio
    from filter_cloudy_tiles import TILE_REJECT_THRESH, patch_validity
    led = pd.read_csv(LEDGER_DIR / f"{station}.csv", dtype={"baseline": str})
    rows = []
    for r in led[led.status == "ok"].itertuples():
        s2 = STAGE / station / "S2L2A" / f"{int(r.date)}.tif"
        cm = STAGE_CM / cat / station / "CloudMask" / f"{int(r.date)}.tif"
        if not cm.exists():
            rows.append(dict(station=station, date=int(r.date), masked_frac=np.nan,
                             verdict="NO_CLOUDMASK"))
            continue
        with rasterio.open(cm) as src:
            mf = float(1.0 - patch_validity(src.read()).mean())
        keep = mf <= TILE_REJECT_THRESH
        if not keep and s2.exists():
            rej = STAGE / station / "rejected"
            rej.mkdir(exist_ok=True)
            s2.replace(rej / s2.name)
        rows.append(dict(station=station, date=int(r.date), masked_frac=round(mf, 4),
                         verdict="keep" if keep else "reject"))
    return rows


def run_filter(only, workers):
    meta = _station_meta()
    stations = sorted(_targets(only).station.unique())
    with Pool(workers) as p:
        out = [r for rows in p.map(_filter_station, [(s, meta[s]["cat"]) for s in stations])
               for r in rows]
    df = pd.DataFrame(out)
    # One file per station: concurrent shard jobs each rewrote one shared manifest, and the last
    # writer would silently erase the others' verdicts (-> "nothing kept" at merge).
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for s, g in df.groupby("station"):
        g.to_csv(MANIFEST_DIR / f"{s}.csv", index=False)
    print(f"filter: {df.verdict.value_counts().to_dict()}  -> {MANIFEST_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true")
    g.add_argument("--harmonise", action="store_true")
    g.add_argument("--filter", action="store_true")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--stations-file", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    st = list(a.stations or [])
    if a.stations_file:
        st += Path(a.stations_file).read_text().split()
    only = set(st) if st else None
    (run_download if a.download else run_harmonise if a.harmonise else run_filter)(only, a.workers)


if __name__ == "__main__":
    main()
