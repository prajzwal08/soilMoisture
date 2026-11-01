"""
backfill_catalogue.py — §50 phase 0: the catalogue, the existing-store audit, the target list
==============================================================================================
Three steps, two conda envs (soilmoisture has pystac but no zarr; terramind the reverse):

  --catalogue   (soilmoisture)  one MPC search per station over its own [start, end] window
                                -> csvs/s2_catalogue_items.csv
                                   station, date, item_id, mgrs, baseline, cloud, n_items_date
  --store       (terramind)     per S2 scene in satellite_zarr: per-band median of non-zero DN,
                                zero fraction, min value, count of values < 0
                                -> csvs/s2_store_scene_stats.csv
  --report      (terramind)     0a answers + 0c target list
                                -> csvs/s2_backfill_targets.csv

0a asks two questions of the EXISTING store, because the backfill must reproduce its true
convention (or both get fixed together):
  (i)  harmonise_s2_pre2022.py added +1000 DN by DATE (< 2022-01-25). Scenes before the cut
       whose catalogue baseline is already >= 04.00 carry the offset natively, so they would
       be DOUBLE offset. Test: their per-band median vs the same station's post-cut scenes.
  (ii) download_s2_mpc.py cast NaN-filled float to int16 without fillna(0). Is nodata 0, or a
       garbage value (-32768)?

Read-only on every store. Usage: sbatch slurm/backfill_phase0.sh [--stations A B ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from splits_config import ALL_CATEGORIES, category_of, station_dir_name  # noqa: E402

RAW_ROOT   = Path("/projects/prjs1968/satellite_zarr")
CAT_CSV    = REPO / "csvs" / "s2_catalogue_items.csv"
STORE_CSV  = REPO / "csvs" / "s2_store_scene_stats.csv"
TARGET_CSV = REPO / "csvs" / "s2_backfill_targets.csv"
DELETE_LOG = REPO / "text" / "cloudy_tile_manifest_delete_log.csv"
HARMONISE_CUT = 20220125
MAX_CLOUD = 75
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def _stations(only):
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    out, seen = [], set()
    for _, r in df[df["cat"].isin(ALL_CATEGORIES)].iterrows():
        n = station_dir_name(r)
        if n in seen or (only and n not in only):
            continue
        seen.add(n)
        out.append(dict(name=n, split=r["split"], cat=r["cat"], lat=float(r["latitude"]),
                        lon=float(r["longitude"]), start=int(r["start_date"]),
                        end=int(r["end_date"])))
    return out


# ── catalogue (soilmoisture) ─────────────────────────────────────────────────

def _catalogue_one(t):
    import planetary_computer  # noqa: F401
    import pystac_client
    from download_s2_mpc import MPC_URL, station_grid
    _, _, bbox = station_grid(t["lat"], t["lon"])
    lo = max(t["start"], 20160101)
    win = f"{str(lo)[:4]}-{str(lo)[4:6]}-{str(lo)[6:]}/{str(t['end'])[:4]}-{str(t['end'])[4:6]}-{str(t['end'])[6:]}"
    for attempt in range(4):
        try:
            cat = pystac_client.Client.open(MPC_URL)
            items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=win,
                                    query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items())
            break
        except Exception:                                   # noqa: BLE001
            time.sleep(10 * (attempt + 1))
    else:
        return [dict(station=t["name"], date=-1, item_id="CATALOGUE_FAILED")]
    rows = [dict(station=t["name"], date=int(i.datetime.strftime("%Y%m%d")), item_id=i.id,
                 mgrs=i.properties.get("s2:mgrs_tile", ""),
                 baseline=str(i.properties.get("s2:processing_baseline", "")),
                 cloud=float(i.properties.get("eo:cloud_cover", np.nan)),
                 boa_offset=i.properties.get("earthsearch:boa_offset_applied", ""))
            for i in items]
    return rows


def run_catalogue(tasks):
    out = []
    with Pool(16) as p:
        for i, rows in enumerate(p.imap_unordered(_catalogue_one, tasks), 1):
            out.extend(rows)
            if i % 50 == 0 or i == len(tasks):
                print(f"  catalogue {i}/{len(tasks)}", flush=True)
    df = pd.DataFrame(out)
    df["n_items_date"] = df.groupby(["station", "date"])["item_id"].transform("count")
    df.to_csv(CAT_CSV, index=False)
    print(f"-> {CAT_CSV}  {len(df):,} items, {df.station.nunique()} stations, "
          f"failed: {int((df.item_id == 'CATALOGUE_FAILED').sum())}")


# ── store stats (terramind) ──────────────────────────────────────────────────

def _store_one(name):
    import zarr
    p = RAW_ROOT / f"{name}.zarr"
    rows = []
    try:
        rg = zarr.open_group(str(p), mode="r")
        if "s2/data" not in rg:
            return rows
        dates = [int(bytes(d).decode()[:8]) for d in rg["s2/dates"][:]]
        arr = rg["s2/data"]
        for i, d in enumerate(dates):
            x = np.asarray(arr[i])                                  # (12,224,224) int16
            nz = x != 0
            med = [float(np.median(x[b][nz[b]])) if nz[b].any() else np.nan for b in range(12)]
            # 1st percentile of positive DN: a harmonised scene bottoms out near the +1000
            # offset (dark water / shadow), a double-offset one near +2000. Needs no
            # same-station reference, unlike the median comparison.
            pos = x > 0
            p1 = [float(np.percentile(x[b][pos[b]], 1)) if pos[b].sum() > 100 else np.nan
                  for b in range(12)]
            rows.append(dict(station=name, date=d, idx=i, min=int(x.min()),
                             n_neg=int((x < 0).sum()), zero_frac=float((~nz).mean()),
                             **{f"med_{BANDS[b]}": med[b] for b in range(12)},
                             **{f"p1_{BANDS[b]}": p1[b] for b in range(12)}))
    except Exception as e:                                          # noqa: BLE001
        rows.append(dict(station=name, date=-1, idx=-1, min=0, n_neg=0, zero_frac=np.nan,
                         error=f"{type(e).__name__}:{e}"[:200]))
    return rows


def run_store(tasks):
    out = []
    with Pool(64) as p:
        for i, rows in enumerate(p.imap_unordered(_store_one, [t["name"] for t in tasks]), 1):
            out.extend(rows)
            if i % 50 == 0 or i == len(tasks):
                print(f"  store {i}/{len(tasks)}", flush=True)
    pd.DataFrame(out).to_csv(STORE_CSV, index=False)
    print(f"-> {STORE_CSV}  {len(out):,} scenes")


# ── report + target list ─────────────────────────────────────────────────────

def _bnum(b):
    try:
        return float(b)
    except (TypeError, ValueError):
        return np.nan


def run_report(tasks):
    cat = pd.read_csv(CAT_CSV)
    cat = cat[cat.item_id != "CATALOGUE_FAILED"].copy()
    cat["bnum"] = cat.baseline.map(_bnum)
    st = pd.read_csv(STORE_CSV)
    st = st[st.date > 0]
    names = {t["name"] for t in tasks}
    cat, st = cat[cat.station.isin(names)], st[st.station.isin(names)]

    # each station's dominant MGRS tile = the tile of the catalogue items matching its stored dates
    m = st[["station", "date"]].merge(cat[["station", "date", "mgrs"]], on=["station", "date"])
    dom = m.groupby("station").mgrs.agg(lambda s: s.value_counts().index[0]).to_dict()

    # ── (ii) nodata convention ──
    print("\n=== 0a(ii) NODATA CONVENTION in the existing store")
    print(f"  scenes {len(st):,};  any value < 0: {int((st.n_neg > 0).sum()):,}  "
          f"min value histogram: {st['min'].clip(upper=0).value_counts().head(6).to_dict()}")
    print(f"  zero fraction > 0.01 in {int((st.zero_frac > 0.01).sum()):,} scenes "
          f"(nodata written as 0 where present)")

    # ── (i) double offset ──
    b_one = (cat.sort_values("bnum").drop_duplicates(["station", "date"], keep="first")
             [["station", "date", "bnum"]])
    s = st.merge(b_one, on=["station", "date"], how="left")
    s["pre"] = s.date < HARMONISE_CUT
    s["grp"] = np.select([s.pre & (s.bnum < 4.0), s.pre & (s.bnum >= 4.0), ~s.pre],
                         ["pre_cut_pb<04 (harmonised, correct)", "pre_cut_pb>=04 (SUSPECT)",
                          "post_cut (native offset)"], "pre_cut_pb_unknown")
    ref = s[~s.pre].groupby("station")[["med_B04", "med_B08", "med_B11"]].median()
    s = s.join(ref, on="station", rsuffix="_ref")
    for b in ("B04", "B08", "B11"):
        s[f"d_{b}"] = s[f"med_{b}"] - s[f"med_{b}_ref"]
    print("\n=== 0a(i) DOUBLE OFFSET: per-scene median DN minus the station's post-cut median")
    print(s.groupby("grp")[["d_B04", "d_B08", "d_B11"]].median().round(0).to_string())
    print("  scene counts:", s.grp.value_counts().to_dict())
    s["p1_dark"] = s[["p1_B02", "p1_B03", "p1_B04"]].min(axis=1)
    print("\n  1st-percentile positive DN, darkest visible band (harmonised ~1000, double ~2000):")
    print(s.groupby("grp")["p1_dark"].describe(percentiles=[.05, .5, .95]).round(0).to_string())
    print(f"  scenes with p1_dark >= 1800 (candidate double offset), by group: "
          f"{s[s.p1_dark >= 1800].grp.value_counts().to_dict()}")
    print(f"  scenes with p1_dark < 900 (candidate MISSING offset), by group: "
          f"{s[s.p1_dark < 900].grp.value_counts().to_dict()}")
    sus = s[s.grp.str.startswith("pre_cut_pb>=04")]
    print(f"  SUSPECT scenes: {len(sus):,} at {sus.station.nunique()} stations; "
          f"baselines {sus.bnum.value_counts().head(6).to_dict()}")
    s.to_csv(STORE_CSV.with_name("s2_store_offset_check.csv"), index=False)

    # ── 0c targets ──
    dl = pd.read_csv(DELETE_LOG, usecols=["station", "date"]) if DELETE_LOG.exists() else pd.DataFrame(columns=["station", "date"])
    have = set(zip(st.station, st.date)) | set(zip(dl.station, dl.date.astype(int)))
    c = cat.copy()
    c["dom"] = c.station.map(dom)
    c = c[~c.apply(lambda r: (r.station, int(r.date)) in have, axis=1)]
    # one item per (station, date): the station's own tile first, then lowest cloud
    c["own_tile"] = (c.mgrs == c.dom).astype(int)
    c = (c.sort_values(["station", "date", "own_tile", "cloud"], ascending=[True, True, False, True])
          .drop_duplicates(["station", "date"], keep="first"))
    c["needs_offset"] = c.bnum < 4.0
    c[["station", "date", "item_id", "mgrs", "own_tile", "baseline", "needs_offset", "cloud"]] \
        .to_csv(TARGET_CSV, index=False)
    print(f"\n=== 0c TARGETS -> {TARGET_CSV}: {len(c):,} scenes at {c.station.nunique()} stations; "
          f"needs +1000 (pb<04): {int(c.needs_offset.sum()):,}; off own tile: "
          f"{int((c.own_tile == 0).sum()):,}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--catalogue", action="store_true")
    g.add_argument("--store", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--stations", nargs="*", default=None)
    a = ap.parse_args()
    tasks = _stations(set(a.stations) if a.stations else None)
    print(f"stations: {len(tasks)}", flush=True)
    (run_catalogue if a.catalogue else run_store if a.store else run_report)(tasks)


if __name__ == "__main__":
    main()
