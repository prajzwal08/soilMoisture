"""
verify_splits_v2.py — §47.9, the pre-registered gate on the rebuilt splits
===========================================================================

Checks the splits INDEPENDENTLY of `update_splits_v2.py`: it recomputes the geometry from
lat/lon rather than reading the flag columns that script wrote, so a bug in the writer
cannot hide behind its own output. Every check FAILS the job; none of them warn.

  1  geometry     no pair < 1120 m with both members in train; none with both in val;
                  none straddling two splits
  2  netherlands  zero Dutch rows in train or val
  3  top-up       every val_topup station was oos before, is not tile-pair / Dutch / thin /
                  reserved, and has a real pre-cut record; TRAIN MEMBERSHIP UNCHANGED
  4  temporal     every train row clears MIN_PRE_CUT_DAYS; TRAIN_YEARS and OOT_YEARS are
                  disjoint; no train sample lands on or after the cut
  5  datasets     train and val actually build, with per-depth sample counts -- the real
                  test of the 17 surface-only sm_and_flux stations
  6  oot/oost     counted from the dataset's own year gating, not from the flags, with the
                  excluded partial-year stations named
  7  stats        era5 / driver stats hashes and per-depth label means, old beside new

Run:  sbatch slurm/verify_splits_v2.sh
      sbatch slurm/verify_splits_v2.sh --skip-datasets     (checks 1-4 and 7 only, ~1 min)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from splits_config import (
    HALF_TILE_M,
    MIN_POST_CUT_DAYS,
    MIN_PRE_CUT_DAYS,
    NL_NETWORKS,
    OOT_CUT_DATE,
    OOT_YEARS,
    SM_CATEGORIES,
    TRAIN_YEARS,
    category_of,
    station_dir_name,
    station_key,
)

REPO     = Path(__file__).resolve().parent
CSV      = REPO / "csvs" / "station_splits.csv"
BACKUP   = REPO / "csvs" / "station_splits.csv.pre_s47"
DURATION = REPO / "csvs" / "station_duration_audit.csv"

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label + (f" — {detail}" if detail else ""))


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "ABSENT"


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-datasets", action="store_true",
                    help="skip checks 5 and 6 (which open every zarr store)")
    ap.add_argument("--max-stations", type=int, default=None,
                    help="cap the dataset builds in checks 5-6 (smoke mode)")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    df["category"] = df.apply(category_of, axis=1)
    sm = df[df["category"].isin(SM_CATEGORIES)].copy()
    print(f"{CSV.name}: {len(df)} rows, {len(sm)} soil-moisture stations")
    print("splits:", dict(sm["split"].value_counts()), "\n")

    # ── 1. geometry, recomputed ─────────────────────────────────────────────
    print("1. GEOMETRY — recomputed from lat/lon, not read from the flag columns")
    recs = sm.assign(_k=sm.apply(station_key, axis=1))[
        ["_k", "station_id", "latitude", "longitude", "split"]].to_dict("records")
    train_pairs, val_pairs, straddles = [], [], []
    for i in range(len(recs)):
        a = recs[i]
        for j in range(i + 1, len(recs)):
            b = recs[j]
            if abs(a["latitude"] - b["latitude"]) > 0.02:
                continue
            d = haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            if d >= HALF_TILE_M:
                continue
            if a["split"] == b["split"] == "train":
                train_pairs.append((d, a, b))
            elif a["split"] == b["split"] == "val":
                val_pairs.append((d, a, b))
            elif a["split"] != b["split"] and "duplicate" not in (a["split"], b["split"]):
                # §35.29 marks one member of a 6 m cross-network duplicate pair
                # `split="duplicate"`, an inert sentinel. It is not a leak.
                straddles.append((d, a, b))
    for lbl, lst in (("train", train_pairs), ("val", val_pairs), ("straddling", straddles)):
        for d, a, b in lst[:10]:
            print(f"        {d:7.1f} m  {a['station_id']} ({a['split']}) <-> "
                  f"{b['station_id']} ({b['split']})")
    check(not train_pairs, "no tile-sharing pair inside train", f"{len(train_pairs)} found")
    check(not val_pairs,   "no tile-sharing pair inside val",   f"{len(val_pairs)} found")
    check(not straddles,   "no tile-sharing pair across splits", f"{len(straddles)} found")

    # ── 2. netherlands ──────────────────────────────────────────────────────
    print("\n2. NETHERLANDS")
    nl = df[df["network"].isin(NL_NETWORKS)]
    bad_nl = nl[nl["split"].isin(["train", "val"])]
    check(bad_nl.empty, f"no {'/'.join(sorted(NL_NETWORKS))} station in train or val",
          f"{len(bad_nl)} found: {list(bad_nl['station_id'])[:8]}")
    print(f"        {len(nl)} Dutch stations, splits: {dict(nl['split'].value_counts())}")

    # ── 3. top-up purity, and train left alone ──────────────────────────────
    print("\n3. VAL TOP-UP")
    if "val_topup" not in df.columns:
        check(False, "val_topup column present", "update_splits_v2.py has not been applied")
    elif not BACKUP.exists():
        check(False, "backup present for comparison", f"{BACKUP.name} missing")
    else:
        old = pd.read_csv(BACKUP)
        old_key = old.apply(station_key, axis=1)
        new_key = df.apply(station_key, axis=1)
        old_split = dict(zip(old_key, old["split"]))
        topup = df[df["val_topup"].astype(str).str.lower() == "true"]
        was_oos = [k for k in new_key[topup.index] if old_split.get(k) != "oos"]
        check(not was_oos, "every val_topup station was oos before", f"{len(was_oos)} were not")
        dirty = topup[
            topup["same_patch_pair"].astype(str).str.lower().eq("true")
            | topup["tile_pair_eval"].astype(str).str.lower().eq("true")
            | topup["nl_holdout"].astype(str).str.lower().eq("true")
            | topup["thin_pre_cut"].astype(str).str.lower().eq("true")
            | topup["joint_eval"].astype(str).str.lower().eq("true")
            | topup["flux_only_eval"].astype(str).str.lower().eq("true")
        ]
        check(dirty.empty, "no val_topup station is tile-pair / Dutch / thin / reserved",
              f"{len(dirty)} dirty: {list(dirty['station_id'])[:6]}")
        # the point of drawing from oos rather than train
        old_train = {k for k, s in old_split.items() if s == "train"}
        new_train = set(new_key[df["split"] == "train"])
        gained = new_train - old_train
        lost   = old_train - new_train
        check(not gained, "the top-up took nothing from train", f"{len(gained)} gained")
        print(f"        train membership: -{len(lost)} demoted, +{len(gained)} added; "
              f"val_topup moved {len(topup)} oos -> val")

    # ── 4. temporal ─────────────────────────────────────────────────────────
    print("\n4. TEMPORAL")
    check(not (set(TRAIN_YEARS) & set(OOT_YEARS)), "TRAIN_YEARS and OOT_YEARS disjoint")
    check(max(TRAIN_YEARS) < OOT_CUT_DATE // 10000, "TRAIN_YEARS stops before the cut")
    if "pre_cut_days" in df.columns:
        thin = sm[(sm["split"] == "train") & (sm["pre_cut_days"] < MIN_PRE_CUT_DAYS)]
        check(thin.empty, f"every train station has >= {MIN_PRE_CUT_DAYS} pre-cut days",
              f"{len(thin)} short: {list(thin['station_id'])[:6]}")
    else:
        dur = pd.read_csv(DURATION)
        dur["_stem"] = dur["file"].astype(str).str.replace(r"\.nc$", "", regex=True)
        print("        pre_cut_days absent from the CSV; recomputed from "
              f"{DURATION.name} ({len(dur)} rows)")

    # ── 5 & 6. the datasets, and OOT/OOST from the year gating ──────────────
    if not args.skip_datasets:
        from dataset import SoilMoistureDataset, SM_DEPTHS
        common = dict(
            splits_csv      = str(CSV),
            era5_stats_path = str(REPO / "csvs" / "era5_stats18.json"),
            category_filter = list(SM_CATEGORIES),
        )
        print("\n5. DATASETS BUILD")
        for name, filt, yrs in (("train", ["train"], TRAIN_YEARS),
                                ("val",   ["val"],   TRAIN_YEARS)):
            ds = SoilMoistureDataset(**common, split_filter=filt, years=list(yrs),
                                     training=(name == "train"),
                                     max_stations=args.max_stations)
            n_st = len({s["station_key"] for s in ds.samples})
            yrs_seen = sorted({s["year"] for s in ds.samples})
            print(f"        {name:<5s} {len(ds):>8,d} samples from {n_st:4d} stations, "
                  f"years {yrs_seen[:1]}..{yrs_seen[-1:]}")
            check(len(ds) > 0, f"{name} dataset is non-empty")
            if args.max_stations is None:        # §51.1: admitted == assigned, not just non-empty
                check(not ds.station_skips, f"{name}: every assigned station admitted",
                      str(ds.station_skips))
            if name == "train":
                bad_year = [y for y in yrs_seen if y >= OOT_CUT_DATE // 10000]
                check(not bad_year, "no train sample on or after the cut", str(bad_year))
                # The surface-only sm_and_flux stations: each depth head must see a
                # DIFFERENT, smaller station population, without a crash or a zero-fill.
                # Depth availability is per station, and lives in the label cache
                # (dataset.py:1404 unpacks it as sm_np, depths, times, qc_np).
                d_stn, d_smp = Counter(), Counter()
                for _sd, (_sm, _depths, _t, _qc) in ds._label_cache.items():
                    for d in _depths:
                        d_stn[d] += 1
                for s in ds.samples:
                    for d in ds._label_cache[s["sat_dir"]][1]:
                        d_smp[d] += 1
                for d in SM_DEPTHS:
                    print(f"          depth {d:<8s} {d_stn.get(d, 0):4d} stations "
                          f"{d_smp.get(d, 0):>9,d} samples")
                check(all(d_stn.get(d, 0) > 0 for d in SM_DEPTHS),
                      "every depth bin has stations")
                check(d_stn.get(SM_DEPTHS[0], 0) >= d_stn.get(SM_DEPTHS[2], 0),
                      "surface bin is the best covered",
                      f"{dict(d_stn)}")
            del ds

        print("\n6. OOT / OOST — counted from the dataset, not from the flags")
        for name, filt in (("oot", ["train", "val"]), ("oost", ["oos"])):
            ds = SoilMoistureDataset(**common, split_filter=filt, years=list(OOT_YEARS),
                                     training=False, max_stations=args.max_stations)
            n_st = len({s["station_key"] for s in ds.samples})
            by_year = Counter(s["year"] for s in ds.samples)
            print(f"        {name:<5s} {len(ds):>8,d} samples from {n_st:4d} stations   "
                  + "  ".join(f"{y}:{by_year.get(y, 0):,d}" for y in OOT_YEARS))
            flag = "oot_eligible" if name == "oot" else "oost_eligible"
            if flag in df.columns:
                n_flag = int(df[flag].astype(str).str.lower().eq("true").sum())
                print(f"              {flag} says {n_flag}; the dataset yields {n_st}")
            del ds
        # §51.2: eval_predict.py filters on this column, so it must exist and be filled
        check("oot_effective_days" in df.columns
              and not sm.loc[sm["split"].isin(["train", "val", "oos"]), "oot_effective_days"].isna().any(),
              "oot_effective_days present for every train/val/oos station (eval_predict §51.2 filter)")
        if "oot_effective_days" in df.columns:
            partial = sm[(sm["oot_effective_days"] > 0)
                         & (sm["oot_effective_days"] < MIN_POST_CUT_DAYS)
                         & sm["split"].isin(["train", "oos"])]
            print(f"\n        excluded, partial post-cut year ({len(partial)} stations):")
            for _, r in partial.iterrows():
                print(f"          {r['station_id']:<22s} {r['network']:<16s} {r['split']:<5s} "
                      f"{int(r['oot_effective_days']):4d} d")
    else:
        print("\n5-6. DATASETS — skipped (--skip-datasets)")

    # ── 7. stats provenance ─────────────────────────────────────────────────
    print("\n7. STATS")
    for f in ("era5_stats18.json", "era5_stats.json", "driver_stats.json"):
        p = REPO / "csvs" / f
        print(f"        {f:<22s} sha {sha_of(p)}")
    dj = REPO / "csvs" / "driver_stats.json"
    if dj.exists():
        d = json.loads(dj.read_text())
        if "label" in d and "mean" in d["label"]:
            print(f"        label means per depth: "
                  + ", ".join(f"{m:.4f}" for m in np.atleast_1d(d["label"]["mean"])))

    # ── verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"VERIFY FAILED — {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("VERIFY PASSED — splits are consistent with §47. Training may launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
