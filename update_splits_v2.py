"""
update_splits_v2.py — §47: all soil moisture in, tiles and the Netherlands out
==============================================================================

Extends §35.29's `update_splits_tile_pairs.py`, which solved one third of this problem:
it demoted every *train* station sharing a tile with another soil-moisture station. Three
things were left, and §46 cannot launch on top of them.

WHAT THIS CHANGES

  RULE 1  tile-sharing            d < 1120 m from another SM station   train/val -> oos
  RULE 2  Netherlands             network in NL_NETWORKS               train/val -> oos
  RULE 3  no pre-cut record       < 365 label days before the cut      train/val -> oos

  then    VAL TOP-UP              34-ish location groups drawn from OOS back into val,
                                  stratified by kg_macro x igbp_macro, whole groups only

  then    ELIGIBILITY             oot_eligible / oost_eligible regenerated from MEASURED
                                  label + ERA5 + S2 coverage, not from `end_date`

Rule 1 is unchanged in definition from §35.29 and, run today, moves NO train station —
that sweep already did it. What it catches now is 18 val stations that share tiles with
each other: 11 FMI probes at three Sodankyla locations and 7 TxSON probes inside one tile.
They were never training data, but they were duplicated sites inside the set that drives
early stopping and the LR schedule.

THE TOP-UP COMES OUT OF OOS, NOT OUT OF TRAIN. val and oos are both station-disjoint from
train, so reassigning between them leaks nothing — it trades test power for model-selection
power and leaves the training set exactly as costed in §47.6. Drawing the replacement from
train would have cost ~180 station-years of training data and broken comparability with
every previous run.

ORDER MATTERS: demote first, then draw. Otherwise a station demoted for being Dutch or for
sharing an FMI patch becomes a candidate for its own replacement. The eligibility mask is
what enforces that; `verify_splits_v2.py` re-checks it independently rather than trusting
this script.

Nothing is deleted. Demotion is to `oos`, which is a real evaluation split; the
`split="duplicate"` sentinel from §35.29 is left exactly as it was.

Usage:  sbatch slurm/update_splits_v2.sh            (dry run, reports only)
        sbatch slurm/update_splits_v2.sh --apply    (writes, backup at .pre_s47)
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from splits_config import (
    DUP_M,
    HALF_TILE_M,
    MIN_CELL_SIZE,
    MIN_POST_CUT_DAYS,
    MIN_PRE_CUT_DAYS,
    NL_NETWORKS,
    OOT_CUT_DATE,
    OOT_YEARS,
    PATCH_M,
    RANDOM_SEED,
    TRAIN_YEARS,
    VAL_FRACTION,
    category_of,
    station_dir_name,
    station_key,
)

REPO      = Path(__file__).resolve().parent
CSV       = REPO / "csvs" / "station_splits.csv"
DURATION  = REPO / "csvs" / "station_duration_audit.csv"
COVERAGE  = REPO / "csvs" / "dataset_coverage.csv"
ERA5_QC   = REPO / "csvs" / "era5_all_station_qc.csv"
BACKUP    = REPO / "csvs" / "station_splits.csv.pre_s47"

TRAIN_START = TRAIN_YEARS[0] * 10000 + 101      # 20160101
TRAIN_END   = TRAIN_YEARS[-1] * 10000 + 1231    # 20221231
OOT_END     = OOT_YEARS[-1] * 10000 + 1231      # 20251231


# ─────────────────────────────────────────────────────────────────────────────
# dates: YYYYMMDD ints in, day counts out. No timezone, no calendar library.
# ─────────────────────────────────────────────────────────────────────────────
def to_ordinal(yyyymmdd) -> int:
    import datetime as _dt
    s = int(yyyymmdd)
    return _dt.date(s // 10000, (s // 100) % 100, s % 100).toordinal()


def overlap_days(start, end, win_start, win_end) -> int:
    """Inclusive day count of [start,end] ∩ [win_start,win_end]."""
    if pd.isna(start) or pd.isna(end):
        return 0
    lo = max(to_ordinal(start), to_ordinal(win_start))
    hi = min(to_ordinal(end), to_ordinal(win_end))
    return max(0, hi - lo + 1)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres. Equirectangular would do at these scales, but
    haversine costs nothing and does not degrade at FMI's 67 N, where cos(lat) is 0.39."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# coverage: how much record a station actually has either side of the cut
# ─────────────────────────────────────────────────────────────────────────────
def attach_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Adds pre_cut_days, post_cut_days, oot_effective_days.

    `post_cut_days` is the LABEL record past the cut. `oot_effective_days` additionally
    requires the model's inputs to reach that far: §22.2 measured `oost_eligible` claiming
    128 stations where only 98 were real, because the flag checked `end_date` and nothing
    else while ERA5 and S2 stopped earlier.
    """
    dur = pd.read_csv(DURATION)
    dur["_stem"] = dur["file"].astype(str).str.replace(r"\.nc$", "", regex=True)
    dur = dur.set_index("_stem")

    cov = pd.read_csv(COVERAGE).set_index("station") if COVERAGE.exists() else None
    era = pd.read_csv(ERA5_QC).set_index("station") if ERA5_QC.exists() else None

    stem = df.apply(station_dir_name, axis=1)
    missing = sorted(set(stem) - set(dur.index))
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} stations absent from {DURATION.name}; "
            f"coverage cannot be measured. First few: {missing[:5]}"
        )

    rec_start = stem.map(dur["start_date"])
    rec_end   = stem.map(dur["end_date"])

    # ERA5 runs contiguously from max(record start, 2016-01-01); n_days therefore pins its
    # end. A shortfall against the record means ERA5 truncates early.
    if era is not None:
        era_n = stem.map(era["n_days"])
        era_start = np.maximum(rec_start.map(to_ordinal), to_ordinal(TRAIN_START))
        era_end_ord = era_start + era_n.fillna(0).astype(int) - 1
    else:
        era_end_ord = rec_end.map(to_ordinal)

    s2_end_ord = (
        stem.map(cov["s2_end"]).map(lambda v: to_ordinal(v) if pd.notna(v) else np.nan)
        if cov is not None else pd.Series(np.nan, index=df.index)
    )

    eff_end_ord = pd.concat(
        [rec_end.map(to_ordinal), pd.Series(era_end_ord, index=df.index), s2_end_ord],
        axis=1,
    ).min(axis=1, skipna=True)

    cut_ord, oot_end_ord = to_ordinal(OOT_CUT_DATE), to_ordinal(OOT_END)

    out = df.copy()
    out["rec_start"] = rec_start.values
    out["rec_end"]   = rec_end.values
    out["pre_cut_days"]  = [
        overlap_days(s, e, TRAIN_START, TRAIN_END) for s, e in zip(rec_start, rec_end)
    ]
    out["post_cut_days"] = [
        overlap_days(s, e, OOT_CUT_DATE, OOT_END) for s, e in zip(rec_start, rec_end)
    ]
    out["oot_effective_days"] = [
        max(0, min(int(ee), oot_end_ord) - cut_ord + 1) if pd.notna(ee) else 0
        for ee in eff_end_ord
    ]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# rule 1: the geometry sweep
# ─────────────────────────────────────────────────────────────────────────────
def tile_sweep(sm: pd.DataFrame):
    """Global O(n^2) sweep over soil-moisture stations.

    Keyed on `station_key`, not `station_id`: the latter is not unique across source
    networks, which is the latent defect in §35.29's columns (§47.10).
    """
    recs = sm.assign(_key=sm.apply(station_key, axis=1))[
        ["_key", "station_id", "network", "source_network",
         "latitude", "longitude", "split"]
    ].to_dict("records")

    same_patch, tile_pair, dups, straddle = set(), set(), [], []
    n_pairs = 0
    for i in range(len(recs)):
        a = recs[i]
        for j in range(i + 1, len(recs)):
            b = recs[j]
            if abs(a["latitude"] - b["latitude"]) > 0.02:   # 1 deg lat ~ 111 km
                continue
            d = haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            n_pairs += 1
            if d >= HALF_TILE_M:
                continue
            if d < PATCH_M:
                same_patch.add(a["_key"]); same_patch.add(b["_key"])
            else:
                tile_pair.add(a["_key"]); tile_pair.add(b["_key"])
            if d < DUP_M and a["source_network"] != b["source_network"]:
                dups.append((d, a, b))
            if a["split"] != b["split"] and "duplicate" not in (a["split"], b["split"]):
                straddle.append((d, a, b))
    return same_patch, tile_pair, dups, straddle, n_pairs


# ─────────────────────────────────────────────────────────────────────────────
# the val top-up
# ─────────────────────────────────────────────────────────────────────────────
def draw_val_topup(df: pd.DataFrame, eligible: pd.Series, is_sm: pd.Series,
                   n_target: int, rng: np.random.Generator) -> set:
    """Draw whole location groups from the eligible OOS pool until val reaches n_target.

    Mirrors `create_evaluation_splits.py:195-225` — stratify by kg_macro x igbp_macro,
    proportional per cell, whole groups only. That routine SKIPS cells below MIN_CELL_SIZE,
    which is right when carving a fraction out of a large pool and wrong here, where a
    skipped cell just leaves the target unmet. So a deterministic second pass fills the
    remainder from whatever eligible groups are left, and reports how many it took.
    """
    # Count val the same way the target is computed: soil-moisture stations only. The 15
    # flux_only val rows can never be trained or validated on, so including them would
    # under-draw the top-up by exactly their count.
    n_val_now = int((is_sm & (df["split"] == "val")).sum())
    need = max(0, n_target - n_val_now)
    if need == 0:
        return set()

    # Whole groups, and a group is only a candidate if EVERY one of its soil-moisture
    # members is eligible and currently oos. Building the group membership from the
    # eligible rows alone silently splits a group whose second member is reserved or
    # tile-sharing, which is the one thing this draw must never do.
    sm_oos = df[is_sm & (df["split"] == "oos")]
    members = {gid: idx.tolist()
               for gid, idx in sm_oos.groupby("location_group_id").groups.items()}
    elig_idx = set(df.index[eligible])
    groups = {gid: idxs for gid, idxs in members.items()
              if all(i in elig_idx for i in idxs)}
    if not groups:
        return set()
    n_partial = len(members) - len(groups)
    if n_partial:
        print(f"  {n_partial} oos group(s) skipped: not every soil-moisture member is eligible")

    pool = df.loc[[i for idxs in groups.values() for i in idxs]].copy()
    rep = {gid: idxs[0] for gid, idxs in groups.items()}
    rep_df = pool.loc[list(rep.values())].copy()

    chosen_groups, taken = [], 0
    for (kg, igbp), cell in rep_df.groupby(["kg_macro", "igbp_macro"], dropna=False):
        if len(cell) < MIN_CELL_SIZE:
            continue
        share = need * len(cell) / len(rep_df)
        n_take = min(len(cell), max(1, int(round(share))))
        picks = rng.choice(cell.index.tolist(), size=n_take, replace=False)
        for r in picks:
            gid = pool.loc[r, "location_group_id"]
            if gid in chosen_groups:
                continue
            chosen_groups.append(gid)
            taken += len(groups[gid])
        if taken >= need:
            break

    n_stratified = len(chosen_groups)
    if taken < need:
        rest = [g for g in groups if g not in chosen_groups]
        for gid in rng.permutation(rest):
            chosen_groups.append(gid)
            taken += len(groups[gid])
            if taken >= need:
                break
    if len(chosen_groups) > n_stratified:
        print(f"  note: stratified pass supplied {n_stratified} groups; "
              f"{len(chosen_groups) - n_stratified} more drawn from the remainder "
              f"to reach the target of {n_target}")

    out = set()
    for gid in chosen_groups:
        out.update(groups[gid])
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the CSV. Without it this is a dry run that only reports.")
    ap.add_argument("--val-target", type=int, default=None,
                    help="target val station count (default: VAL_FRACTION of the SM inventory)")
    args = ap.parse_args()

    if not CSV.exists():
        print(f"FATAL: {CSV} not found (run from the repo root)", file=sys.stderr)
        return 2

    df = pd.read_csv(CSV)
    print(f"read {CSV.name}: {len(df)} rows")

    df["category"] = df.apply(category_of, axis=1)
    is_sm = df["category"].isin(["sm_only", "sm_and_flux"])
    print(f"categories: " + ", ".join(
        f"{k} {v}" for k, v in df['category'].value_counts().items()))
    print(f"soil-moisture stations: {int(is_sm.sum())}")
    print("split counts before:", dict(df.loc[is_sm, "split"].value_counts()), "\n")

    df = attach_coverage(df)

    # ── rule 1 ──────────────────────────────────────────────────────────────
    sm = df[is_sm]
    same_patch, tile_pair, dups, straddle, n_pairs = tile_sweep(sm)
    keys = df.apply(station_key, axis=1)
    print(f"=== RULE 1 — tile geometry ({n_pairs} pairs evaluated after latitude reject) ===")
    print(f"  same-patch  (< {PATCH_M:.0f} m)        : {len(same_patch)} stations")
    print(f"  tile-pair   ({PATCH_M:.0f}-{HALF_TILE_M:.0f} m)      : {len(tile_pair)} stations")
    for d, a, b in straddle:
        print(f"  *** SPLIT STRADDLE: {a['station_id']} ({a['split']}) and "
              f"{b['station_id']} ({b['split']}) are {d:.0f} m apart ***")
    if dups:
        print(f"  cross-network duplicates < {DUP_M:.0f} m: "
              + ", ".join(f"{a['station_id']}/{b['station_id']} ({d:.1f} m)"
                          for d, a, b in dups))

    df["same_patch_pair"] = keys.isin(same_patch)
    df["tile_pair_eval"]  = keys.isin(tile_pair)

    # ── rules 2 and 3 ───────────────────────────────────────────────────────
    df["nl_holdout"]   = df["network"].isin(NL_NETWORKS)
    df["thin_pre_cut"] = df["pre_cut_days"] < MIN_PRE_CUT_DAYS

    print(f"\n=== RULE 2 — Netherlands ({', '.join(sorted(NL_NETWORKS))}) ===")
    print(f"  {int(df['nl_holdout'].sum())} stations, "
          f"{int((df['nl_holdout'] & df['split'].isin(['train','val'])).sum())} of them in train/val")
    print(f"\n=== RULE 3 — under {MIN_PRE_CUT_DAYS} label days before {OOT_CUT_DATE} ===")
    thin_in = df["thin_pre_cut"] & df["split"].isin(["train", "val"]) & is_sm
    for _, r in df[thin_in].iterrows():
        print(f"  {r['station_id']:<22s} {r['network']:<16s} {r['split']:<5s} "
              f"pre-cut {int(r['pre_cut_days']):4d} d   post-cut {int(r['post_cut_days']):4d} d")

    # ── demote ──────────────────────────────────────────────────────────────
    demote = (
        is_sm
        & df["split"].isin(["train", "val"])
        & (df["same_patch_pair"] | df["tile_pair_eval"] | df["nl_holdout"] | df["thin_pre_cut"])
    )
    print(f"\n=== DEMOTING {int(demote.sum())} STATIONS TO oos ===")
    for _, r in df[demote].iterrows():
        why = ("same-patch" if r["same_patch_pair"] else
               "tile-pair"  if r["tile_pair_eval"]  else
               "netherlands" if r["nl_holdout"]     else "thin-pre-cut")
        print(f"  {r['station_id']:<22s} {r['network']:<16s} {r['split']:>5s} -> oos   ({why})")
    df.loc[demote, "split"] = "oos"

    # ── val top-up, from OOS ────────────────────────────────────────────────
    n_target = args.val_target or int(round(is_sm.sum() * VAL_FRACTION))
    for col in ("joint_eval", "flux_only_eval"):
        if col not in df.columns:
            df[col] = False
    eligible = (
        is_sm
        & (df["split"] == "oos")
        & ~df["same_patch_pair"] & ~df["tile_pair_eval"]
        & ~df["nl_holdout"] & ~df["thin_pre_cut"]
        & ~df["joint_eval"].astype(str).str.lower().eq("true")
        & ~df["flux_only_eval"].astype(str).str.lower().eq("true")
    )
    print(f"\n=== VAL TOP-UP — target {n_target}, drawn FROM oos ===")
    print(f"  val after demotions : {int((is_sm & (df['split'] == 'val')).sum())} "
          f"soil-moisture stations")
    print(f"  eligible oos pool   : {int(eligible.sum())} stations, "
          f"{df.loc[eligible, 'location_group_id'].nunique()} location groups")

    rng = np.random.default_rng(RANDOM_SEED)
    topup = draw_val_topup(df, eligible, is_sm, n_target, rng)
    df["val_topup"] = df.index.isin(topup)
    df.loc[df["val_topup"], "split"] = "val"
    print(f"  moved oos -> val    : {len(topup)} stations in "
          f"{df.loc[df['val_topup'], 'location_group_id'].nunique()} groups")

    # whole groups only, as create_evaluation_splits.py:412-414 asserts
    for gid, g in df[is_sm & (df["split"] != "duplicate")].groupby("location_group_id"):
        if g["split"].nunique() != 1:
            raise SystemExit(
                f"FATAL: location group {gid} straddles splits after the top-up: "
                f"{dict(g['split'].value_counts())}"
            )

    # ── eligibility, from measured coverage ─────────────────────────────────
    usable_post = df["oot_effective_days"] >= MIN_POST_CUT_DAYS
    df["oot_eligible"]  = is_sm & (df["split"] == "train") & usable_post
    df["oost_eligible"] = is_sm & (df["split"] == "oos")   & usable_post
    partial = (
        is_sm & df["split"].isin(["train", "oos"])
        & (df["oot_effective_days"] > 0) & ~usable_post
    )
    print(f"\n=== OOT / OOST ELIGIBILITY — measured, not from end_date ===")
    print(f"  oot_eligible  (train, >= {MIN_POST_CUT_DAYS} d past the cut): "
          f"{int(df['oot_eligible'].sum())}")
    print(f"  oost_eligible (oos,   >= {MIN_POST_CUT_DAYS} d past the cut): "
          f"{int(df['oost_eligible'].sum())}")
    print(f"  excluded, partial year (1-{MIN_POST_CUT_DAYS - 1} d): {int(partial.sum())} "
          f"— reported, not deleted")

    # ── report ──────────────────────────────────────────────────────────────
    after = df[is_sm]
    print("\n=== AFTER ===")
    print("split counts:", dict(after["split"].value_counts()))
    for s in ("train", "val", "oos"):
        g = after[after["split"] == s]
        print(f"  {s:<6s} {len(g):4d} stn   "
              f"{g['pre_cut_days'].sum() / 365.25:8.1f} st-yr {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}   "
              f"{g['post_cut_days'].sum() / 365.25:7.1f} st-yr {OOT_YEARS[0]}-{OOT_YEARS[-1]}")

    df = df.drop(columns=["category", "rec_start", "rec_end"])

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    shutil.copy2(CSV, BACKUP)
    df.to_csv(CSV, index=False)
    print(f"\nwrote {CSV}  (backup: {BACKUP})")
    print("REMINDER: csvs/era5_stats18.json and csvs/driver_stats.json are fitted on the "
          "train split and MUST be regenerated now (§47.8 item 3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
