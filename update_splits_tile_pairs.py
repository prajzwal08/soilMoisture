"""
update_splits_tile_pairs.py — hold tile-sharing stations out of training (§35.29)
=================================================================================

Each station gets its OWN 2240 m tile centred on itself, so "two stations share a
tile" means: station B falls inside station A's tile. That splits into two cases
with different consequences, and they must not be treated alike.

  SAME PATCH (< 160 m)
      Both stations land in the SAME TerraMind token. Training on both feeds the
      model two different labels for one input — contradictory supervision, and it
      silently double-weights that site. Their disagreement is also the single most
      useful number available for interpreting §35.10: it is the IRREDUCIBLE NOISE
      FLOOR of any 160 m prediction. If two sensors 100 m apart differ by
      0.04 m3/m3, no model at this resolution can beat 0.04.

  DIFFERENT PATCH, SAME TILE (160 - 1120 m)
      The only DIRECT test of §34's hypothesis that exists in the data. Predict
      station A's tile, read the 160 m patch containing station B, compare against
      B's observation. `diag/patch_map_sd` shows the map is not constant; it cannot
      show the pattern is CORRECT. These pairs can.

      Measured: 26 usable pairs over 25 stations, almost entirely TxSON (19) and
      FMI (6). Low power — treat as a qualitative sign test, not a precise metric.

  > 1120 m
      Outside each other's tile. Ordinary independent stations, nothing to do.

WHAT THIS SCRIPT CHANGES

  1. Adds `same_patch_pair`  (bool)  — station shares a 160 m patch with another
  2. Adds `tile_pair_eval`   (bool)  — station is usable for the within-tile test
  3. Adds `duplicate_of`     (str)   — set on cross-network duplicate records
  4. Moves the tile-pair and same-patch stations OUT of `train`
  5. Marks exact duplicates with split="duplicate" so every existing
     `split_filter` excludes them WITHOUT deleting the inventory row

Nothing is deleted. `split="duplicate"` is inert to dataset.py, which filters with
`splits["split"].isin(split_filter)`, so an unrecognised value is simply never
selected — and the row stays auditable and the change stays reversible.

THE DUPLICATE SWEEP is deliberately global, not restricted to `location_group_id`.
The 6 m VairaRanch / US-Var pair (ISMN's FLUXNET mirror vs AmeriFlux direct) was
found inside a group; nothing guaranteed the grouping caught every such case, and a
duplicated training station quietly doubles that site's weight in the loss.

Usage:  sbatch slurm/update_splits.sh        (--apply to write; default is a dry run)
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

CSV = Path("csvs/station_splits.csv")

TILE_M       = 2240.0
PATCH_M      = 160.0
HALF_TILE_M  = TILE_M / 2.0      # 1120 m: beyond this, B is outside A's tile
DUP_M        = 50.0              # closer than this across networks = same physical site


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Equirectangular is fine at these scales, but
    haversine costs nothing here and does not degrade at high latitude — FMI's SOD
    stations are at 67 N, where the cos(lat) factor is 0.39."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the CSV. Without it this is a dry run that only reports.")
    args = ap.parse_args()

    if not CSV.exists():
        print(f"FATAL: {CSV} not found (run from the repo root)", file=sys.stderr)
        return 2
    df = pd.read_csv(CSV)
    print(f"read {CSV}: {len(df)} rows\n")

    sm = df[df["has_soil_moisture"].astype(str).str.lower() == "true"].copy()
    print(f"soil-moisture stations: {len(sm)}")
    print("split counts before:", dict(sm["split"].value_counts()), "\n")

    # ── pairwise geometry, global (not restricted to location_group_id) ──────
    recs = sm[["station_id", "network", "latitude", "longitude", "split",
               "location_group_id"]].to_dict("records")
    same_patch, tile_pair, dups = set(), set(), []
    n_pairs = 0
    for i in range(len(recs)):
        a = recs[i]
        for j in range(i + 1, len(recs)):
            b = recs[j]
            # cheap reject before the trig: 1 deg lat ~ 111 km
            if abs(a["latitude"] - b["latitude"]) > 0.02:
                continue
            d = haversine_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            n_pairs += 1
            if d < DUP_M and a["network"] != b["network"]:
                dups.append((d, a, b))
            if d < PATCH_M:
                same_patch.add(a["station_id"]); same_patch.add(b["station_id"])
            elif d < HALF_TILE_M:
                tile_pair.add(a["station_id"]); tile_pair.add(b["station_id"])
            if d < HALF_TILE_M and a["split"] != b["split"]:
                print(f"  *** LEAKAGE: {a['station_id']} ({a['split']}) and "
                      f"{b['station_id']} ({b['split']}) are {d:.0f} m apart and in "
                      f"DIFFERENT splits — one tile spans both ***")

    print(f"pairs evaluated (after latitude reject): {n_pairs}")
    print(f"same-patch  (<{PATCH_M:.0f} m)            : {len(same_patch)} stations")
    print(f"tile-pair   ({PATCH_M:.0f}-{HALF_TILE_M:.0f} m)          : {len(tile_pair)} stations\n")

    # ── duplicate sweep ─────────────────────────────────────────────────────
    print(f"=== CROSS-NETWORK DUPLICATE SWEEP (< {DUP_M:.0f} m, different networks) ===")
    dup_drop = {}
    if not dups:
        print("  none found")
    for d, a, b in sorted(dups, key=lambda t: t[0]):
        # Keep whichever is already outside train if that breaks the tie, else keep
        # the first alphabetically so the choice is deterministic and reviewable.
        keep, drop = (a, b) if (a["split"] != "train" and b["split"] == "train") else \
                     ((b, a) if (b["split"] != "train" and a["split"] == "train") else
                      ((a, b) if a["station_id"] < b["station_id"] else (b, a)))
        print(f"  {d:6.1f} m  {a['station_id']:<18s}({a['network']:<20s} {a['split']:<6s}) "
              f"<-> {b['station_id']:<18s}({b['network']:<20s} {b['split']:<6s})   "
              f"KEEP {keep['station_id']}  DROP {drop['station_id']}")
        dup_drop[drop["station_id"]] = keep["station_id"]

    # ── build the new columns ───────────────────────────────────────────────
    df["same_patch_pair"] = df["station_id"].isin(same_patch)
    df["tile_pair_eval"]  = df["station_id"].isin(tile_pair)
    df["duplicate_of"]    = df["station_id"].map(dup_drop).fillna("")

    hold_out = (same_patch | tile_pair)
    moved = df[(df["split"] == "train") & (df["station_id"].isin(hold_out))]
    print(f"\n=== MOVING OUT OF TRAIN: {len(moved)} stations ===")
    for _, r in moved.iterrows():
        why = "same-patch" if r["station_id"] in same_patch else "tile-pair"
        print(f"  {r['station_id']:<18s} {r['network']:<22s} train -> oos   ({why})")
    df.loc[(df["split"] == "train") & (df["station_id"].isin(hold_out)), "split"] = "oos"

    print(f"\n=== MARKING DUPLICATES: {len(dup_drop)} rows ===")
    for drop_id, keep_id in dup_drop.items():
        print(f"  {drop_id:<18s} split -> 'duplicate'   (duplicate_of {keep_id})")
    df.loc[df["station_id"].isin(dup_drop), "split"] = "duplicate"

    after = df[df["has_soil_moisture"].astype(str).str.lower() == "true"]
    print("\nsplit counts after:", dict(after["split"].value_counts()))

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    backup = CSV.with_suffix(".csv.pre_s3529")
    shutil.copy2(CSV, backup)
    df.to_csv(CSV, index=False)
    print(f"\nwrote {CSV}  (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
