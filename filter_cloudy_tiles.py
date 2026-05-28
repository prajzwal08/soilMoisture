"""
filter_cloudy_tiles.py
======================
Phase 1 (--analyze, default): walk all CloudMask TIFs, classify each tile as
keep/reject using the combined cloud+nodata criterion, print survival stats, and
save a rejection manifest CSV.  NO files are deleted.

Phase 2 (--delete): read the manifest written by phase 1 and delete the listed
S2L2A tiles from scratch (and optionally their CloudMask TIFs).

Token-validity criterion (per 16×16 patch):
  - ANY cloud pixel (class 3/4/5) → patch invalid
  - ≥1% nodata pixels (class 255)  → patch invalid
Tile rejection threshold: >50% of 196 patches invalid.

Usage:
    # Phase 1 — analysis only, saves manifest
    python filter_cloudy_tiles.py --analyze
    python filter_cloudy_tiles.py --analyze --manifest text/cloudy_tile_manifest.csv

    # Phase 2 — delete S2L2A tiles listed in manifest
    python filter_cloudy_tiles.py --delete --manifest text/cloudy_tile_manifest.csv

    # Phase 2 — also delete CloudMask TIFs
    python filter_cloudy_tiles.py --delete --manifest text/cloudy_tile_manifest.csv --also-cloudmask
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DATA_DIR    = Path("/gpfs/work3/0/prjs1968/data")
SPLITS      = ("sm_only", "sm_and_flux", "flux_only")

# ── patch validity ────────────────────────────────────────────────────────────
PATCH_SIZE        = 16
TILE_SIZE         = 224
N_SIDE            = TILE_SIZE // PATCH_SIZE   # 14
N_PATCHES         = N_SIDE * N_SIDE           # 196
N_PIX             = PATCH_SIZE * PATCH_SIZE   # 256
NODATA_FRAC_THRESH = 0.01   # ≥1% nodata per patch → invalid
TILE_REJECT_THRESH = 0.50   # >50% invalid patches → reject tile


def patch_validity(arr: np.ndarray) -> np.ndarray:
    """(1,224,224) uint8 cloud mask → (196,) bool valid-token mask."""
    a = arr[0]
    valid = np.zeros(N_PATCHES, dtype=bool)
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            p = a[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE]
            has_cloud   = np.any((p == 3) | (p == 4) | (p == 5))
            nodata_frac = np.sum(p == 255) / N_PIX
            valid[i*N_SIDE + j] = (not has_cloud) and (nodata_frac < NODATA_FRAC_THRESH)
    return valid


def masked_frac(cm_path: Path) -> float | None:
    """Return fraction of invalid patches for one CloudMask TIF, or None on error."""
    try:
        with rasterio.open(cm_path) as src:
            arr = src.read()
    except Exception:
        return None
    return float(1.0 - patch_validity(arr).mean())


# ── manifest helpers ──────────────────────────────────────────────────────────
MANIFEST_FIELDS = [
    "station", "split", "date", "year",
    "masked_frac",
    "s2_path", "cloudmask_path",
]


def _collect_tiles() -> list[dict]:
    """Walk all CloudMask TIFs; return list of dicts with path info."""
    rows = []
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for station_dir in sorted(split_dir.iterdir()):
            cm_dir = station_dir / "CloudMask"
            if not cm_dir.exists():
                continue
            sid = station_dir.name
            for cm_tif in sorted(cm_dir.glob("*.tif")):
                date = cm_tif.stem          # YYYYMMDD
                year = date[:4] if len(date) >= 4 else "????"
                s2_path = SCRATCH_DIR / sid / "S2L2A" / cm_tif.name
                rows.append({
                    "station"       : sid,
                    "split"         : split,
                    "date"          : date,
                    "year"          : year,
                    "cloudmask_path": str(cm_tif),
                    "s2_path"       : str(s2_path),
                })
    return rows


# ── phase 1: analyze ──────────────────────────────────────────────────────────

def run_analyze(manifest_path: Path) -> None:
    print("Phase 1: analyzing all CloudMask TIFs (no files deleted)")
    print(f"Rejection threshold : >{TILE_REJECT_THRESH*100:.0f}% invalid patches per tile")
    print(f"Patch invalid if    : any cloud (cls 3/4/5) OR ≥{NODATA_FRAC_THRESH*100:.0f}% nodata (cls 255)")
    print()

    tiles = _collect_tiles()
    print(f"Total CloudMask TIFs found : {len(tiles):,}")
    print()

    # Score every tile
    rejected_rows: list[dict] = []
    kept_rows:     list[dict] = []

    # per-station counters
    station_total:    dict[str, int]        = defaultdict(int)
    station_kept:     dict[str, int]        = defaultdict(int)
    station_yr_total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    station_yr_kept:  dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    errors = 0
    for row in tqdm(tiles, desc="Scoring", unit="tile"):
        sid  = row["station"]
        year = row["year"]
        frac = masked_frac(Path(row["cloudmask_path"]))
        station_total[sid] += 1
        station_yr_total[sid][year] += 1

        if frac is None:
            errors += 1
            row["masked_frac"] = "ERROR"
            kept_rows.append(row)   # don't reject on read error
            station_kept[sid] += 1
            station_yr_kept[sid][year] += 1
            continue

        row["masked_frac"] = f"{frac:.4f}"
        if frac > TILE_REJECT_THRESH:
            rejected_rows.append(row)
        else:
            kept_rows.append(row)
            station_kept[sid] += 1
            station_yr_kept[sid][year] += 1

    total      = len(tiles)
    n_rejected = len(rejected_rows)
    n_kept     = total - n_rejected

    # ── global summary ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"GLOBAL TILE SURVIVAL  (threshold >{TILE_REJECT_THRESH*100:.0f}% invalid patches)")
    print(f"{'='*65}")
    print(f"  Total tiles    : {total:>8,}")
    print(f"  Kept           : {n_kept:>8,}  ({n_kept/total*100:.1f}%)")
    print(f"  Rejected       : {n_rejected:>8,}  ({n_rejected/total*100:.1f}%)")
    if errors:
        print(f"  Read errors    : {errors:>8,}  (counted as kept)")

    # ── station-level summary ─────────────────────────────────────────────────
    n_stations   = len(station_total)
    stations_zero = [s for s in station_total if station_kept[s] == 0]
    stations_lt10 = [s for s in station_total
                     if 0 < station_kept[s] < 10]

    print(f"\n{'='*65}")
    print(f"STATION SURVIVAL")
    print(f"{'='*65}")
    print(f"  Total stations : {n_stations}")
    print(f"  Stations with 0 tiles after filter : {len(stations_zero)}")
    print(f"  Stations with <10 tiles after filter: {len(stations_lt10)}")

    if stations_zero:
        print(f"\n  Stations losing ALL tiles:")
        for s in sorted(stations_zero):
            print(f"    {s}  ({station_total[s]} tiles rejected)")

    # ── worst-hit stations (most rejected) ────────────────────────────────────
    reject_pct = {s: 100.0 * (station_total[s] - station_kept[s]) / station_total[s]
                  for s in station_total}
    top_rejected = sorted(reject_pct.items(), key=lambda x: -x[1])[:15]

    print(f"\n  Top 15 stations by % tiles rejected:")
    print(f"  {'Station':<45}  {'Kept':>6}  {'Rejected':>8}  {'Rej%':>6}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*8}  {'-'*6}")
    for sid, pct in top_rejected:
        k = station_kept[sid]
        r = station_total[sid] - k
        print(f"  {sid:<45}  {k:>6,}  {r:>8,}  {pct:>5.1f}%")

    # ── per-station per-year table ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"TILES PER STATION PER YEAR  (kept / total)")
    print(f"{'='*65}")

    all_years = sorted({y for s in station_yr_total for y in station_yr_total[s]})
    # header
    year_cols = "  ".join(f"{y:>9}" for y in all_years)
    print(f"  {'Station':<45}  {year_cols}")
    print(f"  {'-'*45}  " + "  ".join(["-"*9]*len(all_years)))

    for sid in sorted(station_total.keys()):
        cols = []
        for yr in all_years:
            tot = station_yr_total[sid].get(yr, 0)
            kpt = station_yr_kept[sid].get(yr, 0)
            if tot == 0:
                cols.append(f"{'':>9}")
            else:
                cols.append(f"{kpt:>4}/{tot:<4}")
        print(f"  {sid:<45}  " + "  ".join(cols))

    # ── write manifest ────────────────────────────────────────────────────────
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rejected_rows)

    print(f"\n{'='*65}")
    print(f"Rejection manifest saved → {manifest_path}")
    print(f"  {len(rejected_rows):,} tiles listed for deletion")
    print(f"  Run with --delete to actually remove them.")
    print(f"{'='*65}")


# ── phase 2: delete ───────────────────────────────────────────────────────────

def run_delete(manifest_path: Path, also_cloudmask: bool) -> None:
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run with --analyze first.")
        sys.exit(1)

    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Phase 2: deleting {len(rows):,} tiles listed in {manifest_path}")
    if also_cloudmask:
        print("Also deleting CloudMask TIFs from project storage.")
    print()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_path = manifest_path.with_name(manifest_path.stem + "_delete_log.csv")

    deleted_s2 = deleted_cm = skipped = errors = 0

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "station", "date", "masked_frac",
                         "s2_deleted", "cloudmask_deleted", "note"])

        for row in tqdm(rows, desc="Deleting", unit="file"):
            s2_path = Path(row["s2_path"])
            cm_path = Path(row["cloudmask_path"])
            s2_ok = cm_ok = False
            note = ""

            # delete S2L2A from scratch
            if s2_path.exists():
                try:
                    s2_path.unlink()
                    s2_ok = True
                    deleted_s2 += 1
                except OSError as exc:
                    note += f"S2 delete failed: {exc}; "
                    errors += 1
            else:
                note += "S2 not found in scratch; "
                skipped += 1

            # optionally delete CloudMask TIF
            if also_cloudmask and cm_path.exists():
                try:
                    cm_path.unlink()
                    cm_ok = True
                    deleted_cm += 1
                except OSError as exc:
                    note += f"CM delete failed: {exc}; "
                    errors += 1

            writer.writerow([ts, row["station"], row["date"], row["masked_frac"],
                             s2_ok, cm_ok, note.strip()])

    print(f"\nDone.")
    print(f"  S2L2A tiles deleted     : {deleted_s2:,}")
    if also_cloudmask:
        print(f"  CloudMask TIFs deleted  : {deleted_cm:,}")
    print(f"  S2 not in scratch       : {skipped:,}  (already purged or never existed)")
    print(f"  Errors                  : {errors:,}")
    print(f"  Delete log saved →  {log_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and optionally delete cloudy S2L2A tiles")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true", default=True,
                      help="Score all tiles and save rejection manifest (default)")
    mode.add_argument("--delete",  action="store_true",
                      help="Delete tiles listed in manifest (run after --analyze)")
    parser.add_argument("--manifest", type=Path,
                        default=Path("text/cloudy_tile_manifest.csv"),
                        help="Path to rejection manifest CSV")
    parser.add_argument("--also-cloudmask", action="store_true",
                        help="(--delete only) also remove CloudMask TIFs from project storage")
    args = parser.parse_args()

    if args.delete:
        run_delete(args.manifest, args.also_cloudmask)
    else:
        run_analyze(args.manifest)


if __name__ == "__main__":
    main()
