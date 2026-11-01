#!/usr/bin/env python
"""
Verify a per-station zarr store by ARRAY CONTENT, not by sentinel files.

§35.1. The 2026-08-26 scratch purge deleted array chunks while leaving
`.zmetadata` intact at each station root. `zarr.open_consolidated` therefore
still exposes e.g. `soil` as a valid array and silently returns `fill_value`
(all zeros) with no exception and no warning. `slurm/restore_zarr.sh` verified
by counting `.complete` markers -- 0-byte files that rsync restores first -- so
it would report success on a store whose every chunk was missing.

This script checks, per station and per array:
  1. the array directory exists and holds a `.zarray`
  2. it holds at least one chunk file (this is what actually failed)
  3. a sampled read is not entirely `fill_value`
  4. targeted invariants: soil > 0, dem non-constant, era5 finite

and at station root:
  5. every `*_l{3,6,9}.npy` memmap matches the shape in its `.json` sidecar
     and matches the corresponding zarr array's shape

Run it on the BACKUP before restoring, and on the LIVE store afterwards.

Usage
-----
    python verify_zarr_store.py --root /projects/prjs1968/zarr_tokens
    python verify_zarr_store.py --root /gpfs/scratch1/shared/pkhanal/zarr
    python verify_zarr_store.py --root <r> --out csvs/verify_backup.csv --workers 64

Exit code is 0 only if every station passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

CATEGORIES = ("sm_only", "sm_and_flux", "flux_only")

# Arrays whose content must be non-degenerate. Masks and date arrays are
# excluded: an all-True token_mask or an all-zero doys slot is legitimate.
DATA_ARRAYS = (
    "soil", "dem", "lulc",
    "era5/values", "labels/sm",
    "s2/l3", "s2/l6", "s2/l9", "s2/l12",
    "s1_asc/l3", "s1_asc/l6", "s1_asc/l9", "s1_asc/l12",
    "s1_desc/l3", "s1_desc/l6", "s1_desc/l9", "s1_desc/l12",
    "sif/values", "twsa/lwe", "labels/le",
)

# Arrays that must merely exist; content is not constrained. The store's schema
# varies legitimately by station, so these are grouped by what actually applies:
#
#   - `s2/token_mask` DOES NOT EXIST by design. Token masks are computed only for
#     s1_asc/s1_desc/dem/lulc (compute_s1_dem_lulc_token_masks.py:10-13); S2 is
#     masked from `cm/masks` instead (dataset.py:270,276).
#   - flux_only stations carry a different label schema (latent-heat flux)
#     rather than soil moisture.
#   - some stations have s1_desc and no s1_asc, or vice versa.
REQUIRED_ALWAYS = (
    "dem", "lulc", "dem_token_mask", "lulc_token_mask", "soil",
    "era5/values", "era5/date_ints", "era5/doys",
    "s2/dates",
)
REQUIRED_LABELS = {
    "sm_only":     ("labels/sm", "labels/dates", "labels/depths", "labels/qc"),
    "sm_and_flux": ("labels/sm", "labels/dates", "labels/depths", "labels/qc"),
    "flux_only":   ("labels/le", "labels/dates_flux", "labels/le_qc"),
}
# Required within an orbit group, but only when that group exists at all.
PER_ORBIT_REQUIRED = ("dates", "token_mask", "l3", "l6", "l9", "l12")


def _chunk_files(array_dir: Path) -> int:
    """Count chunk files in a zarr array directory (dotfiles are metadata)."""
    if not array_dir.is_dir():
        return -1
    try:
        return sum(1 for e in os.scandir(array_dir) if e.is_file() and not e.name.startswith("."))
    except OSError:
        return -1


def _walk_arrays(group, prefix=""):
    """Yield (path, zarr.Array) for every array under `group`."""
    for name, node in group.arrays():
        yield (f"{prefix}{name}", node)
    for name, sub in group.groups():
        yield from _walk_arrays(sub, prefix=f"{prefix}{name}/")


def _sample(arr) -> np.ndarray:
    """Read a small, cheap slice: one leading index for N-D, else the whole thing."""
    if arr.ndim == 0 or arr.shape[0] == 0:
        return np.asarray(arr[...])
    if arr.ndim == 1:
        return np.asarray(arr[: min(arr.shape[0], 4096)])
    return np.asarray(arr[0])


def _is_degenerate(sample: np.ndarray, fill) -> bool:
    """True if every element equals fill_value (or 0 when fill_value is None)."""
    if sample.size == 0:
        return True
    if sample.dtype.kind in "SU":          # date strings
        return bool((sample == b"").all() or (sample == "").all())
    target = 0 if fill is None else fill
    try:
        if isinstance(target, float) and np.isnan(target):
            return bool(np.isnan(sample).all())
        return bool((sample == target).all())
    except (TypeError, ValueError):
        return False


def check_station(job) -> dict:
    root, category, station = job
    sdir = Path(root) / category / station
    rec = {"category": category, "station": station, "ok": True,
           "n_arrays": 0, "n_npy": 0,
           "n_meta_only": 0, "n_data_loss": 0, "n_dir_absent": 0,
           "problems": ""}
    problems: list[str] = []

    try:
        try:
            zg = zarr.open_consolidated(str(sdir), mode="r")
        except (KeyError, ValueError, FileNotFoundError):
            zg = zarr.open_group(str(sdir), mode="r")

        seen = {}
        for apath, arr in _walk_arrays(zg):
            seen[apath] = arr
            rec["n_arrays"] += 1

            adir = sdir / apath
            nchunk = _chunk_files(adir)
            has_meta = (adir / ".zarray").exists()

            # The 2026-08-26 purge removed BOTH kinds of file, but not together:
            #   - `.zarray` (a few hundred bytes, written once in June, never
            #     re-read) was deleted for most arrays
            #   - chunk files survived wherever a read refreshed their atime
            # Distinguishing the two matters: a missing `.zarray` is a ~400-byte
            # repair, a missing chunk is real data loss.
            if nchunk < 0:
                rec["n_dir_absent"] += 1
                problems.append(f"{apath}: ARRAY DIR ABSENT")
                continue
            if not has_meta and nchunk == 0:
                rec["n_data_loss"] += 1
                problems.append(f"{apath}: EMPTY (no .zarray, no chunks) -- DATA LOSS")
                continue
            if not has_meta:
                rec["n_meta_only"] += 1
                problems.append(f"{apath}: META-ONLY (.zarray missing, {nchunk} chunks present)")
                continue
            if nchunk == 0:
                rec["n_data_loss"] += 1
                problems.append(f"{apath}: 0 chunks (reads as fill_value) -- DATA LOSS")
                continue

            if apath in DATA_ARRAYS:
                try:
                    s = _sample(arr)
                except Exception as e:
                    problems.append(f"{apath}: read failed ({type(e).__name__})")
                    continue
                if _is_degenerate(s, arr.fill_value):
                    problems.append(f"{apath}: sample is entirely fill_value")

        # ---- targeted invariants -------------------------------------------
        if "soil" in seen:
            s = np.asarray(seen["soil"][...])
            finite = s[np.isfinite(s)]
            if finite.size == 0:
                problems.append("soil: no finite values")
            elif float(finite.max()) <= 0.0:
                problems.append(f"soil: max<=0 (max={float(finite.max()):.3f}) -- zeroed")
        else:
            problems.append("soil: array absent")

        if "era5/values" in seen:
            e = np.asarray(seen["era5/values"][: min(seen["era5/values"].shape[0], 64)])
            if not np.isfinite(e).any():
                problems.append("era5/values: no finite values")
        else:
            problems.append("era5/values: array absent")

        if "dem" in seen:
            d = np.asarray(seen["dem"][...])
            fin = d[np.isfinite(d)]
            if fin.size and float(fin.std()) == 0.0:
                problems.append("dem: constant")

        for expected in REQUIRED_ALWAYS:
            if expected not in seen:
                problems.append(f"{expected}: absent")

        for expected in REQUIRED_LABELS.get(category, ()):
            if expected not in seen:
                problems.append(f"{expected}: absent")

        # Orbit groups are optional, but must be complete when present.
        for orbit in ("s1_asc", "s1_desc"):
            present = [k for k in seen if k.startswith(f"{orbit}/")]
            if not present:
                continue
            for leaf in PER_ORBIT_REQUIRED:
                if f"{orbit}/{leaf}" not in seen:
                    problems.append(f"{orbit}/{leaf}: absent (group exists)")
        if not any(k.startswith(("s1_asc/", "s1_desc/")) for k in seen):
            problems.append("s1_asc and s1_desc: BOTH absent")

        # ---- .npy memmaps against their sidecars and the zarr --------------
        for npy in sorted(sdir.glob("*_l[369].npy")):
            rec["n_npy"] += 1
            meta = npy.with_suffix(".json")
            if not meta.exists():
                problems.append(f"{npy.name}: sidecar missing")
                continue
            try:
                m = json.loads(meta.read_text())
                shape = tuple(m["shape"])
                itemsize = np.dtype(m.get("dtype", "float16")).itemsize
            except Exception:
                problems.append(f"{npy.name}: sidecar unreadable")
                continue
            expect_bytes = int(np.prod(shape)) * itemsize
            actual = npy.stat().st_size
            if actual != expect_bytes:
                problems.append(f"{npy.name}: {actual}B != {expect_bytes}B from sidecar")
                continue
            key = npy.stem.replace("_l3", "/l3").replace("_l6", "/l6").replace("_l9", "/l9")
            if key in seen and tuple(seen[key].shape) != shape:
                problems.append(f"{npy.name}: shape {shape} != zarr {tuple(seen[key].shape)}")

    except Exception as e:
        problems.append(f"FATAL {type(e).__name__}: {e}")
        rec["traceback"] = traceback.format_exc(limit=3)

    rec["ok"] = not problems
    rec["problems"] = " | ".join(problems)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="store root holding sm_only/ etc.")
    ap.add_argument("--out", default=None, help="CSV report path")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="check only the first N stations")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    jobs = []
    for cat in CATEGORIES:
        cdir = root / cat
        if not cdir.is_dir():
            print(f"  note: {cat}/ absent under {root}")
            continue
        for st in sorted(os.listdir(cdir)):
            if (cdir / st).is_dir():
                jobs.append((str(root), cat, st))
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Verifying {len(jobs)} stations under {root} with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        rows = pool.map(check_station, jobs, chunksize=1)

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path("csvs") / f"verify_{root.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_bad = int((~df["ok"]).sum())
    print(f"\n{'='*72}")
    print(f"stations checked : {len(df)}")
    print(f"PASS             : {int(df['ok'].sum())}")
    print(f"FAIL             : {n_bad}")
    print(f"report           : {out}")
    print(f"\n--- damage class (array-instances summed over stations) ---")
    print(f"  META-ONLY (.zarray gone, chunks present, cheap repair) : {int(df['n_meta_only'].sum())}")
    print(f"  DATA LOSS (chunks gone, must restore from backup)      : {int(df['n_data_loss'].sum())}")
    print(f"  ARRAY DIR ABSENT                                       : {int(df['n_dir_absent'].sum())}")
    n_ml = int((df["n_data_loss"] == 0).sum() & 1) if False else int(((df["n_data_loss"] == 0) & (df["n_meta_only"] > 0)).sum())
    print(f"  stations with META-ONLY damage and NO data loss        : {n_ml}")
    if n_bad:
        print(f"\n--- first 5 failures (truncated; full detail in the CSV) ---")
        for _, r in df[~df["ok"]].head(5).iterrows():
            p = str(r["problems"])
            print(f"  {r['category']}/{r['station']}: {p[:220]}{'...' if len(p) > 220 else ''}")
        # which problems dominate
        kinds: dict[str, int] = {}
        for p in df.loc[~df["ok"], "problems"]:
            for part in str(p).split(" | "):
                key = part.split(":")[0]
                kinds[key] = kinds.get(key, 0) + 1
        print(f"\n--- problem counts by array ---")
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  {v:5d}  {k}")
    print(f"{'='*72}")
    print("VERDICT:", "PASS - store is intact" if n_bad == 0 else "FAIL - do NOT proceed")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
