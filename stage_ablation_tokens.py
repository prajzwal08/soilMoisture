"""§24.13 step 0 — stage the ablation_oos token stores from /projects to scratch.

The frozen U-Net arm reads `dataset_unet.py:40` ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr,
which the purge left as a 24 MB directory skeleton: 993 station dirs, zero regular files, zero
`.complete` sentinels. The full store survived at /projects/prjs1968/zarr_tokens (993 `.complete`,
1.4 TB, read-only).

Staging rather than repointing ZARR_ROOT is deliberate: scratch reads faster, and the frozen arm
then runs UNMODIFIED at its original path, which is the strongest provenance available for
reproducing §24.11.

Only the `sm_only` stations of the ablation_oos subset are needed — `category_filter=["sm_only"]`
(train_unet.py:191) drops the rest — so this copies ~36 stations / ~50 GB, not 1.4 TB.

THE SENTINEL IS THE HAZARD. `_open_zarr` (dataset_unet.py:127-140) gates on `.complete` alone and
returns None without raising when it is absent, so the dataset builds 0 samples silently. A
sentinel sitting beside a HALF-copied station is that same failure in a worse form: partial data
that looks whole. So `.complete` is excluded from the copy and written only after that station's
file count and byte total match the source exactly. Verification never counts sentinels — that is
precisely what a broken copy would also satisfy.

The source is read-only and is never written to. The staged copy is disposable; scratch will be
purged again and /projects remains the only permanent copy.

    python stage_ablation_tokens.py                 # dry run: report what would be copied
    python stage_ablation_tokens.py --execute       # do it
    sbatch slurm/stage_ablation_tokens.sh
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

SRC_ROOT = Path("/projects/prjs1968/zarr_tokens")
DST_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
FLAG_CSV = Path("eval_output/_flag_oos_ablation_oos.csv")
SENTINEL = ".complete"
REPORT   = Path("csvs/stage_ablation_tokens_report.json")


def station_dirs(csv_path: Path, categories=("sm_only",)) -> list[tuple[str, str]]:
    """(category, dir_name) for each row, matching dataset_unet.py:812-820 exactly.

    pandas, never `awk -F,`: one AmeriFlux row has a quoted comma in `station_name` and naive
    field splitting shifts every column after it (see the station_splits parsing note).
    """
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        fl = str(r.get("has_flux",          "False")).lower() == "true"
        cat = "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")
        if cat not in categories:
            continue
        if not bool(r.get("soil_patch_ok", True)):
            continue
        if str(r["source_network"]) == "ISMN":
            dir_name = f"ISMN_{r['network']}_{r['station_name']}"
        else:
            dir_name = f"{r['source_network']}_{r['station_id']}"
        out.append((cat, dir_name))
    return out


def tree_stats(root: Path) -> tuple[int, int]:
    """(n_files, total_bytes) under root, EXCLUDING the sentinel.

    The sentinel is excluded on both sides so source and destination are compared on payload
    only — the destination does not have one yet at verification time.
    """
    n = b = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f == SENTINEL:
                continue
            try:
                b += os.stat(os.path.join(dirpath, f)).st_size
                n += 1
            except FileNotFoundError:
                pass          # a file vanished mid-walk; counted as mismatch downstream
    return n, b


def make_dirs_writable(root: Path) -> None:
    """chmod u+rwx over the DESTINATION's directories.

    The source tree is read-only (dr-xr-x---) and plain `rsync -a` replicates that mode onto the
    destination, after which we cannot create `.complete` inside our own copy — the failure that
    killed job 27051106. Directories only: chmod-ing every zarr chunk would be tens of thousands
    of metadata ops for no benefit, since `--chmod=Fu+rw` covers the files rsync writes.

    Only ever called on DST_ROOT paths. The source is never modified.
    """
    for dirpath, dirnames, _files in os.walk(root):
        for name in (dirpath, *(os.path.join(dirpath, d) for d in dirnames)):
            try:
                os.chmod(name, os.stat(name).st_mode | 0o700)
            except (FileNotFoundError, PermissionError):
                pass


def stage_one(task: tuple[str, str, bool]) -> dict:
    cat, dir_name, execute = task
    src = SRC_ROOT / cat / dir_name
    dst = DST_ROOT / cat / dir_name
    rec = {"category": cat, "station": dir_name, "src": str(src), "dst": str(dst)}
    try:
        return _stage_one(rec, src, dst, execute)
    except Exception as e:                      # noqa: BLE001
        # One station must never abort the pool: imap_unordered re-raises in the parent and the
        # remaining 35 are lost. Report and carry on; no sentinel is written, so a station that
        # failed here stays invisible to _open_zarr.
        return {**rec, "status": "ERROR", "error": f"{type(e).__name__}: {e}"}


def _stage_one(rec: dict, src: Path, dst: Path, execute: bool) -> dict:
    if not src.is_dir():
        return {**rec, "status": "SRC_MISSING"}
    if not (src / SENTINEL).exists():
        return {**rec, "status": "SRC_INCOMPLETE"}

    s_n, s_b = tree_stats(src)
    rec.update(src_files=s_n, src_bytes=s_b)

    if not execute:
        d_n, d_b = tree_stats(dst) if dst.is_dir() else (0, 0)
        rec.update(dst_files=d_n, dst_bytes=d_b, status="DRY_RUN")
        return rec

    t0 = time.time()
    dst.mkdir(parents=True, exist_ok=True)
    make_dirs_writable(dst)      # undo any read-only mode a previous `-a` run stamped on

    # -a           resumable: unchanged files are skipped on a re-run.
    # --chmod      the source is read-only (dr-xr-x---) and plain -a REPLICATES that onto the
    #              destination, after which the sentinel cannot be created inside our own copy.
    # --delete     reconcile dst to src. A killed rsync leaves partial-transfer temp files
    #              (`.s2_l6.npy.a2o4Dw`) that no later rsync removes, and they fail verification
    #              forever. Safe here and nowhere else: dst is a disposable staging area, src is
    #              read-only and is never a --delete target.
    # --exclude    withholds the sentinel until verification passes; with --delete (and without
    #              --delete-excluded) an excluded file is also PROTECTED from deletion, so a
    #              sentinel already earned by a previous run survives.
    cmd = ["rsync", "-a", "--chmod=Du+rwx,Fu+rw", "--delete", "--exclude", SENTINEL,
           f"{src}/", f"{dst}/"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {**rec, "status": "RSYNC_FAILED", "rc": p.returncode,
                "stderr": p.stderr[-2000:]}

    d_n, d_b = tree_stats(dst)
    rec.update(dst_files=d_n, dst_bytes=d_b, seconds=round(time.time() - t0, 1))

    if (d_n, d_b) != (s_n, s_b):
        # No sentinel is written. The station stays invisible to _open_zarr, which is the
        # correct outcome — better absent than silently partial.
        return {**rec, "status": "VERIFY_FAILED"}

    # Sentinel LAST, and written fresh rather than copy2'd: copy2 preserves the source's
    # read-only mode, which would make a re-run unable to overwrite it.
    (dst / SENTINEL).write_bytes((src / SENTINEL).read_bytes())
    os.chmod(dst / SENTINEL, 0o644)
    return {**rec, "status": "OK"}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="actually copy; without it, report sizes and exit")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 64)),
                    help="parallel rsync processes (default: $SLURM_CPUS_PER_TASK, else 64)")
    ap.add_argument("--flag-csv", default=str(FLAG_CSV))
    ap.add_argument("--categories", nargs="+", default=["sm_only"],
                    help="station categories to stage (default: sm_only, matching "
                         "train_unet.py:191 category_filter)")
    args = ap.parse_args()

    stations = station_dirs(Path(args.flag_csv), tuple(args.categories))
    print(f"Source      : {SRC_ROOT}")
    print(f"Destination : {DST_ROOT}")
    print(f"Flag CSV    : {args.flag_csv}")
    print(f"Categories  : {args.categories}")
    print(f"Stations    : {len(stations)}")
    print(f"Workers     : {args.workers}")
    print(f"Mode        : {'EXECUTE' if args.execute else 'DRY RUN'}\n")
    if not stations:
        print("No stations selected — check --flag-csv and --categories.")
        return 1

    tasks = [(c, d, args.execute) for c, d in stations]
    t0 = time.time()
    results = []
    with Pool(min(args.workers, len(tasks))) as pool:
        for i, rec in enumerate(pool.imap_unordered(stage_one, tasks), 1):
            results.append(rec)
            gb = rec.get("src_bytes", 0) / 2**30
            print(f"[{i:3d}/{len(tasks)}] {rec['status']:15s} {rec['station']:45s} {gb:7.2f} GB")
            sys.stdout.flush()

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    total_gb = sum(r.get("src_bytes", 0) for r in results) / 2**30

    print(f"\n{'─' * 70}")
    for k in sorted(by_status):
        print(f"  {k:15s} {by_status[k]}")
    print(f"  {'total payload':15s} {total_gb:.1f} GB")
    print(f"  {'elapsed':15s} {time.time() - t0:.0f} s")

    bad = [r for r in results if r["status"] not in ("OK", "DRY_RUN")]
    for r in bad:
        print(f"  FAILED {r['station']}: {r['status']} "
              f"src=({r.get('src_files')},{r.get('src_bytes')}) "
              f"dst=({r.get('dst_files')},{r.get('dst_bytes')})")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"source": str(SRC_ROOT), "destination": str(DST_ROOT),
         "flag_csv": args.flag_csv, "categories": args.categories,
         "execute": args.execute, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "by_status": by_status, "total_bytes": sum(r.get("src_bytes", 0) for r in results),
         "stations": sorted(results, key=lambda r: r["station"])}, indent=2))
    print(f"\nReport → {REPORT}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to copy.")
        return 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
