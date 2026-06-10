"""
Scan all consolidated .pt token bundles and fix any date/token length mismatches.

A mismatch means the dates list is longer than the tokens tensor — the tokenisation
job recorded a date but failed to save the corresponding token (e.g. job preemption).

Fix: trim the longer side to match the shorter, resave in-place.

Usage:
    python fix_bundle_mismatch.py [--execute] [--workers 16]
"""

import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import torch

DATA_ROOT = Path("/gpfs/work3/0/prjs1968/data")
CATEGORIES = ["sm_only", "sm_and_flux", "flux_only"]


def check_file(pt_file: Path, execute: bool) -> dict | None:
    """Check one .pt file. Returns a result dict if mismatched, else None."""
    name = pt_file.name
    is_bundle = any(f"_L{l}_" in name or f"_L{l}." in name
                    for l in ("3", "6", "9", "12"))
    if not is_bundle:
        return None

    try:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"path": pt_file, "error": str(e)}

    if not isinstance(data, dict) or "tokens" not in data or "dates" not in data:
        return None

    n_tok = data["tokens"].shape[0]
    n_dat = len(data["dates"])

    if n_tok == n_dat:
        return None

    result = {"path": pt_file, "n_tok": n_tok, "n_dat": n_dat, "fixed": False}

    if execute:
        if n_tok > n_dat:
            data["tokens"] = data["tokens"][:n_dat]
        else:
            data["dates"] = data["dates"][:n_tok]
        torch.save(data, pt_file)
        result["fixed"] = True

    return result


def scan_and_fix(execute: bool, workers: int):
    # Collect all .pt files across all categories
    all_files = []
    for cat in CATEGORIES:
        cat_dir = DATA_ROOT / cat
        if cat_dir.exists():
            all_files.extend(cat_dir.rglob("*.pt"))

    print(f"Found {len(all_files):,} .pt files — scanning with {workers} workers...")

    fn = partial(check_file, execute=execute)
    with Pool(workers) as pool:
        results = pool.map(fn, all_files)

    bad = [r for r in results if r is not None]
    errors   = [r for r in bad if "error" in r]
    mismatches = [r for r in bad if "error" not in r]

    for r in mismatches:
        status = "FIXED" if r.get("fixed") else "DRY-RUN"
        print(f"  [{status}] {r['path'].relative_to(DATA_ROOT)}"
              f"  tokens={r['n_tok']}  dates={r['n_dat']}")
    for r in errors:
        print(f"  [ERROR] {r['path'].relative_to(DATA_ROOT)}  — {r['error']}")

    print(f"\n{'='*60}")
    print(f"Scanned  : {len(all_files):,} files")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Errors   : {len(errors)}")
    if execute:
        print(f"Fixed    : {sum(1 for r in mismatches if r['fixed'])}")
    else:
        print("(dry-run — rerun with --execute to fix)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually fix files in-place (default: dry-run)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    print("EXECUTE mode\n" if args.execute else "DRY-RUN mode\n")
    scan_and_fix(args.execute, args.workers)
