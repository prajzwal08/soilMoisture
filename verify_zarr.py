"""
verify_zarr.py
==============
Post-migration verification: for every .zarr store in the archive, check that
the tile counts match the source TIFs and that data can actually be decompressed.

Outputs:
  csvs/zarr_verification.csv   — per-station pass/fail
  printed summary to stdout

Usage:
    python verify_zarr.py
    python verify_zarr.py --workers 16
    python verify_zarr.py --zarr-root /gpfs/work3/0/prjs1968/satellite_zarr
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import zarr

SRC_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/satellite")
ZARR_ROOT = Path("/gpfs/work3/0/prjs1968/satellite_zarr")
OUT_CSV   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/zarr_verification.csv")


def verify_station(args_tuple: tuple) -> dict:
    zarr_path, src_root = args_tuple
    name   = zarr_path.stem   # strip .zarr
    src    = src_root / name
    errors = []

    # 1. Sentinel
    if not (zarr_path / ".complete").exists():
        errors.append("missing .complete sentinel")
        return {"station": name, "ok": False, "errors": "; ".join(errors)}

    # 2. Open store
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        return {"station": name, "ok": False, "errors": f"zarr open failed: {e}"}

    # 3. S2 tile count
    if src.exists() and (src / "S2L2A").exists():
        expected = len(list((src / "S2L2A").glob("*.tif")))
        if expected > 0:
            if "s2" not in root:
                errors.append("s2 group missing")
            else:
                actual = root["s2/data"].shape[0]
                if actual != expected:
                    errors.append(f"s2 count mismatch: got {actual}, expected {expected}")
                else:
                    # spot-read first and last chunk
                    try:
                        _ = root["s2/data"][0]
                        _ = root["s2/data"][-1]
                    except Exception as e:
                        errors.append(f"s2 read error: {e}")

    # 4. S1 tile counts
    if src.exists() and (src / "S1RTC").exists():
        for orbit in ("ASC", "DESC"):
            expected = len([
                f for f in (src / "S1RTC").glob("*.tif")
                if re.search(rf"_{orbit}\.tif$", f.name)
            ])
            grp = f"s1_{orbit.lower()}"
            if expected > 0:
                if grp not in root:
                    errors.append(f"{grp} group missing")
                else:
                    actual = root[f"{grp}/data"].shape[0]
                    if actual != expected:
                        errors.append(f"{grp} count mismatch: got {actual}, expected {expected}")
                    else:
                        try:
                            _ = root[f"{grp}/data"][0]
                        except Exception as e:
                            errors.append(f"{grp} read error: {e}")

    ok = len(errors) == 0
    return {"station": name, "ok": ok, "errors": "; ".join(errors)}


def main():
    parser = argparse.ArgumentParser(description="Verify Zarr migration")
    parser.add_argument("--zarr-root", type=Path, default=ZARR_ROOT)
    parser.add_argument("--src-root",  type=Path, default=SRC_ROOT)
    parser.add_argument("--workers",   type=int,  default=16)
    args = parser.parse_args()

    zarr_stores = sorted(args.zarr_root.glob("*.zarr"))
    n = len(zarr_stores)
    print(f"Verifying {n} Zarr stores with {args.workers} workers ...\n")

    tasks = [(p, args.src_root) for p in zarr_stores]
    results = []

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(verify_station, tasks, chunksize=4), 1):
            results.append(res)
            if i % 100 == 0 or i == n:
                n_ok = sum(1 for r in results if r["ok"])
                print(f"  [{i}/{n}]  OK: {n_ok}", flush=True)

    df = pd.DataFrame(results).sort_values("station").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_ok   = df["ok"].sum()
    n_fail = n - n_ok
    print(f"\n{'='*50}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total stores : {n}")
    print(f"Passed       : {n_ok}")
    print(f"Failed       : {n_fail}")

    if n_fail:
        print(f"\nFailed stations:")
        for row in df[~df["ok"]].itertuples():
            print(f"  {row.station} — {row.errors}")
    else:
        print("\nAll stores verified OK.")

    print(f"\nResults saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
