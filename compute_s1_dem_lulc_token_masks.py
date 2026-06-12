"""
Compute per-patch nodata masks for S1 (asc/desc), DEM, and LULC from raw
satellite_zarr and write them into the token zarr stores.

Mask convention (same as precompute_terramind._nn_fill_and_sanitize):
  True  → nodata_frac < 1%  → valid/usable patch
  False → nodata_frac ≥ 1%  → masked/bad patch (encoder received 0-filled input)

Arrays written per token-zarr station:
  s1_asc/token_mask   (N_asc,  14, 14) bool
  s1_desc/token_mask  (N_desc, 14, 14) bool
  dem_token_mask      (14, 14)         bool
  lulc_token_mask     (14, 14)         bool

Usage:
  python compute_s1_dem_lulc_token_masks.py          # dry-run: counts only
  python compute_s1_dem_lulc_token_masks.py --execute
"""

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

ZARR_ROOT       = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ZARR_ROOT   = Path("/gpfs/work3/0/prjs1968/satellite_zarr")
CATEGORIES      = ("sm_only", "sm_and_flux", "flux_only")
FILLABLE_THRESH = 0.01   # ≥1% nodata pixels in a 16×16 patch → masked
TOKEN_SIZE      = 16
N_SIDE          = 14     # 224 / 16 = 14
N_WORKERS       = 64


def _patch_mask_from_nodata(nodata_2d: np.ndarray) -> np.ndarray:
    """nodata_2d: (224,224) bool — True where pixel is nodata.
    Returns (14,14) bool — True where patch valid (nodata_frac < threshold)."""
    mask = np.ones((N_SIDE, N_SIDE), dtype=bool)
    if not nodata_2d.any():
        return mask
    for r in range(N_SIDE):
        for c in range(N_SIDE):
            rs, re = r * TOKEN_SIZE, (r + 1) * TOKEN_SIZE
            cs, ce = c * TOKEN_SIZE, (c + 1) * TOKEN_SIZE
            frac = nodata_2d[rs:re, cs:ce].mean()
            if frac >= FILLABLE_THRESH:
                mask[r, c] = False
    return mask


def _s1_token_masks(data: np.ndarray) -> np.ndarray:
    """data: (N, C, 224, 224) float16 — raw S1RTC (NaN at swath edges).
    Returns (N, 14, 14) bool."""
    data_f = data.astype(np.float32, copy=False)
    nodata = np.isnan(data_f).any(axis=1)   # (N, 224, 224)
    return np.stack([_patch_mask_from_nodata(nodata[i]) for i in range(len(nodata))])


def _dem_token_mask(data: np.ndarray) -> np.ndarray:
    """data: (1, 224, 224) float32 — raw DEM (NaN = nodata).
    Returns (14, 14) bool."""
    return _patch_mask_from_nodata(np.isnan(data[0]))


def _lulc_token_mask(data: np.ndarray) -> np.ndarray:
    """data: (N_years, 224, 224) uint8 — uses latest year; 0 = nodata.
    Returns (14, 14) bool."""
    return _patch_mask_from_nodata(data[-1] == 0)


def _process_station(args: tuple) -> tuple[str, object]:
    cat, name, execute = args
    tok_path = ZARR_ROOT / cat / name
    sat_path = SAT_ZARR_ROOT / f"{name}.zarr"

    if not sat_path.exists():
        return name, "no_sat_zarr"

    try:
        rg    = zarr.open_group(str(sat_path), mode="r")
        store = zarr.DirectoryStore(str(tok_path))
        root  = zarr.open_group(store=store, mode="a" if execute else "r")

        to_write: dict[str, np.ndarray] = {}
        n_masked = 0

        # ── S1 asc / desc ────────────────────────────────────────────
        for orbit in ("s1_asc", "s1_desc"):
            key = f"{orbit}/token_mask"
            if key in root:
                continue   # already computed — resume-safe
            data_key = f"{orbit}/data"
            if data_key not in rg:
                continue
            tm = _s1_token_masks(rg[data_key][:])
            n_masked += int((~tm).sum())
            to_write[key] = tm

        # ── DEM ──────────────────────────────────────────────────────
        if "dem_token_mask" not in root and "dem/data" in rg:
            tm = _dem_token_mask(rg["dem/data"][:].astype(np.float32))
            n_masked += int((~tm).sum())
            to_write["dem_token_mask"] = tm

        # ── LULC (latest year) ───────────────────────────────────────
        if "lulc_token_mask" not in root and "lulc/data" in rg:
            tm = _lulc_token_mask(rg["lulc/data"][:])
            n_masked += int((~tm).sum())
            to_write["lulc_token_mask"] = tm

        if execute and to_write:
            for key, arr in to_write.items():
                if "/" in key:
                    grp_key, arr_key = key.rsplit("/", 1)
                    grp = root.require_group(grp_key)
                    grp.create_dataset(arr_key, data=arr, dtype=bool,
                                       chunks=True, overwrite=False)
                else:
                    root.create_dataset(key, data=arr, dtype=bool,
                                        chunks=True, overwrite=False)
            zarr.consolidate_metadata(store)

        return name, n_masked

    except Exception as exc:
        return name, f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Write masks to token zarr (default: dry-run)")
    args = parser.parse_args()

    jobs: list[tuple] = []
    for cat in CATEGORIES:
        cat_root = ZARR_ROOT / cat
        if not cat_root.exists():
            continue
        for station_dir in sorted(cat_root.iterdir()):
            if (station_dir / ".complete").exists():
                jobs.append((cat, station_dir.name, args.execute))

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    logging.info(f"{mode}: {len(jobs)} stations, Pool({N_WORKERS})")

    total_masked = 0
    n_errors = 0
    n_no_sat = 0
    with Pool(N_WORKERS) as pool:
        for name, result in pool.imap_unordered(_process_station, jobs,
                                                chunksize=4):
            if isinstance(result, str):
                if result.startswith("ERROR"):
                    logging.warning(f"{name}: {result}")
                    n_errors += 1
                elif result == "no_sat_zarr":
                    n_no_sat += 1
            else:
                total_masked += result

    logging.info(f"Total masked patches: {total_masked:,}")
    logging.info(f"Missing satellite_zarr: {n_no_sat}  |  Errors: {n_errors}")
    if not args.execute:
        logging.info("Dry-run complete — pass --execute to write masks.")


if __name__ == "__main__":
    main()
