"""
Dataset stress test — iterate every sample and verify shapes, dtypes, NaN counts.
Run before submitting training to confirm __getitem__ is clean across all code paths.

Usage:
    python check_dataset.py [--category sm_only] [--split train] [--workers 64] [--batch-size 8]
"""

import argparse
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS, MAX_S2, MAX_S1, MAX_SIF, MAX_TWSA

SPLITS_CSV   = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv"
ERA5_STATS   = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json"

# Expected shapes for every key returned by __getitem__
EXPECTED = {
    "s2_l12"        : (MAX_S2,  196, 768),
    "s2_doys"       : (MAX_S2,),
    "s2_valid"      : (MAX_S2,),
    "s2_token_mask" : (MAX_S2, 14, 14),
    "s2_rel_pos"    : (MAX_S2,),
    "s1_l12"         : (MAX_S1,  196, 768),
    "s1_doys"        : (MAX_S1,),
    "s1_valid"       : (MAX_S1,),
    "s1_token_mask"  : (MAX_S1, 14, 14),
    "s1_rel_pos"     : (MAX_S1,),
    "dem_l12"        : (196, 768),
    "lulc_l12"       : (196, 768),
    "dem_token_mask" : (14, 14),
    "lulc_token_mask": (14, 14),
    "anchor_l3"      : (196, 768),
    "anchor_l6"      : (196, 768),
    "anchor_l9"      : (196, 768),
    "anchor_l12"     : (196, 768),
    "anchor_rel_pos" : (),
    "anchor_orbit"   : (),
    "soil_patch"    : (21, 74, 74),
    "era5"          : (365, 19),
    "era5_doys"     : (365,),
    "sif"           : (MAX_SIF,  1),
    "sif_doys"      : (MAX_SIF,),
    "sif_valid"     : (MAX_SIF,),
    "twsa"          : (MAX_TWSA, 1),
    "twsa_doys"     : (MAX_TWSA,),
    "twsa_valid"    : (MAX_TWSA,),
    "label"         : (3,),
}

# Keys that must NEVER contain NaN
NO_NAN_KEYS = [
    "s2_l12", "s1_l12", "dem_l12", "lulc_l12",
    "anchor_l3", "anchor_l6", "anchor_l9", "anchor_l12",
    "soil_patch", "era5", "sif", "twsa",
]

# rel_pos must be in [0, 364]
REL_POS_KEYS = ["s2_rel_pos", "s1_rel_pos"]


def check_batch(batch: dict, base_idx: int) -> tuple[list[str], int, int]:
    """Vectorized check over a whole batch. Returns (errors, n_all_nan, n_has_obs)."""
    errors = []
    bs = next(v for v in batch.values() if isinstance(v, torch.Tensor)).shape[0]

    # Shape check — same for every item in the batch, check once per key
    for key, expected_shape in EXPECTED.items():
        if key not in batch:
            errors.append(f"  [batch@{base_idx}] MISSING key: {key}")
            continue
        t = batch[key]
        if not isinstance(t, torch.Tensor):
            errors.append(f"  [batch@{base_idx}] {key}: expected Tensor, got {type(t)}")
            continue
        actual = tuple(t.shape[1:])
        if actual != expected_shape:
            errors.append(f"  [batch@{base_idx}] {key}: shape {actual} != expected {expected_shape}")

    # NaN check — vectorized: find which samples in the batch have NaN
    for key in NO_NAN_KEYS:
        if key not in batch:
            continue
        t = batch[key].float()
        nan_per_sample = torch.isnan(t).flatten(1).any(dim=1)  # (B,)
        for b in nan_per_sample.nonzero(as_tuple=True)[0].tolist():
            n = torch.isnan(t[b]).sum().item()
            errors.append(f"  [{base_idx + b}] {key}: contains {n} NaN values")

    # rel_pos range check — vectorized per sample
    for key in REL_POS_KEYS:
        if key not in batch:
            continue
        t = batch[key]                                          # (B, N)
        valid = batch[key.replace("rel_pos", "valid")]          # (B, N)
        for b in range(bs):
            rp_valid = t[b][valid[b]]
            if rp_valid.numel() > 0:
                mn, mx = rp_valid.min().item(), rp_valid.max().item()
                if mn < 0 or mx > 364:
                    errors.append(f"  [{base_idx + b}] {key}: rel_pos out of [0,364]: min={mn} max={mx}")

    # Label stats
    lbl = batch["label"].float()                                # (B, 3)
    n_all_nan = int(torch.isnan(lbl).all(dim=1).sum().item())
    n_has_obs = bs - n_all_nan

    return errors, n_all_nan, n_has_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-samples",  type=int, default=None,
                        help="Check a random subset of this size (default: all samples)")
    parser.add_argument("--category",   type=str, default=None,
                        help="One of sm_only, sm_and_flux, flux_only (None = all)")
    parser.add_argument("--split",      type=str, default=None,
                        help="One of train, val, test (None = train+val)")
    args = parser.parse_args()

    category_filter = [args.category] if args.category else None
    split_names     = [args.split] if args.split else ["train", "val"]

    common_kwargs = dict(
        splits_csv      = SPLITS_CSV,
        era5_stats_path = ERA5_STATS,
        years           = list(range(2016, 2023)),  # 2023 held out for OOT/OOST evaluation
        category_filter = category_filter,
    )

    grand_errors = []

    for split_name in split_names:
        training_flag = split_name == "train"
        print(f"Building dataset ({split_name})...")
        ds = SoilMoistureDataset(**common_kwargs, split_filter=[split_name], training=training_flag)
        n = len(ds)
        print(f"Dataset: {n} samples from {len(ds._zarr_groups)} stations")
        print(f"Dataset ready: {n} samples")

        if args.n_samples is not None and args.n_samples < n:
            import random
            indices = random.sample(range(n), args.n_samples)
            ds = torch.utils.data.Subset(ds, indices)
            n = args.n_samples
            print(f"Checking {n} random samples (subset) with {args.workers} workers, batch_size={args.batch_size}...\n")
        else:
            print(f"Checking all {n} samples with {args.workers} workers, batch_size={args.batch_size}...\n")

        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=False, prefetch_factor=2)

        all_errors  = []
        n_checked   = 0
        label_stats = {"all_nan": 0, "has_obs": 0}

        for batch in loader:
            errs, n_nan, n_obs = check_batch(batch, n_checked)
            all_errors.extend(errs)
            label_stats["all_nan"] += n_nan
            label_stats["has_obs"] += n_obs
            n_checked += next(v for v in batch.values() if isinstance(v, torch.Tensor)).shape[0]

            if n_checked % 50000 == 0:
                print(f"  {n_checked}/{n} checked  errors so far: {len(all_errors)}")

        print(f"\n{'='*60}")
        print(f"Split           : {split_name}")
        print(f"Samples checked : {n_checked}")
        print(f"Label all-NaN   : {label_stats['all_nan']}  (no SM obs that day — fine)")
        print(f"Label has obs   : {label_stats['has_obs']}")
        print(f"Total errors    : {len(all_errors)}")

        if all_errors:
            print(f"\nERRORS in {split_name}:")
            for e in all_errors[:50]:
                print(e)
            if len(all_errors) > 50:
                print(f"  ... and {len(all_errors)-50} more")
        else:
            print(f"\n{split_name} PASSED — clean.")

        grand_errors.extend(all_errors)
        print()

    print(f"{'='*60}")
    if grand_errors:
        print(f"OVERALL: {len(grand_errors)} errors found across all splits.")
    else:
        print("OVERALL: All samples PASSED — dataset is clean.")


if __name__ == "__main__":
    main()
