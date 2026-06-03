"""
Dataset stress test — iterate N random samples and verify shapes, dtypes, NaN counts.
Run before submitting training to confirm __getitem__ is clean across all code paths.

Usage:
    python check_dataset.py [--n-samples 500] [--workers 4]
"""

import argparse
import random
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS, MAX_S2, MAX_S1, MAX_SIF, MAX_TWSA

SPLITS_CSV   = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv"
DATA_ROOT    = "/gpfs/work3/0/prjs1968/data"
ERA5_STATS   = "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json"

# Expected shapes for every key returned by __getitem__
EXPECTED = {
    "s2_l12"        : (MAX_S2,  196, 768),
    "s2_doys"       : (MAX_S2,),
    "s2_valid"      : (MAX_S2,),
    "s2_token_mask" : (MAX_S2, 14, 14),
    "s2_rel_pos"    : (MAX_S2,),
    "s1_l12"        : (MAX_S1,  196, 768),
    "s1_doys"       : (MAX_S1,),
    "s1_valid"      : (MAX_S1,),
    "s1_rel_pos"    : (MAX_S1,),
    "dem_l12"       : (196, 768),
    "lulc_l12"      : (196, 768),
    "skip_l3"       : (196, 768),
    "skip_l6"       : (196, 768),
    "skip_l9"       : (196, 768),
    "recent_is_s1"  : (),
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
    "skip_l3", "skip_l6", "skip_l9",
    "soil_patch", "era5", "sif", "twsa",
]

# rel_pos must be in [0, 364]
REL_POS_KEYS = ["s2_rel_pos", "s1_rel_pos"]


def check_sample(sample: dict, idx: int) -> list[str]:
    errors = []

    for key, expected_shape in EXPECTED.items():
        if key not in sample:
            errors.append(f"  [{idx}] MISSING key: {key}")
            continue

        t = sample[key]
        if not isinstance(t, torch.Tensor):
            errors.append(f"  [{idx}] {key}: expected Tensor, got {type(t)}")
            continue

        if tuple(t.shape) != expected_shape:
            errors.append(f"  [{idx}] {key}: shape {tuple(t.shape)} != expected {expected_shape}")

    for key in NO_NAN_KEYS:
        if key in sample:
            t = sample[key].float()
            if torch.isnan(t).any():
                n = torch.isnan(t).sum().item()
                errors.append(f"  [{idx}] {key}: contains {n} NaN values")

    for key in REL_POS_KEYS:
        if key in sample:
            t = sample[key]
            valid_mask = sample[key.replace("rel_pos", "valid")]
            rp_valid = t[valid_mask]
            if rp_valid.numel() > 0:
                mn, mx = rp_valid.min().item(), rp_valid.max().item()
                if mn < 0 or mx > 364:
                    errors.append(f"  [{idx}] {key}: rel_pos out of [0,364]: min={mn} max={mx}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--workers",   type=int, default=4)
    args = parser.parse_args()

    common_kwargs = dict(
        splits_csv      = SPLITS_CSV,
        data_root       = DATA_ROOT,
        era5_stats_path = ERA5_STATS,
        years           = list(range(2016, 2024)),
        category_filter = ["sm_only"],
    )

    splits = [("train", True), ("val", False)]
    grand_errors = []

    for split_name, training_flag in splits:
        print(f"Building dataset ({split_name})...")
        ds = SoilMoistureDataset(**common_kwargs, split_filter=[split_name], training=training_flag)
        print(f"Dataset ready: {len(ds)} samples")

        n = min(args.n_samples, len(ds))
        indices = random.sample(range(len(ds)), n)
        print(f"Checking {n} random samples with {args.workers} DataLoader workers...\n")

        from torch.utils.data import Subset
        subset = Subset(ds, indices)
        loader = DataLoader(subset, batch_size=1, num_workers=args.workers,
                            shuffle=False, pin_memory=False)

        all_errors  = []
        n_checked   = 0
        label_stats = {"all_nan": 0, "has_obs": 0}

        for i, batch in enumerate(loader):
            sample = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}

            errs = check_sample(sample, indices[i])
            all_errors.extend(errs)
            n_checked += 1

            lbl = sample["label"]
            if torch.isnan(lbl).all():
                label_stats["all_nan"] += 1
            else:
                label_stats["has_obs"] += 1

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n} checked  errors so far: {len(all_errors)}")

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
