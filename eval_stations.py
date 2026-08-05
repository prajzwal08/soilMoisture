"""Standalone per-station validation using a saved checkpoint.

Usage:
    python eval_stations.py --run-name baseline_huber [--ckpt last.pt]
"""
import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel, masked_huber_loss, total_variation_loss
from train import compute_metrics, evaluate, _per_depth_mean, _loss_aggregates

CKPT_ROOT = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")

def worker_init_fn(worker_id):
    np.random.seed(os.getpid() + worker_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ckpt",     default="last.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_ROOT / args.run_name / args.ckpt
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"Epoch: {ckpt['epoch']}  val_loss: {ckpt.get('val_loss', 'N/A')}")

    # Build val dataset (no shm preload — only 74 val stations, L12 fits in RAM directly)
    common_kwargs = dict(
        splits_csv      = config["splits_csv"],
        shm_dir         = None,
        category_filter = config.get("category_filter"),
        era5_stats_path = config.get("era5_stats"),
    )
    val_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["val"], training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        worker_init_fn = worker_init_fn,
        persistent_workers = False,
        prefetch_factor = 2,
    )
    print(f"Val dataset: {len(val_dataset)} samples from {val_dataset.n_stations} stations")

    # Build model and load weights
    model = SoilMoistureModel(
        n_depths = config["n_depths"],
        n_layers = config["n_layers"],
        d_model  = config["d_model"],
        n_heads  = config["n_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Run evaluation
    val_loss, metrics, per_station, depth_sum, depth_cnt = evaluate(
        model, val_loader, device
    )
    depth_loss     = _per_depth_mean(depth_sum, depth_cnt)
    pooled, dmean  = _loss_aggregates(depth_sum, depth_cnt)

    print(f"\n=== Global metrics (epoch {ckpt['epoch']}) ===")
    print(f"val_loss = {val_loss:.6f}   pooled_huber = {pooled:.6f}   depth_mean = {dmean:.6f}")
    for depth, m in metrics.items():
        print(f"  {depth:>8s}  loss={depth_loss[depth]:.6f}  "
              f"MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"ubRMSE={m['ubRMSE']:.4f}  bias={m['bias']:.4f}")

    surface = SM_DEPTHS[0]
    print(f"\n=== Per-station results sorted by {surface} ubRMSE ===")
    ranked = sorted(
        [(st, v) for st, v in per_station.items() if surface in v],
        key=lambda x: x[1][surface]["ubRMSE"], reverse=True,
    )
    print(f"{'Station':<55} {'ubRMSE':>8} {'MAE':>8} {'bias':>8} {'n':>6}")
    print("-" * 90)
    for st, v in ranked:
        m = v[surface]
        print(f"{st:<55} {m['ubRMSE']:>8.4f} {m['MAE']:>8.4f} {m['bias']:>8.4f} {m['n']:>6d}")

if __name__ == "__main__":
    main()
