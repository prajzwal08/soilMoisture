"""
Training script for SoilMoistureModel — Phase 1 (sm_only).

Usage (terramind conda env):
    python train.py [--lr LR] [--batch-size N] [--n-layers N] [--run-name NAME]
                    [--max-stations N]   # smoke-test: limit to N stations

Key paths and hyperparameters are in the CONFIG dict below.
W&B project: soil-moisture-phd
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel, masked_huber_loss, masked_nll_loss

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "splits_csv"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    "data_root"     : "/gpfs/work3/0/prjs1968/data",
    "era5_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json",
    "checkpoint_dir": "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only",

    # Data
    "category_filter": ["sm_only"],
    "years"          : list(range(2016, 2024)),
    "seed"           : 42,

    # Training
    "batch_size"    : 4,
    "num_workers"   : 8,
    "max_epochs"    : 100,
    "lr"            : 1e-4,
    "weight_decay"  : 0.05,
    "lr_patience"   : 10,
    "lr_factor"     : 0.5,
    "grad_clip"     : 1.0,
    "early_stop_patience": 20,

    # Model
    "n_depths"           : 3,
    "d_model"            : 768,
    "n_heads"            : 12,
    "n_layers"           : 6,
    "predict_uncertainty": True,

    # Loss: "nll" (Gaussian NLL, aleatoric uncertainty) or "huber"
    "loss_fn" : "nll",

    # W&B
    "wandb_project": "soil-moisture-phd",
    "run_name"     : "baseline_nll",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker independently so RNG state is not duplicated across workers."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def compute_metrics(preds, targets):
    """
    preds, targets : (N, n_depths) numpy arrays (NaN masked already)
    Returns dict of MSE, MAE, ubRMSE per depth.
    """
    metrics = {}
    for i, depth in enumerate(SM_DEPTHS):
        p = preds[:, i]
        t = targets[:, i]
        mask = ~(np.isnan(p) | np.isnan(t))
        if mask.sum() == 0:
            continue
        p, t = p[mask], t[mask]
        mse    = np.mean((p - t) ** 2)
        mae    = np.mean(np.abs(p - t))
        bias   = np.mean(p - t)
        ubrmse = np.sqrt(np.mean(((p - p.mean()) - (t - t.mean())) ** 2))
        metrics[depth] = {"MSE": mse, "MAE": mae, "ubRMSE": ubrmse, "bias": bias}
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def _compute_loss(mu, log_var, label, loss_fn):
    if loss_fn == "nll":
        return masked_nll_loss(mu, log_var, label)
    return masked_huber_loss(mu, label)


def train_one_epoch(model, loader, optimizer, device, grad_clip, loss_fn):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        mu, log_var = model(batch)
        loss = _compute_loss(mu, log_var, batch["label"], loss_fn)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    all_preds    = []
    all_targets  = []
    all_sigmas   = []   # mean σ at station pixel per batch (NLL mode only)

    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        mu, log_var = model(batch)
        loss = _compute_loss(mu, log_var, batch["label"], loss_fn)
        total_loss += loss.item()
        n_batches  += 1

        all_preds.append(mu[:, :, SROW, SCOL].cpu().numpy())
        all_targets.append(batch["label"].cpu().numpy())

        if log_var is not None:
            sigma = (0.5 * log_var[:, :, SROW, SCOL]).exp().cpu().numpy()
            all_sigmas.append(sigma)

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets)

    mean_sigma = float(np.concatenate(all_sigmas).mean()) if all_sigmas else None
    return total_loss / max(n_batches, 1), metrics, mean_sigma


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── CLI overrides ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--n-layers",     type=int,   default=None)
    parser.add_argument("--run-name",     type=str,   default=None)
    parser.add_argument("--loss-fn",      type=str,   default=None,
                        choices=["nll", "huber"])
    parser.add_argument("--max-stations", type=int,   default=None,
                        help="Limit dataset to N stations (smoke-test mode)")
    args = parser.parse_args()

    if args.lr          is not None: CONFIG["lr"]         = args.lr
    if args.batch_size  is not None: CONFIG["batch_size"] = args.batch_size
    if args.n_layers    is not None: CONFIG["n_layers"]   = args.n_layers
    if args.run_name    is not None: CONFIG["run_name"]   = args.run_name
    if args.loss_fn     is not None: CONFIG["loss_fn"]    = args.loss_fn

    # Sync predict_uncertainty with loss_fn: Huber never needs uncertainty head
    if CONFIG["loss_fn"] == "huber":
        CONFIG["predict_uncertainty"] = False

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(
            project = CONFIG["wandb_project"],
            name    = CONFIG["run_name"],
            config  = {k: v for k, v in CONFIG.items()
                       if not k.endswith("_dir") and not k.endswith("_csv")
                       and not k.endswith("_stats") and k != "wandb_project"},
        )
        use_wandb = True
    except Exception as e:
        print(f"W&B disabled: {e}")
        use_wandb = False

    # ── Datasets (pre-defined splits from station_splits.csv) ─────────
    print("Building datasets...")
    common_kwargs = dict(
        splits_csv       = CONFIG["splits_csv"],
        data_root        = CONFIG["data_root"],
        era5_stats_path  = CONFIG["era5_stats"],
        years            = CONFIG["years"],
        category_filter  = CONFIG["category_filter"],
    )
    train_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["train"], training=True)
    val_dataset   = SoilMoistureDataset(**common_kwargs, split_filter=["val"],   training=False)

    # Smoke-test: limit to first N stations
    if args.max_stations is not None:
        def _limit(ds, n):
            seen = set()
            idx  = []
            for i, s in enumerate(ds.samples):
                seen.add(s["station_key"])
                if len(seen) > n:
                    break
                idx.append(i)
            from torch.utils.data import Subset
            return Subset(ds, idx)
        train_dataset = _limit(train_dataset, args.max_stations)
        val_dataset   = _limit(val_dataset,   max(1, args.max_stations // 5))

    train_loader = DataLoader(
        train_dataset,
        batch_size      = CONFIG["batch_size"],
        shuffle         = True,
        num_workers     = CONFIG["num_workers"],
        pin_memory      = True,
        drop_last       = True,
        worker_init_fn  = worker_init_fn,
        persistent_workers = CONFIG["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size      = CONFIG["batch_size"],
        shuffle         = False,
        num_workers     = CONFIG["num_workers"],
        pin_memory      = True,
        worker_init_fn  = worker_init_fn,
        persistent_workers = CONFIG["num_workers"] > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("Building model...")
    model = SoilMoistureModel(
        n_depths            = CONFIG["n_depths"],
        d_model             = CONFIG["d_model"],
        n_heads             = CONFIG["n_heads"],
        n_layers            = CONFIG["n_layers"],
        predict_uncertainty = CONFIG["predict_uncertainty"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and "bias" not in n and "norm" not in n.lower()]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ("bias" in n or "norm" in n.lower())]
    optimizer = AdamW(
        [{"params": decay_params},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr           = CONFIG["lr"],
        weight_decay = CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor   = CONFIG["lr_factor"],
        patience = CONFIG["lr_patience"],
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss    = float("inf")
    no_improve_count = 0

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, CONFIG["grad_clip"], CONFIG["loss_fn"]
        )
        val_loss, metrics, mean_sigma = evaluate(model, val_loader, device, CONFIG["loss_fn"])
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch:03d}  |  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        for depth, m in metrics.items():
            print(f"  {depth:>8s}  MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"ubRMSE={m['ubRMSE']:.4f}  bias={m['bias']:.4f}")

        if use_wandb:
            log_dict = {
                "epoch"      : epoch,
                "train/loss" : train_loss,
                "val/loss"   : val_loss,
                "lr"         : optimizer.param_groups[0]["lr"],
            }
            for depth, m in metrics.items():
                log_dict[f"val/{depth}/ubRMSE"] = m["ubRMSE"]
                log_dict[f"val/{depth}/MAE"]    = m["MAE"]
                log_dict[f"val/{depth}/bias"]   = m["bias"]
            if mean_sigma is not None:
                log_dict["val/mean_sigma"] = mean_sigma
            wandb.log(log_dict)

        # Checkpoint
        state = {
            "epoch"     : epoch,
            "model"     : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "val_loss"  : val_loss,
            "config"    : CONFIG,
        }
        torch.save(state, ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            no_improve_count = 0
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  New best val_loss={best_val_loss:.4f} — checkpoint saved")
        else:
            no_improve_count += 1
            if no_improve_count >= CONFIG["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {CONFIG['early_stop_patience']} epochs)")
                break

    if use_wandb:
        wandb.finish()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
