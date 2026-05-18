"""
Training script for SoilMoistureModel.

Usage (geo conda env):
    conda run -n geo python train.py

Key paths and splits are configured in the CONFIG dict below.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel, masked_huber_loss

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "metadata_csv"  : "/home/khanalp/data/soilmoisture/level1/station_metadata.csv",
    "satellite_dir" : "/home/khanalp/data/satellite",
    "ismn_dir"      : "/home/khanalp/data/soilmoisture/level1",
    "checkpoint_dir": "/home/khanalp/data/checkpoints/soilmoisture",

    # Data
    "years"         : list(range(2016, 2024)),
    "val_fraction"  : 0.15,       # fraction of stations held out for validation (OOS)
    "seed"          : 42,

    # Training
    "batch_size"    : 4,
    "num_workers"   : 4,
    "max_epochs"    : 100,
    "lr"            : 1e-4,
    "weight_decay"  : 0.05,
    "lr_patience"   : 10,         # ReduceLROnPlateau patience
    "lr_factor"     : 0.5,
    "grad_clip"     : 1.0,
    "early_stop_patience": 20,

    # Model
    "freeze_terramind": True,     # set False to fine-tune TerraMind
    "n_depths"        : 4,
    "d_model"         : 768,
    "n_heads"         : 12,
    "n_layers"        : 6,
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_val_split(dataset: SoilMoistureDataset, val_fraction: float, seed: int):
    """
    Out-of-space split: hold out val_fraction of stations for validation.
    All samples from a given station go entirely to train or val.
    """
    all_stations = list({s["station_key"] for s in dataset.samples})
    rng          = random.Random(seed)
    rng.shuffle(all_stations)

    n_val      = max(1, int(len(all_stations) * val_fraction))
    val_stns   = set(all_stations[:n_val])
    train_stns = set(all_stations[n_val:])

    train_idx = [i for i, s in enumerate(dataset.samples) if s["station_key"] in train_stns]
    val_idx   = [i for i, s in enumerate(dataset.samples) if s["station_key"] in val_stns]

    print(f"Train: {len(train_stns)} stations, {len(train_idx)} samples")
    print(f"Val  : {len(val_stns)} stations,   {len(val_idx)} samples")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


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

def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}

        sm_map = model(batch)                    # (B, n_depths, 224, 224)
        loss   = masked_huber_loss(sm_map, batch["label"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []

    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL

    for batch in loader:
        batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}

        sm_map = model(batch)
        loss   = masked_huber_loss(sm_map, batch["label"])
        total_loss += loss.item()
        n_batches  += 1

        pred_at_station = sm_map[:, :, SROW, SCOL].cpu().numpy()  # (B, n_depths)
        all_preds.append(pred_at_station)
        all_targets.append(batch["label"].cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets)

    return total_loss / max(n_batches, 1), metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────
    print("Building dataset...")
    dataset = SoilMoistureDataset(
        metadata_csv  = CONFIG["metadata_csv"],
        satellite_dir = CONFIG["satellite_dir"],
        ismn_dir      = CONFIG["ismn_dir"],
        years         = CONFIG["years"],
    )

    train_set, val_set = train_val_split(dataset, CONFIG["val_fraction"], CONFIG["seed"])

    train_loader = DataLoader(
        train_set,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("Building model...")
    model = SoilMoistureModel(
        n_depths         = CONFIG["n_depths"],
        d_model          = CONFIG["d_model"],
        n_heads          = CONFIG["n_heads"],
        n_layers         = CONFIG["n_layers"],
        freeze_terramind = CONFIG["freeze_terramind"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────
    # Do not apply weight decay to bias / LayerNorm parameters
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
            model, train_loader, optimizer, device, CONFIG["grad_clip"]
        )
        val_loss, metrics = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        # Print metrics
        print(f"\nEpoch {epoch:03d}  |  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        for depth, m in metrics.items():
            print(f"  {depth:>8s}  MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"ubRMSE={m['ubRMSE']:.4f}  bias={m['bias']:.4f}")

        # Checkpoint
        state = {
            "epoch"      : epoch,
            "model"      : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "val_loss"   : val_loss,
            "config"     : CONFIG,
        }
        torch.save(state, ckpt_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            no_improve_count = 0
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  ✓ New best val_loss={best_val_loss:.4f} — checkpoint saved")
        else:
            no_improve_count += 1
            if no_improve_count >= CONFIG["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for "
                      f"{CONFIG['early_stop_patience']} epochs)")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
