"""
Training script for SoilMoistureModel — Phase 1 (sm_only).

Usage (terramind conda env):
    python train.py [--lr LR] [--batch-size N] [--n-layers N] [--run-name NAME]
                    [--loss-fn nll|huber] [--max-stations N]

Resume behaviour: if {checkpoint_dir}/{run_name}/last.pt exists the run
resumes automatically — no flag needed. Delete last.pt for a fresh start.

W&B project: soil-moisture-phd
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel, masked_huber_loss, masked_nll_loss, total_variation_loss

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "splits_csv"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    "era5_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json",
    # Each run saves checkpoints under {checkpoint_dir}/{run_name}/
    "checkpoint_dir": "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only",

    # Data
    "category_filter": ["sm_only"],
    "years"          : list(range(2016, 2023)),  # 2023 held out for OOT/OOST evaluation
    "seed"           : 42,

    # Training
    "batch_size"    : 64,
    "num_workers"   : 8,
    "prefetch_factor": 4,
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
    "loss_fn"   : "nll",
    "lambda_tv" : 0.1,     # TV regularization weight (0 = disabled)

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


def compute_metrics(preds, targets, station_keys):
    """
    preds, targets : (N, n_depths) numpy arrays
    station_keys   : (N,) array-like of per-sample station identifiers
    Returns dict of MSE, MAE, ubRMSE, bias per depth.

    ubRMSE removes each station's own temporal mean before computing RMSE
    (the standard unbiased-RMSE definition) -- a global mean across all
    stations would otherwise leave cross-station bias in the result.
    """
    station_keys = np.asarray(station_keys)
    metrics = {}
    for i, depth in enumerate(SM_DEPTHS):
        p = preds[:, i]
        t = targets[:, i]
        mask = ~(np.isnan(p) | np.isnan(t))
        if mask.sum() == 0:
            continue
        p, t, sk = p[mask], t[mask], station_keys[mask]
        bias   = float(np.mean(p - t))
        mae    = float(np.mean(np.abs(p - t)))
        mse    = float(np.mean((p - t) ** 2))

        p_anom = np.empty_like(p)
        t_anom = np.empty_like(t)
        for station in np.unique(sk):
            sel = sk == station
            p_anom[sel] = p[sel] - p[sel].mean()
            t_anom[sel] = t[sel] - t[sel].mean()
        ubrmse = float(np.sqrt(np.mean((p_anom - t_anom) ** 2)))

        metrics[depth] = {"MSE": mse, "MAE": mae, "ubRMSE": ubrmse, "bias": bias}
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def _compute_loss(mu, var, label, loss_fn, lambda_tv=0.0):
    if loss_fn == "nll":
        loss = masked_nll_loss(mu, var, label)
    else:
        loss = masked_huber_loss(mu, label)
    tv = total_variation_loss(mu)
    if lambda_tv > 0.0:
        loss = loss + lambda_tv * tv
    return loss, tv.detach()


def _scan_for_nan(tensors: dict, exclude=()) -> dict:
    """Return {key: [bad sample indices]} for float tensors containing NaN/Inf."""
    bad = {}
    for k, v in tensors.items():
        if k in exclude or v is None or not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        bad_mask = torch.isnan(v) | torch.isinf(v)
        if bad_mask.any():
            per_sample = bad_mask.reshape(v.shape[0], -1).any(dim=1)
            bad[k] = torch.where(per_sample)[0].tolist()
    return bad


def _report_nan(tag, batch, bad):
    for k, idx_list in bad.items():
        for i in idx_list[:5]:
            station = batch["station_key"][i] if "station_key" in batch else "?"
            year    = batch["year"][i].item()  if "year" in batch else "?"
            doy     = batch["doy"][i].item()   if "doy" in batch else "?"
            print(f"  [NaN DEBUG] {tag}: {k}[{i}] station={station} year={year} doy={doy}")


def train_one_epoch(model, loader, optimizer, device, grad_clip, loss_fn, lambda_tv=0.0,
                     max_batches=None, debug_nan=False):
    model.train()
    total_loss = 0.0
    total_tv   = 0.0
    n_batches  = 0
    t_prev     = time.perf_counter()

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if debug_nan:
            bad_in = _scan_for_nan(batch, exclude={"label"})
            if bad_in:
                _report_nan(f"batch {n_batches+1:03d} INPUT", batch, bad_in)

        mu, var = model(batch)

        if debug_nan:
            bad_out = _scan_for_nan({"mu": mu, "var": var})
            if bad_out:
                _report_nan(f"batch {n_batches+1:03d} OUTPUT", batch, bad_out)

        loss, tv = _compute_loss(mu, var, batch["label"], loss_fn, lambda_tv)

        optimizer.zero_grad()
        loss.backward()

        if debug_nan:
            bad_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                           for p in model.parameters())
            if bad_grad:
                print(f"  [NaN DEBUG] batch {n_batches+1:03d}: NaN/Inf in gradients")

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if debug_nan:
            bad_param = any(torch.isnan(p).any() for p in model.parameters())
            if bad_param:
                print(f"  [NaN DEBUG] batch {n_batches+1:03d}: NaN parameters after optimizer.step()")

        total_loss += loss.item()
        total_tv   += tv.item()
        n_batches  += 1

        t_now = time.perf_counter()
        print(f"  batch {n_batches:03d}  loss={loss.item():.4f}  tv={tv.item():.5f}  step={1000*(t_now - t_prev):.0f}ms")
        t_prev = t_now

    n = max(n_batches, 1)
    return total_loss / n, total_tv / n


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, max_batches=None):
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []
    all_sigmas  = []
    all_station_keys = []

    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        mu, var = model(batch)
        loss, _ = _compute_loss(mu, var, batch["label"], loss_fn)
        total_loss += loss.item()
        n_batches  += 1

        all_preds.append(mu[:, :, SROW, SCOL].cpu().numpy())
        all_targets.append(batch["label"].cpu().numpy())
        all_station_keys.extend(batch["station_key"])

        if var is not None:
            sigma = var[:, :, SROW, SCOL].sqrt().cpu().numpy()
            all_sigmas.append(sigma)

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets, all_station_keys)

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
    parser.add_argument("--lambda-tv",   type=float, default=None,
                        help="TV regularization weight (default 0.001; 0 to disable)")
    parser.add_argument("--max-stations", type=int,   default=None,
                        help="Limit dataset to N stations (smoke-test mode)")
    parser.add_argument("--max-epochs",   type=int,   default=None,
                        help="Override max_epochs (smoke-test mode)")
    parser.add_argument("--num-workers",     type=int, default=None,
                        help="Override DataLoader num_workers (smoke-test mode)")
    parser.add_argument("--prefetch-factor", type=int, default=None,
                        help="Override DataLoader prefetch_factor (smoke-test mode)")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Limit batches per training epoch (trial/debug mode)")
    parser.add_argument("--max-val-batches",   type=int, default=None,
                        help="Limit batches during validation (trial/debug mode)")
    parser.add_argument("--debug-nan", action="store_true",
                        help="Scan inputs/outputs/grads/params for NaN/Inf each batch (slow)")
    args = parser.parse_args()

    if args.lr          is not None: CONFIG["lr"]         = args.lr
    if args.batch_size  is not None: CONFIG["batch_size"] = args.batch_size
    if args.n_layers    is not None: CONFIG["n_layers"]   = args.n_layers
    if args.run_name    is not None: CONFIG["run_name"]   = args.run_name
    if args.num_workers     is not None: CONFIG["num_workers"]     = args.num_workers
    if args.prefetch_factor is not None: CONFIG["prefetch_factor"] = args.prefetch_factor
    if args.loss_fn     is not None: CONFIG["loss_fn"]    = args.loss_fn
    if args.max_epochs  is not None: CONFIG["max_epochs"] = args.max_epochs
    if args.lambda_tv   is not None: CONFIG["lambda_tv"]  = args.lambda_tv

    if CONFIG["loss_fn"] == "huber":
        CONFIG["predict_uncertainty"] = False

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Each run gets its own subdirectory so runs never clobber each other's checkpoints
    ckpt_dir = Path(CONFIG["checkpoint_dir"]) / CONFIG["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────
    print("Building datasets...")
    common_kwargs = dict(
        splits_csv       = CONFIG["splits_csv"],
        era5_stats_path  = CONFIG["era5_stats"],
        years            = CONFIG["years"],
        category_filter  = CONFIG["category_filter"],
    )
    val_max_stations = max(1, args.max_stations // 5) if args.max_stations is not None else None
    train_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["train"], training=True,
                                         max_stations=args.max_stations)
    val_dataset   = SoilMoistureDataset(**common_kwargs, split_filter=["val"],   training=False,
                                         max_stations=val_max_stations)

    train_loader = DataLoader(
        train_dataset,
        batch_size         = CONFIG["batch_size"],
        shuffle            = True,
        num_workers        = CONFIG["num_workers"],
        pin_memory         = True,
        drop_last          = True,
        worker_init_fn     = worker_init_fn,
        persistent_workers = CONFIG["num_workers"] > 0,
        prefetch_factor    = CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size         = CONFIG["batch_size"],
        shuffle            = False,
        num_workers        = CONFIG["num_workers"],
        pin_memory         = True,
        worker_init_fn     = worker_init_fn,
        persistent_workers = CONFIG["num_workers"] > 0,
        prefetch_factor    = CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None,
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

    # ── Resume from checkpoint (automatic if last.pt exists) ──────────
    start_epoch      = 1
    best_val_loss    = float("inf")
    no_improve_count = 0
    wandb_run_id     = None

    ckpt_last = ckpt_dir / "last.pt"
    if ckpt_last.exists():
        print(f"Checkpoint found — resuming from {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
        no_improve_count = ckpt["no_improve_count"]
        wandb_run_id     = ckpt.get("wandb_run_id")
        print(f"  Resuming from epoch {start_epoch}  "
              f"best_val_loss={best_val_loss:.4f}  no_improve={no_improve_count}")
    else:
        print("No checkpoint found — starting fresh")

    # ── W&B ───────────────────────────────────────────────────────────
    use_wandb = False
    try:
        import wandb
        wandb.init(
            project  = CONFIG["wandb_project"],
            name     = CONFIG["run_name"],
            id       = wandb_run_id,
            resume   = "allow",
            config   = {k: v for k, v in CONFIG.items()
                        if not k.endswith("_dir") and not k.endswith("_csv")
                        and not k.endswith("_stats") and k != "wandb_project"},
        )
        use_wandb = True
    except Exception as e:
        print(f"W&B disabled: {e}")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
        train_loss, train_tv = train_one_epoch(
            model, train_loader, optimizer, device, CONFIG["grad_clip"], CONFIG["loss_fn"],
            lambda_tv=CONFIG["lambda_tv"],
            max_batches=args.max_train_batches, debug_nan=args.debug_nan,
        )
        val_loss, metrics, mean_sigma = evaluate(
            model, val_loader, device, CONFIG["loss_fn"], max_batches=args.max_val_batches
        )
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch:03d}  |  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        for depth, m in metrics.items():
            print(f"  {depth:>8s}  MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"ubRMSE={m['ubRMSE']:.4f}  bias={m['bias']:.4f}")

        if use_wandb:
            log_dict = {
                "epoch"      : epoch,
                "train/loss" : train_loss,
                "train/tv"   : train_tv,   # raw (unweighted) TV for lambda tuning
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

        # ── Checkpoint ────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        state = {
            "epoch"           : epoch,
            "model"           : model.state_dict(),
            "optimizer"       : optimizer.state_dict(),
            "scheduler"       : scheduler.state_dict(),
            "val_loss"        : val_loss,
            "best_val_loss"   : best_val_loss,
            "no_improve_count": no_improve_count,
            "config"          : CONFIG,
            "wandb_run_id"    : wandb.run.id if use_wandb else None,
        }
        torch.save(state, ckpt_last)

        if no_improve_count == 0:
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  New best val_loss={best_val_loss:.4f} — checkpoint saved")

        if no_improve_count >= CONFIG["early_stop_patience"]:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {CONFIG['early_stop_patience']} epochs)")
            break

    if use_wandb:
        wandb.finish()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
