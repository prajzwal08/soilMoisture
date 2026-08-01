"""
Training script for SoilMoistureModel — Phase 1 (sm_only).

Usage (terramind conda env):
    python train.py [--lr LR] [--batch-size N] [--n-layers N] [--run-name NAME]
                    [--max-stations N]

Resume behaviour: if {checkpoint_dir}/{run_name}/last.pt exists the run
resumes automatically — no flag needed. Delete last.pt for a fresh start.

W&B project: soil-moisture-phd
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.multiprocessing

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel, masked_huber_loss, total_variation_loss

# ── Preemption handling ───────────────────────────────────────────────────────
_preempted = False

def _handle_sigterm(signum, frame):
    global _preempted
    _preempted = True

class _Preempted(Exception):
    pass

# ── CUDA prefetcher ───────────────────────────────────────────────────────────

class CudaPrefetcher:
    """Overlaps H2D transfer of batch N+1 with GPU compute on batch N.

    Wraps any DataLoader.  Batches arrive on `device` with tensors already
    transferred; non-tensor fields (station_key, year, doy) pass through as-is.
    """
    def __init__(self, loader, device):
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._iter   = iter(loader)
        self._next   = None
        self._preload()

    def _preload(self):
        try:
            raw = next(self._iter)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self._stream):
            self._next = {
                k: v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream(self._device).wait_stream(self._stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                v.record_stream(torch.cuda.current_stream(self._device))
        self._preload()
        return batch

    def __len__(self):
        return len(self._loader)

# ── /dev/shm L12 preloader ────────────────────────────────────────────────────

def _preload_l12_to_shm(splits_csv: str, category_filter, shm_dir: Path,
                        max_stations: int | None = None) -> None:
    """Rank 0: load all stations' L12 tokens from zarr → /dev/shm tmpfs memmaps.

    All DDP ranks then open the same files via numpy.memmap(mode='r') so the OS
    serves one shared physical copy — cuts node RAM by ~400 GB on the full run.
    max_stations: if set, only preload the first N stations (matches dataset cap,
    so smoke tests don't preload 145 GB of data they'll never use).
    Only train+val splits are preloaded — OOS/test stations are never touched
    during training so preloading them wastes /dev/shm.
    """
    import zarr
    import pandas as pd
    from dataset import ZARR_ROOT

    splits = pd.read_csv(splits_csv)
    if category_filter:
        def _cat(r):
            sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
            fl = str(r.get("has_flux",          "False")).lower() == "true"
            return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")
        splits = splits[splits.apply(_cat, axis=1).isin(category_filter)]
    splits = splits[splits["split"].isin(["train", "val"])]

    n_written = 0
    for _, r in splits.iterrows():
        if not bool(r.get("soil_patch_ok", True)):
            continue
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux",          "False")).lower() == "true"
        cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
        if str(r["source_network"]) == "ISMN":
            dir_name = f"ISMN_{r['network']}_{r['station_name']}"
        else:
            dir_name = f"{r['source_network']}_{r['station_id']}"

        zarr_path = ZARR_ROOT / cat / dir_name
        if not (zarr_path / ".complete").exists():
            continue
        try:
            zg = zarr.open_consolidated(str(zarr_path), mode="r")
        except Exception:
            try:
                zg = zarr.open_group(str(zarr_path), mode="r")
            except Exception:
                continue

        wrote_any = False
        for key in ("s2", "s1_asc", "s1_desc"):
            if f"{key}/l12" not in zg:
                continue
            arr      = zg[f"{key}/l12"][:]
            bin_path = shm_dir / f"{dir_name}__{key}.bin"
            meta_path = shm_dir / f"{dir_name}__{key}.meta.json"
            if bin_path.exists():        # already written (resume case)
                wrote_any = True
                continue
            mm = np.memmap(bin_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
            mm[:] = arr
            del mm                       # flush to tmpfs
            meta_path.write_text(json.dumps({"shape": list(arr.shape),
                                             "dtype": str(arr.dtype)}))
            wrote_any = True
        if wrote_any:
            n_written += 1
            if max_stations is not None and n_written >= max_stations:
                break

    print(f"[SHM] L12 preloaded for {n_written} stations → {shm_dir}")


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
    "batch_size"      : 128,
    "num_workers"     : 8,
    "val_num_workers" : 2,    # val uses 2w×pf2; train uses 8w×pf2
    "prefetch_factor" : 2,
    "max_epochs"      : 100,
    "lr"              : 2e-4,
    "weight_decay"    : 0.05,
    "lr_patience"     : 10,
    "lr_factor"       : 0.5,
    "grad_clip"       : 1.0,
    "early_stop_patience": 20,

    # Model
    "n_depths"      : 3,
    "d_model"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 6,
    "drop_path_rate": 0.1,
    "use_cls_depth" : False,    # if True: per-depth CLS tokens in transformer

    # Loss
    "loss_fn"   : "huber",
    "per_depth_loss" : False,   # if True: equal-weight Huber per depth (vs. pooled)
    "lambda_tv"       : 0.0,    # TV regularization weight (0 = disabled; disabled — TV smooths the 224² map, see Tier-1 verdict)
    "lambda_boundary" : 0.1,    # penalty for SM outside [0, 1]

    # W&B
    "wandb_project": "soil-moisture-phd",
    "run_name"     : "baseline_huber",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker independently so RNG state is not duplicated across workers."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _advise_l369_willneed(*datasets) -> None:
    # Non-blocking: ask the kernel to async-prefetch all L369 memmap pages back into the
    # OS page cache.  Called at every epoch start so pages evicted by competing Snellius
    # jobs between epochs are re-warmed before DataLoader workers start consuming them.
    # NOTE: posix_fadvise is silently ignored by GPFS; madvise(MADV_WILLNEED) on the
    # existing mmap region goes through the Linux VM layer and IS honoured on most GPFS
    # versions — verify empirically if jitter persists.
    # ValueError covers the closed-mmap edge case; OSError covers EINVAL from some GPFS
    # striping configurations.
    import mmap as _mmap
    for ds in datasets:
        for arr_dict in ds._l369_cache.values():
            for arr in arr_dict.values():
                if hasattr(arr, '_mmap') and arr._mmap is not None:
                    try:
                        arr._mmap.madvise(_mmap.MADV_WILLNEED)
                    except (OSError, ValueError):
                        pass


def _fsync_save(obj, path):
    """torch.save + fsync: forces GPFS client to flush to storage server immediately."""
    torch.save(obj, path)
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _log_mem_snapshot(label: str, device, is_main: bool,
                      use_wandb: bool = False, epoch: int | None = None,
                      log_dict: dict | None = None):
    """Print RAM/CPU/per-GPU VRAM snapshot; optionally emit to W&B log_dict."""
    if not is_main:
        return
    lines = [f"\n=== Memory snapshot: {label} ==="]
    if psutil is not None:
        vm           = psutil.virtual_memory()
        ram_used_gb  = (vm.total - vm.available) / 1e9
        ram_total_gb = vm.total / 1e9
        cpu_pct      = psutil.cpu_percent(interval=0.1)
        lines.append(f"  RAM  used : {ram_used_gb:.1f} GB / {ram_total_gb:.1f} GB"
                     f"  ({100 * ram_used_gb / ram_total_gb:.0f}%)")
        lines.append(f"  CPU  util : {cpu_pct:.0f}%")
    else:
        lines.append(f"  RAM / CPU : psutil not installed")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        resv  = torch.cuda.memory_reserved(i) / 1e9
        peak  = torch.cuda.max_memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        lines.append(f"  GPU {i} VRAM: {alloc:.1f} alloc / {resv:.1f} rsv /"
                     f" {peak:.1f} peak / {total:.0f} GB total")
    print("\n".join(lines))
    if use_wandb and epoch is not None and log_dict is not None:
        tag = label.replace(" ", "_")
        if psutil is not None:
            log_dict[f"mem/{tag}/ram_used_gb"] = ram_used_gb
            log_dict[f"mem/{tag}/cpu_pct"]     = cpu_pct
        log_dict[f"mem/{tag}/gpu0_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9


def compute_metrics(preds, targets, station_keys, n_worst=5):
    """
    preds, targets : (N, n_depths) numpy arrays
    station_keys   : (N,) array-like of per-sample station identifiers
    Returns (global_metrics, per_station_metrics) where per_station_metrics
    is a dict {station: {depth: {MSE, MAE, ubRMSE, bias}}}.

    ubRMSE removes each station's own temporal mean before computing RMSE
    (the standard unbiased-RMSE definition) -- a global mean across all
    stations would otherwise leave cross-station bias in the result.
    """
    station_keys = np.asarray(station_keys)
    metrics = {}
    per_station = {}  # station -> depth -> metrics

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
        ub_mask = np.zeros(len(p), dtype=bool)
        for station in np.unique(sk):
            sel = sk == station
            if sel.sum() < 2:
                continue
            p_anom[sel] = p[sel] - p[sel].mean()
            t_anom[sel] = t[sel] - t[sel].mean()
            ub_mask[sel] = True
            st_ubrmse = float(np.sqrt(np.mean((p_anom[sel] - t_anom[sel]) ** 2)))
            st_bias   = float(np.mean(p[sel] - t[sel]))
            st_mae    = float(np.mean(np.abs(p[sel] - t[sel])))
            st_mse    = float(np.mean((p[sel] - t[sel]) ** 2))
            if station not in per_station:
                per_station[station] = {}
            per_station[station][depth] = {"ubRMSE": st_ubrmse, "MAE": st_mae, "bias": st_bias,
                                           "MSE": st_mse, "n": int(sel.sum())}
        ubrmse = float(np.sqrt(np.mean((p_anom[ub_mask] - t_anom[ub_mask]) ** 2))) if ub_mask.any() else float("nan")

        metrics[depth] = {"MSE": mse, "MAE": mae, "ubRMSE": ubrmse, "bias": bias}
    return metrics, per_station


# ── Training loop ─────────────────────────────────────────────────────────────

def _compute_loss(pred, label, lambda_tv=0.0, lambda_boundary=0.0, per_depth=False):
    import torch.nn.functional as F
    loss = masked_huber_loss(pred, label, per_depth=per_depth)
    if lambda_tv > 0.0:
        tv = total_variation_loss(pred)
        loss = loss + lambda_tv * tv
    else:
        tv = pred.new_zeros(1)
    if lambda_boundary > 0.0:
        boundary = F.relu(-pred).mean() + F.relu(pred - 1.0).mean()
        loss = loss + lambda_boundary * boundary
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


def make_resume_loader(full_loader, skip_batches, epoch):
    """
    Return a DataLoader starting at batch skip_batches+1 with ZERO disk IO.
    Reproduces DistributedSampler's deterministic index sequence for `epoch`,
    slices off the first skip_batches*batch_size indices, and wraps the rest
    in a new DataLoader — no data files are touched during the skip.
    """
    from torch.utils.data import Sampler as _Sampler

    class _IndexSampler(_Sampler):
        def __init__(self, idx): self._idx = idx
        def __iter__(self):      return iter(self._idx)
        def __len__(self):       return len(self._idx)

    ds  = full_loader.dataset
    bs  = full_loader.batch_size
    sam = full_loader.sampler          # DistributedSampler

    g = torch.Generator()
    g.manual_seed(sam.seed + epoch)   # mirrors DistributedSampler.__iter__
    n          = len(ds)
    indices    = torch.randperm(n, generator=g).tolist()
    total_size = math.ceil(n / sam.num_replicas) * sam.num_replicas
    indices   += indices[:(total_size - n)]                         # pad
    indices    = indices[sam.rank:total_size:sam.num_replicas]      # subsample
    indices    = indices[: len(indices) - (len(indices) % bs)]      # drop_last

    indices = indices[skip_batches * bs:]                           # zero-IO skip

    return DataLoader(
        ds,
        batch_size         = bs,
        sampler            = _IndexSampler(indices),
        num_workers        = full_loader.num_workers,
        pin_memory         = full_loader.pin_memory,
        drop_last          = False,
        worker_init_fn     = full_loader.worker_init_fn,
        persistent_workers = full_loader.persistent_workers,
        prefetch_factor    = full_loader.prefetch_factor if full_loader.num_workers > 0 else None,
    )


def train_one_epoch(model, loader, optimizer, device, grad_clip, lambda_tv=0.0,
                     lambda_boundary=0.0, per_depth=False, max_batches=None,
                     debug_nan=False, skip_batches=0, mid_ckpt_every=500,
                     mid_ckpt_fn=None):
    """Train one epoch.  If skip_batches > 0, fast-forwards past already-done
    batches (data loads but no GPU compute) then resumes training from that
    point.  Calls mid_ckpt_fn(batches_done) every mid_ckpt_every batches so
    rank 0 can save a recovery checkpoint.
    """
    model.train()
    total_loss   = 0.0
    total_tv     = 0.0
    n_batches    = 0
    data_time    = 0.0
    compute_time = 0.0
    t_data_start = time.perf_counter()

    # Loader is pre-sliced by make_resume_loader — no IO skip needed here.
    # skip_batches is kept as a display/checkpoint offset only.
    if skip_batches > 0:
        print(f"  [mid-epoch resume] resuming from batch {skip_batches + 1}")

    for batch in CudaPrefetcher(loader, device):
        if max_batches is not None and n_batches >= max_batches:
            break

        data_time += time.perf_counter() - t_data_start

        if debug_nan:
            bad_in = _scan_for_nan(batch, exclude={"label"})
            if bad_in:
                _report_nan(f"batch {n_batches+1:03d} INPUT", batch, bad_in)

        t_compute = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)

            if debug_nan:
                bad_out = _scan_for_nan({"mu": mu})
                if bad_out:
                    _report_nan(f"batch {n_batches+1:03d} OUTPUT", batch, bad_out)

            loss, tv = _compute_loss(mu, batch["label"], lambda_tv, lambda_boundary, per_depth)

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

        compute_time += time.perf_counter() - t_compute
        total_loss   += loss.item()
        total_tv     += tv.item()
        n_batches    += 1

        step_ms = 1000 * (data_time + compute_time) / n_batches
        print(f"  batch {skip_batches + n_batches:04d}  loss={loss.item():.4f}  tv={tv.item():.5f}  step={step_ms:.0f}ms")

        # Mid-epoch checkpoint every N batches (rank 0 only, via callback)
        if mid_ckpt_fn is not None and mid_ckpt_every > 0 and n_batches % mid_ckpt_every == 0:
            mid_ckpt_fn(skip_batches + n_batches)

        # SIGTERM preemption: save immediately and exit so --requeue restarts cleanly
        if _preempted:
            if mid_ckpt_fn is not None:
                mid_ckpt_fn(skip_batches + n_batches)
            raise _Preempted()

        t_data_start = time.perf_counter()

    n = max(n_batches, 1)
    return total_loss / n, total_tv / n, data_time, compute_time


@torch.no_grad()
def evaluate(model, loader, device, world_size=1, rank=0, max_batches=None, per_depth=False):
    """Distributed-aware evaluation.

    All ranks process their shard in parallel; loss is all_reduced; predictions
    are gathered to rank 0 for metric computation.  Keeps all GPUs active so
    NCCL watchdog never triggers regardless of GPFS latency.
    """
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []
    all_station_keys = []

    SROW = SoilMoistureModel.STATION_ROW
    SCOL = SoilMoistureModel.STATION_COL

    for batch in CudaPrefetcher(loader, device):
        if max_batches is not None and n_batches >= max_batches:
            break

        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
            loss, _ = _compute_loss(mu, batch["label"], per_depth=per_depth)
        total_loss += loss.item()
        n_batches  += 1

        all_preds.append(mu[:, :, SROW, SCOL].float().cpu().numpy())
        all_targets.append(batch["label"].cpu().numpy())
        all_station_keys.extend(batch["station_key"])

    mean_loss = total_loss / max(n_batches, 1)

    if world_size > 1:
        # Average loss across all ranks
        loss_t = torch.tensor(mean_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        mean_loss = loss_t.item()

        # Gather predictions from all ranks to rank 0 (variable-length safe via pickle)
        n_depths = len(SM_DEPTHS)
        local_preds   = np.concatenate(all_preds,   axis=0) if all_preds   else np.empty((0, n_depths))
        local_targets = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, n_depths))
        gathered_preds   = [None] * world_size
        gathered_targets = [None] * world_size
        gathered_keys    = [None] * world_size
        dist.all_gather_object(gathered_preds,   local_preds)
        dist.all_gather_object(gathered_targets, local_targets)
        dist.all_gather_object(gathered_keys,    all_station_keys)

        if rank == 0:
            preds        = np.concatenate(gathered_preds,   axis=0)
            targets      = np.concatenate(gathered_targets, axis=0)
            station_keys = [k for keys in gathered_keys for k in keys]
            metrics, per_station = compute_metrics(preds, targets, station_keys)
        else:
            metrics, per_station = {}, {}
    else:
        preds   = np.concatenate(all_preds,   axis=0)
        targets = np.concatenate(all_targets, axis=0)
        metrics, per_station = compute_metrics(preds, targets, all_station_keys)

    return mean_loss, metrics, per_station


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── CLI overrides ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--n-layers",     type=int,   default=None)
    parser.add_argument("--run-name",     type=str,   default=None)
    parser.add_argument("--lambda-tv",       type=float, default=None,
                        help="TV regularization weight (default 0.001; 0 to disable)")
    parser.add_argument("--lambda-boundary", type=float, default=None,
                        help="Boundary penalty weight for SM ∈ [0,1] (default 0.1; 0 to disable)")
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
    parser.add_argument("--use-memmap", action="store_true",
                        help="Load anchor L3/L6/L9 from .npy memmaps instead of zarr "
                             "(eliminates zstd decompress; OS page cache warms after epoch 1)")
    parser.add_argument("--per-depth-loss", action="store_true",
                        help="Equal-weight Huber per depth (vs. pooled baseline)")
    parser.add_argument("--use-cls-depth", action="store_true",
                        help="Add per-depth CLS tokens to transformer for depth-specific representations")
    args = parser.parse_args()

    if args.lr          is not None: CONFIG["lr"]         = args.lr
    if args.batch_size  is not None: CONFIG["batch_size"] = args.batch_size
    if args.n_layers    is not None: CONFIG["n_layers"]   = args.n_layers
    if args.run_name    is not None: CONFIG["run_name"]   = args.run_name
    if args.num_workers     is not None: CONFIG["num_workers"]     = args.num_workers
    if args.prefetch_factor is not None: CONFIG["prefetch_factor"] = args.prefetch_factor
    if args.max_epochs  is not None: CONFIG["max_epochs"] = args.max_epochs
    if args.lambda_tv       is not None: CONFIG["lambda_tv"]       = args.lambda_tv
    if args.lambda_boundary is not None: CONFIG["lambda_boundary"] = args.lambda_boundary
    if args.per_depth_loss: CONFIG["per_depth_loss"] = True
    if args.use_cls_depth:  CONFIG["use_cls_depth"]  = True

    # ── L12 shared memory preloading (before DDP init to avoid TCPStore timeout) ──
    # Rank 0 reads ~120 GB from GPFS — can take several minutes. Doing this before
    # dist.init_process_group() means ranks 1-3 spin on a sentinel file rather than
    # inside an NCCL communicator setup that times out after 600 s.
    SHM_DIR  = Path(f"/dev/shm/sm_l12_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    _shm_done = SHM_DIR / ".done"
    _pre_rank = int(os.environ.get("RANK", "0"))
    if _pre_rank == 0:
        SHM_DIR.mkdir(parents=True, exist_ok=True)
        t_shm = time.perf_counter()
        _preload_l12_to_shm(CONFIG["splits_csv"], CONFIG.get("category_filter"), SHM_DIR,
                            max_stations=args.max_stations)
        _shm_done.touch()
        print(f"[SHM] Preload done in {time.perf_counter() - t_shm:.1f}s  ({SHM_DIR})")
    else:
        while not _shm_done.exists():
            time.sleep(2)

    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank, rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    set_seed(CONFIG["seed"] + rank)
    if is_main:
        print(f"Device: {device}  |  world_size: {world_size}")

    # Each run gets its own subdirectory so runs never clobber each other's checkpoints
    ckpt_dir = Path(CONFIG["checkpoint_dir"]) / CONFIG["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────
    if is_main:
        print("Building datasets...")
    common_kwargs = dict(
        splits_csv       = CONFIG["splits_csv"],
        era5_stats_path  = CONFIG["era5_stats"],
        years            = CONFIG["years"],
        category_filter  = CONFIG["category_filter"],
        shm_dir          = SHM_DIR,
        use_mmap         = args.use_memmap,
    )
    val_max_stations = max(1, args.max_stations // 5) if args.max_stations is not None else None
    # file_system strategy avoids fd exhaustion with 32 workers; must be set before workers spawn
    torch.multiprocessing.set_sharing_strategy("file_system")

    train_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["train"], training=True,
                                         max_stations=args.max_stations)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                        shuffle=True, drop_last=True) if is_ddp else None

    # Val dataset on all ranks — DistributedSampler splits it across GPUs
    val_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["val"], training=False,
                                       max_stations=val_max_stations)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False, drop_last=False) if is_ddp else None

    # Freeze all Python objects before DataLoader forks workers.
    # Prevents GC from scanning/dirtying CoW-shared cache pages in worker processes,
    # which would cause the kernel to give each worker a private page copy → RSS blowup.
    gc.freeze()

    # IPC budget (pf=2): train 8w×pf2×bs128×~30MB×4r ≈ 240 GB
    #                     val  2w×pf2×bs128×~30MB×4r ≈  60 GB
    # Boundary peak: 145 (shm) + 159 (heaps) + 240 + 60 = 604 GB → 186 GB headroom vs 790G
    train_loader = DataLoader(
        train_dataset,
        batch_size         = CONFIG["batch_size"],
        shuffle            = (train_sampler is None),
        sampler            = train_sampler,
        num_workers        = CONFIG["num_workers"],
        pin_memory         = True,
        drop_last          = True,
        worker_init_fn     = worker_init_fn,
        persistent_workers = True,    # persistent avoids worker respawn race at epoch boundaries
        prefetch_factor    = CONFIG["prefetch_factor"] if CONFIG["num_workers"] > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size         = CONFIG["batch_size"],
        shuffle            = False,
        sampler            = val_sampler,
        num_workers        = CONFIG.get("val_num_workers", 2),
        pin_memory         = True,
        worker_init_fn     = worker_init_fn,
        persistent_workers = CONFIG.get("val_num_workers", 2) > 0,
        prefetch_factor    = CONFIG["prefetch_factor"] if CONFIG.get("val_num_workers", 2) > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────
    if is_main:
        print("Building model...")
    model = SoilMoistureModel(
        n_depths       = CONFIG["n_depths"],
        d_model        = CONFIG["d_model"],
        n_heads        = CONFIG["n_heads"],
        n_layers       = CONFIG["n_layers"],
        drop_path_rate = CONFIG.get("drop_path_rate", 0.1),
        use_cls_depth  = CONFIG.get("use_cls_depth", False),
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_ddp else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
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
    start_epoch       = 1
    best_val_loss     = float("inf")
    no_improve_count  = 0
    wandb_run_id      = None
    val_pending_epoch = None   # epoch whose training is done but val crashed last time
    saved_train_loss  = None
    saved_train_tv    = None

    ckpt_last = ckpt_dir / "last.pt"
    if ckpt_last.exists():
        if is_main:
            print(f"Checkpoint found — resuming from {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location=device, weights_only=False)
        try:
            raw_model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint key mismatch — architecture likely changed. "
                f"Delete {ckpt_last} to start fresh.\nOriginal error: {e}"
            ) from None
        optimizer.load_state_dict(ckpt["optimizer"])
        for pg in optimizer.param_groups:  # honour CONFIG lr even when resuming
            pg["lr"] = CONFIG["lr"]
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss    = ckpt["best_val_loss"]
        no_improve_count = ckpt["no_improve_count"]
        wandb_run_id     = ckpt.get("wandb_run_id")
        if ckpt.get("val_pending"):
            # Training completed but validation crashed — skip training, run val only
            start_epoch       = ckpt["epoch"]
            val_pending_epoch = ckpt["epoch"]
            saved_train_loss  = ckpt.get("train_loss")
            saved_train_tv    = ckpt.get("train_tv")
            if is_main:
                print(f"  Resuming epoch {start_epoch} — training done, validation pending")
        else:
            start_epoch = ckpt["epoch"] + 1
            if is_main:
                print(f"  Resuming from epoch {start_epoch}  "
                      f"best_val_loss={best_val_loss:.4f}  no_improve={no_improve_count}")
    else:
        if is_main:
            print("No checkpoint found — starting fresh")

    # ── Mid-epoch checkpoint (survives node failure mid-epoch) ────────
    # Saved every 500 batches; allows resuming from last saved point
    # rather than repeating the entire epoch.
    skip_batches = 0
    mid_ckpt_path = ckpt_dir / "mid_epoch.pt"
    if mid_ckpt_path.exists() and not val_pending_epoch:
        mc = torch.load(mid_ckpt_path, map_location=device, weights_only=False)
        if mc.get("epoch") == start_epoch:
            raw_model.load_state_dict(mc["model"])
            optimizer.load_state_dict(mc["optimizer"])
            skip_batches = mc.get("batches_done", 0)
            if is_main:
                print(f"  Mid-epoch checkpoint: epoch {start_epoch}, "
                      f"resuming from batch {skip_batches + 1}")

    if is_ddp:
        dist.barrier()

    # ── W&B ───────────────────────────────────────────────────────────
    use_wandb = False
    if is_main:
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

    # ── Memory snapshot (before first epoch) ─────────────────────────
    _log_mem_snapshot("job_start", device, is_main)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        _advise_l369_willneed(train_dataset, val_dataset)
        torch.cuda.reset_peak_memory_stats(device)
        _log_mem_snapshot(f"epoch_{epoch:03d}_start", device, is_main)

        if epoch == val_pending_epoch:
            # Resuming after a val crash — training already completed, reuse saved metrics
            train_loss = saved_train_loss or 0.0
            train_tv   = saved_train_tv   or 0.0
            data_time = compute_time = 0.0
        else:
            # Mid-epoch checkpoint callback (rank 0 only)
            def _save_mid_ckpt(batches_done):
                _fsync_save({
                    "epoch"       : epoch,
                    "model"       : raw_model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "batches_done": batches_done,
                    "best_val_loss"   : best_val_loss,
                    "no_improve_count": no_improve_count,
                    "config"          : CONFIG,
                    "wandb_run_id"    : wandb.run.id if use_wandb else None,
                }, mid_ckpt_path)

            _skip = skip_batches if epoch == start_epoch else 0
            if _skip > 0:
                if rank == 0:
                    print(f"  [mid-epoch resume] fast-forwarding to batch {_skip + 1} (zero IO)...")
                _loader = make_resume_loader(train_loader, _skip, epoch)
            else:
                _loader = train_loader

            try:
                train_loss, train_tv, data_time, compute_time = train_one_epoch(
                    model, _loader, optimizer, device, CONFIG["grad_clip"],
                    lambda_tv=CONFIG["lambda_tv"],
                    lambda_boundary=CONFIG.get("lambda_boundary", 0.0),
                    per_depth=CONFIG["per_depth_loss"],
                    max_batches=args.max_train_batches, debug_nan=args.debug_nan,
                    skip_batches = _skip,
                    mid_ckpt_every = 500,
                    mid_ckpt_fn    = _save_mid_ckpt if is_main else None,
                )
            except _Preempted:
                if is_main:
                    print("[preempt] SIGTERM received — checkpoint saved, exiting for requeue")
                if is_ddp:
                    dist.destroy_process_group()
                raise SystemExit(0)
            _log_mem_snapshot(f"epoch_{epoch:03d}_post_train", device, is_main)

            # Save post-training checkpoint before validation — epoch not lost if val crashes
            if is_main:
                _fsync_save({
                    "epoch"           : epoch,
                    "model"           : raw_model.state_dict(),
                    "optimizer"       : optimizer.state_dict(),
                    "scheduler"       : scheduler.state_dict(),
                    "train_loss"      : train_loss,
                    "train_tv"        : train_tv,
                    "val_loss"        : float("inf"),
                    "best_val_loss"   : best_val_loss,
                    "no_improve_count": no_improve_count,
                    "config"          : CONFIG,
                    "wandb_run_id"    : wandb.run.id if use_wandb else None,
                    "val_pending"     : True,
                }, ckpt_last)

        # All ranks evaluate their shard in parallel — all_reduce inside evaluate()
        # averages the loss across ranks; all_gather_object collects preds to rank 0.
        # No NCCL timeout risk: all GPUs stay active throughout validation.
        val_loss, metrics, per_station = evaluate(
            model if not is_ddp else model.module,
            val_loader, device,
            world_size=world_size, rank=rank,
            max_batches=args.max_val_batches,
            per_depth=CONFIG["per_depth_loss"],
        )
        _log_mem_snapshot(f"epoch_{epoch:03d}_post_val", device, is_main)
        # Release cached-but-free VRAM each epoch.  With expandable_segments:True this
        # cannot unmap segments (by design), so reserved memory will still grow if the
        # driver is fragmenting.  If growth continues, profile with
        # torch.cuda.memory._snapshot() or disable expandable_segments to isolate.
        torch.cuda.empty_cache()
        if is_ddp and epoch != val_pending_epoch:
            # Reduce train_loss and train_tv to rank 0 for accurate global average logging
            t_loss = torch.tensor(train_loss, device=device)
            t_tv   = torch.tensor(train_tv,   device=device)
            dist.reduce(t_loss, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(t_tv,   dst=0, op=dist.ReduceOp.AVG)
            if is_main:
                train_loss = t_loss.item()
                train_tv   = t_tv.item()
        # val_loss already all_reduced inside evaluate() — same on all ranks, no broadcast needed

        scheduler.step(val_loss)

        if is_main:
            peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_util  = (compute_time / max(data_time + compute_time, 1e-6)) * 100
            print(f"\nEpoch {epoch:03d}  |  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                  f"  data={data_time:.0f}s  compute={compute_time:.0f}s"
                  f"  gpu_util={gpu_util:.0f}%  peak_vram={peak_vram:.1f}GB")
            for depth, m in metrics.items():
                print(f"  {depth:>8s}  MSE={m['MSE']:.4f}  MAE={m['MAE']:.4f}  "
                      f"ubRMSE={m['ubRMSE']:.4f}  bias={m['bias']:.4f}")

            # Per-station ubRMSE — all stations, all depths, sorted by surface ubRMSE
            surface_depth = SM_DEPTHS[0]
            if per_station and surface_depth in next(iter(per_station.values()), {}):
                ranked = sorted(
                    per_station.items(),
                    key=lambda x: x[1].get(surface_depth, {}).get("ubRMSE", 0.0),
                    reverse=True,
                )
                # header
                depth_header = "  ".join(f"{d:>8s}" for d in SM_DEPTHS)
                print(f"\n  Per-station ubRMSE  [{depth_header}]")
                for depth in SM_DEPTHS:
                    ubs = [v[depth]["ubRMSE"] for _, v in ranked if depth in v]
                    if ubs:
                        print(f"    {depth:>8s}  station-mean={np.mean(ubs):.4f}  "
                              f"median={np.median(ubs):.4f}  pooled={metrics[depth]['ubRMSE']:.4f}")
                print()
                for st, v in ranked:
                    vals = "  ".join(
                        f"{v[d]['ubRMSE']:8.4f}" if d in v else f"{'N/A':>8s}"
                        for d in SM_DEPTHS
                    )
                    print(f"    {st:50s}  {vals}")

            # Persist per-station metrics to CSV (append per epoch)
            if per_station:
                csv_path = ckpt_dir / "val_station_metrics.csv"
                rows = [
                    {"epoch": epoch, "station": st, "depth": d,
                     "ubRMSE": m["ubRMSE"], "MAE": m["MAE"],
                     "bias": m["bias"], "MSE": m["MSE"], "n": m["n"]}
                    for st, dv in per_station.items()
                    for d, m in dv.items()
                ]
                pd.DataFrame(rows).to_csv(
                    csv_path, mode="a", header=not csv_path.exists(), index=False
                )

            if use_wandb:
                log_dict = {
                    "epoch"        : epoch,
                    "train/loss"   : train_loss,
                    "train/tv"     : train_tv,
                    "val/loss"     : val_loss,
                    "lr"           : optimizer.param_groups[0]["lr"],
                    "perf/data_s"  : data_time,
                    "perf/compute_s": compute_time,
                    "perf/gpu_util": gpu_util,
                    "perf/peak_vram_gb": peak_vram,
                }
                for depth, m in metrics.items():
                    log_dict[f"val/{depth}/ubRMSE"] = m["ubRMSE"]
                    log_dict[f"val/{depth}/MAE"]    = m["MAE"]
                    log_dict[f"val/{depth}/bias"]   = m["bias"]
                # Worst-5 stations per depth
                if per_station:
                    for depth in SM_DEPTHS:
                        worst = sorted(
                            [(st, v[depth]["ubRMSE"]) for st, v in per_station.items() if depth in v],
                            key=lambda x: x[1], reverse=True,
                        )[:5]
                        for i, (st, ub) in enumerate(worst, 1):
                            log_dict[f"val/{depth}/worst{i}_ubRMSE"]  = ub
                            log_dict[f"val/{depth}/worst{i}_station"]  = st
                _log_mem_snapshot(f"epoch_{epoch:03d}_post_val", device, is_main,
                                  use_wandb=True, epoch=epoch, log_dict=log_dict)
                wandb.log(log_dict)

            # ── Checkpoint (post-validation; overwrites the val_pending checkpoint) ──
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            state = {
                "epoch"           : epoch,
                "model"           : raw_model.state_dict(),
                "optimizer"       : optimizer.state_dict(),
                "scheduler"       : scheduler.state_dict(),
                "val_loss"        : val_loss,
                "best_val_loss"   : best_val_loss,
                "no_improve_count": no_improve_count,
                "config"          : CONFIG,
                "wandb_run_id"    : wandb.run.id if use_wandb else None,
                "val_pending"     : False,
            }
            _fsync_save(state, ckpt_last)
            if mid_ckpt_path.exists():
                mid_ckpt_path.unlink()

            if no_improve_count == 0:
                _fsync_save(state, ckpt_dir / "best.pt")
                print(f"  New best val_loss={best_val_loss:.4f} — checkpoint saved")

        # Broadcast early-stop decision to all ranks so none hang at next DDP sync
        stop_flag = torch.tensor(
            int(is_main and no_improve_count >= CONFIG["early_stop_patience"]),
            device=device,
        )
        if is_ddp:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            if is_main:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {CONFIG['early_stop_patience']} epochs)")
            break

    if is_main:
        if use_wandb:
            wandb.finish()
        print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {ckpt_dir}")

    if rank == 0 and SHM_DIR.exists():
        shutil.rmtree(SHM_DIR, ignore_errors=True)
        print(f"[SHM] Cleaned up {SHM_DIR}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
