"""
Training script for SoilMoistureModel — §48: temporal trunk + fine CNN encoder + U-Net
decoder, soil-moisture heads plus the Landsat ST pattern head.

Usage (terramind conda env):
    python train.py [--lr LR] [--batch-size N] [--n-layers N] [--run-name NAME]
                    [--max-stations N] [--warmup-steps N] [--huber-delta D]
                    [--log-every N] [--lambda-lst auto|FLOAT] [--fine-skips cnn|pool]
                    [--modality-dropout P]

    L = L_sm + lambda * L_lst       L_lst is PATTERN ONLY (alpha = 0, §48.9 item 4)
    lambda "auto" = EMA(g_sm / g_lst), both gradient norms taken at the shared 64-ch map z
    (§46.5 item 28); --lambda-lst 0 is the control that attributes any change to the aux head.

Requires csvs/driver_stats.json (compute_driver_stats.py) — it supplies the per-depth
label means used to initialise the regression-head biases — plus csvs/fine_stats.json and
csvs/lst_stats.json. Missing file = hard error; all four are SHA'd into the checkpoint.

This file REPLACED the patchwise trainer (tags `pw_stage2a-ep9`, `pre-s48-build`).

Resume behaviour: if {checkpoint_dir}/{run_name}/last.pt exists the run
resumes automatically — no flag needed. Delete last.pt for a fresh start.
Resume restores the RNG streams, the global optimizer-step counter (so LR warmup does
not restart) and the per-rank sampler order, so a requeued run is the same experiment
as an uninterrupted one.

Model selection, early stopping and ReduceLROnPlateau all key off the SOIL-MOISTURE
component only (station-mean ubRMSE by default, or val_huber_pooled), NEVER the total
L_sm + lambda*L_lst — otherwise the lambda=0 control would be selected on a different
quantity and could not be compared (§46.5 item 30).

W&B project: soil-moisture-phd
"""

import argparse
import gc
import json
import math
import hashlib
import multiprocessing as mp
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from splits_config import SM_CATEGORIES, TRAIN_YEARS
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.multiprocessing

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import (SoilMoistureModel, masked_huber_loss, lst_pattern_loss, lst_pattern_stats,
                   lst_level_loss, lst_level_stats, lst_level_summary,
                   lst_dT_pixel_loss, lst_dT_pixel_stats)

# ── Preemption handling ───────────────────────────────────────────────────────
# _preempted is set by the SIGTERM handler in whichever process SLURM signalled.
# It is deliberately NOT acted on directly at the batch boundary any more: SLURM
# delivers SIGTERM to every task of the step, but not at the same instant, and the
# handler fires between bytecodes.  Rank 0 could therefore unwind out of the batch
# loop, save, and call destroy_process_group() while rank 2 was still inside
# loss.backward() — the surviving ranks then blocked on the next collective for the
# full 7200 s NCCL timeout while holding four H100s.  The flag is now all_reduce(MAX)'d
# on a fixed cadence (CONFIG["preempt_check_every"]) so every rank leaves on the SAME
# batch index, and the reduction itself is the synchronisation point that rank 0's
# ~600 MB _fsync_save waits behind.
_preempted = False

def _handle_sigterm(signum, frame):
    global _preempted
    _preempted = True

class _Preempted(Exception):
    pass


# ── RNG state (resume reproducibility) ────────────────────────────────────────

def _capture_rng_state() -> dict:
    """Snapshot every RNG stream this process draws from.

    Without this a requeued 120 h run replays a DIFFERENT augmentation stream than an
    uninterrupted one — ERA5 masking, SIF/TWSA dropout and drop-path all draw from
    these generators — so "resumed run" and "fresh run" were never the same experiment
    and a requeue silently changed the training distribution mid-run.
    """
    return {
        "python": random.getstate(),
        "numpy" : np.random.get_state(),
        "torch" : torch.get_rng_state(),
        "cuda"  : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _gather_rng_states(is_ddp: bool, world_size: int) -> list:
    """Collective: return [rng_state_rank0, rng_state_rank1, ...].

    Every rank is seeded differently (set_seed(seed + rank)), so saving only rank 0's
    state and restoring it everywhere would collapse the four ranks onto one stream —
    worse than not restoring at all.  Must be called by ALL ranks.
    """
    local = _capture_rng_state()
    if not is_ddp:
        return [local]
    out = [None] * world_size
    dist.all_gather_object(out, local)
    return out


def _restore_rng_state(states, rank: int, is_main: bool) -> None:
    """Restore this rank's slice of a saved RNG snapshot.  Never fatal.

    Caveat worth knowing: persistent_workers=True means DataLoader workers are seeded
    once, at first iteration, from torch.initial_seed() in the parent.  Restoring the
    parent's torch state BEFORE the first iteration therefore also restores the worker
    seeds; restoring it later would not.  This is called at resume time, before the
    epoch loop, for exactly that reason.
    """
    if not states:
        if is_main:
            print("  [resume] WARNING: checkpoint carries no RNG state (pre-§35.24 "
                  "checkpoint) — the augmentation stream will differ from an "
                  "uninterrupted run.")
        return
    s = states[rank] if rank < len(states) else states[0]
    try:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch"])
        if s.get("cuda") is not None and torch.cuda.is_available():
            # Device count can differ across a requeue onto a different node shape;
            # set_rng_state_all raises rather than truncating, so guard it.
            if len(s["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(s["cuda"])
            elif is_main:
                print(f"  [resume] WARNING: saved CUDA RNG has {len(s['cuda'])} device "
                      f"states but this node exposes {torch.cuda.device_count()} — "
                      f"skipping CUDA RNG restore.")
    except Exception as e:                       # never lose a run over a RNG blob
        if is_main:
            print(f"  [resume] WARNING: RNG restore failed ({e}) — continuing with the "
                  f"freshly seeded stream.")

# ── Sample-index wrapper (val de-duplication) ────────────────────────────────

class IndexedDataset(Dataset):
    """Passthrough wrapper that stamps each item with its dataset index.

    The val sampler is DistributedSampler(drop_last=False), which PADS the last shard by
    repeating the head of the index list so every rank gets an equal count. Those repeated
    samples come back through all_gather_object and were counted a second time in
    compute_metrics — so ubRMSE, bias and the per-station n depended on
    len(val_dataset) % world_size, i.e. on how many GPUs the job happened to get. Up to
    world_size-1 samples, always the same ones (the head of the permutation, and the val
    sampler does not shuffle, so it is literally always the first stations).

    Carrying the index through the batch makes the duplicates identifiable after the
    gather. Wrapping rather than editing dataset.py keeps this fix inside train.py.
    """
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        item["sample_idx"] = int(i)      # default_collate -> (B,) int64 tensor
        return item

    def __getitems__(self, indices):
        # The DataLoader fetcher prefers __getitems__ when hasattr() finds it, and __getattr__
        # below would forward that lookup to the inner dataset — bypassing __getitem__ and
        # never stamping sample_idx, so val de-duplication silently never ran (review B1).
        # Delegate to the inner batch fetch so its I/O thread pool (dataset.io_threads) is used.
        items = (self.ds.__getitems__(list(indices)) if hasattr(self.ds, "__getitems__")
                 else [self.ds[i] for i in indices])
        for it, i in zip(items, indices):
            it["sample_idx"] = int(i)
        return items

    def __getattr__(self, name):
        # Forward anything else (station lists, caches) to the wrapped dataset. Only called
        # for attributes IndexedDataset itself does not define. The explicit "ds" guard
        # prevents infinite recursion if this is ever consulted before __init__ has run
        # (unpickling in a spawn-start worker would do exactly that).
        if name == "ds":
            raise AttributeError(name)
        return getattr(self.ds, name)


# ── Station-balanced samplers (review A2) ────────────────────────────────────
# One full pass is ~1.04M station-days (~2,000 steps at 4x128); the previous model peaked at
# epoch 2, so early stopping had ~2 looks before overfitting and long-record stations
# dominated the gradient while selection weights stations equally. An "epoch" is now at most
# K days per station, redrawn every epoch: ~10x finer early-stop resolution, equal station
# weight. The val subset is drawn ONCE (fixed seed) so the selection metric is comparable
# across epochs; eval_predict.py still reports on the full val set.

def _by_station(samples) -> dict:
    groups: dict = {}
    for i, s in enumerate(samples):
        groups.setdefault(s["station_key"], []).append(i)
    return groups


class StationBalancedSampler(torch.utils.data.Sampler):
    """Each epoch: min(n, K) days per station without replacement, shuffled, then sharded.
    Deterministic in (seed, epoch), so make_resume_loader's list(iter(sampler)) reproduces
    the issued order. Truncated to a multiple of num_replicas: identical length per rank."""

    def __init__(self, dataset, per_station: int, num_replicas: int = 1, rank: int = 0,
                 seed: int = 0):
        self.groups = list(_by_station(dataset.samples).values())
        self.k, self.W, self.rank, self.seed, self.epoch = int(per_station), num_replicas, rank, seed, 0
        n = sum(min(len(g), self.k) for g in self.groups)
        self.num_samples = n // self.W

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        idx = np.concatenate([np.asarray(g)[rng.permutation(len(g))[: self.k]] for g in self.groups])
        idx = idx[rng.permutation(len(idx))][: self.num_samples * self.W]
        return iter(idx[self.rank :: self.W].tolist())

    def __len__(self):
        return self.num_samples


class FixedSubsetDistributedSampler(torch.utils.data.Sampler):
    """A fixed per-station val subset (min(n, K) days, seed 0), sharded like
    DistributedSampler(drop_last=False): padded with the head so every rank has the same
    length; IndexedDataset's sample_idx lets evaluate() drop the padding."""

    def __init__(self, dataset, per_station: int, num_replicas: int = 1, rank: int = 0):
        rng = np.random.default_rng(0)
        idx = sorted(i for g in _by_station(dataset.samples).values()
                     for i in np.asarray(g)[rng.permutation(len(g))[: int(per_station)]].tolist())
        W = num_replicas
        n_pad = (-len(idx)) % W
        idx = idx + idx[:n_pad]
        self.idx = idx[rank::W]

    def set_epoch(self, epoch: int):
        pass

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


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

# There is no /dev/shm L12 preloader any more. The patchwise arm staged tokens there because
# the token store chunks l12 32 acquisitions at a time; §48 reads pooled pyramids from RAM
# and the anchor from a flat per-station memmap (prepare_s48_cache.py), which the page cache
# shares across ranks by itself.


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "splits_csv"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    # §47: dataset.py:80 reads `era5/values18` (18 cols, skt dropped, ssrd/strd added),
    # so it must be z-scored with the 18-column constants. era5_stats.json is the 19-column
    # pre-§43.12 set -- same length mistake aside, its columns 6+ describe skt where the
    # array now holds u10. §43.12 built stats18 but never repointed the trainer.
    "era5_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats18.json",
    # Produced by compute_driver_stats.py.  Supplies SIF/TWSA/soil normalisation to
    # dataset.py and the per-depth head bias to this file (§35.24).  Fail closed: a
    # missing file raises rather than silently training heads from a zero bias, which
    # costs the first ~1k steps just walking the output up to the label mean.
    "driver_stats"  : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/driver_stats.json",
    # TerraMind constants for the fine path (§48.2 item 2) and sigma_ST for the thermal loss
    # (§49). Both are model contracts: SHA'd into the checkpoint beside the two above.
    "fine_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/fine_stats.json",
    "lst_stats"     : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/lst_stats.json",
    # §52 dT_pixel knee + head bias, frozen from the train set by compute_lst_dT_stats.py
    "lst_dT_stats"  : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/lst_dT_stats.json",
    # Each run saves checkpoints under {checkpoint_dir}/{run_name}/
    "checkpoint_dir": "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only",

    # Data
    # §47: both come from splits_config. The cut used to live here AND in
    # create_evaluation_splits.py:27 with nothing tying them, so moving one silently made
    # OOT contaminated or empty (§44.6).
    "category_filter": list(SM_CATEGORIES),
    "years"          : list(TRAIN_YEARS),
    "seed"           : 42,

    # Training
    "batch_size"      : 128,
    "num_workers"     : 12,
    "val_num_workers" : 4,    # val uses 4w×pf4; train uses 12w×pf4 → (12+4)×4 ranks = 64 CPUs
    "prefetch_factor" : 4,
    "max_epochs"      : 150,    # review A2: an epoch is now <=K days per station (~200 steps)
    "lr"              : 2e-4,
    "weight_decay"    : 0.05,
    "lr_patience"     : 3,
    "lr_factor"       : 0.5,
    "grad_clip"       : 1.0,
    "early_stop_patience": 8,
    # Review A2: days per station per epoch (train, redrawn each epoch) and the fixed val
    # subset used for model selection. 0 = the old full pass / full val.
    "train_days_per_station": 180,
    "val_days_per_station"  : 120,
    # Linear LR warmup, in OPTIMIZER STEPS (not epochs).  §35.12: thirteen runs went
    # straight to lr=2e-4 on step 1 with 75.5 M parameters and none of them converged.
    # A transformer that large sees its largest gradients in the first few hundred
    # steps, when the depth heads are still at their bias and every attention row is
    # near-uniform; AdamW's second moment has not warmed up either, so the effective
    # step is at its maximum exactly when the direction is worst.
    "warmup_steps"    : 1000,
    # Cadence, in batches, of the collective preempt check and the batch log line.
    "preempt_check_every": 25,
    "log_every"       : 50,
    # What best.pt / early stopping / ReduceLROnPlateau key off. See _ubrmse_selection:
    # "ubrmse" is the depth-mean of the station-mean ubRMSE — one vote per depth, one vote
    # per station, per-station mean removed — which is the quantity every reported number
    # in this project is stated in. "huber_pooled" keeps selection on the training loss.
    "select_metric"   : "ubrmse",
    # Once-per-val-epoch input-gradient ratio, fine imagery vs everything else (rank 0, one
    # batch, gradients w.r.t. INPUTS only). ~0 means the decoder is not reading the fine path.
    "input_grad_diag" : True,

    # Model
    "n_depths"      : 3,
    "d_model"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 6,
    "drop_path_rate": 0.1,
    # There is no use_cls_depth option: the three disconnected depth heads each read their
    # own CLS row, so the prefix is an invariant of the architecture, not a knob.

    # Fine path (§48). "cnn" = the light encoder; "pool" = §46's masked pool + 1x1, kept as
    # the one-run ablation that says whether the encoder earns its parameters.
    "fine_skips"      : "cnn",
    "modality_dropout": 0.2,    # per-sample P(zero S2 or S1 in the fine path), train only

    # Thermal aux (§46.5 items 27-28, §48.9 item 4). "auto" = EMA(g_sm / g_lst) at the shared
    # 64-ch map, refreshed every lambda_every optimizer steps and all-reduced so every rank
    # optimises the same objective. A number fixes it; 0 is the control.
    "lambda_lst"      : "auto",
    "lambda_every"    : 50,
    "lambda_clamp"    : 10.0,   # review C3: lambda stays within [seed/c, seed*c] of its first post-warmup value
    "lambda_frac"     : 0.3,    # auto lambda = 0.3 x (g_sm/g_lst): LST pulls on the shared map at 30% of SM, not parity
    "lambda_ema"      : 0.9,
    "lst_delta"       : 1.0,    # Huber knee in units of sigma_ST (= 2.71 K, §49.5)
    "lst_level_weight": 0.0,    # §52: weight of the tile LEVEL term (LST - t2m_mean) inside L_lst; 0 = pattern only
    "lst_units"       : "sigma", # §52: "K" = head_lst predicts (LST - t2m) directly in Kelvin (no sigma_ST scaling)
    "lst_target"      : "pattern", # §52: "dT_pixel" = per-cell Huber against (LST_obs - t2m_mean) in K, NOTHING else
    "era5_dropout"    : 0.0,    # §53: P(whole ERA5 window marked missing) per training sample
    "sif_twsa_dropout": 0.5,    # §53: P(whole SIF / whole TWSA window dropped) per training sample (was hard-coded)
    "coarse_dropout"  : 0.0,    # §53: P(all 160 m satellite tokens withheld: anchor + S2/S1 history) per training sample

    # Loss
    "loss_fn"   : "huber",
    "huber_delta": 0.05,        # was a buried default inside masked_huber_loss; SM is in
                                # m3/m3, so 0.05 is ~ one volumetric-percent-times-five —
                                # the knee sits just above the sensor noise floor

    "per_depth_loss" : True,    # equal-weight Huber per depth (default since 2026-08-02;
                                # pooled let the obs-rich 0-10 layer dominate the gradient —
                                # 30-100 barely moved across epochs 1-2 of baseline_huber_notv)

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


def _fsync_save(obj, path):
    """Atomically write a checkpoint, then fsync so GPFS flushes to the storage server.

    Writes to a sibling .tmp and os.replace()s it into place.  Overwriting the live
    file directly is not survivable: these are ~600 MB, and SLURM sends SIGTERM then
    SIGKILL 30 s later (KillWait) on preemption or requeue.  A kill part-way through
    leaves a truncated last.pt/mid_epoch.pt that torch.load rejects on every rank, so
    the job crash-loops on requeue — and a truncated last.pt alongside a best.pt
    written from the same state moments later can lose a multi-day run outright.

    os.replace is atomic within a filesystem, so a reader sees either the whole old
    file or the whole new one.  The directory fsync makes the rename itself durable.
    """
    path = Path(path)
    tmp  = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)            # make the rename durable, not just the data
    finally:
        os.close(dir_fd)


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


def _pearson_r(a, b) -> float:
    """Pearson r with an explicit degenerate-case answer.

    A constant prediction — the collapse mode §35.20 is hunting — has zero variance, and
    np.corrcoef returns nan there with a RuntimeWarning.  nan is the right answer (the
    correlation is undefined, not zero), but it must arrive without a warning storm and
    without depending on numpy's error state.
    """
    if len(a) < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den <= 0.0:
        return float("nan")
    return float((a * b).sum() / den)


def compute_metrics(preds, targets, station_keys, n_worst=5):
    """
    preds, targets : (N, n_depths) numpy arrays
    station_keys   : (N,) array-like of per-sample station identifiers
    Returns (global_metrics, per_station_metrics) where per_station_metrics
    is a dict {station: {MSE, RMSE, MAE, ubRMSE, anomRMSE, bias, r, R2, n}} per depth.

    ubRMSE removes each station's own temporal mean before computing RMSE
    (the standard unbiased-RMSE definition) -- a global mean across all
    stations would otherwise leave cross-station bias in the result.

    §35.10 makes a WITHIN-STATION quantity the primary gate, and until now nothing here
    could see one: MSE/MAE/bias are all pooled and dominated by cross-station offsets, so
    a model that predicts each station's climatological mean and nothing else scores well
    on every one of them.  Three additions close that:

      r        — Pearson correlation of prediction against label *within* a station.  This
                 is the number the gate is about: it is exactly 0 (or nan) for the
                 constant-per-station predictor and is invariant to any per-station affine
                 rescaling the model might have learned instead of dynamics.
      R2       — 1 - SS_res/SS_tot against that station's OWN mean, i.e. skill relative to
                 "always predict this station's climatology".  Unlike r it is not
                 invariant to gain or offset, so r high + R2 negative means the model has
                 the phase but the wrong amplitude — a distinguishable failure.
      anomRMSE — RMSE after centring predictions AND labels on that station's mean.  This
                 is numerically identical to ubRMSE by construction and is emitted under
                 its own name only so the within-station family reads as one block; both
                 keys are kept because ubRMSE is what every earlier log, CSV and figure
                 in this project calls it.

    Global (pooled) rows gain RMSE (√MSE, so the log stops putting a squared quantity
    beside two unsquared ones), the pooled within-station r/R2 computed on the pooled
    anomalies, and the unweighted station-mean of the per-station r/R2 — the pooled
    version weights a station by its sample count, the station-mean does not, and §35.10
    is a statement about stations.
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
        st_r_list, st_r2_list = [], []
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
            st_r      = _pearson_r(p[sel], t[sel])
            # SS_tot uses the station's own label mean -> R2 is skill over that
            # station's climatology, which is the honest null for this problem.
            ss_tot    = float(np.sum(t_anom[sel] ** 2))
            ss_res    = float(np.sum((p[sel] - t[sel]) ** 2))
            st_r2     = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            if station not in per_station:
                per_station[station] = {}
            per_station[station][depth] = {"ubRMSE": st_ubrmse, "anomRMSE": st_ubrmse,
                                           "MAE": st_mae, "bias": st_bias,
                                           "MSE": st_mse, "RMSE": float(np.sqrt(st_mse)),
                                           "r": st_r, "R2": st_r2, "n": int(sel.sum())}
            if math.isfinite(st_r):
                st_r_list.append(st_r)
            if math.isfinite(st_r2):
                st_r2_list.append(st_r2)

        if ub_mask.any():
            ubrmse    = float(np.sqrt(np.mean((p_anom[ub_mask] - t_anom[ub_mask]) ** 2)))
            r_within  = _pearson_r(p_anom[ub_mask], t_anom[ub_mask])
            ss_tot_w  = float(np.sum(t_anom[ub_mask] ** 2))
            ss_res_w  = float(np.sum((p[ub_mask] - t[ub_mask]) ** 2))
            r2_within = (1.0 - ss_res_w / ss_tot_w) if ss_tot_w > 0 else float("nan")
        else:
            ubrmse = r_within = r2_within = float("nan")

        metrics[depth] = {
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mae,
            "ubRMSE": ubrmse, "anomRMSE": ubrmse, "bias": bias,
            "r_within" : r_within,
            "R2_within": r2_within,
            "r_station_mean" : float(np.mean(st_r_list))  if st_r_list  else float("nan"),
            "R2_station_mean": float(np.mean(st_r2_list)) if st_r2_list else float("nan"),
            "n_stations_scored": len(st_r_list),
        }
    return metrics, per_station


# ── Training loop ─────────────────────────────────────────────────────────────

def _compute_loss(pred, label, per_depth=False, return_breakdown=False, delta=0.05,
                  depth_weights=None):
    """SM Huber at the station pixel. Returns (loss, tv) or (loss, tv, depth_sum, depth_cnt).

    `pred` is the model's output dict or its `sm` map (B, 3, 112, 112). This is the
    SOIL-MOISTURE component only — the thermal term is added by the caller, so everything
    that selects or schedules on this function's output stays SM-only (§46.5 item 30).

    `tv` is retained as an always-zero second element purely so the epoch bookkeeping and the
    W&B panels keep their shape. TV = 0 by design (§46.1 row 12), and there is no boundary
    term: Kelvin is not in [0, 1], and the SM heads have no such penalty either.

    depth_sum/depth_cnt are raw per-depth SUMS over this batch (see masked_huber_loss); the
    caller accumulates them over the epoch and all_reduce(SUM)s across ranks, which is only
    correct on sums.  That epoch-level accumulation is the ONLY place a per-depth mean is
    formed in this file — the per-batch 1/n_d(batch) weighting is the model's business and
    is being fixed there (§35.24 item 2).  Nothing here re-normalises by batch counts, so
    there is no double correction to undo when it lands.

    `delta` is the Huber knee, threaded from CONFIG["huber_delta"] / --huber-delta rather
    than left as a default buried in masked_huber_loss's signature: it sets the scale at
    which the loss stops being quadratic, i.e. what counts as an outlier in m3/m3, and a
    run cannot be reproduced from its log if that number is invisible.
    """
    if isinstance(pred, dict):
        pred = pred["sm"]
    if return_breakdown:
        loss, depth_sum, depth_cnt = masked_huber_loss(
            pred, label, delta=delta, per_depth=per_depth,
            depth_weights=depth_weights, return_breakdown=True)
        return loss, pred.new_zeros(1), depth_sum, depth_cnt
    return (masked_huber_loss(pred, label, delta=delta, per_depth=per_depth,
                              depth_weights=depth_weights),
            pred.new_zeros(1))


class LambdaLST:
    """lambda for L = L_sm + lambda * L_lst (§46.5 item 28).

    "auto": lambda <- EMA(g_sm / g_lst), where g_* = ||dL_* / dz|| at the shared 64-channel
    112x112 map both heads read. Taken at z and NOT over parameters: autograd.grad w.r.t. a
    non-leaf touches no AccumulateGrad node, so DDP's reducer sees nothing (the train.py
    precedent of input_grad_ratio). The two norms are all-reduced (SUM) before the ratio, so
    every rank moves to the same lambda on the same step and optimises the same objective;
    refresh steps are keyed to global_step, which is identical across ranks.

    The point is to make two incommensurable scales — m3/m3 Huber and sigma_ST-units Huber —
    contribute comparably to the gradient the decoder sees, without hand-tuning. A number
    fixes lambda; 0 is the control, and then no L_lst gradient is ever formed.
    """

    def __init__(self, spec, every: int = 50, beta: float = 0.9, hold_steps: int = 0,
                 clamp: float = 10.0, frac: float = 1.0):
        self.auto  = (str(spec) == "auto")
        self.value = 0.0 if self.auto else float(spec)
        self.every = max(1, int(every))
        self.beta  = float(beta)
        self.hold_steps = int(hold_steps)
        self.clamp = float(clamp)
        self.frac  = float(frac)     # LST pulls on z at frac x the SM pull (0.3, user 2026-09-29)
        self.n_updates = 0
        self.last_ratio = float("nan")
        self.ema_sm = self.ema_lst = 0.0
        self.seed_value = float("nan")

    @property
    def active(self) -> bool:
        return self.auto or self.value != 0.0

    def due(self, global_step: int) -> bool:
        # Review C3: frozen (lambda = 0) through LR warmup. With zero-weight SM heads and a
        # 1/1000 warmup factor, the first g_sm/g_lst ratios are ~1e-6 of steady state, and an
        # EMA seeded there takes ~1k steps to recover.
        # §53: a FIXED non-zero lambda is never changed, but its push ratio is MEASURED every
        # `every` steps (from step 0; step 0 itself is skipped in update since g_sm == 0) so the
        # log shows whether the fixed value stays near equal pull. Same step on every rank.
        if not self.auto:
            return self.value != 0.0 and global_step % self.every == 0
        if global_step < self.hold_steps:
            return False
        return self.n_updates == 0 or global_step % self.every == 0

    def update(self, l_sm, l_lst, z, ddp_active: bool) -> None:
        g_sm,  = torch.autograd.grad(l_sm,  z, retain_graph=True, allow_unused=True)
        g_lst, = torch.autograd.grad(l_lst, z, retain_graph=True, allow_unused=True)
        norms = torch.stack([
            (g_sm.float().norm()  if g_sm  is not None else z.new_zeros((), dtype=torch.float32)),
            (g_lst.float().norm() if g_lst is not None else z.new_zeros((), dtype=torch.float32)),
        ]).detach()
        if ddp_active:
            dist.all_reduce(norms, op=dist.ReduceOp.SUM)
        g_sm_v, g_lst_v = float(norms[0]), float(norms[1])
        # Either norm zero = nothing to balance this step, so keep lambda and stay "due":
        #   g_lst == 0  no thermal cell on any rank
        #   g_sm  == 0  step 0 — the SM heads are zero-weight-initialised (model.py), so
        #               dL_sm/dz is exactly 0 until the first update. Seeding the EMA with that
        #               0 would hold lambda near zero for the first ~1/(1-beta) refreshes.
        if g_lst_v <= 0.0 or g_sm_v <= 0.0 or not math.isfinite(g_sm_v / g_lst_v):
            return
        self.last_ratio = g_sm_v / g_lst_v
        if not self.auto:
            return                                  # measure only: a fixed lambda stays fixed
        # Review C3: EMA the two norms separately and take the ratio of the EMAs — one
        # noisy small g_lst no longer spikes lambda — then clamp to [seed/c, seed*c] around the
        # first post-warmup value, so lambda cannot run away as the static LST pattern fits
        # and g_lst shrinks (parity would otherwise grow lambda without bound).
        if self.n_updates == 0:
            self.ema_sm, self.ema_lst = g_sm_v, g_lst_v
        else:
            self.ema_sm  = self.beta * self.ema_sm  + (1.0 - self.beta) * g_sm_v
            self.ema_lst = self.beta * self.ema_lst + (1.0 - self.beta) * g_lst_v
        raw = self.ema_sm / self.ema_lst
        if self.n_updates == 0:
            self.seed_value = raw
        self.value = self.frac * min(max(raw, self.seed_value / self.clamp), self.seed_value * self.clamp)
        self.n_updates += 1

    def state_dict(self) -> dict:
        return {"auto": self.auto, "value": self.value, "n_updates": self.n_updates,
                "ema_sm": self.ema_sm, "ema_lst": self.ema_lst, "seed_value": self.seed_value}

    def load_state_dict(self, st: dict | None) -> None:
        if st and bool(st.get("auto")) == self.auto and self.auto:
            self.value, self.n_updates = float(st["value"]), int(st["n_updates"])
            self.ema_sm  = float(st.get("ema_sm", 0.0))
            self.ema_lst = float(st.get("ema_lst", 0.0))
            self.seed_value = float(st.get("seed_value", self.value))


def _per_depth_mean(depth_sum, depth_cnt) -> dict:
    """Sums/counts -> {depth_name: mean loss}, with nan where a depth was never
    observed.  nan rather than 0.0 so an absent depth is visibly absent instead
    of masquerading as a perfect fit (runbook §19.4)."""
    mean = (depth_sum / depth_cnt.clamp(min=1)).tolist()
    cnt  = depth_cnt.tolist()
    return {d: (mean[i] if cnt[i] > 0 else float("nan")) for i, d in enumerate(SM_DEPTHS)}


def _loss_aggregates(depth_sum, depth_cnt):
    """Two flag-independent aggregates of the per-depth Huber sums -> (pooled, depth_mean).

    pooled     = Σsum / Σcnt — one Huber mean over every valid (sample, depth) pair.
                 Its definition does NOT depend on per_depth_loss, so it is
                 comparable across every run, including
                 finished ones.  `val_loss` is not: it means pooled-Huber when
                 per_depth_loss=False and mean-of-depth-means when True, which is why
                 the runbook forbids comparing val_loss across runs (§19.3).
    depth_mean = unweighted mean over observed depths — exactly the average of the
                 per-depth numbers printed above it, so the log reconciles on its face.

    The two differ when depth coverage is uneven: pooled weights each observation
    equally (dominated by 0-10 cm, which has the most stations), depth_mean weights
    each depth equally.  pooled for cross-run comparison, depth_mean for balance.
    """
    tot_s, tot_c = depth_sum.sum().item(), depth_cnt.sum().item()
    pooled = tot_s / tot_c if tot_c > 0 else float("nan")
    per_d  = [s / c for s, c in zip(depth_sum.tolist(), depth_cnt.tolist()) if c > 0]
    depth_mean = sum(per_d) / len(per_d) if per_d else float("nan")
    return pooled, depth_mean


def _inverse_frequency_weights(counts) -> list | None:
    """Per-depth observation counts -> fixed inverse-frequency loss weights, mean 1.

    These must be computed ONCE over the training set and then held fixed.  Deriving them
    per batch (which is what the old per_depth branch effectively did, by dividing each
    depth's sum by that BATCH's count) makes a sample's weight depend on who else happened
    to be in its batch: a 30-100 cm observation that lands alone in a batch of 128 gets 128x
    the weight of one that lands beside three others, and the expected gradient is then not
    the gradient of any fixed objective.  With 43 val stations at 30-100 vs 74 at 0-10, that
    variance is not a rounding error.

    Normalised to mean 1 over the OBSERVED depths so the loss keeps its scale and remains
    readable against previous runs; an unobserved depth gets weight 0.
    """
    c = [float(x) for x in (counts.tolist() if hasattr(counts, "tolist") else counts)]
    inv = [(1.0 / x) if x > 0 else 0.0 for x in c]
    obs = [w for w in inv if w > 0]
    if not obs:
        return None
    m = sum(obs) / len(obs)
    return [w / m for w in inv]


def _ubrmse_selection(per_station) -> float:
    """Depth-mean of the station-mean ubRMSE -> the model-selection scalar.

    Why not Huber.  Every number this project reports is per-station ubRMSE per depth, but
    best.pt was being chosen on a Huber scalar, and the two disagree systematically for two
    compounding reasons:

      * Huber is sample-weighted, so the 0-10 cm layer — which has the most stations and the
        most observations — dominates it. A checkpoint that improved 0-10 while 30-100 got
        worse could win. Averaging over depths first gives each depth one vote.
      * within a depth, Huber is still sample-weighted across stations, so a handful of
        long-record stations set the criterion. Averaging the per-station ubRMSE first gives
        each station one vote, which is what §35.10 is stated in.

    And ubRMSE removes the per-station mean, so a model that only learns station
    climatology cannot win on it — which pooled Huber, dominated by the offset term, will
    happily reward.

    Returns nan if no station/depth had enough samples; the caller falls back to Huber.
    """
    if not per_station:
        return float("nan")
    per_depth = []
    for d in SM_DEPTHS:
        vals = [v[d]["ubRMSE"] for v in per_station.values()
                if d in v and math.isfinite(v[d]["ubRMSE"])]
        if vals:
            per_depth.append(sum(vals) / len(vals))
    return sum(per_depth) / len(per_depth) if per_depth else float("nan")


_NORM_TYPES = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
               torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.InstanceNorm2d)

# Every module type whose parameters are learned *inputs* or *scales* rather than weight
# matrices, and must therefore never be decayed.  Kept separate from _NORM_TYPES because
# test_per_depth_loss.py uses _NORM_TYPES to assert specifically about normalisation
# layers, and widening that constant would make the test tautological.
_NO_DECAY_TYPES = _NORM_TYPES + (torch.nn.Embedding, torch.nn.EmbeddingBag)


def _split_param_groups(model, raw_model):
    """-> (decay_params, no_decay_params) for AdamW.

    Selection is by module TYPE, not by parameter name.  The previous name-based
    filter (`"norm" not in n.lower()`) silently missed every BatchNorm2d in the
    decoder: they sit inside nn.Sequential, so PyTorch names them positionally
    (`decoder.conv1.net.1.weight`) with no "norm" substring anywhere.  Ten BatchNorm
    scale vectors were being decayed toward zero as a result — and each γ is a
    multiplicative gate on an entire decoder feature map, so decaying it attenuates
    the signal rather than constraining capacity.  Their biases were excluded (the
    name contains "bias"), which made the bug harder to spot.

    Matching is by id(), so it is unaffected by DDP's "module." name prefix.

    depth_tokens are excluded too: like a positional embedding they are a learned
    *input*, not a weight matrix, and decaying them pulls the three per-depth queries
    back toward the symmetric state that §18.3 exists to break.

    §35.24: that argument was written for depth_tokens and then applied to depth_tokens
    ALONE, while the model has seven learned embeddings and six of them were being decayed
    at 0.05 — rel_pos_emb, hist_modality_emb, static_modality_emb, and the era5/sif/twsa/
    soil modality tags.  Every one is an nn.Embedding whose rows are added to a token, so
    the exact same reasoning applies verbatim.  Two of them are worse than the depth_tokens
    case, not better:

      * the modality tags are 1- or 2-row tables.  They exist only to make "this token is
        SIF" distinguishable from "this token is TWSA", and the whole signal is the
        DIFFERENCE between rows.  Decay pulls every row toward the origin, i.e. toward
        each other, which is a direct pressure to erase the only thing they encode.
      * rel_pos_emb has 365 rows and, at any given step, a driver history touches only the
        slots it actually has data for.  AdamW's decay is applied to every parameter with
        a grad entry regardless — so the staleness codes for lags that this batch never
        saw still shrink.  Rarely-populated lags decay monotonically toward zero and stop
        being distinguishable from the padded slots.

    Selection is by module type via _NO_DECAY_TYPES, matched by id(), so it stays correct
    for any embedding added later without anyone remembering to update a name list — which
    is exactly what happened in §35.26, when rel_pos_emb was split into a driver table and a
    full-scale rel_pos_emb_hist for the frozen TerraMind stream. Both are covered here with
    no edit, because the rule is the type and not the name.
    """
    no_decay_ids = {id(p) for mod in raw_model.modules() if isinstance(mod, _NO_DECAY_TYPES)
                    for p in mod.parameters(recurse=False)}
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # endswith("bias"), not endswith(".bias"): MultiheadAttention names its packed
        # QKV bias `self_attn.in_proj_bias`, which has no dot before "bias" and would
        # otherwise start being decayed — the mirror image of the BatchNorm bug.
        is_no_decay = (id(p) in no_decay_ids or n.endswith("bias")
                       or n.endswith("depth_tokens"))
        (no_decay if is_no_decay else decay).append(p)
    return decay, no_decay


def _format_depth_line(depth: str, train_loss: float, val_loss: float, m: dict | None) -> str:
    """One per-depth log line.  Extracted so the m=None path — a depth that was trained
    but has no val samples, which compute_metrics drops entirely — is
    reachable from a test instead of only from a rare data layout.

    RMSE is printed next to MSE because the line otherwise put one squared quantity
    (MSE) beside two unsquared ones (MAE, ubRMSE) with no unit marker, and at the
    magnitudes involved — MSE 0.0127 vs MAE 0.0950 — the squared number reads as the
    *smaller* error.  √MSE = 0.113 is the one that is comparable to MAE and ubRMSE.
    Computed here rather than read from m['RMSE'] so old callers passing a 4-key dict
    (and the regression test at test_per_depth_loss.py §7) keep working."""
    if m:
        rmse  = math.sqrt(m["MSE"]) if m.get("MSE") is not None and m["MSE"] >= 0 else float("nan")
        r_w   = m.get("r_within")
        r_str = f"  r={r_w:.3f}" if (r_w is not None and math.isfinite(r_w)) else ""
        stats = (f"MSE={m['MSE']:.4f}  RMSE={rmse:.4f}  MAE={m['MAE']:.4f}  "
                 f"ubRMSE={m['ubRMSE']:.4f}  bias={m['bias']:.4f}{r_str}")
    else:
        stats = "no val samples"
    return f"  {depth:>8s}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  {stats}"


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
    sam = full_loader.sampler          # DistributedSampler under DDP, RandomSampler otherwise

    # Ask the sampler for its own index sequence instead of reimplementing it.
    #
    # The reimplementation was wrong and could not be anything else for long. It open-coded
    # DistributedSampler's drop_last=False branch — ceil(n/W)*W with the head of the
    # permutation appended as padding — while the real sampler is built with drop_last=True,
    # which TRUNCATES to floor(n/W)*W and pads nothing. On a mid-epoch resume every rank
    # therefore got a different index list from the one the sampler had actually issued:
    # shifted by the pad, with W-1 duplicated early samples retrained and the tail skipped.
    # Silent, and only on the resume path.
    #
    # It also read sam.seed / sam.num_replicas / sam.rank unconditionally, which a plain
    # RandomSampler (the single-GPU case, sampler=None -> DataLoader builds one) does not
    # have — so any single-GPU mid-epoch resume died with AttributeError.
    #
    # list(iter(sam)) is the sampler's own answer, correct for whatever flags it was built
    # with, now and after anyone changes them.
    if hasattr(sam, "set_epoch"):
        sam.set_epoch(epoch)           # DistributedSampler: makes the permutation epoch-exact
    indices = list(iter(sam))
    # The loader itself was built with drop_last=True, so it never issues the ragged tail.
    indices = indices[: len(indices) - (len(indices) % bs)]
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


class WarmupPlateauLR:
    """Linear LR warmup composed with an externally-owned ReduceLROnPlateau.

    §35.12: there was no warmup at all.  75.5 M parameters started at lr=2e-4 on step 1
    and thirteen runs never converged.

    Composition is the fiddly part.  ReduceLROnPlateau owns param_group["lr"] — it reads
    the current value at epoch end and multiplies it by `factor`.  Scaling that same field
    for warmup means the plateau reduction compounds with the warmup factor and the run
    silently loses the decay.  So this class keeps the plateau's lr as the authoritative
    `base_lrs` and treats param_group["lr"] as a scratch field:

        set_step(s)  -> param_group["lr"] = base_lr * min(1, (s+1)/warmup_steps)
        before_plateau_step() -> param_group["lr"] = base_lr   (hand the plateau its own
                                 number back, un-warmed, so its factor applies once)
        after_plateau_step()  -> base_lr = param_group["lr"]   (adopt whatever it decided)

    RESUME CORRECTNESS: the step argument is the GLOBAL optimizer step restored from the
    checkpoint, not a counter that starts at 0 each launch.  A --requeue job that dies at
    step 40 000 must not re-warm from 2e-5; equally, one that dies at step 300 must resume
    mid-ramp rather than jumping to full lr.  With warmup driven by a fresh per-process
    counter, a job preempted every few hours would have spent a large fraction of its life
    in warmup and the effective schedule would depend on the cluster's preemption pattern.
    """

    def __init__(self, optimizer, warmup_steps: int):
        self.opt          = optimizer
        self.warmup_steps = max(int(warmup_steps), 0)
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]

    def sync_base_from_optimizer(self):
        """Adopt param_group lrs as the new bases (after a checkpoint restore)."""
        self.base_lrs = [pg["lr"] for pg in self.opt.param_groups]

    def factor(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / self.warmup_steps)

    def set_step(self, step: int) -> float:
        f = self.factor(step)
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            pg["lr"] = base * f
        return f

    def before_plateau_step(self):
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            pg["lr"] = base

    def after_plateau_step(self):
        self.sync_base_from_optimizer()


def train_one_epoch(model, loader, optimizer, device, grad_clip,
                     per_depth=False, max_batches=None,
                     debug_nan=False, skip_batches=0, mid_ckpt_every=500,
                     mid_ckpt_fn=None, huber_delta=0.05, depth_weights=None,
                     global_step=0, warmup=None, is_main=True, log_every=1,
                     ddp_active=False, preempt_check_every=25, use_wandb=False,
                     lam=None, sigma_st=1.0, lst_delta=1.0, lst_level_weight=0.0,
                     dT_sd=1.0, lvl_delta=1.0, lst_target="pattern"):
    """Train one epoch.  If skip_batches > 0, fast-forwards past already-done
    batches (data loads but no GPU compute) then resumes training from that
    point.  Calls mid_ckpt_fn(batches_done) every mid_ckpt_every batches so
    rank 0 can save a recovery checkpoint.

    mid_ckpt_fn must now be passed on EVERY rank, not just rank 0: it performs a
    collective (the RNG all_gather) before rank 0 writes.  Non-main ranks' version is
    expected to participate and then return without writing.

    Returns (mean_loss, mean_tv, data_time, compute_time, depth_sum, depth_cnt,
             global_step, stats) — global_step is the running optimizer-step count that
    drives warmup across requeues; stats carries the gradient-norm summary and the thermal
    term. mean_loss is the SM component only, so it stays comparable with the lambda=0
    control; the thermal term is reported separately (stats["lst_*"]).
    """
    model.train()
    total_loss   = torch.zeros((), device=device)   # kept on-device: see the log throttle
    total_tv     = torch.zeros((), device=device)
    lst_sum      = torch.zeros((), device=device)   # Σ (L_lst x cells), for a cell-weighted mean
    lst_cells    = torch.zeros((), device=device)
    lvl_sum      = torch.zeros((), device=device)   # Σ (L_level x n), §52
    lvl_n        = torch.zeros((), device=device)
    # clip_grad_norm_ RETURNS the pre-clip total norm and it was being thrown away. It is
    # the first number you want when a run diverges or flatlines: a norm two orders of
    # magnitude above grad_clip every step means the reported lr is fiction (every update is
    # really lr * clip / ||g||), and a norm that collapses toward 0 means the model has
    # stopped learning regardless of what the loss curve looks like. Accumulated on-device.
    total_gnorm  = torch.zeros((), device=device)
    max_gnorm    = torch.zeros((), device=device)
    n_clipped    = torch.zeros((), device=device)
    n_batches    = 0
    data_time    = 0.0
    compute_time = 0.0
    t_data_start = time.perf_counter()

    # Under DDP the preempt decision is made ONLY by the collective below, so the cadence
    # must be at least 1 or a signalled job would never stop.
    if ddp_active and preempt_check_every <= 0:
        preempt_check_every = 1

    # Per-depth Huber accumulators — raw sums, reduced with all_reduce(SUM) by the
    # caller (runbook §19.4). Kept on-device and never .item()'d inside the loop.
    n_depths     = len(SM_DEPTHS)
    depth_sum_acc = torch.zeros(n_depths, device=device)
    depth_cnt_acc = torch.zeros(n_depths, device=device)

    # Loader is pre-sliced by make_resume_loader — no IO skip needed here.
    # skip_batches is kept as a display/checkpoint offset only.
    if skip_batches > 0 and is_main:
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
            out = model(batch)

            if debug_nan:
                bad_out = _scan_for_nan({"sm": out["sm"], "lst": out["lst"]})
                if bad_out:
                    _report_nan(f"batch {n_batches+1:03d} OUTPUT", batch, bad_out)

            loss, tv, d_sum, d_cnt = _compute_loss(
                out, batch["label"], per_depth, return_breakdown=True, delta=huber_delta,
                depth_weights=depth_weights)

        depth_sum_acc += d_sum
        depth_cnt_acc += d_cnt

        # Thermal pattern term, outside autocast (lst_pattern_loss works in fp32). Skipped
        # entirely under the lambda=0 control, so the control forms no L_lst gradient at all.
        total = loss
        if lam is not None and lam.active and lst_target == "dT_pixel":
            # §52: the ONLY thermal term — each valid cell vs its own LST_obs - t2m_mean (K).
            l_lst, n_cells = lst_dT_pixel_loss(out["lst"], batch["lst_obs"], batch["lst_t2m"],
                                               delta=lvl_delta, return_count=True)
            if lam.due(global_step):
                lam.update(loss, l_lst, out["z"], ddp_active)
            total = loss + lam.value * l_lst
            lst_sum   += l_lst.detach() * n_cells
            lst_cells += n_cells
        elif lam is not None and lam.active:
            l_lst, n_cells = lst_pattern_loss(out["lst"], batch["lst_obs"], sigma_st,
                                              delta=lst_delta, return_count=True)
            if lst_level_weight > 0:
                # §52: the level term joins the pattern term INSIDE L_lst, so the auto lambda
                # balances the whole thermal pull against SM exactly as before.
                l_lvl, n_lvl = lst_level_loss(out["lst"], batch["lst_obs"], batch["lst_dT"],
                                              sigma_st, dT_sd, delta=lvl_delta, return_count=True)
                l_lst = l_lst + lst_level_weight * l_lvl
                lvl_sum += l_lvl.detach() * n_lvl
                lvl_n   += n_lvl
            if lam.due(global_step):
                lam.update(loss, l_lst, out["z"], ddp_active)
            total = loss + lam.value * l_lst
            lst_sum   += l_lst.detach() * n_cells
            lst_cells += n_cells
        else:
            # head_lst must stay in the graph: DDP is built without find_unused_parameters,
            # and a parameter with no gradient on one step is a hard error there. Zero weight,
            # so the control's gradient is exactly the SM gradient.
            total = loss + 0.0 * out["lst"].float().sum()

        # Warmup is applied per OPTIMIZER STEP, immediately before the step, and is
        # driven by the global counter so a requeue resumes mid-ramp instead of
        # restarting the ramp.
        lr_factor = warmup.set_step(global_step) if warmup is not None else 1.0

        optimizer.zero_grad()
        total.backward()

        if debug_nan:
            bad_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                           for p in model.parameters())
            if bad_grad:
                print(f"  [NaN DEBUG] batch {n_batches+1:03d}: NaN/Inf in gradients")

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach()
        total_gnorm += gnorm
        max_gnorm    = torch.maximum(max_gnorm, gnorm)
        n_clipped   += (gnorm > grad_clip).float()
        optimizer.step()

        if debug_nan:
            bad_param = any(torch.isnan(p).any() for p in model.parameters())
            if bad_param:
                print(f"  [NaN DEBUG] batch {n_batches+1:03d}: NaN parameters after optimizer.step()")

        compute_time += time.perf_counter() - t_compute
        # .detach(), not .item(): the old line forced a device→host sync on EVERY rank on
        # EVERY batch — twice, counting the print below — which drains the pipeline and
        # throws away the H2D/compute overlap CudaPrefetcher exists to create.  The sum
        # stays on device and is read once, at the end of the epoch.
        total_loss   += loss.detach()
        total_tv     += tv.detach().sum()
        n_batches    += 1
        global_step  += 1

        # Per-batch log: rank 0 only, every log_every batches.  Every rank printing every
        # batch produced four interleaved copies of the same line in the SLURM log and
        # cost a sync per rank per step to do it.
        if is_main and (n_batches == 1 or log_every <= 1 or n_batches % log_every == 0):
            step_ms = 1000 * (data_time + compute_time) / n_batches
            _l, _g = loss.item(), gnorm.item()
            # `.3e`, not `.4f`: on pw_stage2a_L3 the train loss fell to 5e-5, and `.4f`
            # printed 467 of 902 logged batches as exactly "0.0001" and 73 as "0.0000".
            # The log is the only per-batch record that survives without W&B, and at
            # `.4f` most of a converged run is quantised into a flat floor.
            _lam = lam.value if (lam is not None and lam.active) else 0.0
            print(f"  batch {skip_batches + n_batches:04d}  loss_sm={_l:.3e}"
                  f"  total={total.item():.3e}  lambda={_lam:.3e}"
                  f"  gnorm={_g:.3f}  lr={optimizer.param_groups[0]['lr']:.3e}"
                  f"  wu={lr_factor:.2f}  step={step_ms:.0f}ms")
            # Step-level W&B (§35.31). Everything logged here was ALREADY synced to host
            # for the print above, on rank 0 only and under the same log_every throttle,
            # so this costs no extra device sync. Without it the only wandb point in an
            # epoch is the epoch summary — on 647 stations that is hours of a flat curve,
            # and a divergence is invisible until it is already over.
            #
            # No explicit step=: the epoch-level wandb.log() uses wandb's auto-increment
            # counter, and mixing an explicit step with an implicit one on the same run
            # makes wandb drop whichever goes backwards. `global_step` rides along as a
            # regular metric so it can be picked as the x-axis in the UI instead.
            if use_wandb:
                # Local import: `import wandb` lives inside main()'s `if is_main` block,
                # so the name does not exist at module scope and a bare wandb.log() here
                # is a NameError on the first logged batch. use_wandb is only ever True
                # after that import succeeded, so this is a sys.modules hit, not a load.
                import wandb
                wandb.log({"train/loss_step"     : _l,
                           "train/lambda_lst"    : _lam,
                           "train/gnorm_step"    : _g,
                           "train/lr"            : optimizer.param_groups[0]["lr"],
                           "train/warmup_factor" : lr_factor,
                           "train/step_ms"       : step_ms,
                           "global_step"         : global_step})

        # Mid-epoch checkpoint every N batches.  Collective on all ranks (RNG gather);
        # every rank reaches the same n_batches because drop_last=True gives all ranks an
        # identical batch count and max_batches is identical too.
        if mid_ckpt_fn is not None and mid_ckpt_every > 0 and n_batches % mid_ckpt_every == 0:
            mid_ckpt_fn(skip_batches + n_batches, global_step)

        # SIGTERM preemption.  The flag is per-process and SLURM does not deliver SIGTERM
        # to every task at the same instant, so acting on the local flag let rank 0 unwind
        # and destroy_process_group() while another rank sat in loss.backward() — the
        # survivors then blocked on their next collective for the full 7200 s NCCL timeout
        # holding four H100s.  all_reduce(MAX) on a fixed cadence makes the decision global
        # and pins it to the SAME batch index on every rank.  The reduction is also the
        # synchronisation the save needs: when it returns, every rank has arrived, so rank
        # 0's ~600 MB _fsync_save cannot start while anyone is still computing.
        stop_now = _preempted
        if ddp_active and preempt_check_every > 0 and n_batches % preempt_check_every == 0:
            flag = torch.tensor([1.0 if _preempted else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop_now = bool(flag.item() > 0)
        elif ddp_active:
            stop_now = False              # only the collective may decide, never the local flag
        if stop_now:
            if ddp_active:
                dist.barrier()            # explicit: nobody is mid-backward past this line
            if mid_ckpt_fn is not None:
                mid_ckpt_fn(skip_batches + n_batches, global_step)
            raise _Preempted()

        t_data_start = time.perf_counter()

    n = max(n_batches, 1)
    stats = {
        "grad_norm_mean": total_gnorm.item() / n,
        "grad_norm_max" : max_gnorm.item(),
        "clip_frac"     : n_clipped.item() / n,   # 1.0 = every step was clipped
        # Raw sums, reduced across ranks by the caller.
        "lst_sum"       : lst_sum.item(),
        "lst_cells"     : lst_cells.item(),
        "lst_level_sum" : lvl_sum.item(),
        "lst_level_n"   : lvl_n.item(),
    }
    if lam is not None:
        stats["lambda_lst"]       = lam.value
        stats["lambda_raw_ratio"] = lam.last_ratio
    return (total_loss.item() / n, total_tv.item() / n, data_time, compute_time,
            depth_sum_acc, depth_cnt_acc, global_step, stats)


def input_grad_ratio(raw_model, batch, device, huber_delta, depth_weights=None):
    """d(L_sm)/d(fine imagery) vs d(L_sm)/d(everything else), RMS per element.

    The check on §48's central claim: the fine encoder exists so the SM map can carry
    structure below 160 m. If this ratio is ~0 the loss is being minimised without reading
    the fine path at all, and every 20 m map is the bottleneck upsampled.

    Norms are divided by sqrt(numel) so tensors of very different sizes compare as RMS per
    element. torch.autograd.grad w.r.t. INPUTS only: no parameter gradient is produced and
    DDP's reducer is untouched. Rank 0 only, one batch. Soil moisture only — the thermal
    head would read the fine path by construction and would say nothing about the SM heads.
    """
    if not batch:
        return {}
    fine_keys = ("fine",)
    rest_keys = ("era5", "soil_patch", "sif", "twsa", "s2_pyr", "s1_pyr", "anchor_l12",
                 "dem_pyr", "lulc_pyr")
    b, leaves = dict(batch), {}
    for k in fine_keys + rest_keys:
        v = b.get(k)
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        t = v.detach().to(device).float().requires_grad_(True)
        b[k], leaves[k] = t, t
    if not leaves:
        return {}

    was_training = raw_model.training
    raw_model.eval()          # drop-path and modality dropout off: attribution, not training
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out  = raw_model(b)
            loss = masked_huber_loss(out["sm"], b["label"].to(device), delta=huber_delta,
                                     per_depth=depth_weights is not None,
                                     depth_weights=depth_weights)
        grads = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
    finally:
        raw_model.train(was_training)

    res = {}
    for (k, t), g in zip(leaves.items(), grads):
        res[k] = (0.0 if g is None
                  else float(g.detach().float().norm().item() / max(t.numel() ** 0.5, 1.0)))
    fs = sum(res.get(k, 0.0) for k in fine_keys)
    rs = sum(res.get(k, 0.0) for k in rest_keys)
    res["fine_sum"] = fs
    res["rest_sum"] = rs
    res["ratio"]    = (fs / rs) if rs > 0 else float("nan")
    return res


class FineAblation:
    """Batch hook for --eval-fine-ablation: replaces everything the fine encoder reads
    (`fine` = 20 m S2/S1/DEM stack with its valid/age flags, and `lulc` at 10 m).

    none    — unchanged.
    zero    — all 19 channels 0 (so every valid flag says "missing") and lulc = LULC_PAD: the
              same thing the model sees for a station with no raw imagery, so on-distribution.
    shuffle — each sample gets the fine inputs of a sample from a DIFFERENT station (same batch
              or the previous one; val batches hold ~4-5 stations since the sampler is sorted).
              Kills site identity while keeping realistic imagery (§24: shuffle, not zero).
    Targets (label, lst_obs) are never touched, so a drop means the model read the fine path.
    """

    def __init__(self, mode: str, seed: int = 0):
        from model import LULC_PAD
        self.mode, self.pad = mode, LULC_PAD
        self.gen  = np.random.default_rng(seed)
        self.prev = None
        self.n_swapped = self.n_kept = 0

    def __call__(self, batch):
        if self.mode == "none":
            return batch
        b = dict(batch)
        if self.mode == "zero":
            b["fine"] = torch.zeros_like(batch["fine"])
            b["lulc"] = torch.full_like(batch["lulc"], self.pad)
            return b
        keys = list(batch["station_key"])
        pf, pl, pk = batch["fine"], batch["lulc"], keys
        if self.prev is not None:
            pf = torch.cat([pf, self.prev[0]]); pl = torch.cat([pl, self.prev[1]]); pk = keys + self.prev[2]
        donors = []
        for i, k in enumerate(keys):
            cand = [j for j, kk in enumerate(pk) if kk != k]
            if cand:
                donors.append(cand[int(self.gen.integers(len(cand)))]); self.n_swapped += 1
            else:
                donors.append(i); self.n_kept += 1
        idx = torch.as_tensor(donors, device=pf.device)
        b["fine"], b["lulc"] = pf[idx], pl[idx]
        self.prev = (batch["fine"], batch["lulc"], keys)
        return b


def run_fine_ablation(raw_model, val_loader, device, world_size, rank, is_main, depth_weights,
                      ckpt_path):
    """--eval-fine-ablation: score val with the fine path intact / shuffled / zeroed. Read-only."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    if is_main:
        print(f"\n=== FINE-PATH ABLATION  ckpt={ckpt_path}  epoch={ckpt.get('epoch')} ===", flush=True)
    rows = []
    for mode in ("none", "shuffle", "zero"):
        hook = FineAblation(mode, seed=rank)
        diag = {}
        _, _, per_station, _, _ = evaluate(
            raw_model, val_loader, device, world_size=world_size, rank=rank,
            per_depth=CONFIG["per_depth_loss"], huber_delta=CONFIG["huber_delta"],
            depth_weights=depth_weights, diag_out=diag, sigma_st=CONFIG["lst_sig_eff"],
            lst_delta=CONFIG["lst_pat_delta"], dT_sd=CONFIG["lst_lvl_scale"],
            lvl_delta=CONFIG["lst_lvl_delta"], lst_target=CONFIG["lst_target"],
            batch_hook=hook)
        if not is_main:
            continue
        nan = float("nan")
        lp, ll, lx = diag.get("lst_pattern") or {}, diag.get("lst_level") or {}, diag.get("lst_px") or {}
        msd = diag.get("map_sd") or [nan] * len(SM_DEPTHS)
        rows.append((mode, _ubrmse_selection(per_station), msd[0],
                     lp.get("r_mean", nan), lp.get("rmse_K", nan), lp.get("skill", nan),
                     ll.get("rmse_K", nan), lx.get("rmse_K", nan), lx.get("r", nan)))
        print(f"  [{mode:7s}] done  (rank-0 swapped {hook.n_swapped}, kept {hook.n_kept})", flush=True)
    if is_main:
        print("\n  mode     SELECT    mapSD0-10  pat_r  pat_RMSE  pat_skill  lvl_RMSE  px_RMSE  px_r")
        for m, s, sd, pr, pe, pk, le, xe, xr in rows:
            print(f"  {m:7s}  {s:.5f}   {sd:.5f}   {pr:.3f}  {pe:.3f} K   {pk:.3f}     "
                  f"{le:.3f} K  {xe:.3f} K  {xr:.3f}")
        print("  pat_* = within-tile LST pattern (centred per scene); lvl = tile mean; px = per pixel dT.\n"
              "  A big drop under shuffle/zero = that output reads the 20 m path.", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, world_size=1, rank=0, max_batches=None, per_depth=False,
             huber_delta=0.05, depth_weights=None, diag_out=None, sigma_st=1.0,
             lst_delta=1.0, dT_sd=1.0, lvl_delta=1.0, lst_target="pattern", batch_hook=None):
    """Distributed-aware evaluation.

    batch_hook: optional callable(batch) -> batch applied to every batch on device before the
            forward (the --eval-fine-ablation input swap). None = unchanged behaviour.

    All ranks process their shard in parallel; loss is all_reduced; predictions
    are gathered to rank 0 for metric computation.  Keeps all GPUs active so
    NCCL watchdog never triggers regardless of GPFS latency.

    `model` is the RAW module (main unwraps DDP before calling), so the per-forward
    diagnostic stashes can be read straight off it.

    diag_out: optional dict, filled in place with the §35.20 collapse diagnostics,
    ALL-REDUCED across ranks before returning.  Passed as an out-parameter rather than
    added to the return tuple because eval_stations.py unpacks exactly five values from
    this function and must keep working.

    The diagnostics MUST be accumulated here and not sampled in main, for two reasons
    that between them made the previous version report nothing at all:

      * the reductions are collectives.  The old entropy log lived inside
        `if is_main: if use_wandb:` — a collective there deadlocks, so it could only ever
        be a rank-0, single-batch snapshot of the LAST val batch on ONE of four shards.
      * `raw_model._last_*` is overwritten by every forward.  Reading it after the loop
        samples one batch out of the epoch; §35.20 needs the epoch.
    """
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_preds   = []
    all_targets = []
    all_station_keys = []
    all_idx     = []          # dataset indices, for de-duplicating the padded val shard


    n_depths      = len(SM_DEPTHS)
    depth_sum_acc = torch.zeros(n_depths, device=device)
    depth_cnt_acc = torch.zeros(n_depths, device=device)

    # ── Diagnostics (§35.24 contract: epoch-wide SUMS, all_reduce'd here) ─────────
    # Allocated up front, not lazily from the first forward: a rank whose val shard is
    # empty would otherwise skip the all_reduce and hang the other three.
    #   depth_ctx   the three CLS outputs, for the inert-use_cls_depth check — now
    #               load-bearing, since the depth heads are disconnected (§46.5 item 34)
    #   map_sd      within-tile SD of each depth's 112x112 SM map, sample-summed. ~0 means
    #               the 20 m map is flat, i.e. nothing below the bottleneck reaches the heads
    #   lst         Σ(L_lst x cells) and Σ cells, the thermal pattern loss on val
    want_diag = diag_out is not None
    diag_dim  = getattr(model, "d_model", 768)
    diag_nd   = getattr(model, "n_depths", n_depths)
    ctx_acc   = torch.zeros(diag_nd, diag_dim, device=device)
    ctx_n_acc = torch.zeros(1, device=device)
    msd_acc   = torch.zeros(n_depths + 1, device=device)    # [:n] Σ SD per depth, [n] count
    lst_acc   = torch.zeros(2, device=device)               # [0] Σ loss*cells, [1] Σ cells
    lst_pat   = torch.zeros(7, device=device)               # lst_pattern_stats sums (spatial pattern r, RMSE K, skill)
    lst_lvl   = torch.zeros(8, device=device)               # [0:6] lst_level_stats, [6] Σ level loss*n, [7] Σ n (§52)
    lst_px    = torch.zeros(8, device=device)               # [0:6] lst_dT_pixel_stats, [6] Σ pixel loss*cells, [7] Σ cells (§52)
    n_ctx_missing = 0

    for batch in CudaPrefetcher(loader, device):
        if max_batches is not None and n_batches >= max_batches:
            break
        if batch_hook is not None:
            batch = batch_hook(batch)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss, _, d_sum, d_cnt = _compute_loss(
                out, batch["label"], per_depth=per_depth, return_breakdown=True,
                delta=huber_delta, depth_weights=depth_weights)
        depth_sum_acc += d_sum
        depth_cnt_acc += d_cnt
        total_loss += loss.item()
        n_batches  += 1

        sm = out["sm"].float()
        msd_acc[:n_depths] += sm.flatten(2).std(dim=2).sum(0)
        msd_acc[n_depths]  += sm.shape[0]
        if "lst_obs" in batch:
            l_lst, n_cells = lst_pattern_loss(out["lst"], batch["lst_obs"], sigma_st,
                                              delta=lst_delta, return_count=True)
            lst_acc[0] += l_lst.detach() * n_cells
            lst_acc[1] += n_cells
            lst_pat += lst_pattern_stats(out["lst"], batch["lst_obs"], sigma_st)
            if "lst_dT" in batch:
                # Always measured, weight or not: the pattern-only run's level is the reference.
                lst_lvl[:6] += lst_level_stats(out["lst"], batch["lst_obs"], batch["lst_dT"],
                                               sigma_st)
                l_lvl, n_lvl = lst_level_loss(out["lst"], batch["lst_obs"], batch["lst_dT"],
                                              sigma_st, dT_sd, delta=lvl_delta, return_count=True)
                lst_lvl[6] += l_lvl.detach() * n_lvl
                lst_lvl[7] += n_lvl
            if "lst_t2m" in batch:
                # Review fix: head output x sigma_st = K in every unit mode (sigma_st is 1.0 under
                # K units); knee dT_sd K in both (dT_sd x lvl_delta), so lst_px is comparable.
                _lst_K = out["lst"].float() * sigma_st
                lst_px[:6] += lst_dT_pixel_stats(_lst_K, batch["lst_obs"], batch["lst_t2m"])
                l_px, n_px = lst_dT_pixel_loss(_lst_K, batch["lst_obs"], batch["lst_t2m"],
                                               delta=dT_sd * lvl_delta, return_count=True)
                lst_px[6] += l_px.detach() * n_px
                lst_px[7] += n_px

        if want_diag:
            if n_batches == 1 and rank == 0:
                # Keep one batch for the input-gradient attribution afterwards.
                diag_out["_first_batch"] = {
                    k: (v.detach() if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
            ctx = getattr(model, "_last_depth_ctx",   None)
            ctn = getattr(model, "_last_depth_ctx_n", None)
            if ctx is None or ctn is None:
                n_ctx_missing += 1
            else:
                c = ctx.detach().float().to(ctx_acc.device)
                if c.shape != ctx_acc.shape:
                    raise RuntimeError(
                        f"[diag] _last_depth_ctx has shape {tuple(c.shape)}, expected "
                        f"{tuple(ctx_acc.shape)} = (n_depths, d_model) SUMMED over the "
                        f"batch, per the §35.24 contract."
                    )
                ctx_acc   += c
                ctx_n_acc += float(ctn)

        # The supervised value is the station pixel of each depth's map.
        all_preds.append(out["sm"][:, :, SoilMoistureModel.STATION_ROW,
                                   SoilMoistureModel.STATION_COL].float().cpu().numpy())
        all_targets.append(batch["label"].cpu().numpy())
        all_station_keys.extend(batch["station_key"])
        if "sample_idx" in batch:
            all_idx.append(batch["sample_idx"].cpu().numpy().reshape(-1))

    mean_loss = total_loss / max(n_batches, 1)

    # Unconditional collectives: every rank allocated these above.
    if world_size > 1:
        dist.all_reduce(msd_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(lst_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(lst_pat, op=dist.ReduceOp.SUM)
        dist.all_reduce(lst_lvl, op=dist.ReduceOp.SUM)
        dist.all_reduce(lst_px, op=dist.ReduceOp.SUM)
        dist.all_reduce(ctx_acc,   op=dist.ReduceOp.SUM)
        dist.all_reduce(ctx_n_acc, op=dist.ReduceOp.SUM)
    if diag_out is not None:
        _n = float(msd_acc[n_depths])
        diag_out["map_sd"]        = ([float(v) / _n for v in msd_acc[:n_depths]] if _n > 0
                                     else [float("nan")] * n_depths)
        diag_out["lst_loss"]      = (float(lst_acc[0] / lst_acc[1]) if lst_acc[1] > 0
                                     else float("nan"))
        diag_out["lst_cells"]     = float(lst_acc[1])
        _p = [float(v) for v in lst_pat]
        diag_out["lst_pattern"] = {
            "n_scenes"    : _p[0],
            "r_mean"      : _p[1] / _p[6] if _p[6] > 0 else float("nan"),
            "r_pos_frac"  : _p[2] / _p[6] if _p[6] > 0 else float("nan"),
            "rmse_K"      : math.sqrt(_p[3] / _p[4]) if _p[4] > 0 else float("nan"),
            "obs_sd_K"    : math.sqrt(_p[5] / _p[4]) if _p[4] > 0 else float("nan"),
            "skill"       : 1.0 - _p[3] / _p[5] if _p[5] > 0 else float("nan"),
        }
        diag_out["lst_level"] = lst_level_summary(lst_lvl[:6].tolist())
        diag_out["lst_level"]["loss"] = (float(lst_lvl[6] / lst_lvl[7]) if lst_lvl[7] > 0
                                         else float("nan"))
        diag_out["lst_px"] = lst_level_summary(lst_px[:6].tolist())
        diag_out["lst_px"]["loss"] = (float(lst_px[6] / lst_px[7]) if lst_px[7] > 0
                                      else float("nan"))
        diag_out["depth_ctx_sum"] = ctx_acc.cpu()
        diag_out["depth_ctx_n"]   = float(ctx_n_acc.item())
        diag_out["n_ctx_missing"] = n_ctx_missing
        diag_out["n_batches"]     = n_batches

    if world_size > 1:
        # Average loss across all ranks. SUM then divide (see the training-side comment):
        # ReduceOp.AVG is the least portable of the reduction ops across NCCL builds and
        # buys nothing here.
        loss_t = torch.tensor(mean_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        mean_loss = loss_t.item() / world_size

        # Per-depth accumulators are raw SUMS, so they reduce with SUM (not AVG
        # like the scalar above) before being divided by the global count.
        dist.all_reduce(depth_sum_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(depth_cnt_acc, op=dist.ReduceOp.SUM)

        # Gather predictions from all ranks to rank 0 (variable-length safe via pickle)
        n_depths = len(SM_DEPTHS)
        local_preds   = np.concatenate(all_preds,   axis=0) if all_preds   else np.empty((0, n_depths))
        local_targets = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, n_depths))
        local_idx     = (np.concatenate(all_idx) if all_idx
                         else np.empty((0,), dtype=np.int64))
        gathered_preds   = [None] * world_size
        gathered_targets = [None] * world_size
        gathered_keys    = [None] * world_size
        gathered_idx     = [None] * world_size
        dist.all_gather_object(gathered_preds,   local_preds)
        dist.all_gather_object(gathered_targets, local_targets)
        dist.all_gather_object(gathered_keys,    all_station_keys)
        dist.all_gather_object(gathered_idx,     local_idx)

        if rank == 0:
            preds        = np.concatenate(gathered_preds,   axis=0)
            targets      = np.concatenate(gathered_targets, axis=0)
            station_keys = [k for keys in gathered_keys for k in keys]
            # De-duplicate the sampler's padding. DistributedSampler(drop_last=False)
            # repeats the head of the index list to make every shard the same length, so
            # up to world_size-1 samples arrive twice — always the same ones, since the val
            # sampler does not shuffle. Counting them twice made every val metric a
            # function of len(val_dataset) % world_size, i.e. of how many GPUs the job got.
            idx = np.concatenate(gathered_idx) if any(len(g) for g in gathered_idx) else None
            if idx is not None and len(idx) == len(preds):
                _, keep = np.unique(idx, return_index=True)   # first occurrence of each
                if len(keep) != len(idx):
                    keep = np.sort(keep)
                    preds        = preds[keep]
                    targets      = targets[keep]
                    station_keys = [station_keys[i] for i in keep]
            elif idx is None:
                print("  [val] WARNING: no sample_idx in the val batches — the padded "
                      "shard cannot be de-duplicated, so up to world_size-1 samples are "
                      "counted twice. Wrap the val dataset in IndexedDataset.")
            metrics, per_station = compute_metrics(preds, targets, station_keys)
        else:
            metrics, per_station = {}, {}
    else:
        # Same empty-shard guard the DDP branch has had all along.  Unguarded,
        # `--max-val-batches 0` or an empty val split raised
        # "need at least one array to concatenate" from inside evaluate() on the
        # single-GPU path — the smoke-test path, which is where it is least welcome.
        preds   = (np.concatenate(all_preds,   axis=0) if all_preds
                   else np.empty((0, n_depths)))
        targets = (np.concatenate(all_targets, axis=0) if all_targets
                   else np.empty((0, n_depths)))
        metrics, per_station = compute_metrics(preds, targets, all_station_keys)

    # Raw sums, not derived means: the caller needs them for both _per_depth_mean and
    # _loss_aggregates, and only sums stay correct under further reduction.
    return mean_loss, metrics, per_station, depth_sum_acc, depth_cnt_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── CLI overrides ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--n-layers",     type=int,   default=None)
    parser.add_argument("--run-name",     type=str,   default=None)
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
    parser.add_argument("--per-depth-loss", action="store_true",
                        help="Equal-weight Huber per depth (vs. pooled baseline)")
    # ── Architecture / thermal aux (§48) ────────────────────────────────────
    parser.add_argument("--fine-skips", choices=["cnn", "pool"], default=None,
                        help="cnn: the light fine encoder (default). pool: §46's masked pool + "
                             "1x1, the ablation")
    parser.add_argument("--modality-dropout", type=float, default=None,
                        help="Per-sample P(zero S2 or S1 in the fine path) in training (0.2)")
    parser.add_argument("--lambda-lst", type=str, default=None,
                        help="'auto' = EMA(g_sm/g_lst) at the shared map (default); a number "
                             "fixes it; 0 is the control (no thermal gradient at all)")
    parser.add_argument("--lst-level-weight", type=float, default=None,
                        help="§52: weight of the thermal LEVEL term (tile LST - t2m_mean, Huber on "
                             "dT / train SD) added to the pattern term inside L_lst (default 0)")
    parser.add_argument("--lst-units", choices=["sigma", "K"], default=None,
                        help="§52: units of the head_lst output. sigma (default) = sigma_ST units as "
                             "in s48_full; K = LST - t2m directly in Kelvin")
    parser.add_argument("--lst-target", choices=["pattern", "dT_pixel"], default=None,
                        help="§52 thermal target. pattern (default, s48_full, + optional "
                             "--lst-level-weight) | dT_pixel = every valid cell against "
                             "LST_obs - t2m_mean, Huber in K; forces --lst-units K")
    parser.add_argument("--era5-dropout", type=float, default=None,
                        help="§53: per-sample P of dropping the WHOLE ERA5 window in training "
                             "(default 0; the 15%% per-day mask always applies)")
    parser.add_argument("--sif-twsa-dropout", type=float, default=None,
                        help="§53: per-sample P of dropping the whole SIF and (independently) the "
                             "whole TWSA window in training (default 0.5, the old hard-coded value)")
    parser.add_argument("--coarse-dropout", type=float, default=None,
                        help="§53: per-sample P of withholding ALL 160 m satellite tokens (anchor -> "
                             "bottleneck, and the S2/S1 history) in training (default 0)")
    parser.add_argument("--lambda-frac", type=float, default=None,
                        help="§53: auto lambda targets g_lst = frac x g_sm at the shared map "
                             "(default 0.3)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Parent folder for {run_name}/ (default CONFIG['checkpoint_dir']); "
                             "keeps ablation arms out of the main run folder")
    # Regularisation overrides — CONFIG keeps the baseline values so comparison runs
    # stay clean; pass these on the sbatch line to make a run self-documenting.
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay (default 0.05; bias/norm always excluded)")
    parser.add_argument("--drop-path-rate", type=float, default=None,
                        help="Stochastic depth rate, linearly scaled across layers (default 0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Epochs without val improvement before stopping (default 8)")
    parser.add_argument("--train-days-per-station", type=int, default=None,
                        help="Days per station per epoch, redrawn each epoch; 0 = full pass (review A2)")
    parser.add_argument("--val-days-per-station", type=int, default=None,
                        help="Fixed val subset per station for selection; 0 = full val (review A2)")
    parser.add_argument("--lr-patience", type=int, default=None,
                        help="ReduceLROnPlateau patience in epochs (default 3)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear LR warmup length in OPTIMIZER STEPS (default 1000). "
                             "0 disables warmup and restores the pre-§35.24 behaviour. "
                             "Driven by the global step counter, so a requeue resumes "
                             "mid-ramp instead of re-warming")
    parser.add_argument("--huber-delta", type=float, default=None,
                        help="Huber knee in m3/m3 (default 0.05, unchanged) — the point "
                             "where the loss turns linear, i.e. what counts as an outlier")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Batches between per-batch log lines, rank 0 only (default 50)")
    parser.add_argument("--select-metric", choices=["ubrmse", "huber_pooled"], default=None,
                        help="What best.pt, early stopping and ReduceLROnPlateau key off. "
                             "ubrmse (default) = depth-mean of station-mean ubRMSE, the "
                             "quantity §35.10 is stated in. huber_pooled = the pooled "
                             "training loss. Both are always logged")
    parser.add_argument("--no-input-grad-diag", action="store_true",
                        help="Disable the once-per-epoch fine-vs-rest input-gradient ratio "
                             "(rank 0, one val batch). On by default")
    parser.add_argument("--eval-fine-ablation", action="store_true",
                        help="EVAL ONLY, writes nothing: load --eval-ckpt of --run-name and score "
                             "val three times -- fine path intact, shuffled across stations, "
                             "zeroed -- printing SM SELECT, map SD and the LST pattern/level/pixel "
                             "stats for each. Answers whether the 20 m path drives LST and SM")
    parser.add_argument("--eval-ckpt", default="best.pt",
                        help="checkpoint file inside the run's checkpoint dir for --eval-fine-ablation")
    args = parser.parse_args()

    if args.lr          is not None: CONFIG["lr"]         = args.lr
    if args.batch_size  is not None: CONFIG["batch_size"] = args.batch_size
    if args.n_layers    is not None: CONFIG["n_layers"]   = args.n_layers
    if args.run_name    is not None: CONFIG["run_name"]   = args.run_name
    if args.num_workers     is not None: CONFIG["num_workers"]     = args.num_workers
    if args.prefetch_factor is not None: CONFIG["prefetch_factor"] = args.prefetch_factor
    if args.max_epochs  is not None: CONFIG["max_epochs"] = args.max_epochs
    if args.per_depth_loss: CONFIG["per_depth_loss"] = True
    if args.fine_skips       is not None: CONFIG["fine_skips"]       = args.fine_skips
    if args.modality_dropout is not None: CONFIG["modality_dropout"] = args.modality_dropout
    if args.lambda_lst       is not None:
        # Validate here, before anything is allocated: a typo would otherwise surface as a
        # float() error on the first training step, after the datasets are built.
        if args.lambda_lst != "auto":
            float(args.lambda_lst)
        CONFIG["lambda_lst"] = args.lambda_lst
    if args.checkpoint_dir   is not None: CONFIG["checkpoint_dir"]   = args.checkpoint_dir
    if args.lst_level_weight is not None: CONFIG["lst_level_weight"] = args.lst_level_weight
    if args.lst_units        is not None: CONFIG["lst_units"]        = args.lst_units
    if args.lst_target       is not None: CONFIG["lst_target"]       = args.lst_target
    if args.era5_dropout     is not None: CONFIG["era5_dropout"]     = args.era5_dropout
    if args.sif_twsa_dropout is not None: CONFIG["sif_twsa_dropout"] = args.sif_twsa_dropout
    if args.coarse_dropout   is not None: CONFIG["coarse_dropout"]   = args.coarse_dropout
    if args.lambda_frac      is not None: CONFIG["lambda_frac"]      = args.lambda_frac
    if CONFIG["lst_target"] == "dT_pixel":
        CONFIG["lst_units"] = "K"                # the target is in K; no sigma_ST anywhere

    # Architecture stamp. ckpt_utils refuses anything else: every earlier arm shares key
    # prefixes with this one, so the stamp is the only reliable discriminator.
    CONFIG["arch"] = "s48"

    # Provenance: no code path recorded a commit SHA into a checkpoint, so a reported number
    # could not be traced to the code that produced it (§35.13).
    try:
        import subprocess as _sp
        CONFIG["git_sha"] = _sp.check_output(["git", "rev-parse", "HEAD"],
                                             text=True, stderr=_sp.DEVNULL).strip()
        CONFIG["git_dirty"] = bool(_sp.check_output(["git", "status", "--porcelain"],
                                                    text=True, stderr=_sp.DEVNULL).strip())
    except Exception:
        CONFIG["git_sha"], CONFIG["git_dirty"] = "unknown", None

    # Normalisation provenance (§35.28). git_sha traces a number to the CODE that produced
    # it; it says nothing about the CONSTANTS. csvs/era5_stats.json and
    # csvs/driver_stats.json are part of the model contract — every ERA5/SIF/TWSA/soil input
    # is z-scored against them and the head biases are initialised from label_mean — and
    # they are regenerated by a separate job. §35.27 overwrote era5_stats.json in place
    # (whole-record -> train-years-only, the OOT-leak fix), which silently invalidated every
    # checkpoint trained before it: eval_predict.py would feed such a model differently
    # normalised inputs and report quietly degraded numbers with no error anywhere.
    #
    # Hash both into the checkpoint so "which normalisation produced this number" is
    # answerable from the artifact rather than from memory.
    for _key, _path in (("era5_stats", CONFIG["era5_stats"]),
                        ("driver_stats", str(Path(CONFIG["era5_stats"]).with_name(
                            "driver_stats.json"))),
                        ("fine_stats", CONFIG["fine_stats"]),
                        ("lst_stats",  CONFIG["lst_stats"]),
                        ("lst_dT_stats", CONFIG["lst_dT_stats"])):
        try:
            CONFIG[f"{_key}_sha"] = hashlib.sha256(
                Path(_path).read_bytes()).hexdigest()[:16]
        except Exception as _e:
            CONFIG[f"{_key}_sha"] = "unknown"
            # `is_main` is not defined until after DDP init, ~90 lines below; this runs
            # during argument parsing. Gate on the env var torchrun already set.
            if int(os.environ.get("RANK", 0)) == 0:
                print(f"  [provenance] WARNING: could not hash {_path} ({_e}) — the "
                      f"checkpoint will not record which normalisation it was trained with.")
    if args.weight_decay        is not None: CONFIG["weight_decay"]        = args.weight_decay
    if args.drop_path_rate      is not None: CONFIG["drop_path_rate"]      = args.drop_path_rate
    if args.early_stop_patience is not None: CONFIG["early_stop_patience"] = args.early_stop_patience
    if args.lr_patience         is not None: CONFIG["lr_patience"]         = args.lr_patience
    if args.train_days_per_station is not None: CONFIG["train_days_per_station"] = args.train_days_per_station
    if args.val_days_per_station   is not None: CONFIG["val_days_per_station"]   = args.val_days_per_station
    if args.warmup_steps        is not None: CONFIG["warmup_steps"]        = args.warmup_steps
    if args.huber_delta         is not None: CONFIG["huber_delta"]         = args.huber_delta
    if args.log_every           is not None: CONFIG["log_every"]           = args.log_every
    if args.select_metric       is not None: CONFIG["select_metric"]       = args.select_metric
    if args.no_input_grad_diag:              CONFIG["input_grad_diag"]     = False

    # sigma_ST (§49): the thermal residual is divided by it, so it is a model contract, not a
    # convenience. Read once here and carried in CONFIG, hence in every checkpoint.
    with open(CONFIG["lst_stats"]) as _f:
        CONFIG["sigma_st"] = float(json.load(_f)["sigma_ST"])

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
    )
    # Val gets a fifth of the smoke-test station cap, as it always has.
    val_max_stations = max(1, args.max_stations // 5) if args.max_stations is not None else None
    # file_system strategy avoids fd exhaustion with 32 workers; must be set before workers spawn
    torch.multiprocessing.set_sharing_strategy("file_system")

    # require_lst whenever the thermal term is on: a thermal head with no target anywhere in
    # the training set would train on nothing and log nothing wrong (§46.8).
    _lam_on = str(CONFIG["lambda_lst"]) == "auto" or float(CONFIG["lambda_lst"]) != 0.0
    train_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["train"], training=True,
                                         max_stations=args.max_stations, require_lst=_lam_on,
                                         era5_dropout=CONFIG["era5_dropout"],
                                         sif_twsa_dropout=CONFIG["sif_twsa_dropout"],
                                         coarse_dropout=CONFIG["coarse_dropout"])
    # §52 level target scale, from the TRAINING samples only. Computed for every run so the
    # pattern-only and control runs report the same val level metrics.
    CONFIG["dT_mu"], CONFIG["dT_sd"], _n_dT = train_dataset.lst_dT_stats()
    if is_main:
        print(f"  thermal level dT = LST_tile - t2m_mean over train: mean {CONFIG['dT_mu']:.3f} K, "
              f"sd {CONFIG['dT_sd']:.3f} K, n {_n_dT}")
    _dT_on = CONFIG["lst_level_weight"] > 0 or CONFIG["lst_target"] == "dT_pixel"
    if _dT_on and _lam_on and not (_n_dT > 100 and CONFIG["dT_sd"] > 0):
        raise RuntimeError(f"dT target on (level weight {CONFIG['lst_level_weight']}, target "
                           f"{CONFIG['lst_target']}) but only {_n_dT} "
                           f"training samples carry a dT target (sd {CONFIG['dT_sd']})")
    if not (CONFIG["dT_sd"] > 0):
        CONFIG["dT_sd"] = 1.0                    # val metric scale only; no level target
    # Units of the thermal head. "sigma": output x sigma_ST = K, pattern knee 1 sigma, level
    # residual / dT_sd (s48_full). "K": output IS K; same knees expressed in K (pattern
    # sigma_ST K, level dT_sd K), so both loss terms are plain Kelvin Huber.
    if CONFIG["lst_units"] == "K":
        CONFIG["lst_sig_eff"], CONFIG["lst_pat_delta"] = 1.0, CONFIG["sigma_st"]
        CONFIG["lst_lvl_scale"], CONFIG["lst_lvl_delta"] = 1.0, CONFIG["dT_sd"]
    else:
        CONFIG["lst_sig_eff"], CONFIG["lst_pat_delta"] = CONFIG["sigma_st"], CONFIG["lst_delta"]
        CONFIG["lst_lvl_scale"], CONFIG["lst_lvl_delta"] = CONFIG["dT_sd"], 1.0
    CONFIG["dT_bias_init"] = CONFIG["dT_mu"]
    if CONFIG["lst_target"] == "dT_pixel":
        # §52: the knee and the head's starting bias are FROZEN from the full training set
        # (csvs/lst_dT_stats.json), not recomputed from whatever stations this run loaded.
        # Fail closed: without the file the knee would silently depend on --max-stations.
        with open(CONFIG["lst_dT_stats"]) as _f:
            _ds = json.load(_f)
        CONFIG["dT_pixel_mean"], CONFIG["dT_pixel_sd"] = (float(_ds["dT_pixel_mean"]),
                                                          float(_ds["dT_pixel_sd"]))
        CONFIG["lst_lvl_delta"] = CONFIG["dT_pixel_sd"]          # Huber knee c, K
        CONFIG["dT_bias_init"]  = CONFIG["dT_pixel_mean"]
        if is_main:
            print(f"  dT_pixel (frozen, {Path(CONFIG['lst_dT_stats']).name}): knee c = "
                  f"{CONFIG['dT_pixel_sd']:.3f} K, head bias = {CONFIG['dT_pixel_mean']:.3f} K")
    if CONFIG["train_days_per_station"]:
        train_sampler = StationBalancedSampler(train_dataset, CONFIG["train_days_per_station"],
                                               num_replicas=world_size, rank=rank, seed=CONFIG["seed"])
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                            shuffle=True, drop_last=True) if is_ddp else None

    # Val dataset on all ranks — DistributedSampler splits it across GPUs.
    # Wrapped so each item carries its dataset index: the val sampler runs drop_last=False
    # and pads the final shard with repeats, which compute_metrics would otherwise count
    # twice (see IndexedDataset).
    val_dataset = IndexedDataset(
        SoilMoistureDataset(**common_kwargs, split_filter=["val"], training=False,
                            max_stations=val_max_stations))
    if CONFIG["val_days_per_station"]:
        val_sampler = FixedSubsetDistributedSampler(val_dataset, CONFIG["val_days_per_station"],
                                                    num_replicas=world_size, rank=rank)
    else:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                          shuffle=False, drop_last=False) if is_ddp else None
    if is_main:
        print(f"  samplers: train {len(train_sampler) if train_sampler is not None else len(train_dataset)}"
              f"/rank per epoch ({CONFIG['train_days_per_station'] or 'all'} d/station), "
              f"val {len(val_sampler) if val_sampler is not None else len(val_dataset)}/rank "
              f"({CONFIG['val_days_per_station'] or 'all'} d/station, fixed)")

    # §51.1: every station the split assigns must be admitted. A dropped station is a silent
    # inventory lie (565 trained vs 573 reported); smoke runs cap stations, so they are exempt.
    if args.max_stations is None:
        for _name, _ds in (("train", train_dataset), ("val", val_dataset.ds)):
            _sk = getattr(_ds, "station_skips", None)
            if _sk is None:
                raise RuntimeError(f"§51.1: {_name} dataset has no station_skips; cannot verify admission")
            if _sk:
                raise RuntimeError(f"§51.1: {_name} dropped stations the split assigns: {_sk}. "
                                   f"Fix the data or demote them in station_splits.csv; do not train around it.")

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

    # ── Head bias initialisation (§35.24) ─────────────────────────────
    # The per-depth regression heads start at bias 0, so epoch 1 opens with the model
    # predicting ~0 m3/m3 against labels centred near 0.25.  The first few hundred steps
    # are then spent doing nothing but walking three scalars up to the label mean — under
    # Huber, whose gradient SATURATES at delta once the residual exceeds the knee, so that
    # walk happens at a constant, slow rate and every other parameter is being updated
    # from a gradient dominated by an offset the head could have been handed for free.
    # Initialising the bias at the training-set mean per depth makes step 1 start at the
    # climatological null instead of below the physical range.
    #
    # Fail closed: no silent fallback to zeros.  A run that quietly trained without this
    # is indistinguishable in the log from one that had it, which is exactly the class of
    # difference that makes two runs incomparable for no visible reason.
    driver_stats_path = Path(CONFIG["driver_stats"])
    if not driver_stats_path.exists():
        raise FileNotFoundError(
            f"{driver_stats_path} not found. It is produced by compute_driver_stats.py "
            f"and carries both the SIF/TWSA/soil normalisation dataset.py needs and the "
            f"per-depth label_mean used here for head bias init. Run "
            f"`python compute_driver_stats.py` (train split, {CONFIG['years'][0]}-"
            f"{CONFIG['years'][-1]}) before training."
        )
    with open(driver_stats_path) as _f:
        _driver_stats = json.load(_f)
    _label_mean = _driver_stats.get("label_mean")
    if not isinstance(_label_mean, dict):
        raise KeyError(
            f"{driver_stats_path} has no 'label_mean' object — it was written by an older "
            f"compute_driver_stats.py. Regenerate it with the current script."
        )
    _missing = [d for d in SM_DEPTHS if d not in _label_mean]
    if _missing:
        raise KeyError(
            f"{driver_stats_path}['label_mean'] is missing depth(s) {_missing}; "
            f"SM_DEPTHS = {SM_DEPTHS}. Regenerate with compute_driver_stats.py."
        )
    head_bias_init = [float(_label_mean[d]) for d in SM_DEPTHS]   # SM_DEPTHS order

    # ── Fixed per-depth loss weights ──────────────────────────────────
    # per_depth_loss must weight each depth by a CONSTANT, not by that batch's count (see
    # _inverse_frequency_weights). Preferred source is a count vector in driver_stats.json,
    # which is a genuine one-off pass over the training set. If it is absent, the weights
    # are derived at the end of epoch 1 from the all-reduced train_depth_cnt — which is also
    # exactly one pass over the training set, just obtained for free — then frozen, written
    # to {ckpt_dir}/depth_weights.json and carried in the checkpoint so a resume reuses the
    # identical vector. Epoch 1 therefore runs unweighted; that is stated in the log rather
    # than hidden, because it makes epoch 1's loss non-comparable with epoch 2's.
    depth_weights_list = None
    _label_count = _driver_stats.get("label_count") or _driver_stats.get("label_n")
    if isinstance(_label_count, dict) and all(d in _label_count for d in SM_DEPTHS):
        depth_weights_list = _inverse_frequency_weights(
            [float(_label_count[d]) for d in SM_DEPTHS])

    # ── Model ─────────────────────────────────────────────────────────
    if is_main:
        print("Building model...")
        print(f"  head_bias_init (from {driver_stats_path.name}): " +
              "  ".join(f"{d}={b:.4f}" for d, b in zip(SM_DEPTHS, head_bias_init)))
    model = SoilMoistureModel(
        n_depths         = CONFIG["n_depths"],
        d_model          = CONFIG["d_model"],
        n_heads          = CONFIG["n_heads"],
        n_layers         = CONFIG["n_layers"],
        drop_path_rate   = CONFIG.get("drop_path_rate", 0.1),
        head_bias_init   = head_bias_init,
        fine_skips       = CONFIG["fine_skips"],
        modality_dropout = CONFIG["modality_dropout"],
    ).to(device)
    if ((CONFIG["lst_level_weight"] > 0 or CONFIG["lst_target"] == "dT_pixel")
            and math.isfinite(CONFIG["dT_bias_init"])):
        # §52: head_lst now predicts (LST - t2m) per cell (K, or sigma_ST units). Start its level at
        # the training mean so step 1 is not a ~15 K error; a resume overwrites this anyway.
        with torch.no_grad():
            model.decoder.head_lst.bias.fill_(CONFIG["dT_bias_init"] / CONFIG["lst_sig_eff"])
    lam = LambdaLST(CONFIG["lambda_lst"], every=CONFIG["lambda_every"],
                    hold_steps=CONFIG["warmup_steps"], clamp=CONFIG["lambda_clamp"],
                    frac=CONFIG["lambda_frac"],
                    beta=CONFIG["lambda_ema"])

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_ddp else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable parameters: {n_params:,}  (fine encoder "
              f"{sum(p.numel() for p in raw_model.fine_encoder.parameters()):,})")
        # Echo the run-defining config. It is saved into the checkpoint too, but a job
        # log should be readable on its own — otherwise which flags a run actually used
        # can only be recovered by torch.load-ing a 600 MB checkpoint.
        _echo = ["run_name", "fine_skips", "modality_dropout", "lambda_lst", "sigma_st",
                 "per_depth_loss", "lr", "warmup_steps", "huber_delta", "lst_delta",
                 "lst_target", "era5_dropout", "sif_twsa_dropout", "coarse_dropout", "lambda_frac", "lst_level_weight", "lst_units", "dT_mu", "dT_sd", "lst_pat_delta", "lst_lvl_delta", "dT_bias_init",
                 "batch_size", "weight_decay", "drop_path_rate",
                 "n_layers", "early_stop_patience", "lr_patience",
                 "select_metric", "input_grad_diag", "git_sha",
                 "era5_stats_sha", "driver_stats_sha", "fine_stats_sha", "lst_stats_sha", "lst_dT_stats_sha"]
        print("CONFIG: " + "  ".join(f"{k}={CONFIG.get(k)}" for k in _echo))

    # ── Optimiser ─────────────────────────────────────────────────────
    decay_params, no_decay_params = _split_param_groups(model, raw_model)
    if is_main:
        print(f"Optimiser groups: {len(decay_params)} decayed, "
              f"{len(no_decay_params)} not decayed (norms, biases, depth_tokens, "
              f"all nn.Embedding)")
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
    # Linear warmup composed with the plateau scheduler (§35.12). It owns base_lrs and
    # treats param_group["lr"] as scratch; the plateau keeps owning the base. Constructed
    # BEFORE the resume block so its bases can be re-synced from whatever lr the
    # checkpoint restores.
    warmup = WarmupPlateauLR(optimizer, CONFIG["warmup_steps"])

    # ── Resume from checkpoint (automatic if last.pt exists) ──────────
    start_epoch       = 1
    best_val_loss     = float("inf")
    no_improve_count  = 0
    wandb_run_id      = None
    val_pending_epoch = None   # epoch whose training is done but val crashed last time
    saved_train_loss  = None
    saved_train_tv    = None
    global_step       = 0      # optimizer steps since the run began; drives warmup
    # Cached from a previous run of this same run_name, if any.
    _dw_path = ckpt_dir / "depth_weights.json"
    if depth_weights_list is None and _dw_path.exists():
        try:
            depth_weights_list = json.loads(_dw_path.read_text())["weights"]
        except Exception as e:
            print(f"  [depth-weights] ignoring unreadable {_dw_path} ({e})")
    # Name of the scalar that selects best.pt and drives the LR plateau. Stamped into
    # every checkpoint so a resume can detect that it is inheriting a best_val_loss
    # measured with a different definition (§35.24 item 1).
    SELECTION_METRIC  = ("val_ubrmse_depth_mean" if CONFIG["select_metric"] == "ubrmse"
                         else "val_huber_pooled")

    ckpt_last = ckpt_dir / "last.pt"
    if ckpt_last.exists():
        if is_main:
            print(f"Checkpoint found — resuming from {ckpt_last}")
        ckpt = torch.load(ckpt_last, map_location=device, weights_only=False)
        # Review fix: the thermal target and head units must match the checkpoint, or a resume
        # silently changes what head_lst means mid-run.
        _cc = ckpt.get("config", {}) or {}
        for _k, _dflt in (("lst_target", "pattern"), ("lst_units", "sigma")):
            if _cc.get(_k, _dflt) != CONFIG[_k]:
                raise RuntimeError(f"resume mismatch: checkpoint {_k}={_cc.get(_k, _dflt)!r}, "
                                   f"this run {_k}={CONFIG[_k]!r}. Use a new --run-name.")
        try:
            raw_model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint key mismatch — architecture likely changed. "
                f"Delete {ckpt_last} to start fresh.\nOriginal error: {e}"
            ) from None
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            # Param-group SIZES changed (e.g. the norm/bias split was corrected), so the
            # saved optimizer state cannot be mapped. The model state_dict already loaded
            # cleanly, which is why the friendly RuntimeError above never fires — without
            # this catch you get a bare ValueError that says nothing about what to do.
            # Continuing with fresh Adam moments is far better than dying: they re-warm
            # within a few hundred steps and the weights are intact.
            if is_main:
                print(f"  [resume] optimizer state incompatible — continuing with fresh "
                      f"Adam moments (weights loaded fine). Reason: {e}")
        scheduler.load_state_dict(ckpt["scheduler"])

        # Restore-then-override.  Optimizer.load_state_dict replaces the whole param-group
        # dict and ReduceLROnPlateau.load_state_dict is a bare __dict__.update, so BOTH
        # silently revert this launch's CLI flags to whatever the checkpoint held.
        # Re-apply ONLY what was passed explicitly.
        #
        # Critically, lr is no longer reset unconditionally. The scheduler's decayed lr
        # lives in param_groups (ReduceLROnPlateau never writes it back on load), so
        # clobbering it meant every requeue of this --requeue/120h job jumped lr from e.g.
        # 5e-5 back to 2e-4 while the restored scheduler.best/num_bad_epochs still believed
        # the decay had happened. Silent, no crash — the run would simply re-diverge.
        if args.lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = CONFIG["lr"]
        if args.weight_decay is not None:
            optimizer.param_groups[0]["weight_decay"] = CONFIG["weight_decay"]  # group 1 stays 0.0
        if args.lr_patience is not None:
            scheduler.patience = CONFIG["lr_patience"]
        # The plateau's lr now lives in param_groups after all the overrides above; adopt
        # it as the warmup base so the very first step of this launch is
        # base * factor(global_step) and not base * 1.0 followed by a warmup-scaled step 2.
        warmup.sync_base_from_optimizer()
        if is_main:
            print(f"  [resume] lr={optimizer.param_groups[0]['lr']:.3e}  "
                  f"wd={optimizer.param_groups[0]['weight_decay']}  "
                  f"lr_patience={scheduler.patience}  "
                  f"(explicit CLI flags re-applied; others kept from checkpoint)")
        best_val_loss    = ckpt["best_val_loss"]
        no_improve_count = ckpt["no_improve_count"]
        wandb_run_id     = ckpt.get("wandb_run_id")

        # Selection metric provenance. best_val_loss used to be the mean-of-batch-means
        # `val_loss`; it is now the pooled Huber. The two are different numbers on the same
        # model, so inheriting one as the threshold for the other would either freeze
        # best.pt forever or overwrite it on epoch 1 for no reason, and early stopping
        # would count from a meaningless baseline. Reset rather than pretend.
        _ckpt_metric = ckpt.get("selection_metric")
        if _ckpt_metric != SELECTION_METRIC:
            if is_main:
                print(f"  [resume] WARNING: checkpoint selected on "
                      f"'{_ckpt_metric or 'val_loss (pre-§35.24 mean-of-batch-means)'}' but "
                      f"this run selects on '{SELECTION_METRIC}'. The two are not "
                      f"comparable, so best_val_loss and no_improve_count are RESET. "
                      f"best.pt on disk is from the old criterion until the next "
                      f"improvement overwrites it.")
            best_val_loss    = float("inf")
            no_improve_count = 0

        # The exact weight vector the run was using, so a requeue does not silently change
        # the objective mid-run.
        if ckpt.get("depth_weights") is not None:
            depth_weights_list = list(ckpt["depth_weights"])
        # Same for lambda: an "auto" run resumes at its EMA, not re-measured from scratch.
        lam.load_state_dict(ckpt.get("lambda_lst"))

        # RNG state, so a requeued run replays the same augmentation stream as an
        # uninterrupted one (§35.24 item 10). Restored on every rank from its own slice,
        # BEFORE the loaders spawn their persistent workers.
        _restore_rng_state(ckpt.get("rng"), rank, is_main)

        # Global optimizer step, for warmup. Old checkpoints have none; estimate it from
        # the epoch count rather than restarting the ramp — a run requeued at epoch 40
        # must not re-warm. len(train_loader) is this rank's batch count, which equals the
        # optimizer-step count per epoch.
        global_step = ckpt.get("global_step")
        if global_step is None:
            global_step = max(0, (ckpt["epoch"] - 1)) * len(train_loader)
            if is_main:
                print(f"  [resume] checkpoint predates global_step — estimating "
                      f"{global_step} from epoch {ckpt['epoch']} x {len(train_loader)} "
                      f"batches (warmup is {CONFIG['warmup_steps']} steps, so this only "
                      f"matters if the run died inside the first epoch)")

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
            _lr_before = [pg["lr"] for pg in optimizer.param_groups]
            try:
                optimizer.load_state_dict(mc["optimizer"])
            except ValueError as e:
                if is_main:
                    print(f"  [mid-epoch] optimizer state incompatible — fresh Adam "
                          f"moments. Reason: {e}")
            # mid_epoch.pt has no scheduler state, so restore whatever lr the
            # epoch-boundary resume above had already settled on.  Without this the
            # mid-epoch path silently reverts to the lr stored in mid_epoch.pt.
            for pg, lr in zip(optimizer.param_groups, _lr_before):
                pg["lr"] = lr
            warmup.sync_base_from_optimizer()
            skip_batches = mc.get("batches_done", 0)
            lam.load_state_dict(mc.get("lambda_lst"))
            # The mid-epoch checkpoint is the more precise source for both of these.
            if mc.get("global_step") is not None:
                global_step = mc["global_step"]
            _restore_rng_state(mc.get("rng"), rank, is_main)
            if is_main:
                print(f"  Mid-epoch checkpoint: epoch {start_epoch}, "
                      f"resuming from batch {skip_batches + 1}, lr={_lr_before[0]:.3e}, "
                      f"global_step={global_step}")

    if is_ddp:
        dist.barrier()

    # ── Eval-only fine-path ablation: runs here, BEFORE W&B and the loop, writes nothing ──
    if args.eval_fine_ablation:
        _dw = (torch.tensor(depth_weights_list, device=device, dtype=torch.float32)
               if (CONFIG["per_depth_loss"] and depth_weights_list) else None)
        run_fine_ablation(raw_model, val_loader, device, world_size, rank, is_main, _dw,
                          ckpt_dir / args.eval_ckpt)
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

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

    # Device tensor form, rebuilt whenever depth_weights_list changes.
    depth_weights = (torch.tensor(depth_weights_list, device=device, dtype=torch.float32)
                     if (CONFIG["per_depth_loss"] and depth_weights_list) else None)
    if is_main:
        if not CONFIG["per_depth_loss"]:
            print("Per-depth loss OFF — depth_weights unused (pooled Huber).")
        elif depth_weights_list:
            print("Per-depth loss weights (fixed, inverse-frequency, mean 1): " +
                  "  ".join(f"{d}={w:.3f}" for d, w in zip(SM_DEPTHS, depth_weights_list)))
        else:
            print("Per-depth loss ON but no weight vector yet — epoch 1 runs UNWEIGHTED "
                  "(equivalent to pooled Huber); the weights are frozen from epoch 1's "
                  "observation counts and used from epoch 2 on. Epoch 1's loss is "
                  "therefore not comparable with epoch 2's.")

    # ── Memory snapshot (before first epoch) ─────────────────────────
    _log_mem_snapshot("job_start", device, is_main)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        torch.cuda.reset_peak_memory_stats(device)
        _log_mem_snapshot(f"epoch_{epoch:03d}_start", device, is_main)

        if epoch == val_pending_epoch:
            # Resuming after a val crash — training already completed, reuse saved metrics.
            # The per-depth vectors must be initialised here too: the DDP reduce below is
            # guarded on `epoch != val_pending_epoch`, but the print/log path is not.
            train_loss = saved_train_loss or 0.0
            train_tv   = saved_train_tv   or 0.0
            data_time = compute_time = 0.0
            train_depth_sum = torch.zeros(len(SM_DEPTHS), device=device)
            train_depth_cnt = torch.zeros(len(SM_DEPTHS), device=device)
            train_stats     = {}
        else:
            # Mid-epoch checkpoint callback.  Called on EVERY rank (see train_one_epoch):
            # the RNG gather is a collective, so it must not be rank-0-only.  Only rank 0
            # writes.  Every rank reaches the same batch index — drop_last=True gives all
            # ranks an identical batch count — so the collective is safe.
            def _save_mid_ckpt(batches_done, gstep):
                rng = _gather_rng_states(is_ddp, world_size)
                if not is_main:
                    return
                _fsync_save({
                    "epoch"       : epoch,
                    "model"       : raw_model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "batches_done": batches_done,
                    "global_step" : gstep,
                    "rng"         : rng,
                    "best_val_loss"   : best_val_loss,
                    "no_improve_count": no_improve_count,
                    "selection_metric": SELECTION_METRIC,
                    "depth_weights"   : depth_weights_list,
                    "lambda_lst"      : lam.state_dict(),
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
                (train_loss, train_tv, data_time, compute_time,
                 train_depth_sum, train_depth_cnt, global_step,
                 train_stats) = train_one_epoch(
                    model, _loader, optimizer, device, CONFIG["grad_clip"],
                    per_depth=CONFIG["per_depth_loss"],
                    max_batches=args.max_train_batches, debug_nan=args.debug_nan,
                    skip_batches = _skip,
                    mid_ckpt_every = 100,   # epochs are ~200 steps now (station-balanced sampler)
                    mid_ckpt_fn    = _save_mid_ckpt,
                    huber_delta    = CONFIG["huber_delta"],
                    depth_weights  = depth_weights,
                    global_step    = global_step,
                    warmup         = warmup,
                    is_main        = is_main,
                    log_every      = CONFIG["log_every"],
                    ddp_active     = is_ddp,
                    preempt_check_every = CONFIG["preempt_check_every"],
                    use_wandb      = use_wandb,
                    lam            = lam,
                    sigma_st       = CONFIG["lst_sig_eff"],
                    lst_delta      = CONFIG["lst_pat_delta"],
                    lst_level_weight = CONFIG["lst_level_weight"],
                    dT_sd          = CONFIG["lst_lvl_scale"],
                    lvl_delta      = CONFIG["lst_lvl_delta"],
                    lst_target     = CONFIG["lst_target"],
                )
            except _Preempted:
                # Every rank arrives here on the same batch (the all_reduce(MAX) in
                # train_one_epoch decided it collectively) and rank 0's save has already
                # completed behind that same barrier, so there is nothing left to
                # synchronise. destroy_process_group is best-effort: a peer that has
                # already exited must not turn a clean preempt into a crash-loop.
                print(f"[preempt] rank {rank}: SIGTERM — checkpoint saved, exiting for requeue")
                if is_ddp:
                    try:
                        dist.destroy_process_group()
                    except Exception as e:
                        print(f"[preempt] rank {rank}: destroy_process_group: {e}")
                raise SystemExit(0)
            _log_mem_snapshot(f"epoch_{epoch:03d}_post_train", device, is_main)

            # Save post-training checkpoint before validation — epoch not lost if val
            # crashes. The RNG gather is collective, so it happens on every rank first.
            # Un-warm param_group["lr"] first: during warmup it holds base*f, and a resume
            # from this checkpoint adopts param_group["lr"] as the new base
            # (sync_base_from_optimizer), permanently lowering the lr. Harmless here — no
            # optimizer step happens before the plateau step, which sets it to base anyway.
            warmup.before_plateau_step()
            _rng_states = _gather_rng_states(is_ddp, world_size)
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
                    "selection_metric": SELECTION_METRIC,
                    "depth_weights"   : depth_weights_list,
                    "lambda_lst"      : lam.state_dict(),
                    "global_step"     : global_step,
                    "rng"             : _rng_states,
                    "config"          : CONFIG,
                    "wandb_run_id"    : wandb.run.id if use_wandb else None,
                    "val_pending"     : True,
                }, ckpt_last)

        # All ranks evaluate their shard in parallel — all_reduce inside evaluate()
        # averages the loss across ranks; all_gather_object collects preds to rank 0.
        # No NCCL timeout risk: all GPUs stay active throughout validation.
        val_diag = {}
        val_loss, metrics, per_station, val_depth_sum, val_depth_cnt = evaluate(
            model if not is_ddp else model.module,
            val_loader, device,
            world_size=world_size, rank=rank,
            max_batches=args.max_val_batches,
            per_depth=CONFIG["per_depth_loss"],
            huber_delta=CONFIG["huber_delta"],
            depth_weights=depth_weights,
            diag_out=val_diag,
            sigma_st=CONFIG["lst_sig_eff"],
            lst_delta=CONFIG["lst_pat_delta"],
            dT_sd=CONFIG["lst_lvl_scale"],
            lvl_delta=CONFIG["lst_lvl_delta"],
            lst_target=CONFIG["lst_target"],
        )

        # ── Fine-path attribution (rank 0, no collectives) ──────────────────────────
        # Gradients w.r.t. INPUTS only, so no parameter gradient is produced and DDP's
        # reducer is untouched. Never fatal — a broken diagnostic must not kill a 120 h run.
        gr_diag = {}
        if CONFIG["input_grad_diag"] and is_main:
            try:
                gr_diag = input_grad_ratio(raw_model, val_diag.get("_first_batch"),
                                           device, CONFIG["huber_delta"],
                                           depth_weights=depth_weights)
            except Exception as e:
                print(f"  [diag] input-gradient attribution failed: {e}")
                gr_diag = {}
            optimizer.zero_grad(set_to_none=True)   # belt and braces
            raw_model.train()                       # restore the state evaluate() left
        val_diag.pop("_first_batch", None)           # release the held batch's VRAM

        _log_mem_snapshot(f"epoch_{epoch:03d}_post_val", device, is_main)
        # Release cached-but-free VRAM each epoch.  With expandable_segments:True this
        # cannot unmap segments (by design), so reserved memory will still grow if the
        # driver is fragmenting.  If growth continues, profile with
        # torch.cuda.memory._snapshot() or disable expandable_segments to isolate.
        torch.cuda.empty_cache()
        if is_ddp and epoch != val_pending_epoch:
            # Reduce train_loss and train_tv to rank 0 for accurate global average logging.
            # SUM then divide, not ReduceOp.AVG: NCCL documents AVG for all_reduce, and the
            # point-to-point `reduce` collective has raised "Cannot use ReduceOp.AVG with
            # boolean/this op" on some builds. SUM is unambiguously supported everywhere and
            # the division is free.
            t_loss = torch.tensor(train_loss, device=device)
            t_tv   = torch.tensor(train_tv,   device=device)
            dist.reduce(t_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(t_tv,   dst=0, op=dist.ReduceOp.SUM)
            t_loss /= world_size
            t_tv   /= world_size
            # Per-depth sums/counts reduce with SUM — all_reduce (not reduce) so every
            # rank stays in sync and the collective count matches the val path.
            dist.all_reduce(train_depth_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_depth_cnt, op=dist.ReduceOp.SUM)
            # Thermal term: raw sums, same rule. Unconditional so every rank joins.
            _lst_t = torch.tensor([train_stats.get("lst_sum", 0.0),
                                   train_stats.get("lst_cells", 0.0),
                                   train_stats.get("lst_level_sum", 0.0),
                                   train_stats.get("lst_level_n", 0.0)], device=device)
            dist.all_reduce(_lst_t, op=dist.ReduceOp.SUM)
            (train_stats["lst_sum"], train_stats["lst_cells"],
             train_stats["lst_level_sum"], train_stats["lst_level_n"]) = _lst_t.tolist()
            if is_main:
                train_loss = t_loss.item()
                train_tv   = t_tv.item()
        # Freeze the fixed per-depth weights from the first completed training epoch's
        # global observation counts. train_depth_cnt is all_reduce(SUM)'d above, so every
        # rank derives the identical vector without another collective. Done once ever:
        # the vector is written to disk and into the checkpoint, so a requeue does not
        # recompute (and therefore cannot change) the objective mid-run.
        if (CONFIG["per_depth_loss"] and depth_weights is None
                and epoch != val_pending_epoch and float(train_depth_cnt.sum()) > 0):
            depth_weights_list = _inverse_frequency_weights(train_depth_cnt)
            if depth_weights_list:
                depth_weights = torch.tensor(depth_weights_list, device=device,
                                             dtype=torch.float32)
                if is_main:
                    _dw_path.write_text(json.dumps({
                        "weights": depth_weights_list,
                        "counts" : [float(c) for c in train_depth_cnt.tolist()],
                        "depths" : SM_DEPTHS,
                        "frozen_at_epoch": epoch,
                    }, indent=2))
                    print(f"  [depth-weights] frozen from epoch {epoch} counts "
                          f"{[int(c) for c in train_depth_cnt.tolist()]}: " +
                          "  ".join(f"{d}={w:.3f}"
                                    for d, w in zip(SM_DEPTHS, depth_weights_list)) +
                          f"  -> {_dw_path.name}. In effect from epoch {epoch + 1}.")

        train_depth_loss = _per_depth_mean(train_depth_sum, train_depth_cnt)
        val_depth_loss   = _per_depth_mean(val_depth_sum,   val_depth_cnt)
        train_pooled, train_depth_mean = _loss_aggregates(train_depth_sum, train_depth_cnt)
        val_pooled,   val_depth_mean   = _loss_aggregates(val_depth_sum,   val_depth_cnt)
        # val_loss already all_reduced inside evaluate() — same on all ranks, no broadcast
        # needed. Caveat: the per-depth Huber sums still include the val sampler's padded
        # repeats (at most world_size-1 samples), because de-duplicating them would need
        # per-sample losses rather than sums. The metric path IS de-duplicated, so the
        # ubRMSE-based selection default is unaffected; the effect on val_pooled is <0.1%
        # at any realistic val size.

        # ── The selection scalar (§35.24 item 1) ─────────────────────────────────────
        # val_loss is a mean of per-batch means: every batch counts the same regardless of
        # how many valid (sample, depth) pairs it held, and the last batch of a shard is
        # usually short.  That makes it a function of BATCH COMPOSITION, not just of the
        # model — it changes with batch_size, with world_size, and with which stations
        # happened to land together.  Selecting best.pt on it, and stepping the LR plateau
        # on it, meant both were partly driven by shuffling.
        #
        # val_pooled = Σsum / Σcnt over the whole epoch and every rank weights each
        # observation once, full stop.  It was already being computed and merely logged.
        # It is now what selects and what schedules; val_loss stays in the log purely for
        # continuity with the runs that reported it.
        val_selection = val_pooled
        val_ubrmse_sel = float("nan")
        if CONFIG["select_metric"] == "ubrmse":
            # per_station only exists on rank 0 (evaluate gathers there), but scheduler.step
            # runs on every rank and must see the same number — so rank 0 computes and
            # broadcasts. float64 so the comparison against best_val_loss is bit-identical
            # everywhere.
            _sel_t = torch.tensor(
                [_ubrmse_selection(per_station) if is_main else 0.0],
                device=device, dtype=torch.float64)
            if is_ddp:
                dist.broadcast(_sel_t, src=0)
            val_ubrmse_sel = _sel_t.item()
            if math.isfinite(val_ubrmse_sel):
                val_selection = val_ubrmse_sel
            else:
                # Identical on every rank (val_pooled is all_reduced), so no divergence.
                if is_main:
                    print("  [select] WARNING: station-mean ubRMSE is undefined this epoch "
                          "(no station had >=2 val samples at any depth) — falling back to "
                          "val_huber_pooled for selection and scheduling THIS EPOCH ONLY. "
                          "best_val_loss is now comparing two different quantities; treat "
                          "any best.pt written this epoch with suspicion.")
        # scheduler.step is called on EVERY rank, so its argument must be identical on
        # every rank. val_pooled derives from val_depth_sum/cnt, which evaluate() already
        # all_reduce(SUM)'d, and val_ubrmse_sel was broadcast from rank 0 — so it is.
        # (Warmup hands the plateau its own un-warmed base lr and adopts whatever it
        # returns; see WarmupPlateauLR.)
        # Not during warmup: with ~200-step epochs, 1000 warmup steps span ~5 epochs, and a
        # plateau counted there could halve the base lr before full lr is ever reached.
        # global_step is identical on every rank, so every rank takes the same branch.
        _in_warmup = global_step < CONFIG["warmup_steps"]
        if not _in_warmup:
            warmup.before_plateau_step()
            scheduler.step(val_selection)
            warmup.after_plateau_step()

        if is_main:
            peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_util  = (compute_time / max(data_time + compute_time, 1e-6)) * 100
            # 6 decimals, not 4: at 4 dp the val_loss printed a flat 0.0022 for four
            # epochs of run 25150428 while it was actually rising 0.002182 -> 0.002230.
            print(f"\nEpoch {epoch:03d}  |  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
                  f"  data={data_time:.0f}s  compute={compute_time:.0f}s"
                  f"  gpu_util={gpu_util:.0f}%  peak_vram={peak_vram:.1f}GB")
            if train_stats:
                # clip_frac near 1.0 => every step was clipped => the effective step size is
                # grad_clip/||g||, not the lr printed anywhere in this log.
                print(f"  {'grad':>8s}  mean={train_stats['grad_norm_mean']:.4f}"
                      f"  max={train_stats['grad_norm_max']:.4f}"
                      f"  clipped={100 * train_stats['clip_frac']:.0f}% of steps"
                      f"  (clip={CONFIG['grad_clip']})")
            # Iterate SM_DEPTHS, not metrics: compute_metrics drops a depth entirely when
            # val has no samples for it, but train and val are different station sets, so
            # a depth can be trained and not validated. Printing from metrics would hide
            # that depth's train loss altogether.
            for depth in SM_DEPTHS:
                print(_format_depth_line(depth, train_depth_loss[depth],
                                         val_depth_loss[depth], metrics.get(depth)))
            # pooled is the only scalar whose definition is flag-independent — compare
            # runs on this, never on val_loss.  depth_mean is the plain average of the
            # three lines above, so the block reconciles without mental arithmetic.
            print(f"  {'pooled':>8s}  train={train_pooled:.6f}  val={val_pooled:.6f}"
                  f"   |  depth_mean  train={train_depth_mean:.6f}  val={val_depth_mean:.6f}")
            _tr_lst = (train_stats["lst_sum"] / train_stats["lst_cells"]
                       if train_stats.get("lst_cells") else float("nan"))
            print(f"  {'thermal':>8s}  train_lst={_tr_lst:.4f}"
                  f"  val_lst={val_diag.get('lst_loss', float('nan')):.4f}"
                  f"  (train_lst = the trained thermal loss, target={CONFIG['lst_target']}; "
                  f"val_lst = centred PATTERN Huber -- compare lst_px for dT_pixel; val cells="
                  f"{int(val_diag.get('lst_cells', 0))})"
                  f"  lambda={train_stats.get('lambda_lst', 0.0):.4e}"
                  f"  raw g_sm/g_lst={train_stats.get('lambda_raw_ratio', float('nan')):.4e}"
                  f"  push lst/sm={train_stats.get('lambda_lst', 0.0) / train_stats['lambda_raw_ratio'] if train_stats.get('lambda_raw_ratio', 0.0) and math.isfinite(train_stats['lambda_raw_ratio']) else float('nan'):.2f}"
                  f"   <-- NOT in the selection scalar")
            _lp = val_diag.get("lst_pattern")
            if _lp:
                # Spatial pattern only (both maps centred per scene). skill = 1 - err^2/obs^2:
                # 0 = no better than a flat map, 1 = perfect pattern.
                print(f"  {'lst_pat':>8s}  r={_lp['r_mean']:.3f}  r>0 in {100 * _lp['r_pos_frac']:.0f}%"
                      f"  RMSE={_lp['rmse_K']:.3f} K vs obs sd {_lp['obs_sd_K']:.3f} K"
                      f"  skill={_lp['skill']:.3f}  (val scenes={int(_lp['n_scenes'])})")
            _ll = val_diag.get("lst_level")
            if _ll and _ll.get("n", 0) >= 2:
                _tr_ll = (train_stats["lst_level_sum"] / train_stats["lst_level_n"]
                          if train_stats.get("lst_level_n") else float("nan"))
                # Pooled over val scenes (stations + seasons mixed), K. skill vs predicting the
                # val mean. Measured whether or not the level is trained (weight 0 = reference).
                print(f"  {'lst_lvl':>8s}  r={_ll['r']:.3f}  RMSE={_ll['rmse_K']:.3f} K"
                      f"  bias={_ll['bias_K']:+.3f} K  skill={_ll['skill']:.3f}"
                      f"  loss train={_tr_ll:.4f} val={_ll['loss']:.4f}"
                      f"  (val scenes={int(_ll['n'])}, weight={CONFIG['lst_level_weight']})")
            _lx = val_diag.get("lst_px")
            if _lx and _lx.get("n", 0) >= 2:
                # Per cell, LST - t2m_mean in K (the dT_pixel target). Pooled over val cells.
                print(f"  {'lst_px':>8s}  r={_lx['r']:.3f}  RMSE={_lx['rmse_K']:.3f} K"
                      f"  bias={_lx['bias_K']:+.3f} K  skill={_lx['skill']:.3f}"
                      f"  loss val={_lx['loss']:.4f}  (val cells={int(_lx['n'])}, "
                      f"target={CONFIG['lst_target']}, units={CONFIG['lst_units']})")
            print(f"  {'SELECT':>8s}  {SELECTION_METRIC}={val_selection:.6f}  <-- drives "
                  f"best.pt, early stopping and ReduceLROnPlateau."
                  f"   |  val_huber_pooled={val_pooled:.6f}"
                  f"  ubrmse_depth_mean={val_ubrmse_sel:.6f}"
                  f"  val_loss={val_loss:.6f} (mean-of-batch-means, batch-composition "
                  f"dependent — logged for continuity only)")
            # Within-station skill (§35.10). r is the number the gate is about: a model
            # that has learned only each station's climatology scores well on MSE/MAE and
            # gets r ~ 0 here.
            _wl = []
            for depth in SM_DEPTHS:
                m = metrics.get(depth)
                if not m:
                    continue
                _wl.append(f"{depth}: r={m['r_within']:.3f}/{m['r_station_mean']:.3f}"
                           f"  R2={m['R2_within']:.3f}/{m['R2_station_mean']:.3f}"
                           f"  anomRMSE={m['anomRMSE']:.4f}  (n_st={m['n_stations_scored']})")
            if _wl:
                print("  within-station  [pooled/station-mean]")
                for _line in _wl:
                    print(f"    {_line}")

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

            # Persist per-station metrics to CSV.
            # mode="a" was wrong on the resume path: a `val_pending` resume re-runs
            # validation for an epoch whose rows are already in the file, so the epoch
            # appeared twice with different numbers and every downstream groupby silently
            # averaged the two. The file is rewritten instead, with this epoch's rows
            # replacing any earlier attempt at the same (epoch, station, depth). It stays
            # small — ~74 stations x 3 depths x 100 epochs — so a full rewrite per epoch is
            # cheaper than the class of bug it removes.
            if per_station:
                csv_path = ckpt_dir / "val_station_metrics.csv"
                rows = [
                    {"epoch": epoch, "station": st, "depth": d,
                     "ubRMSE": m["ubRMSE"], "anomRMSE": m["anomRMSE"],
                     "MAE": m["MAE"], "bias": m["bias"],
                     "MSE": m["MSE"], "RMSE": m["RMSE"],
                     "r": m["r"], "R2": m["R2"], "n": m["n"]}
                    for st, dv in per_station.items()
                    for d, m in dv.items()
                ]
                new_df = pd.DataFrame(rows)
                if csv_path.exists():
                    try:
                        old_df = pd.read_csv(csv_path)
                        # Drop this epoch wholesale, then append: a re-validated epoch may
                        # legitimately cover a different station set than the first attempt
                        # (e.g. --max-val-batches changed), and keeping the stale remainder
                        # would mix two evaluations under one epoch number.
                        old_df = old_df[old_df["epoch"] != epoch]
                        new_df = pd.concat([old_df, new_df], ignore_index=True)
                        new_df = new_df.drop_duplicates(
                            subset=["epoch", "station", "depth"], keep="last")
                    except Exception as e:
                        print(f"  [metrics-csv] could not merge existing {csv_path.name} "
                              f"({e}) — writing this epoch's rows only")
                new_df.to_csv(csv_path, index=False)

            if use_wandb:
                log_dict = {
                    "epoch"        : epoch,
                    "train/loss"   : train_loss,
                    "train/tv"     : train_tv,
                    "val/loss"     : val_loss,
                    # The selection scalar, logged under an unambiguous name so a W&B
                    # panel cannot accidentally plot the non-selecting one.
                    "val/selection": val_selection,
                    "val/selection_name": SELECTION_METRIC,
                    "val/ubrmse_depth_mean": val_ubrmse_sel,
                    "lr"           : optimizer.param_groups[0]["lr"],
                    "opt/global_step": global_step,
                    "opt/warmup_factor": warmup.factor(global_step),
                    "perf/data_s"  : data_time,
                    "perf/compute_s": compute_time,
                    "perf/gpu_util": gpu_util,
                    "perf/peak_vram_gb": peak_vram,
                }
                # Gradient health (rank 0's shard; DDP averages grads so the norms agree
                # across ranks to within reduction order). clip_frac near 1.0 means the
                # effective step is grad_clip/||g||, not the lr logged above.
                for _k, _v in train_stats.items():
                    log_dict[f"opt/{_k}"] = _v
                log_dict["train/lst_loss"] = _tr_lst
                log_dict["val/lst_loss"]   = val_diag.get("lst_loss", float("nan"))
                log_dict["val/lst_cells"]  = val_diag.get("lst_cells", 0.0)
                for _k, _v in (val_diag.get("lst_pattern") or {}).items():
                    log_dict[f"val/lst_pattern/{_k}"] = _v
                for _k, _v in (val_diag.get("lst_level") or {}).items():
                    log_dict[f"val/lst_level/{_k}"] = _v
                for _k, _v in (val_diag.get("lst_px") or {}).items():
                    log_dict[f"val/lst_px/{_k}"] = _v
                if train_stats.get("lst_level_n"):
                    log_dict["train/lst_level_loss"] = (train_stats["lst_level_sum"]
                                                         / train_stats["lst_level_n"])
                for depth, m in metrics.items():
                    log_dict[f"val/{depth}/ubRMSE"] = m["ubRMSE"]
                    log_dict[f"val/{depth}/MAE"]    = m["MAE"]
                    log_dict[f"val/{depth}/bias"]   = m["bias"]
                    log_dict[f"val/{depth}/MSE"]    = m["MSE"]   # printed before, never logged
                    log_dict[f"val/{depth}/RMSE"]   = m["RMSE"]
                    # §35.10 within-station family. r_within/R2_within are pooled over the
                    # station-centred anomalies (sample-weighted); the *_station_mean pair
                    # weights each station equally, which is what the gate is stated in.
                    log_dict[f"val/{depth}/anomRMSE"]        = m["anomRMSE"]
                    log_dict[f"val/{depth}/r_within"]        = m["r_within"]
                    log_dict[f"val/{depth}/R2_within"]       = m["R2_within"]
                    log_dict[f"val/{depth}/r_station_mean"]  = m["r_station_mean"]
                    log_dict[f"val/{depth}/R2_station_mean"] = m["R2_station_mean"]
                    log_dict[f"val/{depth}/n_stations"]      = m["n_stations_scored"]
                # Per-depth Huber loss — the capacity-vs-scarcity diagnostic (runbook §19.1).
                # train high+flat => capacity/information ceiling; train low + val high =>
                # label scarcity.  These need opposite fixes.
                for depth in SM_DEPTHS:
                    log_dict[f"train/{depth}/loss"] = train_depth_loss[depth]
                    log_dict[f"val/{depth}/loss"]   = val_depth_loss[depth]
                _finite_val = [v for v in val_depth_loss.values() if math.isfinite(v)]
                if _finite_val:
                    log_dict["val/worst_depth_loss"] = max(_finite_val)
                # Flag-independent aggregates — see _loss_aggregates. huber_pooled is the
                # cross-run comparable scalar; val/loss is not.
                log_dict["train/huber_pooled"]     = train_pooled
                log_dict["train/huber_depth_mean"] = train_depth_mean
                log_dict["val/huber_pooled"]       = val_pooled
                log_dict["val/huber_depth_mean"]   = val_depth_mean
                # Mechanism check (§18.7 / §19.4): if the depth tokens stay mutually
                # identical, each depth is asking the same attention question and the
                # per-depth CLS prefix is inert.  Cosine should sit near -0.02 at init.
                # Unconditional now that the depth CLS prefix is an invariant.
                with torch.no_grad():
                    dt  = F.normalize(raw_model.depth_tokens.float(), dim=-1)
                    cos = dt @ dt.T
                for a in range(len(SM_DEPTHS)):
                    for b in range(a + 1, len(SM_DEPTHS)):
                        log_dict[f"diag/depth_token_cos_{a}{b}"] = cos[a, b].item()

                # ── Fine-path attribution and map flatness (§48) ──────────────────
                if gr_diag:
                    for _k, _v in gr_diag.items():
                        log_dict[f"diag/grad_{_k}"] = _v
                    print(f"  [diag] input-grad RMS  fine={gr_diag['fine_sum']:.3e}"
                          f"  rest={gr_diag['rest_sum']:.3e}"
                          f"  ratio={gr_diag['ratio']:.4f}   <-- ~0 means the SM loss is "
                          f"being minimised without reading the fine imagery at all")
                _msd = val_diag.get("map_sd")
                if _msd:
                    for _d, _s in zip(SM_DEPTHS, _msd):
                        log_dict[f"diag/map_sd_{_d}"] = _s
                    print("  [diag] within-tile SD of the 112x112 SM map  " +
                          "  ".join(f"{d}={s:.5f}" for d, s in zip(SM_DEPTHS, _msd)) +
                          "   <-- ~0 means the 20 m map is flat")

                # depth_ctx: the transformer OUTPUT for each depth slot, summed over the
                # whole val epoch. The input depth_tokens can stay near-orthogonal while the
                # outputs collapse onto one vector — that is use_cls_depth being inert, and
                # with the depth heads disconnected it is now load-bearing (§46.5 item 34).
                _ctx_n = val_diag.get("depth_ctx_n", 0.0)
                if _ctx_n > 0:
                    with torch.no_grad():
                        _dc = F.normalize(val_diag["depth_ctx_sum"].float() / _ctx_n, dim=-1)
                        _cc = _dc @ _dc.T
                    for a in range(len(SM_DEPTHS)):
                        for b in range(a + 1, len(SM_DEPTHS)):
                            log_dict[f"diag/depth_ctx_cos_{a}{b}"] = float(_cc[a, b])
                    log_dict["diag/depth_ctx_n"] = _ctx_n
                else:
                    print(f"  [diag] WARNING: depth-context sums were empty on "
                          f"{val_diag.get('n_ctx_missing', '?')}/"
                          f"{val_diag.get('n_batches', '?')} val batches — "
                          f"diag/depth_ctx_cos_* is NOT logged this epoch.")
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
        # The RNG gather is collective and therefore sits OUTSIDE the is_main block.
        _rng_states = _gather_rng_states(is_ddp, world_size)
        if is_main:
            # Selection is on val_selection (= val_pooled), NOT val_loss. best_val_loss
            # keeps its name for checkpoint-format continuity but now holds the pooled
            # statistic; "selection_metric" in the state records which it is.
            # Review fix (2026-09-30): save best.pt on a real improvement only. A worse WARMUP
            # epoch leaves no_improve_count at 0, which used to overwrite best.pt with worse
            # weights under the earlier epoch's metric.
            _improved = val_selection < best_val_loss
            if _improved:
                best_val_loss    = val_selection
                no_improve_count = 0
            elif not _in_warmup:          # warmup epochs never count toward early stopping
                no_improve_count += 1

            state = {
                "epoch"           : epoch,
                "model"           : raw_model.state_dict(),
                "optimizer"       : optimizer.state_dict(),
                "scheduler"       : scheduler.state_dict(),
                "val_loss"        : val_loss,
                "val_pooled"      : val_pooled,
                "val_ubrmse_depth_mean": val_ubrmse_sel,
                "selection_metric": SELECTION_METRIC,
                "depth_weights"   : depth_weights_list,
                "lambda_lst"      : lam.state_dict(),
                "best_val_loss"   : best_val_loss,
                "no_improve_count": no_improve_count,
                "global_step"     : global_step,
                "rng"             : _rng_states,
                "config"          : CONFIG,
                "wandb_run_id"    : wandb.run.id if use_wandb else None,
                "val_pending"     : False,
            }
            _fsync_save(state, ckpt_last)
            if mid_ckpt_path.exists():
                mid_ckpt_path.unlink()

            if _improved:
                _fsync_save(state, ckpt_dir / "best.pt")
                print(f"  New best {SELECTION_METRIC}={best_val_loss:.6f} — checkpoint saved")

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


    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
