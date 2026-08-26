"""
Training script for SoilMoistureModel — Phase 1 (sm_only).

Usage (terramind conda env):
    python train.py [--lr LR] [--batch-size N] [--n-layers N] [--run-name NAME]
                    [--max-stations N] [--warmup-steps N] [--huber-delta D]
                    [--log-every N]

Requires csvs/driver_stats.json (compute_driver_stats.py) — it supplies the per-depth
label means used to initialise the regression-head biases. Missing file = hard error.

Resume behaviour: if {checkpoint_dir}/{run_name}/last.pt exists the run
resumes automatically — no flag needed. Delete last.pt for a fresh start.
Resume restores the RNG streams, the global optimizer-step counter (so LR warmup does
not restart) and the per-rank sampler order, so a requeued run is the same experiment
as an uninterrupted one.

Model selection, early stopping and ReduceLROnPlateau all key off val_huber_pooled
(Σ per-depth Huber sums / Σ counts over the whole val epoch), NOT the mean-of-batch-
means `val_loss`, which depends on batch composition. See §35.24.

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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.multiprocessing

from dataset import SoilMoistureDataset, SM_DEPTHS
from model import SoilMoistureModel, masked_huber_loss

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

    def __getattr__(self, name):
        # Forward anything else (station lists, caches) to the wrapped dataset. Only called
        # for attributes IndexedDataset itself does not define. The explicit "ds" guard
        # prevents infinite recursion if this is ever consulted before __init__ has run
        # (unpickling in a spawn-start worker would do exactly that).
        if name == "ds":
            raise AttributeError(name)
        return getattr(self.ds, name)


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
                        max_train_stations: int | None = None,
                        max_val_stations:   int | None = None) -> None:
    """LOCAL rank 0 (one process per NODE): zarr L12 tokens → /dev/shm tmpfs memmaps.

    All ranks on that node then open the same files via numpy.memmap(mode='r') so the
    OS serves one shared physical copy — cuts node RAM by ~400 GB on the full run.
    Only train+val splits are preloaded: OOS/test stations are never touched during
    training, so preloading them wastes /dev/shm.

    Station capping mirrors SoilMoistureDataset EXACTLY (dataset.py, the
    `len(self._zarr_groups) >= max_stations` break).  Two properties matter and the
    old implementation had neither:

      * the caps are PER SPLIT.  train and val are separate SoilMoistureDataset
        instances, each with its own cap (max_stations and max_stations//5).  Walking
        train+val interleaved in CSV order against a single train-sized budget meant a
        smoke run's val stations were usually never reached, so val silently fell back
        to per-rank GPFS reads while train ran off shm — a confounded A/B.
      * a station counts against the cap as soon as it is SEEN (passes the category and
        soil_patch_ok filters), not when it is successfully written.  The dataset
        increments its dict before it knows whether the store opened, so a missing or
        incomplete store consumes a slot there.  Skipping it here without counting
        walked further down the CSV than the dataset ever will.
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

    n_written, n_seen_total = 0, 0
    for split_name, cap in (("train", max_train_stations), ("val", max_val_stations)):
        sub  = splits[splits["split"] == split_name]
        seen = set()                     # (cat, dir_name) — mirrors dataset's sat_dir key
        for _, r in sub.iterrows():
            if not bool(r.get("soil_patch_ok", True)):
                continue
            has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
            has_fl = str(r.get("has_flux",          "False")).lower() == "true"
            cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
            if str(r["source_network"]) == "ISMN":
                dir_name = f"ISMN_{r['network']}_{r['station_name']}"
            else:
                dir_name = f"{r['source_network']}_{r['station_id']}"

            key_seen = (cat, dir_name)
            if key_seen in seen:
                continue                 # dataset caches per sat_dir; extra rows are free
            if cap is not None and len(seen) >= cap:
                break                    # same break point the dataset takes
            seen.add(key_seen)
            n_seen_total += 1

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
                bin_path  = shm_dir / f"{dir_name}__{key}.bin"
                meta_path = shm_dir / f"{dir_name}__{key}.meta.json"
                # The resume check MUST come before the read. `zg[...][:]` pulls the whole
                # L12 array off GPFS — ~120 GB across the full station list — and the old
                # order did that read and then discarded it because the .bin already
                # existed. Every requeue of this --requeue/120 h job therefore paid the
                # entire preload (worst observed: 1901 s) to produce nothing.
                if bin_path.exists() and meta_path.exists():
                    wrote_any = True
                    continue
                arr = zg[f"{key}/l12"][:]
                mm = np.memmap(bin_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
                mm[:] = arr
                del mm                       # flush to tmpfs
                meta_path.write_text(json.dumps({"shape": list(arr.shape),
                                                 "dtype": str(arr.dtype)}))
                wrote_any = True
            if wrote_any:
                n_written += 1

    print(f"[SHM] L12 preloaded for {n_written} stations "
          f"({n_seen_total} scanned; caps train={max_train_stations} "
          f"val={max_val_stations}) → {shm_dir}")


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Paths
    "splits_csv"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv",
    "era5_stats"    : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/era5_stats.json",
    # Produced by compute_driver_stats.py.  Supplies SIF/TWSA/soil normalisation to
    # dataset.py and the per-depth head bias to this file (§35.24).  Fail closed: a
    # missing file raises rather than silently training heads from a zero bias, which
    # costs the first ~1k steps just walking the output up to the label mean.
    "driver_stats"  : "/gpfs/work3/0/prjs1968/soilMoisture/csvs/driver_stats.json",
    # Each run saves checkpoints under {checkpoint_dir}/{run_name}/
    "checkpoint_dir": "/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only",

    # Data
    "category_filter": ["sm_only"],
    "years"          : list(range(2016, 2023)),  # 2023 held out for OOT/OOST evaluation
    "seed"           : 42,

    # Training
    "batch_size"      : 128,
    "num_workers"     : 12,
    "val_num_workers" : 4,    # val uses 4w×pf4; train uses 12w×pf4 → (12+4)×4 ranks = 64 CPUs
    "prefetch_factor" : 4,
    "max_epochs"      : 100,
    "lr"              : 2e-4,
    "weight_decay"    : 0.05,
    "lr_patience"     : 10,
    "lr_factor"       : 0.5,
    "grad_clip"       : 1.0,
    "early_stop_patience": 20,
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
    # Once-per-val-epoch K=196 forward + input-gradient ratio (§35.19). Cheap, rank-0 only,
    # forward/one-backward, and it is the only check on the architecture's central claim.
    "patch_map_diag"  : True,
    "patch_map_diag_stations": 2,   # tiny separate token_sel="all" dataset; ~30 MB/sample

    # Model
    "n_depths"      : 3,
    "d_model"       : 768,
    "n_heads"       : 12,
    "n_layers"      : 6,
    "drop_path_rate": 0.1,
    # There is no use_cls_depth option any more. The per-patch sequence carries the depth
    # CLS tokens as a prefix and the model raises without them, so the flag was a lie in
    # three places at once: CONFIG said False, main() overwrote it to True unconditionally,
    # and the --use-cls-depth argparse entry was parsed and never read. Deleted rather than
    # documented — a knob with one legal value is not a knob.

    # Architecture (§35.18 / text/patchwise_math.md). "unet" is the pooled baseline and must
    # stay byte-identical; "patchwise" is the two-transformer model — T1 encodes the 431
    # tile-level driver tokens ONCE per sample, T2 runs each patch's 106-token sequence and
    # cross-attends into T1's cached K/V. STEP 1 has no decoder: the head predicts at 160 m.
    "driver_mode"   : "memory",  # memory | concat — how the drivers enter each patch sequence
    "driver_layers" : 2,         # T1 depth. NOT a repeat count: T1 runs once whatever its depth
    "token_sel"     : "station",  # "station" = K=1 (token 105), the only supervised setting;
                                  # "all" = K=196, inference only (~30 MB/sample IPC)
    "patch_token_dropout": 0.0,

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
    """Huber on the supervised patch. Returns (loss, tv) or (loss, tv, depth_sum, depth_cnt).

    `tv` is retained as a always-zero second element purely so the epoch bookkeeping and the
    W&B panels keep their shape. There is no TV term and no boundary term any more: both were
    defined on the 224x224 decoder map, which this architecture does not produce. The boundary
    penalty in particular was LIVE at 0.1 and its `.mean()` would have renormalised by
    ~50,176x (50,176 px x 3 depths against K=1 x 3) -- §35.22.

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
    if return_breakdown:
        loss, depth_sum, depth_cnt = masked_huber_loss(
            pred, label, delta=delta, per_depth=per_depth,
            depth_weights=depth_weights, return_breakdown=True)
        return loss, pred.new_zeros(1), depth_sum, depth_cnt
    return (masked_huber_loss(pred, label, delta=delta, per_depth=per_depth,
                              depth_weights=depth_weights),
            pred.new_zeros(1))


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
                     ddp_active=False, preempt_check_every=25):
    """Train one epoch.  If skip_batches > 0, fast-forwards past already-done
    batches (data loads but no GPU compute) then resumes training from that
    point.  Calls mid_ckpt_fn(batches_done) every mid_ckpt_every batches so
    rank 0 can save a recovery checkpoint.

    mid_ckpt_fn must now be passed on EVERY rank, not just rank 0: it performs a
    collective (the RNG all_gather) before rank 0 writes.  Non-main ranks' version is
    expected to participate and then return without writing.

    Returns (mean_loss, mean_tv, data_time, compute_time, depth_sum, depth_cnt,
             global_step, stats) — global_step is the running optimizer-step count that
    drives warmup across requeues; stats carries the gradient-norm summary.
    """
    model.train()
    total_loss   = torch.zeros((), device=device)   # kept on-device: see the log throttle
    total_tv     = torch.zeros((), device=device)
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
            mu = model(batch)

            if debug_nan:
                bad_out = _scan_for_nan({"mu": mu})
                if bad_out:
                    _report_nan(f"batch {n_batches+1:03d} OUTPUT", batch, bad_out)

            loss, tv, d_sum, d_cnt = _compute_loss(
                mu, batch["label"], per_depth, return_breakdown=True, delta=huber_delta,
                depth_weights=depth_weights)

        depth_sum_acc += d_sum
        depth_cnt_acc += d_cnt

        # Warmup is applied per OPTIMIZER STEP, immediately before the step, and is
        # driven by the global counter so a requeue resumes mid-ramp instead of
        # restarting the ramp.
        lr_factor = warmup.set_step(global_step) if warmup is not None else 1.0

        optimizer.zero_grad()
        loss.backward()

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
            print(f"  batch {skip_batches + n_batches:04d}  loss={loss.item():.4f}"
                  f"  gnorm={gnorm.item():.3f}  lr={optimizer.param_groups[0]['lr']:.3e}"
                  f"  wu={lr_factor:.2f}  step={step_ms:.0f}ms")

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
    }
    return (total_loss.item() / n, total_tv.item() / n, data_time, compute_time,
            depth_sum_acc, depth_cnt_acc, global_step, stats)


@torch.no_grad()
def patch_map_diag(raw_model, loader, device):
    """K=196 forward on ONE batch -> across-patch spread of the emitted 14x14 map.

    §35.19, the architecture's central untested claim.  Training supervises K=1 (patch 105)
    and inference asks for K=196, and NOTHING in the loss constrains the other 195 patches.
    If the tile-constant drivers (ERA5, SIF, TWSA, soil) explain most of the label variance,
    the loss is fully minimised by a function that ignores dem_k / lulc_k / the per-patch
    history entirely and emits 196 identical numbers — which would look like a perfectly
    healthy training curve and would make every 160 m map in the thesis a constant.

    pred.std(dim=1) over the patch axis is the direct test.  Near zero = the map is flat =
    the 160 m claim is unsupported by anything the model has actually learned.

    Forward-only and deliberately NOT routed through masked_huber_loss, which refuses
    K != 1 (correctly — supervising several patches needs multi-station labels).
    """
    raw_model.eval()
    for batch in loader:
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = raw_model(b)                                   # (B, K, n_depths)
        mu = mu.float()
        if mu.shape[1] < 2:
            return {"error": f"loader emitted K={mu.shape[1]}; expected 196 "
                             f"(token_sel='all' did not take effect)"}
        sd  = mu.std(dim=1).mean(dim=0)                          # (n_depths,)
        rng = (mu.amax(dim=1) - mu.amin(dim=1)).mean(dim=0)      # (n_depths,)
        return {"K": int(mu.shape[1]),
                "sd":    [float(v) for v in sd],
                "range": [float(v) for v in rng]}
    return {"error": "patch-map loader yielded no batches"}


def input_grad_ratio(raw_model, batch, device, huber_delta, depth_weights=None):
    """d(loss)/d(per-patch inputs) vs d(loss)/d(tile-constant inputs), RMS per element.

    The companion to patch_map_diag and the cheaper of the two: if the ratio is ~0 the model
    is not USING the per-patch inputs, which is the same failure the flat map would show,
    detectable one epoch earlier and without a second dataset.

    Norms are divided by sqrt(numel) so tensors of very different sizes — era5 is
    (B, 365, C) while s2_hist is (B, T, K, 768) — are compared as RMS per element rather
    than by raw magnitude.

    Uses torch.autograd.grad w.r.t. the INPUTS only: parameter AccumulateGrad nodes are not
    on any path to those leaves, so nothing lands in p.grad and DDP's reducer never sees a
    gradient it was not expecting.  Rank 0 only, one batch, K=1 (so the real loss applies).
    """
    if not batch:
        return {}
    per_patch_keys = ("dem_tok", "lulc_tok", "s2_hist", "s1_hist")
    tile_keys      = ("era5", "soil_patch", "sif", "twsa")
    b, leaves = dict(batch), {}
    for k in per_patch_keys + tile_keys:
        v = b.get(k)
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        t = v.detach().to(device).float().requires_grad_(True)
        b[k], leaves[k] = t, t
    if not leaves:
        return {}

    was_training = raw_model.training
    raw_model.eval()          # drop-path off: attribution, not a training step
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = raw_model(b)
            loss = masked_huber_loss(mu, b["label"].to(device), delta=huber_delta,
                                     per_depth=depth_weights is not None,
                                     depth_weights=depth_weights)
        grads = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
    finally:
        raw_model.train(was_training)

    out = {}
    for (k, t), g in zip(leaves.items(), grads):
        out[k] = (0.0 if g is None
                  else float(g.detach().float().norm().item() / max(t.numel() ** 0.5, 1.0)))
    pp = sum(out.get(k, 0.0) for k in per_patch_keys)
    tc = sum(out.get(k, 0.0) for k in tile_keys)
    out["per_patch_sum"] = pp
    out["tile_const_sum"] = tc
    out["ratio"] = (pp / tc) if tc > 0 else float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device, world_size=1, rank=0, max_batches=None, per_depth=False,
             huber_delta=0.05, depth_weights=None, diag_out=None):
    """Distributed-aware evaluation.

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

    # ── Collapse diagnostics (§35.24 contract) ───────────────────────────────────
    # Shapes are taken from the model up front, NOT lazily from the first forward: a rank
    # whose val shard is empty (or --max-val-batches 0) would otherwise have nothing to
    # allocate from and would skip the all_reduce, hanging the other three.
    want_diag   = diag_out is not None
    diag_blocks = getattr(model, "patch_blocks", None)
    diag_dim    = getattr(model, "d_model", None)
    diag_nd     = getattr(model, "n_depths", n_depths)
    if want_diag and (diag_blocks is None or diag_dim is None):
        if rank == 0:
            print("  [diag] WARNING: model exposes no patch_blocks/d_model — the §35.20 "
                  "collapse diagnostics are DISABLED for this run. This is not a "
                  "silent skip: attention-entropy and depth-context collapse are the "
                  "load-bearing evidence for step 1 and will be missing from W&B.")
        want_diag = False
    if want_diag:
        n_layers_diag = len(diag_blocks)
        # (n_layers, 3): [:,0] Σ entropy in nats, [:,1] Σ entropy/log(n_valid), [:,2] count
        ent_acc   = torch.zeros(n_layers_diag, 3, device=device)
        ctx_acc   = torch.zeros(diag_nd, diag_dim, device=device)
        ctx_n_acc = torch.zeros(1, device=device)
        n_ent_missing = 0
        n_ctx_missing = 0

    for batch in CudaPrefetcher(loader, device):
        if max_batches is not None and n_batches >= max_batches:
            break

        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu = model(batch)
            loss, _, d_sum, d_cnt = _compute_loss(
                mu, batch["label"], per_depth=per_depth, return_breakdown=True,
                delta=huber_delta, depth_weights=depth_weights)
        depth_sum_acc += d_sum
        depth_cnt_acc += d_cnt
        total_loss += loss.item()
        n_batches  += 1

        if want_diag:
            if n_batches == 1 and rank == 0:
                # Keep one K=1 batch for the input-gradient attribution afterwards. Costs
                # one batch of VRAM for the epoch and saves re-loading data for it.
                diag_out["_first_batch"] = {
                    k: (v.detach() if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
            ent = getattr(model, "_last_attn_entropy", None)
            if ent is None:
                n_ent_missing += 1
            else:
                e = ent.detach().float().to(ent_acc.device)
                if e.shape != ent_acc.shape:
                    raise RuntimeError(
                        f"[diag] _last_attn_entropy has shape {tuple(e.shape)}, expected "
                        f"{tuple(ent_acc.shape)} = (n_layers, 3) per the §35.24 contract "
                        f"[:,0]=Σnats, [:,1]=Σ(entropy/log n_valid), [:,2]=count."
                    )
                ent_acc += e
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

        # mu is (B, K, n_depths); the value IS the prediction and the dataset already
        # selected patch 105. No map, nothing to index.
        all_preds.append(mu[:, 0, :].float().cpu().numpy())
        all_targets.append(batch["label"].cpu().numpy())
        all_station_keys.extend(batch["station_key"])
        if "sample_idx" in batch:
            all_idx.append(batch["sample_idx"].cpu().numpy().reshape(-1))

    mean_loss = total_loss / max(n_batches, 1)

    if want_diag:
        # Raw SUMS over the whole val epoch on every rank -> SUM-reduce, exactly like the
        # per-depth Huber accumulators.  Unconditional: every rank allocated these above,
        # so no rank can skip the collective.
        if world_size > 1:
            dist.all_reduce(ent_acc,   op=dist.ReduceOp.SUM)
            dist.all_reduce(ctx_acc,   op=dist.ReduceOp.SUM)
            dist.all_reduce(ctx_n_acc, op=dist.ReduceOp.SUM)
        diag_out["attn_entropy_sums"] = ent_acc.cpu()
        diag_out["depth_ctx_sum"]     = ctx_acc.cpu()
        diag_out["depth_ctx_n"]       = float(ctx_n_acc.item())
        diag_out["n_ent_missing"]     = n_ent_missing
        diag_out["n_ctx_missing"]     = n_ctx_missing
        diag_out["n_batches"]         = n_batches
    elif diag_out is not None:
        diag_out["disabled"] = True

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
    # ── Architecture (§35.18) ───────────────────────────────────────────────
    parser.add_argument("--driver-mode", choices=["memory", "concat"], default=None,
                        help="memory: drivers are a read-only cross-attended memory, K/V cached "
                             "once per sample. concat: all 537 tokens in one self-attention stack")
    parser.add_argument("--driver-layers", type=int, default=None,
                        help="Depth of the driver (weather) encoder T1; default 2")
    parser.add_argument("--token-sel", choices=["station", "all"], default=None,
                        help="Which patches the dataset emits. station = K=1 (token 105), the "
                             "only supervised setting; all = K=196, inference only")
    parser.add_argument("--patch-token-dropout", type=float, default=None,
                        help="Per-patch token dropout during training (patchwise only)")
    # Regularisation overrides — CONFIG keeps the baseline values so comparison runs
    # stay clean; pass these on the sbatch line to make a run self-documenting.
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay (default 0.05; bias/norm always excluded)")
    parser.add_argument("--drop-path-rate", type=float, default=None,
                        help="Stochastic depth rate, linearly scaled across layers (default 0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="Epochs without val improvement before stopping (default 20)")
    parser.add_argument("--lr-patience", type=int, default=None,
                        help="ReduceLROnPlateau patience in epochs (default 10)")
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
    parser.add_argument("--input-norm", action="store_true",
                        help="LayerNorm the frozen TerraMind features on the way in. OFF by "
                             "default (§35.26): it deletes token magnitude, 9.3%% of S2's "
                             "TEMPORAL variance rides there even with the registers stripped, "
                             "and the frozen pooled baseline does not do it. Pass this to run "
                             "it as a deliberate ablation.")
    parser.add_argument("--no-patch-map-diag", action="store_true",
                        help="Disable the once-per-epoch K=196 patch-map diagnostic "
                             "(across-patch SD of the emitted map + per-patch vs "
                             "tile-constant input-gradient ratio). On by default")
    args = parser.parse_args()

    if args.lr          is not None: CONFIG["lr"]         = args.lr
    if args.batch_size  is not None: CONFIG["batch_size"] = args.batch_size
    if args.n_layers    is not None: CONFIG["n_layers"]   = args.n_layers
    if args.run_name    is not None: CONFIG["run_name"]   = args.run_name
    if args.num_workers     is not None: CONFIG["num_workers"]     = args.num_workers
    if args.prefetch_factor is not None: CONFIG["prefetch_factor"] = args.prefetch_factor
    if args.max_epochs  is not None: CONFIG["max_epochs"] = args.max_epochs
    if args.per_depth_loss: CONFIG["per_depth_loss"] = True
    if args.driver_mode   is not None: CONFIG["driver_mode"]   = args.driver_mode
    if args.driver_layers is not None: CONFIG["driver_layers"] = args.driver_layers
    if args.token_sel     is not None: CONFIG["token_sel"]     = args.token_sel
    if args.patch_token_dropout is not None:
        CONFIG["patch_token_dropout"] = args.patch_token_dropout

    # token_sel is a startup invariant, not a runtime one. masked_huber_loss refuses K != 1
    # (multi-station labels do not exist yet, §35.19) and evaluate() hardcodes mu[:, 0, :],
    # so --token-sel all cannot train — but it only died on the FIRST BACKWARD, ~30 minutes
    # in, after the shm preload had already read 120 GB and four H100s had been held for the
    # whole of it. Fail here instead, before anything is allocated.
    if CONFIG["token_sel"] != "station":
        raise ValueError(
            f"--token-sel {CONFIG['token_sel']!r} cannot be trained on: the loss requires "
            f"K=1 (one supervised patch per sample) and evaluate() reads mu[:, 0, :]. "
            f"token_sel='all' is an INFERENCE-only setting for 14x14 map figures — use "
            f"eval_predict.py / eval_stations.py for it."
        )

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
    if args.weight_decay        is not None: CONFIG["weight_decay"]        = args.weight_decay
    if args.drop_path_rate      is not None: CONFIG["drop_path_rate"]      = args.drop_path_rate
    if args.early_stop_patience is not None: CONFIG["early_stop_patience"] = args.early_stop_patience
    if args.lr_patience         is not None: CONFIG["lr_patience"]         = args.lr_patience
    if args.warmup_steps        is not None: CONFIG["warmup_steps"]        = args.warmup_steps
    if args.huber_delta         is not None: CONFIG["huber_delta"]         = args.huber_delta
    if args.log_every           is not None: CONFIG["log_every"]           = args.log_every
    if args.select_metric       is not None: CONFIG["select_metric"]       = args.select_metric
    if args.no_patch_map_diag:               CONFIG["patch_map_diag"]      = False

    # ── L12 shared memory preloading (before DDP init to avoid TCPStore timeout) ──
    # Rank 0 reads ~120 GB from GPFS — can take several minutes. Doing this before
    # dist.init_process_group() means ranks 1-3 spin on a sentinel file rather than
    # inside an NCCL communicator setup that times out after 600 s.
    SHM_DIR  = Path(f"/dev/shm/sm_l12_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    _shm_done = SHM_DIR / ".done"
    # LOCAL_RANK, not RANK.  /dev/shm is per NODE, so the preload has to run once per node
    # — and the sentinel it drops is likewise only visible on that node.  Gating on global
    # RANK meant that on a 2-node launch node 1's shm was never populated AND its ranks
    # (global 4-7, local 0-3) waited on a sentinel that could only ever appear on node 0.
    # They then spun the full 3 h _SHM_WAIT_MAX holding four H100s before dying, on a job
    # that was otherwise healthy.  Single-node launches are unaffected: there LOCAL_RANK 0
    # is global rank 0.
    _pre_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _pre_rank       = int(os.environ.get("RANK", "0"))
    # Caps must match the dataset's, which applies max_stations to the train split and
    # max_stations//5 to val INDEPENDENTLY (see val_max_stations below).
    _val_cap = max(1, args.max_stations // 5) if args.max_stations is not None else None
    if _pre_local_rank == 0:
        SHM_DIR.mkdir(parents=True, exist_ok=True)
        # The explicit rmtree at the bottom of main() is unreachable on the preempt path —
        # the _Preempted handler raises SystemExit(0) — so a preempted job stranded ~145 GB
        # of tmpfs on the node, and SLURM does not clear /dev/shm between steps. atexit runs
        # on SystemExit and on a normal return, so it covers both. (It cannot cover SIGKILL;
        # nothing in-process can.) Registered on local rank 0 only: /dev/shm is per node and
        # a non-owner must never delete another rank's live memmaps.
        import atexit as _atexit
        _atexit.register(lambda: shutil.rmtree(SHM_DIR, ignore_errors=True))
        t_shm = time.perf_counter()
        _preload_l12_to_shm(CONFIG["splits_csv"], CONFIG.get("category_filter"), SHM_DIR,
                            max_train_stations=args.max_stations,
                            max_val_stations=_val_cap)
        _shm_done.touch()
        print(f"[SHM] Preload done in {time.perf_counter() - t_shm:.1f}s  ({SHM_DIR})")
    else:
        # Bounded wait. Rank 0's preload reads ~120 GB from GPFS and took 1901 s in the worst
        # observed case, so the ceiling is generous — but it must exist: an unbounded spin
        # means a rank-0 death during preload leaves ranks 1-3 idling for the full 120 h
        # walltime while holding four H100s.
        _SHM_WAIT_MAX = 3 * 3600
        _t_wait = time.perf_counter()
        while not _shm_done.exists():
            if time.perf_counter() - _t_wait > _SHM_WAIT_MAX:
                raise RuntimeError(
                    f"[SHM] rank {_pre_rank} (local {_pre_local_rank}): waited "
                    f"{_SHM_WAIT_MAX/3600:.0f} h for this NODE's local-rank-0 preload "
                    f"sentinel ({_shm_done}) and it never appeared — local rank 0 most "
                    f"likely died during preload. Failing fast instead of holding the GPUs."
                )
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
        token_sel        = CONFIG["token_sel"],
        patch_token_dropout = CONFIG["patch_token_dropout"],
    )
    # Same object the shm preloader was capped with — computed once so the two can never
    # drift apart again (§35.24 item 7).
    val_max_stations = _val_cap
    # file_system strategy avoids fd exhaustion with 32 workers; must be set before workers spawn
    torch.multiprocessing.set_sharing_strategy("file_system")

    train_dataset = SoilMoistureDataset(**common_kwargs, split_filter=["train"], training=True,
                                         max_stations=args.max_stations)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                        shuffle=True, drop_last=True) if is_ddp else None

    # Val dataset on all ranks — DistributedSampler splits it across GPUs.
    # Wrapped so each item carries its dataset index: the val sampler runs drop_last=False
    # and pads the final shard with repeats, which compute_metrics would otherwise count
    # twice (see IndexedDataset).
    val_dataset = IndexedDataset(
        SoilMoistureDataset(**common_kwargs, split_filter=["val"], training=False,
                            max_stations=val_max_stations))
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

    # ── K=196 patch-map diagnostic loader (§35.19) ────────────────────
    # A SEPARATE, deliberately tiny dataset: token_sel="all" restores the ~30 MB/sample IPC
    # payload, so it gets 2 stations, batch_size 2 and num_workers 0, on rank 0 only. It
    # exists to answer one question once per epoch — does the model emit 196 different
    # numbers, or 196 copies of one? — and nothing in the training loss asks that question.
    # Built here rather than lazily so a mistake surfaces before the first epoch, not four
    # hours in. Failure to build is a warning, never fatal: this is a diagnostic.
    patch_map_loader = None
    if CONFIG["patch_map_diag"] and is_main:
        try:
            _pm_kwargs = dict(common_kwargs)
            _pm_kwargs["token_sel"] = "all"
            _pm_kwargs["patch_token_dropout"] = 0.0
            _pm_ds = SoilMoistureDataset(**_pm_kwargs, split_filter=["val"], training=False,
                                         max_stations=CONFIG["patch_map_diag_stations"])
            if len(_pm_ds) == 0:
                raise RuntimeError("patch-map dataset is empty")
            patch_map_loader = DataLoader(_pm_ds, batch_size=2, shuffle=False,
                                          num_workers=0, pin_memory=False)
            print(f"  [diag] patch-map loader ready: {len(_pm_ds)} samples, token_sel=all")
        except Exception as e:
            print(f"  [diag] WARNING: could not build the K=196 patch-map loader ({e}) — "
                  f"diag/patch_map_sd_* will be MISSING. The 160 m claim then has no "
                  f"check at all this run; --no-patch-map-diag silences this deliberately.")
            patch_map_loader = None

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
        n_depths       = CONFIG["n_depths"],
        d_model        = CONFIG["d_model"],
        n_heads        = CONFIG["n_heads"],
        n_layers       = CONFIG["n_layers"],
        drop_path_rate = CONFIG.get("drop_path_rate", 0.1),
        use_cls_depth  = True,   # invariant, not a setting — see the CONFIG comment
        driver_mode    = CONFIG.get("driver_mode", "memory"),
        driver_layers  = CONFIG.get("driver_layers", 2),
        head_bias_init = head_bias_init,
        use_input_norm = args.input_norm,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_ddp else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable parameters: {n_params:,}")
        # Echo the run-defining config. It is saved into the checkpoint too, but a job
        # log should be readable on its own — otherwise which flags a run actually used
        # can only be recovered by torch.load-ing a 600 MB checkpoint.
        _echo = ["run_name", "driver_mode", "driver_layers", "token_sel",
                 "per_depth_loss", "lr", "warmup_steps", "huber_delta", "batch_size",
                 "weight_decay", "drop_path_rate",
                 "n_layers", "early_stop_patience", "lr_patience",
                 "select_metric", "patch_map_diag", "git_sha"]
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
                    mid_ckpt_every = 500,
                    mid_ckpt_fn    = _save_mid_ckpt,
                    huber_delta    = CONFIG["huber_delta"],
                    depth_weights  = depth_weights,
                    global_step    = global_step,
                    warmup         = warmup,
                    is_main        = is_main,
                    log_every      = CONFIG["log_every"],
                    ddp_active     = is_ddp,
                    preempt_check_every = CONFIG["preempt_check_every"],
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
                    "global_step"     : global_step,
                    "rng"             : _rng_states,
                    "config"          : CONFIG,
                    "wandb_run_id"    : wandb.run.id if use_wandb else None,
                    "val_pending"     : True,
                }, ckpt_last)

        # All ranks evaluate their shard in parallel — all_reduce inside evaluate()
        # averages the loss across ranks; all_gather_object collects preds to rank 0.
        # No NCCL timeout risk: all GPUs stay active throughout validation.
        # Arm the entropy diagnostic for the validation pass only: collecting attention
        # weights forces the math kernel and gives up SDPA, so it stays off during training.
        for _blk in raw_model.patch_blocks:
            _blk.collect_entropy = True

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
        )
        # Disarm immediately. Collecting attention weights forces need_weights=True, which gives
        # up the SDPA/flash kernel; leaving it on would silently slow every subsequent TRAINING
        # epoch, not just this validation pass.
        for _blk in raw_model.patch_blocks:
            _blk.collect_entropy = False

        # ── §35.19 patch-map diagnostics (rank 0, no collectives) ───────────────────
        # Both run on the RAW module. The K=196 pass is forward-only under no_grad; the
        # gradient-ratio pass takes grads w.r.t. INPUTS only, so no parameter gradient is
        # produced and DDP's reducer is untouched. Other ranks simply wait a few seconds at
        # the next collective. Never fatal — a broken diagnostic must not kill a 120 h run.
        pm_diag, gr_diag = {}, {}
        if CONFIG["patch_map_diag"] and is_main:
            if patch_map_loader is not None:
                try:
                    pm_diag = patch_map_diag(raw_model, patch_map_loader, device)
                except Exception as e:
                    print(f"  [diag] patch-map forward failed: {e}")
                    pm_diag = {"error": str(e)}
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

                # ── §35.19 patch-map / input-attribution ────────────────────────────
                if pm_diag.get("error"):
                    print(f"  [diag] patch-map: {pm_diag['error']}")
                elif pm_diag:
                    for _d, _sd, _rg in zip(SM_DEPTHS, pm_diag["sd"], pm_diag["range"]):
                        log_dict[f"diag/patch_map_sd_{_d}"]    = _sd
                        log_dict[f"diag/patch_map_range_{_d}"] = _rg
                    log_dict["diag/patch_map_sd_mean"] = sum(pm_diag["sd"]) / len(pm_diag["sd"])
                    print(f"  [diag] patch map (K={pm_diag['K']}) across-patch SD  " +
                          "  ".join(f"{d}={s:.5f}" for d, s in zip(SM_DEPTHS, pm_diag["sd"])) +
                          "   <-- ~0 means the 14x14 map is CONSTANT and the 160 m claim "
                          "is unsupported")
                if gr_diag:
                    for _k, _v in gr_diag.items():
                        log_dict[f"diag/grad_{_k}"] = _v
                    print(f"  [diag] input-grad RMS  per_patch={gr_diag['per_patch_sum']:.3e}"
                          f"  tile_const={gr_diag['tile_const_sum']:.3e}"
                          f"  ratio={gr_diag['ratio']:.4f}   <-- ~0 means the loss is being "
                          f"minimised without reading the per-patch inputs at all")

                # ── §35.20 collapse diagnostics ─────────────────────────────────────
                # Both of these are load-bearing evidence for step 1 and both were
                # reporting nothing:
                #
                #   depth_ctx: read via getattr(raw_model, "_last_depth_ctx", None), which
                #     has been None on every epoch since the U-Net strip (commit fe0dc2c) —
                #     the only surviving producer is model_unet.py. getattr's default
                #     turned a dead diagnostic into an absent W&B key, and an absent key
                #     looks exactly like a key nobody plotted.
                #
                #   attn_entropy: a single-batch, rank-0-only snapshot of whatever
                #     _last_attn_entropy happened to hold after the final forward of one
                #     shard, compared against a FIXED math.log(MAX_S2 + MAX_S1) = log(100)
                #     reference. That reference was wrong in both directions at once: a
                #     sample almost never has all 100 history slots valid, so the true
                #     uniform ceiling is log(n_valid) < log(100) and a genuinely collapsed
                #     row read as "well below uniform"; and nothing constrained the entropy
                #     to be over history keys only, so driver/CLS keys inflated it.
                #
                # Both are now epoch-wide sums accumulated inside evaluate() and
                # all_reduce(SUM)'d there, per the §35.24 contract. The scale-free ratio
                # (entropy / log n_valid, per sample, per head, per readout row) replaces
                # the fixed reference: 1.0 means uniform means collapsed, on any sample
                # whatever its history length.
                if not val_diag or val_diag.get("disabled"):
                    print("  [diag] WARNING: §35.20 collapse diagnostics unavailable this "
                          "epoch — nothing logged under diag/attn_entropy* or "
                          "diag/depth_ctx*.")
                else:
                    ent_sums = val_diag["attn_entropy_sums"]        # (n_layers, 3)
                    if val_diag["n_ent_missing"]:
                        print(f"  [diag] WARNING: _last_attn_entropy was absent on "
                              f"{val_diag['n_ent_missing']}/{val_diag['n_batches']} val "
                              f"batches — entropy below is averaged over the rest. If this "
                              f"is all of them, the model is not populating the stash and "
                              f"the collapse detector is BLIND.")
                    _ent_nats, _ent_ratio = [], []
                    for _i in range(ent_sums.shape[0]):
                        _cnt = float(ent_sums[_i, 2])
                        if _cnt <= 0:
                            continue
                        _nats  = float(ent_sums[_i, 0]) / _cnt
                        _ratio = float(ent_sums[_i, 1]) / _cnt
                        log_dict[f"diag/attn_entropy_L{_i}"]       = _nats
                        log_dict[f"diag/attn_entropy_ratio_L{_i}"] = _ratio
                        _ent_nats.append(_nats)
                        _ent_ratio.append(_ratio)
                    if _ent_nats:
                        log_dict["diag/attn_entropy_mean"]       = sum(_ent_nats) / len(_ent_nats)
                        # THE number to watch: 1.0 = uniform attention over the valid
                        # history = the model is reading a mean, which is precisely what
                        # un-pooling exists to escape.
                        log_dict["diag/attn_entropy_ratio_mean"] = sum(_ent_ratio) / len(_ent_ratio)
                        log_dict["diag/attn_entropy_ratio_max"]  = max(_ent_ratio)
                        print(f"  [diag] attn entropy  mean={log_dict['diag/attn_entropy_mean']:.3f} nats"
                              f"  collapse_ratio mean={log_dict['diag/attn_entropy_ratio_mean']:.4f}"
                              f"  max={log_dict['diag/attn_entropy_ratio_max']:.4f}"
                              f"   (1.0 = uniform = collapsed)")
                    else:
                        print("  [diag] WARNING: attention-entropy counts were all zero — "
                              "collect_entropy did not arm, or no readout row contributed.")

                    # depth_ctx: the transformer OUTPUT for each depth slot, summed over
                    # the whole val epoch. The input depth_tokens can stay near-orthogonal
                    # while the outputs collapse onto one vector — that is exactly
                    # use_cls_depth being inert, and only this pair of cosines shows it.
                    _ctx_n = val_diag["depth_ctx_n"]
                    if _ctx_n > 0:
                        with torch.no_grad():
                            _dc  = F.normalize(val_diag["depth_ctx_sum"].float() / _ctx_n, dim=-1)
                            _cc  = _dc @ _dc.T
                        for a in range(len(SM_DEPTHS)):
                            for b in range(a + 1, len(SM_DEPTHS)):
                                log_dict[f"diag/depth_ctx_cos_{a}{b}"] = float(_cc[a, b])
                        log_dict["diag/depth_ctx_n"] = _ctx_n
                    else:
                        print(f"  [diag] WARNING: _last_depth_ctx / _last_depth_ctx_n were "
                              f"absent on {val_diag['n_ctx_missing']}/"
                              f"{val_diag['n_batches']} val batches and the accumulated "
                              f"count is 0 — diag/depth_ctx_cos_* is NOT logged this "
                              f"epoch. The depth-collapse check is blind until the model "
                              f"populates those attributes.")
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
            if val_selection < best_val_loss:
                best_val_loss    = val_selection
                no_improve_count = 0
            else:
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

            if no_improve_count == 0:
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

    # Local rank 0, matching the preload: /dev/shm is per node, so on a multi-node launch
    # global rank 0 can only ever clean its own node's tmpfs and node 1's ~145 GB would
    # survive the job.
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and SHM_DIR.exists():
        shutil.rmtree(SHM_DIR, ignore_errors=True)
        print(f"[SHM] Cleaned up {SHM_DIR}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
