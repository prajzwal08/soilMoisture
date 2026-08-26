# Soil Moisture Training Runbook

Last updated: 2026-08-05 (Session 20 — §19 per-depth loss reporting + regularisation run)  
Author: Prajwal Khanal

---

## 1. Data Pipeline Flowchart

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        STORAGE TIER                                      │
│                                                                          │
│  GPFS scratch (zarr)                  /dev/shm (tmpfs)                  │
│  /gpfs/scratch1/shared/pkhanal/zarr   /dev/shm/sm_l12_<JOBID>/          │
│                                                                          │
│  Per station:                         Per station × orbit:               │
│  ├─ s2/{dates,l12,cm/masks,...}       <station>__s2.bin   (N,196,768)   │
│  ├─ s1_asc/{dates,l12,token_mask}     <station>__s1_asc.bin             │
│  ├─ s1_desc/{...}                     <station>__s1_desc.bin            │
│  ├─ era5/{values,date_ints,doys}      All memmapped; OS serves ONE      │
│  ├─ sif/{...}  twsa/{...}             physical copy across 4 DDP ranks  │
│  └─ labels/{soil_moisture,...}        Total: ~91 GB shared               │
│                                                                          │
│  WRITES to scratch: none during training (read-only)                    │
└──────────┬───────────────────────────────────────┬───────────────────────┘
           │ one-time bulk reads at init           │ mmap (zero-copy)
           │ (inside DataLoader worker fork)       │
           ▼                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  SoilMoistureDataset.__init__()  (rank 0 ONLY)          │
│                                                                          │
│  Per station (one-time, then CoW-forked to workers):                    │
│  _l12_cache           → /dev/shm memmaps or zarr fallback              │
│  _era5_cache          → (N,19) float32 numpy + date_ints               │
│  _sif_cache           → (N,1) float32 + date_ints                      │
│  _twsa_cache          → (N,1) float32 + date_ints                      │
│  _label_cache         → (n_depths, T) float32 + QC flags               │
│  _static_cache        → DEM/LULC tensors + token masks + soil patch    │
│  _cm_token_mask_cache → (N_cm, 14, 14) bool  ← BULK CM zarr read      │
│  _s1_token_mask_cache → {orbit: (N, 14, 14) bool}                     │
│  _zarr_date_cache     → {orbit: {dates, date_ints, years, doys}}       │
│                                                                          │
│  Total init RAM per rank: ~62 GB (CoW shared; ~15 GB unique per rank)  │
└──────────┬───────────────────────────────────────────────────────────────┘
           │ fork() — Copy-on-Write; workers share init data physically
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              DataLoader WORKER  (×8 per rank, persistent)               │
│                                                                          │
│  __getitem__(idx)  ← called by worker                                   │
│  ├─ load_s2_rolling_zarr()   uses _zarr_date_cache + _cm_token_mask_cache│
│  │    zero GPFS reads; L12 from /dev/shm mmap                          │
│  ├─ load_s1_rolling_zarr()   uses _zarr_date_cache + _s1_token_mask_cache│
│  │    zero GPFS reads; L12 from /dev/shm mmap                          │
│  ├─ select_anchor_zarr()     uses all three caches for date/quality     │
│  │    GPFS reads: anchor L3 + L6 + L9 only (~882 KB/sample)            │
│  ├─ load_era5_rolling()      numpy slice from _era5_cache (RAM only)   │
│  ├─ load_sif_rolling()       numpy slice + vectorized rel_pos           │
│  ├─ load_twsa_rolling()      numpy slice + vectorized rel_pos           │
│  └─ build batch dict  ~30 MB per sample                                 │
│                                                                          │
│  Per-sample GPFS IO: ~882 KB  (down from ~3+ MB; 3.5× reduction)      │
└──────────┬───────────────────────────────────────────────────────────────┘
           │ file_system IPC  (/tmp temp files; file_system strategy)
           │ pf=3: 3 batches buffered per worker
           │ Total IPC RAM:  8 workers × 3 pf × 128 samples × 30 MB × 4 ranks
           │               = 440 GB
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            MAIN PROCESS  (1 per rank, runs on CPU)                      │
│                                                                          │
│  CudaPrefetcher                                                          │
│  ├─ batch = next(iter)          ← ready in prefetch queue               │
│  ├─ .to(device, non_blocking=True)  ← H2D DMA on dedicated CUDA stream │
│  └─ current_stream.wait_stream()   ← ensures batch N+1 arrives before  │
│                                      GPU compute on batch N starts      │
│                                                                          │
│  Overlap: H2D transfer of batch N+1  ‖  GPU compute of batch N         │
└──────────┬───────────────────────────────────────────────────────────────┘
           │ CUDA copy stream (async, non_blocking)
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 GPU  (4× H100, 100 GB VRAM each)                        │
│                                                                          │
│  Compute stream:                                                         │
│  ├─ SoilMoistureModel.forward()   bfloat16 autocast                    │
│  │    ├─ TemporalTransformer (S2/S1/DEM/LULC tokens)                   │
│  │    ├─ Random token masking p=0.5 (training only) — GPU-side         │
│  │    └─ ERA5 / SIF / TWSA fusion                                       │
│  ├─ _compute_loss()                                                      │
│  │    ├─ masked_huber_loss()                                            │
│  │    ├─ total_variation_loss()  λ=0.1                                  │
│  │    └─ boundary_penalty()     λ=0.1  (F.relu(-μ) + F.relu(μ-1))     │
│  ├─ loss.backward()                                                     │
│  ├─ clip_grad_norm_(1.0)                                                │
│  └─ optimizer.step()  (AdamW)                                           │
│                                                                          │
│  DDP: gradient AllReduce across 4 ranks after each backward()           │
│  Peak VRAM: ~48 GB / 100 GB  (gradient checkpointing not needed yet)   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Memory Pool Reference Table

| Pool | Size | Lifetime | Shared across ranks? |
|---|---|---|---|
| `/dev/shm` L12 preload | **145 GB** (measured) | Job lifetime | YES — OS page cache; 1 copy for 4 ranks |
| Dataset init caches (ERA5/SIF/TWSA/labels) | ~62 GB total | Job lifetime | CoW fork — stable with `gc.freeze()` |
| CM token mask cache | ~11 MB | Job lifetime | CoW fork |
| S1 token mask cache | ~9 MB | Job lifetime | CoW fork |
| Zarr date cache | ~4 MB | Job lifetime | CoW fork |
| Static cache (DEM/LULC/soil) | ~2 GB | Job lifetime | CoW fork |
| Train DataLoader IPC | **~16 GB** (8w×pf2×bs128×2MB×4r) | Epoch lifetime | NO — 4 × independent IPC per rank |
| Val DataLoader IPC | **~4 GB** (2w×pf2×bs128×2MB×4r) | Val phase only | NO — 4 × independent |
| CUDA activations (peak) | ~56 GB (measured) | Per-batch | Per-GPU only |
| Model params + Adam states | ~2 GB | Job lifetime | Per-GPU (replicated by DDP) |

**SLURM budget: `--mem=720G` (shared node cap; 790G never schedules on shared nodes)**
**Post pyramid-pool fix: boundary peak ≈ 324 GB → 396 GB headroom vs 720G.**

```
During training epoch:
  /dev/shm L12:          145 GB  (measured; shared across 4 ranks — still needed for pooling)
  4 × rank heaps:        159 GB  (CoW; stable with gc.freeze())
  Train loader IPC:       16 GB  (8w × pf2 × bs128 × ~2 MB × 4r)
  ─────────────────────────────
  Total:                 320 GB  ← 400 GB headroom vs 720 GB limit

During validation:
  /dev/shm L12:          145 GB
  4 × rank heaps:        159 GB
  Val loader IPC:          4 GB  (2w × pf2 × bs128 × ~2 MB × 4r)
  ─────────────────────────────
  Total:                 308 GB  ← very safe

Val→Train boundary (worst case, both IPCs resident):
  /dev/shm L12:          145 GB
  4 × rank heaps:        159 GB
  Train IPC (filling):    16 GB
  Val IPC (draining):      4 GB
  ─────────────────────────────
  Total:                 324 GB  ← 396 GB headroom vs 720 GB limit
```

**Pre-fix history:** pf=2 with 30 MB/sample IPC gave 607 GB boundary (113 GB headroom — tight).
Pyramid pooling fix cut IPC from ~242 GB to ~16 GB; headroom increased from 113 GB to 396 GB.
Can now safely increase pf or batch size if GPU utilisation demands it.

**Smoke test /dev/shm fix (2026-06-15, commit cdb3ef9):** `_preload_l12_to_shm` now accepts
`max_stations` and stops after N stations are written. With `--max-stations 20` the preload
takes ~30s and ~5 GB instead of 17 min and 145 GB. Preload and dataset always use the same
filtered, ordered station list so they stay aligned.

**NLL removal (2026-06-15, Session 9):** Gaussian NLL / aleatoric uncertainty support fully
removed. `masked_nll_loss()` deleted from `model.py`; `predict_uncertainty` param gone from
`UNetDecoder` and `SoilMoistureModel`; decoder head now always outputs `n_depths` channels
(not `2×n_depths`). `--loss-fn` CLI arg removed — Huber is the only loss. `_compute_loss`
simplified; TV term now only computed when `lambda_tv > 0` (was unconditionally run on every
eval batch and discarded). `eval_stations.py` fixed: stale `dropout`/`loss_fn`/`window_size`
kwargs removed from model init; `era5_stats_path` config key corrected to `era5_stats`.
Val loss intentionally stays as pure Huber (no TV/boundary) — correct signal for scheduler
and early-stopping. Smoke test job 23888331 (20 stations, 3 epochs).

**Dataset init cache audit (2026-06-15):** Agent-reviewed all zarr reads in `__init__` loop.
Two fixes applied (dataset.py):
1. `_era5_cache[sat_dir]` → `_era5_cache.get(sat_dir)` — defensive; KeyError impossible with
   current data (audit: 0 missing zarr, 0 no ERA5) but fragile if data changes.
2. S2 year range now read from `_zarr_date_cache` (already in RAM) instead of re-reading
   `s2/dates` from GPFS zarr — eliminates one zarr read per station at init (~600 reads saved).
`__getitem__` was already clean — all heavy data from caches, only anchor L3/L6/L9 from zarr.
Zarr audit: 890/993 fully OK, 103 missing SM labels (all ICOS flux stations, filtered by
category_filter=["sm_only"] so never loaded).

---

## 3. OOM Post-Mortem (Job 23809811)

**What happened:**  
Job 23809811 completed epoch 3 successfully (`val_loss=0.0035`), then was killed by the Linux cgroup OOM killer at the epoch-3→4 boundary. This was NOT a CUDA OOM — it was a host RAM OOM.

**Root cause:**  
SLURM `--mem=720G` + `persistent_workers=False`:
- Epoch 3 val completed: 32 val workers dying (still holding ~262 GB IPC temp files)
- Epoch 4 train start: 32 new train workers spawning (adding ~262 GB IPC)
- Baseline RAM at that point: ~249 GB (L12 + rank processes)
- Total at boundary: 249 + 262 + 262 = **~773 GB > 720 GB cgroup limit → SIGKILL**

**Fix applied:**
1. `persistent_workers=True` — workers persist across epochs, no respawn race
2. Val DataLoader: 8 workers → 2 workers (val IPC 262 GB → 62 GB)
3. `prefetch_factor` 2 → 3 (train IPC 293 GB → 440 GB, but offset by fix #1+2)
4. `--mem=720G` → `--mem=790G` (node has 792.3 GB; leaves 2.3 GB for OS)

---

## 3b. OOM Post-Mortem #2 (Job 23864068) — CoW refcount blowup

**What happened:** First full sm_only run (8 workers/rank × 4 ranks, bs=128, `--mem=720G`).
Preload + dataset build fine; trained ~28 min into epoch 1, then cgroup OOM. `State=OUT_OF_MEMORY`,
3 oom_kill events, all ranks SIGKILL (-9). `MaxRSS=573 GB` (`mem=600756021K`). RSS climbed
*gradually* over the epoch (not a startup spike), crossing 720G when combined with the 145 GB
`/dev/shm` tmpfs (~573 + 145 ≈ 718 GB).

**Root cause — Python copy-on-write breakage across 32 worker processes:**
- Dataset caches (`_era5_cache`, `_sif_cache`, `_twsa_cache`, `_label_cache`, date lists) are
  eager Python dicts/lists built in `__init__` (dataset.py:773–851), *before* the DataLoader
  forks workers. Initially CoW-shared.
- Every time a worker merely *reads* a Python object, CPython mutates that object's refcount in
  its header → writes the page → kernel gives the worker a private copy. Over an epoch, workers
  touch more cached objects → more pages privately duplicated × 32 processes → RSS climbs from
  ~205 GB toward 573 GB and beyond.
- NOT lazy-loading (caches are eager) and NOT classic IPC (IPC reaches steady-state fast). The
  gradual-over-epoch slope scaling with worker count is the CoW signature.
- The runbook's old `--mem=790G` "fix" never applies: 790G will not schedule on a shared node
  (hard cap 720G). Must solve via memory reduction, not a bigger `--mem`.

**Immediate mitigation (job 23866742):** `--num-workers 4 --prefetch-factor 2`. Fewer worker
processes ≈ halves the duplication slope. Worst-case peak ≈ 205 + 4×60 + 145 ≈ 590 GB, fits.
Stopgap only — costs data-loading throughput.

**Proper fixes (apply after baseline converges, ranked):**
1. **Caches → `/dev/shm` raw buffers / `np.memmap`** (like L12 already is). Raw bytes have no
   Python refcount, so they stay CoW-shared regardless of worker count. The scalable fix.
2. **`gc.freeze()`** right after dataset build, before fork — moves existing objects to a
   permanent GC generation so the collector won't scan/dirty their pages. One line, low risk;
   likely lets us return to 8 workers. Also convert date string-lists → int64 numpy arrays
   (numpy data buffers have no per-element refcount).
3. `num_workers=0` — zero duplication but serial loading (bottleneck). Not viable here.
4. Shrink footprint — downcast ERA5 to fp16, preload fewer modalities. Symptom only.

**Recommended order:** (2) first (cheap, lets us go back to 8 workers), then (1) if still tight.
Do NOT do (1) mid-baseline — it changes the frozen reference config.

---

## 3c. OOM Fix #2 — Architectural: keep L12 tokens out of DataLoader IPC

### The intuition (why this was wasteful)

The L12 tokens already live **once** in `/dev/shm` — that 145 GB shared cache is a single
physical copy that every rank and every DataLoader worker can read directly. That is the
entire point of the preload: store the tokens once, share them everywhere.

But `__getitem__` was then taking a **fresh copy** of each sample's token window *out* of
`/dev/shm` and putting it into the returned batch — which got shipped through the DataLoader's
worker→main IPC buffers and held there `num_workers × prefetch_factor × ranks` = 96 times over.
So the very data that was sitting as one tidy shared copy got **re-duplicated ~96× into IPC
(~393 GB) for no reason.** We were copying shared memory into private buffers needlessly, and
that redundant duplication is exactly what pushed RAM past the 720 GB limit.

The fix removes the redundancy: workers pass only the tiny **row indices** ("use rows 5, 12,
30 … of this station's array"); the actual tokens are read straight from the single `/dev/shm`
copy in the main process, just before they go to the GPU. The tokens never enter IPC at all.

### The details

**CORRECTION to 3b's root-cause hypothesis.** After reading the code, the OOM in
job 23864068 is **IPC-payload-dominated**, not primarily CoW-refcount. Each sample
returned by `__getitem__` carried the gathered L12 token windows:
`s2_l12 (60,196,768) fp16 = 17.6 MB` + `s1_l12 (40,196,768) = 11.8 MB` ≈ **29 MB/sample**
(92% of the ~32 MB payload). Even though the L12 source already lives in `/dev/shm`,
`load_s2_rolling_zarr` **sliced a fresh copy** of the window into the returned tensor, which
then flowed through DataLoader worker→main shared-memory IPC and was buffered
`num_workers × prefetch_factor × ranks` deep:

```
8 workers × 3 prefetch × 4 ranks = 96 batches buffered
96 × (128 samples × 32 MB) ≈ 393 GB IPC  +  145 GB /dev/shm  +  ~60 GB caches  →  > 720 GB
```

This is why reducing workers (3b's stopgap) helped, and why "move caches to /dev/shm" (the
originally-proposed fix #1) was **moot** — the caches were already in /dev/shm; the leak was
the per-sample *window copy* shipped through IPC.

**The fix (implemented 2026-06-15):** workers return only lightweight **row indices**; the
heavy L12 gather moves to the **main process**, reading directly from the `/dev/shm` memmaps
so tokens never enter IPC.

Changes:
- `dataset.py load_s2_rolling_zarr` → returns `src_idx (MAX_S2,)` (−1 = pad) instead of the
  `(MAX_S2,196,768)` tensor. NaN-skip preserved (reads one row from /dev/shm for the check,
  does not retain it).
- `dataset.py load_s1_rolling_zarr` → returns `src_idx` + `src_orbit` (0=asc,1=desc,−1=pad).
- `__getitem__` → emits `s2_src_idx`, `s1_src_idx`, `s1_src_orbit` (+ existing `station_key`)
  instead of `s2_l12`/`s1_l12`.
- `dataset._l12_by_key`: `station_key → {s2,s1_asc,s1_desc}` /dev/shm memmaps, for the gather.
- `dataset.gather_l12_from_shm(batch, l12_by_key)`: materialises `s2_l12`/`s1_l12` in the main
  process; padding stays zero; orbit routing per slot.
- `train.py CudaPrefetcher(loader, device, l12_by_key=...)`: calls the gather in `_preload`
  (CPU, overlaps prev batch's GPU compute) before the H2D copy. Wired for train + val loaders.

**Effect:** per-sample IPC 32 MB → ~3 MB (anchors + masks + ERA5; anchor L3/L6/L9 still
shipped). IPC ≈ 393 GB → **~37 GB**. New budget ≈ 145 (shm) + 37 (IPC) + 60 (caches) ≈
**~242 GB**, leaving ~478 GB headroom under 720G — and 8 workers/pf3 are safe again.

**Verification:** `test_gather_equiv.py` builds a real 3-station dataset and asserts the
gathered tokens equal the source memmap rows exactly, padding is zero, S1 orbit routing is
correct, and no NaN leaks. PASS (186 S2 + 280 S1 tokens over 8 samples). GPU end-to-end
validated by smoke job `smoke_gather` (23868316).

**Cost / follow-ups:** the gather is a single-threaded main-process memcpy (~3.8 GB/batch from
/dev/shm, ~0.4–0.8 s) overlapped with GPU compute — watch it doesn't bottleneck. Direct
dataset consumers that bypass the prefetcher (e.g. `demo_plot.py`) now receive `s2_src_idx`
not `s2_l12`; they must call `gather_l12_from_shm` themselves. Optional later: precompute a
per-acquisition NaN mask at preload to drop the worker's NaN read entirely.

---

## 3d. OOM Post-Mortem #3 (Job 23888806, Epoch 4) — Persistent Worker Page Accumulation + IPC Double-Copy

**What happened:**  
Job 23888806 ran 3 clean epochs (val_loss: 0.0035→0.0033→0.0022), then OOM killed at epoch 4
batch ~595. `Detected 2 oom_kill events`. SIGKILL — SIGTERM handler did not fire.

**Memory profile (psutil, first time available):**

| Snapshot | RAM used |
|---|---|
| job_start (after preemption restart) | 222 GB |
| epoch_003_start | 222 GB |
| epoch_003_post_train | 659 GB |
| epoch_003_post_val | 688 GB |
| epoch_004_start | **688 GB** ← only 32 GB headroom vs 720G cgroup limit |
| epoch_004_batch_595 | > 720 GB → **OOM KILL** |

**Why 688 GB base at epoch 4 start — not 222 GB?**  
With `persistent_workers=True`, the 32 worker processes (8 per rank × 4 ranks) stay alive
across epoch boundaries. Over the epoch they each accumulate private memory pages: Python
CoW pages dirtied by refcount mutations, zarr read buffers, numpy temporaries. These pages
are never released because workers never die. After epoch 3: 222 GB base + ~466 GB worker
private accumulation = 688 GB. This is the compounding effect of persistence across epochs.

**Why epoch 4 OOMed when epochs 1–3 did not:**  
Each of epochs 1–3 started after a SLURM preemption (SIGKILL). Preemption kills all
processes → OS reclaims all worker private pages → job restarts clean at 222 GB base. Epoch 3
was the first to complete without a mid-epoch preemption, so the epoch-3→4 transition was
the first time the full accumulated 688 GB base met a fresh DataLoader IPC fill (~437 GB).

**The double-copy problem (root cause of excessive IPC):**  
L12 tokens already live in `/dev/shm` as memmaps — one 145 GB physical copy shared by all
processes. But `__getitem__` slices a fresh window from these memmaps into a new tensor
`(60, 196, 768)` fp16 per S2 and `(40, 196, 768)` per S1, which then crosses the DataLoader
IPC barrier (PyTorch `file_system` shared-memory strategy) into the prefetch queue. So L12
data is copied twice:

```
/dev/shm (145 GB, one copy)
    → worker private tensor (~30 MB/sample)
    → IPC prefetch queue (8w × pf2 × 128 × 30MB × 4r = 437 GB)
    → main process → GPU
```

The 437 GB IPC queue is entirely redundant — all processes could read `/dev/shm` directly.
At epoch 4 start: 688 GB base + 437 GB IPC fill = **>1,100 GB needed** vs 720 GB cgroup.

**Why is so much going through IPC?**  
Inside `model.forward()`, `_pyramid_from_l12()` immediately compresses the `(M, 196, 768)`
L12 tensor down to `(M, 4, 768)` pyramid tokens via `spatial_pyramid_pool()`. So 99% of the
30 MB IPC payload (the full 196-token spatial grid for ~100 historical acquisitions) is
shipped across IPC only to be thrown away after 4 pooling operations on the GPU. Only the
single anchor acquisition truly needs its full 196×768 grid (for U-Net decoder skip
connections). All other acquisitions need only 4×768 summary tokens.

---

## 3e. IMPLEMENTED Fix — Move Pyramid Pooling into `__getitem__` (Session 11, 2026-06-17)

**Why this is the right fix:**  
The pyramid pooling for historical acquisitions does not require model weights to be
on-device — it is a spatial average/max pool over fixed nested windows. Moving it into the
dataset worker (CPU, before IPC) compresses each acquisition from `(196, 768)` → `(4, 768)`
before the data crosses IPC. Per-sample payload drops from 30 MB to ~2 MB. IPC drops from
437 GB to ~13 GB — completely irrelevant.

**Will this slow training?**  
No. Workers run in parallel with GPU compute — while the GPU processes batch N, workers
prepare batch N+1 in the background. Pyramid pooling for 128 samples × 100 acquisitions on
CPU takes ~50 ms per batch; GPU step time is ~2800 ms. Workers have 2800 ms to prepare the
next batch — 50 ms extra is invisible. There may even be a small speedup: the GPU no longer
runs `_pyramid_from_l12()` on large `(B, 100, 196, 768)` tensors, freeing GPU cycles for
the transformer and decoder.

**What about `s2_pyramid_attn` / `s1_pyramid_attn` (trainable weights)?**  
These are `nn.Linear(768, 1)` layers that do learned attention pooling over the 196 spatial
positions. Moving pooling to the CPU worker means we cannot use these layers (they live on
GPU inside the model). We replace them with **static average pooling** (the `attn=None` path
already in `spatial_pyramid_pool()`). Performance cost is expected to be small — the 4-scale
pyramid structure provides the key spatial summarization; the learned attention only
fine-tunes within each scale.

**Implementation plan:**

1. Add `_cpu_pyramid_pool(l12, token_mask)` in `dataset.py`:
   - Input: `(M, 196, 768)` fp16 + `(M, 14, 14)` bool mask
   - Reshape to `(M, 14, 14, 768)`, apply masked mean over 4 nested spatial windows
   - Output: `(M, 4, 768)` fp32
   - Pure numpy/torch, no model dependency

2. Modify `load_s2_rolling_zarr()` and `load_s1_rolling_zarr()` in `dataset.py`:
   - Apply `_cpu_pyramid_pool` to L12 window before returning
   - Change return key from `s2_l12 (M, 196, 768)` → `s2_pyr (M, 4, 768)`
   - Anchor L12 `(196, 768)` stays unchanged (still needed for decoder skip)

3. Modify `SoilMoistureModel.forward()` in `model.py`:
   - Replace `self._pyramid_from_l12(batch["s2_l12"], ...)` with `batch["s2_pyr"].to(device)`
   - Remove `s2_pyramid_attn` / `s1_pyramid_attn` usage (or leave as unused params)
   - `_get_target_spatial_tokens()` unchanged (still uses anchor `(196, 768)`)

**Memory impact:**

| | Before | After |
|---|---|---|
| Per-sample IPC | 30 MB | ~2 MB |
| Train IPC (8w×pf2×128×4r) | 437 GB | ~13 GB |
| Total training RAM | ~660 GB (OOM) | ~240 GB |
| /dev/shm preload | 145 GB unchanged | 145 GB unchanged |

**Implementation — commits 8947f9a, 2f6581b, ec74775 (2026-06-17):**

- `_cpu_pyramid_pool(l12, token_mask)` added to `dataset.py`: compresses `(M,196,768) fp16` →
  `(M,4,768) fp32` using static masked mean over 4 nested square windows (widths `[1,3,5,7]`,
  same formula as old `spatial_pyramid_pool` in model.py)
- `load_s2_rolling_zarr` + `load_s1_rolling_zarr`: return 4-tuple `(pyr, doys, valid, rel_pos)`
  instead of old 5-tuple. Added `training: bool = False` param — when True, applies 50%
  random spatial token dropout to `token_mask` BEFORE pooling (restores ContextFormer
  augmentation that was previously in `model.forward()`)
- `__getitem__`: DEM and LULC also pooled via `_cpu_pyramid_pool` (no random dropout —
  DEM/LULC are always available at inference, masking them trains on unrealistic scenarios)
- Batch keys changed: `s2_l12/s2_token_mask → s2_pyr`, `s1_l12/s1_token_mask → s1_pyr`,
  `dem_l12/dem_token_mask/lulc_l12/lulc_token_mask → dem_pyr/lulc_pyr`
- `model.py`: removed `s2_pyramid_attn`, `s1_pyramid_attn`, `dem_pyramid_attn`,
  `lulc_pyramid_attn` from `__init__`. `forward()` reads pyramid tokens directly from batch.
  `spatial_pyramid_pool`, `_pyramid_from_l12`, `_static_pyramid` kept but marked NOT USED.
- Random token masking in `forward()` removed entirely (moved to workers for S2/S1; DEM/LULC
  not masked by design)

**Consistency:** all four modalities now use identical static masked-mean pyramid pooling.
Previously S2/S1 used learned attention (GPU) while DEM/LULC also used learned attention.
Now all four use static mean (CPU). Learned attention (`nn.Linear(768,1)` per modality) had
minimal expressiveness over 196 patches; the temporal transformer does real spatial reasoning.

**Verification:** `test_pyramid_equiv.py` — 5 checks all pass (shape, zero-slots, determinism,
finite values, center-patch scale consistency). Agent code review: no critical issues.

**New memory budget (pyramid pooling fix):**

| | Before fix | After fix |
|---|---|---|
| Per-sample IPC | ~30 MB | ~2 MB |
| Train IPC (8w×pf2×bs128×4r) | 242 GB | ~16 GB |
| Val IPC (2w×pf2×bs128×4r) | 61 GB | ~4 GB |
| Val→Train boundary | 607 GB (113 GB headroom) | **~324 GB (396 GB headroom)** |
| `/dev/shm` L12 preload | 145 GB | **145 GB unchanged** (workers still read L12 to pool it) |

The `/dev/shm` preload is still needed — workers read L12 from there and pool it CPU-side
before IPC. Only the IPC payload shrinks.

**Phase 2 (after baseline converges): Pre-compute pyramid tokens in zarr**  
Run a one-time offline script: read each station's L12, apply `_cpu_pyramid_pool`, write
`(N_acq, 4, 768)` pyramid arrays back to zarr. Then `__getitem__` loads `(N_acq, 4, 768)`
directly. `/dev/shm` preload drops from 145 GB to ~4 GB. Total training RAM < 100 GB.

---

## 4. Tuning Guide

### Adjusting worker count and prefetch_factor safely

The memory equation for train DataLoader IPC:
```
IPC_GB = num_workers × prefetch_factor × batch_size × per_sample_MB × num_ranks / 1024
per_sample_MB ≈ 30.3 MB  (S2+S1+static tokens + ERA5 + SIF/TWSA)
```

Baseline = 145 GB (/dev/shm measured) + 159 GB (rank heaps) = **304 GB**

Baseline = 145 GB (/dev/shm measured) + 159 GB (rank heaps) = **304 GB**

| num_workers | pf | Train IPC (4r) | Boundary (+61 GB val) | vs 720G (shared) | vs 790G (gcn) |
|---|---|---|---|---|---|
| 8 | 1 | 121 GB | 486 GB | +234 GB safe | +304 GB safe |
| 8 | 2 | 242 GB | 607 GB | +113 GB safe ← **current** | +183 GB safe |
| 8 | 3 | 363 GB | 728 GB | **+8 GB — too tight** | +62 GB safe |
| 8 | 4 | 484 GB | 849 GB | **UNSAFE** | **UNSAFE** |
| 4 | 4 | 242 GB | 607 GB | +113 GB safe | +183 GB safe |

**Rule**: `num_workers × prefetch_factor ≤ 24` on shared nodes (720G); `≤ 28` on gcn nodes (790G).
(Limit = (mem − 304 − 61) / (30.3MB × 128 × 4 / 1024))

### GPU utilisation target

After fixes: `gpu_util ≈ 90–95%` (data_time << compute_time)  
If `gpu_util < 80%`: increase `prefetch_factor` by 1 or `num_workers` by 2  
If OOM: decrease `prefetch_factor` by 1 or switch to `num_workers=4`

---

## 5. Runbook for Common Failures

### 5.1 Linux cgroup OOM (SIGKILL)
**Symptom:** Log ends with `Detected N oom_kill events`, no Python traceback  
**Diagnosis:** Count GPU workers × pf × batch × 30 MB × 4 ranks vs SLURM --mem  
**Fix:** Reduce `prefetch_factor` by 1, or increase `--mem` (max 792G on gcn nodes)

### 5.2 CUDA OOM (torch.cuda.OutOfMemoryError)
**Symptom:** Python traceback with `CUDA out of memory`  
**Diagnosis:** `peak_vram` in epoch summary; check batch size and model config  
**Fix:** Reduce `batch_size`, add gradient checkpointing, or reduce `d_model`

### 5.3 GPFS batch stalls (25+ second steps)
**Symptom:** Batch step times spike to 10–25 seconds intermittently  
**Diagnosis:** GPFS scratch load from cloud mask or token mask zarr reads  
**Fix:** Verify `_cm_token_mask_cache` and `_zarr_date_cache` are populated (check init log)  
Emergency: set `DISABLE_L12_CACHE=0` and ensure `/dev/shm` preload completes

### 5.4 NCCL timeout
**Symptom:** `NCCL communicator watchdog`, all 4 ranks killed  
**Diagnosis:** One rank stalled during `evaluate()` (val too slow or GPFS read)  
**Fix:** `NCCL_TIMEOUT=7200` already set. Reduce val dataset size or add more val workers.  
Long-term: distributed validation (all ranks validate in parallel — see project notes)

### 5.5 Job preemption / requeue
**Symptom:** `SIGTERM` → job requeued  
**Fix:** Automatic — `--requeue` is set. Resume is automatic via `last.pt` checkpoint.  
If `val_pending=True` in checkpoint: epoch completes validation on resume without retraining.

### 5.6 W&B connection failure
**Symptom:** `W&B disabled: ...` in log  
**Impact:** Metrics still logged to stdout and `val_station_metrics.csv`. Training continues.  
**Fix:** Check internet access on compute node; W&B is optional.

---

## 6. Key Commands

```bash
# Submit new training run (Huber is now the only loss — no --loss-fn flag)
sbatch slurm/train.sh --run-name baseline_huber

# Smoke test (20 stations, 3 epochs; preload takes ~30s vs 17 min full)
sbatch slurm/train.sh --run-name smoke_v2 --max-stations 20 --max-epochs 3

# Mmap smoke test — validate flat data_time across epochs before full run
sbatch slurm/train.sh --run-name smoke_anchor_mmap --max-stations 5 --max-epochs 3 --use-memmap
# Pass: epoch 2 data_time ≤ epoch 1; losses match zarr smoke run

# Full run with mmap (after smoke test confirms fix)
sbatch slurm/train.sh --run-name baseline_mmap --use-memmap

# Resume is automatic — just resubmit the same run-name
sbatch slurm/train.sh --run-name baseline_huber

# Start fresh (delete checkpoint first)
rm /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only/baseline_huber/last.pt
sbatch slurm/train.sh --run-name baseline_huber

# Per-station eval from a saved checkpoint
python eval_stations.py --run-name baseline_huber

# Check per-station metrics after training
cat /gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only/baseline_huber/val_station_metrics.csv | head -20
```

---

## 7. Epoch Log Format (post-fix)

```
=== Memory snapshot: epoch_004_start ===
  RAM  used : 690.1 GB / 792.3 GB  (87%)
  CPU  util : 12%
  GPU 0 VRAM: 1.8 alloc / 2.1 rsv / 2.1 peak / 100 GB total
  ...

  batch 0001  loss=0.0004  tv=0.00042  step=620ms
  batch 0002  loss=0.0003  tv=0.00038  step=610ms
  ...

=== Memory snapshot: epoch_004_post_train ===
  RAM  used : 695.3 GB / 792.3 GB  (88%)
  ...

Epoch 004  |  train_loss=0.0003  val_loss=0.0029  data=48s  compute=625s  gpu_util=93%  peak_vram=48.0GB
  0-10      MSE=0.0031  MAE=0.0412  ubRMSE=0.0431  bias=0.0012
  10-30     MSE=0.0022  MAE=0.0310  ubRMSE=0.0298  bias=0.0008
  30-100    MSE=0.0019  MAE=0.0280  ubRMSE=0.0271  bias=0.0003
```

**GPU utilisation target: ≥ 90%**  
`gpu_util = compute_time / (data_time + compute_time)`  
If gpu_util < 80% after the GPFS fix, check anchor L3/L6/L9 read latency.

---

## 8. Deferred Performance Optimizations

These are **speed-only** changes intentionally NOT applied to the first baseline run
(job 23864068, started 2026-06-15). The baseline is frozen on a known, reviewed config
so a future optimized run can be compared against a clean reference. Apply these *after*
the baseline converges, then measure speedup (and any quality delta) against this run.

### 8.1 `_pyramid_from_l12` `.nonzero()` GPU sync — RESOLVED (pyramid moved to CPU)

~~This was HIGH priority~~ — now moot. `_pyramid_from_l12` is dead code since pyramid pooling
moved to `_cpu_pyramid_pool()` in `dataset.py`. The `.nonzero()` GPU sync no longer fires.
`_pyramid_from_l12` and `_static_pyramid` are kept in `model.py` marked NOT USED.

### 8.2 SyncBatchNorm → GroupNorm  (MEDIUM priority)

**Where:** `train.py:674` (`SyncBatchNorm.convert_sync_batchnorm`) + `_ConvBlock` BatchNorm2d
layers (`model.py:148,151` and other BatchNorm2d at `model.py:268,274`).

**Why it's slow:** BatchNorm normalizes using the current batch's mean/variance. With 4 DDP
ranks each holding a different batch slice, `SyncBatchNorm` makes all 4 GPUs exchange
statistics over NCCL (`all_reduce`) for *every* norm layer on *every* forward and backward —
~8 rounds of inter-GPU chatter per forward. It is also the source of the
`Grad strides do not match bucket view strides` warning seen in the logs.

**Fix:** switch `_ConvBlock` (and the other BatchNorm2d sites) to `GroupNorm`, which computes
statistics *within each sample on each GPU independently* — no cross-GPU communication, no
batch-size dependence, and it kills the DDP stride warning. NOTE: this slightly *changes the
normalization math*, so a GroupNorm run is not numerically comparable to this BatchNorm
baseline — that is precisely why it is deferred rather than applied mid-baseline.

### 8.2b DDP load imbalance — length-bucketed batching  (HIGH priority — needs confirmation)

**Symptom observed (job 23864068, epoch 1, 2026-06-15):** high-frequency `nvidia-smi`
sampling (30 samples @ 0.25 s) showed uneven GPU duty cycles across ranks:
```
G0 = 100% busy   G1 = 60%   G2 = 87%   G3 = 30% busy
```
Average effective utilisation ≈ 70%, not the ~95% a sparse single-snapshot suggests.
GPU 0 was pinned while ranks 1 and 3 idled 40–70% of the time.

**Most likely cause — NOT classic data starvation:** in synchronous DDP every rank meets at
the gradient `all_reduce` barrier each step, so the *slowest rank sets the pace* and faster
ranks idle at the barrier. Two contributors:
1. **Variable-length token sequences** — each sample has a different number of satellite
   acquisitions (S2/S1 dates in the rolling window), so a rank drawing heavier batches does
   genuinely more transformer compute; lighter ranks finish early and wait. Inherent to
   ragged multimodal data under `DistributedSampler`.
2. **Per-rank GPFS jitter** — occasional slow anchor L3/L6/L9 reads (~882 KB/sample) stall
   whichever rank hits them.

**CRITICAL measurement caveat:** the code's own `gpu_util = compute/(data+compute)` (epoch
summary) measures only *per-rank data starvation* — it does NOT see cross-rank barrier idle.
So the epoch summary can report a healthy gpu_util while `nvidia-smi` shows real idle. Trust
the `nvidia-smi` duty cycle for the cross-rank picture; trust `data=/compute=` for starvation.

**Confirmation step (do before fixing):** read the epoch-1 `data=Xs compute=Ys gpu_util=Z%`
line. If `data` is small but GPUs still idle per `nvidia-smi` → confirmed imbalance (not
starvation), so prefetch/worker tuning will NOT help.

**Fix (if confirmed):** length-bucketed batching — group samples with similar acquisition
counts into the same batch so per-step compute is balanced across ranks. Bigger lever for
this symptom than 8.1/8.2. Alternative: pad/truncate to a fixed acquisition count (simpler,
wastes some compute on padding).

### 8.3 Learned pyramid attention ablation (after baseline converges)

Current state: all four modalities use static masked-mean pyramid pooling (CPU-side). The
original design used `nn.Linear(768,1)` learned attention for pooling. After baseline
converges, run an ablation: restore learned spatial attention for S2/S1 by pre-computing
pyramid tokens in zarr (Phase 2), then the pooling runs offline and the learned attention
can be applied in the model without any IPC cost. If learned attention improves val metrics
meaningfully (expected small: the temporal transformer does the real spatial reasoning),
adopt it in Phase 2. If not, static mean is simpler and confirmed adequate.

### 8.4 `dist.all_gather_object` → tensor `dist.all_gather` (MEDIUM priority)

**Where:** `evaluate()` in `train.py` — collects variable-length numpy predictions to rank 0
using `dist.all_gather_object` (Python pickle under the hood).

**Why it matters:** Pickle is slow and fragile; NCCL watchdog timeouts can occur if collection
takes longer than `NCCL_TIMEOUT`. With 4 GPUs this is fine, but would break on multi-node.

**Fix:** pad local prediction tensors to a fixed maximum size, use `dist.all_gather` (tensor,
no pickle), slice off padding after. Already noted in deferred task memory.

### 8.5b Anchor L3/L6/L9 → `.npy` memmap (HIGH priority — gpu_util 22%→~65%)

**Root cause:** §3f. Zarr chunks=(32,196,768)+zstd force 9.6 MB GPFS read + decompression per
`zg[key][best_idx]` call. 3 reads per sample → primary DataLoader bottleneck.

**Fix:** convert L3/L6/L9 arrays to uncompressed `.npy` memmap files alongside zarr:
```
{station_dir}/s2_l3.npy      (N, 196, 768) fp16, flat binary, no compression
{station_dir}/s2_l6.npy
{station_dir}/s2_l9.npy
{station_dir}/s1_asc_l3.npy  ...and so on for s1_asc, s1_desc
{station_dir}/{orbit}_{layer}.json   shape metadata
```

Worker access: `np.memmap(path, dtype="float16", mode="r", shape=shape)[best_idx]`
- Reads exactly 0.3 MB (one row) vs 9.6 MB zarr chunk — **32× less GPFS I/O**
- Zero decompression (raw binary)
- OS page cache: after first access, subsequent reads hit RAM (zero I/O from epoch 2+)
- Inode count: 9 files/station × 661 stations = ~5,949 files (vs ~41k zarr chunks) — **fewer files**

**Storage:** ~524 GB uncompressed (S2+S1_asc+S1_desc L3+L6+L9). Verified scratch has space.  
**File count:** ~5,949 .npy + ~5,949 .json = ~12k new files — no inode concern.  
**Memory:** less than zarr (no per-worker zarr chunk LRU cache; shared OS page cache instead).

**Conversion script:** `convert_l369_to_npy.py` (64 workers, resume-safe, dry-run by default)
```bash
python convert_l369_to_npy.py                # dry run — prints what would be written
python convert_l369_to_npy.py --execute      # actually write files (~2–4 hrs on thin node)
```

**Training integration:** `--use-memmap` flag in `train.py` → `use_mmap=True` in dataset.
`_load_layer` checks memmap cache first, falls back to zarr if .npy absent. Both paths in
one script — comparison smoke test: run with/without `--use-memmap`, compare `data_time`.

**Expected improvement:** gpu_util 22% → ~50–65% (anchor reads eliminated; CPU pyramid
pooling becomes new minor bottleneck; DDP barrier imbalance remains).

**Status (2026-06-16): CONVERSION COMPLETE + IMPLEMENTED.**
Job 23927829: 890 stations, 7818 files, ~525 GB, 0 errors.
dataset.py: `_l369_cache` + `_load_layer` cache-first logic. train.py: `--use-memmap` flag.
Smoke test before full run — see §3g for comparison strategy and pass criteria.

### 8.5 Lower-priority items (apply opportunistically)
- Cache `torch.arange(365)` as a `register_buffer` for ERA5 rel_pos (avoids per-call alloc).
- Cache `k` arange in `circular_doy_pe` as a module buffer.
- S1 load: `np.asarray()` → direct `tokens_z[src_i]` indexing.
- GPU timing: add `torch.cuda.synchronize()` before perf_counter for accurate step benchmarks.

### 8.4 Expected payoff
Removing the `.nonzero()` sync (8.1) + SyncBN all_reduce chatter (8.2) is estimated to
reclaim ~10–20% of the ~2.9 s/batch step time. Both are GPU-pipeline overhead, not genuine
model FLOPs — the model itself is compute-heavy at bs=128, but these stalls are avoidable.

---

---

## 3f. GPFS Anchor Bottleneck — Root Cause Analysis (Session 12, 2026-06-16)

**Observed:** Epoch 1 (job 23921632): `data=1429s compute=405s gpu_util=22%`  
DataLoader workers are 3.5× slower than GPU compute. Workers can't keep up.

**What IS in RAM (no GPFS reads in `__getitem__`):**
- `_era5_cache`, `_sif_cache`, `_twsa_cache`, `_label_cache` → fully in Python dicts at init
- `_cm_token_mask_cache`, `_s1_token_mask_cache`, `_zarr_date_cache` → loaded at init
- `_static_cache` (DEM/LULC/soil) → loaded at init
- L12 tokens → `/dev/shm` memmaps (preloaded at job start)

**The actual GPFS read path per `__getitem__`:**  
`select_anchor_zarr._load_layer()` does `zg[key][best_idx]` for L3, L6, L9 — 3 reads per sample.

**Smoking gun — zarr chunk layout:**
```
s2/l3: chunks=(32, 196, 768), dtype=float16, compressor=Blosc(zstd, clevel=3)
s2/l6: chunks=(32, 196, 768)
s2/l9: chunks=(32, 196, 768)
```
A single-index access `zg[key][best_idx]` must fetch the **entire 32-row chunk** (9.6 MB raw)
from GPFS, then zstd-decompress it, to extract one row (0.3 MB). This happens 3× per sample.
With 32 workers hammering GPFS simultaneously: **~28.9 MB GPFS read + 3 decompressions per sample**.

**Time breakdown per `__getitem__` (estimated):**
| Operation | Cost | Bottleneck? |
|---|---|---|
| ERA5/SIF/TWSA/label slicing (RAM) | ~0.1 ms | No |
| L12 mmap reads (/dev/shm) | ~1–2 ms | Mild |
| `_cpu_pyramid_pool` S2+S1 | ~5–10 ms | Moderate |
| Anchor L3/L6/L9 GPFS+zstd | **~50–200 ms** | **PRIMARY BOTTLENECK** |

At 50 ms/sample avg, 32,000 samples/worker/epoch → ~1,600 s data time. Matches observed 1429s.

**Why workers/prefetch don't help:** more workers = more concurrent GPFS requests = more
filesystem contention. 32 workers already saturate GPFS scratch for this access pattern.
Prefetch_factor increases queue depth but not GPFS throughput.

**Workers/prefetch reverted to 8/2** (previous values) after this analysis. The proposed
12-worker / pf=4 change was based on wrong diagnosis and was reverted.

**Minor fix applied (2026-06-16):** DEM/LULC pyramid (`_cpu_pyramid_pool`) moved from
`__getitem__` to `__init__` — precomputed once per station, stored as `dem_pyr`/`lulc_pyr`
in `_static_cache`. Zero RAM cost (~16 MB total). Saves ~10 ms per sample (small vs 50–200 ms
anchor bottleneck, but correct). Fallback to on-the-fly computation if cache missing.

**Options investigated:**
1. Rechunk L3/L6/L9 to `(1,196,768)` → 32× fewer GPFS bytes but ~1.2M chunk files (inode explosion) ✗
2. Preload S2 L3/L6/L9 to /dev/shm → 139 GB extra, total 520 GB (safe), but doesn't cover S1 anchors
3. Preload ALL L3/L6/L9 → 524 GB extra, total 905 GB → OOM ✗
4. **Convert to `.npy` memmap** → single file per array, no compression, OS page cache, 0.3 MB/read ✓

**Chosen fix: `.npy` memmap conversion (see §8.5b). Conversion complete (job 23927829). Implementation wired via `--use-memmap` flag (dataset.py + train.py, 2026-06-16).**

---

## 3g. Cross-Epoch Data-Load Degradation — Mechanism & Fix (Session 12, 2026-06-16)

**Observed pattern:** job 23921632 epoch timing: data=1429s → 1758s → 5200s (monotonically worsening).

### Three compounding causes

**Cause 1 — Zarr decompresses on every access, even on OS page-cache hits**

`_load_layer("l3/l6/l9")` in `select_anchor_zarr` calls `zg[key][best_idx]` which must fetch
a 9.6 MB compressed chunk (chunks=(32,196,768), Blosc/zstd) from GPFS to extract 0.3 MB.
Even when the OS page cache holds the compressed bytes, zarr re-decompresses them in userspace
on every call. The OS cache removes the GPFS I/O latency but cannot remove the CPU
decompression step.

**Cause 2 — GPFS contention accumulates over job lifetime**

The longer the job runs, the more competing cluster jobs access shared GPFS scratch. Lock and
metadata contention grows. Epoch 1 sees a quieter filesystem; epoch 3 runs at peak activity.

**Cause 3 — Zarr's internal chunk cache cannot exploit the full free OS page cache**

Zarr has a small in-process LRU chunk cache. With 32,015 samples in random order
(DistributedSampler reshuffles each epoch), zarr's cache misses constantly. The full 540+ GB
of free OS page cache on the node is invisible to zarr's cache layer.

### How `.npy` memmap solves all three

`np.load(path, mmap_mode='r')` creates a virtual memory mapping — no process heap growth
(~200 bytes metadata per object). When `arr[best_idx]` is accessed:
- **Page NOT in OS page cache:** kernel reads from GPFS, stores raw float16 in free RAM (OS
  page cache, outside process RSS), returns data.
- **Page IN OS page cache:** kernel returns bytes directly from RAM. No decompression. Zero
  extra CPU work.

After epoch 1, every anchor row in the training set has been accessed. The ~525 GB of L3/L6/L9
data fits in the ~540 GB of free RAM (OS page cache). From epoch 2 onwards, all reads serve
from RAM — no GPFS calls, no lock contention, no decompression.

**Per-epoch timing after fix:**

| | zarr (current) | .npy mmap |
|---|---|---|
| Epoch 1 | GPFS + zstd (slow) | GPFS flat read, no decompress (faster) |
| Epoch 2 | GPFS + zstd + contention | OS page cache → pure RAM |
| Epoch 3 | Much worse | ≈ epoch 2 (stable) |

### RAM budget

`mmap_mode='r'` does NOT grow process RSS — OS page cache is kernel-managed free RAM.

| Component | RAM type | Approx |
|---|---|---|
| Process RSS (model + caches + IPC) | Process heap | ~80 GB |
| `/dev/shm` L12 | tmpfs | 145 GB |
| L3/L6/L9 .npy page cache (after epoch-1 warmup) | Kernel page cache (evictable) | up to 525 GB |
| **Process RSS + /dev/shm (cgroup limit applies here)** | | **~225 GB → safe** |

### Implementation

`dataset.py`: `_l369_cache` dict; station-init loop opens `.npy` memmaps via
`np.load(sat_dir / f"{orbit}_{layer}.npy", mmap_mode="r")`; `_load_layer` in
`select_anchor_zarr` checks cache before zarr for l3/l6/l9; fallback to zarr if file absent.

`train.py`: `--use-memmap` flag → `use_mmap=True` to dataset constructor.

### Conversion status

Job 23927829: 890 stations, 7818 files written, 0 errors, ~525 GB total at
`/gpfs/scratch1/shared/pkhanal/zarr/{category}/{station}/{orbit}_{layer}.npy`.

### Comparison strategy

We already have zarr timing from job 23921632 (1429s / 1758s / 5200s epochs 1–3). No need to
re-run zarr. Run mmap smoke test to validate:

```bash
sbatch slurm/train.sh --run-name smoke_anchor_mmap --max-stations 5 --max-epochs 3 --use-memmap
```

Pass criteria: epoch 2 data_time ≤ epoch 1; epoch 3 ≈ epoch 2; losses match zarr run.
If smoke test passes → submit full run: `sbatch slurm/train.sh --run-name baseline_mmap --use-memmap`

---

## 9. Baseline Run — Measured Numbers (job 23864068, 2026-06-15)

These supersede the earlier *estimates* in sections 2–4. The first full sm_only run was
launched fresh at `--mem=720G` (node max for a shared job; 790G never schedules).

| Quantity | Earlier estimate | **Measured (this run)** |
|---|---|---|
| Stations preloaded (train+val) | ~600 | **661** (577 train + 74 val after coverage filter) |
| `/dev/shm` L12 cache | ~91–215 GB | **145 GB** |
| SHM preload time | ~14 min | **17.4 min (1045 s)** — one-time per job start |
| Train samples | — | **1,049,925** |
| Batches/epoch (per rank) | — | **2,051** (1.05M ÷ 128 ÷ 4) |
| Train DataLoader IPC | ~440 GB | **~188 GB** (shared rose 145→333 GB once workers spawned) |
| Peak RAM available | — | **~216 GB free** of 755 GB — no OOM |
| GPU util — single snapshot | 90–95% | 96–100% (misleading; sparse sample) |
| GPU util — dense duty cycle | — | **~70% avg** (G0=100/G1=60/G2=87/G3=30) — see §8.2b imbalance |
| Peak VRAM / GPU | ~48 GB | **~56 GB / 95.8 GB** |
| Step time | 600 ms (post-fix dream) | **~2,880 ms/batch** (compute-bound at bs=128) |
| Epoch time | 35–45 min | **~98 min (~1h 38m) + ~5 min val** |
| 50-epoch wall-clock | — | **~85 h** (inside 120 h `--time`; early-stop likely shorter) |

**Why step time did not fall like the smoke test:** the bs=8 smoke test was IO-bound early
and recovered 6330→320 ms as `/dev/shm` warmed. At bs=128 the run is **compute-bound from
batch 1** (GPUs 96–100%), so there is no IO slack to recover — ~2,880 ms is the real
GPU-limited floor. Section 8 fixes are the path to lowering it.

---

## 10. Current Run — job 23921632 (baseline_huber, started 2026-06-16)

Fresh start after OOM fix (CPU pyramid pooling, §3e). All old checkpoints cleared.

**Config:** 661 train / 74 val stations, batch_size=128, 8 workers/rank, pf=2, Huber loss,
lr=2e-4, lambda_tv=0.1

| Snapshot | RAM used |
|---|---|
| job_start | ~217 GB |
| epoch_001_start | ~217 GB |
| epoch_001_post_train | **358.4 GB** |
| epoch_002_start | **381.5 GB** (47% of 811 GB node RAM) |

Note: script reports node physical RAM (811 GB base-10) not cgroup limit.
Cgroup limit: 720 GiB = 773 GB base-10. Usage at epoch 2 start = 381.5 GB → 391 GB headroom.

**Epoch 1 results:**

| Metric | Value |
|---|---|
| train_loss | 0.0012 |
| val_loss | 0.0022 (new best, checkpoint saved) |
| 0-10cm ubRMSE | **0.0535** ✓ (target < 0.07) |
| 10-30cm ubRMSE | 0.0491 |
| 30-100cm ubRMSE | 0.0550 |
| data_time | **1429 s** |
| compute_time | 405 s |
| gpu_util | **22%** ← DataLoader bottleneck (§3f) |
| peak_vram | 38.6 GB / 96 GB |
| epoch duration | ~80 min train + ~60 min val (cold first epoch) |

**GPU util breakdown:** `data/compute = 3.5×` — DataLoader dominates. Root cause: zarr
chunk reads for anchor L3/L6/L9 (§3f). Not DDP imbalance (data_time >> compute_time).
Fix: `.npy` memmap wired via `--use-memmap` (§3g, §8.5b). Run smoke test before resubmitting.

**VRAM:** 38.6 GB peak (40% of 96 GB) — comfortable. Batch size could be increased to 256
to improve gpu_util (more compute per data fetch: 22%→~36%) while memmap fix is pending.

---

## 11. Current Run — job 23936932 (full run, started 2026-06-16, ongoing)

Fresh run using `.npy` memmap (§3g/§8.5b). `--use-memmap` confirmed active: log shows
`L369 memmap cache: 5142 arrays opened (OS page cache; zero process heap cost)`.

**Config:** 577 train / 74 val stations, 1,049,917 train samples, batch_size=128, 8 workers/rank,
pf=2, Huber loss, lr=2e-4, lambda_tv=0.1

### Epoch results (epochs 1–10, epoch 11 in progress as of 2026-06-17)

| Epoch | train_loss | val_loss | 0-10 ubRMSE | 10-30 ubRMSE | 30-100 ubRMSE | gpu_util | data_time | Best? |
|-------|-----------|---------|------------|-------------|--------------|----------|-----------|-------|
| 1  | 0.0012 | 0.0023 | 0.0525 | 0.0487 | 0.0554 | 25% | 1256s | ✓ |
| 2  | 0.0005 | 0.0021 | 0.0538 | 0.0505 | 0.0566 | 70% | 176s  | ✓ |
| 3  | 0.0003 | 0.0021 | 0.0520 | 0.0484 | 0.0561 | 91% | 41s   | ✓ |
| 4  | 0.0003 | 0.0021 | 0.0527 | 0.0477 | 0.0554 | 86% | 69s   | ✓ |
| 5  | 0.0002 | 0.0021 | 0.0526 | 0.0483 | 0.0566 | 27% | 1131s | — |
| 6  | 0.0002 | 0.0021 | 0.0532 | 0.0482 | 0.0554 | 51% | 396s  | — |
| 7  | 0.0002 | 0.0022 | 0.0530 | 0.0491 | 0.0579 | 35% | 777s  | — |
| 8  | 0.0001 | 0.0021 | 0.0539 | 0.0488 | 0.0565 | 47% | 470s  | — |
| 9  | 0.0001 | 0.0021 | 0.0529 | 0.0482 | 0.0571 | 39% | 629s  | ✓ |
| 10 | 0.0001 | 0.0021 | 0.0527 | 0.0486 | 0.0567 | 58% | 304s  | ✓ |

Peak VRAM: stable at **38.6 GB / 100 GB** every epoch. No OOM.  
Compute time: **constant ~410s/epoch** — GPU not the bottleneck.  
Best val_loss: **0.0021** (plateaued from epoch 2).  
Train loss still descending (0.0012 → 0.0001) while val is flat → early overfitting onset.

### RAM analysis — not a leak

| Snapshot | RAM used |
|---|---|
| epoch_001_start | 257.8 GB |
| epoch_001_post_train | 366.6 GB |
| epoch_001_post_val | **387.8 GB** ← +130 GB jump |
| epoch_002_start | 387.8 GB |
| epoch_002_post_train | 394.1 GB |
| epoch_003–010 | ~391–403 GB (essentially flat, ±5 GB noise) |
| epoch_011_post_train | 400.7 GB |

The +130 GB jump in epoch 1 is **OS page cache filling** as the 5142 memmap arrays are read
from GPFS scratch for the first time — expected, not a leak. From epoch 2 onward, RAM is
flat at ~400 GB. The cgroup limit (720 GB) has 320 GB headroom — safe for the full run.

### VRAM reserved creep — real risk

GPU 0 **reserved** (not allocated) memory grows exactly +1.1 GB/epoch:

| Epoch | allocated | reserved |
|---|---|---|
| 1 | 1.1 GB | 41.1 GB |
| 5 | 1.1 GB | 50.2 GB |
| 10 | 1.1 GB | 61.5 GB |
| **~34** | — | **~100 GB → OOM** |

This is the PyTorch CUDA allocator caching allocator holding its high-water mark. At the
current rate the run OOMs at ~epoch 34.

**Fix applied (train.py):** `torch.cuda.empty_cache()` added after each epoch's post-val
snapshot. **Caveat:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in train.sh)
prevents `empty_cache()` from unmapping segments by design — the call is harmless but may not
fully stop the creep. Monitor `rsv` in the next run. If growth continues, either:
1. Remove `expandable_segments:True` temporarily to test if it's pure fragmentation
2. Profile with `torch.cuda.memory._snapshot()` to find any live allocations growing each epoch
   (likely candidate: `all_gather_object` in `evaluate()` building a larger prediction list)

### GPU utilisation / I/O jitter — root cause and fix

`data_time` varies wildly (41–1256s) because competing Snellius jobs evict the memmap OS page
cache between epochs. When pages are warm: data=41s, gpu_util=91%. When evicted: data=1256s,
gpu_util=25%. Compute is constant at 410s — GPU is never the bottleneck.

**Fix applied (train.py):** `_advise_l369_willneed(train_dataset, val_dataset)` called at the
top of each epoch loop. Calls `arr._mmap.madvise(mmap.MADV_WILLNEED)` on all 5142 open memmap
objects — non-blocking, returns in ~ms, kernel async-prefetches evicted pages in background
during epoch setup.

**Important caveats:**
- `posix_fadvise` was originally considered but is **silently ignored by GPFS** — returns 0
  but is a no-op. `madvise(MADV_WILLNEED)` on the mmap region goes through the Linux VM layer
  and IS honoured on most GPFS (Spectrum Scale) versions, but this is version-dependent.
- Effectiveness must be verified empirically: if `data_time` jitter persists in the next run,
  GPFS on this version may be silently ignoring madvise too.

### Loss calculation — how three depths are combined

`_compute_loss` calls `masked_huber_loss(pred, label)` where:
- `pred = sm_map[:, :, STATION_ROW, STATION_COL]` — extracts the single station pixel from the
  `(B, 3, 224, 224)` output map → shape `(B, 3)`
- `mask = ~torch.isnan(label)` — True where a depth has a valid label (not all samples have
  all 3 depths present)
- `F.huber_loss(pred[mask], label[mask], delta=0.05, reduction="mean")` — **flattens all valid
  (batch × depth) pairs into one vector and takes a single mean**

The three depths are **not weighted or averaged separately** — they are pooled into one scalar.
A sample with 2 valid depths contributes 2 pairs; a sample with 3 valid depths contributes 3.
The logged `train_loss` and `val_loss` are this single scalar. TV and boundary penalties are
added on top but do not change the per-depth weighting.

**Should we weight depths separately?** See §12 below.

---

## 12. Pending Design Decisions (post-baseline)

### 12.1 Per-depth loss weighting

**Current:** all valid (batch × depth) pairs are flattened into one vector, single mean Huber.
Surface (0–10 cm) has the most observations per batch and dominates the gradient signal.

**Problem:** 30–100 cm ubRMSE is consistently ~0.006 higher than 0–10 cm (observed across all
10 epochs of job 23936932). Deeper layers receive proportionally less gradient signal because
they appear in fewer samples.

**Proposed fix:** compute separate Huber loss per depth, then mean across depths:
```python
depth_losses = []
for d in range(pred.shape[1]):
    mask = ~torch.isnan(label[:, d])
    if mask.any():
        depth_losses.append(F.huber_loss(pred[mask,d], label[mask,d],
                                          delta=0.05, reduction="mean"))
loss = torch.stack(depth_losses).mean()   # equal weight per depth
```
This gives each of the 3 depths equal optimization pressure regardless of observation count.

**When to apply:** after job 23936932 converges (use that as the flat-pool baseline).
The two approaches are not numerically comparable — track as a separate run.

**Implementation status (2026-06-17):**
- `model.py`: `masked_huber_loss` already accepts `per_depth: bool = False` — backward compatible
- `train.py` CONFIG: `"per_depth_loss": False` added
- `train.py` `_compute_loss` + `train_one_epoch`: already wired

**Remaining wiring (3 edits in train.py — defer until after baseline):**

1. `evaluate()` — thread `per_depth` through signature and its `_compute_loss` call:
```python
def evaluate(model, loader, device, world_size=1, rank=0,
             max_batches=None, per_depth=False):
    ...
    loss, _ = _compute_loss(mu, batch["label"], per_depth=per_depth)
```

2. argparse — add `--per-depth-loss` flag:
```python
parser.add_argument("--per-depth-loss", action="store_true",
                    help="Equal-weight Huber per depth (vs. pooled baseline)")
if args.per_depth_loss: CONFIG["per_depth_loss"] = True
```

3. Main loop — pass to both callers:
```python
train_one_epoch(..., per_depth=CONFIG["per_depth_loss"], ...)
evaluate(...,        per_depth=CONFIG["per_depth_loss"])
```

**Validation (before full run):**

Step 1 — unit test (seconds, no GPU):
```python
# quick_test_perdepth.py
import torch
from model import SoilMoistureModel, masked_huber_loss
pred = torch.rand(4, 3, 224, 224)
label = torch.rand(4, 3)
label[0, 2] = float('nan')   # simulate missing deep layer
loss = masked_huber_loss(pred, label, per_depth=True)
assert loss.isfinite()
loss.backward()
print(f"per-depth loss={loss.item():.5f}  ok")
```

Step 2 — smoke test (~10 min):
```bash
sbatch slurm/train.sh --run-name smoke_perdepth --max-stations 5 --max-epochs 3 --per-depth-loss
```
Pass criteria: 3 epochs complete, no NaN loss.

**Full run (after smoke only):**
```bash
sbatch slurm/train.sh --run-name perdepth_huber --per-depth-loss
```
Fresh start — do NOT resume from 23936932's checkpoint (loss landscape changed).

---

### 12.2 Depth-specific CLS token architecture (run: `cls_perdepth_huber`)

Run **after** `perdepth_huber` converges. If the 30–100 cm gap narrows with per-depth loss
alone, this is optional upside. If gap persists, this is the next motivated step.

Always use `--per-depth-loss` with this — there is no reason to do CLS tokens without it.

**Problem:** All 3 depths share one temporal context vector from `TemporalTransformer`. The
model cannot learn that deeper layers correlate with different temporal patterns (e.g., slower
seasonal response vs. fast event response at the surface).

**Design: minimal CLS insertion**

Keep the shared U-Net decoder path. Only diverge at the final output stage. Each depth gets
its own FiLM layer just before its own 1×1 head — conditioned on its depth-specific CLS
context vector. The shared conv path is unchanged.

```
TemporalTransformer input:
  [cls_0, cls_1, cls_2, S2_pyr..., S1_pyr..., ERA5..., SIF, TWSA, DEM, LULC, soil]
               ↓ transformer (all tokens attend to all others)
  [ctx_0, ctx_1, ctx_2, ...]   ← per-depth CLS outputs (B, n_depths, d_model)
        ↓ mean ↓
    global_ctx                 ← still used for skip FiLM (unchanged)

UNetDecoder:
  shared bottle→up1/2/3/4 path (global_ctx for FiLM — unchanged)
  ↓
  for d in range(n_depths):
      x_d = depth_film[d](x, ctx_d)   ← per-depth final FiLM
      out[d] = heads[d](x_d)           ← per-depth 1×1 conv
  return stack(out)  → (B, n_depths, 224, 224)
```

**File changes (all in `model.py`):**

`TemporalTransformer.__init__`:
- Add `self.depth_tokens = nn.Parameter(torch.zeros(n_depths, d_model))`

`TemporalTransformer.forward`:
- Prepend depth tokens to sequence; extend padding mask with 3 False columns
- After transformer, extract `out[:, :n_depths, :]` as `depth_ctx`
- Return both `depth_ctx (B, n_depths, d_model)` and existing `(bottleneck, context)` outputs

`UNetDecoder.__init__`:
- Add `self.depth_film = nn.ModuleList([FiLMLayer(d_context, c[3]) for _ in range(n_depths)])`
- Replace `self.head = nn.Conv2d(c[3], n_depths, 1)` with
  `self.heads = nn.ModuleList([nn.Conv2d(c[3], 1, 1) for _ in range(n_depths)])`

`UNetDecoder.forward`:
- Accept `depth_ctx: (B, n_depths, d_context)` in addition to `context`
- At output stage: loop over depths applying `depth_film[d]` then `heads[d]`

`SoilMoistureModel.forward`:
- Pass `depth_ctx` from transformer output to decoder
- Pass `context` (global pool, unchanged) for skip FiLM

**Parameter cost:**
- 3 CLS tokens: 3 × 768 = ~2.3 K params (negligible)
- 3 FiLM layers (d=768, c[3]=64): 3 × 2 × 64 × 768 ≈ 295 K params (small)
- 3 heads vs 1 head: 2 extra Conv2d(64,1,1) = ~128 params (negligible)
- **Total delta: ~300 K params** on top of existing model

**Validation (before full run):**

Step 1 — unit test (seconds, CPU-only):
```python
# quick_test_cls.py
import torch
from model import SoilMoistureModel
model = SoilMoistureModel(n_depths=3, d_model=768, use_cls_depth=True)
# ... synthetic batch with dummy tensors ...
out = model(batch)
assert out.shape == (2, 3, 224, 224)
assert not torch.allclose(out[:, 0], out[:, 1]), "All depths identical — CLS not working"
loss = out.mean()
loss.backward()
assert model.temporal_transformer.depth_tokens.grad is not None
print("CLS forward + backward ok")
```

Step 2 — smoke test (~10 min):
```bash
sbatch slurm/train.sh --run-name smoke_cls --max-stations 5 --max-epochs 3 --per-depth-loss --use-cls-depth
```
Pass criteria: 3 epochs complete, no NaN loss, loss ≤ smoke_perdepth.

**Full run (after smoke only):**
```bash
sbatch slurm/train.sh --run-name cls_perdepth_huber --per-depth-loss --use-cls-depth
```

---

### 12.3 Overfitting audit (2026-06-17)

**Symptom:** train_loss=0.0001, val_loss=0.0021 — 21× gap, stable since epoch 2 across 14
epochs of job 23936932. An external critique prompted a full audit of every suggested fix.

**Verdict on each suggestion:**

| Suggestion | Verdict | Reason |
|---|---|---|
| Weight decay → 0.1 | Low impact | Already 0.05 with correct AdamW (bias/norm excluded). Gap stable 14 epochs — more decay won't move it. |
| Stochastic depth (drop path) | **Missing — highest priority** | Completely absent. 6-layer 768-dim transformer (~85M params). Linear drop-path schedule is the standard ViT regularizer. |
| More dropout | Partial | Transformer has 0.1. UNetDecoder has **zero** dropout. ERA5/SIF/TWSA MLPs have **zero** dropout between linear layers. |
| Early stopping | Already done | patience=20 wired and functional. |
| Spatial/temporal split leakage | **Not the cause** | Agent confirmed: zero location-group overlap between train and val. Split is correctly isolated. 21× gap is genuine overfitting, not data contamination. |
| ERA5 temporal masking | Missing | S2/S1 token dropout already active. ERA5 has all 365 days always visible — no masking. Masking 10–15% of ERA5 timesteps would force generalization. |
| Augmentation (noise/crops) | Low priority | Token dropout covers spatial; temporal masking (ERA5) is the highest-value addition. Gaussian noise on ERA5/SIF/TWSA is secondary. |

**Decision: apply all fixes + CLS depth tokens together in `cls_perdepth_huber`** — since
baseline_huber is the clean reference point, the next run is the full package. Attribution
of individual components is deferred; getting the best model matters more right now.

---

### 12.4 All fixes for `cls_perdepth_huber`

`cls_perdepth_huber` = per-depth loss (§12.1) + all regularization fixes below + CLS depth
tokens (§12.2). Fresh start — do NOT resume from 23936932's checkpoint.

**Fix 1 — Stochastic depth in transformer** (`model.py`)

Replace `nn.TransformerEncoder` with a custom stack using `timm`'s `DropPath` or a manual
implementation. Linear schedule: layer `i` gets `drop_path_rate * i / (n_layers - 1)`.

```python
from timm.models.layers import DropPath

class DropPathTransformerLayer(nn.Module):
    def __init__(self, layer, drop_path_rate=0.0):
        super().__init__()
        self.layer = layer
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        residual = x
        x = self.layer(x, src_key_padding_mask=src_key_padding_mask)
        return residual + self.drop_path(x - residual)
```

CONFIG: `"drop_path_rate": 0.1` (linear schedule, 0 at layer 0 → 0.1 at layer 5).

**Fix 2 — Dropout in UNetDecoder** (`model.py`)

Add `nn.Dropout(0.15)` after the second ReLU in each `_ConvBlock`, and a `nn.Dropout(0.1)`
just before the final `self.head` conv in `UNetDecoder.forward`.

**Fix 3 — Dropout in ERA5/SIF/TWSA MLPs** (`model.py`)

Each auxiliary MLP (`era5_mlp`, `sif_mlp`, `twsa_mlp`) currently has no dropout between
linear layers. Add `nn.Dropout(0.1)` between each pair of linear layers.

**Fix 4 — ERA5 temporal masking** (`dataset.py`)

In `__getitem__`, after loading ERA5 features, randomly zero out 10–15% of time steps
during training (skip at val/test time):

```python
if self.split == "train":
    mask = torch.rand(era5.shape[0]) < 0.15   # 15% of days
    era5[mask] = 0.0
    era5_pad[mask] = True   # mark as padding so transformer ignores them
```

**Validation before full run:**

Step 1 — smoke test (5 stations, 3 epochs, ~10 min):
```bash
sbatch slurm/train.sh --run-name smoke_perdepth --max-stations 5 --max-epochs 3 --per-depth-loss
```
Pass: 3 epochs complete, no NaN loss, train_loss > baseline_huber smoke (regularization
should raise training loss slightly — that is expected and correct).

Step 2 — smoke test (~10 min):
```bash
sbatch slurm/train.sh --run-name smoke_cls --max-stations 5 --max-epochs 3 --per-depth-loss --use-cls-depth
```
Pass: 3 epochs complete, no NaN, train_loss slightly higher than baseline smoke (regularization raises training loss — expected).

Step 3 — full run:
```bash
sbatch slurm/train.sh --run-name cls_perdepth_huber --per-depth-loss --use-cls-depth
```

---

### 12.5 GPFS Checkpoint Buffering Bug — Post-Mortem (Session 14, 2026-06-17)

**Symptom:** Job 23936932 ran 14 epochs and printed "New best val_loss — checkpoint saved" 7 times, but `last.pt` and `best.pt` on disk both contained epoch=3 state. `val_station_metrics.csv` also froze at epoch 3.

**Diagnosis:** GPFS write buffering. `torch.save` on gcn128 writes into the GPFS client cache synchronously (returns immediately), but the data is only flushed to the storage servers asynchronously. When `scancel` terminated the job, ~1.2 GB of dirty checkpoint data per epoch (2 files × 600 MB) was in the cache and discarded. Epoch 3's data happened to survive because the initial cold-write of a 600 MB file triggered a flush due to cache pressure.

**Confirming evidence:**
- Direct I/O (`dd iflag=direct`) still read epoch=3 — rules out login-node page cache; the storage server genuinely had epoch=3.
- `mid_epoch.pt` mtime 17:48 — GPFS writes worked fine for other files during the run; only the periodic 600 MB checkpoint *overwrites* were unbuffered late.
- `val_station_metrics.csv` (58 KB) also froze at epoch=3 — the entire `if is_main:` block was running but I/O wasn't reaching storage.

**Fix:** `_fsync_save(obj, path)` in `train.py` — calls `torch.save` then opens the file and calls `os.fsync(fd)` before returning. Overhead: ~6 s per 600 MB file (98.7 MB/s measured). Three saves per epoch (mid-epoch, pre-val, post-val) → ~18 s overhead on a ~480 s epoch (<4%).

**Impact on 23936932 results:** val_loss plateaued at epoch 2 (0.0021); epoch 3 was already the best checkpoint. Loss of epochs 4–14 is immaterial.

---

### 12.6 Train / Val / OOS Split Audit (Session 14, 2026-06-17)

**Station inventory (sm_only, Phase 1):**

| Split | Count | Purpose |
|---|---|---|
| train | 587 | gradient updates |
| val | 74 | early stopping, LR scheduling |
| oos | 181 | final paper metrics, spatial maps |
| **total sm_only** | **842** | — |

`category_filter: ["sm_only"]` is hardcoded in CONFIG. The remaining 151 stations (sm_and_flux + flux_only) are excluded from Phase 1.

**Distribution balance:**

| Climate (kg_macro) | train | val | oos |
|---|---|---|---|
| A (tropical) | 4 | 1 | 0 |
| B (arid) | 147 | 13 | 45 |
| C (temperate) | 166 | 21 | 60 |
| D (continental) | 267 | 39 | 76 |
| E (polar) | 3 | 0 | 0 |

| Land cover (IGBP macro) | train | val | oos |
|---|---|---|---|
| Forest | 257 (43.8%) | 36 (48.6%) | 76 (42.0%) |
| Grass-Crop | 258 (43.9%) | 32 (43.2%) | 78 (43.1%) |
| Shrub-Savanna | 38 (6.5%) | 4 (5.4%) | 16 (8.8%) |
| Other | 34 (5.8%) | 2 (2.7%) | 11 (6.1%) |

Climate and land cover fractions are proportional — no category is over/under-represented in any split.

Elevation: val is slightly Low-heavy (49% vs 36% in train). Minor; not a concern.

**Spatial stratification:** All 783 location groups fall entirely within one split — no nearby stations leak across train/val/oos. Val and OOS metrics are not inflated by spatial autocorrelation. This is the critical property for defensible paper results.

**Planned evaluation workflow:**
1. Train → early stop on val loss
2. Best checkpoint → OOS inference → ubRMSE / MAE / bias per depth
3. OOS predictions → spatial maps (predicted vs observed SM)
4. Potentially: demo_plot.py for visual results

OOS (181 stations, 21.5% of sm_only) is the number that goes in the paper. Val is internal only.

---

### 12.7 Smoke Test — smoke_cls (job 23956770, 2026-06-17)

**Command:** `sbatch slurm/train.sh --run-name smoke_cls --max-stations 5 --max-epochs 3 --per-depth-loss --use-cls-depth`

**Result: PASSED**

- 3 epochs completed, no NaN, finite loss throughout
- `_fsync_save` worked: "New best val_loss=0.0045 — checkpoint saved" at epoch 3
- VRAM: 42.3 GB peak (vs 38.6 GB baseline — +3.7 GB from CLS FiLM layers, acceptable)
- RAM: 177.8 GB / 811 GB (22%) — healthy

Code is ready for full `cls_perdepth_huber` run.

---

### 12.8 Experiment sequence

| Step | Action | Time | Purpose |
|---|---|---|---|
| 1 | `quick_test_perdepth.py` | seconds | catch NaN/shape bugs in per-depth loss |
| 2 | `quick_test_cls.py` | seconds | catch shape/grad bugs in CLS arch |
| 3 | smoke `smoke_cls` (5 stations, 3 epochs) | ~10 min | confirm all fixes together |
| 4 | **full run `cls_perdepth_huber`** | ~85 h | full model vs `baseline_huber` |

Steps 1–3 gate step 4.

| Run | Changes vs baseline | Compares to |
|---|---|---|
| `baseline_huber` (23936932) | — | — |
| `cls_perdepth_huber` | per-depth loss + stochastic depth + decoder/MLP dropout + ERA5 masking + CLS depth tokens | `baseline_huber` |

---

## 13. Meeting Evaluation & Visualization Plan (Session 15, 2026-06-18)

Comprehensive evaluation and figure generation for the `baseline_huber` best checkpoint. Four new scripts, one SLURM job.

### 13.1 Split Definitions

| Split | `split_filter` | `years` | Purpose |
|-------|---------------|---------|---------|
| OOS  | `["oos"]` | 2016–2022 | Spatial generalization (181 held-out stations) |
| OOT  | `["train","val"]` | `[2023]` | Temporal generalization (2023 is fully held-out year) |
| OOST | `["oos"]` | `[2023]` | Spatial + temporal (hardest condition, 128 stations) |

### 13.2 Scripts

**`evaluate_splits.py`** — GPU inference across all 3 splits.
- Reuses `evaluate()` + `compute_metrics()` from `train.py`
- Adds Pearson R per station (computed from per-station preds/targets post-gather)
- Outputs to `meeting_output/`:
  - `metrics_summary.csv` — split × depth × {ubRMSE, RMSE, MAE, R, bias}
  - `per_station_{oos|oot|oost}.csv` — per-station metrics joined with station metadata (IGBP, climate, lat/lon)

**`plot_timeseries_meeting.py`** — GPU; needs inference.
- Selects 5 best + 5 worst OOS stations by surface ubRMSE from step above
- Per-station multi-panel figure: time series all 3 depths (all available years) + world map inset + metadata table + ERA5 precip bar

**`plot_breakdown_meeting.py`** — CPU only; reads CSVs.
- Violin plots: surface ubRMSE by IGBP macro and Koppen macro
- Grouped bar: mean ubRMSE for OOS vs OOT vs OOST by IGBP macro
- Depth comparison bar: 0-10 / 10-30 / 30-100 ubRMSE per split
- Scatter predicted vs observed (3 depths, OOS, coloured by IGBP macro)

**`plot_satellite_sm_meeting.py`** — GPU; uses zarr L3 tokens.
- Applies to the same best-5 + worst-5 OOS stations
- Loads `s2/l3` + `s1_asc/l3` tokens from zarr → PCA(3) → pseudo-RGB at 14×14 → bicubic 224×224
- Runs inference → (n_depths, 224, 224) SM map
- Figure per station: [S2 pseudo-RGB | S1 pseudo-SAR | SM 0-10 | SM 10-30 | SM 30-100]

**`slurm/evaluate_meeting.sh`** — 1 GPU, 16 CPUs, 90 min.
- Runs evaluate_splits.py → plot_timeseries_meeting.py → plot_satellite_sm_meeting.py sequentially

### 13.3 Publication-Quality Plot Standards

All figures use `plt.style.use(["science", "nature"])` (scienceplots), DPI=300, constrained layout.
- Depth colours: `#e74c3c` (0-10), `#2980b9` (10-30), `#27ae60` (30-100) — consistent across all plots
- Split colours: `#1a6faf` (OOS), `#e8851a` (OOT), `#9b59b6` (OOST)
- SM spatial maps: `viridis` or `YlOrBr_r`, 0–0.5 m³/m³
- All axes labelled with units; N annotated on box/violin plots

### 13.4 Execution

```bash
# Step 1 — GPU job (~40 min)
sbatch slurm/evaluate_meeting.sh

# Step 2 — CPU, after step 1 completes (~5 min)
conda activate terramind && python plot_breakdown_meeting.py

# All outputs in meeting_output/
```

### 13.5 Sanity Checks

After `evaluate_splits.py` completes:
- OOT ubRMSE ≈ val ubRMSE (model saw these stations — temporal shift only)
- OOS ubRMSE > val ubRMSE (novel stations — expected degradation)
- OOST ubRMSE ≥ OOS (hardest: novel stations + novel year)
- `metrics_summary.csv` should have 9 rows (3 splits × 3 depths)

---

## §14. Spatial Resolution Roadmap (future work)

### 14.1 Current Limitation

The model is **point-supervised**: the loss is computed only at the station pixel (112, 112).
The output is `(n_depths, 224, 224)` but the effective spatial resolution is **14×14 tokens**
(one TerraMind ViT token covers 16×16 pixels = 160m×160m at 10m/px).

Consequences:
- All 196 tokens receive the same gradient signal (from the single center pixel)
- Global self-attention homogenises predictions → nearly uniform SM maps
- The spatial decoder produces no meaningful spatial variation
- Current `plot_spatial_sm_meeting.py` shows this: SM panels look like a flat colour

### 14.2 What's Needed for 10m Spatial Maps

Two changes are required simultaneously:

**A. Spatial decoder (architecture change)**

Replace the current output head with a U-Net-style decoder using the L3/L6/L9 skip connections
already stored in zarr:

```
TerraMind L3 (14×14×768) ──► ConvTranspose 2× ──► 28×28×384  ──┐
TerraMind L6 (14×14×768) ──► ConvTranspose 2× ──► 28×28×384  ──┤ FPN merge
TerraMind L9 (14×14×768) ──► ConvTranspose 2× ──► 28×28×384  ──┘
                              ConvTranspose 2× ──► 56×56×192
                              ConvTranspose 2× ──► 112×112×96
                              ConvTranspose 2× ──► 224×224×n_depths  (SM map)
```

The station point still anchors the absolute SM value; the decoder learns spatial variation.

**B. Spatial supervision signals**

One point per station cannot teach spatial variation. Options (weakest → strongest):

| Signal | Resolution | Notes |
|--------|-----------|-------|
| SMAP L4 | 9 km, daily | Coarse; constrains patch-average SM |
| S1 VV/VH backscatter | 10 m | Physically correlated with surface SM; self-supervised spatial regularisation |
| S2 optical indices (NDVI, EVI, BSI) | 10 m | Vegetation/bare-soil proxies for SM retention |
| Multiple ISMN stations in same tile | point | Rare; but exists for dense networks (e.g. SCAN, OzNet) |

### 14.3 Recommended Approach (Phase 2)

1. **Add lightweight U-Net decoder** to `model.py` (reuse L3/L6/L9 tokens already in zarr)
2. **Keep point loss** at (112, 112) — stations still anchor absolute values
3. **Add spatial regularisation loss**: for each sample, penalise large SM gradients in areas
   where S2 spectral similarity is high (similar reflectance → similar SM)
   `L_spatial = mean(|∇SM| * spectral_similarity_mask)`
4. **Weak SMAP constraint**: compute patch-mean predicted SM and penalise deviation from
   nearest SMAP L4 pixel (after bias correction)

This is a meaningful PhD contribution — going from point-supervised SM estimation to
spatially-resolved 10m SM mapping using only freely available satellite data.

### 14.4 Expected Outcome

- SM maps will show land-cover-driven spatial structure (crops vs. forest vs. urban)
- Validation: compare spatial patterns against airborne / drone SM surveys (if available)
  or use SMAP spatial correlation as a proxy metric
- Model SROW/SCOL centre-pixel metrics will remain the primary accuracy benchmark

### 14.5 Current Workaround (for meeting)

The spatial figure (`plot_spatial_sm_meeting.py`) has been updated to show:
- Real S2 true-colour RGB, S1 VV SAR, DEM, LULC (from `/projects/prjs1968/satellite_zarr/`)
- SM panels show model output but spatial variation is limited — frame as point estimate
- Future figures will replace flat SM panels with spatially-resolved maps once Phase 2 is done

---

## §15. TerraMind Embedding Diagnostic (Tier-0) — Session 17, 2026-07-01

**Motivation.** §14 shows SM maps are nearly flat and §12.3 shows a persistent 21× train/val
gap. Before investing in a spatial decoder (§14) or more regularization (§12.4), we must answer
a prior question: **do the frozen TerraMind L12 tokens actually carry scene-tracking,
land-cover-discriminative structure, or do they wash out to a near-constant embedding?** The
answer forks the roadmap:

- **Rich, scene-tracking tokens → Tier 1** — the information exists; the problem is downstream
  flattening (pyramid mean-pool §3e, global attention homogenisation §14.1). Fix the decoder.
- **Flat wash → Tier 2** — the tokenization/input itself is uninformative. No decoder can
  recover signal that isn't there; fix inputs/tokenization first.

Deliverable: a per-station figure (raw scene → token L2 norm → PCA→RGB) read against a Tier-0
checklist, for 4 land-cover-contrasting held-out stations.

### 15.1 Verified data facts (recon 2026-07-01)

Token store: `/gpfs/scratch1/shared/pkhanal/zarr/{category}/{station}/`
(`/scratch-shared/pkhanal` is a symlink to the same path). **Open with
`zarr.open_consolidated(path)`** — plain `open_group` returns empty (store uses `.zmetadata`).
993 stores on disk (sm_only 842, sm_and_flux 48, flux_only 103); no `.complete` sentinels —
treat "dir opens via open_consolidated" as completeness.

Per-station structure (verified on `ISMN_AMMA-CATCH_Banizoumbou`):

| path | shape | dtype | notes |
|---|---|---|---|
| `s2/{l3,l6,l9,l12}` | `(N,196,768)` | fp16 | 196 = 14×14 tokens |
| `s2/dates` | `(N,)` | U8 | `YYYYMMDD` |
| `s1_asc/{l3,l6,l9,l12}` + `dates` + `token_mask (M,14,14) bool` | `(M,196,768)` | fp16 | |
| `s1_desc/*` | — | — | **present only if orbit exists** (absent for this station) |
| `cm/masks` | `(N,224,224)` | uint8 | **0 = clear, nonzero = cloud/shadow** |
| `cm/dates` | `(N,)` | U8 | **index-aligned 1:1 with `s2/dates`** (verified N∩N = N) |
| `dem`, `lulc` | `(196,768)` | fp16 | single static L12 embedding (not replicated over T) |
| `dem_token_mask`, `lulc_token_mask` | `(14,14)` | bool | |

Both original recon blockers are resolved by the data itself: cloud masks are inline and
index-aligned (no external store / SCL fallback), and every modality carries explicit `dates`
(token index → date mapping is guaranteed). Written by `create_token_zarr.py`.

Raw imagery (for the "raw scene" column): `/projects/prjs1968/satellite_zarr/<Network>_<station>.zarr`
— `s2/data (T,12,224,224)` int16 + `s2/dates`; attrs `pixel_size_m=10`, `patch_size_px=224`
→ 2.24 km patch; lat/lon/epsg present.

Station metadata: `csvs/station_splits.csv` (993 rows). On-disk dir name =
`ISMN_{network}_{station_name}` (ISMN) / `{source_network}_{station_id}` (else) — matches 993/993.
sm_only held-out land cover (from §12.6): **oos** Forest 76 / Grass-Crop 78 / Shrub-Savanna 16 /
Other 11; climate B 45 / C 60 / D 76 / A 0 / E 0. **val** has the only tropical-A held-out station.

### 15.2 Decisions (locked with user)

1. **Seasons: climate-aware.** Temperate → DJF/MAM/JJA/SON with Southern-Hemisphere 6-month
   flip; Köppen A/B (tropical/arid) → wet/dry. Derived from latitude + `kg_macro`.
2. **Station pool: held-out only** — `split ∈ {val, oos}`.
3. **Cloud-free: whole-patch fraction** — acquisition qualifies if `mean(cm_mask != 0) < 0.08`
   over the full 224×224 patch.

### 15.3 Procedure

Script `visualize_embeddings.py` (env `terramind` — zarr/torch/rasterio). Output → `embed_viz_output/`.
SLURM wrapper `slurm/visualize_embeddings.sh` (CPU-only; include
`--mail-type=BEGIN,END,FAIL --mail-user=ktm.prajwalkhanal@gmail.com`).

**Phase 1 — station selection (4 contrasting, held-out).** From `split ∈ {val, oos}` with an
openable store, pick 4 contrasting IGBP×Köppen classes: Forest (C/D), Grass-Crop (D),
Shrub-Savanna (B arid), and a tropical-A/wetland/"Other" station. Keep only candidates with
(a) ≥1 whole-patch cloud-free S2 acquisition per climate-aware season, (b) raw+token stores
present, (c) geographic spread; prefer stations with `s1_desc` (degrade gracefully if absent).
Print the 4 for sign-off.

**Phase 2 — date selection.** S2: per season, the acquisition minimizing whole-patch cloud
fraction (<0.08); `s2/dates[i]` → token slice `i` directly; skip ~all-zero slices
(`abs(tok).max()<1e-6`) or mostly-false `token_mask`. S1 (asc + desc if present): no cloud
filter; 4 dates spanning the year, matched to S2 where possible. DEM/LULC: one static embedding.

**Phase 3 — visualization (L12 only).** Per station: rows = modality (S2×4 seasons, S1 asc,
S1 desc if present, DEM, LULC); columns = raw scene → per-token L2 norm (14×14) → PCA→RGB (14×14).
Raw S2 RGB from B4/B3/B2 of the 12-band int16 stack (~/10000, clip; confirm band indices vs
attrs). Reshape `[196]→[14,14]` row-major; verify orientation vs raw scene, transpose/flip if
mirrored. Grey out invalid tokens via `token_mask`/`*_token_mask`. Shared norm/colormap across
the 4 S2 seasons; PCA sign/rotation arbitrary — compare boundaries, not hues. Per-panel metrics:
off-diagonal mean cosine, PCA top-3 variance ratio, neighbor autocorrelation.

**Phase 4 — verdict.** Read each figure against the Tier-0 checklist; record per-station verdict
+ aggregate call (Tier 1 vs Tier 2) here.

### 15.4 Verification checklist

1. Phase 1 dry-run prints 4 stations + per-season cloud-free dates; every season has a qualifying
   acquisition.
2. Assert `s2/dates == cm/dates` element-wise per station at load.
3. Render one station first; fix raw-vs-PCA orientation and S2 RGB band mapping before batching all 4.
4. DEM/LULC panels non-degenerate; S1 asc ≠ desc where both exist.
5. Inspect `embed_viz_output/*.png`; write the Tier verdict in this section.

### 15.5 Notes

- `graphify` CLI is not installed on this host (`command not found`) despite the repo hook;
  code questions answered by reading source directly.

### 15.6 Verdict — spatial structure collapses at L12 (Session 17, 2026-07-01)

`visualize_embeddings.py` written and run on the 4 selected held-out stations (figures in
`embed_viz_output/embed_<station>.png`): PSA7Ruebezahl (Forest, Dfb), Balruddery (Grass-Crop,
Cfb), YucaipaValley (Shrub-Savanna, BSk), DWDBerlin-Spaeth (Other, Dfb). Figure = per-layer
PCA→RGB sweep (raw scene → L3 → L6 → L9 → L12), one modality-acquisition per row.

**Quantitative Tier-0 metrics** — `neighbor_ac` = spatial autocorrelation of the per-token L2
norm map (higher ⇒ tokens vary smoothly like the scene), averaged over the 4 cloud-free S2
seasons per station:

| Station (IGBP) | L3 | L6 | L9 | **L12** |
|---|---|---|---|---|
| PSA7 (Forest) | +0.92 | +0.89 | +0.86 | **+0.10** |
| Balruddery (Grass-Crop) | +0.54 | +0.53 | +0.48 | **−0.01** |
| Yucaipa (Shrub-Savanna) | +0.72 | +0.73 | +0.72 | **+0.23** |
| Berlin-Späth (Other) | +0.59 | +0.53 | +0.51 | **+0.02** |

`offdiag_cos` stays ~0.4–0.65 across layers (tokens are distinct, NOT collapsed to identical);
`pca_top3` captures 0.29–0.65 of variance at L3 (shallow maps are genuine structure, not noise).

**Verdict: Tier 1, not Tier 2.** Scene structure is richly present and spatially coherent through
L3/L6/L9 (autocorr +0.47…+0.92) and **collapses to ~0 at L12** in all 4 stations — deep global
attention scrambles token locality (classic ViT behaviour; matches §14.1). Tokenization/input is
fine. The problem is downstream: the temporal transformer is fed **L12 pyramid tokens** (§3e) —
the layer where spatial structure is gone — while the spatially-rich L3/L6/L9 enter only as U-Net
decoder skips (§8.5b, §14.2). This is the mechanism behind §14's flat SM maps.

**Recommended next step (Phase 2 / Tier-1 fix):** route L3/L6/L9 spatial structure more directly
into the prediction path (e.g. pool/attend over L9 instead of L12 for the transformer, or a
stronger FPN over the L3/L6/L9 skips), rather than relying on L12 for the main representation.

**Caveat on the figure:** per-panel PCA with 2–98% per-channel percentile stretch makes L3–L9
maps *look* like high-frequency "confetti" in thumbnails despite high autocorrelation — trust the
`neighbor_ac` numbers over the RGB appearance. A norm-map column (or smoothed PCA) would show the
coherence more directly if the figure is used for a talk.

- Status: Phases 0–4 complete. Verdict recorded above; Tier-1 routing change deferred to Phase 2.

### 15.7 How the metrics are computed & what we actually did

**What we did (procedure that produced §15.6).**
1. Selected 4 held-out (`split ∈ {val, oos}`) stations with contrasting IGBP×Köppen and 4
   cloud-free climate-aware seasons each (Phase 1 of `visualize_embeddings.py`).
2. For each station, per S2 season, pulled the token grid at **each layer L3/L6/L9/L12**
   (`s2/l{3,6,9,12}[i]`, shape `(196, 768)` = 14×14 tokens × 768 dims).
3. Rendered `embed_<station>.png` (raw scene → per-layer PCA→RGB), and separately dumped the
   three scalar metrics below, **averaged over the 4 seasons** per (station, layer) → the §15.6
   table. Metrics live in `panel_metrics()` / `pca_rgb()` in `visualize_embeddings.py`.

**Metric definitions** (per panel = one acquisition at one layer; N=196 tokens, each 768-d;
padded tokens excluded via `token_mask` where present):

- **`cos` — mean off-diagonal cosine similarity.** L2-normalise every token vector, form the
  196×196 cosine matrix, average the strict upper triangle (all distinct token pairs).
  `cos→1` ⇒ tokens near-identical (collapsed embedding); lower ⇒ tokens distinct.
  Measures *content diversity*, not spatial layout.

- **`ac` — neighbour autocorrelation of the token-norm map.** Reshape the per-token L2 norm to
  14×14; take the Pearson correlation between horizontally-adjacent cells
  (`nm[:, :-1]` vs `nm[:, 1:]`) over finite pairs. High positive ⇒ neighbouring tokens have
  similar magnitude ⇒ smooth, scene-tracking spatial structure; ≈0 ⇒ spatially incoherent.
  NOTE: (a) computed on the scalar **norm**, not the full 768-d vector — it is a *proxy* for
  spatial locality; (b) **horizontal neighbours only** (vertical not yet included).

- **`pca` — top-3 PCA variance ratios.** Center the 196 valid tokens, SVD; report
  `S[k]²/ΣS²` for k=0,1,2. These are the fractions of total token variance shown as R/G/B in the
  PCA→RGB image. Small values ⇒ the RGB shows only a sliver of the structure (looks noisy even
  when tokens are structured — the "confetti" artifact of per-panel PCA + 2–98% percentile
  stretch).

**Measured vs. interpreted (calibration of the §15.6 verdict).**
- *Directly measured:* (i) `cos(L12) ≈ 0.57–0.63` ≈ `cos(L3)` → L12 is **no more collapsed** than
  shallow layers (tokens stay mutually distinct); (ii) `ac(L12) ≈ 0` vs `ac(L3/L6/L9) ≈ +0.5…+0.9`
  → L12's **norm field** loses the spatial autocorrelation the shallow layers have; (iii) decent
  `pca` at L3 → shallow structure is real signal, not noise.
- *Interpretation (not proven by these numbers):* "L12 is spatially **scrambled by global
  attention**." The scramble is inferred from the norm-map proxy; attention was not measured.
  L12 could in principle retain spatial structure in vector directions the norm does not capture.

**Direct test to remove the proxy (recommended before paper claims).** Per panel, compare mean
cosine of **spatially-adjacent** token pairs vs **random** token pairs, on the full 768-d vectors:
shallow layers should show adjacent ≫ random; if L12 shows adjacent ≈ random it *directly*
demonstrates lost spatial locality (no norm proxy, no attention assumption). At scale, corroborate
with **CKA across L3→L12** and a **land-cover linear probe per layer** on the 220 feasible stations.

## §16. Tier-1 Diagnostic — where does spatial variance die downstream? (Session 18, 2026-07-01)

**Goal.** Tier-0 (§15) measured the *frozen tokens* (no model). Tier-1 loads the **trained downstream
weights** and asks which stage of the prediction path flattens the signal, using one forward pass per
station. Checklist:
- 1.1 Are target-day tokens still distinct after the temporal transformer, or homogenised by attention?
- 1.2 Does the 14×14×768 bottleneck have spatial variance, or is it flat before the U-Net starts?
- 1.3 If the bottleneck has variance but the 224² output is smooth → is the **decoder** smoothing?
- 1.4 Net call: loss at **attention** (upstream) vs **conv upsampling** (downstream)?

**Model probed.** `baseline_huber_memmap/best.pt` — the final full-train run (epoch 11, best_val≈0.00209,
`use_cls_depth=False`). Loaded via `ckpt_utils.load_checkpoint` (remaps legacy keys, reads config).

**Pipeline & taps** (all external hooks — `model.py` unchanged):
```
raw scene (S2 RGB / S1 VV)     ← recovered by matching anchor_l12 back to the token zarr
  → L12 anchor tokens [196,768]   batch["anchor_l12"]                       (tap 1)
  → temporal transformer + transformer_norm
  → 196 spatial tokens reshaped = bottleneck [768,14,14]                    (tap 2)
       = forward-PRE-hook input to model.decoder.bottle_proj
  → U-Net decoder conv1/conv2/conv3  feature maps 28²/56²/112² (channel-mean) (taps 3–5)
  → SM map [n_depths,224,224]      model forward return                     (tap 6)
```
Key structural facts confirmed while building the probe:
- **Loss is single-pixel:** `masked_huber_loss` supervises only the station centre pixel (112,112);
  `total_variation_loss` regularises the whole map toward smoothness.
- **Decoder upsampling is BILINEAR, not transposed conv:** all four up-stages are
  `nn.Upsample(mode="bilinear") → _ConvBlock` (`model.py:223-233`) — bilinear interpolation is an
  intrinsically *smoothing* operator, so "distinct bottleneck → smooth output" is the expected decoder-side
  failure mode.
- The **transformer-output** and **bottleneck** taps are the *same tensor* (spatial slice reshaped), so the
  genuine taps are L12 → bottleneck → decoder stages → output. This still cleanly separates the attention
  side (L12→bottleneck) from the decoder side (bottleneck→output).
- Continuity with Tier-0: the L12 fed to the transformer already lost norm-autocorrelation (`neighbor_ac`≈0,
  §15.6) while staying content-distinct (`offdiag_cos`≈0.6). Spatially-rich L3/L6/L9 still enter as decoder
  skips, so output structure (if any) may come from the skips, not the transformer path.

**Metrics** (`tier1_probe.py`, reuses `panel_metrics`/`pca_rgb` from `visualize_embeddings.py`):
- `offdiag_cos`, `neighbor_ac` — Tier-0-comparable token metrics on L12 and bottleneck.
- `rel_spatial_std` — scale-free spatial dispersion of a token grid (per-feature std across the 196 cells /
  per-feature magnitude, averaged).
- `pc1_var_ratio` — fraction of token variance in PC1.
- `norm_std` — normalised spatial std (std/|mean|) of each decoder-stage map and the output SM map.

**Decision table (1.4):**

| Observation | Culprit |
|---|---|
| bottleneck `cos` jumps toward ~0.9 (was ~0.6 at L12) AND `rel_spatial_std` drops sharply | **attention homogenises** — upstream |
| bottleneck stays distinct but decoder-stage `norm_std` collapses toward the output | **decoder smooths** (bilinear + single-pixel loss) — downstream |
| bottleneck already flat despite distinct L12 | reshape/projection between them |

`verdict()` auto-applies this: attention-side if `cos_jump > 0.15` or `disp_ratio < 0.5`, else decoder-side.

**Run.** Same 4 stations as Tier-0. Dataset built on **only those 4 stations** (subset splits CSV → ~4 not
255). GPU SLURM job `slurm/tier1_probe.sh` (`conda run -n terramind`, mail flags). Outputs:
`tier1_output/tier1_probe.png` (row per station: raw scene → L12 → transformer-out → decoder 28/56/112 →
output) + `tier1_output/tier1_metrics.json` + printed verdict. First run: job **24349182**.

**Snags fixed to get the run green** (job 24349536 = final):
- The 4 held-out stores lack the `.complete` marker `dataset._open_zarr` requires (every group is
  actually present). `tier1_probe.patch_open_zarr_no_marker()` bypasses ONLY the marker — no data changed.
- `anchor_l3/l6/l9/l12` are fp16; decoder convs are fp32 → cast all floating batch tensors to fp32 in
  the probe (`_prep`; training used autocast). Integer/bool tensors left untouched.

### §16.1 Result — DECODER-SIDE collapse (Session 18, 2026-07-01)

Job **24349536** COMPLETED. Four stations, anchor = most-recent clear acquisition nearest doy 180
(2× S2, 2× S1_DESC — all dated 2023-06-27 / 2018-06-26). Averaged taps:

| Tap | cos ↓ | rel_spatial_std | note |
|---|---|---|---|
| **L12 in** | 0.561 | 1.164 | distinct tokens, full spatial dispersion |
| **Bottleneck (post-transformer)** | 0.329 | 1.106 | cos **drops** (tokens *more* distinct), dispersion preserved |
| Decoder 28² / 56² / 112² (n-std) | — | 0.089 / 0.095 / 0.034 | structure survives to 56², collapses by 112² |
| **Output SM 224²** (n-std) | — | **0.0065** | essentially uniform |

**1.1** Tokens stay distinct after the transformer — `cos` *falls* 0.56→0.33 and `neighbor_ac` mostly
*rises* (e.g. Yucaipa 0.006→0.236). Attention does **not** homogenise; if anything it re-sharpens.
**1.2** Bottleneck has full spatial variance (`rel_spatial_std` 1.11 ≈ L12's 1.16). Not flat.
**1.3** Decoder feature maps keep structure at 28²/56² (n-std ~0.09) then collapse: 112²→0.034,
output→0.0065 (~14× drop from the mid-decoder to the head). The figure shows a bright **central
hotspot** at the supervised pixel in the decoder maps while the rest goes flat — the network learned
the (112,112) pixel and paints the surround uniform.
**1.4 NET CALL → DECODER-SIDE.** The loss of spatial variance is at **conv upsampling**, not attention.
Bilinear `Upsample`→conv (no transposed conv) + single-pixel (112,112) Huber supervision + TV
smoothness give the decoder neither the mechanism nor the incentive to paint off-centre detail.

**Recommended fix (Phase 2):** add spatial supervision so the whole 224² map is trained, not just the
centre pixel — e.g. multi-pixel / patch loss around the station, an auxiliary dense target (ERA5-Land
SM or a coarse SM product resampled to the tile), and/or replace bilinear upsampling with
learned upsampling. Note this is complementary to §15.6's routing fix (feed L3/L6/L9 structure into the
main path): §15 addresses *what enters*, §16 addresses *what the decoder does with it*.

- Artefacts: `tier1_probe.py`, `slurm/tier1_probe.sh`, `tier1_output/{tier1_probe.png,tier1_metrics.json}`.
- Status: Tier-1 complete. Verdict = decoder-side. Fix deferred to Phase 2.

### §16.2 How to read the figure (`tier1_output/tier1_probe.png`)

4 rows (one per Tier-0 station) × 7 columns = the input→output spatial progression. Each column is
rendered **differently** — do not compare colours across column *types*:

| # | Column | Rendering | Colour means | Units / scale |
|---|---|---|---|---|
| 1 | **Raw scene** | true imagery — S2 RGB (B04/B03/B02) or S1 VV grey | reflectance / backscatter | stretched 2–98% |
| 2 | **L12 in** | **PCA→RGB** of the 196 tokens | token's position in its top-3 PCA space | arbitrary (per-panel PCA) |
| 3 | **Transformer out** | **PCA→RGB** of the 196 tokens | same as col 2 | arbitrary (per-panel PCA) |
| 4–6 | **Decoder 28²/56²/112²** | **channel-mean** of the conv feature map, `viridis` | mean activation at that pixel | real activation (colorbar shown) |
| 7 | **Output SM 224²** | the **actual SM prediction** (0–10 cm), `YlOrBr_r` | soil moisture | m³/m³, fixed `vmin0–vmax0.5` |

Rendering details & caveats:
- **PCA→RGB** (`pca_rgb`, cols 2–3): center the 196×768 tokens, SVD, project onto the top 3 PCs → an
  (R,G,B) per cell, reshaped 14×14, per-channel 2–98% stretch. It makes 768-D structure *visible*.
  Two caveats: (a) **PCA is fit per-panel**, so a red cell in col 2 is NOT the same feature as red in
  col 3 — compare *structure*, not hue; (b) the percentile stretch exaggerates faint structure into
  "confetti", so trust the title numbers (`cos`, `disp`) over vividness.
- **Channel-mean** (cols 4–6): `fmap = tap.mean(axis=0)` over feature channels — a literal activation
  map, no PCA, no stretch. Chosen because the decoder question is "is spatial variance still present?",
  which the mean map answers directly. Title `n-std = std/|mean|` is computed on that exact map.
- **Output** (col 7): not a visualisation of features — the real physical prediction. Near-uniform
  orange = the predicted SM really is ~spatially constant.
- Only cols 2–3 are PCA. Token panels (arbitrary PCA units) and decoder/output panels (real units) are
  not on a shared scale → compare *within* a type; the `n-std` numbers give the quantitative
  cross-stage comparison of flatness.
- **What to look for:** structured cols 2–6 (especially a bright dot at the centre station pixel in the
  decoder maps) next to a flat col 7 = the decoder-side collapse. That contrast IS the result.

### §16.3 Architecture note — why the bottleneck is 14×14, not 4 (clarification)

Common confusion: "we feed 4 levels, so how do we get 14×14×768 out?" Two separate token paths:
- **Target-day anchor** (`anchor_l12` `[196,768]`) enters the transformer at **full 14×14 = 196 tokens,
  un-pooled** (`_get_target_spatial_tokens`). These are *the* spatial tokens.
- **History + static context** (DEM, LULC, soil, every *past* S2/S1 acquisition) is compressed
  `[196,768] → [4,768]` by `_cpu_pyramid_pool` (`dataset.py:190`) — 4 nested concentric masked-mean
  windows centred on the station pixel (a spatial *pyramid*, NOT the L3/L6/L9/L12 transformer layers).
  Reason is purely payload: many history dates × 196 tokens blew up the DataLoader IPC queue (~30 MB→2 MB
  per sample); history needs multi-scale context, not per-pixel detail.

The transformer sequence is `[DEM×4 | LULC×4 | Soil×4 | anchor×196 | S2hist×(N·4) | S1hist×(N·4) |
ERA5×365 | SIF | TWSA]` with `spatial_start = 12` (4+4+4). The bottleneck is recovered by slicing those
196 anchor positions back out and reshaping — nothing is "un-summarised": the 196 rode through the
transformer intact, attention just contextualised them, and `ctx[:, 12:12+196].reshape(14,14,768)` puts
them back on the grid the U-Net expects.

### §16.4 Decoder-side fix — plan (Session 18, 2026-07-01; NOT yet run)

**Decision (with user): the 224² map is a genuine deliverable** (10 m dense SM maps, §14 roadmap), so we
commit to adding real spatial supervision.

**Reframe — the root cause is supervision, not the conv layers.** The map is 224² but trained on exactly
ONE pixel (station at 112,112); ISMN is a *point* measurement, so there is no ground truth anywhere else
on the tile. The bilinear decoder is doing the only rational thing under the loss: nail the supervised
pixel, paint the rest flat to satisfy the TV penalty. **No architecture change creates structure the loss
never asks for.** Encouraging: §15.6 shows the spatially-rich L3/L6/L9 already reach the decoder as skips —
the information is present, the decoder just isn't rewarded for using it.

**Step 0 — capacity check (cheap, do first, ~minutes).** Overfit ONE station for a few hundred steps
against a *synthetic structured* dense target (S1- or NDVI-shaped field), re-run `tier1_probe.py`.
- produces structure ⇒ decoder capacity is fine, supervision is the only lever (expected) → green-light Step 1.
- still flat ⇒ bilinear layers are limiting → architecture (Step 2) must come first.
(The TV-loss ablation is NOT a standalone Step-0 test — it needs retraining, so it lives in Step 1.)

**Step 1 — proxy-consistency spatial supervision (the real work).** Keep point loss, add structure loss,
relax the smoother:
1. **Point loss (unchanged):** Huber at (112,112) vs ISMN — pins the *absolute level*.
2. **Structure loss (new):** build a physical wetness proxy field `W(x)` at 224² from data we already have,
   penalise the predicted map for not matching its *normalised spatial pattern*
   (e.g. `1 − spatial_corr(pred_norm, W_norm)`, or gradient-matching). Proxies, priority order:
   - **S1 VV/VH backscatter** (moisture-sensitive) — anchor S1 scene in `RAW_ROOT`.
   - **Topographic Wetness Index** from DEM (`TWI = ln(a/tanβ)`, static; valleys wetter).
   - NDVI from S2 (optional; vegetation proxy, use with care).
3. **Cut the TV weight** (it currently rewards flatness) — this is where the TV ablation lives.
4. **Fine-tune from `baseline_huber_memmap`** (cheaper than scratch; the probe becomes the before/after test).

**Caveats (on the record):** the proxy is a *weak structural prior*, not truth — S1↔SM is confounded by
vegetation/roughness; TWI is static, no dynamics; and S1 is already a model input (mild circularity).
Validation, since we lack dense truth: (a) point-station ubRMSE must not degrade; (b) qualitative map
inspection via the same probe; (c) COSMOS-UK stations (cosmic-ray, ~hundreds-of-m footprint) give a
semi-areal check.

**Step 2 — architecture, only if Step 1 underperforms.** Swap bilinear → learned upsampling
(transposed conv / pixel-shuffle). Secondary — never instead of Step 1.

- Status: plan agreed, **not started**. Next action = Step 0 capacity check.

---

## §17. `UNetDecoder` — line-by-line reference (Session 19, 2026-08-03)

Reference: `model.py:188-270`, plus `model.py:499-510` (`_get_skip_connections`) and
`model.py:700-720` (`SoilMoistureModel.forward`).

### 17.1 What it does

Takes the transformer's 14×14 spatial bottleneck and upsamples to a 224×224 map with
`n_depths` channels, re-injecting TerraMind L9/L6/L3 on the way up, each FiLM-modulated by
the temporal context vector.

### 17.2 Flow

```
 INPUTS (from SoilMoistureModel.forward, model.py:700-720)
 ┌──────────────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────────────┐
 │ bottleneck           │ │ skip_L9       │ │ skip_L6       │ │ skip_L3       │ │ context      │ │ depth_ctx (opt.)   │
 │ (B,768,14,14)  L707  │ │ (B,768,14,14) │ │ (B,768,14,14) │ │ (B,768,14,14) │ │ (B,768)      │ │ (B,n_depths,768)   │
 │ transformer output   │ │ TerraMind raw │ │ TerraMind raw │ │ TerraMind raw │ │ mean of non- │ │ depth CLS tokens   │
 │ reshaped             │ │ L505-510      │ │               │ │               │ │ spatial toks │ │ L703               │
 └──────────┬───────────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └──┬───┬───┬───┘ └─────────┬──────────┘
            ▼ L246                ▼ L247            ▼ L248            ▼ L249       │   │   │              │
   ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐  ┌──────────────┐     │   │   │              │
   │ bottle_proj     │   │ skip_proj[0] │   │ skip_proj[1] │  │ skip_proj[2] │     │   │   │              │
   │ 1×1 768→512     │   │ 1×1 768→512  │   │ 1×1 768→256  │  │ 1×1 768→128  │     │   │   │              │
   │ (L211)          │   │ (L213)       │   │ (L214)       │  │ (L215)       │     │   │   │              │
   └────────┬────────┘   └──────┬───────┘   └──────┬───────┘  └──────┬───────┘     │   │   │              │
            │                   ▼                  ▼                 ▼             │   │   │              │
            │            ┌──────────────┐   ┌──────────────┐  ┌──────────────┐     │   │   │              │
            │            │ film_s9 L219 │◄──┤ film_s6 L220 │◄─┤ film_s3 L221 │◄────┘   │   │              │
            │            │ scale*x+shift│   │              │  │              │◄────────┘   │              │
            │            └──────┬───────┘   └──────┬───────┘  └──────┬───────┘◄────────────┘              │
            ▼ x (B,512,14,14)   │ s9               │ s6              │ s3                                 │
   ┌─────────────────┐          │                  │                 │                                    │
   │ up1  bilinear×2 │ L251     │                  │                 │                                    │
   │ → (B,512,28,28) │          │                  │                 │                                    │
   └────────┬────────┘          │                  │                 │                                    │
            │      ┌────────────▼───────────┐      │                 │                                    │
            └─────►│ cat( x, interp(s9→28) )│ L252 │                 │                                    │
                   │ → (B,1024,28,28)       │      │                 │                                    │
                   └────────────┬───────────┘      │                 │                                    │
                   ┌────────────▼───────────┐      │                 │                                    │
                   │ conv1 1024→256  L224   │      │                 │                                    │
                   └────────────┬───────────┘      │                 │                                    │
                   ┌────────────▼───────────┐      │                 │                                    │
                   │ up2 → (B,256,56,56)    │ L254 │                 │                                    │
                   └────────────┬───────────┘      │                 │                                    │
                   ┌────────────▼───────────┐      │                 │                                    │
                   │ cat( x, interp(s6→56) )│◄─────┘  L255           │                                    │
                   │ → (B,512,56,56)        │                        │                                    │
                   └────────────┬───────────┘                        │                                    │
                   ┌────────────▼───────────┐                        │                                    │
                   │ conv2 512→128   L227   │                        │                                    │
                   └────────────┬───────────┘                        │                                    │
                   ┌────────────▼───────────┐                        │                                    │
                   │ up3 → (B,128,112,112)  │ L257                   │                                    │
                   └────────────┬───────────┘                        │                                    │
                   ┌────────────▼───────────┐                        │                                    │
                   │ cat( x, interp(s3→112))│◄───────────────────────┘  L258                              │
                   │ → (B,256,112,112)      │                                                             │
                   └────────────┬───────────┘                                                             │
                   ┌────────────▼───────────┐                                                             │
                   │ conv3 256→64    L230   │                                                             │
                   └────────────┬───────────┘                                                             │
                   ┌────────────▼───────────┐  L260  (no skip left — L12 grid exhausted)                  │
                   │ up4 → (B,64,224,224)   │                                                             │
                   └────────────┬───────────┘                                                             │
                   ┌────────────▼───────────┐                                                             │
                   │ conv4 64→64     L261   │                                                             │
                   └────────────┬───────────┘                                                             │
                   ┌────────────▼───────────┐                                                             │
                   │ pre_head_drop   L262   │                                                             │
                   └────────────┬───────────┘                                                             │
                   ┌────────────▼───────────────────────────┐                                             │
                   │  use_cls_depth AND depth_ctx not None? │  L264                                       │
                   └───────┬──────────────────────┬─────────┘                                             │
                      NO   │                      │  YES  loop d = 0..n_depths-1 (L266)                    │
              ┌────────────▼───────┐   ┌──────────▼───────────────────┐                                   │
              │ head: 1×1 64→3     │   │ depth_film[d](x, ────────────────────────────────────────────────┘
              │ (L241, L270)       │   │            depth_ctx[:,d,:]) │ L267
              │ ONE shared feature │   │  then heads[d]: 1×1 64→1     │ L268
              │ map, 3 biased      │   └──────────────┬───────────────┘
              │ slices             │                  ▼ cat over dim=1 (L269)
              └─────────┬──────────┘   ┌──────────────────────────────┐
                        └──────────────┴──────────────┬───────────────┘
                                                      ▼
                                      OUTPUT  (B, n_depths, 224, 224)
```

### 17.3 `__init__` (L197-241)

| Line | What |
|---|---|
| L199-204 | `in_ch=768`, `skip_ch=768`, `dec_ch=(512,256,128,64)`, `n_depths=3`, `d_context=768` |
| L211 | `bottle_proj` — 1×1, channel squeeze 768→512, no spatial mixing |
| L212-216 | 1×1 convs projecting each skip to the channel count of its target stage (512/256/128) |
| L219-221 | One `FiLMLayer` per skip. `FiLMLayer` (L150-168) maps `context (B,768)` → `(B,2C)`, splits into per-channel scale/shift. Zero-init weights, (1,0)-init bias (L158-160) → **identity at step 0**, so training starts from a plain U-Net |
| L223-233 | Four `Upsample(bilinear ×2)` + `_ConvBlock` pairs. `_ConvBlock` (L171-185) = Conv3×3→BN→ReLU ×2 → `Dropout2d(0.15)`. `c[i]+c[i]` in-channels = concat of upsampled `x` and skip. `conv4` has no skip |
| L235 | `Dropout(0.1)` before the head — element-wise, not channel-wise |
| L237-241 | Two head modes. Default (L241): single 1×1 64→`n_depths` = **195 depth-specific params**. `use_cls_depth=True` (L238-239): per-depth FiLM + per-depth 1×1 head |

### 17.4 `forward` (L243-270)

| Line | What |
|---|---|
| L246 | Bottleneck 768→512, still 14×14 |
| L247-249 | Each skip: 1×1 project → FiLM with global `context`. Skips are the **raw** TerraMind L3/L6/L9 of the anchor acquisition (L505-510) — never pass through the transformer, so FiLM is what makes them time-aware |
| L251-252 | 14→28, concat FiLM'd **L9** (deepest skip → coarsest stage), conv → 256 ch |
| L254-255 | 28→56, concat **L6**, conv → 128 ch |
| L257-258 | 56→112, concat **L3** (shallowest skip → finest stage), conv → 64 ch |
| L260-261 | 112→224, conv only |
| L264-269 | Per-depth: FiLM the shared 224² map with that depth's CLS vector, apply that depth's head, concat |
| L270 | Shared: one 1×1 conv emits all depths |

### 17.5 Two structural facts

**(a) Every skip is 14×14 — the `F.interpolate` calls do real work.**
`_get_skip_connections` (L505-510) reshapes 196 tokens → 14×14, so L3/L6/L9 are all at the
same resolution and get bilinearly stretched ×8/×4/×2 (L258/255/252). TerraMind is a *plain*
ViT: patchify once (16×16 → 196 tokens), then every block keeps the same token count. Depth
in a ViT buys **semantic abstraction**, not spatial density.

Contrast with a classic U-Net, where the skip at 112×112 was *computed from the image at
112×112* and carries 12,544 independent spatial measurements. Here the 112×112 skip is
`interp(s3)` — stored as 12,544 numbers but with **spatial rank ≤ 14 per axis, i.e. 196
degrees of freedom**. Same tensor shape, 64× less information.

At 224×224 @ 10 m = 2.24 km (`download_s2_mpc.py:15,57-58`), one token = **160 m × 160 m**.
Effective smoothing length ≈ 2 tokens ≈ 320 m.

Note this is *not* an information bound on the model — bilinear upsampling is injective and
sub-patch structure survives in the 768 channels. `196 × 768 = 224² × 3 = 150,528` exactly
(since `16² × 3 = 768`), so the budget is balanced 1:1 and `PixelShuffle(16)` is the exact
bijection. The problem is that bilinear-then-3×3-conv makes *smooth* the zero-effort default
and nothing in the loss rewards deviating from it.

**(b) 195 depth-specific parameters.** With `use_cls_depth=False`, `64×3+3 = 195` params out
of the whole model distinguish one depth from another. See §18.

### 17.6 Why bilinear+conv rather than transposed conv

Checkerboard artifacts (Odena et al. 2016): `ConvTranspose2d(k=3,s=2)` gives uneven output
overlap → periodic grid pattern. Invisible after `argmax` in segmentation; a visible
artifact in a continuous physical field, compounded over 4 stages (16× total upsampling).
Resize-then-conv is the prescribed fix. Secondary: bilinear is parameter-free, is a fixed
smooth operator so the decoder starts as a sensible interpolator, and decouples geometry
from feature mixing.

Counterpoint on the record: bilinear *is* a strong smoothness prior. But swapping the
upsampler cannot fix smoothness while the loss supervises one pixel — see §18.4.

### 17.7 Reference point — how TerraMind's own decoder gets sharp output

Verified against the installed package
(`site-packages/terratorch/models/backbones/terramind/`), not the paper.

Generation path (`model/terramind_generation.py:278, 364`; `model/generate.py:64-98`):
image → tokenizer `.encode()` → FSQ tokens → transformer predicts target-modality tokens
(MaskGIT-style cosine schedule) → target tokenizer `.decode_tokens()` → pixels.

S2L2A tokenizer (`tokenizer/tokenizer_register.py:157`, authors' recorded training args):

| | |
|---|---|
| Type | `divae` — **diffusion** VQ-VAE (`tokenizer/vqvae.py:545`) |
| Quantizer | `fsq`, `codebook_size="8-8-8-6-5"` = 15,360 codes ≈ **13.9 bits/token** |
| Decoder | `unet_patched`, `patch_size_dec=4` |
| Objective | per-pixel MSE, 1000 timesteps; DDIM 50 steps at inference |

**It is synthesis, not reconstruction.** 196 × 13.9 bits ≈ **341 bytes** vs ~1.2 MB raw —
~3,500× compression. `decode_quant` states generations are stochastic by default
(`vqvae.py:707`); same tokens twice → different images. Detail is sampled from a learned
prior. **Do not route SM through a generative decoder** — publishing spatial structure drawn
from a landscape prior rather than measured is a liability.

`unet_patched` (`tokenizer/models/unet/unet.py:680-740`) — what it actually does:
1. patchify `(B,C,224,224) → (B,C·16,56,56)` — space→depth, **lossless**, = `PixelUnshuffle(4)` (L715)
2. `F.interpolate(tokens, (56,56), mode="nearest")`, concat as extra channels (L721-722)
3. UNet at 56×56, `num_res_blocks=3`, `channel_mult=(1,2,2,2)` (L725)
4. depatchify → `(B,C,224,224)` = `PixelShuffle(4)` (L728)

Correction to an earlier reading: TerraMind **does** interpolate its 14×14 token grid, same
as us. The real difference is that its UNet's *main input* is a full-resolution noisy image
— a carrier with independent values at all 50,176 pixels — which the coarse tokens merely
condition. We have no carrier; the 14×14 grid is simultaneously our signal and our only
source of spatial variation.

| | full-res carrier | coarse conditioner | detail is |
|---|---|---|---|
| TerraMind DiVAE | diffusion latent @224 | 14×14 tokens, nearest-upsampled | hallucinated |
| Our `UNetDecoder` | **none** | 14×14 tokens, bilinear-upsampled | absent (smooth) |
| Proposed (§16 Step 1/2) | raw 10 m S1 pixels @224 | 14×14 tokens, upsampled | **measured** |

Ranked constraints on map sharpness:
1. **single-pixel supervision** — decisive, architecture-independent
2. **no full-resolution carrier** — structural; carrier must be measured pixels, not noise
3. bilinear vs PixelShuffle — real but smaller; rearranges what exists, cannot create a carrier

This supersedes §16 Step 2 ("swap bilinear → learned upsampling") as the *primary*
architectural lever: the carrier matters more than the upsampler.

---

## §18. Planned change — per-depth dynamics (Session 19, 2026-08-03)

### 18.1 Problem

`use_cls_depth=False` (`train.py:206`). All three depths share everything and differ by
**195 parameters** (§17.5b). The decoder gets one `context` vector, mean-pooled identically
for all depths (`model.py:712-717`), so no depth can learn a different temporal response.
Symptom: surface fits, 30–100 cm regresses (run 25150428, stopped e6).

The model never sees soil moisture as an *input* — the sequence is
`[DEM×4 | LULC×4 | Soil×4 | spatial×196 | satellite | era5×365]` (`model.py:519`), confirmed
by the batch keys. Depth dynamics can therefore only be learned from the **loss**, which
means there must be a per-depth parameter for that gradient to land on. Today there is
almost none.

(Excluding SM history is the right call — it would make the model useless at ungauged
locations, which is the point of the work. The cost is a harder learning problem.)

### 18.2 Change 1 — turn on `use_cls_depth`

CLI `--use-cls-depth` (`train.py:617`). Prepends `n_depths` learned tokens to the sequence
(`model.py:411, 689`).

Mechanism, in Q/K/V terms: `W_q`, `W_k`, `W_v` are **shared**; `q_d = W_q · depth_tokens[d]`
differs because `depth_tokens[d]` differs. Keys are unchanged. So
`softmax(q_d · Kᵀ)` gives each depth its **own** weighting over the 365-day ERA5 history.
Attention weights over a time axis are a learnable convolution kernel → a per-depth impulse
response. Depth 0 can concentrate on recent days; depth 2 on a long damped window.

`depth_tokens[d]` is trained by `L_d` via `heads[d]`/`depth_film[d]`, so the *target*
teaches the query where to look — no SM input needed.

Depth-specific params: **195 → ~295 k**. Cost: +3 tokens, ~1–3 % compute.

Bookkeeping already handled correctly in the code: `cls_pad` marks the depth slots valid
(L691-692); `sp_start` shifts by `depth_offset` so the bottleneck slice stays correct
(L705); depth CLS excluded from the `context` mean-pool (L713-714).

Known asymmetry: because the depth tokens are attendable keys, `depth_tokens[d]` also
receives gradient from *all* depths via its effect on the spatial tokens. Isolating that
would need an attention mask. Defensible as-is (cf. DINOv2 register tokens) — noting it so
it isn't rediscovered as a bug.

### 18.3 Change 2 — break the depth-token symmetry

```python
# model.py:411
self.depth_tokens = nn.Parameter(torch.zeros(n_depths, d_model))
nn.init.trunc_normal_(self.depth_tokens, std=0.02)          # ADD
```

Zero-init → all three queries numerically identical at step 0. No positional encoding is
added to those slots (L690), so attention is permutation-equivariant over them and
`depth_ctx[:,0,:] == depth_ctx[:,1,:] == depth_ctx[:,2,:]` **exactly**. They separate only
because `depth_film[d]`/`heads[d]` random inits differ. One line removes an unnecessary
handicap on the mechanism being tested.

### 18.4 Change 3 — star residual across depths

Replace `model.py:264-269`:

```python
base = self.heads[0](self.depth_film[0](x, depth_ctx[:, 0, :]))
out  = [base]
for d in range(1, self.n_depths):
    out.append(base + self.heads[d](self.depth_film[d](x, depth_ctx[:, d, :])))
return torch.cat(out, dim=1)
```

Depth 0 predicts SM absolutely; deeper depths predict an **offset from the surface**.
Zero-init the `d ≥ 1` heads so training starts from "all depths = surface prediction".

**Star (all offsets from depth 0), not chain (each offset from the depth above):** the ≥95 %
coverage filter drops depths per station, so a sparsely-observed middle depth would become a
weak link corrupting everything below it. Star isolates that.

Effects:
- base carries the **common** signal (site wetness); deltas carry the **depth-specific**
  signal (attenuation, lag) — cleaner factorisation, and the deltas are smaller/lower-variance
  than absolute values, so easier to fit.
- `heads[0]` now receives gradient from all three losses. The surface becomes a shared
  baseline — watch for slight degradation in the surface metric.
- Loss unchanged: still Huber on absolute values per depth. Pure output reparameterisation.

Rationale: the depth tokens have no idea the depths are *ordered* or physically coupled —
`depth_tokens[0]` means "surface" only because row 0's gradient always comes from the 0–10 cm
error. Deep SM is roughly a lagged, damped integral of shallow SM; the star residual injects
that prior rather than hoping it's discovered.

### 18.5 Explicitly NOT doing yet — 3×3 per-depth heads

A 1×1 head can only mix the 64 shared channel maps: `y_d = Σ_c (W_d[c]·s_d[c])·x[c,h,w]`.
Different depths *can* get different patterns (different points in that 64-map span), but no
**neighbourhood** operation is possible — no shift, blur, or spatial gradient. A depth offset
field is physically exactly that (lateral flow, diffusion), so a 3×3 is well-motivated in
principle.

Deferred because the loss reads **one pixel** (`model.py:733`): per-depth spatial patterns are
completely unsupervised, so a 3×3 head adds capacity nothing trains and no way to measure it.
Revisit with the §16 Step 1 dense-supervision work.

### 18.6 Run

```bash
sbatch slurm/train.sh --run-name cls_depth_star --use-cls-depth
```

`per_depth_loss` already `True` (`train.py:210`); `lambda_tv` already `0.0` (L213);
`lambda_boundary` `0.1` (L214).

### 18.7 What to watch

| Signal | Expect |
|---|---|
| Per-depth val loss, 30–100 cm | should move across epochs instead of flat-lining |
| Surface val loss | small degradation acceptable (shared baseline); large ⇒ star residual hurting |
| Epoch of overfit | previously e3; more depth-specific capacity may pull this earlier |
| Depth-token divergence | cosine similarity between the 3 rows should drop below 1.0 early |

### 18.8 Scope note

Everything in §18 is on the **depth** axis. Map smoothness is the **spatial** axis (§16, §17.7)
and is untouched by any of it. Order there is unchanged: dense supervision first, carrier
second, upsampler third.

### 18.9 Status

**Implemented 2026-08-03.** Changes landed in `model.py`:

| Change | Location |
|---|---|
| `trunc_normal_(std=0.02)` on `depth_tokens` | `model.py:411-417` |
| Zero-init offset heads (`heads[1:]`) | `model.py:237-247` |
| Star residual in `UNetDecoder.forward` | `model.py:270-281` |

The `use_cls_depth=False` path is untouched, so existing checkpoints still load.

Smoke test (decoder + model instantiation, dummy tensors):

| Check | Result |
|---|---|
| Output shape | `(2, 3, 224, 224)` — unchanged |
| All depths identical at step 0 (star residual + zero-init offsets) | True |
| `depth_tokens` std | 0.0199 |
| `cos(depth_tokens[0], depth_tokens[1])` | **−0.024** (was exactly 1.0) |
| Depth-specific params | **297,795** (was 195) |

~~Next action = launch `sbatch slurm/train.sh --run-name cls_depth_star --use-cls-depth`.~~

**Superseded by §19** (Session 20, 2026-08-05). The launch was deferred to bundle in two more
things: per-depth *loss* reporting (§19.3-19.4) and regularisation (§19.5). Run name is now
`cls_depth_star_reg`.

---

## §19. Run `cls_depth_star_reg` — per-depth loss reporting + regularisation (Session 20, 2026-08-05)

### 19.1 Why

Two things are being fixed at once, both consequences of run `baseline_huber_notv_perdepth`
(job 25150428, W&B `pg7mw3xb`, stopped at epoch 6).

**(a) The architecture fix (§18) has never actually run.** All of §18 landed in `model.py` on
2026-08-03 but the flag was never switched on, so every run to date has had only **195 of
50,050,944 parameters depth-specific**. Symptom: 0-10 and 10-30 cm flat while 30-100 cm
*regressed* — ubRMSE .0559 → .0566 and wet bias +.0146 → +.0245 monotonic across e3→e6.

**(b) There is still no per-depth LOSS anywhere.** `train.py` reports per-depth
MSE/MAE/ubRMSE/bias on *validation only* (L955-957, L1009-1012). The training loss is a single
pooled scalar. That gap matters because the two candidate explanations for a stuck 30-100 cm
need **opposite** fixes and are indistinguishable without it:

| Per-depth train loss | Per-depth val loss | Diagnosis | Fix |
|---|---|---|---|
| high, flat | high | capacity or information ceiling | architecture (cascade), or accept the ceiling |
| low, falling | high, rising | label scarcity / generalisation | depth-aware sampling, regularisation |

Recall the physical caveat (logs.txt S16): S2 optical senses ~1 cm, S1 C-band a few cm. 30-100 cm
information can *only* come from ERA5-Land, soil texture, and temporal memory. If the trunk never
encoded it, no head architecture will extract it. Per-depth train loss is what tells us whether
we are fighting capacity or physics.

**(c) Regularisation.** Val loss bottomed at epoch 3 and rose thereafter while train loss fell
5×. Classic overfit onset. `early_stop_patience=20` would have burned ~20 epochs past best.

### 19.2 Readiness (verified 2026-08-05)

| Item | State |
|---|---|
| §18 model changes | Committed (0d68932); `use_cls_depth=False` path untouched, old checkpoints still load |
| Zarr on scratch | Present — `sm_only` 842, `sm_and_flux` 48, `flux_only` 103 `.complete` markers |
| `.npy` anchor memmaps | Present (7,401 files); `slurm/train.sh:47` hard-wires `--use-memmap` |
| TV / smoothness loss | Off — `lambda_tv: 0.0` (`train.py:213`), disabled since d339085 per the Tier-1 verdict |
| Boundary penalty | On — `lambda_boundary: 0.1`; range penalty on SM ∉ [0,1], not smoothness |
| GPU budget | 730,202 / 800,000 SBU remaining |
| Queue | Empty |
| Checkpoint collision | New run name ⇒ new dir; no stale `last.pt` / `mid_epoch.pt` to resume from |

### 19.3 Change 1 — `masked_huber_loss(..., return_breakdown=True)`

`model.py:745`. When set, additionally returns `(depth_sum, depth_cnt)`, both `(n_depths,)`
float32, on-device, **detached**.

Computed branch-free — better than the sketch in logs.txt L2914, which reused the existing
`if mask_d.any():` pattern and would add a GPU sync per depth per batch:

```python
valid     = ~torch.isnan(label)                                     # (B, D)
lab       = torch.nan_to_num(label, nan=0.0)
elem      = F.huber_loss(pred, lab, delta=delta, reduction="none")  # (B, D)
depth_sum = (elem * valid).sum(0).detach()                          # (D,)
depth_cnt = valid.sum(0).float().detach()                           # (D,)
```

No data-dependent control flow ⇒ no new `.item()` calls, no new syncs.

**The returned scalar `loss` is unchanged.** Both the `per_depth` and pooled branches stay
byte-identical, so `val_loss` remains comparable to `baseline_huber_notv_perdepth`.

**Consequence to remember:** the mean of the new per-depth losses will *not* equal `val/loss`.
The scalar is a mean-of-batch-means-of-depth-means; the breakdown is sample-weighted over the
whole epoch. Sample-weighting is the correct choice here — deep coverage is sparse (43 val
stations at 30-100 vs 74 at 0-10), so a batch-mean would over-weight batches holding two deep
samples. Two different quantities on purpose; do not expect them to reconcile.

`return_breakdown` defaults `False`, so `eval_stations.py` / `demo_plot.py` are unaffected.

### 19.4 Change 2 — accumulate, reduce, log

- `_compute_loss` (`train.py:356`) gains `return_breakdown=False` and forwards through. Two call
  sites only: L476 (train), L543 (val).
- `train_one_epoch` (L434) and `evaluate` (L520) accumulate `depth_sum` / `depth_cnt` as device
  tensors and return the **raw sums**, not means — the DDP reduction is only correct on sums.
- Reduction is `all_reduce(..., op=SUM)` for the new vectors. Note this differs from the existing
  scalar-loss reduction, which is `AVG` (`train.py:556`) and stays as-is.
- Divide as `sum / cnt.clamp(min=1)`, then write `nan` where `cnt == 0`, so a depth absent from
  the batch or from a smoke subset reports cleanly instead of as a misleading `0.0000`.
- **Edge case:** the DDP reduce block (L936) is guarded by `epoch != val_pending_epoch`, and the
  val-pending resume path (L857-861) skips training entirely. Initialise both vectors to zeros in
  that branch or it `NameError`s on resume-after-val-crash.

Reporting:
- Epoch line (L952) → **6 decimals** on `train_loss` / `val_loss`. The 4-decimal print is exactly
  what hid the rising val loss last run: stdout showed a flat `0.0022` for four epochs while the
  checkpoints held 0.002182277 → 0.002230212.
- Per-depth print (L955-957) gains `train_loss` and `val_loss` columns.
- New W&B keys: `train/{depth}/loss`, `val/{depth}/loss`, `val/{depth}/MSE` (computed today but
  never logged), `val/worst_depth_loss`.
- **Mechanism check** (§18.7): log pairwise cosine similarity of the `depth_tokens` rows as
  `diag/depth_token_cos_{01,02,12}`. Should sit near −0.02 at init and stay well below 1.0. If it
  returns to ≈1.0 the depth queries have collapsed and `use_cls_depth` is inert — this is the
  cheapest possible check that the whole premise of the run is holding.

### 19.5 Change 3 — regularisation, as CLI flags

`CONFIG` keeps its baseline values; new flags override them, so the run is reproducible from the
sbatch line and the defaults stay clean for comparison runs.

| Flag | CONFIG key | Baseline | This run | Rationale |
|---|---|---|---|---|
| `--weight-decay` | `weight_decay` (L194) | 0.05 | **0.1** | overfit by e3 |
| `--drop-path-rate` | `drop_path_rate` (L205) | 0.1 | **0.2** | stochastic depth is the strongest regulariser available for a 6-layer ViT and is already plumbed to the model (L735) |
| `--early-stop-patience` | `early_stop_patience` (L198) | 20 | **6** | 20 would burn ~20 epochs × ~440 s past best |
| `--lr-patience` | `lr_patience` (L195) | 10 | **3** | first LR drop should land near the overfit knee, not 10 epochs after it |

`DropPath` has no parameters, so raising `drop_path_rate` does not change the parameter set.
Weight decay already excludes bias/norm via the existing `decay_params` / `no_decay_params` split
(L748-754) — no change needed there.

**Stated confound.** This run changes architecture *and* regularisation together, so a worse
result cannot be cleanly attributed to `use_cls_depth`. Accepted deliberately to save a run cycle.
The mitigation is the new per-depth **train** loss: if regularisation is the culprit, train loss
rises across all three depths together. If the architecture is, the depths move apart.

### 19.6 Run

Smoke first. Note `slurm/train_a100_smoke.sh` does **not** hard-wire `--use-memmap` (unlike
`train.sh:47`), so pass it explicitly:

```bash
sbatch slurm/train_a100_smoke.sh --run-name smoke_cls_star_reg --use-cls-depth --use-memmap \
  --max-stations 20 --max-epochs 2 --max-train-batches 5 --max-val-batches 5 \
  --weight-decay 0.1 --drop-path-rate 0.2
```

Pass criteria:

| Check | Expect |
|---|---|
| `Trainable parameters:` | **50,348,544** (was 50,050,944) — proves the flag took effect |
| Zarr fallback | No `[WARN]` line ⇒ memmaps are being read |
| Per-depth train/val loss | Finite for depths present in the 20-station subset; `nan` (not `0.0000`, not a crash) for any absent depth |
| `diag/depth_token_cos_01` | ≈ −0.02, not 1.0 |
| Epoch 2 completes | Mid-epoch / resume paths untouched |

Then the full run:

```bash
sbatch slurm/train.sh --run-name cls_depth_star_reg --use-cls-depth \
  --weight-decay 0.1 --drop-path-rate 0.2 --early-stop-patience 6 --lr-patience 3
```

H100×4, ~440 s compute/epoch. Data time swings 27-691 s on GPFS contention (§3g) — that variance
is storage, not the model.

### 19.7 What to watch

| Signal | Expect / meaning |
|---|---|
| `train/30-100/loss` | **The key diagnostic.** High + flat ⇒ capacity or information ceiling. Falling while `val/30-100/loss` rises ⇒ label scarcity |
| `val/30-100/ubRMSE`, `bias` | Should stop the monotonic wet drift (+.0146 → +.0245 was the failure mode) |
| `val/0-10/*` | Small degradation acceptable — under the star residual `heads[0]` is now a shared baseline receiving gradient from all three losses. Large degradation ⇒ star residual is hurting |
| `diag/depth_token_cos_*` | Must stay well below 1.0 |
| Epoch of best val | Was e3. More depth capacity may pull it earlier; heavier regularisation should push it later |

**Comparison baseline:** `baseline_huber_notv_perdepth` (W&B `pg7mw3xb`), compared on per-depth
ubRMSE/MAE/bias in m³/m³. **Never compare the `val_loss` scalar across runs with different loss
definitions** — `baseline_huber_memmap`'s 0.00209 pooled all (sample × depth) pairs, which is a
different quantity from per-depth-then-mean.

### 19.8 Follow-up branches

Do **not** respond to a flat 30-100 cm by adding more parameters.

- **If per-depth train loss says capacity** — next lever is a **cascade** (10-30 as a residual on
  0-10, 30-100 as a residual on 10-30) rather than the current star. Physically motivated: deep SM
  is roughly a lagged, damped integral of shallow SM. Star was chosen first because the ≥95 %
  coverage filter drops depths per station, so a sparse middle depth in a chain is a weak link;
  revisit that trade-off only if star has demonstrably plateaued.
- **If it says label scarcity** — depth-aware sampling or per-depth loss weighting.
- **If train loss is high and flat even with 297,795 depth params** — that is the information
  ceiling, and the honest conclusion is that 30-100 cm is not recoverable from this input set.

Still blocked on the single-pixel loss (`model.py:753`), unchanged: dense spatial supervision
(§16.4) and 3×3 per-depth heads (§18.5).

### 19.9 Status

**Implemented 2026-08-05.** Changes landed:

| Change | Location |
|---|---|
| `return_breakdown` on `masked_huber_loss` | `model.py:745-807` |
| `_compute_loss` forwarding | `train.py:357-382` |
| `_per_depth_mean` helper (nan where cnt==0) | `train.py:384` |
| Accumulate + return sums in `train_one_epoch` | `train.py:474-477, 505-510, 550` |
| Accumulate + `all_reduce(SUM)` in `evaluate` | `train.py:570-576, 597-603, 628` |
| `val_pending_epoch` vector init | `train.py:931-932` |
| Train-side `all_reduce(SUM)` + `_per_depth_mean` | `train.py:1014-1021` |
| 6-decimal epoch print, per-depth train/val columns | `train.py:1029-1041` |
| 6-decimal "New best" line | `train.py:1158` |
| CONFIG echo at startup | `train.py:806-812` |
| W&B keys + depth-token cosine diagnostic | `train.py:1097-1118` |
| Regularisation CLI flags | `train.py:667-674, 688-691` |
| `eval_stations.py` updated for new `evaluate` arity | `eval_stations.py:70-78` |

Two issues found while wiring it up, both fixed:
- `eval_stations.py` unpacked `evaluate` into 3 values and would have crashed.
- The per-depth print iterated `metrics`, but `compute_metrics` **drops** a depth entirely when
  val has no samples for it (`train.py:324-325`). Train and val are different station sets, so a
  depth can be trained and not validated — its train loss would have vanished from the log. Now
  iterates `SM_DEPTHS` and prints `no val samples`.

**Offline verification (CPU):**

| Check | Result |
|---|---|
| Scalar loss identical with/without `return_breakdown`, both loss modes | True |
| Absent depth | `nan`, not `0.0` |
| All-NaN label batch | loss 0.0, no crash |
| DDP arithmetic — 4 simulated ranks, uneven depth coverage vs single-pass truth | match to 1e-8 |
| Param count `use_cls_depth` False → True | 50,050,944 → **50,348,544** |
| `drop_path_rate` changes param count | False (as expected — `DropPath` has no params) |
| `depth_tokens` cos 01/02/12 at init | −0.047 / 0.014 / −0.067 |
| `heads[1]` zero-init | True |

**Smoke test — job 25234282** (A100×4, 20 stations, 2 epochs, batch 32, W&B `alm7sze4`): PASS.

| Check | Result |
|---|---|
| `Trainable parameters:` | **50,348,544** ✓ |
| Zarr fallback `[WARN]` | none — 138 L369 memmap arrays opened ✓ |
| Per-depth train/val loss | all finite and independent ✓ |
| DDP grad-stride warning | now `[1, 64, 1, 1]` (per-depth head) vs `[3, 64, 1, 1]` before — independent confirmation the branch switched ✓ |
| `diag/depth_token_cos_01` | −0.047 (init) → 0.088 after 2 epochs — tokens diverging, mechanism live ✓ |
| W&B keys | `train/{d}/loss`, `val/{d}/loss`, `val/{d}/MSE`, `val/worst_depth_loss`, `diag/depth_token_cos_*` all present ✓ |

Epoch 1 per-depth train loss was near-identical across depths (0.0268 / 0.0275 / 0.0273) — correct
by construction: the star residual with zero-init offset heads makes all depths the same prediction
at step 0. By epoch 2 they separate (0.0132 / 0.0116 / 0.0127).

Sanity check on the documented §19.3 caveat: epoch-1 `train_loss` 0.047413 vs per-depth mean
0.02722. The 0.0202 gap is the boundary penalty (`lambda_boundary=0.1` × ~0.2), which the per-depth
breakdown excludes by design. The two numbers are not supposed to reconcile.

**Smoke re-run — job 25234340** (6 stations, 1 epoch): re-verifies the final code, since 25234282
predated the `SM_DEPTHS` print-loop fix, the CONFIG echo, and the 6-dp "New best" line. Fewer
stations to also exercise the `no val samples` branch.

The CONFIG echo confirms every flag reached `CONFIG` — this line is now the single place to check
what a run actually used:

```
CONFIG: run_name=smoke_cls_star_reg2  use_cls_depth=True  per_depth_loss=True  lr=0.0002
        batch_size=32  weight_decay=0.1  drop_path_rate=0.2  n_layers=6  lambda_tv=0.0
        lambda_boundary=0.1  early_stop_patience=6  lr_patience=3
```

Result: PASS, zero errors. Per-depth train loss again near-identical at epoch 1
(0.026945 / 0.026448 / 0.027202) as the star residual predicts. The `no val samples` branch was
still not hit live — both smoke subsets happened to carry all three depths in val — so it remains
covered only by the offline unit test. It is a `.get()` fallback, so the risk is a cosmetic print,
not a crash.

**Full run — job 25234370**, launched then **cancelled** 2026-08-05 at 12 min elapsed, still
inside the SHM preload with no checkpoint written. Cancelled deliberately to fold in §19.10
before spending H100 hours. Nothing was lost.

---

## §19.10 Two gaps closed before the real launch (Session 20, 2026-08-05)

Reviewing §19.9 surfaced three weaknesses. Two are code and are fixed here; the third is
experimental design and is §19.11.

### 19.10.1 Gap A — a branch with no execution evidence

`compute_metrics` **drops** a depth from `metrics` when val has no samples for it
(`train.py:324-325`). The per-depth print therefore needs an `m is None` fallback. Both smokes
(25234282, 25234340) happened to carry all three depths in val, so that fallback never ran.

It matters because train and val are different station sets: a depth with ample training data but
no val stations is exactly when the train number is most wanted, and it was the only line in the
change with zero execution evidence.

**Fix:** extract the line into `_format_depth_line(depth, train_loss, val_loss, m)`
(`train.py:407`), so the branch is reachable from a test instead of from a rare data layout.

### 19.10.2 Gap B — no scalar in the log is comparable across runs

Pre-existing, and it has already cost us. `val_loss` changes **meaning** with `per_depth_loss`:

| `per_depth_loss` | `val_loss` means |
|---|---|
| `False` | pool all (sample × depth) pairs, one Huber mean |
| `True` | Huber per depth, then average the depths |

So `baseline_huber_memmap` 0.00209 and `baseline_huber_notv_perdepth` 0.002182 are different
quantities. §19.7's "never compare val_loss across runs" is a workaround for a missing metric, not
a fix. The per-depth losses added a third definition (sample-weighted, Huber-only, boundary
excluded) — hence epoch 1's `train_loss=0.047413` against a per-depth mean of 0.02722, the 0.0202
gap being `lambda_boundary=0.1 × ~0.2`.

**Fix:** `_loss_aggregates(depth_sum, depth_cnt)` (`train.py:387`) returns two scalars, both free
from the sums already accumulated — no extra compute, no extra GPU sync:

| Key | Formula | Purpose |
|---|---|---|
| `{split}/huber_pooled` | `Σsum / Σcnt` | One Huber mean over every valid (sample, depth) pair. Definition is independent of `per_depth_loss`, `lambda_tv`, `lambda_boundary` — **this is the cross-run comparable scalar** |
| `{split}/huber_depth_mean` | unweighted mean over observed depths | Exactly the average of the printed per-depth lines, so the block reconciles on its face |

They differ under uneven coverage, and the difference is itself informative: `pooled` weights each
observation equally (dominated by 0-10 cm, which has the most stations), `depth_mean` weights each
depth equally (30-100 cm counts as much as the surface). Measured on the test fixture:
pooled 0.014915 vs depth_mean 0.015142.

`evaluate` now returns the raw `(depth_sum, depth_cnt)` rather than a pre-derived dict, so the
caller can compute both. `eval_stations.py` updated to match and now prints all three scalars.

**`val_loss` itself is unchanged** — same formula, same value, still what drives `best.pt`, early
stopping and the LR scheduler. These are metrics, not objective changes.

### 19.10.3 These cannot affect training

Verified, not assumed:

| Check | Result |
|---|---|
| `loss.requires_grad` | True (unchanged — this is what `backward()` runs on) |
| `depth_sum.requires_grad` / `.grad_fn` | False / None — the breakdown uses `pred.detach()` |
| `_loss_aggregates` return type | Python `float`, not tensor |
| Gradient w.r.t. input, with vs without `return_breakdown` | **bit-identical** (`torch.equal`) |

### 19.10.4 `test_per_depth_loss.py` — 24 CPU checks, no GPU/data/DDP needed

```bash
python test_per_depth_loss.py
```

Covers: scalar-loss identity in both loss modes; the breakdown being detached and the gradient
bit-identical; absent depth → `nan` never `0.0`; all-NaN batch not crashing; **DDP sum-reduction
reproducing a single-pass computation to 1e-6 across 4 simulated ranks with uneven depth
coverage**; both aggregates including the no-data and skip-absent-depth cases; and the
`m=None` print branch, which now demonstrably yields:

```
30-100  train_loss=0.027312  val_loss=nan  no val samples
```

Why sums and not means, restated because it is the easiest thing to get wrong: means cannot be
averaged across ranks when per-rank sample counts differ, and with sparse deep coverage they
always differ. Accumulate sums, `all_reduce(SUM)`, divide once at the end.

### 19.11 Open — the confound is NOT fixed by code

§19.5 changes architecture **and** regularisation together. No code change resolves that; it needs
a second run. With the existing baseline that gives a clean three-point design:

| Run | `use_cls_depth` | Regularisation | Isolates |
|---|---|---|---|
| `baseline_huber_notv_perdepth` (W&B `pg7mw3xb`) | off | baseline (wd 0.05, dp 0.1) | reference |
| `cls_depth_star_noreg` — **to launch** | **on** | baseline | the architecture |
| `cls_depth_star_reg` — **to launch** | **on** | wd 0.1, dp 0.2, patience 6/3 | the regularisation, given the row above |

Compare all three on `val/huber_pooled` (§19.10.2) and per-depth ubRMSE — never on `val_loss`.

---

## §19.12 Pre-launch bug hunt — 9 defects found and fixed (Session 20, 2026-08-05)

Before committing H100 hours, `train.py`, `model.py` and `dataset.py` were reviewed
adversarially. Nine real defects surfaced. Two were introduced today; seven were pre-existing
and had been shipping in every run to date. Runs 25234370 and 25234753 were cancelled during
SHM preload (no checkpoint written, no compute wasted) to fix them first.

Summary, worst first:

| # | Defect | Sev | Origin | Where |
|---|---|---|---|---|
| 1 | `nan` silently poisoned the per-depth diagnostic | High | today | `model.py:783` |
| 2 | Every resume reset LR, discarding all scheduler decay | High | pre-existing | `train.py:943` |
| 3 | Checkpoints written non-atomically over the live 600 MB file | High | pre-existing | `train.py:265` |
| 4 | 10 decoder BatchNorm scales were being weight-decayed | Med | pre-existing | `train.py:748` |
| 5 | `--weight-decay` / `--lr-patience` silently reverted on resume | Med | pre-existing | `train.py:942` |
| 6 | Optimizer group-size mismatch → bare `ValueError` crash-loop | Med | today (from #4) | `train.py:942` |
| 7 | Mid-epoch resume reverted the LR | Med | pre-existing | `train.py:1007` |
| 8 | Depth-token cosine measured the wrong tensor | Med | today | `train.py:1194` |
| 9 | `endswith(".bias")` dropped 6 `in_proj_bias` tensors | Med | today (from #4) | `train.py:449` |

### 19.12.1 `nan` poisoned the per-depth diagnostic

```python
depth_sum = (elem * valid).sum(0)     # WRONG: nan * False is nan, not 0
```

`pred` is taken for **all** depths (`sm_map[:, :, row, col]`) while the scalar loss only ever
sees `pred[mask]`. So a non-finite prediction at a depth with **no label** cannot affect
training — but it made `depth_sum` `nan`, survived `all_reduce(SUM)` to all four ranks, and
turned `train/{depth}/loss`, `huber_pooled` and `huber_depth_mean` into `nan` while
`train_loss` still printed a healthy number. That is exactly the signal §19.1 added the
breakdown to obtain.

Fixed with `torch.where(valid, elem, elem.new_zeros(()))`. Guarded by test §4b.

### 19.12.2 Every resume reset the learning rate

The single most likely way this run could have been silently corrupted.

```python
optimizer.load_state_dict(ckpt["optimizer"])
for pg in optimizer.param_groups:      # ran unconditionally
    pg["lr"] = CONFIG["lr"]
```

`ReduceLROnPlateau.load_state_dict` is a bare `__dict__.update` — it never writes back to
`param_groups`, and `_reduce_lr` reads `param_group["lr"]` live. So the decayed LR lives only
in the optimizer, and this loop overwrote it.

`slurm/train.sh` sets `--requeue` and `--time=120:00:00`. Any preemption or requeue after the
first LR drop jumped LR from e.g. 5e-5 back to 2e-4, while the restored
`scheduler.best`/`num_bad_epochs` still believed the decay had happened. No crash, no log
line — the run simply re-diverges, days in.

Fixed: LR is overridden **only** when `--lr` is passed explicitly. Same treatment for
`--weight-decay` (#5) and `--lr-patience` (#5), which `load_state_dict` was also silently
reverting to the checkpoint's values while `CONFIG`, W&B and the startup echo all still
reported the flags you passed. A `[resume]` line now prints the effective lr/wd/patience.

### 19.12.3 Non-atomic checkpoint writes

`_fsync_save` called `torch.save` straight over the live `last.pt`/`mid_epoch.pt` — 600 MB on
GPFS. SLURM sends SIGTERM then SIGKILL after `KillWait` (30 s default; `train.sh` sets no
`--signal=B:TERM@N`). A kill part-way through leaves a truncated checkpoint that `torch.load`
rejects **on all four ranks**, so the job crash-loops on requeue. Worse, a truncated `last.pt`
plus a `best.pt` written from the same state moments later can lose a multi-day run outright.

Fixed: write to `.tmp`, fsync, `os.replace` (atomic within a filesystem), then fsync the
directory so the rename itself is durable. A reader now sees either the whole old file or the
whole new one.

### 19.12.4 Weight decay was hitting BatchNorm scales

The optimizer split was name-based:

```python
decay_params = [p for n, p in model.named_parameters()
                if p.requires_grad and "bias" not in n and "norm" not in n.lower()]
```

Every decoder BatchNorm sits inside an `nn.Sequential`, so PyTorch names it positionally —
`decoder.conv1.net.1.weight` — with no `norm` substring. The filter missed all ten. Their
*biases* were excluded (the name contains `bias`), which is what made it hard to spot.

Only 3,424 scalars of 50 M, but each BatchNorm γ is a **multiplicative gate on an entire
decoder feature map**: decaying it attenuates signal rather than constraining capacity. And
§19.5 had just doubled `weight_decay` 0.05 → 0.1, doubling the effect — inside the very run
meant to measure what regularisation does.

Fixed: `_split_param_groups()` selects by module **type**, matching on `id(p)` so DDP's
`module.` prefix is irrelevant. Newly protected: 8 decoder + 2 soil-encoder BatchNorm scales,
plus `depth_tokens` (a learned *input* like a positional embedding — decaying it pulls the
three per-depth queries back toward the symmetric state §18.3 exists to break).

Group sizes 81/83 → 70/94.

**And a regression caught while fixing it (#9):** the first attempt used `endswith(".bias")`,
which silently dropped protection from six `self_attn.in_proj_bias` tensors — MultiheadAttention's
packed QKV bias has no dot before `bias`. The exact mirror of the original defect. Now
`endswith("bias")`, with a test naming `in_proj_bias` explicitly and another asserting that
nothing protected by the old filter lost protection.

### 19.12.5 The mechanism diagnostic watched the wrong tensor

`diag/depth_token_cos_*` measured `self.depth_tokens` — the learned **input** parameters. The
failure mode it exists to catch is `use_cls_depth` being inert, and that manifests in the
transformer **outputs**: after six bidirectional layers all three CLS slots can carry identical
content while the input parameters remain near-orthogonal (cos ≈ 0, "looks healthy").

Without this fix the run could have gone four days with the diagnostic reading fine and the
per-depth losses locked together, leaving no way to tell whether the CLS mechanism or the
decoder was at fault.

Fixed: `SoilMoistureModel.forward` stashes the batch-mean `depth_ctx` (detached, ~9 KB) and
training now logs `diag/depth_ctx_cos_*` alongside the parameter cosine. **`depth_ctx_cos` is
the one to watch.**

### 19.12.6 Verified clean

`model.py` came back with no blocking defect, verified by execution rather than reading:
`use_cls_depth` index offsets (`depth_ctx` lands exactly on the prepended slots; `sp_start`
correct; 199 positions excluded from the context pool); star-residual gradient routing
(`∂out[:,d]/∂heads[k]` nonzero for exactly `k ∈ {0,d}`); channel order matching `SM_DEPTHS`;
FiLM identity at init; `nan_to_num` giving sums identical to masking first; scalar loss
byte-identical; `F.huber_loss` on autocast's fp32 list so no bf16 accumulation; and the
`use_cls_depth=False` path unchanged, so old checkpoints still load.

`train.py` DDP collective symmetry was traced and is clean — every collective is reached by all
ranks, none sits inside an `is_main` block, and the two new `all_reduce`s are correctly placed
above the rank-0 guard. No unbound-variable path on either resume route.

### 19.12.6b `dataset.py` — clean for this run, two guards added

`dataset.py` was reviewed against the actual data on disk, not just read. Everything that could
corrupt labels is **provably** clean for this run:

| Hazard | Finding |
|---|---|
| Depth→channel mis-mapping | The only depth strings across all 993 stores are exactly `0-10` (889), `10-30` (686), `30-100` (554). No whitespace or format variants, so `depths.index()` cannot collide; a different order is handled by construction, a subset just leaves NaN |
| QC realignment (`dataset.py:184-186`) | 513/890 stations hit the trailing-slice realign. Proved correct: the pre-2016 trim archives contain no `labels_qc` (it was written later, from the untrimmed series), `len(labels_dates) == qc_len - sm_len` **exactly** for all 513, and all start at 20160101 — so the surplus is entirely leading |
| Label values | At `qc == 0` — the only slots used — **0 values > 1.0** out of 5,192,395, 8 negatives, 0 NaNs |
| Date alignment | All-integer YYYYMMDD, no timezones. ERA5 right-alignment verified across all 661 in-scope stations: **0 interior gaps, and the window's last ERA5 day equals the target day for all 1,598,711 observed label days** |
| memmap staleness | All 8,667 `.npy` slots: JSON shape matches zarr layer length and byte size equals `prod(shape)*2` exactly, 0 mismatches |
| Missing → 0.0 confusion | Missingness is always out-of-band (`doys == 0` / `valid=False`), never a sentinel value |
| DDP worker safety | Read-only memmaps use position-independent page faults (no shared fd offset); zarr opens a fresh fd per chunk, so none crosses the fork |

Two guards added anyway, both against burning 120 h × 4 H100s:

- `_load_l12_shm` checked only `bin_path.exists()` then read `meta.json` unconditionally
  (`dataset.py:699`). The preloader creates the `.bin` *before* the `.meta.json`, and its own
  resume check is `if bin_path.exists()`, so a rank-0 death between those statements leaves a
  bin with no meta that never gets repaired → `FileNotFoundError` on all four ranks. Now checks
  both.
- `while not _shm_done.exists(): sleep(2)` (`train.py:798`) had **no timeout**. If rank 0 died
  during preload, ranks 1-3 would spin for the full 120 h walltime holding four H100s. Now
  bounded at 3 h (worst observed preload: 1901 s) with an explicit failure message.

Not applicable to this run, recorded for when scope changes: partial `.npy` coverage is silent
unless a station has *no* npy at all (103/102/78 stations lack s2/s1_asc/s1_desc npy, but **zero**
are in `sm_only` train+val); and with `--max-stations` the preloader and the train dataset select
different station subsets, so smoke runs partly bypass the shared-memory design.

### 19.12.7 Known and accepted

- `evaluate` calls `_compute_loss` **without** `lambda_boundary`, so `val_loss` excludes the
  boundary penalty that `train_loss` includes — model selection optimises a slightly different
  objective than training. Left unchanged: fixing it would change `val_loss`'s definition and
  break comparability with `pg7mw3xb`, and `huber_pooled` (§19.10.2) is already boundary-free
  and consistent across train and val. Flagged for a deliberate decision, not changed silently.
- After a val-crash resume, one epoch's per-depth **train** losses log as `nan` (the old
  checkpoint stores `train_loss` but not the depth vectors). Cosmetic; one row in W&B.
- At step 0, `depth_film[1:]` receives exactly zero gradient (zero-init offset heads × zero-init
  FiLM projection). Self-resolving after one AdamW step, but the deep-depth FiLM starts one step
  behind — worth knowing if the first ~100 batches look flat.

---

## §20 The absolute-level problem — diagnostic before lever (Session 20, 2026-08-05)

Run `cls_depth_star_reg` (job 25235976) answered the question §19 was built to ask, and the
answer was not the one either branch of §19.8 anticipated. This section records the finding,
the diagnostic that must run before any further architecture work, and the two mutually
exclusive plans that follow from its result.

**Nothing here requires stopping run 25235976.** The diagnostic is CPU-only and reads
artefacts that already exist.

### 20.1 The finding: dynamics are learned, absolute level is not

Best epoch so far is e3 (`val_loss=0.002312`). Per-depth, on the 74 held-out val stations:

| Depth | MSE | RMSE (√MSE) | ubRMSE | bias |
|---|---|---|---|---|
| 0-10 | 0.0066 | 0.0812 | 0.0546 | +0.0047 |
| 10-30 | 0.0065 | 0.0806 | 0.0513 | +0.0154 |
| 30-100 | 0.0111 | 0.1054 | 0.0564 | +0.0183 |

`compute_metrics` (`train.py:324`) removes **each station's own temporal mean** from both
prediction and observation before computing ubRMSE. Within a station the anomaly error and
the mean-offset error are orthogonal, so

```
MSE  ≈  ubRMSE²  +  (per-station offset)²
```

Solving for the offset:

| Depth | ubRMSE (dynamics) | implied per-station offset | offset / ubRMSE |
|---|---|---|---|
| 0-10 | 0.0546 | **0.0602** | 1.10 |
| 10-30 | 0.0513 | **0.0622** | 1.21 |
| 30-100 | 0.0564 | **0.0890** | 1.58 |

**The dominant error is getting each station's absolute level wrong, not its dynamics — and
it worsens monotonically with depth.**

The trap: global `bias` at 0-10 is +0.0047, which reads as near-perfect calibration. It is
not. It means the per-station offsets are large but **cancel** — some sites too wet, some too
dry. `val_station_metrics.csv` confirms this directly; the first val station logged
(`ISMN_Berlin_PSA1BerlinerStr`, 0-10) carries `bias = −0.0389` against a global +0.0047.
**Never read global bias as evidence of per-site calibration. Take the RMS of the per-station
`bias` column instead.**

### 20.1b Measured numbers (run 25235976, epochs 1-5)

Per-station offsets, measured directly from `val_station_metrics.csv` at e5 — confirms §20.1's
algebra (0.060 / 0.061 / 0.091):

| Depth | RMS bias | mean bias | range | abs>0.05 | n stations |
|---|---|---|---|---|---|
| 0-10 | **0.0618** | −0.0019 | −0.156 … +0.127 | 29/74 | 74 |
| 10-30 | **0.0611** | +0.0053 | −0.216 … +0.117 | 17/51 | 51 |
| 30-100 | **0.0875** | +0.0079 | −0.219 … +0.244 | 21/43 | 43 |

ubRMSE by epoch:

| Depth | e1 | e2 | e3 | e4 | e5 |
|---|---|---|---|---|---|
| 0-10 | 0.0567 | 0.0552 | 0.0546 | 0.0556 | 0.0542 |
| 10-30 | 0.0517 | 0.0513 | 0.0513 | 0.0511 | 0.0509 |
| 30-100 | 0.0553 | 0.0548 | 0.0564 | 0.0553 | 0.0569 |

bias by epoch:

| Depth | e1 | e2 | e3 | e4 | e5 |
|---|---|---|---|---|---|
| 0-10 | 0.0142 | 0.0158 | 0.0047 | 0.0029 | 0.0027 |
| 10-30 | 0.0131 | 0.0186 | 0.0154 | 0.0151 | 0.0090 |
| 30-100 | 0.0194 | 0.0209 | 0.0183 | 0.0126 | 0.0165 |

- Mean bias ≈ 0 while stations reach ±0.15 → offsets cancel; global bias is meaningless per-site.
- Offsets are large at **all** depths — never was a deep-only problem, just worst at depth.
- ubRMSE spans only 0.006 across depths → deep **dynamics** are fine; only level and the
  generalisation gap (12× vs 5×) fail.
- 30-100 ubRMSE oscillates 0.0548-0.0569 with no trend, best value at e2, while its train loss
  falls 6× → flat anomaly skill, not degrading.
- 30-100 has 43 val stations vs 74 (≥95% coverage filter) → the statable reason if scoping down.
- 0.04 ubRMSE is unreachable here: 0.0567→0.0542 over 5 epochs, decelerating.
- Even at ubRMSE 0.040 with offset 0.062, RMSE ≈ 0.074 → a 0.04 gate certifies **anomaly skill
  only**, not soil moisture prediction. Say so explicitly if used as a criterion.

### 20.2 What the per-depth train/val gap rules out

The per-depth breakdown §19 added exists to separate a capacity/information ceiling from a
generalisation failure. At e3:

| Depth | train | val | gap |
|---|---|---|---|
| 0-10 | 0.000604 | 0.002124 | 3.5× |
| 10-30 | 0.000440 | 0.002105 | 4.8× |
| 30-100 | **0.000388** | **0.002902** | **7.5×** |

**30-100 cm fits the training data BEST and generalises WORST.** §19.7 predicted that a
capacity or information ceiling would show as high-and-flat train loss at 30-100. The opposite
happened: it is the easiest depth to fit. Physically consistent — deep soil moisture is
temporally smooth, so per-station it is nearly a slowly-varying constant, trivial to memorise
on seen stations and worthless on unseen ones.

Three things are therefore ruled out, and should not be attempted:

1. **The cascade** (§19.8) — it adds deep-depth capacity. Capacity is not the constraint;
   30-100 already has the lowest train loss of the three depths.
2. **More regularisation strength** — §19.5 doubled `weight_decay` (0.05→0.1) and
   `drop_path_rate` (0.1→0.2) and the memorisation curve was unchanged versus S16
   (e2 train 0.00067 both runs). This is not a weight-magnitude problem; the network is
   using legitimately-supplied inputs (a unique 74×74 soil patch, DEM, LULC) as a station
   fingerprint. Weight decay cannot suppress that.
3. **More depth-specific parameters** — the S16 lever. `use_cls_depth` raised depth-specific
   params 195 → 297,795 and did not beat the S16 baseline at the same epoch
   (30-100 ubRMSE 0.0564 vs 0.0559; bias +0.0183 vs +0.0146).

### 20.3 Why the model cannot be its own diagnostic

When the network gets absolute level wrong by 0.060, two causes are indistinguishable from
its output alone:

- **(A)** the static data does not determine site moisture level — two sites with identical
  texture and terrain genuinely sit at different levels (drainage, water table, macroporosity,
  sensor calibration), or
- **(B)** the static data does determine it and the network is failing to exploit it.

These demand opposite responses, and each wrong guess costs a ~5-day run. The network cannot
separate them because its failure is confounded with learning dynamics, fusing five
modalities, and overfitting simultaneously.

### 20.4 The diagnostic: is station-mean SM predictable from static features?

Strip the problem to one number per station and ask it in isolation.

**Table.** One row per station, 577 train + 74 val (the stations the run actually used, not
the CSV's 680/90 — token availability differs, and the comparison must be exact).

**Run it as a NESTED LADDER, not one lumped model.** Each block's *increment* attributes the
information, which turns the diagnostic from a yes/no into "here is which input matters".

| Block | Features | What its increment tells you |
|---|---|---|
| B0 | global training mean | null baseline; defines "no skill" |
| B1 | soil (21 channel means over the 74×74 patch + their 21 spatial stds) + DEM mean/std/slope + LULC class fractions | do static site properties determine level? |
| B2 | **+ all 19 `ERA5_VARS` temporal means** over the station's whole record | does the forcing the model *already has* determine it? |
| B3 | + derived water-balance terms: VPD from `t2m`/`d2m`, a temperature-based PET (Hargreaves, from `t2m_min`/`t2m_max`), aridity ratio, seasonal amplitude | could the network have derived these itself? |
| B4 | + lat/lon | how much is merely geographic proximity? |

**Use the 19 ERA5 means, not hand-picked climate summaries.** Selecting features by hand tests
the feature engineering, not the information content. Feeding all 19 makes the probe a fair
upper bound on what is linearly extractable **from what the network actually sees**. Average
over each station's whole record, not the 365-day sample window — the target is its long-term
climate, not one sample's forcing.

**Decisive read:** if **B2 ≫ B1**, the level information is carried by forcing the model
already receives, which makes the failure unambiguously a model failure and hands the question
straight to §20.12. If **B3 ≫ B2**, the network could have derived the term but did not, which
argues for supplying it explicitly rather than trusting a 6-layer transformer to learn PET from
raw fields.

**Deliberately out of scope (decided 2026-08-05):** ERA5-Land `volumetric_soil_water_layer_*`
and `total_evaporation` are **not** downloaded — `download_era5land.py` fetches only
t2m/d2m/skt/u10/v10/sp × {mean,min,max} + tp_sum, and the zarr stores exactly those 19
(`era5/values (N, 19)`). They were considered as a "is the information obtainable at all"
ceiling and an ERA5-Land baseline, and **rejected** — not worth a new acquisition for this
diagnostic. Consequence: the ladder measures what is extractable from **current inputs only**,
so a negative result means "not learnable from what we have", NOT "not learnable in principle".
State it that way in any write-up.

**Also absent from the zarr, though it was downloaded:** `ssrd`/`strd` (solar and thermal
radiation × mean/min/max). `download_era5land.py` writes them into the per-station NetCDFs but
they never reached the zarr, so the model has never seen them. Downwelling radiation is the
primary driver of evaporative demand. If B3's temperature-based PET carries signal, re-ingesting
these six from the existing NetCDFs is the cheap follow-up — a re-ingest, not a download.

**Target.** Each station's mean observed SM, **computed separately per depth** (three
independent regressions). 30-100 cm is where the problem is worst and must be reported
separately.

**Note a depth mismatch to handle explicitly:** the soil product's depths are 0-30 / 30-60 /
60-100 cm (`text/architecture.md:255-263`) while the SM labels are 0-10 / 10-30 / 30-100 cm.
Do not silently pair them 1:1. Feed all 21 channels to every depth's regression and let the
model weight them.

**Two methodological requirements:**

1. **Standardise features.** Ridge's L2 penalty is scale-sensitive; clay in % and elevation
   in metres would otherwise be penalised unequally.
2. **Tune α by GroupKFold on `location_group_id`, within training stations only.** Ordinary
   K-fold would place neighbouring stations in both fold-train and fold-test and yield an
   optimistic α. The main split already enforces **zero** train/val group overlap (verified:
   646 train groups, 69 val groups, 0 shared) and the tuning must match that standard.

**Run twice — with and without lat/lon.** With coordinates the model can spatially
interpolate, which is legitimate but answers "are nearby stations similar", not "does soil
physics determine level". Only the coordinate-free version speaks to transfer into a new
region. Report both.

### 20.5 Why ridge AND gradient boosting, specifically

**Ridge is chosen because it is weak, not because it is good.** With 577 rows, ~50 features
and a real L2 penalty, it is incapable of memorising. Therefore a positive result **cannot be
an artefact** — that is the property required of a measuring instrument. Its coefficients are
also directly readable, and identify which soil properties drive level, which is exactly the
input the §20.7 reparameterisation needs.

**GBM is the nonlinearity control.** Water retention is not linear in texture — moisture rises
with clay, saturates, and interacts with organic carbon. A purely linear probe could report
"no signal" when the signal is real but curved. Shallow trees (depth ≈3) with early stopping
under the same GroupKFold keep it honest.

Neither alone is sufficient: ridge alone confuses *nonlinear* with *absent*; GBM alone leaves
open whether it memorised.

| Ridge | GBM | Interpretation |
|---|---|---|
| works | — | signal exists, roughly linear — cheap to exploit |
| fails | works | signal exists but nonlinear — a small MLP or pedotransfer form is right |
| fails | fails | no signal in these features — §20.6 |

### 20.6 Decision procedure — three numbers

1. **Null baseline** — predict the global training mean for every station. Its RMSE is the
   between-station spread of mean SM. This defines "no skill" and must be computed first.
2. **Ridge / GBM** — RMSE on the 74 val stations, per depth.
3. **The network's error at the same task** — RMS of the per-station `bias` column in
   `checkpoints/.../cls_depth_star_reg/val_station_metrics.csv`, filtered to the best epoch.
   That column IS each station's offset, so this is directly comparable and station-equally
   weighted. Expect ≈0.060 / 0.062 / 0.089 per §20.1.

| Outcome | Conclusion | Go to |
|---|---|---|
| ridge ≈ null | static features do not determine level; the task is unidentifiable from these inputs | §20.7 |
| ridge ≪ network | information is present and the network is discarding it | §20.8 |
| ridge ≈ network | the network already extracts everything available; remainder is irreducible | §20.7 |

All three outcomes are actionable and point somewhere different. That is what makes the
diagnostic worth running before choosing a lever.

### 20.7 BRANCH A — if there is no information

**First, one more cheap check before accepting it.** ISMN aggregates many networks using
different sensor types (capacitance, TDR, neutron probe), each with its own calibration. Two
identical soils instrumented differently read different absolute values. Group the per-station
`bias` by the `network` column of `station_splits.csv`:

- **Offsets cluster strongly by network** → a substantial part of the 0.060 is *instrumentation
  bias in the labels*, not model error. No model can predict it, because it is not a property
  of the soil. This is itself a reportable finding and fully justifies anomaly-based evaluation.
- **Offsets unstructured within networks** → genuine unmeasured site variability (drainage,
  water table, macroporosity).

Either way, three moves:

**A1. Change the target.** Train on per-station standardised anomaly
`z = (θ − μ_station) / σ_station`. Today a large share of the loss is spent on a component
that provably cannot be predicted, which actively competes with dynamics learning. Removing it
should **improve** ubRMSE rather than merely relabel the problem. This is a testable
prediction, not a cosmetic change.

**A2. Change the claim.** Report anomaly skill (ubRMSE, correlation) as the primary result.
This is standard practice, not a retreat — SMAP/SMOS validation leans on anomaly metrics and
triple-collocation methods exist precisely because absolute cross-sensor agreement is known to
be unattainable. "Predicts SM dynamics at unseen sites with ubRMSE ≈0.046 (median station)" is
a clean, defensible claim.

**A3. Offer level where obtainable.** If the application permits even a handful of local
observations, a single offset correction removes essentially all of this error. Frame the
product as transferable dynamics plus a one-sample site calibration.

### 20.8 BRANCH B — if the information exists

**Do NOT rebuild the model first. Test post-hoc on the existing checkpoint — it costs nothing
and sizes the prize.**

**B1. Post-hoc offset correction (no GPU, no retraining).** Take the best checkpoint's val
predictions, use the fitted ridge to predict each val station's mean from its static features,
and shift that station's predictions to match. Recompute MSE, ubRMSE, bias.

- ubRMSE is unchanged by construction (a constant shift cancels in anomalies) — this is a
  correctness check on the implementation, not a result.
- MSE should fall by roughly the recovered offset variance. If 30-100's MSE drops from 0.0111
  toward ~0.004, the fix is demonstrated using an artefact that already exists.
- **If MSE barely moves, stop.** The ridge's apparent skill did not transfer to the actual
  predictions, and no rebuild is justified. This gate exists to prevent a 5-day run on a
  result that looked good only in aggregate.

**B2. Two-headed model** (only if B1 succeeds):

- **Head A** predicts `z(t)`, the normalised dynamics — what the network already does well.
- **Head B** predicts `μ_station` (and `σ_station`) from **static features only**. Small,
  low-capacity, deliberately incapable of memorising. Warm-start from the fitted ridge.
- Reconstruct `θ̂(t) = μ̂ + σ̂ · ẑ(t)`. Keep the Huber loss on `θ̂` so `val_loss` stays
  comparable with every prior run (§19.10.2 discipline).

**B3. The critical addition is an auxiliary loss directly on `μ̂`** against each training
station's observed mean. This is what is missing today: the level currently receives only an
implicit, diluted gradient tangled with dynamics. A dedicated target plus a dedicated
low-capacity pathway converts it into a supervised problem the model can actually solve.

**B4. Bundle static-feature dropout** — randomly zero the soil patch, DEM and LULC during
training. This prevents Head A from re-learning the station fingerprint and forces level
information to flow through Head B, where it belongs. Cheap, few lines, and attacks the
memorisation mechanism identified in §20.2 directly.

### 20.9 Implementation spec for the diagnostic script

```
scripts: station_mean_probe.py          (new)
inputs:  csvs/station_splits.csv        (splits, network, location_group_id)
         zarr soil/ patches             (21, 74, 74) per station
         label NetCDFs                  (per-station mean SM per depth)
         checkpoints/.../cls_depth_star_reg/val_station_metrics.csv  (network comparison)
outputs: csvs/station_mean_probe.json   (all scores)
         plots/station_mean_probe.png   (pred vs true station mean, per depth)
runtime: < 1 min, CPU only, Pool(64) per feedback_multiprocessing
```

Must report, per depth: null-baseline RMSE, ridge RMSE (with and without coordinates), GBM
RMSE, the network's RMS per-station bias, the top-10 ridge coefficients by magnitude, and the
per-network mean/std of station bias for the §20.7 clustering check.

**No silent caps** — if any station is dropped for missing features, log the count and the
reason. A quietly reduced station set would make every number optimistic.

### 20.10 Caveats to carry

- Each station's mean SM is computed over its **own record period**, and those differ. Some
  between-station variance is therefore climate-of-the-period rather than site character. This
  adds noise to the target and makes the test **conservative**: a positive result stays
  trustworthy, but a marginal negative deserves a re-run restricted to a common period.
- The soil composite is single-epoch 2020-2022 while labels span 2016-2025. Static soil
  properties are genuinely near-static, so this is acceptable — but it means land-use change
  within the record is invisible to the probe.
- 19 stations have fully-NaN soil patches and are already excluded via `soil_patch_ok`; confirm
  none re-enter through the probe's own feature construction.

### 20.11 Status

Diagnostic NOT yet run. Run 25235976 continues — e4 turned (`val_loss` 0.002312 → 0.002336,
first non-improvement, `no_improve_count` 1 of 6) with best still at e3. The run should be
left to finish or early-stop on its own; its `val_station_metrics.csv` is an input to §20.6
and improves with every epoch logged.

### §20.12 Localising the failure — decoder or temporal transformer?

§20 asks *whether* the level information exists. This asks *where it is lost*. The two are
complementary and share a target, a split and a metric, so their numbers are directly
comparable.

**Method: measure the same quantity at three points along the pipeline.**

```
raw static features   →   context vector   →   final prediction
      (input)             (decoder input)          (output)
```

| Point | How | Answers |
|---|---|---|
| Input | §20.4 ridge on soil / terrain / climate → station mean | is the information present at all? |
| Decoder boundary | ridge on the 768-d context vector → station mean | did the transformer preserve it? |
| Output | RMS of per-station `bias` in `val_station_metrics.csv` | did the decoder use it? |

Whichever **adjacent pair** shows the drop localises the failure:

| input | context | output | Verdict |
|---|---|---|---|
| good | good | bad | transformer encoded it; **the DECODER is discarding it** |
| good | bad | bad | **the TRANSFORMER destroyed it**; the decoder never had a chance |
| bad | — | bad | no information — §20.7 Branch A; neither component is at fault |

All three probes use the frozen best checkpoint. **No retraining, no GPU-days.**

#### 20.12.1 FiLM check — CLOSED, premise was wrong (amended 2026-08-05)

**Both halves of the original rule were false. Do not quote the superseded version.**

| Claim as published | Status |
|---|---|
| "FiLM is the only route by which transformer context modulates the decoder" | **FALSE** |
| "Still ≈0 → the decoder is ignoring the context vector" | **FALSE**, follows from the above |

- The 196 target-day spatial tokens are taken from the transformer **output** (`model.py:732-733`)
  and go straight into `decoder.bottle_proj` (Conv2d 768→512, `model.py:211/252`). Transformer
  context therefore reaches the decoder **spatially, unconditionally and FiLM-free**.
- FiLM modulates only the **frozen TerraMind L3/L6/L9 skip** branch. It is a side-channel.
- So a zero FiLM weight would have meant only "the pooled summary vector adds no sample-dependent
  modulation to the skips" — never "the decoder ignores context".

**Measured anyway from `cls_depth_star_reg/best.pt` (e6) — FiLM has trained:**

| layer | shape | ‖W‖_F | element RMS |
|---|---|---|---|
| film_s9 | [1024,768] | 11.28 | 1.27e-2 |
| film_s6 | [512,768] | 6.22 | 9.9e-3 |
| film_s3 | [256,768] | 4.68 | 1.06e-2 |
| depth_film.0 | [128,768] | 1.43 | **4.6e-3** |
| depth_film.1 | [128,768] | 3.45 | 1.10e-2 |
| depth_film.2 | [128,768] | 3.72 | 1.19e-2 |

- Weights moved off zero → gradient signal **did** reach FiLM. The zero-init-trap hypothesis is dead.
- But element RMS ≈1.3e-2 vs AdamW's decay equilibrium `1/λ = 10` → **~790× below it**, and ~0.5%
  of the ballistic bound (`Σlr ≈ 2e-4 × 12,300 steps`) → the gradient has been near-sign-random.
- Weight decay is **not** the explanation: the undecayed biases (`_split_param_groups` excludes
  `endswith("bias")`) also barely moved from (1, 0).
- **Compare element RMS, never ‖W‖_F** — the Frobenius ordering is an artefact of parameter count.
- Real finding: `depth_film.0` is 2.5× smaller than `.1`/`.2`, and depth 0 is the star-residual
  base that predicts absolute SM rather than an offset.

**Consequence:** FiLM is demoted. Do not spend a GPU job on it. Answering decoder-vs-transformer
needs a **bottleneck** ablation (freeze `decoder.bottle_proj`'s input to a dataset mean), not a
FiLM one — and that is only worth running if §20.14 shows the information exists at all.

#### 20.12.2 The context probe

One forward pass over the 74 val stations with the frozen checkpoint, caching the context
vector, then ridge from context → station mean. **Use §20.4's methodology exactly** —
standardised features, GroupKFold on `location_group_id`, same 577/74 split — so the number is
commensurable with the input-side ridge. A drop between the two localises the loss to the
encoder/transformer.

#### 20.12.3 The bypass head — proof rather than inference

Cache context vectors for train+val once (a single forward pass), then train a tiny MLP
`context → 3 scalars` with everything else frozen. Minutes on cached features.

If the bypass head's RMS per-station bias is materially lower than the full decoder's, the
decoder is **provably** the bottleneck — and the run also yields a working alternative readout
for free.

#### 20.12.4 Why this matters beyond the level problem

The Tier-1 verdict (2026-07-01) already traced over-smooth SM maps to the decoder side —
bilinear upsampling plus single-pixel loss — rather than to attention. If the decoder is also
where absolute level dies, that is **the same root cause surfacing twice**, and it materially
raises the priority of dense spatial supervision (§16.4, currently deferred): one fix would
then address both the smoothness and the level failure.

Conversely, if FiLM has trained and the context probe is strong, the decoder is exonerated
here and §16.4 stays a separate, independently-motivated piece of work.

#### 20.12.5 Modality dropout — what is actually withheld during training

**CORRECTION.** An earlier version of this section stated modality dropout was not
implemented. That was wrong. It **is** implemented — entirely in `dataset.py`, never in
`model.py`. Grepping `model.py` alone finds only `DropPath` and ordinary `nn.Dropout` and
misses all of it. Look in `dataset.py` for anything of this kind.

| Modality | Treatment during training | Location |
|---|---|---|
| ERA5 | 15% of timesteps zeroed and marked as padding | `dataset.py:1054-1058` |
| SIF | dropped entirely, **p = 0.5** per sample | `dataset.py:1064-1065` |
| TWSA | dropped entirely, **p = 0.5** per sample | `dataset.py:1071-1072` |
| Satellite spatial tokens | 50% random token dropout before pyramid pooling | `dataset.py:306, 397` |
| **Soil / DEM / LULC** | **never dropped — always present** | — |

None is applied at val/test time.

**Soil is therefore ruled out as a direct cause**, which was this check's purpose: the input
carrying site-level information is always available. That conclusion survives the correction.

**But two consequences the original draft missed:**

1. **TWSA is withheld half the time, and it is the one input about water *storage* rather than
   forcing.** GRACE terrestrial water storage anomaly is conceptually the closest thing in the
   input set to "how much water is in this column". Dropping it at p=0.5 explicitly trains the
   model to discount it. Its ~300 km footprint and monthly cadence limit how well it can
   discriminate *between* nearby stations — many val stations share a GRACE cell — so this is a
   hypothesis to test, not a conclusion. If the §20.4 ladder shows a TWSA/SIF block carrying
   real level information, lowering or removing TWSA's dropout is a one-line experiment, far
   cheaper than anything in §20.8.
2. **ERA5's 15% masking is immaterial to the level problem.** It drops individual days at
   random from a 365-day window; a temporal *mean* over that window is barely perturbed. It
   targets temporal generalisation, not the station-level signal.

**Method note worth generalising:** the original error came from grepping one file and
concluding absence. For any "is X implemented" question spanning the data path, check
`dataset.py`, `model.py` and `train.py` before recording a negative.

### §20.13 Plan once ubRMSE 0.04 is off the table (2026-08-05)

Run 25235976 at e6: val 0.002230 (best), ubRMSE 0.0543 / 0.0504 / 0.0552, still improving but
dynamics near-flat — 6 epochs bought 4% ubRMSE at the surface, ~0% at depth, while val_loss
fell 11%. **Level is what's moving, not dynamics.** 0.04 needs ~50 more epochs at a
decelerating rate → not reachable by this run.

**Order of work:**

1. **Diagnostics first** (§20.12.1 FiLM norm, §20.4 ladder) — minutes each. If FiLM ≈0, ubRMSE
   is capped by wiring not physics, and everything below is premature.
2. **Per-station standardised anomaly target** (§20.7-A1) — the single change most likely to
   move ubRMSE. ~Half the loss is currently spent on the station offset, unlearnable at unseen
   sites; removing it sends all capacity to dynamics. Testable prediction: ubRMSE **improves**,
   not merely gets relabelled.
3. **Dense spatial supervision** (§16.4) — largest untapped structural lever. Only pixel
   (112,112) of 224² is supervised; the Tier-1 verdict already traced over-smooth maps to this,
   and it targets the decoder that §20.12 may be about to implicate.

**Reframe the bar.** 0.04 is SMAP's mission requirement against *core validation sites*. Broad
ISMN point-scale validation of SMAP/SMOS typically lands nearer 0.05-0.06 — **confirm against
current literature before quoting** — in which case ubRMSE ≈0.054 at stations the model has
never seen is competitive, not a failure. Claim becomes: transferable SM *dynamics* at unseen
sites, ubRMSE ≈0.05, absolute level explicitly out of scope with §20 as the evidence.

**Ruled out with evidence — do not spend a run on these:**

| Lever | Why ruled out |
|---|---|
| More regularisation | §19.5 doubled wd and drop_path; memorisation curve unchanged vs S16 |
| More depth capacity / cascade | 30-100 has the LOWEST train loss of three depths (§20.2) |
| Raising Huber `delta` | train residuals ≈0.020-0.027, already inside delta=0.05 → delta is inactive where gradients are produced; and it targets offset, which ubRMSE removes by construction |

---

## §20.14 Station-mean probe — the gate before any model work (2026-08-05)

**Runs before §20.12's decoder-vs-transformer question.** If level is not predictable from
available inputs, that question is moot and any GPU job is wasted. CPU-only, <1 min, data
already on disk.

### 20.14.1 Why

- Offsets (0.0618 / 0.0611 / 0.0875) exceed ubRMSE (0.0543 / 0.0504 / 0.0552) at every depth.
- Two causes are indistinguishable from model output — (a) inputs don't determine level,
  (b) they do and the network discards it — and they demand opposite responses.
- Each wrong guess costs a ~5-day run. → strip to **one number per station** and ask in isolation.

### 20.14.2 Design

- **Target:** per station per depth, mean observed SM over `qc == 0` only. Three independent
  regressions.
- **Rows:** the stations the run actually used — replicate `dataset.py:813-835`
  (`category_filter=["sm_only"]`, `split ∈ {train,val}`, `soil_patch_ok`, zarr opens). ≈577/74.
  **Assert** the val list matches `val_station_metrics.csv` or the comparison is invalid.

| Block | Features | Mechanism it isolates |
|---|---|---|
| B0 | global train mean | null |
| B1 | 21 soil channel means + 21 spatial stds | water-holding capacity |
| B2 | + `elevation_m`, `elevation_band` | drainage / water table |
| B3 | + IGBP, lc_cci one-hot | rooting depth, ET demand |
| B4 | + 19 `ERA5_VARS` means + köppen one-hot | climate forcing the model already has |
| B5 | + VPD, Hargreaves PET, aridity, seasonal amplitude of t2m/tp | derivable terms |
| B6 | + lat/lon | geographic proximity |

**Blocks are split, not lumped** — soil / terrain / land cover are different mechanisms and a
lumped B1 cannot attribute between them. **köppen sits in B4, not B1**: it is a climate
classification, and leaving it in the static block leaks climate into B1 and shrinks the B4
increment the ladder turns on.

- **Ridge** chosen because it is **weak** — ~600 rows under a real L2 penalty cannot memorise, so
  a positive result cannot be an artefact. **GBM** (depth 3, early stop) is the nonlinearity
  control: ridge fails + GBM works → curved, not absent.
- **GroupKFold on `location_group_id`, training stations only** — plain K-fold puts neighbours in
  both folds. Main split has 0 train/val group overlap (646/69 groups); tuning must match.
- Standardise features — ridge's penalty is scale-sensitive.
- Run **with and without lat/lon**: only the coordinate-free version speaks to transfer.

### 20.14.3 Decision table (pre-registered)

Reference: network RMS per-station bias = 0.0618 / 0.0611 / 0.0875.

| Outcome | Reading | Next |
|---|---|---|
| ridge ≈ null | level not determined by available inputs | §20.7 Branch A — retarget to anomaly, reframe per §20.13 |
| ridge ≪ network | information present, network discarding it | §20.8 Branch B — post-hoc correction on the existing checkpoint FIRST as a gate |
| ridge ≈ network | network already extracts what exists | §20.7; remainder irreducible |
| **B2 ≫ B1** | level carried by ERA5 the model already has | model failure → revives decoder-vs-transformer, now worth a GPU job |
| **B3 ≫ B2** | model could have derived it but didn't | supply derived terms explicitly |

### 20.14.4 Implementation

- `station_mean_probe.py` (new, CPU, `Pool(64)`), `slurm/station_mean_probe.sh` (genoa,
  `--cpus-per-task=64`, mail flags). Conventions from `tier1_probe.py:322-391`.
- **Reuse, do not reimplement:** `dataset._load_zarr_labels` (`:174-187`, handles the qc/sm
  realign affecting 513 stations), `_open_zarr`, `fill_soil_nans` (`:539`), `ZARR_ROOT`,
  `ERA5_VARS`, `SM_DEPTHS`.
- **Do NOT instantiate `SoilMoistureDataset`** — it eagerly loads L12 (~16 GB, minutes of GPFS).
  Read zarr per station directly.
- Outputs: `csvs/station_mean_probe.json`, `figures/station_mean_probe.png`, stdout `VERDICT:`.
- Report ridge's top-10 coefficients — they name which soil properties drive level, the input
  §20.8's reparameterisation needs.
- **No silent caps** — log every dropped station and why.

### 20.14.5 Verification

- B0 RMSE must equal the between-station std of mean SM, else the target is built wrong.
- B4 ≥ B3 expected; a large jump = geographic interpolation, not physics.
- Val station list must match `val_station_metrics.csv`.
- Runtime >1 min ⇒ the L12 path is being touched.

### 20.14.6 Scope decisions

- `swvl`/ET **rejected** (not downloaded, 2026-08-05) → a negative means "not learnable from what
  we have", NOT "not learnable in principle". State it that way.
- `ssrd`/`strd` re-ingest only if B3 shows PET signal.
- Train-split context caching for §20.12.2 deferred until this ladder fixes the feature blocks and
  α-tuning it must share.

### 20.14.7 RESULT — job 25246010 (2026-08-05, genoa, 64 cores, ~2 min)

Inputs verified complete: 661 `sm_only` train+val stations (587/74), **0 dropped** — all have
`soil`, `era5`, `labels`. Val count 74 matches `val_station_metrics.csv`. `labels/qc` present for
all; **403 of 661 need the length realign** `_load_zarr_labels` handles. ERA5 raw (t2m 264-302 K,
tp 3e-4..3.7e-2 m). Soil NaN mean 1.3%, max 44%, none >50%.

RMSE on the 74 held-out val stations, m³/m³ (ridge / GBM):

| Block | nfeat | 0-10 | 10-30 | 30-100 |
|---|---|---|---|---|
| B1 soil | 42 | 0.0698 / 0.0621 | 0.0675 / 0.0625 | 0.0851 / 0.0830 |
| B2 +terrain | 46 | 0.0698 / 0.0610 | 0.0647 / 0.0607 | 0.0832 / 0.0827 |
| B3 +landcover | 80 | 0.0697 / 0.0612 | 0.0633 / 0.0610 | 0.0817 / 0.0827 |
| B4 +climate | 118 | 0.0658 / 0.0627 | 0.0599 / 0.0609 | 0.0820 / 0.0829 |
| B5 +derived | 123 | 0.0646 / **0.0599** | 0.0599 / **0.0588** | 0.0836 / **0.0815** |
| B6 +latlon | 125 | 0.0647 / 0.0597 | 0.0599 / 0.0604 | 0.0836 / 0.0819 |
| **network** | — | **0.0618** | **0.0611** | **0.0875** |
| **null** | — | **0.0752** | **0.0763** | **0.0877** |

- **Station-mean SM is only weakly predictable**: best beats null by 21% / 23% / **7%**. Level is
  largely *not* determined by the inputs we have → **§20.7 Branch A**.
- **0-10 and 10-30: the network already matches the probe** (0.0618 vs 0.0597; 0.0611 vs 0.0588).
  No untapped signal. Its remaining offset is close to irreducible from these inputs.
- **30-100 is the exception. The network sits exactly at null (0.0875 vs 0.0877) — it extracts
  NOTHING about deep absolute level — while the probe reaches 0.0815.** Small pocket (7%), but the
  network's contribution there is zero, not merely suboptimal.
- **GBM > ridge at the surface** (0.0597 vs 0.0647): the relation is **nonlinear**, and a
  ridge-only probe understates it by ~8%. The nonlinearity control earned its place — do not run
  this ladder ridge-only.
- **lat/lon adds ≈0** → the signal is physical, not geographic interpolation. Good news for
  transfer.
- **B5 derived helps GBM but not ridge** → VPD/PET/aridity carry real curved signal.
- Top ridge coefficients: soil means dominate at 10-30 / 30-100; `koppen_16` is the single largest
  at 30-100; **VPD is negative at all three depths**.

**Caveat on the printed VERDICT.** The pre-registered rule ("Branch B if best < 0.90 × network")
printed "network already extracts what exists" for all three depths, including 30-100 where the
ratio is 0.932. That is **misleading at 30-100** — the rule compares probe-vs-network but never
network-vs-null, and the network is at null there. **Fix the rule to test network vs null first;
do not quote the 30-100 verdict line as printed.**

**Terrain null is uninformative** — B2 is essentially one scalar (`elevation_m`). The zarr stores
DEM only as TerraMind tokens, so slope / aspect / TWI are unavailable. A flat B2 is not evidence
that terrain does not matter.

Artefacts: `station_mean_probe.py`, `slurm/station_mean_probe.sh`,
`csvs/station_mean_probe.json`, `figures/station_mean_probe.{png,pdf}`.
Log: `logs/station_mean_probe_25246010.out`.

### 20.14.8 What follows

1. **Branch A is the main path** (§20.7): retarget to per-station standardised anomaly, report
   ubRMSE as the headline, absolute level explicitly out of scope with this table as the evidence.
2. **§20.12's decoder-vs-transformer question is largely moot at 0-10 / 10-30** — there is no
   withheld signal to localise. Do not spend a GPU job there.
3. **The one live Branch-B pocket is 30-100.** The network is at null while ~7% is extractable.
   Cheapest test: post-hoc per-station offset correction on the existing checkpoint using the
   fitted GBM, deep depth only. No retraining, no GPU.
4. Do **not** re-run this ladder ridge-only, and do not add `swvl`/ET without re-reading §20.14.6.

---

## §21 SMAP + ECOSTRESS — ceiling tests before any build (2026-08-05)

Two independent failures, per §20.14:

| Failure | Size (m³/m³) | Cause | Status |
|---|---|---|---|
| Per-station offset (level) | 0.062 / 0.061 / 0.088 | information not in current inputs | ~irreducible from what we have |
| ubRMSE (dynamics) | 0.054 / 0.050 / 0.055 | untested; likely single-pixel supervision | **unmeasured** |

Proposed: **SMAP L3 36 km** for level, **ECOSTRESS LST 70 m** as a dense supervision target. Both
plausible, both large builds. §20.14's cheap-test-before-expensive-lever discipline just saved a
5-day run — apply it again. **Nothing in §21 trains a model.**

### 21.1 Critique of the proposal — record before testing

**Right:** SMAP is categorically different from anything §20.14 tested. Soil / terrain / land cover
/ ERA5 are all *proxies* for wetness; SMAP L-band **measures** it. §20.14's negative was explicitly
scoped to "not learnable from what we have" — SMAP is the thing we don't have. The finding
motivates the idea rather than contradicting it. Likewise LST beats §16.4's own priority-1 proxy
(S1 VV/VH) because S1 **is already a model input** — circularity the runbook itself flags.

**The serious objection — 36 km cannot see the offset we are chasing.** A SMAP pixel is ~1,300 km².
It gives the **regional mean**; our error is the station's **deviation from its region** — sub-pixel
heterogeneity, exactly what SMAP averages away. SMAP can only fix the between-region share of the
offset variance.

**Prior from data in hand: lat/lon added ≈0 to the §20.14 ladder** → station means are not strongly
spatially organised at the scale our stations sample. Lowers the prior on SMAP. Not decisive — SMAP
measures *at* the station rather than interpolating between training stations.

**Emphasis is backwards.** SMAP's strong suit is **level** (L-band measures water content; nothing
else in our inputs does). For **dynamics/ubRMSE** its marginal value over ERA5-Land is likely small
— ERA5 already carries regional wetting/drying at **9 km, finer than SMAP**. Do not assume it fixes
both.

**It changes the thesis.** With SMAP as input this becomes *SMAP downscaling*, not *retrieval from
S1/S2/TerraMind*. Respectable and publishable, but different — and the baseline hardens from
"predict the global mean" (0.075) to "resample SMAP to the station". Decide deliberately.

### 21.2 T2 — does SMAP break the station-mean wall?

Add SMAP as **block B7** in `station_mean_probe.py`. One number per station (temporal mean of SMAP
surface SM) suffices.

- Product: **SMAP L3 36 km radiometer retrieval** — keeps the "retrieval" claim; L4 is a land-model
  assimilation, closer to ERA5 than to a measurement.
- **Try GEE first** — `download_era5land_gee.py` already exists, so if SMAP L3 is in the GEE catalog
  this is a reducer query, not a new Earthdata pipeline. If only the 9 km *Enhanced* L3 is there,
  that still satisfies the independence intent — record the substitution.
- Wall: null **0.0752 / 0.0763 / 0.0877**; network 0.0618 / 0.0611 / 0.0875.
- **Read:** drives station-mean RMSE toward ~0.03 → SMAP validated before any model code. Barely
  moves → sub-pixel heterogeneity dominates, months avoided.
- Run a **separate** SMAP-anomaly → station-anomaly check; the dynamics claim needs its own evidence.

### 21.3 T3 — does ECOSTRESS have enough coverage?

- **Hard constraint: ECOSTRESS launched July 2018.** Training years are 2016-2022 → covers at most
  ~5 of 7 years, **zero of 2016-2017**. Any dense-supervision arm is masked to the post-2018 subset.
- Census via Earthdata **CMR metadata API** — no granule downloads. ECOSTRESS is on LP DAAC, **not**
  Microsoft Planetary Computer, so no existing downloader applies.
- Report usable clear-sky overpasses **per station per year** (per-year, so the 2016-2017 gap is
  visible), median per station-year.
- **Read:** too few → dense supervision falls back to the §16.4 S1/NDVI proxy, whose raw 224² pixels
  already exist in `/projects/prjs1968/satellite_zarr`.

### 21.4 Deferred: T1 and Phase 0

- **T1 (is the offset regional or sub-pixel?)** — semivariogram / Moran's I of per-station offset vs
  separation distance. **Deferred**: T2 measures the same question *directly*, and T1 needs Phase
  0's oos+oost stations for power. 74 val stations = 69 location groups → far too sparse. **Do not
  run T1 on val alone.** When run, report n-pairs per distance bin: a nugget-dominated variogram
  with few short-range pairs is evidence of **no power**, not of white noise.
- **Phase 0 (real oos/oot/oost evaluation)** — `meeting_output/` currently holds a **10-station
  smoke run** (Jun 18), not real numbers. Needs run 25235976 finished so `best.pt` is final.
  `sbatch slurm/evaluate_meeting.sh cls_depth_star_reg best.pt`.
  - **OOT is the counterfactual**: `split_filter=["train","val"], years=[2023]` — stations the model
    memorised, unseen year. Small OOT offset → the model *can* hold level on a seen station → the
    failure is purely station transfer and §20.14's conclusion is airtight. Large → representation
    problem, architecture question reopens.
  - Fixes needed: add `network`/`source_network` to `META_COLS` (`evaluate_splits.py:39-43`) for the
    sensor-calibration check; report `train.compute_metrics` alongside its own (pooled vs
    mean-across-stations — only the former is comparable to `val_station_metrics.csv`); post-filter
    OOT to `oot_eligible` (358) since the script does not.

### 21.5 Decision gates

| T2 | T3 | Action |
|---|---|---|
| SMAP breaks the wall | — | build the SMAP input path — highest-value lever |
| SMAP flat | — | **drop SMAP**; level is sub-pixel and irreducible → §20.7 Branch A |
| SMAP flat | ECOSTRESS rich | skip SMAP, go straight to dense supervision |

Dense supervision gets its **own plan after T3**, as a clean ablation with the current single-pixel
run as control. Design note: prefer a **multi-task LST head** over a proxy-consistency loss — LST is
a real measurement and not a model input.

### 21.6 Carried forward

- **§16.4 Step 0 already PASSED** — `capacity_check.py` job 24352962: decoder norm-std 0.0061 →
  0.2504, corr 1.000 vs a synthetic dense target. The decoder **can** paint structure; flat maps are
  a supervision problem. Step 1 green-lit and unstarted.
- **§17.7 supersedes §16.4 Step 2**: the missing piece is a full-resolution **carrier** (measured
  pixels as decoder input), ranked above the bilinear→PixelShuffle swap. Order: dense supervision
  first, carrier second, upsampler third.
- `lambda_boundary` is the only active loss term touching off-centre pixels, and its gradient is
  exactly zero inside [0,1] — it is **not** spatial supervision.
- **TWI is not viable** as a proxy: it needs upslope contributing area, not computable from an
  isolated 2.24 km tile.

### 21.7 RESULT — SMAP does not break the wall (job 25250245, 2026-08-05)

`NASA/SMAP/SPL3SMP_E/006`, 661 stations, 2016-2022 per-year means. **651/661 with
retrieval, 0 missing for brightness temperature.** Standard 36 km `SPL3SMP` is NOT in GEE —
Enhanced 9 km substituted (same footprints, Backus-Gilbert interpolated).

| Block | 0-10 | 10-30 | 30-100 |
|---|---|---|---|
| B6 +latlon | 0.0647 / 0.0597 | 0.0599 / 0.0604 | 0.0836 / 0.0819 |
| B7 +smap_sm | 0.0631 / 0.0607 | 0.0597 / 0.0604 | 0.0812 / **0.0782** |
| B8 +smap_tb | 0.0627 / **0.0576** | 0.0603 / **0.0576** | 0.0806 / 0.0795 |
| **network (e16)** | **0.0585** | **0.0584** | **0.0851** |
| null | 0.0752 | 0.0763 | 0.0877 |

- **VERDICT: drop the SMAP integration.** Gains only 3.5 / 2.0 / 4.5% over the SMAP-free
  best; level still just 23 / 25 / 11% over null; worth ~3% on total RMSE. Not worth a new
  acquisition pipeline plus reframing the thesis as SMAP downscaling.
- **0-10 and 10-30: network ≈ probe** (0.0585/0.0584 vs 0.0576/0.0576). Nothing left to
  extract. **Compare like-for-like weighting** — an earlier note claimed the network had
  *passed* the probe, but that compared sample-weighted `sqrt(MSE − ubRMSE²)` against the
  probe's station-equal RMSE. It is a tie, not a win.
- **30-100 is the one live pocket**: network 0.0851 vs probe 0.0782 = **8.1% unused**, and
  the network sits at null (0.0877). Fired the pre-registered Branch-B rule correctly here.
- **B8 > B7 at both shallow depths** (5.1% / 4.6%): raw TB normalised by ERA5 `skt` beats
  SMAP's own retrieval. Evidence the L3 retrieval's climatological tau-omega assumptions
  discard information the observable retains. **Holds independently of whether SMAP is ever
  ingested — worth writing up on its own.** At depth the retrieval wins instead
  (0.0782 vs 0.0795), consistent with vegetation/roughness corrections mattering more where
  the surface signal is weakest.
- `smap_sm_am` is the LARGEST single ridge coefficient at 10-30 and third-largest elsewhere
  — SMAP contributes, there is just little total signal to contribute to.
- **Caution:** TB_h ranges down to 76.8 K, implausible for land (L-band land TB is 200-280 K).
  Water-contaminated coastal/lake cells. Noise in B8; do not treat raw TB extremes as physical.

### 21.8 Run cls_depth_star_reg — final

Cancelled at e16 by a pre-agreed rule (stop unless e16 beat e13's 0.002187 by ≥1%; it
reached 0.002177 = 0.46%, a new best but below threshold). **`best.pt` = epoch 16,
val_loss 0.00217653.** 10h13m, 16 epochs.

| | 0-10 | 10-30 | 30-100 |
|---|---|---|---|
| ubRMSE | 0.0539 | **0.0500** | 0.0552 |
| RMS per-station offset | 0.0585 | 0.0584 | 0.0851 |

- **I called this run converged at e5 and again at e9. Wrong both times** — it produced four
  more new bests (e11, e12, e13, e16) after apparent plateaus. Do not predict this run's
  stopping point from a short plateau.
- **10-30 is the best depth, not 0-10** (ubRMSE 0.0500 vs 0.0539). Counterintuitive since
  satellites sense the surface; likely physical — 0-10 responds sharply to rain events and
  dries within days (high temporal variance), 10-30 is damped.
- Overfitting reached **21×** (train 0.000099 vs val 0.002177). This is station
  memorisation, structural — §19.5 proved regularisation cannot touch it. The fix is
  anomaly retargeting (§20.7-A1), which removes the memorisable quantity from the target.

### 21.9 Next, in order

1. **Retarget to per-station standardised anomaly** (§20.7-A1). Hits all three problems at
   once: removes the unlearnable level component, removes the memorisable quantity driving
   the 21× gap, frees capacity for dynamics. **Testable prediction: ubRMSE improves.**
2. **Dense spatial supervision** (§16.4 Step 1). Step 0 PASSED 13 months ago (job 24352962:
   decoder norm-std 0.0061 → 0.2504, corr 1.000). Only 1 of 50,176 pixels is supervised.
   Use the S1/NDVI proxy already at 224² in `satellite_zarr` — no new download.
3. **Reality-check the 0.04 ubRMSE bar** — it is SMAP's requirement against *core*
   validation sites, not broad ISMN point-scale validation (typically 0.05-0.06). If that
   holds, 0.0539 / 0.0500 at *unseen* stations may already be competitive.
4. **Cheap:** 30-100 post-hoc offset correction via the fitted GBM (no retrain); real
   oos/oot/oost evaluation — **now specified in §22** (Session 21, 2026-08-06).

### 21.10 GEE operational notes (cost real time — do not rediscover)

- Default `earthengine authenticate` needs `gcloud` (absent on Snellius). `notebook` mode
  needs interactive paste-back, which a FIFO cannot deliver (the write blocks). **Working
  pattern:** split across two processes — p1 `flow = oauth.Flow(auth_mode="notebook")`,
  persist `flow.code_verifier`; p2 `oauth._obtain_and_write_token(code, verifier, scopes)`.
- **Sort stations geographically before chunking.** Globally-scattered chunks trigger
  "User memory limit exceeded" — EE must hold tiles from everywhere in one request.
- **Do not ask EE for `collection.mean()` before sampling.** Reducing ~4000 images
  server-side stalls indefinitely. Sample per year, average locally — identical result.
- `getInfo()` on a FeatureCollection aborts past **5000 elements**. For 661 stations ×
  ~2500 days use `ee.batch.Export.table.toDrive`, not `getInfo`.

---

## §22 Held-out evaluation — OOS / OOT / OOST (Session 21, 2026-08-06)

**Written before the job runs.** §22.1–22.8 are the design and the pre-registered
predictions; §22.9 is filled only after the numbers exist. Nothing here is provisional.

### 22.1 Why now

`cls_depth_star_reg` `best.pt` = epoch 16 is the final Phase-1 model (§21.8). Its only
reported numbers are on **val**: ubRMSE 0.0539 / 0.0500 / 0.0552. Val drove early stopping
and LR scheduling, so **it cannot appear in the paper**. The three held-out splits were
pre-registered in §13.1 and implemented in `evaluate_splits.py:33-37` but have **never been
run for real** — `meeting_output/` still holds the 2026-06-18 smoke run (10 stations,
against the older `baseline_huber`). This section closes §21.9 item 4.

### 22.2 Verified station counts

Probed all 842 `sm_only` zarr stores directly (ERA5 coverage, S2 coverage, ≥30 `qc==0`
observed days) rather than trusting the CSV `end_date`, which runs to 2025 while ERA5/S2
often stop earlier:

| Split | Filter | Years | Stations |
|---|---|---|---|
| OOS  | `split == "oos"` | 2016–2022 | **180** |
| OOT  | `split ∈ {train, val}` | 2023 | **399** (355 train + 44 val) |
| OOST | `split == "oos"` | 2023 | **98** |

All 842 stores present and `.complete` on scratch (survived the purge/restore).

- **Do not filter OOT on `oot_eligible`.** It is `True` only for `split=="train"` rows by
  construction (`create_evaluation_splits.py:335`), so filtering would silently drop the
  44 val stations. The dataset's own year gating (`dataset.py:950-967`) already removes
  stations lacking 2023 ERA5/S2/labels — the counts above are what it will actually yield.
  Carry a `train_split` column instead so val can be sliced out at analysis time.
- 98 < the 128 flagged `oost_eligible`: the flag checks only `end_date`, not whether ERA5
  and S2 actually reach 2023.
- OOST stations are a **subset** of the OOS stations (579 unique stations, 677 station×split
  combinations).

### 22.3 Input context is not label leakage

Predicting a 2023 day needs the trailing **365-day input window**, which for early-2023
targets reaches back into 2022. This is automatic — no code change:
`__getitem__` passes `(year, doy)` to `load_s2_rolling_zarr` / `load_s1_rolling_zarr` /
`load_era5_rolling` (`dataset.py:1005-1045`), which slice the station's **full** cached
arrays. `years=[2023]` selects which days become *samples* (`dataset.py:955`); it does not
truncate input history.

**This is not leakage.** Inputs are satellite tokens, ERA5, SIF, TWSA, static soil. Soil
moisture observations are **only ever targets** — no SM history enters `__getitem__`. Using
2022 weather and imagery to predict January 2023 is what is available at deployment time.

The honest asymmetry: for an OOT station those 2022 input tokens were also seen in training
(as context for 2022 targets). So seen-context fraction decays across 2023:

| target date | input window | seen context |
|---|---|---|
| 2023-01-01 | 2022-01-02 … 2023-01-01 | ~100% |
| 2023-07-01 | 2022-07-02 … 2023-07-01 | ~50% |
| 2023-12-31 | 2023-01-01 … 2023-12-31 | ~0% |

Inherent to the OOT definition (same station, novel year), not a defect — but it motivates
§22.7.

**Decision:** OOT metrics *and* figures use 2023 only. No training-period context top-up
pass. One GPU pass total.

### 22.4 Architecture — dump predictions once, everything else is CPU

The decoder emits a full `(3,224,224)` map but the metric path reads only the station pixel
`(112,112)` (`model.py:348-349`). **Spatial maps are not required for metrics** — same
forward pass, zero extra cost, nothing spatial written.

```
GPU once (~70 min):  eval_predict.py  → eval_output/predictions_{oos,oot,oost}.parquet
CPU seconds:         eval_metrics.py  → per_station_{split}.csv, metrics_summary.csv
CPU seconds:         plot_eval_scatter.py, plot_eval_timeseries.py
GPU separate:        spatial maps (deferred, §22.8)
```

**Why the parquet, and not the existing pattern:** every current plot script rebuilds the
dataset and re-runs GPU inference (`plot_timeseries_meeting.py:56` infers per station),
~10 min per re-plot. Long format (one row per station×day×depth, ~1.2M rows, ~25 MB
compressed) makes per-depth groupby trivial and keeps depth-coverage differences explicit.

**Store it on work3, not scratch.** Scratch has already been purged once. The parquet is
the durable record of what this checkpoint predicted — every later comparison (notably
anomaly retargeting, §21.9 item 1) is a per-station per-day diff against it, impossible if
only summary metrics survive. Stamp `eval_output/manifest.json` with run name, checkpoint
path, epoch, val_loss, split definitions, station counts, run date.

### 22.5 Metric definitions

Per station × depth: `n`, `RMSE`, `ubRMSE`, `MAE`, `bias`, `offset` (= `mean(p) − mean(t)`),
`R`, `R2_pearson`, `NSE`, `NSE_anom`. Require `n ≥ 5`; ubRMSE additionally `n ≥ 2`.
ubRMSE removes **each station's own temporal mean** (`train.py:350-368`).

Two traps this section exists to avoid:

- **NSE and `R2_pearson` are reported separately, never merged as "R²."** A pure level
  offset drives NSE strongly negative while leaving Pearson untouched. Given §20.1 (level
  is the dominant error, RMS offsets 0.0585/0.0584/0.0851 vs ubRMSE 0.0539/0.0500/0.0552),
  **the gap between them is the expected headline result**, not a nuisance.
- **Every summary metric is emitted twice** — `*_stn` (mean across stations, each station
  once; what `evaluate_splits.py:103` does) and `*_pool` (over all samples; what
  `train.py:324` does, so comparable to the val numbers). §21.7 records that silently
  mixing these already produced one wrong conclusion ("it is a tie, not a win"). Also emit
  `RMS_offset`; §20.1 warns global `bias` reads as near-perfect calibration when per-site
  calibration is poor.

### 22.6 Pre-registered sanity checks (from §13.5)

Registered **before seeing any number**, so a FAIL is a finding rather than something
rationalised afterwards:

1. OOT ubRMSE ≈ val ubRMSE — same stations, novel year only
2. OOS ubRMSE > val ubRMSE — novel stations
3. OOST ubRMSE ≥ OOS — hardest condition
4. `metrics_summary.csv` has exactly 9 rows (3 splits × 3 depths)

**Hard gate before any of the above is believed:** run the metric code over a
`split=val, years=2016–2022` dump and reproduce **0.0539 / 0.0500 / 0.0552** from
`val_station_metrics.csv` epoch 16. This proves the new code agrees with
`train.py:compute_metrics`. Until it passes, no OOS/OOT/OOST number means anything.

### 22.7 Pre-registered diagnostic — OOT error vs day-of-year

Plot OOT mean absolute error binned by day-of-year in 2023, one line per depth. Free: a
groupby on the parquet, no GPU.

Motivated by §22.3's 100%→0% seen-context decay. It separates two memorisation modes that
§21.8 lumps together under the 21× train/val gap:

- **flat** → skill does not depend on having seen the input context = genuine temporal
  generalisation. Consistent with pure *station-level* memorisation (offset, no DOY trend).
- **rising** → skill decays as seen context leaves the window = *input-sequence*
  memorisation.

**The control is OOST**, overlaid on the same axes. OOST stations have 0% seen context at
every DOY, so they carry the seasonal shape alone. OOT rising while OOST is flat = memory
decay. Both tracing the same shape = seasonality, and the diagnostic says nothing.
Soil moisture error genuinely is seasonal — **do not read a rising OOT curve without the
OOST control.**

Suggestive, not decisive. The decisive test is an ablation: zero the pre-2023 portion of
the input window and check whether early-2023 predictions degrade. Costs a second GPU pass.
**Run it only if this free diagnostic shows a signal.**

### 22.8 Deliverables

| file | role |
|---|---|
| `eval_predict.py` | GPU pass → parquet |
| `eval_metrics.py` | parquet → per-station + summary CSVs |
| `plot_eval_scatter.py` | pred-vs-obs 3×3, station-mean, metric distributions, ubRMSE-vs-offset, OOT error-vs-DOY |
| `plot_eval_timeseries.py` | best-10 / worst-10 per split; 3 depth panels, obs dots vs pred line |
| `slurm/eval_predict.sh` | `gpu_h100`, 1 GPU, 16 CPU, 120 G, **4 h** (not the 2 h in `evaluate_meeting.sh`) |

```bash
# smoke first
python eval_predict.py --run-name cls_depth_star_reg --max-stations 5 --splits oos
# full
sbatch slurm/eval_predict.sh cls_depth_star_reg best.pt
```

Outputs land in **`eval_output/`** — the stale June `meeting_output/` is left intact rather
than silently overwritten.

- Load checkpoints with `ckpt_utils.load_checkpoint` **only**. The loaders in
  `eval_stations.py:35` and `demo_plot.py:44` build the model without
  `use_cls_depth`/`drop_path_rate` and call `load_state_dict` strictly — both crash on this
  checkpoint. (`eval_stations.py` has been broken for this run since §19.)
- Time-series station ranking needs an `n ≥ 100` guard; `n` spans 5 to ~2500 across
  stations, so an unguarded "best" list selects short records, not good ones.
- Spatial maps (`plot_spatial_sm_meeting.py`, already written) are deferred to a separate
  GPU pass — scope set after §22.9.

**Memory — the reason this is two jobs, not one (measured, do not rediscover).**
`SoilMoistureDataset` eagerly preloads every station's L12 tokens into RAM. Uncompressed
sizes from the zarr array shapes:

| split | stations | L12 in RAM |
|---|---|---|
| val | 74 | 18 GB |
| oos / oost | 181 | 39 GB |
| **oot (train+val)** | 774 | **156 GB** |

- `evaluate_meeting.sh`'s `--mem=120G` is **not enough for OOT** — it would OOM. Job A
  (val+oos+oost) runs at 120 G; job B (oot) needs 300 G.
- The split loop must `del` the previous dataset and `gc.collect()` before building the
  next, or oos+oot are resident together and peak near 195 GB.
- **`DISABLE_L12_CACHE=1` is not the fix.** Measured on 8 OOT stations: RAM cache 1:32 /
  7.3 GB RSS vs lazy zarr 5:59 / 6.3 GB RSS — **3.9× slower** (the §3f GPFS chunk-read
  penalty) for ~0.13 GB/station saved. Extrapolated to OOT's ~120k samples the lazy path
  needs ~4.3 h of inference and busts the 4 h wall. Buy the memory, not the disk reads.
- **The cost is init, not inference.** Measured on the 2026-08-06 run: OOT spent ~65 min
  loading 156 GB off GPFS before the first batch, against ~15 min of actual inference
  (observed throughput ~179 samples/s). Chunk it — `eval_predict.py` takes
  `--csv-start-idx/--csv-end-idx/--tag` (same pattern as `precompute_terramind.py`), and
  `eval_metrics.py` globs `predictions_{split}_*.parquet` back together, de-duplicating on
  `(station, year, doy, depth)`:

```bash
for i in 0 1 2 3; do
  lo=$((i*195)); hi=$(((i+1)*195))
  sbatch --mem=100G slurm/eval_predict.sh cls_depth_star_reg best.pt \
      --splits oot --csv-start-idx $lo --csv-end-idx $hi --tag c$i
done   # ~20 min wall instead of ~80, same total GPU cost
```

### 22.9 Results (jobs 25283775 + 25284407, 2026-08-06)

Checkpoint `cls_depth_star_reg/best.pt` e16. Station counts hit the §22.2 probe exactly
(val 74, oos 180, oot 399, oost 98). Wall: val 11.9 + oos 29.7 + oost 4.0 min (job A),
oot 15.6 min inference after ~66 min init (job B).

**Gate PASSED exactly** — three independent quantities reproduced to 4 dp:
pooled ubRMSE 0.0539/0.0500/0.0552 (§21.8), station-equal 0.0490/0.0459/0.0490
(`val_station_metrics.csv` e16), RMS offset 0.0585/0.0584/0.0851 (§21.8). The metric code
agrees with `train.py:compute_metrics`.

Station-equal means (`_stn`); `metrics_summary.csv` also carries `_pool`.

| split | depth | ubRMSE | RMSE | r² | NSE | NSE_anom | RMS offset | n |
|---|---|---|---|---|---|---|---|---|
| val  | 0-10 | 0.0490 | 0.0721 | 0.598 | −1.39 | +0.23 | 0.0585 | 74 |
| val  | 10-30 | 0.0459 | 0.0688 | 0.589 | −11.3 | −1.03 | 0.0584 | 51 |
| val  | 30-100 | 0.0490 | 0.0868 | 0.495 | −17.2 | −1.29 | 0.0851 | 43 |
| **oos** | 0-10 | **0.0507** | 0.0762 | 0.611 | −0.76 | **+0.35** | 0.0639 | 180 |
| **oos** | 10-30 | **0.0489** | 0.0742 | 0.564 | −2.40 | −0.07 | 0.0646 | 125 |
| **oos** | 30-100 | **0.0499** | 0.0899 | 0.480 | −20.8 | −1.76 | 0.0865 | 106 |
| oot | 0-10 | 0.0397 | 0.0472 | 0.674 | −0.36 | +0.17 | **0.0303** | 399 |
| oot | 10-30 | 0.0357 | 0.0431 | 0.668 | −0.85 | +0.05 | **0.0289** | 350 |
| oot | 30-100 | 0.0369 | 0.0490 | 0.622 | −30.5 | −3.55 | **0.0427** | 287 |
| oost | 0-10 | 0.0542 | 0.0798 | 0.579 | −4.48 | −0.68 | 0.0665 | 98 |
| oost | 10-30 | 0.0465 | 0.0768 | 0.594 | −16.1 | −1.65 | 0.0713 | 82 |
| oost | 30-100 | 0.0569 | 0.0974 | 0.476 | −40.6 | −1.98 | 0.0906 | 67 |

#### Pre-registered checks: 5 of 6 pass

1. OOT ≈ val — PASS at all depths, but **the check is badly written**: it is two-sided at
   25% tolerance and OOT came in 19-25% *better*, not merely close. A PASS here means much
   less than it appears. Rewrite as a signed test before reusing.
2. OOS > val — PASS at all three depths.
3. OOST ≥ OOS — PASS at 0-10 and 30-100, **FAIL at 10-30** (0.0465 vs 0.0489). Survives a
   like-for-like recut on the 98 common stations (0.0465 vs 0.0481, 3.3%), so it is a real
   small improvement in *dynamics*, not a composition artefact. Level error still degrades
   in 2023 at every depth (0.0653→0.0665, 0.0637→0.0713, 0.0823→0.0906). **Temporal
   extrapolation costs absolute level, not tracking.**
4. Nine held-out rows — PASS.

#### The level failure is TRANSFER, not representation

The counterfactual `logs.txt:3474` asked for. OOT and OOST are both 2023, so year is
controlled; they differ only in whether the station was seen in training:

| RMS offset | 0-10 | 10-30 | 30-100 |
|---|---|---|---|
| OOT — 2023, **seen** stations | 0.0303 | 0.0289 | 0.0427 |
| OOST — 2023, **novel** stations | 0.0665 | 0.0713 | 0.0906 |
| OOS — 2016-22, novel stations | 0.0639 | 0.0646 | 0.0865 |

**Level error is 2.2× larger on novel stations in the same year.** OOST sits with OOS, not
with OOT — so 2023 is not an easy year, station-seenness is the whole effect. The network
*can* represent per-station level (it does so, to 0.030, for stations it has seen); it
cannot *infer* it for novel stations. This corroborates §20.14/§21.7 from the opposite
direction: the level information is not recoverable from the inputs. Confirms the §21.8
attribution of the 21× train/val gap to **station-level memorisation**, and strengthens the
case for §21.9 item 1 (anomaly retargeting), which removes exactly the memorised quantity.

#### NSE vs r² — why §22.5 insisted on both

At OOS 0-10: r² = 0.611 but NSE = −0.76, purely because the level is wrong; NSE_anom is
positive, i.e. the model beats each station's own climatology on dynamics. Reporting a
single "R²" would have hidden either the skill or the failure.

**CORRECTION (same day). Never quote mean NSE — it is unbounded below and a handful of
catastrophic stations dominate it.** Use the median and the fraction beating climatology:

| OOS | mean NSE_anom | **median** | % stations > 0 | p10 |
|---|---|---|---|---|
| 0-10 | +0.347 | **+0.555** | 88.9% | −0.06 |
| 10-30 | −0.073 | **+0.476** | 82.4% | −0.26 |
| 30-100 | −1.761 | **+0.214** | 58.5% | −2.49 |

An earlier draft of this section read "30-100 does not work … worse than predicting the
station mean" off the mean. **That was wrong.** The median station at 30-100 beats its own
climatology and 58.5% of stations do. 30-100 is the **weakest** depth with a **heavy failure
tail**, not a broken one. Same pattern on OOST (0-10 mean −0.676 vs median +0.516).

Per-station `bias` IS the level offset (`mean(p) − mean(t)`); there is no separate `offset`
column. The §20.1 cancellation warning is dramatic here: OOS 10-30 mean bias is −0.0001
(reads as flawless calibration) while RMS offset is 0.0646 and 42/125 stations are off by
>0.05. **Always RMS the per-station bias.**

#### SNOTEL — raw ubRMSE is worse, but the model is RELATIVELY BETTER there

**CORRECTED (same day). Do NOT exclude SNOTEL — it would be cherry-picking in reverse.**

| OOS 0-10 | non-SNOTEL | SNOTEL |
|---|---|---|
| ubRMSE | 0.0433 | 0.0616 (**+42%**) |
| observed SD | 0.0623 | 0.0898 (**+44%**) |
| ubRMSE / SD (lower better) | 0.682 | **0.630** |
| median NSE_anom (higher better) | +0.535 | **+0.602** |

SNOTEL ubRMSE is 33-42% higher at every depth — but their observed variability is 44-57%
higher. **Normalised for that, SNOTEL is the split the model handles best** (better
ubRMSE/SD and better median NSE_anom at all three depths). Snowmelt-driven mountain sites
simply swing harder, and ubRMSE is an absolute metric. An earlier draft proposed excluding
SNOTEL to report "OOS 0-10 = 0.0433" — that would drop the stations where the model
performs *relatively best*. **Report ubRMSE alongside a variance-normalised metric
(NSE_anom or ubRMSE/SD) instead of excluding anything.**

Freeze masking is also not the lever it looked like. ERA5 `skt` is in the zarr in Kelvin
(confirmed, 100% join to prediction rows), so the SMAP-style flag `skt_mean < 273.15` is
free to compute. It flags 23-32% of OOS rows — but frozen days are only **1.05-1.11×**
worse, so masking them moves overall MAE by ~2.5%. The SNOTEL gap is a variance effect,
**not** a frozen-soil effect. Do not attribute it to freezing without checking.

#### §22.7 diagnostic — FIRED. Seasonality excluded by two independent controls

`figures/eval/oot_error_vs_doy.png`. Quantified as the trend in |error| anomaly (per-station
mean removed) per 100 days of year; 95% CI from a station-clustered bootstrap.
**Positive = error grows through the year.** OOS 2016-22 is the strong seasonality control
— 7 years × 180 novel stations, so year-specific weather averages out.

| | 0-10 | 10-30 | 30-100 |
|---|---|---|---|
| **OOT 2023 — SEEN stations** | **+0.0033** [+0.0023,+0.0041] | **+0.0045** [+0.0032,+0.0055] | **+0.0032** [+0.0011,+0.0048] |
| OOST 2023 — novel | −0.0039 [−0.0074,−0.0002] | −0.0016 (ns) | +0.0022 (ns) |
| OOS 2016-22 — novel, 7 yr | −0.0025 [−0.0038,−0.0012] | −0.0013 (ns) | −0.0007 (ns) |

- **OOT rises at all three depths, significantly. Both novel-station controls go the other
  way** (flat or declining), with non-overlapping intervals at 0-10 and 10-30. Seasonality
  would move all three rows together. It does not. **Seasonality is excluded.**
- Direction matches the mechanism: on 2023-01-01 the 365-day input window is entirely 2022
  inputs seen in training; by 2023-12-31 it is entirely novel.
- Magnitude ≈ +0.012 m³/m³ across the year — **~30% of OOT's 0.0397 ubRMSE**. So a
  meaningful part of OOT's apparent advantage over val is borrowed from having seen the
  inputs, and it decays. **Do not quote OOT as clean temporal generalisation.**
- **Residual confound:** if OOT station records end mid-2023 more often than OOST's,
  late-year samples have sparser valid tokens, which mimics decay. Not yet checked.
- Decisive test remains the masking ablation (zero the pre-2023 window, re-predict, second
  GPU pass) — immune to that confound. **Not yet run**, but now justified rather than
  speculative.

#### The NSE_anom failure tail — an AMPLITUDE problem, not a data bug

Diagnosed on OOS after the correction above. Stations with NSE_anom < 0 are
20/180 (0-10), 22/125 (10-30), **44/106 (30-100)**. They are not broken sites:

| OOS 0-10 | predicted SD | observed SD | pred/obs |
|---|---|---|---|
| NSE_anom < 0 | 0.0759 | 0.0456 | **1.67** |
| NSE_anom ≥ 0 | 0.0730 | 0.0757 | 0.98 |

- Failing stations have **1.7-2.6× lower observed variability**; 84-90% of them sit below
  the median for variability. Their correlation is still fine (R ≈ 0.62-0.65) — the shape
  is right, the **amplitude** is wrong.
- **The model emits nearly the same spread at every station** (0.0759 vs 0.0730 at 0-10 —
  essentially identical) regardless of how variable the site actually is. At 30-100 the
  observed SD differs 2.6× between groups while the model's output differs only 1.24×.
- **Conclusion: the network has learned one typical soil-moisture rhythm and applies it
  everywhere.** It is not failing at quiet stations so much as failing to know they are
  quiet. This is the same class of defect as the level failure — an inability to infer a
  per-station property — one moment up (σ) instead of the first (μ).
- Weak secondary clustering: SCAN fails 18% at 0-10 and 40% at 30-100; Shrub-Savanna 31%
  at 0-10. The variance effect explains most of it; do not chase the network signal first.

#### GATE BEFORE ANOMALY RETARGETING — is σ_station predictable?

Retargeting to `z = (θ − μ_station)/σ_station` removes both the unlearnable level and the
miscalibrated amplitude, and the loss code is unchanged. **But converting a prediction back
to θ needs that station's μ and σ, which a novel station does not have** — computing them
from the test period is exactly the leakage §22 exists to prevent. §20.14/§21 proved μ is
not inferable from the inputs.

So run the cheap probe first: **adapt `station_mean_probe.py` (which already does this for
μ) to target σ.** CPU-only, ~an afternoon.
- σ predictable → fix the amplitude directly, no leakage, most of the win without a reframe.
- σ not predictable → retargeting is a **reframe, not a fix**: it changes the task to
  "given a site's climatology, predict its anomalies" (legitimate — that is what drought
  monitoring and data assimilation consume) and the paper must say so explicitly rather
  than imply absolute prediction at unknown sites.

Decide this before spending a training run.

#### Artefacts

`eval_output/`: `predictions_{val,oos,oot,oost}.parquet` (6.7 MB total, work3, purge-proof),
`per_station_{split}.csv`, `metrics_summary.csv`, `gate.json`, `manifest.json`,
`timeseries/{split}/` (60 figures + contact PDFs).
`figures/eval/`: 5 scatter figures (PNG+PDF).

---

## §23 Spatial heterogeneity diagnostic — does the predicted field vary in space? (Session 22, 2026-08-10)

**Status: PLANNED, not yet run.** Approved 2026-08-10. Diagnostic only — no retraining, no
loss change, no touching `plot_spatial_sm_meeting.py`.

### 23.0 Why this exists

`SoilMoistureModel.forward()` (`model.py:670`) returns `(B, 3, 224, 224)` — the model has
**only ever emitted a 2D field**. There is no separate spatial head to enable. Every
evaluation path in the repo indexes the centre pixel and discards the rest:

| file | line | code |
|---|---|---|
| `eval_predict.py` | 75 | `preds.append(mu[:, :, srow, scol]...)` |
| `evaluate_splits.py` | 61 | `all_preds.append(mu[:, :, SROW, SCOL]...)` |
| `demo_plot.py` | 79 | `mu[0, :, model.STATION_ROW, model.STATION_COL]` |

and `masked_huber_loss` (`model.py:751`) supervises **1 pixel of 50,176**, with
`lambda_tv = 0.0` (`train.py:213`, disabled per the §16.1 verdict).

Two prior results bracket the question:

- **§16.1** (`tier1_probe.py`, epoch 11 of the older `baseline_huber`): output map
  norm-std = **0.0065**; decoder taps 0.089 / 0.095 / **0.034** at 28²/56²/112². Structure
  survives to 56² and collapses by 112². Verdict: **decoder-side**, not attention-side.
- **§16.4** (`capacity_check.py`, job 24352962): froze everything up to the bottleneck and
  overfit only the decoder against a dense synthetic target — norm-std **0.0061 → 0.2504,
  corr 1.000**. The decoder *can* paint structure. Flatness is a **supervision artefact,
  not a capacity limit**.

What is missing is the same measurement on the **final Phase-1 model**
(`cls_depth_star_reg/best.pt`, epoch 16), at stations where the model demonstrably works,
across seasons, next to the satellite scene the model actually consumed.

**Expected outcome, stated up front so the figure is not oversold: the maps will be
near-uniform.** The value is that it is measured on the delivered model, at good stations,
against a landscape that demonstrably has structure, with a flat-terrain control that kills
the "the landscape is just flat" rebuttal. It converts an assertion into a figure with
numbers, and it is the evidence base for Phase 2 dense supervision.

### 23.1 Station selection — ubRMSE alone is a trap

Ranking `eval_output/per_station_oos.csv` purely on `ubRMSE_0_10` returns:

| station | ubRMSE₀₋₁₀ | IGBP | LULC classes | DEM relief |
|---|---|---|---|---|
| ISMN_NGARI_ALI02 | 0.0172 | **WAT (lake)** | 2 | 301 m |
| ISMN_USCRN_Tucson-11-W | 0.0219 | SAV | 2 | 249 m |
| ISMN_SCAN_Crossroads | 0.0265 | GRA | **1** | **9.5 m** |

A lake and flat deserts — sites where a uniform map is arguably *correct*. Publishing that
hands a reviewer the rebuttal for free.

**Split: `oos`** (spatially held out, 2016-2022, so a full seasonal cycle fits inside one
year). Not `oot`/`oost` — 2023-only, `n = 365`, and a weaker generalisation claim. Not
`val` — those stations were seen in training.

**Gates, applied before ranking.** Each one blocks a specific objection:

| gate | blocks |
|---|---|
| `n_0_10 ≥ 700` (~2 yr daily) | "the score is a fluke of a short record" |
| `R_0_10 ≥ 0.6` | "low ubRMSE because nothing ever moves" |
| `std(obs) ≥ 0.03` at 0-10, from `predictions_oos.parquet` | explicit dynamic-range floor |
| `IGBP ∉ {WAT, SNO}` | drops the lake |
| `category == sm_only` | `train.py:184` — the model never saw anything else |

**Three stations, three roles** (this is the design, not a count):

1. **best gated ubRMSE** — the model at its strongest;
2. **most heterogeneous footprint** among the gated set (`#LULC ≥ 3` or relief ≥ 30 m) —
   e.g. `ISMN_RSMN_Iasi`, 6 classes / 75 m;
3. **flat control** `ISMN_SCAN_Crossroads` (1 class, 9.5 m). **Keep this one.** If the map
   is equally flat at Iasi and at Crossroads, the flatness is the model, not the landscape.
   That contrast is the punchline and it costs one extra station of runtime.

### 23.2 Figure design

**As shipped** (v2 — the per-panel anomaly block of the first draft was dropped in favour of
the time series, so that one figure carries both the spatial and the temporal validation):

```
HEADER      station, network, lat/lon, elevation, IGBP + Köppen spelled out, record, ubRMSE
CONTEXT     [ S2 RGB | NDVI | DEM | land cover w/ NAMED classes ]   structure available to paint
SEASON GRID (4 rows DJF/MAM/JJA/SON)
            [ anchor scene (MODEL INPUT, orbit+date+lag) | SM 0-10 | 10-30 | 30-100 ]
            one shared 0-0.5 scale; σ, ñ, Δ printed in each map title
TIME SERIES [ 0-10 | 10-30 | 30-100 ]  whole target year at the station pixel,
            predicted line + observed dots, the four map dates marked
BOTTOM      [ ñ per season vs §16.1/§16.4 lines | SYMBOL GLOSSARY | spatial summary ]
CAPTION     checkpoint, split, year, anchor-lag caveat, what the cyan cross means
```

Design decisions worth keeping:

- **The time series is swept through the model, not read from the parquet.** `eval_predict.py:110`
  drops rows with NaN `obs`, so the parquet cannot show a depth the station never measured.
  `infer_year()` runs every sample day of the target year, so all three depths appear and
  unmeasured ones are labelled *"no in-situ sensor at this depth — prediction shown,
  unvalidated"*. At Iasi that is 2 of 3 depths; hiding it would have been misleading.
- **σ, ñ and Δ moved onto the map titles** when Block B was dropped — the statistics survive
  even though the anomaly rendering does not, and the anomaly maps are still available in
  `heterogeneity_metrics.json`.
- **`interpolation="nearest"` on every SM panel.** `plot_spatial_sm_meeting.py:312` uses
  `bilinear`, which manufactures precisely the smoothness under investigation.
- **DEM/LULC/NDVI appear once**, in the context strip. The existing renderer redraws them per
  row (`:285-303`) — half the ink, none of the information.
- **Land cover is labelled with class NAMES**, from `text/modality_bands.txt:59-81`: the zarr
  stores the *remapped TerraMind indices* (0 No data, 1 Water, 2 Trees, 3 Flooded veg.,
  4 Crops, 5 Built area, 6 Bare ground, 7 Snow/ice, 8 Clouds, 9 Rangeland), **not** raw ESRI
  values, and ESRI v1 "Grass" (3) and "Scrub/Shrub" (6) are both merged into Rangeland
  (`_LULC_REMAP = [0,1,2,9,3,4,9,5,6,7,8,9]`). This is why Crossroads reads 100% class 9 and
  why reading the stored value as raw ESRI would wrongly call it Snow/Ice.
- **Row heights track panel width.** Equal-aspect `imshow` in a short cell leaves a white band
  in every row; `h_row ≈ figure_width / n_cols` removes it.

*(Superseded first draft: paired absolute + per-panel-anomaly blocks with amplitude bars under
each anomaly panel. The anomaly rendering was the right tool for "is it flat?", but once the
answer turned out to be "no, and the structure is wrong" the diagnostic value moved to the
correlations and the persistence statistic, which are numbers rather than pictures.)*

### 23.3 Metrics — so it is not an eyeball argument

Per SM panel: `σ_s = sm.std()`, `ñ = norm_std(sm)` (**`tier1_probe.norm_std`, L123, verbatim**
so it is directly comparable to the §16.1 0.0065), `Δ = p98 − p2`.

Five references, all computed in the same script and written to `heterogeneity_metrics.json`:

1. **§16.1 `ñ = 0.0065`** and **§16.4 ceiling `0.2504`** as dashed lines in the metric strip,
   with provenance in the footer. These two lines turn the strip into a verdict.
2. **Structure available in the same footprint** — `norm_std(DEM)`, `norm_std(NDVI)`,
   `#LULC classes`. "The map is flat" means nothing without "and here is what was there to
   paint".
3. **σ_space / σ_time — the headline scalar.** σ_time = std of the centre-pixel prediction
   across the target year, from `predictions_oos.parquet`. A ratio ≈ 0.02 licenses the
   sentence *"the model moves the whole field up and down in time 50× more than it varies
   across space"* — i.e. a temporally modulated constant.
4. **Physical plausibility** — Pearson r between the SM anomaly and z-scored DEM / NDVI, at
   224² and after 16×16 block-averaging to the **14×14 token grid** (the meaningful one: the
   bottleneck's native resolution, free of upsample smoothing).
   - **Report r only. No p-values** — 50,176 spatially autocorrelated pixels, effective
     n ≈ 196; every p-value would read `< 1e-300`.
   - Also report `r(anom_0-10, anom_30-100)`, and **label it a caveat, not a finding**: the
     star residual (`model.py:276-280`) makes deeper depths `base + offset`, so cross-depth
     spatial similarity is an architectural fact.
5. **Effective resolution** — block-average to 14×14, re-upsample, report the fraction of
   spatial variance retained. If ≈100%, state plainly: effective resolution is **160 m, not
   10 m**.

### 23.4 Scene selection — show what the model actually saw

`dataset.py:503` sets `anchor_rel_pos = 364 − (target_date − acq_date).days`, so

```python
anchor_date  = target_date - timedelta(days=364 - int(item["anchor_rel_pos"]))
anchor_orbit = int(item["anchor_orbit"])          # 0 = s2, 1 = s1_asc, 2 = s1_desc
```

The token store and the raw store `/projects/prjs1968/satellite_zarr/{key}.zarr` carry
**identical, index-aligned `dates`** (verified on `ISMN_RSMN_Iasi`: 164 entries, equal
element-for-element), so an exact string match retrieves the true input scene.

**Do not reuse `get_s2_rgb` (`plot_spatial_sm_meeting.py:77`)** — it takes the nearest scene
within 30 days, which is *not* what the model consumed.

**The anchor can be up to 364 days stale.** `select_anchor_zarr` (`dataset.py:488-492`)
takes the most recent fully-clear acquisition in a 365-day window, else the most recent. A
winter row may legitimately display a summer scene. **Always print the lag in days** — a
200-day lag is a finding, not a bug. Panel titles read
`S2 anchor 2021-06-27 · lag 12 d · MODEL INPUT`; when the anchor is S1 the VV panel carries
that label and the seasonally-matched RGB moves to the context strip marked
`NOT model input`.

Cloud mask is **QC only**, from the token store (`cm/masks (N,224,224) uint8`,
chunks `(128,224,224)` — read once, index in memory). Use `mean(isin(cm,(3,4,5)))`;
**not** `mean(cm != 0)` (`visualize_embeddings.py:118`), which counts water and snow as cloud
and would mark every lakeside or winter station permanently cloudy.

Season centres `--season-doys 15 105 196 288`, nearest sample within **30** days (tighter
than the existing 45 at `:193`), requiring a non-NaN label. Year = the one filling the most
season slots (replaces the blind median-year pick at `:469`), ties broken by recency.

### 23.5 Reuse — and the two loaders that must not be used

| from | function | why |
|---|---|---|
| `ckpt_utils.py:33` | `load_checkpoint` | **Only** loader honouring `use_cls_depth` (`:44`) and calling `model.eval()` (`:59`; required — `UNetDecoder.pre_head_drop` is `Dropout(0.1)`, `model.py:235`) |
| `plot_spatial_sm_meeting.py` | `_sat_zarr_path` 63, `_closest_idx` 68, `get_s1_vv` 108, `get_dem` 136, `get_lulc` 148, `_make_key` 358, `_crosshair` 336, `_no_data` 342, `_hcbar` 348 | readers + panel furniture |
| `plot_spatial_sm_meeting.py:163` | `infer_spatial_for_dates` | adapt into `infer_seasons`, also returning `anchor_rel_pos`, `anchor_orbit`, centre pixel |
| `tier1_probe.py` | `norm_std` 123, `subset_splits_csv` 99, `patch_open_zarr_no_marker` 75 | comparable metric; 3-station dataset init; held-out stores lack the `.complete` marker |

**Do not use `demo_plot.py:44` or `plot_satellite_sm_meeting.py:47`.** Both are duplicate
loaders predating `use_cls_depth`; with `strict=False` they will silently build the wrong
architecture for `cls_depth_star_reg` and swallow the mismatch. Adding this diagnostic into
`plot_spatial_sm_meeting.py` is how those two duplicates came to exist — it is Step 3 of
`slurm/evaluate_meeting.sh:44-48` and stays untouched. (Separately noted: its
`resolve_stations` at `:389` reads the stale `meeting_output/per_station_oos.csv`, not
`eval_output/`, and its default `--run-name baseline_huber` is out of date.)

### 23.6 Verification — the figure must be checked, not merely produced

1. **Selection** — `--dry-run-selection` on the login node (CPU; reads CSVs, parquet and
   zarr *metadata* only, no torch import on that path). Review every gate value before
   spending GPU time.
2. **Centre-pixel identity against `predictions_oos.parquet`**, joined on
   `(station_key, year, doy, depth)`.
   **Do not assert bit-exact equality — it will false-alarm.** `eval_predict.py:74` ran under
   `autocast(bfloat16)` at batch 128; this runs batch 1, and kernel selection is batch-size
   dependent. Run **with** autocast bf16, assert `|Δ| ≤ 2e-3` (~2.5 bf16 ulp at 0.2), and
   always print the max abs diff. **> 5e-3 means something real is wrong**: wrong checkpoint,
   wrong sample index, `model.train()` leaking dropout, or a different `category_filter`.
   Second trap: `eval_predict.py:110` drops rows with NaN `obs`, so a missing
   (station, doy, depth) must **skip with a warning**, not fail.
3. **Anchor identity, two independent derivations** — (a) the `anchor_rel_pos` arithmetic;
   (b) re-implement the candidate loop of `select_anchor_zarr` (`dataset.py:445-494`) in the
   verifier. Assert `(date, orbit)` agree. For an S2 anchor also recompute dataset.py's own
   validity rule (`isin(cm.reshape(14,16,14,16),[3,4,5,255]).mean(axis=(1,3)) <= 0.01`,
   `:471`) and assert 196 valid tokens whenever a fully-clear candidate exists.
4. **Renderer self-test** (`--selftest`) — push a synthetic checkerboard through the same
   panel-drawing function; assert the printed σ/ñ/Δ match numpy and that the anomaly panel
   shows the checkerboard. This catches *"the panel is flat because imshow got the wrong
   array"* — the failure mode that would make the entire figure a lie in the safe direction.
5. **Single source array** — assert Block A and Block B derive from the same `sm` object.
6. **Effective-resolution round-trip** doubles as a check that the map is not being read at
   the wrong scale.

### 23.7 Artefacts and commands

New: `plot_spatial_heterogeneity.py`, `slurm/spatial_heterogeneity.sh` (copy the
`slurm/tier1_probe.sh` header: `gpu_h100`, 1 GPU, 16 cpus, `--mem=120G`, **`--time=00:40:00`**,
`ulimit -n 65536`, mail flags, `conda run -n terramind`).
Outputs: `figures/spatial_heterogeneity/{station}_heterogeneity.{png,pdf}`,
`heterogeneity_metrics.json`, `selection.csv`.

Use the **`terramind`** env, not `soilmoisture` — the latter has no pyarrow, so the parquet
verification step fails there.

```bash
# 1. selection only, login node, seconds, no GPU
python plot_spatial_heterogeneity.py --dry-run-selection

# 2. one-station smoke with both verifiers, ~5 min
sbatch slurm/spatial_heterogeneity.sh --station ISMN_RSMN_Iasi --selftest \
       --verify-against eval_output/predictions_oos.parquet

# 3. full 3-station run, expected < 10 min
sbatch slurm/spatial_heterogeneity.sh
```

Runtime is dominated by dataset init and a few `(12,224,224)` int16 scene reads — inference
is 3 stations × 4 seasons = **12 batch-1 forward passes**. §16.1's `tier1_probe` built 3,430
samples over 4 stations well inside a 30-minute wall.

### 23.8 Non-goals

- **No retraining, no touching `lambda_tv`.** §16.4 already proved the decoder has the
  capacity; a retrain would answer a question that is already answered.
- **No change to `plot_spatial_sm_meeting.py`** or the meeting pipeline.
- **Not a fix.** If it confirms near-uniform output, the follow-on is dense spatial
  supervision (Phase 2) — separate work, and gated on the §22 anomaly-retargeting decision
  first.

### 23.9 Result

**The §23.0 prediction was wrong, and the real failure is worse.**

Jobs 25408177 (smoke, Iasi) and 25408407 (full, 3 stations), 2026-08-10. All verifications
passed: selftest PASS, anchors `exact=True` with lags 0-7 d, centre pixel vs
`predictions_oos.parquet` **max |Δ| = 4.88e-04** against a 2e-3 tolerance (16 matched, 20
absent because those station-depths carry no observations) — the rendered map is provably the
same forward pass that produced the §22 evaluation numbers.

**The maps are not flat. §16.1's magnitude does not hold for `cls_depth_star_reg` e16.**

| station | role | LULC | relief | ñ (0-10, over seasons) | σ | Δ p2-p98 |
|---|---|---|---|---|---|---|
| ISMN_USCRN_Tucson-11-W | best ubRMSE | 2 | **249 m** | 0.093 – 0.201 | 0.0107 | 0.044 |
| ISMN_RSMN_Iasi | heterogeneous | **5** | 75 m | 0.133 – 0.282 | 0.0246 | 0.089 |
| ISMN_SCAN_Crossroads | **flat control** | **1** | **9.5 m** | **0.317 – 0.524** | 0.0325 | **0.135** |

That is **14–80×** the §16.1 baseline of 0.0065, with several panels at or above the §16.4
dense-supervision ceiling of 0.2504. Not a like-for-like regression — §16.1 measured
`baseline_huber` epoch 11 over 4 stations — but the gap is far too large to be sampling.

**The control station inverted the expected relationship, and that inversion is the finding.**
The *flattest* footprint (1 land-cover class, 9.5 m relief, DEM ñ = 0.001, uniform NDVI)
receives the **most** spatial variation — ñ 0.52, Δ = 0.188 m³/m³ in DJF at 0-10. The footprint
with 5 land-cover classes and a river valley receives less; the one with 249 m of relief
receives least. **Structure painted is anti-correlated with structure present.**

Corroborating numbers across all four seasons and all three stations:

- `r(anomaly, DEM)` at 14×14 = **−0.04 to +0.21**; `r(anomaly, NDVI)` = **−0.10 to +0.06**.
  What is being painted is neither the terrain nor the vegetation.
- `r(season~season)` = **+0.34 to +0.84** (0-10: 0.34–0.56), against +0.03 for independent
  fields and +1.00 for a frozen template (both verified synthetically in `--selftest`). The
  pattern is **substantially persistent but not perfectly static** — it responds to inputs
  somewhat, just not to anything physical in the footprint.
- Variance retained at 14×14 = **64–83%**: most structure lives on the token grid, so
  **effective resolution ≈ 160 m, not 10 m**.
- σ_space/σ_time = **0.46–0.89**. The spatial spread is of the same order as the centre
  pixel's entire annual range — not a small perturbation on a constant.
- At Crossroads the painted spread (Δ = 0.135) is **5× that station's ubRMSE (0.0265)** and
  **3.7× its σ_time (0.0365)**.
- Visible edge artefact: a strong band down the left edge of the Crossroads DJF/SON maps.
  Decoder padding, not hydrology.

**Reframing for Phase 2.** One-pixel supervision does not produce a blank field. It leaves
50,175 pixels *unconstrained*, and unconstrained is not empty — the decoder fills them with
confident, physically meaningless structure of up to 0.19 m³/m³, strongest exactly where the
landscape is most uniform. So (a) any claim or figure about this model's spatial output is
currently indefensible, and (b) dense supervision is needed to **constrain** structure that
already exists, not to **create** structure that is missing. Different argument from §16.1's,
and a stronger one.

**Do not quote 0.0065 as the current model's behaviour anywhere.** §16.1's *mechanism* finding
(decoder-side, not attention-side) stands; its *magnitude* is obsolete.

Artefacts: `plot_spatial_heterogeneity.py`, `slurm/spatial_heterogeneity.sh`,
`figures/spatial_heterogeneity/{ISMN_USCRN_Tucson-11-W,ISMN_RSMN_Iasi,ISMN_SCAN_Crossroads}_heterogeneity.{png,pdf}`,
`heterogeneity_metrics.json`, `selection.csv`.
Logs: `logs/spatial_het_25408177.out`, `logs/spatial_het_25408407.out`.

Follow-ups, cheapest first:
1. **Is the lattice the positional embedding?** Correlate the anomaly fields *across stations*.
   High r ⇒ a fixed decoder/position artefact independent of input entirely. One-line addition
   to `pattern_persistence`, no new GPU work beyond what is already cached.
2. **Does dense supervision suppress it or merely overwrite it?** Re-run `capacity_check.py`'s
   frozen-decoder setup and report ñ *and* r(DEM) afterwards, not ñ alone.
3. **The left-edge band** points at padding in `UNetDecoder`; ten minutes with a synthetic
   constant input would settle it.

---

## §24 Is the satellite branch doing anything? — modality shuffling (Session 22, 2026-08-10)

**Status: PLANNED, implementation starting.** No retraining. One eval pass per condition.

### 24.0 The gap

**A no-satellite ablation has never been run.** The runbook contains a TV-loss ablation
(§16.4), a pyramid-attention ablation (§8.3), a bottleneck ablation (§20.12), and the
temporal masking ablation (§22.7) — all unrun — but nothing has ever tested the premise of
the whole thesis: that the TerraMind satellite tokens contribute to the prediction. It is the
first question a reviewer asks about a multimodal model, and we cannot answer it.

Three independent results now suggest the answer may be "very little":

1. **§20.14** — a ridge/GBM over soil, terrain, land cover, climate, lat/lon and SMAP, using
   **no imagery at all**, reaches station-mean RMSE ≈ 0.0576 / 0.0576 / 0.0782. The trained
   network's RMS station bias is 0.0618 / 0.0611 / 0.0875: **it ties at the surface and is
   worse at 30-100 cm than tabular regression.**
2. **§22.10** — the model beats climatology on novel-station level by 8-22%; the static probe
   got ~23% off the null. The same number, reached without imagery.
3. **§23** — the predicted spatial field is uncorrelated with terrain and vegetation and
   *anti*-correlated with landscape complexity. Not what a model consuming useful imagery
   produces.

Counter-argument to keep in view: S1 backscatter genuinely is sensitive to surface soil
moisture — it is the basis of operational Sentinel-1 SM retrievals. So a null result means
"we are not extracting it", not "it is not there".

`csvs/station_splits.csv` already carries `ablation_train` (143 stations) and `ablation_oos`
(50) — a stratified cheap-ablation subset someone designed and never used.

### 24.1 Design — shuffle, do not zero

Keep the trained checkpoint, the temporal transformer and every non-satellite input exactly
as they are. Swap only the satellite bundle between samples and re-run the evaluation.

**Zeroing is the wrong perturbation.** It puts the input off the training distribution, so a
collapse would show that the model dislikes zeros, not that it uses imagery. Shuffling holds
every marginal distribution fixed — same token statistics, magnitudes, sparsity — and destroys
only the *correspondence* between imagery and the station being predicted. That is what makes
a null result interpretable.

Two conditions, which decompose what the tokens carry:

| donor | destroys | preserves | question |
|---|---|---|---|
| **different station, same season** (±15 d) | station identity | seasonality | do the tokens say *where* we are? |
| **same station, different season** (>60 d) | temporal state | station identity | do the tokens say *when* we are? |

The first is aimed at the §20.14/§22 level failure. The second asks whether the imagery
carries any dynamic signal, i.e. whether S1 is being used as a moisture observable.

**The evidence is asymmetric, and that is fine.** No change under shuffling is *conclusive* —
the model is not using the tokens. A large drop is *ambiguous*, because breaking
correspondence is itself a perturbation; that case escalates to a retrain on the 143/50
ablation subset.

### 24.2 Modality groups — a modality moves as a whole

From `dataset.py:1083-1125`:

```
S2 history   s2_pyr, s2_doys, s2_valid, s2_rel_pos
S1 history   s1_pyr, s1_doys, s1_valid, s1_rel_pos
static img   dem_pyr, lulc_pyr
anchor       anchor_l3, anchor_l6, anchor_l9, anchor_l12, anchor_rel_pos, anchor_orbit
--- not satellite ---
soil         soil_patch
ERA5         era5, era5_doys          <- the positive control
SIF / TWSA   sif*, twsa*
```

Permuting `s2_pyr` without `s2_rel_pos` hands the model tokens whose declared time offsets
belong to a different acquisition — that tests **incoherence, not absence**. The anchor group
is the same: `anchor_l3/6/9/12` must travel with `anchor_rel_pos` and `anchor_orbit`, or the
§23 arithmetic that recovers the anchor date becomes meaningless.

### 24.3 The trap: do NOT permute along the batch dimension

`eval_predict.py:70` iterates a **non-shuffled** loader, so a batch is typically consecutive
days *from one station*. Permuting within the batch therefore yields a **within-station date
shuffle** while the run is labelled cross-station. Both conditions would come out
indistinguishable and both mislabelled — a silent, unfalsifiable error.

Use an explicit donor map instead, built from `ds.samples` metadata (`station_key`, `year`,
`doy`) before any I/O.

### 24.4 Implementation

```python
class AblationDataset(torch.utils.data.Dataset):
    """Wraps SoilMoistureDataset; replaces one modality with another sample's."""
    def __init__(self, base, modality, mode, seed=0):
        self.base, self.keys = base, MODALITY_KEYS[modality]
        rng = np.random.default_rng(seed)
        s = base.samples                       # metadata only -- no token reads
        self.donor = np.empty(len(s), dtype=np.int64)
        for i, si in enumerate(s):
            if mode == "cross_station":        # different site, same season
                cand = [j for j in doy_bucket[si["doy"] // 15]
                        if s[j]["station_key"] != si["station_key"]]
            else:                              # same site, different season
                cand = [j for j in station_index[si["station_key"]]
                        if abs(s[j]["doy"] - si["doy"]) > 60]
            self.donor[i] = rng.choice(cand) if cand else i

    def __getitem__(self, i):
        item = self.base[i]
        d    = self.base[int(self.donor[i])]
        for k in self.keys:
            item[k] = d[k]
        return item
```

Cross-station donors are season-matched (±15 d) so `rel_pos`/`doys` stay plausible and the
perturbation is confined to station identity rather than "everything at once".

`eval_predict.py` gains `--ablate {none,sat,s2,s1,anchor,dem,lulc,era5}`,
`--ablate-mode {cross_station,within_station}`, `--seed`, wraps the dataset when
`--ablate != none`, and tags output as
`predictions_{split}__{ablate}_{mode}_s{seed}.parquet`. Everything downstream
(`eval_metrics.py`, `plot_eval_boxplot.py`, `verify_level_claim.py`) then works unchanged and
yields per-station ubRMSE, bias, r, NSE_anom and level-vs-climatology skill for free.

**Cost:** `__getitem__` doubles for ablated conditions (two token bundles per sample). That is
the honest price of an exact donor map; accept it.

### 24.5 Positive control — run FIRST, not last

Shuffle **ERA5** with the identical machinery. We are confident ERA5 forcing matters.

- ERA5 shuffle collapses skill, satellite shuffle does not → the null is real.
- **ERA5 shuffle changes nothing → the harness is broken. Stop and debug.** The permutation is
  not reaching the model, and every satellite condition would be a false negative.

Same discipline as §23's `--selftest` and the reason it caught nothing silently: without this,
"nothing changed" is indistinguishable from "nothing was applied".

### 24.6 Run order

```bash
# 1. positive control first
sbatch slurm/eval_predict.sh --splits oos --ablate era5 --ablate-mode cross_station --seed 0
# 2. the actual question, three seeds -- one permutation can be lucky
sbatch slurm/eval_predict.sh --splits oos --ablate sat --ablate-mode cross_station --seed 0
#    ... seeds 1, 2
# 3. only if 2 shows a drop: identity or temporal content?
sbatch slurm/eval_predict.sh --splits oos --ablate sat --ablate-mode within_station --seed 0
# 4. attribution, if warranted
#    --ablate s2 | s1 | anchor
```

OOS only to start: 180 stations, a fraction of the 4-hour `eval_predict.sh` budget.
Baseline is already on disk and reproduces §22 to 4.88e-04 (§23.9), so it needs no re-run.

### 24.7 Interpretation

| result | reading | consequence |
|---|---|---|
| sat shuffle ≈ baseline | the model does not use the imagery | the multimodal claim does not survive as written; the question becomes *why* the tokens are ignored — and §20.14, §22.10, §23 all become one story |
| sat shuffle ≫ baseline error | genuine reliance, or perturbation shock | escalate to the retrain on the 143/50 ablation subset for clean attribution |
| era5 shuffle ≈ baseline | harness broken | stop; no satellite result is valid |
| cross-station drops, within-station does not | tokens carry site identity but no dynamics | supports the §22 level story from a new direction |

### 24.8 Couple it to §23

Re-run `plot_spatial_heterogeneity.py` under the `sat` shuffle. If the predicted map is
unchanged when the imagery comes from a *different station*, the §23 blob lattice is confirmed
as input-independent decoration, and the spatial and multimodal stories close together.

### 24.9 What else this flag unlocks

Built properly, the same mechanism covers §22.7's unrun masking ablation (zero/replace the
pre-2023 input window to test memorised-context reliance) — add `--ablate-window` later rather
than writing a second harness.

### 24.10 Non-goals

- No retraining in this section. Attribution by retrain is a separate decision, gated on 24.7.
- No change to the checkpoint, the splits, or the §22 baseline artefacts.

### 24.11 Result

**POSITIVE CONTROL PASSED (job 25412088, 2026-08-10).** ERA5 shuffled cross-station on the
`ablation_oos` subset, paired against the baseline over 112,791 rows / 36 stations:

| depth | n | ubRMSE base | ubRMSE shuffled | Δ median | r base → shuffled |
|---|---|---|---|---|---|
| 0-10 | 36 | 0.0497 | 0.0714 | **+0.0227 (+46%)** | 0.81 → 0.47 |
| 10-30 | 28 | 0.0464 | 0.0661 | **+0.0174 (+38%)** | 0.70 → 0.46 |
| 30-100 | 21 | 0.0357 | 0.0524 | **+0.0117 (+33%)** | 0.64 → 0.50 |

mean |Δpred| 0.035-0.050 m³/m³. **The permutation reaches the model and materially changes
its output, so a null result on the satellite conditions is interpretable** (§24.5 satisfied).

**SATELLITE CONDITION: THE TOKENS MATTER — MORE THAN ERA5.** Jobs 25412085/86/87,
`sat` cross_station, seeds 0/1/2. Median Δ ubRMSE vs baseline, same 36 stations, paired:

| depth | ERA5 shuffled | satellite shuffled (s0/s1/s2) |
|---|---|---|
| 0-10 | +0.0227 | **+0.0351 / +0.0343 / +0.0344** |
| 10-30 | +0.0174 | **+0.0333 / +0.0315 / +0.0316** |
| 30-100 | +0.0117 | **+0.0413 / +0.0411 / +0.0382** |

Median r: baseline 0.81/0.70/0.64 → ERA5 0.47/0.46/0.50 → **satellite 0.35/0.32/0.23**.
mean |Δpred| 0.071-0.078 (satellite) vs 0.035-0.050 (ERA5). Seed spread < 0.004.

**The §24.0 expectation was wrong: the model is not ignoring TerraMind.** Two caveats
before this is quoted as a clean win, both anticipated in §24.7:

1. **Perturbation sizes are not matched** — 16 satellite keys (including the anchor
   L3/L6/L9/L12 skips that feed the U-Net decoder directly) against 2 ERA5 keys. A larger
   input disturbance producing a larger output disturbance is not automatically "more
   informative". A fairer contrast would ablate one satellite sub-block at a time
   (`--ablate s2 | s1 | anchor`), now justified.
2. **cross_station conflates identity with content.** The model may be using the tokens as
   a **site fingerprint** rather than as a moisture observable — which would be entirely
   consistent with §20.14 and §22.10 (level memorised for seen stations, not inferable for
   novel ones). `--ablate sat --ablate-mode within_station` (job 25413066, submitted) is
   what separates them: same station's imagery, wrong date. Collapse ⇒ genuine temporal
   content; little movement ⇒ mostly an identity key.

**This does not contradict the level findings.** ubRMSE and r are anomaly quantities: the
imagery can drive day-to-day variation while still failing to supply a novel station's
absolute level. Read §24 as being about dynamics, §22.10 as being about level.

### 24.12 within_station — identity vs content, and the depth gradient (Session 23, 2026-08-11)

Job 25413066 (`sat`, `within_station`, seed 0) finished 2026-08-10 18:29 and is analysed
here. Donors: 99.8% assigned, 73 fallbacks, 0.0% from a different station, median |Δdoy|
146 d — i.e. same site, wrong date, exactly the intended perturbation.

**It is not a null.** Paired over the same 112,791 rows / 36 stations as every other
condition:

| depth | n | Δ ubRMSE, same site / wrong date | stations worse | Δ ubRMSE, wrong station | temporal share |
|---|---|---|---|---|---|
| 0-10 | 36 | **+0.0136** | 94% | +0.0351 | 39% |
| 10-30 | 28 | **+0.0089** | 93% | +0.0333 | 27% |
| 30-100 | 21 | **+0.0049** | 67% | +0.0413 | 12% |

Median r: baseline 0.81 / 0.70 / 0.64 → within_station 0.49 / 0.38 / 0.35 → cross_station
0.37 / 0.32 / 0.20. Median NSE_anom: 0.517 / 0.365 / 0.004 → 0.186 / 0.091 / −0.147 →
−0.477 / −0.819 / −4.663.

**Reading — both §24.7 rows are partly true, and which one dominates depends on depth.**

1. **The tokens do carry genuine temporal content**, and the test that shows it is
   *conservative*. `select_anchor_zarr` (`dataset.py:488`) already accepts the most recent
   fully-clear acquisition within a **365-day** window, so the model is routinely fed stale
   imagery; a 146-day median donor gap is close to normal operation. A degradation this
   large under a perturbation the model is habituated to is signal, not perturbation shock.
   This is the one condition where the asymmetry of §24.7 runs in our favour: the bias was
   toward a null and we did not get one.
2. **Site identity is nonetheless the larger component, and its share grows with depth.**
   At 30-100 the date barely matters (+0.0049, only 67% of stations worse) while the station
   matters enormously (+0.0413, 100% worse): deep prediction is close to a pure site
   fingerprint. At 0-10 the split is nearer even. That gradient is physically sensible —
   surface moisture is what optical/SAR imagery can actually observe — and it lines up with
   §20.14 and §22.10, which are about level and are strongest at depth.

**Metric trap avoided, worth recording.** Baseline median NSE_anom at 30-100 on these 36
stations is **0.004**, against 0.214 for the full 180-station OOS pool (§22). The
`ablation_oos` subset is a harder slice at depth. Its *paired deltas* are valid — that is
what it was built for — but **never quote its absolute numbers as OOS performance.**

**Harness re-verified on a fresh code path.** `compare_ablation.py` reproduces the §24.11
cross-station and ERA5 deltas to within 0.0007, independently of the one-off script used on
2026-08-10.

Tooling: `compare_ablation.py` (paired inner join on `station_key/year/doy/depth`, per-station
metrics, station-equal medians, fraction-worse); summary table in
`eval_output/ablation_summary.csv`.

```bash
python compare_ablation.py eval_output/predictions_oos_sat_within_station_s0.parquet \
                           eval_output/predictions_oos_sat_cross_station_s0.parquet \
                           eval_output/predictions_oos_era5_cross_station_s0.parquet \
                           --csv eval_output/ablation_summary.csv
```

**§24 is closed as a question.** What remains from it is optional attribution
(`--ablate s2 | s1 | anchor`, which also fixes the unmatched-perturbation problem of
§24.11 caveat 1) and the §24.8 coupling to §23. Neither is on the critical path; the open
gate is still §22's "is σ_station predictable?".

---

## §26 TxSON network run — per-station time series at every pixel + composite SM maps (Session 24, 2026-08-11)

**STATUS: RUN 2026-08-11 (jobs 25461484 smoke, 25461750 full).** The §26.1–26.9 text below was
written *before* code existed, for critique; its pre-run numbers are measured from data on disk
(station coordinates, tile bounds, zarr labels, `eval_output/manifest.json` throughput). Results
are in §26.11 and the verification PASS is back-filled into §26.6 item 3. The shipped scripts
took generalised names — `build_network_readouts.py`, `combine_network.py`, `plot_network_map.py`,
`plot_network_timeseries.py`, `plot_tile_context.py` — rather than the `*_txson.py` names planned
in §26.5. **`mosaic_txson.py` (§26.5 step 4) does not exist**, so the 417 k-overlap-pixel
disagreement map, the decisive §26.3 test, is still unbuilt and §26.7's gate is still open.

### 26.1 Motivation

Every station in this project is predicted at exactly **one pixel** — `(112, 112)` of its own
224×224 @ 10 m tile (`model.py:348-349`, `masked_huber_loss` at `model.py:751`, readout at
`eval_predict.py:76`). The model emits a full `(B, 3, 224, 224)` map (`model.py:672`), so
**50,175 of 50,176 pixels have never been compared to anything.**

§23 measured that map and found it is *not* flat (norm-std 0.093–0.524) but that the painted
structure is **anti-correlated with the landscape**: the flattest control station gets the most
structure, `r(anomaly, DEM) ≈ 0`, and 64–83 % of the variance sits on the 14×14 token grid
(effective resolution ≈160 m). §23 could not decide whether that structure is *real but
mis-scaled* or a *decoder positional artefact*, because there was no ground truth off-centre and
no second opinion on the same ground.

TxSON supplies both. **40 stations inside a 33 × 33 km domain** with 2.24 km tiles means
**27 of the 40 fall inside at least one other station's tile**, and the tiles overlap each other
heavily. One forward pass on one tile predicts up to 6 stations at once.

### 26.2 Verified facts (measured 2026-08-11, read-only)

Time-series side:

| | |
|---|---|
| TxSON stations | 40 — 14 train / 8 val / 18 oos; all have tiles + tokens |
| Network extent | 32.9 × 32.6 km — no single 2.24 km tile covers it |
| **(tile, station) readouts** | **96** = 40 centre + 56 off-centre |
| Estimates per station | 13 ×1, 15 ×2, 3 ×3, 3 ×4, 4 ×5, 2 ×6 |
| Densest tiles | CR200-18 holds 6 stations; CR200-3 holds 5; CR200-26 holds 4 |
| Depths | **all 40 are surface-only (`0-10`)** — no 10-30 / 30-100 labels anywhere in TxSON |
| Records | essentially gap-free: 2503 daily obs, 2016-01-01 → 2022-11-07 for most |
| ERA5 span | 2016-01-01 → 2022-12-31 for all 40; 53–201 S2 acquisitions per station |
| **Anchor-day bottleneck** | **none.** Samples are keyed to the anchor's observed days (`dataset.py:950-977`); checked on all 3 multi-station tiles, every member's observed days are a **subset** of its anchor's (0–1 days lost). |

Composite side — **all 40 tiles are EPSG:32614**, so the mosaic is a paste, no reprojection:

| | |
|---|---|
| Mosaic grid | 34.8 × 35.0 km = **3482 × 3499 px @ 10 m** (12.2 M px) |
| Covered by ≥1 tile | **131 km² = 10.7 %** of the bounding box |
| Covered by ≥2 tiles | **417 k px = 31.9 % of covered**, up to 6 tiles deep |
| Shape | **17 disconnected islands** — NOT a continuous map |

Overlap histogram (tiles per covered pixel): 1→892,080 · 2→257,834 · 3→83,425 · 4→40,558 ·
5→24,019 · 6→11,115.

The four islands worth mapping. Each is **homogeneous in split**, because the 3.0 km location
grouping in `create_evaluation_splits.py:22,72-92` cut them that way:

| island | area | size | tiles | split | stations |
|---|---|---|---|---|---|
| 1 | 19.79 km² | 7.09 × 3.94 km | 7 | **oos** | CR1000-6, CR200-13, CR200-19, CR200-22, CR200-28, CR200-29, LCRA-2 |
| 3 | 19.52 km² | 4.73 × 5.52 km | 8 | **oos** | CR1000-1, CR200-1, CR200-14, CR200-21, CR200-26, CR200-3, CR200-4, CR200-9 |
| 11 | 12.97 km² | 4.64 × 3.15 km | 7 | val | CR1000-2, CR200-15, CR200-18, CR200-24, CR200-25, CR200-6, CR200-7 |
| 9 | 11.57 km² | 3.81 × 4.24 km | 4 | train | CR1000-3, CR200-16, CR200-17, CR200-2 |

Island 13 is a two-tile pair (6.84 km², CR200-11 + CR200-5, train). The remaining **12 islands are
isolated single tiles** (5.02 km² each): CR1000-4, CR1000-5, CR200-8, CR200-10, CR200-12,
CR200-23, LCRA-1, LCRA-3, LCRA-4, LCRA-5, LCRA-6, LCRA-7. That is 4 large + 1 pair + 12 singles =
17. **Islands 1 and 3 are fully out-of-sample** — those are the publishable composites.

### 26.3 The overlap region is the sharpest thing in this run

417 k pixels are predicted by 2–6 *different tiles*. Two tiles seeing the **same ground pixel**
must paint the same value if §23's structure is real and driven by the land surface. If the
structure is a decoder positional artefact it is locked to **tile** coordinates, and the two tiles
will disagree — with the disagreement aligned to tile boundaries.

**This is a decisive test of §23 and it requires no ground truth at all.** It falls out of the
same forward passes for free. It is the reason this plan runs all 40 tiles rather than a minimal
covering set.

### 26.4 Compute — and why NOT to deduplicate tiles

Measured throughput from `eval_output/manifest.json`: OOS did 317,521 samples in 29.7 min ⇒
**~10,700 samples/min**.

| variant | tiles | forward passes | GPU |
|---|---|---|---|
| all 40 station tiles | 40 | ~96,000 | **~9 min** |
| greedy minimum cover | 24 | ~57,600 | ~5.4 min |

Deduplicating saves **4 minutes on a 4-hour allocation** and costs the entire §26.3 analysis:
27 stations would drop from 2–6 estimates to 1. **Run all 40.** The greedy cover is still computed
and shipped as an `in_min_cover` column, so it stays a one-line filter.

### 26.5 Implementation

1. **`build_txson_readouts.py`** (CPU) — for each of the 40 tiles read `bounds_utm`/`epsg` from
   `/projects/prjs1968/satellite_zarr/{dir}.zarr/.zattrs`, project every TxSON station with
   `pyproj.Transformer(..., always_xy=True)`, keep `0 ≤ row,col < 224` where
   `col = floor((x − west)/10)`, `row = floor((north − y)/10)`.
   → `csvs/txson_readouts.csv` (96 rows: `tile, tile_split, station, station_split, row, col,
   offset_px, dist_m, is_centre, in_min_cover, lat, lon`)
   → `csvs/txson_mosaic_grid.json` (mosaic origin/size, per-tile paste offsets, island id)
   Takes `--network TxSON` so Walnut Gulch / FMI Sodankylä reuse it later.

2. **`eval_predict.py` — add `--pixel-csv` and `--save-maps`.** Both guarded, so behaviour is
   unchanged when absent and §22/§24 artefacts stay reproducible. `--pixel-csv` replaces the
   single readout at `eval_predict.py:76` with a per-sample gather:
   ```python
   idx  = rows_b * 224 + cols_b                                             # (B, K)
   vals = mu.reshape(B, D, -1).gather(2, idx[:, None, :].expand(-1, D, -1))  # (B, D, K)
   ```
   `rows_b`/`cols_b` looked up per batch element from `station_key` (already in the sample dict,
   `dataset.py:1083-1131`); K = 6, padded with the centre index and masked. Station selection
   reuses the temp-splits-CSV pattern at `eval_predict.py:242-259`.
   → `eval_output/predictions_txson.parquet`
   `--save-maps DATES` dumps whole `(3,224,224)` maps for **12–24 listed dates only** — dense
   full-domain storage is 146 MB/date, so all 2503 dates would be 366 GB.
   → `eval_output/txson_maps/{tile}_{date}.npy` fp16

3. **`combine_txson.py`** (CPU) — join observations via `dataset._load_zarr_labels`
   (`dataset.py:174-188`), `qc == 0` only (matching `dataset.py:961-965`). **Call that function,
   never read `labels/*` by hand**: `labels/qc` is longer than `labels/sm` in many stations
   (`trim_pre2016.py` trimmed sm/dates but not qc) and lines 184-186 realign it by taking the
   trailing `n_sm` days.
   → `eval_output/txson_timeseries.parquet` (long: `station, date, depth, tile, row, col,
   offset_px, is_centre, pred, obs, station_split`, plus per-station-day `pred_own_centre,
   pred_mean_all, n_estimates, spread`)
   → `eval_output/txson_per_station.csv` via `eval_metrics.metrics_from_arrays`
   (`eval_metrics.py:53-89`), separately for own-centre and each off-centre estimate
   → tile-consistency table for the 27 stations with ≥2 estimates

4. **`mosaic_txson.py`** (CPU) — paste into the 3482 × 3499 grid; **drop 8 px from every tile
   border** before pasting (§23 found a left-edge decoder-padding artefact and the border is
   exactly where it lives); blend the 32 % overlap by distance-to-tile-centre linear taper; keep
   the unblended per-tile stack for the disagreement map. GeoTIFF via `rasterio`, EPSG:32614,
   10 m, with a `count` band and nodata over the 89 % gap.
   → `eval_output/txson_mosaic/{date}_{depth}.tif`
   → `eval_output/txson_mosaic/disagreement_{date}.tif` + summary CSV of tile-to-tile std by
   overlap depth (2,3,4,5,6)

5. **`plot_txson.py`** (CPU) — (a) 40 per-station time-series panels, obs vs own-centre vs
   neighbour-tile preds coloured by source tile; (b) island composites for the two OOS islands
   alongside S2 RGB from `/projects/prjs1968/satellite_zarr` (reuse `plot_spatial_heterogeneity.py`
   / `plot_spatial_sm_meeting.py`); (c) disagreement map with tile-boundary polygons overlaid;
   (d) ubRMSE / NSE_anom vs `offset_px`.

6. **`slurm/eval_txson.sh`** — copy `slurm/eval_predict.sh`; 1 × H100, `--cpus-per-task=16`,
   `--time=01:00:00`, mem can drop to 64 G (40 stations, not 774); keep
   `--mail-type=BEGIN,END,FAIL --mail-user=ktm.prajwalkhanal@gmail.com`.
   Checkpoint `cls_depth_star_reg/best.pt` (e16, `use_cls_depth=True`), years 2016–2022.

### 26.6 Verification (run before trusting any number)

1. **Self-test** — projecting a station into its own tile must return exactly `(112, 112)` for all
   40. Any failure invalidates everything downstream.
2. **Symmetry** — for reciprocal tiles, `(row_{A→B} − 112) == −(row_{B→A} − 112)` within 1 px.
3. **Centre-pixel reproduction** — `is_centre` rows must match existing
   `eval_output/predictions_{val,oos}.parquet` on `(station_key, year, doy, depth)`.

   **RUN 2026-08-11 on CR200-18 (job 25461484): PASS.** 2500 of 2503 rows are *bit-identical*
   (Δ = 0 through the 99th percentile). The 3 that differ do so by **exactly 0.000977 = 2⁻¹⁰,
   one bfloat16 ULP** at that magnitude — a single-ULP rounding difference, because the run is
   1 station × 2503 samples where the original was 74 stations, so the tail batch has a different
   shape and cuBLAS picks a different bf16 kernel. It is not a readout error: a standalone unit
   test shows the gather `mu.reshape(B,D,H*W).gather(2, idx)` is `torch.equal` to direct
   `mu[b,:,row,col]` indexing.

   **The gate is therefore "≥99.5 % of rows bit-identical AND max |Δ| ≤ 1 bf16 ULP", not
   `max |Δ| < 1e-4`.** The original threshold was set before measuring and is unachievable for
   any autocast path whose batching changes; do not re-tighten it.
4. **Readout count** — exactly 96 distinct `(tile, station)` pairs in the parquet.
5. **Mosaic registration** — paste a synthetic tile that is 1 at its centre pixel and 0 elsewhere;
   the mosaic must light up exactly at that station's known UTM coordinate.
6. **Smoke run** — `--pixel-csv` on CR200-18 alone (6 readouts, 1 year) before the full job.

### 26.7 Decision gates

| tile-to-tile disagreement on the 417 k overlap pixels | reading |
|---|---|
| small, and not aligned to tile boundaries | §23's structure is **ground-driven** → the 224² map carries real spatial information; §14's dense-SM deliverable is live and §16.4 Step 1 is refinement, not rescue |
| large, and aligned to tile boundaries | §23's structure is a **decoder positional artefact** locked to tile coordinates → the model outputs a point estimate dressed as a map; §16.4 Step 1 becomes mandatory and no 10 m map claim may be made |
| large, not aligned to boundaries | structure is input-driven but unstable → investigate anchor-date staleness (`select_anchor_zarr`, `dataset.py:402-505`) before anything else |

Independently, `ubRMSE` vs `offset_px` on the 56 off-centre readouts answers whether skill decays
with distance from the supervised pixel.

### 26.8 What this will NOT show — state these in any write-up

- **Nothing at 10-30 or 30-100 cm.** TxSON is surface-only. Deep predictions come out of the model
  but have no ground truth here.
- **The composite is 10.7 % of the domain in 17 islands.** It is not a continuous SM map of TxSON
  and must not be presented as one.
- **Split contamination.** 14 of 40 stations are `train` (memorised) and 8 are `val` — no
  gradients, but `best.pt` was *selected* on val loss, and the whole Fredericksburg cluster
  (island 11) is val. Only the 18 `oos` stations, including islands 1 and 3, are fully clean.
  Carry the split label on every table and figure; do not pool into one headline number.
- **One catchment.** Results characterise TxSON, not the model in general.

### 26.9 Deliberately out of scope

A **continuous** mosaic over the full 33 × 33 km domain needs ~225 gridded tile centres with fresh
S2/S1/DEM/LULC downloads plus TerraMind tokenization — a separate project. §26 mosaics only tiles
that already exist on disk.

### 26.10 Related colocation inventory (measured, superset of TxSON)

The same projection sweep across **all** 993 pipeline stations plus the 482 level-1 stations that
never entered the pipeline gives **33 colocation clusters covering 105 stations**, 206 ordered
in-tile pairs, 164 with an unseen SM target, **95 usable** at ≥365 d record overlap (49 anchors,
53 targets). Restricting to ≥160 m separation (>1 token) leaves **33 unique pairs / 61 directed
runs**, of which 25 are TxSON, 13 are val-only, and just 2 have a *trained* anchor tile with a
never-trained target (`ISMN_ARM_Lamont-CF1 → AmeriFlux_US-ARM` at 351 m;
`ISMN_LABFLUX_Bussolenobosco → ISMN_LABFLUX_Bussolenoprato` at 176 m). Temporal overlap, not
geometry, is the binding constraint: 64 in-tile pairs are lost to non-overlapping records,
including the 7-node SOILSCAPE deployment inside Tonzi Ranch's tile.

### 26.11 Results (run 2026-08-11, jobs 25461484 smoke / 25461750 full)

96 readouts (40 own-centre, 56 off-centre) over 40 tiles, `cls_depth_star_reg/best.pt` e16,
2016–2022, surface only. `eval_output/manifest.json` records the TxSON split: 40 stations,
657,354 rows, 90,779 samples, `nan_pred: 0`, 7.5 min, `n_readouts: 96`, `n_offcentre: 56` —
matching the §26.2 prediction exactly.

**Own-centre vs off-centre** (`eval_output/txson_per_readout.csv`):

| arm | n | stations | ubRMSE | RMSE | RMS bias | R | NSE_anom | pred_sd | obs_sd |
|---|---|---|---|---|---|---|---|---|---|
| own_centre | 40 | 40 | 0.0301 | 0.0512 | 0.0511 | 0.868 | 0.727 | 0.0575 | 0.0578 |
| off_centre | 56 | 27 | 0.0345 | 0.0606 | 0.0623 | 0.825 | 0.577 | 0.0549 | 0.0549 |

Paired on the 27 multi-estimate stations: median ΔubRMSE (off − own) = **+0.0025**, 74% worse
off-centre. Split-stratified own-centre ubRMSE: train **0.0110** (memorised) vs val 0.0299 vs
oos 0.0386 — and off-centre for those same train stations degrades to 0.0500. **The
memorisation does not travel off the supervised pixel.**

**The across-station spread is the finding.** Per `eval_output/txson_timeseries.parquet`,
depth 0-10, the four densest tiles:

| tile | n_stn | between-station SD pred | obs | % of obs | level range pred / obs | r(pred level, obs level) |
|---|---|---|---|---|---|---|
| CR200-18 | 6 | 0.0116 | 0.0661 | **17%** | 0.024 / **0.171** | **−0.135** |
| CR200-3  | 5 | 0.0111 | 0.0578 | **19%** | 0.013 / 0.120 | +0.012 |
| CR200-26 | 4 | 0.0103 | 0.0552 | **19%** | 0.008 / 0.123 | **−0.589** |
| CR1000-2 | 6 | 0.0102 | 0.0661 | **15%** | 0.012 / 0.171 | −0.077 |

Within-station (temporal) SD is reproduced almost perfectly (0.051 vs 0.051; 0.058 vs 0.056)
while between-station SD is reproduced at 15–19%, and the *ordering* of station levels is
uncorrelated to anti-correlated with truth. Every tile falls under
`plot_network_timeseries.py`'s 0.35 threshold ⇒ **"the map repeats ~one series over the whole
tile."** Figures: `figures/tile_context/ISMN_TxSON_CR200-18.{png,pdf}`,
`figures/network_timeseries/ISMN_TxSON_CR200-18_0-10.{png,pdf}`.

**§26.3 remains untested.** `mosaic_txson.py` was never written, so the tile-to-tile
disagreement on the 417 k overlap pixels — the test that decides whether §23's structure is
ground-driven or a decoder positional artefact — has not been run. §26.7's gate is open.

---

## §27 Is sub-km SM contrast recoverable from TerraMind embeddings? (Session 25, 2026-08-12)

**STATUS: DESIGNED, NOTHING RUN.** Full derivation, flowchart and limitations in
`text/subkm_design.pdf` (source `text/subkm_design.tex`). Written before code exists, for
critique. This section is the runbook summary; the PDF is authoritative.

### 27.1 What §26.11 leaves open

§26.11 shows the model reproduces temporal variance at a station almost perfectly and
between-station variance at 15–19% with the wrong ordering. §23 showed the painted map has
structure but anti-correlated with the landscape. §24.12 showed site identity dominates.
None of these say **whether the information to do better exists in the inputs at all.**

Measured 2026-08-12 from `eval_output/predictions_oos.parquet` (depth 0-10, station-mean level):

| scale | n | sd_obs | sd_pred | ratio | r |
|---|---|---|---|---|---|
| between networks | 21 | 0.0578 | 0.0525 | 0.91 | **+0.85** |
| within USCRN | 12 | 0.0967 | 0.0851 | 0.88 | +0.89 |
| within SCAN | 33 | 0.1029 | 0.0868 | 0.84 | +0.78 |
| within SNOTEL | 73 | 0.0715 | 0.0442 | 0.62 | +0.31 |
| within COSMOS-UK | 6 | 0.0525 | 0.0356 | 0.68 | −0.05 |
| within TxSON | 18 | 0.0633 | 0.0158 | **0.25** | **−0.11** |
| within one tile | 6 | 0.0601 | 0.0113 | **0.19** | **−0.18** |

**Level skill decays monotonically with separation and vanishes below ~10 km.** It is not a
readout artefact: reading each TxSON station from its OWN centred patch — the training
configuration — still gives sd_pred 0.0158 vs sd_obs 0.0633, r = −0.11.

For the six CR200-18 stations, ~95% of the input is shared: ERA5-Land (~9 km) is byte-identical
(`dataset.py:1045`) and supplies 365 of ~600 tokens; SIF and TWSA identical; OpenLandMap soil
varies by 3% clay / 4% sand / 1 pH unit; and **the finest satellite scale the model sees is
320 m, not 160 m** — `dataset.py:212` gives `widths=[1,3,5,7]` ⇒ 2×2/6×6/10×10/14×14 token
windows, contradicting the `model.py:109` docstring, with `dataset.py:306` dropping 50% of
tokens *before* pooling. A homogeneous prediction is close to Bayes-optimal given these inputs.

### 27.2 Design — pair differences

Scope is TerraMind embeddings ONLY. Terrain derivatives, 30 m optics, POLARIS soil,
multi-station supervision and any loss change are out of scope and stay deferred.

Model `y_i = μ(x_i) + b_i` (station offset b: calibration, installation, bulk density),
field `μ(x) = m(z(x)) + ε(x)` with z regional on scale L≈100 km. For a pair at separation
d ≪ L, `Δμ = ∇m·Δz + Δε → Δε`: **the pair difference is a high-pass filter** that annihilates
exactly anything tile-constant — climate, ERA5 cell, season, biome, network, sensor vendor,
calibration epoch, and (Arm A) the whole image context of the forward pass. Hence
`Cov(Δe,Δy) = Cov(Δe,Δε) + Cov(Δe,Δb)` with the second term zero in expectation, since sensor
idiosyncrasy is not visible from orbit. **A significant association at small d can only come
from genuine short-range field information in the embedding.**

Ceiling: `Var(Δy) = 2γ_ε(d) + 2σ_b²` ⇒ **`R²_max(d) = γ_ε(d)/(γ_ε(d)+σ_b²)`**. As d→0 only
instrument noise remains, so below some separation NO covariate can predict Δy. σ_b² is the
variogram nugget and is estimable from labels alone — **this runs first**; a null reported
without it is uninterpretable. For CR200-18, γ̂(0.6 km) ≈ 0.0601² = 0.0036.

Three arms: **A** both tokens from one tile's 14×14 grid (what TerraMind knows); **B** each
station's own patch centre token (what the model is fed); **C** the 4-scale pooled pyramid
(what pooling leaves). Pooling is a fixed map P, so by the data-processing inequality
`I(P·e;y) ≤ I(e;y)` ⇒ **ρ_C ≤ ρ_A necessarily** — the comparison is one-sided and cannot
reverse by chance. `ρ_A > 0` with `ρ_C ≈ 0` *proves* pooling is the bottleneck.

Statistic: distance correlation, `dCor = 0 ⟺ independence`, no fitted parameters (overfitting
structurally impossible), exact permutation null under label exchangeability. Then a 1-dof
scalar test `corr(‖Δe‖, |Δy|)`, then RidgeCV (no intercept) / PCA(16)→GBM with GroupKFold on
tile and network. Antisymmetry (both orderings emitted) makes `E[Δy]=E[Δe]=0` exactly, so a
constant predictor sits exactly at the null.

Verified storage: `s2/{l3,l6,l9,l12}` and `s1_{asc,desc}/{l3,l6,l9,l12}` all present as
`(N,196,768)` fp16; **`dem` and `lulc` are L12-only** `(196,768)`. Sweep = 14 combinations
× 3 arms. Hypothesis: **early layers beat L12 on within-tile contrast even though L12 beats
them on site identity** (§24.12).

### 27.3 Sample size — the number that constrains everything

A great-circle sweep over the 993 pipeline stations gives 126 pairs < 2.24 km across 12
networks (TxSON 55, FMI 22, SNOTEL 13, AmeriFlux 11, SCAN 5, ICOS 4, ARM 3, FLUXNET-AMERIFLUX
3, SOILSCAPE 3, iRON 2, Berlin 1, NGARI 1), then 178 at 2.24–5 km, 283 at 5–10, 1420 at 10–30,
4287 at 30–100, 486,234 beyond.

**But §26.10 already established that geometry is not the binding constraint.** Of 206 ordered
in-tile pairs, only 95 survive the ≥365 d record-overlap requirement, and requiring ≥160 m
separation — necessary for Δe ≠ 0 in Arm A, since a token is 160 m — leaves **33 unique pairs**.

> **Arm A's real n is ≈33 (66 antisymmetrised), not 126.** This is the single most important
> number in the design. It is why the ladder starts with a parameter-free statistic and an
> exact permutation test, and why Arm B and the ρ(d) curve (n in the thousands) carry the
> statistical weight. A null on Arm A is underpowered, NOT evidence that TerraMind cannot.

The ~62 sub-token pairs are not wasted: in Arm A they have **Δe ≡ 0 exactly by construction**,
making them simultaneously a perfect negative control and a direct upper bound on σ_b² via
`Var(Δy | same token) = 2γ_ε(d<160m) + 2σ_b²`.

**One structural advantage over probing the model:** the TerraMind encoder is frozen and was
never trained on soil moisture, so train/val/test leakage does not apply. Every colocated
station is usable regardless of split — which matters when colocated stations are this scarce.

### 27.4 Controls

Antisymmetry (constant predictor ≡ null) · label shuffle within folds (all 14 combinations must
collapse) · same-token pairs (Δe ≡ 0, any skill is a bug) · **far distance bins must be
significant — the built-in positive control, since between-network r = +0.85 is already
measured** · random 768-d vector (dCor ≈ 0) · independent positive control on Δelevation and
Δsoil-texture at the station pixel, the analogue of §24's ERA5 shuffle.

Pre-registered primary test: **depth 0-10, Arm A, S2, all distance bins.** Everything else is
exploratory under Benjamini-Hochberg FDR. Do not gate on a max-over-tests.

### 27.5 Decision gate

Read jointly with `R²_max(d)` — a probe that reaches the ceiling has succeeded even if its
absolute r looks modest.

| Arm A | Arm C | reading | next move |
|---|---|---|---|
| signal | signal | tokens carry it, model is fed it, model fails to use it | training/loss problem → multi-station supervision, level/anomaly split |
| signal | none | **pooling destroys it** (one-sided, by the DPI) | restore a true 1×1 centre scale (`dataset.py:212`), exempt centre token from the 50% dropout (`dataset.py:306`) — cheapest fix in the project |
| none | none, R²_max ≫ 0 | TerraMind at 160 m does not resolve sub-km SM contrast | state the resolution limit; reframe on ubRMSE / NSE_anom |
| none | none, R²_max ≈ 0 | the *observations* cannot resolve it — offset noise exceeds field contrast | not a model finding; report the nugget and stop |

Valid only if the far bins are significant and the random-vector control gives dCor ≈ 0.

### 27.6 What gets built (nothing yet)

`build_network_readouts.py --all-networks` → `csvs/all_readouts.csv` (reuse the `:214` centre
assertion and `:230` symmetry check; must reproduce `csvs/txson_readouts.csv` on TxSON) ·
`probe_variogram.py` (labels only, minutes, **runs first**) · `probe_terramind_subkm.py`
(CPU-only, `Pool(64)`, opens zarr directly — never instantiates `SoilMoistureDataset`, which
eagerly loads ~16 GB of L12 tokens; reuses `_load_zarr_labels` and `_cpu_pyramid_pool`) ·
`slurm/probe_terramind_subkm.sh`. Outputs `csvs/terramind_subkm_probe.json`,
`figures/subkm_layer_sweep.png`, `figures/subkm_rho_curve.png`.

DEM/LULC at L3/L6/L9 do not exist; getting them needs a GPU re-run of
`precompute_terramind.py`. **Gated** — do not pay unless the S2/S1 sweep shows early layers
matter.

Verification before any number is trusted: (i) every station projects into its own tile at
exactly (112,112); (ii) the six CR200-18 means reproduce 0.1367 / 0.1197 / 0.1826 / 0.2323 /
0.1857 / 0.2865 to 1e-3; (iii) a centre station's Arm A and Arm B tokens are bit-identical;
(iv) ‖Δe‖ = 0 exactly on same-token pairs; (v) Σ Δy = 0 over the antisymmetrised table;
(vi) all 14 combinations collapse under permuted labels.

### 27.7 Scope narrowed: it is not "is the embedding diverse", it is "is it the RIGHT diversity"

**Decision, 2026-08-12.** An earlier draft of this section proposed a label-free pre-check
(do the 196 tokens inside a tile differ at all; is the variation structured or noise) before
spending the scarce colocated stations. **That pre-check is dropped.** §15's Tier-0
diagnostic already measured per-token L2 norm over the 14×14 grid, PCA→RGB structure,
off-diagonal cosine and neighbour autocorrelation (`visualize_embeddings.py`,
`embed_viz_output/`). Token diversity within a tile is established; re-measuring it would
answer a question we have already answered.

The open question is narrower and is the one §27 exists for:

> The tokens are diverse. Is that diversity **wide enough, and of the right kind, to cover
> the diversity of soil moisture** — or is it diversity about vegetation and terrain that
> happens not to track wetness?

This is exactly the Δe-vs-Δy pair test of §27.2. What changes is one cheap addition that
makes a null interpretable.

**The reference axis.** Run the identical estimator, on the identical tokens and the
identical pairs, with two *other* targets alongside soil moisture:

| target | source | role |
|---|---|---|
| Δ(soil moisture level) | station labels | the question |
| Δ(land-cover class) at the station's own 160 m token footprint | `lulc` raster, `/projects/prjs1968/satellite_zarr/{station}.zarr` | reference axis |
| Δ(elevation, slope) at the same footprint | `dem` raster, same store | reference axis |

Token `(r, c)` covers pixels `[16r:16r+16, 16c:16c+16]`; the tile mean must be removed from
both sides, or the probe wins by recognising *which image* it is looking at rather than
*where in the image*.

Three readings, and the third is the one worth having:

| SM | land cover / terrain | reading |
|---|---|---|
| signal | signal | the diversity covers soil moisture; the failure is the model's |
| none | **signal** | **the diversity is real but is about vegetation and terrain, not wetness** — the honest answer to "is it diverse enough", and a publishable negative |
| none | none | the probe or the pairing is broken; fix before interpreting anything |

The land-cover axis doubles as a **guaranteed positive control**: the `lulc` tokens are
TerraMind's own encoding of the land-cover raster, so `LULC token → LULC class` at 160 m
must score near-perfectly. If it does not, the fault is in the pipeline, not in the
representation. It also sets the ceiling against which the S2/S1 tokens are measured.

Cost: nil beyond §27 as already specified — same stations, same tokens, same estimator, two
extra target columns read from rasters already on disk.

### 27.8 Correction to the pooling claim in §27.1

§27.1 states "the finest satellite scale the model sees is 320 m, not 160 m". Reading
`model.py:486-530`, that is **wrong as a blanket statement**:

| what the model receives | resolution | where |
|---|---|---|
| anchor acquisition, L12 | **all 196 tokens, unpooled**, + 2-D positional encoding | `_get_target_spatial_tokens`, `model.py:497-505` |
| anchor acquisition, L3/L6/L9 | **all 196 tokens** → (768, 14, 14) decoder skips | `_get_skip_connections`, `model.py:519-530` |
| S2/S1 **history** | pooled to 4 scales, finest 2×2 = 320 m | `_pyramid_from_l12`, `model.py:446-475` |
| **DEM and LULC** | pooled to 4 scales, finest 2×2 = 320 m | `_cpu_pyramid_pool`, `dataset.py:190-220` |

The 320 m claim holds for the history and for DEM/LULC; the anchor image arrives at full
160 m in four layers. This sharpens rather than weakens the hypothesis:

> **DEM and LULC are precisely the static terrain and land-cover signals that would drive
> persistent within-tile wetness differences — and they are exactly the ones pooled away to
> ≥320 m.** The anchor, which *is* delivered at 160 m, is one date's reflectance:
> informative about vegetation state, far less about persistent wetness.

Arm C of §27.2 is therefore a targeted measurement, not a blanket one: run the §27.7 probe
on DEM and LULC twice — raw 196 tokens, then through `_cpu_pyramid_pool` with and without
the 50% dropout of `dataset.py:306` — and measure the drop. Pooling can only lose
information, so the comparison is one-sided by construction. The `model.py:109` docstring
claiming a 1×1 (160 m) centre scale remains wrong and should be corrected when that file is
next touched.

## §27a Token PCA→RGB maps, and the massive-activation registers they exposed (Session 25, 2026-08-12)

**STATUS: RUN 2026-08-12.** Jobs 25521508 / 25521717 / 25521875 (figures), 25523711 (§27a.3),
25525365 (§27a.4), 25525765 (§27a.5), 25526680 (§27a.6).

Code: `plot_token_pca.py`, `audit_static_token_outliers.py`, `audit_layernorm_compression.py`,
`audit_register_dim_variance.py`, `audit_register_across_modalities.py`, each with a
`slurm/` wrapper (CPU, `--cpus-per-task=64`, `Pool(64)`).
Figures: `figures/token_pca/ISMN_TxSON_CR200-18_{s2,s1_asc,s1_desc,static}.{png,pdf}`.
Tables: `csvs/{static_token_outliers,layernorm_compression,register_dim_variance,register_across_modalities}.*`

Motivation (§27.7): not "are the embeddings diverse" — §15 Tier-0 settled that — but **is the
diversity of the right kind and magnitude to cover soil moisture**. Look before probing.

### 27a.0 Method

Tile `ISMN_TxSON_CR200-18`, 2019–2020, six stations in **six distinct tokens** (105, 62, 100,
20, 44, 172), observed mean SM 0.1197–0.2865. One PCA basis per (modality, layer) fitted
across all four seasons at once, deterministic component sign, each season centred by its own
valid-token mean, one colour scale per row spanning raw and pooled. `visualize_embeddings.pca_rgb`
does none of this — it refits per panel with an arbitrary SVD sign, so its colours are not
comparable between panels. Pooling replicated in numpy and **asserted equal to
`dataset._cpu_pyramid_pool`**; the 14×14 window equals the plain masked mean. Every symbol on
the figure is defined in a printed legend block.

### 27a.1 Raw tokens are spatially heterogeneous, and the detail sits in the early layers

**Two statistics were dropped from this section on 2026-08-12 and should not be reinstated:**
the per-token L2 norm map and Moran's I computed on it. Both are magnitude summaries, and
§27a.3-27a.6 showed a single register coordinate dominates a token's magnitude by construction
(one entry at -1671 while every other sits within ±82). They therefore measure the register
more than the token. Nothing below depends on them.

What the tile figure shows, on evidence independent of any norm:

**1. The tokens are genuinely heterogeneous within the 2.24 km tile.** The 196 tokens differ
from one another, and PCA variance is spread across many components rather than concentrated
in one direction — i.e. high effective spatial rank, with sink tokens excluded from the fit:

| modality/layer | PCA top-3 variance ratio |
|---|---|
| S2 L3 | 0.13 / 0.07 / 0.06 |
| S2 L6 | 0.10 / 0.06 / 0.05 |
| S2 L9 | 0.08 / 0.07 / 0.04 |
| S2 L12 | 0.45 / 0.05 / 0.04 with sinks, **0.09 / 0.07 / 0.05** without (4 sink tokens) |
| DEM L12 | 0.65 / 0.05 / 0.04 with sinks, **0.14 / 0.11 / 0.06** without |
| LULC L12 | 0.36 / 0.13 / 0.07 with sinks, **0.20 / 0.10 / 0.08** without |

The apparent "one dominant component, low spatial rank" at L12 and on the statics was
**entirely the register artefact**. Once it is excluded, every layer looks high-rank and
mutually comparable — S2 goes 0.13 / 0.10 / 0.08 / 0.09 across L3 / L6 / L9 / L12. Note the
sink tokens appear only at L12 (4 of them for S2; none at L3/L6/L9), which is the same depth
signature §27a.6 finds independently.

**2. The heterogeneity is landscape-correlated** — the two positive controls, both passed
visually: the LULC PCA→RGB reproduces the land-cover map (rangeland 72%, crops 15%, trees 11%),
and the DEM PCA→RGB traces the drainage seen in the hillshade (435-481 m, sd 7.9 m).

**3. Spatial detail is richest in the early layers.** L3/L6 show visibly finer texture than
L12, consistent with the §27.7 prediction logged before running and with §27a.6, where the
high-magnitude register token is an L12 phenomenon (S2 1.5x at L9 -> 7.2x at L12) while L3/L6
stay clean.

So: **TerraMind embeddings do carry within-tile spatial heterogeneity at 160 m, and it tracks
terrain and land cover.** What is NOT shown here is that the heterogeneity tracks soil
moisture -- that is §27b.

### 27a.2 Pooling destroys 97–98% of within-tile position information

`variance kept` = fraction of within-tile variance surviving when every token is replaced by
its smallest containing nested window — everything the 4 pooled vectors can express about
position inside the tile:

| modality | variance kept |
|---|---|
| S2 L3 / L6 / L9 | **3%** (all four seasons) |
| S2 L12 | **2%** |
| DEM | **1.5%** (3.1% excluding the §27a.3 sink token) |
| LULC | **2.6%** (3.8% excluding it) |

The pooled panels are four flat nested squares, indistinguishable across seasons — pooling
removes the seasonal signal along with the spatial one. The 50% token dropout adds a
draw-to-draw spread of 2–4% of the colour range on top.

Expected in hindsight and no less damning: after the tile mean is removed, three nested annuli
remain, and they can only express structure that is radially symmetric about the station.

**Scope it correctly (§27.8):** the anchor acquisition enters **unpooled** — 196 tokens
(`model.py:497-505`) plus L3/L6/L9 as decoder skips (`model.py:519-530`). Only the S2/S1
**history** and **DEM/LULC** are pooled.

### 27a.3 One token holds most of the variance — and the raw raster is pristine

| modality | top token | ‖e‖ | median ‖e‖ | share of within-tile variance |
|---|---|---|---|---|
| DEM | (r10, c2) | 1682.2 | 123.1 | **64.8%** |
| LULC | (r8, c3) | 999.1 | 124.6 | 35.7% |

**The input is fine.** The DEM under that token is 452.88–454.55 m, 254 distinct values, smooth,
**zero NaN**. The LULC footprint is uniform class 9 (rangeland). No nodata, no fill, no
resampling artefact.

**The embedding is not.** `min = −1671.00, max = +82.19, mean = +1.756` — the whole norm is one
coordinate. That is a ViT massive-activation / attention-sink register.

Sweep over 993 stations: **970 (97.7%)** have a DEM token >3× the median norm; median ratio
**13.0×**, p90 34×, max 152×; **median variance share 66.7%**, p90 93.1%. LULC: 920 (92.6%),
median 9.2×, median share 40.4%. **Zero are masked by `dem_token_mask` / `lulc_token_mask`.**
Only 1–4% sit on the patch border, so it is not a padding artefact.

It contaminated our own numbers: excluding sinks from the PCA fit moved the DEM basis from
`0.65/0.05/0.04` → `0.14/0.11/0.06` and LULC `0.36/0.13/0.07` → `0.20/0.10/0.08`. The apparent
"one dominant component, low spatial rank" was **entirely the artefact**.

### 27a.4 LayerNorm is per-token, so pooling is what does the damage

LayerNorm normalises each token over its own 768 features, so a sink token cannot contaminate
its neighbours — it only wrecks itself (post-LN, **98.9%** of its squared norm is register), and
it is 1 token in 196. Harmless on the unpooled path.

But pooling is a mean, and a mean does not remove a common offset. Using the **trained**
`transformer_layers.0.layer.norm1` γ/β from `cls_depth_star_reg/best.pt`:

**DEM** — |sink|/σ_other, compression, and post-LayerNorm share in the register dims:

| window | \|sink\|/σ_other | compression | post-LN share |
|---|---|---|---|
| 2×2 (320 m) | 29.3 | 1.46× | **0.489** |
| 6×6 | 41.7 | 1.83× | 0.670 |
| 10×10 | 45.7 | 1.98× | 0.714 |
| 14×14 | 46.9 | 2.00× | **0.720** |

LULC is milder (0.138 → 0.284; compression 1.09–1.22×). The register coordinate is **dim 328**,
shared by **100%** of affected stations in both modalities — median 4 spiking coordinates per
tile for DEM, 2 for LULC.

### 27a.5 The register is a near-constant offset: dilution, not signal

Detected here by a different criterion — dims with a large **mean across stations**, i.e. a
constant offset in every token of every tile. That finds **87 and 126** (DEM), **126** (LULC),
which are *not* the same as the within-token spike dim 328.

**DEM, 14×14 window:** dims 87+126 carry **88.4% of the mean vector's magnitude** but only
**13.9% of the across-station variance**; CV 0.315 vs **2.00** for a typical dim (6× less
variable). Decisively:

> **Median pairwise cosine between stations, post-LayerNorm: 0.783 → 0.157 when those 2 of 768
> dims are zeroed.** Holds at every window (0.577→0.083 at 2×2).

Two coordinates are what make every station's DEM representation look like every other's.
LULC is milder: 35.4% of magnitude, 12.5% of variance, 0.694 → 0.604.

Not established: that the residual 16%-cosine content is *informative about soil moisture*. A
lower cosine means the vectors are no longer dominated by a common direction; it does not prove
the remainder is useful. That is still §27b's question.

### 27a.6 It is NOT a DEM/LULC problem — it is encoder-wide, and worst at L12

The obvious explanation was that registers get recruited in low-information patches, and
DEM/LULC are single-band, static and often near-uniform. **That is not what the data says.**
Sweep over all 993 stations, one acquisition each, every modality × layer:

| modality/layer | register dims | mag share | var share | token max/median L2 norm* |
|---|---|---|---|---|
| dem/l12 | 87, 126, 328 | 0.884 | 0.139 | **13.0** |
| lulc/l12 | 126 | 0.354 | 0.125 | **9.1** |
| s2/l12 | 9, 87, 126, 328, 329 … | **0.940** | 0.083 | **7.2** |
| s2/l9 | 9, 126, 329, 723 | 0.615 | 0.041 | 1.4 |
| s2/l6 | 9, 126, 329, 437 | 0.603 | 0.040 | 1.5 |
| s2/l3 | 9, 126 | 0.586 | 0.017 | 1.5 |
| s1_asc/l12 | 87, 126, 716 | **0.912** | 0.038 | 1.2 |
| s1_asc/l9 | 87, 126, 329 | 0.723 | 0.105 | 1.3 |
| s1_asc/l6 | 126, 329 | 0.687 | 0.138 | 1.3 |
| s1_asc/l3 | 126, 459 | 0.715 | 0.038 | 1.2 |
| s1_desc/* | as s1_asc | 0.69–0.91 | 0.04–0.16 | 1.2–1.4 |

\* the norm column is a **detector**, not a content measure — it is the cheapest way to see the
L12 spike emerge, but a norm is dominated by the register coordinate by construction, which is
why §27a.1 no longer reports norm-based statistics.

Three conclusions:

1. **Dim 126 is a register in EVERY modality and EVERY layer.** Dim 87 in every modality at
   L9/L12. Dim 328 at L12 in dem and s2. These are properties of the TerraMind encoder, not of
   the input.
2. **The high-norm TOKEN is an L12 phenomenon.** S2 jumps 1.5× (L9) → **7.2×** (L12); S1 never
   develops it at any layer (1.2–1.4×); DEM is the extreme at 13.0×. Classic depth signature of
   massive activations.
3. **The dilution is universal and worst where it hurts most.** Magnitude share is 0.59–0.94
   everywhere while variance share stays 0.02–0.16 — near-constant directions dominating the
   magnitude in every modality. **S2 L12 is the worst of all at 0.940**, and S2 L12 is exactly
   what `_get_target_spatial_tokens` (`model.py:497`) feeds in as the 196 spatial tokens that
   become the **decoder bottleneck**. L3/L6/L9 are markedly cleaner (0.59–0.72).

Input redundancy is at best a weak modulator, not the cause: `r(DEM relief, register strength)`
= **−0.217** (flattest quartile 17.4× vs most-varied 10.2× — right direction, weak), and
`r(LULC purity, strength)` = **−0.018**, i.e. nothing. Explanation (A), encoder-wide, with a
minor input modulation for DEM only.

### 27a.7 What this changes

- The earlier framing "a DEM/LULC pooling problem" was too narrow. **The decoder bottleneck
  (S2 L12, 196 unpooled tokens) is 94% register by magnitude**, which sits directly upstream of
  §23's finding that the painted 224² structure is anti-correlated with the landscape.
- **L3/L6 are cleaner on both counts** — more within-tile spatial structure (§27a.1) and far
  less register dominance (§27a.6). Anything aiming at fine resolution should be built from
  them, not from L12.
- **The fix already exists in this codebase.** ERA5 is globally z-scored per variable
  (`csvs/era5_stats.json`, `dataset.py:1046-1048`); the satellite and static embeddings get **no
  normalisation at all**. Per-dimension standardisation over the dataset — the same thing
  `compute_era5_stats.py` does for ERA5 — would put dims 9/87/126/328/329 on the same footing as
  every other coordinate. It discards nothing: standardisation keeps whatever across-station
  variance those dims carry, it just stops them dominating the magnitude. Zero training-time cost.
- **Open item:** `dem_token_mask` / `lulc_token_mask` flag none of these tokens. Whether sink
  tokens should be excluded from the pooled masked-mean is a separate, cheap decision.

### 27a.8 What this does NOT show

Nothing here shows the token structure tracks **soil moisture**. It demonstrably tracks terrain
and land cover (the positive controls). Whether wetter stations sit on systematically different
tokens is §27b, and it is now worth running because §27a.1 removed the reason it might have been
pointless — but it should lead with **L3/L6**, not L12.

## §27b Do TerraMind embeddings predict mean soil moisture? (Session 25, 2026-08-12)

**STATUS: DESIGNED. Scale A being built; B and C designed, not built.** Written before code
exists, for critique.

Scales are labelled **A / B / C**, not S1/S2/S3 — S1 and S2 already mean Sentinel-1 and
Sentinel-2 throughout this project.

### 27b.1 What §27a leaves open

§27a established that the embeddings **do** carry within-tile spatial heterogeneity at 160 m
and that it tracks the landscape (DEM PCA→RGB traces the drainage; LULC reproduces the tree
patches). Sink-excluded PCA ratios are high-rank at every depth: S2 L3 0.13 · L6 0.10 ·
L9 0.08 · L12 0.09. So "the imagery just looks the same at those six spots" is dead.

Nothing in §27a shows the heterogeneity tracks **soil moisture** — it demonstrably tracks
terrain and land cover. That is what §27b measures.

### 27b.2 One measurement, three scales

The direct framing (token → mean SM) and the pair framing (Δe vs Δy, dCor) are **not
alternatives**. Differencing a pair *is* removing a group mean. Same measurement, group made
progressively smaller, trading confound-removal against sample size:

| scale | confound removal | n | question |
|---|---|---|---|
| **A — global** | none | ~993 stations | is SM predictable from the embedding at all? |
| **B — within-network** | subtract the network mean from X and y | ~993 stations | does it survive removal of climate and region? |
| **C — within-tile** | pair differences inside one 2.24 km tile, dCor | **~33 pairs** | does it separate two places a few hundred metres apart? |

A is well-powered but will partly succeed *via climate* — L12 separates Texas from Norway.
C removes climate, the ERA5 cell, season, biome, network, vendor and calibration epoch
**exactly**, because both tokens come out of one forward pass on one image; but §26.10 caps it
at 33 usable pairs (206 in-tile pairs → 95 with ≥365 d overlap → 33 above 160 m separation,
the minimum for Δe ≠ 0 since a token *is* 160 m). B is the bridge and would carry the weight.

The scale at which predictability disappears is the result, and it is directly comparable to
the model's own behaviour: level skill r = **+0.85** between networks, **−0.11** within TxSON,
**−0.18** within one tile.

### 27b.3 Features — the embedding of the pixel the station stands on

**X = the station's own token.** A station sits at pixel (112, 112) of its own patch = token
(7, 7) = **index 105**, the 160 m cell it stands in. That single 768-vector is the feature. No
neighbourhood averaging, no hand-built covariates. Verified on CR200-18: its six stations
occupy six *distinct* tokens (105, 62, 100, 20, 44, 172) under `(row//16)*14 + (col//16)`.

Sweep: **S2 / S1_ASC / S1_DESC × L3 / L6 / L9 / L12** plus **DEM / LULC at L12** = 14
combinations. Dynamic modalities use the **multi-year mean** of that token across acquisitions,
because the target is a multi-year mean and the aggregation must match.

**Lead with L3/L6, not L12** — §27a.1 showed they carry more within-tile structure and §27a.6
showed far less register dominance.

**Temporal aggregation: multi-year mean, not yearly.** The target is one number per station, so
a yearly-resolved feature (one row per station-year, ~5000 rows) is **pseudo-replication** — it
multiplies rows without adding information about a per-station target, and rows within a station
are near-duplicates. `GroupKFold` would keep the CV honest but the effective sample size stays
~993. Use the multi-year mean, with two corrections that do matter:

1. **Restrict acquisitions to the station's own label window.** A station with 2016-2018 labels
   but 2016-2023 imagery would otherwise average the embedding over a different period than the
   soil moisture. Take the window from `dataset._load_zarr_labels` and filter `{mod}/dates`.
2. **Two secondary blocks:** seasonal means (4 × 768 — phenology relates to moisture regime, and
   one annual mean cannot express it) and the temporal SD of the token (1 × 768 — a pixel whose
   appearance swings through the year is plausibly a different regime from a stable one).

**Cloud-sampling caution.** S2 acquisitions surviving cloud filtering are not an unbiased sample
of the year: cloudy days are preferentially wet days, so the S2 multi-year mean is really a mean
*fair-weather* token. **S1 is cloud-immune**, so S1/S2 agreement is the check that this bias is
not driving the answer. Report them separately, never pooled.

**Yearly-resolved is a different experiment**, not a better version of this one: predicting
*yearly* mean SM from *yearly* mean embedding with station means removed asks whether the
embedding tracks year-to-year wetness at a fixed site. §24.12 showed the model's temporal skill
is reasonable while its site-level skill is not, so the gap worth probing is spatial. Later.

### 27b.3b Scale A part 1 — does the embedding space organise itself by wetness?

Asked **before any model is fitted**: take the station tokens, look at how they arrange
themselves, and ask whether wet stations sit near other wet stations. No regression, no
regularisation choice, no p ≫ n concern — the structure is either there or it is not. This is
the primary presentation of Scale A; the §27b.4 ladder is its quantitative backbone.

**The figure.** 2-D projection of the 993 station tokens, plotted twice: coloured by **mean soil
moisture**, and coloured by **network / Köppen class**. The second panel is not decoration — it
is the confound made visible. If the SM colouring looks organised only because the climate
colouring is organised, the eye sees it immediately.

**Projection: UMAP, with PCA alongside.** `umap-learn` is **not installed** (checked
2026-08-12: `sklearn` 1.8.0 present; `umap`, `openTSNE`, `pacmap` absent) — needs
`pip install umap-learn` (pulls `numba` + `pynndescent`) and a line in `environment.yml`.
Preferred over t-SNE because t-SNE deliberately distorts inter-cluster distances, which is
exactly the claim being made here; UMAP preserves global structure far better. Set
`random_state` for reproducibility. Show linear PCA in the same figure as the honest baseline —
if the structure appears only under UMAP, that is worth knowing.

Two rules that must not be broken:

> **Every statistic is computed in the original 768-d space, never on the 2-D projection.**
> UMAP/t-SNE coordinates are a visualisation, not data; clustering the 2-D output and reporting
> it as a result is a standard and serious error.

> **Unsupervised fit, supervised colouring.** `umap-learn` accepts `fit_transform(X, y=sm)`,
> which uses the target to shape the projection — wet and dry would separate *by construction*
> and the figure would be circular. Supervised UMAP is legitimate only as a feature extractor
> evaluated on held-out stations inside CV, which is a different experiment.

**Three statistics, cheapest first:**

1. **k-NN in embedding space** — predict each station's mean SM as the average of its k nearest
   neighbours (cosine, leave-one-out, k ≈ 10). Report RMSE. Non-parametric, essentially nothing
   fitted, and it lands on the **same scale as the §27b.4 ladder** (null 0.0752, best tabular
   0.0576), so it is comparable without any modelling assumption.
2. **Cluster composition** — k-means into K clusters (sweep K), then one-way ANOVA of station
   mean SM across clusters: `η² = SS_between / SS_total`, with a permutation null (10,000
   shuffles of the SM labels). Answers "do wet places fall in the same clusters" with an exact
   p-value.
3. **Neighbourhood purity vs distance** — median |ΔSM| between a station and its k-th embedding
   neighbour as a function of k, against the same for random pairs. Shows how far the
   organisation extends, not just whether it exists.

**All three repeated on within-network residuals** (SM minus its network mean, tokens likewise).
That is Scale B applied to the same statistics, and it separates "the embedding knows Norway is
wet" from "the embedding knows *this field* is wet". Expect a large drop; **the size of the drop
is the result**.

**Secondary contrast, free:** rerun with the 4-scale pooled pyramid in place of the centre
token — what the model actually receives for the history and for DEM/LULC. §27a.2 measured that
as retaining 2–3% of within-tile variance. Centre token predicting while pooled does not ⇒
pooling is the bottleneck, and the comparison is one-sided by construction.

**No separate normalisation job is needed.** `station_mean_probe.py` already puts a
`StandardScaler` inside the CV, fitted on the training fold only. This matters: §27a.5 found
register dims holding 59–94% of the magnitude but 2–16% of the across-station variance, and
RidgeCV penalises coordinates equally, so unstandardised features would be dominated by them.

### 27b.4 Scale A — extend the existing ladder, against a baseline that already exists

`station_mean_probe.py` gains a **`B9 +terramind`** block. The §20.14 ladder already
establishes exactly the right baseline (depth 0-10, RMSE of station-mean SM):

| | RMSE |
|---|---|
| null (predict the global mean) | 0.0752 |
| B1 soil | 0.0621 |
| B5 +derived | 0.0599 |
| B8 +smap_tb (soil + terrain + land cover + climate + lat/lon + SMAP) | **0.0576** |
| the trained network's own RMS per-station bias | 0.0618 |

> **The question is concrete and falsifiable: does adding TerraMind tokens beat 0.0576?**

If a foundation-model embedding cannot beat a tabular stack of soil, terrain, climate and SMAP,
that is a publishable finding about foundation models for soil moisture. If it does, that is
the justification for the architecture.

Reuse as-is: `fit_block()` (`station_mean_probe.py:199`), `GroupKFold` on `location_group_id`
(`:212-215`, which already prevents neighbouring stations straddling train/test), RidgeCV as
the deliberately weak learner, `HistGradientBoostingRegressor` as the nonlinearity control, the
JSON ladder output, and `NETWORK_RMS_BIAS` (`:37-39`).

### 27b.5 Decision gate for Scale A

| result | reading | next |
|---|---|---|
| beats 0.0576 clearly | embeddings carry SM information beyond the tabular stack | run B, then C |
| ties 0.0576 | embeddings re-encode what soil/terrain/climate already say | run B — does it add anything *locally*? |
| loses to 0.0576 | at station-mean scale the embeddings add nothing | report it; C only with a specific reason to expect sub-km to differ |

**A strong Scale A result does NOT imply sub-km skill** — L12 separating Texas from Norway
would produce it. That is why B and C exist and why A alone must not be over-read.

### 27b.6 Limits, on record before running

- Part of every station's mean is instrument, not landscape — calibration, installation depth,
  and the point-vs-160 m support mismatch. That bounds every scale from above.
- Tokens are frozen and never saw soil moisture, so **train/val/test leakage does not apply**;
  every station is usable regardless of split.
- 768 dims against ~993 stations: RidgeCV under a real L2 penalty is the point, not a
  limitation — a positive result cannot be memorisation. GBM runs on PCA(32) as the
  nonlinearity control.
- Pre-register **depth 0-10, S2 L3, centre token** as primary; the rest exploratory under
  Benjamini–Hochberg FDR.

### 27b.7 Verification

1. The ladder reproduces the existing B1–B8 numbers with the token block omitted.
2. Station token index equals `(row//16)*14 + (col//16)`; CR200-18's six give 105, 62, 100, 20,
   44, 172 with the centre at 105.
3. Six CR200-18 station means reproduce 0.1367 / 0.1197 / 0.1826 / 0.2323 / 0.1857 / 0.2865
   to 1e-3.
4. Every block collapses to the null RMSE under a permuted target.

### 27b.8 Scale A results (run 2026-08-12, jobs 25530917 / 25531584 / 25532410)

`probe_token_sm_structure.py` + `slurm/probe_token_sm_structure.sh` →
`csvs/token_sm_structure.{csv,json}`, `figures/token_sm/umap_*.png`.
**842 `sm_only` stations**, depth 0-10, station's own centre token (index 105), multi-year mean
over acquisitions inside each station's own label window, 24-date subsample.

**Restricted to `sm_only` deliberately.** It is the population the model was trained and
evaluated on (`eval_output/manifest.json` `category_filter`), and it is the only category free of
the non-finite tokens found below. With `sm_only` the run has **zero** non-finite warnings.

| modality/layer | null | kNN10 | **skill** | η²(20) | p | null_w | kNN10_w | **skill_w** | η²_w | p_w |
|---|---|---|---|---|---|---|---|---|---|---|
| **s2/l12** | 0.0798 | 0.0720 | **9.8%** | **0.156** | 0.0005 | 0.0734 | 0.0713 | 3.0% | 0.038 | 0.034 |
| s1_desc/l9 | 0.0784 | 0.0718 | 8.4% | 0.052 | 0.0015 | 0.0725 | 0.0694 | **4.2%** | 0.038 | 0.072 |
| s1_desc/l12 | 0.0784 | 0.0721 | 8.0% | 0.076 | 0.0005 | 0.0725 | 0.0702 | 3.2% | 0.029 | 0.227 |
| s1_asc/l12 | 0.0797 | 0.0740 | 7.2% | 0.055 | 0.0005 | 0.0735 | 0.0725 | 1.4% | 0.043 | 0.014 |
| s2/l9 | 0.0798 | 0.0743 | 6.9% | 0.109 | 0.0005 | 0.0734 | 0.0725 | 1.2% | 0.038 | 0.031 |
| dem/l12 | 0.0798 | 0.0744 | 6.8% | 0.101 | 0.0005 | 0.0734 | 0.0713 | 2.9% | **0.075** | 0.0005 |
| s1_asc/l9 | 0.0797 | 0.0750 | 6.0% | 0.055 | 0.0015 | 0.0735 | 0.0727 | 1.1% | 0.023 | 0.473 |
| s2/l3 | 0.0798 | 0.0766 | 4.0% | 0.047 | 0.0035 | 0.0734 | 0.0746 | **−1.6%** | 0.016 | 0.842 |
| s2/l6 | 0.0798 | 0.0769 | 3.6% | 0.061 | 0.0005 | 0.0734 | 0.0746 | **−1.6%** | 0.023 | 0.412 |
| lulc/l12 | 0.0798 | 0.0803 | **−0.7%** | 0.062 | 0.0005 | 0.0734 | 0.0753 | **−2.5%** | 0.041 | 0.015 |

(`_w` = within-network residuals: network mean removed from features and target. s1_asc/l3, l6
and s1_desc/l3, l6 omitted for space; all ≤3.8% global, ≤0.1% within-network.)

**Three readings.**

1. **The embeddings predict SM, but far worse than the tabular stack.** Like-for-like skill:
   §20.14's B8 (soil + terrain + land cover + climate + lat/lon + SMAP) is 0.0752 → 0.0576 =
   **23.4%**. The best token is 0.0798 → 0.0720 = **9.8%**. The tabular covariates are ~**2.4×
   better**. *Caveat: k-NN is a weaker learner than the ladder's RidgeCV/GBM, so 9.8% is a FLOOR,
   not a ceiling — §27b.4's `B9 +terramind` block is still needed before this becomes a claim.*
2. **Remove climate and it nearly vanishes.** Within-network skill 0–4.2%, with **four of
   fourteen combinations negative** (worse than predicting the network mean). η² for s2/l12 falls
   0.156 → 0.038. This matches the model's own behaviour exactly: level skill r = +0.85 between
   networks, −0.11 within TxSON, −0.18 within one tile.
3. **The layer ordering flips versus §27a.** L12/L9 beat L3/L6 here (s2: 9.8%, 6.9% vs 4.0%,
   3.6%) — the opposite of the within-tile spatial-detail ordering. Consistent: station-mean SM
   at global scale is largely a climate/biome task, and L12 is where climate lives. §27a's "lead
   with L3/L6" therefore applies to the **sub-km** question, not this one.

**The figure says it without statistics.** `figures/token_sm/umap_s2_l12_0-10.png`: UMAP and PCA
fitted **unsupervised** on the tokens, soil moisture used only for colour. The Köppen panel shows
D, C and B classes in distinct regions; the soil-moisture panel is near-uniform scatter.
**The embedding organises itself by climate, not by wetness.**

`umap-learn` 0.5.12 was installed into the `terramind` env for this (`pip install umap-learn`,
pulls `numba` + `pynndescent`); add it to `environment.yml`.

**Data-quality item.** Four stations have non-finite S1 tokens and are **all `sm_and_flux`**:
`AmeriFlux_US-Ton`, `AmeriFlux_US-Var`, `ICOS_DE-HoH`, `ICOS_FI-Var`. `precompute_terramind.py:427`
raises on non-finite at write time, so these arrived by another path. Worth tracking down before
any run that uses `sm_and_flux`.

---

## §28 Replace the U-Net decoder with a per-token head — an honest 160 m map (DESIGNED, not built)

**STATUS: DESIGNED 2026-08-12, NOTHING BUILT.** Recorded before code exists, for critique.

### 28.1 The two defects, stated separately

They are independent, and only the first is pooling:

| | does the information arrive? | is it used? | fix |
|---|---|---|---|
| **DEM / LULC / S2-S1 history** | **NO** — pooled to 4 nested-window vectors before the transformer sees them (§27a.2: 1.5% / 2.6% / 2–3% of within-tile variance retained) | — | feed their 196-token grids |
| **anchor S2/S1** | yes — 196 tokens unpooled, 4 layers (`model.py:497-530`) | **NO** — U-Net decoder to 224², only pixel (112,112) supervised (`model.py:779`) | per-token head |

Supporting measurements: §23 — the 224² map's structure is *anti-correlated with the landscape*
and **64–83% of its variance already sits on the 14×14 grid**, i.e. effective resolution ≈160 m,
with the verdict *"do not present this model as producing 10 m soil-moisture maps"*. §26.11 —
off-centre readouts degrade (median ΔubRMSE +0.0025, 74% of stations worse) and train-station
memorisation does not travel off the supervised pixel (0.0110 → 0.0500).

### 28.2 The change

Delete `UNetDecoder`. Apply a small **shared** head to the 196 transformer output tokens the model
already computes (`spatial_ctx`, `model.py:732`). Supervise at the token containing the station.
At inference the same head runs on all 196 in one pass → a **14×14 map at 160 m**, matching
TerraMind's patch size and the resolution §23 measured the model as actually having.

**Cost goes DOWN, not up.** The 196 contextualised tokens already exist; we are attaching a head
to a tensor, not adding 196 forward passes. What is deleted is the dominant cost: four upsampling
stages to 224² with channels (512, 256, 128, 64) plus three FiLM'd skips. The 224²×64 activation
is ~3.2 M values per sample per depth; the new head is ~200 K. Expect lower memory and a faster
step. Only eval storage grows (196 values per sample instead of 1).

### 28.3 What the head receives, per token k

| input | shape | source | why |
|---|---|---|---|
| `h_k` contextualised spatial token | 768 | `ctx[:, sp_start+k, :]` (`model.py:732`) | carries the global context attention built |
| `anchor_l3_k / l6_k / l9_k` | 3 × 768 | `batch["anchor_l3/6/9"][:, k, :]` — **the decoder skips, in token space** | §27a: L3/L6 hold the most within-tile detail and least register dominance; today reachable only via the decoder |
| `depth_ctx[:, d, :]` | 768 per depth | `ctx[:, :n_depths, :]` (`model.py:723`) | preserves the depth mechanism and star residual |

```
u_k    = Linear(4·768 → 768)( concat[h_k, l3_k, l6_k, l9_k] )      # SHARED across all k
base   = head_0( FiLM_0(u_k, depth_ctx[:,0,:]) )                   # depth 0 absolute
out[d] = base + head_d( FiLM_d(u_k, depth_ctx[:,d,:]) )  for d>0   # zero-init offsets
```

Weight sharing across k is the whole mechanism: supervising one token teaches the mapping at
every token, which is what removes the 50,175-unconstrained-pixels problem. `context`
(`model.py:743`) is dropped — `h_k` already attends to those tokens; keep it as an ablation flag.

### 28.4 Pooling is KEPT, as context

A per-token head is not a per-token model. `_build_sequence` (`model.py:532-574`) already gives
six layers of **global self-attention** over
`[DEM×4 | LULC×4 | Soil×4 | spatial×196 | S2 hist | S1 hist | ERA5×365 | SIF | TWSA]`, so each
output token is contextualised by its own embedding, every other spatial token, the pooled
statics, the pooled history and the full ERA5 series. Nothing is removed; pooling simply stops
being the *sole* spatial channel for DEM/LULC.

**Cheap addition, ablatable on its own:** fuse the per-token DEM/LULC grids into the spatial
tokens by **addition** rather than appending 392 tokens (sequence ~600 → ~1000):

```
spatial_tokens += W_dem · dem_tokens[:, k, :] + W_lulc · lulc_tokens[:, k, :]
```

Same pattern as the positional / modality / staleness embeddings at `model.py:503-514`. Zero
sequence-length cost, and the cheapest available fix for §27a.2.

### 28.5 The trap: position leakage

Patches are station-centred, so the supervised token is **always index 105**. With 2-D positional
encodings (`model.py:501-505`) the model can learn *"read (7,7)"* rather than a position-general
mapping — exactly the off-centre degradation §26.11 measures. Needs at least the first two:

1. **Translation augmentation in token space** — feed a random sub-window of the 14×14 grid
   (e.g. 10×10) so the station lands at varying positions. A slice of an array already in memory;
   **no re-tokenisation**.
2. **Multi-station supervision** — k stations in a tile give k supervised tokens from one forward
   pass. `csvs/txson_readouts.csv` already holds the indices (96 readouts, 56 off-centre).
3. **Ablate the spatial positional encoding** as a diagnostic.

### 28.6 The risk: global attention may homogenise

`training_runbook.md:1500` (§14) already records *"Global self-attention homogenises predictions →
nearly uniform SM maps"*. Pushback: the gradient now **penalises** flatness (under the decoder
only one pixel was supervised, so nothing did); and it is directly measurable — log across-token
SD of the predicted map and spatial attention entropy each epoch, so collapse is visible
immediately. **Fallback:** windowed (Swin-style) spatial attention over a k×k neighbourhood while
keeping full attention to the pooled/ERA5 prefix. Hold in reserve; do not build up front.

### 28.7 Honest limit

§27b.8 measured within-network skill from a station's own token at **0–4.2%**. **This change
cannot manufacture information.** What it delivers regardless: an output resolution matching the
input resolution, no unconstrained pixels, and a map that is *validatable* — per-token
predictions make §26.3's 417 k-pixel tile-overlap disagreement test clean, and TxSON's 56
off-centre readouts become genuine held-out ground truth rather than an out-of-distribution
probe. Frame it as removing a defect and making the product honest, not as a fix for
heterogeneity. State the expected gain on the spread ratio as low **before** the GPU is spent.

### 28.8 Numbers it must beat (TxSON, same checkpointed comparison as §26.11)

| metric | current | target |
|---|---|---|
| own-centre ubRMSE (0-10) | 0.0301 | ≤ |
| off-centre ubRMSE | 0.0345 | materially closer to own-centre |
| **between-station spread as % of observed** | **15–19%** | **> 35%** (`plot_network_timeseries.py`'s "resolves the tile" threshold) |
| r(pred level, obs level), CR200-18 | −0.175 | > 0 |
| tile-to-tile disagreement on overlaps | unmeasured | small, and NOT aligned to tile boundaries |

### 28.9 Implementation and verification

`model.py` — add `TokenHead`, gate on `--head token|unet` so existing checkpoints stay
reproducible; `forward()` returns `(B, n_depths, 14, 14)` in token mode.
`masked_huber_loss` — read `pred[:, :, tok_idx]` with a **per-sample** token index instead of the
hardcoded `(112,112)`; accept padded (token, label) lists for multi-station supervision, as
`eval_predict.py`'s `PixelMap` already does. `dataset.py` — emit `token_idx = (row//16)*14 +
(col//16)` and the optional translation crop. `eval_predict.py` — `--pixel-csv` gather switches
from 224² to token indices.

Verify: (i) with augmentation off and one station per tile, token-mode predictions at index 105
are statistically comparable to the current `(112,112)`; (ii) after augmentation, error must not
depend on which token the station occupies — regress error on `offset_px`, slope ≈ 0 against
§26.11's current positive slope; (iii) CR200-18's six tokens are 105, 62, 100, 20, 44, 172 with
the centre at 105; (iv) measure step time and peak memory before committing — the claim is that
both improve; (v) smoke on one tile, one year, before any multi-day allocation.

---

## §29 Does land-surface temperature show within-tile heterogeneity? (Phase A RUN 2026-08-13 — YES it does, but it does NOT track soil moisture)

**STATUS: Phase A RUN 2026-08-13. 246 Landsat scenes downloaded (tile + full AOI), station
series extracted, tests executed, both figures made. Result in §29.13 — the answer is NO, and
the reason the naive pooled test said otherwise is instructive.**

**Build order, decided 2026-08-13: two strict phases.** Phase A is Landsat end to end — download,
analysis, both figures, write-up. It needs no credentials, resolves 30 m rather than 70 m, and
spans the full 2016-01 → 2022-11 label window, so **it answers the question standalone.** Phase B
is ECOSTRESS and starts only once A is written up; it adds exactly one thing Landsat physically
cannot provide — night LST, and therefore the diurnal temperature range — and reuses the Phase A
code with `--sensor ecostress`.

### 29.1 Why LST is the right next input to try

§27a/§27b say the *reflectance* embeddings encode the landscape but not its wetness: within-tile
heterogeneity is real and tracks terrain and land cover, yet within-network SM skill is 0–4.2%
and the UMAP organises by Köppen class, not by soil moisture.

**LST is a different physical channel, not another view of the same one.** Wet soil is cooler
through evaporative cooling, so surface temperature is a direct thermodynamic consequence of soil
moisture rather than a correlate of vegetation and terrain. If LST shows within-tile structure
tracking the station-to-station SM differences where the embeddings do not, that is a clear,
physically-grounded finding about what a soil-moisture model should be fed.

Scope is small: TxSON, 40 stations in a 33 × 33 km domain, labels 2016-01-01 → 2022-11-07.
The reference case throughout is tile `ISMN_TxSON_CR200-18`, which holds **six** stations —
CR200-18 at the centre pixel (112,112), then CR200-25 at 405 m, CR1000-2 at 684 m, CR200-24 at
865 m, CR200-15 at 925 m, CR200-6 at 936 m. Observed mean SM spans 0.1197–0.2865, spread
**0.0601**; the model predicts spread **0.0113** with **r = −0.175** (§26.11).

### 29.2 Sensor choice — the decisive difference is day/night, not revisit

| | Landsat 8/9 C2 L2 ST | ECOSTRESS (`ECO_L2T_LSTE` C2) |
|---|---|---|
| resolution | **30 m** → 76 × 76 px inside a tile | ~70 m → 33 × 33 px |
| repeat | 16 d per satellite, **halved by the sidelap (§29.3)** | none fixed — ISS orbit precesses |
| overpass | fixed ~11:00 CST, **DAYTIME ONLY** | **DAY AND NIGHT** — drifts through the diurnal cycle |
| record vs TxSON labels | **full** | 2018-07 → (~63% of the period) |
| access | MPC, anonymous — routes this repo already uses | NASA Earthdata via `earthaccess` (credentials needed) |

Landsat is sun-synchronous and the C2 L2 ST product is generated only for daytime scenes.
**Landsat can never give a diurnal range.** That splits the choice by physical signal:

- **Daytime only (Landsat):** midday is the single best time for *spatial* wet/dry contrast —
  maximum solar forcing gives maximum evaporative-cooling difference. Classic and strong.
- **Day + night (ECOSTRESS):** unlocks the **diurnal temperature range**, a thermal-inertia proxy
  and arguably the strongest LST-based soil-moisture signal — wet soil has high thermal inertia
  and a small diurnal swing, dry soil swings widely. Night LST is also physically cleaner: no
  slope/aspect illumination effects, no transpiration confound.

70 m is not a limitation here — CR200-18's stations are 405–936 m apart = **6–13 ECOSTRESS
pixels**. Landsat is nonetheless built first because it is unblocked, finer, and covers the
whole label window.

### 29.3 Step 0 — the catalogue census, run live 2026-08-13

Measured rather than argued. Both queries are metadata-only and need no credentials.

**Landsat.** MPC `landsat-c2-l2` over the TxSON bbox `(-98.97, 30.15, -98.59, 30.48)`,
2016-01-01 → 2022-11-07, returns **625 items**. The decisive finding:
**path 027/row 039 = 313 and path 028/row 039 = 312 — TxSON sits in a WRS-2 sidelap, so the
effective revisit is halved.** Landsat-8 = 303, Landsat-9 = 35, Landsat-7 = 287 (excluded);
Tier-1 = 298, Tier-2 = 40. L8/9 per year 2016:46, 2017:45, 2018:46, 2019:46, 2020:44, 2021:49,
2022:62. All acquisitions at UTC hour 17 ≈ 11:00 CST. A scene-level `eo:cloud_cover < 20` would
retain only 141 scenes over seven years — too aggressive to use as the download gate (see §29.5).

Three things the census corrected that would otherwise have cost a debugging cycle:

1. **The asset is `lwir11`, not `ST_B10`.** MPC exposes `lwir11` (Surface Temperature, uint16,
   scale 0.00341802, offset 149.0, nodata 0, kelvin), `qa` (= ST_QA, int16, scale 0.01, nodata
   −9999, kelvin) and `qa_pixel` (uint16). **Landsat-7 exposes `lwir` instead and will KeyError** —
   another reason to exclude it beyond its SLC-off wedges.
2. **`proj:epsg` is absent** from these items — only `proj:code` and `proj:transform`. `stackstac
   0.5.1` still works provided `epsg=` is passed explicitly.
3. **Grid alignment.** All 90 L8/9 scenes checked are EPSG:32614 with `proj:transform` origin
   ≡ **(15, 15) mod 30**, identically for both paths. The 10 m TxSON grid is *not* on that grid;
   an unsnapped AOI silently costs a 15 m shift.

Smoke-tested end to end: a 75 × 75 window on 2019-08-16 returned LST 313.7 / 318.3 / 331.0 K,
ST_QA 2.84 K, `qa_pixel = 21824` (= clear, all confidences "low"). Plausible Texas midday August
values with the right phase. **The download path is proven.**

**ECOSTRESS.** CMR metadata needs no auth. `ECO_L2T_LSTE` v002 over the same bbox,
2018-01-01 → 2022-11-07: **1984 granules** (2018:40, 2019:426, 2020:637, 2021:476, 2022:405),
**DAY 944 / NIGHT 1040**. The local-hour histogram is essentially flat — 61 to 107 granules across
all 24 bins — so **the day+night advantage is real and is not an artefact of a midday cluster.**
Two MGRS tiles appear, but **`14RNU` alone covers all of TxSON** (bbox 29.74/−99.00 →
30.73/−97.85); `14RMU` only clips the western sliver of the search box and adds ~995 redundant
granules. 989 granules on 14RNU, median 5.07 MB. `day_night_flag` is returned directly in the
granule metadata — do not try to infer it. One trap: CMR link entries do **not** all carry a
`title` key, so filter layer links by `href` basename suffix.

The one number still unmeasured, and the one that sizes the DTR test: **how many dates carry both
a DAY and a NIGHT granule after cloud filtering.** The census script writes it.

**Tooling, now settled.** Downloads run in **`soilmoisture`** (py3.10 — pystac-client,
planetary-computer, stackstac, rasterio, rioxarray; **no pyarrow, no earthaccess**). Analysis and
plotting run in **`terramind`** (py3.11 — rasterio, rioxarray, pyarrow, zarr, scipy, sklearn;
**no STAC, no EDL**). The split is a hard constraint: never combine download and analysis in one
job. `~/.netrc` currently holds only `machine api.wandb.ai` and must be **appended to**, never
overwritten.

**Script:** `census_lst_sources.py` → `csvs/lst_census_landsat_scenes.csv`,
`csvs/lst_census_ecostress_granules.csv`, `csvs/lst_census_summary.json`. It asserts on every
retained Landsat item that `proj:code == "EPSG:32614"`, that
`(transform[2] % 30, transform[5] % 30) == (15, 15)`, and that `lwir11` is present — the
invariants the AOI snap depends on.

### 29.4 Phase A — the Landsat downloader

`download_landsat_st_mpc.py`, forked from `download_s2_mpc.py`. Reuse verbatim: `with_retry`
(`:123-136`), `save_geotiff` (`:139-145`, float32), `load_checkpoint`/`append_checkpoint_row`
(`:152-166`), `setup_logging` (`:173-184`), `os.environ.pop("PROJ_DATA", None)` (`:27`), the
`ThreadPoolExecutor` orchestration (`:400-421`), and — load-bearing — the
**re-sign-inside-the-loader-closure** idiom (`:217-225`); SAS tokens expire, so `sign_inplace`
must never be hoisted out of the retry.

**Download one AOI over the whole TxSON domain, not 40 patches** — the tiles overlap heavily
(§26.2: 131 km² in 17 islands), so per-station patches would re-fetch the same bytes repeatedly.
Drop `station_grid`, `center_crop`, `process_station`, `download_dem`, `STATION_CSV`; iterate over
*scenes* instead of stations. Snap `csvs/txson_mosaic_grid.json` `origin_utm` onto the Landsat grid
with `snap(v, res=30, off=15)`:

> **AOI: EPSG:32614, bounds `(503685.0, 3336015.0, 538545.0, 3371055.0)` = `1162 × 1168` px @ 30 m.**
> **The CR200-18 tile window inside it: `(527835.0, 3344895.0, 530115.0, 3347175.0)` = `76 × 76` px
> at `row0 = 796`, `col0 = 805`.**

`stackstac.stack(assets=["lwir11","qa_pixel","qa"], epsg=32614, resolution=30, bounds=AOI,
rescale=False, resampling=Resampling.nearest, dtype="float64", fill_value=np.nan)`.
**Nearest, never bilinear** — bilinear would smear cloud edges into the QA band and interpolate
across ST nodata.

Scale on write: `lst_k = where(dn == 0, nan, dn * 0.00341802 + 149.0)`,
`st_qa_k = where(dn == -9999, nan, dn * 0.01)`, and keep **`qa_pixel` as raw DN** so the mask
threshold can be revisited without re-downloading. One 3-band float32 GeoTIFF per scene,
`[lst_kelvin, st_qa_kelvin, qa_pixel_dn]` — uint16 ≤ 65535 < 2²⁴, so float32 round-trips exactly
and no mixed-dtype hack is needed. Output
`/gpfs/scratch1/shared/pkhanal/lst/landsat_st/txson/{YYYYMMDD}_{LC08|LC09}_{path}{row}.tif` plus
`aoi.json`; path/row and platform go in the filename because the sidelap plus L8+L9 makes same-day
collisions possible. Resume through `csvs/landsat_st_download_log.csv`, skipping when the file
exists **or** the item id already has `status in {done, no_data}`.

**298 Tier-1 scenes ≈ 1.5–2 GB, ~20–30 min at `--workers 12`.** `--smoke` (one scene, tile window)
must reproduce the 2019-08-16 read above before anything is submitted.

### 29.5 How cloud is filtered — three tiers, and why the product QA is the right tool

This is where thermal data goes wrong. **A cloud-contaminated LST pixel is not merely noisy — it is
several kelvin cold, and it would masquerade as wet soil, manufacturing exactly the negative
correlation §29.7 is looking for.** Note this is *not* the CloudSEN12 route the repo uses for S2
(`cloud_masking_inference.py` → `filter_cloudy_tiles.py`): thermal products ship their own QA, and
a learned RGB cloud mask would be the wrong instrument.

**Tier 1 — scene level, at search time.** `query={"eo:cloud_cover": {"lt": 80}}`. Deliberately
loose: `eo:cloud_cover` describes a ~180 × 180 km scene, so it says almost nothing about a 2.24 km
tile. A 70%-cloudy scene is often perfectly clear over TxSON; a 15%-cloudy one can have its one
cloud sitting on the network. Filter loose here, hard at pixel level.

**Tier 2 — per pixel, Landsat `QA_PIXEL` (uint16).** Bits: 0 Fill, 1 Dilated Cloud, 2 Cirrus,
3 Cloud, 4 Cloud Shadow, 5 Snow, 6 **Clear**, 7 Water, 8–9 Cloud Confidence, 10–11 Cloud Shadow
Confidence, 12–13 Snow/Ice Confidence, 14–15 Cirrus Confidence (0 none → 3 high).

```python
q      = qa.astype(np.uint16)
single = ~np.any([(q >> b) & 1 for b in (0, 1, 2, 3, 4, 5, 7)], axis=0)   # incl. water
conf   = (((q >> 8) & 3) <= 1) & (((q >> 10) & 3) <= 1) & (((q >> 14) & 3) <= 1)
clear  = single & conf & (((q >> 6) & 1) == 1)
```
The live-read value 21824 passes (bit 6 set, all confidences "low"). Two filters the bit table does
not give and which are both necessary: **`ST_QA ≤ 3 K`** per-pixel uncertainty (the 2019-08-16 read
gave 2.84 K, so 3.0 sits near the mode — run 2/3/5 K and report the sensitivity rather than
defending one threshold), and a **range guard `250 < LST < 350 K`**.

**Tier 3 — ECOSTRESS.** Separate `cloud` and `water` layers (uint8, 1 = flagged) plus `QC` bits 0–1
(Mandatory QA, `00` = produced, best quality), and `view_zenith < 25°`. Bits above 3 are
ASTER-heritage and their layout is **unverified**; treat them as advisory until the LP DAAC v002
user guide is checked. The mask is instead validated *empirically* by §29.10's no-cold-tail
histogram check, which is independent of any bit table being right.

**Download-time gating.** For **Landsat, do not gate** — 298 scenes in ~25 min, and a two-pass on
`qa_pixel` would save bytes while doubling the HTTP round-trips that actually cost the time; keeping
raw QA DN on disk is worth more than the saving. Do compute and log `frac_clear` per tile at write
time so analysis-side filtering is free. For **ECOSTRESS, gate** — 989 granules × 6 layers ≈ 5900
fetches over 6–12 h is the long pole of the whole plan, so fetch `_cloud.tif` first (uint8, a small
fraction of the granule), compute clear fraction over the **whole AOI** (not per tile — one AOI
serves all 17 islands, and a granule clear over half the domain is still worth having), and skip the
other five layers if it fails `--min-aoi-clear` (default 0.30). Log every skipped granule with its
clear fraction so the decision is auditable and re-runnable at a looser threshold. Expect roughly
half to fail, roughly halving the job.

### 29.6 Phase B — the ECOSTRESS downloader

`download_ecostress_lste.py`. Add `earthaccess` and `pyarrow` to the `pip:` block of
`environment-download.yml`, then `conda env update -n soilmoisture`.

**User action, the only hard external blocker in the plan (~3 min):** register at
`urs.earthdata.nasa.gov`, then Applications → Authorized Apps → approve **"LP DAAC Data Pool"** and
**"LP DAAC Cumulus (LPCLOUD)"** (downloads 403 without this even with valid credentials), then
**append** a `machine urs.earthdata.nasa.gov` block to `~/.netrc` — the `api.wandb.ai` entry must
survive — and `chmod 600`. Worth doing during Phase A so Phase B starts unblocked.

`earthaccess.search_data(short_name="ECO_L2T_LSTE", version="002", bounding_box=…, temporal=…,
count=-1)`, filtered to `--tiles 14RNU`. Keep layers `_LST, _LST_err, _QC, _cloud, _water,
_view_zenith`; `earthaccess.download()` fetches *all* granule files, so build an EDL session
(`get_requests_https_session()`) and stream only the selected hrefs through `with_retry`. LST is
uint16, **scale 0.02 K, fill 0** — the only factor hardcoded; for `LST_err` and `view_zenith`,
**read `src.scales`/`src.offsets`/`src.nodatavals` from the COG** and cross-check the user guide
rather than assuming.

**No reprojection is needed.** MGRS tile 14RNU is UTM zone 14N = EPSG:32614, the same CRS as
everything else in this project. Assert `crs.to_epsg() == 32614`, derive `off = transform.c % 70`
**from the file** (40 expected, unverified), snap the mosaic bounds onto that grid → roughly
500 × 501 px, with the CR200-18 window at 33 × 33. Assert every later granule shares that exact
transform; `reproject_match` with nearest and log it if one does not.

Output one 6-band float32 GeoTIFF per granule,
`{YYYYMMDD}T{HHMMSS}_{DAY|NIGHT}_{orbit}_{tile}.tif`, ~1–2 MB deflated; delete the raw per-layer
downloads after repacking. Keep `--workers 8` — LP DAAC throttles above that. **~1.5 GB retained,
6–8 GB transient, 6–12 h.**

**If Earthdata never materialises**, say so plainly: there is **no substitute** for the night half.
Landsat C2 L2 ST is daytime-only by construction. Run §29 Landsat-only — it still answers the core
question, at *better* resolution — and mark the DTR test deferred. MODIS/VIIRS LST is on MPC
anonymously with day+night, but at **1 km** it is ~2 pixels across the entire tile and cannot
resolve 405–936 m separation; it is a domain-mean sanity check, not a substitute.

### 29.7 Analysis

`analyze_lst_heterogeneity.py`, env `terramind`. Reuse the geometry that already exists rather than
recomputing it: `build_network_readouts.py:59-70` `load_tile_geometry()` reads `epsg`/`bounds_utm`
from the satellite zarr attrs, and `:75-90` `station_pixel()` returns `(row, col, x, y)` — the
`x, y` is exactly what the LST grids need, so the projection is shared with `csvs/txson_readouts.csv`
**by construction rather than by coincidence**. Then `ls_col = floor((x − AOI_W)/30)`,
`ls_row = floor((AOI_N − y)/30)`, and the same at 70 m.

**Work in LST anomaly, not absolute LST.** Subtract the tile mean on each date — the same
within-tile differencing used throughout §27, which cancels season, air mass and overpass time
exactly, leaving the local contrast. Require `tile_frac_clear ≥ 0.70` or drop the date. Compare the
anomaly magnitude against the sensor noise floor (~1–2 K, and against the measured per-pixel
`ST_QA`); **if the within-tile SD sits at the floor, the answer is immediate.**

**Observations come from `eval_output/txson_timeseries.parquet`, not the level-1 NetCDFs — this is
a trap that will otherwise look like a failed verification.** The parquet reproduces §29.10's six
reference means exactly; the raw `/projects/prjs1968/raw_soil_moisture/TxSON_*.nc` with `qc == 0`
gives 0.1390 / 0.1209 / 0.1841 / 0.2361 / 0.1857 / 0.2918 instead, because those records start
2014-10/11, before the 2016-01-01 window. Assert the six means to 1e-3 before doing anything else.
Keep `combine_network._obs_from_zarr` as fallback only, with a loud warning.

**The tests, in increasing order of statistical power:**

- **Headline (directly comparable to §26.11).** Per tile, per sensor, per day/night: station mean
  LST anomaly vs station mean observed SM, n = 6 for CR200-18. Report `r`, the LST-anomaly spread
  in K, and the SM spread 0.0601 — set beside the model's spread 0.0113, r = −0.175, and §27b's
  0–4.2%.
- **Per date.** For each (tile, sensor, date, day/night) with ≥ 4 valid stations,
  `r_date = corr(LST anomaly, observed SM)`; aggregate mean/median `r`, `frac(r < 0)`, a binomial
  sign test, and a one-sample t on Fisher-z. **The prediction is a NEGATIVE correlation.** The sign
  is itself a strong control: zero or positive means the signal is not evaporative-cooling driven
  and the interpretation changes completely.
- **Pooled.** Across all station-dates, `lst_anom_k` against `obs − tile_mean_obs(date)`. One
  number, maximum power, no n=6 fragility.

> **AMENDED 2026-08-13 after running it — read §29.14 before trusting the two tests above.**
> Neither the per-date nor the pooled test as written is a valid test of the hypothesis: both are
> dominated by *between-station* offsets, and on the real data they return `+0.244` and `+0.167`
> while the mechanism they were meant to detect is absent. The station-identity component must be
> removed first — **de-mean both variables by station** (fixed effects), which turns `+0.167` into
> `−0.077`. "Maximum power, no n=6 fragility" was wrong: the 546 records are five stations
> replicated 116 times, not 546 independent samples, so the p-value was inflated too.
- **Diurnal range (Phase B).** On dates carrying both DAY and NIGHT, `DTR = lst_day − lst_night`,
  then `corr(DTR anomaly, SM)`. **Prediction: negative** — wet soil has high thermal inertia and a
  small swing.

**Controls, all required.** (1) **Label shuffle** — permute station labels within (tile, date),
1000 draws, empirical p; must destroy the signal. (2) **Clear-sky dry bias** — retained vs dropped
SM distribution, mean/median/KS, retained counts per year. (3) **Noise floor** — within-tile
anomaly SD vs median `ST_QA`/`LST_err`. (4) **LST–NDVI (TVDI-style)** using the S2 NDVI already on
disk, and the partial correlation of LST anomaly with SM controlling for NDVI, since vegetation
cover confounds the LST–moisture relationship. (5) **Terrain** — partial out elevation, and
slope/aspect for daytime Landsat; night ECOSTRESS should show *less* terrain dependence, and that
contrast is itself evidence. (6) **ST_QA sensitivity** at 2/3/5 K.

Outputs: `csvs/lst_station_pixels.csv`, `csvs/lst_station_timeseries.csv`,
`csvs/lst_per_date_corr.csv`, `csvs/lst_level_correlations.csv`, `csvs/lst_controls.json`,
`csvs/lst_summary.json`.

### 29.8 Figures — the two deliverables

Both in `terramind`, both importing `mark_stations` (`:132`), `scale_bar` (`:150`), `layout_panel`
(`:227`), `STATION_COLOURS` (`:45`), `open_raw` (`:61`), `hillshade` (`:84`) from
`plot_tile_context.py`, and following its conventions (`Agg`, `GridSpec`, 8.2 pt left-aligned
titles, monospace summary block, dpi 190, save png **and** pdf, the `assert centre == 112` guard).

**`plot_tile_lst.py` → `figures/tile_lst/{tile}.{png,pdf}` — LST at native resolution.** The trick
that makes native resolution work with the existing station markers: station `row`/`col` in
`txson_readouts.csv` are 10 m tile pixels, so render each LST panel with `interpolation="nearest"`
and an `extent=` mapping the 76 × 76 (or 33 × 33) native grid onto the 224 px axis via its UTM
bounds. **The blocky 30 m / 70 m pixels stay blocky — that is the entire point** — while
`mark_stations` works unchanged and the 14 × 14 token-grid overlay still lines up, so §29 can be
read directly against §26/§27. Rows: (1) Landsat absolute, 4 dates auto-picked as the 2 wettest and
2 driest by tile-mean observed SM among scenes ≥ 90% clear, `inferno`, per-panel colorbar since
season dominates absolute values; (2) Landsat **anomaly**, same dates, `RdBu_r` symmetric about 0
with one shared colorbar — this is the row where the heterogeneity either is or is not visible;
(3) ECOSTRESS day/night absolute and anomaly; (4) DTR, DTR anomaly, day-vs-night scatter, and a
Landsat-vs-ECOSTRESS cross-sensor check; (5) **the answer three ways** — observed mean SM,
predicted mean SM, and mean LST anomaly as three `layout_panel` squares in the identical `YlGnBu`
style, plus the n = 6 scatter of station mean LST anomaly against observed mean SM, **which is the
scientific claim**; (6) the monospace summary block. Caption must note that the colour scales run
in opposite senses on purpose — red = hot = expected dry, dark blue = wet.

**`plot_lst_timeseries.py` → `figures/lst_timeseries/{tile}.{png,pdf}` — the whole record.** One
row per station, ordered by `offset_px` as in `plot_network_timeseries.py`, sharing an x axis over
the full 2016–2022 label window. Left panel, twin axis: observed SM as a line, LST **anomaly** as
markers (filled = Landsat day, open = ECOSTRESS day, triangle = ECOSTRESS night) about a zero line
— anomaly rather than absolute, so all six panels share one y range and the vertical offsets
between stations are directly readable. Shade the ECOSTRESS-absent years (2016 → mid-2018) so the
coverage gap is honest. Right panel, narrow: that station's LST anomaly against its SM anomaly over
all retained dates, with `r` and `n`. Titles carry the §26 supervision language — the centre panel
reads "centre pixel (112,112) — SUPERVISED", the others "pixel (r,c) — N px off centre". Bottom
strip: tile-mean absolute LST across the record (the seasonal cycle the anomaly removes) as the
§29.10 plausibility check made visual, plus a per-year retained-date rug exposing the clear-sky
sampling.

### 29.9 LST as dense auxiliary supervision (forward-looking; design only)

> **STATUS after Phase A ran:** §29's main hypothesis failed, but **this idea survived it in better
> shape than it entered** — see §29.15. The seasonal climatology shows the 30 m LST field is
> 0.967-coherent across all twelve months, i.e. a stable 5–9 K landscape texture. That is nearly
> worthless as a model *input* (DEM and LULC already supply static structure) but it is a virtue in
> a *target*: a consistent field is easier to learn than a noisy one, and the decoder's problem is
> that nothing supplies within-tile structure at all. This is now the strongest remaining §29
> thread, ahead of Phase B.

Recorded now because it changes what "success" in §29 means. If the correlation comes back
negative-as-predicted, the natural follow-on is not to feed LST in as another input but to
**supervise on it**.

**The arithmetic is the argument.** The model currently receives **one supervised pixel per tile
per day** — (112,112). One Landsat scene supplies **76 × 76 = 5776 pixels per tile** (ECOSTRESS
33 × 33 = 1089). Even at ~25% clear-date retention that is an enormous increase in *spatial*
supervision density, and it targets precisely the failure §26.11 measured: the decoder produces
almost no within-tile structure because nothing has ever asked it to. It pairs naturally with §28's
per-token head — an honest 160 m map needs a target at that resolution, and LST is the only one
available.

**Input or target? The asymmetry decides it.** As an *input*, LST exists on only ~20–40% of days,
needs a missingness mask, and is unavailable at inference exactly when it would be wanted. As an
*auxiliary target* it is training-only: cloudy days contribute no LST term and inference is
untouched. **Target is the cleaner design.**

Masked loss, `loss = (per_px * mask).sum() / mask.sum().clamp(min=1.0)`, with three things that
bite. (i) **Empty masks** — a fully-clouded sample gives `mask.sum() == 0`, NaN without the clamp
and a silently zero gradient with it; under DDP every rank must contribute or gradients go
inconsistent, so reduce `sum(loss)` and `sum(mask)` **across ranks** and divide once rather than
averaging per-rank means. (ii) **Weighting** — clear pixels per tile range 0–5776, so a per-sample
mean weights a 3%-clear tile equally with a fully clear one; global sum/sum is usually right, but
state which was chosen. (iii) **The dry-sky bias becomes a *training* bias, not merely a reporting
caveat** — LST supervision exists only on clear days, which skew dry, so the model would learn
thermal structure conditioned on dry conditions. Measure it; do not assume it benign.

### 29.10 Honest limits

Clear-sky sampling is biased toward dry days — the same bias flagged for S2 in §27b.3, and it works
*against* detecting wet anomalies; report retained-date counts and the SM distribution on retained
versus dropped dates. **n = 6 per date gives a 95% CI of roughly ±0.7 on a single-date `r`** —
one date proves nothing, power comes only from aggregating hundreds of dates plus the pooled test
plus the sign test, and no single impressive `r` may be quoted. One catchment (TxSON rangeland,
Texas), exactly as §26.8 states for the model. LST responds to albedo, vegetation, roughness, slope
and aspect as well as moisture — station-pair differencing removes what is tile-constant but not
these. Timing mismatch: Landsat samples ~11:00 CST instantaneously, labels are daily means; for
ECOSTRESS nights, report the headline under both `--night-assign same` and `prev`, since a 02:00
overpass arguably reflects the previous day's drydown. ECOSTRESS covers only ~63% of the label
window and CR1000-2's record ends 2021-06 (n = 1983 against 2503 for the others), so report
per-station retained-date counts, not tile totals.

### 29.11 Verification

Station pixels project correctly into the LST grid — recomputing at 10 m must reproduce all 96 rows
of `csvs/txson_readouts.csv` exactly, and the six CR200-18 stations must land at (112,112),
(72,105), (114,43), (25,109), (62,33), (193,65). Absolute LST plausible for Texas (~270–330 K) with
the right seasonal phase and amplitude. Cloud-flagged pixels excluded — **the retained-LST histogram
must have no cold tail**, which validates the mask independently of any bit table. The six CR200-18
SM means must reproduce **0.1367 / 0.1197 / 0.1826 / 0.2323 / 0.1857 / 0.2865** to 1e-3 **from the
parquet**. Shuffle control — permuting station labels within a tile must destroy `r_date`. Census
assertions on every Landsat item: EPSG 32614, `(15,15) mod 30` origin, `lwir11` present. ECOSTRESS:
`crs.to_epsg() == 32614` and an identical transform on every 14RNU granule, with `day_night_flag`
consistent with local hour. And `--smoke` must reproduce the already-verified 2019-08-16 read —
LST ≈ 313.7 / 318.3 / 331.0 K, ST_QA ≈ 2.84 K, `qa_pixel = 21824`.

### 29.12 Sequencing

**Phase A.** (0) Census — `slurm/census_lst.sh`, < 5 min, both sensors since neither half needs
auth. (1) `--smoke`, then `jobs/landsat_st_mpc.sh` — 298 scenes, 1.5–2 GB, ~25 min at 12 workers.
(2) `slurm/lst_heterogeneity.sh --sensor landsat` — minutes with Pool(64). (3) `slurm/plot_lst.sh`
→ both figures. (4) Write up. **Phase A alone answers the question.**

**Phase B.** (5) Earthdata registration — do it during Phase A. (6) `conda env update` for
`earthaccess` + `pyarrow`. (7) `--smoke`, then `jobs/ecostress_lste.sh` — the long pole at 6–12 h,
roughly halved by the `_cloud.tif` gate. (8) Re-run the analysis and figures with `--sensor both`;
the DTR test and figure rows 3–4 light up. (9) Extend the write-up.

All jobs on `rome` with `--mail-type=BEGIN,END,FAIL --mail-user=ktm.prajwalkhanal@gmail.com`;
download jobs `conda activate soilmoisture` per the `jobs/` idiom, analysis jobs
`conda run -n terramind` per the `slurm/` idiom. Total scratch < 15 GB against 1.9 P free.

> **What actually happened (2026-08-13).** Phase A ran end to end. Deviations from the plan above,
> all recorded so the estimates can be trusted next time:
> * **Speed.** Both downloads finished in **under one minute**, not ~25. The full 1162×1168 AOI is
>   962 MB; the 76×76 tile window is 7.7 MB. HTTP round-trips, not bytes, are the cost, and 246
>   scenes at 12 workers is nothing.
> * **Scene count.** 247 after platform+tier+asset+`cloud<80`, not 298 — the 298 figure was before
>   the cloud filter. 246 downloaded, 1 failed (a corrupt `QA_PIXEL.TIF` on
>   `LC09_L2SP_028039_20220519`, source-side, not retryable).
> * **Script names** differ from the plan: `download_landsat_st_mpc.py` (as planned),
>   `extract_lst_timeseries.py` + `analyze_lst_heterogeneity.py` (planned as one
>   `analyze_lst_heterogeneity.py`), `plot_lst.py` + `plot_lst_seasonal.py` (planned as
>   `plot_tile_lst.py` + `plot_lst_timeseries.py`). SLURM: `slurm/landsat_st.sh`,
>   `slurm/lst_pipeline.sh`, `slurm/plot_lst.sh`, `slurm/plot_lst_seasonal.sh`.
> * **The census (step 0) was never run as a separate job** — its Landsat half was answered during
>   planning and its assertions were folded into the downloader (`assert_grid_invariants`), so
>   `census_lst_sources.py` does not exist. The ECOSTRESS half is still unrun and is the first
>   thing Phase B needs.
> * **Two coordinate systems, one bug.** `extract_lst_timeseries.py` initially wrote the 30 m LST
>   raster indices into `row`/`col`, which the plotting code read as 10 m tile pixels — putting
>   every station marker in the top-left corner. Now written explicitly as `lst_row`/`lst_col` and
>   `tile_row`/`tile_col`, with an assert in the plotter. Never let those two share a column name.


### 29.13 RESULT — Phase A, run 2026-08-13

**Answer: no. Within-tile LST anomaly does not track station-to-station soil moisture.** The
apparent positive pooled correlation is a Simpson's-paradox artefact of station identity.

**What ran.** 246 Landsat 8/9 C2 L2 Tier-1 scenes over 2016-01-12 → 2022-10-27, downloaded twice
— once as the 76×76 CR200-18 tile window and once as the full 1162×1168 AOI (962 MB). Both jobs
finished in **under one minute**, not the 20–30 min estimated in §29.12. 116 dates survived the
clear-sky filter; 546 station-date records carry both clear LST and same-day observed SM.

**Verification, all passed.** All 96 readout pixels round-trip exactly; the six CR200-18 reference
pixels land at (112,112) (72,105) (114,43) (25,109) (62,33) (193,65); the six reference SM means
reproduce to 1e-3 **from the parquet**; LST spans 280.9–332.0 K with correct seasonal phase;
median per-pixel ST_QA 2.13 K.

**The thermal pattern is real and well resolved.** Station-mean anomalies span **1.80 K** with a
standard error of ~0.07 K — **27× the standard error**. Single-date within-tile spread is 2.26 K
median against a ~2 K noise floor, so no single date means much, but the persistent pattern is
unambiguous. LST at 30 m *is* spatially heterogeneous inside the tile, exactly as §27a found for
the reflectance embeddings.

**But it does not track soil moisture.**

| test | r | verdict |
|---|---|---|
| station level (n = 5, spatial) | **+0.327** (p = 0.59) | not significant |
| pooled station-dates (n = 546) | **+0.167** (p = 9e-5) | **confounded — see below** |
| per-date across stations (115 dates) | mean **+0.244**, 83% positive | same confound |
| **within-station (fixed effects)** | **−0.077** (p = 0.07) | ~zero, marginally negative |

Per-station, over time: CR1000-2 −0.39, CR200-15 −0.09, CR200-18 +0.06, CR200-24 −0.12,
CR200-25 +0.32 — mean −0.042, three of five negative.

**The confound, stated plainly.** The pooled and per-date tests mix two different questions:
*between* stations (does a persistently warmer pixel sit on persistently wetter soil?) and *within*
a station (when this pixel runs warm for its own average, is its soil drier?). Station identity
dominates: de-meaning both variables by station moves the correlation from **+0.167 to −0.077**, a
swing of 0.244 that is entirely between-station. **§29.7's per-date test as designed was not a
valid test of the hypothesis** — it must be run with station fixed effects, and the runbook design
should be read as amended. The shuffle control (empirical p = 0.0000) confirms the pooled number
is not noise; it is real structure answering a question nobody asked.

**A product limitation that costs the most informative station.** **18.2% of the CR200-18 tile has
no Landsat ST retrieval at all** — permanently, identically across all 246 scenes from *two*
different WRS-2 path/rows, so it is ground-fixed, and confirmed present as zeros in the raw
`lwir11` COG at source (1027/5776), not introduced by this pipeline. The hole occupies the southern
part of the tile, and the only station inside it is **CR200-6 — the wettest of the six at 0.2865**.
So the test ran on 5 stations spanning SM 0.114–0.221 instead of 6 spanning 0.114–0.287, losing
both the extreme and a third of the dynamic range. A side effect worth remembering: `tile_frac_clear`
can never exceed **0.818** for this tile, so any absolute clear-fraction threshold must be taken
relative to that ceiling, not to 1.0.

**Clear-sky bias measured, and it is mild.** Retained dates 116 of 2503; mean SM on retained days
0.1837 versus 0.1913 on dropped days — retained days are **drier by 0.0076 m³/m³** (KS = 0.054,
p = 0.049). Real, in the expected direction, small.

**What this means for the model.** LST does not rescue the §26.11 failure the way §29.1 hoped. The
model's within-tile spread is 0.0213 against an observed 0.1076 (20%), and LST offers no signal to
close that gap through this route. The finding does *not* touch **§29.9** — LST as a dense
auxiliary target is about supervision density (5776 px/tile against 1), not about LST correlating
with SM, and remains the more promising use.

**Honest limits on this negative.** Five stations in one tile in one catchment. Landsat samples
~11:00 CST instantaneously against daily-mean labels. The skin/5 cm decoupling is unaddressed and
is the most likely physical explanation — LST sees the top millimetres, TxSON sensors sit at 5 cm,
and a dry crust over moist soil breaks the chain. The NDVI/TVDI and terrain partials of §29.7 have
**not** been run. Phase B (ECOSTRESS night LST and diurnal range) is untouched and is the arm that
would test thermal inertia rather than instantaneous midday cooling.

**Artefacts.** `download_landsat_st_mpc.py`, `extract_lst_timeseries.py`,
`analyze_lst_heterogeneity.py`, `plot_lst.py`; `slurm/landsat_st.sh`, `slurm/lst_pipeline.sh`,
`slurm/plot_lst.sh`; `csvs/lst_station_timeseries.csv`, `csvs/lst_per_date_corr.csv`,
`csvs/lst_level_correlations.csv`, `csvs/lst_summary.json`;
`figures/tile_lst/ISMN_TxSON_CR200-18.png`, `figures/lst_timeseries/ISMN_TxSON_CR200-18.png`;
rasters at `/gpfs/scratch1/shared/pkhanal/lst/landsat_st/txson/{aoi,ISMN_TxSON_CR200-18}/`.

### 29.14 The confound, named — and why the two results are one story

§29.13 reports a pooled `r = +0.167` and a within-station `r = −0.077` on the *same 546 records*.
Both are arithmetically correct. They differ because they answer different questions, and the
design in §29.7 asked the wrong one.

The five stations each carry a nearly fixed offset in **both** variables:

| station | mean LST anomaly | mean SM anomaly |
|---|---|---|
| CR1000-2 | +1.43 K | ≈ +0.010 |
| CR200-15 | +1.35 K | ≈ +0.015 |
| CR200-18 | −0.37 K | ≈ −0.030 |
| CR200-25 | −0.27 K | ≈ −0.045 |
| CR200-24 | −0.28 K | ≈ +0.060 |

Read down the columns and the two persistently warm pixels sit on slightly wetter soil while two
of the three cool pixels sit on drier soil — a positive **between-station** relationship. The
pooled correlation is measuring that, i.e. **five points replicated 116 times each**, and its
significance (p = 9e-5) is inflated by treating 546 non-independent records as independent.

Read *within* one station over time and the relationship is gone: CR1000-2 −0.39, CR200-15 −0.09,
CR200-18 +0.06, CR200-24 −0.12, CR200-25 +0.32; mean −0.042.

**Which one tests the hypothesis?** §29.1 proposed a *mechanism* — wet soil evaporates, so it
cools. A mechanism acts through time at a place: if it is real, a pixel must be cooler on its own
wet days than on its own dry days. That is the within-station number, and it is ≈ 0. The
between-station number cannot distinguish evaporative cooling from any static covariate — soil
type, slope, canopy — that happens to co-vary with mean wetness across five points.

This is the standard Simpson's-paradox correction: **de-mean both variables by station before
pooling.** §29.7's per-date and pooled tests must be read as amended; the per-date test in
particular inherits the same confound, which is why 83% of its dates came out positive while the
mechanism it was meant to test is absent.

### 29.15 Seasonal climatology — the pattern is landscape, not weather (run 2026-08-13)

Prompted by the obvious follow-up: is the tile heterogeneous at all times of year, or only in
summer? All 246 scenes pooled by calendar month, mean anomaly per month, native 30 m
(`plot_lst_seasonal.py` → `figures/tile_lst_seasonal/ISMN_TxSON_CR200-18.png`).

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n scenes | 17 | 9 | 12 | 12 | 11 | 13 | 17 | 10 | 5 | 16 | 3 | 11 |
| mean LST (K) | 293 | 295 | 305 | 309 | 313 | 321 | 324 | 318 | 311 | 306 | 298 | 294 |
| spatial SD (K) | 2.10 | 2.31 | 2.13 | 2.34 | 2.65 | 2.66 | **2.96** | 2.59 | 2.37 | 2.53 | 2.07 | 2.08 |
| spatial r vs annual | +.959 | +.955 | +.950 | +.988 | +.971 | +.956 | +.955 | +.977 | +.976 | +.983 | +.968 | +.963 |

**The tile is heterogeneous in every month** — p95−p5 spread runs from 5.6 K in December to 9.5 K
in July across just 2.24 km, always above the ~2.1 K single-date noise floor.

**And it is the same pattern in every month.** Spatial correlation against the annual mean map is
**+0.95 to +0.99 in all twelve months** (mean **+0.967**). The cool region east of centre and the
warm western and southern margins are present in January, July and December alike; only the
*amplitude* changes, scaling with insolation (SD 2.1 K midwinter → 3.0 K midsummer). Station ranks
never flip: CR1000-2 and CR200-15 are the warm pair in all twelve months, CR200-18/24/25 the cool
trio in all twelve. Annual-mean pattern SD = 2.32 K.

**This closes §29 more firmly than §29.13's correlation did, and it explains it.** A pattern that
is 0.967-coherent across the seasonal cycle is a *static landscape property* — terrain, canopy,
soil, land cover — that breathes with solar forcing but does not move. **A static field cannot
track a dynamic variable.** It will correlate with the time-invariant component of soil moisture
(station identity — hence the pooled +0.167) and with nothing else (hence within-station −0.077).
The two results of §29.13 are not in tension; they are the same fact seen twice.

**It also flips the sign of the argument for §29.9.** For LST-as-*input* to a soil-moisture model,
a static field is nearly worthless — it duplicates what DEM and LULC already supply. But for
LST-as-*auxiliary-dense-target*, stability is a virtue: the model's within-tile spread is 20% of
observed because nothing supplies spatial structure at all, and a high-contrast (5–9 K),
physically-grounded, temporally *consistent* 30 m texture is easier to learn than a noisy one.
§29.9 survives this negative result better than it entered it.

**Caveats.** Nov (n = 3) and Sep (n = 5) are thin; their maps are noisier than the rest and their
SD is correspondingly less certain. The permanent 18.2% nodata region blanks the southern third of
every panel, so "the pattern" is characterised over 82% of the tile. One tile, one catchment.
---

## §30 Per-location processor — resolving within-tile heterogeneity (DESIGNED 2026-08-13, nothing built)

Written before code exists, for critique.

### 30.1 The problem

The model reproduces **temporal** variation almost perfectly and **spatial** variation barely at all.
§26.11: within-station temporal SD 0.051 vs observed 0.051; between-station SD **15–19% of observed**;
station level ordering **anti-correlated** with truth (r = −0.175 at CR200-18).

Three measured facts define it:

- **Pooling destroys most within-tile position information** — but only for S2/S1 *history* and
  DEM/LULC (`_cpu_pyramid_pool`, `dataset.py:190-220`; 97–98% of variance lost, §27a.2). The anchor
  already arrives un-pooled (`model.py:497-530`). Killing pooling is necessary, not sufficient.
- **The map is not flat, it is wrong.** §23: norm-std 0.093–0.524, strongest where the landscape is
  most uniform (flat control SCAN Crossroads Δ 0.188 m³/m³, 5× its own ubRMSE); `r(anomaly, DEM)` =
  −0.04…+0.21.
- **The temporal path is spatially constant.** Verified against `model.py:736-743`: `ctx_mask` starts
  as `~key_mask`, then zeroes the depth-CLS slots and the 196 spatial slots, and `context` is the
  masked mean of what remains — every valid non-CLS, non-spatial, non-pad token. FiLM applies it as a
  uniform per-channel scale/shift. Two precisions the first draft of this section got wrong:
  (i) the code comment calls it "temporal context for FiLM", but the mean **also includes the pooled
  static tokens** at indices 0–11 (DEM/LULC/soil pyramids); (ii) it is not "100% of the temporal
  signal" — the anchor contributes one date's spatial snapshot with a staleness embedding, so the
  bottleneck *is* date-dependent. The correct statement is that **all of the time-series signal —
  ERA5's 365 days, S2/S1 history, SIF, TWSA — reaches the decoder only through that one spatially
  constant vector**. The spatial pattern can change between samples; the **response to weather cannot
  vary across the tile**. Two stations 500 m apart therefore cannot have different *response
  functions*, only different means — exactly the observed signature: temporal SD perfect,
  between-station SD 15–19%.

**Why §27b does not close this off.** §27b time-averaged each station's centre token and asked
whether it predicts that station's multi-year mean SM (9.8% global, 0–4.2% within-network).
Time-averaging deletes the wetness signal and leaves a landscape descriptor — hence the UMAP
organising by Köppen class. Station-mean SM at global scale is a climate/texture/sensor-depth
quantity, not the within-tile anomaly. Within-network is ~100 km, not 500 m. And it tested
**TerraMind embeddings, not terrain derivatives** — TerraMind was trained on a generic reconstruction
objective, so there is no reason L12 linearly exposes convergence or insolation. Independent claims;
only the first was tested.

**The data is already on disk.** `/projects/prjs1968/satellite_zarr` — **993/993 stations**,
station-centred, 224×224 @ 10 m: `dem/data (1,224,224) float32` (real elevation, 0 NaN),
`lulc/data (4,224,224) uint8`, `s1_asc/data (N,2,224,224)`, `s2/data (N,12,224,224)`. Plus
`soil (21,74,74)` @ 30 m in `zarr_tokens`. All 40 TxSON stations, relief 46–88 m, mean slope
2.5–4.1°. `plot_tile_context.py` already reads this store. **Zero downloads** except MERIT Hydro.

### 30.2 Current architecture and where it fails

```
[DEM×4 | LULC×4 | Soil×4 | anchor×196 | S2hist×4N | S1hist×4N | ERA5×365 | SIF×50 | TWSA×12]
                            ~1035 tokens @ 768,  spatial_start = 12
                                          |
                          6 × full self-attention over ALL of it
                                          |
                    +---------------------+---------------------+
          bottleneck (B,768,14,14)                      context (B,768)
          = ctx[:, 12:208] reshaped                     = masked mean of all
                                                          NON-spatial tokens
                                          |
                    UNetDecoder  14->28->56->112->224 (bilinear)
                    skips L9/L6/L3 (14×14, interpolated), FiLM'd by `context`
                                          |
                              1×1 conv -> (B,3,224,224)
                                          |
                            loss reads (112,112) — 1 of 50,176
```

### 30.3 Proposed architecture

Contextformer's split (Benson et al., CVPR 2024, GreenEarthNet): context encoded once, temporal model
run **per location in parallel attending over time only**, weather shared.

```
BLOCK 1 — CONTEXT ENCODER              (once per sample, whole tile)
  anchor L12 (196,768)
  + dem_tok, lulc_tok (196,768)        additive, zero-init
  + terrain_stem(static_hr) stride-16  additive, zero-init
  + soil, pooled pyramid (tile context)
        |  6 × self-attention
        v
  S (B,196,768)  one context vector PER LOCATION
  g (B,768)      tile-level summary

BLOCK 2 — WEATHER ENCODER              (once per sample, SHARED across locations)
  ERA5 (B,365,19), SIF, TWSA  -->  W (B,T,768)      never replicated per location

BLOCK 3 — PROCESSOR                    (per location k, weights SHARED)
  seq_k = [ S[:,k,:] , g , s2_hist[:,:,k,:] , s1_hist[:,:,k,:] ]   ~102 tokens
            varies   const    varies             varies
        |  L × ( self-attn over seq_k -> cross-attn into W )
        v
  h_k (B,768)

BLOCK 4 — PER-PIXEL HEAD               (no upsampling anywhere)
  input(i,j) = [ S[:, i//16, j//16, :] , raster_stack[:, :, i, j] ]
                 nearest GATHER (index op)   measured 10 m pixels
        v  shared MLP / 1×1 conv
  (B,3,224,224)
```

**What varies per location — the whole mechanism.**

| fed to the processor at location k | varies with k? |
|---|---|
| `S[:, k, :]` own context token | **yes** |
| `s2_hist[:, :, k, :]`, `s1_hist[:, :, k, :]` own history | **yes** |
| `g` tile summary | no |
| `W` weather | no |

The encoders run **once**; the *slice* differs. If everything fed to the processor were
tile-constant, every location would emit an identical trajectory and today's failure would be rebuilt
with more machinery. **The fix is not per-pixel inputs, it is per-pixel *temporal* inputs.** Static
per-location features can only shift a location's mean; what lets two stations respond differently to
the same rain is that each carries its own S2/S1 history column.

**Why one-pixel supervision suffices.** The processor is one shared f(local context, weather) →
SM(t), fitted at supervised locations across **993 stations** spanning a huge context range. Running
it at an unsupervised location inside a tile is **interpolation in context space**, not
extrapolation. The training signal for within-tile heterogeneity comes from between-station variation
across the whole dataset — no mean-zero anomaly constraint and no dense LST supervision needed to
make it honest.

**Terrain enters at the context encoder, not at the end.** Conv stem stride-16 over the 224²
`static_hr` stack → `(768,14,14)`, added into the 196 spatial tokens like `dem_tok`. Only then can
terrain change the **response function** — a hollow should drain more slowly after rain, not merely
sit wetter. Bolted on after the fact it is a static offset. A *learned* stem beats hand-picked
pooling because curvature is a signed second derivative whose block mean largely cancels. MERIT's
90 m aggregates into 160 m tokens with no meaningful loss. Expect **`dem_tok` to become redundant** —
explicit slope/curvature/TWI encodes terrain far more directly than a TerraMind reconstruction
embedding, and §27a measured that DEM token as 13× register-dominated with 1.5% of variance surviving
pooling. Feed both, ablate `dem_tok`.

**No upsampling — and why Contextformer's head does not transfer.** L12 is 14×14 by construction
(16×16 px per token), so 160 m is a hard floor and upsampling adds nothing — §23's
64–83%-of-variance-at-14×14 is that measured. The **raw rasters are different**: `s1_asc` 10 m native
RTC, `s2` 10 m (20 m red-edge/SWIR), `lulc` 10 m, `dem` 30 m in a 10 m array, `soil` 30 m native.
Those are genuine carriers. Contextformer's per-token unpatchify head works because **every pixel has
an NDVI label**. Ours is one point sensor in 50,176 pixels, so unpatchify would invent sub-token
structure with nothing to check it — §17.7's objection to the DiVAE, and what §23 caught the decoder
doing.

| | 14×14 token out | unpatchify | **token gather + 10 m rasters** |
|---|---|---|---|
| resolution | 160 m | 224 | 224 |
| sub-token detail from | — | token embedding | measured pixels |
| defensible under point supervision | yes | **no** | **yes** |

**S1 is the source that matters** — DEM/soil/LULC are static and can only paint a fixed offset; S1
backscatter responds to surface wetness *and changes date to date*, the only **time-varying 10 m**
carrier available. Mild circularity (S1 already enters via tokens) but at 160 m, so the 10 m pixels
are new information. Handle **speckle** (multi-look or short temporal median) or the added detail is
variance, not signal.

**Blockiness becomes a diagnostic.** With nearest gather, inert fine rasters show as visible 16-px
blocking. Bilinear hides exactly that failure behind a smooth ramp — which is how the current map
looks plausible while being anti-correlated with the landscape. Do not smooth it away; report it.
**Do not add a within-block coordinate feature** `(i%16, j%16)` — it lets the model paint arbitrary
sub-token patterns from position alone.

### 30.3a Which channels earn a place in Block 4 — MEASURED 2026-08-13

A channel belongs at 224 only if it varies *within* a 160 m token; otherwise the token already
carries it and putting it in Block 4 is redundancy. Exact variance decomposition against the token
grid (total = between-token + within-token; the residual is orthogonal to the block means by
construction):

```python
tot    = a.var()                                   # over all 224x224 pixels
blk    = a.reshape(14,16,14,16).mean(axis=(1,3))   # mean inside each 16x16 token -> (14,14)
coarse = np.repeat(np.repeat(blk,16,0),16,1)
frac   = (a - coarse).var() / tot                  # = 1 - R^2 of "predict pixel by its token mean"
```

Averaged over 4 dense TxSON tiles (CR200-18, CR1000-2, CR200-3, CR200-26):

| channel | sub-token fraction | verdict |
|---|---|---|
| `curv_lap` | **0.96** | tier 1 |
| `curv_plan` | **0.95** | tier 1 |
| `tpi_100m` | **0.79** | tier 1 |
| **S1 VV (latest)** | **0.76** | tier 1 — and the only time-varying one |
| soil `socd` / `soc` | 0.54 / 0.50 | tier 2 |
| northness | 0.51 | tier 2 |
| S2 B08 NIR | 0.47 | tier 2 |
| `bd` | 0.44 | tier 2 |
| slope | 0.42 | tier 2 |
| LULC | 0.40 | tier 2 |
| `ph` | 0.36 | tier 3 |
| soil clay / sand / silt | 0.29 / 0.27 / 0.26 | context encoder only |
| **elevation** | **0.05** | **DROP from Block 4** — the token already has all of it |

**This metric measures non-redundancy, NOT usefulness.** Curvature is a second derivative, i.e. a
high-pass operator whose token means are ~0, so ~0.95 is close to tautological. It proves a token
cannot carry the channel; it does not prove the channel predicts soil moisture. Necessary, not
sufficient — sufficiency needs a regression of observed within-tile SM contrast on these channels.
Further caveats: **S1's 0.76 includes speckle**, which is sub-token noise by definition, so it is an
upper bound that multi-looking will lower; **soil is mildly inflated** by the nearest-neighbour
74→224 upsample used in the measurement; and 4 semi-arid Texas tiles at 46–88 m relief are not
globally representative. The elevation (0.05) and curvature (0.95) results are structural and hold
anywhere; the S1 and LULC numbers are landscape-dependent.

**Correction to `subkm_design.tex`:** it calls soil "noise" on the basis of 3% clay across CR200-18's
six stations. Across the **full tile** the range is 11 wt% clay and 22.75 wt% sand. The "noise"
verdict applies to those six points, not to the tile.

**Latest-clear imagery needs two extra inputs.** Feed S1 as **(latest, long-term median)** so the
model can difference them itself — current minus baseline *is* the wetness anomaly, while the
baseline is roughness and vegetation. And feed **days-since-acquisition**, because "latest clear" has
variable staleness; reuse the staleness embedding already at `model.py:505-514` rather than letting
stale imagery read as current.

**Channel budget (C = 24).** curv_plan/prof/lap (3) · tpi 30/100/300 m (3) · slope/northness/eastness
(3) · TWI/HAND (2) · S1 vv_now/vh_now/vv_med/vh_med/days_since (5) · S2 ndvi/SWIR-index/days_since
(3) · LULC embedding (2) · soil soc/socd/bd (3). Elevation, soil texture and the 30–60 / 60–100 cm
layers stay in Block 1.

### 30.3b The current model already has S and g — what it lacks is Block 3

Both ingredients exist today, wired wrong. `context` (B,768) at `model.py:736-743` is the analogue of
**g**, a tile summary; the 196 spatial tokens at `model.py:731-734` are the analogue of **S**.
Nothing ever combines a *specific location's* context with a *specific location's* history — they
meet only inside FiLM, where the temporal half is broadcast uniformly. **We have context and
per-location tokens; we have no per-location processing.** That is the entire delta of §30.

**Cost.**

| | training | inference |
|---|---|---|
| context encoder | full tile, once | once |
| weather encoder | once | once |
| processor | **1 location** | 196 |
| per-pixel head | **1 pixel** | 50,176 (one batched 1×1 conv) |

Un-pooling **reduces** payload: emit the supervised location's own history column
`s2_tok_k (60,768)` = 92 KB against `s2_pyr (60,4,768)` = 184 KB. The ~30 MB/sample IPC blowup
`_cpu_pyramid_pool` was built to prevent only occurs if all 196 columns ship, and training needs one.
`UNetDecoder` is deleted (four upsampling stages at 512/256/128/64 + three FiLM'd skips); §28.2's
figure is ~3.2 M activations/sample/depth against ~200 K. **Training should get cheaper than today.**

### 30.4 Terrain derivatives

**Slope: Horn (1981). Curvature: Zevenbergen & Thorne (1987).**

```python
s  = max(1, int(round(native_m / px_m)))          # dilate stencil: 3 px
h  = px_m * s                                      # effective spacing = 30 m
zp = np.pad(z.astype(np.float64), s, mode="edge")
def sh(di, dj): return zp[s+di*s : s+di*s+H, s+dj*s : s+dj*s+W]
z1,z2,z3 = sh(-1,-1), sh(-1,0), sh(-1,1)           # row 0 = north
z4,z5,z6 = sh( 0,-1), sh( 0,0), sh( 0,1)
z7,z8,z9 = sh( 1,-1), sh( 1,0), sh( 1,1)

p = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8*h)       # Horn dz/d(east)
q = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8*h)       # Horn dz/d(north)
D = ((z4 + z6)/2 - z5) / h**2                           # ZT 1/2 d2z/dx2
E = ((z2 + z8)/2 - z5) / h**2                           # ZT 1/2 d2z/dy2
F = (-z1 + z3 + z7 - z9) / (4*h**2)                     # ZT     d2z/dxdy

g2, g = p*p + q*q, np.hypot(p, q)
flat  = g2 < 1e-8                                       # curvature UNDEFINED on flat ground
g2s   = np.where(flat, 1.0, g2); gs = np.where(g < 1e-8, 1.0, g)
slope_deg = np.degrees(np.arctan(g))
northness = np.where(g < 1e-8, 0.0, -q/gs)              # +1 faces north
eastness  = np.where(g < 1e-8, 0.0, -p/gs)
curv_plan = np.where(flat, 0.0,  2*(D*q*q + E*p*p - F*p*q) / g2s)   # <0 convergent
curv_prof = np.where(flat, 0.0, -2*(D*p*p + E*q*q + F*p*q) / g2s)
curv_lap  = 2*(D + E)                                   # no denominator — always safe
tpi_r     = z - uniform_filter(z, size=int(round(2*r_m/px_m))|1, mode="nearest")
```

Plus **TWI = ln(upa / tan β)** and **HAND** from MERIT Hydro (`MERIT/Hydro/v1_0_1` on GEE). Upslope
contributing area is non-local and genuinely not computable from an isolated 2.24 km tile, but it is
precomputed globally, so **no wide DEM window is needed**. The earlier rejection ("not computable
from an isolated 2.24 km tile") is true of the tile, not of the problem.

Four gotchas: **(i)** the `s=3` dilation is load-bearing — `download_dem_cdse.py:202` fetched
`COPERNICUS_30 + resample_spatial(10m, bilinear)`, so the array is 10 m but the information is 30 m;
a 10 m stencil differentiates the interpolant. **(ii)** curvature divides by (p²+q²) and TxSON mean
slope is 2.5–4.1°, so near-flat pixels give ±10⁶ spikes without the `flat` mask. **(iii)** aspect as
sin/cos, never degrees. **(iv)** verify plan curvature is negative in a known hollow — ZT and ArcGIS
conventions differ.

**Which grid to derive on.** Correct order is derive at native resolution, then resample. Decimating
the 10 m array does not recover the native grid: GLO-30 ships in geographic coordinates at 1
arc-second and `download_dem_cdse.py` reprojects per-station to UTM (`get_utm_epsg`), so the 10 m
grid is not aligned to the source postings. The risk is artefacts, not accuracy — bilinear is
piecewise-linear inside each source cell, so curvature can acquire a periodic moiré against the
rotated UTM grid, and §23 already caught the decoder painting a padding artefact. **Decide by
measurement (~20 min):** curvature both ways on 3–4 TxSON tiles, look for a spectral peak at the
source-cell spacing. Clean → dilated stencil. Dirty → Gaussian low-pass at σ ≈ 15 m, or re-fetch
native GLO-30.

### 30.5 Not doing

**A water-movement loss.** A prior pushing predicted SM toward convergent terrain **asserts** the
relationship rather than measuring it — a plausible map with no evidence it is right, i.e. §23's
failure better dressed, and the liability §17.7 flagged. The honest version is TWI/HAND **as inputs**.

**Off-centre supervision as a load-bearing element.** The **train** split has only **12** off-centre
pairs on 12 tiles, and `location_group_id` never spans splits (0 of 906), so no reshuffle creates
more. Build the `pixel_idx` machinery (§28.9) because it is ~80 lines and eval already gathers this
way (`eval_predict.py:167-169`), but do not expect it to carry the run.

### 30.6 Risks

- **Position leakage.** Patches are station-centred, so the supervised location is always k=105 and
  the pixel always (112,112). The model can learn "read the centre" instead of the mapping. Requires
  a token-space translation crop (random 10×10 sub-window of the 14×14 grid — a slice of tensors
  already in memory), with `spatial_row_emb`/`spatial_col_emb` (`model.py:403-404`) indexed by
  **absolute** row/col. Not optional.
- **Weather may dominate.** 365 shared weather tokens against ~102 local ones. Log across-location SD
  of predictions every epoch; near-zero means the cross-attention is being ignored.
- **§27b's ceiling may be real.** If a station's own token genuinely carries 0–4.2% within-network
  skill, no architecture manufactures information. Fallback is §29 LST — but **amended 2026-08-13
  by §29.13**, which ran Phase A and found daytime Landsat LST does *not* track within-tile soil
  moisture (within-station fixed effects **−0.077**; the apparent +0.167 pooled correlation is a
  Simpson's-paradox artefact of station identity). So the fallback can no longer be justified by
  "tied to wetness by evaporative cooling" — that mechanism was tested and is absent at this tile,
  most likely through skin/5 cm decoupling. It survives on **§29.9's** grounds instead: LST as a
  **dense auxiliary target** is about supervision density (5776 px/tile against 1), not about LST
  correlating with SM. §16.4 proved the decoder paints correct structure under dense supervision
  (norm-std 0.0061 → 0.2504, corr 1.000). Phase B (ECOSTRESS night LST + diurnal range) is untouched
  and is the only arm that could still recover a direct LST↔SM link.

### 30.7 Files

| File | Change |
|---|---|
| `build_static_stack.py` | **new** — terrain derivatives + soil + LULC + TWI/HAND → `static_hr (C,224,224)` fp16 in `satellite_zarr`; `Pool(64)` |
| `compute_static_stats.py` | **new** — per-channel stats; template `compute_era5_stats.py` |
| `download_merit_hydro_gee.py` | **new** — `upa`/`hnd`; template `download_smap_gee.py`; `soilmoisture` env |
| `plot_architecture.py` | **new** — `figures/architecture_{current,proposed}.{png,pdf}` |
| `model.py` | split into context / weather / processor blocks; per-pixel gather head; delete `UNetDecoder` behind `--head token\|pixel\|unet`; `pixel_idx` in `masked_huber_loss` (replaces hardcoded `model.py:779`) |
| `dataset.py` | emit `s2_tok_k`/`s1_tok_k`, `dem_tok`/`lulc_tok`, `static_hr`, per-sample supervised index; translation crop; `anchor_valid` (`dataset.py:483-486` silently returns zero anchors as a real S2 anchor) |
| `train.py` | flags; `load_static_stats` after `.to(device)` (`train.py:899`) |
| `ckpt_utils.py` | new param names into the `new_keys` allowlist (`ckpt_utils.py:52`) |

Reuse: `load_tile_geometry()`/`station_pixel()` (`build_network_readouts.py:59-75`), the zarr read
pattern in `plot_tile_context.py`, `compute_era5_stats.py` as the stats template, the gather at
`eval_predict.py:167-169`.

**All heavy work via `sbatch`** — nothing long-running on the login/compute node. CPU jobs `Pool(64)`
+ `--cpus-per-task=64`; every job carries `--mail-type=BEGIN,END,FAIL` and
`--mail-user=ktm.prajwalkhanal@gmail.com`. Env: `terramind` for build/train, `soilmoisture` for
downloads.

### 30.8 Plan of action

**Phase 0 — rollback point + write-up, no compute.** Commit and tag the current tree before any
surgery, then branch `feat/per-location-processor` and push after each phase. Rollback has three
layers: **code** via git, **the trained model** (`checkpoints/cls_depth_star_reg/best.pt`, on disk,
not in git), and the `--head unet` flag, which keeps the old architecture runnable *inside* the new
codebase rather than only in history. Then this section, then the architecture figures.

**Phase 1 — data prep (CPU, sbatch).** Grid check (§30.4) → `download_merit_hydro_gee.py` →
`build_static_stack.py` + `compute_static_stats.py` over 993 stations, target **C ≤ 24**.

**Phase 2 — model surgery, all behind flags.** `dataset.py` emissions + translation crop +
`anchor_valid`; then `model.py` block split, processor, per-pixel gather head, `pixel_idx` loss.

**Phase 3 — validate before spending the allocation.** Regression gates (§30.9) → smoke (20 stations,
3 epochs), **measuring `data=` and peak RAM against the 540 GB baseline** → full run, budget ~42
GPU-h ≈ 10.5 h wallclock on one 4×H100 node (measured, job 25235976); expect less, decoder gone.

**Phase 4 — evaluate.** `sbatch slurm/eval_txson.sh` → `combine_network.py` →
`plot_network_timeseries.py`; re-run §23 `plot_spatial_heterogeneity.py`; leakage check; ablate the
flags; result back here as §30.n.

### 30.9 Verification

**Regression gates.** All flags off → one training step bit-identical to `main`; zero-init projections
and stem loaded from `best.pt` → forward bit-identical; `--head unet` reproduces
`cls_depth_star_reg`. Reuse the §26 provenance gate unchanged — **≥99.5% of rows bit-identical AND
max |Δ| ≤ 1 bf16 ULP** (0.000977); **do not re-tighten it.**

| metric | current | target |
|---|---|---|
| own-centre ubRMSE (0-10) | 0.0301 | ≤ |
| off-centre ubRMSE | 0.0345 | materially closer to own-centre |
| between-station spread, % of observed | **15–19%** | **> 35%** |
| r(pred level, obs level), CR200-18 | **−0.175** | **> 0** |

**Report 14×14 and 224 separately.** If 224 helps only via the static branch, that is a
resolution-limited offset, not resolved dynamics — §23 is explicit the distinction must not be
blurred.

**Physicality, not just variance.** norm-std should *drop* at SCAN Crossroads (the flat control
currently receiving the most painted variation, 0.52) and `r(anomaly, DEM)` should rise from
−0.04…+0.21. A run that increases spread while `r(anomaly, DEM)` stays near zero has made the map
more variable, not more correct, and **must not be reported as success**.

**Leakage.** Regress per-readout error on `offset_px` (a column in `csvs/txson_readouts.csv`);
§26.11 shows a positive slope today. Slope must fall to ~0 under the translation crop. If spread
improves while the slope stays positive, the gain is memorisation.

**Sanity constants.** CR200-18's six station tokens are **105, 62, 100, 20, 44, 172**; observed means
**0.1367 / 0.1197 / 0.1826 / 0.2323 / 0.1857 / 0.2865**.


## §31 Per-location processor at 32 m — the specified architecture (SPECIFIED 2026-08-17, nothing built)

Supersedes §30 wholesale. **The full spec, with flowcharts, is
`text/architecture_per_location.txt`** (1286 lines, structured summary → symbols → diagrams →
detail, written to be read cold). This section is the runbook record: what changed, why, and what
was measured. Figures: `figures/architecture_{current,proposed}.{png,pdf}` from
`plot_architecture.py`, regenerated to match.

Nothing built. No GPU spent. No pipeline code changed.

### 31.1 The diagnosis, unchanged

Verified at `model.py:736-743`: `context` is the masked mean of every valid non-CLS, non-spatial,
non-pad token, applied by FiLM as a **uniform** per-channel scale/shift. All time-series signal —
ERA5's 365 days, S2/S1 history, SIF, TWSA — reaches the decoder through **one spatially constant
vector**. The spatial pattern may change between samples; the *response to weather* cannot vary
across the tile. Two stations 500 m apart can be given different means, never different behaviour.

Signature: within-station temporal SD 0.051 against observed 0.051; between-station SD **15–19%**;
r(pred level, obs level) at CR200-18 **−0.175**; §23 norm-std 0.093–0.524, **highest at the flattest
site**. The map is not flat, it is wrong.

§30.3b named the delta correctly and it still stands: the model has `S` (196 spatial tokens,
`model.py:731-734`) and `g` (`context`), and nothing that combines a *particular* location's context
with *that* location's history.

### 31.2 Four changes from §30, each argued rather than assumed

**(a) Output is 70×70 @ 32 m**, not 224×224 @ 10 m. 30 m does not divide the 2240 m tile —
2240/30 = 74.67, so each 160 m token would straddle a ragged 5.33 cells. 32 m gives 2240/32 = 70
exactly and 70/14 = 5 exactly, so every token covers precisely a 5×5 block and the gather is an
integer divide. 32 m is 6.7% coarser, immaterial when the thermal band is 100 m and TWI/HAND are
30 m. It also removes the asterisk from the resolution claim: at 32 m every channel is at or below
the output resolution except the token itself.

**(b) Block 4 was dropped, then reinstated.** Dropped when terrain was MERIT-only at 90 m — with no
real sub-160 m carrier, a fine output is an upsampled coarse field, i.e. §17.7's DiVAE objection.
Reinstated once terrain moved to our own 30 m routing *and* it was recognised that S1/S2 at 10 m
were on disk the whole time. §30.3's criterion was never resolution; it was whether sub-token detail
is **measured** or **invented**. C = 17 measured channels.

**(c) MERIT becomes validator and calibrator, not product.** `upa` inside the tile says whether the
tile contains a channel; `upa` at the station sizes the window; our accumulation against MERIT's at
the station is a **per-station boundary-capture pass/fail with ground truth**; `hnd ≈ 0` calibrates
our stream threshold instead of us inventing one; and it is the fallback where no window is
practical. TWI/HAND are computed from our own conditioned wide DEM at native 30 m.

**(d) LST becomes a live component**, not merely the §29 fallback — see §31.5.

### 31.3 The architecture

```
BLOCK 1  CONTEXT ENCODER   once per sample, whole tile
  anchor_l12 + spatial_pe + modality/staleness + terrain_tokens   (zero-init, ADDED)
  + dem_pyr, lulc_pyr, soil, histories, ERA5, SIF, TWSA
  -> 6 x self-attention -> S (B,196,768) per-location ; g (B,768) tile summary

BLOCK 2  WEATHER ENCODER   once, SHARED   ERA5+SIF+TWSA -> W.  Never replicated:
                                          ERA5 is 9 km, identical over the tile.

BLOCK 3  PROCESSOR  per location k, weights SHARED          <<< THE DELTA >>>
  seq_k = [ S[:,k,:] | g | s2_tok_k (60) | s1_tok_k (40) ]   ~102 tokens
             varies   (=)      varies          varies
  L x ( self-attn over seq_k -> cross-attn into W )  -> h_k (B,768) @ 160 m
  train 1 location / infer 196

BLOCK 4  PER-CELL HEAD   no upsampling
  out(i,j) = MLP([ h[i//5, j//5] , fine[:,i,j] ])   -> (B,3,70,70) @ 32 m
  train 1 cell / infer 4,900 (one batched 1x1 conv)
```

**Terrain enters at Block 1, additively, zero-init.** Three consequences, each load-bearing:
bit-identity is free (adding zero changes nothing, so the day-0 forward equals `best.pt` and the
regression gate passes trivially); token k's terrain lands on token k, whereas appending pooled
terrain tokens the `dem_pyr`/`lulc_pyr` way would make it tile-level and able only to shift a mean;
and `g` receives terrain anyway through attention, since `context` is computed from the encoder
*output*.

**The cross-attention into `W` is the only place terrain can change the response function.**

**The U-Net is deleted from the forward path, and there is now a second, architectural reason.**
Beyond the supervision argument, `_get_skip_connections` returns `anchor_l3/l6/l9`
(`model.py:530`) — **all at 14×14**, `F.interpolate`d up at `model.py:258,261,264`. TerraMind is a
ViT and never downsamples, so there is **no high-resolution skip path at all**. A CNN U-Net's power
comes from skips at 224/112/56; ours had none. Kept runnable behind `--head unet`.

The supervision arm of the same argument was already measured, both ways: §16.4 gave the decoder
dense targets and it reproduced structure (norm-std 0.0061 → 0.2504, corr 1.000); §23 gave it one
label per tile and it produced maps anti-correlated with the terrain. Same decoder.

**Blockiness is the readout, not a defect.** With nearest gather, inert fine channels show as 5×5
blocks — which is the map saying Block 4 is contributing nothing. Bilinear hides exactly that behind
a smooth ramp, which is how the current map looks plausible while being anti-correlated with the
landscape. Do not smooth it. No `(i%5, j%5)` coordinate feature either.

### 31.4 Producing TWI and HAND

**MEASURED 2026-08-17: the lowest cell lies on the tile boundary in 25 of 30 sampled tiles
(83.3%)**, median relief 188 m (13.5–797 m). Flow leaves the 2.24 km window, so neither quantity is
computable from the current tile. TWI's error would be *aligned with the signal* — a valley cell
that really drains 50 km² receives only edge leakage, so the underestimate is largest exactly where
TWI should be high. HAND would often be **undefined**: a headwater tile may hold no cell above any
sensible threshold, and lowering it until one qualifies invents a stream.

Recipe: condition first (breach, not fill), **MFD/D∞ for accumulation but D8 for the HAND trace**
(under MFD flow splits, so there is no single stream cell to trace), `a = A_cells·cellsize`,
`beta = Horn(dem)` with `s=1` since the wide DEM is fetched at native 30 m, slope floor 1e-3, `A`
floored at one cell. Condition on a buffer then crop, or the window edge acts as a wall. Assert
whether the library returns cell counts or area — a factor of 30 inside a log is a silent offset.

**Window sizing.** The non-local error is largely **common-mode within a tile**: two cells on one
hillslope both miss the same upstream inflow, so their *difference* survives, and a tile-constant
offset is removed by standardisation. It breaks specifically at **channel cells**. Hence a checkable
criterion: **a tile is at risk iff it contains a channel**, read from MERIT `upa` inside the tile.
But `a` still varies within a tile and cannot be left at 90 m — across one hillslope `Δln(a) ≈ 2.8`
against `Δln(tanβ) ≈ 1.6`. Proposed window 10 km at 30 m = 333² cells ≈ 0.44 MB/station, **~440 MB
for all 993**. Compute was never the constraint.

**Largest correctness risk: GLO-30 is a DSM.** Canopy and buildings are in it, so a 20 m tree line
dams a stream and conditioning routes flow around a ridge that does not exist — precisely why MERIT
Hydro exists. Prefer **FABDEM**; confirm the access route rather than assuming it. Failing that,
breach aggressively. **None of `pyflwdir` / `richdem` / `whitebox` / `pysheds` is installed in
either env** (verified).

### 31.5 The theory, and why it dictates Block 3

TOPMODEL (Beven & Kirkby 1979): `R·a = T₀e^(−fz)·tanβ` ⇒ `z = (1/f)[ln(T₀/R) − TWI]`. **Depth to
water table is linear in −TWI.** That is the entire theoretical content — TWI is a
water-table-depth index, not a soil moisture index. HAND (Rennó 2008; Nobre 2011) approximates the
same quantity with fewer assumptions and is numerically more robust, being a height difference where
TWI's `a` is very sensitive to the flow algorithm.

**Neither predicts *surface* moisture except through water-table coupling.** Capillary rise is
~0.1–0.3 m in sand, 1–2 m in silt, 3–10 m in clay; above a few metres HAND the top 10 cm is
hydraulically decoupled and governed by infiltration and evaporative demand — a 1-D vertical
process. TxSON has 46–88 m relief.

**Grayson et al. (1997), and Tarrawarra (Western, Grayson & Blöschl): terrain control of surface
moisture is STATE-DEPENDENT.** Wet → lateral flow organises the pattern and terrain explains much
("nonlocal control"). Dry → soil and vegetation dominate and terrain explains almost nothing
("local control").

**This is a stronger argument for Block 3 than anything in §30.** A state-dependent effect requires
terrain **interacted with wetness state**. Cross-attention provides that; a uniform FiLM scale/shift
structurally cannot express it. It also means a *static* terrain input is the wrong functional form
— it would fit the average of two regimes and match neither. And it means **the sufficiency gate
must be split wet/dry**, or a pooled regression averages the two regimes into nothing.

Stated fairly: if lateral flow does not reach 0–10 cm, heterogeneity still comes from Grayson's
local controls — texture, vegetation, aspect. §30.3a measured these varying within our tiles
(texture 0.26–0.29, socd/soc 0.50–0.54, northness 0.51). **TWI and HAND address only one of the two
candidate mechanisms.**

### 31.6 The LST auxiliary

A **side branch off `S`, not off `h_k`** — `h_k` would require running Block 3 at all 196 locations
every step and destroy the reason training is cheap, whereas `S` already exists. `Linear(768→1)`
over the 196 vectors → (14,14) predicted anomaly. **Target = Landsat ST averaged per token, minus
the tile mean for that date.** Subtracting the mean is load-bearing: absolute LST is season and
time-of-day, readable off ERA5, so predicting it teaches nothing spatial — and removing it kills
exactly the Simpson's-paradox artefact §29 found. `L = L_SM + λ·L_LST`, λ balanced by **gradient
magnitude, not pixel count**, decayed over training. Nothing downstream reads it; it only makes
gradient, and at inference it is not computed.

**It is smaller than §30.6 implied.** With the U-Net, a dense target was load-bearing — the only
thing stopping the decoder inventing structure (§16.4). Block 4 is not a decoder and has nothing to
paint with, so that problem largely dissolves. The auxiliary's job changes from *constraining the
decoder* to *giving Block 1 dense gradient*. Real, but nice-to-have.

Four honest caveats. **(i)** §29.9's "5776 px vs 1" is the grid, not the information — TIRS is
100 m, so ~22×22 ≈ **500** independent values per tile. Still ~500× better than 1. **(ii)** The
pattern is **static** (+0.967 month-to-month), so it disciplines the *static* branch, not the
response function. **(iii)** The 18.2% no-retrieval region **contains the wettest station** —
missingness correlated with the target. **(iv)** If the target is a fixed map per station and the
model can identify the station — §24.12 found identity dominates at depth, §27b's UMAP organised by
Köppen — it may **memorise 993 maps** rather than learn structure. Likely, not hypothetical: test
LST-head skill on validation against training stations in the first smoke run.

Separately, the **time-averaged anomaly map goes in as a Block-4 input channel** unconditionally —
cheap, no λ, no contamination risk.

### 31.7 Files

| File | Change |
|---|---|
| `download_merit_hydro_gee.py` | **new** — `upa`/`hnd`; validator, calibrator, fallback |
| `download_dem_cdse.py` | extend: wide-window, **native 30 m** mode (reuses the openEO job manager and the 1°-boundary buffer at `:105-116`) |
| `build_twi_hand.py` | **new** — condition → route → threshold → TWI/HAND → QC → `terrain_lo`/`terrain_hi` |
| `build_fine_stack.py` | **new** — S1 (speckle, medians, days_since), S2, LULC, soil, LST anomaly at 32 m |
| `dataset.py` | per-location history columns; terrain + fine stacks; `pixel_idx`/`token_idx`; translation crop; `anchor_valid` |
| `model.py` | terrain stem; block split; Block 3; Block 4 gather head; LST head; `UNetDecoder` behind `--head` |
| `train.py`, `ckpt_utils.py` | flags, stats loading, `new_keys` allowlist |

**Download template is `download_soil_openlandmap.py`** (`station_bbox_wgs84()` `:147`,
`read_patch()` `:171`, `process_station()` `:207`) — **not** `download_smap_gee.py`, which samples
points via `reduceRegions` and has no retry, resume or checkpointing. Take only its
`ee.Initialize(project=...)` lazy-import pattern (`:83`). Use **~8–16 workers with backoff, not
`Pool(64)`** — that rule is for local CPU scans and will throttle against a remote API.

### 31.8 Gates

**Regression, before any GPU.** All flags off → one training step bit-identical to `main`;
`terrain_stem` zero-init on `best.pt` weights → forward bit-identical; `--head unet` reproduces
`cls_depth_star_reg`. Reuse the §26 provenance gate **unchanged** (≥99.5% rows bit-identical, max
|Δ| ≤ 1 bf16 ULP = 0.000977). **Do not re-tighten it.**

**Terrain derivation.** Per-station boundary capture against MERIT `upa` — failures masked or fall
back, never silently used. Stream threshold calibrated against `hnd ≈ 0`; a threshold that cannot be
made to agree means conditioning failed, not that it needs more tuning. Forested vs open station
comparison to confirm the DSM problem is solved. TWI high in hollows / low on ridges against
`plot_tile_context.py`'s hillshade — **verify against a known hollow, sign conventions differ
between references**. Report the per-tile slope-floor fraction. Measure the sub-token variance
fraction of TWI/HAND (§30.3a's decomposition) — **not yet done for these channels**; it decides
whether they belong in Block 4 at all.

**Sufficiency — no GPU, runs on MERIT 90 m in parallel with the 30 m build.** §30.3a says outright
that its channel ranking "measures non-redundancy, NOT usefulness" and that sufficiency is **not yet
run**. Regress observed ΔSM across the **62 colocated pairs** (13 at 0–50 m, 4 at 50–160 m, 16 at
160–500 m, 29 at 500–1120 m; 8 networks) on ΔTWI/ΔHAND under station fixed effects, **split
wet/dry** per §31.5. The pair list was computed in-session in §26 and **never persisted** —
recompute and write `csvs/colocated_pairs.csv` this time. §29 Phase A was this same gate and it
killed that arm before the build.

**Skill targets** are §30.9's, unchanged. **Physicality**: norm-std must *drop* at SCAN Crossroads;
`r(anomaly, TWI)` positive **and stronger wet than dry** — a flat wet/dry profile means a static
offset dressed as terrain. More variance still anti-correlated with the landscape is §23 repeated
and **must not be reported as success**. Report 14×14 and 70×70 separately. Log across-location SD
of predictions every epoch: near-zero means Block 3's cross-attention is ignored and the
architecture has silently reverted — the failure that looks like success until you plot a map.

### 31.9 Sequence

> **SUPERSEDED BY §32.7** (2026-08-23). The architecture in §31.1–31.8 stands unchanged; only this
> ordering is replaced. MERIT no longer runs first as a window-sizer — it is now a **required gate**
> on the derived terrain (§32.6), and the DEM is fetched per *region* rather than per station.
> **Step 0 below is also wrong**: branch from `main`, not from the tag — the tag and `main` carry
> byte-identical training code, so branching from the tag would only discard §31/§32 and the logs.
> See §32.7 step 0.

0. Branch `feat/per-location-processor` from tag `pre-s30-architecture` (61c1773).
1. `download_merit_hydro_gee.py` — needed in **every** scenario, so start first.
2. **Plot the `upa` distribution across 993 stations** — the per-station go/no-go for self-computing
   terrain at all. Report how many are tractable at 5 / 10 / 50 km.
3. Wide-DEM fetch at native 30 m. **Confirm FABDEM access first.**
4. `build_twi_hand.py` + `build_fine_stack.py`, with §31.8's derivation gates.
5. Sufficiency gate, in parallel with 3–4.
6. `dataset.py` → `model.py` → regression gates → smoke (20 stations, 3 epochs, measuring `data=`
   and peak RAM against the 540 GB baseline) → full run ~42 GPU-h.
7. Evaluate, ablate, write back as §31.n.

### 31.10 Risks

**The physics may not reach 0–10 cm** — the largest risk and independent of DEM resolution. Same
shape as §29, which tested evaporative cooling and measured −0.077. This is why the gate runs on
90 m data *in parallel* with the 30 m build rather than after it.

**§27b's ceiling may be real** — own-token within-network skill 0–4.2%.

**But a terrain failure does not kill the project.** The ablation row that matters is **Block 3 on,
terrain off**: it isolates the architecture from the hydrology, because the per-location S2/S1
history columns do not depend on terrain at all. Unlike §29, a negative result prunes **one input**.

**The §29 fallback is weaker than §30.6 claimed.** §29.13 found daytime Landsat LST does not track
within-tile SM (within-station −0.077; the +0.167 pooled figure is a Simpson's-paradox artefact of
station identity). It survives only on §29.9's supervision-density grounds. Phase B (ECOSTRESS night
LST, diurnal range) is untouched and is the only arm that could still recover a direct LST↔SM link.

**Latent bug, fix regardless:** `dataset.py:483-486` returns all-zero anchors with `orbit_id=0`,
indistinguishable from a real S2 anchor (`dataset.py:504`), so a missing anchor silently receives
the "S2" modality embedding at `model.py:498,508`.

## §32 Terrain, DEM-first — compute TWI/HAND ourselves, MERIT as a required gate (PLANNED 2026-08-23, nothing built)

Refines §31's **sequence**, not its architecture. §31.1–31.7 stand unchanged: Block 3, the 70×70
@ 32 m output, terrain entering Block 1 zero-init, the LST side branch. What changes is the order
of the terrain build and what MERIT is for.

Nothing built. No GPU spent. No pipeline code changed.

### 32.1 What changed from §31.9, and why

§31.9 put `download_merit_hydro_gee.py` first, because MERIT `upa` at the station was to size each
DEM window. Three measurements this session invert that, and a fourth changes MERIT's role.

**(a) GEE is currently unusable.** The stored refresh token returns
`invalid_grant: Token has been expired or revoked` (`download_era5land_gee.py:45` credential path,
project `1066500857818`). Every MERIT route is blocked behind a manual `earthengine authenticate`.
The DEM route is blocked behind nothing.

**(b) Window sizing dissolves once the processing unit is a region rather than a station.** See
§32.3. Boundary capture then has a self-contained pre-test — trace the station's upslope mask and
check whether it reaches the region edge — which front-loads the cheap half of the MERIT gate but
does **not** replace it.

**(c) The sufficiency gate is materially stronger at 30 m than at MERIT's 90 m.** Recomputing the
colocated pairs from `station_splits.csv` (KD-tree on ECEF, 1120 m cutoff): **75 pairs across 84
distinct stations** — 15 at 0–50 m, 6 at 50–160 m, 25 at 160–500 m, 29 at 500–1120 m. **21 of 75
are separated by under 160 m**, i.e. under two MERIT cells, so ΔTWI is ≈0 for them by construction;
the 25 at 160–500 m span 2–5 cells. At 90 m the gate is really only powered on the 29 pairs beyond
500 m. Our own 30 m terrain restores ~50 usable pairs. §31.8's gate would have risked a **false
negative that killed a valid arm**.

(§31.8 records 62 pairs as 13/4/16/29. The recomputation gives 75 as 15/6/25/29, so §26 applied a
further filter — most plausibly overlapping observation periods. Reconcile and record it when
`csvs/colocated_pairs.csv` is finally written.)

**(d) MERIT is promoted, not demoted — from window-sizer to necessary condition.** It no longer
runs first, but **no station's TWI/HAND may be used by anything until its accumulation has been
checked against MERIT `upa`**. §32.6. This is a stronger requirement than §31.8's, which framed the
comparison as a per-station pass/fail among several derivation gates.

### 32.2 DEM source — GLO-30, and the TerraMind consistency argument

Same product already in the pipeline: `COPERNICUS_30` via `download_dem_cdse.py:197`, stored per
station as 224×224 @ 10 m bilinear, now living in `/projects/prjs1968/satellite_zarr/*.zarr/dem`.

**TerraMind pretrained on this same surface — verified this session.** TerraMesh's DEM modality is
Copernicus WorldDEM-30 ("produced using Copernicus WorldDEM-30 © DLR e.V."), and the shipped
tokenizer config carries `domain="dem@264"`, `data_path="./data/TerraMesh/train"`
(`terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py:240`). So `dem_L12.pt` and
`dem_pyr` sit on exactly the surface the frozen encoder expects. **The DEM channel stays GLO-30**;
substituting FABDEM there would introduce a pretraining distribution shift for no gain.

**That argument carries no weight for the terrain derivation, and the split is load-bearing.**
TerraMind used elevation as texture and context; it never routed water through it. GLO-30 is
officially a **DSM** — TanDEM-X X-band scatters near canopy top — so a tree line dams a stream and
conditioning routes flow around a ridge that does not exist. This is precisely why MERIT Hydro
exists as a separate product.

In our pipeline the wide DEM is an **intermediate, not an input**: nothing downstream sees those
elevations, only TWI and HAND. So the mitigation costs nothing in consistency — breach aggressively
per §31.4, run §32.7's OSM stream check, and if that fails, swap **only the derivation step** to
FABDEM. FABDEM access was probed and confirmed: `zip+https://data.bris.ac.uk/datasets/
s5hqmjcdj8yo2ibzi9b4ew3sn/N50E000-N60E010_FABDEM_V1-2.zip!N52E006_FABDEM_V1-2.tif` opens in 0.4 s
and a 10 km window reads in 0.6 s, no auth, HTTP range supported. **§31.4's "confirm FABDEM access
first" is discharged.** 49 zips ≈ 39 GB would mirror every station at 50 km half-width if wanted.

### 32.3 Transport — same product, and the openEO job manager is not needed

GLO-30 is published as **public COGs on AWS**, verified this session:
`https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N52_00_E006_00_DEM/…_DEM.tif`
opens in 0.2 s and a 10 km window reads in 0.3 s, **no auth, no request signing**. So the wide
fetch reuses `download_soil_openlandmap.py`'s windowed-COG pattern (`station_bbox_wgs84()` `:147`,
`read_patch()` `:171` with 3-attempt exponential backoff, `process_station()` `:207`) rather than
`download_dem_cdse.py`'s 993-job `MultiBackendJobManager`. `download_dem_cdse.py` is left untouched
and keeps serving the existing 10 m tiles; CDSE remains the fallback.

**Two traps in the GLO-30 grid, both measured.** The N52 tile is 2400×3600 at
(0.000417° lon, 0.000278° lat) — GLO-30 **decimates longitude poleward**, 1.5 arcsec in x above
50°, changing again at 60/70/80°. Cells are not square in metres anywhere, and **56 stations are
above 60°N**. Flow routing and Horn slope assume square cells. Second, a 10 km window straddles 1°
tile boundaries often; `download_dem_cdse.py:105-116` buffers the bbox for CDSE, but direct COG
reads must open and mosaic up to four neighbouring tiles.

**Global routing was sized and rejected.** Global land at 30 m is 166 × 10⁹ cells: 0.66 TB for the
DEM array and ~4.6 TB working set against ~1 TB on a Snellius fat node — and conditioning is
non-local, so it cannot be naively tiled. The download would be fine; the routing is not. MERIT
Hydro *is* this problem solved at 90 m.

**The processing unit is a region.** Single-linkage clustering of the 993 stations:

| linkage | regions | cells | DEM | working set | largest region |
|---|---|---|---|---|---|
| 25 km | 537 | 0.34 e9 | 1.4 GB | 9 GB | 155 km |
| **50 km** | **353** | **0.84 e9** | **3.4 GB** | **24 GB** | **559 km** |
| 100 km | 202 | 2.76 e9 | 11 GB | 77 GB | 1112 km |
| global | — | 166 e9 | 0.66 TB | 4.6 TB | — |

**50 km / 353 regions** is chosen: the largest single region is ~3.5 × 10⁸ cells ≈ 10 GB working,
so every region fits an ordinary node under `Pool(16)`, while the ocean-and-empty-land overhead
stays near 2× the tight-window area. Per region: bbox + 10 km buffer → mosaic the covering GLO-30
COGs → reproject to a **per-region Lambert Azimuthal Equal Area at exactly 30 m**. A UTM zone is
only ~500 km wide and cannot carry a 559 km region.

Regions beat per-station windows on three counts, the third being the reason: boundary capture
becomes self-testable before MERIT arrives; **the 75 colocated pairs land inside one continuous
flow field**, so ΔTWI carries no differential boundary error exactly where the gate is most
sensitive; and there are no seams inside a region.

### 32.4 Producing TWI and HAND — the recipe, with the reason each choice is not the obvious one

```python
dem_raw  = read(region_dem)              # equal-area, 30 m, SQUARE cells
dem_cond = breach_least_cost(dem_raw)    # whitebox; then fill residual pits
flw   = pyflwdir.from_dem(dem_cond, nodata=..., transform=..., latlon=False)
acc   = flw.upstream_area(unit="m2")     # <- assert unit, do not trust the name
a     = acc / 30.0                       # m^2 -> specific catchment area, units of LENGTH
beta  = horn_slope(dem_raw, 30.0)        # RAW dem -- conditioning flattens valleys
twi   = np.log(a / np.clip(np.tan(beta), 1e-3, None))
streams = acc > (10 * 1e4)               # 10 ha = 111 cells at 30 m
hand    = flw.hand(drain=streams, elevtn=dem_raw)
```

1. **Breach, do not fill.** Filling raises a pit until it spills over its rim; breaching carves
   down through the obstruction. Our obstructions are canopy, because GLO-30 is a DSM — a 20 m tree
   line across a valley is a 20 m dam. Filling floods the valley into a fake lake and routes flow
   around the hill; breaching cuts a notch, which is what the real stream does under the canopy.
   `BreachDepressionsLeastCost`, then fill residuals.
2. **MFD for accumulation, D8 for the HAND trace.** D8 sends all water to the steepest of 8
   neighbours (compare steepest *slope* — diagonals are 42.4 m away, not 30), which stripes smooth
   hillslopes. MFD (Quinn 1991, slope^p, p≈1.1) splits among all downslope neighbours and is more
   physical for `a`. But under MFD there is no single downstream path, and HAND is *defined* by
   following one — hence a second D8-only field purely for tracing.
3. **Accumulation** is one topological-order pass, high to low. Assert cells-vs-area: a factor of
   900 is a constant 6.8 inside a log — harmless after standardisation, poisonous when mixing
   conventions between regions or comparing against MERIT.
4. **`a` is area per unit contour width**, so `a = A_cells · cellsize`, units of **metres, not m²**.
   Ridge 30 m, valley bottom thousands — that spread *is* §31.4's Δln(a) ≈ 2.8.
5. **Slope from the RAW DEM.** Horn 3×3, `β = atan(√((dz/dx)²+(dz/dy)²))`, 8·Δ denominators.
   Conditioning deliberately flattens things; taking slope off the conditioned surface puts
   artificial zeros exactly in the valleys where TWI matters most. Route on conditioned, measure
   slope on raw.
6. **Floor `tan β` at 1e-3**, and report the per-tile fraction hitting the floor — a large fraction
   means TWI is degenerate there, not merely clipped.
7. **HAND** = elevation − elevation of the first stream cell on the D8 trace. Zero on streams,
   **≥ 0 everywhere**; any negative value is a conditioning bug, not a feature. `log1p` before
   standardising.

Run once per region on the buffered mosaic, then crop each station's 2.24 km tile out of the
regional rasters — the region edge is the only wall and it is 10 km beyond the buffer. Outputs
`terrain_lo` (3,28,28) @ 80 m and `terrain_hi` (3,70,70) @ 32 m, channels `[TWI, HAND, valid_mask]`.

**Only the non-local terms need the wide DEM.** `a` must be integrated over tens of km, and HAND's
trace must reach a stream that lies outside the tile in **25 of 30** sampled cases (§31.4). The
local term `tan β` needs a one-cell halo and is already covered by the 10 m `dem` in the zarr.
Taking `a` from MERIT at 90 m and keeping only β fine was reconsidered and rejected again:
Δln(a) ≈ 2.8 across a hillslope against Δln(tan β) ≈ 1.6, so `a` is simultaneously the non-local
term *and* the larger sub-tile gradient. That combination is what makes it expensive.

**Environment split.** None of `pyflwdir` / `richdem` / `whitebox` / `pysheds` is installed in
either env (re-verified). `pyflwdir` 0.5.12 needs numba, which exists **only in `terramind`**
(0.67.0; `soilmoisture` has none). So `download_wide_dem.py` and `download_merit_hydro_gee.py` run
in `soilmoisture`, `build_twi_hand.py` in `terramind`. `whitebox` 2.3.6 as an independent second
opinion on breaching.

*API caveat:* `pyflwdir.from_dem` returns D8 with its own internal filling, so MFD accumulation
likely needs WhiteboxTools `FD8FlowAccumulation` / `DInfFlowAccumulation`. Verify in the pilot.

### 32.5 Validation — five tiers, cheapest and most decisive first

Two questions kept apart: **is the derivation correct** (tiers 1–5), and **does terrain matter for
soil moisture** (§32.8). A perfectly correct TWI can still fail the sufficiency gate.

**Tier 1 — synthetic DEMs with analytic answers.** Seconds to run; catches what real terrain hides.

| surface | required result |
|---|---|
| inclined plane, slope s | `a` per row = distance from top edge; slope exactly s; TWI linear downslope |
| cone (divergent) | `a` ≈ one cell everywhere; TWI low and near-uniform |
| V-valley | TWI maximal on the axis and **symmetric** — asymmetry = direction-encoding bug |
| single pit in a plane | after breaching, zero cells lack a downstream neighbour |

Plus the only exact test in the pipeline: **mass conservation** — accumulation summed over all
outlets equals the total cell count, to the integer.

**Tier 2 — internal consistency, no external truth.** HAND ≥ 0 and non-increasing downstream;
accumulation non-decreasing downstream; `a` ≥ 30 m everywhere; **buffer-doubling invariance**
(recompute at 20 km instead of 10 km — TWI/HAND at the stations must not move); catchment-inside-
region for every station; slope-floor fraction per tile.

**Tier 3 — two independent implementations.** pyflwdir vs WhiteboxTools on the identical
conditioned DEM, correlating ln(a) and HAND. The codebases share no lineage. Disagreement on flats
and where MFD/D8 genuinely differ is expected; disagreement on ordinary hillslopes is a bug.

**Tier 4 — independent external data, no GEE needed.** The tier that tests our largest exposure.
**OpenStreetMap waterways overlay** — does the derived stream network follow mapped rivers? If
canopy has dammed a valley, the derived stream visibly leaves the mapped channel. Plus **TWI over
hillshade** for ~20 stations spanning flat/steep/forested/arid, reusing `hillshade()` at
`plot_tile_context.py:84` — no numeric test catches a global sign flip, a person looking at a
picture does; verify against a *known* hollow, since sign conventions differ between references.
Plus a forested-vs-open station on the same landform, to size the DSM error.

**Tier 5 — the MERIT gate.** §32.6.

### 32.6 THE MERIT GATE — necessary condition, no station passes without it

**Access verified 2026-08-23** after re-auth in notebook mode (`gcloud` is absent on the login node
and the default auth mode shells out to it; `--auth_mode=notebook` avoids it). `ee.Initialize`
succeeds against project `1066500857818` via `~/.config/earthengine/credentials`, which is exactly
`_CREDENTIALS_FILE` at `download_era5land_gee.py:43`. `MERIT/Hydro/v1_0_1` is 3 arcsec (90 m),
bands `elv, dir, wth, wat, upa, upg, hnd, viswth`; a 25 km window returns 271×436.

**`upa` IS IN km², NOT m² — measured, not assumed.** `upa/upg` matches the 3-arcsec cell area in
km² to four decimals across latitudes 30–52°N (0.005292 vs 0.005283 at 52°N; 0.007389 vs 0.007402
at 30°N), and the ratio *tracking cell area with latitude* is what makes it decisive. `upg` is the
pixel count. This is §32.4's cells-vs-area trap landing on the reference side: a factor of **10⁶**
inside a log, not 900. Convert explicitly at the comparison, and assert it in a test.

Aggregate our 30 m accumulation to MERIT's 90 m grid and compare **over the tile, not at a point**:

- **Do not sample MERIT at the station coordinate.** Verified failure mode: four points placed on
  named large rivers all returned `upg` of 1–8, i.e. hillslope cells — the coordinates missed the
  channels by a couple of hundred metres. At 90 m, two cells off a river drops `upa` by orders of
  magnitude. Some ISMN stations sit near channels, so a point comparison would manufacture spurious
  failures out of georeferencing error.
- Compare instead across the 2.24 km footprint: **spatial correlation of ln(upa)** between ours and
  MERIT's, plus **magnitude agreement at matched quantiles**. This is robust to sub-cell
  registration and uses far more information than one cell. Where a station-level number is wanted,
  snap to the nearest MERIT stream cell within a stated radius and report the snap distance.
- Boundary capture stays **per-station pass/fail**. A station whose upslope area disagrees with
  MERIT in *magnitude* did not capture its catchment, whatever the region-edge pre-test said.
  Failures are **masked or fall back to MERIT — never silently used** (§31.8's rule, unchanged).
- Report the pass fraction and the distribution of `ln(our upa) − ln(MERIT upa)`. A tile-constant
  offset is tolerable and removed by standardisation; a station-varying one is not.
- **Stream threshold calibrated against `hnd ≈ 0`**, not invented and not swept as a substitute.
  Sweep {1, 5, 10, 50} ha additionally to report HAND's sensitivity alongside the calibrated value.
  §31.8's rule stands: a threshold that cannot be made to agree means conditioning failed, not that
  it needs more tuning.
- MERIT is a comparison at 90 m carrying its own errors, so disagreement in sub-90 m **structure**
  is expected and fine. The gate is on the **magnitude of upslope area** — exactly the non-local
  quantity the region was built to capture.

Nothing downstream — not the sufficiency gate, not `dataset.py` — consumes TWI/HAND for a station
that has not passed this.

`download_merit_hydro_gee.py` is new: `upa`, `upg`, `hnd`, `elv`, `dir` from `MERIT/Hydro/v1_0_1`
at 90 m, 25 km window per station (`upg` fetched alongside `upa` so the unit assertion is
reproducible offline). Template is `download_soil_openlandmap.py`, **not** `download_smap_gee.py`
(which samples points via `reduceRegions` with no retry, resume or checkpointing); take only its
lazy `ee.Initialize(project=...)` pattern `:83`, plus `_gee_credentials()` from
`download_era5land_gee.py:45`. **8–16 workers with backoff, not `Pool(64)`** — remote API.
`sampleRectangle` is fine at 25 km (118k cells, under the request cap); larger windows would need
`getDownloadURL` / `computePixels`.

### 32.7 Sequence

0. Branch `feat/per-location-processor` **from `main`, NOT from the tag.** §31.9 step 0 said to
   branch from `pre-s30-architecture` (61c1773); that is wrong and was carried over unchecked.
   **Verified 2026-08-23:** `model.py`, `dataset.py`, `train.py`, `ckpt_utils.py` and `utils.py` are
   **byte-identical** between the tag and `main` — everything that changed since is docs, figures
   and analysis scripts (`analyze_lst_heterogeneity.py`, `plot_lst*.py`, `plot_architecture.py`,
   the runbook, the architecture doc, `logs.txt`). Branching from the tag would discard §31, §32,
   `text/architecture_per_location.txt`, every session log and the whole §29 LST analysis, while
   gaining **nothing** for §31.8's bit-identity gate, since main's training code already *is* the
   tag's training code.
   **Branch strategy:** one long-lived branch, merged to `main` once the §32 gates pass — it exists
   so `main` stays runnable mid-rebuild, not for isolation. **Rollback is by TAG, not branch:** tag
   every run that produces a reported number (`run/<run_name>`) and every milestone worth returning
   to (`s32/…`), and record the tag name in this runbook beside the result. `git switch --detach
   <tag>` then reaches any past state without a zoo of branches.
1. Install `pyflwdir` 0.5.12 + `whitebox` 2.3.6 into `terramind`.
2. ~~`earthengine authenticate`~~ **DONE 2026-08-23**, notebook mode. Then
   `download_merit_hydro_gee.py`, in parallel with 3. No longer blocking.
3. `download_wide_dem.py` — 353 regions, GLO-30 from AWS, LAEA @ 30 m. Provenance-check against
   the zarr `dem` at the same footprint for ~5 stations.
4. `build_twi_hand.py` + Tier-1/2/3 validation.
5. Pilot on ~12 stations spanning flat (TxSON) / steep / forested / arid; Tier 4; sub-token
   variance fraction of TWI/HAND (§30.3a's decomposition, **never yet computed for these
   channels**) — it decides whether they belong in Block 4 at all.
6. **The MERIT gate**, §32.6.
7. `csvs/colocated_pairs.csv` — persist, and reconcile 75-vs-62 against §26.
8. **Sufficiency gate** on the 84 pair stations that passed §32.6: ΔSM on ΔTWI/ΔHAND under station
   fixed effects, **split wet/dry** per §31.5. No GPU. This is the science go/no-go.
9. Only if 8 passes: `build_fine_stack.py`, then `dataset.py` → `model.py` → §31.8's regression
   gates → smoke (20 stations, 3 epochs, measuring `data=` and peak RAM against the 540 GB
   baseline) → full run ~42 GPU-h. Evaluate, ablate, write back as §32.n.

### 32.8 Risks, unchanged from §31.10 except where noted

**The physics may not reach 0–10 cm** — still the largest risk, still independent of DEM
resolution, still the same shape as §29 (which tested evaporative cooling and measured −0.077).
§32's contribution is that the gate now runs on 30 m terrain over ~50 resolvable pairs instead of
90 m terrain over 29, so a negative result is trustworthy rather than possibly an artefact of
resolution.

**A terrain failure does not kill the project.** Ablation row 2 — Block 3 on, terrain off —
isolates the architecture from the hydrology, because the per-location S2/S1 history columns do not
depend on terrain at all. A negative result prunes **one input**.

**The DSM problem is the failure most likely to actually occur**, and no care in the routing code
fixes it. Decision rule: if Tier 4's OSM overlay shows streams leaving mapped channels in forested
regions, switch **those regions' derivation** to FABDEM rather than tuning the breaching.
Everything downstream is unaffected, because the wide DEM is an intermediate and the encoder's DEM
channel stays GLO-30 regardless (§32.2).

## §32.9 Build log — steps 0–4 executed (2026-08-24)

Branch `feat/per-location-processor` off `main` @ `a64b1e9`. §32.7 steps 0–3 are complete and
step 4 is running. No GPU spent; no training code touched.

### 32.9.1 What ran, and what it cost

| step | result | cost |
|---|---|---|
| 1. `pyflwdir` 0.5.12 + `whitebox` 2.3.6 into `terramind` | installed; WBT ships binary v2.4.0 | — |
| 2. `download_merit_hydro_gee.py` | **993/993 stations, 0 failures** | ~5 min, 24 CPU |
| 3. `build_dem_regions.py` → `download_wide_dem.py` | **353/353 regions, 0 failures**, 2.0 GB on disk | **2 min 10 s**, 10.9 GB peak RSS |
| 4. `test_terrain_tier1.py` | **all checks pass** | 7 s |
| 4. `build_twi_hand.py` | Tier 2 passes on 2 pilot regions; full run queued | 15 s and 42 s per 742² region |

`build_dem_regions.py` reproduces §32.3's clustering exactly: **353 regions at 50 km**, largest
420 × 562 km (§32.3 said 559). Cell count comes out 0.98e9 rather than 0.84e9 because the bbox
now includes the station tile half-width and snaps outward to the 30 m grid; 3.9 GB uncompressed,
2.0 GB deflated.

### 32.9.2 Measurements that changed the recipe

**`upa` is km², confirmed across all 993 stations, not just the four probes.** `upa/upg` against
the analytic 3-arcsec cell area: median **0.99974**, range 0.99552–1.00424. The assertion is
re-derived per station and written into every GeoTIFF's tags, so the 10⁶ factor cannot come back.

**MERIT's grid is centre-registered, and snapping to 1/1200° is half a cell wrong.** The asset's
`crs_transform` origin is `(-180.000416666667, 84.999583333333)`, i.e. cell *centres* sit on exact
multiples of 1/1200 and the *edges* are offset by half a cell. Snapping a window to multiples of
1/1200 — the obvious move, and what the first implementation did — put the GeoTIFF transform
**46.6 m** out in latitude, a systematic bias on every footprint comparison in §32.6. Snapping to
the asset's own edges instead: **0.34 m**, verified against `pixelLonLat` on 40 stations.

**§32.5's cone row is wrong and would have failed correct code.** It asks for "a ≈ one cell
everywhere; TWI low and near-uniform". On a cone the upslope area is πr² and the contour width is
2πr, so **a = r/2 exactly** — measured median accumulation 18.3 cells, not ~1. Divergence bounds
the contour-width *ratio*, not `a`. Replaced with the exact analytic value, which is a strictly
stronger test.

**§32.5's pit row cannot be posed through pyflwdir.** `from_dem` fills internally (§32.4's own API
caveat), so the raw DEM reports zero pits before conditioning and the test passes vacuously. Now
posed against a direct `interior_sinks()` count, with the caveat itself asserted as a test.

**§32.4's HAND definition is internally inconsistent, and the conditioned surface wins.** It asks
for both `elevtn=dem_raw` and "HAND ≥ 0 everywhere". Breaching carves a notch, so on the raw
surface a cell behind the obstruction sits *below* the raw elevation of the stream its flow path
reaches. Measured: carving reached **65 m and 152 m** on the two pilot regions, and raw-surface
HAND reached **−31.5 m and −84.9 m** over 0.7% and 1.3% of cells. HAND is now measured on the
conditioned surface — the one flow was actually routed on — which makes ≥ 0 and
non-increasing-downstream true by construction rather than aspirational. The raw-surface version
is still computed per region so the discrepancy stays measured. **Slope still comes from the raw
DEM** (§32.4 point 5 stands): there the raw surface is the honest one.

**`BREACH_DIST_CELLS` 100 → 20.** Least-cost breaching is ~O(dist²) per pit and GLO-30's flat sea
surface gives coastal regions enormous pit counts (region 123: 153,233 raw pits). At dist=100 a
single 742² region had not finished conditioning after 3 minutes; at dist=20 it takes 30 s. A
canopy dam is 1–3 cells wide at 30 m, so "breach aggressively" means willing to *cut*, not willing
to cut 3 km.

**Pits and flats are different things.** A pit has every neighbour strictly higher and conditioning
must remove it; a flat merely has an equal neighbour and is what a sea surface, lake or plateau
looks like in a DSM. Counting flats as sinks made Tier 2 fire on correct output — both pilot
regions reported exactly 5 "sinks", all of them flats.

### 32.9.3 Tooling traps worth not rediscovering

- **`earthengine authenticate --auth_mode=notebook`** — the default mode shells out to `gcloud`,
  absent on the login node. (Recorded in §32.6; it held.)
- **GLO-30 ocean tiles 404 rather than returning zeros**, and the COGs carry `nodata=None`, so
  sea level 0.0 is indistinguishable from no data by value. Absent tiles become NaN and the valid
  mask comes from tile coverage. Only 1 of 353 regions has a missing tile (1.6% NaN) and 1 crosses
  a longitude-decimation band; neither produced a NaN stripe.
- **WhiteboxTools' python wrapper cannot report failure.** `run_tool()` returns 0 unconditionally,
  so a Rust panic reads as success; it discards the tool's stdout when `verbose=False`, which is
  where the panic text goes; and it `chdir()`s the whole process for the duration of the call.
  `terrain_ops.run_wbt()` invokes the binary directly instead.
- **WhiteboxTools rejects a GeoTIFF with no geokeys** — synthetic grids need a CRS.
- **`FillDepressions` panics intermittently on identical input** ("Error unwrapping output",
  rc=101, ~1 call in 6). It is now conditional on a measured pit count — breaching with `fill=True`
  already leaves zero pits — so the flaky tool is off the common path for all 353 regions. A
  bounded retry sits underneath; it fired twice during the pilot and succeeded both times.
- **`rasterio`'s `Window.round_lengths()` rounds to nearest, not up** (≥1.4), which can clip a
  window's far edge by a cell and leave a one-pixel NaN stripe between two source groups — a fake
  ditch through the flow field. `download_wide_dem.py` does the window arithmetic explicitly.

### 32.9.4 Open, not fixed blind

**Tier 3 agreement is lower than §32.5 expects.** pyflwdir vs WhiteboxTools D8 accumulation on the
*identical* conditioned DEM: r(ln a) = **0.856** and **0.714**, and **0.69 on hillslopes**, with a
median log ratio of exactly 0 — dispersion, not bias. §32.5 says "disagreement on ordinary
hillslopes is a bug". The likely cause is that `pyflwdir.from_dem` resolves flats its own way, so
the two are not in fact operating on the same surface; the worse of the two regions is the one
where conditioning touched **32%** of cells against 2.5%. `--tier3` therefore runs on all 345 bulk
regions, where the sample is large enough to settle it. **This must be resolved before the
sufficiency gate**, because if D8 direction on hillslopes is unstable then so is HAND's trace.

**Coastal regions churn.** Region 123 (Puerto Rico): conditioning touched 32% of cells and 26.4%
of the region sits on the tan-slope floor — GLO-30's flat sea. Land stations should be unaffected,
because ocean is downstream of everything and so never contributes upslope area, but this wants
confirming in the pilot rather than assuming. A water mask from MERIT's `wat` band is the fallback.

**Still never computed: the sub-token variance fraction of TWI/HAND** (§32.7 step 5). It decides
whether these channels belong in Block 4 at all, and nothing measured so far speaks to it.

## §32.10 The sufficiency gate RAN, and TERRAIN FAILS IT (2026-08-24)

§32.7 step 8 executed. No GPU. `gate_sm_vs_terrain.py`, SLURM 25999866, 93 s.

### 32.10.1 The result

**890 stations** carry both derived terrain and ≥180 observed days. **75 colocated pairs
across 84 stations** — reproducing §32.9's recomputation exactly (15 / 6 / 25 / 29 at
0–50 / 50–160 / 160–500 / 500–1120 m) and confirming §31.8's 62 was the same set under a
further filter. **§32.7 step 7 is discharged**: `csvs/colocated_pairs.csv` is written, and
the 75-vs-62 discrepancy is resolved — §26 additionally required overlapping observation
periods. Enforcing ≥120 common observed days leaves **49 usable pairs**.

| test | r | p | n |
|---|---|---|---|
| ΔSM(0–10) ~ ΔHAND | **−0.102** | 0.480 | 49 |
| ΔSM(0–10) ~ ΔTWI | +0.049 | 0.737 | 49 |
| ΔSM(wet third) ~ ΔHAND | −0.104 | 0.472 | 49 |
| ΔSM(dry third) ~ ΔHAND | −0.101 | 0.487 | 49 |
| Köppen C (temperate) | −0.112 | 0.512 | 36 |
| Köppen D (continental) | −0.137 | 0.696 | 10 |

95% CI on the headline r: **[−0.373, +0.184]**. At n = 49 the detectable effect is
\|r\| > 0.28, so a *strong* terrain control is excluded; a weak one is not.

### 32.10.2 Why this is a fail and not an underpowered null

**The single most diagnostic prediction is absent.** Saturation excess is a wet-state
mechanism — §31.5 built the wet/dry split precisely to test it. Measured: wet −0.104,
dry −0.101. **No contrast whatsoever.** Whatever weak negative slope exists is not
behaving like saturation excess.

**It does not survive a power-matched restriction — it gets weaker.** Median \|ΔHAND\|
over usable pairs is only 3.91 m, so the obvious objection is that most pairs have no
terrain contrast. Restricting to the pairs that do:

| restriction | r | p | n |
|---|---|---|---|
| all | −0.102 | 0.480 | 49 |
| \|ΔHAND\| ≥ 2 m | −0.103 | 0.584 | 30 |
| \|ΔHAND\| ≥ 5 m | −0.065 | 0.764 | 23 |
| \|ΔHAND\| ≥ 10 m | **+0.002** | 0.995 | 11 |

A real effect strengthens when restricted to high-contrast pairs. This one vanishes.

**The response is not the problem.** Paired stations differ by a median
**\|ΔSM\| = 0.047 m³/m³**, and that difference holds the same sign on **95.8% of days**
(median across pairs). ΔSM is large and highly reproducible. The regressor has range,
the response is clean, and they are unrelated.

**HOBE did not generalise.** The 5-station HOBE look gave r = −0.637 with the right sign,
stronger when wet and stronger at 10–30 cm. In the full pairwise analysis Köppen D — which
contains HOBE — gives −0.137 over 10 pairs. The HOBE result was n = 5.

**Depth is the one question left genuinely open.** Only 4 pairs have 10–30 cm at both
stations and 3 have 30–100 cm, so the strongest HOBE signal (−0.776 at 10–30) could not be
tested globally. If terrain reaches soil moisture at all, it is below the surface layer,
and our labels are overwhelmingly 0–10 cm.

### 32.10.3 What this prunes, and what it does not

Exactly what §31.10 and §32.8 pre-committed to: **a terrain failure prunes ONE INPUT, not
the architecture.** Ablation row 2 — Block 3 on, terrain off — isolates them, because the
per-location S2/S1 history columns do not depend on terrain. Steps 1–7 of §32.7 all
succeeded and their products stand: 353 regions of 30 m terrain, MERIT for 993 stations,
the region design validated by 0/990 truncated catchments.

**The positive finding matters more than the negative one.** Stations ~400 m apart differ
by 0.047 m³/m³ with 95.8% day-to-day sign consistency. That is a large, reproducible,
per-location signal — precisely what §31's diagnosis said the FiLM context vector cannot
represent, and precisely what Block 3 exists to capture. It is simply not terrain.

**§32.7 step 9 proceeds with terrain OFF.** `build_fine_stack.py` drops `terrain_lo` /
`terrain_hi`; Block 1's zero-init terrain path (§31.3) is not built. The §32.6 MERIT gate
becomes unnecessary for the model and is demoted to a validation artefact, since nothing
downstream now consumes accumulation.

### 32.10.4 Caveats recorded against this verdict

- The **§32.6 MERIT gate never ran**, so no station's accumulation was validated. HAND is
  far less exposed to that than TWI, and HAND is what failed, so this does not rescue it.
- **TWI was independently disqualified** on stability (§32.9.4) before it was tested here,
  so its +0.049 is doubly uninformative.
- **49 pairs is a small n.** The verdict is "no strong terrain control on 0–10 cm soil
  moisture", not "terrain is irrelevant to hydrology".
- The **depth question is untested** for want of deep labels at paired stations.

## §32.11 The dynamics hypothesis, tested and closed (2026-08-24)

§32.10 rejected terrain on the LEVEL. One alternative remained: that HAND controls the
soil moisture DYNAMICS rather than its mean — a low-HAND site need not sit wetter, but
might drain more slowly. It is now tested and rejected, and the sequence is worth
recording because the first look said the opposite of the last one.

### 32.11.1 What the small sample said

`probe_drydown_dynamics.py` over the same 49 colocated pairs, on common dates:

| Δmetric vs ΔHAND | all 49 | ≥2 m | ≥5 m |
|---|---|---|---|
| Δ drydown τ | −0.173 | −0.255 | **−0.398, p = 0.046** (n=23) |
| Δ lag-1 memory | −0.234 (p = 0.10) | | |
| Δ recession floor | −0.016 | | |
| Δ wetting response | +0.080 | | |
| Δ mean level (§32.10) | −0.102 | −0.103 | −0.065 → +0.002 at ≥10 m |

The τ column **strengthened** monotonically with terrain contrast where the level column
**evaporated** — the exact diagnostic used to dismiss the level result, pointing the
other way. Both τ and memory carried the predicted negative sign.

### 32.11.2 Pre-registered checking, criteria fixed before running

**Split-half replication: PASS.** Splitting each pair's record in two by date gave
r = −0.388 (h1) and −0.378 (h2) at \|ΔHAND\| ≥ 5 m. And τ is a real station property, not
a fitting artefact: **τ(first half) vs τ(second half) correlates at r = +0.663, p < 0.001,
n = 97 station-halves.** This retired the worry that τ's narrow IQR (3.2 d, 2.9–3.6)
meant it was measuring sensor noise.

**Permutation over the whole search: FAIL.** Shuffling ΔHAND 10,000 times and rebuilding
the entire 6 × 3 grid: observed best \|r\| = 0.426 against a null median of 0.324 and 95th
percentile of **0.527**. **Corrected p = 0.196.** Searching 18 cells with 49 pairs
produces correlations of that size routinely.

So the limitation was power, not artefact — which justified exactly one more test.

### 32.11.3 THE decisive test

Pairing was bought to control climate, protocol and sensor type. **Network fixed effects
buy the same control over 890 stations instead of 49 pairs.** One pre-specified
regression, no thresholds, no strata, no metric selection:

> **τ(0–10 cm) ~ HAND, within-network demeaned. PASS = r < 0 and p < 0.05.**

**Result: r = +0.0881, p = 0.0099, n = 877 across 25 networks. FAIL.**

Statistically significant and **the wrong sign** — higher above drainage dries *more
slowly*, the reverse of the prediction — at r² = 0.8%. The 49-pair estimate of −0.398
reverses to +0.088 at n = 877, which is what a small-sample fluke looks like. Detectable
\|r\| at this n was 0.067, so the test had ample power.

Secondaries, all null and carrying no weight: memory −0.019, recession floor −0.003,
mean level −0.047, temporal sd +0.009. Köppen D gives τ +0.144 (p = 0.005), also wrong
signed.

### 32.11.4 Verdict

**Terrain is closed for this dataset at this depth.** Neither the level nor the dynamics
of 0–10 cm soil moisture is controlled by HAND, and TWI was independently disqualified on
stability (§32.9.4). §32.10's pruning stands without an asterisk.

Still formally untested, and honestly so: **depth.** Only 4 of 49 pairs carry 10–30 cm
labels and 3 carry 30–100 cm. If terrain reaches soil moisture it is below the surface
layer, and the labels are overwhelmingly 0–10 cm. That is a data limitation, not a
result, and it does not change what gets built now.

**What was worth the two days:** §32.7 steps 1–7 all succeeded and their products stand —
353 regions of validated 30 m terrain, MERIT for 993 stations, `colocated_pairs.csv`, and
a region design vindicated by 0 of 990 truncated catchments. And the finding that outlives
the terrain arm: **stations ~400 m apart differ by a median 0.047 m³/m³ with the same sign
on 95.8% of days.** A large, reproducible, per-location signal that FiLM's uniform context
vector provably cannot represent, and that Block 3 exists to capture. It is real. It is
simply not terrain.

## §33 Sentinel-1 in the decoder — a measured carrier at every scale (DESIGNED 2026-08-24, nothing built)

§32 closed terrain and left the question it was asked to answer still open. This section
answers it with a different carrier.

### 33.1 What §32 leaves standing

The positive finding survives intact and is the whole motivation here:

> **Stations ~400 m apart differ by a median 0.047 m³/m³, with the same sign on 95.8% of
> days.** (§32.10.2)

Large, reproducible, per-location. The signal exists. §32.10 and §32.11 established only
that **terrain is not what carries it** — HAND fails on level (r = −0.102, no wet/dry
contrast, vanishing at high ΔHAND) and on dynamics (r = **+0.088**, wrong-signed,
p = 0.0099, n = 877), and TWI was disqualified earlier on stability (§32.9.4).

A note on that stability probe, because the two results are easy to conflate:
`csvs/terrain_stability_probe.json` shows **HAND is the stable quantity** (r ≥ 0.96 under
1 m of DEM noise; ΔHAND across pairs 0.97–1.0000) while **TWI is not** (ΔTWI goes
0.339 → −0.163 → −0.106 under 0.05/0.2/1.0 m). It also shows **FABDEM is worse than
GLO-30**, so §32's pre-committed FABDEM swap is not indicated. None of this rescues
terrain: HAND is reproducible *and* irrelevant to 0–10 cm soil moisture. Stable ≠ useful.

So: **what varies at 400 m, changes with weather, and is measured everywhere?**

### 33.2 The diagnosis, restated in terms of what the decoder can see

`spatial_ctx` is computed at `model.py:732` and the decoder is driven by `context`
(`model.py:736-743`), the masked mean of every non-spatial token — one spatially constant
vector. §16.1 localised the collapse to the decoder, not the encoder (rel_spatial_std
1.164 → 1.106 through attention, then 0.089 / 0.095 / 0.034 → **0.0065** through the
decoder).

The mechanism is simpler than a modelling failure. **TerraMind is a ViT: it patches once
and never downsamples**, so `L3`, `L6`, `L9`, `L12` are all 14×14 (`model.py:530`). At
`model.py:257-264` those skips are interpolated up. **Between 160 m and 20 m the decoder
holds no measurement at all**, and §23 measured what fills the vacuum: confident structure
up to 0.19 m³/m³, strongest at the *flattest* site.

### 33.3 Only Sentinel-1 can carry it

| source | varies in SPACE | varies in TIME |
|---|---|---|
| ERA5 / SIF / TWSA | no (9 km) | yes |
| DEM / soil / texture | yes | no |
| LST | yes | no — +0.967 month-to-month (§29.15) |
| S2 | yes | yes, but clouds, and it sees the canopy |
| **S1 σ⁰** | **yes** | **yes** |

Per-station S1 counts, listed as NEVER MEASURED in earlier planning, are now measured:

```
   s1_asc   993 stations   median 212   p10  90   max 958
   s1_desc  910 stations   median 238   p10   5   max 956
   asc+desc                median 440   p10 144   only 9 stations below 50
   /projects/prjs1968/satellite_zarr/{station}.zarr/s1_{asc,desc}/
       data (N,2,224,224) fp16 dB [VV,VH]   dates (N,) b'YYYYMMDD'
```

§30.3a already flagged the same thing from the other direction: **S1 VV sub-token variance
fraction 0.76 — the only time-varying tier-1 channel** in the entire fine stack.

### 33.4 The decomposition

Raw backscatter is a sum of three things the network cannot separate on its own:

```
   sigma0(p,k)  =   c_k      +   (r_p - mu)   +   d(p,k)
                    permanent    how wet the      how wet THIS spot is
                    texture      whole tile is    versus its own norm
```

Computed per station × relative-orbit group × polarisation, at each decoder level ℓ:

```
   P        = 10^(sigma_dB/10)                        LINEAR power
   T_l(p,k) = 10*log10( mean of P over the f x f block ),  f = 224/l
   mu  = mean over all (p,k)      r_p = mean over k      c_k = mean over p
   d(p,k) = T_l(p,k) - r_p - c_k + mu
```

which is the double-centring projection
`D = (I - 11'/n) T (I - 11'/l²)`, giving `Σ_k d = 0 ∀p` and `Σ_p d = 0 ∀k`, and an exact
variance split `Var(T) = Var(r) + Var(c) + Var(d)` — verified numerically to 2e-07 dB².

**Why feed the decomposition and not raw σ⁰.** The three components sum back to raw σ⁰, so
nothing is lost. But `c_k` **cannot be derived from a single sample** — it needs the
multi-year archive. A network given raw σ⁰ could only recover it by memorising a per-pixel
mean for each of 993 stations, which is the memorisation this design exists to avoid and
which does not transfer out of sample. Computing `c_k` offline injects information the
model cannot otherwise obtain, and it *does* transfer, because at inference it is computed
from the new station's own archive.

Full derivation, equations and worked example: **`text/s1_decoder_input.pdf`**.

### 33.5 The four things that were measured before committing

**(a) Multilook in LINEAR power, never in dB.** Averaging dB gives the geometric mean of
power. Measured on CA-Cbo: mean bias **0.53 dB**, max **7.5 dB**, r = 0.952, worst in
heterogeneous cells — exactly where the signal is.

**(b) Group by relative orbit, `ord(date) mod 12`.** Mixing orbits injects viewing
geometry. The grouping validates itself:

```
   split-half r(c_k) WITHIN a group  (the ceiling)   0.9979  0.9981  0.9982
   BETWEEN groups   0 vs 1  0.9632   different orbit -> keep apart
                    0 vs 7  0.9552   different orbit -> keep apart
                    1 vs 7  0.9952   AT ceiling: S1A/S1B same orbit -> MERGE
```

Mixing orbits inflates the interaction term from ~17% to 18.3%. Drop groups with n < 20.

**(c) The interaction term is real and is not speckle.**

```
station                  n   season%  static%  INTER%   ac1     ac1(shuffled)
AmeriFlux_CA-Cbo       116    19.6     63.7     16.7    +0.240    -0.006
ISMN_TxSON_CR1000-1    113    37.7     50.1     12.2    +0.371    -0.011
ISMN_SCAN_Abrams       104    40.7     21.7     37.7    +0.478    -0.002
```

**The variance share is NOT the evidence.** Double-centring pure noise leaves 94–99% in
the interaction term. The evidence is that the pattern *repeats between passes* (+0.24 to
+0.48) while shuffled controls do not (−0.01). Report the autocorrelation against a
location-shuffled control everywhere this quantity is discussed; never the share alone.

**(d) Vegetation is present but is not what makes it coherent.** The cross-ratio
`CR = VH − VV` is the standard C-band vegetation descriptor (Vreugdenhil et al. 2018;
Copernicus SSM). Measured: `r(d_VV, d_CR)` = −0.40 to −0.58, so vegetation accounts for
16–34%. But three things say the coherent part is not phenology:

- the ACF decays to zero by ~6 passes (~72 d, e-folding 24–36 d) — a soil-moisture
  timescale, not a canopy one;
- a per-location annual harmonic explains only **9.8–22.3%** of `d_VV`, so 78–90% is
  episodic;
- **removing vegetation makes the signal stronger, not weaker** — lag-1 ACF rises
  0.240→0.328, 0.371→0.428, 0.478→0.546.

So the build carries `d* = d_VV − β·d_CR`, with β accepted **only if the residual's
autocorrelation increases**; otherwise shrink β to zero. Log both per group.

### 33.6 What is fed to each decoder stage

Three channels per stage, from the **most recent available pass** `p*`:

| stage | grid | cell | channels |
|---|---|---|---|
| bottleneck | 14×14 | 160 m | `c_14`, `d_14(p*)`, `w_p*` |
| `up1` | 28×28 | 80 m | `c_28`, `d_28(p*)`, `w_p*` |
| `up2` | 56×56 | 40 m | `c_56`, `d_56(p*)`, `w_p*` |
| `up3` | 112×112 | 20 m | `c_112`, `d_112(p*)`, `w_p*` |

`w_p = (r_p − r̄)/sd(r)` is the **catchment wetness state** — one standardised scalar per
date, broadcast across the grid. It must be fed back explicitly: double-centring deletes it
from `d` by construction, and it is what lets a conv express *response × wetness* rather
than a flat offset. Plus two housekeeping scalars: the age of `p*` in days, and a validity
flag (zero ⇒ `d = 0`, the natural "behaving like the tile average" value).

```python
x = self.bottle_proj(bottleneck)
x = torch.cat([x, c_14, d_14, w], 1)                                    #  512+3
x = self.up1(x); x = self.conv1(torch.cat([x, film9(skip_L9), c_28,  d_28,  w],1))
x = self.up2(x); x = self.conv2(torch.cat([x, film6(skip_L6), c_56,  d_56,  w],1))
x = self.up3(x); x = self.conv3(torch.cat([x, film3(skip_L3), c_112, d_112, w],1))
```

Each `in_channels` grows by 3 (≈12 k parameters). **Zero-init the new weight slices**
(`conv.block[0].weight[:, -3:] = 0`) so the forward pass is bit-identical to `best.pt` at
initialisation and §26's provenance gate applies unchanged.

**Output grid 112×112 @ 20 m**, not 224×224 @ 10 m. `2240/20 = 112` and `112/14 = 8`
exactly, and 20 m is where S1 actually resolves (IW is 5 × 20 m native; the RTC product is
*gridded* at 10 m but multilooked coarser). The `up4` stage is dropped — it is the most
expensive conv in the decoder and the only stage with no measured input. Claiming 10 m
would be claiming the grid rather than the resolution, which is precisely the flaw in
arXiv:2505.00265 (`RMSE 0.06–0.08`, validated only at points, no spatial validation at
all, final product resolution never stated).

### 33.7 LST as the target, at 100 m

The two sensors take **disjoint roles**: S1 is input only, LST is target only. Nothing
appears on both sides, so no leakage discipline is needed anywhere.

Grid: `2240/22 = 101.8 m`, a 22×22 array — the finest grid that divides the tile exactly
and stays coarser than TIRS's ~100 m, so every cell holds at least one independent thermal
sample and nothing is oversampled.

```python
p = self.head_lst(up1_feat)                        # Conv2d(256,1,1) -> (B,1,28,28)
p = F.interpolate(p, size=(22,22), mode='area')    # 80 m -> 101.8 m, AGGREGATION only
L_lst = huber(p[m], d_LST[m])
```

Both sides of the comparison are aggregations — the target goes 30 m → 101.8 m by
averaging, the prediction 80 m → 101.8 m — so nothing is invented on either side. Target
built by the same double-centring, **grouped by WRS-2 path/row** (different paths image at
different local times, hence different thermal states). Masked for cloud *and* the
no-retrieval region.

**Why it earns its place despite being coarser than three of the four S1 levels:** not
resolution — **cross-sensor disambiguation**. The open question on `d` is whether it tracks
soil moisture or differential surface roughness after rain; radar cannot separate those and
thermal can, because roughness does not cool a surface and evaporation does. A model handed
radar and required to produce a thermal field must learn the moisture mechanism; the
roughness explanation cannot pass that test.

**Prerequisite:** a global Landsat ST pull. §29.12 clocked the TxSON job at under a minute
for 246 scenes with a 7.7 MB per-station window, so 993 stations is ≈7.6 GB and 30–60 min
with `Pool(64)` — a scoped job, not a project. Caveat to report: clear-sky days are
measurably drier (§29: 0.1837 vs 0.1913, KS p = 0.049), so LST supervision lands
preferentially on dry days; report LST-head skill split by wetness tercile.

### 33.8 What is deliberately NOT in this design

- **No S1 target.** S1 is an input. The hidden-pass masking machinery of earlier drafts
  exists only when a quantity is on both sides; it is not needed here.
- **No colocated-pair loss.** 19 train pairs is too few to supervise with, and keeping
  pairs entirely out of the objective makes them an *uncontaminated* metric — any rise in
  between-station SD is then attributable to S1 alone, with no second term to argue about.
- **No terrain.** §32.10 and §32.11, closed.
- **No physics.** No θ_r, AWC, pedotransfer, transmissivity or TOPMODEL storage. The
  water-balance design in `text/architecture_state_model.tex` was considered and rejected:
  too many physical parameters, and its own §10.2/§10.5 concede equifinality and the
  wrong-physics failure mode.
- **No second transformer.** More parameters means more data, and the spatial dimension is
  where the data is thin.
- **No coordinates.** No latitude, longitude or elevation scalar is fed, and none is to be
  added. Verified 2026-08-25: the batch carries no coordinate of any kind — `era5` is 19 pure
  meteorological channels, and nothing else encodes position. Splits are **by station**, so
  coordinates would let the model interpolate a spatial climatology between training stations,
  inflating val and collapsing on the 219 oos stations in unseen regions.

  This is **not** the same as claiming the model cannot identify a station. `dem_pyr` (4,768),
  `lulc_pyr` (4,768) and especially `soil_patch` (21,74,74) are per-station constants and
  across 993 stations are almost certainly a unique fingerprint. §33.4's memorisation argument
  does not depend on the absence of an identifier: it is that recovering `c_k` by memorisation
  costs a 993-entry table of per-pixel means, and — decisively — **cannot transfer**, because a
  new station's fingerprint was never seen. Computing `c_k` offline is the right move whether
  or not an identifier exists.

### 33.9 Gates before any GPU

1. **`d`-vs-SM probe (CPU, an afternoon).** Regress observed SM anomaly on `d` at the
   station's own cell, **with station fixed effects** — §29 died on exactly that
   distinction. Include a persistence control (`d(t)` from `d(t−1)`) and a **land-cover and
   season split**: if `d` tracks SM only on cropland in the growing season it is phenology.
   This is the only gate that can kill the design. Reuse `station_mean_probe.py`'s ladder
   (`fit_block`, GroupKFold on `location_group_id`, RidgeCV, HGBR).
2. **Multilook assertion** — the build never averages dB directly.
3. **Same-orbit assertion** — no centring statistic or `d` value mixes ASC/DESC or
   relative orbits; validate the mod-12 grouping per station against its own split-half
   ceiling.
4. **Centring identities** — `max|mean_k d|` and `max|mean_p d|` < 1e-4 for every station.
5. **`c_k` sample size** — n ≥ 20 floor, n ≥ 50 comfortable (measured: RMSE falls to 14.5%
   of the `c_k` spread at n = 20, 7.6% at n = 50). Also check seasonal balance; a lopsided
   record gives a `c_k` that absorbs part of the interaction.

### 33.10 Open risks, stated before building

- **`d` may be real but not soil moisture.** 12–38% of σ⁰ variance being dynamic and
  spatial does not make it wetness. Differential surface roughness after rain would look
  the same to radar. Gate 1 and the LST target both attack this; neither is guaranteed.
- **Nothing forces the decoder to use the new channels.** With the ISMN probe as the only
  soil-moisture target, the conv may learn to ignore `c` and `d`. **Ablation: zero them at
  test time.** If the metrics barely move, the channels are inert — and that is precisely
  when a dense `d` target earns its place as a follow-up.
- **Sub-token validation is thin.** Of the 21 pairs inside one 160 m token, **15 are closer
  than one 20 m output cell** and cannot test sub-token skill at all. Only 6 remain
  (2 train, 4 oos, **0 val**). Those 15 do measure something useful — the irreducible
  representativeness floor, the disagreement between two probes in one output cell — but
  the honest claim chain is: `d`↔SM validated at points, `d` predicted at 20 m on held-out
  stations, SM pattern at 20 m **inferred, not verified**.
- **Inference needs an archive.** `c_k` requires ~20–50 prior passes, i.e. 8–20 months of
  S1 at any new tile. This is standard — the TU Wien change-detection method behind the
  Copernicus SSM product needs multi-year dry/wet references per pixel for the same reason
  — but it must be stated as a scope condition.
- **Staleness mismatch.** Training must randomise the age of `p*` and pass that age as a
  feature, or the model sees a distribution at inference it never trained on. Drop `p*`
  entirely on a fraction of samples so the S1-absent fallback is learned, following the
  existing SIF/TWSA convention (`dataset.py:1064-1072`).

### 33.11 Files

**Modified.** `model.py` — three extra input channels per decoder stage at `:257-264`,
zero-initialised; `up4` dropped; SM head at 112×112; `masked_huber_loss` (`:751`) takes a
per-sample cell index in place of the hardcoded `(112,112)` at `:779`; LST head on `up1`.
`train.py` — `_compute_loss` (`:376-400`) gains `L_lst`; per-epoch between-station-SD and
`r(pred level, obs level)` reporting in `evaluate` (`:642`). `dataset.py` — `__getitem__`
returns `c`, `d`, `w`, age and validity at four levels, plus the LST target when a clear
scene falls within ±1 day; add a second sample stream keyed on clear-scene dates that
carries `L_lst` only.

**New.** `build_s1_interaction.py` (offline, `Pool(64)`, one sbatch, ~130 GB read, ~13 GB
written, 30–60 min), `probe_s1_dvs_sm.py`, `download_landsat_st_global.py`,
`slurm/s1_interaction.sh`.

**Reused unchanged.** `station_mean_probe.py`, `eval_predict.py:112-189`
(`PixelMap`, `run_split_pixels`), `csvs/colocated_pairs.csv` (§32.7 step 7, discharged),
the frozen TerraMind tokens and the zarr loader.

### 33.12 Amendments (2026-08-25)

Design session on the S1 decomposition and the LST target. **Nothing built.** §33.9 gate 1 is
still unrun and is still the only thing that can kill the design; everything below is wiring for
a design that is not yet validated.

Full derivation, worked examples and the flowchart: **`text/s1processing.md`**.

**(a) Seven channels per stage, not three. β is removed from the build.**

§33.6's channel table and §33.5(d)'s `d* = d_VV − β·d_CR` are superseded. β is a coefficient
fitted per station × group × polarisation — roughly 993 × 4 offline numbers, the same class of
object the water-balance design was rejected for (§33.8). Instead both terms are fed and the
conv's own weights do the combining: one shared trained weight instead of thousands of fitted
ones, covered by the existing training story.

| channel | spatial? | source |
|---|---|---|
| `c_VV` | ℓ×ℓ | archive |
| `d_VV` | ℓ×ℓ | pass `p*` |
| `d_CR` | ℓ×ℓ | `d_VH − d_VV`, free |
| `w_VV` | scalar, broadcast | pass `p*` |
| `w_CR` | scalar, broadcast | pass `p*` |
| `age` | scalar, broadcast | days since `p*` |
| `valid` | scalar, broadcast | 0 ⇒ the four above are 0; `c_VV` stays |

Three maps and four broadcast scalars. §33.6's prose already named `age` and `valid` but its code
snippet and parameter count did not carry them; they are now explicitly wired at every stage.
Broadcasting a scalar costs nothing, and a conv that sees `d` and `age` in one tensor can
downweight stale `d` locally, which it cannot do from a global vector.

β survives as the **diagnostic that earned the design** — `r(d_VV, d_CR)` = −0.40…−0.58, the
9.8–22.3% annual-harmonic share, and the ACF rise 0.240→0.328 / 0.371→0.428 / 0.478→0.546 — not
as code.

`c_VH` and `μ_VH` are still computed and stored: they are needed to build `d_VH`, hence `d_CR`.
`c_VH` is **not** fed — it is a static land-cover map, redundant with LULC and `c_VV`, and `d_CR`
is already referenced by construction so `c_CR` adds nothing to interpreting it. Judgement call,
cheap to revisit.

Two identities, both verified numerically at all four levels:

- `d_CR = d_VH − d_VV` **exactly**, because double-centring is linear:
  `H_N (T_VH − T_VV) H_M = H_N T_VH H_M − H_N T_VV H_M`.
- `w_CR ≠ w_VH − w_VV`. Standardisation is affine with channel-specific constants, so difference
  the raw ρ first and standardise second.

Zero-init widens from `conv.block[0].weight[:, -3:] = 0` to `[:, -7:]`. Getting that slice width
wrong silently breaks §26's provenance gate.

**(b) `w` is built from the full-tile linear mean, not the row mean of `T_ℓ`.**

`r^(ℓ)` averages logs of block means, so it drifts with ℓ (Jensen), yet §33.6 broadcasts a single
`w` to four stages. Keep `r^(ℓ)` inside the centring — it is what makes the variance split exact
— and define `w` from the level-independent

```
   rho_q(p) = 10*log10( mean of P_q(p) over the full 224 x 224 )
   w_VV     = (rho_VV - rho_bar_VV) / s_VV
   w_CR     = (rho_VH - rho_VV - rho_bar_CR) / s_CR
```

**(c) Correction: the §33.5 `season%` column is not a vegetation fraction.**

The three columns are a variance partition and sum to 100 — `Var(r)`, `Var(c)`, `Var(d)` as
shares of `Var(T)`. CA-Cbo 19.6+63.7+16.7, TxSON 37.7+50.1+12.2, Abrams 40.7+21.7+37.7. They say
how much of the tile variance the pass-level term carries, **not** how much of that term is
canopy. So vegetation contamination of `w` is plausible but **unmeasured**, unlike `d` where
16–34% is measured directly against `d_CR`. `w_CR` therefore rests on an argument, not a number.
State it that way.

**(d) The LST aggregation was ragged. Fixed by moving the head to `up3`.**

§33.7's `F.interpolate(28→22, mode='area')` is adaptive pooling with a non-integer ratio
(28/22 = 1.27). Measured: output cells span **2 or 3** source cells and **20 of 28** source cells
feed more than one output. It is an overlapping blur, not a partition, so "nothing is invented on
either side" does not hold and the 484 residuals are correlated — the LST term's effective weight
against the SM term is not what is set.

Replace with an exact partition:

```python
p = self.head_lst(up3_feat)                     # (B,1,112,112) @ 20 m
p = p[..., 1:111, 1:111]                        # 110x110 @ 20 m = 2200 m
p = F.avg_pool2d(p, kernel_size=5, stride=5)    # 22x22 @ EXACTLY 100.0 m, disjoint
```

`110 = 5 × 22`. Every output cell is the mean of 25 disjoint 20 m cells — a true 100.0 m box,
uniform support, no overlap. Cost is a 20 m ring at the tile edge (2240 → 2200 m), irrelevant
with the station at centre. Bonus: 100.0 m rather than 101.8 m, landing on TIRS nominal.

Target side must partition too. Landsat ST arrives at 30 m (TIRS acquires at 100 m; USGS
Collection 2 L2 delivers it resampled to 30 m — there is no 100 m download). Warp to the same
2200 m / 22×22 / 100 m grid with **`Resampling.average`** (exact area-weighted aggregation), not
`nearest` and not `bilinear`. Mask a cell whose valid contributing fraction falls below
threshold, before the double-centring by WRS-2 path/row.

Accepted trade-off: predicting at 20 m and pooling to 100 m leaves 20 m sub-structure
unsupervised by the LST loss, where `up1` left an 80→100 m gap. Taken anyway, because that
sub-structure is unsupervised regardless — only the centre SM pixel constrains it — and §33.10
already states the claim correctly. The alternative that keeps the head on `up1` is 14×14 @ 160 m
(pool 2, exact), but 160 m is the token scale and tests no sub-token skill at all.

**(e) LST loss wiring.**

```python
sm  = self.head_sm(up3_feat)                    # (B, n_depths, 112, 112)
lst = self.head_lst(up3_feat)                   # (B, 1, 112, 112)

L_sm  = huber(sm[:, :, 56, 56][m_sm], y_sm[m_sm])

p     = F.avg_pool2d(lst[..., 1:111, 1:111], 5, 5)
p     = p - p[m_lst].mean()                     # centre over VALID cells, per sample
L_lst = huber(p[m_lst], d_lst_norm[m_lst])

L = L_sm + lam * L_lst
```

- **Normalise `d_LST` by division only.** It is already double-centred, so its mean is exactly
  zero; subtracting again is a no-op at best and breaks §33.9 gate 4's identities at worst.
  Divide by **one global σ**, pooled over the **train split only**. Not per-station: dividing by
  each station's own σ amplifies retrieval noise at stations with weak thermal contrast.
- **Centre the prediction over the valid cells.** The target is zero-sum by construction, so the
  loss should be invariant to a per-sample offset. Cloud masking means the surviving target cells
  do not sum to exactly zero, hence centring both sides over the same subset.
- **Zero-init the LST head's final 1×1 conv.** With `W = 0` the gradient into the trunk is
  `Wᵀg = 0`, so the head calibrates before it starts steering the decoder.
- **Pick λ by matching gradient norms into `up3`, not loss values.** SM residuals are ~0.03 in
  m³/m³ so `L_sm ≈ 5e-4` against `L_lst ≈ 0.5` at unit variance — value-matching would give
  λ ≈ 1e-3 and silently disable the term. Measure `g_sm / g_lst` on a warm batch, then sweep ×3
  and ÷3, logging the ratio per epoch.
- **Guard `m_lst.sum() == 0`** and average over samples that have any valid cell. Landsat 8+9
  gives ~8-day revisit and roughly half is lost to cloud, so **~6% of samples carry an LST
  gradient**; without the guard the effective λ fluctuates with batch composition. Log the count
  per epoch.
- **112 is even, so there is no centre cell.** The station falls between 55 and 56 — a 10 m
  offset. The current loss inherits the same problem at `(112,112)` in a 224 grid. Choose one
  cell or average 55:57 deliberately.

The division of labour: **LST supervises the pattern** (dense, 100 m, zero-sum, relative);
**SM supervises the level** (one point, 20 m, absolute). They do not compete for the same degree
of freedom, which is why the dual objective is coherent rather than two losses bolted together.

**Ablation that decides whether it earned its place:** train λ = 0 versus λ > 0 and compare **SM**
skill. Pair with §33.10's existing test (zero the S1 channels at test time) so "the channels are
inert" can be told apart from "the auxiliary loss did not help".

**(f) New gate — `d_LST` must pass the same coherence test as `d`.**

§33.5(c) established that double-centring pure noise leaves 94–99% in the interaction term. That
applies to the **target** as much as the input. Landsat ST retrieval noise runs ~1–2 K, and after
removing the static map and the pass offset `d_LST` may not be much larger. If it is mostly noise
the decoder is being trained to fit 484 noise cells per sample — actively harmful, not merely
useless.

**Run the lag-1 ACF of `d_LST` between consecutive passes against a location-shuffled control,
before using it as a target.** Same protocol as §33.5(c), applied to the label side. Add as
§33.9 gate 6.

**(g) Reference and leakage discipline.**

1. **Global constants from the train split only.** `SIGMA_LST`, and audit
   `csvs/era5_stats.json` — if it was built over all 993 stations that contamination is already
   in the baseline.
2. **Per-station references stay per-station.** True by construction (splits are by station,
   references are per station, so nothing crosses the boundary) — assert it in code so it cannot
   drift.
3. **Freeze and version the references.** If a new pass is appended to the archive, `d` for a
   given date depends on when the job ran and reported numbers stop being reproducible. Recompute
   on a schedule, version the output.
4. **Log `n` per group.** Self-inclusion bias is 1/N and is worst exactly at the `n ≥ 20` floor,
   where `c_k` is already least trustworthy (RMSE 14.5% of the `c_k` spread at n = 20, 7.6% at
   n = 50, §33.9 gate 5).
5. **Chronological split-half of `c_k`, as a new diagnostic.** The random split-half already
   computed for the ±6 orbit merge gives the noise ceiling (0.998). Recompute it splitting the
   record at its **midpoint** instead. If `ρ_chrono ≈ ρ_random` the scene is temporally stable
   and the full-archive `c_k` is correct. If it falls materially below, that station's scene
   changed: **flag and exclude from the spatial claims, do not auto-detect change points** across
   993 × 4 groups.
6. **Balance the averaging set rather than shortening it.** Merged S1A/S1B groups are ~2× denser
   before Dec 2021, so `c_k` is implicitly weighted toward the pre-2021 scene. Weight to equalise
   per-year contribution — negligible variance cost, removes a known bias. Log the pre/post-2021
   pass ratio per merged group. Seasonal balance is already gated (§33.9 #5); this is the
   temporal counterpart. A trailing rolling window is **rejected**: it would absorb genuine
   multi-year drying into the reference and delete it from `d`.

**(h) OPEN, to resolve before any reported number — `c_k` and temporal look-ahead.**

`c_k` for an evaluation station is computed from that station's S1 archive, which spans years the
model never trained on. Raised at the end of the session and **not resolved**.

Where it stands:

- It is **not label leakage**. `c_k` is built from S1; the label is an in-situ SM probe. No path
  from label to reference. Splits are by station and references are per station, so nothing
  crosses the split.
- It **is** temporal look-ahead *within* a station: the input at time `t` depends on acquisitions
  after `t`. Arguments that it is mild — `c_k` estimates a time-invariant quantity, carries no
  information about the target date specifically, and the self-inclusion effect is 1/N — are
  arguments, not measurements.
- The **chronological split-half in (g)(5) settles it empirically**: if `c_k` is temporally
  stable, causal and full-archive `c_k` must agree, and the convention stops mattering. Run both
  and report the gap rather than arguing the case.
- Provisional position: full-archive `c_k` is defensible for a held-out-station evaluation
  provided the convention is stated; anything operational or forecast-shaped needs an
  expanding-window causal `c_k`. **Revisit before publishing any number.**

**(i) Housekeeping.**

- `nanmean` at the block step, with a valid-fraction floor per cell. A partly masked block biases
  `c_k` with no visible symptom.
- **Measure the magnitudes of `c` and `d` before deciding whether to standardise them** for
  concatenation. §33.5 reports variance *shares*, not absolute dB. If they are comparable, skip
  the standardisation; if not, standardise per station-group or the zero-init is undone by
  whatever the conv learns first.
- Nothing to change on the download side. MPC `sentinel-1-rtc` is derived from IW GRDH and is
  already multilooked 5×1 (ENL ≈ 4.4, ~20 m resolution on a 10 m grid). At TerraMind's 16×16
  patch that is ENL ≈ 280, residual speckle ≈ 0.26 dB. dB-on-disk costs ≈ −0.5 dB of bias which
  is static per station-orbit and therefore absorbed. **Do not re-download, do not pre-average.**
  §33.5(a)'s rule stands: never average dB.

**Supersedes in §33.11:** "three extra input channels per decoder stage" → **seven**; "LST head
on `up1`" → **`up3`**.

### 33.13 `MAX_AGE` — measured, not chosen (SPECIFIED 2026-08-25, unrun)

`valid` is defined entirely by one constant that no document gives a value to. `s1processing.md`
§9.9: *"if no pass in the group falls within the maximum age, set `d = w = 0`, `valid = 0`"* — and
`MAX_AGE` appears nowhere else. It is not a free parameter to be picked by taste: it decides when
feeding `d` is worse than feeding zero, and it sets the age distribution that §33.10's staleness
randomisation has to span. Measure it, on the CPU, as **a second axis on gate 1** — the same
regression, the same samples, one extra loop.

**Do not use naturally-varying lag.** Binning samples by whatever `Δ = t − date(p*)` happens to be
and comparing skill across bins is confounded. Long `Δ` arises when passes are missing, and
missingness is not random in time: the merged groups run ~6-day revisit before Dec 2021 and ~12-day
after (§33.12(g)(6)), so `Δ` correlates with era, and era correlates with scene state. The estimated
decay would be part staleness and part 2021.

**Construct the lag instead.** A label date does not have one prior pass, it has a ladder of them
at the group's revisit spacing:

```
for each (station, label date t) with >= K prior passes in group g,  K = 4:
    for k = 1..K:
        d_k   = d built from the k-th most recent pass at or before t
        age_k = t - date(pass_k)
    regress SM_anomaly(t) ~ d_k at the station's own cell
        station fixed effects + persistence control, as gate 1
    -> partial skill as a function of age, on IDENTICAL samples
```

Every sample contributes at every `k`, so stations, seasons, land cover and revisit era are held
constant across the curve by construction. What varies is age and nothing else.

**Decision rule.** `MAX_AGE` = the age at which the partial contribution of `d_k` over the
persistence control stops being distinguishable from zero. Fit on **train stations only** and freeze
before val/test is touched — sweeping the cutoff and then reporting the skill it maximises is
selection on the evaluation set.

**Prior bound already in hand.** §33.12 measured `d`'s lag-1 ACF across consecutive passes in the
group at **0.240 / 0.371 / 0.478** (0.328 / 0.428 / 0.546 after the vegetation correction). If `d`
retains only a quarter to a half of itself across one revisit, a two-revisit-old `d` carries little.
That points at one to two revisits, not a month. It is a bound, not the answer: the ACF measures how
fast `d` changes, the lag curve measures how fast its *relevance to SM* decays, and those need not
coincide.

**Two readings, and the second is the useful one.**

- **Decays fast** → short cutoff, as the ACF suggests. Note that because `age` is fed as a channel
  (§33.12(a)) the conv can discount stale `d` on its own, so a smooth long-tailed decay argues for a
  *generous* cutoff plus the `age` channel rather than a tight threshold.
- **Does not decay at all** → this is a **failure signal, not a licence for a long cutoff**. If
  skill at `k = 1` is indistinguishable from `k = 4`, `d` is acting as a quasi-static map — residual
  texture that survived the double-centring — rather than a dynamic anomaly. That is the §29 LST
  failure mode exactly: a real spatial pattern that does not track soil moisture (+0.967
  month-to-month, pooled +0.167 by Simpson's paradox, within-station −0.077). Gate 1 must treat a
  flat lag curve as disqualifying.

**What the result also settles.** The invalid-case value of `age` (§33.12's table leaves it
undefined), the training age-randomisation range in §33.10, and whether the tile-level
valid-fraction floor on `p*` needs to be strict — a pass that is mostly masked and a pass that is
stale are the same kind of degradation and should be scored on the same curve.

**Where it lives:** `probe_s1_dvs_sm.py`, alongside gate 1. No new job, no GPU.

---

## §34 Patchwise temporal transformer — give every patch its own history (SPECIFIED 2026-08-25, nothing built)

§33 attacks the vacuum *below* 160 m by feeding measured S1 into the decoder. §34 attacks a
different and larger defect: the model has almost no per-patch information *at* 160 m either,
because the satellite history is spatially pooled before the transformer ever sees it.

Built in three steps, each with its own gate. **Step 1 has no decoder at all.**

### 34.1 The defect, measured

`_cpu_pyramid_pool` runs in the dataset worker and collapses the patch axis:

```
S2:   L12 (60, 196, 768)  ->  (60, 4, 768)
S1:   L12 (40, 196, 768)  ->  (40, 4, 768)
DEM:      (196, 768)      ->  (4, 768)
LULC:     (196, 768)      ->  (4, 768)
```

§27a.2 measured what survives: **1.5% of within-tile variance for DEM, 2.6% for LULC, 2-3% for
satellite history.** That is not compression, it is destruction. §28.1 already lists it as the first
of two independent defects, with the fix stated as "feed their 196-token grids".

The consequence is stark. After pooling, the **only** per-patch information anywhere in the model is
the anchor - `anchor_l12`, a single acquisition (`model.py:497`). A patch knows what it looked like
on one date, and knows the *tile's* history. It does not know its own.

The cost is already being paid: the worker loads the full `(60, 196, 768)` tensor and *then* pools
it away. Un-pooling adds no disk I/O.

### 34.2 Why pooling was there, and what removes the need for it

Sequence length. 196 patches x 100 acquisitions = **19,600 tokens** in one global sequence, against
~990 today. Attention is quadratic, so ~380x. Infeasible, and pooling was the obvious escape.

**Running the temporal transformer patchwise removes the constraint entirely.** Each patch is an
independent sequence over time; the patch axis moves into the batch dimension:

```
(B, 196, T, 768)  ->  (B*196, T, 768)  ->  attention over T only  ->  (B, 196, 768)
```

Cost is `196 x 100^2 ~= 2.0 M` attention pairs, and the global stage *shrinks* (196 summaries
replace 196 anchor tokens plus 400 pooled history tokens, ~990 -> ~580 ~= 0.35 M). Total ~= **2.4x**
the current attention, not 380x.

This is the factorisation Contextformer uses (Benson et al., CVPR 2024), and its justification
transfers verbatim: *"for ecosystem processes, spatial context is crucial but does not change
dynamically. Therefore, separating spatial and temporal processing enhances efficiency."* They
report a 16x memory reduction from the same move; their temporal encoder follows Presto's.

Two differences worth recording. Contextformer runs its **vision backbone first**, then the temporal
stage - for us that backbone is TerraMind, frozen and precomputed per acquisition, so the spatial
step is already done and we begin where their temporal stage begins. And they keep a **4 x 4 px
local context** inside the temporal encoder purely to absorb Sentinel-2 sub-pixel geolocation drift.
Our patches are 16 x 16 px = 160 m, far coarser than that error, so per-token is defensible - but
check `csvs/register_across_modalities.csv` before assuming. If registration is loose, the fix is
theirs: a 3 x 3 token neighbourhood in the temporal stage.

### 34.3 The patchwise temporal transformer

Per patch `k`, its own sequence and nothing else:

```
[ static_k | L12 date_1 | L12 date_2 | ... | L12 date_100 ]
  ^ W_dem . dem_tok[k] + W_lulc . lulc_tok[k]
                each + rel_pos_emb (staleness) + hist_modality_emb (S2 / S1asc / S1desc)

N layers of self-attention over T only, weights SHARED across patches, pad + cloud masked
-> learned CLS per patch -> (B, 196, 768)
```

- **No patch sees any other patch.** Spatial context came from TerraMind, within each acquisition.
- **Statics go in as a prefix, not appended to the summary.** As a prefix the temporal attention can
  condition drydown on land cover and terrain - *"this patch dries fast because it is sandy"*. Added
  after the summary, that interaction is unexpressible. Soil stays tile-level: SoilGrids is 250 m
  native (~9 x 9 real cells per tile) and is itself predicted from covariates including DEM and land
  cover, so per-patch soil is largely redundant with what the prefix already carries.
- **The masks already exist.** `token_mask` (`model.py:465`) tracks per-patch validity per
  acquisition, `rel_pos_emb` (`:584`) gives staleness, `hist_modality_emb` (`:588`) separates S2
  from S1.
- **Residual formulation, so the baseline is preserved:**

```python
spatial_tokens = anchor_l12 + delta      # delta = stage-1 CLS, output projection zero-init
```

With the projection zero-initialised, **step 0 is bit-identical to the current model** and §26's
provenance gate holds unchanged. Zeroing `delta` at test time is a clean ablation of exactly this
stage. Same trick as Contextformer's "predict deviations from the last cloud-free observation" - our
anchor *is* that observation.

Output is `(B, 196, 768)`, the same shape `spatial_tokens` has today (`model.py:573`), so
`spatial_start` stays 12 and nothing downstream changes signature.

### 34.4 STEP 1 - token head at 160 m, no decoder. This is the gate.

Attach §28's shared per-token head directly to the 196 outputs and supervise at the station's token.
Output: a **14 x 14 map at 160 m**. `2240 / 14 = 160`. One SM value per patch, 196 per tile.

**This is a regression, and saying so is the point.** One shared function maps
*(patch history, patch statics) -> SM*, applied independently at all 196 positions. Weight sharing
is the entire mechanism: supervising one token teaches the mapping at every token, which removes
§28's 50,175-unconstrained-pixels problem. No upsampling, nothing invented, output resolution equals
input resolution.

It is also the honest resolution. §23 measured **64-83% of the current 224^2 map's variance already
sitting on the 14 x 14 grid**, with the verdict *"do not present this model as producing 10 m
soil-moisture maps"*.

**And it is validatable.** §32.10.2's colocated stations are 405-936 m apart - **2.5 to 6 cells** -
so a 160 m map resolves them. Contrast §33.10: 15 of 21 sub-token pairs are closer than one 20 m
output cell, leaving 6 usable and **zero in val**. Step 1 predicts at exactly the scale the evidence
can check.

**What it answers:** does per-patch history carry within-tile SM information at all? Every later
step assumes it does. §27b.8 measured own-token within-network skill at **0-4.2%** with the *pooled*
history - this is the first test of whether un-pooling changes that.

**Numbers to beat** (§28.8, TxSON, same checkpointed comparison as §26.11):

| metric | current | target |
|---|---|---|
| own-centre ubRMSE (0-10) | 0.0301 | <= |
| off-centre ubRMSE | 0.0345 | materially closer to own-centre |
| between-station spread as % of observed | 15-19% | **> 35%** |
| r(pred level, obs level), CR200-18 | -0.175 | > 0 |

**Position leakage, unchanged from §28.5 and mandatory here.** Patches are station-centred, so the
supervised token is always index 105 and the model can learn *"read (7,7)"* rather than a
position-general mapping - exactly the off-centre degradation §26.11 measures (0.0110 -> 0.0500).
Needs translation augmentation in token space (a slice of an array already in memory, no
re-tokenisation) plus multi-station supervision; `csvs/txson_readouts.csv` already holds 96
readouts, 56 of them off-centre.

### 34.5 STEP 2 - add the decoder, 160 m -> 20 m

Only if step 1 passes **and** §33.9 gate 1 has passed.

```
(B, 196, 768) -> (B, 768, 14, 14) = bottleneck

x = cat[ bottle_proj(bottleneck), s12,  S1_7ch@14  ]    14x14   @ 160 m
x = up1 -> conv1( cat[x, s9^,           S1_7ch@28  ] )  28x28   @  80 m
x = up2 -> conv2( cat[x, s6^,           S1_7ch@56  ] )  56x56   @  40 m
x = up3 -> conv3( cat[x, s3^,           S1_7ch@112 ] )  112x112 @  20 m
no up4

  ^ = bilinear upsample of a 14x14 skip
  7 ch = c_VV, d_VV, d_CR (maps) + w_VV, w_CR, age, valid (broadcast), §33.12(a)
  zero-init conv.block[0].weight[:, -7:]
```

`s12` is new - `anchor_l12` as a fourth skip, 1x1 proj + FiLM, zero-init. It gives the decoder the
raw L12 alongside `bottleneck`, which is L12 *after* the transformer: a residual path around the
transformer, cheap.

`up4` is dropped. Without raw S2 10 m bands (user decision, `open_items.md` §E) there is no measured
time-varying input at 10 m - WorldCover is static and S1 at l=224 is one look per cell (ENL ~= 4.4,
~2.7 dB speckle). That is §33.6's original reason for dropping it, re-derived from the other
direction.

**The division of labour is now clean, and it makes the ablation sharp.** Above 160 m everything is
settled by the patchwise stage - each cell knows its own history, terrain and cover. Below 160 m the
skips give nothing, since all four TerraMind layers are 14 x 14 stretched. So the **only**
information available for the 160 m -> 20 m step is the seven S1 channels. Zero them and everything
below 160 m is invention, by construction.

**Honesty condition on step 2.** At 160 m the prediction is checkable; at 20 m it is not. The claim
chain stays exactly as §33.10 states it: `d` <-> SM validated at points, `d` predicted at 20 m on
held-out stations, **SM pattern at 20 m inferred, not verified**. Publishable, provided it is said
that way.

### 34.6 STEP 3 - thermal supervision

Per §33.7 as amended, and only after §33.9 gate 6 and the static-map sufficiency gate
(`open_items.md` C1-C2):

```
head_sm    -> (B, n_depths, 112, 112) @ 20 m     the product
head_therm -> (B, 2, 112, 112) day / night       auxiliary, deleted at inference
                avg_pool2d(7,7) -> 16x16 @ EXACTLY 140 m    112 = 7 x 16, no crop
                ECOSTRESS 70 m -> 2x2 average -> 140 m      exact both sides

L = L_sm + lam_day.L_day + lam_night.L_night + lam_dtr.L_dtr
    lam by matching GRADIENT norms into up3, not loss values
```

Two channels rather than one DTR channel, so unpaired passes still supervise
(`open_items.md` D7). ISMN supervises the **level** at one pixel, absolute; thermal supervises the
**pattern**, dense and zero-sum. They do not compete for the same degree of freedom. At inference
only `head_sm` runs - no Landsat, no ECOSTRESS needed.

### 34.7 Cost

| | now | §34 |
|---|---|---|
| attention pairs | ~0.96 M | ~2.3 M (~2.4x) |
| global sequence | ~990 | ~580 |
| disk I/O | - | **unchanged** - the worker already reads `(60,196,768)` and pools it away |
| host->device | - | grows on the history tensors |
| decoder params | - | +12 k per stage from the 7 channels (step 2 only) |

Peak memory must be measured on a smoke run before any allocation: `B x 196 x 100 x 768` fp16 is
~30 MB per sample for the input alone, and at batch 8 the attention sees 1,568 sequences of length
100.

### 34.8 Open decisions before code

1. **Where ERA5 sits.** As drawn it stays in the global sequence, shared by all patches. Presto -
   and therefore Contextformer's temporal encoder - puts the meteorological series *inside* the
   per-patch temporal stage. That is the difference between every patch getting the same weather and
   every patch integrating the same weather differently, which is the *response x wetness* mechanism
   §33.6 currently hopes a conv approximates from a broadcast `w`. Strong argument for moving it;
   costs sequence length in the per-patch stage.
2. **Station cell at 112 x 112** (step 2 only). Even grid, no centre - the station falls between 55
   and 56. Choose one or average deliberately (`open_items.md` A3).
3. **`MAX_AGE` and the invalid-`age` value** (`open_items.md` B2-B4), from §33.13.
4. **Per-patch registration** - check `csvs/register_across_modalities.csv` before assuming
   per-token is safe; 3 x 3 neighbourhood if not.
5. **Crop-train / full-infer** for the decoder, available as a pure optimisation once step 2 exists:
   train on a token neighbourhood around the station, infer on the full tile. Provably identical
   weights, ~8-20x less decoder compute. Requires a margin for the bilinear skip interpolation, and
   an assertion that the cropped forward pass reproduces the full-tile one at the station pixel.

### 34.9 What §34 does not do

It recovers the full information available **at 160 m**. It does not create sub-token information:
every TerraMind layer is 14 x 14, so `skip_L3` at `up3` is a 16x interpolation carrying nothing it
did not carry at 160 m. §33's S1 decomposition remains the only route below 160 m, and §33.9 gate 1
remains the only thing that can kill it.

---

## §35 Restore, then build the patchwise processor (PHASE 0 DONE + BUILD APPROVED 2026-08-26)

**STATUS: PLANNED 2026-08-26. Phase 0 in progress, nothing else built.** Session 32. Written after
a first draft was put through four adversarial critiques (architecture, science, implementation,
red-team). Two independent reviewers found the draft's founding premise misquoted, a third found
two guaranteed crashes, and the fourth found the training store is currently corrupt. This section
records the corrected plan and, as importantly, the corrections themselves — so they are not
re-derived.

### 35.0 The premise §34 was built on is wrong

§34 and the first draft of this section both open with: *"§27b.8 measured own-token within-network
skill at 0-4.2% **with the pooled history**; this is the first test of whether un-pooling moves
it."*

That is not what §27b.8 measured. `probe_token_sm_structure.py:63` sets
`CENTRE_TOK = (GRID//2)*GRID + GRID//2`, `:78` asserts `CENTRE_TOK == 105`, `:113` and `:286`
describe *"the station's own centre token (index 105)"*, and `training_runbook.md:5602-5610` states
the setup verbatim. **§27b.8 measured the UN-POOLED per-patch token — the exact object §34 proposes
to introduce — and got the null.** The pooled-pyramid contrast was designed as a secondary
comparison (§27b.3, "Secondary contrast, free") and is not in the results table; it was never run.

So spatial un-pooling at station-mean scale has already been tested. What remains genuinely
untested is narrower, and different:

1. **The temporal axis of that token.** §27b.8 collapsed ~100 acquisitions to a multi-year mean.
   Does the token's *variation over time* track SM *anomaly*?
2. **Between-patch differences.** Do differences between two patches track differences between two
   stations? §27's Arm A was designed for exactly this and never run.

Both are answerable on CPU with machinery that already exists. Neither needs a new architecture.
This is why §35 stages the work: **the cheap things that can invalidate the expensive thing run
first.**

### 35.1 PHASE 0 — the training store is corrupt (BLOCKING)

Verified 2026-08-26 on `ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr` (`dataset.py:39`):

```
.complete sentinels, whole tree ............ 0      (of 993)
soil/ array dir (ISMN_TWENTE_Hupsel) ....... EMPTY - no .zarray, no chunks
era5/values ................................ chunk 0.0 present, .zarray GONE
L3/L6/L9 .npy memmaps ...................... gone for 953 of 993 stations
station dirs still present ................. 842 (sm_only)
```

**The damage classifies into two very different kinds** (`verify_zarr_store.py`, 12-station sample
2026-08-26 — 215 vs 166 array-instances):

| class | count | arrays affected | repair |
|---|---|---|---|
| **META-ONLY** — `.zarray` deleted, chunks intact | 215 | `s2/{l3,l6,l9,l12}`, `s1_asc/{l3,l6,l9,l12}`, `s2/dates`, `s1_asc/dates`, `dem`, `lulc`, `dem_token_mask`, `lulc_token_mask`, `labels/*`, sometimes `era5/*` | ~400 bytes per array |
| **DATA LOSS** — chunks gone too | 166 | `soil`, `cm/*`, `era5/*` (most stations), `sif/*`, `twsa/*`, `s1_asc/token_mask`, `s2/token_mask` | restore from backup |

**The expensive part survived.** Every large TerraMind token array is META-ONLY — the ~730 GB of
L3/L6/L9/L12 chunks are on disk; only the tiny `.zarray` headers were purged. That is exactly what
an age-based purge produces: headers were written once in June and never re-read, while chunk files
had their atime refreshed by training and eval reads. What is genuinely lost is the small auxiliary
arrays.

Consequence for the restore: **the rsync merge transfers far less than 1.4 TB**, because `rsync -a`
skips files that already match. It restores the `.zarray` headers plus the small arrays and leaves
the token chunks untouched.

The scratch purge ate array *contents* while `.zmetadata` survived at each station root. So
`zarr.open_consolidated` still exposes `soil` as a valid array and returns `fill_value` — **all
zeros, no exception, no warning**. Sampled 40/40 stations: scratch `min=0.0 max=0.0`, backup
`min=8.0 max=150.0`. With zero `.complete` markers, `dataset.py:132-134` drops every station, so
the dataset is empty; `train.py:140-141` preloads 0 stations and touches `.done` anyway.
`slurm/train.sh:43` hardcodes `--use-memmap`, and only the 40 TxSON stations kept their memmaps —
their atime was refreshed by the August eval.

**The actual bug is the silence.** A `fill_value` read is indistinguishable from real data
downstream, and soil is §20.14's strongest tabular block. A run started today would train on zeroed
soil and report a plausible number.

**This was an age-based purge, not a quota eviction** — scratch usage is 8.8% of an 8 TiB quota.
Restoring resets the clock; it *will* recur. Restore-and-verify is a recurring operation.

**Backup status, verified.** `/gpfs/work3/0/prjs1968/zarr_tokens` = `/projects/prjs1968/zarr_tokens`
(**the same directory**, device 43 inode 1561365504 — the `restore_zarr.sh` `SRC` path is correct).
All 993 stations (842 + 48 + 103), **1.4 TB** measured (1.2 T + 86 G + 66 G), all 842 `sm_only`
`.complete` markers present. For the station checked array-by-array, chunk counts and mtimes match
the live store and `.zmetadata` is the same generation (Jun 12 10:04). **Caveat: the backup's newest
chunks are Jun 26-29**, so anything fixed on scratch after that (`fix_bad_s1_tokens.py`,
`test_resanitize_s1.py`, `find_bad_s1_acquisitions.py` all exist) would be silently rolled back.

**Three storage tiers — the memmaps are only a partial duplicate.** Exactly nine arrays,
`{s2, s1_asc, s1_desc} x {l3, l6, l9}`, **no L12**:

| tier | holds | why | rebuilt from |
|---|---|---|---|
| zarr on scratch | everything (all `l12`, soil, era5, labels, dem, lulc, cm, sif, twsa) | source of truth | the backup |
| `.npy` memmaps | L3/L6/L9 only | anchor reads ONE random date; zarr chunks `[32,196,768]` amplify 32x | zarr |
| `/dev/shm` preload | L12 only, ~145 GB | history reads many dates -> load wholesale once per job | zarr |

Tiers 2 and 3 are derived and rebuildable; the zarr is not. **Zarr is not redundant** — it is the
only copy of `soil` and `era5/values`. `convert_l369_to_npy.py:68` writes `mm[:] = arr[:]`, i.e. the
**full** `(N,196,768)`, not an anchor cache (Hupsel `s2_l3.npy` = 35.2 MB = 117x196x768x2) — so the
memmaps are the complete L3/L6/L9 history, which matters if §35.2's P1 picks L3/L6 over L12.

#### 35.1.1 Procedure — reversible at every step

**Governing rule (user instruction, 2026-08-26): no step may destroy the only copy of anything.**
Every deletion happens *after* its replacement is verified, never before.

1. **Protect the source.** `chmod -R a-w /gpfs/work3/0/prjs1968/zarr_tokens`. Verify a test write
   fails. Nothing else touches the backup for the rest of the procedure.
2. **Verify the BACKUP before touching scratch.** This ordering is the point — the first draft wiped
   scratch first, which would destroy the only other copy if the backup proved incomplete. Full
   per-array sweep over 993 stations: `.zarray` present, chunk count > 0, sampled read is not
   `fill_value`, `soil.min() > 0`, `era5/values` non-empty, `.npy` shapes match their sidecars.
   `Pool(64)`, sbatch. **If this fails anywhere, stop.**
3. **Check headroom.** 8 TiB quota, 0.71 TiB used, +1.4 TiB restore = ~2.1 TiB ~ 26%. Comfortable.
4. **Snapshot scratch state** (seconds, no data copied):
   `find $ZARR_ROOT -type f -printf '%P %s %T@\n' | sort > scratch_manifest_pre.txt`.
   `rsync -a` overwrites when size/mtime differ, so this is how we prove afterwards whether a
   post-June scratch file was replaced by the June backup.
5. **Merge-restore, no wipe, no rename.** `slurm/restore_zarr.sh` was read and is sound in the ways
   that matter: `rsync -a "$SRC/$rel/" "$DST/$rel/"` per station, 64-way via `xargs`, **no
   `--delete` anywhere**, both paths hardcoded rather than arguments, idempotent, `staging`
   partition, correct mail flags. A merge is **strictly safer than the wipe-and-restore first
   approved**, because nothing is ever deleted. Two defects to fix first:
   - **Verification is unsound** — it counts `.complete` markers, which come *from* the backup, so
     it reports `sm_only: 842` even if every array chunk failed. Replace with step 2's content check.
   - **It can fail silently** — `set -uo pipefail` without `-e`, no exit-code check on the individual
     `rsync`s inside `xargs`. Collect per-station exit codes.
6. **Verify the restored store**, same sweep as step 2, plus a manifest diff. Any file whose mtime
   moved backwards means a post-June version was overwritten — investigate before accepting.
7. **Nothing to delete.** The merge leaves old files in place.
8. **Add a store-integrity assertion to `dataset.__init__`** so a `fill_value` read can never again
   pass as data. This is the durable fix; the restore is only the immediate one.

**Backup protection decision:** `chmod` only, no second copy. The realistic threat is our own
restore command, which `chmod` blocks for zero cost. A 730 GB second copy would sit on the same
`wstor_work3` filesystem and add little. Nothing here is irreplaceable — soil is re-downloadable
from OpenLandMap, tokens recomputable from the intact `satellite_zarr` raw imagery (on `work3`, not
purge-exposed) — so worst case is weeks of recompute. Revisit SURF Data Archive (`/archive` exists,
`dmftar` installed, `/archive/pkhanal` would need requesting) nearer the 2027-02-16 expiry.

**Until Phase 0 passes, every number produced since the purge is suspect.**

### 35.2 PHASE 1 — CPU probes (SUPERSEDED by §35.7: NOT a gate; the build was approved. Retained for reference; two items kept as build inputs)

All reuse existing machinery. `Pool(64)`, `--cpus-per-task=64`, sbatch, mail flags per project rule.

**P0. The ceiling — `probe_variogram.py`.** §27.2 pre-registered this as *"this runs first; a null
reported without it is uninterpretable"* and it was never run. First-cut estimate from
`csvs/gate_pair_deltas.csv`:

```
E[dSM^2] pairs  <160 m (n=15) = 0.002340 -> sd_b + micro = 0.0342
E[dSM^2] pairs >=160 m (n=34) = 0.007701 -> sd_total     = 0.0621
=> 30.4% of the >=160 m between-station variance sits BELOW one token cell
=> max achievable between-station spread ratio ~ sqrt(0.696) = 0.83
```

Do it properly: bin all 75 pairs of `csvs/colocated_pairs.csv` by separation, bootstrap the
variogram, report `R2_max(160 m)` with a CI. Every gate number then becomes a fraction of what is
physically achievable. Labels only, minutes.

**P1. The dynamic pair-difference probe — the decisive test, never run.** `de_t = e_i(t) - e_j(t)`
versus `dSM_t` over colocated pairs separated by **more than** 160 m, pair mean removed. Both tokens
come from one forward pass on one image, so climate, ERA5 cell, season, biome, network, sensor
vendor and calibration epoch are annihilated by construction — the confounds that killed §29 and
§32 cannot occur. The *level* version has n ~ 33 and is hopeless; the *dynamic* version has ~33-54
pairs x ~1500 days. Control: the same statistic from the pooled tile mean, which must be ~0 by
construction. Statistic: distance correlation with an exact permutation null (§27.3).
**Sweep all 14 modality x layer combinations**, which settles P3 as a side effect.

**P2. Within-station daily anomaly — the temporal question §27b.8 did not ask.** Regress SM anomaly
on the station's own token 105 *per acquisition date*, station fixed effects, GroupKFold on
`location_group_id`, with a persistence control.

**P3. Which layer — decided by data, not inherited.** §27a.7: *"L3/L6 are cleaner on both counts —
more within-tile spatial structure and far less register dominance. **Anything aiming at fine
resolution should be built from them, not from L12.**"* S2 L12 register magnitude share is
**0.940**, the worst of any modality x layer; L3/L6 are 0.586/0.603. The first draft built the whole
history path on L12 without argument, conflating "drop the four-layer *anchor*" (an I/O argument,
legitimate) with "the *history* must be L12" (never argued). P1's sweep decides it.

**P4. Position-code probe.** **TerraMind bakes 2-D sin-cos positional embeddings into every token** —
`terratorch/models/backbones/terramind/model/encoder_embeddings.py:151`, `tm_utils.py:51
build_2d_sincos_posemb`, `sincos_pos_emb: bool = True` by default, propagating through all 12 blocks.
So the claim that dropping `spatial_row_emb`/`spatial_col_emb` makes §28.5's position leakage
*unrepresentable* is **false**: it removes the explicit channel and leaves the implicit one frozen
into the features, where — unlike an `nn.Embedding` — it cannot be ablated at all. Worse than the
status quo. Probe: linear regression of (row, col) from L12 tokens, ~50 tiles x 196 tokens, held-out
tiles, one hour. §34.4's translation augmentation must be reinstated regardless, since it was deleted
on the strength of the false claim.

**P5. The additive baseline — the competitor a referee will demand.**
`SM(patch k, t) = f(ERA5, soil, tile drivers)_t + g(dem_k, lulc_k)`. No transformer, no per-patch
history, no GPU, CPU-minutes. Motivated by measurement, not parsimony: from
`csvs/gate_pair_deltas.csv`, for pairs >=160 m apart the *sign* of the between-station SM difference
holds on **98.6% of days** (§32.11 measured 95.8%). The target is very nearly a static field. If the
additive baseline matches, the transformer is unjustified.

**P6. Two cheap audits that change Phase 2's design.** (a) Covariate shift over all 993 tiles: LULC
histogram and DEM stats of token 105 vs all 196; report the fraction of patches whose LULC class
never occurs at any station's token 105. If ~30% are out of support, the 14x14 map needs a **mask,
not a caveat**. Note §G7's trigger reads this in TxSON, 72% rangeland — where patch diversity is
smallest and shift least detectable. (b) Within-tile variance of the 21 OpenLandMap soil channels,
which decides the soil treatment in §35.3.

#### 35.2.1 Exit criteria — pre-registered, written before P1 is submitted

- **P1 fires** (significant against the permutation null on some layer) -> Phase 2 justified, and
  P1 names the layer.
- **P1 null, P2 fires** -> the signal is temporal-only at tile scale; the within-tile ambition is
  not supported, and the honest next step is a better tile-level temporal model.
- **P1 and P2 both null** -> per-patch tokens carry neither between-patch nor within-station SM
  information. **Do not build Phase 2.** Redirect to §33, which uses a *measured* radar quantity
  rather than a frozen embedding.
- **P5 matches whatever Phase 2 could achieve** -> build nothing; report the additive model.

### 35.3 Corrections to §34 / open_items §G, recorded so they are not re-derived

**The sink audit was already done.** `csvs/static_token_outliers.csv` (993 rows) already carries
`dem_argmax_r/c/i` and `lulc_argmax_r/c/i` per station, `argmax_i == r*14+c` on 100% of rows. Token
105 is the sink in **0/993 (DEM)** and **1/993 (LULC)**; sinks cluster in rows 11-12 at the bottom
edge. That is the "proceed" branch. The claim that the file held only summary statistics was wrong.

**But the register problem was scoped wrongly.** Mitigating only `dem_k`/`lulc_k` covers **2 of 141
tokens** while `s2/l12` has register magnitude share **0.940** and **102 of 141 tokens are L12**.
LayerNorm is per-token and does not remove a shared direction across tokens — §27a.5 measured
post-LayerNorm median pairwise cosine **0.783 -> 0.157** when two of 768 dims are zeroed. For a
*temporal* transformer this is acute: if all 100 history keys point nearly the same way, `q.k` is
near-constant and **attention over the history collapses to near-uniform** — the model gets a mean,
which is what un-pooling was supposed to escape. §27a.7 already prescribes the fix and it was
silently dropped: **per-dimension standardisation over the dataset, train-split only**, *"It
discards nothing"*, *"Zero training-time cost"*. Adopt it; compute the stats in Phase 1.

**Rejected fixes, do not revisit.** *Inpainting the sink patch and re-running TerraMind*: the input
there is pristine (§27a.3 — DEM 452.88-454.55 m, 254 distinct values, zero NaN; LULC uniform
rangeland); the -1671 coordinate is manufactured by the ViT, which needs somewhere to park global
information, so replacing the patch relocates the sink rather than removing it — at the cost of a
full GPU re-tokenisation and destroying good data. *Changing input normalisation*: already correct
(`precompute_terramind.py:72-75`, `:124-140` z-score with `v1_pretraining_{mean,std}`; LULC v1_5
stats are identity), and massive activations occur in correctly-normalised in-distribution ViTs
regardless.

**Cost arithmetic was wrong in both directions.** §34.7, §G2 and the first draft counted *attention
pairs*, ignoring that FFN/projection cost scales with `K.T`. Per-layer per-sample FLOPs at d=768 are
`12.T.d^2 + 2.T^2.d`, and the linear term dominates:

| | T | total/layer | vs baseline |
|---|---|---|---|
| baseline | 1038 | 9.00e9 | - |
| patchwise K=1 | 141 | 1.03e9 | **8.8x cheaper** (not 54x) |
| patchwise K=196 | - | 2.02e11 | **22x more expensive** (not 3.6x) |

Attention is 3% of the patchwise cost. And **epochs are data-bound** (data 250-630 s vs compute
483 s), so an 8.8x compute cut applies to under half the epoch — realistic ceiling **1.6-2x**. The
145 GB `/dev/shm` preload is unchanged at **17-32 min per job start** (project total: 44 starts,
8.5 h wall ~ 34 GPU-h).

**Soil keeps the exact defect §35 exists to remove.** It is **OpenLandMap, not SoilGrids**
(`download_soil_openlandmap.py:62-64`, `PATCH_PX=74`, `RES_M=30` -> `(21,74,74)`), and `SoilEncoder`
(`model.py:328-331`) collapses it to four **concentric centre-symmetric means** — after the tile mean
is removed only radially-symmetric structure survives, the identical criticism §27a.2 levels at
pyramid pooling. Per-patch soil at 14x14 is **21 x 196 x 4 B ~ 16 KB against the 460 KB currently
kept — 29x smaller**. The "redundant with DEM/LULC" argument was a non-sequitur: SoilGrids being
*predicted from* DEM does not mean the model can recover it from TerraMind's frozen,
register-dominated *encoding* of DEM.

**Other corrections.** `h_k` was undefined — the 141-token sequence has no readout; add an explicit
per-patch CLS, and note the depth CLS changes meaning (3 x 196 per sample, so §19.3's
`_last_depth_ctx` diagnostic no longer applies). Multi-station supervision is **not** free —
`dataset.py:1076` has one label vector per sample and `station_key` is a pure function of station
identity (`:818-824`), so there is no mechanism to attach two labels to one forward pass;
`location_group_id` exists in `station_splits.csv` (906 groups) but `dataset.py` never reads it.
The anchor-drop membership test is insufficient: the anchor is selected as *most recent with all 196
patches valid* (`:418-421`, `:488-492`) — a tile-level criterion the per-token mask does not encode —
its pool spans S2/S1-asc/S1-desc (`:448`) and is not truncated to `MAX_S2`/`MAX_S1` while the history
is, and **68.2% of station-years exceed `MAX_S1 = 40`**. The "three separate resamplers for clean
ablation" justification is false: masking a modality's inputs gives exact ablation in a shared
resampler and is already the code path (`:1064-1065`, `:1071-1072`); separation instead discards what
a cross-modal interaction would have used. **`best.pt` IS resumable** — byte-identical to `last.pt`
(both 604,402,070 B, epoch 16); §G6's claim was wrong. **The rollback tag is on the wrong commit** —
`baseline-unet-temporal` = `a46efaa`, a docs-only commit; the code that trained `cls_depth_star_reg`
is `0313f25`. Retag, and write the commit SHA into every checkpoint.

### 35.4 PHASE 2 — build (SUPERSEDED by §35.7-§35.13, which are the approved build plan. Retained for the reasoning behind staging)

**Do not ship eleven changes in one run.** The first draft bundled: patchwise encoder, drop anchor,
drop L3/L6/L9, independent depth heads, drop ERA5 `skt`, drop spatial embeddings, three resamplers,
new loss readout, new token masks, `--arch` flag, `hist_modality_emb` widening. If the gate fails,
nothing is attributable.

The sharpest staging argument: **at K=1 the driver resampler saves nothing at training time** — raw
drivers give 536 tokens, still 3.7x cheaper than today's 1035. It is an inference-cost optimisation
for K=196, and the gate's inference is TxSON only. So the most bug-prone new component (learned null
token, `-1e4` masking, the ~25% both-blank path with no precedent in this codebase) would enter the
highest-stakes run for no benefit that run needs.

| stage | contains |
|---|---|
| **2a** | patchwise encoder, raw drivers, layer from P1, register standardisation, per-depth heads |
| **2b** | driver resamplers (inference optimisation only) |
| **2c** | ERA5 `skt` removal, `hist_modality_emb`->3, translation augmentation |

**Required arms — one run answers nothing.** Same code, same seed, one flag apart: *full*;
*history-ablated* (zero the 100 history tokens — does per-patch history contribute at all?);
*re-pooled* (history replaced by its tile mean — isolates un-pooling, the actual hypothesis);
*statics-only* (zero `dem_k`/`lulc_k` — §27b.8 says `dem/l12` is the strongest within-network token
signal, eta^2_w = 0.075, p = 0.0005); and a **patch-shuffle negative control** (permute per-patch
history across the 196 patches at eval — spread and off-centre skill must collapse; §29.7 mandated a
shuffle control and the first draft had none). `ablation.py`'s `MODALITY_KEYS` (`:28-38`) is the
instrument.

**Depth heads: independent, not §18.4's star residual** (user decision 2026-08-26). Measured label
mass over a 120-station sample: 0-10 at 100% of stations / 41.0% of station-days, 10-30 at 75.5% /
32.6%, 30-100 at 60.8% / 26.4%; combinations all-three 68, `0-10+10-30` 25, `0-10` only 22,
`0-10+30-100` 5. The star residual was an inductive bias for sample efficiency, not a data
necessity. Caveat to carry: depth coverage is **not missing at random** — sensor configuration is a
network property confounded with climate, so the deep heads train on a systematically different
station population. Compare the depth-1/depth-2 subsets against the full set on Koppen,
`igbp_macro`, `elevation_band` before trusting per-depth attribution.

#### 35.4.1 The gate must be rebuilt — the §28.8 version is not usable

- **">35% spread" misquotes its source.** `plot_network_timeseries.py:156-158`: **0.6** is *"the map
  resolves the tile"*; **0.35** is only the floor below which it *"repeats ~one series"*. Passing
  bought the verdict *"partly resolves"*. Misquoted in four places in this runbook.
- **`r > 0` on CR200-18 is a coin flip.** n = 6; 95% CI on the current -0.175 is [-0.864, +0.742].
  Across §26.11's four densest tiles the current model gives -0.135, +0.012, -0.589, -0.077 —
  **CR200-3 already passes** — and P(>=1 of 4 passing by chance) = 0.94. The tile was chosen post hoc.
- **The reference numbers are not an artefact.** §28.8 records no run, checkpoint, epoch, SHA or
  W&B id; its `r = -0.175` contradicts §26.11's own table (-0.135); the adjudicating parquet is
  gitignored; spread appears as 15-19%, 17%, 18.8% and 20% in four places. Regenerate from a named
  checkpoint or drop the comparison.
- **The baseline mixes memorised training stations with held-out ones.** §26.11 split-stratified:
  train 0.0110 / val 0.0299 / oos 0.0386. TxSON's 40 stations are 14/8/18, so the pooled 0.0301 is
  not a generalisation metric; the honest held-out bar is ~0.036.

**Replacement, pre-registered before any run.** **Primary endpoint: a within-station criterion,
which the old gate lacked entirely.** At off-centre readouts, de-mean prediction and observation *by
station*, then score the residuals — does the predicted map's *anomaly pattern* move with time in
step with observed station-to-station differences on a given day? Without it, passing is compatible
with having learned a static landscape map, which §27a/§27b say is exactly what these tokens encode
and which the 98.6% sign-stability measurement says would suffice. **That is the §29 failure
repeated**; §29.15's own verdict was *"a static field cannot track a dynamic variable"*.

Secondary: the four §28.8 metrics, each split-stratified, each with a CI bootstrapped over
**stations** not samples, each expressed as a fraction of P0's ceiling. Spread and correlation gated
**jointly** — spread alone is unbounded above and rises whenever the model emits per-patch variation,
correct or not. Fix the tile set in advance (all TxSON tiles with >=4 readouts, not one). Give
"materially closer off-centre" a number and a paired test. State a **collapse criterion**, not just a
log, and add **temporal** attention entropy over the history block — the register-driven collapse
shows up there, not in the map SD. One primary endpoint, secondaries under Benjamini-Hochberg (as
§27b.6 did correctly); *the gate is read on run 1 with default flags, every later run is exploratory
and labelled so.*

**Delete the "~2M supervised samples is ample" argument** (§G7 reason 2). §27b.3 already rejected it
as *"pseudo-replication — it multiplies rows without adding information about a per-station
target"*, and §29.14 measured the damage. Effective n for the spatial question is 993 stations, and
generously: 33 networks, top-5 = 74.8%, SNOTEL alone 38.8%; drydown tau ~ 3.2 d, so consecutive days
are ~1/3 of an independent observation.

**No full run has ever converged.** Thirteen 4xGPU runs: five OOM-killed, three cancelled by human
judgement, the rest superseded — ~412 GPU-h on abandoned runs, 15 jobs with `oom_kill` events.
Best-val epochs were 3, 3, 3, and **16 of 16 still improving**; the recorded note is *"I called this
run converged at e5 and again at e9. Wrong both times."* **Set the stopping rule before submitting.**
And `val_loss` is not comparable across `per_depth_loss` settings — changing the depth-head structure
invalidates every val-loss comparison to a prior run unless recomputed.

**Budget is not the constraint**: 79,592 of 800,000 GPU SBU used in six months, expiring 2027-02-16,
~47 full runs in hand. The scarce resource is **calendar time** — which argues for Phase 1, not
against it.

#### 35.4.2 Implementation defects, verified against source

Two guaranteed crashes: **(1) DDP** — `train.py:902` uses `find_unused_parameters=False`, so leaving
the unet path constructed-but-unused raises `RuntimeError` on step 1 of every DDP run; gate
*construction* on `arch` and smoke on **2 GPUs**, since it is invisible on one. **(2) `token_mask` is
`(T,14,14)`, not `(T,196)`** (`dataset.py:244`, `:328`), so `token_mask[:, sel]` indexes the row axis
and silently returns `(T,K,14)`.

Silent-wrongness: padded and NaN slots are marked **valid** (`token_mask` inits to `ones`, median S2
acquisitions/station-year is 36 against `MAX_S2 = 60`, NaN slots `continue` at `:289-290`, no-cloud-mask
dates have no `else` at `:295`) — the mask must be `token_mask.reshape(T,196)[:, sel] & (doys>0)[:,None]`;
two zero-fallback branches (`:248-249`, `:371-372`) break collation for stations with no S2/S1 in
window; "everything after `model.py:779` is shape-agnostic" is false (`label` is `(B,n_depths)`, and
in readouts mode the K tokens are different stations with different labels); `lambda_boundary = 0.1`
silently changes meaning and `total_variation_loss` raises `IndexError` under patchwise; the §G3
resampler sketch does not run (six defects, incl. `circular_doy_pe` needing 1-D input and an
`nn.ModuleList` called as a function); `FiLMLayer` is 4-D only (`model.py:162-168`);
`--use-cls-depth` is not default-on (`train.py:207`); `slurm/train.sh:43` hardcodes `--use-memmap`;
`--arch` must land in `CONFIG` or every eval script silently rebuilds the unet (note `demo_plot.py:44-48`
and `plot_satellite_sm_meeting.py:47-50` carry duplicate `load_checkpoint` implementations bypassing
`ckpt_utils`); `token_sel=all` reinstates the ~437 GB DataLoader IPC OOM that pooling was added to fix
(`_cpu_pyramid_pool` docstring `:194-196`); two stations within 160 m collapse to the same token
(19 of 75 pairs); and the `patch_mode=all`[105] == `patch_mode=station` smoke invariant cannot be
bitwise on GPU — use `allclose`, or CPU fp32 under `model.eval()`.

Also missing from the first draft's change list: `check_dataset.py:37-58` (already stale),
`tier1_probe.py:195,205`, `plot_spatial_heterogeneity.py`, `plot_architecture.py:208`; `STATION_ROW`
/ literal `112` in a further eight files (`plot_tile_context.py:440` **already** computes
`(row//16)*14+(col//16)` — reuse it); **six independent copies of `ERA5_VARS`** plus three hardcoded
`19`s, and `station_mean_probe.py:140` reads `ERA5_VARS.index("skt_mean")` directly and breaks
outright on the `skt` drop; and 15 `SoilMoistureDataset(...)` construction sites.

### 35.5 The failure mode this staging exists to avoid

Three to four weeks of engineering; the run trains against a restored-but-unverified store; the gate
returns unchanged because three of its four numbers measure a quantity three independent studies
(§27b.8, §29.13, §32.10) have already measured as absent; eleven simultaneous changes make the
result unattributable against a bar that is a prose cross-reference contradicting its own source
table; the decision on step 2 cannot be made; §36 gets designed. **That loop is in the record four
times in thirteen days** (§30 -> §31 -> §33 -> §34). The one escape this project has found is what
it did to §32: a cheap, pre-registered, CPU-only gate allowed to kill its own arm.

Phase 1 is that gate.

---

### 35.6 PHASE 0 EXECUTED (2026-08-26) — what actually happened

Commits `102d23c`, `aa8a07a`, `db14008`, `696badb` on `feat/per-location-processor`.

| | before | after |
|---|---|---|
| `.complete` markers | 0 | **993** (842 + 48 + 103) |
| `.npy` memmaps | 360 partial | **7,818**, user-writable |
| `soil` | all zeros | real (Hupsel 7-293, Banizoumbou 8-150, CA-Cbo 14-364) |
| `era5/values` | no chunks | 100% finite |
| store verification | 993 FAIL | **993 PASS**, zero damage of any class |
| dataset end-to-end | empty (0 stations) | 5,383 samples from 4 stations, sample loads |

**Damage classification, measured** (`verify_zarr_store.py`, 12-station sample: 215 vs 166
array-instances). The purge took `.zarray` headers everywhere and, for the *small* arrays, the
chunks too. **Every large token array was META-ONLY** — the ~730 GB of L3/L6/L9/L12 chunks were on
disk the whole time, only their ~400-byte headers were gone. Age-based purge: headers written once
in June and never re-read; chunk files had atime refreshed by training and eval.

**Procedure used, and why it differs from the first draft.** The draft said wipe-and-restore. Reading
`slurm/restore_zarr.sh` showed `rsync -a` with **no `--delete`**, so a *merge* cannot remove
anything and is strictly safer. A `--dry-run --itemize-changes` over 28 stations then showed the
merge was purely additive — 2,403 files created, **0 content overwrites** — which also closed the
staleness worry (a June backup cannot roll back a post-June fix if nothing is overwritten). Restore
job 26058778: 9 min 36 s, no station-level rsync failures, self-verified 993/993.

Backup is `chmod -R a-w` and verified intact 993/993. Leave it that way. Verified end-to-end: a real
`rm` of a live `.zarray` in the backup is refused.

**Three bugs found, all the same shape — SILENT SUCCESS:**
1. `zarr.open_consolidated` returning `fill_value` with no exception (the original corruption).
2. `restore_zarr.sh` verifying by counting `.complete` markers — 0-byte files that rsync restores
   first, **and which come from the backup**, so it reported `sm_only: 842` on a gutted store.
3. `RSYNC_OPTS=(-a --chmod=u+rwX)` as a bash array. **Bash cannot export arrays**, and `copy_one`
   runs inside `bash -c` under `xargs`, so rsync ran with **no flags at all**: no recursion
   (`skipping directory .` per station), no `--chmod`. Job 26058649 looked healthy while doing
   nothing. Cancelled at 64 s; the partial copy was additive and harmless.

Plus one latent bug caught by the dry-run: the backup is now read-only, and `rsync -a` preserves
source permissions, so the restore would have handed the live store `r--r-----` — surfacing days
later as a permission error during memmap regeneration, far from its cause. `--chmod=u+rwX` added.

**The generalisable lesson:** every check that verified a *proxy* instead of the thing itself passed
while the thing was broken. `verify_zarr_store.py` exists to verify contents — `.zarray` present,
chunk count > 0, sampled read != `fill_value`, `soil.min() > 0` — and is schema-aware (S2 has no
`token_mask` by design; flux-only stations carry `labels/{le,dates_flux,le_qc}`; some stations have
`s1_desc` and no `s1_asc`).

**Recurrence is certain.** Scratch is purged **by age, not quota** — usage is 8.8% of an 8 TiB
allowance, so nothing was evicted for space. Restore-and-verify is a recurring operation. The
durable fix is a store-integrity pre-flight (below), not the restore.

**Not on scratch and never was:** raw S1 dB + S2 imagery live in
`/gpfs/work3/0/prjs1968/satellite_zarr` (998 stores, ~150 GB, on work3, not purge-exposed). §33's
decomposition needs that raw dB, not the tokens.

### 35.7 DECISION 2026-08-26 — BUILD THE PATCHWISE MODEL

**User decision, taken after the four critiques were presented. §35.2's Phase 1 is NOT a gate and
does not block the build.** Two items are retained from it as build *inputs*, both cheap:

- **Register standardisation stats** (§27a.7, train-split only, `Pool(64)`, "zero training-time
  cost"). **Not optional.** 102 of 141 tokens are L12 and `s2/l12` has register magnitude share
  **0.940**; LayerNorm is per-token and does not remove a direction shared *across* tokens (§27a.5:
  post-LN median pairwise cosine 0.783 -> 0.157 when two of 768 dims are zeroed). If all 100 history
  keys point nearly the same way, `q.k` is near-constant and **temporal attention collapses to a
  mean** — silently defeating the entire point of un-pooling.
- **Position-code probe** (~1 h CPU). TerraMind bakes 2-D sin-cos position into every token
  (`terratorch/.../encoder_embeddings.py:151`, `tm_utils.py:51`, `sincos_pos_emb=True` default), so
  dropping `spatial_row/col_emb` removes only the *explicit* channel and leaves the implicit one
  frozen into the features where it **cannot be ablated at all**. Decides whether translation
  augmentation lands in stage 2a.

**Honest scoping of what this build can answer.** §27b.8 measured own-token within-network SM skill
at **0-4.2%** on the **un-pooled centre token** (`probe_token_sm_structure.py:63,78`,
`CENTRE_TOK == 105`) — the same object this build introduces. The *spatial* question has already
been answered once, negatively. What is untested is the **temporal** axis: §27b.8 collapsed ~100
acquisitions to a multi-year mean; this keeps them. Consequently the arms in §35.9 are **not
optional** — without history-ablated and re-pooled, a null cannot distinguish "un-pooling does not
help" from "we re-measured what was already known".

**Layer:** default **L12** (status quo), with `--hist-layer {3,6,12}` exposed so §27a.7's "build fine
resolution from L3/L6, not L12" is tested as an ablation rather than assumed either way. The `.npy`
memmaps already hold the full `(N,196,768)` for L3/L6/L9.

### 35.8 Build order

| # | step | note |
|---|---|---|
| 0 | Store-integrity pre-flight | `slurm/train.sh` calls `verify_zarr_store.py` before training; a 3-line check in `_open_zarr` additionally covers the other 14 dataset construction sites |
| 1 | Register standardisation stats | not optional, see §35.7 |
| 2 | Position-code probe | ~1 h CPU |
| 3 | Anchor-redundancy check | fraction of samples whose anchor date is not in that sample's history, **per modality** |
| 4 | **Stage 2a** | dataset slicing, `PatchwiseEncoder`, per-depth heads, raw drivers, `--arch`, loss readout |
| 5 | Smoke on **2 GPUs** | the DDP bug is invisible on one |
| 6 | Four arms + shuffle control | §35.9 |
| 7 | Pre-register the gate, then run | §35.10 |
| 8 | Stage 2b (resamplers), 2c (`skt`, ASC/DESC, per-patch soil, translation aug) | each its own ablation |

**Staging rationale.** At K=1 the driver resampler *saves nothing at training time* — raw drivers
give 536 tokens, still 3.7x cheaper than today's 1035. It is an inference-cost optimisation for
K=196, and the gate's inference is TxSON only. So the most bug-prone new component (learned null
token, `-1e4` masking, the ~25% both-blank path with no precedent in this codebase) must NOT go into
the highest-stakes run for no benefit that run needs.

**Target sequence**, per patch k, patch axis folded into batch, weights shared:

```
[ depth_CLS x3 | dem_k | lulc_k | soil x4 | era5 | sif | twsa | hist_k x100 | CLS ]
        N transformer layers over THIS sequence only
        token head -> SM for patch k      196 patches -> 14x14 map @ 160 m
```

Statics enter as a **prefix, not appended to the summary**, so temporal attention can condition
drydown on cover and terrain. An **explicit per-patch CLS is required** — §34.3 says "learned CLS
per patch" but §G's 141-token count has no such token, leaving `h_k` undefined.

**Dropped:** `anchor_l12` and `anchor_l3/l6/l9` (the anchor is normally already one of the 100
history tokens with its own staleness; it existed *because* the history was pooled — verify per
modality first, step 3); `spatial_row_emb`/`spatial_col_emb`/`spatial_modality_emb`; `scale_emb`
(indexes pyramid levels, meaningless un-pooled). **Kept:** `rel_pos_emb`, `hist_modality_emb`,
`static_modality_emb`, the block modality tags, `circular_doy_pe`, `depth_tokens`.

**Depth heads: independent, NOT §18.4's star residual** (user decision). Measured label mass
(120-station sample): 0-10 at 100% of stations / 41.0% of station-days; 10-30 at 77.5% / 32.6%;
30-100 at 60.8% / 26.4%. The star residual was a sample-efficiency bias, not a data necessity.
**Caveat for reporting:** depth coverage is **not missing at random** — sensor configuration is a
network property confounded with climate, so deep heads train on a systematically different station
population. `FiLMLayer` (`model.py:162-168`) is 4-D only; a 1-D variant is needed, and
`depth_ctx[:, d, :]` is `(B,768)` against `u_k`'s `(B*K,768)` so it needs `repeat_interleave`.

### 35.9 Arms — one run answers nothing

Same code, same seed, one flag apart. `ablation.py`'s `MODALITY_KEYS` (`:28-38`) is the instrument.

| arm | change | answers |
|---|---|---|
| full | - | the headline |
| **history-ablated** | zero the 100 history tokens | does per-patch history contribute at all? |
| **re-pooled** | history replaced by its tile mean | isolates *un-pooling* — the actual hypothesis |
| **statics-only** | zero `dem_k`/`lulc_k` | §27b.8: `dem/l12` is the strongest within-network token signal (eta^2_w = 0.075, p = 0.0005). If this matches full, the finding is "un-pooling the *statics* helps" |
| **patch-shuffle** (negative control) | permute per-patch history across the 196 patches at eval | spread and off-centre skill **must** collapse. §29.7 mandated a shuffle control; §34 had none |

### 35.10 The gate must be rebuilt — §28.8's version is not usable

Four confirmed defects:

- **">35% spread" misquotes its source.** `plot_network_timeseries.py:156-158`: **0.6** is *"the map
  resolves the tile"*; **0.35** is only the floor below which it *"repeats ~one series"*. Passing
  §28.8's gate buys the verdict *"partly resolves"*. Misquoted in four places in this runbook.
- **`r > 0` on CR200-18 is a coin flip.** n = 6; the 95% CI on the current -0.175 is
  [-0.864, +0.742]. Across §26.11's four densest tiles the current model gives -0.135, +0.012,
  -0.589, -0.077 — **CR200-3 already passes** — and P(>=1 of 4 passing by chance) = 0.94. The
  reference tile was chosen post hoc.
- **The reference is not an artefact.** §28.8 records no run, checkpoint, epoch, SHA or W&B id; its
  `r = -0.175` contradicts §26.11's own table (-0.135); spread appears as 15-19% / 17% / 18.8% /
  20% in four places; the adjudicating parquet is gitignored. Regenerate from a named checkpoint or
  drop the comparison.
- **The baseline mixes memorised train stations with held-out ones.** §26.11 split-stratified:
  train 0.0110 / val 0.0299 / oos 0.0386. TxSON's 40 stations are 14/8/18, so the pooled 0.0301 is
  not a generalisation metric; the honest held-out bar is ~0.036.

**Replacement, pre-registered before run 1.**

**Primary endpoint — a within-station criterion, which the old gate lacked entirely.** At off-centre
readouts, de-mean prediction and observation *by station*, then score the residuals. Without this,
passing is compatible with having learned a **static landscape map** — which §27a/§27b say is what
these tokens encode, and which the following measurement says would suffice: from
`csvs/gate_pair_deltas.csv`, for pairs >=160 m apart **the sign of the between-station SM difference
holds on 98.6% of days** (§32.11 measured 95.8% and called it "the finding that outlives the terrain
arm"). A static per-patch offset captures nearly all of the target. **That is the §29 failure
repeated**, and §29.15's own verdict was *"a static field cannot track a dynamic variable"*.

Secondary: the four §28.8 metrics, **split-stratified** (train/val/oos), CIs bootstrapped over
**stations** not samples, each expressed as a fraction of the achievable ceiling. **The ceiling was
never measured** though §27.2 pre-registered it as *"this runs first; a null reported without it is
uninterpretable"* — first-cut from `csvs/gate_pair_deltas.csv`: ~30.4% of the >=160 m
between-station variance sits **below one token cell**, so max spread ratio ~ 0.83.

Also: spread and correlation gated **jointly** (spread alone is unbounded above and rises whenever
the model emits per-patch variation, correct or not); fix the tile set in advance; give "materially
closer off-centre" a number and a paired test; state a **collapse criterion** not just a log, and log
**temporal** attention entropy over the history block (the register-driven collapse shows up there,
not in the map SD); one primary endpoint with secondaries under Benjamini-Hochberg, as §27b.6 did
correctly.

**Delete the "~2M supervised samples is ample" argument** (§G7 reason 2). §27b.3 already rejected it
as *"pseudo-replication — it multiplies rows without adding information about a per-station
target"*, and §29.14 measured the damage. Effective n for the spatial question is 993 stations, and
generously: 33 networks, top-5 = 74.8%, SNOTEL alone 38.8%; drydown tau ~ 3.2 d so consecutive days
are ~1/3 of an independent observation.

**Set a stopping rule before submitting.** Thirteen 4xGPU runs have never converged: five OOM-killed,
three cancelled by human judgement, ~412 GPU-h on abandoned runs, 15 jobs with `oom_kill` events.
Best-val epochs were 3, 3, 3, and **16 of 16 still improving**. The recorded note: *"I called this
run converged at e5 and again at e9. Wrong both times."* And `val_loss` is **not** comparable across
`per_depth_loss` settings — changing the depth-head structure invalidates every val-loss comparison
to a prior run unless recomputed.

### 35.11 Implementation defects to fix, all verified against source

**Two guaranteed crashes.** (1) `train.py:902` uses DDP with `find_unused_parameters=False`, so
leaving the unet path *constructed* but unused raises `RuntimeError` on step 1 of every DDP run —
gate **construction** on `arch`, and smoke on **2 GPUs** since this is invisible on one.
(2) `token_mask` is `(T,14,14)` not `(T,196)` (`dataset.py:244`, `:328`), so `token_mask[:, sel]`
indexes the row axis and silently returns `(T,K,14)`.

**Silent wrongness.** Padded and NaN slots are marked **valid** — `token_mask` inits to `ones` and is
written only for filled slots matching a cloud-mask date; median S2 acquisitions/station-year is
**36** against `MAX_S2 = 60`; NaN slots `continue` at `:289-290`; no-cloud-mask dates have no `else`
at `:295`. Mask must be `token_mask.reshape(T,196)[:, sel] & (doys > 0)[:, None]`. Two zero-fallback
branches (`:248-249`, `:371-372`) break collation for stations with no S2/S1 in window.
"Everything after `model.py:779` is shape-agnostic" is **false** — `label` is `(B,n_depths)` and
never flattened, and in `readouts` mode the K tokens are *different stations with different labels*.
`lambda_boundary = 0.1` silently changes meaning (50,176 px x 3 depths today vs K=1 x 3 patchwise)
and `total_variation_loss` raises `IndexError` under patchwise. `--use-cls-depth` is not default-on
(`train.py:207`) but the sequence requires it. `slurm/train.sh:43` hardcodes `--use-memmap`.
`--arch` must land in **`CONFIG`** (saved `:1088`/`:1132`; `ckpt_utils.py:38-45` rebuilds from
`ckpt["config"]`) or every eval script silently constructs the unet — note `demo_plot.py:44-48` and
`plot_satellite_sm_meeting.py:47-50` carry **duplicate** `load_checkpoint` implementations that
bypass `ckpt_utils` entirely. `token_sel=all` reinstates the *"~437 GB DataLoader IPC queue that
caused epoch-boundary OOM kills"* recorded in `_cpu_pyramid_pool`'s own docstring (`:194-196`) — cap
eval batch size. Two stations within 160 m collapse to the same token (19 of 75 colocated pairs) and
get identical predictions; nothing guards this.

**Also missing from §34's change list:** `check_dataset.py:37-58` (already stale),
`tier1_probe.py:195,205`, `plot_spatial_heterogeneity.py`, `plot_architecture.py:208`; `STATION_ROW`
or literal `112` in eight further files (`plot_tile_context.py:440` **already** computes
`(row//16)*14 + (col//16)` — reuse it); **six independent copies of `ERA5_VARS`** plus three
hardcoded `19`s, with `station_mean_probe.py:140` reading `ERA5_VARS.index("skt_mean")` directly and
breaking outright on the `skt` drop; and 15 `SoilMoistureDataset(...)` construction sites.

**Dead/broken tests:** `test_gather_equiv.py` imports a function that no longer exists;
`test_pyramid_equiv.py` tests `_cpu_pyramid_pool` and dies with the refactor;
`test_per_depth_loss.py` breaks in four places.

### 35.12 Cost — the §34.7 / §G2 arithmetic measures the wrong quantity

Per-layer per-sample FLOPs at d=768 are `12.T.d^2 + 2.T^2.d`; the **linear term dominates**, so
counting attention pairs is wrong:

| | T | total/layer | vs baseline |
|---|---|---|---|
| baseline | 1038 | 9.00e9 | - |
| patchwise K=1 | ~141 | 1.03e9 | **8.8x cheaper** (not 54x) |
| patchwise K=196 | - | 2.02e11 | **22x more expensive** (not 3.6x) |

Attention is 3% of the patchwise cost. And **epochs are data-bound** (data 250-630 s vs compute
483 s), so an 8.8x compute cut applies to under half the epoch — realistic gain **1.6-2x**, not 54x.
The 145 GB `/dev/shm` preload is unchanged at **17-32 min per job start** (project total: 44 starts,
8.5 h wall ~ 34 GPU-h). Budget is not the constraint: 79,592 of 800,000 GPU SBU used in six months,
expiring 2027-02-16, ~47 full runs in hand. **The scarce resource is calendar time.**

### 35.13 Rollback — checked, no retag needed

The red-team claimed `baseline-unet-temporal` sat on the wrong commit and that the real code was
`0313f25`. **Both halves are wrong**, verified 2026-08-26: `0313f25` is itself **docs-only**
(touches `text/logs.txt` alone), and while the tag's commit `a46efaa` has a docs-only *subject*, the
**code state** is what matters — `git diff b81faf7 a46efaa -- model.py train.py dataset.py
ckpt_utils.py` is **empty**, the only code delta being that `eval_predict.py` was *added*.
`b81faf7` ("Pre-launch bug hunt: fix 9 defects", 2026-08-05) is the last code commit before the
`cls_depth_star_reg` run. So the tag holds exactly the baseline training code plus eval tooling —
**leave it alone**. The reviewer inferred from a commit subject without checking the tree: the same
class of error as trusting a `.complete` marker.

**The real gap does stand:** no code path records a commit SHA into a checkpoint, so a reported
number cannot be traced to the code that produced it. Stamp `git rev-parse HEAD` into `CONFIG`
during stage 2a. Also, `best.pt` **is** resumable — byte-identical to `last.pt` (both 604,402,070 B,
epoch 16) — so §G6's claim was wrong; the real gap is that there is no second copy.

Branch: **`feat/patchwise-temporal`**, created off `feat/per-location-processor` so the Phase 0
store-verification work travels with it. `baseline-unet-temporal` remains the rollback point.

### 35.14 The Perceiver resampler is DROPPED — the cost it was solving does not bind

§G2/§G3 specified three Perceiver-style resamplers compressing the tile-level drivers
(ERA5 365 + SIF 50 + TWSA 12 = **427 tokens**) to 32 latents, justified by inference cost: each
patch's sequence carries the drivers, so at K=196 you pay 427 driver tokens **196 times**, quoted
as "59x". That framing never asks whether the *absolute* cost binds. It does not.

Raw drivers give a per-patch sequence of `3 depth_CLS + 1 dem + 1 lulc + 4 soil + 365 + 50 + 12 +
100 hist + 1 CLS = 537`. Per tile-day, forward, 6 layers, d=768:

```
linear     12 . T . d^2 = 12 x 537 x 768^2 = 3.80e9   per patch per layer
attention   2 . T^2 . d =  2 x 537^2 x 768 = 0.44e9
                                        ->  4.24e9 x 196 patches x 6 layers ~= 5.0 TFLOP
```

**~0.05 s per tile-day on an H100 even at 10% of peak.** A full year of 14x14 maps over all 40
TxSON tiles is ~12 minutes. The resampler would make that ~3 minutes.

Where K=196 is actually needed is narrower than §G2 assumes:

| use | K | needs the resampler? |
|---|---|---|
| training | 1 | no — raw drivers at 537 tokens are already ~3.7x cheaper than today's 1035 |
| gate evaluation (off-centre readouts) | <= 6 | no |
| 14x14 map figures | 196 | no — 0.05 s/tile-day |

**And deferring it to "stage 2b" was not free, which the staging argument missed.** Training 2a with
raw drivers and then introducing a resampler is a *different model* — the weights do not transfer,
so 2b would be a full retrain rather than an add-on. A retrain to save 9 minutes of figure
rendering is unjustifiable.

Two further reasons the design was weak, from the critiques:

- **"Three separate resamplers, for clean ablation" rests on a false premise.** Exact ablation of a
  modality is achieved by masking its *inputs* and letting the null token absorb it — and that is
  already the live code path at `dataset.py:1064-1065`, `:1071-1072`, which blanks SIF and TWSA on
  ~50% of training samples by design. Separation therefore buys nothing that masking does not
  already give, while costing real cross-modal capacity: modality-blind compression discards what a
  cross-modal interaction would have used, so "same expressiveness, interaction moved downstream"
  is wrong.
- **The 32-latent budget contradicts its own derivation.** §G2 derives 32 from a *meteorological*
  argument (14 daily + 8 weekly + 10 monthly) and then hands 8 of the 32 to SIF and TWSA — which the
  same document calls *"tile-constant, so neither can contribute to within-tile pattern"*. Step 1's
  entire question is within-tile pattern.

**Decision: no resampler. Raw drivers everywhere.** This removes the most bug-prone component in the
plan (learned null token, `-1e4` masking, the ~25% both-blank path with no precedent in this
codebase) and removes a retrain. The property worth preserving from the idea — drivers living
*inside* each patch's attention rather than broadcast as one vector, so each patch can integrate the
same weather differently (§34.8's "response x wetness") — is preserved *better* by raw tokens than
by 32 latents.

### 35.15 Tile-level tokens become a read-only cross-attended memory (the real inference lever)

The genuine inefficiency at K=196 is not the number of driver tokens, it is that **427 of each
patch's 537 tokens are byte-identical across all 196 patches, and self-attention recomputes their
projections 196 times.**

```
per patch:   [ depth_CLS x3 | dem_k | lulc_k | hist_k x100 | CLS ]     106 tokens, SELF-attention
tile-level:  [ era5 365 | sif 50 | twsa 12 | soil 4 ]                  431 tokens, K/V computed ONCE
             every patch CROSS-attends into that memory
```

Per layer, d=768, K=196:

| design | FLOPs/layer |
|---|---|
| naive, everything in self-attention (T=537) | 8.32e11 |
| self-attn over 106 + cross-attn to 431 | **1.89e11** |
| | **~4.4x cheaper** |

Breakdown of the cross-attention arm: Q/O projections `2 . 106 . d^2` per patch = 2.45e10 over 196;
attention `2 . 106 . 431 . d` = 1.38e10 over 196; and K/V projections `2 . 431 . d^2` computed
**once per tile** = 5.1e8. Activation memory falls with it — `196 x 106` instead of `196 x 537`.

**This subsumes the resampler and beats it.** The resampler bought ~3.8x by *discarding* 427 driver
tokens down to 32. Cross-attention gets ~4.4x while keeping every driver token intact, needs no new
module to train, and introduces none of the null-token machinery.

**There is a physical argument for it, not merely an arithmetic one.** Meteorology is exogenous:
ERA5 at 9 km is not modified by which 160 m patch is being predicted. A patch must read the weather;
the weather need not read the patch. Read-only drivers encode that. What is given up is drivers
being contextualised *by patch content* — which the current global model permits, but for which no
mechanism has ever been articulated.

`soil` (tile-level, 4 tokens) gets the same treatment for free. `dem_k` / `lulc_k` are per-patch and
stay in the self-attention sequence.

**OPEN DECISION, and it must be taken before stage 2a trains.** Cross-attention changes the
encoder's shape, so retrofitting it later means a retrain — the identical trap the resampler had.
Either build it into 2a from the start, or accept the naive form permanently. Not a decision to
defer.

### 35.16 Training speed — the model is not the bottleneck, and patchwise makes that worse

Measured from the last full run: `data = 250-630 s` against `compute = 483 s`, GPU utilisation
43-46%. Patchwise cuts compute ~8.8x at K=1, so compute lands near ~55 s and the epoch becomes
**almost entirely dataloader-bound**. Optimising the model buys nothing; the data path is where the
time is. Three wins, all unlocked *by* patchwise:

1. **Read patch k directly instead of copying whole acquisitions.** The loader currently does
   `tok = torch.from_numpy(tokens_z[src_i])` — a full `(196,768)` = 294 KB — writes it into a
   `(T,196,768)` buffer, and only then slices. Per sample that is 100 x 294 KB ~= **29 MB of memcpy
   to use 150 KB of it**. `l12` is C-contiguous, so element `[i,k,:]` is a contiguous 768-float run
   and `tokens_z[src_i, sel, :]` reads **1.5 KB** directly. ~196x less memory traffic, and the large
   per-sample buffer allocation disappears.
2. **The `/dev/shm` preload can shrink ~196x.** It holds **145 GB** because the pooled path needed
   every token. If training only ever reads token 105, the preload needs one column: **145 GB ->
   ~0.74 GB.** That removes the **17-32 min startup cost per job start** (project total: 44 starts,
   30,626 s = 8.5 h wall ~= 34 GPU-h on preload alone) and frees ~145 GB of host RAM — which is what
   four of the five OOM kills were about.
   **Dependency:** this only holds if training reads a *fixed* token set. If the position-code probe
   says translation augmentation is required, k varies per sample and a neighbourhood (or all 196)
   must be resident. A k x k window around centre is the middle option. **The probe result sizes
   this win.**
3. **IPC payload drops ~8x** — `(T,4,768)` fp32 pooled = 1.2 MB/sample against `(T,1,768)` fp16 =
   154 KB.

Second-order, after the data path: `torch.compile` on the encoder, and confirming
`nn.TransformerEncoderLayer` reaches SDPA/flash rather than the math fallback.

### 35.17 Stage 2a progress (2026-08-26)

Branch `feat/patchwise-temporal`. Commits `d4470d7` (dataset), `476065d` (tests), plus the Phase 0
set.

**Done — build step 0 and step 4 part 1/4.**
- `slurm/train.sh` runs `verify_zarr_store.py` before training and refuses a damaged store.
- `dataset.py`: `_finalise_history()` / `_empty_history()` shared by both loaders, so the S2 and S1
  paths cannot drift — they already differ subtly (S2 uses `enumerate(win_idx)` so NaN skips leave
  gaps; S1 increments `out_i` only on success). `--arch unet` is unchanged (`token_sel=None`).
  Patchwise emits **distinct keys** (`s2_hist`, `s2_hist_valid`, `dem_tok`, `lulc_tok`, `token_idx`,
  `token_valid`) and **removes** the pooled ones, so a stale consumer raises `KeyError` rather than
  silently reading `(T,K,768)` as `(T,4,768)`. All three silent-wrongness bugs of §35.11 fixed, each
  with a regression test in `test_patchwise_dataset.py`.

**Confirmed against source while implementing — both §35.11 claims hold:**
- `circular_doy_pe` (`model.py:75-91`) uses `len(doys)` and `doys.unsqueeze(1)`, so it **requires
  1-D input**; a `(B,N)` call raises. Every existing call flattens first.
- `FiLMLayer.forward` (`model.py:162-168`) does `params[:, :C].unsqueeze(-1).unsqueeze(-1)` —
  **4-D only**. The token head needs a 1-D variant, and `depth_ctx[:, d, :]` is `(B,768)` against
  `u_k`'s `(B*K,768)`, so it needs `repeat_interleave(K, 0)`.

**Next:** `model.py` (`PatchwiseEncoder`, construction gated on `arch` so the DDP unused-parameter
crash cannot occur; 1-D FiLM; explicit per-patch CLS), then `train.py` (loss gather over
`token_idx`, `--arch` into `CONFIG`, guard `lambda_boundary` and `total_variation_loss`), then
`eval_predict.py`. **§35.15's cross-attention decision gates the encoder shape and must be settled
first.**

### 35.18 The §35.15 decision is TAKEN — read-only driver memory, both modes behind a flag (2026-08-26)

**Full derivation, with every dimension and a worked numeric example: `text/patchwise_math.md`.**
That document is the reference; this section records the decision and what it changes.

**Decision.** Build both driver wirings behind `--driver-mode {memory, concat}`, default `memory`,
and train stage 2a with `memory`. All 431 tile-level driver tokens (era5 365 + sif 50 + twsa 12 +
soil 4) are kept **whole** in both modes — the Perceiver resampler stays dropped per §35.14, and
with it the learned latent queries, the null token and the `-1e4` masking.

```
memory (default)
  self:   [ depth_CLS x3 | dem_k | lulc_k | hist_k x100 | CLS ]   L = 106   per patch
  cross:  [ era5 365 | sif 50 | twsa 12 | soil 4 ]                M = 431   K/V once per tile-day
concat
          all T = 537 in one self-attention stack
```

**Why it had to be decided before the encoder is written, proved rather than asserted.** Under
`concat` the driver tokens sit in the `(431,106)` block of the score matrix — the weather *reads the
patch*. `text/patchwise_math.md` §3.4 works this through numerically at d=2: two patches are handed
a byte-identical driver token `[1,1]`, and after **one** layer it has become `[1.000, 0.670]` for
one patch and `[0.670, 1.000]` for the other. The input redundancy is real; one layer of full
self-attention destroys it. So there is nothing to cache from layer 2 onward, and retrofitting the
cache later is a retrain — the identical trap the resampler had.

**Three corrections to §35.14–§35.17 that this derivation forces.**

1. **The 4.4x is mostly the FFN, not the attention.** Per layer the linear term is `12·n·d²` and the
   attention term `2·n²·d`; at n=537, d=768 that is 3.80e9 vs 0.44e9, so the linear term is ~90%.
   Under `concat` the FFN runs on all 537 tokens for each of the 196 patches; under `memory` on 106.
   Any cost claim in this runbook derived from *attention pairs* — §34.7 and `open_items.md` §G2
   both — is wrong by a large factor and should be re-derived from `12·n·d² + 2·n²·d`.
2. **§35.15's "12 min vs 3 min" is 2x low.** It treated multiply-adds as FLOPs. Corrected: 0.10 s vs
   0.023 s per tile-day, i.e. **~24 min vs ~5.6 min** for a year over 40 TxSON tiles. This changes
   nothing, and that is the point — **the cost argument is not the reason to prefer `memory`**.
   §35.14 killed the resampler by showing the absolute cost does not bind; consistency requires
   applying the same test here, and `memory` fails it too.
3. **The reason that does survive is optimisation, not expressiveness.** `memory-form` is a strict
   *subset* of `concat-form` — concat can express everything memory can, plus the deleted block, so
   trained perfectly concat is never worse. What differs is what the optimiser is asked to do: the
   patch-specific share of what the per-patch CLS attends over is **102/537 = 19%** under concat
   against **102/106 = 96%** under memory. Step 1 asks whether per-patch history carries within-tile
   SM information at all; if concat returns a null, *"un-pooling does not help"* and *"the optimiser
   never dug the signal out of an 81% constant background"* are indistinguishable. That is the same
   class of confound §35.9's five arms exist to remove, and §35.12 names calendar time — not GPU
   SBU — as the scarce resource, so an uninterpretable null is the expensive outcome.

The `--driver-mode` flag exists so that whether this mattered is a **measurement**, not an
assertion. FFN, norms, heads and the driver encoder are shared; only the attention wiring differs,
so the second mode is ~30 lines.

**Two required components that were not in §35.15.**

- **A driver self-encoder**, 1-2 layers over the 431 tokens, run once per tile-day *before* they
  become K/V. Read-only memory alone never lets driver days interact: a cross-attention head can
  form weighted sums (so "total rain over 14 days" is expressible) but not sequential structure
  (drydown since the last event). Cost 6.7e9 MAC against ~1.1e12 for the patch stack.
- **Explicit projections, not stock `nn.MultiheadAttention`.** MHA runs `in_proj` on whatever it is
  given, so `m.expand(196,431,768)` would re-project the same 431 tokens 196 times and return every
  bit of the duplication. The block needs its own `q_proj`/`k_proj`/`v_proj`/`o_proj` with
  `F.scaled_dot_product_attention`, and the `k_proj`/`v_proj` calls lifted out of the patch loop.

**Cache and batching.**

```
cache = { (Kc_l, Vc_l) : l = 1..6 }     6 x 2 x (431,768) fp16 = 7.9 MB per tile-day

Kc (D,431,768) -> (D,12,431,64)          one cache per DAY
Q  (D,P,106,768) -> (D,P,12,106,64)      one query set per (day, patch)
S = einsum('dphlx,dhmx->dphlm', Q, Kc)   d on BOTH, p on Q ONLY -- that asymmetry IS the sharing
```

**One model instance, patches in the batch dimension** — not 196 copies. Weight sharing is what lets
supervision at the station's single token teach the mapping at all 196 (§34.4), and it is why
training at K=1 and inference at K=196 use the *same checkpoint*: `token_sel` changes only the batch
width. Separate instances appear only across SLURM array tasks.

**The silent bug to guard.** The cache is per *day*, not per patch. A wrong `(D,P)` view or a stray
`repeat_interleave` on the wrong axis makes every patch read another day's weather — nothing
crashes, the model is merely mediocre. Assert it in `test_patchwise_model.py`.

**Inference is I/O-bound, not compute-bound, and the fix is loop order.** Re-reading a
`(100,196,768)` fp16 window per tile-day is 30 MB x 14,600 = ~438 GB. But a station's *entire* L12
record is ~35 MB per modality (Hupsel, 117 dates), ~100 MB across S2/S1asc/S1desc — **~4 GB for all
40 TxSON tiles, ~100 GB for all 993** on a 720 GB node. **Station outer, day inner** with the array
resident (`/dev/shm` memmap, `dataset.py:749-765`) turns 438 GB into one 4 GB read plus in-RAM
slicing. Shard across tasks **by station, never by day**: days are independent computationally (the
model is not recurrent), but consecutive days share ~99% of their history window, so day-sharding
re-reads each station array once per task. `slurm/eval_predict.sh` already has
`--csv-start-idx`/`--csv-end-idx`; reuse it. Only after the loop order is right does adding GPUs
help — otherwise it just adds GPFS contention.

**Build order is unchanged from §35.8**, with the encoder shape now settled: steps 1-3 (register
standardisation stats, position-code probe, anchor-redundancy check) run as CPU sbatch jobs
concurrently with step 4 coding; then 2-GPU smoke, then the five arms, then the §35.10 gate. Full
plan in `/home/pkhanal/.claude/plans/for-implementing-the-new-enumerated-comet.md`.

### 35.19 Capacity and data budget — measured, and what it can and cannot answer (2026-08-26)

**Parameter counts, measured not estimated** (`SoilMoistureModel(n_depths=3, use_cls_depth=True)`
instantiated and summed):

**Estimated 2026-08-26, then MEASURED after the build — the estimate was 3.5 M low** because it
omitted the depth heads: `FiLM1d(768, 768)` is a `Linear(768, 1536)` = 1.18 M, three of them.

| configuration | params | vs baseline |
|---|---|---|
| **`--arch unet` (baseline)** | **50.35 M** | — |
| — 6 transformer layers | 42.53 M | |
| — UNet decoder | 6.70 M | |
| — SoilEncoder + era5/sif/twsa MLPs + embeddings | 1.12 M | |
| **`patchwise --driver-mode memory --n-layers 6 --driver-layers 2`** | **75.53 M** | **1.50x** |
| `patchwise --driver-mode concat --n-layers 6 --driver-layers 2` | 61.34 M | 1.22x |
| `patchwise --driver-mode memory --n-layers 4 --driver-layers 2` | **56.62 M** | **1.12x** |
| `patchwise --driver-mode memory --n-layers 6 --driver-layers 6` | 103.88 M | 2.06x |

`concat` is 14.2 M smaller than `memory` because it has no cross-attention block — 4 x d^2 per
layer x 6. That is a real confound for the `--driver-mode` arm and must be reported with it.

Per-layer blocks at d=768: self-attn 2.36 M, cross-attn 2.36 M, FFN 4.72 M. **The patch decoder alone
exceeds the entire current model**, because every layer now carries two attention blocks instead of
one.

**Size should be chosen to remove a confound, not to save time.** §35.16 measured epochs as
data-bound (data 250-630 s vs compute 483 s); patchwise cuts compute ~8.8x at K=1, so compute lands
near ~55 s and the epoch becomes almost entirely dataloader-bound. Shrinking the model saves
essentially nothing in wall-clock; growing it costs essentially nothing. What size *does* affect is
attribution: if patchwise runs at 2x the baseline's parameters and wins, architecture and capacity
are confounded.

**DECIDED (user, 2026-08-26): `--n-layers 6 --driver-layers 2` = 75.53 M measured.** T2 stays at 6 to match
the baseline's 6 transformer layers, keeping the architecture comparison clean. The
**capacity-parity variant** `--n-layers 4 --driver-layers 2` (**56.62 M** measured, against the
baseline's 50.35 M — 1.12x, the closest parity available) is deferred to **ablation, not run 1**. Note the two parities are
mutually exclusive: the baseline layer is `self + FFN`, ours is `self + cross + FFN`, so equal
depth means unequal parameters and vice versa.

**`N_drv` is a depth, not a repeat count** — T1 runs **once per tile-day** whatever its depth.
Do **not** default the weather encoder to 6 layers: an earlier draft argued for that on
Vaswani equal-depth convention, but matching a convention is worth less than an interpretable
comparison, and the extra 28 M buys depth only on a **tile-constant** input that cannot contribute
within-tile pattern at all.

**Data budget — the answer differs by question.**

*For fitting the model: sufficient.* 993 stations x ~2000 days = ~2 M samples; with tau ~ 3.2 d,
consecutive days are ~1/3 of an independent observation, so ~625 k effective station-days against
75.5 M parameters. This is the same regime the current 50 M model already trains in without
overfitting — one run had val still improving at **16 of 16 epochs**. Compute is not a constraint:
79,592 of 800,000 GPU SBU used, ~47 full runs in hand.

*For the question §34/§35 exists to answer: thin exactly where it matters.* At K=1 the model
supervises **one patch per sample — token 105, the station's own cell. There is never a label for
any off-centre patch.** The mapping is learned from station patches and *assumed* to transfer to the
other 195; nothing in training tests that. The evidence that can test it:

```
colocated station pairs                        75 pairs / 84 stations
  usable (>=120 common days)                   49
  minus 19 pairs closer than 160 m (same token, identical prediction)   ~56 distinct-token
TxSON readouts                                 96, of which 56 off-centre, over 40 tiles
```

Tens of pairs and 40 tiles. That is precisely why §35.10 requires a **within-station** primary
endpoint, CIs bootstrapped over **stations** not samples, and a measured ceiling — scored on pooled
samples, a null here is under-powered rather than informative. Covariate shift compounds it:
stations sit in accessible, flat, agricultural places, so off-centre patches are drawn from a
different distribution than anything ever supervised (this is also §G7's named trigger for
revisiting pretraining).

**The one change that would materially help is not more data — it is multi-station supervision.**
A tile containing two stations yields **two labels for two different patches in one forward pass**,
which is the *only* direct supervision of the spatial mapping anywhere in this dataset. §35.3 already
flags that it is not free today:

- `dataset.py:1076` has one label vector per sample
- `station_key` is a pure function of station identity (`:818-824`)
- **`location_group_id` exists in `csvs/station_splits.csv` — 906 groups for 993 stations — and
  `dataset.py` never reads it**

So ~87 stations share a tile with another, plus the 84 colocated ones. Wiring this turns the
strongest evidence in the dataset from a *validation* signal into a *training* signal. **Promote it
from a stage-2c nicety to the first item after 2a**, and note the §35.11 hazard it interacts with:
two stations within 160 m collapse to the same token and get identical predictions, so the loss
gather must handle duplicate `token_idx` rather than silently averaging two labels onto one patch.

**Depth is not a law, and ours is not derived.** `N = 6` comes from Vaswani et al. 2017, chosen
empirically for WMT at that compute budget (their Table 3 ablates N = 2, 4, 6, 8), and propagated by
inheritance. BERT-base is 12, BERT-large 24, ViT-Base 12 (TerraMind's own backbone — hence
L3/L6/L9/L12), T5-base 12+12, GPT-3 96. This project's `n_layers = 6` (`train.py:206`) copied
ViT-Base's width (768, 12 heads) and halved the depth. So `--n-layers` should be swept once 2a has a
baseline; it is cheap, because epochs are dataloader-bound and a smaller model costs the same
wall-clock.

**The model is fully bidirectional; there is no causal mask anywhere**, which is correct because we
regress one value from a fixed window rather than generating a sequence. It does not leak: the
window is strictly backward-looking (`rel_pos = 364 - (target - acq).days`, `dataset.py:89-96`), so
attention running "forwards" inside it still only sees the past. The one open exception is §33.12's
`c_k` look-ahead — a §33 decoder issue, untouched by stage 2a, still not closed.

### 35.20 STAGE 2a — the executable build plan (2026-08-26)

Settled inputs: `--arch patchwise`, `--driver-mode memory` (with `concat` behind the same flag),
`--n-layers 6`, `--driver-layers 2`, 72.0 M parameters. Derivation in `text/patchwise_math.md`.
Ordering is by dependency, not by importance.

**Step 0 — DONE.** Store pre-flight (`slurm/train.sh:38-58`); dataset per-patch history (`d4470d7`,
`476065d`).

**Step 1 — register standardisation: DROPPED for run 1 (user decision, 2026-08-26).**
§35.7 called it "not optional". That was over-claimed, and checking the actual measurements shows
why. `csvs/register_across_modalities.json` (993 stations) separates two quantities that §35.3 and
§35.7 conflated:

| key | register dims | `mean_magnitude_share` | `median_token_max_over_median` |
|---|---|---|---|
| `dem/l12` | 87, 126, 328 | 0.884 | **13.02** (sink) |
| `lulc/l12` | 126 | 0.354 | **9.15** (sink) |
| `s2/l12` | 9, 87, 126, 328, 329, 723 | **0.940** | **7.19** (sink) |
| `s1_asc/l12` | 87, 126, 716 | 0.912 | 1.23 (no sink) |
| `s1_desc/l12` | 87, 126, 716 | 0.911 | 1.24 (no sink) |
| `s2/l3` | 9, 126 | 0.586 | 1.51 |

- The **sink** (one token position dominating) is a DEM / LULC / s2-L12 phenomenon; S1 has none. It
  was already audited and cleared — token 105 is the sink in 0/993 DEM, 1/993 LULC (§35.3).
- The **register** (a few dimensions dominating, shared across tokens) is broad, not DEM-only, and
  dims 87 and 126 recur across nearly every modality.

**But the number the argument rested on is DEM-only.** Post-LayerNorm median pairwise cosine
0.783 -> 0.157 lives in `csvs/register_dim_variance.json`, which contains **only `dem` and `lulc`**
— it is the DEM p14 row. Post-LN collapse has **never been measured for s2 or s1**, and nobody has
measured the quantity that actually decides this: post-LN pairwise cosine between the ~100 history
tokens of the *same patch across time*. Every existing measurement is across stations or across
token positions; temporal attention cares about across-acquisitions.

**Why deferring is safe here, unlike the driver-mode decision.** §35.10 already mandates logging
**temporal attention entropy over the history block**, which measures whether attention actually
collapsed in the trained model — strictly better evidence than an offline cosine proxy. And
retrofitting standardisation later forces a retrain only in the world where entropy came back
collapsed, i.e. where the run was invalid anyway. There is no scenario in which deferring loses a
*good* run. Cross-attention was a one-way door; this is not.

**Binding condition:** temporal attention entropy logging is now **load-bearing**, not a
diagnostic nicety. It is the only thing separating "un-pooling does not help" from "attention
collapsed to a mean". Log it per layer against the uniform reference (log 100 = 4.605 nats). If it
fires, build `compute_register_stats.py` (train-split per-dimension mean/std, `Pool(64)`, subsampled
~20 acquisitions/station -> ~2.7 M vectors for 768 statistics) plus a `--standardise-tokens` flag,
and rerun.

**Step 2 — `model.py`, additive only; `--arch unet` must not move.**
- `DriverMemoryEncoder` — reuses `era5_mlp`/`sif_mlp`/`twsa_mlp`/`SoilEncoder`, `circular_doy_pe`,
  `rel_pos_emb` and the modality embeddings verbatim (`model.py:604-662`), then `N_drv = 2`
  `DropPathTransformerLayer`s. Returns `m (B,431,768)` and `pad (B,431)`. Runs once per sample.
- `PatchwiseBlock(d, h, driver_mode)` — pre-norm self-attention over 106; under `memory` an explicit
  cross-attention with its **own** `q_proj/k_proj/v_proj/o_proj` and `F.scaled_dot_product_attention`
  (NOT `nn.MultiheadAttention` — it runs `in_proj` on an `.expand()`ed memory and re-projects 196
  times, returning all the duplication); shared FFN and norms. Under `concat` the memory is
  concatenated into the self sequence and no cross-attention is constructed.
- `PatchwiseEncoder` — folds K into the batch, builds `x_k`, **explicit per-patch CLS**, 6 layers,
  `FiLM1d` depth conditioning, **independent depth heads** (not §18.4's star residual), returns
  `(B,K,n_depths)`. Statics enter as a **prefix**.
- Cache contract: `k_proj`/`v_proj` are called **once per sample** outside the patch loop, producing
  `[(Kc_l, Vc_l)] * 6`. The cache is indexed by **day, not patch** — a wrong `(D,P)` view makes every
  patch read another day's weather with no crash.
- Construction gated on `arch` so the U-Net decoder is never built under patchwise; otherwise
  `train.py:902` (`find_unused_parameters` unset, default `False`) raises on step 1 of every DDP run.

**Step 3 — `train.py`.**
`--arch {unet,patchwise}`, `--token-sel {station,all}`, `--driver-mode {memory,concat}`,
`--driver-layers`, all into **`CONFIG`** (`:176-220`) so they reach all three checkpoint saves
(`:1088`, `:1132`, `:1314`). Pass `token_sel`/`patch_token_dropout` through `common_kwargs`
(`:835-841`). Stamp `git rev-parse HEAD` into `CONFIG`. Force `use_cls_depth` under patchwise. Loss
gathers over `token_idx` rather than `sm_map[:,:,112,112]` — an `arch`-aware branch inside
`masked_huber_loss`, not a second loss function. Guard `lambda_boundary` (its `.mean()` renormalises
by ~50,176x off a 224² map) and make `total_variation_loss` raise a clear error. Add `arch` /
`driver_mode` to the config echo (`:910-915`).

**Step 4 — the silent-failure fixes, all verified against source.**
- `ckpt_utils.py:38-48` — read `arch` from `cfg`, construct the right class, `strict=True` for
  patchwise. Today `strict=False` yields a randomly-initialised U-Net that runs and prints plausible
  numbers. Then the three divergent copies: `demo_plot.py:44-56`,
  `plot_satellite_sm_meeting.py:47-61`, `eval_stations.py:60-65`.
- `ablation.py:28-38` — `MODALITY_KEYS` gains `s2_hist`/`s2_hist_valid`, `s1_hist`/`s1_hist_valid`,
  `dem_tok`, `lulc_tok`; replace the `if k in d` guard (`:144`) with a hard error. Today
  `--ablate sat` on patchwise ablates **nothing** and still reports a healthy donor fraction.
- `eval_predict.py` — recover `arch` from the checkpoint `cfg`, arch-aware readout, reject
  `--pixel-csv` under patchwise, cap `--batch-size` when `token_sel="all"`.

**Step 5 — tests.** Delete `test_gather_equiv.py` (imports `gather_l12_from_shm`, gone). Keep
`test_pyramid_equiv.py` as the unet guard. Make `test_per_depth_loss.py` arch-aware (`:196-198`
assert `BatchNorm2d`, decoder-only). New `test_patchwise_model.py`: shapes; `memory` and `concat`
agree on output shape; **the cache is indexed by day** (shuffle the day axis and assert predictions
change); no parameter left ungradiented.

**Step 6 — smoke on 2 GPUs, not 1.** The DDP unused-parameter failure is invisible on one device.

**Step 7 — arms (§35.9), one calendar slot.** full / history-ablated / re-pooled / statics-only /
patch-shuffle. `--driver-mode concat` and `--n-layers 4` are later ablations, not run 1.

**Step 8 — pre-register the §35.10 gate before submitting run 1.**

**Deferred, in order:** multi-station supervision (§35.19 — the only direct supervision of the
spatial mapping that exists); position-code probe; anchor-redundancy check; the `/dev/shm` preload
shrink (§35.16, sized by the probe).

### 35.21 STAGE 2a BUILT AND SMOKED (2026-08-26) — what was done and what it cost

Branch `feat/patchwise-temporal`. Smoke job **26070202**, 2 GPUs, COMPLETED.

**Built.** `model.py`: `FiLM1d`, `PatchwiseBlock`, `_row_entropy`, `_build_driver_tokens` (T1's 431
tokens, copied verbatim from the existing ERA5/SIF/TWSA construction so the arm stays comparable),
`_build_patch_seq` (T2's 106), `_forward_patchwise`. Cross-attention is hand-written with explicit
`q/k/v/o` + `F.scaled_dot_product_attention`; `k_proj`/`v_proj` are called by the parent **once per
sample**, outside the patch loop. Queries fold to `(B, h, K*L, dh)` rather than expanding the memory
to `(B*K, ...)` — exact, and it avoids ~2 GB at K=196. Construction is gated on `arch`, so
`decoder`, `transformer_layers`, `scale_emb` and the three spatial embeddings do not exist under
patchwise (required: `find_unused_parameters` is unset).

`train.py`: `--arch`, `--driver-mode`, `--driver-layers`, `--token-sel`, `--patch-token-dropout`,
all into `CONFIG`; `token_sel` finally passed through `common_kwargs`; **`git rev-parse HEAD` and a
dirty flag stamped into every checkpoint** (§35.13's open gap, now closed). Under patchwise,
`use_cls_depth` is forced on, `token_sel` defaults to `"station"`, and `lambda_tv`/`lambda_boundary`
are forced to 0 **with a printed override** — `lambda_boundary` was live at 0.1 and its `.mean()`
would have renormalised by ~50,176x.

`dataset.py`: `anchor_l3/l6/l9/l12` dropped on the patchwise path (~1.15 MB/sample of dead IPC).

**The loss got simpler, not more complex.** `sm_map[:, :, 112, 112]` existed only because the U-Net
emits a map. Patchwise emits `(B, K, n_depths)` where the value IS the prediction, so the branch is
`sm_map.ndim == 3 -> mu[:, 0, :]` and everything downstream (`per_depth`, `return_breakdown`, the
depth accumulators) is reused untouched. There is **no gather over `token_idx`** — an earlier draft
said there would be; `token_idx` is needed only at inference.

**Three silent-failure fixes.** `ckpt_utils` is arch-aware, `strict=True` for patchwise, and raises
if patchwise keys appear as "unexpected" — previously a patchwise checkpoint loaded as a
**randomly-initialised U-Net that ran and printed plausible numbers**, and every eval script funnels
through that one function. The three divergent copies (`demo_plot.py`, `plot_satellite_sm_meeting.py`,
`eval_stations.py`) now delegate to it or carry the full kwargs, and the two plotting ones refuse a
patchwise checkpoint rather than rendering a wrong map. `ablation.py` raises when a modality matches
no key instead of reporting a healthy donor fraction for an ablation that never happened — with the
pooled keys popped, `--ablate sat` on patchwise was a **total no-op**, which would have made every
§35.9 arm return "no effect".

**Measured, not estimated.**

| run | params | note |
|---|---|---|
| `--arch unet` (smoke, `use_cls_depth=False`) | 50,050,944 | baseline unmoved, val 0.0058 -> 0.0051 |
| `patchwise memory --n-layers 6 --driver-layers 2` | **75,526,208** | 1.50x |
| `patchwise concat --n-layers 6 --driver-layers 2` | 61,342,784 | 1.22x |

The 75.5 M is 3.5 M above the 72.0 M estimate in §35.19: the estimate omitted the depth heads, and
`FiLM1d(768, 768)` is a `Linear(768, 1536)` = 1.18 M, three of them. **`concat` is 14.2 M smaller
than `memory`** because it has no cross-attention block (4 x d^2 x 6 layers) — a real confound that
must be reported alongside the `--driver-mode` arm.

**Smoke results, 3 stations / 2 epochs / 2 GPUs.** All three passed. Patchwise peak VRAM 2.4 GB
against unet's 3.6 GB, and **`data=2-4 s` against `compute=1 s`** — already dataloader-bound at
smoke scale, exactly as §35.16 predicted, which is what sizes the next optimisation. Checkpoint
round-trip verified: a patchwise checkpoint rebuilds the patchwise class, and forcing `arch=unet`
on it now **raises** instead of returning a random model.

**Tests.** `test_gather_equiv.py` deleted (imported `gather_l12_from_shm`, removed long ago). New
`test_patchwise_model.py`, 30 checks, all passing — including the two that would otherwise be
silent: permuting the weather across the batch **must** change predictions (the cache is indexed by
sample, not patch) and every parameter must receive a gradient (the DDP precondition). It is
deliberately built at `d_model=64, n_heads=4, n_layers=2`: every property tested is width- and
depth-independent, and at 768/12/6 the CPU forward+backward took minutes, which broke the standing
rule that nothing heavy runs on a login node.

**Two bugs of my own, found and fixed during the build.** (1) `collect_entropy` was armed before
validation and never disarmed, so every subsequent *training* epoch would have kept collecting
attention weights and given up SDPA. (2) `slurm/smoke_patchwise.sh` cleared `checkpoints/smoke_*`
via a **relative** path, but checkpoints live under `CONFIG["checkpoint_dir"]` — so the `rm` matched
nothing, and since `train.py` auto-resumes from `last.pt` a second smoke run would have resumed the
first one's weights while looking like a clean pass.

**CHECKPOINTS HAVE NO BACKUP — verified, and it is not what it looks like.**
`/projects/prjs1968/checkpoints` and `/gpfs/work3/0/prjs1968/checkpoints` are **the same directory**
(device 49, inode 1561052160), exactly the trap the zarr store had. `cls_depth_star_reg/best.pt` and
`last.pt` are byte-identical but sit in one directory on one filesystem, so that is not redundancy
either. The rollback for *code* is the tag `baseline-unet-temporal`; the comparison baseline's 604 MB
of unique weights exist **exactly once**. §35.13 flagged this in words; it is still open.

**Next:** §35.9's arms and the §35.10 gate, pre-registered before run 1.

### 35.22 PATCHWISE-ONLY REFACTOR — the live modules stop carrying two architectures (2026-08-26)

**User decision.** `model.py`, `dataset.py`, `train.py` become patchwise-only: no `arch` flag, no
branches, no dead paths. The baseline is preserved as a **frozen snapshot** instead of as a second
code path. Baseline comparability is explicitly not a design constraint — *"we do not have to look
for baseline, if we have to we think about it then."*

**Why this is the right call, on the session's own evidence.** Every one of the four
silent-failure bugs found while building §35.21 was the same shape — *the other path's assumption
leaked*: `ckpt_utils`'s `strict=False` (written for one 2026-06 unet checkpoint) turning a
patchwise checkpoint into a randomly-initialised U-Net that ran; `ablation.py`'s `if k in d`
(written for pooled keys) making `--ablate sat` a total no-op; two plot scripts assuming a 224²
map. Branching is where the bugs live. It is also where the I/O goes: the wide read exists only to
keep the pooled path alive.

**The snapshot** — four self-contained files, FROZEN, never edited again:

```
model_unet.py       git show HEAD:model.py
dataset_unet.py     git show d4470d7~1:dataset.py     <- HEAD already carries the patchwise work
train_unet.py       git show HEAD:train.py
ckpt_utils_unet.py  git show HEAD:ckpt_utils.py
```

Imports repointed at each other so the set does not depend on the live modules. Tag
`baseline-unet-temporal` still holds the same code state; this is the same thing addressable
without a checkout.

**Deleted from the live modules.** `model.py`: `UNetDecoder`, `_ConvBlock`, `FiLMLayer` (4-D,
decoder-only — `FiLM1d` replaces it), `spatial_pyramid_pool`, `_pyramid_from_l12`,
`_static_pyramid`, `_get_target_spatial_tokens`, `_get_skip_connections`, `_build_sequence`,
`total_variation_loss`, `STATION_ROW`/`STATION_COL`, and the unet-only embeddings
(`spatial_row/col_emb`, `spatial_modality_emb`, `scale_emb`, `transformer_layers`,
`transformer_norm`). `masked_huber_loss` loses the station-pixel arguments entirely: it takes
`(B, K, n_depths)` and nothing else. `train.py`: `--arch`, `--use-memmap`, `--lambda-tv`,
`--lambda-boundary` and both regularisers — each is defined on a 224² map that no longer exists.

**Two consequences that were not obvious until the deletions were traced.**

1. **The I/O fix stops being a special case.** With no wide path to preserve, `zeros(max_acq, K,
   768)` and `tokens_z[src_i, sel, :]` are simply what the loader does — **1.5 KB per acquisition
   (one memmap page) instead of 294 KB (72 pages)**, and the slice at `dataset.py:266` becomes the
   identity. No `if token_sel is None` guard needed anywhere.
2. **The entire `.npy` memmap machinery retires with the anchor.** `_l369_cache`, `use_mmap`, the
   `{orbit}_{layer}.npy` loading at `dataset.py:928-946` exist **only** to serve the anchor
   L3/L6/L9 reads, and `--arch patchwise` STEP 1 has no decoder and therefore no anchor. That also
   retires `--use-memmap` from `slurm/train.sh:65` and the `ulimit -n 65536` its comment justifies,
   and means `convert_l369_to_npy.py`'s output is no longer needed for training.

**One behaviour change, recorded.** `torch.isnan(tok).any()` currently sees the whole tile and
skips an acquisition if ANY of the 196 patches is NaN. Narrowed to patch k it sees only that patch.
That is the more defensible rule for a patchwise model — an acquisition should not be discarded
because a far corner of the tile is bad — but it IS a change, and `test_patchwise_dataset.py` pins
the old one.

**RUN: `memory` ONLY.** `--driver-mode concat` stays in the code and is **not submitted**. It costs
one branch in `PatchwiseBlock` and ~30 lines, and it is the one choice that cannot be retrofitted
without a retrain: under `concat` the driver tokens are updated by each patch from layer 1, so
there is no cache to add later (`text/patchwise_math.md` §3.4, worked numerically at d=2). Keeping
the flag makes "was it the read-only restriction?" a flag rather than a rewrite. An option held
open, not an experiment being run.

Recorded for whenever `concat` IS run: it is **14.2 M smaller** (no cross-attention block,
4.d^2 x 6 layers), so a `concat` loss could be capacity rather than wiring and a
`concat --n-layers 8` arm (~72 M) would be needed to separate them; and it is `4.24e9` MAC/layer
against `memory`'s `0.963e9`, which starts to matter once the epoch is no longer dataloader-bound.

**Verification that is not optional:** the frozen snapshot must still import and still load
`cls_depth_star_reg/best.pt`. It is the only thing standing between us and being unable to touch
the baseline at all — and per §35.21 those weights exist exactly once, with no backup.

### 35.23 REFACTOR VERIFIED (2026-08-26) — and the epoch is no longer dataloader-bound

Job **26071036**, 2 GPUs, COMPLETED, `ALL VERIFICATION PASSED`.

| # | check | result |
|---|---|---|
| 1 | unit tests (`test_patchwise_model`, `test_patchwise_dataset`) | no failures |
| 2 | **frozen snapshot still loads the baseline** | imports OK; `cls_depth_star_reg` loads, epoch 16, 50,348,544 params, decoder present |
| 3 | no unet remnants by grep in the live modules | clean |
| 4 | smoke, both driver modes, 2 GPUs | `memory` 71,981,888 params; `concat` 57,798,464 |
| 5 | per-sample IPC payload | 635.2 KB, no pooled or anchor keys |

Check 2 is the one that mattered: §35.21 established those baseline weights exist exactly once
with no backup, and after this refactor the `_unet` snapshot is the only code that can read them.

**THE MEASURED WIN — the epoch flipped from dataloader-bound to compute-bound.** Same smoke, same
3 stations, before and after the refactor:

```
before   data=4s   compute=1s     gpu_util=27%
after    data=0s   compute=1-2s   gpu_util=93-95%
```

That is what the narrow read bought: 1.5 KB per acquisition (one memmap page) instead of 294 KB
(72 pages), plus the anchor drop. §35.16 predicted the epoch would become almost entirely
dataloader-bound under patchwise and that the data path was where the time was; it was, and it has
now moved. **Caveat: 3 stations and 20 batches does not extrapolate to 993 stations**, where the
`/dev/shm` preload dominates — treat this as direction, not magnitude.

**Two consequences to carry forward.**

1. **`concat`'s 4.4x compute is now a real cost.** While the epoch was dataloader-bound the extra
   `4.24e9` vs `0.963e9` MAC/layer was nearly free. It is not any more. Add that to the §35.22
   list of things to weigh whenever that arm is run.
2. **`soil_patch` is now the dominant payload** — 460 KB of the 635 KB total; everything satellite
   is down to ~153 KB. §35.3 records that per-patch soil at 14x14 would be 16 KB against 460 KB
   (29x smaller) and that `SoilEncoder`'s four concentric centre-symmetric means keep the exact
   defect §35 exists to remove. That is the next target after this.

**FiLM AND THE PATCH CLS ARE GONE — the depth heads now predict directly (user decision).**

```
was   out_i = head_i( FiLM1d_i(h, depth_ctx[:, i, :]) )     h = a shared patch CLS row
now   out_i = head_i( depth_ctx[:, i, :] )                  each depth CLS IS its readout
```

FiLM earned its place in the U-Net by broadcasting one context vector across a `(B,C,H,W)` map;
here both operands are `(N,768)` vectors from the same transformer, so modulation bought nothing a
direct readout does not have. It also cost 3 x 1.18 M parameters and, being identity-initialised,
started all three depths reading the **identical** vector — a soft echo of the §18.4 star residual
the design had explicitly rejected. Each depth CLS is already a full readout over all tokens with
its own learned query, so the separate patch CLS was strictly redundant and went too.

```
sequence   106 -> 105 tokens   [ depth_CLS x3 | dem_k | lulc_k | hist_k x100 ]
params     75,526,208 -> 71,981,888     (-3,543,552 FiLM, -768 patch_cls)
```

which lands on the 72.0 M §35.19 first estimated, before the depth heads were counted.

Consequence handled: `_row_entropy` was measuring the patch CLS row, which no longer exists. It now
averages the **depth CLS** rows — arguably what the collapse detector wanted all along, since those
are the rows whose attention over the history is the thing at risk.

**L12 ONLY.** Nothing in `model.py` / `dataset.py` / `train.py` touches L3/L6/L9 any more — they
existed solely for the U-Net skip connections. `convert_l369_to_npy.py`'s ~7,800 `.npy` files are
no longer needed for training, and `--use-memmap` plus its `ulimit -n 65536` are obsolete. One
latent crash was caught on the way: `train.py`'s `_advise_l369_willneed` still read
`ds._l369_cache` after the cache was deleted, and would have raised at epoch 1.

**`skt` KEPT, deliberately.** Dropping ERA5 skin temperature "because ECOSTRESS gives it later"
confuses two roles: `skt` is an **input driver** (9 km, daily, complete), while ECOSTRESS LST is a
**supervision target** in §34.6 step 3 / §33.7. The replacement is also not in hand — §29 measured
Landsat LST against SM at within-station r = -0.077 and ECOSTRESS DTR has not passed §33.9 gate 6.
It stays as a stage-2c ablation (§35.8), not a refactor casualty.

**Three bugs of mine that the first verification run caught**, all silent-shaped: `CONFIG["token_sel"]`
left at `None` after the arch-fixup block was removed (crashed both smokes); a stale comment
mentioning a deleted function tripping the remnant grep; and `test_patchwise_dataset.py` still
asserting the old `_finalise_history` contract, which used to slice and now receives an
already-narrowed buffer. The test now also pins the condition that matters most for §35.22: **the
narrow read must return exactly what the wide read would have** — if those ever diverge, every
number is computed on the wrong patch and nothing crashes.

**PUSHED FOR REVIEW, NOT SUBMITTED.** `feat/patchwise-temporal` on
`github.com/prajzwal08/soilMoisture`:

```
1dc43e7  Freeze the pooled U-Net baseline as a self-contained snapshot
fe0dc2c  Patchwise-only: strip the U-Net path, narrow the read, drop FiLM
```

Split deliberately: the first commit is additions only and can be reviewed on its own; the second
is the whole refactor. `text/patchwise_math.md` and `test_patchwise_model.py` were force-added —
`.gitignore` has `*.md` and `test_*.py`, the same trap `csvs/dem_regions.csv` hit in §32.9.

`slurm/train.sh` no longer passes `--use-memmap` (the flag no longer exists in `train.py`, so it
would have failed at argparse). The submission line is:

```
sbatch slurm/train.sh --driver-mode memory --driver-layers 2 --n-layers 6 \
                      --run-name patchwise_2a
```

**Two things must happen before that result is looked at.**

1. **Review the diff.** The two places where a mistake is silent rather than loud:
   `model.py _forward_patchwise` — the `kv = [(blk.k_proj(m), blk.v_proj(m)) ...]` line and how
   `kc`/`vc` reach the block; a wrong `(D,P)` view makes every patch read another sample's weather
   with no error. And `dataset.py _read_patch_tokens` — if the narrow and wide reads ever diverge,
   every number is computed on the wrong patch and nothing crashes. Both are pinned by tests, but
   both are worth human eyes.
2. **PRE-REGISTER the §35.10 gate**, before run 1 finishes rather than after. Choosing the
   threshold with the answer in view is precisely what §28.8's gate did wrong. Primary endpoint is
   the **within-station** criterion — de-mean prediction and observation by station, then score the
   residuals — with CIs bootstrapped over **stations**, not samples, and spread and correlation
   gated jointly. Log temporal attention entropy per layer: if it sits at the uniform value
   (log 100 = 4.605 nats) the run is uninterpretable whatever its loss.

`concat` remains an unexercised flag (§35.22).

**Deferred, in priority order:** back up `cls_depth_star_reg` (604 MB, exists once, no copy);
`/dev/shm` preload 145 GB -> ~0.74 GB (valid only while `k` is fixed, i.e. while translation
augmentation stays undecided); per-patch soil (460 KB -> 16 KB, and it removes the concentric-mean
defect §35.3 names); multi-station supervision via `location_group_id`, which
`csvs/station_splits.csv` has had all along (906 groups for 993 stations) and `dataset.py` has
never read — §35.19 argues it is the only direct supervision of the spatial mapping that exists.

---

## §35.24 Two-round audit of dataset.py / model.py / train.py — 30 defects fixed, none of which crashed (Session 33, 2026-08-26)

**Nothing in this section has been executed.** Every fix was made by reading, per the
no-login-node rule. The state is "carefully reviewed, unverified"; the first real check is
`sbatch slurm/run_tests.sh`. Rollback tag: `pre-s33-audit-fixes`.

Two independent audits were run against `text/patchwise_math.md`: four agents over the three
files plus their interfaces, and a second external review. The **design survived** — memory vs
concat, the read-only K/V cache, the two-transformer shape and the §6 cost argument are all as
derived. What did not survive was the implementation. Not one of the 30 defects would have
raised an exception.

### The three classes

**1. Fail-open data paths.** Four separate inputs defaulted a missing value to "everything is
valid", and the effect of each was to fabricate ground truth:

```
missing cloud-mask group  ->  all 60 S2 acquisitions marked cloud-free
missing S1 token_mask     ->  border / shadow / layover patches marked valid
missing QC variable       ->  gap-filled climatology trained as observed truth
missing DEM nodata mask   ->  an all-zero token read as a real elevation embedding
all-NaN soil channel      ->  NaN through SoilEncoder -> Kc/Vc -> the whole cross-attention
```

All are now fail-closed **and counted and printed**. That second half is the point: after §32's
corrupt-store incident, a run that silently trains on a degraded subset is indistinguishable
from a healthy one. `create_token_zarr.py` now writes `QC_NO_SOURCE = 255` instead of defaulting
QC to zeros, and the dataset drops stations whose QC is entirely sentinel.

**2. Diagnostics that could not see what they were built to see.** §35.20 makes attention
entropy the *sole* evidence separating "un-pooling does not help" from "attention collapsed",
and stakes the deferral of register standardisation on it. As written it:

- head-averaged before computing entropy — twelve sharply-peaked heads average to something
  numerically indistinguishable from uniform;
- spanned the five non-history prefix columns as well as the history;
- was compared against a fixed `log(100) = 4.605` although the median station-year carries ~36
  of 60 S2 slots, so a **fully collapsed** row over ~50 valid keys scored ~4.0 and passed;
- was overwritten every forward, so what reached W&B was one batch on rank 0.

The companion depth-collapse metric (`_last_depth_ctx`) had been reading a `getattr` default
since the U-Net strip in `fe0dc2c` and had logged **nothing at all** — while the surviving
fallback measured the input cosine, which the code's own comment says is the insufficient one,
and which is weight-decay-protected so it will look healthy indefinitely.

Both are rebuilt as epoch-accumulated, `all_reduce(SUM)`-ed sums with a per-sample
`log(n_valid)` reference. **This supersedes the "log 100 = 4.605 nats" instruction at the end of
§35.23**: the reported quantity is now a scale-free ratio where **1.0 means collapsed**,
whatever that sample's valid-slot count. The history slice is bounded at `hist_end` so concat
mode does not fold its 431 driver columns into the history entropy — without that the two
`--driver-mode` arms' numbers are not comparable, which is the one comparison the arms exist for.

**3. Scale and initialisation.** `nn.Embedding` defaults to `normal_(0, 1)` while `era5_mlp`
emits sigma ~0.22, so the driver token entering T1 was ~98% calendar and ~2% weather, with
`driver_norm` then normalising that mixture. All seven tables are now `trunc_normal_(0.02)` —
the ViT/BERT convention, and what `depth_tokens` already used. That fix is only coherent
alongside input LayerNorms on the frozen TerraMind features (`s2_norm`, `s1_norm`, `dem_norm`,
`lulc_norm`): a positional code at std 0.02 against an unnormalised, register-dominated frozen
feature is invisible, which would have traded the driver-token problem for the same problem on
the history.

### The one that would have been hardest to find

ERA5 was the only modality whose staleness came from its **slot index** rather than its real
date, in a window that `load_era5_rolling` *compacts* rather than calendar-aligns. Any interior
missing day shifted every earlier slot's recency code; and because the admission guard was
year-granular, a station whose ERA5 ended 2023-03-15 still admitted an August 2023 target and
placed the March row at slot 364, embedded as "today's weather". `circular_doy_pe` carried the
true DOY throughout, so the two clocks actively disagreed and nothing said so. The dataset now
emits `era5_rel_pos` from real dates and the guard is day-granular.

### Per-file summary

| file | what changed |
|---|---|
| `model.py` | embedding init; DOY code geometric + capped at 26 harmonics (was aliasing above Nyquist k=182) and precomputed as a buffer; input LayerNorms on frozen features; residual + cross-attention dropout in T2; DropPath per-branch rate deflated so effective per-layer drop matches the schedule; T1's dpr scaled against T2's depth; `era5_rel_pos`; 3-way `hist_modality_emb` (S2 / S1-ASC / S1-DESC); `dem_valid`/`lulc_valid` in the static pad; T1 no longer built in concat mode; `head_bias_init`; fp32 readout; device from parameters, not from the batch; `per_depth` loss switched to fixed `depth_weights` |
| `dataset.py` | the five fail-closed paths above, each counted; `era5_rel_pos`; `s1_orbit`; SIF/TWSA/soil z-scored from `driver_stats.json`; narrow L12 preload; per-key shm/zarr merge; ERA5 train mask no longer removes tokens from the sequence; S2 history compacts like S1; per-reason skip tally; five dead keys removed; `CM_BAD_CLASSES` documented |
| `train.py` | selection and LR schedule off the mean-of-batch-means, default `--select-metric ubrmse`; linear warmup, resume-correct on `global_step`; every `nn.Embedding` excluded from weight decay; `head_bias_init` + `depth_weights` wired; both collapse diagnostics rebuilt and reduced; shm preload gated on LOCAL_RANK and skip-check moved above the 120 GB read; `atexit` shm cleanup; preempt flag `all_reduce(MAX)`; RNG state checkpointed; per-station r / R2 / anomaly-RMSE; val shard de-duplicated; `ReduceOp.AVG` -> SUM-then-divide |
| `compute_driver_stats.py` (new), `compute_era5_stats.py` | train-split, train-years-only statistics; **the ERA5 NetCDF source no longer exists** (0 of 842 `sm_only` station dirs still have `ERA5Land/`), so both now read `era5/values` from the token zarr — which also fixes the old `nc_files[0]` partial read at the root |

### Architecture — the finding that matters most

Training runs K=1; inference runs K=196. The 431 driver tokens are tile-constant by
construction — that is what makes the cache exact — so **the loss is fully minimised by a
function that ignores `dem_k`, `lulc_k` and `hist_k` entirely**, and at K=196 that function
emits 196 identical values. Nothing in the loss, the data, or the diagnostics constrained it.

§35.9's five arms and the §35.10 gate were always the test for this. What was missing is that
**there are two orthogonal collapse modes and only one had a detector**:

```
temporal collapse   attention spreads uniformly over the history   -> entropy ratio -> 1.0
spatial invariance  the per-patch pathway is ignored altogether    -> map SD -> 0
```

§35.20 correctly says map SD does not catch attention collapse. The converse is equally true
and had not been written down: a model that ignores per-patch inputs can have perfectly sharp,
low-entropy attention — it simply attends sharply to the weather. Neither detector substitutes
for the other. Both now run once per val epoch, alongside a third: the ratio of gradient norm
w.r.t. the per-patch inputs against the tile-constant ones.

Two further points, neither previously recorded. The **K=196 path had never been executed end
to end** — `masked_huber_loss` raises on K != 1 and `evaluate` hardcodes `mu[:, 0, :]`, so
"same checkpoint works at both" was true of the weights and untested of the code. And at K=1 the
model only ever sees patch 105 in training, so at inference it is fed patches whose statics and
history are **out of distribution** — open water, urban, swath edge. That covariate shift sits
on top of the invariance risk, and it is why the `dem_valid`/`lulc_valid` masks matter far more
at inference than at training.

### Tests

The old suite asserted shapes and no-crash. Every test batch set every validity tensor all-True
and `era5_doys` never 0, so **no test batch contained a single padded token**: deleting a `~`
anywhere in the masking chain passed the entire suite, and so did permuting `depth_heads`. Both
chains were in fact correct; nothing defended them. The new tests are written to fail on the
specific defect — randomised values in masked slots must not change the output; a `[x, nan,
nan]` label must give gradient to `depth_heads[0]` only; the K/V cache must be per day, not per
patch; the entropy ratio must be invariant to valid-slot count.

### New hard dependency

`csvs/driver_stats.json` does not exist yet. **Both `dataset.py` and `train.py` raise without
it**, by design — fail closed rather than silently fall back to identity normalisation and a
zero head bias. Run order is therefore:

```
sbatch slurm/run_tests.sh        # must be green first
sbatch slurm/driver_stats.sh     # writes csvs/era5_stats.json + csvs/driver_stats.json
sbatch slurm/train.sh --driver-mode memory --driver-layers 2 --n-layers 6 --run-name patchwise_2a
```

### Open, deliberately

`CM_MAX_BAD_FRAC = 0.01` over a 16x16 block admits at most **two** bad pixels (3 px = 1.17% and
fails), i.e. it is zero-tolerance rather than the 1% it reads as. Left unchanged and now
documented as deliberate. `era5_require_full_window` defaults False: the trailing-edge date test
is unconditional (that was the bug), but requiring full 365-day coverage would delete the first
year of every station for something that is no longer a correctness issue once `era5_rel_pos`
carries real staleness. The would-be-dropped count prints either way.

Register standardisation stays deferred (§35.20) — but that deferral was only ever defensible
if the detector worked, so it is now contingent on the rebuilt one rather than on the broken
one. `CLAUDE.md` is untracked and matched by `.gitignore`'s `*.md`; it cannot be rolled back.

### Three things writing the tests exposed

Written last, and each is a defect the tests could not have caught by running — they were found
by trying to write an assertion for a contract and discovering there was nothing to assert
against.

1. **`_load_driver_stats` validated three blocks of four.** It checked `sif`, `twsa` and `soil`
   but not `label_mean`, so a `driver_stats.json` carrying the normalisation blocks and nothing
   else would pass the fail-closed gate and then leave `head_bias_init=None` — silently
   reinstating the default-bias-in-Huber's-linear-regime failure the field was added to prevent.
   Now validated in the same place, per depth.

2. **`_last_attn_entropy` was `(n_armed, 3)`, not `(n_layers, 3)`.** `ent` collects only blocks
   with `collect_entropy` set, so arming a subset returned a shorter tensor whose rows train.py
   would have logged under the wrong layer index. `_forward_patchwise` now refuses a partial arm.

3. **`slurm/verify_patchwise_refactor.sh` step 1 had become a no-op that always passed.** It ran
   `python test_patchwise_model.py`; against a pytest module that imports and exits 0 without
   executing a test. Worse, the *old* file would have failed against the current model anyway —
   it asserted `ent.numel() == len(patch_blocks)` and `ent.max() <= log(105)`, both stale against
   the sums contract. Now routed through `python -m pytest`.

`pytest` is not in the built `terramind` env. `slurm/run_tests.sh` pip-installs it on the compute
node and exits 2 with a clear message if that fails; `environment-terramind.yml` now pins it so
the fallback stops being load-bearing.

---

## §35.25 How big are the frozen tokens? — the input LayerNorm, measured rather than assumed (Session 33, 2026-08-26)

§35.24 added an input LayerNorm on the frozen TerraMind L12 features (`model.py` s2_norm /
s1_norm / dem_norm / lulc_norm) on an argument that was never checked: that a positional code at
`EMB_INIT_STD = 0.02` would be invisible against a raw token. `measure_token_scale.py` +
`slurm/token_scale.sh` measure it. 120 stations, 4 acquisitions each, ~25 s.
Result: `csvs/token_scale.json`.

```
modality  per-elem std  (no reg)      L2  tag share  reg share  agree  tile mag  mag-noreg
s2               4.652     3.172   131.8      0.43%      79.0% 100.0%     33.7%       2.7%
s1_asc           4.210     2.765   117.9      0.48%      55.7% 100.0%      1.2%       2.6%
s1_desc          4.250     2.808   119.0      0.47%      52.8% 100.0%      1.3%       2.7%
dem              4.417     3.165   123.4      0.45%      83.4% 100.0%     56.6%       2.9%
lulc             3.998     3.600   111.7      0.50%      67.7% 100.0%     34.0%       2.4%
```

`per-elem std` is the spread across the 768 features inside one token — exactly what LayerNorm
divides by. `tag share` = 0.02 / that, i.e. what an annotation is worth against the token today;
after LayerNorm it is 2.00% by construction. `tile mag` is the share of WITHIN-TILE variance
across the 196 patches carried by token magnitude, which LayerNorm deletes; `mag-noreg` is the
same with the six register coordinates zeroed.

### VERDICT: keep the LayerNorms

The risk that motivated the measurement was that LayerNorm deletes token magnitude, and magnitude
looked like it carried a third of S2's within-tile variance — the very signal §34 exists to find.
**Stripping the registers takes S2 from 33.7% to 2.7% and DEM from 56.6% to 2.9%.** The
across-patch magnitude variation was almost entirely register variation. LayerNorm is deleting
the sink, not the signal: it costs ~2.7% of within-tile content and gains ~4.6x on annotation
visibility. No code change; `model.py` already does this.

### Three things to carry forward

**The justification was right, the numbers were not.** §35.24 asserted the tag would be worth
~0.04% against a raw token. It is 0.43% — ten times better. Quiet, not invisible. The LayerNorm
survives on a thinner margin than was claimed for it, which is worth remembering the next time an
architecture change is argued from an unmeasured order of magnitude.

**One shared register direction, corpus-wide.** `agree = 100%` in every modality: all 120
stations have the same top-1 register coordinate. §35.3 inferred a shared direction; this
measures it on the current tokens. Register share of summed square: DEM 83.4%, S2 79.0%,
LULC 67.7%, S1 ~53-56% — DEM highest, consistent with §27a.3's 13x norm ratio.

**§27a.4's compression question is closed for this architecture.** That analysis worried
LayerNorm would divide informative coordinates by a sink-inflated sigma and crush the content —
but it was reasoning about the POOLED U-Net path, where a pooled vector inherits
`sink_value / n_tokens`. Patchwise does not pool. Measured inflation is
`4.652 / 3.172 = 1.47x`, which is mild. The concern was real for the architecture that has since
been deleted, and is not for this one.

This does not settle register standardisation (§35.20), which stays deferred — it characterises
the registers better and shows that per-token LayerNorm already neutralises their effect on
within-tile magnitude. It says nothing about whether they still flatten q.k, which remains the
rebuilt entropy detector's job.

### Two defects in the measurement script itself, both fail-quiet

Worth recording because they are the same pattern §35.24 spent the day removing. Run 1 looked for
`dem/l12` and `lulc/l12`; the statics are top-level `(196, 768)` arrays, so **DEM and LULC were
simply absent from the output table rather than flagged**. And the register-stripped statistic —
the one the verdict turns on — was promised in the script's docstring and never implemented. Both
fixed before the reported run. A third, louder failure (an invented `station` column in
`station_splits.csv`) killed run 0 in 19 s, which is the failure mode one wants.

---

## §35.26 One scale cannot serve two streams — splitting the staleness tables, and taking the input LayerNorm back out (Session 33, 2026-08-26)

This **supersedes part of §35.24**. That section set all seven `nn.Embedding` tables to
`trunc_normal_(0.02)` and called it a fix. It was half a fix, and the other half caused the
next two problems.

### The measurement that was missing when §35.24 was written

`measure_token_scale.py` gained a temporal axis after §35.25 was committed, so §35.25's table
carries only the spatial columns. The full result:

```
modality  per-elem std  (no reg)      L2  tag share  reg share  agree  tile mag  mag-noreg  TIME mag   noreg
s2               4.644     3.171   131.4      0.43%      77.1% 100.0%     29.4%       2.7%      9.6%    9.3%
s1_asc           4.218     2.772   118.1      0.47%      53.2% 100.0%      1.2%       2.6%      0.9%    2.5%
s1_desc          4.245     2.807   118.8      0.47%      55.7% 100.0%      1.4%       2.8%      0.9%    2.4%
dem              4.417     3.165   123.4      0.45%      83.4% 100.0%     56.6%       2.9%       n/a     n/a
lulc             3.998     3.600   111.7      0.50%      67.7% 100.0%     34.0%       2.4%       n/a     n/a
```

`tile mag` is the share of variance ACROSS THE 196 PATCHES carried by token magnitude;
`TIME mag` is the share across ACQUISITIONS at the station token. `noreg` is the same with the
six register coordinates zeroed.

**The asymmetry is the whole argument.** Spatially, magnitude collapses to ~2.7% once the
registers are stripped — so what an input LayerNorm deletes there is sink, and deleting it is
a gain. Temporally it does **not** collapse: 9.6% -> 9.3%. That is real content, on the axis
where wetness would live (wet soil is darker in SWIR). An input LayerNorm deletes it.

### The actual bug §35.24 half-fixed

```
                      content std   annotation std   annotation share
drivers (era5_mlp)          0.22             1.0             ~450%   <- genuinely broken
history (TerraMind L12)     4.65             1.0              ~21%   <- entirely fine
```

`rel_pos_emb` is a **single shared table**: the same 365x768 tensor annotates ERA5/SIF/TWSA
days *and* satellite acquisitions, whose content differs in magnitude by ~21x. One std cannot
serve both. §35.24 set it to 0.02 — correct for the drivers, and it dropped history staleness
from a healthy 21% to 0.43%. The input LayerNorms were then added to bring history back to
2%, and a `log|token|` feature after that to hand back the magnitude the LayerNorm had just
deleted. Three changes, two of them undoing the first.

**Fix the cause.** Two tables, two scales:

```
rel_pos_emb        drivers   EMB_INIT_STD      = 0.02      era5, sif, twsa
rel_pos_emb_hist   history   HIST_EMB_INIT_STD = 1.0       s2, s1
static_modality_emb, hist_modality_emb -> history scale
soil/era5/sif/twsa_modality_emb        -> driver scale
```

Cost: one extra 365x768 table, 280 K parameters, ~0.4% of the model. In exchange the input
LayerNorm becomes unnecessary, the scale feature disappears entirely, and parity with the
frozen pooled baseline (`model_unet.py`, which normalises nothing) is restored.

`use_input_norm` survives as a flag, **default off**, `nn.Identity` when off so no parameter
can leave the DDP graph. Revisit it when a decoder exists: a decoder broadcasts a token over
a 16x16 block, which is precisely where the 97%-register spatial magnitude becomes visible
artefacts. Today there is no decoder and it buys nothing.

### driver_norm now applies in both modes

§35.24 made `driver_enc` **and** `driver_norm` memory-only in one conditional. `driver_enc`
memory-only is right — §4.3's argument holds, concat already has the 431x431 block inside its
joint stack and running T1 there would give it that block twice. But tying the norm to the
same conditional meant the two `--driver-mode` arms differed in **contextualisation** (the
hypothesis) *and* in **normalisation** (an accident of the code), so a difference in result
could not be attributed to either. The norm is now unconditional.

Worth recording what does NOT bite: the softmax budget is safe either way, because
`norm_self` is per-token and normalises all 536 tokens independently before Q/K/V. The
asymmetry lived in the residual stream, not in the attention.

### label_count — epoch 1 no longer trains under a different objective

`train.py` prefers `driver_stats.json["label_count"]` for the fixed inverse-frequency
`depth_weights`, and otherwise freezes epoch 1's own counts — which means epoch 1 runs under
uniform weights, i.e. a different objective from every epoch after it.
`compute_driver_stats.py` now emits per-depth qc==0 counts over the same sample set it uses
for `label_mean`, so the objective is fixed before the first gradient step and cannot change
on requeue.

### The driver row order was documented wrong, and §35.9 was about to walk into it

`_build_driver_tokens` appends **soil first**:

```
soil  [  0,   4)
era5  [  4, 369)
sif   [369, 419)
twsa  [419, 431)
```

Both `model.py`'s docstring and this runbook's §2.3 listing said "era5 365 + sif 50 + twsa 12
+ soil 4". Nothing crashes — `mem_pad` is built in the same order as `toks` — but §35.9's
ablation arms are the next thing to be built, and an arm masking "the ERA5 block" as
`m[:, 0:365]` would silently ablate the four soil tokens plus `era5[0:361]`, and report a
number. Corrected in `model.py` and in `text/patchwise_math.md` §2.3, with the offsets.

### Verification

`slurm/run_tests.sh`: 117 passed / 1 skipped (the skipped one needs the real zarr store and
`csvs/driver_stats.json`, which does not exist until `slurm/driver_stats.sh` has run). Four
new tests pin this section: the per-stream table split, `use_input_norm` off by default with
magnitude surviving, `use_input_norm=True` restoring exact scale-invariance, and `driver_norm`
present in both driver modes.

One test failure in the first run was the right kind — `test_concat_does_not_build_T1`
asserted `driver_norm` is `nn.Identity` in concat mode, which is exactly the behaviour this
section reverses. The assertion was updated, not the code.

**Still unverified.** The suite is synthetic tensors only. `dataset.py`'s fail-closed paths,
the day-granular ERA5 guard, the `/dev/shm` preload, DDP, warmup-on-resume and every §35.24
diagnostic have never run against real data. Order stands: `run_tests.sh` ->
`driver_stats.sh` -> a short `train.sh --max-stations` smoke run -> anything long.

---

## §35.27 The audit fix that cost 62% of the training split, and the stats that came out of fixing it (Session 33, 2026-08-26)

### What happened

`slurm/driver_stats.sh` ran for the first time and reported:

```
stations contributing : 219 / 587
362  labels-qc-length-mismatch
```

That was §35.24 audit item 9. The old `_load_zarr_labels` realigned a length-mismatched
`labels/qc` by taking its trailing `n` columns; the audit called that an unverifiable guess —
if the truncation were at the back instead, every QC flag would be offset and the loader would
train gap-filled days as observed — and made it fatal under `strict`. The audit note asserted
the path was "dead for current stores". It was not dead. It fired for 62% of the train split.

### The trailing slice was correct, and it is verifiable

```
station                        labels/sm  labels/qc  labels/dates   diff
ISMN_AMMA-CATCH_Banizoumbou         1095       1825          1095    730
ISMN_COSMOS-UK_Sheepdrove           2557       3287          2557    730
ISMN_HOBE_1.07                      1144       1612          1144    468
```

`dates` matches `sm` everywhere; only `qc` is long. `station_splits.csv` closes it exactly for
Banizoumbou: `actual_start_date` 2014-01-01, `start_date` 2016-01-01, `end_date` 2018-12-30 —
1825 days untrimmed, 1095 trimmed, difference 730. `trim_pre2016.py` trims `labels/sm` and
`labels/dates` from the FRONT and leaves `labels/qc` at its original length.

So the two directions are different problems and must not share a branch:

```
qc LONGER  than sm   front-trim.  Recoverable, and now VERIFIED by requiring `labels/dates`
                     to be a gapless daily span — that is what makes "same end date, earlier
                     start" imply the trailing n columns align.
qc SHORTER than sm   no alignment exists.  Always fatal, strict or not.
```

`strict` now governs only a genuinely unverifiable case (a non-contiguous date index), which
no current store exhibits.

### Two lessons, both about this session's own method

The audit finding was right in principle and wrong in calibration: an unverified trailing slice
IS a guess, and the code did assert something false about `trim_pre2016.py`. But converting
"unverified" straight to "fatal" without measuring how often it fires is the same error as
arguing from initialisation scales nobody had measured (§35.25, §35.26). Verify first, then
choose the severity.

The reason it surfaced in 30 seconds rather than inside a training run is the OTHER audit fix.
§35.24 required every fail-closed path to be **counted and printed**. The per-reason skip tally
is what turned a silent 62% data loss into a one-line diagnosis. The audit caught its own
regression.

### The stats, finally

```
                       before fix      after fix
ERA5 stations          225 / 587       587 / 587
driver stations        219 / 587       577 / 587
station-days             365,956       1,183,871
label n (0-10)           302,020       1,049,213
```

The 10 stations still out are `no-sample-producing-year-in-2016-2022` — SNOTEL high-altitude
sites and Berlin urban plots with no valid year in the training window. A real filter.

```
SIF     n=  503,270  (572 stations)  mean= 0.477111  std= 0.605864
TWSA    n=   26,229  (507 stations)  mean=-10.536151  std=43.760724
soil    21 channels, 3,159,652 px/channel   (ch03 mean 331.9 std 285.2 in the first run)

LABEL MEAN (m3/m3, qc==0 only — initialises the per-depth head bias)
   0-10     n=1,049,213  mean=0.172005  std=0.115675
   10-30    n=  855,965  mean=0.192178  std=0.114307
   30-100   n=  691,531  mean=0.186037  std=0.119728
```

Two things worth noting. The label means barely moved when the missing 362 stations came back
(0-10: 0.1727 -> 0.1720), so the constants are robust to the sample — but the COUNTS tripled,
and those drive the inverse-frequency `depth_weights`, so the objective would have been wrong.
And the means are ~0.17-0.19, not the ~0.25 assumed throughout §35.24's head-bias reasoning;
the bias init is now measured rather than guessed.

TWSA at -10.5 +/- 43.8 and soil channels in the hundreds, against ERA5 tokens at exactly
N(0,1), is the §35.24 item-7 argument in numbers.

### Open: nothing binds a checkpoint to its stats

`csvs/era5_stats.json` was overwritten in this session (whole-record -> train-years-only, the
OOT-leak fix). Any checkpoint trained against the old file is now silently mismatched: the
normalisation constants are part of the model contract and `train.py` stamps only `git_sha`.
Regenerating the stats after a run degrades `eval_predict.py` with no error. Stamp a hash of
both stats files into the checkpoint and verify it at eval before the first reported run.

---

## §35.29 Tile-sharing stations held out of training — and location_group_id does not catch them all (Session 33, 2026-08-26)

Each station gets its OWN 2240 m tile centred on itself, so "two stations share a tile" means
station B falls inside station A's tile. That splits into two cases with different
consequences, and they must not be treated alike.

```
SAME PATCH   (< 160 m)        both land in the SAME TerraMind token
DIFFERENT PATCH, SAME TILE    (160 - 1120 m)   B is in A's tile, a different token
> 1120 m                      outside each other's tile — ordinary stations
```

**Same-patch pairs are contradictory supervision.** Training on both feeds the model two
different labels for one input and silently double-weights the site. They are also the single
most useful number available for interpreting §35.10: their disagreement is the **irreducible
noise floor** of any 160 m prediction. If two sensors 100 m apart differ by 0.04 m3/m3, no
model at this resolution can beat 0.04. Nothing else in the data gives that number.

**Different-patch pairs are the only DIRECT test of §34's hypothesis.** Predict station A's
tile, read the 160 m patch containing station B, compare against B's observation.
`diag/patch_map_sd` shows the map is not constant; it cannot show the pattern is CORRECT.
These can. 26 usable pairs, almost entirely TxSON and FMI — low power, so a qualitative sign
test rather than a precise metric.

### The finding: location_group_id misses most of them

`csvs/station_splits.csv` has carried `location_group_id` all along and §35.24 verified that no
group straddles a split. But the grouping is not the same relation as "shares a tile", and a
global O(n^2) sweep finds far more:

```
                        via location_group_id      global sweep
same-patch stations                    17                   26
tile-pair stations                     25                   44
train stations affected                 5                   17
```

It missed `Lamont-CF1`, the SMOSMANIA pair (Pezenas, Prades-le-Lez), the Carpeneto STEMS pair,
`TonziRanch`/`US-Ton`, and four further TxSON stations. Anyone reasoning about co-location from
`location_group_id` alone — as §35.19's deferred multi-station-supervision item does — is
working from an undercount.

**No leakage was present.** The sweep checks every pair under 1120 m for a split mismatch and
found none, so despite the incomplete grouping nothing was training on a tile it would later be
evaluated on. Checked rather than assumed.

### Applied

`update_splits_tile_pairs.py` + `slurm/update_splits.sh` (dry run by default, `--apply` writes;
backup at `csvs/station_splits.csv.pre_s3529`). Three new columns — `same_patch_pair`,
`tile_pair_eval`, `duplicate_of` — so evaluation can find these without recomputing geometry.

```
train 612 -> 595   (17 moved to oos, 2.8%)
oos   202 -> 218
val    76 ->  76
duplicate        1
```

Nothing is deleted. The one cross-network duplicate found globally under 50 m —
`VairaRanch` (ISMN's FLUXNET-AMERIFLUX mirror) and `US-Var` (AmeriFlux direct), **6.1 m apart,
the same physical site ingested twice** — is marked `split="duplicate"`, which every existing
`split_filter` excludes without touching the inventory row.

`TonziRanch`/`US-Ton` is the same mirror pattern but sits between 50 and 160 m, so it is not
auto-marked; both are held out as same-patch anyway. The 50 m threshold is deliberate:
`Lamont-CF2`/`US-ARM` at 152 m are genuinely DIFFERENT instruments at the ARM SGP site and
marking one a duplicate would be wrong.

### Consequence: the stats had to be refitted

`csvs/era5_stats.json` and `csvs/driver_stats.json` were fitted on the old train split, which
included the 17 stations now held out. Regenerated: 563/573 stations contributing (was
577/587); label means moved only 0.1720 -> 0.1715 at 0-10 cm, so the constants stay robust to
the sample.

Both hashes changed, which is the §35.28 provenance mechanism proving itself on a real change:

```
era5_stats    7d72c8f82bc7d2f9  ->  572028af6cd66199
driver_stats  0eb7ade028686033  ->  34ade6b95d913b52
```

Any checkpoint trained before this — `smoke`, `smoke2` — now correctly reports a mismatch at
load. That is the intended behaviour, not a fault.
