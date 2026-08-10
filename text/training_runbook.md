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

*(to be filled in — per condition and seed: median per-station ubRMSE / bias / r / NSE_anom at
each depth on OOS, Δ from baseline, and the level-vs-climatology skill from
`verify_level_claim.py`.)*
