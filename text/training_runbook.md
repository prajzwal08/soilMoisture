# Soil Moisture Training Runbook

Last updated: 2026-06-17 (Session 11 — CPU pyramid pooling implemented)  
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
