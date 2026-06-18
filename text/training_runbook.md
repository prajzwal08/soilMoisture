# Soil Moisture Training Runbook

Last updated: 2026-06-18 (Session 16 — §14 spatial resolution roadmap added)  
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
