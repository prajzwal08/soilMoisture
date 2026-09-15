"""Parallel L12 → /dev/shm staging, shared by train.py and eval_predict.py (§35.33).

Extracted from train.py so evaluation gets the §35.31 fix it never had. The measured
cost of NOT having it, on eval job 26091958 (2026-08-27):

    VAL split, 74 stations:  ~8 min to build the dataset, ~46 s to run the model.

`SoilMoistureDataset.__init__` falls back to `zg["s2/l12"][:, tsl, :]` per station on one
core when no shm_dir is given. The store is chunked (32, 196, 768), so the token axis is
ONE chunk and reading patch 105 decompresses all 196 — dataset.py's own comment says the
narrowing "buys memory, not startup time". Startup comes from fanning the stations out.

Two things stay exactly as train.py had them, because they are load-bearing:

  * Selection is STRICTLY SERIAL and mirrors the dataset's own cap semantics — a station
    counts against the cap as soon as it is SEEN (passes category + soil_patch_ok), not
    when it is successfully written, because the dataset increments its dict before it
    knows whether the store opened.
  * Caps are PER SPLIT. Walking splits interleaved against one budget silently starved
    val, which then fell back to per-rank GPFS reads while train ran off shm.

The generalisation is only that the split list is now a PARAMETER rather than the
hardcoded [("train", cap), ("val", cap)] — eval needs ("oos", None), ("oot", None) and so
on, over its own EVAL_SPLITS filters.
"""
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


def _category_of(r) -> str:
    """The sat_dir category, resolved exactly as dataset.py resolves it."""
    sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
    fl = str(r.get("has_flux",          "False")).lower() == "true"
    return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")


def _dir_name_of(r) -> str:
    if str(r["source_network"]) == "ISMN":
        return f"ISMN_{r['network']}_{r['station_name']}"
    return f"{r['source_network']}_{r['station_id']}"


def preload_one_station(task):
    """One station, one worker. Returns 1 if any key was staged, else 0.

    Module-level (not a closure) so the fork Pool can pickle it.
    """
    cat, dir_name, shm_dir, tsl = task
    import zarr
    from dataset import ZARR_ROOT

    # Parallelism is at the PROCESS level here, so every nested thread pool is pure
    # oversubscription: blosc defaults to one decompression thread per core, which
    # across 64 forked workers is 64x64 threads fighting for 64 cores. Pin them to 1.
    try:
        import numcodecs.blosc as _blosc
        _blosc.set_nthreads(1)
    except Exception:
        pass

    zarr_path = ZARR_ROOT / cat / dir_name
    if not (zarr_path / ".complete").exists():
        return 0
    try:
        zg = zarr.open_consolidated(str(zarr_path), mode="r")
    except Exception:
        try:
            zg = zarr.open_group(str(zarr_path), mode="r")
        except Exception:
            return 0

    wrote_any = False
    for key in ("s2", "s1_asc", "s1_desc"):
        if f"{key}/l12" not in zg:
            continue
        bin_path  = shm_dir / f"{dir_name}__{key}.bin"
        meta_path = shm_dir / f"{dir_name}__{key}.meta.json"
        # The resume check MUST come before the read.  `zg[...][:]` pulls the whole
        # L12 array off GPFS, and the old order did that read and then discarded it
        # because the .bin already existed.  Every requeue of a --requeue job therefore
        # paid the entire preload (worst observed: 1901 s) to produce nothing.
        if bin_path.exists() and meta_path.exists():
            wrote_any = True
            continue
        # NARROWED (§35.31): one of 196 token columns, not the full (N,196,768) slab.
        # That was 153.6 GB of tmpfs across 647 stations for data of which exactly one
        # column is read. tmpfs is RESIDENT ram, so full width was never free.
        arr = (zg[f"{key}/l12"][:, tsl, :] if tsl is not None
               else zg[f"{key}/l12"][:])
        mm = np.memmap(bin_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        del mm                       # flush to tmpfs
        # `narrowed` travels WITH the array.  A stale full-width bin from an older run
        # carries no such key, defaults False, and is read at full width — widths are
        # resolved per key on the consumer side and never assumed.
        meta_path.write_text(json.dumps({"shape": list(arr.shape),
                                         "dtype": str(arr.dtype),
                                         "narrowed": tsl is not None}))
        wrote_any = True
    return 1 if wrote_any else 0


def preload_l12_to_shm(splits_csv: str, category_filter, shm_dir: Path,
                       split_caps, token_sel: str = "station",
                       workers: int = 64, label: str = "SHM") -> int:
    """Stage L12 tokens for the named splits into `shm_dir`. Returns stations written.

    split_caps: sequence of (split_name, cap) or (iterable_of_split_names, cap).  The
                second form is what eval needs — its OOT split is `["train", "val"]`.
                cap=None means no cap, which is the normal evaluation case.

    Callers must create shm_dir and are responsible for removing it.
    """
    import pandas as pd

    splits = pd.read_csv(splits_csv)
    if category_filter:
        splits = splits[splits.apply(_category_of, axis=1).isin(category_filter)]

    # The token slice, resolved the same way the dataset resolves it so the two can never
    # disagree about which column patch 105 is.  A non-contiguous selection yields None
    # and the preload falls back to full width, which is always safe.
    from dataset import _token_slice, STATION_TOKEN, N_TOKENS
    _sel = (np.array([STATION_TOKEN], dtype=np.int64) if token_sel == "station"
            else np.arange(N_TOKENS, dtype=np.int64))
    tsl  = _token_slice(_sel)

    # ── Phase 1: SELECTION, strictly serial ───────────────────────────────────
    # Pure CSV iteration (fast), and its per-split cap semantics are load-bearing — see
    # the module docstring.  Walking it out of order, or in parallel, would break the
    # "counts as SEEN, not as WRITTEN" property that keeps this in lockstep with
    # SoilMoistureDataset.
    n_seen_total = 0
    targets: list[tuple[str, str]] = []
    staged: set = set()          # across splits: OOS and OOST share every station
    for split_names, cap in split_caps:
        names = [split_names] if isinstance(split_names, str) else list(split_names)
        sub   = splits[splits["split"].isin(names)]
        seen  = set()            # (cat, dir_name) — mirrors the dataset's sat_dir key
        for _, r in sub.iterrows():
            if not bool(r.get("soil_patch_ok", True)):
                continue
            key_seen = (_category_of(r), _dir_name_of(r))
            if key_seen in seen:
                continue         # dataset caches per sat_dir; extra rows are free
            if cap is not None and len(seen) >= cap:
                break            # same break point the dataset takes
            seen.add(key_seen)
            n_seen_total += 1
            # Dedup ACROSS splits too: staging a station twice would re-read it from
            # GPFS for nothing (the per-station resume check would catch it, but only
            # after the Pool had already scheduled the task).
            if key_seen not in staged:
                staged.add(key_seen)
                targets.append(key_seen)

    if not targets:
        print(f"[{label}] nothing to preload for {split_caps}", flush=True)
        return 0

    # ── Phase 2: READ + WRITE, fanned out ─────────────────────────────────────
    # Measured serial: 2733 s for 647 stations, one process averaging 10.6% CPU at
    # 56 MB/s aggregate — blocked on GPFS, not computing.  Stations are independent and
    # nothing here touches CUDA, so a fork Pool is safe as long as the caller runs it
    # before any CUDA context exists.
    workers = max(1, min(workers, len(targets)))
    t0 = time.perf_counter()
    print(f"[{label}] Preloading {len(targets)} stations with {workers} workers "
          f"(token_sel={token_sel!r}, narrowed={tsl is not None}) ...", flush=True)

    tasks = [(cat, dir_name, shm_dir, tsl) for cat, dir_name in targets]
    n_written = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        # imap_unordered so progress reflects completions, not submission order.
        for i, got in enumerate(pool.imap_unordered(preload_one_station, tasks,
                                                    chunksize=1), start=1):
            n_written += got
            if i % 100 == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                print(f"[{label}]   {i}/{len(tasks)} stations  {el:6.1f}s  "
                      f"({i/max(el, 1e-6):.1f} st/s)", flush=True)

    print(f"[{label}] L12 preloaded for {n_written} stations "
          f"({n_seen_total} scanned) → {shm_dir}", flush=True)
    return n_written
