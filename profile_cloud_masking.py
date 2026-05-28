"""
profile_cloud_masking.py
========================
Profile SEnSeIv2 GPU inference to find the optimal batch size for
cloud_masking_inference.py before the full 1,028-station run.

Measures per batch-size:
  - Throughput (tiles/sec)
  - Peak VRAM (GB)
  - Mean time per tile (ms)
  - GPU utilisation (via nvidia-smi, if available)

Separates I/O time from pure GPU compute time so you know which is
the real bottleneck.

Usage:
    conda activate sensei
    python profile_cloud_masking.py
    python profile_cloud_masking.py --station ISMN_SNOTEL_Truckee#2 --n-tiles 128
    python profile_cloud_masking.py --batch-sizes 1 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import rasterio

from cloud_masking_inference import load_sensei, S2_L2A_DESCRIPTORS, SCRATCH_DIR

# ── helpers ───────────────────────────────────────────────────────────────────

def gpu_memory_gb() -> tuple[float, float]:
    """Returns (allocated_GB, reserved_GB) for CUDA device 0."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved  = torch.cuda.memory_reserved(0) / 1e9
    return allocated, reserved


def nvidia_smi_util() -> str:
    """Return GPU utilisation % and memory from nvidia-smi (best-effort)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=2
        ).decode().strip().split("\n")[0]
        util, mem_used, mem_total = [x.strip() for x in out.split(",")]
        return f"{util}% util  {mem_used}/{mem_total} MB"
    except Exception:
        return "n/a"


def load_tiles(station_dir: Path, n_tiles: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read up to n_tiles S2L2A tiles into RAM. Returns (arrays, nodata_masks)."""
    tifs = sorted((station_dir / "S2L2A").glob("*.tif"))[:n_tiles]
    if not tifs:
        raise FileNotFoundError(f"No S2L2A tiles found in {station_dir / 'S2L2A'}")
    arrays, nodata_masks = [], []
    for p in tifs:
        with rasterio.open(p) as src:
            arr_int = src.read()
        nodata_masks.append((arr_int == 0).all(axis=0))
        arr = arr_int.astype(np.float32)
        arr /= 10_000.0
        arrays.append(arr)
    print(f"  Loaded {len(arrays)} tiles  "
          f"shape={arrays[0].shape}  "
          f"dtype={arrays[0].dtype}")
    return arrays, nodata_masks


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark_io(station_dir: Path, n_tiles: int) -> float:
    """Time raw I/O without any inference. Returns tiles/sec."""
    tifs = sorted((station_dir / "S2L2A").glob("*.tif"))[:n_tiles]
    t0 = time.perf_counter()
    for p in tifs:
        with rasterio.open(p) as src:
            arr_int = src.read()
        arr = arr_int.astype(np.float32)
        arr /= 10_000.0
    elapsed = time.perf_counter() - t0
    return len(tifs) / elapsed


def benchmark_batch_size(
    model, arrays: list[np.ndarray], batch_size: int,
    device: torch.device, n_warmup: int = 2
) -> dict:
    """
    Run inference over all arrays with given batch_size.
    Returns timing and memory stats.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # ── warmup: prevent first-call CUDA JIT overhead skewing results ──────────
    warmup_batch = torch.from_numpy(
        np.stack(arrays[:min(n_warmup, len(arrays))])
    ).to(device)
    with torch.no_grad():
        _ = model.model(warmup_batch,
                        [S2_L2A_DESCRIPTORS] * warmup_batch.shape[0])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # ── timed run ─────────────────────────────────────────────────────────────
    n = len(arrays)
    t0 = time.perf_counter()

    for i in range(0, n, batch_size):
        chunk = arrays[i : i + batch_size]
        batch = torch.from_numpy(np.stack(chunk)).to(device)
        with torch.no_grad():
            preds = model.model(batch, [S2_L2A_DESCRIPTORS] * len(chunk))
        _ = preds.cpu()   # force D2H transfer (same as real pipeline)
        del batch, preds

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "batch_size":    batch_size,
        "n_tiles":       n,
        "elapsed_s":     elapsed,
        "tiles_per_sec": n / elapsed,
        "ms_per_tile":   elapsed / n * 1000,
        "peak_vram_gb":  peak_vram_gb,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station",     default="ISMN_SNOTEL_Truckee#2",
                        help="Station name (must exist in scratch)")
    parser.add_argument("--n-tiles",     type=int, default=128,
                        help="Number of tiles to use (more = more stable estimate)")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="Batch sizes to sweep")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found. Run this on a GPU node.")
        return

    device     = torch.device("cuda")
    device_str = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("=" * 62)
    print(f"  GPU        : {device_str}")
    print(f"  Total VRAM : {total_vram:.1f} GB")
    print(f"  Station    : {args.station}")
    print(f"  Tiles      : {args.n_tiles}")
    print("=" * 62)

    station_dir = args.scratch_dir / args.station
    if not station_dir.exists():
        print(f"ERROR: {station_dir} does not exist.")
        return

    # ── I/O baseline ─────────────────────────────────────────────────────────
    print("\n[1/3] Measuring raw I/O throughput (no inference)...")
    io_tps = benchmark_io(station_dir, args.n_tiles)
    print(f"  I/O ceiling : {io_tps:.1f} tiles/sec  "
          f"({1000/io_tps:.1f} ms/tile)\n")

    # ── load tiles into RAM ───────────────────────────────────────────────────
    print("[2/3] Loading tiles into RAM...")
    arrays, _ = load_tiles(station_dir, args.n_tiles)

    # ── load model ───────────────────────────────────────────────────────────
    print("\n[3/3] Loading SEnSeIv2...")
    model = load_sensei(device="cuda")
    print(f"  Model VRAM (idle) : {gpu_memory_gb()[1]:.2f} GB reserved\n")

    # ── sweep ─────────────────────────────────────────────────────────────────
    print(f"{'Batch':>6}  {'Tiles/s':>9}  {'ms/tile':>8}  "
          f"{'Peak VRAM':>10}  {'vs bs=1':>7}")
    print("-" * 56)

    results = []
    baseline_tps = None

    for bs in args.batch_sizes:
        if bs > len(arrays):
            print(f"  batch_size={bs} > n_tiles={len(arrays)}, skipping.")
            continue
        try:
            r = benchmark_batch_size(model, arrays, bs, device)
            results.append(r)
            if baseline_tps is None:
                baseline_tps = r["tiles_per_sec"]
            speedup = r["tiles_per_sec"] / baseline_tps
            print(f"{r['batch_size']:>6}  "
                  f"{r['tiles_per_sec']:>9.1f}  "
                  f"{r['ms_per_tile']:>8.1f}  "
                  f"{r['peak_vram_gb']:>9.2f}G  "
                  f"{speedup:>6.1f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {'OOM':>9}")
            break

    # ── recommendation ────────────────────────────────────────────────────────
    if results:
        best = max(results, key=lambda r: r["tiles_per_sec"])
        # "knee": last batch size that achieves ≥ 95% of peak throughput
        knee = next(
            (r for r in reversed(results)
             if r["tiles_per_sec"] >= 0.95 * best["tiles_per_sec"]),
            best
        )
        print()
        print("=" * 62)
        print(f"  I/O ceiling        : {io_tps:.1f} tiles/sec")
        print(f"  Peak GPU throughput: {best['tiles_per_sec']:.1f} tiles/sec "
              f"(batch_size={best['batch_size']})")
        if io_tps < best["tiles_per_sec"]:
            print(f"  *** I/O is the bottleneck — GPU is faster than disk can feed it.")
            print(f"  *** Consider pre-loading tiles or using more CPU workers.")
        print(f"\n  Recommended --batch-size : {knee['batch_size']}")
        print(f"  (≥95% of peak throughput, {knee['peak_vram_gb']:.2f} GB VRAM)")
        print("=" * 62)


if __name__ == "__main__":
    main()
