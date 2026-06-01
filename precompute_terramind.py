"""
precompute_terramind.py
=======================
Run TerraMind on every satellite .tif in scratch and save features to
permanent project storage.

Input  (scratch – purged ~14-30 days):
    /gpfs/scratch1/shared/pkhanal/satellite/{station}/

Output (project – permanent):
    /gpfs/work3/0/prjs1968/data/{sm_only|sm_and_flux|flux_only}/{station}/

Saved per temporal acquisition (S2L2A, S1RTC):
    {stem}_L12.pt   (196, 768) fp16   semantic tokens → history bottleneck
    {stem}_L9.pt    (196, 768) fp16   U-Net skip connection
    {stem}_L6.pt    (196, 768) fp16   U-Net skip connection
    {stem}_L3.pt    (196, 768) fp16   U-Net skip connection
    {stem}_geo.json              CRS, affine transform, bounds

Saved per static modality (DEM, LULC) — L12 only, no skips needed:
    dem_L12.pt    (196, 768) fp16
    dem_geo.json
    lulc_L12.pt   (196, 768) fp16
    lulc_geo.json

geo.json schema:
    {
      "crs"      : "EPSG:32631",
      "transform": [a, b, c, d, e, f],   // GDAL-order affine (pixel → world)
      "bounds"   : [west, south, east, north]
    }
    Each of the 196 tokens covers a 16×16 pixel (160 m) block.
    Token (row i, col j) has world-space centre:
        Affine(*geo["transform"]) * (j*16 + 8, i*16 + 8)

Resume-safe: skips any acquisition whose _L12.pt already exists in the output dir.

Usage:
    python precompute_terramind.py                          # all stations
    python precompute_terramind.py --station ISMN_SCAN_Abo  # single station (test)
    python precompute_terramind.py --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
from affine import Affine
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.utils.data
from terratorch import BACKBONE_REGISTRY

sys.path.insert(0, str(Path(__file__).parent))
from dataset import S2_BAND_INDICES

# ── TerraMind encoder ────────────────────────────────────────────────────────

# Z-score normalization stats from v1_pretraining_{mean,std} in terramind_register.py.
# The backbone does not normalize internally; the generation model does (terramind_generation.py:265).
# We must apply the same (value - mean) / std before passing to the backbone.
# LULC: v1_5 stats are mean=0, std=1 → identity, so no normalization needed.
_NORM_MEAN = {
    "S2L2A": [1390.458, 1503.317, 1718.197, 1853.91,  2199.1,   2779.975,
              2987.011, 3083.234, 3132.22,  3162.988, 2424.884, 1857.648],
    "S1RTC": [-10.93, -17.329],
    "DEM"  : [670.665],
}
_NORM_STD = {
    "S2L2A": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
              1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    "S1RTC": [4.391, 4.459],
    "DEM"  : [951.272],
}


class TerraMindEncoder(nn.Module):
    """
    Wraps TerraMind Base and exposes intermediate layer outputs via hooks.
    All outputs are (B, 196, 768).
    """

    HOOK_LAYERS = {"L3": 2, "L6": 5, "L9": 8, "L12": 11}

    MODALITY_MAP = {
        "S2L2A": "untok_sen2l2a@224",
        "S1RTC": "untok_sen1rtc@224",
        "DEM"  : "untok_dem@224",
        "LULC" : "untok_lulc@224",
    }

    def __init__(self, frozen: bool = True, compile: bool = False):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_base",
            pretrained=True,
            modalities=list(self.MODALITY_MAP.values()),
        )
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if compile and hasattr(torch, "compile"):
            print("Compiling backbone with torch.compile(mode='reduce-overhead')...")
            self.backbone = torch.compile(self.backbone, mode="reduce-overhead")
        self._feats   = {}
        self._handles = []
        for name, idx in self.HOOK_LAYERS.items():
            self._handles.append(
                self.backbone.encoder[idx].register_forward_hook(self._make_hook(name))
            )
        # Pre-build normalisation buffers; shape (C,1,1) for broadcast over (B,C,H,W)
        self._norm: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for mod in ("S2L2A", "S1RTC", "DEM"):
            m = torch.tensor(_NORM_MEAN[mod], dtype=torch.float32).view(-1, 1, 1)
            s = torch.tensor(_NORM_STD[mod],  dtype=torch.float32).view(-1, 1, 1)
            self._norm[mod] = (m, s)

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self._feats[name] = output if output.dim() == 3 else output[0]
        return hook

    def forward(self, patch: torch.Tensor, modality: str) -> dict:
        """patch: (B, C, 224, 224) → dict L3/L6/L9/L12 → (B, 196, 768)"""
        if modality in self._norm:
            mean, std = self._norm[modality]
            patch = (patch - mean.to(patch)) / std.to(patch)
        self._feats = {}
        with torch.no_grad():
            self.backbone({self.MODALITY_MAP[modality]: patch})
        return {k: v.clone() for k, v in self._feats.items()}


# ── Performance tracker ───────────────────────────────────────────────────────

class PerfTracker:
    """Accumulates timing and GPU memory stats across batches for trial runs."""

    def __init__(self, device: torch.device):
        self.device   = device
        self.batches  = 0
        self.images   = 0
        self.t_load   = 0.0
        self.t_encode = 0.0
        self.t_save   = 0.0
        self.peak_mb  = 0.0

    def record(self, n_images: int, t_load: float, t_encode: float,
               t_save: float, peak_bytes: int) -> None:
        self.batches  += 1
        self.images   += n_images
        self.t_load   += t_load
        self.t_encode += t_encode
        self.t_save   += t_save
        self.peak_mb   = max(self.peak_mb, peak_bytes / 1e6)

    def report(self) -> str:
        if self.images == 0:
            return "  (no images processed)"
        total = self.t_load + self.t_encode + self.t_save
        img_s = self.images / total if total > 0 else 0
        lines = [
            f"  images      : {self.images}  in {self.batches} batch(es)",
            f"  load        : {self.t_load:.2f}s  ({100*self.t_load/total:.0f}%)",
            f"  encode      : {self.t_encode:.2f}s  ({100*self.t_encode/total:.0f}%)",
            f"  save        : {self.t_save:.2f}s  ({100*self.t_save/total:.0f}%)",
            f"  total       : {total:.2f}s   →  {img_s:.2f} img/s",
            f"  peak GPU mem: {self.peak_mb:.0f} MB",
        ]
        return "\n".join(lines)


# ── Constants ─────────────────────────────────────────────────────────────────

SCRATCH_DIR = Path("/gpfs/scratch1/shared/pkhanal/satellite")
DATA_DIR    = Path("/gpfs/work3/0/prjs1968/data")
BATCH_SIZE  = 8

TEMPORAL_LAYERS = ("L12", "L9", "L6", "L3")
IMAGE_SIZE      = 224   # pixel dimensions of each input patch
TOKEN_SIZE      = 16    # each token covers TOKEN_SIZE × TOKEN_SIZE pixels → 14×14 = 196 tokens
N_SIDE          = IMAGE_SIZE // TOKEN_SIZE   # 14
FILLABLE_THRESH = 0.01  # patches with < 1% nodata pixels are NN-filled; ≥ 1% are left as-is

# ESRI Annual LULC v2 raw values → TerraMind indices (applied after re-download).
# 0=NoData 1=Water 2=Trees 3=FloodedVeg 4=Crops 5=BuiltArea 6=BareGround 7=Snow/Ice 8=Clouds 9=Rangeland
_LULC_REMAP = np.array([0, 1, 2, 9, 3, 4, 9, 5, 6, 7, 8, 9], dtype=np.uint8)


def _remap_lulc(arr: np.ndarray) -> np.ndarray:
    """Remap ESRI Annual LULC v2 raw class values to TerraMind indices (uint8)."""
    raw = np.asarray(arr, dtype=np.int16)
    out = np.where((raw >= 0) & (raw < len(_LULC_REMAP)),
                   _LULC_REMAP[np.clip(raw, 0, len(_LULC_REMAP) - 1)],
                   np.uint8(0))
    return out.astype(np.uint8)


# ── geo helpers ───────────────────────────────────────────────────────────────

def _read_geo(src: rasterio.DatasetReader,
              crop_top: int = 0, crop_left: int = 0) -> dict:
    """Build geo dict from an open rasterio dataset, adjusted for any crop."""
    T = src.transform * Affine.translation(crop_left, crop_top)
    bounds = rasterio.transform.array_bounds(IMAGE_SIZE, IMAGE_SIZE, T)
    return {
        "crs"      : src.crs.to_string(),
        "transform": [T.a, T.b, T.c, T.d, T.e, T.f],
        "bounds"   : list(bounds),      # [west, south, east, north]
    }


def _save_geo(geo: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(geo, indent=2))
    tmp.rename(path)


# ── tif loader ────────────────────────────────────────────────────────────────

def _load_tif(path: Path,
              band_indices: list[int] | None = None,
              do_crop: bool = False,
              modality: str | None = None) -> tuple[torch.Tensor | None, dict | None]:
    """Load .tif → (C, H, W) float32 tensor + geo dict. Returns (None, None) on I/O failure."""
    try:
        with rasterio.open(path) as src:
            nodata = src.nodata
            h, w   = src.shape
            if do_crop and (h < IMAGE_SIZE or w < IMAGE_SIZE):
                print(f"    [warn] {path.name}: image {h}×{w} smaller than IMAGE_SIZE={IMAGE_SIZE}; skipping")
                return None, None
            if not do_crop and (h != IMAGE_SIZE or w != IMAGE_SIZE):
                print(f"    [warn] {path.name}: expected {IMAGE_SIZE}×{IMAGE_SIZE}, got {h}×{w}; skipping")
                return None, None
            crop_top  = (h - IMAGE_SIZE) // 2 if do_crop else 0
            crop_left = (w - IMAGE_SIZE) // 2 if do_crop else 0
            geo = _read_geo(src, crop_top, crop_left)
            if do_crop:
                window = rasterio.windows.Window(crop_left, crop_top, IMAGE_SIZE, IMAGE_SIZE)
                arr = src.read(window=window).astype(np.float32)
            else:
                arr = src.read().astype(np.float32)
    except (OSError, rasterio.errors.RasterioIOError) as exc:
        print(f"    [warn] cannot load {path.name}: {exc}")
        return None, None

    if nodata is not None:
        mask = np.isclose(arr, nodata) if np.issubdtype(arr.dtype, np.floating) else (arr == nodata)
        arr[mask] = 0.0
    if band_indices is not None:
        arr = arr[band_indices]

    if modality == "LULC":
        arr = _remap_lulc(arr).astype(np.float32)

    if modality is not None:
        arr = _nn_fill_and_sanitize(arr, modality)

    return torch.from_numpy(arr), geo


def _nn_fill_and_sanitize(arr: np.ndarray, modality: str) -> np.ndarray:
    """NN-fill nodata pixels in patches with < 1% nodata; replace remaining NaN with 0.

    Per-patch rule (16×16 = 256 pixels):
      0 < nodata_frac < FILLABLE_THRESH  → NN-fill those pixels from nearest valid neighbour
      nodata_frac >= FILLABLE_THRESH     → leave as-is (bad token, masked at training time)

    After patching, any remaining NaN is replaced with 0 so the encoder never receives
    non-finite input (applies to S1RTC / DEM patches with ≥ 1% nodata).

    Nodata convention:
      S2L2A        → pixel == 0  (standard S2 nodata; valid reflectance ≥ 1 post-harmonisation)
      S1RTC / DEM  → np.isnan   (float NaN from orbit gaps / missing elevation)
    """
    if modality == "S2L2A":
        nodata_2d = (arr == 0).any(axis=0)          # (H, W)
    elif modality == "LULC":
        nodata_2d = (arr == 0).all(axis=0)          # valid classes ≥ 10; 0 = background
    else:
        nodata_2d = np.isnan(arr).any(axis=0)

    if not nodata_2d.any():
        return arr  # fast path: no nodata

    out = arr.copy()

    # Identify which nodata pixels belong to fillable patches (0 < frac < 1%)
    fill_2d = np.zeros(nodata_2d.shape, dtype=bool)  # (H, W)
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            rs, re = i * TOKEN_SIZE, (i + 1) * TOKEN_SIZE
            cs, ce = j * TOKEN_SIZE, (j + 1) * TOKEN_SIZE
            patch  = nodata_2d[rs:re, cs:ce]
            frac   = patch.sum() / (TOKEN_SIZE * TOKEN_SIZE)
            if 0 < frac < FILLABLE_THRESH:
                fill_2d[rs:re, cs:ce] = patch

    if fill_2d.any():
        # For each nodata pixel in fill_2d, copy value from nearest valid neighbour
        _, nn_idx = distance_transform_edt(nodata_2d, return_indices=True)
        for c in range(out.shape[0]):
            out[c][fill_2d] = out[c][nn_idx[0][fill_2d], nn_idx[1][fill_2d]]

    # Sanitize remaining NaN → 0 (S1/DEM patches with ≥ 1% nodata)
    if modality != "S2L2A":
        np.nan_to_num(out, nan=0.0, copy=False)

    return out


# ── async data loading ────────────────────────────────────────────────────────

class _TifDataset(torch.utils.data.Dataset):
    """Wraps _load_tif for async prefetch via DataLoader workers."""

    def __init__(self, paths: list[Path],
                 band_indices: list[int] | None,
                 do_crop: bool,
                 modality: str | None = None):
        self.paths        = paths
        self.band_indices = band_indices
        self.do_crop      = do_crop
        self.modality     = modality

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        tensor, geo = _load_tif(self.paths[idx], self.band_indices,
                                self.do_crop, self.modality)
        if tensor is None:
            return torch.empty(0), {}
        return tensor, geo


def _collate_tif(batch: list) -> tuple[list, list]:
    """Return (patches, geos) as plain lists — no stacking."""
    tensors, geos = zip(*batch)
    patches = [t if t.numel() > 0 else None for t in tensors]
    geos    = [g if g else None for g in geos]
    return patches, geos


# ── batch runner ──────────────────────────────────────────────────────────────

def _amp_ctx(device: torch.device):
    """AMP context: fp16 autocast on CUDA, no-op on CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda")


def _save_feats(feats: dict, out_idx: int, src_idx: int,
                src_paths: list[Path], geos: list[dict | None],
                out_dir: Path, layers: tuple[str, ...]) -> None:
    """Save one item's features to disk. Raises ValueError on non-finite tensors."""
    stem = src_paths[src_idx].stem
    for layer in layers:
        t = feats[layer][out_idx]
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {layer} for {stem}")
        tmp = out_dir / f"{stem}_{layer}.tmp"
        torch.save(t.half().cpu(), tmp)
        tmp.rename(out_dir / f"{stem}_{layer}.pt")
    if geos[src_idx] is not None:
        _save_geo(geos[src_idx], out_dir / f"{stem}_geo.json")


def _run_batch(encoder:   TerraMindEncoder,
               patches:   list[torch.Tensor | None],
               geos:      list[dict | None],
               src_paths: list[Path],
               out_dir:   Path,
               modality:  str,
               layers:    tuple[str, ...],
               device:    torch.device,
               failures:  list[str],
               perf:      PerfTracker | None = None,
               dry_run:   bool = False) -> int:
    """Forward a batch through TerraMind, save .pt and _geo.json. Returns count saved."""
    valid_i = [i for i, p in enumerate(patches) if p is not None]
    if not valid_i:
        return 0

    if dry_run:
        shapes = [tuple(patches[i].shape) for i in valid_i]
        print(f"    [dry-run] {modality} batch of {len(valid_i)}: shapes {shapes}")
        return len(valid_i)

    t0_encode = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch = torch.stack([patches[i] for i in valid_i]).float().to(device)
    oom = False
    try:
        with torch.no_grad(), _amp_ctx(device):
            feats = encoder(batch, modality)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
            raise
        oom = True

    t_encode = time.perf_counter() - t0_encode
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    if not oom:
        out_dir.mkdir(parents=True, exist_ok=True)
        t0_save = time.perf_counter()
        saved = 0
        for out_idx, src_idx in enumerate(valid_i):
            try:
                _save_feats(feats, out_idx, src_idx, src_paths, geos, out_dir, layers)
                saved += 1
            except ValueError as exc:
                print(f"    [warn] {exc}")
                failures.append(str(src_paths[src_idx]))
        t_save = time.perf_counter() - t0_save
        if perf is not None:
            perf.record(len(valid_i), 0.0, t_encode, t_save, peak_bytes)
        return saved

    print(f"    [warn] GPU OOM on batch of {len(valid_i)}; retrying one-at-a-time")
    torch.cuda.empty_cache()
    del batch
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for src_idx in valid_i:
        single = patches[src_idx].float().unsqueeze(0).to(device)
        t0_enc1 = time.perf_counter()
        try:
            with torch.no_grad(), _amp_ctx(device):
                sf = encoder(single, modality)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc2:
            print(f"    [warn] OOM on single {src_paths[src_idx].name}: {exc2}")
            failures.append(str(src_paths[src_idx]))
            continue
        t_enc1 = time.perf_counter() - t0_enc1
        pk1 = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        t0_sv1 = time.perf_counter()
        try:
            _save_feats(sf, 0, src_idx, src_paths, geos, out_dir, layers)
            saved += 1
        except ValueError as exc2:
            print(f"    [warn] {exc2}")
            failures.append(str(src_paths[src_idx]))
        if perf is not None:
            perf.record(1, 0.0, t_enc1, time.perf_counter() - t0_sv1, pk1)
    return saved


# ── per-modality processors ───────────────────────────────────────────────────

def process_temporal(encoder:      TerraMindEncoder,
                     src_dir:      Path,
                     out_dir:      Path,
                     modality:     str,
                     device:       torch.device,
                     batch_size:   int,
                     band_indices: list[int] | None = None,
                     do_crop:      bool = False,
                     failures:     list[str] | None = None,
                     perf:         PerfTracker | None = None,
                     dry_run:      bool = False,
                     num_workers:  int = 0) -> int:
    """
    Process all .tif files for S2L2A or S1RTC.
    Saves L3/L6/L9/L12 + geo.json per acquisition to out_dir.
    num_workers > 0 enables async DataLoader prefetch (overlaps I/O with GPU compute).
    Returns count of newly processed acquisitions.
    """
    if failures is None:
        failures = []
    if not src_dir.exists():
        return 0

    pending = [
        f for f in sorted(src_dir.glob("*.tif"))
        if dry_run or not all((out_dir / f"{f.stem}_{lay}.pt").exists()
                              for lay in TEMPORAL_LAYERS)
    ]
    if not pending:
        return 0

    if dry_run:
        pending = pending[:batch_size]

    dataset = _TifDataset(pending, band_indices, do_crop, modality=modality)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = num_workers,
        collate_fn        = _collate_tif,
        persistent_workers= (num_workers > 0),
        pin_memory        = False,  # patches may be None; pin_memory requires uniform tensors
    )

    processed = 0
    for batch_num, (patches, geos) in enumerate(loader):
        batch_start = batch_num * batch_size
        batch_files = pending[batch_start : batch_start + len(patches)]
        n_valid     = sum(1 for p in patches if p is not None)
        t0_load     = time.perf_counter()
        t_load      = time.perf_counter() - t0_load   # ~0: DataLoader prefetched
        saved = _run_batch(
            encoder, patches, geos,
            batch_files, out_dir, modality, TEMPORAL_LAYERS, device, failures, perf,
            dry_run=dry_run,
        )
        if perf is not None and n_valid > 0:
            perf.t_load += t_load
        processed += saved

    return processed


def process_static(encoder:  TerraMindEncoder,
                   src_path: Path,
                   out_dir:  Path,
                   modality: str,
                   out_stem: str,
                   device:   torch.device,
                   do_crop:  bool = False,
                   failures: list[str] | None = None) -> None:
    """
    Process a single static .tif (DEM or LULC). Saves L12 + geo.json only.
    Skip connections (L3/L6/L9) are not needed for static modalities.
    """
    if failures is None:
        failures = []
    out_pt = out_dir / f"{out_stem}_L12.pt"
    if out_pt.exists() or not src_path.exists():
        return

    patch, geo = _load_tif(src_path, do_crop=do_crop, modality=modality)
    if patch is None:
        return

    with torch.no_grad(), _amp_ctx(device):
        feats = encoder(patch.float().unsqueeze(0).to(device), modality)

    t = feats["L12"][0]
    if not torch.isfinite(t).all():
        print(f"    [warn] non-finite values in L12 for {out_stem} ({src_path.name})")
        failures.append(str(src_path))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{out_stem}_L12.tmp"
    torch.save(t.half().cpu(), tmp)
    tmp.rename(out_pt)

    if geo is not None:
        _save_geo(geo, out_dir / f"{out_stem}_geo.json")


def _latest_lulc_tif(lulc_dir: Path) -> Path | None:
    """Return the most recent year-named .tif in a LULC directory (e.g. 2024.tif)."""
    tifs = sorted(lulc_dir.glob("[0-9][0-9][0-9][0-9].tif"))
    return tifs[-1] if tifs else None


def _station_data_dir(station_id: str, data_dir: Path) -> Path | None:
    """Find the permanent data directory for a station across sm_only/sm_and_flux/flux_only."""
    for subfolder in ("sm_only", "sm_and_flux", "flux_only"):
        d = data_dir / subfolder / station_id
        if d.exists():
            return d
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute TerraMind features")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR,
                        help="Root of raw satellite tifs (scratch)")
    parser.add_argument("--data-dir",    type=Path, default=DATA_DIR,
                        help="Root of per-station data folders (sm_only/sm_and_flux/flux_only)")
    parser.add_argument("--batch-size",  type=int,  default=BATCH_SIZE)
    parser.add_argument("--station",     type=str,  default=None,
                        help="Process a single station by name (for testing)")
    parser.add_argument("--start-idx",   type=int,  default=0,
                        help="First station index, inclusive (for SLURM array slicing)")
    parser.add_argument("--end-idx",     type=int,  default=None,
                        help="Last station index, exclusive (for SLURM array slicing)")
    parser.add_argument("--trial",       type=int,  default=None, metavar="N",
                        help="Trial mode: process only first N stations and print detailed "
                             "timing + GPU memory breakdown (default: off)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Load tiles and print shapes without running the encoder. "
                             "Fast CPU smoke test to verify paths and data before GPU submission.")
    parser.add_argument("--num-workers", type=int,  default=0,
                        help="DataLoader workers for async I/O prefetch (default: 0 = sync). "
                             "Set to number of allocated CPUs minus 1 (e.g. 3 for 4 CPUs).")
    parser.add_argument("--compile",     action="store_true",
                        help="Wrap backbone with torch.compile(mode='reduce-overhead') for "
                             "kernel fusion. Adds ~60s warmup on first batch; speeds up all "
                             "subsequent batches by ~10-30%.")
    args = parser.parse_args()

    # Speed up rasterio on GPFS: disable per-file directory scans and enable read cache
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_MAX_RAW_BLOCK_CACHE_SIZE", "200000000")
    os.environ.setdefault("GDAL_SWATH_SIZE", "200000000")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    # Prevent thread thrashing: each DataLoader worker handles one file at a time,
    # so internal thread parallelism inside GDAL/OpenMP/MKL is counterproductive.
    # Without this, num_workers=3 on 4 CPUs can spawn 3×N threads, causing context-switch overhead.
    os.environ.setdefault("OMP_NUM_THREADS",  "1")
    os.environ.setdefault("MKL_NUM_THREADS",  "1")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial   = args.trial
    dry_run = args.dry_run

    print(f"Device      : {device}")
    print(f"Scratch     : {args.scratch_dir}")
    print(f"Data dir    : {args.data_dir}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Num workers : {args.num_workers}")
    print(f"Compile     : {args.compile}")
    if dry_run:
        print("Mode        : DRY-RUN (load + shape check only, no inference)")
    elif trial:
        print(f"Trial mode  : first {trial} stations — detailed timing enabled")
    print()

    if dry_run:
        encoder = None
    else:
        encoder = TerraMindEncoder(frozen=True, compile=args.compile).to(device).eval()

    if args.station:
        station_dirs = [args.scratch_dir / args.station]
    else:
        station_dirs = sorted(d for d in args.scratch_dir.iterdir() if d.is_dir())
        station_dirs = station_dirs[args.start_idx : args.end_idx]

    if trial:
        station_dirs = station_dirs[:trial]

    n = len(station_dirs)
    print(f"Stations to process: {n}\n")

    total_s2 = total_s1 = 0
    skipped: list[str] = []
    failures: list[str] = []

    perf_s2 = PerfTracker(device) if trial else None
    perf_s1 = PerfTracker(device) if trial else None

    for idx, src_station in enumerate(station_dirs, 1):
        t0          = time.time()
        sid         = src_station.name
        out_station = _station_data_dir(sid, args.data_dir)
        if out_station is None:
            print(f"[{idx:4d}/{n}] {sid}  [skip — no data directory found]")
            skipped.append(sid)
            continue

        s2_new = process_temporal(
            encoder,
            src_dir      = src_station / "S2L2A",
            out_dir      = out_station / "S2L2A",
            modality     = "S2L2A",
            device       = device,
            batch_size   = args.batch_size,
            band_indices = S2_BAND_INDICES,
            do_crop      = True,
            failures     = failures,
            perf         = perf_s2,
            dry_run      = dry_run,
            num_workers  = args.num_workers,
        )
        s1_new = process_temporal(
            encoder,
            src_dir     = src_station / "S1RTC",
            out_dir     = out_station / "S1RTC",
            modality    = "S1RTC",
            device      = device,
            batch_size  = args.batch_size,
            do_crop     = False,
            failures    = failures,
            perf        = perf_s1,
            dry_run     = dry_run,
            num_workers = args.num_workers,
        )
        if not dry_run:
            process_static(
                encoder,
                src_path = src_station / "DEM" / "dem.tif",
                out_dir  = out_station / "DEM",
                modality = "DEM",
                out_stem = "dem",
                device   = device,
                do_crop  = True,
                failures = failures,
            )
            lulc_tif = _latest_lulc_tif(src_station / "LULC")
            if lulc_tif:
                process_static(
                    encoder,
                    src_path = lulc_tif,
                    out_dir  = out_station / "LULC",
                    modality = "LULC",
                    out_stem = "lulc",
                    device   = device,
                    do_crop  = False,
                    failures = failures,
                )

        total_s2 += s2_new
        total_s1 += s1_new

        elapsed = time.time() - t0
        if s2_new or s1_new:
            print(f"[{idx:4d}/{n}] {sid}  S2={s2_new}  S1={s1_new}  ({elapsed:.1f}s)")
        elif idx % 50 == 0:
            print(f"[{idx:4d}/{n}] ... (up to date)")

    if trial:
        print("\n" + "="*60)
        print("TRIAL SUMMARY")
        print("="*60)
        print(f"\nS2L2A (batch_size={args.batch_size}):")
        print(perf_s2.report())
        print(f"\nS1RTC (batch_size={args.batch_size}):")
        print(perf_s1.report())
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"\nGPU total VRAM : {total_vram:.1f} GB")
        print("="*60)

    print(f"\nDone.  New acquisitions → S2: {total_s2},  S1: {total_s1}")
    if skipped:
        tail = f" … and {len(skipped) - 10} more" if len(skipped) > 10 else ""
        print(f"Skipped {len(skipped)} stations (no data dir): {', '.join(skipped[:10])}{tail}")
    if failures:
        manifest = args.data_dir / "failures.log"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with manifest.open("a") as fh:
            for path in failures:
                fh.write(f"{ts}  {path}\n")
        print(f"{len(failures)} file(s) failed (NaN/OOM/IO) — appended to {manifest}")


if __name__ == "__main__":
    main()
