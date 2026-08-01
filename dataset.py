"""
SoilMoistureDataset
====================
Loads pre-computed TerraMind features, ERA5-Land meteo, and ISMN labels
for one station × year × day-of-year sample.

All data is read from Zarr stores on scratch:
  {ZARR_ROOT}/{sm_only|sm_and_flux|flux_only}/{station}/
      s2/             dates, l3, l6, l9, l12, token_mask, cm
      s1_asc/         dates, l3, l6, l9, l12, token_mask
      s1_desc/        dates, l3, l6, l9, l12, token_mask  (if available)
      dem/            l12
      dem_token_mask/
      lulc/           l12
      lulc_token_mask/
      era5/           values, dates
      sif/            values, dates
      twsa/           values, dates
      labels/         soil_moisture, depth, time
      soil/           soil_patch (21, 74, 74)
"""

import json
import os
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
import zarr
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

ZARR_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")

# torch.from_numpy on a read-only /dev/shm memmap triggers a non-writable warning;
# the tensor is immediately copied into a pre-allocated output buffer so mutation is safe.
warnings.filterwarnings("ignore", message=".*not writable.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*non-writeable.*", category=UserWarning)

# Set DISABLE_L12_CACHE=1 to skip eager L12 RAM caching and force the lazy
# zarr chunk-read fallback in load_s2_rolling_zarr / load_s1_rolling_zarr.
DISABLE_L12_CACHE = os.environ.get("DISABLE_L12_CACHE", "") == "1"

# ── Constants ────────────────────────────────────────────────────────────────

ERA5_VARS = [
    "t2m_mean",  "t2m_min",  "t2m_max",
    "d2m_mean",  "d2m_min",  "d2m_max",
    "skt_mean",  "skt_min",  "skt_max",
    "u10_mean",  "u10_min",  "u10_max",
    "v10_mean",  "v10_min",  "v10_max",
    "sp_mean",   "sp_min",   "sp_max",
    "tp_sum",
]  # 19 features

SM_DEPTHS = ["0-10", "10-30", "30-100"]  # n_depths = 3

S2_BAND_INDICES = list(range(12))  # all 12 S2L2A bands (no B10)

PREC_IDX = ERA5_VARS.index("tp_sum")  # index computed once at import, not per sample

MAX_S2 = 60
MAX_S1 = 40


# ── Helpers ──────────────────────────────────────────────────────────────────

def center_crop(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Crop (C, H, W) array from centre to size×size."""
    _, h, w = arr.shape
    top  = (h - size) // 2
    left = (w - size) // 2
    return arr[:, top:top + size, left:left + size]


def _rel_pos(acq_doy: int, acq_year: int, target_doy: int, target_year: int) -> int:
    """
    0-indexed position in the 365-day rolling window (0=oldest, 364=today).
    Uses datetime subtraction so leap-year DOY 366 never overflows rel_pos_emb(365).
    """
    acq_dt    = datetime(acq_year,    1, 1) + timedelta(days=acq_doy    - 1)
    target_dt = datetime(target_year, 1, 1) + timedelta(days=target_doy - 1)
    return 364 - (target_dt - acq_dt).days


def _window_datetimes(year: int, target_doy: int) -> tuple[datetime, datetime]:
    """
    Return (window_start, target_date) for the 365-day rolling window.
    Uses timedelta arithmetic — always exactly 365 days inclusive regardless of leap years.
    """
    target_date  = datetime(year, 1, 1) + timedelta(days=target_doy - 1)
    window_start = target_date - timedelta(days=364)
    return window_start, target_date


def _in_window(date_str: str, window_start: datetime, target_date: datetime) -> bool:
    """True if date_str (YYYYMMDD) falls in [window_start, target_date]."""
    try:
        dt = datetime.strptime(date_str[:8], "%Y%m%d")
        return window_start <= dt <= target_date
    except ValueError:
        return False



def _date_to_int(dt) -> int:
    """Convert datetime-like to YYYYMMDD int."""
    return dt.year * 10000 + dt.month * 100 + dt.day


def _window_ints(year: int, target_doy: int) -> tuple[int, int]:
    """Return (start_int, end_int) as YYYYMMDD ints for the 365-day rolling window."""
    ws, td = _window_datetimes(year, target_doy)
    return _date_to_int(ws), _date_to_int(td)



# ── Zarr helpers ─────────────────────────────────────────────────────────────

def _open_zarr(station_dir: Path, category: str) -> zarr.Group | None:
    """
    Open zarr group for a station from scratch SSD.
    Uses consolidated metadata when available (single .zmetadata read at open).
    Returns None if zarr store is not yet complete.
    """
    path = ZARR_ROOT / category / station_dir.name
    if not (path / ".complete").exists():
        return None
    try:
        return zarr.open_consolidated(str(path), mode="r")
    except KeyError:
        return zarr.open_group(str(path), mode="r")


def _load_zarr_era5(zg: zarr.Group):
    """Load ERA5 from zarr → same tuple format as _load_era5_nc()."""
    if "era5/values" not in zg:
        return None
    return (
        zg["era5/values"][:],
        zg["era5/date_ints"][:],
        zg["era5/doys"][:],
    )


def _load_zarr_sif(zg: zarr.Group):
    """Load SIF from zarr → same tuple format as _load_sif_nc()."""
    if "sif/values" not in zg:
        return None
    return (
        zg["sif/values"][:],
        zg["sif/date_ints"][:],
        zg["sif/doys"][:],
    )


def _load_zarr_twsa(zg: zarr.Group):
    """Load TWSA from zarr → same tuple format as _load_twsa_nc()."""
    if "twsa/lwe" not in zg:
        return None
    return (
        zg["twsa/lwe"][:],
        zg["twsa/date_ints"][:],
        zg["twsa/doys"][:],
    )


def _load_zarr_labels(zg: zarr.Group):
    """Load labels from zarr → (sm_np, depths, times, qc_np).
    qc_np: 0=observed, 1=gap-filled, 2=still missing. None if not present."""
    if "labels/sm" not in zg:
        return None
    sm_np  = zg["labels/sm"][:]                        # (n_depths, n_days) float32
    depths = [str(d) for d in zg["labels/depths"][:]]
    dates  = [str(d) for d in zg["labels/dates"][:]]
    times  = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    qc_np  = zg["labels/qc"][:] if "labels/qc" in zg else None  # (n_depths, n_days) uint8
    if qc_np is not None and qc_np.shape[1] != sm_np.shape[1]:
        # trim_pre2016.py trimmed sm/dates but not qc — take trailing n days to realign
        qc_np = qc_np[:, -sm_np.shape[1]:]
    return sm_np, depths, times, qc_np


def _cpu_pyramid_pool(l12: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked spatial pyramid pooling on CPU inside dataset workers.

    Compresses history L12 tokens from (M, 196, 768) → (M, 4, 768) before the
    IPC barrier, cutting per-sample payload from ~30 MB to ~2 MB and eliminating
    the ~437 GB DataLoader IPC queue that caused epoch-boundary OOM kills.

    Uses the same 4-scale nested-window logic as model.spatial_pyramid_pool()
    but with static masked-mean pooling (no learned attention weights) so it
    can run on CPU without model access.

    l12:        (M, 196, 768) fp16 — zero-padded for empty slots
    token_mask: (M, 14, 14)   bool — True = valid/clear patch
    returns:    (M, 4, 768)   fp32
    """
    M, N_tok, D = l12.shape
    G    = int(N_tok ** 0.5)                        # 14 for 196 tokens
    g    = l12.float().reshape(M, G, G, D)          # (M, 14, 14, 768)
    v    = token_mask.float().unsqueeze(-1)          # (M, 14, 14, 1)

    half   = G // 2
    widths = [max(1, G * (i + 1) // 8) for i in range(4)]

    def _pool(w):
        rs, re = half - w, half + w
        rg = g[:, rs:re, rs:re, :]                  # (M, 2w, 2w, 768)
        rv = v[:, rs:re, rs:re, :]                  # (M, 2w, 2w, 1)
        return (rg * rv).sum(dim=(1, 2)) / rv.sum(dim=(1, 2)).clamp(min=1)  # (M, 768)

    return torch.stack([_pool(w) for w in widths], dim=1)   # (M, 4, 768)


def load_s2_rolling_zarr(zg: zarr.Group, year: int, target_doy: int,
                          max_acq: int = MAX_S2,
                          l12_np: np.ndarray | None = None,
                          date_cache: dict | None = None,
                          cm_token_mask: np.ndarray | None = None,
                          training: bool = False):
    """
    Load S2 L12 tokens for the 365-day rolling window and compress to pyramid tokens.

    date_cache:    precomputed per-orbit date info from _zarr_date_cache (eliminates zarr date reads)
    cm_token_mask: precomputed (N_cm, 14, 14) bool quality array from _cm_token_mask_cache
    l12_np:        preloaded L12 tokens from RAM/shm (eliminates chunk reads for history tokens)
    training:      if True, applies ContextFormer-style 50% random spatial token dropout to
                   token_mask before pyramid pooling — restores the augmentation that was
                   previously done in model.forward() before pyramid pooling moved to CPU.

    Returns (pyr, doys, valid, rel_pos) where pyr is (max_acq, 4, 768) fp32 — already
    pyramid-pooled on CPU so the full 196×768 tensors never cross the IPC barrier.
    """
    l12        = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    token_mask = torch.ones(max_acq, 14, 14, dtype=torch.bool)
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)

    if "s2/l12" not in zg:
        pyr = torch.zeros(max_acq, 4, 768, dtype=torch.float32)
        return pyr, doys, doys > 0, rel_pos

    # Date arrays — use precomputed cache (no zarr read) or fall back to zarr
    if date_cache is not None and "s2" in date_cache:
        s2_dc      = date_cache["s2"]
        all_dates  = s2_dc["dates"]
        date_ints  = s2_dc["date_ints"]
        doys_arr   = s2_dc["doys"]
        years_arr  = s2_dc["years"]
        start_int, end_int = _window_ints(year, target_doy)
        win_idx    = np.where((date_ints >= start_int) & (date_ints <= end_int))[0][-max_acq:].tolist()
    else:
        all_dates  = [str(d) for d in zg["s2/dates"][:]]
        ws, td     = _window_datetimes(year, target_doy)
        win_idx    = [i for i, d in enumerate(all_dates) if _in_window(d, ws, td)][-max_acq:]
        doys_arr   = None
        years_arr  = None

    # CM date→index lookup — prefer precomputed, fall back to zarr read
    if date_cache is not None and "cm" in date_cache:
        cm_d2i = date_cache["cm"].get("date_to_idx", {})
    elif "cm/masks" in zg and "cm/dates" in zg:
        cm_d2i = {str(d): i for i, d in enumerate(zg["cm/dates"][:])}
    else:
        cm_d2i = {}

    tokens_z = l12_np if l12_np is not None else zg["s2/l12"]
    cm_z     = None if (cm_token_mask is not None or not ("cm/masks" in zg)) else zg["cm/masks"]

    for out_i, src_i in enumerate(win_idx):
        date_str = all_dates[src_i]
        if doys_arr is not None:
            acq_doy  = int(doys_arr[src_i])
            acq_year = int(years_arr[src_i])
        else:
            dt       = datetime.strptime(date_str[:8], "%Y%m%d")
            acq_doy  = dt.timetuple().tm_yday
            acq_year = dt.year

        tok = torch.from_numpy(tokens_z[src_i])
        if torch.isnan(tok).any():
            continue
        l12[out_i]     = tok
        doys[out_i]    = acq_doy
        rel_pos[out_i] = _rel_pos(acq_doy, acq_year, target_doy, year)

        if date_str in cm_d2i:
            cm_idx = cm_d2i[date_str]
            if cm_token_mask is not None:
                token_mask[out_i] = torch.from_numpy(cm_token_mask[cm_idx])
            elif cm_z is not None:
                cm    = cm_z[cm_idx]
                cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
                bad_frac = np.isin(cm_4d, [3, 4, 5, 255]).mean(axis=(1, 3))
                token_mask[out_i] = torch.from_numpy(bad_frac <= 0.01)

    if training:
        token_mask = token_mask & (torch.rand(max_acq, 14, 14) >= 0.5)
    pyr = _cpu_pyramid_pool(l12, token_mask)  # (max_acq, 4, 768) fp32
    return pyr, doys, doys > 0, rel_pos


def load_s1_rolling_zarr(zg: zarr.Group, year: int, target_doy: int,
                          max_acq: int = MAX_S1,
                          l12_asc_np: np.ndarray | None = None,
                          l12_desc_np: np.ndarray | None = None,
                          date_cache: dict | None = None,
                          s1_token_mask_cache: dict | None = None,
                          training: bool = False):
    """Load S1 L12 tokens (ASC + DESC merged) from zarr and compress to pyramid tokens.

    date_cache:          precomputed per-orbit date info (eliminates zarr date reads)
    s1_token_mask_cache: precomputed {orbit: (N,14,14) bool} from _s1_token_mask_cache
    l12_asc_np / l12_desc_np: preloaded RAM arrays; eliminates L12 chunk reads.
    training:            if True, applies 50% random spatial token dropout before pooling.
    Returns (pyr, doys, valid, rel_pos) 4-tuple matching S2's signature.
    """
    l12        = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    token_mask = torch.ones(max_acq, 14, 14, dtype=torch.bool)
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)

    start_int, end_int = _window_ints(year, target_doy)
    ws, td = _window_datetimes(year, target_doy)
    entries: list[tuple[str, object, int, str, int | None, int | None]] = []

    l12_np_map = {"s1_asc": l12_asc_np, "s1_desc": l12_desc_np}

    # S1 token masks — use precomputed cache or fall back to zarr read
    tm_np_map: dict[str, np.ndarray | None] = {}
    for orbit_key in ("s1_asc", "s1_desc"):
        if s1_token_mask_cache is not None:
            tm_np_map[orbit_key] = s1_token_mask_cache.get(orbit_key)
        else:
            mk = f"{orbit_key}/token_mask"
            tm_np_map[orbit_key] = np.asarray(zg[mk][:]) if mk in zg else None

        if f"{orbit_key}/l12" not in zg:
            continue

        # Date arrays — precomputed cache or zarr read
        if date_cache is not None and orbit_key in date_cache:
            dc         = date_cache[orbit_key]
            orbit_dates = dc["dates"]
            di          = dc["date_ints"]
            doys_a      = dc["doys"]
            years_a     = dc["years"]
            idx_arr     = np.where((di >= start_int) & (di <= end_int))[0]
            _src = l12_np_map[orbit_key]
            tokens_src  = _src if _src is not None else zg[f"{orbit_key}/l12"]
            for i in idx_arr:
                entries.append((orbit_dates[i], tokens_src, int(i), orbit_key,
                                 int(doys_a[i]), int(years_a[i])))
        else:
            orbit_dates = [str(d) for d in zg[f"{orbit_key}/dates"][:]]
            _src = l12_np_map[orbit_key]
            tokens_src  = _src if _src is not None else zg[f"{orbit_key}/l12"]
            for i, d in enumerate(orbit_dates):
                if _in_window(d, ws, td):
                    entries.append((d, tokens_src, i, orbit_key, None, None))

    if not entries:
        pyr = torch.zeros(max_acq, 4, 768, dtype=torch.float32)
        return pyr, doys, doys > 0, rel_pos

    entries.sort(key=lambda x: x[0])
    entries = entries[-max_acq:]

    out_i = 0
    for date_str, tokens_z, src_i, orbit_key, cached_doy, cached_year in entries:
        tok = torch.from_numpy(np.asarray(tokens_z[src_i]))
        if torch.isnan(tok).any():
            continue
        if cached_doy is not None:
            acq_doy  = cached_doy
            acq_year = cached_year
        else:
            dt       = datetime.strptime(date_str[:8], "%Y%m%d")
            acq_doy  = dt.timetuple().tm_yday
            acq_year = dt.year
        l12[out_i]     = tok
        doys[out_i]    = acq_doy
        rel_pos[out_i] = _rel_pos(acq_doy, acq_year, target_doy, year)
        if tm_np_map[orbit_key] is not None:
            token_mask[out_i] = torch.from_numpy(tm_np_map[orbit_key][src_i])
        out_i += 1

    if training:
        token_mask = token_mask & (torch.rand(max_acq, 14, 14) >= 0.5)
    pyr = _cpu_pyramid_pool(l12, token_mask)  # (max_acq, 4, 768) fp32
    return pyr, doys, doys > 0, rel_pos


def select_anchor_zarr(zg: zarr.Group, year: int, target_doy: int,
                        l12_cache: dict | None = None,
                        l369_cache: dict | None = None,
                        date_cache: dict | None = None,
                        cm_token_mask: np.ndarray | None = None,
                        s1_token_mask_cache: dict | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select the best anchor acquisition for the U-Net skip connections and
    target spatial tokens.

    date_cache:          precomputed per-orbit date info (eliminates zarr date reads)
    cm_token_mask:       precomputed (N_cm, 14, 14) bool from _cm_token_mask_cache
    s1_token_mask_cache: precomputed {orbit: (N,14,14) bool} from _s1_token_mask_cache

    Priority:
      1. Most recent acquisition with ALL 196 patches valid (fully clear/no-NaN).
      2. If none fully clear, fall back to most recent acquisition regardless.
    All zero if no in-window acquisition found.
    """
    zeros  = torch.zeros(196, 768, dtype=torch.float16)
    start_int, end_int = _window_ints(year, target_doy)
    ws, td = _window_datetimes(year, target_doy)

    # CM date→index lookup
    if date_cache is not None and "cm" in date_cache:
        cm_d2i = date_cache["cm"].get("date_to_idx", {})
    elif "cm/masks" in zg and "cm/dates" in zg:
        cm_d2i = {str(d): i for i, d in enumerate(zg["cm/dates"][:])}
    else:
        cm_d2i = {}
    cm_z = None if cm_token_mask is not None else (zg["cm/masks"] if "cm/masks" in zg else None)

    # S1 token masks — precomputed or zarr fallback
    tm_np: dict[str, np.ndarray | None] = {}
    for ok in ("s1_asc", "s1_desc"):
        if s1_token_mask_cache is not None:
            tm_np[ok] = s1_token_mask_cache.get(ok)
        else:
            mk = f"{ok}/token_mask"
            tm_np[ok] = np.asarray(zg[mk][:]) if mk in zg else None

    # ── Collect all in-window candidates ─────────────────────────────────
    candidates: list[tuple[str, str, int, int]] = []

    for orbit in ("s2", "s1_asc", "s1_desc"):
        dates_key = f"{orbit}/dates"
        if dates_key not in zg or f"{orbit}/l12" not in zg:
            continue
        # Date filtering — precomputed cache or zarr read
        if date_cache is not None and orbit in date_cache:
            dc          = date_cache[orbit]
            orbit_dates = dc["dates"]
            idx_arr     = np.where((dc["date_ints"] >= start_int) &
                                   (dc["date_ints"] <= end_int))[0]
            in_window_pairs = [(orbit_dates[i], int(i)) for i in idx_arr]
        else:
            orbit_dates = [str(d) for d in zg[dates_key][:]]
            in_window_pairs = [(d, i) for i, d in enumerate(orbit_dates)
                               if _in_window(d, ws, td)]

        for d, i in in_window_pairs:
            if orbit == "s2":
                if d in cm_d2i:
                    cm_idx = cm_d2i[d]
                    if cm_token_mask is not None:
                        n_valid = int(cm_token_mask[cm_idx].sum())
                    elif cm_z is not None:
                        cm    = cm_z[cm_idx]
                        cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
                        bad   = np.isin(cm_4d, [3, 4, 5, 255]).mean(axis=(1, 3))
                        n_valid = int((bad <= 0.01).sum())
                    else:
                        n_valid = 196
                else:
                    n_valid = 196
            else:
                n_valid = int(tm_np[orbit][i].sum()) if tm_np[orbit] is not None else 196
            candidates.append((d, orbit, i, n_valid))

    if not candidates:
        return (zeros, zeros.clone(), zeros.clone(), zeros.clone(),
                torch.tensor(0, dtype=torch.long),
                torch.tensor(0, dtype=torch.long))   # default orbit: S2=0

    # ── Select anchor ─────────────────────────────────────────────────────
    # Priority 1: most recent fully-clear (n_valid == 196)
    fully_clear = [c for c in candidates if c[3] == 196]
    best = max(fully_clear, key=lambda c: c[0]) if fully_clear \
           else max(candidates, key=lambda c: c[0])   # fallback: most recent

    best_date, best_orbit, best_idx, _ = best
    if date_cache is not None and best_orbit in date_cache:
        dc_o     = date_cache[best_orbit]
        acq_doy  = int(dc_o["doys"][best_idx])
        acq_year = int(dc_o["years"][best_idx])
    else:
        dt       = datetime.strptime(best_date[:8], "%Y%m%d")
        acq_doy  = dt.timetuple().tm_yday
        acq_year = dt.year
    rp      = _rel_pos(acq_doy, acq_year, target_doy, year)
    orbit_id = {"s2": 0, "s1_asc": 1, "s1_desc": 2}[best_orbit]

    # ── Load L3 / L6 / L9 / L12 for chosen anchor ────────────────────────
    def _load_layer(layer: str) -> torch.Tensor:
        # L3/L6/L9: prefer .npy memmap (flat binary, no decompression).
        # On OS page-cache hit (epoch 2+) this is a pure RAM read.
        if layer in ("l3", "l6", "l9") and l369_cache:
            arr = l369_cache.get(f"{best_orbit}_{layer}")
            if arr is not None and best_idx < len(arr):
                return torch.from_numpy(np.asarray(arr[best_idx]))
        # Fallback: zarr (compressed GPFS read)
        key = f"{best_orbit}/{layer}"
        if key not in zg or best_idx >= zg[key].shape[0]:
            return zeros.clone()
        return torch.from_numpy(np.asarray(zg[key][best_idx]))

    # Use preloaded L12 RAM cache (zero disk I/O) when available
    l12_np_for_orbit = (l12_cache or {}).get(best_orbit)
    if l12_np_for_orbit is not None and best_idx < len(l12_np_for_orbit):
        anchor_l12 = torch.from_numpy(np.asarray(l12_np_for_orbit[best_idx]))
    else:
        anchor_l12 = _load_layer("l12")

    return (
        _load_layer("l3"),
        _load_layer("l6"),
        _load_layer("l9"),
        anchor_l12,
        torch.tensor(rp,       dtype=torch.long),
        torch.tensor(orbit_id, dtype=torch.long),  # 0=S2, 1=S1_asc, 2=S1_desc
    )


# ── Soil patch helpers ───────────────────────────────────────────────────────

def fill_soil_nans(patch: np.ndarray) -> np.ndarray:
    """
    Fill NaN pixels in a soil patch via nearest-neighbour propagation.
    patch : (21, 74, 74) float32
    Operates channel-by-channel; channels with no NaN are untouched.
    """
    out = patch.copy()
    for c in range(out.shape[0]):
        mask = np.isnan(out[c])
        if mask.any():
            _, idx = distance_transform_edt(mask, return_indices=True)
            out[c] = out[c][tuple(idx)]
    return out



# ── ERA5 rolling slicer (no file I/O — works on pre-loaded numpy arrays) ────

def load_era5_rolling(cache_entry, year: int, target_doy: int):
    """
    Slice pre-loaded ERA5 arrays for the 365-day rolling window.

    Args:
        cache_entry: (values (N,19) float32, date_ints (N,) int32, doys (N,) int32) or None
    Returns:
        era5 : (365, 19) float32 numpy array
        doys : (365,) int64 numpy array
    """
    if cache_entry is None:
        return (np.zeros((365, len(ERA5_VARS)), dtype=np.float32),
                np.zeros(365, dtype=np.int64))

    values, date_ints, doy_arr = cache_entry
    start_int, end_int = _window_ints(year, target_doy)
    mask = (date_ints >= start_int) & (date_ints <= end_int)

    era5_win = values[mask]
    doys_win = doy_arr[mask].astype(np.int64)

    n = 365
    out_era5 = np.zeros((n, len(ERA5_VARS)), dtype=np.float32)
    out_doys = np.zeros(n, dtype=np.int64)
    l = min(len(doys_win), n)
    out_era5[-l:] = era5_win[-l:]
    out_doys[-l:] = doys_win[-l:]
    return out_era5, out_doys



# ── SIF rolling slicer (no file I/O) ─────────────────────────────────────────

MAX_SIF  = 50
MAX_TWSA = 12


def load_sif_rolling(cache_entry, year: int, target_doy: int):
    """
    Slice pre-loaded SIF arrays for the 365-day rolling window.

    Args:
        cache_entry: (values (N,) float32, date_ints (N,) int32, doys (N,) int32) or None
    Returns:
        vals    : (MAX_SIF, 1) float32
        doys    : (MAX_SIF,) long  -- absolute day-of-year (for sinusoidal_pe)
        rel_pos : (MAX_SIF,) long  -- 0..364 rolling-window position (for rel_pos_emb)
        valid   : (MAX_SIF,) bool
    """
    vals    = torch.zeros(MAX_SIF, 1, dtype=torch.float32)
    doys    = torch.zeros(MAX_SIF, dtype=torch.long)
    rel_pos = torch.zeros(MAX_SIF, dtype=torch.long)
    valid   = torch.zeros(MAX_SIF, dtype=torch.bool)

    if cache_entry is None:
        return vals, doys, rel_pos, valid

    values, date_ints, doy_arr = cache_entry
    start_int, end_int = _window_ints(year, target_doy)
    mask = (date_ints >= start_int) & (date_ints <= end_int)

    win_vals  = values[mask]
    win_doys  = doy_arr[mask]
    win_dates = date_ints[mask]
    n_win = min(len(win_vals), MAX_SIF)
    win_vals  = win_vals[-n_win:]
    win_doys  = win_doys[-n_win:]
    win_dates = win_dates[-n_win:]

    vals[:n_win, 0] = torch.from_numpy(win_vals)
    doys[:n_win]    = torch.from_numpy(win_doys.astype(np.int64))
    valid[:n_win]   = True
    if n_win > 0:
        acq_years = (win_dates // 10000).astype(np.int32)
        target_dt = datetime(year, 1, 1) + timedelta(days=target_doy - 1)
        rp = np.array([
            364 - (target_dt - (datetime(int(acq_years[i]), 1, 1)
                                + timedelta(days=int(win_doys[i]) - 1))).days
            for i in range(n_win)
        ], dtype=np.int64)
        rel_pos[:n_win] = torch.from_numpy(rp)

    return vals, doys, rel_pos, valid


# ── TWSA rolling slicer (no file I/O) ────────────────────────────────────────

def load_twsa_rolling(cache_entry, year: int, target_doy: int):
    """
    Slice pre-loaded TWSA arrays for the 365-day rolling window.
    TWSA is monthly; typically ≤ 12 observations per year.

    Args:
        cache_entry: (values (N,) float32, date_ints (N,) int32, doys (N,) int32) or None
    Returns:
        vals    : (MAX_TWSA, 1) float32
        doys    : (MAX_TWSA,) long  -- absolute day-of-year (for sinusoidal_pe)
        rel_pos : (MAX_TWSA,) long  -- 0..364 rolling-window position (for rel_pos_emb)
        valid   : (MAX_TWSA,) bool
    """
    vals    = torch.zeros(MAX_TWSA, 1, dtype=torch.float32)
    doys    = torch.zeros(MAX_TWSA, dtype=torch.long)
    rel_pos = torch.zeros(MAX_TWSA, dtype=torch.long)
    valid   = torch.zeros(MAX_TWSA, dtype=torch.bool)

    if cache_entry is None:
        return vals, doys, rel_pos, valid

    values, date_ints, doy_arr = cache_entry
    start_int, end_int = _window_ints(year, target_doy)
    mask = (date_ints >= start_int) & (date_ints <= end_int)

    win_vals  = values[mask]
    win_doys  = doy_arr[mask]
    win_dates = date_ints[mask]
    n_win = min(len(win_vals), MAX_TWSA)
    win_vals  = win_vals[-n_win:]
    win_doys  = win_doys[-n_win:]
    win_dates = win_dates[-n_win:]

    vals[:n_win, 0] = torch.from_numpy(win_vals)
    doys[:n_win]    = torch.from_numpy(win_doys.astype(np.int64))
    valid[:n_win]   = True
    if n_win > 0:
        acq_years = (win_dates // 10000).astype(np.int32)
        target_dt = datetime(year, 1, 1) + timedelta(days=target_doy - 1)
        rp = np.array([
            364 - (target_dt - (datetime(int(acq_years[i]), 1, 1)
                                + timedelta(days=int(win_doys[i]) - 1))).days
            for i in range(n_win)
        ], dtype=np.int64)
        rel_pos[:n_win] = torch.from_numpy(rp)

    return vals, doys, rel_pos, valid


def _load_l12_shm(dir_name: str, shm_dir: Path) -> dict[str, np.ndarray] | None:
    """Return memmapped L12 arrays from /dev/shm (shared across all DDP ranks)."""
    result = {}
    for key in ("s2", "s1_asc", "s1_desc"):
        bin_path  = shm_dir / f"{dir_name}__{key}.bin"
        meta_path = shm_dir / f"{dir_name}__{key}.meta.json"
        if not bin_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        result[key] = np.memmap(bin_path, dtype=meta["dtype"], mode="r",
                                shape=tuple(meta["shape"]))
    return result or None


# ── Dataset ──────────────────────────────────────────────────────────────────

class SoilMoistureDataset(Dataset):
    """
    One sample = one (station, year, day-of-year) triple.

    All data is read from ZARR_ROOT (/gpfs/scratch1/shared/pkhanal/zarr).
    Zarr layout per station:
        {ZARR_ROOT}/{sm_only|sm_and_flux|flux_only}/{station}/
            s2/          dates, l3, l6, l9, l12, token_mask, cm
            s1_asc/      dates, l3, l6, l9, l12, token_mask
            s1_desc/     dates, l3, l6, l9, l12, token_mask  (if available)
            dem/         l12
            dem_token_mask/
            lulc/        l12
            lulc_token_mask/
            era5/        values, dates
            sif/         values, dates
            twsa/        values, dates
            labels/      soil_moisture, depth, time
            soil/        soil_patch (21, 74, 74)

    Args:
        splits_csv       : path to station_splits.csv
        era5_stats_path  : path to csvs/era5_stats.json  (from compute_era5_stats.py)
        years            : list of years to include (default 2016–2023)
        min_obs          : minimum observed SM days per year to include
        category_filter  : list of categories to include, e.g. ["sm_only"]  (None = all)
        split_filter     : list of split values to include, e.g. ["train"]  (None = all)
        training         : if True, apply SIF/TWSA modality dropout (p=0.5 each)
        max_stations     : if set, stop scanning splits once this many stations have
                           been cached (smoke-test mode)
    """

    def __init__(
        self,
        splits_csv:      str,
        era5_stats_path: str,
        years=None,
        min_obs:         int        = 30,
        category_filter: list | None = None,
        split_filter:    list | None = None,
        training:        bool        = True,
        max_stations:    int | None  = None,
        shm_dir:         Path | None = None,
        use_mmap:        bool        = False,
    ):
        self.training   = training
        self._use_mmap  = use_mmap
        self.years     = years or list(range(2016, 2024))

        # ERA5 normalisation stats
        with open(era5_stats_path) as f:
            era5_stats = json.load(f)
        self._era5_means      = np.array(era5_stats["means"],  dtype=np.float32)
        self._era5_stds       = np.array(era5_stats["stds"],   dtype=np.float32)
        self._era5_log1p_prec = bool(era5_stats.get("log1p_precip", False))

        splits = pd.read_csv(splits_csv)

        # Category filter using has_soil_moisture / has_flux columns
        if category_filter is not None:
            def _cat(r):
                sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
                fl = str(r.get("has_flux",          "False")).lower() == "true"
                return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")
            splits = splits[splits.apply(_cat, axis=1).isin(category_filter)]

        if split_filter is not None:
            splits = splits[splits["split"].isin(split_filter)]

        self.samples = []

        # Per-station in-memory caches (numpy arrays, no open file handles).
        # Populated once in __init__; DataLoader workers inherit via fork (copy-on-write).
        # _zarr_groups: sat_dir → open zarr.Group (or None if zarr not available)
        # _l12_cache:   sat_dir → {"s2": (N,196,768) fp16, "s1_asc": ..., "s1_desc": ...}
        #               Preloads L12 token arrays into RAM so __getitem__ does 0 disk reads
        #               for history tokens. CoW fork: one physical copy across all workers.
        # ERA5/SIF/TWSA/label caches: same format, all populated from zarr on scratch.
        self._zarr_groups  : dict[Path, zarr.Group | None]       = {}
        self._l12_cache    : dict[Path, dict[str, np.ndarray]]   = {}
        self._era5_cache   : dict[Path, tuple | None] = {}
        self._sif_cache    : dict[Path, tuple | None] = {}
        self._twsa_cache   : dict[Path, tuple | None] = {}
        self._label_cache  : dict[Path, tuple]        = {}
        # Static-per-station tensors (DEM, LULC, soil, token masks).
        # Loaded once at init; workers inherit as shared CoW pages (read-only).
        self._static_cache : dict[Path, dict[str, torch.Tensor]] = {}
        # Precomputed zarr data — eliminates all GPFS reads per __getitem__:
        #   _cm_token_mask_cache : (N_cm, 14, 14) bool quality array per station
        #   _s1_token_mask_cache : {orbit: (N, 14, 14) bool} per station
        #   _zarr_date_cache     : {orbit: {"dates", "date_ints", "doys", "years"}} per station
        self._cm_token_mask_cache : dict[Path, np.ndarray | None] = {}
        self._s1_token_mask_cache : dict[Path, dict]               = {}
        self._zarr_date_cache     : dict[Path, dict]               = {}
        # L3/L6/L9 memmap cache: sat_dir → {"s2_l3": memmap(N,196,768), ...}
        # Populated when use_mmap=True; workers inherit via CoW fork (read-only, no duplication).
        # OS page cache backs the memmaps — after epoch-1 warmup, reads serve from free RAM.
        self._l369_cache          : dict[Path, dict[str, np.ndarray]] = {}

        for _, r in splits.iterrows():
            has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
            has_fl = str(r.get("has_flux",          "False")).lower() == "true"
            cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")

            # Build directory name matching the on-disk convention
            if str(r["source_network"]) == "ISMN":
                dir_name = f"ISMN_{r['network']}_{r['station_name']}"
            else:
                dir_name = f"{r['source_network']}_{r['station_id']}"

            sat_dir = ZARR_ROOT / cat / dir_name

            if not bool(r.get("soil_patch_ok", True)):
                continue

            # Load per-station data into memory once (subsequent rows reuse caches)
            if sat_dir not in self._zarr_groups:
                if max_stations is not None and len(self._zarr_groups) >= max_stations:
                    break
                zg = _open_zarr(sat_dir, cat)
                self._zarr_groups[sat_dir] = zg
                if zg is not None:
                    self._era5_cache[sat_dir]  = _load_zarr_era5(zg)
                    self._sif_cache[sat_dir]   = _load_zarr_sif(zg)
                    self._twsa_cache[sat_dir]  = _load_zarr_twsa(zg)
                    # Prefer /dev/shm memmaps (one physical copy shared across DDP ranks)
                    if shm_dir is not None:
                        shm_l12 = _load_l12_shm(dir_name, shm_dir)
                        if shm_l12:
                            self._l12_cache[sat_dir] = shm_l12
                    if sat_dir not in self._l12_cache and not DISABLE_L12_CACHE:
                        self._l12_cache[sat_dir] = {
                            k: zg[f"{k}/l12"][:]
                            for k in ("s2", "s1_asc", "s1_desc")
                            if f"{k}/l12" in zg
                        }
                    # L3/L6/L9 memmap — opened lazily; OS page cache warms on epoch-1 access.
                    # Files live at sat_dir/{orbit}_{layer}.npy (same tree as zarr, flat binary).
                    if self._use_mmap:
                        l369: dict[str, np.ndarray] = {}
                        for _orbit in ("s2", "s1_asc", "s1_desc"):
                            for _layer in ("l3", "l6", "l9"):
                                _p  = sat_dir / f"{_orbit}_{_layer}.npy"
                                _jp = sat_dir / f"{_orbit}_{_layer}.json"
                                if _p.exists() and _jp.exists():
                                    import json as _json
                                    _shape = tuple(_json.loads(_jp.read_text())["shape"])
                                    l369[f"{_orbit}_{_layer}"] = np.memmap(
                                        str(_p), dtype="float16", mode="r", shape=_shape
                                    )
                        if not l369:
                            print(f"[WARN] use_mmap=True but no .npy files found for {sat_dir} "
                                  f"— falling back to zarr (run convert_l369_to_npy.py first)")
                        self._l369_cache[sat_dir] = l369
                    _dem_l12  = (torch.from_numpy(zg["dem"][:])
                                 if "dem" in zg
                                 else torch.zeros(196, 768, dtype=torch.float16))
                    _lulc_l12 = (torch.from_numpy(zg["lulc"][:])
                                 if "lulc" in zg
                                 else torch.zeros(196, 768, dtype=torch.float16))
                    _dem_tm   = (torch.from_numpy(np.asarray(zg["dem_token_mask"][:]))
                                 if "dem_token_mask" in zg
                                 else torch.ones(14, 14, dtype=torch.bool))
                    _lulc_tm  = (torch.from_numpy(np.asarray(zg["lulc_token_mask"][:]))
                                 if "lulc_token_mask" in zg
                                 else torch.ones(14, 14, dtype=torch.bool))
                    self._static_cache[sat_dir] = {
                        "dem":            _dem_l12,
                        "lulc":           _lulc_l12,
                        "dem_token_mask": _dem_tm,
                        "lulc_token_mask":_lulc_tm,
                        "dem_pyr":  _cpu_pyramid_pool(_dem_l12.unsqueeze(0),  _dem_tm.unsqueeze(0)).squeeze(0),
                        "lulc_pyr": _cpu_pyramid_pool(_lulc_l12.unsqueeze(0), _lulc_tm.unsqueeze(0)).squeeze(0),
                        "soil":           (torch.from_numpy(fill_soil_nans(zg["soil"][:]))
                                           if "soil" in zg
                                           else torch.zeros(21, 74, 74, dtype=torch.float32)),
                    }
                    lc = _load_zarr_labels(zg)
                    if lc is not None:
                        self._label_cache[sat_dir] = lc

                    # ── Precompute GPFS-hot-path data once at init ──────────────
                    # (a) CM token-mask quality: (N_cm, 14, 14) bool — one bulk read
                    cm_qm = None
                    if "cm/masks" in zg and "cm/dates" in zg:
                        cm_all = zg["cm/masks"][:]
                        cm_4d  = cm_all[:, :224, :224].reshape(len(cm_all), 14, 16, 14, 16)
                        bad    = np.isin(cm_4d, [3, 4, 5, 255]).mean(axis=(2, 4))
                        cm_qm  = (bad <= 0.01)
                    self._cm_token_mask_cache[sat_dir] = cm_qm

                    # (b) S1 token masks per orbit: {orbit: (N, 14, 14) bool}
                    s1_tm: dict[str, np.ndarray] = {}
                    for _ok in ("s1_asc", "s1_desc"):
                        _mk = f"{_ok}/token_mask"
                        if _mk in zg:
                            s1_tm[_ok] = np.asarray(zg[_mk][:])
                    self._s1_token_mask_cache[sat_dir] = s1_tm

                    # (c) Date arrays + precomputed date_ints/years/doys per orbit
                    _dc: dict = {}
                    for _orbit in ("s2", "s1_asc", "s1_desc", "cm"):
                        _dkey = f"{_orbit}/dates"
                        if _dkey not in zg:
                            continue
                        _dates     = [str(d) for d in zg[_dkey][:]]
                        _date_ints = np.array([int(d[:8]) for d in _dates], dtype=np.int32)
                        _years_a   = (_date_ints // 10000).astype(np.int16)
                        _doys_a    = np.array([
                            datetime(int(d[:4]), int(d[4:6]), int(d[6:8])).timetuple().tm_yday
                            for d in _dates
                        ], dtype=np.int16)
                        _dc[_orbit] = {"dates": _dates, "date_ints": _date_ints,
                                        "years": _years_a, "doys": _doys_a}
                    if "cm" in _dc:
                        _dc["cm"]["date_to_idx"] = {d: i for i, d in enumerate(_dc["cm"]["dates"])}
                    self._zarr_date_cache[sat_dir] = _dc

            # ERA5 year range from cache (fast int arithmetic — no file I/O)
            era5_entry = self._era5_cache.get(sat_dir)
            if era5_entry is None:
                continue
            era5_start_year = int(era5_entry[1][0])  // 10000
            era5_end_year   = int(era5_entry[1][-1]) // 10000

            # S2 year range: from date cache (no zarr I/O)
            _s2_dc = self._zarr_date_cache.get(sat_dir, {}).get("s2")
            if _s2_dc is None:
                continue
            s2_years = (int(_s2_dc["date_ints"][0]) // 10000,
                        int(_s2_dc["date_ints"][-1]) // 10000)

            if sat_dir not in self._label_cache:
                continue
            sm_np, depths, times, qc_np = self._label_cache[sat_dir]

            for year in self.years:
                if not (era5_start_year <= year <= era5_end_year):
                    continue
                if s2_years is None or not (s2_years[0] <= year <= s2_years[1]):
                    continue

                year_mask    = times.year == year
                if not year_mask.any():
                    continue

                year_indices = np.where(year_mask)[0]
                # Only train on directly observed values (qc==0); gap-filled (qc==1) excluded
                if qc_np is not None:
                    valid_days = np.any(qc_np[:, year_indices] == 0, axis=0)
                else:
                    valid_days = np.any(~np.isnan(sm_np[:, year_indices]), axis=0)
                if valid_days.sum() < min_obs:
                    continue

                for day_idx in np.where(valid_days)[0]:
                    doy = times[year_indices[day_idx]].day_of_year
                    self.samples.append({
                        "sat_dir"    : sat_dir,
                        "year"       : year,
                        "doy"        : doy,
                        "time_idx"   : year_indices[day_idx],
                        "station_key": dir_name,
                    })

        print(f"Dataset: {len(self.samples)} samples from "
              f"{len(set(s['station_key'] for s in self.samples))} stations")
        if self._use_mmap:
            n_mmap = sum(len(v) for v in self._l369_cache.values())
            print(f"L369 memmap cache: {n_mmap} arrays opened (OS page cache; zero process heap cost)")

    def __len__(self):
        return len(self.samples)

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]

    def __getitem__(self, idx):
        s       = self.samples[idx]
        sat_dir = s["sat_dir"]
        year    = s["year"]
        doy     = s["doy"]
        zg      = self._zarr_groups.get(sat_dir)   # zarr group or None

        if zg is not None:
            # ── Zarr path — all GPFS data served from precomputed caches ──
            _l12     = self._l12_cache.get(sat_dir, {})
            _dc      = self._zarr_date_cache.get(sat_dir, {})
            _cm_tm   = self._cm_token_mask_cache.get(sat_dir)
            _s1_tm   = self._s1_token_mask_cache.get(sat_dir, {})

            s2_pyr, s2_doys, s2_valid, s2_rel_pos = \
                load_s2_rolling_zarr(zg, year, doy,
                                     l12_np=_l12.get("s2"),
                                     date_cache=_dc,
                                     cm_token_mask=_cm_tm,
                                     training=self.training)

            s1_pyr, s1_doys, s1_valid, s1_rel_pos = \
                load_s1_rolling_zarr(zg, year, doy,
                                     l12_asc_np=_l12.get("s1_asc"),
                                     l12_desc_np=_l12.get("s1_desc"),
                                     date_cache=_dc,
                                     s1_token_mask_cache=_s1_tm,
                                     training=self.training)

            _static  = self._static_cache.get(sat_dir, {})
            dem_pyr  = _static.get("dem_pyr",  _cpu_pyramid_pool(
                           _static.get("dem",  torch.zeros(196, 768, dtype=torch.float16)).unsqueeze(0),
                           _static.get("dem_token_mask", torch.ones(14, 14, dtype=torch.bool)).unsqueeze(0)
                       ).squeeze(0))   # (4, 768) — pre-computed at init, fallback for missing cache
            lulc_pyr = _static.get("lulc_pyr", _cpu_pyramid_pool(
                           _static.get("lulc", torch.zeros(196, 768, dtype=torch.float16)).unsqueeze(0),
                           _static.get("lulc_token_mask", torch.ones(14, 14, dtype=torch.bool)).unsqueeze(0)
                       ).squeeze(0))   # (4, 768)

            anchor_l3, anchor_l6, anchor_l9, anchor_l12, anchor_rel_pos, anchor_orbit = \
                select_anchor_zarr(zg, year, doy,
                                   l12_cache=_l12,
                                   l369_cache=self._l369_cache.get(sat_dir),
                                   date_cache=_dc,
                                   cm_token_mask=_cm_tm,
                                   s1_token_mask_cache=_s1_tm)

        # ── Soil patch (static, from cache) ──────────────────────────
        soil_patch = self._static_cache.get(sat_dir, {}).get(
            "soil", torch.zeros(21, 74, 74, dtype=torch.float32)
        )

        # ── ERA5 — rolling 365-day window, numpy slice from cache ─────
        # load_era5_rolling returns numpy directly; single torch.from_numpy at end
        era5_np, era5_doys_np = load_era5_rolling(self._era5_cache.get(sat_dir), year, doy)
        if self._era5_log1p_prec:
            era5_np[:, PREC_IDX] = np.log1p(era5_np[:, PREC_IDX].clip(0))
        era5_np  = (era5_np - self._era5_means) / (self._era5_stds + 1e-8)
        era5     = torch.from_numpy(era5_np)
        era5_doys = torch.from_numpy(era5_doys_np)

        # Mask 15% of ERA5 timesteps during training — forces temporal generalisation.
        # Only masks non-padded slots (era5_doys > 0); never applied at val/test time.
        if self.training:
            valid_slots = era5_doys > 0
            mask = (torch.rand(era5.shape[0]) < 0.15) & valid_slots
            era5[mask] = 0.0
            era5_doys[mask] = 0   # treated as padding by the transformer

        # ── SIF — optional sparse modality, numpy slice from cache ───
        sif_vals, sif_doys, sif_rel_pos, sif_valid = load_sif_rolling(
            self._sif_cache.get(sat_dir), year, doy
        )
        if self.training and random.random() < 0.5:
            sif_valid[:] = False

        # ── TWSA — optional sparse modality, numpy slice from cache ──
        twsa_vals, twsa_doys, twsa_rel_pos, twsa_valid = load_twsa_rolling(
            self._twsa_cache.get(sat_dir), year, doy
        )
        if self.training and random.random() < 0.5:
            twsa_valid[:] = False

        # ── ISMN labels — observed values only (qc==0) ───────────────
        sm_np, depths, _, qc_np = self._label_cache[s["sat_dir"]]
        label = torch.full((len(SM_DEPTHS),), float("nan"), dtype=torch.float32)
        for i, depth_str in enumerate(SM_DEPTHS):
            if depth_str in depths:
                d_idx = depths.index(depth_str)
                if qc_np is None or qc_np[d_idx, s["time_idx"]] == 0:
                    label[i] = float(sm_np[d_idx, s["time_idx"]])

        return {
            # S2 — pyramid-pooled on CPU (30 MB → ~2 MB per sample across IPC)
            "s2_pyr"        : s2_pyr,            # (MAX_S2, 4, 768) fp32
            "s2_doys"       : s2_doys,           # (MAX_S2,) long
            "s2_valid"      : s2_valid,          # (MAX_S2,) bool
            "s2_rel_pos"    : s2_rel_pos,        # (MAX_S2,) long

            # S1 — pyramid-pooled on CPU
            "s1_pyr"        : s1_pyr,            # (MAX_S1, 4, 768) fp32
            "s1_doys"       : s1_doys,           # (MAX_S1,) long
            "s1_valid"      : s1_valid,          # (MAX_S1,) bool
            "s1_rel_pos"    : s1_rel_pos,        # (MAX_S1,) long

            # Static — pyramid-pooled on CPU (consistent with S2/S1)
            "dem_pyr"       : dem_pyr,           # (4, 768) fp32
            "lulc_pyr"      : lulc_pyr,          # (4, 768) fp32

            # Anchor (best clear + recent acquisition) — used for U-Net skip + spatial target
            "anchor_l3"       : anchor_l3,         # (196, 768) fp16
            "anchor_l6"       : anchor_l6,         # (196, 768) fp16
            "anchor_l9"       : anchor_l9,         # (196, 768) fp16
            "anchor_l12"      : anchor_l12,        # (196, 768) fp16
            "anchor_rel_pos"  : anchor_rel_pos,    # () long
            "anchor_orbit"    : anchor_orbit,      # () long  0=S2, 1=S1_asc, 2=S1_desc

            # Soil (static)
            "soil_patch"    : soil_patch,        # (21, 74, 74) fp32 — NaN-free

            # ERA5 (z-scored)
            "era5"          : era5,              # (365, 19) fp32
            "era5_doys"     : era5_doys,         # (365,) long

            # SIF (optional, sparse)
            "sif"           : sif_vals,          # (MAX_SIF, 1) fp32
            "sif_doys"      : sif_doys,          # (MAX_SIF,) long
            "sif_rel_pos"   : sif_rel_pos,       # (MAX_SIF,) long
            "sif_valid"     : sif_valid,         # (MAX_SIF,) bool

            # TWSA (optional, sparse)
            "twsa"          : twsa_vals,         # (MAX_TWSA, 1) fp32
            "twsa_doys"     : twsa_doys,         # (MAX_TWSA,) long
            "twsa_rel_pos"  : twsa_rel_pos,      # (MAX_TWSA,) long
            "twsa_valid"    : twsa_valid,        # (MAX_TWSA,) bool

            # Labels
            "label"         : label,             # (3,) — NaN where depth absent

            # Debug identifiers (non-tensor; for NaN tracing)
            "station_key"   : s["station_key"],
            "year"          : s["year"],
            "doy"           : s["doy"],
        }
