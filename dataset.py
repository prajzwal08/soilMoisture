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

# Token grid is 14x14 over a 224px tile, so patch k covers pixels [16k, 16k+16).
# Every tile is station-centred at (112,112), hence the supervised token is
# always index 105 = (112//16)*14 + (112//16). §35.8.
TOKEN_GRID   = 14
N_TOKENS     = TOKEN_GRID * TOKEN_GRID          # 196
STATION_TOKEN = (112 // 16) * TOKEN_GRID + (112 // 16)   # 105

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


def _token_slice(token_sel):
    """A basic slice when the selection is contiguous, else None.

    Contiguity matters: `arr[i, slice, :]` on a numpy memmap faults in only the pages the
    slice touches, whereas fancy indexing materialises through a temporary. Both live
    selections are contiguous -- [STATION_TOKEN] and arange(196).
    """
    sel = np.asarray(token_sel)
    if sel.size == 1 or np.all(np.diff(sel) == 1):
        return slice(int(sel[0]), int(sel[-1]) + 1)
    return None


def _read_patch_tokens(src, i: int, tsl, sel):
    """Acquisition i, patches `sel` only -> (K, 768).

    THE point of the patchwise loader. The old code did `src[i]` -- a full (196,768) fp16
    slab, 294 KB spanning ~72 memmap pages -- and then threw 195/196 of it away. l12 is
    C-contiguous, so patch k is a contiguous 768-float run: 1.5 KB, one page.
    """
    if tsl is not None:
        return np.asarray(src[i, tsl, :])
    return np.asarray(src[i])[sel]


def _finalise_history(l12, token_mask, doys, rel_pos, training,
                      token_sel, dropout_p: float = 0.0):
    """Turn the (T,K,768) buffer into what the model wants.

    Returns (feat (T,K,768) fp16, doys, valid_acq, rel_pos, hist_valid (T,K)).

    Two correctness points that bit the first draft (§35.11):
      * `token_mask` is (T,14,14), NOT (T,196) -- indexing it directly with token indices
        silently selects ROWS and returns (T,K,14). Hence the explicit reshape.
      * `token_mask` is initialised to ones and written only for slots that were actually
        filled AND matched a cloud-mask date. Padded slots (median 36 S2 acquisitions per
        station-year against MAX_S2=60), NaN acquisitions, and dates with no cloud-mask
        entry therefore read back VALID. Poisonous as a per-patch key mask -- hence the
        explicit `& valid_acq`.
    """
    valid_acq = doys > 0                                    # (T,)
    if training and dropout_p > 0:
        keep = torch.rand(token_mask.shape, dtype=torch.float32) >= dropout_p
        token_mask = token_mask & keep

    T  = token_mask.shape[0]
    tm = token_mask.reshape(T, N_TOKENS)[:, token_sel]      # (T,K)
    tm = tm & valid_acq[:, None]                            # padded/NaN -> invalid
    return l12, doys, valid_acq, rel_pos, tm


def _empty_history(max_acq: int, token_sel):
    """Zero return for a station with no acquisition in window.

    Must match the shape of the normal path or default_collate raises "stack expects each
    tensor to be equal size" on any batch mixing a station that has S2/S1 in window with one
    that does not (§35.11).
    """
    doys    = torch.zeros(max_acq, dtype=torch.long)
    rel_pos = torch.zeros(max_acq, dtype=torch.long)
    K       = len(token_sel)
    return (torch.zeros(max_acq, K, 768, dtype=torch.float16),
            doys, doys > 0, rel_pos, torch.zeros(max_acq, K, dtype=torch.bool))


def load_s2_rolling_zarr(zg: zarr.Group, year: int, target_doy: int,
                          max_acq: int = MAX_S2,
                          l12_np: np.ndarray | None = None,
                          date_cache: dict | None = None,
                          cm_token_mask: np.ndarray | None = None,
                          training: bool = False,
                          token_sel=None,
                          patch_token_dropout: float = 0.5):
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
    sel        = np.asarray(token_sel)
    tsl        = _token_slice(sel)
    K          = len(sel)
    l12        = torch.zeros(max_acq, K, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    # (14,14) of bools is 196 BYTES against 294 KB of tokens, so it stays full and is
    # indexed down in _finalise_history. It was never the reason for the wide read.
    token_mask = torch.ones(max_acq, 14, 14, dtype=torch.bool)
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)

    if "s2/l12" not in zg:
        return _empty_history(max_acq, token_sel)

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

        # Narrowed read: patch k only. BEHAVIOUR CHANGE, recorded in §35.22 -- the NaN test
        # now sees only this patch, where it used to see the whole tile and drop the whole
        # acquisition if any of the 196 patches was NaN. Discarding an acquisition because a
        # far corner of the tile is bad is not defensible for a per-patch model.
        tok = torch.from_numpy(_read_patch_tokens(tokens_z, src_i, tsl, sel))
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

    return _finalise_history(l12, token_mask, doys, rel_pos, training,
                             token_sel, patch_token_dropout)


def load_s1_rolling_zarr(zg: zarr.Group, year: int, target_doy: int,
                          max_acq: int = MAX_S1,
                          l12_asc_np: np.ndarray | None = None,
                          l12_desc_np: np.ndarray | None = None,
                          date_cache: dict | None = None,
                          s1_token_mask_cache: dict | None = None,
                          training: bool = False,
                          token_sel=None,
                          patch_token_dropout: float = 0.5):
    """Load S1 L12 tokens (ASC + DESC merged) from zarr and compress to pyramid tokens.

    date_cache:          precomputed per-orbit date info (eliminates zarr date reads)
    s1_token_mask_cache: precomputed {orbit: (N,14,14) bool} from _s1_token_mask_cache
    l12_asc_np / l12_desc_np: preloaded RAM arrays; eliminates L12 chunk reads.
    training:            if True, applies 50% random spatial token dropout before pooling.
    Returns (pyr, doys, valid, rel_pos) 4-tuple matching S2's signature.
    """
    sel        = np.asarray(token_sel)
    tsl        = _token_slice(sel)
    K          = len(sel)
    l12        = torch.zeros(max_acq, K, 768, dtype=torch.float16)
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
        return _empty_history(max_acq, token_sel)

    entries.sort(key=lambda x: x[0])
    entries = entries[-max_acq:]

    out_i = 0
    for date_str, tokens_z, src_i, orbit_key, cached_doy, cached_year in entries:
        tok = torch.from_numpy(_read_patch_tokens(tokens_z, src_i, tsl, sel))
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

    return _finalise_history(l12, token_mask, doys, rel_pos, training,
                             token_sel, patch_token_dropout)


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
        # Both must exist: the preloader creates the .bin via np.memmap(mode="w+") BEFORE
        # writing .meta.json, so a rank-0 death between those two statements leaves a bin
        # with no meta — and the preloader's own resume check is `if bin_path.exists()`,
        # so it never repairs it. Checking only the bin here would then raise
        # FileNotFoundError and kill all four ranks at dataset init.
        if not (bin_path.exists() and meta_path.exists()):
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
        token_sel:       str         = "station",
        patch_token_dropout: float   = 0.0,
    ):
        self.training   = training

        # Which patches to read. This is the ONLY place the store is narrowed, and everything
        # downstream inherits it: the loaders allocate (T,K,768) and read K rows per
        # acquisition rather than all 196 (§35.22).
        #   "station" -> patch 105 only, the supervised token. Training uses this.
        #   "all"     -> all 196, for 14x14 map figures. INFERENCE ONLY: it restores a
        #                ~30 MB/sample IPC payload, so cap the eval batch size.
        if token_sel == "station":
            self._token_sel = np.array([STATION_TOKEN], dtype=np.int64)
        elif token_sel == "all":
            self._token_sel = np.arange(N_TOKENS, dtype=np.int64)
        else:
            raise ValueError(f"token_sel must be 'station' or 'all', got {token_sel!r}")
        # The old 50% spatial token dropout degraded a POOLED mean, which is a mild
        # augmentation. Here it would delete half of patch k's acquisitions outright, so it
        # defaults OFF and has to be asked for.
        self._patch_token_dropout = patch_token_dropout
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

            s2_hist, s2_doys, s2_valid, s2_rel_pos, s2_hist_valid = \
                load_s2_rolling_zarr(zg, year, doy,
                                     l12_np=_l12.get("s2"),
                                     date_cache=_dc,
                                     cm_token_mask=_cm_tm,
                                     training=self.training,
                                     token_sel=self._token_sel,
                                     patch_token_dropout=self._patch_token_dropout)

            s1_hist, s1_doys, s1_valid, s1_rel_pos, s1_hist_valid = \
                load_s1_rolling_zarr(zg, year, doy,
                                     l12_asc_np=_l12.get("s1_asc"),
                                     l12_desc_np=_l12.get("s1_desc"),
                                     date_cache=_dc,
                                     s1_token_mask_cache=_s1_tm,
                                     training=self.training,
                                     token_sel=self._token_sel,
                                     patch_token_dropout=self._patch_token_dropout)

            # DEM/LULC enter patch k's sequence DIRECTLY, not as four nested-window means.
            # §27a.2 measured that pooling retains 1.5% (DEM) / 2.6% (LULC) of within-tile
            # variance -- that destruction is the defect this whole build exists to remove.
            _static  = self._static_cache.get(sat_dir, {})
            _sel     = self._token_sel
            dem_tok  = _static.get("dem",  torch.zeros(N_TOKENS, 768, dtype=torch.float16))[_sel]
            lulc_tok = _static.get("lulc", torch.zeros(N_TOKENS, 768, dtype=torch.float16))[_sel]


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
            # ── Per-patch satellite history — the point of the whole architecture ──
            # (T, K, 768) fp16, K=1 in training (patch 105). Only these K patches were ever
            # read from the store; the other 195 never enter the process (§35.22).
            "s2_hist"       : s2_hist,           # (MAX_S2, K, 768) fp16
            "s2_hist_valid" : s2_hist_valid,     # (MAX_S2, K) bool  — cloud mask AND doy>0
            "s2_doys"       : s2_doys,           # (MAX_S2,) long
            "s2_valid"      : s2_valid,          # (MAX_S2,) bool    — acquisition-level
            "s2_rel_pos"    : s2_rel_pos,        # (MAX_S2,) long    — staleness, 364 = today

            "s1_hist"       : s1_hist,           # (MAX_S1, K, 768) fp16
            "s1_hist_valid" : s1_hist_valid,     # (MAX_S1, K) bool
            "s1_doys"       : s1_doys,           # (MAX_S1,) long
            "s1_valid"      : s1_valid,          # (MAX_S1,) bool
            "s1_rel_pos"    : s1_rel_pos,        # (MAX_S1,) long

            # ── Per-patch statics ─────────────────────────────────────────
            "dem_tok"       : dem_tok,           # (K, 768) fp16
            "lulc_tok"      : lulc_tok,          # (K, 768) fp16
            "token_idx"     : torch.from_numpy(self._token_sel),   # (K,) long
            "token_valid"   : torch.ones(len(self._token_sel), dtype=torch.bool),

            # ── Tile-level drivers: identical for every patch, hence cacheable ──
            "soil_patch"    : soil_patch,        # (21, 74, 74) fp32 — NaN-free
            "era5"          : era5,              # (365, 19) fp32 — z-scored
            "era5_doys"     : era5_doys,         # (365,) long
            "sif"           : sif_vals,          # (MAX_SIF, 1) fp32
            "sif_doys"      : sif_doys,          # (MAX_SIF,) long
            "sif_rel_pos"   : sif_rel_pos,       # (MAX_SIF,) long
            "sif_valid"     : sif_valid,         # (MAX_SIF,) bool
            "twsa"          : twsa_vals,         # (MAX_TWSA, 1) fp32
            "twsa_doys"     : twsa_doys,         # (MAX_TWSA,) long
            "twsa_rel_pos"  : twsa_rel_pos,      # (MAX_TWSA,) long
            "twsa_valid"    : twsa_valid,        # (MAX_TWSA,) bool

            # ── Labels and identity ───────────────────────────────────────
            "label"         : label,             # (3,) — NaN where the depth has no obs
            "station_key"   : s["station_key"],
            "year"          : s["year"],
            "doy"           : s["doy"],
        }
