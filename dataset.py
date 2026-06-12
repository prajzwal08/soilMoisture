"""
SoilMoistureDataset
====================
Loads pre-computed TerraMind features, ERA5-Land meteo, and ISMN labels
for one station × year × day-of-year sample.

Directory layout (consolidated format):
  {data_dir}/{sm_only|sm_and_flux|flux_only}/{station}/
      S2L2A/   {station}_L{3,6,9,12}_{start}_{end}.pt   dict{tokens[N,196,768], dates, layer, geo}
      S1RTC/   {station}_{ASC|DESC}_L{3,6,9,12}_{start}_{end}.pt
      DEM/     dem_L12.pt                                plain tensor [196,768] fp16
      LULC/    lulc_L12.pt                               plain tensor [196,768] fp16
      CloudMask/ {station}_{start}_{end}.pt              dict{masks[N,H,W] uint8, dates, geo}
      ERA5Land/  meteo_{start}_{end}.nc
      SIF/       sif_{start}_{end}.nc
      TWSA/      twsa_{start}_{end}.nc
      labels.nc

See data_structure.txt for full schema.
"""

import json
import os
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


def _load_pt(path: Path) -> dict:
    """Load a consolidated .pt dict, using mmap when available (PyTorch ≥ 2.1)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def _year_range_from_stem(stem: str) -> tuple[int, int]:
    """Parse start/end year from a consolidated filename stem.
    e.g. 'ISMN_X_L12_20160120_20241215' → (2016, 2024)
         'meteo_20140101_20241231'        → (2014, 2024)
    """
    parts = stem.split("_")
    return int(parts[-2][:4]), int(parts[-1][:4])


def _date_to_int(dt) -> int:
    """Convert datetime-like to YYYYMMDD int."""
    return dt.year * 10000 + dt.month * 100 + dt.day


def _window_ints(year: int, target_doy: int) -> tuple[int, int]:
    """Return (start_int, end_int) as YYYYMMDD ints for the 365-day rolling window."""
    ws, td = _window_datetimes(year, target_doy)
    return _date_to_int(ws), _date_to_int(td)


def _resolve_pt_paths(sat_dir: Path) -> dict:
    """
    Pre-resolve all consolidated .pt file paths for a station.
    Called once per station in __init__; eliminates all .glob() calls from __getitem__.
    """
    def _first(d: Path, pattern: str) -> Path | None:
        if not d.exists():
            return None
        matches = sorted(d.glob(pattern))
        return matches[0] if matches else None

    s2  = sat_dir / "S2L2A"
    s1  = sat_dir / "S1RTC"
    cm  = sat_dir / "CloudMask"
    dem = sat_dir / "DEM"  / "dem_L12.pt"
    lc  = sat_dir / "LULC" / "lulc_L12.pt"

    return {
        "s2_l12"      : _first(s2, "*_L12_*.pt"),
        "s2_l3"       : _first(s2, "*_L3_*.pt"),
        "s2_l6"       : _first(s2, "*_L6_*.pt"),
        "s2_l9"       : _first(s2, "*_L9_*.pt"),
        "cm"          : _first(cm, "*_*.pt"),
        "s1_asc_l12"  : _first(s1, "*_ASC_L12_*.pt"),
        "s1_asc_l3"   : _first(s1, "*_ASC_L3_*.pt"),
        "s1_asc_l6"   : _first(s1, "*_ASC_L6_*.pt"),
        "s1_asc_l9"   : _first(s1, "*_ASC_L9_*.pt"),
        "s1_desc_l12" : _first(s1, "*_DESC_L12_*.pt"),
        "s1_desc_l3"  : _first(s1, "*_DESC_L3_*.pt"),
        "s1_desc_l6"  : _first(s1, "*_DESC_L6_*.pt"),
        "s1_desc_l9"  : _first(s1, "*_DESC_L9_*.pt"),
        "dem_l12"     : dem if dem.exists() else None,
        "lulc_l12"    : lc  if lc.exists()  else None,
    }


# ── Per-station NC pre-loaders (called once in __init__, not in __getitem__) ─

def _load_era5_nc(era5_dir: Path):
    """Load ERA5Land NC fully into memory. Returns (values (N,19), date_ints (N,), doys (N,)) or None."""
    nc_files = sorted(era5_dir.glob("meteo_*_*.nc")) if era5_dir.exists() else []
    if not nc_files:
        return None
    ds = xr.open_dataset(nc_files[0])
    values = np.stack([ds[v].values for v in ERA5_VARS], axis=-1).astype(np.float32)
    times  = pd.DatetimeIndex(ds["time"].values)
    ds.close()
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.day_of_year for t in times], dtype=np.int32)
    return values, date_ints, doys


def _load_sif_nc(sif_dir: Path):
    """Load SIF NC fully into memory. Returns (values (N,), date_ints (N,), doys (N,)) or None."""
    nc_files = sorted(sif_dir.glob("sif_*_*.nc")) if sif_dir.exists() else []
    if not nc_files:
        return None
    ds = xr.open_dataset(nc_files[0])
    times  = pd.to_datetime(ds["time"].values)
    values = ds["sif"].values.astype(np.float32)
    ds.close()
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    return values, date_ints, doys


def _load_twsa_nc(twsa_dir: Path):
    """Load TWSA NC fully into memory. Returns (values (N,), date_ints (N,), doys (N,)) or None."""
    nc_files = sorted(twsa_dir.glob("twsa_*_*.nc")) if twsa_dir.exists() else []
    if not nc_files:
        return None
    ds = xr.open_dataset(nc_files[0])
    times  = pd.to_datetime(ds["time"].values)
    values = ds["lwe"].values.astype(np.float32)
    ds.close()
    date_ints = np.array([_date_to_int(t) for t in times], dtype=np.int32)
    doys      = np.array([t.timetuple().tm_yday for t in times], dtype=np.int32)
    return values, date_ints, doys


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
    """Load labels from zarr → same tuple format as label_cache."""
    if "labels/sm" not in zg:
        return None
    sm_np  = zg["labels/sm"][:]                        # (n_depths, n_days) float32
    depths = [str(d) for d in zg["labels/depths"][:]]
    dates  = [str(d) for d in zg["labels/dates"][:]]
    times  = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return sm_np, depths, times


def load_s2_rolling_zarr(zg: zarr.Group, year: int, target_doy: int,
                          max_acq: int = MAX_S2,
                          l12_np: np.ndarray | None = None):
    """
    Load S2 L12 tokens and cloud mask from zarr for the 365-day rolling window.
    l12_np: if provided (preloaded into RAM), L12 tokens are served from RAM
    with zero disk I/O. Otherwise falls back to chunk reads from zarr.
    """
    l12        = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    token_mask = torch.ones(max_acq, 14, 14, dtype=torch.bool)
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)

    if "s2/l12" not in zg:
        return l12, doys, doys > 0, token_mask, rel_pos

    all_dates = [str(d) for d in zg["s2/dates"][:]]
    ws, td    = _window_datetimes(year, target_doy)
    win_idx   = [i for i, d in enumerate(all_dates) if _in_window(d, ws, td)]
    win_idx   = win_idx[-max_acq:]

    cm_date_idx: dict[str, int] = {}
    has_cm = "cm/masks" in zg and "cm/dates" in zg
    if has_cm:
        cm_date_idx = {str(d): i for i, d in enumerate(zg["cm/dates"][:])}

    tokens_z = l12_np if l12_np is not None else zg["s2/l12"]
    cm_z     = zg["cm/masks"] if has_cm else None

    for out_i, src_i in enumerate(win_idx):
        date_str = all_dates[src_i]
        dt       = datetime.strptime(date_str[:8], "%Y%m%d")
        acq_doy  = dt.timetuple().tm_yday

        l12[out_i]     = torch.from_numpy(tokens_z[src_i])   # RAM slice or chunk read
        doys[out_i]    = acq_doy
        rel_pos[out_i] = _rel_pos(acq_doy, dt.year, target_doy, year)

        if date_str in cm_date_idx:
            cm    = cm_z[cm_date_idx[date_str]]               # one chunk read
            cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
            bad_frac = np.isin(cm_4d, [3, 4, 5, 255]).mean(axis=(1, 3))
            token_mask[out_i] = torch.from_numpy(bad_frac <= 0.01)

    return l12, doys, doys > 0, token_mask, rel_pos


def load_s1_rolling_zarr(zg: zarr.Group, year: int, target_doy: int,
                          max_acq: int = MAX_S1,
                          l12_asc_np: np.ndarray | None = None,
                          l12_desc_np: np.ndarray | None = None):
    """Load S1 L12 tokens (ASC + DESC merged) from zarr.
    l12_asc_np / l12_desc_np: preloaded RAM arrays; eliminates chunk reads for L12.
    Returns (l12, doys, valid, token_mask, rel_pos) matching S2's 5-tuple signature.
    """
    l12        = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    token_mask = torch.ones(max_acq, 14, 14, dtype=torch.bool)  # True = valid
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)

    ws, td  = _window_datetimes(year, target_doy)
    entries: list[tuple[str, object, int, str]] = []

    l12_np_map = {"s1_asc": l12_asc_np, "s1_desc": l12_desc_np}
    # Preload per-orbit token_mask arrays (small: N×14×14 bool)
    tm_np_map: dict[str, np.ndarray | None] = {}
    for orbit_key in ("s1_asc", "s1_desc"):
        mk = f"{orbit_key}/token_mask"
        tm_np_map[orbit_key] = np.asarray(zg[mk][:]) if mk in zg else None
        if f"{orbit_key}/l12" not in zg:
            continue
        orbit_dates = [str(d) for d in zg[f"{orbit_key}/dates"][:]]
        tokens_src  = l12_np_map[orbit_key] if l12_np_map[orbit_key] is not None \
                      else zg[f"{orbit_key}/l12"]
        for i, d in enumerate(orbit_dates):
            if _in_window(d, ws, td):
                entries.append((d, tokens_src, i, orbit_key))

    if not entries:
        return l12, doys, doys > 0, token_mask, rel_pos

    entries.sort(key=lambda x: x[0])
    entries = entries[-max_acq:]

    out_i = 0
    for date_str, tokens_z, src_i, orbit_key in entries:
        tok = torch.from_numpy(np.asarray(tokens_z[src_i]))   # one chunk read
        if torch.isnan(tok).any():
            # Corrupted TerraMind acquisition (whole-tile NaN) -- skip rather
            # than letting NaN propagate into the model.
            continue
        dt      = datetime.strptime(date_str[:8], "%Y%m%d")
        acq_doy = dt.timetuple().tm_yday
        l12[out_i]     = tok
        doys[out_i]    = acq_doy
        rel_pos[out_i] = _rel_pos(acq_doy, dt.year, target_doy, year)
        if tm_np_map[orbit_key] is not None:
            token_mask[out_i] = torch.from_numpy(tm_np_map[orbit_key][src_i])
        out_i += 1

    return l12, doys, doys > 0, token_mask, rel_pos


def select_anchor_zarr(zg: zarr.Group, year: int, target_doy: int,
                        l12_cache: dict | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select the best anchor acquisition for the U-Net skip connections and
    target spatial tokens.

    Priority:
      1. Most recent acquisition with ALL 196 patches valid (fully clear/no-NaN).
      2. If none fully clear, fall back to most recent acquisition regardless.

    Quality sources:
      S2  — cm/masks cloud raster (bad class ∈ {3,4,5,255}; bad_frac ≤ 1% → valid)
      S1  — s1_asc/token_mask | s1_desc/token_mask (written by compute_s1_dem_lulc_token_masks.py)
      When a quality array is absent, the acquisition is treated as fully valid.

    Returns:
      anchor_l3, anchor_l6, anchor_l9, anchor_l12 : (196, 768) fp16 tensors
      anchor_rel_pos                               : () long scalar tensor
      anchor_is_s1                                 : () bool scalar tensor
    All zero if no in-window acquisition found.
    """
    zeros  = torch.zeros(196, 768, dtype=torch.float16)
    ws, td = _window_datetimes(year, target_doy)

    # ── Preload S2 cloud-mask index ───────────────────────────────────────
    cm_date_idx: dict[str, int] = {}
    cm_z = None
    if "cm/masks" in zg and "cm/dates" in zg:
        cm_date_idx = {str(d): i for i, d in enumerate(zg["cm/dates"][:])}
        cm_z = zg["cm/masks"]

    # ── Preload S1 token-mask arrays (small: N×14×14 bool) ───────────────
    tm_np: dict[str, np.ndarray | None] = {}
    for ok in ("s1_asc", "s1_desc"):
        mk = f"{ok}/token_mask"
        tm_np[ok] = np.asarray(zg[mk][:]) if mk in zg else None

    # ── Collect all in-window candidates ─────────────────────────────────
    # Each entry: (date_str, orbit_key, src_idx, n_valid_patches [0..196])
    candidates: list[tuple[str, str, int, int]] = []

    for orbit in ("s2", "s1_asc", "s1_desc"):
        dates_key = f"{orbit}/dates"
        if dates_key not in zg or f"{orbit}/l12" not in zg:
            continue
        orbit_dates = [str(d) for d in zg[dates_key][:]]
        for i, d in enumerate(orbit_dates):
            if not _in_window(d, ws, td):
                continue
            if orbit == "s2":
                if d in cm_date_idx:
                    cm    = cm_z[cm_date_idx[d]]
                    cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
                    bad   = np.isin(cm_4d, [3, 4, 5, 255]).mean(axis=(1, 3))
                    n_valid = int((bad <= 0.01).sum())
                else:
                    n_valid = 196   # no cloud mask → assume clear
            else:
                if tm_np[orbit] is not None:
                    n_valid = int(tm_np[orbit][i].sum())
                else:
                    n_valid = 196   # mask not yet computed → assume clear
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
    dt      = datetime.strptime(best_date[:8], "%Y%m%d")
    acq_doy = dt.timetuple().tm_yday
    rp      = _rel_pos(acq_doy, dt.year, target_doy, year)
    orbit_id = {"s2": 0, "s1_asc": 1, "s1_desc": 2}[best_orbit]

    # ── Load L3 / L6 / L9 / L12 for chosen anchor ────────────────────────
    def _load_layer(layer: str) -> torch.Tensor:
        key = f"{best_orbit}/{layer}"
        if key not in zg or best_idx >= zg[key].shape[0]:
            return zeros.clone()
        return torch.from_numpy(np.asarray(zg[key][best_idx]))

    # Use preloaded L12 RAM cache (zero disk I/O) when available
    l12_np_for_orbit = (l12_cache or {}).get(
        best_orbit if best_orbit == "s2" else best_orbit   # "s2"|"s1_asc"|"s1_desc"
    )
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


def load_soil_patch(path: Path) -> torch.Tensor | None:
    """
    Load soil_patch.tif → (21, 74, 74) float32 tensor, NaN-filled.
    Returns None if the file does not exist.
    """
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        patch  = src.read().astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        patch[patch == nodata] = np.nan
    patch = fill_soil_nans(patch)
    return torch.from_numpy(patch)


# ── ERA5 rolling slicer (no file I/O — works on pre-loaded numpy arrays) ────

def load_era5_rolling(cache_entry, year: int, target_doy: int):
    """
    Slice pre-loaded ERA5 arrays for the 365-day rolling window.

    Args:
        cache_entry: (values (N,19) float32, date_ints (N,) int32, doys (N,) int32) or None
    Returns:
        era5 : (365, 19) float32
        doys : (365,) long
    """
    if cache_entry is None:
        return (torch.zeros(365, len(ERA5_VARS), dtype=torch.float32),
                torch.zeros(365, dtype=torch.long))

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
    return torch.from_numpy(out_era5), torch.from_numpy(out_doys)


# ── L12 loaders ──────────────────────────────────────────────────────────────

def load_s2_rolling(l12_pt: Path | None, cm_pt: Path | None,
                    year: int, target_doy: int,
                    max_acq: int = MAX_S2):
    """
    Load pre-computed S2 L12 features within the 365-day rolling window.

    Args:
        l12_pt, cm_pt: pre-resolved paths from _resolve_pt_paths() — no glob in hot loop
    Returns:
        l12        : (max_acq, 196, 768) float16
        doys       : (max_acq,) long
        valid      : (max_acq,) bool
        token_mask : (max_acq, 14, 14) bool — True = cloud-free token
        rel_pos    : (max_acq,) long
    """
    l12        = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    token_mask = torch.ones(max_acq, 14, 14, dtype=torch.bool)
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)

    if l12_pt is None:
        return l12, doys, doys > 0, token_mask, rel_pos

    data       = _load_pt(l12_pt)
    tokens_all = data["tokens"]
    all_dates  = data["dates"]

    ws, td = _window_datetimes(year, target_doy)
    win_idx = [i for i, d in enumerate(all_dates) if _in_window(d, ws, td)]
    win_idx = win_idx[-max_acq:]

    cm_masks    = None
    cm_date_idx: dict[str, int] = {}
    if cm_pt is not None:
        cm_data     = _load_pt(cm_pt)
        cm_masks    = cm_data["masks"]
        cm_date_idx = {d: i for i, d in enumerate(cm_data["dates"])}

    for out_i, src_i in enumerate(win_idx):
        date_str = all_dates[src_i]
        dt       = datetime.strptime(date_str[:8], "%Y%m%d")
        acq_doy  = dt.timetuple().tm_yday

        l12[out_i]     = tokens_all[src_i]
        doys[out_i]    = acq_doy
        rel_pos[out_i] = _rel_pos(acq_doy, dt.year, target_doy, year)

        if cm_masks is not None and date_str in cm_date_idx:
            cm    = cm_masks[cm_date_idx[date_str]].numpy()           # (224, 224) uint8
            cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
            # Valid classes: 0=land, 1=water, 2=snow/ice
            # Invalid: 3=thin cloud, 4=thick cloud, 5=shadow, 255=nodata
            # Token valid if ≤1% of its 256 pixels are in the invalid classes
            bad_frac = np.isin(cm_4d, [3, 4, 5, 255]).mean(axis=(1, 3))   # (14, 14)
            token_mask[out_i] = torch.from_numpy(bad_frac <= 0.01)

    return l12, doys, doys > 0, token_mask, rel_pos


def load_s1_rolling(asc_l12_pt: Path | None, desc_l12_pt: Path | None,
                    year: int, target_doy: int,
                    max_acq: int = MAX_S1):
    """
    Load pre-computed S1 L12 features (ASC + DESC merged) within the 365-day window.

    Args:
        asc_l12_pt, desc_l12_pt: pre-resolved paths from _resolve_pt_paths()
    Returns:
        l12     : (max_acq, 196, 768) float16
        doys    : (max_acq,) long
        valid   : (max_acq,) bool
        rel_pos : (max_acq,) long
    """
    l12     = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys    = torch.zeros(max_acq, dtype=torch.long)
    rel_pos = torch.zeros(max_acq, dtype=torch.long)

    ws, td = _window_datetimes(year, target_doy)
    entries: list[tuple[str, torch.Tensor, int]] = []
    for pt in (asc_l12_pt, desc_l12_pt):
        if pt is None:
            continue
        data = _load_pt(pt)
        for i, d in enumerate(data["dates"]):
            if _in_window(d, ws, td):
                entries.append((d, data["tokens"], i))

    if not entries:
        return l12, doys, doys > 0, rel_pos

    entries.sort(key=lambda x: x[0])
    entries = entries[-max_acq:]

    for out_i, (date_str, tokens_all, src_i) in enumerate(entries):
        dt      = datetime.strptime(date_str[:8], "%Y%m%d")
        acq_doy = dt.timetuple().tm_yday
        l12[out_i]     = tokens_all[src_i]
        doys[out_i]    = acq_doy
        rel_pos[out_i] = _rel_pos(acq_doy, dt.year, target_doy, year)

    return l12, doys, doys > 0, rel_pos


# ── Skip-connection feature loader ───────────────────────────────────────────

def load_recent_skip_features(paths: dict, year: int, target_doy: int):
    """
    Load precomputed L3/L6/L9 skip features for the most-recent acquisition
    (S2 or S1) in the rolling 365-day window.

    Args:
        paths: pre-resolved paths dict from _resolve_pt_paths()
    Returns:
        skip_l3, skip_l6, skip_l9 : each (196, 768) float16 — zeros if unavailable
        recent_is_s1               : bool
    """
    zeros    = torch.zeros(196, 768, dtype=torch.float16)
    ws, td   = _window_datetimes(year, target_doy)

    def _most_recent(l12_pt: Path | None) -> str | None:
        if l12_pt is None:
            return None
        data = _load_pt(l12_pt)
        for d in reversed(data["dates"]):
            if _in_window(d, ws, td):
                return d
        return None

    s2_date  = _most_recent(paths.get("s2_l12"))
    s1_date  = None
    s1_orbit = None
    for orbit, key in (("asc", "s1_asc_l12"), ("desc", "s1_desc_l12")):
        d = _most_recent(paths.get(key))
        if d is not None and (s1_date is None or d > s1_date):
            s1_date, s1_orbit = d, orbit

    recent_is_s1 = bool(s1_date and (not s2_date or s1_date > s2_date))
    recent_date  = s1_date if recent_is_s1 else s2_date
    if recent_date is None:
        return zeros, zeros.clone(), zeros.clone(), False

    skips = []
    for layer in ("l3", "l6", "l9"):
        pt = paths.get(f"s1_{s1_orbit}_{layer}") if recent_is_s1 else paths.get(f"s2_{layer}")

        if pt is None:
            skips.append(zeros.clone())
            continue

        data = _load_pt(pt)
        if recent_date in data["dates"]:
            idx = data["dates"].index(recent_date)
            # Guard: some bundles have dates list longer than tokens tensor
            # (partial consolidation) — fall back to zeros rather than crash
            if idx < data["tokens"].shape[0]:
                skips.append(data["tokens"][idx])
            else:
                skips.append(zeros.clone())
        else:
            skips.append(zeros.clone())

    return skips[0], skips[1], skips[2], recent_is_s1


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
    for i in range(n_win):
        acq_year   = int(win_dates[i]) // 10000
        rel_pos[i] = _rel_pos(int(win_doys[i]), acq_year, target_doy, year)

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
    for i in range(n_win):
        acq_year   = int(win_dates[i]) // 10000
        rel_pos[i] = _rel_pos(int(win_doys[i]), acq_year, target_doy, year)

    return vals, doys, rel_pos, valid


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
    ):
        self.training  = training
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
        self._zarr_groups : dict[Path, zarr.Group | None]       = {}
        self._l12_cache   : dict[Path, dict[str, np.ndarray]]   = {}
        self._era5_cache  : dict[Path, tuple | None] = {}
        self._sif_cache   : dict[Path, tuple | None] = {}
        self._twsa_cache  : dict[Path, tuple | None] = {}
        self._label_cache : dict[Path, tuple]        = {}

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
                    if not DISABLE_L12_CACHE:
                        self._l12_cache[sat_dir] = {
                            k: zg[f"{k}/l12"][:]
                            for k in ("s2", "s1_asc", "s1_desc")
                            if f"{k}/l12" in zg
                        }
                    lc = _load_zarr_labels(zg)
                    if lc is not None:
                        self._label_cache[sat_dir] = lc

            # ERA5 year range from cache (fast int arithmetic — no file I/O)
            era5_entry = self._era5_cache[sat_dir]
            if era5_entry is None:
                continue
            era5_start_year = int(era5_entry[1][0])  // 10000
            era5_end_year   = int(era5_entry[1][-1]) // 10000

            # S2 year range: from zarr dates
            zg = self._zarr_groups.get(sat_dir)
            if zg is None or "s2/dates" not in zg:
                continue
            s2_dates_zarr = [str(d) for d in zg["s2/dates"][:]]
            s2_years = (int(s2_dates_zarr[0][:4]), int(s2_dates_zarr[-1][:4]))

            if sat_dir not in self._label_cache:
                continue
            sm_np, depths, times = self._label_cache[sat_dir]

            for year in self.years:
                if not (era5_start_year <= year <= era5_end_year):
                    continue
                if s2_years is None or not (s2_years[0] <= year <= s2_years[1]):
                    continue

                year_mask    = times.year == year
                if not year_mask.any():
                    continue

                year_indices = np.where(year_mask)[0]
                sm_slice     = sm_np[:, year_indices]
                valid_days   = np.any(~np.isnan(sm_slice), axis=0)
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

    def __getitem__(self, idx):
        s       = self.samples[idx]
        sat_dir = s["sat_dir"]
        year    = s["year"]
        doy     = s["doy"]
        zg      = self._zarr_groups.get(sat_dir)   # zarr group or None

        if zg is not None:
            # ── Zarr path — L12 served from RAM cache (zero disk I/O for history) ──
            _l12 = self._l12_cache.get(sat_dir, {})
            s2_l12, s2_doys, s2_valid, s2_token_mask, s2_rel_pos = \
                load_s2_rolling_zarr(zg, year, doy, l12_np=_l12.get("s2"))

            s1_l12, s1_doys, s1_valid, s1_token_mask, s1_rel_pos = \
                load_s1_rolling_zarr(zg, year, doy,
                                     l12_asc_np=_l12.get("s1_asc"),
                                     l12_desc_np=_l12.get("s1_desc"))

            dem_l12  = (torch.from_numpy(zg["dem"][:])
                        if "dem" in zg else torch.zeros(196, 768, dtype=torch.float16))
            lulc_l12 = (torch.from_numpy(zg["lulc"][:])
                        if "lulc" in zg else torch.zeros(196, 768, dtype=torch.float16))

            dem_token_mask  = (torch.from_numpy(np.asarray(zg["dem_token_mask"][:]))
                               if "dem_token_mask" in zg
                               else torch.ones(14, 14, dtype=torch.bool))
            lulc_token_mask = (torch.from_numpy(np.asarray(zg["lulc_token_mask"][:]))
                               if "lulc_token_mask" in zg
                               else torch.ones(14, 14, dtype=torch.bool))

            anchor_l3, anchor_l6, anchor_l9, anchor_l12, anchor_rel_pos, anchor_orbit = \
                select_anchor_zarr(zg, year, doy, l12_cache=_l12)

        # ── Soil patch (static, NaN-filled) ──────────────────────────
        if "soil" in zg:
            soil_patch = torch.from_numpy(fill_soil_nans(zg["soil"][:]))
        else:
            soil_patch = torch.zeros(21, 74, 74, dtype=torch.float32)

        # ── ERA5 — rolling 365-day window, numpy slice from cache ─────
        era5, era5_doys = load_era5_rolling(self._era5_cache.get(sat_dir), year, doy)

        # Z-score normalisation: log1p precipitation then standardise all vars
        era5_np = era5.numpy()
        if self._era5_log1p_prec:
            prec_idx = ERA5_VARS.index("tp_sum")
            era5_np[:, prec_idx] = np.log1p(era5_np[:, prec_idx].clip(0))
        era5_np = (era5_np - self._era5_means) / (self._era5_stds + 1e-8)
        era5    = torch.from_numpy(era5_np)

        # ── SIF — optional sparse modality, numpy slice from cache ───
        sif_vals, sif_doys, sif_rel_pos, sif_valid = load_sif_rolling(
            self._sif_cache.get(sat_dir), year, doy
        )
        # Modality dropout: per-sample coin flip using PyTorch RNG (worker-safe)
        if self.training and torch.rand(1).item() < 0.5:
            sif_valid[:] = False

        # ── TWSA — optional sparse modality, numpy slice from cache ──
        twsa_vals, twsa_doys, twsa_rel_pos, twsa_valid = load_twsa_rolling(
            self._twsa_cache.get(sat_dir), year, doy
        )
        if self.training and torch.rand(1).item() < 0.5:
            twsa_valid[:] = False

        # ── ISMN labels — index pre-loaded numpy array ────────────────
        sm_np, depths, _ = self._label_cache[s["sat_dir"]]
        label = torch.full((len(SM_DEPTHS),), float("nan"), dtype=torch.float32)
        for i, depth_str in enumerate(SM_DEPTHS):
            if depth_str in depths:
                d_idx    = depths.index(depth_str)
                label[i] = float(sm_np[d_idx, s["time_idx"]])

        return {
            # S2 — pre-computed L12 features
            "s2_l12"        : s2_l12,           # (MAX_S2, 196, 768) fp16
            "s2_doys"       : s2_doys,           # (MAX_S2,) long
            "s2_valid"      : s2_valid,          # (MAX_S2,) bool
            "s2_token_mask" : s2_token_mask,     # (MAX_S2, 14, 14) bool
            "s2_rel_pos"    : s2_rel_pos,        # (MAX_S2,) long

            # S1 — pre-computed L12 features
            "s1_l12"        : s1_l12,            # (MAX_S1, 196, 768) fp16
            "s1_doys"       : s1_doys,           # (MAX_S1,) long
            "s1_valid"      : s1_valid,          # (MAX_S1,) bool
            "s1_token_mask" : s1_token_mask,     # (MAX_S1, 14, 14) bool
            "s1_rel_pos"    : s1_rel_pos,        # (MAX_S1,) long

            # Static
            "dem_l12"       : dem_l12,           # (196, 768) fp16
            "lulc_l12"      : lulc_l12,          # (196, 768) fp16
            "dem_token_mask" : dem_token_mask,   # (14, 14) bool
            "lulc_token_mask": lulc_token_mask,  # (14, 14) bool

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
