"""
SoilMoistureDataset
====================
Loads pre-computed TerraMind features, ERA5-Land meteo, and ISMN labels
for one station × year × day-of-year sample.

Requires precompute_terramind.py to have been run first so that
  {satellite_dir}/{station}/S2L2A/{YYYYMMDD}_L{3,6,9,12}.pt  and
  {satellite_dir}/{station}/S1RTC/{stem}_L{3,6,9,12}.pt
exist. The raw .tif files are no longer read at training time.

Directory layout:
  {satellite_dir}/{network}_{station}/
      S2L2A/   YYYYMMDD_L{3,6,9,12}.pt  (196×768 fp16 per layer)
      S1RTC/   {stem}_L{3,6,9,12}.pt    (196×768 fp16 per layer)
      DEM/     dem_pyramid.pt            (4×768 fp32)
      CloudMask/ YYYYMMDD.tif                    (cloud labels)
      ERA5Land/  meteo_YYYY.nc

ISMN label files: {ismn_dir}/{network}_{station}_{start}_{end}.nc
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

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

SM_DEPTHS = ["0-10", "10-20", "20-40", "40-100"]  # n_depths = 4

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


def date_to_doy(stem: str, year: int):
    """Parse YYYYMMDD (possibly followed by _ASC etc.) → day-of-year, or None."""
    date_str = stem[:8]
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.timetuple().tm_yday if dt.year == year else None
    except ValueError:
        return None


def _rel_pos(acq_doy: int, acq_year: int, target_doy: int, target_year: int) -> int:
    """0-indexed position in the 365-day rolling window (0=oldest, 364=today)."""
    if acq_year == target_year:
        return (365 - target_doy) + (acq_doy - 1)
    else:
        return acq_doy - target_doy - 1


def _files_in_window(patch_dir: Path, year: int, target_doy: int) -> list[Path]:
    """
    Return sorted .tif files in the 365-day rolling window ending at target_doy.
    Chronological order (prev-year first, then curr-year); last entry = most recent.
    """
    if not patch_dir.exists():
        return []
    prev = sorted(
        f for f in patch_dir.glob("*.tif")
        if f.stem[:4] == str(year - 1)
        and date_to_doy(f.stem, year - 1) is not None
        and date_to_doy(f.stem, year - 1) > target_doy
    )
    curr = sorted(
        f for f in patch_dir.glob("*.tif")
        if f.stem[:4] == str(year)
        and date_to_doy(f.stem, year) is not None
        and date_to_doy(f.stem, year) <= target_doy
    )
    return prev + curr


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


# ── ERA5 loader ──────────────────────────────────────────────────────────────

def load_era5_rolling(sat_dir: Path, year: int, target_doy: int):
    """
    Load 365 ERA5 tokens for the rolling window [T-364, T].

    Returns:
        era5 : (365, 19) float32
        doys : (365,) long — actual calendar DoY of each token
    """
    n_curr = target_doy
    n_prev = 365 - target_doy

    curr_file = sat_dir / "ERA5Land" / f"meteo_{year}.nc"
    ds_curr   = xr.open_dataset(curr_file)
    era5_curr = np.stack([ds_curr[v].values[:n_curr] for v in ERA5_VARS], axis=-1).astype(np.float32)
    doys_curr = np.arange(1, n_curr + 1, dtype=np.int64)

    if n_prev == 0:
        return torch.from_numpy(era5_curr), torch.from_numpy(doys_curr)

    prev_file = sat_dir / "ERA5Land" / f"meteo_{year - 1}.nc"
    if prev_file.exists():
        ds_prev   = xr.open_dataset(prev_file)
        era5_prev = np.stack([ds_prev[v].values for v in ERA5_VARS], axis=-1).astype(np.float32)
        era5_prev = era5_prev[-n_prev:]
    else:
        era5_prev = np.zeros((n_prev, len(ERA5_VARS)), dtype=np.float32)
    doys_prev = np.arange(365 - n_prev + 1, 366, dtype=np.int64)

    era5 = np.concatenate([era5_prev, era5_curr], axis=0)
    doys = np.concatenate([doys_prev, doys_curr], axis=0)
    return torch.from_numpy(era5), torch.from_numpy(doys)


# ── L12 loaders (replace raw-patch loaders) ──────────────────────────────────

def load_s2_rolling(patch_dir: Path, cloud_mask_dir: Path,
                    year: int, target_doy: int,
                    max_acq: int = MAX_S2):
    """
    Load pre-computed S2 L12 features within the 365-day rolling window.

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

    all_tifs = _files_in_window(patch_dir, year, target_doy)

    for i, tif in enumerate(all_tifs[:max_acq]):
        l12_path = patch_dir / f"{tif.stem}_L12.pt"
        if not l12_path.exists():
            continue   # precompute not yet run for this acquisition → skip

        file_year = int(tif.stem[:4])
        acq_doy   = date_to_doy(tif.stem, file_year)

        l12[i]     = torch.load(l12_path, weights_only=True, map_location="cpu")
        doys[i]    = acq_doy
        rel_pos[i] = _rel_pos(acq_doy, file_year, target_doy, year)

        cm_path = cloud_mask_dir / f"{tif.stem[:8]}.tif"
        if cm_path.exists():
            with rasterio.open(cm_path) as src:
                cm = src.read(1).astype(np.uint8)
            cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
            token_mask[i] = torch.from_numpy((cm_4d == 0).all(axis=(1, 3)))

    valid = doys > 0
    return l12, doys, valid, token_mask, rel_pos


def load_s1_rolling(patch_dir: Path, year: int, target_doy: int,
                    max_acq: int = MAX_S1):
    """
    Load pre-computed S1 L12 features within the 365-day rolling window.

    Returns:
        l12     : (max_acq, 196, 768) float16
        doys    : (max_acq,) long
        valid   : (max_acq,) bool
        rel_pos : (max_acq,) long
    """
    l12     = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys    = torch.zeros(max_acq, dtype=torch.long)
    rel_pos = torch.zeros(max_acq, dtype=torch.long)

    all_tifs = _files_in_window(patch_dir, year, target_doy)

    for i, tif in enumerate(all_tifs[:max_acq]):
        l12_path = patch_dir / f"{tif.stem}_L12.pt"
        if not l12_path.exists():
            continue

        file_year = int(tif.stem[:4])
        acq_doy   = date_to_doy(tif.stem, file_year)

        l12[i]     = torch.load(l12_path, weights_only=True, map_location="cpu")
        doys[i]    = acq_doy
        rel_pos[i] = _rel_pos(acq_doy, file_year, target_doy, year)

    valid = doys > 0
    return l12, doys, valid, rel_pos


# ── Skip-connection feature loader ───────────────────────────────────────────

def load_recent_skip_features(sat_dir: Path, year: int, target_doy: int):
    """
    Load precomputed L3/L6/L9 skip features for the most-recent acquisition
    (S2 or S1) in the rolling 365-day window.

    Returns:
        skip_l3, skip_l6, skip_l9 : each (196, 768) float16 — zeros if unavailable
        recent_is_s1               : bool
    """
    s2_files = _files_in_window(sat_dir / "S2L2A", year, target_doy)
    s1_files = _files_in_window(sat_dir / "S1RTC", year, target_doy)

    # Determine which modality is more recent
    recent_is_s1 = False
    if s2_files and s1_files:
        s2_tif  = s2_files[-1]
        s1_tif  = s1_files[-1]
        s2_year = int(s2_tif.stem[:4])
        s1_year = int(s1_tif.stem[:4])
        s2_rel  = _rel_pos(date_to_doy(s2_tif.stem, s2_year), s2_year, target_doy, year)
        s1_rel  = _rel_pos(date_to_doy(s1_tif.stem, s1_year), s1_year, target_doy, year)
        recent_is_s1 = s1_rel > s2_rel
    elif s1_files:
        recent_is_s1 = True

    zeros         = torch.zeros(196, 768, dtype=torch.float16)
    recent_files  = s1_files if recent_is_s1 else s2_files
    if not recent_files:
        return zeros, zeros.clone(), zeros.clone(), recent_is_s1

    tif       = recent_files[-1]
    patch_dir = sat_dir / ("S1RTC" if recent_is_s1 else "S2L2A")
    skips = []
    for layer in ("L3", "L6", "L9"):
        pt = patch_dir / f"{tif.stem}_{layer}.pt"
        skips.append(
            torch.load(pt, weights_only=True, map_location="cpu")
            if pt.exists() else zeros.clone()
        )
    return skips[0], skips[1], skips[2], recent_is_s1


# ── Dataset ──────────────────────────────────────────────────────────────────

class SoilMoistureDataset(Dataset):
    """
    One sample = one (station, year, day-of-year) triple.

    Args:
        metadata_csv   : path to station_metadata.csv
        satellite_dir  : root of satellite data directories
        ismn_dir       : root of processed ISMN NetCDF files
        years          : list of years to include (default 2016–2023)
        min_obs        : minimum observed SM days in a year to include it
    """

    def __init__(
        self,
        metadata_csv:   str,
        satellite_dir:  str,
        ismn_dir:       str,
        years=None,
        min_obs:        int         = 30,
        station_filter: list | None = None,
        soil_data_root: str | None  = None,
    ):
        self.satellite_dir = Path(satellite_dir)
        self.ismn_dir      = Path(ismn_dir)
        self.years         = years or list(range(2016, 2024))

        # Build soil patch lookup: (network, station_id) → (Path, ok)
        soil_lookup: dict[tuple, tuple[Path, bool]] = {}
        if soil_data_root is not None:
            _soil_root = Path(soil_data_root)
            _splits    = pd.read_csv(_soil_root / "station_splits.csv")
            for _, r in _splits.iterrows():
                has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
                has_fl = str(r.get("has_flux",          "False")).lower() == "true"
                cat    = ("sm_and_flux" if (has_sm and has_fl)
                          else ("sm_only" if has_sm else "flux_only"))
                folder = (f"{r['source_network']}_{r['network']}_{r['station_id']}"
                          if r["source_network"] != r["network"]
                          else f"{r['network']}_{r['station_id']}")
                path   = _soil_root / cat / folder / "soil" / "soil_patch.tif"
                ok     = bool(r.get("soil_patch_ok", True))
                soil_lookup[(str(r["network"]), str(r["station_id"]))] = (path, ok)

        meta = pd.read_csv(metadata_csv)
        self.samples     = []
        self._ds_cache   = {}   # label_file → xr.Dataset (opened once per worker)

        for _, row in meta.iterrows():
            station_key = f"{row['network']}_{row['station']}"
            if station_filter is not None and station_key not in station_filter:
                continue

            # Soil patch check — skip the 19 stations with no valid soil data
            soil_path, soil_ok = soil_lookup.get(
                (str(row["network"]), str(row["station"])), (None, True)
            )
            if not soil_ok:
                continue

            sat_dir = self.satellite_dir / station_key

            if not sat_dir.exists():
                continue

            label_file = Path(row["filepath"])
            if not label_file.exists():
                continue

            ds_label = xr.open_dataset(label_file)

            for year in self.years:
                era5_file = sat_dir / "ERA5Land" / f"meteo_{year}.nc"
                if not era5_file.exists():
                    continue

                s2_files = list((sat_dir / "S2L2A").glob("*.tif")) if (sat_dir / "S2L2A").exists() else []
                if not any(f.stem[:4] == str(year) for f in s2_files):
                    continue

                year_mask = ds_label["date_time"].dt.year.values == year
                if not year_mask.any():
                    continue

                sm_year     = ds_label["soil_moisture"].values
                year_indices = np.where(year_mask)[0]
                sm_slice    = sm_year[:, year_indices]
                valid_days  = np.any(~np.isnan(sm_slice), axis=0)
                if valid_days.sum() < min_obs:
                    continue

                for day_idx in np.where(valid_days)[0]:
                    t_val = ds_label["date_time"].values[year_indices[day_idx]]
                    doy   = pd.Timestamp(t_val).day_of_year

                    self.samples.append({
                        "station_key" : station_key,
                        "sat_dir"     : sat_dir,
                        "label_file"  : label_file,
                        "year"        : year,
                        "doy"         : doy,
                        "time_idx"    : year_indices[day_idx],
                        "soil_path"   : soil_path,       # Path | None
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

        # ── S2L2A — pre-computed L12 + cloud mask ─────────────────────
        s2_l12, s2_doys, s2_valid, s2_token_mask, s2_rel_pos = load_s2_rolling(
            patch_dir      = sat_dir / "S2L2A",
            cloud_mask_dir = sat_dir / "CloudMask",
            year           = year,
            target_doy     = doy,
        )

        # ── S1RTC — pre-computed L12 ───────────────────────────────────
        s1_l12, s1_doys, s1_valid, s1_rel_pos = load_s1_rolling(
            patch_dir  = sat_dir / "S1RTC",
            year       = year,
            target_doy = doy,
        )

        # ── DEM pyramid (pre-computed, static) ────────────────────────
        dem_pyramid_path = sat_dir / "DEM" / "dem_pyramid.pt"
        if dem_pyramid_path.exists():
            dem_pyramid = torch.load(dem_pyramid_path, weights_only=True,
                                     map_location="cpu")          # (4, 768) fp32
        else:
            dem_pyramid = torch.zeros(4, 768, dtype=torch.float32)

        # ── Skip connection features (precomputed L3/L6/L9) ──────────
        skip_l3, skip_l6, skip_l9, recent_is_s1 = load_recent_skip_features(
            sat_dir, year, doy
        )

        # ── Soil patch (static, NaN-filled) ──────────────────────────
        soil_patch = load_soil_patch(s["soil_path"]) if s["soil_path"] else None
        if soil_patch is None:
            soil_patch = torch.zeros(21, 74, 74, dtype=torch.float32)

        # ── ERA5 — rolling 365-day window ─────────────────────────────
        era5, era5_doys = load_era5_rolling(sat_dir, year, doy)

        # ── ISMN labels ───────────────────────────────────────────────
        label_file = s["label_file"]
        if label_file not in self._ds_cache:
            self._ds_cache[label_file] = xr.open_dataset(label_file)
        ds_label = self._ds_cache[label_file]

        label          = torch.full((len(SM_DEPTHS),), float("nan"), dtype=torch.float32)
        depths_in_file = list(ds_label["depth"].values)

        for i, depth_str in enumerate(SM_DEPTHS):
            if depth_str in depths_in_file:
                d_idx    = depths_in_file.index(depth_str)
                val      = float(ds_label["soil_moisture"].values[d_idx, s["time_idx"]])
                label[i] = val

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
            "s1_rel_pos"    : s1_rel_pos,        # (MAX_S1,) long

            # Static
            "dem_pyramid"   : dem_pyramid,       # (4, 768) fp32

            # Skip connection features (most-recent acquisition, precomputed)
            "skip_l3"       : skip_l3,           # (196, 768) fp16
            "skip_l6"       : skip_l6,           # (196, 768) fp16
            "skip_l9"       : skip_l9,           # (196, 768) fp16
            "recent_is_s1"  : torch.tensor(recent_is_s1, dtype=torch.bool),

            # Soil (static)
            "soil_patch"    : soil_patch,        # (21, 74, 74) fp32 — NaN-free

            # ERA5
            "era5"          : era5,              # (365, 19) fp32
            "era5_doys"     : era5_doys,         # (365,) long

            # Labels
            "label"         : label,             # (4,) — NaN where depth absent
            "target_doy"    : torch.tensor(doy, dtype=torch.long),
        }
