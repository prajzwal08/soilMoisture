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
    """0-indexed position in the 365-day rolling window (0=oldest, 364=today)."""
    if acq_year == target_year:
        return (365 - target_doy) + (acq_doy - 1)
    else:
        return acq_doy - target_doy - 1


def _in_window(date_str: str, year: int, target_doy: int) -> bool:
    """True if date_str (YYYYMMDD) falls in the 365-day window ending at (year, target_doy)."""
    try:
        dt  = datetime.strptime(date_str[:8], "%Y%m%d")
        doy = dt.timetuple().tm_yday
        return (dt.year == year and doy <= target_doy) or \
               (dt.year == year - 1 and doy > target_doy)
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
    from datetime import timedelta
    era5_dir = sat_dir / "ERA5Land"
    nc_files = sorted(era5_dir.glob("meteo_*_*.nc")) if era5_dir.exists() else []
    if not nc_files:
        era5 = np.zeros((365, len(ERA5_VARS)), dtype=np.float32)
        doys = np.zeros(365, dtype=np.int64)
        return torch.from_numpy(era5), torch.from_numpy(doys)

    ds = xr.open_dataset(nc_files[0])

    target_date = datetime(year, 1, 1) + timedelta(days=target_doy - 1)
    window_start = datetime(year - 1, 1, 1) + timedelta(days=target_doy)
    ds_win = ds.sel(time=slice(str(window_start.date()), str(target_date.date())))

    era5 = np.stack([ds_win[v].values for v in ERA5_VARS], axis=-1).astype(np.float32)
    doys = np.array([pd.Timestamp(t).day_of_year for t in ds_win.time.values], dtype=np.int64)

    # Pad to exactly 365 rows (window start may be missing if data starts later)
    n = 365
    out_era5 = np.zeros((n, len(ERA5_VARS)), dtype=np.float32)
    out_doys = np.zeros(n, dtype=np.int64)
    l = min(len(doys), n)
    out_era5[-l:] = era5[-l:]
    out_doys[-l:]  = doys[-l:]
    return torch.from_numpy(out_era5), torch.from_numpy(out_doys)


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

    pt_files = sorted(patch_dir.glob("*_L12_*.pt")) if patch_dir.exists() else []
    if not pt_files:
        return l12, doys, doys > 0, token_mask, rel_pos

    data       = _load_pt(pt_files[0])
    tokens_all = data["tokens"]      # [N, 196, 768]
    all_dates  = data["dates"]

    # Filter to rolling window, keep most recent max_acq
    win_idx = [i for i, d in enumerate(all_dates) if _in_window(d, year, target_doy)]
    win_idx = win_idx[-max_acq:]

    # Cloud mask lookup (stacked .pt)
    cm_masks    = None
    cm_date_idx: dict[str, int] = {}
    cm_files = sorted(cloud_mask_dir.glob("*_*.pt")) if cloud_mask_dir.exists() else []
    if cm_files:
        cm_data     = _load_pt(cm_files[0])
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
            cm    = cm_masks[cm_date_idx[date_str]].numpy()
            cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
            token_mask[out_i] = torch.from_numpy((cm_4d == 0).all(axis=(1, 3)))

    return l12, doys, doys > 0, token_mask, rel_pos


def load_s1_rolling(patch_dir: Path, year: int, target_doy: int,
                    max_acq: int = MAX_S1):
    """
    Load pre-computed S1 L12 features (ASC + DESC merged) within the 365-day window.

    Returns:
        l12     : (max_acq, 196, 768) float16
        doys    : (max_acq,) long
        valid   : (max_acq,) bool
        rel_pos : (max_acq,) long
    """
    l12     = torch.zeros(max_acq, 196, 768, dtype=torch.float16)
    doys    = torch.zeros(max_acq, dtype=torch.long)
    rel_pos = torch.zeros(max_acq, dtype=torch.long)

    # Merge ASC and DESC into a single date-sorted list
    entries: list[tuple[str, torch.Tensor, int]] = []
    for orbit in ("ASC", "DESC"):
        pt_files = sorted(patch_dir.glob(f"*_{orbit}_L12_*.pt")) if patch_dir.exists() else []
        if not pt_files:
            continue
        data = _load_pt(pt_files[0])
        for i, d in enumerate(data["dates"]):
            if _in_window(d, year, target_doy):
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

def load_recent_skip_features(sat_dir: Path, year: int, target_doy: int):
    """
    Load precomputed L3/L6/L9 skip features for the most-recent acquisition
    (S2 or S1) in the rolling 365-day window.

    Returns:
        skip_l3, skip_l6, skip_l9 : each (196, 768) float16 — zeros if unavailable
        recent_is_s1               : bool
    """
    zeros = torch.zeros(196, 768, dtype=torch.float16)

    def _most_recent_in_window(mod_dir: Path, glob_pattern: str) -> str | None:
        """Return the most recent date string in the rolling window, or None."""
        pts = sorted(mod_dir.glob(glob_pattern)) if mod_dir.exists() else []
        if not pts:
            return None
        data = _load_pt(pts[0])
        for d in reversed(data["dates"]):   # already sorted ascending → reverse = newest first
            if _in_window(d, year, target_doy):
                return d
        return None

    s2_dir = sat_dir / "S2L2A"
    s1_dir = sat_dir / "S1RTC"

    s2_date  = _most_recent_in_window(s2_dir, "*_L12_*.pt")
    s1_date  = None
    s1_orbit = None
    for orbit in ("ASC", "DESC"):
        d = _most_recent_in_window(s1_dir, f"*_{orbit}_L12_*.pt")
        if d is not None and (s1_date is None or d > s1_date):
            s1_date, s1_orbit = d, orbit

    recent_is_s1 = bool(s1_date and (not s2_date or s1_date > s2_date))
    recent_date  = s1_date if recent_is_s1 else s2_date
    if recent_date is None:
        return zeros, zeros.clone(), zeros.clone(), False

    # Load L3/L6/L9 for that acquisition
    skips = []
    for layer in ("L3", "L6", "L9"):
        if recent_is_s1:
            pts = sorted(s1_dir.glob(f"*_{s1_orbit}_{layer}_*.pt"))
        else:
            pts = sorted(s2_dir.glob(f"*_{layer}_*.pt"))

        if not pts:
            skips.append(zeros.clone())
            continue

        data = _load_pt(pts[0])
        if recent_date in data["dates"]:
            idx = data["dates"].index(recent_date)
            skips.append(data["tokens"][idx])
        else:
            skips.append(zeros.clone())

    return skips[0], skips[1], skips[2], recent_is_s1


# ── SIF loader ───────────────────────────────────────────────────────────────

MAX_SIF  = 50
MAX_TWSA = 12


def load_sif_rolling(sat_dir: Path, year: int, target_doy: int):
    """
    Load TROPOMI SIF values for the 365-day rolling window.

    Returns:
        vals  : (MAX_SIF, 1) float32
        doys  : (MAX_SIF,) long
        valid : (MAX_SIF,) bool
    """
    vals  = torch.zeros(MAX_SIF, 1, dtype=torch.float32)
    doys  = torch.zeros(MAX_SIF, dtype=torch.long)
    valid = torch.zeros(MAX_SIF, dtype=torch.bool)

    sif_dir  = sat_dir / "SIF"
    nc_files = sorted(sif_dir.glob("sif_*_*.nc")) if sif_dir.exists() else []
    if not nc_files:
        return vals, doys, valid

    ds  = xr.open_dataset(nc_files[0])
    times = pd.to_datetime(ds["time"].values)
    sif_vals = ds["sif"].values.astype(np.float32)
    ds.close()

    entries = []
    for i, t in enumerate(times):
        d = str(t.date()).replace("-", "")
        if _in_window(d + "T00", year, target_doy):
            entries.append((t.timetuple().tm_yday, t.year, float(sif_vals[i])))

    entries = entries[-MAX_SIF:]
    for out_i, (acq_doy, acq_year, v) in enumerate(entries):
        vals[out_i, 0] = v
        doys[out_i]    = acq_doy
        valid[out_i]   = True

    return vals, doys, valid


# ── TWSA loader ──────────────────────────────────────────────────────────────

def load_twsa_rolling(sat_dir: Path, year: int, target_doy: int):
    """
    Load GRACE TWSA (liquid water equivalent anomaly) for the 365-day window.
    TWSA is monthly; typically ≤ 12 observations per year.

    Returns:
        vals  : (MAX_TWSA, 1) float32
        doys  : (MAX_TWSA,) long
        valid : (MAX_TWSA,) bool
    """
    vals  = torch.zeros(MAX_TWSA, 1, dtype=torch.float32)
    doys  = torch.zeros(MAX_TWSA, dtype=torch.long)
    valid = torch.zeros(MAX_TWSA, dtype=torch.bool)

    twsa_dir = sat_dir / "TWSA"
    nc_files = sorted(twsa_dir.glob("twsa_*_*.nc")) if twsa_dir.exists() else []
    if not nc_files:
        return vals, doys, valid

    ds   = xr.open_dataset(nc_files[0])
    times = pd.to_datetime(ds["time"].values)
    lwe   = ds["lwe"].values.astype(np.float32)
    ds.close()

    entries = []
    for i, t in enumerate(times):
        d = str(t.date()).replace("-", "")
        if _in_window(d + "T00", year, target_doy):
            entries.append((t.timetuple().tm_yday, float(lwe[i])))

    entries = entries[-MAX_TWSA:]
    for out_i, (acq_doy, v) in enumerate(entries):
        vals[out_i, 0] = v
        doys[out_i]    = acq_doy
        valid[out_i]   = True

    return vals, doys, valid


# ── Dataset ──────────────────────────────────────────────────────────────────

class SoilMoistureDataset(Dataset):
    """
    One sample = one (station, year, day-of-year) triple.

    Args:
        splits_csv       : path to station_splits.csv
        data_root        : root of /gpfs/…/data  (contains sm_only/, sm_and_flux/, flux_only/)
        era5_stats_path  : path to csvs/era5_stats.json  (from compute_era5_stats.py)
        years            : list of years to include (default 2016–2023)
        min_obs          : minimum observed SM days per year to include
        category_filter  : list of categories to include, e.g. ["sm_only"]  (None = all)
        split_filter     : list of split values to include, e.g. ["train"]  (None = all)
        training         : if True, apply SIF/TWSA modality dropout (p=0.5 each)
    """

    def __init__(
        self,
        splits_csv:      str,
        data_root:       str,
        era5_stats_path: str,
        years=None,
        min_obs:         int        = 30,
        category_filter: list | None = None,
        split_filter:    list | None = None,
        training:        bool        = True,
    ):
        import random as _random
        self._rng     = _random.Random()
        self.training = training
        self.years    = years or list(range(2016, 2024))
        self.data_root = Path(data_root)

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

        self.samples   = []
        self._ds_cache = {}   # label_path → xr.Dataset (opened once per worker)

        for _, r in splits.iterrows():
            has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
            has_fl = str(r.get("has_flux",          "False")).lower() == "true"
            cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")

            # Build directory name matching the on-disk convention
            if str(r["source_network"]) == "ISMN":
                dir_name = f"ISMN_{r['network']}_{r['station_name']}"
            else:
                dir_name = f"{r['source_network']}_{r['station_id']}"

            sat_dir    = self.data_root / cat / dir_name
            label_file = sat_dir / "labels.nc"
            soil_path  = sat_dir / "soil" / "soil_patch.tif"

            if not sat_dir.exists() or not label_file.exists():
                continue

            # Skip stations with no valid soil patch
            if not bool(r.get("soil_patch_ok", True)):
                continue

            ds_label = xr.open_dataset(label_file)

            # Fast year-range check via filename stems (avoids opening ERA5 NetCDF)
            era5_files = sorted((sat_dir / "ERA5Land").glob("meteo_*_*.nc")) \
                         if (sat_dir / "ERA5Land").exists() else []
            era5_years = (_year_range_from_stem(era5_files[0].stem)
                          if era5_files else None)

            s2_pt_files = sorted((sat_dir / "S2L2A").glob("*_L12_*.pt")) \
                          if (sat_dir / "S2L2A").exists() else []
            s2_years = (_year_range_from_stem(s2_pt_files[0].stem)
                        if s2_pt_files else None)

            for year in self.years:
                if era5_years is None or not (era5_years[0] <= year <= era5_years[1]):
                    continue
                if s2_years is None or not (s2_years[0] <= year <= s2_years[1]):
                    continue

                # Use "date_time" or fall back to "time" depending on labels.nc version
                time_coord = "date_time" if "date_time" in ds_label else "time"
                year_mask  = ds_label[time_coord].dt.year.values == year
                if not year_mask.any():
                    continue

                sm_vals      = ds_label["soil_moisture"].values   # (n_depths, n_days)
                year_indices = np.where(year_mask)[0]
                sm_slice     = sm_vals[:, year_indices]
                valid_days   = np.any(~np.isnan(sm_slice), axis=0)
                if valid_days.sum() < min_obs:
                    continue

                for day_idx in np.where(valid_days)[0]:
                    t_val = ds_label[time_coord].values[year_indices[day_idx]]
                    doy   = pd.Timestamp(t_val).day_of_year

                    self.samples.append({
                        "sat_dir"    : sat_dir,
                        "label_file" : label_file,
                        "year"       : year,
                        "doy"        : doy,
                        "time_idx"   : year_indices[day_idx],
                        "soil_path"  : soil_path if soil_path.exists() else None,
                        "station_key": dir_name,
                    })

            ds_label.close()

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

        # ── DEM L12 (pre-computed, static) ────────────────────────────
        dem_l12_path = sat_dir / "DEM" / "dem_L12.pt"
        dem_l12 = (torch.load(dem_l12_path, weights_only=True, map_location="cpu")
                   if dem_l12_path.exists()
                   else torch.zeros(196, 768, dtype=torch.float16))

        # ── LULC L12 (pre-computed, static) ───────────────────────────
        lulc_l12_path = sat_dir / "LULC" / "lulc_L12.pt"
        lulc_l12 = (torch.load(lulc_l12_path, weights_only=True, map_location="cpu")
                    if lulc_l12_path.exists()
                    else torch.zeros(196, 768, dtype=torch.float16))

        # ── Skip connection features (precomputed L3/L6/L9) ──────────
        skip_l3, skip_l6, skip_l9, recent_is_s1 = load_recent_skip_features(
            sat_dir, year, doy
        )

        # ── Soil patch (static, NaN-filled) ──────────────────────────
        soil_patch = load_soil_patch(s["soil_path"]) if s["soil_path"] else None
        if soil_patch is None:
            soil_patch = torch.zeros(21, 74, 74, dtype=torch.float32)

        # ── ERA5 — rolling 365-day window (raw) ───────────────────────
        era5, era5_doys = load_era5_rolling(sat_dir, year, doy)

        # Z-score normalisation: log1p precipitation then standardise all vars
        era5_np = era5.numpy()
        if self._era5_log1p_prec:
            prec_idx = ERA5_VARS.index("tp_sum")
            era5_np[:, prec_idx] = np.log1p(era5_np[:, prec_idx].clip(0))
        era5_np = (era5_np - self._era5_means) / (self._era5_stds + 1e-8)
        era5    = torch.from_numpy(era5_np)

        # ── SIF — optional sparse modality ───────────────────────────
        sif_vals, sif_doys, sif_valid = load_sif_rolling(sat_dir, year, doy)
        if self.training and self._rng.random() < 0.5:
            sif_valid[:] = False

        # ── TWSA — optional sparse modality ──────────────────────────
        twsa_vals, twsa_doys, twsa_valid = load_twsa_rolling(sat_dir, year, doy)
        if self.training and self._rng.random() < 0.5:
            twsa_valid[:] = False

        # ── ISMN labels ───────────────────────────────────────────────
        label_file = s["label_file"]
        if label_file not in self._ds_cache:
            self._ds_cache[label_file] = xr.open_dataset(label_file)
        ds_label = self._ds_cache[label_file]

        label          = torch.full((len(SM_DEPTHS),), float("nan"), dtype=torch.float32)
        depths_in_file = [str(d) for d in ds_label["depth"].values]

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
            "dem_l12"       : dem_l12,           # (196, 768) fp16
            "lulc_l12"      : lulc_l12,          # (196, 768) fp16

            # Skip connection features (most-recent acquisition, precomputed)
            "skip_l3"       : skip_l3,           # (196, 768) fp16
            "skip_l6"       : skip_l6,           # (196, 768) fp16
            "skip_l9"       : skip_l9,           # (196, 768) fp16
            "recent_is_s1"  : torch.tensor(recent_is_s1, dtype=torch.bool),

            # Soil (static)
            "soil_patch"    : soil_patch,        # (21, 74, 74) fp32 — NaN-free

            # ERA5 (z-scored)
            "era5"          : era5,              # (365, 19) fp32
            "era5_doys"     : era5_doys,         # (365,) long

            # SIF (optional, sparse)
            "sif"           : sif_vals,          # (MAX_SIF, 1) fp32
            "sif_doys"      : sif_doys,          # (MAX_SIF,) long
            "sif_valid"     : sif_valid,         # (MAX_SIF,) bool

            # TWSA (optional, sparse)
            "twsa"          : twsa_vals,         # (MAX_TWSA, 1) fp32
            "twsa_doys"     : twsa_doys,         # (MAX_TWSA,) long
            "twsa_valid"    : twsa_valid,        # (MAX_TWSA,) bool

            # Labels
            "label"         : label,             # (3,) — NaN where depth absent
            "target_doy"    : torch.tensor(doy, dtype=torch.long),
        }
