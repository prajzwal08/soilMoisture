"""
SoilMoistureDataset
====================
One sample = one (station, year, day-of-year) triple, for the §48 model: temporal trunk over
pooled TerraMind pyramids + the anchor's L12, a light CNN on the most recent raw imagery, and
two targets — ISMN soil moisture at the station pixel and the Landsat ST pattern at 100 m.

This file REPLACED the patchwise loader (§34/§35, tags `pw_stage2a-ep9`, `pre-s48-build`).
Everything below the sample path — labels, QC, soil, ERA5/SIF/TWSA, the audit counters — is
the canonical code unchanged; see §35.24 for why each of those fails closed.

FOUR stores, all read-only here:

  ZARR_ROOT   token store (scratch)      era5/values18, sif, twsa, labels, soil
  CACHE_ROOT  §48 cache (scratch)        built once by prepare_s48_cache.py from ZARR_ROOT:
              {cat}/{station}/pyr.npz        per-acquisition pyramids (N,4,768), valid-token
                                             counts, date ints; DEM/LULC pyramids
              {cat}/{station}/{orbit}_l12.npy  (N,196,768) fp16 — the anchor, one row per read
              {cat}/{station}/s2_cm.npy      (N_s2,224,224) u1 pixel cloud classes aligned to
                                             s2 dates, 255 where no mask exists (fail closed)
              The token store chunks l12 32 acquisitions at a time and cm/masks as ONE chunk
              per station, so reading either per sample decompresses ~9 MB / ~5 MB to use
              0.3 MB. The cache exists for that reason only; it holds no new information.
  RAW_ROOT    raw imagery (scratch)      {station}.zarr  s2/data (N,12,224,224) i2 DN,
                                         s1_{asc,desc}/data (N,2,224,224) f2 dB, dem/data,
                                         lulc/data (Y,224,224) u1 TerraMind indices
  LST_ROOT    Landsat target (work3)     {cat}/{station}/LANDSAT_ST/{station}_lst22.npz,
                                         built by consolidate_landsat_st.py

"Most recent" always means ON OR BEFORE day D (§48.9 item 2). The Landsat scene on day D is a
TARGET, never an input (§48.9 item 3).

§35.24 audit. Everything this loader does about MISSING data fails closed: an acquisition with
no cloud mask is invalid rather than clear, an orbit with no token_mask contributes nothing,
a station with no QC source is dropped, and a missing driver_stats.json raises. Each of those
removes data silently by construction, so each is counted and printed at the end of __init__.
"""

import json
import os
import random
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from splits_config import TRAIN_YEARS, category_of, station_dir_name

ZARR_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
CACHE_ROOT_GPFS = Path("/gpfs/scratch1/shared/pkhanal/s48cache")
# Training reads a /dev/shm copy staged by stage_shm.py (train+val stations, scenes <= 2022):
# GPFS small random reads capped the loader at ~40 samples/s per node (2026-09-29).
CACHE_ROOT = Path(os.environ.get("S48_CACHE_ROOT", str(CACHE_ROOT_GPFS)))
RAW_ROOT   = Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr")
LST_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
FINE_STATS_PATH = Path(__file__).resolve().parent / "csvs" / "fine_stats.json"

# torch.from_numpy on a read-only memmap triggers a non-writable warning; every such tensor
# is copied into a fresh buffer before it leaves __getitem__.
warnings.filterwarnings("ignore", message=".*not writable.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*non-writeable.*", category=UserWarning)

# ── Constants ────────────────────────────────────────────────────────────────

# §43.12: 19 -> 18.  `skt_{mean,min,max}` dropped -- it is a modelled diagnostic in
# ERA5-Land, largely determined by t2m plus the radiation forcings now added, and it
# is the Landsat ST target at 9 km, which makes the thermal head's level term trivial.
# `ssrd_sum`/`strd_sum` added: accumulations get a daily SUM, not mean/min/max, because
# each `_hourly` value is already J m-2 over that hour and because a 24 h total is
# nearly insensitive to where the UTC day boundary cuts the local diurnal cycle.
ERA5_VARS = [
    "t2m_mean",  "t2m_min",  "t2m_max",
    "d2m_mean",  "d2m_min",  "d2m_max",
    "u10_mean",  "u10_min",  "u10_max",
    "v10_mean",  "v10_min",  "v10_max",
    "sp_mean",   "sp_min",   "sp_max",
    "tp_sum", "ssrd_sum", "strd_sum",
]  # 18 features

# Which zarr array the 18 columns live in.  `splice_era5_radiation.py` writes
# `era5/values18` BESIDE the original `era5/values` (19 cols, skt included) rather
# than over it -- zarr_tokens is the only copy of the drivers.  Point this at
# "era5/values" only to read the pre-§43.12 set, and then ERA5_VARS must be reverted
# to match or every column is silently misnamed.
ERA5_ARRAY = "era5/values18"

SM_DEPTHS = ["0-10", "10-30", "30-100"]  # n_depths = 3

S2_BAND_INDICES = list(range(12))  # all 12 S2L2A bands (no B10); precompute_terramind.py reads it

PREC_IDX = ERA5_VARS.index("tp_sum")  # index computed once at import, not per sample

# Token grid is 14x14 over a 224px tile, so patch k covers pixels [16k, 16k+16). Kept for the
# analysis scripts that import them; the §48 model reads the whole grid.
TOKEN_GRID    = 14
N_TOKENS      = TOKEN_GRID * TOKEN_GRID          # 196
STATION_TOKEN = (112 // 16) * TOKEN_GRID + (112 // 16)   # 105

MAX_S2 = 60
MAX_S1 = 40

HIST_ORBITS = ("s2", "s1_asc", "s1_desc")
ANCHOR_ORBIT_ID = {"s2": 0, "s1_asc": 1, "s1_desc": 2}

# ── Cloud-mask class table ───────────────────────────────────────────────────
# cm/masks is written by cloud_masking_inference.py, which runs SEnSeIv2-SegFormerB2 (the
# TerraMesh cloud model). It is a SEVEN-class product and NOT Sentinel-2 SCL:
#
#     0   land / clear
#     1   water
#     2   snow / ice
#     3   thin cloud          <- bad
#     4   thick cloud         <- bad
#     5   cloud shadow        <- bad
#     255 nodata              <- bad
#
# Writing this down because the obvious misreading is expensive. Under Sentinel-2 SCL the
# same integers mean something almost opposite -- 4 = vegetation and 5 = not-vegetated, i.e.
# the two classes you most want to KEEP. The list is correct for THIS product.
CM_BAD_CLASSES = [3, 4, 5, 255]

# Fraction of a 16x16 = 256-pixel patch that may be bad before the token is rejected.
# 0.01 of 256 is 2.56 px, so this admits at most TWO bad pixels. Deliberate at 10 m: a token
# is 160 m across and thin cirrus at its edge contaminates the whole 768-d embedding.
CM_MAX_BAD_FRAC = 0.01

# S1 orbit identity. RTC backscatter differs systematically between ascending and descending
# (incidence angle, look azimuth) by an amount comparable to the moisture signal, so an orbit
# switch must never be readable as a wetting event (§35.24 audit item 8).
ORBIT_ASC, ORBIT_DESC = 0, 1
_ORBIT_ID = {"s1_asc": ORBIT_ASC, "s1_desc": ORBIT_DESC}

# labels/qc sentinel written by create_token_zarr.py when the source NetCDF carried NEITHER
# `soil_moisture_qc` NOR `quality_flag` (§35.24 audit item 4).
QC_OBSERVED   = 0
QC_NO_SOURCE  = 255

# A soil channel that is NaN over the whole 74x74 patch cannot be nearest-neighbour filled;
# it is set to 0 (== the dataset mean once z-scored) and flagged dead. More than two dead
# channels and the station's soil block is fiction, so it is dropped (§35.24 audit item 6).
MAX_DEAD_SOIL_CHANNELS = 2

# ── Fine imagery (§48.3, §48.9 item 1) — must match model.py's FINE_* layout ────
FINE_CH        = 19          # S2 10 bands + valid + age | S1 VV VH valid age orbit | DEM valid
LULC_PAD       = 10          # model.py LULC_PAD; TerraMind index 0 (nodata) and >9 map here
LST_N          = 22
LST_LEVEL_MIN_CELLS = 10     # §52: min valid 100 m cells for a tile-mean (level) target
T2M_MEAN_IDX   = ERA5_VARS.index("t2m_mean")


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    if ERA5_ARRAY not in zg:
        return None
    return (
        zg[ERA5_ARRAY][:],
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


def _load_zarr_labels(zg: zarr.Group, strict: bool = False):
    """Load labels from zarr → (sm_np, depths, times, qc_np).
    qc_np: 0=observed, 1=gap-filled, 2=still missing, 255=no QC source. None if not present.

    §35.24 audit item 9 made a length mismatch fatal under `strict`, on the grounds that the
    old code's trailing-slice realignment was an unverifiable guess. That was half right and
    cost 62% of the training split: `slurm/driver_stats.sh` dropped 362 of 587 train stations
    on `labels-qc-length-mismatch` before anyone noticed the path was not dead.

    §35.27: it IS verifiable, and the trailing slice is correct. trim_pre2016.py trims
    `labels/sm` and `labels/dates` from the FRONT and leaves `labels/qc` at its original
    length, so qc is longer by exactly the pre-2016 span and its last n columns are the ones
    that align. Measured on ISMN_AMMA-CATCH_Banizoumbou: sm/dates 1095, qc 1825, difference
    730 — and station_splits.csv gives actual_start_date 2014-01-01, start_date 2016-01-01,
    end_date 2018-12-30, i.e. 1825 days untrimmed and 1095 trimmed. Exact.

    So the two directions are not the same problem and must not share a branch:

      qc LONGER  than sm  ->  front-trim.  Recoverable, and verified here by requiring the
                              date index to be contiguous daily: if `dates` covers a gapless
                              daily span ending at the record's end, then qc — the same
                              station's record over a longer span with the same end — aligns
                              on its trailing n columns by construction.
      qc SHORTER than sm  ->  no alignment exists.  Always fatal, `strict` or not.

    `strict` now governs only the case that stays genuinely unverifiable (a non-contiguous
    date index), which no current store exhibits.
    """
    if "labels/sm" not in zg:
        return None
    sm_np  = zg["labels/sm"][:]                        # (n_depths, n_days) float32
    depths = [str(d) for d in zg["labels/depths"][:]]
    dates  = [str(d) for d in zg["labels/dates"][:]]
    times  = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    qc_np  = zg["labels/qc"][:] if "labels/qc" in zg else None  # (n_depths, n_days) uint8

    if len(dates) != sm_np.shape[1]:
        raise ValueError(
            f"labels/dates has {len(dates)} entries but labels/sm has {sm_np.shape[1]} "
            f"columns in {getattr(zg.store, 'path', '<zarr>')} — the label block is "
            f"internally inconsistent, no alignment can be recovered."
        )
    if qc_np is not None and qc_np.shape[1] != sm_np.shape[1]:
        n_sm, n_qc = sm_np.shape[1], qc_np.shape[1]
        where = getattr(zg.store, 'path', '<zarr>')

        if n_qc < n_sm:
            # No front-trim can make a SHORTER qc align with sm; there is nothing to recover.
            raise ValueError(
                f"labels/qc has {n_qc} columns against {n_sm} in labels/sm ({where}). "
                f"qc is SHORTER than sm, so no trim can align them — the label block is "
                f"corrupt. Re-run create_token_zarr.py for this station."
            )

        # qc is longer: the trim_pre2016.py front-trim. Verify before relying on it — the
        # trailing slice is correct only if `dates` is a gapless daily span, which is what
        # makes "same end date, longer start" imply trailing alignment.
        contiguous = (len(times) >= 2
                      and (times[-1] - times[0]).days + 1 == len(times)
                      and bool((np.diff(times.values.astype("datetime64[D]")).astype(int) == 1).all()))
        if not contiguous:
            msg = (f"labels/qc has {n_qc} columns against {n_sm} in labels/sm ({where}) and "
                   f"labels/dates is NOT a contiguous daily span, so the front-trim that "
                   f"normally explains the difference cannot be verified. Refusing to guess "
                   f"the offset — a wrong guess trains gap-filled days as observed.")
            if strict:
                raise ValueError(msg)
            print(f"  [labels] WARNING: {msg}")

        qc_np = qc_np[:, -n_sm:]
    return sm_np, depths, times, qc_np


def _load_driver_stats(path):
    """Load csvs/driver_stats.json — the SIF / TWSA / soil analogue of era5_stats.json.

    §35.24 audit item 7: ERA5 was z-scored at load time while SIF (~0-3 mW/m2/sr/nm),
    TWSA (~±40 cm of equivalent water) and the 21 soil channels (pH x10, clay %, bulk
    density in kg/m3, ...) went into their MLPs raw. A 1400-magnitude bulk-density channel
    next to a 0.6-magnitude ERA5 z-score does not "just get learned around": it dominates
    the first-layer gradient and the soil block becomes a constant offset. Fail closed —
    a missing stats file must stop the run, never silently restore identity scaling.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"driver stats not found at {p}. SIF, TWSA and the soil patch must be z-scored "
            f"exactly as ERA5 is — run compute_driver_stats.py to produce "
            f"csvs/driver_stats.json before training. Refusing to fall back to identity "
            f"normalisation (§35.24 audit item 7)."
        )
    with open(p) as f:
        st = json.load(f)
    for k in ("sif", "twsa", "soil"):
        if k not in st or "mean" not in st[k] or "std" not in st[k]:
            raise KeyError(
                f"{p} is missing a complete '{k}' block (needs 'mean' and 'std') — "
                f"regenerate it with compute_driver_stats.py."
            )
    # label_mean is validated HERE even though only train.py consumes it, because a
    # driver_stats.json carrying the three normalisation blocks but no label_mean would sail
    # through this fail-closed gate and then leave head_bias_init=None — which is silently the
    # exact defect §35.24 added it to prevent: a head bias of U(±0.036) against targets of
    # ~0.25 opens training deep in Huber's linear regime, where the gradient is a constant
    # ±delta carrying no information about the size of the error. One validator, one file,
    # one place to regenerate.
    missing = [d for d in SM_DEPTHS if d not in st.get("label_mean", {})]
    if missing:
        raise KeyError(
            f"{p} is missing label_mean for {missing} (needs one entry per SM_DEPTHS "
            f"bin: {SM_DEPTHS}) — regenerate it with compute_driver_stats.py. train.py "
            f"reads this to initialise the per-depth head biases (§35.24 audit item 5)."
        )
    return st



# ── Pyramids and the §48 cache ───────────────────────────────────────────────

def _cpu_pyramid_pool(l12: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked 4-scale nested-window pooling of L12 tokens, as the frozen U-Net arm did it
    (dataset_unet.py:193), so the trunk's history tokens mean the same thing they did there.

    l12:        (M, 196, 768) fp16
    token_mask: (M, 14, 14)   bool — True = valid/clear patch
    returns:    (M, 4, 768)   fp32   centre ~2x2 / 4x4 / 10x10 / 14x14 tokens
    """
    M, N_tok, D = l12.shape
    G    = int(N_tok ** 0.5)
    g    = l12.float().reshape(M, G, G, D)
    v    = token_mask.float().unsqueeze(-1)

    half   = G // 2
    widths = [max(1, G * (i + 1) // 8) for i in range(4)]

    def _pool(w):
        rs, re = half - w, half + w
        rg = g[:, rs:re, rs:re, :]
        rv = v[:, rs:re, rs:re, :]
        return (rg * rv).sum(dim=(1, 2)) / rv.sum(dim=(1, 2)).clamp(min=1)

    return torch.stack([_pool(w) for w in widths], dim=1)


def _cm_token_mask(cm: np.ndarray) -> np.ndarray:
    """(N, 224, 224) u1 cloud classes -> (N, 14, 14) bool, True = <= CM_MAX_BAD_FRAC bad."""
    n   = cm.shape[0]
    bad = np.isin(cm[:, :224, :224].reshape(n, 14, 16, 14, 16), CM_BAD_CLASSES).mean(axis=(2, 4))
    return bad <= CM_MAX_BAD_FRAC


def build_station_cache(zg: zarr.Group, out_dir: Path) -> dict:
    """Write one station's §48 cache from its token store. Used by prepare_s48_cache.py.

    Fail closed throughout, as the patchwise loader was (§35.24 audit items 2, 3):
      * S2 acquisition with no cloud-mask entry  -> token mask all False, cm row all 255
      * S1 orbit with no stored token_mask       -> token mask all False
      * any non-finite token in an acquisition   -> token mask all False, l12 row zeroed
    An acquisition whose mask is all False has nvalid == 0 and is never read: not as
    history, not as the anchor.

    Returns a small report dict (counts), never raises for a missing modality.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, rep = {}, {}

    cm_arr, cm_idx = None, {}
    if "cm/masks" in zg and "cm/dates" in zg:
        cm_arr = zg["cm/masks"][:]                                   # one chunk anyway
        cm_idx = {str(d): i for i, d in enumerate(zg["cm/dates"][:])}

    for orbit in HIST_ORBITS:
        if f"{orbit}/l12" not in zg or f"{orbit}/dates" not in zg:
            continue
        dates = [str(d) for d in zg[f"{orbit}/dates"][:]]
        l12   = np.asarray(zg[f"{orbit}/l12"][:])                     # (N, 196, 768) f16
        N     = len(dates)
        if orbit == "s2":
            cm_al = np.full((N, 224, 224), 255, dtype=np.uint8)
            for i, d in enumerate(dates):
                j = cm_idx.get(d)
                if j is not None:
                    cm_al[i] = cm_arr[j, :224, :224]
            tm = _cm_token_mask(cm_al)
            np.save(out_dir / "s2_cm.npy", cm_al)
            rep["s2_no_cm"] = int(sum(d not in cm_idx for d in dates))
        elif f"{orbit}/token_mask" in zg:
            tm = np.asarray(zg[f"{orbit}/token_mask"][:]).astype(bool)
        else:
            tm = np.zeros((N, 14, 14), dtype=bool)
            rep[f"{orbit}_no_token_mask"] = N

        finite = np.isfinite(l12.reshape(N, -1)).all(axis=1)
        tm     = tm & finite[:, None, None]
        l12    = np.where(finite[:, None, None], l12, np.float16(0))
        rep[f"{orbit}_nonfinite"] = int((~finite).sum())

        pyr = np.empty((N, 4, 768), dtype=np.float16)
        for b0 in range(0, N, 64):
            b1 = min(b0 + 64, N)
            pyr[b0:b1] = _cpu_pyramid_pool(torch.from_numpy(l12[b0:b1]),
                                           torch.from_numpy(tm[b0:b1])).numpy()
        np.save(out_dir / f"{orbit}_l12.npy", l12)
        arrays[f"{orbit}_pyr"]       = pyr
        arrays[f"{orbit}_nvalid"]    = tm.reshape(N, -1).sum(1).astype(np.int16)
        arrays[f"{orbit}_date_ints"] = np.array([int(d[:8]) for d in dates], dtype=np.int32)
        rep[f"{orbit}_n"] = N

    for key in ("dem", "lulc"):
        tm_key = f"{key}_token_mask"
        ok = key in zg and tm_key in zg
        l12 = np.asarray(zg[key][:]) if key in zg else np.zeros((196, 768), np.float16)
        tm  = (np.asarray(zg[tm_key][:]).astype(bool) if ok else np.zeros((14, 14), bool))
        tm  = tm & bool(np.isfinite(l12).all())
        arrays[f"{key}_pyr"] = _cpu_pyramid_pool(
            torch.from_numpy(np.nan_to_num(l12))[None], torch.from_numpy(tm)[None]
        )[0].numpy().astype(np.float16)
        arrays[f"{key}_ok"] = np.array(bool(tm.any()))
        rep[f"{key}_ok"] = bool(tm.any())

    # pyr.npz is written LAST and renamed into place, so its presence is the completion
    # marker: a job killed mid-station leaves no pyr.npz and the loader skips the station.
    tmp = out_dir / "pyr.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.rename(out_dir / "pyr.npz")
    return rep


def _load_station_cache(cache_dir: Path, fine: bool = True) -> dict | None:
    """pyr.npz fully in RAM (a few MB), l12 / cm as read-only memmaps. None if absent."""
    p = cache_dir / "pyr.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        c = {k: z[k] for k in z.files}
    for orbit in HIST_ORBITS:
        f = cache_dir / f"{orbit}_l12.npy"
        if f"{orbit}_pyr" in c and f.exists():
            c[f"{orbit}_l12"] = np.load(f, mmap_mode="r")
    f = cache_dir / "s2_cm.npy"
    if f.exists():
        c["s2_cm"] = np.load(f, mmap_mode="r")
    if fine:
        _attach_fine(c, cache_dir)
    return c


def _fine_stats_sha() -> str:
    import hashlib
    return hashlib.sha256(FINE_STATS_PATH.read_bytes()).hexdigest()[:16]


def _attach_fine(c: dict, cache_dir: Path) -> None:
    """Precomputed fine inputs (prepare_fine_cache.py). fine_meta.npz is written LAST, so its
    presence marks a complete set. Refuses a set built under different fine_stats constants."""
    p = cache_dir / "fine_meta.npz"
    if not p.exists():
        return
    with np.load(p) as z:
        meta = {k: z[k] for k in z.files}
    if str(meta["fine_stats_sha"]) != _fine_stats_sha():
        raise RuntimeError(f"{p}: built with fine_stats sha {meta['fine_stats_sha']}, current "
                           f"{_fine_stats_sha()} — rerun prepare_fine_cache.py")
    cand = meta["s2_cand"].astype(np.int64)                      # (K, 3) date, raw_row, tok_row
    c["fine_s2_cands"] = [(int(d), int(r), int(t), k) for k, (d, r, t) in enumerate(cand)]
    if len(cand):
        c["fine_s2"] = np.load(cache_dir / "fine_s2.npy", mmap_mode="r")
    else:
        c["fine_s2"] = np.zeros((0, 11, 112, 112), np.float16)
    for key in ("s1_asc", "s1_desc"):
        c[f"fine_{key}_dates"] = meta[f"{key}_dates"].astype(np.int64)
        f = cache_dir / f"fine_{key}.npy"
        if len(c[f"fine_{key}_dates"]):
            c[f"fine_{key}"] = np.load(f, mmap_mode="r")
    c["fine_dem"]        = meta["dem"]
    c["fine_lulc_years"] = meta["lulc_years"].astype(np.int32)
    c["fine_lulc"]       = meta["lulc"]


def _int_to_date(d: int) -> datetime:
    return datetime(d // 10000, (d // 100) % 100, d % 100)


def _target_date(year: int, doy: int) -> datetime:
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def _candidates(cache: dict, orbits, start_int: int, end_int: int):
    """(date_int, orbit, idx, nvalid) for every usable acquisition in [start, end]."""
    out = []
    for orbit in orbits:
        di = cache.get(f"{orbit}_date_ints")
        if di is None:
            continue
        nv  = cache[f"{orbit}_nvalid"]
        idx = np.where((di >= start_int) & (di <= end_int) & (nv > 0))[0]
        out.extend((int(di[i]), orbit, int(i), int(nv[i])) for i in idx)
    return out


def load_history(cache: dict, orbits, year: int, doy: int, max_acq: int):
    """Pooled-pyramid history over the 365-day window ending ON day D.

    Compact, oldest-first, padding at the tail — the convention the patchwise loaders agreed
    on (§35.24b item 3). Orbits are merged by date; `orbit` records which pass each slot is.

    Returns (pyr (T,4,768) fp16, doys (T,), valid (T,) bool, rel_pos (T,), orbit (T,) long).
    """
    start_int, end_int = _window_ints(year, doy)
    ent = sorted(_candidates(cache, orbits, start_int, end_int))[-max_acq:]
    pyr     = torch.zeros(max_acq, 4, 768, dtype=torch.float16)
    doys    = torch.zeros(max_acq, dtype=torch.long)
    rel_pos = torch.zeros(max_acq, dtype=torch.long)
    orbit   = torch.zeros(max_acq, dtype=torch.long)
    for k, (dint, orb, i, _) in enumerate(ent):
        dt = _int_to_date(dint)
        pyr[k]     = torch.from_numpy(np.array(cache[f"{orb}_pyr"][i]))
        doys[k]    = dt.timetuple().tm_yday
        rel_pos[k] = _rel_pos(doys[k].item(), dt.year, doy, year)
        orbit[k]   = _ORBIT_ID.get(orb, 0)
    return pyr, doys, doys > 0, rel_pos, orbit


def select_anchor(cache: dict, year: int, doy: int):
    """The anchor: most recent FULLY CLEAR acquisition (all 196 tokens valid) on or before D,
    across S2 and both S1 orbits; else the most recent usable one. Fail closed — an S2 date with
    no cloud mask is not a candidate (the frozen U-Net arm counted it as fully clear).

    Returns (l12 (196,768) fp16, rel_pos, orbit_id, found bool).
    """
    start_int, end_int = _window_ints(year, doy)
    cands = _candidates(cache, HIST_ORBITS, start_int, end_int)
    if not cands:
        return torch.zeros(196, 768, dtype=torch.float16), 0, 0, False
    clear = [c for c in cands if c[3] == N_TOKENS]
    dint, orb, i, _ = max(clear or cands, key=lambda c: (c[0], -ANCHOR_ORBIT_ID[c[1]]))
    dt  = _int_to_date(dint)
    l12 = torch.from_numpy(np.array(cache[f"{orb}_l12"][i]))
    return l12, _rel_pos(dt.timetuple().tm_yday, dt.year, doy, year), ANCHOR_ORBIT_ID[orb], True


# ── Fine imagery (§46.3 worker order, §48.3) ─────────────────────────────────

def _load_fine_stats(path: Path = FINE_STATS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — it carries the TerraMind normalisation "
                                f"constants for the fine path and is SHA'd into CONFIG.")
    st = json.loads(path.read_text())
    keep = st["s2"]["keep_idx"]
    return {
        "s2_keep": np.asarray(keep, dtype=np.int64),
        "s2_mean": np.asarray(st["s2"]["mean"], np.float32)[keep][:, None, None],
        "s2_std":  np.asarray(st["s2"]["std"],  np.float32)[keep][:, None, None],
        "s1_mean": np.asarray(st["s1"]["mean"], np.float32)[:, None, None],
        "s1_std":  np.asarray(st["s1"]["std"],  np.float32)[:, None, None],
        "dem_mean": float(st["dem"]["mean"][0]),
        "dem_std":  float(st["dem"]["std"][0]),
        "dem_nodata_below": float(st["dem"]["nodata_below_m"]),
        "max_age": float(st["max_age_days"]),
    }


_IO_POOL, _IO_POOL_KEY = None, None


def _io_pool(n: int):
    """One ThreadPoolExecutor per PROCESS (keyed on pid): a pool inherited across the
    DataLoader's fork has no live threads, so each worker builds its own on first use."""
    global _IO_POOL, _IO_POOL_KEY
    key = (os.getpid(), n)
    if _IO_POOL_KEY != key:
        from concurrent.futures import ThreadPoolExecutor
        _IO_POOL, _IO_POOL_KEY = ThreadPoolExecutor(max_workers=n), key
    return _IO_POOL


DEM_ASINH_KNEE_M = 1.0   # fine DEM channel: asinh((elev - tile mean) / 1 m) / 4, clipped +/-2.5
DEM_ASINH_SCALE  = 4.0
DEM_ASINH_CLIP   = 2.5


def _pool2(x: np.ndarray, m: np.ndarray):
    """(C,224,224) values, (224,224) bool valid -> masked 2x2 mean (C,112,112), frac (112,112).

    Masked, so a 20 m cell with one cloudy 10 m pixel is the mean of the other three rather
    than diluted toward zero. Cells with no valid pixel come back exactly 0.0 — step 8 of
    §46.3 (zero AFTER normalisation) falls out of the masking.
    """
    C  = x.shape[0]
    mf = m.astype(np.float32)
    s  = (x * mf).reshape(C, 112, 2, 112, 2).sum(axis=(2, 4))
    n  = mf.reshape(112, 2, 112, 2).sum(axis=(1, 3))
    out = np.where(n > 0, s / np.maximum(n, 1.0), 0.0).astype(np.float32)
    return out, (n / 4.0).astype(np.float32)


def _raw_dates(rg, key: str) -> np.ndarray:
    if rg is None or f"{key}/dates" not in rg:
        return np.zeros(0, dtype=np.int32)
    return np.array([int(bytes(d).decode()[:8]) if isinstance(d, (bytes, np.bytes_))
                     else int(str(d)[:8]) for d in rg[f"{key}/dates"][:]], dtype=np.int32)


# ── Fine per-scene maths, shared by build_fine and prepare_fine_cache.py ──────
# Each helper writes a float32 buffer exactly as build_fine used to write `fine` (the same
# float64 -> float32 assignment), so storing buffer.astype(float16) and later assigning it
# back reproduces build_fine's final fine.astype(float16) bit-for-bit.

def fine_s2_scene(x_raw, cm, fs: dict) -> np.ndarray:
    """Raw S2 (12,224,224) + its (224,224) cloud-class mask -> (11,112,112) f32:
    10 z-scored bands masked-2x2-pooled, then the valid fraction. Use only if [10].any()."""
    out = np.zeros((11, 112, 112), dtype=np.float32)
    x = np.asarray(x_raw, dtype=np.float32)[fs["s2_keep"]]                   # (10,224,224)
    m = (x != 0).all(axis=0) & ~np.isin(np.asarray(cm), CM_BAD_CLASSES)
    x = (x - fs["s2_mean"]) / fs["s2_std"]
    pooled, frac = _pool2(x, m)
    out[0:10] = pooled
    out[10] = frac
    return out


def fine_s1_scene(x_raw, fs: dict) -> np.ndarray:
    """Raw S1 (2,224,224) dB -> (3,112,112) f32: z-scored VV, VH (2x2 mean in LINEAR power),
    then the valid fraction. Use only if [2].any()."""
    out = np.zeros((3, 112, 112), dtype=np.float32)
    x = np.asarray(x_raw, dtype=np.float32)
    m = np.isfinite(x).all(axis=0) & (x != 0).all(axis=0)
    lin = np.where(m, np.power(10.0, np.where(m, x, 0.0) / 10.0), 0.0)
    pooled, frac = _pool2(lin, m)
    if frac.any():
        db = 10.0 * np.log10(np.maximum(pooled, 1e-10))
        z = (db - fs["s1_mean"]) / fs["s1_std"]
        out[0:2] = np.where(frac > 0, z, 0.0)
        out[2] = frac
    return out


def fine_dem_static(x_raw, fs: dict) -> np.ndarray:
    """Raw DEM (224,224) m -> (2,112,112) f32: asinh relative relief, then the valid fraction.
    Relative relief, asinh-compressed (2026-09-29; figures/dem_asinh): the global z-score
    (elev - 671)/951 left a flat tile's relief at ~0.002 sd and a Tibetan tile as one flat
    block; absolute elevation already reaches the trunk through the DEM tokens. asinh is
    linear inside |dh| < DEM_ASINH_KNEE_M and logarithmic beyond. Use only if [1].any()."""
    out = np.zeros((2, 112, 112), dtype=np.float32)
    x = np.asarray(x_raw, dtype=np.float32)[None]
    m = np.isfinite(x[0]) & (x[0] > fs["dem_nodata_below"])
    if not np.any(x[0][m] != 0):
        m[:] = False                          # an all-zero raster is the fill value, not sea level
    pooled, frac = _pool2(np.where(m, x, 0.0), m)                # metres, masked 2x2 mean
    if frac.any():
        v = frac > 0
        rel = np.arcsinh((pooled[0] - pooled[0][v].mean()) / DEM_ASINH_KNEE_M) / DEM_ASINH_SCALE
        out[0] = np.where(v, np.clip(rel, -DEM_ASINH_CLIP, DEM_ASINH_CLIP), 0.0)
        out[1] = frac
    return out


def lulc_remap(a) -> np.ndarray:
    a = np.asarray(a)
    return np.where((a >= 1) & (a <= 9), a, LULC_PAD).astype(np.uint8)


def _pick_s2(dates_ok: list, nv, start_int: int, end_int: int):
    """dates_ok: [(date_int, raw_row, tok_row[, storage_row])] -> chosen tuple or None (clear
    preferred). Dates are unique per orbit, so max() orders by date exactly as before."""
    cands = [c for c in dates_ok if start_int <= c[0] <= end_int and nv[c[2]] > 0]
    if not cands:
        return None
    clear = [c for c in cands if nv[c[2]] == N_TOKENS]
    return max(clear or cands)


def _pick_s1(dates_by_key: dict, start_int: int, end_int: int):
    best = None
    for key in ("s1_asc", "s1_desc"):
        d = dates_by_key[key]
        sel = np.where((d >= start_int) & (d <= end_int))[0]
        if len(sel):
            ri = int(sel[np.argmax(d[sel])])
            cand = (int(d[ri]), -_ORBIT_ID[key], key, ri)
            best = cand if best is None or cand > best else best
    return best


def _pick_lulc_year(ly, year: int):
    before = np.where(ly < year)[0]
    yi = int(before[np.argmax(ly[before])]) if len(before) else int(np.argmin(ly))
    return yi, not len(before)


def build_fine(raw: dict | None, cache: dict, year: int, doy: int, fs: dict):
    """The 19-channel fine tensor + the 10 m LULC raster for (station, day D).

    raw: {"zg", "s2", "s1_asc", "s1_desc" (date-int arrays), "lulc_years", "has_dem"} or None

    S2   most recent raw S2 scene on or before D (365-day window) whose date has a cloud mask;
         fully clear by the token rule preferred, else most recent. Pixel valid = all kept
         bands non-zero AND cloud class not in CM_BAD_CLASSES.
    S1   most recent pass on or before D, either orbit. Valid = finite and non-zero in both
         bands. The 2x2 mean is taken in LINEAR power (dB -> linear -> mean -> dB), which is
         also the speckle reduction (§48.2 item 7).
    DEM  static; valid = finite and above the nodata floor.
    LULC the latest year strictly before D's year (causal); if the store has none that early,
         the earliest year it has — a mild look-ahead, reported in `lulc_lookahead`.

    Two paths, identical output (verify_fine_cache.py): if the station cache carries the
    precomputed per-scene arrays (prepare_fine_cache.py, 2026-09-29) they are looked up and
    NOTHING is read from the raw store; otherwise the raw scene is read and pooled here.
    Only the D-dependent channels (ages, orbit) and the scene choice happen per sample.

    Returns (fine (19,112,112) fp16, lulc (224,224) u1, info dict).
    """
    fine = np.zeros((FINE_CH, 112, 112), dtype=np.float32)
    lulc = np.full((224, 224), LULC_PAD, dtype=np.uint8)
    info = {"s2": False, "s1": False, "dem": False, "lulc": False, "lulc_lookahead": False}
    pre = "fine_s2" in cache
    if raw is None and not pre:
        return torch.from_numpy(fine.astype(np.float16)), torch.from_numpy(lulc), info
    start_int, end_int = _window_ints(year, doy)
    tdate = _target_date(year, doy)

    # ── S2 ──
    if pre:
        pick = _pick_s2(cache["fine_s2_cands"], cache["s2_nvalid"], start_int, end_int)
        # cands are (date, raw_row, tok_row, storage_row): the same ordering as the raw path
        s2 = None if pick is None else np.asarray(cache["fine_s2"][pick[3]], dtype=np.float32)
    else:
        s2_dates, pick, s2 = raw["s2"], None, None
        if len(s2_dates) and "s2_date_ints" in cache and "s2_cm" in cache:
            tok_idx = {int(d): i for i, d in enumerate(cache["s2_date_ints"])}
            ok = [(int(d), ri, tok_idx[int(d)]) for ri, d in enumerate(s2_dates) if int(d) in tok_idx]
            pick = _pick_s2(ok, cache["s2_nvalid"], start_int, end_int)
            if pick is not None:
                s2 = fine_s2_scene(raw["zg"]["s2/data"][pick[1]], cache["s2_cm"][pick[2]], fs)
    if s2 is not None and s2[10].any():
        fine[0:11] = s2
        fine[11]   = (s2[10] > 0) * (tdate - _int_to_date(pick[0])).days / fs["max_age"]
        info["s2"] = True

    # ── S1 ──
    s1_dates = ({k: cache[f"fine_{k}_dates"] for k in ("s1_asc", "s1_desc")} if pre
                else {k: raw[k] for k in ("s1_asc", "s1_desc")})
    best = _pick_s1(s1_dates, start_int, end_int)
    if best is not None:
        dint, _, key, ri = best
        s1 = (np.asarray(cache[f"fine_{key}"][ri], dtype=np.float32) if pre
              else fine_s1_scene(raw["zg"][f"{key}/data"][ri], fs))
        if s1[2].any():
            fine[12:15] = s1
            fine[15]    = (s1[2] > 0) * (tdate - _int_to_date(dint)).days / fs["max_age"]
            fine[16]    = (s1[2] > 0) * float(_ORBIT_ID[key])
            info["s1"]  = True

    # ── DEM ──
    dem = (np.asarray(cache["fine_dem"], dtype=np.float32) if pre
           else (fine_dem_static(raw["zg"]["dem/data"][0], fs) if raw["has_dem"] else None))
    if dem is not None and dem[1].any():
        fine[17:19] = dem
        info["dem"] = True

    # ── LULC ──
    ly = cache["fine_lulc_years"] if pre else raw["lulc_years"]
    if len(ly):
        yi, info["lulc_lookahead"] = _pick_lulc_year(ly, year)
        lulc = (np.asarray(cache["fine_lulc"][yi]) if pre else lulc_remap(raw["zg"]["lulc/data"][yi]))
        info["lulc"] = bool((lulc != LULC_PAD).any())

    return torch.from_numpy(fine.astype(np.float16)), torch.from_numpy(lulc), info


def _open_raw(dir_name: str) -> dict | None:
    """Raw imagery handle + its date tables, read once at init. None if the store is absent."""
    p = RAW_ROOT / f"{dir_name}.zarr"
    if not p.exists():
        return None
    try:
        rg = zarr.open_consolidated(str(p), mode="r")
    except KeyError:
        rg = zarr.open_group(str(p), mode="r")        # 0 of 998 carry .zmetadata (§46.5 item 9)
    return {
        "zg": rg,
        "s2": _raw_dates(rg, "s2") if "s2/data" in rg else np.zeros(0, np.int32),
        "s1_asc": _raw_dates(rg, "s1_asc") if "s1_asc/data" in rg else np.zeros(0, np.int32),
        "s1_desc": _raw_dates(rg, "s1_desc") if "s1_desc/data" in rg else np.zeros(0, np.int32),
        "lulc_years": (np.asarray(rg["lulc/years"][:], dtype=np.int32)
                       if "lulc/data" in rg and "lulc/years" in rg else np.zeros(0, np.int32)),
        "has_dem": "dem/data" in rg,
    }


def _load_lst22(cat: str, dir_name: str):
    """{date_int: row} + (n,22,22) f16 Kelvin, or None if consolidate_landsat_st.py wrote none."""
    p = LST_ROOT / cat / dir_name / "LANDSAT_ST" / f"{dir_name}_lst22.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        arr, dates = z["lst22"], z["dates"]
    return {int(d): i for i, d in enumerate(dates)}, arr

# ── Soil patch helpers ───────────────────────────────────────────────────────

def fill_soil_nans_with_validity(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill NaN pixels in a soil patch via nearest-neighbour propagation.
    patch : (21, 74, 74) float32
    Returns (filled (21,74,74) float32, channel_valid (21,) bool).

    §35.24 audit item 6. `distance_transform_edt(mask, return_indices=True)` on an ALL-True
    mask has no non-NaN pixel to point at: it returns the identity index field, `out[c]`
    comes back exactly as NaN as it went in, and the function reports success. That NaN then
    flows through SoilEncoder into the soil tokens, and because the drivers are a shared
    cross-attention memory (§35.18) the K/V cache for the whole SAMPLE goes NaN — every one
    of the 196 patches, and after the first backward pass every parameter. One dead channel
    in one station poisons the run and the traceback points at the loss, not at here.

    So a dead channel is set to 0.0 explicitly and flagged. 0.0 is the right filler only
    because the caller z-scores immediately afterwards and re-zeroes these channels, which
    puts them exactly at the training-set mean — the least informative value available.
    """
    out   = patch.astype(np.float32, copy=True)
    valid = np.ones(out.shape[0], dtype=bool)
    for c in range(out.shape[0]):
        mask = np.isnan(out[c])
        if not mask.any():
            continue
        if mask.all():
            out[c]   = 0.0
            valid[c] = False
            continue
        _, idx = distance_transform_edt(mask, return_indices=True)
        out[c] = out[c][tuple(idx)]
    return out, valid


def fill_soil_nans(patch: np.ndarray) -> np.ndarray:
    """Array-only wrapper. Kept because station_mean_probe.py:58 calls it positionally."""
    return fill_soil_nans_with_validity(patch)[0]



# ── ERA5 rolling slicer (no file I/O — works on pre-loaded numpy arrays) ────

def load_era5_rolling(cache_entry, year: int, target_doy: int):
    """
    Slice pre-loaded ERA5 arrays for the 365-day rolling window.

    Args:
        cache_entry: (values (N,18) float32, date_ints (N,) int32, doys (N,) int32) or None
    Returns:
        era5    : (365, 18) float32 numpy array
        doys    : (365,) int64 numpy array — absolute DOY, 0 = padding
        rel_pos : (365,) int64 numpy array — TRUE staleness, 364 = the target day

    §35.24 audit item 1 — why rel_pos has to be returned at all.

    This function COMPACTS: it takes whatever rows fell inside the window, in order, and
    right-aligns them (`out_era5[-l:] = era5_win[-l:]`). The model, meanwhile, added
    `rel_pos_emb(arange(365))` — it read the SLOT INDEX as the staleness. Those two agree
    only when the record is gapless and ends on the target day. They disagree whenever it
    is not:

        record ends 2021-03-14, target is 2021-09-30
          -> the last real row lands in slot 364 and is labelled "today"
          -> a 200-day-old temperature is presented as this morning's

        a 40-day hole in the middle of the window
          -> every row after the hole is shifted 40 slots later than it belongs
          -> the whole post-gap half of the year is systematically labelled too recent

    Neither crashes, neither shows up in a loss curve, and both corrupt exactly the signal
    the drivers exist to carry: how long ago it last rained. So the staleness now comes from
    the REAL row date and travels with the row.
    """
    n = 365
    if cache_entry is None:
        return (np.zeros((n, len(ERA5_VARS)), dtype=np.float32),
                np.zeros(n, dtype=np.int64),
                np.zeros(n, dtype=np.int64))

    values, date_ints, doy_arr = cache_entry
    start_int, end_int = _window_ints(year, target_doy)
    mask = (date_ints >= start_int) & (date_ints <= end_int)

    era5_win  = values[mask]
    doys_win  = doy_arr[mask].astype(np.int64)
    dates_win = date_ints[mask]

    out_era5 = np.zeros((n, len(ERA5_VARS)), dtype=np.float32)
    out_doys = np.zeros(n, dtype=np.int64)
    out_rel  = np.zeros(n, dtype=np.int64)
    l = min(len(doys_win), n)
    if l == 0:
        return out_era5, out_doys, out_rel
    out_era5[-l:] = era5_win[-l:]
    out_doys[-l:] = doys_win[-l:]

    # Absolute day number per row, vectorised. A 365-day window spans at most two calendar
    # years, so the year->ordinal lookup is a 1- or 2-entry table and searchsorted beats
    # constructing 365 datetimes.
    yrs  = (dates_win[-l:] // 10000).astype(np.int64)
    uy   = np.unique(yrs)
    base = np.array([datetime(int(y), 1, 1).toordinal() for y in uy], dtype=np.int64)
    abs_day    = base[np.searchsorted(uy, yrs)] + doys_win[-l:] - 1
    target_ord = (datetime(year, 1, 1) + timedelta(days=target_doy - 1)).toordinal()
    # 364 = the target day itself, 0 = 364 days before it. Padded slots keep 0, which the
    # model already ignores because era5_doys == 0 there.
    out_rel[-l:] = np.clip(364 - (target_ord - abs_day), 0, 364)
    return out_era5, out_doys, out_rel



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

TWSA_AVAILABLE_AFTER_DAYS = 45   # review A1: GRACE month usable from time_start + 45 d


def _ints_to_dt64(a) -> np.ndarray:
    """YYYYMMDD ints -> datetime64[D] (vectorised)."""
    a = np.asarray(a, dtype=np.int64)
    ym = (np.asarray(a // 10000 - 1970, dtype="timedelta64[Y]") + np.datetime64("1970", "Y")
          ).astype("datetime64[M]") + np.asarray(a // 100 % 100 - 1, dtype="timedelta64[M]")
    return ym.astype("datetime64[D]") + np.asarray(a % 100 - 1, dtype="timedelta64[D]")


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
    # Review A1: a GRACE value is a MONTHLY mean stamped with the period's time_start (the
    # 1st for a regular month), so "stamp <= D" let a sample on 3 March see the March mean —
    # up to ~30 days of future storage. A month is usable only from time_start + 45 d, which
    # also covers the irregular mid-month GRACE-FO periods. The window itself stays on the
    # stamps, so rel_pos (from the mid-month doys) stays within 0..364.
    avail = _ints_to_dt64(date_ints) + np.timedelta64(TWSA_AVAILABLE_AFTER_DAYS, "D")
    mask = (date_ints >= start_int) & (avail <= _ints_to_dt64([end_int])[0])

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



# ── Dataset ──────────────────────────────────────────────────────────────────

class SoilMoistureDataset(Dataset):
    """
    One sample = one (station, year, day-of-year) triple.

    Args:
        splits_csv       : path to station_splits.csv
        era5_stats_path  : path to csvs/era5_stats18.json  (from compute_era5_stats.py)
        driver_stats_path: path to csvs/driver_stats.json (from compute_driver_stats.py).
                           None -> driver_stats.json next to era5_stats_path. Required;
                           a missing file raises rather than silently skipping the SIF /
                           TWSA / soil z-scoring (§35.24 audit item 7).
        years            : list of years to include (default TRAIN_YEARS, never a silent
                           window straddling the §47 cut)
        min_obs          : minimum observed SM days per year to include
        category_filter  : list of categories to include, e.g. ["sm_only"]  (None = all)
        split_filter     : list of split values to include, e.g. ["train"]  (None = all)
        training         : if True, apply ERA5 value masking and SIF/TWSA modality dropout.
                           (The fine-path S2/S1 dropout is in model.py, on the device.)
        max_stations     : if set, stop scanning once this many stations have ADMITTED AT
                           LEAST ONE SAMPLE (smoke-test mode; §35.24 audit item 11).
        era5_require_full_window
                         : admit a sample only if all 365 window days are inside the
                           station's ERA5 record. Off by default; the count is printed.
        require_lst      : fail construction if no admitted station has an lst22 bundle —
                           a thermal head with nothing to supervise it is a silent no-op.
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
        driver_stats_path: str | None = None,
        era5_require_full_window: bool = False,
        require_lst:     bool        = False,
    ):
        self.training = training
        # §47: no silent 2016-2023 fallback. A caller that forgets `years` used to get a
        # window straddling the OOT cut, which is exactly how a temporal holdout leaks.
        self.years    = list(years) if years else list(TRAIN_YEARS)

        # ERA5 normalisation stats
        with open(era5_stats_path) as f:
            era5_stats = json.load(f)
        self._era5_means      = np.array(era5_stats["means"],  dtype=np.float32)
        self._era5_stds       = np.array(era5_stats["stds"],   dtype=np.float32)
        self._era5_log1p_prec = bool(era5_stats.get("log1p_precip", False))
        if len(self._era5_means) != len(ERA5_VARS):
            raise ValueError(
                f"{era5_stats_path} has {len(self._era5_means)} columns but {ERA5_ARRAY} has "
                f"{len(ERA5_VARS)}. The 19-column era5_stats.json belongs to the frozen "
                f"U-Net arm; this loader needs era5_stats18.json (§48.9 item 5).")

        # SIF / TWSA / soil normalisation stats. Raises if absent — see _load_driver_stats.
        if driver_stats_path is None:
            driver_stats_path = Path(era5_stats_path).with_name("driver_stats.json")
        _ds = _load_driver_stats(driver_stats_path)
        self._sif_mean   = float(_ds["sif"]["mean"])
        self._sif_std    = float(_ds["sif"]["std"])
        self._twsa_mean  = float(_ds["twsa"]["mean"])
        self._twsa_std   = float(_ds["twsa"]["std"])
        self._soil_mean  = np.asarray(_ds["soil"]["mean"], dtype=np.float32)[:, None, None]
        self._soil_std   = np.asarray(_ds["soil"]["std"],  dtype=np.float32)[:, None, None]
        self.driver_stats = _ds

        self._fine_stats = _load_fine_stats()

        # Fail loud if the scratch roots were purged (§46.5 item 8): a purge used to yield 0
        # samples and no error, because every station quietly returned None.
        for root, what in ((ZARR_ROOT, "token store"), (CACHE_ROOT, "§48 cache")):
            if not root.exists() or not any(root.glob("*/*")):
                raise FileNotFoundError(
                    f"{what} root {root} is missing or empty. Re-stage (restage_store.py) / "
                    f"rebuild (prepare_s48_cache.py) before constructing the dataset.")

        splits = pd.read_csv(splits_csv)
        if category_filter is not None:
            splits = splits[splits.apply(category_of, axis=1).isin(category_filter)]
        if split_filter is not None:
            splits = splits[splits["split"].isin(split_filter)]

        self.samples = []

        # Per-station caches, filled once in __init__; DataLoader workers inherit them by fork
        # (copy-on-write). The memmaps in _cache share page cache across every rank.
        self._zarr_groups  : dict[Path, zarr.Group | None] = {}
        self._era5_cache   : dict[Path, tuple | None] = {}
        self._sif_cache    : dict[Path, tuple | None] = {}
        self._twsa_cache   : dict[Path, tuple | None] = {}
        self._label_cache  : dict[Path, tuple]        = {}
        self._static_cache : dict[Path, dict]         = {}
        self._cache        : dict[Path, dict]         = {}   # §48 cache (pyramids, l12, cm)
        self._raw          : dict[Path, dict | None]  = {}   # raw imagery handle + dates
        self._lst          : dict[Path, tuple | None] = {}   # (date->row, (n,22,22) f16)

        # ── Audit bookkeeping (§35.24 audit item 11) ────────────────────────────
        skips        = Counter()
        sample_skips = Counter()
        era5_reject_by_station : dict[str, int] = defaultdict(int)
        n_dead_soil_ch = 0
        n_no_raw       = 0
        n_raw_missing  = Counter()        # modality -> stations whose raw store lacks it
        n_no_lst       = 0
        n_no_dem_pyr   = 0
        n_no_lulc_pyr  = 0
        admitted_dirs: set = set()
        rejected_dirs: set = set()

        for _, r in splits.iterrows():
            if max_stations is not None and len(admitted_dirs) >= max_stations:
                break

            cat      = category_of(r)
            dir_name = station_dir_name(r)
            sat_dir  = ZARR_ROOT / cat / dir_name

            if sat_dir in rejected_dirs:
                continue

            if not bool(r.get("soil_patch_ok", True)):
                skips["soil_patch_not_ok (splits_csv)"] += 1
                rejected_dirs.add(sat_dir)
                continue

            if sat_dir not in self._zarr_groups:
                zg = _open_zarr(sat_dir, cat)
                self._zarr_groups[sat_dir] = zg
                if zg is None:
                    skips["zarr_store_incomplete"] += 1
                    rejected_dirs.add(sat_dir)
                    continue

                cache = _load_station_cache(CACHE_ROOT / cat / dir_name)
                if cache is None:
                    skips["no_s48_cache (run prepare_s48_cache.py)"] += 1
                    self._zarr_groups[sat_dir] = None
                    rejected_dirs.add(sat_dir)
                    continue
                self._cache[sat_dir] = cache
                n_no_dem_pyr  += not bool(cache.get("dem_ok", False))
                n_no_lulc_pyr += not bool(cache.get("lulc_ok", False))

                raw = _open_raw(dir_name)
                self._raw[sat_dir] = raw
                if raw is None:
                    n_no_raw += 1
                else:
                    for k in ("s2", "s1_asc", "s1_desc", "lulc_years"):
                        n_raw_missing[k] += not len(raw[k])
                    n_raw_missing["dem"] += not raw["has_dem"]

                lst = _load_lst22(cat, dir_name)
                self._lst[sat_dir] = lst
                n_no_lst += lst is None

                self._era5_cache[sat_dir] = _load_zarr_era5(zg)

                # SIF / TWSA z-scored ONCE on the cached arrays (§35.24 item 7).
                _sif = _load_zarr_sif(zg)
                if _sif is not None:
                    _sif = ((np.asarray(_sif[0], dtype=np.float32) - self._sif_mean)
                            / (self._sif_std + 1e-8), _sif[1], _sif[2])
                self._sif_cache[sat_dir] = _sif

                _tw = _load_zarr_twsa(zg)
                if _tw is not None:
                    _tw = ((np.asarray(_tw[0], dtype=np.float32) - self._twsa_mean)
                           / (self._twsa_std + 1e-8), _tw[1], _tw[2])
                self._twsa_cache[sat_dir] = _tw

                # ── Soil: fill, check for dead channels, z-score ────────────
                if "soil" in zg:
                    _soil_np, _soil_ok = fill_soil_nans_with_validity(zg["soil"][:])
                else:
                    _soil_np = np.zeros((21, 74, 74), dtype=np.float32)
                    _soil_ok = np.zeros(21, dtype=bool)
                _n_dead = int((~_soil_ok).sum())
                if _n_dead > MAX_DEAD_SOIL_CHANNELS:
                    skips[f"soil_{_n_dead}_dead_channels"] += 1
                    self._zarr_groups[sat_dir] = None
                    rejected_dirs.add(sat_dir)
                    continue
                n_dead_soil_ch += _n_dead
                _soil_np = (_soil_np - self._soil_mean) / (self._soil_std + 1e-8)
                # Re-zero AFTER the z-score: 0.0 post-normalisation is the training mean.
                _soil_np[~_soil_ok] = 0.0
                self._static_cache[sat_dir] = {
                    "soil":     torch.from_numpy(np.ascontiguousarray(_soil_np)),
                    "dem_pyr":  torch.from_numpy(np.asarray(cache["dem_pyr"], np.float32)),
                    "lulc_pyr": torch.from_numpy(np.asarray(cache["lulc_pyr"], np.float32)),
                }

                # strict=True: refuse to guess an alignment between labels/qc and labels/sm.
                try:
                    lc = _load_zarr_labels(zg, strict=True)
                except ValueError:
                    skips["labels_length_mismatch (sm/dates/qc)"] += 1
                    self._zarr_groups[sat_dir] = None
                    rejected_dirs.add(sat_dir)
                    continue
                if lc is not None:
                    # QC fail-closed (§35.24 audit item 4): no QC source means climatology
                    # could be labelled as observation, so the station is refused.
                    _qc = lc[3]
                    if _qc is None:
                        skips["labels_qc_absent"] += 1
                        self._zarr_groups[sat_dir] = None
                        rejected_dirs.add(sat_dir)
                        continue
                    if bool(np.all(_qc == QC_NO_SOURCE)):
                        skips["labels_qc_no_source_sentinel"] += 1
                        self._zarr_groups[sat_dir] = None
                        rejected_dirs.add(sat_dir)
                        continue
                    self._label_cache[sat_dir] = lc

            era5_entry = self._era5_cache.get(sat_dir)
            if era5_entry is None:
                skips["no_era5"] += 1
                self._zarr_groups[sat_dir] = None
                rejected_dirs.add(sat_dir)
                continue
            era5_date_ints  = era5_entry[1]
            era5_first_int  = int(era5_date_ints[0])
            era5_last_int   = int(era5_date_ints[-1])
            era5_start_year = era5_first_int // 10000
            era5_end_year   = era5_last_int  // 10000

            _s2_di = self._cache[sat_dir].get("s2_date_ints")
            if _s2_di is None or not len(_s2_di):
                skips["no_s2_dates"] += 1
                self._zarr_groups[sat_dir] = None
                rejected_dirs.add(sat_dir)
                continue
            s2_years = (int(_s2_di[0]) // 10000, int(_s2_di[-1]) // 10000)

            if sat_dir not in self._label_cache:
                skips["no_sm_labels"] += 1
                self._zarr_groups[sat_dir] = None
                rejected_dirs.add(sat_dir)
                continue
            sm_np, depths, times, qc_np = self._label_cache[sat_dir]

            n_year_ok = 0
            for year in self.years:
                if not (era5_start_year <= year <= era5_end_year):
                    sample_skips["year_outside_era5_record"] += 1
                    continue
                if not (s2_years[0] <= year <= s2_years[1]):
                    sample_skips["year_outside_s2_record"] += 1
                    continue

                year_mask = times.year == year
                if not year_mask.any():
                    sample_skips["year_has_no_label_rows"] += 1
                    continue

                year_indices = np.where(year_mask)[0]
                assert qc_np is not None, (
                    f"{dir_name}: labels/qc is None after the QC admission check — the "
                    f"fail-closed guard in __init__ was bypassed.")
                # Only directly observed values (qc==0); gap-filled (qc==1) excluded.
                valid_days = np.any(qc_np[:, year_indices] == QC_OBSERVED, axis=0)
                if valid_days.sum() < min_obs:
                    sample_skips["year_below_min_obs"] += 1
                    continue

                for day_idx in np.where(valid_days)[0]:
                    doy = times[year_indices[day_idx]].day_of_year
                    # ERA5 admission, day-granular (§35.24 audit item 1).
                    target_int = _date_to_int(times[year_indices[day_idx]])
                    if not (era5_first_int <= target_int <= era5_last_int):
                        sample_skips["era5_target_day_outside_record"] += 1
                        era5_reject_by_station[dir_name] += 1
                        continue
                    ws_int, _ = _window_ints(year, int(doy))
                    if ws_int < era5_first_int:
                        sample_skips["era5_window_not_fully_covered"] += 1
                        if era5_require_full_window:
                            era5_reject_by_station[dir_name] += 1
                            continue

                    self.samples.append({
                        "sat_dir"    : sat_dir,
                        "year"       : year,
                        "doy"        : doy,
                        "time_idx"   : year_indices[day_idx],
                        "date_int"   : target_int,
                        "station_key": dir_name,
                    })
                    n_year_ok += 1

            if n_year_ok > 0:
                admitted_dirs.add(sat_dir)
            else:
                skips["no_sample_survived_year_filters"] += 1
                rejected_dirs.add(sat_dir)

        # ── Audit report (§35.24) ───────────────────────────────────────────────
        n_stations = len(set(s["station_key"] for s in self.samples))
        n_lst_samples = sum(
            1 for s in self.samples
            if self._lst.get(s["sat_dir"]) is not None and s["date_int"] in self._lst[s["sat_dir"]][0])
        self.n_lst_samples = n_lst_samples
        self.station_skips = dict(skips)          # §51.1: callers assert this is empty
        print(f"Dataset: {len(self.samples)} samples from {n_stations} stations; "
              f"{n_lst_samples} ({100.0 * n_lst_samples / max(1, len(self.samples)):.1f}%) "
              f"carry a Landsat ST target on their own day")

        if skips:
            print("  stations dropped, by reason:")
            for reason, n in skips.most_common():
                print(f"    {n:6d}  {reason}")
        if sample_skips:
            print("  station-years / samples dropped, by reason:")
            for reason, n in sample_skips.most_common():
                print(f"    {n:6d}  {reason}")

        n_era5_rej = sum(era5_reject_by_station.values())
        if n_era5_rej:
            worst = sorted(era5_reject_by_station.items(), key=lambda kv: -kv[1])[:20]
            print(f"  ERA5 coverage rejected {n_era5_rej} samples across "
                  f"{len(era5_reject_by_station)} stations"
                  f"{' (full-window mode ON)' if era5_require_full_window else ''}; worst:")
            for k, v in worst:
                print(f"    {v:6d}  {k}")

        print(f"  fine path: {n_no_raw} stations with no raw imagery store; raw store lacks "
              f"{dict(n_raw_missing)} (stations per modality) — those channels arrive zeroed "
              f"with valid = 0.")
        print(f"  statics: {n_no_dem_pyr} stations with no valid DEM pyramid, {n_no_lulc_pyr} "
              f"with no valid LULC pyramid; {n_dead_soil_ch} all-NaN soil channels zeroed.")
        print(f"  thermal: {n_no_lst} stations have no lst22 bundle.")
        if require_lst and n_lst_samples == 0:
            raise RuntimeError("require_lst=True but no admitted sample has a Landsat ST target "
                               "— run consolidate_landsat_st.py, or the thermal head trains on "
                               "nothing.")

    def __len__(self):
        return len(self.samples)

    # Threads per DataLoader worker for building a batch (2026-09-29). One sample costs ~1 s,
    # almost all of it waiting on ~4 cold GPFS reads (raw S2/S1 chunks, anchor memmap row)
    # of 100-300 ms each; file reads and blosc release the GIL, so a worker can overlap its
    # batch's reads. Every cache __getitem__ touches is built in __init__ and only read, so
    # samples are independent. 1 = the old sequential behaviour. Set after construction.
    io_threads: int = 1

    def __getitems__(self, indices):
        if self.io_threads <= 1 or len(indices) <= 1:
            return [self.__getitem__(i) for i in indices]
        return list(_io_pool(self.io_threads).map(self.__getitem__, indices))

    def _lst_dT(self, sat_dir, date_int: int, lst_obs) -> float:
        """Tile-mean LST - t2m_mean on date_int (K), or NaN.

        NaN when the scene has < LST_LEVEL_MIN_CELLS valid cells (a level from a few cells is
        a few pixels' temperature, not the tile's) or ERA5 has no row for that day.
        """
        f = np.asarray(lst_obs, dtype=np.float32)
        v = np.isfinite(f)
        if v.sum() < LST_LEVEL_MIN_CELLS:
            return float("nan")
        return float(f[v].mean()) - self._t2m(sat_dir, date_int)

    def _t2m(self, sat_dir, date_int: int) -> float:
        """Raw ERA5-Land t2m_mean (K) on date_int from the cache, or NaN if there is no row."""
        era = self._era5_cache.get(sat_dir)
        if era is None:
            return float("nan")
        values, date_ints, _ = era
        k = int(np.searchsorted(date_ints, date_int))
        if k >= len(date_ints) or int(date_ints[k]) != int(date_int):
            return float("nan")
        return float(values[k, T2M_MEAN_IDX])

    def lst_dT_stats(self) -> tuple[float, float, int]:
        """(mean, std, n) of the dT target over THIS dataset's samples.

        Called on the training set only; the values go into CONFIG and the checkpoint so val,
        eval and a resume use the training distribution. Deterministic, so a resume that
        recomputes it gets the same numbers.
        """
        vals = []
        for s in self.samples:
            lst = self._lst.get(s["sat_dir"])
            if lst is None:
                continue
            row = lst[0].get(s["date_int"])
            if row is None:
                continue
            d = self._lst_dT(s["sat_dir"], s["date_int"], lst[1][row])
            if np.isfinite(d):
                vals.append(d)
        if len(vals) < 2:
            return float("nan"), float("nan"), len(vals)
        a = np.asarray(vals, dtype=np.float64)
        return float(a.mean()), float(a.std()), len(a)

    def __getitem__(self, idx):
        s       = self.samples[idx]
        sat_dir = s["sat_dir"]
        year    = s["year"]
        doy     = s["doy"]
        if self._zarr_groups.get(sat_dir) is None:
            raise RuntimeError(
                f"sample {idx} ({s['station_key']} {s['year']}-{s['doy']}) has no open zarr "
                f"group. __init__ only appends samples for stations it cached, so the group "
                f"was cleared after construction.")
        cache = self._cache[sat_dir]

        # ── Trunk inputs: pooled history + the anchor, all on or before day D ──
        s2_pyr, s2_doys, s2_valid, s2_rel_pos, _ = load_history(cache, ("s2",), year, doy, MAX_S2)
        s1_pyr, s1_doys, s1_valid, s1_rel_pos, s1_orbit = load_history(
            cache, ("s1_asc", "s1_desc"), year, doy, MAX_S1)
        anchor_l12, anchor_rp, anchor_orbit, anchor_found = select_anchor(cache, year, doy)

        _static    = self._static_cache[sat_dir]
        soil_patch = _static["soil"]

        # ── Fine path: most recent raw imagery on or before day D ─────────
        fine, lulc, _ = build_fine(self._raw.get(sat_dir), cache, year, doy, self._fine_stats)

        # ── Thermal target: the Landsat scene on day D itself, or all-NaN ──
        lst_obs = torch.full((LST_N, LST_N), float("nan"), dtype=torch.float32)
        lst = self._lst.get(sat_dir)
        if lst is not None:
            row = lst[0].get(s["date_int"])
            if row is not None:
                lst_obs = torch.from_numpy(np.asarray(lst[1][row], dtype=np.float32))
        # §52 thermal LEVEL target: dT = tile-mean LST - ERA5-Land t2m_mean on day D (K).
        # Raw t2m from the cache, never the z-scored / 15%-masked input window.
        lst_dT = torch.tensor(self._lst_dT(sat_dir, s["date_int"], lst_obs), dtype=torch.float32)
        # §52 per-pixel target is lst_obs - lst_t2m; NaN off overpass days (no cell is valid then).
        lst_t2m = torch.tensor(self._t2m(sat_dir, s["date_int"]) if lst is not None
                               and lst[0].get(s["date_int"]) is not None else float("nan"),
                               dtype=torch.float32)

        # ── ERA5 — rolling 365-day window, numpy slice from cache ─────
        era5_np, era5_doys_np, era5_rel_np = load_era5_rolling(
            self._era5_cache.get(sat_dir), year, doy)
        if self._era5_log1p_prec:
            era5_np[:, PREC_IDX] = np.log1p(era5_np[:, PREC_IDX].clip(0))
        era5_np   = (era5_np - self._era5_means) / (self._era5_stds + 1e-8)
        era5      = torch.from_numpy(era5_np)
        era5_doys = torch.from_numpy(era5_doys_np)
        era5_rel_pos = torch.from_numpy(era5_rel_np)

        # Mask 15% of ERA5 VALUES during training (the rows stay, with their DOY and
        # staleness) — §35.24b item 2. Never at val/test time.
        if self.training:
            # Mask 15% of days as MISSING (doy 0 = key padding in model.py), not as value 0:
            # a z-scored 0 reads as "an average day" (~0.8 mm/d rain), which the model would
            # learn as real weather and never meet at eval (review #10, 2026-09-29).
            mask = (torch.rand(era5.shape[0]) < 0.15) & (era5_doys > 0)
            era5[mask] = 0.0
            era5_doys[mask] = 0

        sif_vals, sif_doys, sif_rel_pos, sif_valid = load_sif_rolling(
            self._sif_cache.get(sat_dir), year, doy)
        if self.training and random.random() < 0.5:
            sif_valid[:] = False

        twsa_vals, twsa_doys, twsa_rel_pos, twsa_valid = load_twsa_rolling(
            self._twsa_cache.get(sat_dir), year, doy)
        if self.training and random.random() < 0.5:
            twsa_valid[:] = False

        # ── ISMN labels — observed values only (qc==0) ───────────────
        sm_np, depths, _, qc_np = self._label_cache[sat_dir]
        label = torch.full((len(SM_DEPTHS),), float("nan"), dtype=torch.float32)
        for i, depth_str in enumerate(SM_DEPTHS):
            if depth_str in depths:
                d_idx = depths.index(depth_str)
                if qc_np[d_idx, s["time_idx"]] == QC_OBSERVED:
                    label[i] = float(sm_np[d_idx, s["time_idx"]])

        return {
            # ── Trunk: pooled pyramids (T,4,768) + the anchor's L12 ──
            "s2_pyr"        : s2_pyr,            # (MAX_S2, 4, 768) fp16
            "s2_doys"       : s2_doys,           # (MAX_S2,) long
            "s2_valid"      : s2_valid,          # (MAX_S2,) bool
            "s2_rel_pos"    : s2_rel_pos,        # (MAX_S2,) long — 364 = day D
            "s1_pyr"        : s1_pyr,            # (MAX_S1, 4, 768) fp16
            "s1_doys"       : s1_doys,
            "s1_valid"      : s1_valid,
            "s1_rel_pos"    : s1_rel_pos,
            "s1_orbit"      : s1_orbit,          # (MAX_S1,) long — 0 = ASC, 1 = DESC
            "anchor_l12"    : anchor_l12,        # (196, 768) fp16
            "anchor_rel_pos": torch.tensor(anchor_rp, dtype=torch.long),
            "anchor_orbit"  : torch.tensor(anchor_orbit, dtype=torch.long),  # 0 S2, 1 asc, 2 desc
            "anchor_found"  : torch.tensor(bool(anchor_found)),  # False = zero map, key-padded in model.py
            "dem_pyr"       : _static["dem_pyr"],    # (4, 768) fp32
            "lulc_pyr"      : _static["lulc_pyr"],   # (4, 768) fp32

            # ── Fine path (model.py FINE_* layout) ──
            "fine"          : fine,              # (19, 112, 112) fp16
            "lulc"          : lulc,              # (224, 224) uint8, LULC_PAD = nodata

            # ── Drivers ──
            "soil_patch"    : soil_patch,        # (21, 74, 74) fp32 — NaN-free, z-scored
            "era5"          : era5,              # (365, 18) fp32 — z-scored
            "era5_doys"     : era5_doys,         # (365,) long
            "era5_rel_pos"  : era5_rel_pos,      # (365,) long — TRUE staleness, 364 = today
            "sif"           : sif_vals,          # (MAX_SIF, 1) fp32 — z-scored
            "sif_doys"      : sif_doys,
            "sif_rel_pos"   : sif_rel_pos,
            "sif_valid"     : sif_valid,
            "twsa"          : twsa_vals,         # (MAX_TWSA, 1) fp32 — z-scored
            "twsa_doys"     : twsa_doys,
            "twsa_rel_pos"  : twsa_rel_pos,
            "twsa_valid"    : twsa_valid,

            # ── Targets and identity ──
            "label"         : label,             # (3,) — NaN where the depth has no obs
            "lst_obs"       : lst_obs,           # (22, 22) Kelvin — all NaN off overpass days
            "lst_dT"        : lst_dT,            # ()  K, tile LST - t2m_mean; NaN if no level
            "lst_t2m"       : lst_t2m,           # ()  K, raw t2m_mean on a scene day, else NaN
            "station_key"   : s["station_key"],
            "year"          : s["year"],
            "doy"           : s["doy"],
        }
