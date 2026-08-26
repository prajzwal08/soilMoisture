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
      labels/         soil_moisture, depth, time, qc
      soil/           soil_patch (21, 74, 74)

§35.24 audit. Everything this loader does about MISSING data now fails closed: an
acquisition with no cloud mask is invalid rather than clear, an orbit with no token_mask
contributes nothing rather than everything, a station with no QC source is dropped rather
than assumed observed, and a missing driver_stats.json raises rather than quietly leaving
SIF/TWSA/soil unnormalised. Each of those removes data silently by construction, so each is
counted and printed at the end of __init__.
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
# the two classes you most want to KEEP. A reviewer who assumes SCL reads the line below as
# "throw away all the good pixels" and either 'fixes' it or, worse, copies the pattern into
# a new file. The list is correct for THIS product.
CM_BAD_CLASSES = [3, 4, 5, 255]

# Fraction of a 16x16 = 256-pixel patch that may be bad before the token is rejected.
# 0.01 of 256 is 2.56 px, so this admits at most TWO bad pixels: 3 px = 1.17% and fails.
# That is effectively zero tolerance, and it is deliberate at 10 m resolution -- a token is
# 160 m across, thin cirrus at its edge contaminates the whole 768-d embedding, and S2 is
# the modality the patchwise hypothesis rests on. Flagged for review in the §35.24 audit;
# not changed here.
CM_MAX_BAD_FRAC = 0.01

# S1 orbit identity, emitted as `s1_orbit` so the model can tell an ascending pass from a
# descending one. §35.24 audit item 8: the loader merged both orbits into one date-sorted
# list and threw the key away, so a VV backscatter step that is pure geometry (different
# incidence angle, different azimuth) was indistinguishable from a wetting event.
ORBIT_ASC, ORBIT_DESC = 0, 1
_ORBIT_ID = {"s1_asc": ORBIT_ASC, "s1_desc": ORBIT_DESC}

# labels/qc sentinel written by create_token_zarr.py when the source NetCDF carried NEITHER
# `soil_moisture_qc` NOR `quality_flag`. The producer used to default to zeros, i.e. "every
# day directly observed", which made climatological gap-fill indistinguishable from a real
# measurement and trained the model on it. §35.24 audit item 4.
QC_OBSERVED   = 0
QC_NO_SOURCE  = 255

# A soil channel that is NaN over the whole 74x74 patch cannot be nearest-neighbour filled;
# it is set to 0 (== the dataset mean once z-scored) and flagged dead. One or two dead
# channels is a tolerable hole in a 21-channel stack; more than that and the station's soil
# block is fiction, so the station is dropped. §35.24 audit item 6.
MAX_DEAD_SOIL_CHANNELS = 2


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


def _narrowed_for(flag, key: str) -> bool:
    """Resolve the per-source narrowing flag for one modality key.

    The L12 cache is assembled PER KEY (§35.24b item 1): a station can legitimately have its
    s2 array from a full-width /dev/shm memmap and its s1_desc array from the narrowed zarr
    read in the same dict. A single per-station bool would then be wrong for one of them and
    silently index the wrong patch axis, so the flag travels as {key: bool}. A bare bool is
    still accepted for direct callers.
    """
    if isinstance(flag, dict):
        return bool(flag.get(key, False))
    return bool(flag)


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
      * `token_mask` now arrives initialised to FALSE (§35.24 audit items 2 and 3). It is
        written only for slots that were actually filled AND matched a cloud-mask entry
        (S2) or a stored token_mask row (S1), so an acquisition the quality layer never
        saw stays invalid instead of being reported clear. The `& valid_acq` below is
        still required on top of that for padded and NaN-skipped slots, which have a
        mask row but no data.
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
                          patch_token_dropout: float = 0.0,
                          l12_narrowed=False):
    """
    Load S2 L12 tokens for patches `token_sel` over the 365-day rolling window.

    date_cache:    precomputed per-orbit date info from _zarr_date_cache (eliminates zarr date reads)
    cm_token_mask: precomputed (N_cm, 14, 14) bool quality array from _cm_token_mask_cache
    l12_np:        preloaded L12 tokens from RAM/shm (eliminates chunk reads for history tokens)
    l12_narrowed:  bool, or {key: bool}. True when `l12_np` has already been sliced down to
                   the K selected patches, i.e. its shape is (N, K, 768) and not
                   (N, 196, 768) (§35.24 audit item 12). The /dev/shm memmaps written by
                   train.py are still full width, so this is per-source and never assumed.
    training:      if True, applies random per-patch dropout to token_mask. Defaults OFF —
                   see the note on `patch_token_dropout` in SoilMoistureDataset.__init__.
    patch_token_dropout: defaults to 0.0. It used to default to 0.5 here while the dataset
                   and train.py both passed 0.0, so any caller that reached the loader
                   directly (eval scripts, probes) silently deleted half of patch 105's
                   history. §35.24 audit item 10.

    Returns (l12, doys, valid_acq, rel_pos, hist_valid) — l12 is (max_acq, K, 768) fp16.
    """
    sel        = np.asarray(token_sel)
    tsl        = _token_slice(sel)
    K          = len(sel)
    l12        = torch.zeros(max_acq, K, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    # (14,14) of bools is 196 BYTES against 294 KB of tokens, so it stays full and is
    # indexed down in _finalise_history. It was never the reason for the wide read.
    #
    # FAIL CLOSED (§35.24 audit item 2). This used to be torch.ones, and it is written only
    # inside `if date_str in cm_d2i`. A station whose zarr has no cm/masks group therefore
    # reported all 60 S2 acquisitions cloud-free — thick cumulus scored as bare soil, with
    # nothing anywhere raising. False means "no quality evidence for this acquisition",
    # which is the only defensible default for a mask.
    token_mask = torch.zeros(max_acq, 14, 14, dtype=torch.bool)
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
    narrowed = _narrowed_for(l12_narrowed, "s2") and l12_np is not None
    cm_z     = None if (cm_token_mask is not None or not ("cm/masks" in zg)) else zg["cm/masks"]

    # out_i is advanced only on a SUCCESSFUL write, so a NaN acquisition is skipped rather
    # than leaving a hole. §35.24b item 3: this loop used to be `enumerate(win_idx)`, which
    # burned a slot per NaN acquisition while load_s1_rolling_zarr compacted. Both were
    # masked correctly by `doys > 0`, so neither produced a wrong number — but a station
    # with several NaN acquisitions silently carried fewer than MAX_S2 usable ones, and two
    # loaders with the same signature and different indexing semantics is a trap for the
    # next person. They now agree: compact, oldest-first, padding at the tail.
    out_i = 0
    for src_i in win_idx:
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
        tok_np = (np.asarray(tokens_z[src_i]) if narrowed
                  else _read_patch_tokens(tokens_z, src_i, tsl, sel))
        tok = torch.from_numpy(tok_np)
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
                # Lazy per-acquisition fallback, used only when the station's 14x14 cache
                # was not built. CM_BAD_CLASSES is the SEnSeIv2-SegFormerB2 class list, NOT
                # Sentinel-2 SCL — see the table at the top of this file before touching it.
                cm    = cm_z[cm_idx]
                cm_4d = cm[:224, :224].reshape(14, 16, 14, 16)
                bad_frac = np.isin(cm_4d, CM_BAD_CLASSES).mean(axis=(1, 3))
                token_mask[out_i] = torch.from_numpy(bad_frac <= CM_MAX_BAD_FRAC)

        out_i += 1

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
                          patch_token_dropout: float = 0.0,
                          l12_narrowed=False):
    """Load S1 L12 tokens (ASC + DESC merged) for patches `token_sel`.

    date_cache:          precomputed per-orbit date info (eliminates zarr date reads)
    s1_token_mask_cache: precomputed {orbit: (N,14,14) bool} from _s1_token_mask_cache
    l12_asc_np / l12_desc_np: preloaded RAM arrays; eliminates L12 chunk reads.
    l12_narrowed:        bool, or {key: bool}. ASC and DESC are resolved SEPARATELY — one
                         orbit can come from a full-width shm memmap while the other came
                         from the narrowed zarr read (§35.24b item 1).
    training:            if True, applies random per-patch dropout before masking.
    patch_token_dropout: defaults to 0.0, same reasoning as the S2 loader (§35.24 item 10).

    Returns (l12, doys, valid_acq, rel_pos, hist_valid, orbit) — the trailing `orbit` is
    (max_acq,) long, 0 = ASC and 1 = DESC (§35.24 audit item 8).
    """
    sel        = np.asarray(token_sel)
    tsl        = _token_slice(sel)
    K          = len(sel)
    l12        = torch.zeros(max_acq, K, 768, dtype=torch.float16)
    doys       = torch.zeros(max_acq, dtype=torch.long)
    # FAIL CLOSED (§35.24 audit item 3). tm_np_map[orbit] is None whenever the store has no
    # `{orbit}/token_mask` array — compute_s1_dem_lulc_token_masks.py writes it, and a store
    # that predates that script has none. With a ones-init the whole S1 history then read
    # back valid, layover/shadow and all. Zeros means an orbit with no stored mask
    # contributes nothing rather than contributing garbage.
    token_mask = torch.zeros(max_acq, 14, 14, dtype=torch.bool)
    rel_pos    = torch.zeros(max_acq, dtype=torch.long)
    orbit      = torch.zeros(max_acq, dtype=torch.long)

    start_int, end_int = _window_ints(year, target_doy)
    ws, td = _window_datetimes(year, target_doy)
    entries: list[tuple[str, object, int, str, int | None, int | None, bool]] = []

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
            _narrow     = _narrowed_for(l12_narrowed, orbit_key) and _src is not None
            for i in idx_arr:
                entries.append((orbit_dates[i], tokens_src, int(i), orbit_key,
                                 int(doys_a[i]), int(years_a[i]), _narrow))
        else:
            orbit_dates = [str(d) for d in zg[f"{orbit_key}/dates"][:]]
            _src = l12_np_map[orbit_key]
            tokens_src  = _src if _src is not None else zg[f"{orbit_key}/l12"]
            _narrow     = _narrowed_for(l12_narrowed, orbit_key) and _src is not None
            for i, d in enumerate(orbit_dates):
                if _in_window(d, ws, td):
                    entries.append((d, tokens_src, i, orbit_key, None, None, _narrow))

    if not entries:
        return (*_empty_history(max_acq, token_sel), orbit)

    entries.sort(key=lambda x: x[0])
    entries = entries[-max_acq:]

    out_i = 0
    for date_str, tokens_z, src_i, orbit_key, cached_doy, cached_year, narrowed in entries:
        tok_np = (np.asarray(tokens_z[src_i]) if narrowed
                  else _read_patch_tokens(tokens_z, src_i, tsl, sel))
        tok = torch.from_numpy(tok_np)
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
        # Orbit identity travels with the acquisition, not with the slot: `entries` is sorted
        # by DATE across both orbits, so slot i is ASC or DESC depending on which satellite
        # pass happened to be that day.
        orbit[out_i]   = _ORBIT_ID[orbit_key]
        if tm_np_map[orbit_key] is not None:
            token_mask[out_i] = torch.from_numpy(tm_np_map[orbit_key][src_i])
        out_i += 1

    return (*_finalise_history(l12, token_mask, doys, rel_pos, training,
                               token_sel, patch_token_dropout), orbit)


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
        cache_entry: (values (N,19) float32, date_ints (N,) int32, doys (N,) int32) or None
    Returns:
        era5    : (365, 19) float32 numpy array
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
        driver_stats_path: path to csvs/driver_stats.json (from compute_driver_stats.py).
                           None -> driver_stats.json next to era5_stats_path. Required;
                           a missing file raises rather than silently skipping the SIF /
                           TWSA / soil z-scoring (§35.24 audit item 7).
        years            : list of years to include (default 2016–2023)
        min_obs          : minimum observed SM days per year to include
        category_filter  : list of categories to include, e.g. ["sm_only"]  (None = all)
        split_filter     : list of split values to include, e.g. ["train"]  (None = all)
        training         : if True, apply SIF/TWSA modality dropout (p=0.5 each)
        max_stations     : if set, stop scanning splits once this many stations have
                           ADMITTED AT LEAST ONE SAMPLE (smoke-test mode). It used to count
                           entries in _zarr_groups, which includes stations whose store is
                           incomplete or whose labels never survived the filters, so
                           `--max-stations 20` could yield 11 (§35.24 audit item 11).
        era5_require_full_window
                         : if True, a sample is admitted only when all 365 days of its
                           window fall inside the station's ERA5 record. Off by default —
                           the trailing edge (target day covered) is always enforced, and
                           era5_rel_pos now declares a short window honestly, so the
                           leading edge costs samples without fixing a correctness bug.
                           The count that WOULD be dropped is printed either way.
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
        driver_stats_path: str | None = None,
        era5_require_full_window: bool = False,
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
        # Contiguous in both cases -- slice(105,106) or slice(0,196) -- so the L12 preload
        # below can narrow the store with a basic slice instead of fancy indexing (§35.24
        # audit item 12).
        self._tsl = _token_slice(self._token_sel)
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

        # SIF / TWSA / soil normalisation stats. Raises if absent — see _load_driver_stats.
        if driver_stats_path is None:
            driver_stats_path = Path(era5_stats_path).with_name("driver_stats.json")
        _ds = _load_driver_stats(driver_stats_path)
        self._sif_mean   = float(_ds["sif"]["mean"])
        self._sif_std    = float(_ds["sif"]["std"])
        self._twsa_mean  = float(_ds["twsa"]["mean"])
        self._twsa_std   = float(_ds["twsa"]["std"])
        # (21,1,1) so it broadcasts straight onto the (21,74,74) patch
        self._soil_mean  = np.asarray(_ds["soil"]["mean"], dtype=np.float32)[:, None, None]
        self._soil_std   = np.asarray(_ds["soil"]["std"],  dtype=np.float32)[:, None, None]
        self.driver_stats = _ds

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
        # True when _l12_cache[sat_dir] holds (N,K,768) arrays rather than (N,196,768).
        # The /dev/shm memmaps are written full-width by train.py::_preload_l12_to_shm, so
        # this is per-station and the loaders are told explicitly (§35.24 audit item 12).
        self._l12_narrowed : dict[Path, dict[str, bool]] = {}

        # ── Audit bookkeeping (§35.24 audit item 11) ────────────────────────────
        # Six independent `continue`s used to drop stations with no output at all beyond a
        # surviving-station count, so "993 stations became 641" was unattributable. Every
        # rejection now lands in a counter and the tally is printed.
        skips        = Counter()          # reason -> stations dropped
        sample_skips = Counter()          # reason -> individual samples dropped
        era5_reject_by_station : dict[str, int] = defaultdict(int)
        n_no_cm_group   = 0               # stations with no cm/masks group at all
        n_s2_acq_no_cm  = 0               # S2 acquisitions with no cloud-mask entry
        n_s2_acq_total  = 0
        n_s1_no_tm      = Counter()       # orbit -> stations with no {orbit}/token_mask
        n_s1_acq_no_tm  = 0
        n_s1_acq_total  = 0
        n_dead_soil_ch  = 0               # all-NaN soil channels seen, across kept stations
        n_l12_shm_partial = Counter()     # key -> stations where shm was partial and zarr filled in
        n_dem_missing   = 0
        n_lulc_missing  = 0
        admitted_dirs: set = set()        # stations that contributed >= 1 sample
        # station_splits.csv can carry several rows per station. Once a station has been
        # rejected it must not be re-examined, or the second row lands in a DIFFERENT
        # counter (the caches it needs were never filled) and the tally double-counts.
        rejected_dirs: set = set()

        for _, r in splits.iterrows():
            # max_stations counts ADMITTED stations, not opened ones. Checked here rather
            # than inside the cache-fill branch so a station that is opened and then thrown
            # out by a later filter does not consume one of the N slots.
            if max_stations is not None and len(admitted_dirs) >= max_stations:
                break

            has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
            has_fl = str(r.get("has_flux",          "False")).lower() == "true"
            cat    = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")

            # Build directory name matching the on-disk convention
            if str(r["source_network"]) == "ISMN":
                dir_name = f"ISMN_{r['network']}_{r['station_name']}"
            else:
                dir_name = f"{r['source_network']}_{r['station_id']}"

            sat_dir = ZARR_ROOT / cat / dir_name

            if sat_dir in rejected_dirs:
                continue

            if not bool(r.get("soil_patch_ok", True)):
                skips["soil_patch_not_ok (splits_csv)"] += 1
                rejected_dirs.add(sat_dir)
                continue

            # Load per-station data into memory once (subsequent rows reuse caches)
            if sat_dir not in self._zarr_groups:
                zg = _open_zarr(sat_dir, cat)
                self._zarr_groups[sat_dir] = zg
                if zg is None:
                    # _open_zarr returns None when the store has no .complete sentinel.
                    # Named explicitly so it stops being conflated with "no ERA5" below,
                    # which is where every one of these used to land.
                    skips["zarr_store_incomplete"] += 1
                    rejected_dirs.add(sat_dir)
                    continue
                if zg is not None:
                    self._era5_cache[sat_dir]  = _load_zarr_era5(zg)

                    # SIF / TWSA are z-scored ONCE here rather than per sample: the slicers
                    # work on these cached arrays, so normalising the cache is identical to
                    # normalising every window and costs one pass per station (§35.24 item 7).
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

                    # ── L12 token source, assembled PER KEY (§35.24b item 1) ────
                    # This used to be per STATION:
                    #     if shm_l12:                 <- truthy if ANY key was found
                    #         self._l12_cache[sat_dir] = shm_l12
                    #     if sat_dir not in self._l12_cache and ...:
                    #         ... zarr fallback ...   <- now unreachable
                    # and _load_l12_shm deliberately tolerates a PARTIALLY written station
                    # (it documents the rank-0-died-between-bin-and-meta case). So a station
                    # with s2 in /dev/shm but no s1_desc got no cache entry for s1_desc at
                    # all, and every __getitem__ fell back to lazy GPFS chunk reads for that
                    # orbit — a large throughput cliff with nothing in the logs. Merge key by
                    # key, and count every key that had to fall back.
                    _l12_src: dict[str, np.ndarray] = {}
                    _l12_nar: dict[str, bool]       = {}
                    _shm_keys: set = set()
                    if shm_dir is not None:
                        shm_l12 = _load_l12_shm(dir_name, shm_dir)
                        if shm_l12:
                            # Full width, but a memmap costs no resident RAM until touched
                            # and _read_patch_tokens faults only the page patch k lives on.
                            _l12_src.update(shm_l12)
                            _shm_keys = set(shm_l12)
                            for _k in shm_l12:
                                _l12_nar[_k] = False
                    if not DISABLE_L12_CACHE:
                        for _k in ("s2", "s1_asc", "s1_desc"):
                            if _k in _l12_src or f"{_k}/l12" not in zg:
                                continue
                            if _shm_keys:
                                # shm covered this station but not this key.
                                n_l12_shm_partial[_k] += 1
                            # §35.24 audit item 12. This used to be `zg[f"{k}/l12"][:]` —
                            # the whole (N,196,768) fp16 slab, ~145 GB across the 993
                            # stations, resident, of which training reads exactly one of the
                            # 196 columns. §35.16 promised the narrowing and only the READ
                            # side landed. Slicing the token axis here takes it to ~0.74 GB.
                            # Chunking is (T_TOKENS, 196, 768), so the token axis is one
                            # chunk and DECOMPRESSION cost is unchanged — this buys resident
                            # memory, not startup time.
                            _l12_src[_k] = (zg[f"{_k}/l12"][:, self._tsl, :]
                                            if self._tsl is not None
                                            else zg[f"{_k}/l12"][:])
                            _l12_nar[_k] = self._tsl is not None
                    if _l12_src:
                        self._l12_cache[sat_dir]    = _l12_src
                        self._l12_narrowed[sat_dir] = _l12_nar

                    # ── DEM / LULC, with their validity masks (§35.24 audit item 5) ──
                    # The masks were already being loaded into _static_cache and then never
                    # emitted, so the model treated a nodata-filled DEM token as terrain.
                    # And a station with NO dem array at all was handed torch.zeros(196,768)
                    # — a fabricated token that is not flat ground, it is whatever the
                    # decoder decides an all-zero L12 vector means. Both now come with a
                    # per-patch validity flag, and the fabricated case is flagged invalid.
                    _has_dem  = "dem"  in zg
                    _has_lulc = "lulc" in zg
                    n_dem_missing  += (not _has_dem)
                    n_lulc_missing += (not _has_lulc)
                    _dem_l12  = (torch.from_numpy(zg["dem"][:]) if _has_dem
                                 else torch.zeros(N_TOKENS, 768, dtype=torch.float16))
                    _lulc_l12 = (torch.from_numpy(zg["lulc"][:]) if _has_lulc
                                 else torch.zeros(N_TOKENS, 768, dtype=torch.float16))
                    # Fail closed: no stored mask -> no evidence the token is real.
                    # verify_zarr_store.py:70 lists dem_token_mask / lulc_token_mask as
                    # REQUIRED for a complete store, so this branch should be unreachable
                    # on a verified store and is not costing valid terrain.
                    _dem_tm   = (torch.from_numpy(np.asarray(zg["dem_token_mask"][:]).astype(bool))
                                 if ("dem_token_mask" in zg and _has_dem)
                                 else torch.zeros(14, 14, dtype=torch.bool))
                    _lulc_tm  = (torch.from_numpy(np.asarray(zg["lulc_token_mask"][:]).astype(bool))
                                 if ("lulc_token_mask" in zg and _has_lulc)
                                 else torch.zeros(14, 14, dtype=torch.bool))

                    # ── Soil: fill, check for dead channels, z-score ────────────
                    if "soil" in zg:
                        _soil_np, _soil_ok = fill_soil_nans_with_validity(zg["soil"][:])
                    else:
                        _soil_np = np.zeros((21, 74, 74), dtype=np.float32)
                        _soil_ok = np.zeros(21, dtype=bool)
                    _n_dead = int((~_soil_ok).sum())
                    if _n_dead > MAX_DEAD_SOIL_CHANNELS:
                        # More than a couple of the 21 channels are pure invention. Drop the
                        # station rather than feed the SoilEncoder a stack that is mostly
                        # dataset means (§35.24 audit item 6).
                        skips[f"soil_{_n_dead}_dead_channels"] += 1
                        self._zarr_groups[sat_dir] = None
                        rejected_dirs.add(sat_dir)
                        continue
                    n_dead_soil_ch += _n_dead
                    _soil_np = (_soil_np - self._soil_mean) / (self._soil_std + 1e-8)
                    # Re-zero AFTER the z-score: a dead channel filled with 0.0 and then
                    # normalised would sit at -mean/std, an extreme value the encoder would
                    # read as a strong signal. 0.0 post-normalisation is the training mean.
                    _soil_np[~_soil_ok] = 0.0

                    self._static_cache[sat_dir] = {
                        "dem":            _dem_l12,
                        "lulc":           _lulc_l12,
                        "dem_token_mask": _dem_tm,
                        "lulc_token_mask":_lulc_tm,
                        "soil":           torch.from_numpy(np.ascontiguousarray(_soil_np)),
                        "soil_ch_valid":  torch.from_numpy(_soil_ok),
                    }

                    # strict=True: refuse to guess an alignment between labels/qc and
                    # labels/sm. A station we cannot align is dropped, counted, not fudged.
                    try:
                        lc = _load_zarr_labels(zg, strict=True)
                    except ValueError:
                        skips["labels_length_mismatch (sm/dates/qc)"] += 1
                        self._zarr_groups[sat_dir] = None
                        rejected_dirs.add(sat_dir)
                        continue
                    if lc is not None:
                        # ── QC fail-closed (§35.24 audit item 4) ────────────────
                        # create_token_zarr.py used to default labels/qc to zeros, i.e.
                        # "every day directly observed", whenever the source NetCDF carried
                        # neither soil_moisture_qc nor quality_flag. The preprocessing
                        # pipeline gap-fills with a month-day climatology, so those zeros
                        # meant the model trained on climatology labelled as observation —
                        # which is exactly a station-mean predictor wearing a ground-truth
                        # badge. The producer now writes 255 for that case and the dataset
                        # refuses the station rather than trusting the flag.
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

                    # ── Precompute GPFS-hot-path data once at init ──────────────
                    # (a) CM token-mask quality: (N_cm, 14, 14) bool.
                    #
                    # What is RESIDENT here is the 14x14 bool derivation — 196 bytes per
                    # acquisition, ~39 KB for a 200-date station — not the raw masks. That
                    # matters because every DDP rank builds this cache independently (only
                    # L12 is shared through /dev/shm), so anything kept here is multiplied
                    # by the rank count. The raw (N,224,224) uint8 array is ~10 MB/station
                    # and would be ~40 GB of pure duplication across 1000 stations x 4 ranks
                    # if it were held; it is ~250x smaller as a token mask (§35.24b item 4).
                    #
                    # The read is now BLOCKED rather than `zg["cm/masks"][:]`. The old form
                    # was never resident, but it did decompress the whole array into one
                    # transient allocation; going block by block bounds that peak to one
                    # chunk-row and keeps `bad_frac` bit-identical (the reduction is per
                    # acquisition, so blocking cannot change a single value).
                    cm_qm = None
                    if "cm/masks" in zg and "cm/dates" in zg:
                        _cm_arr = zg["cm/masks"]
                        _n_cm   = int(_cm_arr.shape[0])
                        cm_qm   = np.empty((_n_cm, 14, 14), dtype=bool)
                        _blk    = 128
                        for _b0 in range(0, _n_cm, _blk):
                            _b1    = min(_b0 + _blk, _n_cm)
                            _chunk = _cm_arr[_b0:_b1, :224, :224]
                            _c4    = _chunk.reshape(_b1 - _b0, 14, 16, 14, 16)
                            # CM_BAD_CLASSES is SEnSeIv2-SegFormerB2 (3 thin cloud, 4 thick
                            # cloud, 5 cloud shadow, 255 nodata) — NOT Sentinel-2 SCL, where
                            # 4 and 5 are the good vegetation classes. See the class table
                            # at the top of this file.
                            _bad   = np.isin(_c4, CM_BAD_CLASSES).mean(axis=(2, 4))
                            cm_qm[_b0:_b1] = (_bad <= CM_MAX_BAD_FRAC)
                    else:
                        n_no_cm_group += 1
                    self._cm_token_mask_cache[sat_dir] = cm_qm

                    # (b) S1 token masks per orbit: {orbit: (N, 14, 14) bool}
                    s1_tm: dict[str, np.ndarray] = {}
                    for _ok in ("s1_asc", "s1_desc"):
                        _mk = f"{_ok}/token_mask"
                        if _mk in zg:
                            s1_tm[_ok] = np.asarray(zg[_mk][:])
                        elif f"{_ok}/l12" in zg:
                            # Orbit has tokens but no quality mask — every one of its
                            # acquisitions is now dropped by the fail-closed init in
                            # load_s1_rolling_zarr. Count it so the loss is visible.
                            n_s1_no_tm[_ok] += 1
                            n_s1_acq_no_tm  += int(zg[f"{_ok}/l12"].shape[0])
                    self._s1_token_mask_cache[sat_dir] = s1_tm
                    for _ok in ("s1_asc", "s1_desc"):
                        if f"{_ok}/l12" in zg:
                            n_s1_acq_total += int(zg[f"{_ok}/l12"].shape[0])

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

                    # How much S2 the fail-closed cloud mask actually costs this station:
                    # an acquisition whose date has no cm entry now contributes nothing.
                    if "s2" in _dc:
                        _cm_keys = set(_dc.get("cm", {}).get("date_to_idx", {}))
                        n_s2_acq_total  += len(_dc["s2"]["dates"])
                        n_s2_acq_no_cm  += sum(1 for d in _dc["s2"]["dates"] if d not in _cm_keys)

            # ERA5 year range from cache (fast int arithmetic — no file I/O)
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

            # S2 year range: from date cache (no zarr I/O)
            _s2_dc = self._zarr_date_cache.get(sat_dir, {}).get("s2")
            if _s2_dc is None:
                skips["no_s2_dates"] += 1
                self._zarr_groups[sat_dir] = None
                rejected_dirs.add(sat_dir)
                continue
            s2_years = (int(_s2_dc["date_ints"][0]) // 10000,
                        int(_s2_dc["date_ints"][-1]) // 10000)

            if sat_dir not in self._label_cache:
                skips["no_sm_labels"] += 1
                self._zarr_groups[sat_dir] = None
                rejected_dirs.add(sat_dir)
                continue
            sm_np, depths, times, qc_np = self._label_cache[sat_dir]

            n_year_ok = 0
            for year in self.years:
                # Cheap year-level pre-filter only — the binding ERA5 test is the
                # day-granular one inside the day loop below.
                if not (era5_start_year <= year <= era5_end_year):
                    sample_skips["year_outside_era5_record"] += 1
                    continue
                if s2_years is None or not (s2_years[0] <= year <= s2_years[1]):
                    sample_skips["year_outside_s2_record"] += 1
                    continue

                year_mask    = times.year == year
                if not year_mask.any():
                    sample_skips["year_has_no_label_rows"] += 1
                    continue

                year_indices = np.where(year_mask)[0]
                # Only train on directly observed values (qc==0); gap-filled (qc==1) excluded.
                # qc_np can no longer be None — a station with no QC array was dropped above
                # (§35.24 audit item 4) — so the old `~isnan` fallback is gone. It was the
                # branch that let a gap-filled climatology through as an observation.
                assert qc_np is not None, (
                    f"{dir_name}: labels/qc is None after the QC admission check — the "
                    f"fail-closed guard in __init__ was bypassed."
                )
                valid_days = np.any(qc_np[:, year_indices] == QC_OBSERVED, axis=0)
                if valid_days.sum() < min_obs:
                    sample_skips["year_below_min_obs"] += 1
                    continue

                for day_idx in np.where(valid_days)[0]:
                    doy = times[year_indices[day_idx]].day_of_year

                    # ── ERA5 admission, day-granular (§35.24 audit item 1) ─────
                    # The guard used to be YEAR-granular: era5_start_year <= year <=
                    # era5_end_year. A station whose ERA5 record stops on 2021-03-14 still
                    # admitted every observed day of 2021, and load_era5_rolling right-aligns
                    # whatever it finds, so a 14-March row was placed in slot 364 and read by
                    # the model as "today's weather" for a target in September. Compare the
                    # actual dates instead.
                    target_int = _date_to_int(times[year_indices[day_idx]])
                    if not (era5_first_int <= target_int <= era5_last_int):
                        sample_skips["era5_target_day_outside_record"] += 1
                        era5_reject_by_station[dir_name] += 1
                        continue
                    ws_int, _ = _window_ints(year, int(doy))
                    if ws_int < era5_first_int:
                        # The 365-day window reaches back before the record starts. Not a
                        # correctness bug any more — era5_rel_pos declares the short window
                        # honestly and the empty slots are masked — so this is opt-in.
                        sample_skips["era5_window_not_fully_covered"] += 1
                        if era5_require_full_window:
                            era5_reject_by_station[dir_name] += 1
                            continue

                    self.samples.append({
                        "sat_dir"    : sat_dir,
                        "year"       : year,
                        "doy"        : doy,
                        "time_idx"   : year_indices[day_idx],
                        "station_key": dir_name,
                    })
                    n_year_ok += 1

            if n_year_ok > 0:
                admitted_dirs.add(sat_dir)
            else:
                skips["no_sample_survived_year_filters"] += 1
                rejected_dirs.add(sat_dir)

        # ── Audit report (§35.24) ───────────────────────────────────────────────
        # Every fail-closed decision above is silent by construction: it removes data and
        # changes nothing that a loss curve would show. So it gets printed.
        n_stations = len(set(s["station_key"] for s in self.samples))
        print(f"Dataset: {len(self.samples)} samples from {n_stations} stations")

        if skips:
            print("  stations dropped, by reason:")
            for reason, n in skips.most_common():
                print(f"    {n:6d}  {reason}")
        if sample_skips:
            # "year_*" rows count STATION-YEARS rejected wholesale; "era5_*" rows count
            # individual target days.
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

        print(f"  cloud mask: {n_no_cm_group} stations have no cm/masks group at all; "
              f"{n_s2_acq_no_cm}/{n_s2_acq_total} S2 acquisitions have no cloud-mask entry "
              f"and are now dropped (they used to read back cloud-free).")
        print(f"  S1 token mask: {dict(n_s1_no_tm)} stations per orbit have tokens but no "
              f"token_mask; {n_s1_acq_no_tm}/{n_s1_acq_total} S1 acquisitions dropped for it.")
        print(f"  statics: {n_dem_missing} stations with no DEM, {n_lulc_missing} with no "
              f"LULC (their patches are emitted with dem_valid/lulc_valid = False); "
              f"{n_dead_soil_ch} all-NaN soil channels zeroed in kept stations.")
        if n_l12_shm_partial:
            # Not an error — the zarr fallback covered it — but it means rank 0's shm
            # preload was incomplete for these keys, so check /dev/shm capacity and the
            # preloader's exit status before blaming the loader for slow epochs.
            print(f"  L12 cache: /dev/shm was partial for {dict(n_l12_shm_partial)} "
                  f"(station, key) pairs; those keys were filled from zarr instead.")

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

        # §35.24 audit item 14. Everything from s2_hist to dem_tok used to be bound only
        # inside `if zg is not None:` while the return dict read them unconditionally, so a
        # None group would have raised UnboundLocalError from a line that mentions none of
        # the loaders. It is unreachable — __init__ never appends a sample for a station
        # whose group is None — so make that an invariant instead of a dangling branch.
        if zg is None:
            raise RuntimeError(
                f"sample {idx} ({s['station_key']} {s['year']}-{s['doy']}) has no open zarr "
                f"group. __init__ only appends samples for stations it successfully cached, "
                f"so this means the group was cleared after construction."
            )

        # ── Zarr path — all GPFS data served from precomputed caches ──
        _l12       = self._l12_cache.get(sat_dir, {})
        # {key: bool} — resolved per modality inside each loader, because shm and the zarr
        # fallback can cover different keys of the same station (§35.24b item 1).
        _narrowed  = self._l12_narrowed.get(sat_dir, {})
        _dc        = self._zarr_date_cache.get(sat_dir, {})
        _cm_tm     = self._cm_token_mask_cache.get(sat_dir)
        _s1_tm     = self._s1_token_mask_cache.get(sat_dir, {})

        # s2_doys / s2_valid are unpacked but no longer emitted — they are consumed only
        # inside _finalise_history, which folds them into s2_hist_valid (§35.24 item 13).
        s2_hist, _s2_doys, _s2_valid, s2_rel_pos, s2_hist_valid = \
            load_s2_rolling_zarr(zg, year, doy,
                                 l12_np=_l12.get("s2"),
                                 date_cache=_dc,
                                 cm_token_mask=_cm_tm,
                                 training=self.training,
                                 token_sel=self._token_sel,
                                 patch_token_dropout=self._patch_token_dropout,
                                 l12_narrowed=_narrowed)

        s1_hist, _s1_doys, _s1_valid, s1_rel_pos, s1_hist_valid, s1_orbit = \
            load_s1_rolling_zarr(zg, year, doy,
                                 l12_asc_np=_l12.get("s1_asc"),
                                 l12_desc_np=_l12.get("s1_desc"),
                                 date_cache=_dc,
                                 s1_token_mask_cache=_s1_tm,
                                 training=self.training,
                                 token_sel=self._token_sel,
                                 patch_token_dropout=self._patch_token_dropout,
                                 l12_narrowed=_narrowed)

        # DEM/LULC enter patch k's sequence DIRECTLY, not as four nested-window means.
        # §27a.2 measured that pooling retains 1.5% (DEM) / 2.6% (LULC) of within-tile
        # variance -- that destruction is the defect this whole build exists to remove.
        _static  = self._static_cache.get(sat_dir, {})
        _sel     = self._token_sel
        dem_tok  = _static.get("dem",  torch.zeros(N_TOKENS, 768, dtype=torch.float16))[_sel]
        lulc_tok = _static.get("lulc", torch.zeros(N_TOKENS, 768, dtype=torch.float16))[_sel]
        # (14,14) -> (196,) -> the K selected patches. §35.24 audit item 5: these masks were
        # built into _static_cache at init and then never left __getitem__, so a DEM token
        # over a nodata void was fed to the model as terrain with nothing marking it.
        dem_valid  = _static.get("dem_token_mask",
                                 torch.zeros(14, 14, dtype=torch.bool)).reshape(N_TOKENS)[_sel]
        lulc_valid = _static.get("lulc_token_mask",
                                 torch.zeros(14, 14, dtype=torch.bool)).reshape(N_TOKENS)[_sel]

        # ── Soil patch (static, from cache — already z-scored at init) ──
        soil_patch = _static.get("soil", torch.zeros(21, 74, 74, dtype=torch.float32))

        # ── ERA5 — rolling 365-day window, numpy slice from cache ─────
        # load_era5_rolling returns numpy directly; single torch.from_numpy at end
        era5_np, era5_doys_np, era5_rel_np = load_era5_rolling(
            self._era5_cache.get(sat_dir), year, doy)
        if self._era5_log1p_prec:
            era5_np[:, PREC_IDX] = np.log1p(era5_np[:, PREC_IDX].clip(0))
        era5_np  = (era5_np - self._era5_means) / (self._era5_stds + 1e-8)
        era5     = torch.from_numpy(era5_np)
        era5_doys = torch.from_numpy(era5_doys_np)
        # True staleness per row, replacing the model's arange(365) (§35.24 audit item 1).
        era5_rel_pos = torch.from_numpy(era5_rel_np)

        # Mask 15% of ERA5 timesteps during training — forces temporal generalisation.
        # Only masks non-padded slots (era5_doys > 0); never applied at val/test time.
        #
        # VALUES ONLY (§35.24b item 2). This used to also set era5_doys[mask] = 0, which the
        # model reads as PADDING — so the masked rows left the sequence entirely. Two things
        # broke. First, training saw ~310 valid driver tokens and validation saw 365: a
        # train/eval shift in sequence LENGTH, not just in content, which is not what input
        # dropout is supposed to do. Second, the attention-entropy collapse detector
        # normalises by per-sample log(n_valid_keys) (§35.24 contract), so train and val
        # were being divided by different references and the diagnostic could not be
        # compared across them. Standard input dropout keeps the token and masks the value:
        # the row stays in the sequence, keeps its DOY and its staleness, and carries a zero
        # feature vector — which, post z-score, is the climatological mean for that variable.
        if self.training:
            valid_slots = era5_doys > 0
            mask = (torch.rand(era5.shape[0]) < 0.15) & valid_slots
            era5[mask] = 0.0

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
        # `qc_np is None` used to be treated as "assume observed"; __init__ now drops those
        # stations outright, so the only surviving path is an explicit QC_OBSERVED flag
        # (§35.24 audit item 4). Anything else — gap-filled, still-missing, or the 255
        # no-QC-source sentinel — leaves the depth as NaN and the loss masks it.
        sm_np, depths, _, qc_np = self._label_cache[s["sat_dir"]]
        label = torch.full((len(SM_DEPTHS),), float("nan"), dtype=torch.float32)
        for i, depth_str in enumerate(SM_DEPTHS):
            if depth_str in depths:
                d_idx = depths.index(depth_str)
                if qc_np[d_idx, s["time_idx"]] == QC_OBSERVED:
                    label[i] = float(sm_np[d_idx, s["time_idx"]])

        return {
            # ── Per-patch satellite history — the point of the whole architecture ──
            # (T, K, 768) fp16, K=1 in training (patch 105). Only these K patches were ever
            # read from the store; the other 195 never enter the process (§35.22).
            # s2_doys / s2_valid / s1_doys / s1_valid / token_valid were emitted here and
            # read by nothing: model.py uses s2_hist / s2_hist_valid / s2_rel_pos only, and
            # they are folded into hist_valid before they leave the loader. Dropped in
            # §35.24 audit item 13. (check_dataset.py's EXPECTED table still names them, but
            # that table also names s2_l12 and anchor_l3, which this dataset has not emitted
            # since §35.14 — it is stale against the patchwise arm either way. ablation.py
            # lists them in MODALITY_KEYS but tolerates absent keys as long as one matches,
            # and s2_hist / s2_hist_valid / s2_rel_pos still do.)
            "s2_hist"       : s2_hist,           # (MAX_S2, K, 768) fp16
            "s2_hist_valid" : s2_hist_valid,     # (MAX_S2, K) bool  — cloud mask AND doy>0
            "s2_rel_pos"    : s2_rel_pos,        # (MAX_S2,) long    — staleness, 364 = today

            "s1_hist"       : s1_hist,           # (MAX_S1, K, 768) fp16
            "s1_hist_valid" : s1_hist_valid,     # (MAX_S1, K) bool
            "s1_rel_pos"    : s1_rel_pos,        # (MAX_S1,) long
            "s1_orbit"      : s1_orbit,          # (MAX_S1,) long    — 0 = ASC, 1 = DESC

            # ── Per-patch statics ─────────────────────────────────────────
            "dem_tok"       : dem_tok,           # (K, 768) fp16
            "lulc_tok"      : lulc_tok,          # (K, 768) fp16
            "dem_valid"     : dem_valid,         # (K,) bool — False = fabricated / nodata
            "lulc_valid"    : lulc_valid,        # (K,) bool
            # .copy(): self._token_sel is a single array shared by every worker and every
            # sample, and from_numpy would hand out a VIEW of it. Any in-place write on the
            # returned tensor — a collate that reuses the buffer, an ablation that permutes
            # it — would rewrite the dataset's own patch selection (§35.24 item 13).
            "token_idx"     : torch.from_numpy(self._token_sel.copy()),   # (K,) long

            # ── Tile-level drivers: identical for every patch, hence cacheable ──
            "soil_patch"    : soil_patch,        # (21, 74, 74) fp32 — NaN-free, z-scored
            "era5"          : era5,              # (365, 19) fp32 — z-scored
            "era5_doys"     : era5_doys,         # (365,) long
            "era5_rel_pos"  : era5_rel_pos,      # (365,) long — TRUE staleness, 364 = today
            "sif"           : sif_vals,          # (MAX_SIF, 1) fp32 — z-scored
            "sif_doys"      : sif_doys,          # (MAX_SIF,) long
            "sif_rel_pos"   : sif_rel_pos,       # (MAX_SIF,) long
            "sif_valid"     : sif_valid,         # (MAX_SIF,) bool
            "twsa"          : twsa_vals,         # (MAX_TWSA, 1) fp32 — z-scored
            "twsa_doys"     : twsa_doys,         # (MAX_TWSA,) long
            "twsa_rel_pos"  : twsa_rel_pos,      # (MAX_TWSA,) long
            "twsa_valid"    : twsa_valid,        # (MAX_TWSA,) bool

            # ── Labels and identity ───────────────────────────────────────
            "label"         : label,             # (3,) — NaN where the depth has no obs
            "station_key"   : s["station_key"],
            "year"          : s["year"],
            "doy"           : s["doy"],
        }
