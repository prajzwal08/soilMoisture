"""
Regression tests for the patchwise loader (§35.8, §35.22, §35.24).

WHAT THIS FILE IS FOR
---------------------
Every check here corresponds to a failure that does NOT crash. `dataset.py`'s §35.24 audit
turned four fail-OPEN paths into fail-CLOSED ones:

    missing cloud mask       ->  all 60 S2 acquisitions were marked cloud-free
    missing S1 token mask    ->  border/shadow/layover patches were marked valid
    missing QC variable      ->  gap-filled climatology was trained as observed truth
    missing DEM nodata mask  ->  an all-zero token was read as a real elevation embedding

Each of those is a silent degradation: it removes or fabricates data and changes nothing a
loss curve would show. So each one is asserted here, in the "absent input" direction — the
direction a future edit would regress.

It also pins the two things the model half of the suite cannot see on its own:

    * the polarity of `*_hist_valid` / `dem_valid` / `sif_valid` at the producer end
      (True = VALID here; model.py negates it into nn.MultiheadAttention's True = IGNORE)
    * that label column i is SM_DEPTHS[i] regardless of the order `labels/depths` happens
      to be stored in — a permutation there would report the 30-100 cm model's error under
      the 0-10 cm label, silently

Fixtures are small synthetic zarr stores in a tmp dir. The real store is only touched by one
test, which skips cleanly when it is absent.

    sbatch slurm/run_tests.sh          # the only supported way to run this
"""
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from torch.utils.data import default_collate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds_mod                                              # noqa: E402
from dataset import (                                                 # noqa: E402
    CM_BAD_CLASSES,
    ERA5_VARS,
    MAX_DEAD_SOIL_CHANNELS,
    MAX_S1,
    MAX_S2,
    MAX_SIF,
    MAX_TWSA,
    N_TOKENS,
    ORBIT_ASC,
    ORBIT_DESC,
    QC_NO_SOURCE,
    QC_OBSERVED,
    SM_DEPTHS,
    STATION_TOKEN,
    SoilMoistureDataset,
    _empty_history,
    _finalise_history,
    _load_driver_stats,
    _load_zarr_labels,
    _read_patch_tokens,
    _token_slice,
    fill_soil_nans_with_validity,
    load_era5_rolling,
    load_s1_rolling_zarr,
    load_s2_rolling_zarr,
)

# ── fixture geometry ─────────────────────────────────────────────────────────

STATION = "STA"
NETWORK = "TESTNET"
DIR_NAME = f"ISMN_{NETWORK}_{STATION}"
CATEGORY = "sm_only"

REC_START = datetime(2020, 1, 1)
REC_END = datetime(2021, 12, 31)
TARGET_YEAR, TARGET_DOY = 2021, 365                  # 2021-12-31

S2_DATES = ["20210110", "20210210", "20210310", "20210410", "20210510", "20210610",
            "20210710", "20210810", "20210910", "20211010", "20211110", "20211210"]
N_S2_CLEAR = 6                                       # first 6 clear, last 6 thick cloud

S1_ASC_DATES = ["20210601", "20210613"]
S1_DESC_DATES = ["20210607", "20210619"]
# merged and date-sorted: 0601 asc, 0607 desc, 0613 asc, 0619 desc
S1_EXPECT_ORBIT = [ORBIT_ASC, ORBIT_DESC, ORBIT_ASC, ORBIT_DESC]

# `labels/depths` is written DELIBERATELY out of SM_DEPTHS order, with a distinct value per
# depth, so any permutation between the store, the loader and the model shows up immediately.
ZARR_DEPTH_ORDER = ["30-100", "0-10", "10-30"]
DEPTH_VALUE = {"0-10": 0.11, "10-30": 0.21, "30-100": 0.31}

SOIL_MEAN, SOIL_STD = 5.0, 2.0                       # non-trivial, so z-scoring is visible


def _date_ints(dates):
    return np.array([int(d) for d in dates], dtype=np.int32)


def _doys(dates):
    return np.array([datetime.strptime(d, "%Y%m%d").timetuple().tm_yday for d in dates],
                    dtype=np.int32)


def _daily_dates():
    n = (REC_END - REC_START).days + 1
    return [(REC_START + timedelta(days=i)).strftime("%Y%m%d") for i in range(n)]


def build_store(root: zarr.Group, *,
                with_cm=True, with_s1_token_mask=True,
                with_dem=True, with_lulc=True,
                with_dem_mask=True, with_lulc_mask=True,
                qc_mode="observed",           # observed | sentinel | absent | short
                dead_soil_channels=0,
                broken_label_dates=False,
                n_observed_days=40,
                rng=None):
    """Write one synthetic station store. Deliberately minimal but structurally complete."""
    rng = rng or np.random.default_rng(0)
    dates = _daily_dates()
    n = len(dates)

    # ── ERA5 (daily, gapless) ──
    era5 = rng.standard_normal((n, len(ERA5_VARS))).astype(np.float32)
    root.array("era5/values", era5, overwrite=True)
    root.array("era5/date_ints", _date_ints(dates), overwrite=True)
    root.array("era5/doys", _doys(dates), overwrite=True)

    # ── labels: stored in a NON-SM_DEPTHS order on purpose ──
    sm = np.zeros((3, n), dtype=np.float32)
    for i, d in enumerate(ZARR_DEPTH_ORDER):
        sm[i, :] = DEPTH_VALUE[d]
    root.array("labels/sm", sm, overwrite=True)
    root.array("labels/depths", np.array(ZARR_DEPTH_ORDER, dtype="U20"), overwrite=True)
    lab_dates = dates[:-1] if broken_label_dates else dates
    root.array("labels/dates", np.array(lab_dates, dtype="U8"), overwrite=True)

    if qc_mode != "absent":
        qc = np.ones((3, n), dtype=np.uint8)                     # 1 = gap-filled
        if qc_mode == "sentinel":
            qc[:] = QC_NO_SOURCE
        else:
            # observed on n_observed_days spread through 2021
            y2021 = [i for i, d in enumerate(dates) if d.startswith("2021")]
            step = max(1, len(y2021) // n_observed_days)
            qc[:, y2021[::step][:n_observed_days]] = QC_OBSERVED
        if qc_mode == "short":
            # trim_pre2016.py used to trim sm/dates WITHOUT trimming qc, so qc ends up longer
            # than sm and the offset between them is unrecoverable.
            qc = np.concatenate(
                [np.full((3, 5), QC_OBSERVED, dtype=np.uint8), qc], axis=1)
        root.array("labels/qc", qc, overwrite=True)

    # ── S2 tokens + dates ──
    l12 = rng.standard_normal((len(S2_DATES), N_TOKENS, 768)).astype(np.float16)
    root.array("s2/l12", l12, overwrite=True)
    root.array("s2/dates", np.array(S2_DATES, dtype="U8"), overwrite=True)

    if with_cm:
        cm = np.zeros((len(S2_DATES), 224, 224), dtype=np.uint8)
        cm[N_S2_CLEAR:] = CM_BAD_CLASSES[1]                      # thick cloud everywhere
        root.array("cm/masks", cm, overwrite=True)
        root.array("cm/dates", np.array(S2_DATES, dtype="U8"), overwrite=True)

    # ── S1, two orbits ──
    for key, dts in (("s1_asc", S1_ASC_DATES), ("s1_desc", S1_DESC_DATES)):
        root.array(f"{key}/l12",
                   rng.standard_normal((len(dts), N_TOKENS, 768)).astype(np.float16),
                   overwrite=True)
        root.array(f"{key}/dates", np.array(dts, dtype="U8"), overwrite=True)
        if with_s1_token_mask:
            root.array(f"{key}/token_mask",
                       np.ones((len(dts), 14, 14), dtype=bool), overwrite=True)

    # ── statics ──
    if with_dem:
        root.array("dem", rng.standard_normal((N_TOKENS, 768)).astype(np.float16),
                   overwrite=True)
        if with_dem_mask:
            root.array("dem_token_mask", np.ones((14, 14), dtype=bool), overwrite=True)
    if with_lulc:
        root.array("lulc", rng.standard_normal((N_TOKENS, 768)).astype(np.float16),
                   overwrite=True)
        if with_lulc_mask:
            root.array("lulc_token_mask", np.ones((14, 14), dtype=bool), overwrite=True)

    soil = rng.standard_normal((21, 74, 74)).astype(np.float32) + SOIL_MEAN
    for c in range(dead_soil_channels):
        soil[c] = np.nan
    root.array("soil", soil, overwrite=True)
    return soil


def make_env(tmp_path, *, driver_stats=True, driver_stats_complete=True, **store_kw):
    """Build ZARR_ROOT/<cat>/<station>, station_splits.csv and the two stats files."""
    zroot = tmp_path / "zarr"
    sdir = zroot / CATEGORY / DIR_NAME
    sdir.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(sdir), mode="w")
    soil = build_store(root, **store_kw)
    zarr.consolidate_metadata(root.store)
    (sdir / ".complete").touch()

    csv = tmp_path / "station_splits.csv"
    pd.DataFrame([{
        "source_network": "ISMN", "network": NETWORK,
        "station_id": STATION, "station_name": STATION,
        "has_soil_moisture": True, "has_flux": False,
        "split": "train", "soil_patch_ok": True,
    }]).to_csv(csv, index=False)

    era5_stats = tmp_path / "era5_stats.json"
    era5_stats.write_text(json.dumps({
        "vars": ERA5_VARS,
        "means": [0.0] * len(ERA5_VARS),
        "stds": [1.0] * len(ERA5_VARS),
        "log1p_precip": False,
    }))

    dstats = tmp_path / "driver_stats.json"
    if driver_stats:
        blob = {
            "sif": {"mean": 0.0, "std": 1.0},
            "twsa": {"mean": 0.0, "std": 1.0},
            "soil": {"mean": [SOIL_MEAN] * 21, "std": [SOIL_STD] * 21},
            "label_mean": {d: DEPTH_VALUE[d] for d in SM_DEPTHS},
            "years": [2020, 2021], "split": "train", "n_stations": 1,
        }
        if not driver_stats_complete:
            blob.pop("soil")
        dstats.write_text(json.dumps(blob))

    return {"zroot": zroot, "sdir": sdir, "csv": str(csv),
            "era5_stats": str(era5_stats), "driver_stats": str(dstats), "soil": soil}


def make_dataset(env, monkeypatch, **kw):
    monkeypatch.setattr(ds_mod, "ZARR_ROOT", env["zroot"])
    kw.setdefault("years", [TARGET_YEAR])
    kw.setdefault("min_obs", 5)
    kw.setdefault("training", False)
    return SoilMoistureDataset(env["csv"], env["era5_stats"],
                               driver_stats_path=env["driver_stats"], **kw)


def open_store(env):
    return zarr.open_group(str(env["sdir"]), mode="r")


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS AND THE NARROW READ
# ═════════════════════════════════════════════════════════════════════════════

def test_token_geometry_constants():
    assert N_TOKENS == 196 and STATION_TOKEN == 105
    assert STATION_TOKEN == (112 // 16) * 14 + (112 // 16)


def test_sm_depths_order_is_the_contract():
    assert SM_DEPTHS == ["0-10", "10-30", "30-100"]


def test_cm_bad_classes_are_the_sensei_product_not_scl():
    """Under Sentinel-2 SCL, 4 = vegetation and 5 = not-vegetated, i.e. the two classes you
    most want to KEEP. This list is the SEnSeIv2-SegFormerB2 seven-class product."""
    assert CM_BAD_CLASSES == [3, 4, 5, 255]


@pytest.mark.parametrize("sel", [np.array([STATION_TOKEN]),
                                 np.arange(N_TOKENS),
                                 np.array([0, 105, 195])])
def test_narrow_read_matches_the_wide_read(sel):
    """If these diverge, every number the model produces is computed on the wrong patch."""
    full = np.random.default_rng(0).standard_normal((5, N_TOKENS, 768)).astype(np.float16)
    got = _read_patch_tokens(full, 3, _token_slice(sel), sel)
    assert np.array_equal(got, full[3][sel])


def test_token_slice_contiguity():
    assert _token_slice(np.array([STATION_TOKEN])) == slice(105, 106)
    assert _token_slice(np.arange(N_TOKENS)) == slice(0, 196)
    assert _token_slice(np.array([0, 105, 195])) is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. MASK POLARITY AT THE PRODUCER END
# ═════════════════════════════════════════════════════════════════════════════
#
# `hist_valid` is True = VALID. model.py negates it. Get it backwards here and the model
# reads exactly the cloudy and padded slots.

def test_finalise_history_is_exactly_token_mask_AND_doy_gt_zero():
    rng = np.random.default_rng(7)
    T, K = 12, 3
    sel = np.array([0, STATION_TOKEN, 195], dtype=np.int64)
    tm = torch.from_numpy(rng.random((T, 14, 14)) > 0.4)
    doys = torch.zeros(T, dtype=torch.long)
    doys[:7] = torch.tensor([10, 40, 70, 100, 130, 160, 190])

    _, _, valid_acq, _, hv = _finalise_history(
        torch.randn(T, K, 768, dtype=torch.float16), tm, doys,
        torch.zeros(T, dtype=torch.long), training=False, token_sel=sel, dropout_p=0.0)

    expect = tm.reshape(T, N_TOKENS)[:, sel] & (doys > 0)[:, None]
    assert tuple(hv.shape) == (T, K), tuple(hv.shape)
    assert hv.dtype == torch.bool
    assert torch.equal(hv, expect), (
        "hist_valid is not `token_mask[selected patches] AND (doy > 0)`. Either the polarity "
        "flipped or the (T,14,14) mask was indexed on the wrong axis."
    )
    assert torch.equal(valid_acq, doys > 0)
    # the (T,14,14) -> (T,196) reshape: indexing the mask directly with token indices
    # silently selects ROWS and yields (T,K,14)
    assert hv.ndim == 2


def test_finalise_history_padded_slots_are_invalid_even_with_an_all_true_mask():
    T, K = 8, 1
    sel = np.array([STATION_TOKEN], dtype=np.int64)
    doys = torch.zeros(T, dtype=torch.long)
    doys[:3] = 100
    _, _, _, _, hv = _finalise_history(
        torch.randn(T, K, 768, dtype=torch.float16),
        torch.ones(T, 14, 14, dtype=torch.bool), doys,
        torch.zeros(T, dtype=torch.long), training=False, token_sel=sel, dropout_p=0.0)
    assert hv[:3].all() and not hv[3:].any()


def test_patch_token_dropout_is_off_unless_asked_and_only_in_training():
    T, K = 40, 1
    sel = np.array([STATION_TOKEN], dtype=np.int64)
    tm = torch.ones(T, 14, 14, dtype=torch.bool)
    doys = torch.full((T,), 100, dtype=torch.long)
    args = (torch.zeros(T, K, 768, dtype=torch.float16), tm, doys,
            torch.zeros(T, dtype=torch.long))
    # training=True but p=0 (the production default): nothing dropped
    assert _finalise_history(*args, training=True, token_sel=sel, dropout_p=0.0)[4].all()
    # p>0 but not training: nothing dropped
    assert _finalise_history(*args, training=False, token_sel=sel, dropout_p=0.9)[4].all()
    # both: something dropped
    torch.manual_seed(0)
    assert not _finalise_history(*args, training=True, token_sel=sel, dropout_p=0.9)[4].all()


def test_empty_history_matches_the_normal_path_and_collates():
    sel = np.array([0, STATION_TOKEN, 195], dtype=np.int64)
    T, K = 8, len(sel)
    ef, ed, ev, er, ehv = _empty_history(MAX_S2, sel)
    assert tuple(ef.shape) == (MAX_S2, K, 768)
    assert tuple(ehv.shape) == (MAX_S2, K)
    assert not ehv.any() and not ev.any()
    assert ed.dtype == torch.long and er.dtype == torch.long

    real = _finalise_history(torch.randn(T, K, 768, dtype=torch.float16),
                             torch.ones(T, 14, 14, dtype=torch.bool),
                             torch.full((T,), 100, dtype=torch.long),
                             torch.zeros(T, dtype=torch.long),
                             training=False, token_sel=sel, dropout_p=0.0)
    empty = _empty_history(T, sel)
    batched = default_collate([{"h": real[0], "v": real[4]},
                               {"h": empty[0], "v": empty[4]}])
    assert tuple(batched["h"].shape) == (2, T, K, 768)


# ═════════════════════════════════════════════════════════════════════════════
# 3. ERA5 STALENESS COMES FROM REAL DATES, NOT THE SLOT INDEX
# ═════════════════════════════════════════════════════════════════════════════
#
# load_era5_rolling COMPACTS and right-aligns. The model used to add rel_pos_emb(arange(365)),
# i.e. it read the slot index as the staleness. Those agree only when the record is gapless
# and ends on the target day.

def _era5_cache(dates):
    di = _date_ints(dates)
    return (np.zeros((len(dates), len(ERA5_VARS)), dtype=np.float32), di, _doys(dates))


def _ordinal(d):
    return datetime.strptime(d, "%Y%m%d").toordinal()


def test_era5_rel_pos_slot_364_is_the_target_day():
    dates = _daily_dates()
    _, doys, rel = load_era5_rolling(_era5_cache(dates), TARGET_YEAR, TARGET_DOY)
    assert rel.shape == (365,) and rel.dtype == np.int64
    assert rel[364] == 364, "slot 364 must be the target day"
    assert doys[364] == TARGET_DOY
    # gapless and ending on the target day is the ONE case where slot index == staleness
    assert np.array_equal(rel, np.arange(365))


def test_era5_rel_pos_survives_an_interior_gap():
    """The defect: a 30-day hole shifts every post-gap row 30 slots later than it belongs, so
    the whole post-gap half of the year is systematically labelled too recent."""
    gap_start, gap_len = "20210601", 30
    g0 = _ordinal(gap_start)
    dates = [d for d in _daily_dates()
             if not (g0 <= _ordinal(d) < g0 + gap_len)]
    _, doys, rel = load_era5_rolling(_era5_cache(dates), TARGET_YEAR, TARGET_DOY)

    target_ord = _ordinal("20211231")
    in_window = [d for d in dates if _ordinal(d) > target_ord - 365]
    n = len(in_window)
    assert n == 365 - gap_len

    # every filled slot carries the staleness implied by its REAL date
    want = np.array([364 - (target_ord - _ordinal(d)) for d in in_window], dtype=np.int64)
    assert np.array_equal(rel[-n:], want)
    # padded head slots stay at 0 and are masked by doy == 0
    assert (rel[:365 - n] == 0).all() and (doys[:365 - n] == 0).all()

    # the slot immediately before the gap: its rel_pos must be its slot index MINUS the gap
    pre_gap_pos = max(i for i, d in enumerate(in_window) if _ordinal(d) < g0)
    slot = (365 - n) + pre_gap_pos
    assert rel[slot] == slot - gap_len, (
        f"pre-gap slot {slot} carries staleness {rel[slot]}; the slot index would say {slot} "
        f"and the true staleness is {slot - gap_len} — arange(365) is back"
    )
    assert rel[slot] != slot


def test_era5_rel_pos_declares_a_record_that_stops_early():
    """record ends 2021-03-14, target 2021-09-30: the last real row must NOT land in slot 364
    labelled 'today'. A 200-day-old temperature presented as this morning's is exactly the
    signal the drivers exist to carry."""
    dates = [d for d in _daily_dates() if _ordinal(d) <= _ordinal("20210314")]
    doy = datetime(2021, 9, 30).timetuple().tm_yday
    _, doys_out, rel = load_era5_rolling(_era5_cache(dates), 2021, doy)
    last = int(np.max(np.nonzero(doys_out)))
    true_stale = 364 - (_ordinal("20210930") - _ordinal("20210314"))
    assert rel[last] == true_stale, (rel[last], true_stale)
    assert rel[last] < 364


def test_era5_rel_pos_is_zero_where_there_is_no_record():
    e, d, rel = load_era5_rolling(None, TARGET_YEAR, TARGET_DOY)
    assert e.shape == (365, len(ERA5_VARS)) and not e.any()
    assert not d.any() and not rel.any()


# ═════════════════════════════════════════════════════════════════════════════
# 4. FAIL-CLOSED LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def test_s2_cloud_mask_polarity_clear_valid_cloudy_invalid(tmp_path):
    env = make_env(tmp_path)
    zg = open_store(env)
    sel = np.array([STATION_TOKEN], dtype=np.int64)
    _, doys, valid_acq, rel, hv = load_s2_rolling_zarr(
        zg, TARGET_YEAR, TARGET_DOY, token_sel=sel)

    assert int(valid_acq.sum()) == len(S2_DATES), "all 12 acquisitions fall in the window"
    got = hv[:len(S2_DATES), 0]
    want = torch.tensor([i < N_S2_CLEAR for i in range(len(S2_DATES))])
    assert torch.equal(got, want), (
        f"cloud-mask polarity is inverted: clear acquisitions {got[:N_S2_CLEAR].tolist()} / "
        f"thick-cloud acquisitions {got[N_S2_CLEAR:].tolist()}"
    )
    # staleness from the real dates
    tord = _ordinal("20211231")
    assert rel[0].item() == 364 - (tord - _ordinal(S2_DATES[0]))
    assert not hv[len(S2_DATES):].any(), "padded tail must be invalid"


def test_missing_cloud_mask_group_marks_every_s2_acquisition_invalid(tmp_path):
    """FAIL CLOSED. token_mask used to be torch.ones and is written only inside
    `if date_str in cm_d2i`, so a station with no cm/masks reported all 60 acquisitions
    cloud-free — thick cumulus scored as bare soil, with nothing anywhere raising."""
    env = make_env(tmp_path, with_cm=False)
    zg = open_store(env)
    _, doys, valid_acq, _, hv = load_s2_rolling_zarr(
        zg, TARGET_YEAR, TARGET_DOY, token_sel=np.array([STATION_TOKEN]))
    assert int(valid_acq.sum()) == len(S2_DATES), "the acquisitions themselves are present"
    assert not hv.any(), (
        "S2 with no cloud-mask evidence read back VALID — the quality mask must fail closed"
    )


def test_s1_orbit_is_tagged_per_acquisition_not_per_slot(tmp_path):
    """`entries` is sorted by DATE across both orbits, so slot i is ASC or DESC depending on
    which satellite pass happened that day. RTC backscatter differs between the two by an
    amount comparable to the moisture signal."""
    env = make_env(tmp_path)
    zg = open_store(env)
    _, doys, valid_acq, rel, hv, orbit = load_s1_rolling_zarr(
        zg, TARGET_YEAR, TARGET_DOY, token_sel=np.array([STATION_TOKEN]))

    n = len(S1_EXPECT_ORBIT)
    assert int(valid_acq.sum()) == n
    assert orbit[:n].tolist() == S1_EXPECT_ORBIT, (
        f"orbit tags {orbit[:n].tolist()} != expected {S1_EXPECT_ORBIT} for the date-sorted "
        f"merge of ASC {S1_ASC_DATES} and DESC {S1_DESC_DATES}"
    )
    assert orbit.dtype == torch.int64 and tuple(orbit.shape) == (MAX_S1,)
    assert set(orbit[:n].tolist()) == {ORBIT_ASC, ORBIT_DESC}
    assert hv[:n, 0].all() and not hv[n:].any()


def test_missing_s1_token_mask_marks_that_orbit_invalid(tmp_path):
    """FAIL CLOSED. With a ones-init the whole S1 history read back valid, layover and
    shadow and all."""
    env = make_env(tmp_path, with_s1_token_mask=False)
    zg = open_store(env)
    _, _, valid_acq, _, hv, orbit = load_s1_rolling_zarr(
        zg, TARGET_YEAR, TARGET_DOY, token_sel=np.array([STATION_TOKEN]))
    assert int(valid_acq.sum()) == len(S1_EXPECT_ORBIT)
    assert not hv.any(), "S1 with no stored token_mask read back VALID"


def test_all_nan_soil_channel_is_flagged_not_silently_nan():
    """distance_transform_edt on an ALL-True mask returns the identity index field: `out[c]`
    comes back exactly as NaN as it went in, the function reports success, and because the
    drivers are a shared cross-attention memory the K/V cache for the whole SAMPLE goes NaN —
    every parameter after the first backward. The traceback points at the loss, not here."""
    patch = np.random.default_rng(0).standard_normal((21, 74, 74)).astype(np.float32)
    patch[3] = np.nan                                    # wholly dead
    patch[7, :10, :10] = np.nan                          # partially dead, fillable
    out, valid = fill_soil_nans_with_validity(patch)
    assert np.isfinite(out).all(), "a dead soil channel leaked NaN into the driver memory"
    assert not valid[3]
    assert valid[7] and valid[0]
    assert (out[3] == 0.0).all()
    assert int((~valid).sum()) == 1


# ═════════════════════════════════════════════════════════════════════════════
# 5. FAIL-CLOSED STATS AND LABEL ALIGNMENT
# ═════════════════════════════════════════════════════════════════════════════

def test_missing_driver_stats_raises_and_names_the_producer(tmp_path):
    env = make_env(tmp_path, driver_stats=False)
    with pytest.raises(FileNotFoundError, match="compute_driver_stats"):
        _load_driver_stats(env["driver_stats"])


def test_incomplete_driver_stats_raises(tmp_path):
    env = make_env(tmp_path, driver_stats_complete=False)
    with pytest.raises(KeyError):
        _load_driver_stats(env["driver_stats"])


def test_dataset_refuses_to_build_without_driver_stats(tmp_path, monkeypatch):
    """Fail closed — never silently fall back to identity normalisation. A 1400-magnitude
    bulk-density channel next to a 0.6-magnitude ERA5 z-score does not 'just get learned
    around'."""
    env = make_env(tmp_path, driver_stats=False)
    with pytest.raises(FileNotFoundError):
        make_dataset(env, monkeypatch)


def test_label_qc_length_mismatch_is_refused_not_guessed(tmp_path):
    """The old code GUESSED a front-trim and took the trailing n columns. If the truncation
    was at the back instead, every QC flag is offset and the loader trains on gap-filled days
    believing they are observed."""
    env = make_env(tmp_path, qc_mode="short")
    zg = open_store(env)
    with pytest.raises(ValueError, match="trim_pre2016|different passes"):
        _load_zarr_labels(zg, strict=True)
    # the permissive default still realigns, for the analysis scripts that documented it
    sm, depths, times, qc = _load_zarr_labels(zg, strict=False)
    assert qc.shape[1] == sm.shape[1]


def test_label_dates_sm_mismatch_always_raises(tmp_path):
    env = make_env(tmp_path, broken_label_dates=True)
    zg = open_store(env)
    for strict in (False, True):
        with pytest.raises(ValueError, match="internally inconsistent"):
            _load_zarr_labels(zg, strict=strict)


# ═════════════════════════════════════════════════════════════════════════════
# 6. THE DATASET, END TO END
# ═════════════════════════════════════════════════════════════════════════════

def test_happy_path_emits_the_contract(tmp_path, monkeypatch):
    ds = make_dataset(make_env(tmp_path), monkeypatch)
    assert len(ds) > 0, "the fixture itself is broken — every other dataset test depends on it"
    s = ds[0]

    expect = {
        "s2_hist": ((MAX_S2, 1, 768), torch.float16),
        "s2_hist_valid": ((MAX_S2, 1), torch.bool),
        "s2_rel_pos": ((MAX_S2,), torch.int64),
        "s1_hist": ((MAX_S1, 1, 768), torch.float16),
        "s1_hist_valid": ((MAX_S1, 1), torch.bool),
        "s1_rel_pos": ((MAX_S1,), torch.int64),
        "s1_orbit": ((MAX_S1,), torch.int64),
        "dem_tok": ((1, 768), torch.float16),
        "lulc_tok": ((1, 768), torch.float16),
        "dem_valid": ((1,), torch.bool),
        "lulc_valid": ((1,), torch.bool),
        "token_idx": ((1,), torch.int64),
        "soil_patch": ((21, 74, 74), torch.float32),
        "era5": ((365, 19), torch.float32),
        "era5_doys": ((365,), torch.int64),
        "era5_rel_pos": ((365,), torch.int64),
        "sif": ((MAX_SIF, 1), torch.float32),
        "sif_valid": ((MAX_SIF,), torch.bool),
        "twsa": ((MAX_TWSA, 1), torch.float32),
        "twsa_valid": ((MAX_TWSA,), torch.bool),
        "label": ((len(SM_DEPTHS),), torch.float32),
    }
    for k, (shape, dtype) in expect.items():
        assert k in s, f"missing key {k}"
        assert tuple(s[k].shape) == shape, f"{k}: {tuple(s[k].shape)} != {shape}"
        assert s[k].dtype == dtype, f"{k}: {s[k].dtype} != {dtype}"

    # §35.24 item 13 removed these; anything still reading them is on the old contract
    for k in ("s2_doys", "s2_valid", "s1_doys", "s1_valid", "token_valid"):
        assert k not in s, f"{k} was dropped in §35.24 item 13 but is still emitted"

    assert s["token_idx"].tolist() == [STATION_TOKEN]
    assert bool(s["dem_valid"].all()) and bool(s["lulc_valid"].all())
    assert torch.isfinite(s["soil_patch"]).all()
    assert torch.isfinite(s["era5"]).all()
    assert int(s["era5_doys"].max()) > 0


def test_label_column_i_is_sm_depths_i_whatever_the_store_order(tmp_path, monkeypatch):
    """`labels/depths` is written as ["30-100", "0-10", "10-30"] on purpose. A permutation
    between the store, the loader and the model reports the 30-100 cm model's error under the
    0-10 cm label, and nothing anywhere raises."""
    ds = make_dataset(make_env(tmp_path), monkeypatch)
    label = ds[0]["label"]
    for i, depth in enumerate(SM_DEPTHS):
        assert float(label[i]) == pytest.approx(DEPTH_VALUE[depth], abs=1e-5), (
            f"label column {i} should be {depth} = {DEPTH_VALUE[depth]}, got {float(label[i])}. "
            f"The store holds depths in the order {ZARR_DEPTH_ORDER}."
        )


def test_token_idx_is_a_copy_not_a_view_of_the_datasets_own_selection(tmp_path, monkeypatch):
    """from_numpy would hand out a VIEW of self._token_sel, which every worker and every
    sample shares — an in-place write would rewrite the dataset's patch selection."""
    ds = make_dataset(make_env(tmp_path), monkeypatch)
    t = ds[0]["token_idx"]
    t[0] = 7
    assert ds._token_sel.tolist() == [STATION_TOKEN]
    assert ds[1]["token_idx"].tolist() == [STATION_TOKEN]


def test_soil_patch_is_z_scored_with_driver_stats(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    ds = make_dataset(env, monkeypatch)
    got = ds[0]["soil_patch"].numpy()
    want = (env["soil"] - SOIL_MEAN) / (SOIL_STD + 1e-8)
    assert np.allclose(got, want, atol=1e-4), (
        "soil_patch is not z-scored with csvs/driver_stats.json — it went into SoilEncoder raw"
    )


def test_all_sentinel_qc_drops_the_station(tmp_path, monkeypatch):
    """create_token_zarr.py used to default labels/qc to zeros, i.e. 'every day directly
    observed'. The preprocessing pipeline gap-fills with a month-day climatology, so those
    zeros trained the model on climatology wearing a ground-truth badge."""
    ds = make_dataset(make_env(tmp_path, qc_mode="sentinel"), monkeypatch)
    assert len(ds) == 0, (
        f"a station whose QC is entirely the {QC_NO_SOURCE} no-source sentinel produced "
        f"{len(ds)} samples"
    )


def test_absent_qc_array_drops_the_station(tmp_path, monkeypatch):
    ds = make_dataset(make_env(tmp_path, qc_mode="absent"), monkeypatch)
    assert len(ds) == 0


def test_qc_misalignment_drops_the_station(tmp_path, monkeypatch):
    ds = make_dataset(make_env(tmp_path, qc_mode="short"), monkeypatch)
    assert len(ds) == 0


def test_gap_filled_days_never_become_samples(tmp_path, monkeypatch):
    """qc == 1 is climatological gap-fill. It must not be a training target."""
    n_obs = 11
    ds = make_dataset(make_env(tmp_path, n_observed_days=n_obs), monkeypatch, min_obs=5)
    assert len(ds) == n_obs, f"{len(ds)} samples from {n_obs} observed days"
    for i in range(len(ds)):
        assert torch.isfinite(ds[i]["label"]).all()


@pytest.mark.parametrize("kw,key", [
    (dict(with_dem=False), "dem_valid"),
    (dict(with_lulc=False), "lulc_valid"),
    (dict(with_dem_mask=False), "dem_valid"),
    (dict(with_lulc_mask=False), "lulc_valid"),
])
def test_missing_static_or_its_mask_is_marked_invalid(tmp_path, monkeypatch, kw, key):
    """A station with no DEM was handed torch.zeros(196,768) — a fabricated token that is not
    flat ground, it is whatever the model decides an all-zero L12 vector means. Terrain is the
    §34.3 mechanism the architecture rests on."""
    ds = make_dataset(make_env(tmp_path, **kw), monkeypatch)
    assert len(ds) > 0, "a missing static must not drop the station, only flag its token"
    s = ds[0]
    assert not s[key].any(), f"{key} is True although the fixture omitted {list(kw)[0]}"
    tok = "dem_tok" if key.startswith("dem") else "lulc_tok"
    if list(kw)[0] in ("with_dem", "with_lulc"):
        assert not s[tok].any(), "the fabricated token should be the all-zero placeholder"


@pytest.mark.parametrize("dead,kept", [(0, True), (1, True),
                                       (MAX_DEAD_SOIL_CHANNELS, True),
                                       (MAX_DEAD_SOIL_CHANNELS + 1, False)])
def test_dead_soil_channels_are_tolerated_then_the_station_is_dropped(
        tmp_path, monkeypatch, dead, kept):
    ds = make_dataset(make_env(tmp_path, dead_soil_channels=dead), monkeypatch)
    if kept:
        assert len(ds) > 0, f"{dead} dead channels should be tolerated"
        sp = ds[0]["soil_patch"]
        assert torch.isfinite(sp).all()
        for c in range(dead):
            assert (sp[c] == 0.0).all(), (
                "a dead channel must be re-zeroed AFTER the z-score, so it sits at the "
                "training mean rather than at -mean/std"
            )
    else:
        assert len(ds) == 0, (
            f"{dead} > MAX_DEAD_SOIL_CHANNELS={MAX_DEAD_SOIL_CHANNELS} channels of pure "
            f"invention should drop the station"
        )


def test_absent_sif_and_twsa_are_marked_invalid_not_zero_valued(tmp_path, monkeypatch):
    """The fixture writes no SIF/TWSA groups. A zero value with valid=True is a fabricated
    observation at the training mean."""
    s = make_dataset(make_env(tmp_path), monkeypatch)[0]
    assert not s["sif_valid"].any() and not s["twsa_valid"].any()
    assert not s["sif"].any() and not s["twsa"].any()


def test_era5_input_dropout_masks_values_only_never_the_slot(tmp_path, monkeypatch):
    """§35.24b item 2. It used to also set era5_doys[mask] = 0, which the model reads as
    PADDING — so training saw ~310 valid driver tokens and validation 365. That is a
    train/eval shift in sequence LENGTH, and it makes the entropy detector's per-sample
    log(n_valid) reference incomparable between the two."""
    env = make_env(tmp_path)
    eval_ds = make_dataset(env, monkeypatch, training=False)
    train_ds = make_dataset(env, monkeypatch, training=True)
    n_eval = int((eval_ds[0]["era5_doys"] > 0).sum())
    n_train = [int((train_ds[0]["era5_doys"] > 0).sum()) for _ in range(5)]
    assert all(n == n_eval for n in n_train), (
        f"training changed the number of unpadded ERA5 slots ({n_train} vs {n_eval}) — "
        f"input dropout must mask the VALUE and keep the token"
    )
    assert n_eval == 365


def test_token_sel_all_widens_every_per_patch_tensor(tmp_path, monkeypatch):
    ds = make_dataset(make_env(tmp_path), monkeypatch, token_sel="all")
    s = ds[0]
    assert tuple(s["s2_hist"].shape) == (MAX_S2, N_TOKENS, 768)
    assert tuple(s["s2_hist_valid"].shape) == (MAX_S2, N_TOKENS)
    assert tuple(s["s1_hist"].shape) == (MAX_S1, N_TOKENS, 768)
    assert tuple(s["dem_tok"].shape) == (N_TOKENS, 768)
    assert tuple(s["dem_valid"].shape) == (N_TOKENS,)
    assert s["token_idx"].tolist() == list(range(N_TOKENS))
    # the station patch of the wide read must be the narrow read
    narrow = make_dataset(make_env(tmp_path / "b"), monkeypatch, token_sel="station")[0]
    assert torch.equal(s["dem_tok"][STATION_TOKEN], narrow["dem_tok"][0])


def test_bad_token_sel_is_refused(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    with pytest.raises(ValueError, match="token_sel"):
        make_dataset(env, monkeypatch, token_sel="centre")


def test_max_stations_counts_admitted_stations(tmp_path, monkeypatch):
    """It used to count entries in _zarr_groups, which includes stations whose store is
    incomplete or whose labels never survived the filters, so --max-stations 20 could yield
    11 (§35.24 audit item 11)."""
    ds = make_dataset(make_env(tmp_path), monkeypatch, max_stations=1)
    assert len({s["station_key"] for s in ds.samples}) == 1


def test_incomplete_store_without_the_sentinel_is_skipped(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    (env["sdir"] / ".complete").unlink()
    ds = make_dataset(env, monkeypatch)
    assert len(ds) == 0, "a store with no .complete sentinel must not be read"


# ═════════════════════════════════════════════════════════════════════════════
# 7. THE REAL STORE (skipped when absent)
# ═════════════════════════════════════════════════════════════════════════════

REAL_ROOT = ds_mod.ZARR_ROOT
REAL_SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "csvs", "station_splits.csv")
REAL_ERA5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "csvs", "era5_stats.json")
REAL_DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "csvs", "driver_stats.json")


@pytest.mark.skipif(not (REAL_ROOT.exists() and os.path.exists(REAL_SPLITS)
                         and os.path.exists(REAL_ERA5) and os.path.exists(REAL_DRIVER)),
                    reason="real zarr store / csvs/driver_stats.json not present")
def test_real_store_sample_satisfies_the_contract(capsys):
    ds = SoilMoistureDataset(REAL_SPLITS, REAL_ERA5, driver_stats_path=REAL_DRIVER,
                             split_filter=["val"], training=False,
                             max_stations=1, token_sel="station")
    if len(ds) == 0:
        pytest.skip("no val sample survived the filters on this store")
    s = ds[0]
    assert tuple(s["era5_rel_pos"].shape) == (365,)
    assert int(s["era5_rel_pos"].max()) <= 364
    assert tuple(s["s1_orbit"].shape) == (MAX_S1,)
    assert set(s["s1_orbit"].unique().tolist()) <= {ORBIT_ASC, ORBIT_DESC}
    assert s["dem_valid"].dtype == torch.bool
    assert torch.isfinite(s["soil_patch"]).all()
    assert torch.isfinite(s["era5"]).all()
    assert not torch.isnan(s["s2_hist"][s["s2_hist_valid"]]).any()
    with capsys.disabled():
        nbytes = sum(v.nbytes for v in s.values() if hasattr(v, "nbytes"))
        print(f"\n  real store: {len(ds)} samples, per-sample payload {nbytes / 1024:.1f} KB")
