"""
compute_driver_stats.py — normalisation constants for the non-ERA5 drivers (§35.24)
===================================================================================

Writes csvs/driver_stats.json, the file dataset.py z-scores `sif`, `twsa` and
`soil_patch` with, and train.py reads `label_mean` from to initialise the
per-depth head biases.  It is the exact analogue of compute_era5_stats.py, and
the two are meant to be run together (slurm/driver_stats.sh runs both).

Emitted schema — BINDING, dataset.py and train.py are written against it:

    {
      "sif":  {"mean": <float>, "std": <float>},
      "twsa": {"mean": <float>, "std": <float>},
      "soil": {"mean": [21 floats], "std": [21 floats]},
      "label_mean": {"0-10": <float>, "10-30": <float>, "30-100": <float>},
      "years": [2016, 2022],
      "split": "train",
      "n_stations": <int>
    }

Three properties this script exists to guarantee
------------------------------------------------

1.  NO TEST LEAK.  Statistics come from `split == "train"` stations only, and
    only from the training years 2016-2022 — the same `CONFIG["years"] =
    list(range(2016, 2023))` train.py trains on.  2023 is the held-out OOT year
    and must not touch a normalisation constant; a constant fitted on 2023 is a
    (weak, but real) channel from the test year into every training gradient.
    This is the defect that was found in compute_era5_stats.py and fixed there
    at the same time as this file was written.

2.  PARITY WITH THE DATASET.  Every array is read through dataset.py's own
    loaders (`_load_zarr_sif`, `_load_zarr_twsa`, `_load_zarr_labels(strict=True)`)
    and the soil patch goes through dataset.py's own
    `fill_soil_nans_with_validity` before it is measured — including its
    channel-validity mask, so a dead channel's placeholder 0.0 never sets the
    scale for the stations that do have that channel.  A statistic computed over
    a differently-prepared array is not a normalisation constant, it is a bias.
    Hence the underscore-prefixed imports below: they are deliberate, dataset.py
    is the single source of truth for "what the model actually sees" and this
    script must not grow a second, drifting copy of that logic.  The station
    admission gates it applies (§35.24 audit items 4 and 6) live in
    `admit_station` here and are imported by compute_era5_stats.py, so both
    constant files are fitted on one population.

3.  `label_mean` IS THE MEAN OF WHAT THE LOSS SEES.  It initialises the model's
    per-depth head bias, so it is computed over exactly the sample set
    SoilMoistureDataset builds: qc == 0 days only (gap-filled qc==1, missing
    qc==2 and out-of-range qc==3 are all excluded — see patch_qc_out_of_range.py
    for the flag scheme), at stations that pass the soil_patch_ok filter and the
    dataset's soil / QC admission gates, on days that clear the same S2 year
    gate, the same day-granular ERA5 record test and the same `min_obs`
    threshold the dataset applies.  Units are m3/m3, straight from labels/sm —
    no rescaling anywhere in the chain.

Numerics
--------
Streaming Chan-Golub-LeVeque moment merging in float64: each worker returns
(n, mean, M2) per quantity and the parent folds the partials together.  Nothing
is concatenated — 587 train stations x ~7 years x 365 days of labels plus
21 x 74 x 74 soil pixels per station would be a needless multi-GB allocation,
and a naive `concat(...).std()` on top of it loses precision on the large-mean /
small-variance channels (sp_mean-like cases exist in the soil stack too).

Run (never on the login node — see slurm/driver_stats.sh):
    conda run -n terramind python compute_driver_stats.py
"""

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# ── Parity imports ───────────────────────────────────────────────────────────
# See point 2 in the docstring.  These are dataset.py's own loaders; if
# dataset.py changes how a driver is prepared, this script must follow it
# automatically rather than silently keep measuring the old preparation.
from dataset import (  # noqa: E402
    MAX_DEAD_SOIL_CHANNELS,
    QC_NO_SOURCE,
    QC_OBSERVED,
    ZARR_ROOT,
    SM_DEPTHS,
    _open_zarr,
    _load_zarr_era5,
    _load_zarr_labels,
    _load_zarr_sif,
    _load_zarr_twsa,
    fill_soil_nans_with_validity,
)

REPO       = Path("/gpfs/work3/0/prjs1968/soilMoisture")
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
OUT_PATH   = REPO / "csvs" / "driver_stats.json"

# Must match train.py CONFIG["years"] = list(range(2016, 2023)).  2023 is OOT.
TRAIN_YEARS = list(range(2016, 2023))

# Must match SoilMoistureDataset.__init__ defaults / train.py usage.
MIN_OBS         = 30            # min qc==0 days per station-year for that year to be used
N_SOIL_CHANNELS = 21            # OpenLandMap stack depth, soil_patch is (21, 74, 74)
N_WORKERS       = 64            # project standard: Pool(64) + --cpus-per-task=64

# A std below this is treated as degenerate (a constant channel).  Dividing by it
# would turn float noise into O(1e6) features, so we clamp to 1.0 and shout.
STD_FLOOR = 1e-12


# ── Moment helpers (float64 throughout) ──────────────────────────────────────

def _moments(v: np.ndarray) -> tuple[int, float, float]:
    """(n, mean, M2) over the finite entries of a 1-D array. M2 = sum((x-mean)^2)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return (0, 0.0, 0.0)
    mean = float(v.mean())
    m2   = float(((v - mean) ** 2).sum())
    return (n, mean, m2)


def _merge(a: tuple, b: tuple) -> tuple[int, float, float]:
    """Chan-Golub-LeVeque parallel variance merge of two (n, mean, M2) partials."""
    na, ma, m2a = a
    nb, mb, m2b = b
    if nb == 0:
        return a
    if na == 0:
        return b
    n     = na + nb
    delta = mb - ma
    mean  = ma + delta * nb / n
    m2    = m2a + m2b + delta * delta * na * nb / n
    return (n, mean, m2)


def _std(acc: tuple, label: str) -> float:
    """Sample std (ddof=1, as in compute_era5_stats.py) with a loud degenerate guard."""
    n, _, m2 = acc
    if n < 2:
        print(f"  !! {label}: only {n} observation(s) — std forced to 1.0 "
              f"(this feature will be centred but not scaled)")
        return 1.0
    s = float(np.sqrt(m2 / (n - 1)))
    if s < STD_FLOOR:
        print(f"  !! {label}: std {s:.3e} is degenerate over {n} observations — "
              f"forced to 1.0 (constant channel?)")
        return 1.0
    return s


def _empty(k: int | None = None):
    """Zero accumulator: a single (n, mean, M2) triple, or a list of k of them."""
    return (0, 0.0, 0.0) if k is None else [(0, 0.0, 0.0) for _ in range(k)]


# ── Station admission — the dataset's own quality gates ──────────────────────

def admit_station(zg) -> tuple[str | None, dict]:
    """
    Decide whether SoilMoistureDataset would keep this station, and return the
    two blocks that decision is made from so the caller does not read them twice.

    Returns (reason, payload):
        reason  : None if the station is admitted, else a short skip reason
        payload : {"soil": (21,74,74) float32 filled | None,
                   "soil_valid": (21,) bool | None,
                   "labels": (sm, depths, times, qc) | None}

    This mirrors SoilMoistureDataset.__init__ (§35.24 audit items 4 and 6) and is
    imported by compute_era5_stats.py so both constant files are fitted on exactly
    the same station population:

      * more than MAX_DEAD_SOIL_CHANNELS all-NaN soil channels  -> dropped, because
        the encoder would then be reading mostly invented values.  A station with no
        `soil` array at all lands here too: the dataset substitutes a zeros patch
        with all 21 channels flagged invalid, which is > 2 dead and therefore a drop.
      * labels/qc that cannot be aligned to labels/sm  -> dropped (strict=True; the
        old code guessed a front-trim, and a wrong guess trains on gap-filled days
        while believing they are observed).
      * labels/qc absent, or entirely the 255 "no QC source" sentinel  -> dropped,
        because "observed" cannot be distinguished from climatological gap-fill.

    A station dropped here contributes NOTHING to any statistic — not its soil, not
    its SIF/TWSA, not its labels, not its ERA5. It never becomes a training sample,
    so letting it move a normalisation constant would be measuring the wrong
    population.
    """
    payload = {"soil": None, "soil_valid": None, "labels": None}

    # ── Soil ──────────────────────────────────────────────────────────────
    if "soil" in zg:
        soil_np, soil_ok = fill_soil_nans_with_validity(np.asarray(zg["soil"][:]))
    else:
        soil_np = np.zeros((N_SOIL_CHANNELS, 74, 74), dtype=np.float32)
        soil_ok = np.zeros(N_SOIL_CHANNELS, dtype=bool)
    n_dead = int((~soil_ok).sum())
    if n_dead > MAX_DEAD_SOIL_CHANNELS:
        return f"soil-{n_dead}-dead-channels", payload
    if soil_np.shape[0] != N_SOIL_CHANNELS:
        return f"soil-has-{soil_np.shape[0]}-channels", payload
    payload["soil"], payload["soil_valid"] = soil_np, soil_ok

    # ── Labels + QC ───────────────────────────────────────────────────────
    try:
        lc = _load_zarr_labels(zg, strict=True)
    except ValueError:
        return "labels-qc-length-mismatch", payload
    if lc is None:
        return "no-sm-labels", payload
    qc = lc[3]
    if qc is None:
        return "labels-qc-absent", payload
    if bool(np.all(qc == QC_NO_SOURCE)):
        return "labels-qc-no-source-sentinel", payload
    payload["labels"] = lc

    return None, payload


# ── Per-station scan (runs in a Pool worker) ─────────────────────────────────

def scan_station(task: tuple) -> dict:
    """
    Measure one station's contribution to the driver statistics.

    Returns a dict with a `status` field:
        "ok"      — partials in "sif"/"twsa"/"soil"/"label" are usable
        "skip:<reason>" — the station contributes nothing (and why)
        "err:<msg>"     — an exception; the station is dropped, loudly

    Every skip reason is reported by the parent.  A silent skip here would show
    up as a subtly wrong constant months later, which is exactly the class of
    bug §35.24 was opened to clear out.
    """
    cat, dir_name = task
    out = {
        "station": dir_name,
        "category": cat,
        "status": "ok",
        "years": [],
        "sif": _empty(),
        "twsa": _empty(),
        "soil": _empty(N_SOIL_CHANNELS),
        "label": _empty(len(SM_DEPTHS)),
        "soil_dead_channels": 0,
    }

    try:
        station_dir = ZARR_ROOT / cat / dir_name
        zg = _open_zarr(station_dir, cat)
        if zg is None:
            out["status"] = "skip:zarr-not-complete"
            return out

        # ── The dataset's own soil / QC quality gates ─────────────────────
        reason, payload = admit_station(zg)
        if reason is not None:
            out["status"] = f"skip:{reason}"
            return out
        soil_patch, soil_valid = payload["soil"], payload["soil_valid"]
        sm_np, depths, times, qc_np = payload["labels"]
        label_years     = np.asarray(times.year)
        # YYYYMMDD per label day — needed for the day-granular ERA5 test below.
        label_date_ints = (np.asarray(times.year)  * 10000
                           + np.asarray(times.month) * 100
                           + np.asarray(times.day))

        # ── Reproduce the dataset's station/year admission gates ──────────
        # SoilMoistureDataset drops a station-year unless ERA5 covers it, S2
        # covers it, and it has >= MIN_OBS directly observed label days.  The
        # label mean has to be taken over that same set, not over the raw
        # NetCDF record, or the head bias is initialised for a population the
        # loss never sees.
        era5_entry = _load_zarr_era5(zg)
        if era5_entry is None:
            out["status"] = "skip:no-era5"
            return out
        era5_date_ints  = np.asarray(era5_entry[1])
        era5_first_int  = int(era5_date_ints[0])
        era5_last_int   = int(era5_date_ints[-1])
        era5_start_year = era5_first_int // 10000
        era5_end_year   = era5_last_int  // 10000

        if "s2/dates" not in zg:
            out["status"] = "skip:no-s2-dates"
            return out
        s2_dates      = [str(d) for d in zg["s2/dates"][:]]
        if not s2_dates:
            out["status"] = "skip:empty-s2-dates"
            return out
        s2_year_start = int(s2_dates[0][:4])
        s2_year_end   = int(s2_dates[-1][:4])

        # ── Which (station, year) pairs actually produce training samples ──
        selected_years: list[int] = []
        label_acc = _empty(len(SM_DEPTHS))

        for year in TRAIN_YEARS:
            if not (era5_start_year <= year <= era5_end_year):
                continue
            if not (s2_year_start <= year <= s2_year_end):
                continue

            year_indices = np.where(label_years == year)[0]
            if year_indices.size == 0:
                continue

            # Same rule as dataset.py: a day counts if ANY depth is directly
            # observed (qc == QC_OBSERVED).  Gap-filled (1), still-missing (2)
            # and out-of-range (3) days never become samples.  qc is guaranteed
            # non-None here — admit_station drops the station otherwise.
            valid_days = np.any(qc_np[:, year_indices] == QC_OBSERVED, axis=0)
            if int(valid_days.sum()) < MIN_OBS:
                continue

            day_idx = year_indices[valid_days]

            # ── Day-granular ERA5 admission (§35.24 audit item 1) ─────────
            # The year-level test above is only a cheap pre-filter in
            # dataset.py; the binding one is per day.  A station whose ERA5
            # record stops on 2021-03-14 used to admit every observed day of
            # 2021, with the 14-March row right-aligned into slot 364 and read
            # as "today's weather" for a September target.  Those days are not
            # samples, so their labels are not part of the head-bias population.
            # (era5_require_full_window is opt-in and defaults off in the
            # dataset, so the "window reaches back before the record" case is
            # NOT filtered here either — it stays a sample.)
            in_era5 = ((label_date_ints[day_idx] >= era5_first_int) &
                       (label_date_ints[day_idx] <= era5_last_int))
            day_idx = day_idx[in_era5]
            if day_idx.size == 0:
                continue

            selected_years.append(year)

            # Per-depth: only the depths this station reports, and only on the
            # days where THAT depth is qc==0 — dataset.py leaves the rest NaN
            # and masked_huber_loss drops them, so they must not enter the mean.
            for i, depth_str in enumerate(SM_DEPTHS):
                if depth_str not in depths:
                    continue
                d_idx = depths.index(depth_str)
                vals  = sm_np[d_idx, day_idx].astype(np.float64)
                vals  = vals[qc_np[d_idx, day_idx] == QC_OBSERVED]
                label_acc[i] = _merge(label_acc[i], _moments(vals))

        if not selected_years:
            out["status"] = "skip:no-sample-producing-year-in-2016-2022"
            return out

        out["years"] = selected_years
        out["label"] = label_acc

        # ── SIF / TWSA — restricted to the training years ─────────────────
        # Note on the rolling window: a 2016 sample can legitimately look back
        # into 2015 (the 365-day history), so a handful of observations the
        # model sees are outside the window measured here.  That asymmetry is
        # deliberate — the constraint that matters is the forward one, that no
        # 2023 value enters a constant used during training.
        year_set = set(selected_years)

        sif_entry = _load_zarr_sif(zg)
        if sif_entry is not None:
            vals, date_ints, _ = sif_entry
            keep = np.isin((np.asarray(date_ints) // 10000).astype(int), list(year_set))
            out["sif"] = _moments(np.asarray(vals)[keep])

        twsa_entry = _load_zarr_twsa(zg)
        if twsa_entry is not None:
            vals, date_ints, _ = twsa_entry
            keep = np.isin((np.asarray(date_ints) // 10000).astype(int), list(year_set))
            out["twsa"] = _moments(np.asarray(vals)[keep])

        # ── Soil — static, per channel, AFTER the dataset's NaN fill ──────
        # fill_soil_nans_with_validity is nearest-neighbour propagation, so it
        # changes the pixel distribution wherever the OpenLandMap stack had
        # holes.  The dataset z-scores THAT filled patch and feeds it to
        # SoilEncoder, so the filled patch is what has to be measured.
        #
        # The `soil_valid` mask matters: an all-NaN channel comes back filled
        # with a literal 0.0 (dataset.py sets it so, then re-zeroes it after the
        # z-score).  That 0.0 is a placeholder, not a measurement — counting it
        # would drag the channel's mean toward zero and shrink its std, i.e. the
        # invented value would set the scale for the stations that do have data.
        # Dead channels therefore contribute nothing here and are counted.
        soil_acc = []
        for c in range(N_SOIL_CHANNELS):
            if not bool(soil_valid[c]):
                soil_acc.append(_empty())
                continue
            soil_acc.append(_moments(soil_patch[c]))
        out["soil"] = soil_acc
        out["soil_dead_channels"] = int((~soil_valid).sum())

        return out

    except Exception as e:  # noqa: BLE001 — one bad station must not kill the scan
        out["status"] = f"err:{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=4)
        return out


# ── Driver ───────────────────────────────────────────────────────────────────

def select_stations(splits_csv: Path, categories: list[str]) -> list[tuple]:
    """
    The train-split stations dataset.py would actually instantiate.

    Mirrors SoilMoistureDataset.__init__: the category is derived from
    has_soil_moisture / has_flux (not from a column), the directory name
    follows the ISMN vs flux-network convention, and soil_patch_ok == False
    stations are dropped before anything is read.
    """
    splits = pd.read_csv(splits_csv)

    def _cat(r):
        sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        fl = str(r.get("has_flux",          "False")).lower() == "true"
        return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")

    tasks, dropped_soil = [], 0
    for _, r in splits.iterrows():
        if str(r["split"]).strip() != "train":
            continue
        cat = _cat(r)
        if cat not in categories:
            continue
        if not bool(r.get("soil_patch_ok", True)):
            dropped_soil += 1
            continue
        if str(r["source_network"]) == "ISMN":
            dir_name = f"ISMN_{r['network']}_{r['station_name']}"
        else:
            dir_name = f"{r['source_network']}_{r['station_id']}"
        tasks.append((cat, dir_name))

    print(f"Station selection: {len(tasks)} train stations in categories {categories} "
          f"({dropped_soil} dropped by soil_patch_ok == False)")
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--splits-csv", type=Path, default=SPLITS_CSV)
    ap.add_argument("--out",        type=Path, default=OUT_PATH)
    ap.add_argument("--workers",    type=int,  default=N_WORKERS)
    ap.add_argument("--categories", nargs="+", default=["sm_only"],
                    help="Station categories to include. Default sm_only, matching "
                         "train.py CONFIG['category_filter'].")
    args = ap.parse_args()

    print("=" * 78)
    print("compute_driver_stats.py  —  SIF / TWSA / soil / label-mean constants (§35.24)")
    print("=" * 78)
    print(f"Zarr root   : {ZARR_ROOT}")
    print(f"Splits      : {args.splits_csv}")
    print(f"Output      : {args.out}")
    print(f"Years       : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} (2023 held out for OOT — never read)")
    print(f"Split       : train")
    print(f"Categories  : {args.categories}")
    print(f"min_obs     : {MIN_OBS} qc==0 days per station-year")
    print(f"Workers     : {args.workers}")
    print()

    tasks = select_stations(args.splits_csv, args.categories)
    if not tasks:
        sys.exit("FATAL: no train stations selected — check --splits-csv / --categories.")

    sif_acc   = _empty()
    twsa_acc  = _empty()
    soil_acc  = _empty(N_SOIL_CHANNELS)
    label_acc = _empty(len(SM_DEPTHS))

    n_ok = 0
    n_soil_dead_ch = 0
    n_sif_stations = 0
    n_twsa_stations = 0
    skips: dict[str, list[str]] = {}
    errors: list[str] = []

    print(f"Scanning {len(tasks)} stations with Pool({args.workers})...")
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(scan_station, tasks, chunksize=1), 1):
            status = res["status"]
            if status.startswith("skip:"):
                skips.setdefault(status[5:], []).append(res["station"])
            elif status.startswith("err:"):
                errors.append(f"{res['station']}: {status[4:]}")
                print(f"  ERROR {res['station']}: {status[4:]}")
                if "traceback" in res:
                    print(res["traceback"])
            else:
                n_ok += 1
                sif_acc  = _merge(sif_acc,  res["sif"])
                twsa_acc = _merge(twsa_acc, res["twsa"])
                n_sif_stations  += 1 if res["sif"][0]  > 0 else 0
                n_twsa_stations += 1 if res["twsa"][0] > 0 else 0
                for c in range(N_SOIL_CHANNELS):
                    soil_acc[c] = _merge(soil_acc[c], res["soil"][c])
                for d in range(len(SM_DEPTHS)):
                    label_acc[d] = _merge(label_acc[d], res["label"][d])
                n_soil_dead_ch += int(res["soil_dead_channels"])

            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i:4d}/{len(tasks)}]  ok={n_ok}  "
                      f"skipped={sum(len(v) for v in skips.values())}  errors={len(errors)}",
                      flush=True)

    # ── Skip / error report — loud on purpose ────────────────────────────
    print()
    print("── Stations not contributing ─────────────────────────────────────")
    if not skips and not errors:
        print("  (none)")
    for reason, stations in sorted(skips.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(stations):4d}  {reason}")
        for s in stations[:10]:
            print(f"          {s}")
        if len(stations) > 10:
            print(f"          ... and {len(stations) - 10} more")
    if errors:
        print(f"  {len(errors):4d}  EXCEPTIONS:")
        for e in errors:
            print(f"          {e}")
    if n_soil_dead_ch:
        print(f"  NOTE: {n_soil_dead_ch} station-channel(s) among the ADMITTED stations were "
              f"all-NaN (<= {MAX_DEAD_SOIL_CHANNELS} per station, so the station was kept) and "
              f"contributed no soil pixels — dataset.py re-zeroes them after the z-score.")

    if n_ok == 0:
        sys.exit("FATAL: no station produced usable statistics — refusing to write "
                 "driver_stats.json.")

    # ── Fail closed on anything the model would then divide by ───────────
    if sif_acc[0] == 0:
        sys.exit("FATAL: zero SIF observations in the training years — dataset.py would "
                 "z-score with a fabricated constant. Check the sif/ arrays in the zarr store.")
    if twsa_acc[0] == 0:
        sys.exit("FATAL: zero TWSA observations in the training years — see above.")
    for d, depth_str in enumerate(SM_DEPTHS):
        if label_acc[d][0] == 0:
            sys.exit(f"FATAL: no qc==0 observations for depth {depth_str} across the train "
                     f"split — head_bias_init for that depth would be undefined.")
    dead_soil = [c for c in range(N_SOIL_CHANNELS) if soil_acc[c][0] == 0]
    if dead_soil:
        sys.exit(f"FATAL: soil channels {dead_soil} have no finite pixels anywhere in the "
                 f"train split — refusing to emit a fabricated mean/std for them.")

    result = {
        "sif":  {"mean": sif_acc[1],  "std": _std(sif_acc,  "sif")},
        "twsa": {"mean": twsa_acc[1], "std": _std(twsa_acc, "twsa")},
        "soil": {
            "mean": [soil_acc[c][1] for c in range(N_SOIL_CHANNELS)],
            "std":  [_std(soil_acc[c], f"soil[{c}]") for c in range(N_SOIL_CHANNELS)],
        },
        "label_mean": {depth: label_acc[d][1] for d, depth in enumerate(SM_DEPTHS)},
        # Per-depth qc==0 observation counts over the same sample set. train.py turns these
        # into the FIXED inverse-frequency `depth_weights` that masked_huber_loss(per_depth=)
        # needs. Without them train.py falls back to freezing epoch 1's own counts, which
        # means epoch 1 trains under uniform weights — i.e. under a different objective from
        # every epoch after it. Emitting them here removes that wrinkle entirely: the
        # objective is fixed before the first gradient step and cannot change on requeue.
        "label_count": {depth: int(label_acc[d][0]) for d, depth in enumerate(SM_DEPTHS)},
        "years": [TRAIN_YEARS[0], TRAIN_YEARS[-1]],
        "split": "train",
        "n_stations": n_ok,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("── Summary ───────────────────────────────────────────────────────")
    print(f"  stations contributing : {n_ok} / {len(tasks)}")
    print(f"  SIF   : n={sif_acc[0]:>10,}  ({n_sif_stations} stations)  "
          f"mean={result['sif']['mean']:.6f}  std={result['sif']['std']:.6f}")
    print(f"  TWSA  : n={twsa_acc[0]:>10,}  ({n_twsa_stations} stations)  "
          f"mean={result['twsa']['mean']:.6f}  std={result['twsa']['std']:.6f}")
    print(f"  SOIL  : {N_SOIL_CHANNELS} channels, "
          f"n={soil_acc[0][0]:,} pixels/channel (channel 0)")
    for c in range(N_SOIL_CHANNELS):
        print(f"          ch{c:02d}  n={soil_acc[c][0]:>9,}  "
              f"mean={result['soil']['mean'][c]:12.4f}  std={result['soil']['std'][c]:12.4f}")
    print("  LABEL MEAN (m3/m3, qc==0 only — initialises the per-depth head bias):")
    for d, depth in enumerate(SM_DEPTHS):
        n, mean, _ = label_acc[d]
        print(f"          {depth:<8s} n={n:>9,}  mean={mean:.6f}  "
              f"std={_std(label_acc[d], f'label[{depth}]'):.6f}")
    print()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
