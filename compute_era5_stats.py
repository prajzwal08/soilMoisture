"""
compute_era5_stats.py — per-variable ERA5-Land normalisation statistics
=======================================================================

Writes csvs/era5_stats.json, which dataset.py z-scores the (365, 19) ERA5
window with.  Companion to compute_driver_stats.py (SIF / TWSA / soil /
label-mean); slurm/driver_stats.sh runs both.

Output schema — UNCHANGED, dataset.py reads it as-is:
  {
    "vars":         ["t2m_mean", ...],   # 19 variable names, in feature order
    "means":        [float, ...],
    "stds":         [float, ...],
    "log1p_precip": true,                # tp_sum is log1p-transformed before z-scoring
    "precip_var":   "tp_sum"
  }

§35.24 rewrite — three things changed, none of them the schema
--------------------------------------------------------------

1.  TRAINING YEARS ONLY (the audited defect).  The previous version accumulated
    each train station's ENTIRE record, which includes 2023 — the held-out OOT
    year.  That is a genuine train/test boundary crossing: every training
    gradient was scaled by a constant that had seen the evaluation year.  Now
    restricted to TRAIN_YEARS = 2016-2022, matching train.py's
    CONFIG["years"] = list(range(2016, 2023)).

2.  ALL OF EACH STATION'S ERA5, NOT THE FIRST FILE (the second audited defect).
    The previous version did `nc_files = sorted(era5_dir.glob("meteo_*_*.nc"))`
    and then read `nc_files[0]` only, so any station stored as per-year files
    contributed one year and any station with an overlapping merged + per-year
    layout contributed an arbitrary slice.

3.  SOURCE IS THE ZARR STORE, NOT THE NetCDFs.  This is forced, not a
    preference: the ERA5Land NetCDF directories the old code globbed
    (/gpfs/work3/0/prjs1968/data/{category}/{station}/ERA5Land) no longer exist
    — verified 2026-08-26, zero of 842 sm_only station directories still have
    one; only MERIT/ and terrain/ survive there.  The surviving copy of ERA5 is
    era5/values in the token zarr store, which create_token_zarr.write_era5()
    built by merging ALL meteo_*.nc files per station (_merge_nc, with a
    dedup on the time axis) — so reading it is also what fixes defect 2 at the
    root, and it is byte-identical to the array dataset.py normalises at
    training time.  Variable order in era5/values is create_token_zarr's
    ERA5_VARS, which is the same list, in the same order, as dataset.ERA5_VARS.

4.  SAME STATION POPULATION AS driver_stats.json.  Station selection and the
    soil / label-QC admission gates are imported from compute_driver_stats.py
    (select_stations, admit_station) rather than reimplemented, so a station the
    dataset refuses to instantiate cannot shape either constant file.

Numerics: nan-aware per-station moments in float64 merged with the
Chan-Golub-LeVeque formula — same scheme as compute_driver_stats.py.  The old
per-row Python Welford loop was ~33M interpreted iterations; this is vectorised
per station and exact to the same precision.

Run (never on the login node — see slurm/driver_stats.sh):
    conda run -n terramind python compute_era5_stats.py
"""

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Parity import: dataset.py owns the definition of "the ERA5 array the model
# sees", including where the zarr store lives and how it is opened (the
# .complete sentinel check).  Duplicating either here is how the two drift.
from dataset import ZARR_ROOT, _open_zarr, _load_zarr_era5  # noqa: E402

# admit_station applies the dataset's soil / label-QC drops (§35.24 audit items
# 4 and 6).  Imported rather than reimplemented so era5_stats.json and
# driver_stats.json are fitted on the SAME station population — otherwise the
# two halves of the input normalisation would disagree about which stations
# "train" refers to, and `n_stations` in one file would not explain the other.
from compute_driver_stats import admit_station, select_stations  # noqa: E402

REPO       = Path("/gpfs/work3/0/prjs1968/soilMoisture")
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
OUT_PATH   = REPO / "csvs" / "era5_stats.json"

ERA5_VARS = [
    "t2m_mean", "t2m_min", "t2m_max",
    "d2m_mean", "d2m_min", "d2m_max",
    "skt_mean", "skt_min", "skt_max",
    "u10_mean", "u10_min", "u10_max",
    "v10_mean", "v10_min", "v10_max",
    "sp_mean",  "sp_min",  "sp_max",
    "tp_sum",
]
PRECIP_IDX = ERA5_VARS.index("tp_sum")

# Must match train.py CONFIG["years"] = list(range(2016, 2023)).  2023 is the
# held-out OOT year and must not reach a normalisation constant.
TRAIN_YEARS = list(range(2016, 2023))

N_WORKERS = 64      # project standard: Pool(64) + --cpus-per-task=64
STD_FLOOR = 1e-12


# ── Moment helpers (float64) ─────────────────────────────────────────────────

def _moments_cols(arr: np.ndarray) -> list[tuple[int, float, float]]:
    """(n, mean, M2) per column of a (rows, 19) array, ignoring non-finite entries."""
    out = []
    for j in range(arr.shape[1]):
        v = arr[:, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            out.append((0, 0.0, 0.0))
            continue
        mean = float(v.mean())
        out.append((int(v.size), mean, float(((v - mean) ** 2).sum())))
    return out


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


# ── Per-station scan (runs in a Pool worker) ─────────────────────────────────

def scan_station(task: tuple) -> dict:
    """
    Return this station's per-variable (n, mean, M2) partials over TRAIN_YEARS.

    status is "ok", "skip:<reason>" or "err:<msg>"; the parent reports every
    non-ok station rather than swallowing it in a bare counter.
    """
    cat, dir_name = task
    out = {"station": dir_name, "status": "ok", "n_rows": 0,
           "cols": [(0, 0.0, 0.0)] * len(ERA5_VARS)}
    try:
        zg = _open_zarr(ZARR_ROOT / cat / dir_name, cat)
        if zg is None:
            out["status"] = "skip:zarr-not-complete"
            return out

        # Same soil / label-QC drops the dataset applies — a station it refuses
        # never produces a training sample, so its meteorology must not shape the
        # constant every training sample is scaled by.
        reason, _ = admit_station(zg)
        if reason is not None:
            out["status"] = f"skip:{reason}"
            return out

        entry = _load_zarr_era5(zg)
        if entry is None:
            out["status"] = "skip:no-era5-in-zarr"
            return out

        values, date_ints, _ = entry
        values    = np.asarray(values, dtype=np.float64)     # (N, 19)
        date_ints = np.asarray(date_ints)                    # (N,) YYYYMMDD
        if values.ndim != 2 or values.shape[1] != len(ERA5_VARS):
            out["status"] = f"skip:era5-shape-{values.shape}"
            return out

        # ── Defect 1 fix: training years only ────────────────────────────
        years = (date_ints // 10000).astype(int)
        keep  = np.isin(years, TRAIN_YEARS)
        if not keep.any():
            out["status"] = "skip:no-rows-in-2016-2022"
            return out
        arr = values[keep]

        # log1p precipitation, exactly as dataset.py applies it before z-scoring
        # (dataset.py: era5_np[:, PREC_IDX] = np.log1p(era5_np[:, PREC_IDX].clip(0))).
        arr[:, PRECIP_IDX] = np.log1p(np.clip(arr[:, PRECIP_IDX], 0, None))

        out["n_rows"] = int(arr.shape[0])
        out["cols"]   = _moments_cols(arr)
        return out

    except Exception as e:  # noqa: BLE001 — one bad station must not kill the scan
        out["status"] = f"err:{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=4)
        return out


# ── Driver ───────────────────────────────────────────────────────────────────
# Station selection lives in compute_driver_stats.select_stations — same
# train-split / category / soil_patch_ok rules, one implementation.

def main():
    ap = argparse.ArgumentParser(description="ERA5-Land normalisation statistics (§35.24)")
    ap.add_argument("--splits-csv", type=Path, default=SPLITS_CSV)
    ap.add_argument("--out",        type=Path, default=OUT_PATH)
    ap.add_argument("--workers",    type=int,  default=N_WORKERS)
    ap.add_argument("--categories", nargs="+", default=["sm_only"])
    args = ap.parse_args()

    print("=" * 78)
    print("compute_era5_stats.py  —  ERA5-Land normalisation constants (§35.24)")
    print("=" * 78)
    print(f"Zarr root  : {ZARR_ROOT}")
    print(f"Splits     : {args.splits_csv}")
    print(f"Output     : {args.out}")
    print(f"Years      : {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} (2023 held out for OOT — never read)")
    print(f"Split      : train")
    print(f"Categories : {args.categories}")
    print(f"Workers    : {args.workers}")
    print()

    tasks = select_stations(args.splits_csv, args.categories)
    if not tasks:
        sys.exit("FATAL: no train stations selected — check --splits-csv / --categories.")

    acc = [(0, 0.0, 0.0) for _ in ERA5_VARS]
    n_ok, n_rows_total = 0, 0
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
                n_rows_total += res["n_rows"]
                for j in range(len(ERA5_VARS)):
                    acc[j] = _merge(acc[j], res["cols"][j])

            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i:4d}/{len(tasks)}]  ok={n_ok}  "
                      f"skipped={sum(len(v) for v in skips.values())}  errors={len(errors)}",
                      flush=True)

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

    if n_ok == 0:
        sys.exit("FATAL: no station produced usable ERA5 statistics — refusing to "
                 "overwrite era5_stats.json.")

    # Fail closed: a variable with no observations would otherwise be written as
    # mean 0 / std 1 and silently pass through as an unnormalised feature.
    dead = [ERA5_VARS[j] for j in range(len(ERA5_VARS)) if acc[j][0] == 0]
    if dead:
        sys.exit(f"FATAL: ERA5 variables {dead} have no finite values in the training "
                 f"years — refusing to emit fabricated constants.")

    means = [acc[j][1] for j in range(len(ERA5_VARS))]
    stds  = []
    for j, v in enumerate(ERA5_VARS):
        n, _, m2 = acc[j]
        s = float(np.sqrt(m2 / (n - 1))) if n > 1 else 1.0
        if s < STD_FLOOR:
            print(f"  !! {v}: std {s:.3e} is degenerate over {n} values — forced to 1.0")
            s = 1.0
        stds.append(s)

    result = {
        "vars":         ERA5_VARS,
        "means":        means,
        "stds":         stds,
        "log1p_precip": True,
        "precip_var":   "tp_sum",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("── Summary ───────────────────────────────────────────────────────")
    print(f"  stations contributing : {n_ok} / {len(tasks)}")
    print(f"  station-days (2016-2022): {n_rows_total:,}")
    print(f"  min finite count over the 19 variables: {min(a[0] for a in acc):,}")
    print("\nPer-variable stats (mean ± std):")
    for v, m, s, a in zip(ERA5_VARS, means, stds, acc):
        flag = " [log1p]" if v == "tp_sum" else ""
        print(f"  {v:<14s}  {m:14.4f} ± {s:12.4f}   n={a[0]:>10,}{flag}")
    print()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
