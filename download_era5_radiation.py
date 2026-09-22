#!/usr/bin/env python
"""
download_era5_radiation.py
==========================
Fetch ONLY the two ERA5-Land downward radiation bands for all 993 stations.

§43.12: the driver stack goes 19 -> 18 -- `skt_{mean,min,max}` out, `ssrd_sum` +
`strd_sum` in.  The other 15 features already live in `era5/values` in the token
store, so this script deliberately fetches **two bands, not seven**.  Re-fetching
the other five variables would resample a different GEE snapshot with re-probed
strategies and silently change the SM model's inputs for reasons unrelated to
this change.  `splice_era5_radiation.py` joins the two new columns onto the
existing array by date.

THE RULE (§43.12):  state variables get mean/min/max, accumulations get a daily
sum.  Radiation is an accumulation.  Each `_hourly` value is already J m-2 over
that hour, so 24 of them summed is the day's incoming energy.  `ssrd_min` is ~0
every night, and -- decisively -- a 24 h total is nearly insensitive to where the
UTC day boundary cuts the local diurnal cycle, while a daily max is not.

BAND NAMES.  ECMWF/ERA5_LAND/HOURLY carries both an accumulated band and a
`_hourly` de-accumulated variant.  We need the `_hourly` one: the plain band
accumulates from 00 UTC, so summing it gives a running total, not a daily flux.
This is the same trap `download_era5land_gee.py` already dodges for
`total_precipitation_hourly`.

TWO DIFFERENCES FROM `download_era5land_gee.py`, both deliberate:

  1. `detect_strategy` is resolved ONCE PER STATION, not once per station-year.
     The original sets `job["strategy"] = None` for every job (:211) and re-probes
     inside `process_station_year` (:378-380), so the same coordinate is probed
     once per year.  Radiation must also use the *same* strategy the original 19
     used, or a coastal station ends up with valid t2m and NaN radiation.

  2. Buffer pixels are collapsed per timestamp BEFORE the daily aggregation.
     `getRegion` on a 25 km buffer returns one row per (pixel, time).
     `download_era5land_gee.py` never groups by time, so its `.resample("1D").sum()`
     for `tp_sum` sums across pixels as well as hours -- inflating precipitation by
     roughly the pixel count (~16 at 0.1 deg in a 25 km buffer).  Measured blast
     radius: THREE stations -- PortGraham, Cape-Charles-5-ENE, Combate -- whose own
     ERA5-Land cell is ocean-masked; the 22 log rows are station-YEARS.  Every other
     station uses a point query on its own cell.  That is a pre-existing bug in the
     stored `tp_sum` for those three; it is
     NOT fixed here (fixing it would mean re-fetching the other 15 columns), but it
     must not be inherited by the radiation sums.

Usage
-----
    python download_era5_radiation.py --stations A,B,C    # smoke
    python download_era5_radiation.py                     # full 993
    sbatch jobs/era5_radiation.sh

Resume is automatic: a station-year whose `rad_{year}.nc` exists is skipped, and
writes are tmp->rename so a partial file never satisfies the skip test.  A
resubmit therefore costs nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd
import xarray as xr

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_era5land_gee import (  # noqa: E402
    ERA5_BUFFER_M,
    GEE_PROJECT,
    GEE_SCALE,
    N_WORKERS,
    STRATEGY_BUFFER,
    STRATEGY_ERA5,
    STRATEGY_EXCLUDE,
    _gee_credentials,
    _gee_getregion_with_retry,
    _getregion_to_df,
    detect_strategy,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT   = Path("/gpfs/work3/0/prjs1968/data")
REPO_ROOT   = Path("/gpfs/work3/0/prjs1968/soilMoisture")

# The AUTHORITATIVE station list -- 993 rows.  `download_era5land_gee.py:62` reads
# `{DATA_ROOT}/station_splits.csv` instead, which has 1010 rows; CLAUDE.md and
# `create_token_zarr.py:45` both name the csvs/ copy as the real one.
STATION_CSV = REPO_ROOT / "csvs" / "station_splits.csv"

LOG_DIR  = DATA_ROOT / "logs"
LOG_FILE = REPO_ROOT / "csvs" / "era5_radiation_log.csv"

GEE_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"

# The de-accumulated variants.  Verified present in the GEE catalogue, units J/m2,
# documented as "disaggregated from the original cumulative values into hourly
# values".  `meteodata_ERA5land_GEE.md:159-160` already names both.
RAD_BANDS   = [
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
]
RAD_SHORT   = ["ssrd", "strd"]
RAD_OUT     = ["ssrd_sum", "strd_sum"]

LOG_COLS = ["station_id", "year", "status", "strategy", "n_px", "n_days",
            "n_nan_days", "error_msg", "timestamp"]

# Pixels that STRATEGY_BUFFER stations are pinned to, resolved once per station and
# persisted so every year of a station uses the identical footprint.
PIXEL_JSON = REPO_ROOT / "csvs" / "era5_buffer_pixels.json"

# Months probed when resolving that footprint -- one per season, so a pixel only
# survives if it is populated year-round.
PROBE_MONTHS = [(2021, 1), (2021, 4), (2021, 7), (2021, 10)]

_log_lock = threading.Lock()


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "era5_radiation.log"),
        ],
    )


def append_log(row: dict) -> None:
    with _log_lock:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        header = not LOG_FILE.exists()
        pd.DataFrame([{c: row.get(c, "") for c in LOG_COLS}]).to_csv(
            LOG_FILE, mode="a", header=header, index=False
        )


# ============================================================
# STATIONS
# ============================================================

def load_stations() -> pd.DataFrame:
    """The 993 authoritative stations, with folder name and output directory."""
    df = pd.read_csv(STATION_CSV)

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"

    def _dir(r):
        has_sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        has_fl = str(r.get("has_flux", "False")).lower() == "true"
        cat = "sm_and_flux" if (has_sm and has_fl) else ("sm_only" if has_sm else "flux_only")
        return DATA_ROOT / cat / r["folder"]

    df["folder"] = df.apply(_folder, axis=1)
    df["station_dir"] = df.apply(_dir, axis=1)
    return df.reset_index(drop=True)


def build_job_list(df: pd.DataFrame) -> tuple[list[dict], int]:
    """One job = one (station, year) with no `rad_{year}.nc` yet."""
    log = logging.getLogger(__name__)
    jobs, n_skipped = [], 0

    for _, row in df.iterrows():
        try:
            start_year = int(str(row["start_date"])[:4])
            end_year   = int(str(row["end_date"])[:4])
        except (ValueError, TypeError):
            log.warning(f"  {row['folder']}: unparseable date range, skipping station")
            continue

        era5_dir = Path(row["station_dir"]) / "ERA5Land"
        for year in range(start_year, end_year + 1):
            out = era5_dir / f"rad_{year}.nc"
            if out.exists():
                n_skipped += 1
                continue
            jobs.append({
                "folder":     row["folder"],
                "station_id": row["station_id"],
                "lat":        float(row["latitude"]),
                "lon":        float(row["longitude"]),
                "year":       year,
                "output_path": out,
            })
    return jobs, n_skipped


# ============================================================
# FETCH
# ============================================================

def _getregion_with_pixels(collection, geometry, scale: int) -> pd.DataFrame:
    """Like `_getregion_to_df` but KEEPS pixel identity.

    `_getregion_to_df` drops longitude/latitude, which is fine for a point query
    (one pixel) but destroys the information a buffer query needs: getRegion
    returns one row per (pixel, hour), and which pixels carry data CHANGES DAY TO
    DAY.  Averaging whatever is valid today makes the series track the ERA5-Land
    mask rather than the weather -- measured at Combate, where the mid-March mean
    halved because extra, darker pixels appeared for 16 days.
    """
    raw = _gee_getregion_with_retry(collection, geometry, scale)
    if len(raw) <= 1:
        return pd.DataFrame(columns=["time", "px"] + RAD_SHORT)
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["px"] = (df["longitude"].astype(float).round(4).astype(str) + ","
                + df["latitude"].astype(float).round(4).astype(str))
    df = df.rename(columns=dict(zip(RAD_BANDS, RAD_SHORT)))
    for c in RAD_SHORT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time", "px"] + RAD_SHORT]


def _buffer_collection(year: int, month: int):
    return (ee.ImageCollection(GEE_COLLECTION)
            .filterDate(_month_start(year, month), _month_end(year, month))
            .select(RAD_BANDS))


def resolve_buffer_pixels(folder: str, lat: float, lon: float) -> list[str]:
    """The pixels inside the 25 km buffer that are valid in EVERY probed hour.

    Intersected across four months spanning the seasons, so a pixel only survives
    if it is populated year-round.  Persisted to csvs/era5_buffer_pixels.json so
    every year of a station uses the identical footprint -- otherwise the pathology
    returns as a step between years.
    """
    log = logging.getLogger(__name__)
    cache = json.loads(PIXEL_JSON.read_text()) if PIXEL_JSON.exists() else {}
    if folder in cache:
        return cache[folder]["pixels"]

    geom = ee.Geometry.Point([lon, lat]).buffer(ERA5_BUFFER_M)
    keep: set[str] | None = None
    seen = 0
    for (y, m) in PROBE_MONTHS:
        df = _getregion_with_pixels(_buffer_collection(y, m), geom, GEE_SCALE)
        if df.empty:
            continue
        seen = max(seen, df["px"].nunique())
        frac = df.groupby("px")[RAD_SHORT].apply(lambda g: g.notna().all(axis=1).mean())
        good = set(frac[frac >= 1.0].index)
        keep = good if keep is None else (keep & good)

    pixels = sorted(keep or [])
    log.info(f"  {folder}: {len(pixels)} of {seen} buffer pixels valid year-round")
    if not pixels:
        raise RuntimeError(
            f"{folder}: no pixel in the 25 km buffer is valid across all probe "
            f"months. This station has no stable ERA5-Land footprint and must be "
            f"excluded rather than averaged over a shifting mask.")

    cache[folder] = {"pixels": pixels, "n_candidates": seen,
                     "probe_months": [f"{y}-{m:02d}" for y, m in PROBE_MONTHS],
                     "lat": lat, "lon": lon}
    PIXEL_JSON.write_text(json.dumps(cache, indent=2))
    return pixels


def _fetch_month_rad(lat: float, lon: float, year: int, month: int,
                     strategy: str, pixels: list[str] | None = None) -> pd.DataFrame:
    """One month of hourly de-accumulated radiation at (lat, lon).

    Returns a time-indexed frame with pixels already collapsed to one row per hour.
    """
    if strategy == STRATEGY_EXCLUDE:
        return pd.DataFrame(columns=["time"] + RAD_SHORT)

    if strategy == STRATEGY_ERA5:
        # ECMWF/ERA5/HOURLY carries only the ACCUMULATED radiation bands, not the
        # `_hourly` de-accumulated variants, so summing them would give a running
        # total.  De-accumulating by hand is possible (difference consecutive
        # steps, reset at 00 UTC) but has never been needed: text/logs.txt:537-570
        # records that all three known coastal stations resolved via STRATEGY_BUFFER
        # and strategy 3 has never run in production.  Fail loudly rather than
        # silently writing accumulated values into a column called `_sum`.
        raise RuntimeError(
            "STRATEGY_ERA5 (ECMWF/ERA5/HOURLY) has no *_hourly radiation bands. "
            "Manual de-accumulation is required before this station can be used; "
            "see download_era5_radiation.py for the reasoning."
        )

    collection = _buffer_collection(year, month)

    if strategy == STRATEGY_BUFFER:
        geometry = ee.Geometry.Point([lon, lat]).buffer(ERA5_BUFFER_M)
        df = _getregion_with_pixels(collection, geometry, GEE_SCALE)
        if pixels:
            df = df[df["px"].isin(pixels)]
        # Collapse the FIXED pixel set to one row per hour.
        return df.groupby("time", as_index=False)[RAD_SHORT].mean()

    geometry = ee.Geometry.Point([lon, lat])
    raw = _gee_getregion_with_retry(collection, geometry, GEE_SCALE)
    return _getregion_to_df(raw, RAD_BANDS, RAD_SHORT)


def _month_start(year: int, month: int) -> str:
    return f"{year}-{month:02d}-01"


def _month_end(year: int, month: int) -> str:
    return f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"


def process_station_year(job: dict, strategy: str,
                         pixels: list[str] | None = None) -> dict:
    """Fetch, aggregate to daily sums, write rad_{year}.nc atomically."""
    log = logging.getLogger(__name__)
    out: Path = job["output_path"]
    result = {"station_id": job["folder"], "year": job["year"],
              "strategy": strategy, "n_px": len(pixels) if pixels else 1,
              "status": "error", "n_days": 0, "n_nan_days": 0,
              "error_msg": "", "timestamp": ""}

    tmp = out.with_suffix(".tmp.nc")
    try:
        if strategy == STRATEGY_EXCLUDE:
            result["status"] = "exclude"
            result["error_msg"] = "all ERA5 strategies returned NaN - ocean pixel"
            return result

        monthly = [_fetch_month_rad(job["lat"], job["lon"], job["year"], m,
                                    strategy, pixels)
                   for m in range(1, 13)]
        df = pd.concat(monthly, ignore_index=True).dropna(subset=["time"])

        if df.empty:
            result["status"] = "no_data"
            result["error_msg"] = "no rows returned for any month"
            return result

        # Pixels are already collapsed inside _fetch_month_rad (to the PINNED set
        # for buffer stations, trivially for a point query), so this is one row
        # per hour.
        df = df.groupby("time", as_index=True)[RAD_SHORT].mean().sort_index()

        # min_count=24: a day missing ANY hour becomes NaN rather than a quiet
        # partial sum.  Without it `.sum()` skips absent hours and a half-day
        # reads as a plausible half-value that no range check can catch.  A NaN
        # day is picked up by the splice's gap policy, and at load time it takes
        # the same path as dataset.py's existing 15% ERA5 dropout -- value 0 and
        # doy 0, which the transformer treats as padding.
        daily = pd.concat(
            [df[c].resample("1D").sum(min_count=24).rename(o)
             for c, o in zip(RAD_SHORT, RAD_OUT)],
            axis=1,
        )
        daily = daily[daily.index.year == job["year"]]
        if daily.empty:
            result["status"] = "no_data"
            result["error_msg"] = f"no rows fell inside {job['year']}"
            return result

        daily.index = daily.index.tz_localize(None)
        ds = xr.Dataset.from_dataframe(daily)
        for v, long in zip(RAD_OUT, ("surface solar radiation downwards daily sum",
                                     "surface thermal radiation downwards daily sum")):
            ds[v].attrs["units"] = "J m**-2"
            ds[v].attrs["long_name"] = long
        ds.attrs.update({
            "station_id": job["folder"],
            "latitude":   job["lat"],
            "longitude":  job["lon"],
            "strategy":   strategy,
            "source":     GEE_COLLECTION,
            "bands":      ", ".join(RAD_BANDS),
            "created":    datetime.now(timezone.utc).isoformat(),
        })

        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(str(tmp))
        tmp.rename(out)          # atomic on POSIX

        n_nan = int(daily[RAD_OUT].isna().any(axis=1).sum())
        result["status"] = "done"
        result["n_days"] = int(len(daily))
        result["n_nan_days"] = n_nan
        log.info(f"  {job['folder']} {job['year']}: {len(daily)} days "
                 f"({n_nan} incomplete) [{strategy}, {result['n_px']} px]")

    except Exception as exc:
        result["error_msg"] = str(exc)[:300]
        log.error(f"  {job['folder']} {job['year']}: {exc}")
        for p in (tmp, out):
            if p.exists():
                p.unlink()
    finally:
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        append_log(result)

    return result


# ============================================================
# STRATEGY RESOLUTION (once per station)
# ============================================================

def resolve_strategies(jobs: list[dict], workers: int) -> dict[str, str]:
    """Probe GEE once per DISTINCT station, not once per station-year."""
    log = logging.getLogger(__name__)
    coords = {}
    for j in jobs:
        coords.setdefault(j["folder"], (j["lat"], j["lon"]))

    log.info(f"Resolving fetch strategy for {len(coords)} stations "
             f"({len(jobs)} station-years)…")

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(detect_strategy, lat, lon): folder
                for folder, (lat, lon) in coords.items()}
        for fut in as_completed(futs):
            folder = futs[fut]
            try:
                out[folder] = fut.result()
            except Exception as exc:
                log.error(f"  strategy probe failed for {folder}: {exc}")
                out[folder] = STRATEGY_EXCLUDE

    counts = pd.Series(list(out.values())).value_counts().to_dict()
    log.info(f"Strategies: {counts}")
    return out


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", type=str, default=None,
                    help="comma-separated folder names, for a smoke run")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of stations processed")
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help=f"concurrent GEE calls (default {N_WORKERS}; raised "
                         f"values risk 429)")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    ee.Initialize(credentials=_gee_credentials(), project=GEE_PROJECT)
    log.info("GEE initialised")

    df = load_stations()
    log.info(f"Stations in {STATION_CSV.name}: {len(df)}")

    if args.stations:
        wanted = {s.strip() for s in args.stations.split(",") if s.strip()}
        df = df[df["folder"].isin(wanted)]
        missing = wanted - set(df["folder"])
        if missing:
            log.error(f"unknown station folders: {sorted(missing)}")
            return 1
    if args.limit:
        df = df.head(args.limit)
    log.info(f"Stations selected: {len(df)}")

    jobs, n_skipped = build_job_list(df)
    log.info(f"Station-years: {len(jobs)} to fetch, {n_skipped} already on disk")
    if not jobs:
        log.info("nothing to do")
        return 0

    strategies = resolve_strategies(jobs, args.workers)

    # Pin every STRATEGY_BUFFER station to a footprint that does not move.
    pixels: dict[str, list[str]] = {}
    buffered = sorted({j["folder"] for j in jobs
                       if strategies[j["folder"]] == STRATEGY_BUFFER})
    if buffered:
        log.info(f"Resolving a fixed pixel footprint for {len(buffered)} "
                 f"buffer station(s)…")
        for folder in buffered:
            lat, lon = next((j["lat"], j["lon"]) for j in jobs if j["folder"] == folder)
            pixels[folder] = resolve_buffer_pixels(folder, lat, lon)

    t0 = datetime.now(timezone.utc)
    done = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_station_year, j, strategies[j["folder"]],
                          pixels.get(j["folder"]))
                for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["status"] == "done":
                done += 1
            else:
                errors += 1
            if i % 100 == 0 or i == len(futs):
                el = (datetime.now(timezone.utc) - t0).total_seconds()
                log.info(f"[{i}/{len(futs)}] ok={done} problem={errors} "
                         f"elapsed={el / 60:.1f} min ({el / max(i, 1):.1f} s/job)")

    log.info(f"finished: {done} ok, {errors} problem, log -> {LOG_FILE}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
