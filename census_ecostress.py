#!/usr/bin/env python
"""
§36 TIER 1 -- ECOSTRESS day/night pair census.

ONE pass.  For every station in range it does, together:
    part A   CMR granule list + solar geometry          (no pixels)
    part B   windowed reads of QC / cloud / water / view_zenith  (pixels, EDL required)
    part C   QA + cloud filtering, then day/night pairing

Output is the definitive per-station QUALITY-PAIR inventory.  It does NOT read _LST --
that is Tier 2 (`download_ecostress_lste.py`), scoped by what this qualifies.

Collection is ECO_L2T_LSTE **v002** (concept id C2076090826-LPCLOUD).  v003 exists but
has ZERO granules before late 2025, so it cannot cover the label window -- see §36.8.

NEVER run this on a login node.  Always sbatch:
    sbatch slurm/ecostress_census.sh selftest
    sbatch slurm/ecostress_census.sh dryrun
    sbatch slurm/ecostress_census.sh controls
    sbatch slurm/ecostress_census.sh sample 20
    sbatch slurm/ecostress_census.sh full

Runbook §36.13-§36.16.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIG
# ============================================================

ROOT        = Path("/gpfs/work3/0/prjs1968/soilMoisture")
STATION_CSV = ROOT / "csvs" / "station_splits.csv"
OUT_GRAN    = ROOT / "csvs" / "ecostress_census_granules.csv"
OUT_PAIRS   = ROOT / "csvs" / "ecostress_census_pairs.csv"
LOG_FILE    = ROOT / "csvs" / "ecostress_census_log.csv"
LOG_DIR     = ROOT / "logs"
COOKIE_JAR  = Path(os.environ.get("TMPDIR", "/tmp")) / "edl_cookies.txt"

CMR_URL       = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
CONCEPT_ID    = "C2076090826-LPCLOUD"          # ECO_L2T_LSTE v002 -- §36.7
MISSION_START = "2018-07-09T00:00:00Z"
# Latitude limit.  NOT 52.0.  The catalogue PAGE says "52N to 52S" in prose, but the
# machine-readable collection metadata for C2076090826-LPCLOUD declares
#     N=54  S=-54  W=-180  E=180
# The ISS inclination is 51.6 deg, but the swath reaches past the sub-satellite track.
# Using the prose figure dropped 38 stations in the 52-54 band, one of them a flux tower.
# Trust the metadata, and let --controls measure where coverage ACTUALLY stops.
LAT_LIMIT     = 54.0
LAT_NOMINAL   = 52.0                           # the prose figure, kept only for reporting
CONTROL_LO    = 50.0                           # --controls sweeps this band to find the
CONTROL_HI    = 62.0                           # real edge from data rather than assuming it

TILE_M        = 2240.0                         # the station window, matching the model tile
PAGE_SIZE     = 2000
HTTP_WORKERS  = 8                              # NOT 64: CMR and LP DAAC both throttle
MAX_RETRIES   = 5
RETRY_WAITS   = [2, 5, 15, 30, 60]
_NO_RETRY_HTTP = {400, 401, 403, 404}
TIMEOUT       = 120

LP_BASE       = ("https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
                 "ECO_L2T_LSTE.002")

# §36.3a -- windows in solar geometry, and the crossover exclusion
DAY_LO, DAY_HI       = 0.5, 3.5       # hours after solar noon
NIGHT_HALF           = 2.0            # hours either side of solar midnight
CROSSOVER_LO         = -5.0           # solar elevation, degrees
CROSSOVER_HI         = 10.0

# Well-phased reporting (NOT a filter).  Surface temperature lags forcing by ~1-2 h, so
# the thermal peak sits after solar noon; a night pass is best near the pre-dawn minimum.
THERMAL_PEAK_LAG_H     = 2.0    # hours after solar noon where the day half is ideal
WELL_PHASED_DAY_H      = 2.5    # day half within this of the peak
WELL_PHASED_NIGHT_TST  = 2.0    # night half at or after this local solar hour (pre-dawn)

# §36.12 -- image-level floor; swept in analysis, this is only the default headline
CLEAR_FRAC_MIN       = 0.5

# The window we expect over a 2.24 km box at ECOSTRESS's 70 m grid.  Used to DETECT the
# silent clipping rasterio does when a window overruns the raster (H5).
PIXEL_M              = 70.0
N_PX_EXPECTED        = int(round(TILE_M / PIXEL_M)) ** 2      # 32 x 32 = 1024
WINDOW_FRAC_MIN      = 0.75           # below this the read is rejected, not merely flagged

# §36.14 -- the pairing sensitivity grid, three axes.
#
# next_day_only is the strict rule: the night pass must fall on the LOCAL SOLAR DATE
# following the day pass.  It is an AXIS, not a hardcoded constraint, so the census
# reports what it costs instead of assuming the cost is acceptable.
#
# Note "following date" and "< 24 h" are NOT the same constraint -- a 13:30 day pass
# paired to anything on D+1 spans ~10 h to ~34 h -- which is why both axes exist.
#
# The strict arm's known cost: the night window is solar midnight +/- NIGHT_HALF, so its
# 22:00-23:59 half sits on date D, the same evening as the day pass.  next_day_only=True
# discards those.  Compare the two arms to see how many.
DT_TOLERANCES_H      = [18, 24, 36, 60]
NEXT_DAY_MODES       = [True, False]

# Headline reported in the log; the full grid always goes to the CSV.
HEADLINE_DT_H        = 24
HEADLINE_NEXT_DAY    = True

GRAN_COLS = [
    "station_id", "network", "lat", "lon", "elevation_m", "kg_macro",
    "granule_ur", "orbit", "scene", "tile", "utc", "day_night_flag",
    "tst", "hours_from_solar_noon", "solar_elev", "year", "doy", "solar_date_str",
    "phase",                 # day | night | crossover | off
    "read_ok", "n_px", "n_px_expected", "window_frac", "clipped",
    "clear_frac", "valid_frac", "vza_mean_abs", "vza_max_abs",
    "frac_mand00", "frac_mand01", "frac_lstacc_ge2", "frac_water", "frac_cloud",
    "passed_qc", "granule_ur_used", "error",
]
PAIR_COLS = [
    "station_id", "day_ur", "night_ur", "day_utc", "night_utc", "dt_hours",
    "day_tst", "night_tst", "day_elev", "night_elev", "elev_drop",
    "day_solar_date", "night_solar_date", "well_phased",
    "day_clear", "night_clear", "quality",
]
LOG_COLS = ["station_id", "status", "n_hits", "n_overpasses", "n_dupe_orbit",
            "n_dupe_reproc", "n_inwindow", "n_read_ok", "n_passed",
            "n_pairs_quality", "n_pairs_well_phased", "error", "timestamp"]


# ============================================================
# SOLAR GEOMETRY  (§36.14)
# ============================================================

def solar_geometry(utc_iso: str, lat: float, lon: float):
    """-> (true_solar_time_h, hours_from_solar_noon, solar_elevation_deg, year, doy).

    Mean solar time PLUS the equation of time.  Dropping E costs up to +/-16 min --
    harmless for 3 h bins, not harmless for the empirical window curve, so it is in.
    """
    ts  = pd.Timestamp(utc_iso)
    doy = int(ts.dayofyear)
    utc_h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0

    decl = 23.45 * math.sin(math.radians(360.0 * (284 + doy) / 365.0))
    b    = math.radians(360.0 * (doy - 81) / 364.0)
    eot  = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)   # minutes

    tst = (utc_h + lon / 15.0 + eot / 60.0) % 24.0
    dt_noon = tst - 12.0
    h = math.radians(15.0 * dt_noon)

    sin_elev = (math.sin(math.radians(lat)) * math.sin(math.radians(decl))
                + math.cos(math.radians(lat)) * math.cos(math.radians(decl)) * math.cos(h))
    elev = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))
    return tst, dt_noon, elev, int(ts.year), doy


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def iso_end_date(raw) -> str:
    """station_splits.csv stores dates as BASIC ISO -- `20181230`, 8 chars, no dashes.

    CMR's `temporal` parameter requires EXTENDED ISO 8601 and answers HTTP 400 otherwise.
    400 is in _NO_RETRY_HTTP, so a malformed date makes every station log status=error
    and the run finish cleanly reporting ZERO granules -- and --controls then prints
    "CONTROL PASSED: zero granules", because zero is exactly what total failure produces.
    The bug validates itself through the test meant to catch it, which is why this
    function asserts rather than trusting a slice.
    """
    s = str(raw or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return "2026-12-31"
    try:
        out = pd.Timestamp(s).strftime("%Y-%m-%d")
    except Exception as exc:                                        # noqa: BLE001
        raise ValueError(f"un-parseable end_date {raw!r}: {exc}") from exc
    if not _ISO_DATE.match(out):
        raise ValueError(f"end_date {raw!r} did not normalise to YYYY-MM-DD (got {out!r})")
    return out


def solar_date(utc_iso: str, lon: float):
    """The LOCAL SOLAR date of an acquisition.

    Load-bearing for the next-day pairing rule.  UTC date is NOT a substitute: at
    lon -98.78 a 01:30 local pass is already 08:00 UTC the next day, while at lon +150
    the two diverge the other way.  Comparing UTC dates would silently mis-pair whole
    regions of the network.
    """
    ts  = pd.Timestamp(utc_iso)
    doy = int(ts.dayofyear)
    b   = math.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)   # minutes
    offset_h = lon / 15.0 + eot / 60.0
    return (ts + pd.Timedelta(hours=offset_h)).normalize()


def solar_noon_utc(solar_day: pd.Timestamp, lon: float) -> pd.Timestamp:
    """UTC instant at which TST == 12 on the given LOCAL SOLAR date.

    Upper bound for the nocturnal-period pairing rule: a night granule belongs to the
    night FOLLOWING a day pass only if it lands before the next day's solar noon.  Using
    solar noon rather than a fixed clock hour makes the rule correct at every latitude
    and season -- "10:30 day -> 22:30 through 04:30" falls out automatically.
    """
    doy = int(pd.Timestamp(solar_day).dayofyear)
    b   = math.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)   # minutes
    offset_h = lon / 15.0 + eot / 60.0          # local solar = utc + offset
    return pd.Timestamp(solar_day).normalize() + pd.Timedelta(hours=12.0 - offset_h)


def classify_phase(tst: float, elev: float) -> str:
    """day | night | crossover | off.  Solar geometry, never clock hour (§36.14).

    The crossover test comes FIRST: near sunrise/sunset the wet-dry thermal contrast
    passes through zero, so those granules carry no moisture signal however clear.
    """
    # SOLAR ELEVATION ALONE.  The clock-hour windows this used to apply (noon+0.5..+3.5 h
    # for day, solar midnight +/-2 h for night) discarded genuinely usable acquisitions:
    # at Banizoumbou the ONLY night granule in the whole record sits at TST 04:38 with the
    # sun 19.8 deg below the horizon -- unambiguously night, and near the pre-dawn minimum
    # that classical thermal-inertia work actually uses -- yet it fell 4.6 h from solar
    # midnight and was classified 'off'.  That single exclusion produced zero pairs at
    # that station.  Elevation is the physical definition of illumination state and is
    # correct at every latitude and season.
    #
    # Phase WITHIN day/night is recorded (tst, hours_from_solar_noon, solar_elev) but NOT
    # enforced -- comparability is handled by reporting well-phased pairs separately, not
    # by discarding acquisitions up front.
    if CROSSOVER_LO <= elev <= CROSSOVER_HI:
        return "crossover"
    if elev > CROSSOVER_HI:
        return "day"
    return "night"


# ============================================================
# HELPERS  (idioms from download_landsat_st_mpc.py:187-228)
# ============================================================

def with_retry(fn, max_retries=MAX_RETRIES, waits=RETRY_WAITS):
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in _NO_RETRY_HTTP:
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt])
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(waits[attempt])


def load_done() -> set[str]:
    """Stations that completed a FULL pass.  status='dryrun' is deliberately excluded --
    a dry run reads no pixels, so its rows have empty clear_frac/passed_qc and must not
    satisfy a later census run (H2)."""
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE, dtype=str)
        return set(df.loc[df["status"] == "ok", "station_id"])
    return set()


def truncate_outputs(log):
    """--fresh must clear the DATA files too, not just the checkpoint (H3).

    append_rows() only writes a header when the file is absent and otherwise appends, so
    a --fresh re-run over existing CSVs silently duplicates every granule and pair row.
    Nothing downstream dedupes, so per-station pair counts double and the §36.16 '>=20
    quality pairs' floor would be cleared on phantom data.
    """
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    for p in (OUT_GRAN, OUT_PAIRS, LOG_FILE):
        if p.exists():
            bak = p.with_suffix(f".{stamp}.bak.csv")
            p.rename(bak)
            log.info("--fresh: moved %s -> %s", p.name, bak.name)


def append_rows(path: Path, cols: list[str], rows: list[dict]):
    if not rows:
        return
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def setup_logging(name: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"{name}.log")],
    )


def configure_gdal():
    """GDAL settings for /vsicurl against LPCLOUD.

    EDL bounces through a redirect on a different host; GDAL needs the netrc entry AND a
    cookie jar to carry the session across it.  READDIR_ON_OPEN must be EMPTY_DIR or GDAL
    lists the whole granule directory on every open -- one wasted request per layer.
    """
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
    os.environ["GDAL_HTTP_COOKIEFILE"] = str(COOKIE_JAR)
    os.environ["GDAL_HTTP_COOKIEJAR"] = str(COOKIE_JAR)
    os.environ["GDAL_HTTP_NETRC"] = "YES"
    os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
    os.environ["VSI_CACHE"] = "TRUE"
    os.environ["VSI_CACHE_SIZE"] = "33554432"


# ============================================================
# PART A -- CMR
# ============================================================

def cmr_granules(session, lon: float, lat: float, end_date: str,
                 start: str = MISSION_START) -> list[dict]:
    """UMM-G records whose footprint contains (lon, lat).

    point=, not bounding_box= -- §29.3's bbox pulled a second MGRS tile and ~995
    redundant granules.  Pages with CMR-Search-After, not the deprecated page_num.
    """
    params = {
        "collection_concept_id": CONCEPT_ID,
        "point": f"{lon},{lat}",
        "temporal": f"{start},{end_date}",
        "page_size": PAGE_SIZE,
    }
    items, headers = [], {}
    while True:
        def _get():
            r = session.get(CMR_URL, params=params, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        resp = with_retry(_get)
        batch = resp.json().get("items", [])
        items.extend(batch)
        sa = resp.headers.get("CMR-Search-After")
        if not sa or len(batch) < PAGE_SIZE:
            break
        headers["CMR-Search-After"] = sa
    return items


def parse_ur(ur: str):
    """ECOv002_L2T_LSTE_{orbit}_{scene}_{tile}_{YYYYMMDDTHHMMSS}_{build}_{ver}

    orbit = field 4, scene = 5, tile = 6 (0-indexed 3/4/5).  Getting this wrong silently
    breaks orbit dedup, which is why §36.20 checks the drop rate lands near 25%.
    """
    p = ur.split("_")
    return (p[3], p[4], p[5]) if len(p) >= 7 else ("", "", "")


def dedupe(items: list[dict]):
    """-> (records, n_dropped_reprocessing, n_dropped_orbit).  §36.9."""
    recs = []
    for it in items:
        umm = it.get("umm", {})
        ur  = umm.get("GranuleUR", "")
        dg  = umm.get("DataGranule", {}) or {}
        tmp = umm.get("TemporalExtent", {}) or {}
        utc = ((tmp.get("RangeDateTime") or {}).get("BeginningDateTime")
               or tmp.get("SingleDateTime"))
        if not ur or not utc:
            continue
        orbit, scene, tile = parse_ur(ur)
        recs.append({"granule_ur": ur, "orbit": orbit, "scene": scene, "tile": tile,
                     "utc": utc, "day_night_flag": dg.get("DayNightFlag", ""),
                     "production": dg.get("ProductionDateTime", "") or ""})

    best = {}
    for r in recs:                                   # reprocessing: keep newest build
        k = (r["tile"], r["utc"])
        if k not in best or r["production"] > best[k]["production"]:
            best[k] = r
    n_reproc = len(recs) - len(best)

    # ONE OVERPASS PER (orbit, scene) -- NOT per (tile, orbit).  A station sitting in an
    # MGRS tile overlap receives the SAME acquisition as two granules, one per tile:
    # measured at Banizoumbou, which straddles 31PDQ and 31PDR and returned 10 + 12
    # granules for the same overpasses.  Keying on tile let both survive and inflated
    # every count.  §29.3 warned about exactly this for TxSON (14RNU vs 14RMU).
    #
    # The duplicate tiles are NOT discarded outright: the station box may be clipped by
    # the edge of one tile and whole in the other, so alternates are carried on the kept
    # record and read_station_window falls back to them if the primary window is clipped.
    per_pass = {}
    for r in sorted(best.values(), key=lambda x: (x["orbit"], x["scene"], x["tile"])):
        k = (r["orbit"], r["scene"])
        if k not in per_pass:
            r["alt_urs"] = []
            per_pass[k] = r
        else:
            per_pass[k]["alt_urs"].append(r["granule_ur"])
    n_orbit = len(best) - len(per_pass)

    return sorted(per_pass.values(), key=lambda x: x["utc"]), n_reproc, n_orbit


# ============================================================
# PART B -- windowed reads
# ============================================================

def layer_url(ur: str, layer: str) -> str:
    return f"{LP_BASE}/{ur}/{ur}_{layer}.tif"


def decode_qc(qc: np.ndarray):
    """§36.10/§36.12.  QC is uint16, bit 0 least significant.

    bits 1&0 mandatory QA, 3&2 data quality, 15&14 LST accuracy.
    Bits 5&4 (Cloud/Ocean) are NOT SET in v002 -- cloud comes from the cloud layer, and
    the guide is explicit that QC 00 "may or may not be cloudy".
    """
    mand    = qc & 0b11
    dataq   = (qc >> 2) & 0b11
    lst_acc = (qc >> 14) & 0b11

    # MEASURED 2026-09-15, not assumed.  A probe of granule 16576 (TxSON, 2021-06-08)
    # returns lst_acc == 01 for 100% of pixels -- i.e. ECOSTRESS's actual delivered LST
    # accuracy here is "1.5-2 K, marginal".  The first version of this filter demanded
    # `>= 2` (<=1.5 K, "good"), which rejected EVERY pixel and produced clear_frac = 0.000
    # network-wide while looking like a cloud problem.  Require <=2 K, i.e. drop only the
    # `00` (">2 K, poor") bin.
    LST_ACC_MIN = 1

    # An ALL-ZERO QC word means the layer is UNPOPULATED, not that every field is at its
    # worst value.  Probe of granule 00375 (2018-07-30, three weeks post-launch) returns
    # a single distinct value 0x0000 over the whole window -- which would decode as "slow
    # convergence, warm humid air, silicate rocks, poor accuracy" for 100% of a Texas
    # rangeland tile.  Treat it as unknown: trust bits 1&0 (which read 00 = produced) and
    # do not let the unset accuracy field veto the pixel.
    unpopulated = (qc == 0)

    keep_qc = (np.isin(mand, [0, 1]) & (dataq == 0)
               & ((lst_acc >= LST_ACC_MIN) | unpopulated))
    return mand, dataq, lst_acc, keep_qc


def read_station_window(ur: str, lon: float, lat: float):
    """Windowed read of the mask layers over the 2.24 km station box.

    Opens QC first to get CRS/transform, reprojects the station point into it, then reads
    the identical window from every other layer.  Returns summary statistics only -- no
    arrays are retained.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    out = {"read_ok": 0, "n_px": 0, "n_px_expected": N_PX_EXPECTED, "window_frac": np.nan,
           "clipped": 0, "clear_frac": np.nan, "valid_frac": np.nan,
           "vza_mean_abs": np.nan, "vza_max_abs": np.nan,
           "frac_mand00": np.nan, "frac_mand01": np.nan, "frac_lstacc_ge2": np.nan,
           "frac_water": np.nan, "frac_cloud": np.nan, "passed_qc": 0, "error": ""}
    try:
        with rasterio.open(f"/vsicurl/{layer_url(ur, 'QC')}") as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
            half = TILE_M / 2.0
            win = from_bounds(cx - half, cy - half, cx + half, cy + half,
                              src.transform).round_offsets().round_lengths()
            qc = src.read(1, window=win)

        if qc.size == 0:
            out["error"] = "empty window -- station outside the tile footprint"
            return out

        # H5: rasterio's read(boundless=False) CLAMPS a window that overruns the raster --
        # windows.crop() does min/max against the bounds with no pad, no warning, no error.
        # A station within TILE_M/2 of an MGRS tile edge therefore gets a SMALLER array, and
        # clear_frac would be a fraction of the clipped area only: a granule covering 18% of
        # the station box could score clear_frac = 1.0 and passed_qc = 1.  Detect it.
        n = int(qc.size)
        out["n_px"] = n
        out["window_frac"] = n / float(N_PX_EXPECTED)
        if n < N_PX_EXPECTED:
            out["clipped"] = 1
            if out["window_frac"] < WINDOW_FRAC_MIN:
                out["error"] = (f"window clipped to {n}/{N_PX_EXPECTED} px "
                                f"({out['window_frac']:.2f}) -- station too near a tile edge")
                return out

        # The three remaining layers are three SEPARATE files, each needing its own TLS
        # open through the EDL redirect at ~3 s.  Read them concurrently: it does not
        # raise granules-per-second (we are connection-limited, not latency-limited), but
        # it cuts per-granule wall from ~12 s to ~3 s, which is what decides whether an
        # array task finishes inside its wall.
        bands, nodata, scales = {}, {}, {}

        def _read_layer(lyr):
            with rasterio.open(f"/vsicurl/{layer_url(ur, lyr)}") as src:
                # H4 / §36.15: fill and scale FROM THE COG, never assumed.
                return (lyr, src.read(1, window=win), src.nodata,
                        (src.scales[0] if src.scales else 1.0))

        with ThreadPoolExecutor(max_workers=3) as lp:
            for lyr, arr, nd, sc in lp.map(_read_layer, ("cloud", "water", "view_zenith")):
                bands[lyr], nodata[lyr], scales[lyr] = arr, nd, sc

        mand, dataq, lst_acc, keep_qc = decode_qc(qc)
        cloud = bands["cloud"]
        water = bands["water"]

        # view_zenith: apply the file's own scale, mask with the file's own nodata, and
        # take the ABSOLUTE angle -- VZA is signed to indicate side of nadir, so a raw
        # mean would let a downstream `vza_mean < 30` pass every west-side granule
        # however oblique (-55 < 30 is True).
        vza = bands["view_zenith"].astype("float64") * float(scales["view_zenith"] or 1.0)
        vmask = np.isfinite(vza)
        if nodata["view_zenith"] is not None:
            vmask &= (vza != float(nodata["view_zenith"]) * float(scales["view_zenith"] or 1.0))
        vza_abs = np.abs(vza[vmask])

        cloud_fill = 255 if nodata["cloud"] is None else int(nodata["cloud"])
        water_fill = 255 if nodata["water"] is None else int(nodata["water"])
        cloud_valid = cloud != cloud_fill
        water_valid = water != water_fill
        keep = keep_qc & (cloud == 0) & (water == 0) & cloud_valid & water_valid

        # clear_frac is scored against the EXPECTED window, not the clipped one, so a
        # partial read can never inflate it.
        denom = float(N_PX_EXPECTED)
        out.update({
            "read_ok": 1,
            "clear_frac": float(keep.sum()) / denom,
            "valid_frac": float((mand <= 1).sum()) / denom,     # separates cloud from no-data
            "vza_mean_abs": float(vza_abs.mean()) if vza_abs.size else np.nan,
            "vza_max_abs": float(vza_abs.max()) if vza_abs.size else np.nan,
            "frac_mand00": float((mand == 0).sum()) / denom,
            "frac_mand01": float((mand == 1).sum()) / denom,
            "frac_lstacc_ge2": float((lst_acc >= 2).sum()) / denom,
            "frac_water": float((water == 1).sum()) / denom,
            "frac_cloud": float((cloud == 1).sum()) / denom,
        })
        out["passed_qc"] = int(out["clear_frac"] >= CLEAR_FRAC_MIN)
    except Exception as exc:                                        # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ============================================================
# PART C -- pairing
# ============================================================

def pair_station(rows: list[dict], station_id: str, lon: float) -> list[dict]:
    """Pair each day pass with a night pass from THE NIGHT THAT FOLLOWS IT.

    The rule, stated the way it was specified: a 10:30 day acquisition may pair with a
    night acquisition from ~22:30 that evening through ~04:30 the next morning -- i.e.
    ONE nocturnal cooling period, the one that follows the observed heating.

    Implemented in solar geometry rather than clock hours so it is correct at every
    latitude and season:

        night_utc  >  day_utc                          the night FOLLOWS the day pass
        night_utc  <  solar_noon(day_solar_date + 1)   and is THAT night, not the next one
        day.elev   >  +10 deg,   night.elev  <  -5 deg  (set in classify_phase)

    The upper bound is next-day solar noon, not a fixed hour, so the admissible window
    stretches and shrinks with the season automatically.  A granule at 06:00 local with
    the sun still below the horizon is admitted; the following EVENING is not, because it
    lies past the next solar noon.

    Greedy one-to-one: a night granule is consumed by at most one day pass per arm, so
    counts are honest rather than inflated by reuse.  Where several nights qualify the
    LATEST is taken, not the nearest -- the later it is in the nocturnal period, the
    closer the surface is to its pre-dawn minimum and the larger the true diurnal range.

    Reported per pair so the phase spread is visible rather than assumed away:
        day_elev, night_elev   solar elevation of each half
        elev_drop              day_elev - night_elev, the illumination contrast
        dt_hours               separation
        well_phased            1 if the day half is within WELL_PHASED_DAY_H of the
                               thermal peak AND the night half is in the pre-dawn
                               approach; these are the pairs where DTR means one thing
    """
    days   = [r for r in rows if r["phase"] == "day"]
    nights = [r for r in rows if r["phase"] == "night"]
    out: list[dict] = []
    used: set[str] = set()

    for d in sorted(days, key=lambda r: r["utc"]):
        td    = pd.Timestamp(d["utc"])
        dsd   = d["solar_date"]
        limit = solar_noon_utc(dsd + pd.Timedelta(days=1), lon)

        best, best_dt = None, None
        for n in nights:
            if n["granule_ur"] in used:
                continue
            tn = pd.Timestamp(n["utc"])
            if tn <= td or tn >= limit:
                continue
            dt = (tn - td).total_seconds() / 3600.0
            # take the LATEST qualifying night: closest to the pre-dawn minimum
            if best_dt is None or dt > best_dt:
                best, best_dt = n, dt
        if best is None:
            continue
        used.add(best["granule_ur"])

        d_elev = float(d.get("solar_elev", float("nan")))
        n_elev = float(best.get("solar_elev", float("nan")))
        d_off  = abs(float(d.get("hours_from_solar_noon", 0.0)) - THERMAL_PEAK_LAG_H)
        n_tst  = float(best.get("tst", 0.0))
        well   = int(d_off <= WELL_PHASED_DAY_H and n_tst >= WELL_PHASED_NIGHT_TST)

        out.append({
            "station_id": station_id,
            "day_ur": d["granule_ur"], "night_ur": best["granule_ur"],
            "day_utc": d["utc"], "night_utc": best["utc"],
            "dt_hours": round(best_dt, 3),
            "day_tst": round(float(d.get("tst", float("nan"))), 3),
            "night_tst": round(n_tst, 3),
            "day_elev": round(d_elev, 2),
            "night_elev": round(n_elev, 2),
            "elev_drop": round(d_elev - n_elev, 2),
            "day_solar_date": str(dsd.date()),
            "night_solar_date": str(best["solar_date"].date()),
            "well_phased": well,
            "day_clear": d.get("clear_frac"),
            "night_clear": best.get("clear_frac"),
            "quality": int(bool(d.get("passed_qc")) and bool(best.get("passed_qc"))),
        })
    return out


# ============================================================
# SELFTEST
# ============================================================

def probe_granule(log, ur: str, lon: float = -98.78, lat: float = 30.30):
    """Dump the RAW distributions §36.12 asks for: 'histogram bits 15&14 before locking
    the threshold'.  Collapsing them to one boolean fraction is what made the
    clear_frac=0 result undiagnosable.

    Also reports each layer's dtype / nodata / scales straight from the COG, which is the
    §36.15 instruction ('never hardcode') and the only way to settle whether view_zenith
    is float-with-NaN or integer-with-fill.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    log.info("=" * 74)
    log.info("PROBE  %s", ur)
    log.info("=" * 74)

    with rasterio.open(f"/vsicurl/{layer_url(ur, 'QC')}") as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        half = TILE_M / 2.0
        win = from_bounds(xs[0] - half, ys[0] - half, xs[0] + half, ys[0] + half,
                          src.transform).round_offsets().round_lengths()
        qc = src.read(1, window=win)
        log.info("QC  dtype=%s nodata=%s scales=%s  shape=%s",
                 src.dtypes[0], src.nodata, src.scales, qc.shape)

    uniq, cnt = np.unique(qc, return_counts=True)
    log.info("QC unique values (%d distinct):", len(uniq))
    for v, c in sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda t: -t[1])[:12]:
        log.info("    0x%04X = %-6d  n=%4d (%.1f%%)  bits15..0 = %s",
                 v & 0xFFFF, v, c, 100.0 * c / qc.size, format(v & 0xFFFF, "016b"))

    fields = [("1&0   mandatory QA", qc & 0b11),
              ("3&2   data quality", (qc >> 2) & 0b11),
              ("5&4   cloud/ocean ", (qc >> 4) & 0b11),
              ("7&6   iterations  ", (qc >> 6) & 0b11),
              ("9&8   atmos opacity", (qc >> 8) & 0b11),
              ("11&10 MMD         ", (qc >> 10) & 0b11),
              ("13&12 emis accuracy", (qc >> 12) & 0b11),
              ("15&14 LST ACCURACY", (qc >> 14) & 0b11)]
    log.info("2-bit field histograms (fraction of %d px):", qc.size)
    log.info("    %-20s %8s %8s %8s %8s", "field", "00", "01", "10", "11")
    for name, arr in fields:
        log.info("    %-20s %8.3f %8.3f %8.3f %8.3f", name,
                 (arr == 0).mean(), (arr == 1).mean(),
                 (arr == 2).mean(), (arr == 3).mean())

    # The window is derived from QC's transform and then reused for every other layer.
    # That is only valid if all layers share an identical grid.  If view_zenith (say) is
    # written on a different extent, the same window object points somewhere else in the
    # file and returns nodata -- which looks exactly like "the layer is empty".  Check.
    with rasterio.open(f"/vsicurl/{layer_url(ur, 'QC')}") as src:
        ref = (src.transform, src.width, src.height, src.crs)
    log.info("GRID CHECK (all layers must match QC's grid for the shared window to be valid)")
    log.info("    %-12s %-9s %-24s %s", "layer", "WxH", "origin (x, y)", "res")
    for lyr in ("QC", "cloud", "water", "view_zenith", "LST"):
        with rasterio.open(f"/vsicurl/{layer_url(ur, lyr)}") as src:
            t = src.transform
            same = (t, src.width, src.height, src.crs) == ref
            log.info("    %-12s %4dx%-4d (%.1f, %.1f)  %.1f m  %s",
                     lyr, src.width, src.height, t.c, t.f, t.a,
                     "MATCH" if same else "*** DIFFERS FROM QC ***")

    for lyr in ("cloud", "water", "view_zenith"):
        with rasterio.open(f"/vsicurl/{layer_url(ur, lyr)}") as src:
            a = src.read(1, window=win)
            fin = np.isfinite(a.astype("float64"))
            log.info("%-12s dtype=%-8s nodata=%-8s scales=%s  finite=%.3f  min=%s max=%s",
                     lyr, src.dtypes[0], src.nodata, src.scales, fin.mean(),
                     (a[fin].min() if fin.any() else "n/a"),
                     (a[fin].max() if fin.any() else "n/a"))
            u, c = np.unique(a, return_counts=True)
            if len(u) <= 8:
                log.info("             values: %s",
                         {float(k): int(v) for k, v in zip(u.tolist(), c.tolist())})


def selftest(log):
    """Prove the mechanics on a handful of real granules before any scale is committed.

    CMR reachable -> ids parse -> solar geometry sane -> EDL auth works -> windowed read
    returns the expected shape -> QC decode is plausible.
    """
    log.info("=" * 74)
    log.info("SELFTEST -- TxSON CR200-18, lon=-98.78 lat=30.30, June 2021")
    log.info("=" * 74)

    session = requests.Session()
    items = cmr_granules(session, -98.78, 30.30, "2021-06-30T23:59:59Z")
    log.info("[1] CMR returned %d granules", len(items))

    recs, n_reproc, n_orbit = dedupe(items)
    log.info("[2] dedupe: %d -> %d overpasses  (reprocessing %d, same-orbit %d = %.1f%%)",
             len(items), len(recs), n_reproc, n_orbit,
             100.0 * n_orbit / len(items) if items else 0.0)

    log.info("[3] solar geometry + phase, first 10 overpasses:")
    log.info("      %-20s %-6s %7s %8s %8s  %s", "utc", "flag", "TST", "dt_noon", "elev", "phase")
    for r in recs[:10]:
        tst, dtn, elev, _, _ = solar_geometry(r["utc"], 30.30, -98.78)
        log.info("      %-20s %-6s %7.2f %8.2f %8.2f  %s",
                 r["utc"][:19], r["day_night_flag"], tst, dtn, elev, classify_phase(tst, elev))

    rows = []
    for r in recs:
        tst, dtn, elev, year, doy = solar_geometry(r["utc"], 30.30, -98.78)
        sd = solar_date(r["utc"], -98.78)
        rows.append({**r, "tst": tst, "hours_from_solar_noon": dtn, "solar_elev": elev,
                     "year": year, "doy": doy, "solar_date": sd,
                     "phase": classify_phase(tst, elev)})
    phases = {}
    for r in rows:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    log.info("[4] phase tally over all %d: %s", len(rows), phases)

    inwin = [r for r in rows if r["phase"] in ("day", "night")]
    demo = pair_station(inwin, "SELFTEST", -98.78)
    seen = {"pairs": len(demo), "well_phased": sum(p["well_phased"] for p in demo),
            "elev_drop_min": min((p["elev_drop"] for p in demo), default=0),
            "elev_drop_max": max((p["elev_drop"] for p in demo), default=0)}
    log.info("      %s", seen)
    log.info("      %-20s %-20s %7s %8s %8s %9s %s", "day_utc", "night_utc", "dt_h",
             "day_elev", "nt_elev", "elev_drop", "well")
    for p in demo[:10]:
        log.info("      %-20s %-20s %7.2f %8.1f %8.1f %9.1f %d",
                 p["day_utc"][:19], p["night_utc"][:19], p["dt_hours"],
                 p["day_elev"], p["night_elev"], p["elev_drop"], p["well_phased"])

    log.info("[5] windowed read + QC decode on up to 3 granules:")
    n_ok = 0
    for r in recs[:3]:
        t0 = time.time()
        res = read_station_window(r["granule_ur"], -98.78, 30.30)
        dt = time.time() - t0
        if res["read_ok"]:
            n_ok += 1
            log.info("      OK  %s", r["granule_ur"])
            log.info("          %.1fs  %d/%d px (window_frac %.2f, clipped=%d)",
                     dt, res["n_px"], res["n_px_expected"], res["window_frac"],
                     res["clipped"])
            log.info("          clear=%.3f  valid=%.3f  |vza| mean=%.1f max=%.1f deg",
                     res["clear_frac"], res["valid_frac"],
                     res["vza_mean_abs"], res["vza_max_abs"])
            log.info("          mand00=%.3f mand01=%.3f lstacc>=2=%.3f cloud=%.3f water=%.3f",
                     res["frac_mand00"], res["frac_mand01"], res["frac_lstacc_ge2"],
                     res["frac_cloud"], res["frac_water"])
        else:
            log.error("      FAIL %s -- %s", r["granule_ur"], res["error"])

    if n_ok == 0:
        log.error("SELFTEST FAILED: no granule could be read. Check ~/.netrc and that BOTH "
                  "'LP DAAC Data Pool' AND 'LP DAAC Cumulus (LPCLOUD)' are authorised at "
                  "urs.earthdata.nasa.gov -- a missing app authorisation 403s with valid "
                  "credentials.")
        raise SystemExit(1)

    exp = int(TILE_M // 70)
    log.info("[6] expected window ~%d x %d px at 70 m = ~%d px", exp, exp, exp * exp)
    log.info("SELFTEST PASSED -- %d/%d granules read", n_ok, min(3, len(recs)))
    log.info("")
    log.info("EYEBALL THESE BEFORE TRUSTING A LARGER RUN:")
    log.info("  * TST spans 0-24; day-phase near 12.5-15.5, night-phase near 22-02")
    log.info("  * elev POSITIVE for Day-flagged rows, NEGATIVE for Night-flagged")
    log.info("  * same-orbit dedup near 25%% (§36.20)")
    log.info("  * clear_frac must NOT be 0.000 or 1.000 on every granule -- that would")
    log.info("    mean the mask is inverted or the bit decode is wrong")
    log.info("  * in [4b], next_day_only=True must give FEWER pairs than False at the same")
    log.info("    tolerance, and pair counts must rise monotonically with dt_tol (§36.20)")
    log.info("  * every next_day_only=True pair must have night_solar_date exactly one day")
    log.info("    after day_solar_date")


# ============================================================
# MAIN
# ============================================================

def load_stations(args) -> pd.DataFrame:
    # pandas only -- 6 AmeriFlux rows have quoted commas in station_name, and awk -F,
    # silently shifts every later field.
    df = pd.read_csv(STATION_CSV)
    df["abs_lat"] = df["latitude"].abs()

    if args.controls:
        # Do NOT just sample above the assumed limit -- that only confirms the assumption.
        # Sweep a band straddling it so the data shows where coverage actually stops.
        # The prose figure is 52; the collection metadata says 54; 38 stations sit between.
        band = df[(df["abs_lat"] >= CONTROL_LO) & (df["abs_lat"] <= CONTROL_HI)]
        return band.sort_values("abs_lat").groupby(
            pd.cut(band["abs_lat"], bins=np.arange(CONTROL_LO, CONTROL_HI + 2, 2.0)),
            observed=True).head(2)

    if args.station:
        return df[df["station_id"].astype(str) == args.station]

    df = df[df["abs_lat"] <= LAT_LIMIT].reset_index(drop=True)

    # SLURM array slice.  Taken AFTER the latitude filter and AFTER reset_index, so the
    # slice indices are stable and contiguous over the in-range set regardless of where
    # the excluded stations sat in the original file.
    if args.start_idx >= 0:
        lo = max(0, args.start_idx)
        hi = len(df) if args.end_idx < 0 else min(len(df), args.end_idx)
        return df.iloc[lo:hi]

    return df.head(args.sample) if args.sample else df


def main():
    ap = argparse.ArgumentParser(description="§36 Tier 1 ECOSTRESS day/night pair census")
    ap.add_argument("--selftest", action="store_true", help="~3 granules, verify mechanics")
    ap.add_argument("--probe", type=str, default="",
                    help="dump raw QC bit histograms + layer dtypes for ONE granule_ur")
    ap.add_argument("--dry-run", action="store_true",
                    help="part A only: CMR + solar geometry, no pixels, no EDL")
    ap.add_argument("--sample", type=int, default=0, help="first N in-range stations")
    ap.add_argument("--station", type=str, default="", help="one station_id")
    ap.add_argument("--controls", action="store_true",
                    help=">52 deg stations; MUST return zero granules (§36.20)")
    ap.add_argument("--workers", type=int, default=HTTP_WORKERS)
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint")
    ap.add_argument("--start-idx", type=int, default=-1,
                    help="SLURM array slice: first station row (inclusive)")
    ap.add_argument("--end-idx", type=int, default=-1,
                    help="SLURM array slice: last station row (exclusive)")
    ap.add_argument("--all-phases", action="store_true",
                    help="read masks for EVERY granule, not just day/night-window ones. "
                         "Required to sweep the window half-widths later (§36.14), but "
                         "~3x the reads. Default off.")
    args = ap.parse_args()

    # Per-task output files.  44 array tasks appending to one CSV would interleave and
    # corrupt it -- csv.DictWriter gives no atomicity across processes.  Merge afterwards
    # with slurm/ecostress_census_array.sh merge.
    global OUT_GRAN, OUT_PAIRS, LOG_FILE
    if args.start_idx >= 0:
        tag = f".{args.start_idx:05d}_{args.end_idx:05d}"
        OUT_GRAN  = OUT_GRAN.with_suffix(f"{tag}.csv")
        OUT_PAIRS = OUT_PAIRS.with_suffix(f"{tag}.csv")
        LOG_FILE  = LOG_FILE.with_suffix(f"{tag}.csv")

    setup_logging("ecostress_census")
    log = logging.getLogger(__name__)
    configure_gdal()

    if args.probe:
        probe_granule(log, args.probe)
        return

    if args.selftest:
        selftest(log)
        return

    if args.fresh:
        truncate_outputs(log)
    stations = load_stations(args)
    done = set() if args.fresh else load_done()
    todo = [r for _, r in stations.iterrows() if str(r["station_id"]) not in done]
    log.info("stations=%d  done=%d  todo=%d  workers=%d  dry_run=%s",
             len(stations), len(done), len(todo), args.workers, args.dry_run)

    if args.controls and not todo:
        log.error("CONTROL VACUOUS: no stations to test -- every control is already marked "
                  "'ok' in the checkpoint, so a PASS here would mean nothing. Re-run with "
                  "--fresh.")
        raise SystemExit(1)

    session = requests.Session()
    session.headers.update({"User-Agent": "soilMoisture-census/1.0 (runbook 36)"})
    totals = {"hits": 0, "overpasses": 0, "dupe_orbit": 0, "dupe_reproc": 0, "inwindow": 0,
              "read_ok": 0, "passed": 0, "pairs": 0, "pairs_loose": 0}

    def work(row):
        sid = str(row["station_id"])
        meta = {"station_id": sid, "network": row.get("network", ""),
                "lat": float(row["latitude"]), "lon": float(row["longitude"]),
                "elevation_m": row.get("elevation_m", ""),
                "kg_macro": row.get("kg_macro", "")}
        # Query the STATION's own window, not the whole mission.  A station whose record
        # begins in 2020 gains nothing from 2018-2019 granules -- there is no label to
        # pair them against -- and fetching them is pure cost.  Clamp to mission start.
        end = iso_end_date(row.get("end_date"))
        start = iso_end_date(row.get("actual_start_date") or row.get("start_date"))
        start = max(start, MISSION_START[:10])
        try:
            items = cmr_granules(session, meta["lon"], meta["lat"],
                                 f"{end}T23:59:59Z", start=f"{start}T00:00:00Z")
            recs, n_reproc, n_orbit = dedupe(items)

            rows = []
            for r in recs:
                tst, dtn, elev, year, doy = solar_geometry(r["utc"], meta["lat"], meta["lon"])
                sd = solar_date(r["utc"], meta["lon"])
                rows.append({**meta, **r, "tst": round(tst, 4),
                             "hours_from_solar_noon": round(dtn, 4),
                             "solar_elev": round(elev, 3), "year": year, "doy": doy,
                             "solar_date": sd, "solar_date_str": str(sd.date()),
                             "phase": classify_phase(tst, elev)})

            inwin = rows if args.all_phases else [r for r in rows if r["phase"] in ("day", "night")]
            if not args.dry_run:
                for r in inwin:
                    # Try the primary tile; if the station box is clipped by a tile edge,
                    # fall back to the alternate tile covering the same overpass.
                    res = read_station_window(r["granule_ur"], meta["lon"], meta["lat"])
                    for alt in r.get("alt_urs", []):
                        if res.get("read_ok") and not res.get("clipped"):
                            break
                        alt_res = read_station_window(alt, meta["lon"], meta["lat"])
                        if alt_res.get("read_ok") and (
                                (alt_res.get("window_frac") or 0) > (res.get("window_frac") or 0)):
                            alt_res["granule_ur_used"] = alt
                            res = alt_res
                    r.update(res)
            pairs = [] if args.dry_run else pair_station(inwin, sid, meta["lon"])
            return sid, rows, inwin, pairs, len(items), n_reproc, n_orbit, None
        except Exception as exc:                                    # noqa: BLE001
            return sid, [], [], [], 0, 0, 0, f"{type(exc).__name__}: {exc}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (sid, rows, inwin, pairs, n_hits, n_rep, n_orb, err) in enumerate(
                pool.map(work, todo), 1):
            if err:
                log.warning("[%d/%d] %s FAILED -- %s", i, len(todo), sid, err)
                append_rows(LOG_FILE, LOG_COLS, [{
                    "station_id": sid, "status": "error", "error": err,
                    "timestamp": pd.Timestamp.utcnow().isoformat()}])
                continue

            append_rows(OUT_GRAN, GRAN_COLS, rows)
            append_rows(OUT_PAIRS, PAIR_COLS, pairs)
            n_pass = sum(int(r.get("passed_qc", 0)) for r in inwin)
            n_read = sum(int(r.get("read_ok", 0)) for r in inwin)
            # pairs = one nocturnal period per day pass.  "quality" = both halves passed
            # image QC.  "well_phased" = day half near the thermal peak AND night half in
            # the pre-dawn approach -- reported separately so the cost of insisting on
            # comparability is visible rather than assumed.
            n_head  = sum(1 for p in pairs if p["quality"] == 1)
            n_loose = sum(1 for p in pairs if p["quality"] == 1 and p["well_phased"] == 1)

            append_rows(LOG_FILE, LOG_COLS, [{
                "station_id": sid,
                "status": "dryrun" if args.dry_run else "ok",
                "n_hits": n_hits,
                "n_overpasses": len(rows), "n_dupe_orbit": n_orb, "n_dupe_reproc": n_rep,
                "n_inwindow": len(inwin), "n_read_ok": n_read, "n_passed": n_pass,
                "n_pairs_quality": n_head, "n_pairs_well_phased": n_loose, "error": "",
                "timestamp": pd.Timestamp.utcnow().isoformat()}])

            for k, v in (("hits", n_hits), ("overpasses", len(rows)),
                         ("dupe_orbit", n_orb), ("dupe_reproc", n_rep),
                         ("inwindow", len(inwin)), ("read_ok", n_read),
                         ("passed", n_pass), ("pairs", n_head), ("pairs_loose", n_loose)):
                totals[k] += v

            if i % 10 == 0 or i == len(todo):
                log.info("[%d/%d] %s  overpasses=%d in-window=%d read=%d passed=%d "
                         "pairs_hl=%d  (%.1f min)", i, len(todo), sid,
                         totals["overpasses"], totals["inwindow"], totals["read_ok"],
                         totals["passed"], totals["pairs"], (time.time() - t0) / 60.0)

    h = totals["hits"]
    log.info("DONE in %.1f min", (time.time() - t0) / 60.0)
    log.info("granules returned : %d", h)
    log.info("unique overpasses : %d", totals["overpasses"])
    # M5: the §36.20 reference (236 -> 178 = 24.6%) is TOTAL drop against RAW granules.
    # n_dupe_orbit alone is measured after reprocessing dedup, so dividing it by raw hits
    # understates the rate whenever both reprocessing builds are in CMR.  Report the
    # comparable total, and the two components against their own bases.
    total_drop = h - totals["overpasses"]
    log.info("dropped, TOTAL    : %d  (%.1f%% of raw -- §36.20 expects ~25%%)",
             total_drop, 100.0 * total_drop / h if h else 0.0)
    log.info("  of which reproc : %d", totals["dupe_reproc"])
    log.info("  of which orbit  : %d", totals["dupe_orbit"])
    log.info("in a day/night win: %d", totals["inwindow"])
    log.info("mask read ok      : %d", totals["read_ok"])
    log.info("passed image QC   : %d", totals["passed"])
    log.info("-" * 60)
    log.info("QUALITY PAIRS (both halves pass image QC)   : %d", totals["pairs"])
    log.info("  of which WELL-PHASED (day near thermal    : %d", totals["pairs_loose"])
    log.info("   peak AND night in the pre-dawn approach)")
    if totals["pairs"]:
        log.info("  -> insisting on well-phased costs %d pairs (%.1f%%). Per-pair day_elev,"
                 " night_elev and elev_drop are in %s so the tradeoff is inspectable.",
                 totals["pairs"] - totals["pairs_loose"],
                 100.0 * (totals["pairs"] - totals["pairs_loose"]) / totals["pairs"],
                 OUT_PAIRS.name)
    log.info("-" * 60)

    if args.controls:
        # This is a MEASUREMENT, not a pass/fail on an assumed limit.  Report where
        # coverage actually stops so LAT_LIMIT can be set from data.
        errs = 0
        if LOG_FILE.exists():
            dfl = pd.read_csv(LOG_FILE, dtype=str)
            errs = int((dfl["status"] == "error").sum())
        if errs:
            log.error("CONTROL INVALID: %d control stations ERRORED. Zero granules from a "
                      "failed query is indistinguishable from zero granules from no "
                      "coverage -- fix the errors before reading anything into this.", errs)
            raise SystemExit(1)
        log.info("CONTROL RESULT -- coverage by latitude band (prose says %.1f, collection "
                 "metadata says %.1f):", LAT_NOMINAL, LAT_LIMIT)
        if OUT_GRAN.exists():
            g = pd.read_csv(OUT_GRAN)
            if len(g):
                g["abs_lat"] = g["lat"].abs()
                g["band"] = (g["abs_lat"] // 2 * 2).astype(int)
                for b, sub in g.groupby("band"):
                    log.info("    %2d-%2d deg : %5d granules across %d stations",
                             b, b + 2, len(sub), sub["station_id"].nunique())
        log.info("Set LAT_LIMIT to the highest band that still returns granules.")


if __name__ == "__main__":
    main()
