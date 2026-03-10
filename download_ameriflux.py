"""
Download and filter AmeriFlux FLUXNET flux data.

Mirrors the workflow of the R `amerifluxr` package:
  amf_site_info()          -> get_site_info()
  amf_download_fluxnet()   -> download_fluxnet()

AmeriFlux account required: https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx
Data policy: https://ameriflux.lbl.gov/data/data-policy/

FLUXNET vs BASE:
  - FLUXNET: gap-filled, quality-flagged, daily/monthly aggregates (NEE, H, LE, etc.)
  - BASE:    raw half-hourly observations
  - Not all sites with BASE data have FLUXNET data — filter on fluxnet_data years
  - FLUXNET variants: "SUBSET" (key variables) or "FULLSET" (all variables)
"""

import json
import os
import re
import time
import zipfile
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# AmeriFlux endpoints
# ---------------------------------------------------------------------------
# Site info is embedded as JS in the site-search page (no standalone REST API)
SITE_SEARCH_PAGE = "https://ameriflux.lbl.gov/sites/site-search/"
DOWNLOAD_URL     = "https://amfcdn.lbl.gov/api/v1/data_download"

# ---------------------------------------------------------------------------
# Credentials & request metadata — loaded from .env (never commit credentials)
# Create a .env file in this directory with:
#   AMERIFLUX_USER_ID=your_username
#   AMERIFLUX_USER_EMAIL=your_email@example.com
# ---------------------------------------------------------------------------
USER_ID   = os.environ.get("AMERIFLUX_USER_ID", "")
USER_EMAIL = os.environ.get("AMERIFLUX_USER_EMAIL", "")

if not USER_ID or not USER_EMAIL:
    raise EnvironmentError(
        "AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL must be set in .env or environment."
    )

DATA_POLICY        = "CC-BY-4.0"    # "CC-BY-4.0" or "Legacy"
INTENDED_USE       = "remote_sensing"  # synthesis | model | remote_sensing | other_research | education | other
INTENDED_USE_TEXT  = "PhD research: multimodal soil moisture prediction"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path("/home/khanalp/data/ameriflux_raw")


def get_site_info() -> pd.DataFrame:
    """
    Fetch the full AmeriFlux site list by scraping the site-search page.

    AmeriFlux embeds all site data as a JSON string in the page HTML
    (variable `jsonSites`), so we parse it from there.

    Returns a DataFrame with columns matching the embedded JSON fields, plus
    convenience aliases: SITE_ID, SITE_NAME, IGBP, LOCATION_LAT,
    LOCATION_LONG, LOCATION_ELEV, DATA_POLICY, TOWER_BEGAN, TOWER_END,
    DATA_START, DATA_END, MAT, MAP, CLIMATE_KOEPPEN.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(SITE_SEARCH_PAGE, headers=headers, timeout=30)
    resp.raise_for_status()

    # The page embeds: const jsonSites = '[{...}]'
    # Keys/values are escaped with \" inside the JS string, so unescape first.
    match = re.search(r"const\s+jsonSites\s*=\s*'(\[.*?\])'", resp.text, re.DOTALL)
    if not match:
        match = re.search(r'const\s+jsonSites\s*=\s*"(\[.*?\])"', resp.text, re.DOTALL)
    if not match:
        raise RuntimeError(
            "Could not find jsonSites in the AmeriFlux site-search page. "
            "The page structure may have changed."
        )

    # The JS single-quoted string escapes " as \" and ' as \'.
    # Use negative lookbehind so we don't corrupt \\\" (literal backslash + quote).
    raw_js = match.group(1)
    raw_json = re.sub(r'(?<!\\)\\"', '"', raw_js)
    raw_json = re.sub(r"(?<!\\)\\'", "'", raw_json)
    sites = pd.DataFrame(json.loads(raw_json))

    # Rename to consistent uppercase aliases used elsewhere in this script
    rename = {
        "site_id":   "SITE_ID",
        "site_name": "SITE_NAME",
        "igbp":      "IGBP",
        "country":   "COUNTRY",
        "state":     "STATE",
        "latitude":  "LOCATION_LAT",
        "longitude": "LOCATION_LONG",
        "elevation": "LOCATION_ELEV",
        "data_policy": "DATA_POLICY",
        "tower_start": "TOWER_BEGAN",
        "tower_end":   "TOWER_END",
        "mat":       "MAT",
        "map":       "MAP",
        "koeppen":   "CLIMATE_KOEPPEN",
    }
    sites = sites.rename(columns={k: v for k, v in rename.items() if k in sites.columns})

    def _year_range(val):
        years = [int(y) for y in (val if isinstance(val, list) else []) if str(y).isdigit()]
        return (min(years), max(years)) if years else (None, None)

    # Derive DATA_START / DATA_END from base_data years
    if "base_data" in sites.columns and "DATA_START" not in sites.columns:
        sites[["DATA_START", "DATA_END"]] = sites["base_data"].apply(
            lambda v: pd.Series(_year_range(v))
        )

    # Derive FLUXNET_START / FLUXNET_END from fluxnet_data years
    if "fluxnet_data" in sites.columns:
        sites[["FLUXNET_START", "FLUXNET_END"]] = sites["fluxnet_data"].apply(
            lambda v: pd.Series(_year_range(v))
        )
        for col in ["FLUXNET_START", "FLUXNET_END"]:
            sites[col] = pd.to_numeric(sites[col], errors="coerce")

    # Coerce numeric columns
    for col in ["LOCATION_LAT", "LOCATION_LONG", "LOCATION_ELEV",
                "TOWER_BEGAN", "TOWER_END", "DATA_START", "DATA_END",
                "MAT", "MAP"]:
        if col in sites.columns:
            sites[col] = pd.to_numeric(sites[col], errors="coerce")

    return sites


def filter_sites(
    sites: pd.DataFrame,
    igbp: list[str] | None = None,
    country: list[str] | None = None,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    data_policy: list[str] | None = None,
    min_years: int | None = None,
    min_fluxnet_years: int | None = None,
    fluxnet_year_range: tuple[int, int] | None = None,
    fluxnet_only: bool = True,
    active_only: bool = False,
) -> pd.DataFrame:
    """
    Filter the site list.

    Parameters
    ----------
    igbp               : IGBP vegetation types to keep, e.g. ["CRO", "GRA", "DBF"]
    country            : Full country name, e.g. ["USA", "Canada"]
    lat_range          : (min_lat, max_lat) in decimal degrees
    lon_range          : (min_lon, max_lon) in decimal degrees
    data_policy        : ["CC-BY-4.0"] or ["Legacy"] or both
    min_years          : Minimum years of published BASE data (DATA_END - DATA_START)
    min_fluxnet_years  : Minimum number of FLUXNET years within fluxnet_year_range
                         (or overall if fluxnet_year_range is None)
    fluxnet_year_range : (start_year, end_year) inclusive — count only FLUXNET years
                         that fall within this window; combined with min_fluxnet_years
                         e.g. (2015, 2025) with min_fluxnet_years=2 keeps sites that
                         have at least 2 years of data anywhere in 2015–2025
    fluxnet_only       : If True (default), keep only sites that have FLUXNET data
    active_only        : If True, keep only currently active towers (TOWER_END is NaN)
    """
    mask = pd.Series(True, index=sites.index)

    if igbp:
        mask &= sites["IGBP"].isin(igbp)
    if country:
        mask &= sites["COUNTRY"].isin(country)
    if lat_range:
        mask &= sites["LOCATION_LAT"].between(*lat_range)
    if lon_range:
        mask &= sites["LOCATION_LONG"].between(*lon_range)
    if data_policy:
        mask &= sites["DATA_POLICY"].isin(data_policy)
    if min_years is not None:
        duration = sites["DATA_END"] - sites["DATA_START"]
        mask &= duration >= min_years
    if fluxnet_only and "FLUXNET_START" in sites.columns:
        mask &= sites["FLUXNET_START"].notna()

    if "fluxnet_data" in sites.columns and (fluxnet_year_range is not None or min_fluxnet_years is not None):
        yr_min, yr_max = fluxnet_year_range if fluxnet_year_range else (0, 9999)

        def _count_years_in_range(val):
            years = [int(y) for y in (val if isinstance(val, list) else []) if str(y).isdigit()]
            return sum(yr_min <= y <= yr_max for y in years)

        year_counts = sites["fluxnet_data"].apply(_count_years_in_range)

        if fluxnet_year_range is not None:
            mask &= year_counts > 0          # at least 1 year in the window
        if min_fluxnet_years is not None:
            mask &= year_counts >= min_fluxnet_years

    if active_only:
        mask &= sites["TOWER_END"].isna()

    return sites[mask].reset_index(drop=True)


def download_fluxnet(
    site_ids: list[str],
    out_dir: Path = OUT_DIR,
    data_variant: str = "FULLSET",
    data_policy: str = "CCBY4.0",
    intended_use: str = INTENDED_USE,
    intended_use_text: str = INTENDED_USE_TEXT,
    user_id: str = USER_ID,
    user_email: str = USER_EMAIL,
    batch_size: int = 10,
) -> list[Path]:
    """
    Download AmeriFlux FLUXNET zip files via the amfcdn API.

    Uses data_product="FLUXNET" (the current AmeriFlux FLUXNET product,
    covering 2015-onward data). Sites with no available data are reported.

    Parameters
    ----------
    site_ids     : List of AmeriFlux site IDs, e.g. ["US-Var", "US-Ton"]
    out_dir      : Directory to save downloaded zip files
    data_variant : "FULLSET" (all variables) or "SUBSET" (key variables)
    data_policy  : "CCBY4.0" or "LEGACY"
    batch_size   : Sites per API request (large batches may time out)

    Returns
    -------
    List of paths to downloaded zip files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    n_batches = (len(site_ids) - 1) // batch_size + 1

    for i in range(0, len(site_ids), batch_size):
        batch = site_ids[i : i + batch_size]
        batch_num = i // batch_size + 1
        payload = {
            "user_id":      user_id,
            "user_email":   user_email,
            "site_ids":     batch,
            "data_product": "FLUXNET",
            "data_variant": data_variant,
            "data_policy":  data_policy,
            "intended_use": intended_use,
            "description":  intended_use_text,
        }

        print(f"Batch {batch_num}/{n_batches}: {batch}")
        try:
            resp = requests.post(DOWNLOAD_URL, json=payload,
                                 headers={"Content-Type": "application/json"}, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"  API ERROR: {e}")
            continue

        data_urls = resp.json().get("data_urls", [])
        found_ids = {e["site_id"] for e in data_urls}
        missing = [s for s in batch if s not in found_ids]
        if missing:
            print(f"  No data available for: {missing}")

        for entry in data_urls:
            site_id  = entry["site_id"]
            url      = entry["url"]
            size_mb  = entry.get("download_size", 0) / 1024 / 1024
            out_path = out_dir / f"{site_id}_FLUXNET_{data_variant}.zip"

            print(f"  Downloading {site_id} ({size_mb:.1f} MB) -> {out_path.name}")
            try:
                dl = requests.get(url, timeout=600, stream=True)
                dl.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                downloaded.append(out_path)
                print(f"    Saved.")
            except requests.HTTPError as e:
                print(f"    DOWNLOAD ERROR: {e}")

    return downloaded


def download_badm(
    site_ids: list[str],
    out_dir: Path = OUT_DIR,
    data_policy: str = "CCBY4.0",
    intended_use: str = INTENDED_USE,
    intended_use_text: str = INTENDED_USE_TEXT,
    user_id: str = USER_ID,
    user_email: str = USER_EMAIL,
    batch_size: int = 10,
) -> list[Path]:
    """
    Download AmeriFlux BASE-BADM zip files (contains BIF Excel + BASE CSV).

    Each zip includes:
      - AMF_{SITE_ID}_BIF_{date}.xlsx  — BADM metadata (sensor depths, canopy height, ...)
      - AMF_{SITE_ID}_BASE_HH_*.csv    — raw half-hourly flux data

    Parameters
    ----------
    site_ids   : List of AmeriFlux site IDs
    out_dir    : Directory to save downloaded zip files
    batch_size : Sites per API request

    Returns
    -------
    List of paths to downloaded zip files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    n_batches = (len(site_ids) - 1) // batch_size + 1

    for i in range(0, len(site_ids), batch_size):
        batch = site_ids[i : i + batch_size]
        batch_num = i // batch_size + 1
        payload = {
            "user_id":      user_id,
            "user_email":   user_email,
            "site_ids":     batch,
            "data_product": "BASE-BADM",
            "data_policy":  data_policy,
            "intended_use": intended_use,
            "description":  intended_use_text,
        }

        print(f"Batch {batch_num}/{n_batches}: {batch}")
        try:
            resp = requests.post(DOWNLOAD_URL, json=payload,
                                 headers={"Content-Type": "application/json"}, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"  API ERROR: {e}")
            continue

        data_urls = resp.json().get("data_urls", [])
        found_ids = {e["site_id"] for e in data_urls}
        missing = [s for s in batch if s not in found_ids]
        if missing:
            print(f"  No data available for: {missing}")

        for entry in data_urls:
            site_id  = entry["site_id"]
            url      = entry["url"]
            size_mb  = entry.get("download_size", 0) / 1024 / 1024
            out_path = out_dir / f"{site_id}_BASE-BADM.zip"

            print(f"  Downloading {site_id} ({size_mb:.1f} MB) -> {out_path.name}")
            try:
                dl = requests.get(url, timeout=600, stream=True)
                dl.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                downloaded.append(out_path)
                print(f"    Saved.")
            except requests.HTTPError as e:
                print(f"    DOWNLOAD ERROR: {e}")

    return downloaded


def _parse_bif(zip_path: Path) -> dict[str, dict]:
    """
    Parse the BIF (BADM Interchange Format) Excel inside a BASE-BADM zip.

    The BIF has a single sheet 'AMF-BIF' with columns:
      SITE_ID, GROUP_ID, VARIABLE_GROUP, VARIABLE, DATAVALUE

    Returns a dict mapping VARIABLE_GROUP -> list of {VARIABLE: DATAVALUE} dicts
    (one dict per GROUP_ID, i.e. one per sensor installation / measurement event).
    """
    import zipfile, openpyxl
    from collections import defaultdict

    with zipfile.ZipFile(zip_path) as zf:
        bif_names = [n for n in zf.namelist() if n.endswith(".xlsx")]
        if not bif_names:
            return {}
        with zf.open(bif_names[0]) as f:
            wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
            ws = wb["AMF-BIF"]
            rows = list(ws.iter_rows(values_only=True))

    if not rows:
        return {}

    hdr = rows[0]  # (SITE_ID, GROUP_ID, VARIABLE_GROUP, VARIABLE, DATAVALUE)
    raw = [dict(zip(hdr, r)) for r in rows[1:]]

    # Pivot: group_key = (VARIABLE_GROUP, GROUP_ID) -> {VARIABLE: DATAVALUE}
    pivoted: dict[tuple, dict] = defaultdict(dict)
    for r in raw:
        key = (r["VARIABLE_GROUP"], r["GROUP_ID"])
        pivoted[key][r["VARIABLE"]] = r["DATAVALUE"]

    # Reorganise by VARIABLE_GROUP -> list of pivoted dicts
    by_group: dict[str, list[dict]] = defaultdict(list)
    for (vgrp, _), d in pivoted.items():
        by_group[vgrp].append(d)

    return dict(by_group)


def extract_badm_metadata(zip_paths: list[Path]) -> pd.DataFrame:
    """
    Extract key ancillary metadata from BASE-BADM zip files.

    For each site, returns:
      SITE_ID              — AmeriFlux site ID
      SWC_DEPTHS_CM        — sorted list of unique sensor depth ranges as
                             "MIN-MAX" strings (cm), e.g. ["0-15", "15-30"]
      CANOPY_HEIGHT_M      — most recent mean canopy/vegetation height (m),
                             or NaN if not reported
      CANOPY_HEIGHT_DATE   — date of that measurement (YYYYMMDD string)
      CANOPY_HEIGHT_VEGTYPE— vegetation type label for the canopy measurement

    Notes
    -----
    EC measurement height is not stored in BADM for most sites and is therefore
    not included here. It can often be inferred as 1.5-2× canopy height (forests)
    or found in site publications.

    Parameters
    ----------
    zip_paths : Paths to downloaded BASE-BADM zip files.

    Returns
    -------
    DataFrame with one row per site.
    """
    records = []

    for zp in zip_paths:
        site_id = zp.name.split("_")[0]  # e.g. "US-Var" from "US-Var_BASE-BADM.zip"
        by_group = _parse_bif(zp)

        # ── SM sensor depth ranges ────────────────────────────────────────────
        depth_ranges: set[str] = set()
        for g in by_group.get("GRP_SWC", []):
            d_min = g.get("SWC_PROFILE_MIN")
            d_max = g.get("SWC_PROFILE_MAX")
            if d_min is not None and d_max is not None:
                try:
                    depth_ranges.add(f"{int(float(d_min))}-{int(float(d_max))}")
                except (ValueError, TypeError):
                    pass
        swc_depths = sorted(depth_ranges, key=lambda s: int(s.split("-")[0]))

        # ── Canopy height — most recent Mean entry ────────────────────────────
        canopy_h = float("nan")
        canopy_date = None
        canopy_vegtype = None

        mean_entries = [
            g for g in by_group.get("GRP_HEIGHTC", [])
            if g.get("HEIGHTC_STATISTIC", "").lower() == "mean"
            and g.get("HEIGHTC") is not None
        ]
        if mean_entries:
            # Sort by date descending; entries without a date go last
            def _hc_date(g):
                try:
                    return int(g.get("HEIGHTC_DATE") or 0)
                except (ValueError, TypeError):
                    return 0

            most_recent = max(mean_entries, key=_hc_date)
            try:
                canopy_h = float(most_recent["HEIGHTC"])
            except (ValueError, TypeError):
                pass
            canopy_date = most_recent.get("HEIGHTC_DATE")
            canopy_vegtype = most_recent.get("HEIGHTC_VEGTYPE")

        records.append({
            "SITE_ID":               site_id,
            "SWC_DEPTHS_CM":         swc_depths,
            "CANOPY_HEIGHT_M":       canopy_h,
            "CANOPY_HEIGHT_DATE":    canopy_date,
            "CANOPY_HEIGHT_VEGTYPE": canopy_vegtype,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("SITE_ID").reset_index(drop=True)
    return df


def extract_fluxnet(zip_paths: list[Path], out_dir: Path | None = None) -> dict[str, list[Path]]:
    """
    Extract downloaded FLUXNET zip files.

    Each zip contains multiple CSVs at different temporal resolutions:
      HH  — half-hourly
      DD  — daily
      WW  — weekly
      MM  — monthly
      YY  — yearly

    Returns a dict mapping resolution code -> list of CSV paths.
    """
    by_res: dict[str, list[Path]] = {}
    for zp in zip_paths:
        dest = out_dir or zp.parent / zp.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                # FLUXNET filename pattern: AMF_CC-Sss_FLUXNET_SUBSET_DD_YYYY-YYYY_*.csv
                parts = Path(name).stem.split("_")
                res = next((p for p in parts if p in ("HH", "DD", "WW", "MM", "YY")), "OTHER")
                by_res.setdefault(res, []).append(dest / name)
    return by_res


def read_fluxnet(csv_path: Path, resolution: str = "DD") -> pd.DataFrame:
    """
    Read an extracted AmeriFlux FLUXNET CSV file.

    Parameters
    ----------
    csv_path   : Path to the CSV file
    resolution : Expected temporal resolution for timestamp parsing:
                 "HH" (half-hourly, YYYYMMDDHHMM),
                 "DD" (daily, YYYYMMDD),
                 "WW"/"MM"/"YY" (weekly/monthly/yearly, YYYYMMDD start)

    FLUXNET CSVs have a single header row followed by data rows.
    Key daily variables: NEE_VUT_REF, H_F_MDS, LE_F_MDS, SW_IN_F, TA_F, P_F,
                         SWC_F_MDS_* (soil moisture at various depths), TIMESTAMP.
    """
    df = pd.read_csv(csv_path, na_values=[-9999, -9999.0])

    # Parse TIMESTAMP — format depends on resolution
    if "TIMESTAMP" in df.columns:
        ts_fmt = "%Y%m%d%H%M" if resolution == "HH" else "%Y%m%d"
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"].astype(str), format=ts_fmt)
        df = df.set_index("TIMESTAMP").sort_index()
    elif "TIMESTAMP_START" in df.columns:
        ts_fmt = "%Y%m%d%H%M" if resolution == "HH" else "%Y%m%d"
        df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"].astype(str), format=ts_fmt)
        if "TIMESTAMP_END" in df.columns:
            df["TIMESTAMP_END"] = pd.to_datetime(df["TIMESTAMP_END"].astype(str), format=ts_fmt)
        df = df.set_index("TIMESTAMP_START").sort_index()

    return df


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=== 1. Fetch site list ===")
    sites = get_site_info()
    print(f"Total sites: {len(sites)}")
    print(sites[["SITE_ID", "SITE_NAME", "IGBP", "COUNTRY",
                  "FLUXNET_START", "FLUXNET_END", "DATA_POLICY"]].head(10).to_string(index=False))

    print("\n=== 2. Filter sites — global, ≥2 FLUXNET years within 2015–2025 ===")
    filtered = filter_sites(
        sites,
        fluxnet_only=True,
        fluxnet_year_range=(2015, 2025),
        min_fluxnet_years=2,
    )
    print(f"Sites after filtering: {len(filtered)}")
    print(filtered[["SITE_ID", "SITE_NAME", "IGBP", "COUNTRY", "LOCATION_LAT",
                     "LOCATION_LONG", "FLUXNET_START", "FLUXNET_END"]].to_string(index=False))

    # Save filtered site list
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUT_DIR / "filtered_sites.csv", index=False)
    print(f"\nSaved filtered site list -> {OUT_DIR / 'filtered_sites.csv'}")

    # Submits request to AmeriFlux API — download link will arrive by email.
    print("\n=== 3. Request FLUXNET FULLSET data ===")
    site_ids = filtered["SITE_ID"].tolist()
    download_fluxnet(site_ids, data_variant="FULLSET")

    # print("\n=== 4. Extract zip files ===")
    # by_res = extract_fluxnet(zips)
    # hh_csvs = by_res.get("HH", [])
    # print(f"  Half-hourly CSV files: {len(hh_csvs)}")
    #
    # print("\n=== 5. Read one half-hourly file as a check ===")
    # if hh_csvs:
    #     df = read_fluxnet(hh_csvs[0], resolution="HH")
    #     flux_cols = [c for c in ["NEE_VUT_REF", "H_F_MDS", "LE_F_MDS", "SW_IN_F", "TA_F", "P_F"]
    #                  if c in df.columns]
    #     print(df[flux_cols].head())

    print("\n=== 6. Download BASE-BADM (sensor depths + canopy height) ===")
    badm_zips = download_badm(site_ids)

    print("\n=== 7. Extract ancillary metadata ===")
    # badm_zips = sorted(OUT_DIR.glob("*_BASE-BADM.zip"))  # if already downloaded
    meta = extract_badm_metadata(badm_zips)
    meta.to_csv(OUT_DIR / "site_badm_metadata.csv", index=False)
    print(meta[["SITE_ID", "SWC_DEPTHS_CM", "CANOPY_HEIGHT_M"]].to_string(index=False))
    print(f"\nSaved -> {OUT_DIR / 'site_badm_metadata.csv'}")
