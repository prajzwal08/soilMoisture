"""
Download and filter ICOS Ecosystem gap-filled FLUXNET-format flux data.

Mirrors the AmeriFlux workflow in download_ameriflux.py:
  get_site_info()        -> station list with metadata
  filter_sites()         -> filter by theme, country, lat/lon
  get_data_inventory()   -> fetch L2 Fluxnet data objects + summarise years
  download_icos()        -> save original CSV files per station

Data product: "ETC L2 Fluxnet (half-hourly)" — spec etcL2Fluxnet
  Gap-filled variables: NEE_VUT_REF, LE_F_MDS, H_F_MDS, GPP_DT_VUT_REF
  Same variable naming as AmeriFlux FLUXNET.

Authentication required for data download:
  from icoscp_core.icos import auth
  auth.init_config_file()   # one-time interactive setup

ICOS Carbon Portal: https://data.icos-cp.eu/portal/
Account: https://cpauth.icos-cp.eu/
"""

from pathlib import Path

import pandas as pd
from icoscp.station import station as station_module
from icoscp.dobj import Dobj
from icoscp_core.icos import data as icos_data

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path("/home/khanalp/data/icos_raw")

# ---------------------------------------------------------------------------
# ICOS Ecosystem gap-filled FLUXNET-format half-hourly product
# Variables: NEE_VUT_REF, LE_F_MDS, H_F_MDS, GPP_DT_VUT_REF, TA_F, P_F ...
# ---------------------------------------------------------------------------
ECO_FLUXNET_SPEC = "http://meta.icos-cp.eu/resources/cpmeta/etcL2Fluxnet"


def get_site_info() -> pd.DataFrame:
    """
    Fetch the full ICOS station list.

    Returns a DataFrame with columns:
        SITE_ID, SITE_NAME, COUNTRY, THEME, SITE_TYPE,
        LOCATION_LAT, LOCATION_LONG, LOCATION_ELEV, ICOS_CLASS, URI
    """
    df = station_module.getIdList()

    rename = {
        "id":        "SITE_ID",
        "name":      "SITE_NAME",
        "country":   "COUNTRY",
        "theme":     "THEME",
        "siteType":  "SITE_TYPE",
        "lat":       "LOCATION_LAT",
        "lon":       "LOCATION_LONG",
        "elevation": "LOCATION_ELEV",
        "icosclass": "ICOS_CLASS",
        "uri":       "URI",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ["LOCATION_LAT", "LOCATION_LONG", "LOCATION_ELEV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_sites(
    sites: pd.DataFrame,
    theme: str = "ES",
    country: list[str] | None = None,
    site_type: list[str] | None = None,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    icos_class: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter the ICOS station list on basic metadata (no data-availability check).

    Parameters
    ----------
    theme      : 'ES' (Ecosystem), 'AT' (Atmosphere), 'OC' (Ocean). Default 'ES'.
    country    : Two-letter country codes, e.g. ['DE', 'FR', 'SE']
    site_type  : e.g. ['Forest', 'Grassland', 'Cropland', 'Wetland', 'Shrubland']
    lat_range  : (min_lat, max_lat) decimal degrees
    lon_range  : (min_lon, max_lon) decimal degrees
    icos_class : ICOS certification class, e.g. ['1', '2']
    """
    mask = pd.Series(True, index=sites.index)

    if theme and "THEME" in sites.columns:
        mask &= sites["THEME"] == theme
    if country:
        mask &= sites["COUNTRY"].isin(country)
    if site_type and "SITE_TYPE" in sites.columns:
        mask &= sites["SITE_TYPE"].isin(site_type)
    if lat_range:
        mask &= sites["LOCATION_LAT"].between(*lat_range)
    if lon_range:
        mask &= sites["LOCATION_LONG"].between(*lon_range)
    if icos_class and "ICOS_CLASS" in sites.columns:
        mask &= sites["ICOS_CLASS"].isin(icos_class)

    return sites[mask].reset_index(drop=True)


def get_data_inventory(
    site_ids: list[str],
    year_range: tuple[int, int] | None = None,
    min_years: int | None = None,
    data_level: str = "2",
) -> pd.DataFrame:
    """
    Fetch L2 data objects for each station and summarise available years.

    Uses icoscp_core.icos.meta to fetch all etcL2Fluxnet objects in one bulk
    query (avoids the broken station.get() per-station API), then filters by
    station ID, year window, and minimum coverage.

    Parameters
    ----------
    site_ids   : List of ICOS station IDs, e.g. ['DE-Hai', 'SE-Nor']
    year_range : (start_year, end_year) inclusive — keep only data objects
                 whose period overlaps this window
    min_years  : Minimum number of years with data within year_range
    data_level : ICOS data level ('2' = quality-controlled, default)

    Returns
    -------
    DataFrame with columns:
        SITE_ID, N_YEARS, YEARS, DATA_OBJECTS
        (one row per station, only stations passing the filters)
    """
    from icoscp_core.icos import meta as icos_meta

    yr_min, yr_max = year_range if year_range else (0, 9999)

    print("  Fetching all ETC L2 Fluxnet data objects from Carbon Portal ...")
    try:
        # Returns list[DataObjectLite]; limit=1000 safely covers all ICOS stations
        all_dobjs = icos_meta.list_data_objects(datatype=ECO_FLUXNET_SPEC, limit=1000)
    except Exception as e:
        print(f"  ERROR fetching data objects: {e}")
        return pd.DataFrame(columns=["SITE_ID", "N_YEARS", "YEARS", "DATA_OBJECTS"])

    if not all_dobjs:
        print("  No data objects found for this spec.")
        return pd.DataFrame(columns=["SITE_ID", "N_YEARS", "YEARS", "DATA_OBJECTS"])

    print(f"  {len(all_dobjs)} FLUXNET_HH objects found (Final + Interim).")

    # station_uri looks like "…/ES_DE-Hai"; extract the short ID after "ES_"
    def _extract_id(uri: str | None) -> str:
        if not uri:
            return ""
        part = uri.split("/")[-1]          # e.g. "ES_DE-Hai"
        return part.split("_", 1)[-1]      # e.g. "DE-Hai"

    # Build a lookup: short_id -> list of DataObjectLite
    from collections import defaultdict
    by_station: dict[str, list] = defaultdict(list)
    for dobj in all_dobjs:
        sid = _extract_id(dobj.station_uri)
        by_station[sid].append(dobj)

    records = []
    window_start = pd.Timestamp(f"{yr_min}-01-01", tz="UTC") if year_range else None
    window_end   = pd.Timestamp(f"{yr_max}-12-31", tz="UTC") if year_range else None

    for site_id in site_ids:
        dobjs = by_station.get(site_id, [])
        if not dobjs:
            continue

        # Filter to year window
        if year_range is not None:
            dobjs = [
                d for d in dobjs
                if d.time_end >= window_start and d.time_start <= window_end
            ]
        if not dobjs:
            continue

        # Collect distinct years covered within window
        years = set()
        for d in dobjs:
            for y in range(d.time_start.year, d.time_end.year + 1):
                if yr_min <= y <= yr_max:
                    years.add(y)

        if min_years is not None and len(years) < min_years:
            continue

        records.append({
            "SITE_ID":      site_id,
            "N_YEARS":      len(years),
            "YEARS":        sorted(years),
            "DATA_OBJECTS": [d.uri for d in dobjs],
        })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["SITE_ID", "N_YEARS", "YEARS", "DATA_OBJECTS"]
    )


def download_icos(
    inventory: pd.DataFrame,
    out_dir: Path = OUT_DIR,
) -> dict[str, list[Path]]:
    """
    Download ICOS L2 data files for each station in the inventory.

    Each data object is saved as-is (original CSV from the Carbon Portal).
    Files are organised into per-station subdirectories.

    Parameters
    ----------
    inventory : DataFrame returned by get_data_inventory()
    out_dir   : Root output directory

    Returns
    -------
    Dict mapping SITE_ID -> list of downloaded file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, list[Path]] = {}

    for _, row in inventory.iterrows():
        site_id = row["SITE_ID"]
        dobj_uris = row["DATA_OBJECTS"]
        site_dir = out_dir / site_id
        site_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {site_id} ({len(dobj_uris)} file(s)) ...")
        site_files = []
        for uri in dobj_uris:
            try:
                icos_data.save_to_folder(uri, str(site_dir))
                # Find the file just written (most recently modified)
                files = sorted(site_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                if files:
                    site_files.append(files[-1])
                    print(f"  Saved -> {files[-1].name}")
            except Exception as e:
                print(f"  ERROR {uri}: {e}")

        if site_files:
            downloaded[site_id] = site_files

    return downloaded


def read_icos(csv_path: Path) -> pd.DataFrame:
    """
    Read a downloaded ICOS ETC L2 Fluxnet (half-hourly) CSV.

    Parses the TIMESTAMP column and replaces -9999 with NaN.

    Key gap-filled flux columns (same naming as AmeriFlux FLUXNET):
        NEE_VUT_REF     — Net Ecosystem Exchange, gap-filled (µmol m⁻² s⁻¹)
        NEE_VUT_REF_QC  — QC flag (0=measured, 1=good gap-fill, 2=medium, 3=poor)
        LE_F_MDS        — Latent heat flux, gap-filled (W m⁻²)
        H_F_MDS         — Sensible heat flux, gap-filled (W m⁻²)
        GPP_DT_VUT_REF  — GPP daytime partitioning (µmol m⁻² s⁻¹)
        GPP_NT_VUT_REF  — GPP nighttime partitioning (µmol m⁻² s⁻¹)
        TA_F            — Air temperature, gap-filled (°C)
        SW_IN_F         — Incoming shortwave radiation, gap-filled (W m⁻²)
        VPD_F           — VPD, gap-filled (hPa)
        P_F             — Precipitation, gap-filled (mm)
    """
    df = pd.read_csv(csv_path, na_values=[-9999, -9999.0], comment="#")

    for ts_col in ["TIMESTAMP", "Date", "date"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)
            df = df.set_index(ts_col).sort_index()
            break

    return df


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Authenticate once (interactive, stores credentials locally)
    # from icoscp_core.icos import auth
    # auth.init_config_file()

    print("=== 1. Fetch ICOS station list ===")
    sites = get_site_info()
    print(f"Total stations: {len(sites)}")
    print(sites[["SITE_ID", "SITE_NAME", "THEME", "COUNTRY",
                  "LOCATION_LAT", "LOCATION_LONG"]].head(10).to_string(index=False))

    print("\n=== 2. Filter — Ecosystem stations globally ===")
    filtered = filter_sites(
        sites,
        theme="ES",   # Ecosystem only
    )
    print(f"Ecosystem stations: {len(filtered)}")
    print(filtered[["SITE_ID", "SITE_NAME", "COUNTRY", "SITE_TYPE",
                     "LOCATION_LAT", "LOCATION_LONG"]].to_string(index=False))

    print("\n=== 3. Get data inventory — ≥2 years within 2015–2025 ===")
    inventory = get_data_inventory(
        filtered["SITE_ID"].tolist(),
        year_range=(2015, 2025),
        min_years=2,
    )
    print(f"Sites with sufficient data: {len(inventory)}")
    print(inventory[["SITE_ID", "N_YEARS", "YEARS"]].to_string(index=False))

    # Save inventory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.drop(columns=["DATA_OBJECTS"]).to_csv(
        OUT_DIR / "icos_filtered_sites.csv", index=False
    )
    print(f"\nSaved -> {OUT_DIR / 'icos_filtered_sites.csv'}")

    # Uncomment to actually download (requires authentication):
    print("\n=== 4. Download L2 flux data ===")
    downloaded = download_icos(inventory)
    #
    # print("\n=== 5. Read one file as a check ===")
    # for site_id, files in list(downloaded.items())[:1]:
    #     df = read_icos(files[0])
    #     flux_cols = [c for c in ["NEE_VUT_REF", "LE_F_MDS", "H_F_MDS", "GPP_DT_VUT_REF", "TA_F", "P_F"] if c in df.columns]
    #     print(f"{site_id}: {df[flux_cols].head()}")
