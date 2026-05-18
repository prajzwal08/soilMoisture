"""
Compile a unified station inventory CSV from all processed stations across:
  - ISMN         (1324 soil-moisture-only stations)
  - ICOS         (63 flux sites: 37 with SWC, 26 flux-only)
  - AmeriFlux    (91 flux sites: 12 with SWC, 79 flux-only)

Output: /home/khanalp/data/soilmoisture/station_inventory.csv

Koppen-Geiger for ICOS is set to N/A — no raster/package available locally.
To fill later: sample Beck et al. 2018 KG GeoTIFF with rasterio at each lat/lon.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ISMN_META_CSV  = Path("/home/khanalp/data/soilmoisture/level1/station_metadata.csv")
ISMN_PY_META   = Path(
    "/home/khanalp/data/ISMNsoilMoisture/"
    "Data_separate_files_header_20140101_20251231_13107_18mx_20260208/"
    "python_metadata/"
    "Data_separate_files_header_20140101_20251231_13107_18mx_20260208.csv"
)
ICOS_SWC_DIR   = Path("/home/khanalp/data/soilmoisture/icos_level1/swc_and_flux")
ICOS_FLUX_DIR  = Path("/home/khanalp/data/soilmoisture/icos_level1/flux_only")
ICOS_AVAIL_CSV = Path("/home/khanalp/data/flux/icos_raw/ICOS_available_station.csv")
ICOS_INFO_CSV  = Path("/home/khanalp/data/flux/ICOS_site_info.csv")
FLX15_CSV      = Path("/home/khanalp/data/flux/FLX15_site_info.csv")
AMF_SWC_DIR    = Path("/home/khanalp/data/soilmoisture/ameriflux_level1/swc_and_flux")
AMF_FLUX_DIR   = Path("/home/khanalp/data/soilmoisture/ameriflux_level1/flux_only")
OUT_CSV        = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_inventory.csv")

# ---------------------------------------------------------------------------
# ESA CCI LC → IGBP mapping
# ---------------------------------------------------------------------------
CCI_TO_IGBP = {
    10: "CRO", 11: "CRO", 12: "CRO", 20: "CRO", 30: "MF",  40: "MF",
    50: "EBF",
    60: "DBF", 61: "DBF", 62: "DBF",
    70: "ENF", 71: "ENF", 72: "ENF",
    80: "DNF", 81: "DNF", 82: "DNF",
    90: "MF", 100: "MF",
    110: "SAV", 120: "SAV", 121: "SAV", 122: "SAV",
    130: "GRA", 140: "GRA",
    150: "BSV",
    160: "WET", 170: "WET", 180: "WET",
    190: "URB",
    200: "BSV", 210: "WAT", 220: "SNO",
}


# ---------------------------------------------------------------------------
# ISMN
# ---------------------------------------------------------------------------
def build_ismn_rows() -> pd.DataFrame:
    # --- station list (only saved stations) ---
    meta = pd.read_csv(ISMN_META_CSV)
    meta = meta[meta["status"] == "saved"].reset_index(drop=True)

    # --- python_metadata: multi-index header, one row per sensor file ---
    py = pd.read_csv(ISMN_PY_META, header=[0, 1], low_memory=False)

    # flatten to useful columns
    net_col   = ("network",    "val")
    sta_col   = ("station",    "val")
    kg_col    = ("climate_KG", "val")
    lc_col    = ("lc_2010",    "val")
    elev_col  = ("elevation",  "val")

    py_flat = py[[net_col, sta_col, kg_col, lc_col, elev_col]].copy()
    py_flat.columns = ["network", "station", "climate_KG", "lc_2010", "elevation_m"]

    # one row per (network, station): take first non-null value
    py_station = (
        py_flat.replace("", np.nan)
        .groupby(["network", "station"], as_index=False)
        .first()
    )

    merged = meta.merge(py_station, on=["network", "station"], how="left")

    rows = []
    for _, r in merged.iterrows():
        cci = r["lc_2010"]
        try:
            cci_int = int(float(cci))
            igbp = CCI_TO_IGBP.get(cci_int, "N/A")
            lc_cci = cci_int
        except (TypeError, ValueError):
            igbp = "N/A"
            lc_cci = np.nan

        n_days = float(r["n_days"]) if pd.notna(r["n_days"]) else np.nan
        n_years = round(n_days / 365.25, 1) if not np.isnan(n_days) else np.nan

        rows.append({
            "source_network":   "ISMN",
            "network":          r["network"],
            "station_id":       r["station"],
            "station_name":     r["station"],
            "latitude":         r["latitude"],
            "longitude":        r["longitude"],
            "elevation_m":      r.get("elevation_m", np.nan),
            "start_date":       str(r["start_date"]),
            "end_date":         str(r["end_date"]),
            "n_years":          n_years,
            "IGBP":             igbp,
            "lc_cci":           lc_cci,
            "koppen_geiger":    r["climate_KG"] if (pd.notna(r.get("climate_KG")) and r.get("climate_KG") not in ("", "masked")) else "N/A",
            "has_soil_moisture": True,
            "has_flux":          False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ICOS
# ---------------------------------------------------------------------------
def _parse_dates_from_filename(nc_path: Path):
    """Extract start_date, end_date from '{site_id}_{YYYYMMDD}_{YYYYMMDD}.nc'."""
    stem = nc_path.stem
    parts = stem.split("_")
    # last two underscore-separated tokens are dates
    return parts[-2], parts[-1]


# Koppen-Geiger for ICOS2020 sites absent from all station CSVs (lat/lon from FLX15)
ICOS_MANUAL_KG = {
    "CH-Cha": "Cfb",
    "CH-Fru": "Cfb",
    "CH-Lae": "Cfb",
    "CH-Oe2": "Cfb",
    "DE-Obe": "Dfb",
    "RU-Fyo": "Dfb",
}


def _load_icos_fallback() -> pd.DataFrame:
    """Load lat/lon/IGBP/elevation fallback from three source CSVs.

    Priority: ICOS_site_info.csv → ICOS_available_station.csv → FLX15_site_info.csv
    """
    # 1. ICOS_site_info.csv: site_id, site_name, lat, lon, IGBP
    info = pd.read_csv(ICOS_INFO_CSV)[["site_id", "site_name", "lat", "lon", "IGBP"]].copy()
    info = info.rename(columns={"lat": "fb_lat", "lon": "fb_lon",
                                "site_name": "fb_name", "IGBP": "fb_igbp"})

    # 2. ICOS_available_station.csv: SITE_ID, LOCATION_LAT, LOCATION_LONG, LOCATION_ELEV
    avail = pd.read_csv(ICOS_AVAIL_CSV)[
        ["SITE_ID", "LOCATION_LAT", "LOCATION_LONG", "LOCATION_ELEV"]
    ].rename(columns={
        "SITE_ID": "site_id",
        "LOCATION_LAT": "fb_lat2", "LOCATION_LONG": "fb_lon2", "LOCATION_ELEV": "fb_elev"
    })

    # 3. FLX15_site_info.csv: covers FLUXNET2015 sites absent from ICOS CSVs
    flx15 = pd.read_csv(FLX15_CSV)[["site_id", "site_name", "lat", "lon", "IGBP"]].copy()
    flx15 = flx15.rename(columns={"lat": "fb_lat3", "lon": "fb_lon3",
                                   "site_name": "fb_name3", "IGBP": "fb_igbp3"})

    fb = info.merge(avail, on="site_id", how="outer").merge(flx15, on="site_id", how="outer")
    fb["fb_lat"]  = fb["fb_lat"].fillna(fb["fb_lat2"]).fillna(fb["fb_lat3"])
    fb["fb_lon"]  = fb["fb_lon"].fillna(fb["fb_lon2"]).fillna(fb["fb_lon3"])
    fb["fb_name"] = fb["fb_name"].fillna(fb["fb_name3"])
    fb["fb_igbp"] = fb["fb_igbp"].fillna(fb["fb_igbp3"])
    return fb.set_index("site_id")


def build_icos_rows() -> pd.DataFrame:
    fallback = _load_icos_fallback()
    rows = []
    for nc_path in sorted(list(ICOS_SWC_DIR.glob("*.nc")) + list(ICOS_FLUX_DIR.glob("*.nc"))):
        ds = xr.open_dataset(nc_path)
        a = ds.attrs
        ds.close()

        site_id = a.get("site_id", nc_path.stem)
        fb = fallback.loc[site_id] if site_id in fallback.index else None

        def _val(key, fb_key=None, default=np.nan):
            v = a.get(key, "")
            if v != "" and not (isinstance(v, float) and np.isnan(v)):
                return v
            if fb is not None and fb_key and pd.notna(fb.get(fb_key)):
                return fb[fb_key]
            return default

        start_date, end_date = _parse_dates_from_filename(nc_path)
        n_days = (
            pd.to_datetime(end_date, format="%Y%m%d")
            - pd.to_datetime(start_date, format="%Y%m%d")
        ).days + 1
        n_years = round(n_days / 365.25, 1)

        has_sm = str(a.get("has_soil_moisture", "False")).lower() == "true"

        igbp = _val("IGBP", "fb_igbp", "N/A")
        if igbp == "":
            igbp = "N/A"

        lat  = float(_val("latitude",   "fb_lat",  np.nan))
        lon  = float(_val("longitude",  "fb_lon",  np.nan))
        elev = float(_val("elevation_m","fb_elev", np.nan))
        kg   = ICOS_MANUAL_KG.get(site_id, "N/A")

        rows.append({
            "source_network":   "ICOS",
            "network":          "ICOS",
            "station_id":       site_id,
            "station_name":     _val("site_name", "fb_name", ""),
            "latitude":         lat,
            "longitude":        lon,
            "elevation_m":      elev,
            "start_date":       start_date,
            "end_date":         end_date,
            "n_years":          n_years,
            "IGBP":             igbp,
            "lc_cci":           np.nan,
            "koppen_geiger":    kg,
            "has_soil_moisture": has_sm,
            "has_flux":          True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AmeriFlux
# ---------------------------------------------------------------------------
def build_ameriflux_rows() -> pd.DataFrame:
    rows = []
    for nc_path in sorted(list(AMF_SWC_DIR.glob("*.nc")) + list(AMF_FLUX_DIR.glob("*.nc"))):
        ds = xr.open_dataset(nc_path)
        a = ds.attrs
        ds.close()

        start_date, end_date = _parse_dates_from_filename(nc_path)
        n_days = (
            pd.to_datetime(end_date, format="%Y%m%d")
            - pd.to_datetime(start_date, format="%Y%m%d")
        ).days + 1
        n_years = round(n_days / 365.25, 1)

        has_sm = str(a.get("has_soil_moisture", "False")).lower() == "true"

        rows.append({
            "source_network":   "AmeriFlux",
            "network":          "AmeriFlux",
            "station_id":       a.get("site_id", nc_path.stem),
            "station_name":     a.get("site_name", ""),
            "latitude":         float(a.get("latitude", np.nan)),
            "longitude":        float(a.get("longitude", np.nan)),
            "elevation_m":      float(a.get("elevation_m", np.nan)),
            "start_date":       start_date,
            "end_date":         end_date,
            "n_years":          n_years,
            "IGBP":             a.get("IGBP", "N/A") or "N/A",
            "lc_cci":           np.nan,
            "koppen_geiger":    a.get("climate_koeppen", "N/A") or "N/A",
            "has_soil_moisture": has_sm,
            "has_flux":          True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building ISMN rows...")
    ismn = build_ismn_rows()
    print(f"  {len(ismn)} ISMN stations")

    print("Building ICOS rows...")
    icos = build_icos_rows()
    print(f"  {len(icos)} ICOS sites  ({icos['has_soil_moisture'].sum()} with SWC)")

    print("Building AmeriFlux rows...")
    amf = build_ameriflux_rows()
    print(f"  {len(amf)} AmeriFlux sites  ({amf['has_soil_moisture'].sum()} with SWC)")

    df = pd.concat([ismn, icos, amf], ignore_index=True)

    # diagnostics
    print(f"\nTotal: {len(df)} stations")
    print(f"  NaN latitude:     {df['latitude'].isna().sum()}")
    print(f"  NaN elevation_m:  {df['elevation_m'].isna().sum()}")
    print(f"  IGBP missing:     {(df['IGBP'].isin(['N/A', '']) | df['IGBP'].isna()).sum()}")
    print(f"  Koppen missing:   {(df['koppen_geiger'].isin(['N/A', '']) | df['koppen_geiger'].isna()).sum()}")
    print("\nIGBP distribution:")
    print(df["IGBP"].value_counts().to_string())
    print("\nKoppen distribution (top 15):")
    print(df["koppen_geiger"].value_counts().head(15).to_string())

    # ensure start/end dates are zero-padded strings not floats
    df["start_date"] = df["start_date"].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and x != "" else ""
    )
    df["end_date"] = df["end_date"].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and x != "" else ""
    )
    # replace "N/A" sentinel with empty string so pandas doesn't silently re-parse it
    df = df.replace("N/A", "")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, na_rep="")
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
