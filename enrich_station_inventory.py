"""
Enrich station_inventory.csv with elevation, IGBP, and Koppen-Geiger
for rows that currently have missing values.

Data sources:
  elevation_m    -- SRTM (USGS/SRTMGL1_003, 30 m) via GEE
  IGBP           -- MODIS MCD12C1 2020 (Majority_Land_Cover_Type_1) via GEE
  koppen_geiger  -- Kottek et al. 2006 ASCII table (0.5° grid, downloaded once)
                   Fallback: Beck et al. 2018 GeoTIFF if saved locally as BECK_TIF

Run with the geo conda env (has ee + rasterio + xarray):
  /home/khanalp/miniforge3/envs/geo/bin/python enrich_station_inventory.py
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import ee
from google.oauth2.credentials import Credentials

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INVENTORY_CSV   = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_inventory.csv")
AUX_DIR          = Path("/home/khanalp/data/aux")
BECK_TIF         = AUX_DIR / "beck_kg_present_1km.tif"   # optional local override
KOTTEK_CSV       = AUX_DIR / "kottek_kg_2006.csv"         # cached after first download
KOTTEK_URL       = "https://koeppen-geiger.vu-wien.ac.at/data/Koeppen-Geiger-ASCII.zip"
CREDENTIALS_FILE = Path.home() / ".config/earthengine/credentials"
GEE_PROJECT      = "1066500857818"

# ---------------------------------------------------------------------------
# Koppen code lookup (Beck et al. 2018 pixel value → KG code)
# ---------------------------------------------------------------------------
BECK_CODES = {
     1: "Af",  2: "Am",  3: "Aw",
     4: "BWh", 5: "BWk",
     6: "BSh", 7: "BSk",
     8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET",  30: "EF",
}

# MODIS MCD12C1 Type-1 (IGBP) pixel value → IGBP code
MODIS_IGBP = {
     1: "ENF",  2: "EBF",  3: "DNF",  4: "DBF",  5: "MF",
     6: "CSH",  7: "OSH",  8: "WSA",  9: "SAV", 10: "GRA",
    11: "WET", 12: "CRO", 13: "URB", 14: "MF",  15: "SNO",
    16: "BSV", 17: "WAT",
}

# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------
def _gee_credentials() -> Credentials:
    with open(CREDENTIALS_FILE) as f:
        d = json.load(f)
    return Credentials(
        token=None,
        refresh_token=d["refresh_token"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )


def _init_gee():
    ee.Initialize(credentials=_gee_credentials(), project=GEE_PROJECT)


def _sample_gee(lats: list, lons: list, indices: list) -> pd.DataFrame:
    """Sample SRTM elevation and MODIS IGBP at given coordinates via GEE.

    Returns DataFrame with columns: idx, elevation_gee, igbp_gee.
    """
    srtm  = ee.Image("USGS/SRTMGL1_003").select("elevation")
    modis = (
        ee.ImageCollection("MODIS/061/MCD12C1")
        .filterDate("2020-01-01", "2021-01-01")
        .first()
        .select("Majority_Land_Cover_Type_1")
    )
    combined = srtm.addBands(modis)

    CHUNK = 400  # GEE getInfo limit per request
    records = []
    for start in range(0, len(lats), CHUNK):
        end = min(start + CHUNK, len(lats))
        features = [
            ee.Feature(
                ee.Geometry.Point([float(lons[i]), float(lats[i])]),
                {"idx": int(indices[i])}
            )
            for i in range(start, end)
        ]
        fc = ee.FeatureCollection(features)
        sampled = combined.sampleRegions(
            collection=fc, scale=500, geometries=False, tileScale=4
        )
        info = sampled.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            records.append({
                "idx":          p["idx"],
                "elevation_gee": p.get("elevation"),
                "igbp_code_gee": p.get("Majority_Land_Cover_Type_1"),
            })
        print(f"  GEE: sampled {end}/{len(lats)} points")

    df = pd.DataFrame(records)
    df["igbp_gee"] = df["igbp_code_gee"].map(MODIS_IGBP)
    return df

# ---------------------------------------------------------------------------
# Beck et al. Koppen raster helpers
# ---------------------------------------------------------------------------
def _sample_beck(lats: list, lons: list) -> list:
    """Sample Beck et al. 2018 local GeoTIFF (optional higher-res override)."""
    results = []
    with rasterio.open(BECK_TIF) as src:
        for lat, lon in zip(lats, lons):
            try:
                row, col = src.index(lon, lat)
                val = int(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
                results.append(BECK_CODES.get(val, ""))
            except Exception:
                results.append("")
    return results


# ---------------------------------------------------------------------------
# Kottek et al. 2006 ASCII lookup (primary Koppen source, 0.5° grid)
# ---------------------------------------------------------------------------
def _load_kottek() -> pd.DataFrame:
    """Download (once) and cache the Kottek et al. 2006 KG ASCII table."""
    if KOTTEK_CSV.exists():
        return pd.read_csv(KOTTEK_CSV)

    print(f"Downloading Kottek et al. 2006 Koppen table from {KOTTEK_URL} ...")
    AUX_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(KOTTEK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        fname = next(n for n in z.namelist() if n.endswith(".txt"))
        content = z.read(fname).decode("latin-1")

    kg = pd.read_csv(io.StringIO(content), sep=r"\s+", header=0,
                     names=["lat", "lon", "kg"])
    kg.to_csv(KOTTEK_CSV, index=False)
    print(f"  Cached → {KOTTEK_CSV}  ({len(kg):,} grid cells)")
    return kg


def _sample_kottek(kg_grid: pd.DataFrame, lats: list, lons: list) -> list:
    """Nearest-neighbour lookup at 0.5° grid spacing."""
    # Round to nearest 0.25° (grid centres are at ±0.25, ±0.75, ...)
    def _snap(v, step=0.5):
        return np.round((np.array(v) - step / 2) / step) * step + step / 2

    snap_lats = _snap(lats)
    snap_lons = _snap(lons)

    lookup = kg_grid.set_index(["lat", "lon"])["kg"]
    results = []
    for slat, slon in zip(snap_lats, snap_lons):
        try:
            results.append(lookup.loc[(slat, slon)])
        except KeyError:
            results.append("")
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(INVENTORY_CSV)
    original_len = len(df)

    # identify rows needing each field
    need_elev  = df["elevation_m"].isna()
    need_igbp  = df["IGBP"].isna() | (df["IGBP"] == "")
    need_koppen = df["koppen_geiger"].isna() | (df["koppen_geiger"] == "")

    print(f"Stations needing elevation:     {need_elev.sum()}")
    print(f"Stations needing IGBP:          {need_igbp.sum()}")
    print(f"Stations needing Koppen-Geiger: {need_koppen.sum()}")

    # -----------------------------------------------------------------------
    # 1. GEE: fill elevation and IGBP
    # -----------------------------------------------------------------------
    gee_mask = need_elev | need_igbp
    if gee_mask.any():
        print("\nInitialising GEE ...")
        _init_gee()

        sub = df[gee_mask].reset_index()  # 'index' column = original df index
        print(f"Sampling {len(sub)} points from GEE (SRTM + MODIS) ...")
        gee_df = _sample_gee(
            lats=sub["latitude"].tolist(),
            lons=sub["longitude"].tolist(),
            indices=sub["index"].tolist(),
        )
        gee_df = gee_df.set_index("idx")

        for orig_idx, row in sub.iterrows():
            i = row["index"]
            if i not in gee_df.index:
                continue
            g = gee_df.loc[i]
            if need_elev.iloc[i] and pd.notna(g["elevation_gee"]):
                df.at[i, "elevation_m"] = float(g["elevation_gee"])
            if need_igbp.iloc[i] and g["igbp_gee"]:
                df.at[i, "IGBP"] = g["igbp_gee"]

        n_elev_filled  = (need_elev & df["elevation_m"].notna()).sum()
        n_igbp_filled  = (need_igbp & df["IGBP"].notna() & (df["IGBP"] != "")).sum()
        print(f"  Filled elevation: {n_elev_filled}")
        print(f"  Filled IGBP:      {n_igbp_filled}")
    else:
        print("\nNo GEE lookups needed.")

    # -----------------------------------------------------------------------
    # 2. Koppen-Geiger
    #    Priority: local Beck et al. GeoTIFF → ERA5 climatology via GEE
    # -----------------------------------------------------------------------
    need_koppen = df["koppen_geiger"].isna() | (df["koppen_geiger"] == "")
    if need_koppen.any():
        sub = df[need_koppen]

        if BECK_TIF.exists():
            # Higher-resolution override (1 km) — user placed it manually
            print(f"\nBeck raster found: {BECK_TIF}")
            print(f"Sampling {len(sub)} points from Beck raster ...")
            codes = _sample_beck(sub["latitude"].tolist(), sub["longitude"].tolist())
        else:
            # Primary source: Kottek et al. 2006 ASCII table (0.5°, auto-download)
            kg_grid = _load_kottek()
            print(f"Sampling {len(sub)} points from Kottek et al. 2006 table ...")
            codes = _sample_kottek(kg_grid, sub["latitude"].tolist(), sub["longitude"].tolist())

        df.loc[need_koppen, "koppen_geiger"] = codes
        filled = sum(c != "" for c in codes)
        print(f"  Filled Koppen: {filled} / {len(codes)}")
    else:
        print("\nNo Koppen lookups needed.")

    # -----------------------------------------------------------------------
    # Summary and save
    # -----------------------------------------------------------------------
    assert len(df) == original_len
    still_missing_elev   = df["elevation_m"].isna().sum()
    still_missing_igbp   = (df["IGBP"].isna() | (df["IGBP"] == "")).sum()
    still_missing_koppen = (df["koppen_geiger"].isna() | (df["koppen_geiger"] == "")).sum()
    print(f"\nRemaining missing after enrichment:")
    print(f"  elevation_m:    {still_missing_elev}")
    print(f"  IGBP:           {still_missing_igbp}")
    print(f"  koppen_geiger:  {still_missing_koppen}")

    df.to_csv(INVENTORY_CSV, index=False, na_rep="")
    print(f"\nUpdated → {INVENTORY_CSV}")


if __name__ == "__main__":
    main()
