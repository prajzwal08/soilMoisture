"""Netherlands land-only tile grid for the wall-to-wall run (plan §68, user 2026-10-05).

Regular 2.24 km grid (one model tile = 224 px x 10 m) in RD New (EPSG:28992), clipped to the CBS
generalised land-part boundaries from PDOK, which exclude the IJsselmeer, Wadden Sea and large
lakes. A tile is kept when >= MIN_LAND_FRAC of it is land; `centre_on_land` flags kept tiles whose
centre (the only pixel the model predicts) falls on water.

Input : data/nl/landsdeel_2024.json  (PDOK WFS gebiedsindelingen:landsdeel_gegeneraliseerd, 2024)
Output: csvs/nl_tiles.csv            (one row per grid cell, kept or not, with land_frac)
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

LAND_FILE = "data/nl/landsdeel_2024.json"
PROV_FILE = "data/nl/provincie_2024.json"  # PDOK gebiedsindelingen:provincie_gegeneraliseerd 2024
OUT_CSV = "csvs/nl_tiles.csv"
TILE_M = 2240.0
MIN_LAND_FRAC = 0.5


def main():
    land = gpd.read_file(LAND_FILE).to_crs(28992)
    land_union = land.union_all()
    print(f"land parts: {len(land)}  ({', '.join(land['statnaam'])})")
    print(f"land area: {land_union.area / 1e6:,.0f} km2")

    xmin, ymin, xmax, ymax = land_union.bounds
    # snap the grid origin to a multiple of the tile size so the grid is reproducible
    x0 = np.floor(xmin / TILE_M) * TILE_M
    y0 = np.floor(ymin / TILE_M) * TILE_M
    xs = np.arange(x0, xmax, TILE_M)
    ys = np.arange(y0, ymax, TILE_M)
    cells = gpd.GeoDataFrame(
        {"ix": np.repeat(np.arange(len(xs)), len(ys)), "iy": np.tile(np.arange(len(ys)), len(xs))},
        geometry=[box(x, y, x + TILE_M, y + TILE_M) for x in xs for y in ys], crs=28992)
    cells = cells[cells.intersects(land_union)].copy()

    cells["land_frac"] = cells.geometry.intersection(land_union).area / TILE_M**2
    cells["x_rd"] = cells.geometry.centroid.x
    cells["y_rd"] = cells.geometry.centroid.y
    cells["centre_on_land"] = [land_union.contains(Point(x, y)) for x, y in zip(cells.x_rd, cells.y_rd)]
    cells["keep"] = cells["land_frac"] >= MIN_LAND_FRAC
    ll = gpd.GeoSeries(gpd.points_from_xy(cells.x_rd, cells.y_rd), crs=28992).to_crs(4326)
    cells["latitude"] = ll.y.values.round(6)
    cells["longitude"] = ll.x.values.round(6)
    # province = the one containing the tile centre (centre on water -> nearest province)
    prov = gpd.read_file(PROV_FILE).to_crs(28992)[["statnaam", "geometry"]]
    centres = gpd.GeoDataFrame(geometry=gpd.points_from_xy(cells.x_rd, cells.y_rd), crs=28992,
                               index=cells.index)
    cells["province"] = gpd.sjoin_nearest(centres, prov, how="left").groupby(level=0)["statnaam"].first()
    cells["station_name"] = [f"NL_{ix:03d}_{iy:03d}" for ix, iy in zip(cells.ix, cells.iy)]
    cells["network"] = "NL_GRID"

    cols = ["network", "station_name", "province", "latitude", "longitude", "x_rd", "y_rd", "ix", "iy",
            "land_frac", "centre_on_land", "keep"]
    pd.DataFrame(cells[cols]).round({"land_frac": 4}).to_csv(OUT_CSV, index=False)

    kept = cells[cells.keep]
    print(f"cells touching land: {len(cells):,}")
    print(f"kept (land_frac >= {MIN_LAND_FRAC}): {len(kept):,}  "
          f"= {len(kept) * TILE_M**2 / 1e6:,.0f} km2 of tile area")
    print(f"kept with centre on water: {(~kept.centre_on_land).sum():,}")
    print(f"land covered by kept tiles: "
          f"{kept.geometry.intersection(land_union).area.sum() / land_union.area:.1%}")
    print("kept tiles per province:")
    print(kept.groupby("province").size().sort_values(ascending=False).to_string())
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
