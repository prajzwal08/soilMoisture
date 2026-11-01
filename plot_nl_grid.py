"""Map of the Netherlands land-only tile grid (plan §68). Reads csvs/nl_tiles.csv from make_nl_grid.py.

Kept tiles dark blue, dropped tiles (< 50 % land) grey, kept tiles whose centre is on water
orange, Overijssel tiles filled light blue (province outline bold), TWENTE stations as black dots. Paper style (plot_style_bw.RC_PAPER), no title.
Output: figures/nl_grid/nl_tiles.{png,pdf}
"""
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box

import plot_style_bw as ps

LAND_FILE = "data/nl/landsdeel_2024.json"
PROV_FILE = "data/nl/provincie_2024.json"
FOCUS = "Overijssel"  # pilot province, contains all 8 TWENTE stations
TILES_CSV = "csvs/nl_tiles.csv"
OUT = "figures/nl_grid/nl_tiles"
TILE_M = 2240.0


def main():
    plt.rcParams.update(ps.RC_PAPER)
    land = gpd.read_file(LAND_FILE).to_crs(28992)
    prov = gpd.read_file(PROV_FILE).to_crs(28992)
    focus = prov[prov.statnaam == FOCUS]
    t = pd.read_csv(TILES_CSV)
    h = TILE_M / 2
    tiles = gpd.GeoDataFrame(t, geometry=[box(x - h, y - h, x + h, y + h) for x, y in zip(t.x_rd, t.y_rd)],
                             crs=28992)
    kept, dropped = tiles[tiles.keep], tiles[~tiles.keep]
    kept_focus = kept[kept.province == FOCUS]
    wet_centre = kept[~kept.centre_on_land]

    st = pd.read_csv("csvs/station_splits.csv")
    st = st[st.network == "TWENTE"]
    st = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.longitude, st.latitude), crs=4326).to_crs(28992)

    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    land.plot(ax=ax, color="0.88", edgecolor="none", zorder=1)
    kept_focus.plot(ax=ax, facecolor=ps.GREEN, edgecolor="none", zorder=2)
    dropped.plot(ax=ax, facecolor="none", edgecolor="0.55", linewidth=0.35, zorder=3)
    kept.plot(ax=ax, facecolor="none", edgecolor=ps.BLUE, linewidth=0.35, zorder=4)
    wet_centre.plot(ax=ax, facecolor=ps.RED, edgecolor=ps.RED, linewidth=0.35, zorder=5)
    gpd.GeoSeries([land.union_all()], crs=28992).boundary.plot(ax=ax, color="black", linewidth=0.6, zorder=6)
    focus.boundary.plot(ax=ax, color="black", linewidth=1.8, zorder=7)
    st.plot(ax=ax, color="black", markersize=14, zorder=8)

    ax.set_aspect("equal")
    ax.set_xlabel("RD x (km)")
    ax.set_ylabel("RD y (km)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    handles = [
        Patch(facecolor="none", edgecolor=ps.BLUE, linewidth=1.2, label=f"Kept, ≥50 % land ({len(kept):,})"),
        Patch(facecolor=ps.GREEN, edgecolor=ps.BLUE, linewidth=1.2, label=f"{FOCUS} ({len(kept_focus):,})"),
        Patch(facecolor="none", edgecolor="0.55", linewidth=1.2, label=f"Dropped, <50 % land ({len(dropped):,})"),
        Patch(facecolor=ps.RED, edgecolor=ps.RED, label=f"Kept, centre on water ({len(wet_centre):,})"),
        Line2D([], [], marker="o", color="black", linestyle="none", markersize=5,
               label=f"TWENTE stations ({len(st)})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    print(f"{FOCUS}: {len(kept_focus):,} kept tiles, province area {focus.area.sum() / 1e6:,.0f} km2")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print(f"wrote {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
