"""Map of a dense station network: tile footprints, station locations, overlap structure.

The point of the figure is the distinction the readout table encodes:

  * CENTRE stations  — predicted at pixel (112,112) of their own tile.  This is what the model
                       was supervised on and what every previous evaluation reports.
  * OFF-CENTRE       — the same station predicted from a *neighbouring* station's tile, at an
                       arbitrary pixel that received no supervision.  These are the additional
                       pixels the model has never been tested on.

Panels
------
A  context inset — the network's position within the continent
B  full domain   — all tile footprints, overlap depth, stations coloured by split
C+ island zooms  — the largest multi-tile islands, with tile-centre -> off-centre-station arrows

Usage
-----
    python plot_network_map.py --network TxSON
    python plot_network_map.py --network TxSON --islands 1 3 --out figures/txson
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pyproj import Transformer

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
SHP = next(iter(sorted(
    Path("/gpfs/home5/pkhanal/miniforge3/envs").glob(
        "*/lib/python3.1[01]/site-packages/pyogrio/tests/fixtures/"
        "naturalearth_lowres/naturalearth_lowres.shp"))), None)

SPLIT_CFG = {
    "train": dict(color="#2166ac", marker="o", label="train (seen in training)"),
    "val":   dict(color="#f4a620", marker="s", label="val (selected best.pt)"),
    "oos":   dict(color="#1a9850", marker="^", label="oos (fully held out)"),
}
# sequential, colour-blind safe; index = number of tiles covering the pixel
DEPTH_COLOURS = ["#f7fbff", "#d5e5f4", "#a7cae4", "#6aaed6", "#3080bd", "#08529c"]


def load(network: str, csv_dir: Path):
    tag = network.lower().replace(" ", "_")
    r = pd.read_csv(csv_dir / f"{tag}_readouts.csv")
    g = json.loads((csv_dir / f"{tag}_mosaic_grid.json").read_text())
    return r, g


def coverage_raster(grid: dict) -> np.ndarray:
    ny, nx = grid["shape"]["ny"], grid["shape"]["nx"]
    p = grid["patch_px"]
    cov = np.zeros((ny, nx), np.uint8)
    for off in grid["tile_offsets"].values():
        cov[off["row0"]:off["row0"] + p, off["col0"]:off["col0"] + p] += 1
    return cov


def utm_extent(grid: dict) -> tuple[float, float, float, float]:
    o = grid["origin_utm"]
    return o["west"], o["east"], o["south"], o["north"]


def station_utm(readouts: pd.DataFrame, epsg: int) -> pd.DataFrame:
    st = readouts.drop_duplicates("station")[
        ["station", "station_name", "station_split", "lat", "lon"]].copy()
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    st["x"], st["y"] = tr.transform(st.lon.values, st.lat.values)
    n_est = readouts.groupby("station").size()
    st["n_estimates"] = st.station.map(n_est)
    st["n_offcentre"] = st.station.map(
        readouts[~readouts.is_centre].groupby("station").size()).fillna(0).astype(int)
    return st


def tile_squares(grid: dict, ax, **kw):
    o = grid["origin_utm"]
    side = grid["patch_px"] * grid["res_m"]
    for off in grid["tile_offsets"].values():
        x = o["west"] + off["col0"] * grid["res_m"]
        y = o["north"] - off["row0"] * grid["res_m"] - side
        ax.add_patch(mpatches.Rectangle((x, y), side, side, fill=False, **kw))


# ---------------------------------------------------------------------------
def panel_context(ax, st: pd.DataFrame, network: str):
    if SHP is None:
        ax.text(.5, .5, "basemap unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    world = gpd.read_file(SHP)
    world.plot(ax=ax, color="#f0ede6", edgecolor="#b0b0b0", linewidth=0.4)
    ax.set_facecolor("#dbe9f6")
    lon0, lat0 = st.lon.mean(), st.lat.mean()
    ax.set_xlim(lon0 - 26, lon0 + 26)
    ax.set_ylim(lat0 - 18, lat0 + 18)
    ax.plot(lon0, lat0, marker="*", ms=17, color="#d62728", mec="white", mew=0.9, zorder=9)
    ax.annotate(f"{network}\n{lat0:.2f}°N, {abs(lon0):.2f}°W",
                (lon0, lat0), textcoords="offset points", xytext=(11, -20),
                fontsize=8.5, weight="bold",
                bbox=dict(fc="white", ec="#999", alpha=.85, pad=1.8, lw=.5))
    ax.set_title("A  Location", loc="left", fontsize=10, weight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=.3, alpha=.4, color="white")


def panel_domain(ax, grid: dict, st: pd.DataFrame, network: str):
    cov = coverage_raster(grid)
    west, east, south, north = utm_extent(grid)
    kmax = int(cov.max())
    cmap = ListedColormap(DEPTH_COLOURS[:kmax])
    cmap.set_under(alpha=0)
    im = ax.imshow(np.ma.masked_equal(cov, 0), extent=(west, east, south, north),
                   origin="upper", cmap=cmap,
                   norm=BoundaryNorm(np.arange(.5, kmax + 1.5), cmap.N), zorder=2)

    tile_squares(grid, ax, ec="#444444", lw=.6, zorder=3)
    for split, cfg in SPLIT_CFG.items():
        s = st[st.station_split == split]
        ax.scatter(s.x, s.y, c=cfg["color"], marker=cfg["marker"], s=42,
                   ec="white", lw=.6, zorder=6)
    # ring the stations that are ALSO predicted from a neighbour's tile
    extra = st[st.n_offcentre > 0]
    ax.scatter(extra.x, extra.y, facecolors="none", edgecolors="#d62728",
               s=120, lw=1.2, zorder=7)

    span_km = (east - west) / 1000
    ax.set_title(f"B  {network} domain · {len(st)} stations · "
                 f"{len(grid['tile_offsets'])} tiles of 2.24 km · "
                 f"{span_km:.0f} × {(north-south)/1000:.0f} km",
                 loc="left", fontsize=10, weight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("UTM easting (m)", fontsize=8)
    ax.set_ylabel("UTM northing (m)", fontsize=8)
    scale_bar(ax, west, south, east, north, 5)
    return im, kmax


def panel_island(ax, grid: dict, readouts: pd.DataFrame, st: pd.DataFrame,
                 island: int, label: str):
    o, res, side = grid["origin_utm"], grid["res_m"], grid["patch_px"] * grid["res_m"]
    tiles = [d for d, off in grid["tile_offsets"].items() if off["island"] == island]
    if not tiles:
        ax.set_axis_off()
        return

    xs, ys = [], []
    for d in tiles:
        off = grid["tile_offsets"][d]
        x = o["west"] + off["col0"] * res
        y = o["north"] - off["row0"] * res - side
        xs += [x, x + side]
        ys += [y, y + side]
        ax.add_patch(mpatches.Rectangle((x, y), side, side, fill=False,
                                        ec="#555", lw=.7, ls="--", zorder=3))

    sub = readouts[readouts.tile.isin(tiles)]
    pos = st.set_index("station")
    for r in sub[~sub.is_centre].itertuples():
        t, s = pos.loc[r.tile], pos.loc[r.station]
        ax.annotate("", xy=(s.x, s.y), xytext=(t.x, t.y), zorder=4,
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=.8, alpha=.55,
                                    shrinkA=5, shrinkB=5))

    members = st[st.station.isin(sub.station.unique())]
    for split, cfg in SPLIT_CFG.items():
        m = members[members.station_split == split]
        ax.scatter(m.x, m.y, c=cfg["color"], marker=cfg["marker"], s=58,
                   ec="white", lw=.7, zorder=6)
    # rotate the label anchor through 4 positions down the y-axis so near-coincident
    # stations (TxSON has pairs 180 m apart) do not overprint each other
    PLACES = [(8, 5, "left"), (-8, -12, "right"), (8, -12, "left"), (-8, 5, "right")]
    for k, m in enumerate(members.sort_values("y", ascending=False).itertuples()):
        dx, dy, ha = PLACES[k % 4]
        ax.annotate(m.station_name, (m.x, m.y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=6.6, zorder=8,
                    bbox=dict(fc="white", ec="none", alpha=.7, pad=.9))

    pad = 160
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)
    n_off = int((~sub.is_centre).sum())
    splits = "/".join(sorted(set(members.station_split)))
    ax.set_title(f"{label}  island {island} · {len(tiles)} tiles · {len(members)} stations\n"
                 f"    {n_off} off-centre readouts · {splits}",
                 loc="left", fontsize=9, weight="bold", pad=6)
    scale_bar(ax, *ax.get_xlim(), *ax.get_ylim(), 1)


def scale_bar(ax, x0, x1, y0, y1, km):
    w = km * 1000
    x = x0 + (x1 - x0) * .06
    y = y0 + (y1 - y0) * .06
    ax.add_patch(mpatches.Rectangle((x, y), w, (y1 - y0) * .008,
                                    fc="black", ec="black", zorder=9))
    ax.annotate(f"{km} km", (x + w / 2, y), textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=7, zorder=9)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="TxSON")
    ap.add_argument("--csv-dir", default=str(REPO / "csvs"))
    ap.add_argument("--out", default=None, help="output stem (default figures/{network}_map)")
    ap.add_argument("--islands", type=int, nargs="*", default=None,
                    help="island ids to zoom (default: the 2 largest multi-tile islands)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    readouts, grid = load(args.network, Path(args.csv_dir))
    st = station_utm(readouts, grid["epsg"])

    islands = args.islands
    if not islands:
        islands = [i["island"] for i in grid["islands"] if i["n_tiles"] > 1][:2]

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[.85, 1.3, 1.3],
                  height_ratios=[1.1, 1], hspace=.30, wspace=.26,
                  left=.05, right=.985, top=.915, bottom=.06)

    ax_ctx = fig.add_subplot(gs[0, 0])
    ax_dom = fig.add_subplot(gs[0, 1:])
    panel_context(ax_ctx, st, args.network)
    im, kmax = panel_domain(ax_dom, grid, st, args.network)

    for k, isl in enumerate(islands[:2]):
        panel_island(fig.add_subplot(gs[1, k]), grid, readouts, st, isl, "CD"[k])

    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.set_axis_off()
    cov = grid["coverage"]
    handles = [Line2D([], [], ls="", marker=c["marker"], color=c["color"], ms=8,
                      mec="white", label=c["label"]) for c in SPLIT_CFG.values()]
    handles += [
        Line2D([], [], ls="", marker="o", mfc="none", mec="#d62728", ms=13, mew=1.5,
               label="also predicted from a neighbour's tile"),
        Line2D([], [], color="#d62728", lw=1, alpha=.6,
               label="tile centre → off-centre station"),
        mpatches.Patch(fc="none", ec="#555", ls="--", label="2.24 km tile footprint"),
    ]
    ax_leg.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5,
                  bbox_to_anchor=(-.05, 1.02))
    n_off = int((~readouts.is_centre).sum())
    ax_leg.text(-.05, .46, (
        f"readouts        {len(readouts)}\n"
        f"  centre pixel  {int(readouts.is_centre.sum())}  (supervised)\n"
        f"  off-centre    {n_off}  (never supervised)\n\n"
        f"tile coverage   {cov['covered_km2']} km² "
        f"({100*cov['covered_frac']:.1f}% of bbox)\n"
        f"  ≥2 tiles     {100*cov['multi_frac_of_covered']:.1f}% of covered\n"
        f"  islands       {len(grid['islands'])}\n\n"
        f"estimates/station\n  " + "  ".join(
            f"{k}×:{v}" for k, v in
            sorted(readouts.groupby('station').size().value_counts().items()))),
        transform=ax_leg.transAxes, fontsize=8.2, family="monospace", va="top")

    cax = ax_dom.inset_axes([1.025, .04, .017, .40])
    cb = fig.colorbar(im, cax=cax, orientation="vertical",
                      ticks=np.arange(1, kmax + 1))
    cb.set_label("tiles covering the pixel", fontsize=7.5, labelpad=3)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(.5)

    fig.suptitle(
        f"{args.network}: testing the model at pixels it was never centred on",
        x=.05, ha="left", fontsize=13.5, weight="bold", y=.972)

    out = Path(args.out) if args.out else REPO / "figures" / f"{args.network.lower()}_map"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
