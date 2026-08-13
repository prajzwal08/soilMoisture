"""Build the (tile, station) pixel-readout table for a dense station network.

Every station in this project is predicted at exactly one pixel — (112, 112) of its own
224x224 @ 10 m tile.  In a dense network the tiles overlap, so one tile's 224x224 map also
covers *other* stations.  This script finds, for every tile, which stations of the network
fall inside it and at which pixel.

Outputs
-------
csvs/{network}_readouts.csv     one row per (tile, station) pair that lands inside the tile
csvs/{network}_mosaic_grid.json mosaic geometry: origin, size, per-tile paste offsets, islands

Usage
-----
    python build_network_readouts.py --network TxSON
    python build_network_readouts.py --network TxSON --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage

REPO       = Path(__file__).resolve().parent
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
SAT_ZARR   = Path("/projects/prjs1968/satellite_zarr")
OUT_DIR    = REPO / "csvs"

PATCH_PX = 224     # tile side in pixels          (download_s2_mpc.py:57)
RES_M    = 10      # ground sampling distance     (download_s2_mpc.py:58)
CENTRE   = 112     # supervised pixel             (model.py:348-349)


# ---------------------------------------------------------------------------
# Station table
# ---------------------------------------------------------------------------
def station_dir(row: pd.Series) -> str:
    """On-disk directory name.  Mirrors dataset.py:819-823 exactly."""
    if str(row["source_network"]) == "ISMN":
        return f"ISMN_{row['network']}_{row['station_name']}"
    return f"{row['source_network']}_{row['station_id']}"


def load_network(network: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_CSV)
    sub = df[df["network"] == network].copy()
    if sub.empty:
        sys.exit(f"no stations with network == {network!r} in {SPLITS_CSV}")
    sub["dir"] = sub.apply(station_dir, axis=1)
    return sub.reset_index(drop=True)


def load_tile_geometry(dirs: list[str]) -> dict[str, dict]:
    """epsg + bounds_utm per station, straight from the satellite zarr attrs."""
    geo = {}
    for d in dirs:
        p = SAT_ZARR / f"{d}.zarr" / ".zattrs"
        if not p.exists():
            print(f"  WARNING no tile on disk for {d} — excluded")
            continue
        a = json.loads(p.read_text())
        geo[d] = {"epsg": int(a["epsg"]), "bounds": [float(v) for v in a["bounds_utm"]]}
    return geo


# ---------------------------------------------------------------------------
# Projection: station lat/lon -> pixel inside a given tile
# ---------------------------------------------------------------------------
def station_pixel(lon: float, lat: float, epsg: int, bounds: list[float],
                  _cache: dict = {}) -> tuple[int, int, float, float]:
    """Return (row, col, x, y).  row/col may fall outside [0, 224)."""
    tr = _cache.get(epsg)
    if tr is None:
        tr = _cache[epsg] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tr.transform(lon, lat)
    west, _south, _east, north = bounds
    col = int(np.floor((x - west) / RES_M))
    row = int(np.floor((north - y) / RES_M))
    return row, col, x, y


def build_readouts(tx: pd.DataFrame, geo: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, tile in tx.iterrows():
        g = geo.get(tile["dir"])
        if g is None:
            continue
        epsg, bounds = g["epsg"], g["bounds"]
        tx_row, tx_col, cx, cy = station_pixel(tile.longitude, tile.latitude, epsg, bounds)
        for _, st in tx.iterrows():
            row, col, x, y = station_pixel(st.longitude, st.latitude, epsg, bounds)
            if not (0 <= row < PATCH_PX and 0 <= col < PATCH_PX):
                continue
            rows.append({
                "tile":          tile["dir"],
                "tile_station":  tile.station_name,
                "tile_split":    tile.split,
                "tile_epsg":     epsg,
                "station":       st["dir"],
                "station_name":  st.station_name,
                "station_split": st.split,
                "row":           row,
                "col":           col,
                "drow":          row - CENTRE,
                "dcol":          col - CENTRE,
                "offset_px":     int(round(np.hypot(row - CENTRE, col - CENTRE))),
                "dist_m":        round(float(np.hypot(x - cx, y - cy)), 1),
                "is_centre":     bool(st["dir"] == tile["dir"]),
                "lat":           float(st.latitude),
                "lon":           float(st.longitude),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Greedy minimum tile cover  (shipped as a flag; NOT used to skip compute)
# ---------------------------------------------------------------------------
def greedy_min_cover(readouts: pd.DataFrame) -> set[str]:
    members = {t: set(g.station) for t, g in readouts.groupby("tile")}
    uncovered = set(readouts.station.unique())
    chosen: set[str] = set()
    while uncovered:
        tile, gain = max(
            ((t, len(uncovered & m)) for t, m in members.items() if t not in chosen),
            key=lambda kv: kv[1],
            default=(None, 0),
        )
        if not tile or gain == 0:
            break
        chosen.add(tile)
        uncovered -= members[tile]
    return chosen


# ---------------------------------------------------------------------------
# Mosaic geometry + island labelling
# ---------------------------------------------------------------------------
def build_mosaic_grid(tx: pd.DataFrame, geo: dict[str, dict]) -> dict:
    epsgs = {g["epsg"] for g in geo.values()}
    if len(epsgs) != 1:
        print(f"  WARNING tiles span {len(epsgs)} UTM zones {sorted(epsgs)} — "
              f"mosaicking only the majority zone")
    epsg = pd.Series([g["epsg"] for g in geo.values()]).mode()[0]
    tiles = {d: g for d, g in geo.items() if g["epsg"] == epsg}

    west  = min(g["bounds"][0] for g in tiles.values())
    north = max(g["bounds"][3] for g in tiles.values())
    east  = max(g["bounds"][2] for g in tiles.values())
    south = min(g["bounds"][1] for g in tiles.values())
    nx = int(round((east - west) / RES_M))
    ny = int(round((north - south) / RES_M))

    offsets, cover = {}, np.zeros((ny, nx), np.uint8)
    for d, g in tiles.items():
        c0 = int(round((g["bounds"][0] - west) / RES_M))
        r0 = int(round((north - g["bounds"][3]) / RES_M))
        offsets[d] = {"row0": r0, "col0": c0}
        cover[r0:r0 + PATCH_PX, c0:c0 + PATCH_PX] += 1

    labels, n_islands = ndimage.label(cover > 0)
    for d, off in offsets.items():
        off["island"] = int(labels[off["row0"] + CENTRE, off["col0"] + CENTRE])

    covered = int((cover > 0).sum())
    islands = []
    for i in range(1, n_islands + 1):
        m = labels == i
        ys, xs = np.where(m)
        members = sorted(d for d, o in offsets.items() if o["island"] == i)
        islands.append({
            "island":  i,
            "km2":     round(float(m.sum()) * RES_M ** 2 / 1e6, 2),
            "h_km":    round(float(ys.max() - ys.min() + 1) * RES_M / 1000, 2),
            "w_km":    round(float(xs.max() - xs.min() + 1) * RES_M / 1000, 2),
            "n_tiles": len(members),
            "tiles":   members,
        })
    islands.sort(key=lambda d: -d["km2"])

    return {
        "epsg": int(epsg),
        "origin_utm": {"west": west, "north": north, "east": east, "south": south},
        "shape": {"ny": ny, "nx": nx},
        "res_m": RES_M,
        "patch_px": PATCH_PX,
        "tile_offsets": offsets,
        "coverage": {
            "covered_px":     covered,
            "total_px":       int(nx * ny),
            "covered_frac":   round(covered / (nx * ny), 4),
            "covered_km2":    round(covered * RES_M ** 2 / 1e6, 2),
            "multi_px":       int((cover > 1).sum()),
            "multi_frac_of_covered": round(float((cover > 1).sum()) / covered, 4),
            "depth_histogram": {int(k): int(v)
                                for k, v in zip(*np.unique(cover[cover > 0], return_counts=True))},
        },
        "islands": islands,
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def selftest(readouts: pd.DataFrame) -> int:
    fails = 0

    centre = readouts[readouts.is_centre]
    bad = centre[(centre.row != CENTRE) | (centre.col != CENTRE)]
    if len(bad):
        fails += 1
        print(f"  FAIL own-tile centre: {len(bad)} stations not at ({CENTRE},{CENTRE})")
        print(bad[["tile", "row", "col"]].to_string(index=False))
    else:
        print(f"  PASS own-tile centre: all {len(centre)} stations land exactly at "
              f"({CENTRE},{CENTRE})")

    idx = {(r.tile, r.station): r for r in readouts.itertuples()}
    checked = worst = 0
    for (a, b), fwd in idx.items():
        rev = idx.get((b, a))
        if rev is None or a == b:
            continue
        checked += 1
        worst = max(worst, abs(fwd.drow + rev.drow), abs(fwd.dcol + rev.dcol))
    if worst > 1:
        fails += 1
        print(f"  FAIL symmetry: max |offset_A->B + offset_B->A| = {worst} px (> 1)")
    else:
        print(f"  PASS symmetry: {checked} reciprocal pairs, max residual {worst} px")

    return fails


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="TxSON", help="value of the `network` column")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--selftest", action="store_true",
                    help="run geometry checks and exit non-zero on failure")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.network.lower().replace(" ", "_")

    print(f"=== {args.network} readout table ===")
    tx = load_network(args.network)
    print(f"stations in splits csv : {len(tx)}  ({dict(tx.split.value_counts())})")

    geo = load_tile_geometry(tx["dir"].tolist())
    print(f"tiles on disk          : {len(geo)}")

    readouts = build_readouts(tx, geo)
    if readouts.empty:
        sys.exit("no readouts produced — check network name and tile availability")

    cover = greedy_min_cover(readouts)
    readouts["in_min_cover"] = readouts.tile.isin(cover)

    grid = build_mosaic_grid(tx, geo)
    island_of = {d: o["island"] for d, o in grid["tile_offsets"].items()}
    readouts["island"] = readouts.tile.map(island_of)

    readouts = readouts.sort_values(["tile", "offset_px"]).reset_index(drop=True)

    print(f"\nreadouts               : {len(readouts)}  "
          f"({int(readouts.is_centre.sum())} centre + "
          f"{int((~readouts.is_centre).sum())} off-centre)")
    print(f"estimates per station  : "
          f"{dict(readouts.groupby('station').size().value_counts().sort_index())}")
    print(f"greedy minimum cover   : {len(cover)} tiles cover all "
          f"{readouts.station.nunique()} stations (of {readouts.tile.nunique()} tiles)")

    cov = grid["coverage"]
    print(f"\nmosaic  epsg={grid['epsg']}  {grid['shape']['ny']} x {grid['shape']['nx']} px "
          f"@ {RES_M} m")
    print(f"  covered   : {cov['covered_km2']} km2 = {100*cov['covered_frac']:.1f}% of bbox")
    print(f"  >=2 tiles : {cov['multi_px']} px = "
          f"{100*cov['multi_frac_of_covered']:.1f}% of covered")
    print(f"  depth hist: {cov['depth_histogram']}")
    print(f"  islands   : {len(grid['islands'])}")
    for isl in grid["islands"][:5]:
        print(f"     #{isl['island']:<3d} {isl['km2']:>6.2f} km2  "
              f"{isl['h_km']:.2f} x {isl['w_km']:.2f} km  {isl['n_tiles']} tiles")

    if args.selftest:
        print("\n=== self-tests ===")
        if selftest(readouts):
            sys.exit(1)

    r_path = out_dir / f"{tag}_readouts.csv"
    g_path = out_dir / f"{tag}_mosaic_grid.json"
    readouts.to_csv(r_path, index=False)
    g_path.write_text(json.dumps(grid, indent=1))
    print(f"\nwrote {r_path}")
    print(f"wrote {g_path}")


if __name__ == "__main__":
    main()
