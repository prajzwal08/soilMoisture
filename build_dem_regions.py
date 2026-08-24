"""
Define the processing regions for the wide-DEM terrain build (§32.3).

The processing unit for TWI/HAND is a REGION, not a station: flow accumulation is
non-local, so every station must sit inside one continuous flow field with its
catchment well away from the edge. Regions come from single-linkage clustering of
the station coordinates — single linkage is the right rule because it merges two
groups whenever ANY pair is within the threshold, which is exactly the condition
under which two stations could share upslope area.

Distances are computed in 3-D geocentric (ECEF) space, so the antimeridian and the
poles need no special casing; only the per-region bounding box does, and that is
built in a per-region equal-area projection rather than in degrees.

Each region gets its own Lambert Azimuthal Equal Area projection at exactly 30 m.
A UTM zone is ~500 km wide and the largest region is ~560 km across, so UTM cannot
carry it; LAEA centred on the region keeps cells square in metres, which flow
routing and Horn slope both assume.

Outputs:
  csvs/dem_regions.csv        one row per region: LAEA proj string, snapped 30 m
                              grid bounds, size, cell count, GLO-30 tiles needed
  csvs/station_dem_region.csv one row per station: region_id + LAEA x/y + the
                              station's own 2.24 km tile bounds in region coords

Usage:
  python build_dem_regions.py                      # 50 km linkage, 10 km buffer
  python build_dem_regions.py --buffer-km 20       # Tier-2 buffer-doubling check
  python build_dem_regions.py --linkage-km 25 --dry-run   # size an alternative
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

REPO         = Path(__file__).resolve().parent
STATION_CSV  = REPO / "csvs" / "station_splits.csv"
OUT_REGIONS  = REPO / "csvs" / "dem_regions.csv"
OUT_STATIONS = REPO / "csvs" / "station_dem_region.csv"

LINKAGE_KM   = 50.0    # §32.3: 353 regions, 0.84e9 cells, largest 559 km
BUFFER_KM    = 10.0    # margin beyond the station tiles, per region edge
RES_M        = 30.0    # GLO-30 native resolution; the LAEA grid is exactly this
TILE_HALF_M  = 1120.0  # station tile is 2.24 km across (70 x 32 m)

# GLO-30 is decimated in longitude poleward, so cells are never square in metres
# and the tiling is 1 degree x 1 degree regardless. Bands matter for the download,
# not here, but record where a region crosses one so download_wide_dem.py can warn.
GLO30_LAT_BANDS = (50.0, 60.0, 70.0, 80.0)


def load_stations() -> pd.DataFrame:
    """Station ids in the same folder convention the rest of the pipeline uses."""
    df = pd.read_csv(STATION_CSV)

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"

    df["station_id"] = df.apply(_folder, axis=1)
    return df[["station_id", "latitude", "longitude"]].reset_index(drop=True)


def cluster_regions(lat: np.ndarray, lon: np.ndarray, linkage_km: float) -> np.ndarray:
    """
    Single-linkage clustering on ECEF chord distance. Returns 1-based labels.

    Chord vs great-circle: at 50 km the chord is short by 0.5 mm, so the
    threshold is used directly rather than corrected.
    """
    to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x, y, z = to_ecef.transform(lon, lat, np.zeros_like(lat))
    pts = np.column_stack([x, y, z])
    Z = linkage(pdist(pts), method="single")
    return fcluster(Z, t=linkage_km * 1000.0, criterion="distance")


def region_centre(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Spherical centroid via ECEF mean — safe across the antimeridian."""
    latr, lonr = np.radians(lat), np.radians(lon)
    x = np.mean(np.cos(latr) * np.cos(lonr))
    y = np.mean(np.cos(latr) * np.sin(lonr))
    z = np.mean(np.sin(latr))
    return (math.degrees(math.atan2(z, math.hypot(x, y))),
            math.degrees(math.atan2(y, x)))


def laea_proj(c_lat: float, c_lon: float) -> str:
    return (f"+proj=laea +lat_0={c_lat:.6f} +lon_0={c_lon:.6f} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def snap_bounds(xmin, ymin, xmax, ymax, res):
    """Snap outward to a res-aligned grid so the raster grid is exactly on res."""
    return (math.floor(xmin / res) * res, math.floor(ymin / res) * res,
            math.ceil(xmax / res) * res,  math.ceil(ymax / res) * res)


def wgs84_envelope(proj: str, xmin, ymin, xmax, ymax, n=200):
    """
    Densely sample the LAEA rectangle's boundary and unproject, because the
    rectangle's WGS84 envelope is bounded by its curved edges, not its corners.
    """
    inv = Transformer.from_crs(CRS.from_proj4(proj), "EPSG:4326", always_xy=True)
    t = np.linspace(0.0, 1.0, n)
    xs = np.concatenate([xmin + t * (xmax - xmin), np.full(n, xmax),
                         xmax - t * (xmax - xmin), np.full(n, xmin)])
    ys = np.concatenate([np.full(n, ymin), ymin + t * (ymax - ymin),
                         np.full(n, ymax), ymax - t * (ymax - ymin)])
    lon, lat = inv.transform(xs, ys)
    return lon, lat


def glo30_tiles(lon: np.ndarray, lat: np.ndarray) -> tuple[list[str], bool]:
    """
    1-degree GLO-30 tile names covering the boundary envelope, plus a flag for
    antimeridian straddle. Tiles are named by their SW corner.
    """
    straddles = (lon.max() - lon.min()) > 180.0
    if straddles:
        lon = np.where(lon < 0, lon + 360.0, lon)

    lo_lon = int(math.floor(lon.min()))
    hi_lon = int(math.floor(lon.max()))
    lo_lat = int(math.floor(lat.min()))
    hi_lat = int(math.floor(lat.max()))

    names = []
    for ilat in range(lo_lat, hi_lat + 1):
        for ilon in range(lo_lon, hi_lon + 1):
            wlon = ((ilon + 180) % 360) - 180
            ns = "N" if ilat >= 0 else "S"
            ew = "E" if wlon >= 0 else "W"
            names.append(f"{ns}{abs(ilat):02d}{ew}{abs(wlon):03d}")
    return names, straddles


def main() -> None:
    ap = argparse.ArgumentParser(description="Define wide-DEM processing regions")
    ap.add_argument("--linkage-km", type=float, default=LINKAGE_KM)
    ap.add_argument("--buffer-km", type=float, default=BUFFER_KM)
    ap.add_argument("--res-m", type=float, default=RES_M)
    ap.add_argument("--out-regions", type=Path, default=OUT_REGIONS)
    ap.add_argument("--out-stations", type=Path, default=OUT_STATIONS)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the size table only; write nothing.")
    args = ap.parse_args()

    st = load_stations()
    lat = st["latitude"].to_numpy(float)
    lon = st["longitude"].to_numpy(float)
    print(f"{len(st)} stations from {STATION_CSV.name}")

    labels = cluster_regions(lat, lon, args.linkage_km)
    uniq = np.unique(labels)
    print(f"single linkage @ {args.linkage_km:g} km -> {len(uniq)} regions")

    # Renumber by descending station count so region_id 0 is the biggest cluster.
    order = sorted(uniq, key=lambda u: (-(labels == u).sum(), u))
    remap = {old: new for new, old in enumerate(order)}
    st["region_id"] = [remap[l] for l in labels]

    # margin = station tile half-width + the requested buffer beyond it
    margin_m = TILE_HALF_M + args.buffer_km * 1000.0

    region_rows, station_rows = [], []
    for rid, grp in st.groupby("region_id", sort=True):
        glat = grp["latitude"].to_numpy(float)
        glon = grp["longitude"].to_numpy(float)
        c_lat, c_lon = region_centre(glat, glon)
        proj = laea_proj(c_lat, c_lon)

        fwd = Transformer.from_crs("EPSG:4326", CRS.from_proj4(proj), always_xy=True)
        sx, sy = fwd.transform(glon, glat)

        xmin, ymin, xmax, ymax = snap_bounds(
            sx.min() - margin_m, sy.min() - margin_m,
            sx.max() + margin_m, sy.max() + margin_m, args.res_m)

        width_px  = int(round((xmax - xmin) / args.res_m))
        height_px = int(round((ymax - ymin) / args.res_m))

        elon, elat = wgs84_envelope(proj, xmin, ymin, xmax, ymax)
        tiles, straddles = glo30_tiles(elon, elat)

        # A region spanning a GLO-30 longitude-decimation boundary mosaics tiles
        # of differing pixel width; the LAEA warp absorbs it but flag it anyway.
        crosses_band = any(elat.min() < b < elat.max() for b in GLO30_LAT_BANDS)

        region_rows.append({
            "region_id":     rid,
            "n_stations":    len(grp),
            "centre_lat":    round(c_lat, 6),
            "centre_lon":    round(c_lon, 6),
            "laea_proj4":    proj,
            "res_m":         args.res_m,
            "x_min":         xmin, "y_min": ymin, "x_max": xmax, "y_max": ymax,
            "width_px":      width_px,
            "height_px":     height_px,
            "width_km":      round((xmax - xmin) / 1000.0, 2),
            "height_km":     round((ymax - ymin) / 1000.0, 2),
            "n_cells":       width_px * height_px,
            "dem_gb":        round(width_px * height_px * 4 / 1e9, 3),
            "lat_min":       round(float(elat.min()), 6),
            "lat_max":       round(float(elat.max()), 6),
            "lon_min":       round(float(elon.min()), 6),
            "lon_max":       round(float(elon.max()), 6),
            "n_glo30_tiles": len(tiles),
            "glo30_tiles":   ";".join(tiles),
            "antimeridian":  straddles,
            "crosses_lat_band": crosses_band,
            "buffer_km":     args.buffer_km,
            "linkage_km":    args.linkage_km,
        })

        for (sid, la, lo, x, y) in zip(grp["station_id"], glat, glon, sx, sy):
            station_rows.append({
                "station_id": sid, "region_id": rid,
                "latitude": la, "longitude": lo,
                "laea_x": round(float(x), 3), "laea_y": round(float(y), 3),
                # station tile in region grid coords, snapped to the same 30 m grid
                "tile_x_min": math.floor((x - TILE_HALF_M) / args.res_m) * args.res_m,
                "tile_y_min": math.floor((y - TILE_HALF_M) / args.res_m) * args.res_m,
                "tile_x_max": math.ceil((x + TILE_HALF_M) / args.res_m) * args.res_m,
                "tile_y_max": math.ceil((y + TILE_HALF_M) / args.res_m) * args.res_m,
            })

    reg = pd.DataFrame(region_rows)
    sta = pd.DataFrame(station_rows)

    total_cells = int(reg["n_cells"].sum())
    biggest = reg.loc[reg["n_cells"].idxmax()]
    print(f"cells      {total_cells/1e9:.3f}e9   DEM {reg['dem_gb'].sum():.1f} GB float32")
    print(f"largest    region {int(biggest['region_id'])}: "
          f"{biggest['width_km']:.0f} x {biggest['height_km']:.0f} km, "
          f"{biggest['n_cells']/1e6:.0f}e6 cells, {biggest['dem_gb']:.2f} GB "
          f"({int(biggest['n_stations'])} stations)")
    print(f"GLO-30     {len(set(t for ts in reg['glo30_tiles'] for t in ts.split(';')))} "
          f"distinct 1-deg tiles, {int(reg['n_glo30_tiles'].sum())} tile-fetches")
    print(f"flags      antimeridian={int(reg['antimeridian'].sum())}  "
          f"crosses_lat_band={int(reg['crosses_lat_band'].sum())}")
    print("stations/region: " + "  ".join(
        f"{k}={v}" for k, v in reg["n_stations"].describe()[["min", "50%", "max"]].items()))

    if args.dry_run:
        print("dry run — nothing written")
        return

    args.out_regions.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(args.out_regions, index=False)
    sta.to_csv(args.out_stations, index=False)
    print(f"wrote {args.out_regions}  ({len(reg)} regions)")
    print(f"wrote {args.out_stations}  ({len(sta)} stations)")


if __name__ == "__main__":
    main()
