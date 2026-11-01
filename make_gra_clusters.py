#!/usr/bin/env python
"""Step 1 of the grassland thermal-figure plan: build the cluster table.

Which stations share a tile, which tile to render them in, and where each one lands in
that tile's pixel grid.

GEOGRAPHIC CLUSTERING, not `location_group_id`.  That column is near-unique per station
-- the six TxSON probes inside one window carry ids 744/742/739/735/736/746 -- so
grouping on it finds nothing (39.2).  Stations are linked when within --link-km of each
other (tiles are 2.24 km wide, so 1.12 km guarantees the two windows overlap by at least
half) and connected components of that graph are the clusters.

Two outputs:

  csvs/gra_thermal_clusters.csv   one row per cluster, including the representative
                                  station whose 2.24 km window the figure is drawn in
  csvs/gra_thermal_members.csv    one row per station: its pixel position in that
                                  window, its 160 m patch index, its 70 m ECOSTRESS cell

The patch index is the point.  "Which pairs are on different patches" is a question
about grid cells, not about distance -- two probes 170 m apart may share a patch or not
depending on where the grid lines fall, because two points inside one 160 m patch can be
up to 160*sqrt(2) = 226 m apart.

Env: terramind (pandas, numpy, pyproj).  Login node runs nothing -- submit it.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

REPO       = Path("/gpfs/work3/0/prjs1968/soilMoisture")
SAT_ZARR   = Path("/projects/prjs1968/satellite_zarr")
SPLITS     = REPO / "csvs" / "station_splits.csv"
BUNDLES    = REPO / "csvs" / "ecostress_dtr_bundles.all.csv"

TILE_PX     = 224      # S2 grid
TILE_M      = 2240.0
S2_PIXEL_M  = 10.0
PATCH_PX    = 16       # 160 m TerraMind patch = 16 S2 pixels
TOKEN_GRID  = 14
ECO_PIXEL_M = 70.0     # 32 x 32 over the same 2240 m

log = logging.getLogger("clusters")


def tile_attrs(folder: str) -> dict | None:
    """Root .zattrs of the raw-imagery store: epsg, bounds_utm, pixel_size_m.

    Read as plain JSON -- that store has no .zmetadata, and we need nothing but attrs.
    """
    p = SAT_ZARR / f"{folder}.zarr" / ".zattrs"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    if "epsg" not in a or "bounds_utm" not in a:
        return None
    return a


def local_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat-earth km offsets, same approximation as analyse_s1_spatial_broad.py."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    dy = (la[:, None] - la[None, :]) * 6371.0
    dx = (lo[:, None] - lo[None, :]) * 6371.0 * np.cos(la.mean())
    return dx, dy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--igbp", default="GRA")
    ap.add_argument("--link-km", type=float, default=1.12)
    ap.add_argument("--min-usable", type=int, default=1,
                    help="minimum n_pairs_usable in the ECOSTRESS bundle")
    ap.add_argument("--out-clusters", default=str(REPO / "csvs" / "gra_thermal_clusters.csv"))
    ap.add_argument("--out-members",  default=str(REPO / "csvs" / "gra_thermal_members.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # pandas, never awk -- 6 AmeriFlux rows carry quoted commas in station_name and a
    # naive split shifts every later field (IGBP among them).
    st = pd.read_csv(SPLITS)
    log.info("station_splits: %d rows", len(st))

    st = st[st["IGBP"].astype(str) == args.igbp].copy()
    log.info("IGBP == %s: %d stations", args.igbp, len(st))

    bun = pd.read_csv(BUNDLES)
    bun = bun[bun["n_pairs_usable"] >= args.min_usable]
    log.info("bundles with >= %d usable pair(s): %d", args.min_usable, len(bun))

    df = st.merge(bun[["station_id", "folder", "category", "n_pairs",
                       "n_pairs_usable", "median_valid_px"]],
                  on="station_id", how="inner")
    log.info("%s stations WITH a usable ECOSTRESS bundle: %d", args.igbp, len(df))
    if df.empty:
        raise SystemExit("nothing to cluster")

    df = df.reset_index(drop=True)
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    dx, dy = local_xy(lat, lon)
    D = np.sqrt(dx ** 2 + dy ** 2)
    adj = D <= args.link_km

    seen, comps = set(), []
    for i in range(len(df)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack.extend(np.where(adj[k])[0].tolist())
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    log.info("connected components at link <= %.2f km: %d  (sizes %s)",
             args.link_km, len(comps), [len(c) for c in comps])

    crows, mrows = [], []
    for ci, comp in enumerate(comps):
        sub = df.iloc[comp]
        net = str(sub["network"].iloc[0])
        cid = f"{net}_{len(comp)}st_{ci:02d}"
        ext = float(D[np.ix_(comp, comp)].max()) if len(comp) > 1 else 0.0

        # Representative = the member whose own 2.24 km window contains the most others.
        # Ties broken by usable-pair count, then by month coverage proxy (n_pairs).
        half_km = TILE_M / 2000.0
        inside = [(int(((np.abs(dx[np.ix_([k], comp)]) <= half_km) &
                        (np.abs(dy[np.ix_([k], comp)]) <= half_km)).sum()),
                   int(df.at[k, "n_pairs_usable"]), int(df.at[k, "n_pairs"]), k)
                  for k in comp]
        inside.sort(reverse=True)
        rep = int(inside[0][3])
        rep_id, rep_folder = df.at[rep, "station_id"], df.at[rep, "folder"]

        att = tile_attrs(rep_folder)
        if att is None:
            log.warning("%s: no usable .zattrs for rep %s -- trying other members",
                        cid, rep_folder)
            for _, _, _, k in inside[1:]:
                att = tile_attrs(df.at[k, "folder"])
                if att is not None:
                    rep = int(k)
                    rep_id, rep_folder = df.at[rep, "station_id"], df.at[rep, "folder"]
                    break
        if att is None:
            log.error("%s: NO member has a raw-imagery store -- skipped", cid)
            continue

        epsg = int(att["epsg"])
        w, s, e, n = [float(v) for v in att["bounds_utm"]]
        fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

        n_in = 0
        for k in comp:
            x, y = fwd.transform(float(df.at[k, "longitude"]), float(df.at[k, "latitude"]))
            col = (x - w) / S2_PIXEL_M
            row = (n - y) / S2_PIXEL_M                     # row 0 at the north edge
            ir, ic = int(math.floor(row)), int(math.floor(col))
            in_tile = (0 <= ir < TILE_PX) and (0 <= ic < TILE_PX)
            n_in += bool(in_tile)
            mrows.append(dict(
                cluster_id=cid, station_id=df.at[k, "station_id"],
                folder=df.at[k, "folder"], category=df.at[k, "category"],
                network=df.at[k, "network"], is_rep=int(k == rep),
                latitude=float(df.at[k, "latitude"]),
                longitude=float(df.at[k, "longitude"]),
                utm_x=float(x), utm_y=float(y), epsg=epsg,
                row=ir, col=ic, in_tile=int(in_tile),
                patch_row=ir // PATCH_PX if in_tile else -1,
                patch_col=ic // PATCH_PX if in_tile else -1,
                patch_idx=((ir // PATCH_PX) * TOKEN_GRID + (ic // PATCH_PX))
                          if in_tile else -1,
                eco_row=int(math.floor(row * S2_PIXEL_M / ECO_PIXEL_M)) if in_tile else -1,
                eco_col=int(math.floor(col * S2_PIXEL_M / ECO_PIXEL_M)) if in_tile else -1,
                dist_m_from_rep=float(np.hypot(dx[rep, k], dy[rep, k]) * 1000.0),
                n_pairs_usable=int(df.at[k, "n_pairs_usable"]),
                median_valid_px=float(df.at[k, "median_valid_px"]),
            ))

        crows.append(dict(
            cluster_id=cid, network=net, n_stations=len(comp),
            members=";".join(df.iloc[comp]["station_id"].tolist()),
            extent_km=round(ext, 3),
            centroid_lat=float(sub["latitude"].mean()),
            centroid_lon=float(sub["longitude"].mean()),
            rep_station=rep_id, rep_folder=rep_folder, rep_epsg=epsg,
            members_inside_tile=n_in,
            total_usable_pairs=int(sub["n_pairs_usable"].sum()),
        ))

    C = pd.DataFrame(crows).sort_values(["n_stations", "cluster_id"],
                                        ascending=[False, True])
    M = pd.DataFrame(mrows)
    Path(args.out_clusters).parent.mkdir(parents=True, exist_ok=True)
    C.to_csv(args.out_clusters, index=False)
    M.to_csv(args.out_members, index=False)

    log.info("")
    log.info("=== clusters (%d) ===", len(C))
    for _, r in C.iterrows():
        flag = "" if r.members_inside_tile == r.n_stations else \
               f"   <-- only {r.members_inside_tile}/{r.n_stations} fit the 2.24 km window"
        log.info("  %-22s n=%2d  extent %5.2f km  rep %-24s pairs %3d%s",
                 r.cluster_id, r.n_stations, r.extent_km, r.rep_station,
                 r.total_usable_pairs, flag)

    # Pair separability.  TWO dimensions, because neither alone is the answer.
    #
    # Grid separability is a patch index, not a distance: two points inside one 160 m
    # patch can be up to 160*sqrt(2) = 226 m apart, so distance alone is ambiguous
    # between 160 and 226 m.  But the converse artifact is sharper and matters more --
    # VairaRanch and US-Var are 6 m apart and land in DIFFERENT patches purely because a
    # grid line falls between rows 111 and 112.  Counting that as "resolvable at 160 m"
    # would be nonsense: at 6 m any difference between two probes is installation and
    # instrument noise, not a spatial gradient (35.30 measured SOD071/SOD073 at 9.0 m
    # disagreeing by 0.058).  So the two are crossed and the artifact is named.
    log.info("")
    log.info("=== within-cluster pairs: grid cell x physical separation ===")
    BANDS = [(0, 20, "<20 m"), (20, 160, "20-160 m"), (160, 500, "160-500 m"),
             (500, 1120, "500-1120 m"), (1120, 1e9, ">1120 m")]
    tally: dict[tuple[str, str], int] = {}
    artifacts = []
    for cid, g in M[M.in_tile == 1].groupby("cluster_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                d = float(np.hypot(a.row - b.row, a.col - b.col) * S2_PIXEL_M)
                same = a.patch_idx == b.patch_idx
                band = next(nm for lo, hi, nm in BANDS if lo <= d < hi)
                tally[(band, "same patch" if same else "diff patch")] = \
                    tally.get((band, "same patch" if same else "diff patch"), 0) + 1
                if not same and d < 160.0:
                    artifacts.append((cid, a.station_id, b.station_id, d))
    log.info("  %-12s %12s %12s", "separation", "same patch", "diff patch")
    for _, _, nm in BANDS:
        s, dd = tally.get((nm, "same patch"), 0), tally.get((nm, "diff patch"), 0)
        if s or dd:
            log.info("  %-12s %12d %12d", nm, s, dd)
    usable = sum(v for (nm, k), v in tally.items()
                 if k == "diff patch" and nm not in ("<20 m", "20-160 m"))
    log.info("  -> %d pair(s) are BOTH on different patches AND >160 m apart "
             "(the only ones a 160 m map can honestly separate)", usable)
    if artifacts:
        log.info("  grid artifacts -- different patch but <160 m apart, NOT resolvable:")
        for cid, a, b, d in artifacts:
            log.info("      %-26s %-24s %-24s %5.0f m", cid, a, b, d)

    log.info("")
    log.info("wrote %s  (%d rows)", args.out_clusters, len(C))
    log.info("wrote %s  (%d rows)", args.out_members, len(M))
    log.info("accounted for %d of %d %s stations", len(M), len(df), args.igbp)


if __name__ == "__main__":
    main()
