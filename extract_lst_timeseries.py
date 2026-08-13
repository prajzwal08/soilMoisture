"""
Extract station-pixel Landsat ST time series from the §29 Phase A downloads.

For every scene GeoTIFF written by download_landsat_st_mpc.py, and every TxSON station that
falls inside it, record the clear-sky LST at that station's pixel plus the tile-mean LST on
that date, and the anomaly (station - tile mean).  Working in anomaly is the whole point:
it cancels season, air mass and overpass time exactly, leaving only the within-tile contrast
(§29.7).

Pixel indexing takes the affine transform FROM THE GEOTIFF, never from aoi.json — stackstac
snaps bounds outward, so the nominal nx/ny there is typically 1 px short.

Verification built in (--verify, on by default):
  * every station's (row, col) recomputed at 10 m must reproduce csvs/txson_readouts.csv exactly
  * the six CR200-18 stations must land at (112,112) (72,105) (114,43) (25,109) (62,33) (193,65)

Usage:
  conda run -n terramind python extract_lst_timeseries.py --extent tile
  conda run -n terramind python extract_lst_timeseries.py --extent aoi --workers 64
"""

import argparse
import json
import math
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

REPO      = Path(__file__).resolve().parent
READOUTS  = REPO / "csvs" / "txson_readouts.csv"
LST_ROOT  = Path("/gpfs/scratch1/shared/pkhanal/lst/landsat_st/txson")
OUT_CSV   = REPO / "csvs" / "lst_station_timeseries.csv"
PIX_CSV   = REPO / "csvs" / "lst_station_pixels.csv"

TILE_PX, TILE_RES_M = 224, 10
CENTRE = 112

# §29.11 reference — must reproduce exactly
REF_PIXELS = {
    "ISMN_TxSON_CR200-18": (112, 112), "ISMN_TxSON_CR200-25": (72, 105),
    "ISMN_TxSON_CR1000-2": (114, 43),  "ISMN_TxSON_CR200-24": (25, 109),
    "ISMN_TxSON_CR200-15": (62, 33),   "ISMN_TxSON_CR200-6":  (193, 65),
}

LST_MIN_K, LST_MAX_K = 250.0, 350.0


# ------------------------------------------------------------------
# QA  (identical bit logic to download_landsat_st_mpc.landsat_clear)
# ------------------------------------------------------------------

def landsat_clear(qa_dn: np.ndarray) -> np.ndarray:
    q = np.nan_to_num(qa_dn, nan=1.0).astype(np.uint16)
    single = ~np.any([((q >> b) & 1).astype(bool) for b in (0, 1, 2, 3, 4, 5, 7)], axis=0)
    conf = (((q >> 8) & 3) <= 1) & (((q >> 10) & 3) <= 1) & (((q >> 14) & 3) <= 1)
    return single & conf & (((q >> 6) & 1) == 1)


# ------------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------------

def station_table(readouts: Path = READOUTS) -> pd.DataFrame:
    """Unique stations with lat/lon/UTM, plus the tile each one is the centre of."""
    df = pd.read_csv(readouts)
    epsg = int(df.tile_epsg.iloc[0])
    assert (df.tile_epsg == epsg).all(), "mixed EPSG in readouts"
    st = (df[["station", "station_name", "station_split", "lat", "lon"]]
          .drop_duplicates("station").reset_index(drop=True))
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xy = [fwd.transform(lo, la) for lo, la in zip(st.lon, st.lat)]
    st["utm_x"] = [p[0] for p in xy]
    st["utm_y"] = [p[1] for p in xy]
    st["epsg"] = epsg
    return st


def tile_bounds_utm(st: pd.DataFrame, tile: str) -> tuple:
    """UTM bounds of a 224 px @ 10 m station-centred tile (download_s2_mpc.py:98-109)."""
    row = st[st.station == tile]
    if row.empty:
        raise KeyError(tile)
    cx, cy = float(row.utm_x.iloc[0]), float(row.utm_y.iloc[0])
    half = TILE_PX * TILE_RES_M / 2
    return (cx - half, cy - half, cx + half, cy + half)


def verify_readout_pixels(st: pd.DataFrame, readouts: Path = READOUTS):
    """Round-trip every (tile, station) pair in txson_readouts.csv. Assert-and-exit."""
    df = pd.read_csv(readouts)
    bad = []
    for tile, grp in df.groupby("tile"):
        w, s, e, n = tile_bounds_utm(st, tile)
        for _, r in grp.iterrows():
            sx = st.loc[st.station == r.station, "utm_x"].iloc[0]
            sy = st.loc[st.station == r.station, "utm_y"].iloc[0]
            col = int(math.floor((sx - w) / TILE_RES_M))
            row = int(math.floor((n - sy) / TILE_RES_M))
            if (row, col) != (int(r.row), int(r.col)):
                bad.append((tile, r.station, (row, col), (int(r.row), int(r.col))))
    if bad:
        raise SystemExit(f"PIXEL ROUND-TRIP FAILED on {len(bad)}/{len(df)} rows, e.g. {bad[:5]}")

    ref_tile = "ISMN_TxSON_CR200-18"
    w, s, e, n = tile_bounds_utm(st, ref_tile)
    for stn, (er, ec) in REF_PIXELS.items():
        sx = st.loc[st.station == stn, "utm_x"].iloc[0]
        sy = st.loc[st.station == stn, "utm_y"].iloc[0]
        col = int(math.floor((sx - w) / TILE_RES_M))
        row = int(math.floor((n - sy) / TILE_RES_M))
        if (row, col) != (er, ec):
            raise SystemExit(f"§29.11 reference pixel FAILED {stn}: got {(row,col)} want {(er,ec)}")
    print(f"  verify: {len(df)} readout pixels round-trip exactly; "
          f"6 CR200-18 reference pixels match §29.11")


# ------------------------------------------------------------------
# PER-SCENE EXTRACTION
# ------------------------------------------------------------------

def scene_records(tif: Path, st: pd.DataFrame, tiles: list, max_st_qa: float,
                  min_tile_clear: float, agg: str) -> list:
    with rasterio.open(tif) as src:
        arr = src.read()
        T, crs = src.transform, src.crs
        tags = src.tags()
    lst, stqa, qap = arr[0], arr[1], arr[2]
    H, W = lst.shape

    clear = (landsat_clear(qap) & np.isfinite(lst)
             & (lst > LST_MIN_K) & (lst < LST_MAX_K)
             & (np.isfinite(stqa) & (stqa <= max_st_qa)))

    date = tags.get("datetime_utc", "")[:10]
    out = []

    for tile in tiles:
        w, s, e, n = tile_bounds_utm(st, tile)
        r0, c0 = rowcol(T, w, n, op=math.floor)
        r1, c1 = rowcol(T, e, s, op=math.ceil)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(H, r1), min(W, c1)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        sub_lst, sub_clear = lst[r0:r1, c0:c1], clear[r0:r1, c0:c1]
        frac = float(sub_clear.mean())
        if frac < min_tile_clear:
            continue
        tile_mean = float(np.nanmean(sub_lst[sub_clear]))

        for _, stn in st.iterrows():
            rr, cc = rowcol(T, stn.utm_x, stn.utm_y, op=math.floor)
            if not (r0 <= rr < r1 and c0 <= cc < c1):
                continue
            if agg == "point":
                ok, v, q = bool(clear[rr, cc]), float(lst[rr, cc]), float(stqa[rr, cc])
            else:
                k = 1 if agg == "med3" else 2
                sl = (slice(max(0, rr - k), min(H, rr + k + 1)),
                      slice(max(0, cc - k), min(W, cc + k + 1)))
                m = clear[sl]
                ok = bool(m.any())
                v = float(np.nanmedian(lst[sl][m])) if ok else np.nan
                q = float(np.nanmedian(stqa[sl][m])) if ok else np.nan
            if not ok or not np.isfinite(v):
                continue
            out.append({
                "sensor": "landsat", "scene": tif.name, "date": date,
                "platform": tags.get("platform", ""),
                "wrs": f"{tags.get('wrs_path','')}/{tags.get('wrs_row','')}",
                "tile": tile, "station": stn.station,
                "station_name": stn.station_name, "split": stn.station_split,
                # lst_row/col index the 30 m LST raster; tile_row/col index the 224 px
                # 10 m tile grid used by txson_readouts.csv and every §26/§27 figure.
                # Keeping both named explicitly - conflating them puts every station
                # marker in the top-left corner of a tile plot.
                "lst_row": int(rr), "lst_col": int(cc),
                "tile_row": int(math.floor((n - stn.utm_y) / TILE_RES_M)),
                "tile_col": int(math.floor((stn.utm_x - w) / TILE_RES_M)),
                "lst_k": round(v, 3), "st_qa_k": round(q, 3),
                "tile_mean_k": round(tile_mean, 3),
                "lst_anom_k": round(v - tile_mean, 3),
                "tile_frac_clear": round(frac, 4),
            })
    return out


def _worker(tif, st, tiles, max_st_qa, min_tile_clear, agg):
    try:
        return scene_records(tif, st, tiles, max_st_qa, min_tile_clear, agg)
    except Exception as exc:
        print(f"  !! {tif.name}: {exc}")
        return []


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extent", choices=["tile", "aoi"], default="tile")
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--root", type=Path, default=LST_ROOT)
    ap.add_argument("--max-st-qa", type=float, default=3.0)
    ap.add_argument("--min-tile-clear", type=float, default=0.70)
    ap.add_argument("--agg", choices=["point", "med3", "med5"], default="point")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    st = station_table()
    print(f"station table: {len(st)} TxSON stations, EPSG:{st.epsg.iloc[0]}")

    if not args.no_verify:
        verify_readout_pixels(st)

    st.to_csv(PIX_CSV, index=False)

    sub = args.tile if args.extent == "tile" else "aoi"
    scene_dir = args.root / sub
    tifs = sorted(p for p in scene_dir.glob("*.tif") if not p.name.endswith(".tmp.tif"))
    if not tifs:
        raise SystemExit(f"no scenes in {scene_dir}")
    print(f"{len(tifs)} scenes in {scene_dir}")

    tiles = ([args.tile] if args.extent == "tile"
             else sorted(pd.read_csv(READOUTS).tile.unique()))
    print(f"{len(tiles)} tile footprint(s); agg={args.agg} "
          f"max_st_qa={args.max_st_qa} min_tile_clear={args.min_tile_clear}")

    fn = partial(_worker, st=st, tiles=tiles, max_st_qa=args.max_st_qa,
                 min_tile_clear=args.min_tile_clear, agg=args.agg)
    with Pool(args.workers) as pool:
        chunks = pool.map(fn, tifs)

    rows = [r for c in chunks for r in c]
    if not rows:
        raise SystemExit("no station-date records survived the clear-sky filter")

    df = pd.DataFrame(rows).sort_values(["tile", "date", "station"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 72)
    print(f"WROTE {args.out}   {len(df)} station-date records")
    print(f"  scenes contributing : {df.scene.nunique()} / {len(tifs)} "
          f"({100*df.scene.nunique()/len(tifs):.1f}% survive tile_frac_clear>={args.min_tile_clear})")
    print(f"  dates               : {df.date.nunique()}  "
          f"({df.date.min()} .. {df.date.max()})")
    print(f"  stations            : {df.station.nunique()}")
    print(f"  LST      min/med/max: {df.lst_k.min():.1f} / {df.lst_k.median():.1f} / "
          f"{df.lst_k.max():.1f} K")
    print(f"  ST_QA    median     : {df.st_qa_k.median():.2f} K")
    print(f"  |anomaly| median    : {df.lst_anom_k.abs().median():.2f} K   "
          f"p95 = {df.lst_anom_k.abs().quantile(0.95):.2f} K")

    print("\n  per-station mean LST anomaly (K):")
    g = df.groupby("station_name").lst_anom_k.agg(["mean", "std", "count"])
    for name, r in g.sort_values("mean").iterrows():
        print(f"    {name:<12s} {r['mean']:+7.3f}  sd {r['std']:5.3f}  n={int(r['count'])}")

    print("\n  within-tile anomaly spread per date (K):")
    sp = df.groupby(["tile", "date"]).lst_anom_k.agg(lambda v: v.max() - v.min())
    print(f"    median {sp.median():.2f}   p90 {sp.quantile(0.9):.2f}   max {sp.max():.2f}")
    print(f"    (sensor noise floor ~1-2 K; median ST_QA {df.st_qa_k.median():.2f} K)")
    print("=" * 72)


if __name__ == "__main__":
    main()
