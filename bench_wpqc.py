#!/usr/bin/env python
"""Where does the per-granule time go?  ONE config per process, DISJOINT granules.

The first version of this benchmark ran all configs in one process over the same 64
granules and reported 6 ms opens -- that is GDAL's /vsicurl chunk cache answering from
RAM, not the network, so every config after the first was measuring a warm cache.  Two
changes make the numbers trustworthy:

  * --config runs exactly ONE config, so env vars that curl reads at connection-creation
    time (HTTP_VERSION, MULTIPLEX) cannot leak from a previous config;
  * --offset hands each config a DISJOINT slice of granules, so nothing it reads has
    been touched by another config.

Configs:
  A_percall    today's code: a fresh ThreadPoolExecutor(3) per granule, 4 layers
  B_shared     the fix: one long-lived layer pool, so curl handles survive the granule
  C_seq        no inner pool at all, 4 layers sequential in the worker thread
  E_sharednovza  B minus view_zenith, which is reported but never enters passed_qc
  D_tuned      E plus HTTP/2 + multiplex + 64 KB header prefetch
"""
from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import ROOT, STATION_CSV, TILE_M, configure_gdal, layer_url  # noqa: E402

T: dict[str, list[float]] = {}
_POOL: ThreadPoolExecutor | None = None


def rec(k, dt):
    T.setdefault(k, []).append(dt)


def shared_pool(n):
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=n)
    return _POOL


def read_one(ur, lon, lat, layers, mode, inner_n):
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    t0 = time.perf_counter()
    src = rasterio.open(f"/vsicurl/{layer_url(ur, 'QC')}")
    rec("open_QC", time.perf_counter() - t0)
    try:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        h = TILE_M / 2.0
        win = from_bounds(xs[0] - h, ys[0] - h, xs[0] + h, ys[0] + h,
                          src.transform).round_offsets().round_lengths()
        t0 = time.perf_counter()
        qc = src.read(1, window=win)
        rec("read_QC", time.perf_counter() - t0)
    finally:
        src.close()
    if qc.size == 0:
        return False

    def _layer(lyr):
        t0 = time.perf_counter()
        with rasterio.open(f"/vsicurl/{layer_url(ur, lyr)}") as s:
            rec("open_layer", time.perf_counter() - t0)
            t1 = time.perf_counter()
            s.read(1, window=win)
            rec("read_layer", time.perf_counter() - t1)

    if mode == "percall":
        with ThreadPoolExecutor(max_workers=3) as lp:
            list(lp.map(_layer, layers))
    elif mode == "shared":
        list(shared_pool(inner_n).map(_layer, layers))
    else:
        for lyr in layers:
            _layer(lyr)
    return True


L4 = ("cloud", "water", "view_zenith")
L3 = ("cloud", "water")
CONFIGS = {
    "A_percall":     dict(layers=L4, mode="percall", tuned=False),
    "B_shared":      dict(layers=L4, mode="shared",  tuned=False),
    "C_seq":         dict(layers=L4, mode="seq",     tuned=False),
    "E_sharednovza": dict(layers=L3, mode="shared",  tuned=False),
    "D_tuned":       dict(layers=L3, mode="shared",  tuned=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    configure_gdal()
    if cfg["tuned"]:
        os.environ["GDAL_HTTP_VERSION"] = "2"
        os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
        os.environ["GDAL_INGESTED_BYTES_AT_OPEN"] = "65536"
    else:
        for k in ("GDAL_HTTP_VERSION", "GDAL_HTTP_MULTIPLEX", "GDAL_INGESTED_BYTES_AT_OPEN"):
            os.environ.pop(k, None)

    pairs = pd.read_csv(ROOT / "csvs" / "ecostress_census_pairs.stats.csv")
    pairs = pairs[pairs["well_phased"] == 1].drop_duplicates(
        subset=["station_id", "day_ur", "night_ur"])
    sdf = pd.read_csv(STATION_CSV).set_index("station_id")[["latitude", "longitude"]]
    pairs = pairs[pairs["station_id"].isin(sdf.index)]
    # Stride the file so every config's slice spans the whole network, then take a
    # DISJOINT window of it.  Slices never overlap, so no config warms another's cache.
    step = max(len(pairs) // 4000, 1)
    pool_rows = pairs.iloc[::step].reset_index(drop=True)
    sl = pool_rows.iloc[args.offset: args.offset + args.n]
    tasks = [(u, float(sdf.loc[s, "longitude"]), float(sdf.loc[s, "latitude"]))
             for s, u in zip(sl["station_id"], sl["day_ur"])]

    # Warm-up: EDL hands out a session cookie on first contact; that one-off cost belongs
    # to neither config.  Two throwaway granules from OUTSIDE every measured slice.
    warm = pool_rows.iloc[3000:3002]
    for s, u in zip(warm["station_id"], warm["day_ur"]):
        try:
            read_one(u, float(sdf.loc[s, "longitude"]), float(sdf.loc[s, "latitude"]),
                     cfg["layers"], cfg["mode"], 3 * args.workers)
        except Exception:                                           # noqa: BLE001
            pass
    T.clear()

    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def w(t):
            try:
                return read_one(t[0], t[1], t[2], cfg["layers"], cfg["mode"],
                                3 * args.workers)
            except Exception:                                       # noqa: BLE001
                return False
        for r in pool.map(w, tasks):
            ok += int(bool(r))
    el = time.time() - t0

    nlay = len(cfg["layers"]) + 1
    print(f"=== {args.config}  layers={nlay} mode={cfg['mode']} tuned={cfg['tuned']} "
          f"offset={args.offset} workers={args.workers}", flush=True)
    print(f"    {len(tasks)} granules in {el:.1f}s -> {len(tasks)/el:.2f} gran/s ({ok} ok)",
          flush=True)
    for k in sorted(T):
        v = T[k]
        print(f"    {k:<12} n={len(v):4d} median={stats.median(v)*1000:7.0f} ms "
              f"total={sum(v):7.1f} s", flush=True)
    conn = sum(sum(v) for k, v in T.items() if k.startswith("open_"))
    read = sum(sum(v) for k, v in T.items() if k.startswith("read_"))
    print(f"    CONNECT {conn:7.1f} s  READ {read:7.1f} s  -> {100*conn/max(conn+read,1e-9):.0f}%"
          f" opening   |  projected 55792 granules: {55792/max(len(tasks)/el,1e-9)/3600:.2f} h",
          flush=True)


if __name__ == "__main__":
    main()
