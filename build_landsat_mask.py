#!/usr/bin/env python
"""Build the supervision mask for the Landsat ST head, once, as the single source of truth.

THE RULE, decided 2026-09-22 (reasoning in text/landsat_st_download.md):

    supervise a pixel  <=>  qa_decode(qa_pixel30)      QA_PIXEL bits 0-5 rejected, cloud/shadow/
                                                       cirrus confidence <= low, bit 6 NOT
                                                       required, bit 7 water KEPT (flagged)
                        AND isfinite(lst30)            the no-retrieval mask
                        AND 250 K < lst30 < 360 K      360, not §29.5's 350: bare desert genuinely
                                                       reads 354.5 K at Stovepipe Wells.  This
                                                       also excludes DN 65535 -> 372.99994 K, the
                                                       uint16 saturation sentinel, which is not a
                                                       temperature.
                        AND cdist30 > 1.0 km           the nearest cloud must be MORE than 1 km
                                                       away.  NO ST_QA GATE: ST_QA is very nearly
                                                       a readout of distance to cloud (5.875 K
                                                       adjacent, 1.625 K beyond 20 km), so gating
                                                       on both double-charges -- and ST_QA <= 3 K
                                                       alone costs 107 of 993 stations while
                                                       excluding only 5 scenes at its 8 K tail.

Masks are np.packbits'd: 76*76 bits is 722 bytes per scene, so the whole thing is small against
the 4.68 GB archive.

OUTPUT  {DATA_ROOT}/{cat}/{folder}/LANDSAT_ST/{folder}_st30mask_{start}_{end}.npz
        csvs/landsat_mask_index.csv    one row per (station, date) -- what dataset.py selects on
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, qa_decode

OUT_CSV = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/landsat_mask_index.csv")
LST_LO, LST_HI = 250.0, 360.0
CDIST_MIN_KM = 1.0
GRID_N = 76
NPX = GRID_N * GRID_N

RULE = (f"qa_decode(qa_pixel30) & isfinite(lst30) & {LST_LO:g} < lst30 < {LST_HI:g} K "
        f"& cdist30 > {CDIST_MIN_KM:g} km ; NO ST_QA gate ; water kept and flagged ; "
        f"QA_PIXEL bit 6 not required")


def one(args):
    path_str, overwrite = args
    path = Path(path_str)
    out = path.with_name(path.name.replace("_st30_", "_st30mask_"))

    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    lst, cd, qap = z["lst30"], z["cdist30"], z["qa_pixel30"]
    n = lst.shape[0]

    clear, water = qa_decode(qap.astype("float64"))
    mask = (clear & np.isfinite(lst) & (lst > LST_LO) & (lst < LST_HI)
            & np.isfinite(cd) & (cd > CDIST_MIN_KM))

    flat = mask.reshape(n, -1)
    n_px = flat.sum(axis=1)
    n_clear = (clear & np.isfinite(lst)).reshape(n, -1).sum(axis=1)

    if not (out.exists() and not overwrite):
        payload = {
            # unpack with:
            #   np.unpackbits(m, axis=1, count=5776).reshape(-1, 76, 76).astype(bool)
            "mask":       np.packbits(flat, axis=1),
            "water":      np.packbits((water & np.isfinite(lst)).reshape(n, -1), axis=1),
            "n_px":       n_px.astype("int32"),
            "n_px_clear": n_clear.astype("int32"),
            "dates":      z["dates"],
            "meta": np.array([json.dumps({
                "station_id": meta["station_id"], "folder": meta["folder"],
                "grid": f"{GRID_N}x{GRID_N} @ 30 m, packbits over the flattened scene",
                "unpack": f"np.unpackbits(m, axis=1, count={NPX}).reshape(-1,{GRID_N},"
                          f"{GRID_N}).astype(bool)",
                "rule": RULE,
                "cdist_min_km": CDIST_MIN_KM,
                "cdist_sense": "distance from EACH PIXEL to the nearest cloud; larger is "
                               "further from cloud, so > 1 km is the strict choice",
                "lst_range_K": [LST_LO, LST_HI],
            })]),
        }
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.rename(out)

    return {"station_id": meta["station_id"], "folder": meta["folder"],
            "n_scenes": n, "n_supervised": int((n_px > 0).sum()),
            "px": int(n_px.sum()), "mb": round(out.stat().st_size / 1e6, 3),
            "_rows": [(meta["station_id"], str(d), int(c), int(p))
                      for d, c, p in zip(z["dates"], n_clear, n_px)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths = sorted(str(p) for p in DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    print(f"{len(paths)} bundles")
    with Pool(args.workers) as pool:
        res = pool.map(one, [(p, args.overwrite) for p in paths], chunksize=2)

    idx = pd.DataFrame([x for r in res for x in r["_rows"]],
                       columns=["station_id", "date", "n_px_clear", "n_px_supervised"])
    idx["supervised"] = (idx.n_px_supervised > 0).astype("int8")
    idx.to_csv(OUT_CSV, index=False)

    d = pd.DataFrame([{k: v for k, v in r.items() if k != "_rows"} for r in res])
    sup = int(d.n_supervised.sum())
    print("\n" + "=" * 78)
    print(f"SUPERVISION MASK BUILT   (CDIST > {CDIST_MIN_KM:g} km, no ST_QA gate)")
    print("=" * 78)
    print(f"  mask files          : {len(d)}")
    print(f"  total size          : {d.mb.sum()/1000:.3f} GB")
    print(f"  scenes              : {int(d.n_scenes.sum()):,}")
    print(f"  scenes SUPERVISED   : {sup:,}")
    print(f"  supervised pixels   : {int(d.px.sum()):,}")
    print(f"  mean supervised px per supervised scene : {d.px.sum()/max(sup,1):.0f} of {NPX}")
    s = d.n_supervised
    print(f"  per station         : min {s.min()}  median {s.median():.0f}  max {s.max()}")
    print(f"  stations < 20 scenes: {int((s < 20).sum())}")
    print(f"\n  wrote {OUT_CSV}  ({len(idx):,} rows)")
    print("=" * 78)


if __name__ == "__main__":
    main()
