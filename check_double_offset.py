"""
check_double_offset.py — per-scene evidence for §50.8's double-offset suspects
===============================================================================
The group statistic (pre-cut scenes with catalogue baseline >= 04.00: B04 +1,290 DN) is real on
average, but the old harmonise skip guard (min non-zero DN >= 1000 -> skip) left many of those
scenes correctly single-offset. The repair smoke proved it: 20211004 scenes at p1 = 1001 were
fine and a blanket −1000 broke them. So each suspect is decided by comparison with the source:

  --fetch    (soilmoisture)  re-download every catalogue item for each suspect (station, date),
                             same grid/crop/bands as download_s2_mpc.py, NaN -> 0, NO offset
                             -> s2_repair_check/{station}/{date}__{item_id}.npy
  --compare  (terramind)     stored raw row vs each fresh item, over pixels non-zero in both:
                             delta = median(stored - fresh). The best-matching item (smallest
                             MAD of the residual after removing delta) is the one we stored.
                             delta ~ +1000  -> double_offset (repair: −1000)
                             delta ~    0   -> fine (drop from the repair list)
                             otherwise      -> unknown (listed, not repaired)
                             -> csvs/s2_double_offset_check.csv; rewrites s2_repair_targets.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = Path("/gpfs/scratch1/shared/pkhanal/s2_repair_check")
RES = REPO / "csvs" / "s2_double_offset_check.csv"


def suspects():
    t = pd.read_csv(REPO / "csvs" / "s2_repair_targets.csv")
    return t


def fetch():
    import planetary_computer
    import pystac_client
    import stackstac
    from rasterio.enums import Resampling
    from download_s2_mpc import MPC_URL, RES_M, S2_BANDS, center_crop, station_grid
    from splits_config import station_dir_name
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    ll = {station_dir_name(r): (float(r.latitude), float(r.longitude)) for _, r in df.iterrows()}
    t = suspects()
    t = t[t.fix.str.contains("double_offset")]
    cat = pystac_client.Client.open(MPC_URL)
    for r in t.itertuples():
        epsg, bounds, bbox = station_grid(*ll[r.station])
        d = str(int(r.date))
        day = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=day).items())
        (OUT / r.station).mkdir(parents=True, exist_ok=True)
        for it in items:
            p = OUT / r.station / f"{d}__{it.id}.npy"
            if p.exists():
                continue
            for a in range(3):
                try:
                    si = planetary_computer.sign(it)
                    da = stackstac.stack([si], assets=S2_BANDS, epsg=epsg, resolution=RES_M,
                                         bounds=bounds, rescale=False,
                                         resampling=Resampling.bilinear).squeeze("time").compute()
                    x = center_crop(da).fillna(0).values.astype(np.int16)
                    np.save(p, x)
                    print(f"  {r.station} {d} {it.id} pb={it.properties.get('s2:processing_baseline')}", flush=True)
                    break
                except Exception as e:                            # noqa: BLE001
                    print(f"  !! {r.station} {d} {it.id}: {type(e).__name__} {str(e)[:100]}", flush=True)


def compare():
    import zarr
    from backfill_merge import RAW_ROOT, _dates_int
    t = suspects()
    rows = []
    for st, g in t[t.fix.str.contains("double_offset")].groupby("station"):
        z = zarr.open_group(str(RAW_ROOT / f"{st}.zarr"), mode="r")
        dates = _dates_int(z["s2/dates"][:])
        for d in g.date.astype(int):
            stored = z["s2/data"][dates.index(d)].astype(np.int32)
            best = None
            for p in sorted((OUT / st).glob(f"{d}__*.npy")):
                fresh = np.load(p).astype(np.int32)
                m = (stored != 0) & (fresh != 0)
                if m.sum() < 1000:
                    continue
                res = (stored - fresh)[m]
                delta = float(np.median(res))
                mad = float(np.median(np.abs(res - delta)))
                if best is None or mad < best[2]:
                    best = (p.name.split("__")[1][:-4], delta, mad, float(m.mean()))
            if best is None:
                rows.append(dict(station=st, date=d, item_id="", delta=np.nan, mad=np.nan,
                                 overlap=0.0, verdict="unknown_no_fresh"))
                continue
            item, delta, mad, ov = best
            v = ("double_offset" if abs(delta - 1000) <= 50 and mad <= 50 else
                 "fine" if abs(delta) <= 50 and mad <= 50 else "unknown")
            rows.append(dict(station=st, date=d, item_id=item, delta=delta, mad=mad,
                             overlap=round(ov, 3), verdict=v))
    res = pd.DataFrame(rows)
    res.to_csv(RES, index=False)
    print(res.verdict.value_counts().to_string())
    print(res.groupby("verdict")[["delta", "mad"]].median().round(1).to_string())
    print(res[res.verdict.str.startswith("unknown")].head(20).to_string(index=False))
    # the repair list keeps negatives + CONFIRMED double offsets only
    t = suspects()
    keep_dbl = set(zip(res[res.verdict == "double_offset"].station, res[res.verdict == "double_offset"].date))
    t["fix"] = [("+".join(f for f in fx.split("+")
                          if f != "double_offset" or (s, int(d)) in keep_dbl))
                for s, d, fx in zip(t.station, t.date, t.fix)]
    t = t[t.fix != ""]
    t.to_csv(REPO / "csvs" / "s2_repair_targets.csv", index=False)
    print(f"-> repair list now {len(t)} scenes: {t.fix.value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    fetch() if a.fetch else compare()
