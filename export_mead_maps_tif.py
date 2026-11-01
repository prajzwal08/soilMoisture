# ============================================================
# Mead: daily predicted SM maps -> GeoTIFF, for stepping through dates in QGIS (runbook §69)
# ============================================================
# Input: maps_<tile>_2017-2024.npz from predict_mead_pixels.py --save-maps
#        sm (days, 3, 112, 112) fp16 at 20 m, dates (days,) -- the frozen §66 model, every day.
# Georeference: the tile's bounds_utm + epsg from the raw imagery store .zattrs (north-up UTM,
# 2240 m, the same grid the S2 224 px tile sits on; 112 px map = 20 m).
#
# Writes, per tile, under /gpfs/work3/0/prjs1968/data_mead/maps_tif/<year>/:
#   SM_<Ne1|Ne2>_<depth>_<year>.tif   one band per day, band description = YYYY-MM-DD
#   daily/SM_<Ne1|Ne2>_<YYYYMMDD>.tif one file per day, 3 bands = 0-10 / 10-30 / 30-100 cm
# Float32, m3 m-3, nodata NaN. Ne1 and Ne2 tiles overlap (546 m east-west shift): kept separate.
#
# Usage:
#   python export_mead_maps_tif.py [--year 2019]
# ============================================================

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

PRED    = Path("/gpfs/work3/0/prjs1968/data_mead/pred")
RAW     = Path("/projects/prjs1968/satellite_zarr")
OUT     = Path("/gpfs/work3/0/prjs1968/data_mead/maps_tif")
TILES   = {"Ne1": "AmeriFlux_US-Ne1", "Ne2": "AmeriFlux_US-Ne2"}
DEPTHS  = ["0-10", "10-30", "30-100"]                    # SM_DEPTHS order, load-bearing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    y = ap.parse_args().year
    out = OUT / str(y)
    (out / "daily").mkdir(parents=True, exist_ok=True)

    for short, st in TILES.items():
        z = np.load(PRED / f"maps_{st}_2017-2024.npz")
        dates = z["dates"].astype("datetime64[D]")
        i = np.where(dates.astype("datetime64[Y]").astype(int) + 1970 == y)[0]
        sm, d = z["sm"][i].astype(np.float32), dates[i]
        attrs = json.loads((RAW / f"{st}.zarr" / ".zattrs").read_text())
        x0, y0, x1, y1 = attrs["bounds_utm"]
        n, _, h, w = sm.shape
        prof = dict(driver="GTiff", dtype="float32", width=w, height=h, nodata=np.nan,
                    crs=f"EPSG:{attrs['epsg']}", transform=from_bounds(x0, y0, x1, y1, w, h),
                    compress="deflate", predictor=3, tiled=False)
        ds = [str(x) for x in d]
        missing = 365 + (y % 4 == 0) - n
        print(f"{short}: {n} days {ds[0]} .. {ds[-1]} ({missing} missing), "
              f"bounds {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f} EPSG:{attrs['epsg']}, "
              f"SM range {np.nanmin(sm):.3f}..{np.nanmax(sm):.3f}")

        for k, dep in enumerate(DEPTHS):
            f = out / f"SM_{short}_{dep}_{y}.tif"
            with rasterio.open(f, "w", count=n, **prof) as dst:
                dst.write(sm[:, k])
                for b, s in enumerate(ds, start=1):
                    dst.set_band_description(b, s)
                dst.update_tags(station=st, depth_cm=dep, units="m3 m-3", model="§66 nolst_L3_wu200_20261005")
            print(f"  wrote {f}")
        for t, s in enumerate(ds):
            f = out / "daily" / f"SM_{short}_{s.replace('-', '')}.tif"
            with rasterio.open(f, "w", count=3, **prof) as dst:
                dst.write(sm[t])
                for b, dep in enumerate(DEPTHS, start=1):
                    dst.set_band_description(b, f"{dep} cm")
                dst.update_tags(date=s, station=st, units="m3 m-3")
        print(f"  wrote {n} daily files to {out / 'daily'}")


if __name__ == "__main__":
    main()
