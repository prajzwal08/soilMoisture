"""§21.2 — per-station SMAP L3 extraction from Google Earth Engine.

Ceiling test for the absolute-level problem: does SMAP predict per-station MEAN soil
moisture, which soil / terrain / land cover / ERA5 largely cannot (§20.14)?

Product: NASA/SMAP/SPL3SMP_E/006 — L3 Enhanced 9 km, 2015-03-31 -> present.
  A RADIOMETER RETRIEVAL, deliberately chosen over L4: L4 (SPL4SMGP) is a land-model
  assimilation, which would weaken the independence claim and turn the thesis into
  "downscaling a model product". L3 keeps it a retrieval.

  The standard 36 km SPL3SMP is NOT in the GEE catalog (verified 2026-08-05). Enhanced
  9 km is the same 36 km radiometer footprints Backus-Gilbert interpolated onto a 9 km
  grid — same information content, finer posting, and the finer grid works in our favour
  against the sub-pixel-heterogeneity objection in §21.1.

  SURFACE ONLY (~top 5 cm). It maps to the 0-10 cm label bin and gives nothing for
  10-30 or 30-100. That is not a defect of the test: 0-10 is SMAP's BEST case, so a
  failure there kills the idea without needing the deeper product.

  tb_h/v_corrected are included because L1C_TB is not a separate GEE collection, and
  these are the corrected brightness temperatures — usable if we later want to learn a
  retrieval rather than inherit SMAP's.

Writes:
    csvs/smap_station.csv    one row per station: temporal means over --years
    csvs/smap_daily.parquet  per-station daily series (--daily), for the anomaly test

    python download_smap_gee.py [--years 2016 2022] [--daily]

Runs in the `soilmoisture` conda env — earthengine-api lives there, not in terramind.
"""
import argparse
from pathlib import Path

import pandas as pd

GEE_PROJECT = "1066500857818"          # same project as download_era5land_gee.py:42
L3 = "NASA/SMAP/SPL3SMP_E/006"

# am = 06:00 descending (the retrieval-preferred overpass), pm = 18:00 ascending.
BANDS = ["soil_moisture_am", "soil_moisture_pm",
         "tb_h_corrected_am", "tb_v_corrected_am",
         "vegetation_water_content_am"]

CATEGORY = "sm_only"


def station_table(splits_csv):
    """The same 661-station selection station_mean_probe.py uses (dataset.py:813-835)."""
    df = pd.read_csv(splits_csv)

    def _cat(r):
        sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        fl = str(r.get("has_flux", "False")).lower() == "true"
        return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")

    df = df[df.apply(_cat, axis=1) == CATEGORY]
    df = df[df["split"].isin(["train", "val"])]
    df = df[df["soil_patch_ok"].astype(str).str.lower() == "true"].reset_index(drop=True)
    df["dir_name"] = [
        (f"ISMN_{r['network']}_{r['station_name']}" if str(r["source_network"]) == "ISMN"
         else f"{r['source_network']}_{r['station_id']}")
        for _, r in df.iterrows()
    ]
    return df[["dir_name", "split", "latitude", "longitude", "network"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-csv", default="csvs/station_splits.csv")
    ap.add_argument("--years", nargs=2, type=int, default=[2016, 2022],
                    help="inclusive; default matches the training window (train.py:185)")
    ap.add_argument("--out", default="csvs")
    ap.add_argument("--chunk", type=int, default=150,
                    help="stations per getInfo() call — keeps payload under EE limits")
    ap.add_argument("--daily", action="store_true",
                    help="also pull daily series, for the SMAP-anomaly test")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=GEE_PROJECT)
    print(f"EE initialized (project {GEE_PROJECT})")

    st = station_table(args.splits_csv)
    y0, y1 = args.years
    start, end = f"{y0}-01-01", f"{y1}-12-31"
    print(f"Stations: {len(st)}  ({(st.split=='train').sum()} train / "
          f"{(st.split=='val').sum()} val)")
    print(f"Product : {L3}  (L3 Enhanced 9 km, radiometer retrieval, SURFACE only)")
    print(f"Window  : {start} .. {end}\n")

    # One temporal-mean composite; evaluated lazily, only at the sampled points.
    mean_img = ee.ImageCollection(L3).filterDate(start, end).select(BANDS).mean()

    rows = []
    for i in range(0, len(st), args.chunk):
        part = st.iloc[i:i + args.chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                       {"dir_name": r.dir_name})
            for r in part.itertuples()
        ])
        sampled = mean_img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                         scale=9000).getInfo()
        rows += [f["properties"] for f in sampled["features"]]
        print(f"  sampled {min(i + args.chunk, len(st))}/{len(st)}")

    out = st.merge(pd.DataFrame(rows), on="dir_name", how="left")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "smap_station.csv"
    out.to_csv(p, index=False)
    print(f"\nSaved: {p}   ({len(out)} rows)")

    # No silent caps — report coverage per band before anything downstream uses it.
    for c in BANDS:
        if c not in out.columns:
            print(f"   {c:<30} ** ABSENT from response **")
            continue
        vals = out[c].dropna()
        rng = f"[{vals.min():.4f}, {vals.max():.4f}]" if len(vals) else "--"
        print(f"   {c:<30} missing={out[c].isna().sum():4d}  range={rng}")

    if args.daily:
        _daily(ee, st, start, end, outdir, args.chunk)


def _daily(ee, st, start, end, outdir, chunk):
    """Per-station daily series for the SMAP-anomaly -> station-anomaly test.

    Separate from the mean because the dynamics claim needs its own evidence: ERA5-Land
    already carries regional wetting/drying at 9 km, so SMAP's marginal value for ubRMSE
    is likely small and must be measured rather than assumed (§21.1).
    """
    print("\nPulling daily series (slow part)...")
    ic = (ee.ImageCollection(L3).filterDate(start, end)
          .select(["soil_moisture_am", "soil_moisture_pm"]))
    frames = []
    for i in range(0, len(st), chunk):
        part = st.iloc[i:i + chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                       {"dir_name": r.dir_name})
            for r in part.itertuples()
        ])

        def _samp(img):
            d = img.date().format("YYYY-MM-dd")
            return (img.reduceRegions(fc, ee.Reducer.first(), 9000)
                       .map(lambda f: f.set("date", d)))

        recs = ic.map(_samp).flatten().getInfo()
        frames += [f["properties"] for f in recs["features"]]
        print(f"  daily {min(i + chunk, len(st))}/{len(st)}")
    df = pd.DataFrame(frames)
    p = outdir / "smap_daily.parquet"
    df.to_parquet(p, index=False)
    print(f"Saved: {p}   ({len(df)} rows)")


if __name__ == "__main__":
    main()
