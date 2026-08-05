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
    ap.add_argument("--stride-days", type=int, default=5,
                    help="use only days 1..N of each month (seasonally unbiased "
                         "subsample). 0 = every day, which stalls EE at this point count.")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=GEE_PROJECT)
    print(f"EE initialized (project {GEE_PROJECT})")

    st = station_table(args.splits_csv)
    # Sort geographically before chunking. Chunks of globally-scattered points force EE
    # to hold tiles from everywhere in one request -> "User memory limit exceeded".
    # Sorting by coarse lat/lon cell makes each chunk spatially compact, so a request
    # touches a handful of tiles instead of hundreds.
    st = (st.assign(_cell_lat=(st.latitude // 5), _cell_lon=(st.longitude // 5))
            .sort_values(["_cell_lat", "_cell_lon", "latitude", "longitude"])
            .drop(columns=["_cell_lat", "_cell_lon"])
            .reset_index(drop=True))
    y0, y1 = args.years
    start, end = f"{y0}-01-01", f"{y1}-12-31"
    print(f"Stations: {len(st)}  ({(st.split=='train').sum()} train / "
          f"{(st.split=='val').sum()} val)")
    print(f"Product : {L3}  (L3 Enhanced 9 km, radiometer retrieval, SURFACE only)")
    print(f"Window  : {start} .. {end}\n")

    # PER-YEAR composites, not one mean over the whole window. A single .mean() over
    # ~4000 images sampled at 661 points is one enormous server-side reduction and EE
    # stalls on it. Per year it is ~580 images per request, which returns in seconds,
    # and the yearly means are averaged locally afterwards. Same answer, bounded requests.
    # Sample the PIXEL per image, then average LOCALLY.
    #
    # The earlier version asked EE for `collection.mean()` and sampled that. Reducing
    # ~4000 global images server-side before extracting 661 point values is an enormous
    # computation and it stalled indefinitely. Reducer.first() at 9 km was never doing
    # spatial aggregation anyway — it is just "the pixel containing the station" — so the
    # mean does not need to happen on EE's side at all. Per-image sampling is a cheap
    # lookup, and averaging in pandas afterwards gives the identical number.
    #
    # Sampling per image also yields the daily series the §21.2 anomaly test needs, so
    # the dynamics check comes free instead of costing a second pass.
    import time
    frames = []
    for y in range(y0, y1 + 1):
        ic = (ee.ImageCollection(L3)
              .filterDate(f"{y}-01-01", f"{y}-12-31")
              .select(BANDS))
        if args.stride_days:
            # Days 1..N of each month: ~N*12 samples/yr, evenly spread over the seasonal
            # cycle, so the climatological mean is estimated with far fewer images and
            # WITHOUT the seasonal bias that picking whole months would introduce.
            ic = ic.filter(ee.Filter.calendarRange(1, args.stride_days, "day_of_month"))
        # One mean image per year, then read THE PIXEL CONTAINING THE STATION.
        # Reducer.first() at the native 9 km scale is exactly that — no neighbourhood,
        # no aggregation. Per-year keeps each server-side reduction small, and the yearly
        # means are averaged locally afterwards.
        img = ic.mean()
        t0, rows = time.time(), []
        for i in range(0, len(st), args.chunk):
            part = st.iloc[i:i + args.chunk]
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                           {"dir_name": r.dir_name})
                for r in part.itertuples()
            ])
            sampled = img.reduceRegions(fc, ee.Reducer.first(), 9000,
                                        tileScale=16).getInfo()
            rows += [f["properties"] for f in sampled["features"]]
        df = pd.DataFrame(rows)
        df["year"] = y
        frames.append(df)
        got = df[BANDS[0]].notna().sum() if BANDS[0] in df.columns else 0
        print(f"  {y}: {len(df)} stations, {got} with data   ({time.time()-t0:.0f}s)",
              flush=True)

    per_year = pd.concat(frames, ignore_index=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    per_year.to_csv(Path(args.out) / "smap_station_yearly.csv", index=False)
    print(f"Saved: {Path(args.out)/'smap_station_yearly.csv'}   ({len(per_year)} rows)")

    # NaN-aware station mean: a station missing some years is not penalised, one missing
    # every year stays NaN so the coverage report below catches it rather than silently
    # becoming an imputed median downstream.
    agg = per_year.groupby("dir_name")[
        [b for b in BANDS if b in per_year.columns]].mean().reset_index()
    out = st.merge(agg, on="dir_name", how="left")
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
