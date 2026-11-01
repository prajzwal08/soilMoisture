"""§68.5 gates S1/S2 for the imagery downloads: completeness for every smoke tile, and identity of
the CONTROL tiles (centred exactly on a TWENTE station) against that station's existing raw store.

For each control tile and modality, on the dates present in BOTH the fresh download and the store:
  S2  int16 DN : fraction of identical pixels, max |diff|        (store = cloud-filtered subset)
  S1  dB       : median / p99 |diff| (store is float16)
  DEM m        : median |diff|                                   (MPC vs CDSE, both GLO-30)
Also: no negative S2 DN (the NaN->int16 defect), S2 failed-scene count in metadata.json.

Usage: python verify_nl_download.py --csv <root>/station_splits.csv --scratch <nl/satellite> [--modalities s2 s1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr

STORE = Path("/gpfs/work3/0/prjs1968/satellite_zarr")


def _dates(arr):
    return [d.decode() if isinstance(d, bytes) else str(d) for d in arr[:]]


def check_s2(tile_dir, store, out):
    tifs = sorted((tile_dir / "S2L2A").glob("*.tif"))
    out["s2_n"] = len(tifs)
    meta = json.loads((tile_dir / "metadata.json").read_text()) if (tile_dir / "metadata.json").exists() else {}
    out["s2_failed"] = meta.get("S2_failed_scenes", "n/a")
    neg = 0
    for t in tifs[:: max(1, len(tifs) // 20)]:                      # sample ~20 scenes for negatives
        with rasterio.open(t) as src:
            neg += int((src.read() < 0).sum())
    out["s2_negative_px_sampled"] = neg
    if store is None:
        return
    sd = _dates(store["s2/dates"])
    fresh = {t.stem: t for t in tifs}
    shared = [d for d in sd if d in fresh]
    out["s2_store_n"], out["s2_shared"] = len(sd), len(shared)
    out["s2_store_missing_in_fresh"] = len(set(sd) - set(fresh))
    same, maxd, bad = [], 0, []
    scenes = meta.get("S2L2A", {})
    for d in shared:                                                # every shared scene, so mismatches are listed
        with rasterio.open(fresh[d]) as src:
            a = src.read().astype(np.int32)
        b = store["s2/data"][sd.index(d)].astype(np.int32)
        same.append(float((a == b).mean()))
        maxd = max(maxd, int(np.abs(a - b).max()))
        if same[-1] < 0.99:
            m = (a != 0) & (b != 0)
            pb = str(scenes.get(d, {}).get("processing_baseline", "?"))
            off = int(np.median((b - a)[m])) if m.any() else None
            # KNOWN store defect (logs 2026-10-06): harmonize_s2_pre2022.py skipped bright pre-2022 scenes whose min DN
            # was already >= 1000, so the store is exactly 1000 LOWER on those. Accepted; anything else fails.
            known = (off == -1000 and d < "20220125" and pb not in ("", "?") and float(pb) < 4.0
                     and bool(((b - a)[m] == -1000).all()))
            bad.append(f"{d}:pb{pb}:store-fresh={off if off is not None else 'nodata'}"
                       f":{'KNOWN_skip_defect' if known else 'UNEXPLAINED'}")
    out["s2_identical_frac_min"] = round(min(same), 4) if same else None
    out["s2_maxabs"] = maxd
    out["s2_n_mismatch"] = len(bad)
    out["s2_n_unexplained"] = sum(x.endswith("UNEXPLAINED") for x in bad)
    if bad:
        print(f"{tile_dir.name} S2 mismatching scenes ({len(bad)}): " + " ".join(bad))


def check_s1(tile_dir, store, out):
    for orb, key in (("ASC", "s1_asc"), ("DESC", "s1_desc")):
        tifs = {t.stem.split("_")[0]: t for t in (tile_dir / "S1RTC").glob(f"*_{orb}.tif")}
        out[f"{key}_n"] = len(tifs)
        if store is None or key not in store:
            continue
        sd = _dates(store[f"{key}/dates"])
        shared = [d for d in sd if d in tifs]
        out[f"{key}_store_n"], out[f"{key}_shared"] = len(sd), len(shared)
        diffs = []
        for d in shared[:: max(1, len(shared) // 25)]:
            with rasterio.open(tifs[d]) as src:
                a = src.read().astype(np.float32)
            b = store[f"{key}/data"][sd.index(d)].astype(np.float32)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                diffs.append(np.abs(a[m] - b[m]))
        if diffs:
            dd = np.concatenate(diffs)
            out[f"{key}_med_absdiff_db"] = round(float(np.median(dd)), 4)
            out[f"{key}_p99_absdiff_db"] = round(float(np.percentile(dd, 99)), 4)


def check_dem(tile_dir, store, out):
    f = tile_dir / "DEM" / "dem.tif"
    out["dem"] = f.exists()
    if not f.exists() or store is None:
        return
    with rasterio.open(f) as src:
        a = src.read(1).astype(np.float32)
    b = store["dem/data"][0].astype(np.float32)
    out["dem_med_absdiff_m"] = round(float(np.median(np.abs(a - b))), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--modalities", nargs="+", default=["s2", "s1"])
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    rows = []
    for r in df.itertuples():
        out = {"tile": r.dir_name, "control_of": r.control_of if isinstance(r.control_of, str) else ""}
        tile_dir = Path(args.scratch) / r.dir_name
        store = None
        if out["control_of"]:
            p = STORE / f"{out['control_of']}.zarr"
            store = zarr.open(str(p), mode="r") if p.exists() else None
            out["store"] = p.exists()
        if "s2" in args.modalities:
            check_s2(tile_dir, store, out)
            check_dem(tile_dir, store, out)
        if "s1" in args.modalities:
            check_s1(tile_dir, store, out)
        rows.append(out)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.T.to_string())

    ok = True
    if "s2" in args.modalities:
        ok &= bool((res.s2_n > 0).all()) and bool((res.s2_negative_px_sampled == 0).all())
        ok &= bool((res.s2_failed.astype(str) == "0").all())        # "n/a" = metadata lost -> unverified, FAIL
        c = res[res.control_of != ""]
        if "s2_n_unexplained" in c:
            ok &= bool((c.s2_n_unexplained.fillna(1) == 0).all())       # every mismatch must be the known skip defect
        if "dem_med_absdiff_m" in c:
            ok &= bool((c.dem_med_absdiff_m.fillna(99) <= 1.0).all())
    if "s1" in args.modalities:
        ok &= bool(((res.s1_asc_n + res.s1_desc_n) > 0).all())
        c = res[res.control_of != ""]
        for k in ("s1_asc_med_absdiff_db", "s1_desc_med_absdiff_db"):
            if k in c:
                ok &= bool((c[k].fillna(99) <= 0.05).all())
    print("GATE", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
