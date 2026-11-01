"""
probe_s2_missing.py — WHY were these S2 scenes never obtained? (Session 44)
===========================================================================
For a few stations, re-attempts scenes the audit (csvs/s2_coverage_audit.csv) found missing,
plus a few scenes the station DID obtain as a control, through the downloader's own path
(planetary_computer.sign_inplace + stackstac.stack + center_crop). ONE attempt per scene, no
retry, so the raw failure is recorded. Nothing is saved except csvs/s2_probe_missing.csv.

Outcome per scene:  OK (loads now -> the original failure was transient)
                    EXC:<type>:<msg>  (fails now -> a persistent reason)
                    OK_ALLNAN / OK_PARTIAL  (loads but carries no / partial data)
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import stackstac
from rasterio.enums import Resampling

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from download_s2_mpc import MAX_CLOUD, MPC_URL, S2_BANDS, RES_M, center_crop, station_grid  # noqa: E402
from splits_config import station_dir_name  # noqa: E402

STATIONS = ["ISMN_SCAN_Price", "ISMN_SNOTEL_Coldfoot", "ISMN_SNOTEL_MedBow",
            "ISMN_SCAN_ReeseCenter", "ISMN_SMOSMANIA_Condom", "ISMN_TWENTE_Hupsel"]
N_MISSING, N_CONTROL = 12, 4

store = json.loads((REPO / "csvs" / "_s2_store_dates.json").read_text())
dl = pd.read_csv(REPO / "text" / "cloudy_tile_manifest_delete_log.csv", usecols=["station", "date"])
df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
cat = pystac_client.Client.open(MPC_URL)
rng = np.random.default_rng(0)
rows = []

for name in STATIONS:
    r = next(r for _, r in df.iterrows() if station_dir_name(r) == name)
    epsg, bounds, bbox = station_grid(float(r.latitude), float(r.longitude))
    lo, hi = max(int(r.start_date), 20160101), int(r.end_date)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                            datetime="2016-01-01/2025-12-31",
                            query={"eo:cloud_cover": {"lt": MAX_CLOUD}}).items())
    got = set(store.get(name) or []) | set(dl[dl.station == name].date.astype(int))
    by_date = {}
    for it in items:
        d = int(it.datetime.strftime("%Y%m%d"))
        if lo <= d <= hi:
            by_date.setdefault(d, []).append(it)
    missing = sorted(d for d in by_date if d not in got)
    present = sorted(d for d in by_date if d in got)
    pick_m = sorted(rng.choice(missing, min(N_MISSING, len(missing)), replace=False)) if missing else []
    pick_c = sorted(rng.choice(present, min(N_CONTROL, len(present)), replace=False)) if present else []
    print(f"\n=== {name}: window {lo}-{hi}, catalogue dates {len(by_date)}, missing {len(missing)}, "
          f"obtained {len(present)}", flush=True)
    # How the downloader ordered them: newest first? Where did the missing ones sit?
    order = [int(it.datetime.strftime("%Y%m%d")) for it in items if lo <= int(it.datetime.strftime("%Y%m%d")) <= hi]
    print(f"    catalogue order: first {order[:3]} ... last {order[-3:]}  "
          f"(newest-first = {order[0] > order[-1]})", flush=True)

    for kind, dates in (("missing", pick_m), ("control", pick_c)):
        for d in dates:
            it = by_date[int(d)][0]
            t0 = time.time()
            try:
                planetary_computer.sign_inplace(it)
                da = stackstac.stack([it], assets=S2_BANDS, epsg=epsg, resolution=RES_M,
                                     bounds=bounds, rescale=False,
                                     resampling=Resampling.bilinear).squeeze("time").compute()
                da = center_crop(da)
                v = da.values
                nan = float(np.isnan(v).mean()) if np.issubdtype(v.dtype, np.floating) else 0.0
                zero = float((np.nan_to_num(v) == 0).mean())
                out = "OK_ALLNAN" if nan > 0.99 else ("OK_PARTIAL" if nan > 0.01 else "OK")
                detail = f"nan={nan:.3f} zero={zero:.3f} shape={tuple(v.shape)}"
            except Exception as e:                                    # noqa: BLE001
                out = f"EXC:{type(e).__name__}"
                detail = str(e).replace("\n", " ")[:300]
            rows.append(dict(station=name, kind=kind, date=int(d), n_items=len(by_date[int(d)]),
                             baseline=it.properties.get("s2:processing_baseline", ""),
                             tile=it.properties.get("s2:mgrs_tile", ""),
                             outcome=out, seconds=round(time.time() - t0, 1), detail=detail))
            print(f"    {kind:7s} {d}  pb={rows[-1]['baseline']:>5s}  tile={rows[-1]['tile']}  "
                  f"{out:22s} {detail[:140]}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(REPO / "csvs" / "s2_probe_missing.csv", index=False)
print("\n=== outcome by kind")
print(res.groupby(["kind", "outcome"]).size().to_string())
print("\n=== outcome by processing baseline (missing only)")
print(res[res.kind == "missing"].groupby(["baseline", "outcome"]).size().to_string())
