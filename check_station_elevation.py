#!/usr/bin/env python
"""
check_station_elevation.py
==========================
§45.12.  Cross-check `elevation_m` in csvs/station_splits.csv against the MERIT
DEM, for all 990 stations.

Motivated by ISMN_ROMPS_Baytik, which carries `elevation_m = 0.0` -- the only
station in the file with elevation 0, the worst sp/ISA outlier at 0.825, and a
site in the Kyrgyz Ala-Too whose measured surface pressure inverts to ~1,591 m.
The question is whether it is a one-off or the visible member of a class.

THREE NUMBERS, KEPT DISTINCT.  Collapsing them is how a smoothing artefact gets
mistaken for an error:

    elevation_m           --        station-reported metadata
    ERA5 sp -> ISA        ~9 km     ERA5-Land MODEL OROGRAPHY, not the station
    MERIT elv             ~90 m     actual terrain at the point

MERIT and the pressure estimate both landing far from the metadata settles a
station: two independent sources agreeing rules out coincidence.  MERIT agreeing
with the metadata while ERA5 disagrees is ordinary orography smoothing -- which
is what the >3,000 m SNOTEL stations show, and they are the control here.

CIRCULARITY, DECLARED.  `elevation_m` is station-reported for most rows (ISMN
python_metadata, ICOS attrs, AmeriFlux LOCATION_ELEV), but
`enrich_station_inventory.py:224` back-filled the NaN rows from SRTM via GEE and
left no record of which.  MERIT is itself SRTM3-derived, so for that subset this
check is quasi-circular.  Stations matching MERIT to within SRTM_SUSPECT_M are
therefore reported as PROBABLY SRTM-FILLED, not as independently confirmed.

A 90 m DEM at a reported station coordinate can differ from a valid station
elevation by tens of metres in steep terrain without either being wrong, so the
3x3 neighbourhood range is reported alongside the point value and a station is
only called bad when the metadata falls outside the local relief.

Usage
-----
    sbatch slurm/check_station_elevation.sh

Env: `terramind` (rasterio).
"""
from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from era5_qc_all_stations import (DATA_ROOT, REPO_ROOT, FIG, station_rows,
                                  isa_pressure_Pa)

OUT_CSV = REPO_ROOT / "csvs" / "station_elevation_check.csv"
QC_CSV  = REPO_ROOT / "csvs" / "era5_all_station_qc.csv"

ELV_BAND = 4          # BANDS = ["upa","upg","hnd","elv","dir"] -- elv is 4th, metres
SRTM_SUSPECT_M = 1.0  # metadata this close to MERIT is probably an SRTM back-fill
BAD_MARGIN_M   = 50.0 # metadata must miss the 3x3 relief by this much to be "bad"


def isa_elevation_m(sp_Pa: float) -> float:
    """Invert the ISA profile: pressure -> the elevation that would produce it."""
    if not np.isfinite(sp_Pa) or sp_Pa <= 0:
        return np.nan
    return (1.0 - (sp_Pa / 101325.0) ** (1.0 / 5.25588)) / 2.25577e-5


def sample(task) -> dict:
    folder, cat, lat, lon, elev_meta, kg, split = task
    o = {"station": folder, "cat": cat, "lat": lat, "lon": lon,
         "elev_meta": elev_meta, "koppen": kg, "split": split,
         "merit_elv": np.nan, "merit_min3": np.nan, "merit_max3": np.nan,
         "in_window": False, "status": "error", "msg": ""}
    try:
        tif = DATA_ROOT / cat / folder / "MERIT" / "merit_hydro_25km.tif"
        if not tif.exists():
            o["status"] = "no-tif"; return o
        with rasterio.open(tif) as ds:
            if ds.count < ELV_BAND:
                o["status"] = "bad-bands"; o["msg"] = f"{ds.count} bands"; return o
            r, c = ds.index(lon, lat)
            h, w = ds.height, ds.width
            o["in_window"] = bool(0 <= r < h and 0 <= c < w)
            if not o["in_window"]:
                o["status"] = "outside-window"
                o["msg"] = f"row={r} col={c} of {h}x{w}"
                return o
            r0, r1 = max(0, r - 1), min(h, r + 2)
            c0, c1 = max(0, c - 1), min(w, c + 2)
            win = ds.read(ELV_BAND,
                          window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
            point = float(win[r - r0, c - c0])
            # the tif is self-describing; assert we opened the right station's tile
            tags = ds.tags()
            try:
                o["tag_dist_km"] = float(np.hypot(
                    (float(tags["station_lat"]) - lat) * 111.32,
                    (float(tags["station_lon"]) - lon) * 111.32 * math.cos(math.radians(lat))))
            except (KeyError, ValueError, TypeError):
                o["tag_dist_km"] = np.nan

        if not np.isfinite(point):
            o["status"] = "nodata"; return o
        o["merit_elv"]  = point
        o["merit_min3"] = float(np.nanmin(win))
        o["merit_max3"] = float(np.nanmax(win))
        o["status"] = "ok"
    except Exception as exc:
        o["msg"] = str(exc)[:200]
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    rows = station_rows()
    tasks = list(rows.itertuples(index=False, name=None))
    print(f"sampling MERIT band {ELV_BAND} (elv, m) for {len(tasks)} stations")

    with Pool(args.workers) as pool:
        d = pd.DataFrame(pool.map(sample, tasks, chunksize=1))

    # join the ERA5 pressure leg
    if QC_CSV.exists():
        qc = pd.read_csv(QC_CSV)[["station", "sp_mean", "sp_isa", "sp_ratio"]]
        d = d.merge(qc, on="station", how="left")
        d["elev_from_sp"] = d["sp_mean"].apply(isa_elevation_m)
    else:
        d["sp_mean"] = d["sp_ratio"] = d["elev_from_sp"] = np.nan
        print(f"WARNING: {QC_CSV} missing -- pressure leg unavailable")

    d["resid_merit"] = d["elev_meta"] - d["merit_elv"]
    d["resid_sp"]    = d["elev_meta"] - d["elev_from_sp"]
    d["relief3"]     = d["merit_max3"] - d["merit_min3"]
    # only "bad" when the metadata misses the LOCAL RELIEF, not the exact pixel
    below = d["elev_meta"] < (d["merit_min3"] - BAD_MARGIN_M)
    above = d["elev_meta"] > (d["merit_max3"] + BAD_MARGIN_M)
    d["metadata_bad"] = (below | above).fillna(False)
    d["srtm_suspect"] = (d["resid_merit"].abs() < SRTM_SUSPECT_M).fillna(False)

    d.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(d)} rows)\n")

    print("--- status ---")
    print(d["status"].value_counts().to_string())
    bad_tool = d[d["status"] != "ok"]
    if len(bad_tool):
        print("\nTOOLING FAILURES (not evidence of anything about the data):")
        print(bad_tool[["station", "status", "msg"]].to_string(index=False))

    g = d[d["status"] == "ok"].copy()
    far = g["tag_dist_km"] > 1.0
    if far.any():
        print(f"\nWARNING: {int(far.sum())} station(s) >1 km from their tif's own "
              f"station_lat/lon tag -- wrong tile?")
        print(g.loc[far, ["station", "tag_dist_km"]].to_string(index=False))

    print(f"\n=== metadata vs MERIT, {len(g)} stations ===")
    print(f"  residual (meta - MERIT):  median {g['resid_merit'].median():+.1f} m   "
          f"MAD {(g['resid_merit'] - g['resid_merit'].median()).abs().median():.1f} m")
    print(f"  |residual| > 100 m     :  {int((g['resid_merit'].abs() > 100).sum())} stations")
    print(f"  metadata outside 3x3 relief +/- {BAD_MARGIN_M:.0f} m : "
          f"{int(g['metadata_bad'].sum())} stations")
    print(f"\n  PROBABLY SRTM-BACK-FILLED (|resid| < {SRTM_SUSPECT_M} m, so NOT "
          f"independent evidence): {int(g['srtm_suspect'].sum())} stations")
    print(f"  independently consistent  : "
          f"{int((~g['srtm_suspect'] & ~g['metadata_bad']).sum())} stations")

    print("\n=== 15 worst |metadata - MERIT| ===")
    w = g.reindex(g["resid_merit"].abs().sort_values(ascending=False).index).head(15)
    print(w[["station", "lat", "lon", "elev_meta", "merit_elv", "merit_min3",
             "merit_max3", "elev_from_sp", "resid_merit", "metadata_bad"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\n=== the station that started this ===")
    b = g[g["station"] == "ISMN_ROMPS_Baytik"]
    if len(b):
        r = b.iloc[0]
        print(f"  ISMN_ROMPS_Baytik  ({r['lat']:.5f} N, {r['lon']:.5f} E)")
        print(f"    metadata elevation_m : {r['elev_meta']:.1f} m")
        print(f"    MERIT elv at point   : {r['merit_elv']:.1f} m   "
              f"(3x3 range {r['merit_min3']:.1f} .. {r['merit_max3']:.1f} m)")
        print(f"    ERA5 sp -> ISA       : {r['elev_from_sp']:.1f} m   "
              f"(9 km model orography)")
        print(f"    verdict              : "
              f"{'METADATA IS WRONG' if r['metadata_bad'] else 'metadata consistent'}")

    print("\n=== control: the >3000 m stations must AGREE with MERIT while "
          "disagreeing with ERA5 ===")
    ctl = g[g["elev_meta"] > 3000].reindex(
        g[g["elev_meta"] > 3000]["sp_ratio"].sort_values(ascending=False).index).head(8)
    print(ctl[["station", "elev_meta", "merit_elv", "resid_merit",
               "elev_from_sp", "resid_sp", "sp_ratio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    plot(g)
    return 0


def plot(g: pd.DataFrame) -> None:
    C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
    fig, a = plt.subplots(1, 3, figsize=(17, 5.4))
    bad = g["metadata_bad"]

    lim = [-200, max(g["elev_meta"].max(), g["merit_elv"].max()) + 300]
    a[0].plot(lim, lim, "k--", lw=1, zorder=1, label="1:1")
    a[0].scatter(g.loc[~bad, "merit_elv"], g.loc[~bad, "elev_meta"], s=14, c=C1,
                 alpha=0.6, lw=0, label="consistent")
    a[0].scatter(g.loc[bad, "merit_elv"], g.loc[bad, "elev_meta"], s=46, c=C2,
                 lw=0.6, edgecolor="k", zorder=4,
                 label=f"metadata outside local relief (n={int(bad.sum())})")
    for _, r in g[bad].iterrows():
        a[0].annotate(r["station"].split("_", 1)[-1], (r["merit_elv"], r["elev_meta"]),
                      fontsize=6, xytext=(4, 3), textcoords="offset points")
    a[0].set_xlim(lim); a[0].set_ylim(lim)
    a[0].set_xlabel("MERIT elv at station (m, ~90 m)")
    a[0].set_ylabel("station_splits.csv elevation_m")
    a[0].set_title("(a) metadata vs DEM", loc="left", fontsize=10)
    a[0].legend(fontsize=8, frameon=False, loc="upper left")

    a[1].plot(lim, lim, "k--", lw=1, zorder=1, label="1:1")
    a[1].scatter(g["merit_elv"], g["elev_from_sp"], s=14, c=C3, alpha=0.6, lw=0)
    a[1].set_xlim(lim); a[1].set_ylim(lim)
    a[1].set_xlabel("MERIT elv (m, ~90 m)")
    a[1].set_ylabel("ERA5 sp inverted through ISA (m, ~9 km)")
    a[1].set_title("(b) the two DEM-independent legs — scatter here is\n"
                   "    orography smoothing, not error", loc="left", fontsize=10)
    a[1].legend(fontsize=8, frameon=False, loc="upper left")

    a[2].axhline(0, color="k", lw=0.8)
    a[2].axvline(1.0, color="k", lw=0.8)
    a[2].scatter(g.loc[~bad, "sp_ratio"], g.loc[~bad, "resid_merit"], s=14, c=C1,
                 alpha=0.6, lw=0)
    a[2].scatter(g.loc[bad, "sp_ratio"], g.loc[bad, "resid_merit"], s=46, c=C2,
                 lw=0.6, edgecolor="k", zorder=4)
    a[2].set_xlabel("sp / ISA(elevation_m)")
    a[2].set_ylabel("elevation_m − MERIT elv  (m)")
    a[2].set_title("(c) the separation: bad metadata leaves the origin\n"
                   "    in BOTH axes; smoothing moves only x", loc="left", fontsize=10)

    fig.suptitle("§45.12 — station elevation vs MERIT DEM", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "station_elevation_check.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
