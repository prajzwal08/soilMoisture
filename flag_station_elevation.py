#!/usr/bin/env python
"""
flag_station_elevation.py
=========================
§45.12.  FLAG disagreements between `elevation_m` in csvs/station_splits.csv and
the MERIT DEM.  Report only -- this script CANNOT write station_splits.csv, and
that is deliberate.

THE POLICY: flag everything, replace nothing.

  * `elevation_m` is a SENTINEL (-99.9, 0.0, -999, -9999, NaN)  -> flag it, and
    record what MERIT says AS A SUGGESTION in a separate column. The suggestion
    is never applied; a human decides.
  * `elevation_m` is a genuine value that DISAGREES with MERIT   -> flag it.
    The station operator knows where their instrument is; a 90 m DEM sampled at a
    reported coordinate does not. A disagreement is evidence that either the
    elevation OR THE COORDINATE is wrong, and which one is a human question.

The original design had an `--execute` branch that patched station_splits.csv.
It was REMOVED rather than left disabled, because `elevation_band` is a
stratification key: `split` was drawn from the ORIGINAL bands, so silently
correcting a value would leave a split whose stratification no longer matches
the column it was built from, and rebalancing would move stations between train
and oos and invalidate every existing trained model. Nothing here should be able
to do that by accident.

WHY THE SENTINELS SURVIVED.  `enrich_station_inventory.py:192` gates its SRTM
back-fill on `df["elevation_m"].isna()`. -99.9 and 0.0 are not NaN, so the fill
skips them and the sentinel flows straight through
`create_evaluation_splits.py:351-355` into `elevation_band`, where
`elev_band(-99.9)` returns "Low".

Usage
-----
    sbatch slurm/flag_station_elevation.sh

Env: `terramind`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO      = Path("/gpfs/work3/0/prjs1968/soilMoisture")
SPLITS    = REPO / "csvs" / "station_splits.csv"
ELEV_CSV  = REPO / "csvs" / "station_elevation_check.csv"
FLAGS_CSV = REPO / "csvs" / "station_elevation_flags.csv"

# Classic nodata codes that are NOT NaN and therefore survive an .isna() gate.
SENTINELS    = [-9999.0, -999.9, -999.0, -99.99, -99.9, -9.999, 0.0]
SENTINEL_TOL = 0.05
DISAGREE_M   = 100.0
BAD_MARGIN_M = 50.0


def is_sentinel(v) -> bool:
    if pd.isna(v):
        return True
    return any(abs(float(v) - s) < SENTINEL_TOL for s in SENTINELS)


def elev_band(e) -> str:
    """Verbatim from create_evaluation_splits.py:351-355."""
    if pd.isna(e) or e < 500:
        return "Low"
    if e < 1500:
        return "Mid"
    return "High"


def main() -> int:
    sp = pd.read_csv(SPLITS)
    ev = pd.read_csv(ELEV_CSV)
    print(f"station_splits.csv : {len(sp)} rows  (READ ONLY -- never written)")
    print(f"elevation check    : {len(ev)} rows, {int((ev['status']=='ok').sum())} sampled ok")

    def _folder(r):
        if r["source_network"] != r["network"]:
            return f"{r['source_network']}_{r['network']}_{r['station_id']}"
        return f"{r['network']}_{r['station_id']}"
    sp["_folder"] = sp.apply(_folder, axis=1)

    m = sp.merge(ev[["station", "merit_elv", "merit_min3", "merit_max3",
                     "elev_from_sp", "status", "srtm_suspect"]],
                 left_on="_folder", right_on="station", how="left", suffixes=("", "_ev"))
    assert len(m) == len(sp), "join changed row count"
    if int(m["merit_elv"].isna().sum()):
        print(f"WARNING: {int(m['merit_elv'].isna().sum())} rows had no MERIT sample")

    orig = m["elevation_m"].astype(float)
    sent = orig.apply(is_sentinel)
    have = m["merit_elv"].notna() & (m["status"] == "ok")
    below = orig < (m["merit_min3"] - BAD_MARGIN_M)
    above = orig > (m["merit_max3"] + BAD_MARGIN_M)
    outside = (below | above).fillna(False) & ~sent
    diff = (orig - m["merit_elv"]).abs()

    def _flag(i) -> str:
        if sent.iloc[i]:
            return "SENTINEL__MERIT_SUGGESTS" if have.iloc[i] else "SENTINEL__NO_DEM"
        if outside.iloc[i]:
            return "DISAGREES_OUTSIDE_LOCAL_RELIEF"
        if pd.notna(diff.iloc[i]) and diff.iloc[i] > DISAGREE_M:
            return "DISAGREES"
        if bool(m["srtm_suspect"].iloc[i]):
            return "PROBABLY_SRTM_BACKFILL"
        return "OK"

    m["elev_flag"] = [_flag(i) for i in range(len(m))]
    # a SUGGESTION only -- nothing downstream reads this, and nothing applies it
    m["merit_suggested_m"] = np.where(sent & have, m["merit_elv"], np.nan)
    m["band_now"] = orig.apply(elev_band)
    m["band_if_merit_used"] = pd.Series(m["merit_suggested_m"]).fillna(orig).apply(elev_band)
    m["band_would_change"] = m["band_now"] != m["band_if_merit_used"]

    print("\n=== flags ===")
    print(m["elev_flag"].value_counts().to_string())

    s = m[sent & have]
    print(f"\n=== {len(s)} SENTINELS — value is a nodata code, not an elevation ===")
    print(s[["_folder", "split", "elevation_m", "merit_suggested_m", "elev_from_sp",
             "band_now", "band_if_merit_used"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    dis = m[m["elev_flag"].str.startswith("DISAGREES")].copy()
    dis["diff"] = dis["elevation_m"] - dis["merit_elv"]
    print(f"\n=== {len(dis)} GENUINE VALUES THAT DISAGREE — KEPT AS-IS ===")
    print("    (wrong elevation, or wrong COORDINATE? a coordinate error would also")
    print("     mis-centre the satellite tiles and the ERA5 pixel — worth a look)")
    print(dis.reindex(dis["diff"].abs().sort_values(ascending=False).index)[
        ["_folder", "split", "elevation_m", "merit_elv", "elev_from_sp", "diff"]]
        .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    ch = m[m["band_would_change"]]
    print(f"\n=== {len(ch)} stations sit in the WRONG elevation_band today ===")
    if len(ch):
        print(ch.groupby(["band_now", "band_if_merit_used", "split"]).size().to_string())
        print("\n  `elevation_band` is a stratification key and `split` was DRAWN from")
        print("  the CURRENT (wrong) bands. Correcting the value later does not")
        print("  retroactively rebalance that split, and rebalancing would move")
        print("  stations between train and oos — invalidating every existing result.")
        print("  Recording the bias is the safe action; re-drawing is a separate call.")

    out = m[["_folder", "split", "elevation_m", "merit_elv", "merit_min3", "merit_max3",
             "elev_from_sp", "elev_flag", "merit_suggested_m", "band_now",
             "band_if_merit_used", "band_would_change"]].rename(
                 columns={"_folder": "station", "elevation_m": "elevation_m_current"})
    out.to_csv(FLAGS_CSV, index=False)
    print(f"\nwrote {FLAGS_CSV}")
    print(f"{SPLITS} was NOT modified — this script has no write path to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
