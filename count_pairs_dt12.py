#!/usr/bin/env python
"""Re-pair ECOSTRESS day/night under a CLEAR-FIRST, dt<=12 h rule and count images.

The shipped census (census_ecostress.py:613) pairs BEFORE cloud is known and bounds the
night half by next-day solar noon.  This script asks the other question:

    filter each granule on clarity FIRST, then pair a day pass with the night that
    follows it within DT_MAX hours

Filtering first matters because a day pass whose greedy partner was cloudy is free to
re-pair with a different, clear night -- which post-hoc filtering of the shipped pair
inventory cannot express.

Variants reported
  V0  baseline rule, no cloud filter          reconciliation against the 60,461 census pairs
  V1  dt<=DT_MAX, no cloud filter             GEOMETRIC ceiling of the new rule
  V2  clear-first, then dt<=DT_MAX            the answer, over granules whose cloud is KNOWN
  V3  shipped usable pairs cut to dt<=DT_MAX  what post-hoc filtering alone would give

Cloud is known only for the 119,566 granules that entered a census pair; granules that
never paired were never read.  V2 is therefore a LOWER BOUND and the script prints the
coverage fraction so the gap is visible rather than assumed away.
"""
from __future__ import annotations

import glob
import os
from bisect import bisect_left, bisect_right
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = "/gpfs/work3/0/prjs1968/soilMoisture"
GRAN = f"{ROOT}/csvs/ecostress_census_granules.stats.csv"
READS = sorted(glob.glob(f"{ROOT}/csvs/ecostress_wp_reads.wp*.csv"))
PAIRS = f"{ROOT}/csvs/ecostress_wp_pairs.wp.csv"

# (lo, hi) in hours: the night half must follow the day pass by dt in [lo, hi)
DT_WINDOWS = [(6.0, 9.0), (0.0, 12.0), (12.0, 24.0), (15.0, 19.0), (6.0, 19.0),
              (0.0, 26.0)]

OUT_BANDS = f"{ROOT}/csvs/ecostress_dt_bands_by_station.csv"
OUT_DT = f"{ROOT}/csvs/ecostress_clearfirst_dt.csv"
CLEAR_MIN = 0.5  # clear_frac threshold, matching the §36 QC pass
NPROC = 64

HOUR_NS = 3600.0 * 1e9


# ------------------------------------------------------------------
# pairing
# ------------------------------------------------------------------
def greedy_pair(day_t, day_ur, night_t, night_ur, lo_t, hi_t):
    """Greedy one-to-one, days ascending, LATEST qualifying night wins.

    Identical policy to census_ecostress.py:654-672 -- only the bounds change.
    `lo_t[i]` / `hi_t[i]` are the inclusive lower / exclusive upper bound in ns for day i.
    nights must be sorted.
    """
    used = np.zeros(len(night_t), dtype=bool)
    out = []
    for i in range(len(day_t)):
        td = day_t[i]
        lo_j = bisect_left(night_t, lo_t[i])  # at or after the lower bound
        hi_j = bisect_left(night_t, hi_t[i])  # strictly before the upper bound
        # latest first: the later in the nocturnal period, the nearer the pre-dawn min
        for j in range(hi_j - 1, lo_j - 1, -1):
            if not used[j]:
                used[j] = True
                out.append((day_ur[i], night_ur[j], (night_t[j] - td) / HOUR_NS))
                break
    return out


def station_job(arg):
    sid, df, clear_ok = arg
    d = df[df["phase"] == "day"].sort_values("t")
    n = df[df["phase"] == "night"].sort_values("t")

    # STEP 2 -- QC on each granule individually, BEFORE any pairing happens
    dc = d[d["granule_ur"].isin(clear_ok)]
    nc = n[n["granule_ur"].isin(clear_ok)]

    res = {}

    def run(dd, nn, lo_h, hi_h):
        if len(dd) == 0 or len(nn) == 0:
            return []
        dt_ns = dd["t"].to_numpy()
        if hi_h is None:
            # baseline: next-day solar noon, reconstructed from hours_from_solar_noon
            lo = dt_ns + 1  # strictly after the day pass
            hi = dt_ns - (dd["hfsn"].to_numpy() * HOUR_NS) + 24.0 * HOUR_NS
        else:
            lo = dt_ns + lo_h * HOUR_NS
            hi = dt_ns + hi_h * HOUR_NS
        return greedy_pair(dt_ns.tolist(), dd["granule_ur"].tolist(),
                           nn["t"].to_numpy().tolist(), nn["granule_ur"].tolist(),
                           lo.tolist(), hi.tolist())

    res["baseline"] = run(d, n, None, None)
    for lo_h, hi_h in DT_WINDOWS:
        res[f"geom_{lo_h:g}_{hi_h:g}"] = run(d, n, lo_h, hi_h)
        res[f"clear_{lo_h:g}_{hi_h:g}"] = run(dc, nc, lo_h, hi_h)

    known = set(df.loc[df["known"], "granule_ur"])
    urs = {u for k, p in res.items() if k.startswith("geom_") for x in p for u in x[:2]}
    res["_cov"] = (len(urs & known), len(urs))
    return sid, res


# ------------------------------------------------------------------
def main():
    print(f"DT_WINDOWS={DT_WINDOWS}   CLEAR_MIN={CLEAR_MIN}   nproc={NPROC}", flush=True)

    # ---- cloud verdicts (only for granules that entered a census pair) ----
    rd = pd.concat([pd.read_csv(f, usecols=["station_id", "granule_ur", "read_ok",
                                            "clear_frac"]) for f in READS],
                   ignore_index=True)
    rd = rd.drop_duplicates(subset=["station_id", "granule_ur"], keep="last")
    rd_ok = rd[rd["read_ok"] == 1]
    print(f"cloud reads: {len(rd):,} rows, {len(rd_ok):,} read_ok, "
          f"{rd_ok['station_id'].nunique():,} stations", flush=True)
    clear_by_st = (rd_ok[rd_ok["clear_frac"] >= CLEAR_MIN]
                   .groupby("station_id")["granule_ur"].apply(set).to_dict())
    known_by_st = rd.groupby("station_id")["granule_ur"].apply(set).to_dict()

    # ---- in-window granule inventory ----
    g = pd.read_csv(GRAN, usecols=["station_id", "granule_ur", "utc", "phase",
                                   "hours_from_solar_noon"])
    n_raw = len(g)
    g = g.drop_duplicates(subset=["station_id", "granule_ur"])
    print(f"granules: {n_raw:,} rows -> {len(g):,} unique (station, granule); "
          f"{g['station_id'].nunique():,} stations", flush=True)
    g = g[g["phase"].isin(("day", "night"))].copy()
    g["t"] = pd.to_datetime(g["utc"], format="ISO8601", utc=True).astype("int64")
    g = g.rename(columns={"hours_from_solar_noon": "hfsn"})
    print(f"in-window (day|night): {len(g):,}  "
          f"day={int((g['phase'] == 'day').sum()):,} "
          f"night={int((g['phase'] == 'night').sum()):,}", flush=True)

    jobs = []
    for sid, df in g.groupby("station_id", sort=False):
        df = df.copy()
        df["known"] = df["granule_ur"].isin(known_by_st.get(sid, set()))
        jobs.append((sid, df, clear_by_st.get(sid, set())))

    with Pool(NPROC) as pool:
        results = pool.map(station_job, jobs, chunksize=4)

    # ---- phase of the night half, so the regime is visible not assumed ----
    pw = pd.read_csv(PAIRS, usecols=["station_id", "day_ur", "night_ur", "dt_hours",
                                     "night_tst", "quality", "both_read"])
    pw["noff"] = np.where(pw["night_tst"] < 12.0, pw["night_tst"], pw["night_tst"] - 24.0)
    phase_of = dict(zip(zip(pw["station_id"], pw["night_ur"]), pw["noff"]))

    def tally(key):
        npairs = st = imgs = 0
        urs, pre, eve, unk = set(), 0, 0, 0
        for sid, res in results:
            p = res.get(key) or []
            if not p:
                continue
            st += 1
            npairs += len(p)
            imgs += 2 * len(p)
            for a, b, _ in p:
                urs.add(a)
                urs.add(b)
                o = phase_of.get((sid, b))
                if o is None:
                    unk += 1
                elif o >= 0:
                    pre += 1
                else:
                    eve += 1
        return npairs, st, imgs, len(urs), pre, eve, unk

    print("\n" + "=" * 88)
    print(f"{'variant':<30}{'pairs':>9}{'stations':>9}{'images':>10}{'uniq':>9}"
          f"{'pre-dawn':>10}{'evening':>9}{'unk':>7}")
    print("=" * 88)
    b = tally("baseline")
    print(f"{'baseline rule, no cloud':<30}{b[0]:>9,}{b[1]:>9,}{b[2]:>10,}{b[3]:>9,}"
          f"{b[4]:>10,}{b[5]:>9,}{b[6]:>7,}")
    for lo_h, hi_h in DT_WINDOWS:
        for pre_k, lab in (("geom", "no cloud"), ("clear", "CLEAR-FIRST")):
            t = tally(f"{pre_k}_{lo_h:g}_{hi_h:g}")
            name = f"dt [{lo_h:g},{hi_h:g}) h  {lab}"
            print(f"{name:<30}{t[0]:>9,}{t[1]:>9,}{t[2]:>10,}{t[3]:>9,}"
                  f"{t[4]:>10,}{t[5]:>9,}{t[6]:>7,}")
        print("-" * 88)

    use = pw[(pw["quality"] == 1) & (pw["both_read"] == 1)]
    print(f"(shipped usable inventory, all dt: {len(use):,} pairs / "
          f"{use['station_id'].nunique():,} stations / {2 * len(use):,} images)")

    cov_k = sum(r["_cov"][0] for _, r in results)
    cov_n = sum(r["_cov"][1] for _, r in results)
    print(f"cloud coverage of the no-cloud arms' granules: {cov_k:,}/{cov_n:,} "
          f"= {100.0 * cov_k / max(cov_n, 1):.1f}%")

    # ---- dumps for the bias diagnostics ----
    meta = (pd.read_csv(GRAN, usecols=["station_id", "network", "lat", "lon",
                                       "elevation_m", "kg_macro"])
            .drop_duplicates(subset=["station_id"]).set_index("station_id"))

    per_st = {}
    for sid, res in results:
        row = {}
        for lo_h, hi_h in DT_WINDOWS:
            row[f"n_{lo_h:g}_{hi_h:g}"] = len(res.get(f"clear_{lo_h:g}_{hi_h:g}") or [])
        per_st[sid] = row
    bands = pd.DataFrame.from_dict(per_st, orient="index")
    bands.index.name = "station_id"
    bands = bands.join(meta, how="left").reset_index()
    bands.to_csv(OUT_BANDS, index=False)
    print(f"\nwrote {OUT_BANDS}  ({len(bands):,} stations)")

    # every clear-first pair from the UNRESTRICTED window, for the dt histogram
    rows_dt = []
    for sid, res in results:
        for _, nur, dt in (res.get("clear_0_26") or []):
            rows_dt.append((sid, round(dt, 3), phase_of.get((sid, nur), np.nan)))
    dtdf = pd.DataFrame(rows_dt, columns=["station_id", "dt_hours", "night_offset"])
    dtdf.to_csv(OUT_DT, index=False)
    print(f"wrote {OUT_DT}  ({len(dtdf):,} pairs)")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
