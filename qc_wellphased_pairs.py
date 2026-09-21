#!/usr/bin/env python
"""§36.24 -- run the IMAGE QC pass over the WELL-PHASED pairs only.

Why this exists.  census_ecostress.py in its full (non --dry-run) form opens a COG for
every in-window overpass: 988,065 of them.  Only ~121k of those granules ever enter a
candidate pair, and only ~56k enter a WELL-PHASED pair -- the census reads ~17x more
than the question needs.  Since --dry-run already wrote the full pair inventory to disk,
the cheap move is to read the masks for exactly the granules named in that file and join
the result back.  No CMR query is repeated; pairing is not recomputed.

What it answers: of the 28,119 well-phased pairs, how many survive the QC + cloud +
water masks on BOTH halves -- i.e. the first honest estimate of the usable DTR sample.

Threshold sweep, not a single verdict: clear_frac is recorded per granule, so pair
survival is reported at several CLEAR_FRAC_MIN values rather than fixing 0.5 by fiat.

Resume-safe: reads are appended to the reads CSV as they land and skipped on restart,
so a wall-clock kill costs nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from census_ecostress import (  # noqa: E402
    ROOT, STATION_CSV, LOG_DIR, N_PX_EXPECTED, WINDOW_FRAC_MIN, CLEAR_FRAC_MIN,
    configure_gdal, read_station_window, setup_logging,
)

READ_COLS = ["station_id", "granule_ur", "half", "lat", "lon",
             "read_ok", "n_px", "window_frac", "clipped", "clear_frac", "valid_frac",
             "vza_mean_abs", "vza_max_abs", "frac_mand00", "frac_mand01",
             "frac_lstacc_ge2", "frac_water", "frac_cloud", "passed_qc", "error"]

SWEEP = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]


def append_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=READ_COLS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "csvs" / "ecostress_census_pairs.stats.csv"))
    ap.add_argument("--out-tag", default="wp")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="read only the first N granules -- smoke test")
    ap.add_argument("--all-pairs", action="store_true",
                    help="do NOT restrict to well_phased==1")
    ap.add_argument("--fresh", action="store_true", help="ignore the reads checkpoint")
    ap.add_argument("--shard", type=int, default=0, help="this task's index, 0..nshards-1")
    ap.add_argument("--nshards", type=int, default=1,
                    help=">1 spreads the reads over that many SLURM array tasks")
    ap.add_argument("--report-only", action="store_true",
                    help="skip reading; merge every shard CSV and print the tables")
    args = ap.parse_args()

    setup_logging("qc_wellphased")   # configures root; returns None
    log = logging.getLogger("qc_wellphased")
    configure_gdal()

    # One CSV PER SHARD.  Eight tasks appending to one file would interleave partial
    # lines -- a checkpoint that corrupts itself under the very parallelism it exists to
    # support.  The report step merges them.
    base = ROOT / "csvs" / f"ecostress_wp_reads.{args.out_tag}.csv"
    reads_csv = (base if args.nshards <= 1
                 else ROOT / "csvs" / f"ecostress_wp_reads.{args.out_tag}.s{args.shard}.csv")
    shard_glob = sorted(ROOT.glob(f"csvs/ecostress_wp_reads.{args.out_tag}*.csv"))
    out_csv = ROOT / "csvs" / f"ecostress_wp_pairs.{args.out_tag}.csv"
    if args.fresh and reads_csv.exists():
        reads_csv.unlink()

    pairs = pd.read_csv(args.pairs)
    n_all = len(pairs)
    # The census appends per station and a station reprocessed across a checkpoint resume
    # lands twice; 223 of the well-phased rows are exact duplicates.  Drop them here or
    # every count downstream is overstated.
    pairs = pairs.drop_duplicates(subset=["station_id", "day_ur", "night_ur"])
    if not args.all_pairs:
        pairs = pairs[pairs["well_phased"] == 1].copy()
    log.info("pairs file        : %s", args.pairs)
    log.info("pairs total       : %d", n_all)
    log.info("pairs selected    : %d  (%s)", len(pairs),
             "all" if args.all_pairs else "well_phased==1")
    log.info("stations selected : %d", pairs["station_id"].nunique())

    st = pd.read_csv(STATION_CSV)
    coord = st.set_index("station_id")[["latitude", "longitude"]].to_dict("index")
    missing = sorted(set(pairs["station_id"]) - set(coord))
    if missing:
        log.warning("%d pair stations absent from station_splits.csv: %s",
                    len(missing), missing[:5])
        pairs = pairs[~pairs["station_id"].isin(missing)]

    # One read task per (station, granule).  NOT deduped across stations: the read is a
    # 2.24 km window centred on THAT station, so the same granule at two stations is two
    # different windows and two different answers.
    tasks: dict[tuple[str, str], dict] = {}
    for half, col in (("day", "day_ur"), ("night", "night_ur")):
        for sid, ur in zip(pairs["station_id"], pairs[col]):
            key = (sid, ur)
            if key not in tasks:
                c = coord[sid]
                tasks[key] = {"station_id": sid, "granule_ur": ur, "half": half,
                              "lat": float(c["latitude"]), "lon": float(c["longitude"])}

    # The checkpoint is EVERY shard's file plus the pre-shard run, so restarting with a
    # different shard count never re-reads what is already measured.
    done: set[tuple[str, str]] = set()
    for f in sorted(ROOT.glob(f"csvs/ecostress_wp_reads.{args.out_tag}*.csv")):
        prev = pd.read_csv(f, usecols=["station_id", "granule_ur"])
        done |= set(zip(prev["station_id"], prev["granule_ur"]))
    log.info("checkpoint        : %d reads already on disk across %d file(s)",
             len(done), len(shard_glob))

    # Shard on a STABLE HASH of the task key, not on position in the todo list.  Position
    # sharding silently loses work: each task reads the checkpoint at ITS OWN start time,
    # so a shard launched a minute later sees a shorter todo list, and todo[i::n] then
    # indexes a DIFFERENT partition -- granules fall between the slices.  That is what
    # left 7,785 of 119,566 reads unattempted on array 26800268.  A hash of the key gives
    # every task the same verdict for the same granule no matter when it started.
    if args.nshards > 1:
        todo = [t for k, t in tasks.items()
                if k not in done
                and int(hashlib.md5(f"{k[0]}|{k[1]}".encode()).hexdigest(), 16)
                % args.nshards == args.shard]
        log.info("shard             : %d of %d (stable hash) -> %d reads",
                 args.shard, args.nshards, len(todo))
    else:
        todo = [t for k, t in tasks.items() if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    log.info("granule reads     : %d total, %d todo, workers=%d",
             len(tasks), len(todo), args.workers)

    if args.report_only:
        todo = []
        log.info("report-only: merging %d shard file(s), no reads", len(shard_glob))

    t0 = time.time()
    buf: list[dict] = []
    n_ok = n_err = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def work(t):
                res = read_station_window(t["granule_ur"], t["lon"], t["lat"])
                return {**t, **{k: v for k, v in res.items() if k in READ_COLS}}

            for i, row in enumerate(pool.map(work, todo), 1):
                buf.append(row)
                n_ok += int(row.get("read_ok") or 0)
                n_err += int(not row.get("read_ok"))
                if len(buf) >= 500:
                    append_rows(reads_csv, buf)
                    buf = []
                if i % 2000 == 0 or i == len(todo):
                    el = (time.time() - t0) / 60.0
                    rate = i / max(el * 60.0, 1e-9)
                    eta = (len(todo) - i) / max(rate, 1e-9) / 60.0
                    log.info("[%d/%d] ok=%d err=%d  %.1f/s  %.1f min elapsed, ETA %.0f min",
                             i, len(todo), n_ok, n_err, rate, el, eta)
    append_rows(reads_csv, buf)
    log.info("read pass done in %.1f min", (time.time() - t0) / 60.0)

    # ---------------- join back and report ----------------
    # Read back EVERY shard, not just this task's, so a single-shard run still reports on
    # the whole measured set.  A shard reporting on itself alone would look like massive
    # attrition when the other seven simply live in other files.
    frames = [pd.read_csv(f) for f in
              sorted(ROOT.glob(f"csvs/ecostress_wp_reads.{args.out_tag}*.csv"))]
    if not frames:
        log.warning("no reads on disk -- nothing to report")
        return
    reads = pd.concat(frames, ignore_index=True)
    reads = reads.drop_duplicates(subset=["station_id", "granule_ur"], keep="last")
    log.info("=" * 70)
    log.info("GRANULE-LEVEL  (n=%d)", len(reads))
    log.info("  read_ok            : %d  (%.1f%%)", int(reads["read_ok"].sum()),
             100.0 * reads["read_ok"].mean())
    log.info("  failed             : %d", int((reads["read_ok"] == 0).sum()))
    if (reads["read_ok"] == 0).any():
        top = (reads.loc[reads["read_ok"] == 0, "error"].fillna("")
               .str.slice(0, 60).value_counts().head(5))
        for msg, n in top.items():
            log.info("      %5d  %s", n, msg)
    ok = reads[reads["read_ok"] == 1]
    log.info("  clipped (kept)     : %d  (window_frac >= %.2f)",
             int(ok["clipped"].sum()), WINDOW_FRAC_MIN)
    if len(ok):
        q = ok["clear_frac"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
        log.info("  clear_frac deciles : p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f",
                 q[0.1], q[0.25], q[0.5], q[0.75], q[0.9])
        log.info("  mean cloud frac    : %.3f   mean water frac : %.3f",
                 ok["frac_cloud"].mean(), ok["frac_water"].mean())

    cf = reads.set_index(["station_id", "granule_ur"])["clear_frac"].to_dict()
    okset = set(reads.loc[reads["read_ok"] == 1].set_index(
        ["station_id", "granule_ur"]).index)

    pairs["day_clear"] = [cf.get((s, u), np.nan) for s, u in
                          zip(pairs["station_id"], pairs["day_ur"])]
    pairs["night_clear"] = [cf.get((s, u), np.nan) for s, u in
                            zip(pairs["station_id"], pairs["night_ur"])]
    both_read = pd.Series(
        [( (s, d) in okset ) and ( (s, n) in okset ) for s, d, n in
         zip(pairs["station_id"], pairs["day_ur"], pairs["night_ur"])],
        index=pairs.index)
    pairs["both_read"] = both_read.astype(int)
    pairs["quality"] = ((pairs["both_read"] == 1)
                        & (pairs["day_clear"] >= CLEAR_FRAC_MIN)
                        & (pairs["night_clear"] >= CLEAR_FRAC_MIN)).astype(int)

    log.info("=" * 70)
    log.info("PAIR-LEVEL")
    log.info("  pairs selected     : %d   stations %d",
             len(pairs), pairs["station_id"].nunique())
    log.info("  both halves read   : %d  (%.1f%%)", int(pairs["both_read"].sum()),
             100.0 * pairs["both_read"].mean())
    log.info("  %-18s %10s %10s %10s %10s", "clear_frac >=", "scenes", "scene%", "pairs", "stations")
    scenes_ok = len(ok)
    for thr in SWEEP:
        s_pass = int((ok["clear_frac"] >= thr).sum()) if scenes_ok else 0
        m = ((pairs["both_read"] == 1) & (pairs["day_clear"] >= thr)
             & (pairs["night_clear"] >= thr))
        exp = (s_pass / scenes_ok) ** 2 * int(pairs["both_read"].sum()) if scenes_ok else 0
        log.info("  %-18.2f %10d %9.1f%% %10d %10d   (independent would give %d)",
                 thr, s_pass, 100.0 * s_pass / max(scenes_ok, 1),
                 int(m.sum()), pairs.loc[m, "station_id"].nunique(), int(exp))
    log.info("=" * 70)
    log.info("HEADLINE at CLEAR_FRAC_MIN=%.2f : %d usable pairs across %d stations",
             CLEAR_FRAC_MIN, int(pairs["quality"].sum()),
             pairs.loc[pairs["quality"] == 1, "station_id"].nunique())

    pairs.to_csv(out_csv, index=False)
    log.info("wrote %s  (%d rows)", out_csv, len(pairs))
    log.info("wrote %s  (%d rows)", reads_csv, len(reads))


if __name__ == "__main__":
    main()
