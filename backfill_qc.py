"""
backfill_qc.py — §50 post-backfill quality control (read-only; writes CSVs + scratch .npy only)
================================================================================================
backfill_verify.py proved the stores are internally consistent. It did NOT prove the new pixels
are right, and its harmonise check (median 1st-percentile dark DN) is a pixel statistic — the
same kind that misled us both ways in §50.8. Four checks close that:

  A  offset    each sampled new scene: stored raw row vs a FRESH re-download of the same item_id
               (no offset). stored - fresh must be exactly +1000 (baseline < 04.00) or 0, on
               pixels non-zero in both. Sample = ALL new pre-2022 scenes of the verify-flagged
               stations (Nivolet, Demokeya) + up to 2 offset / 1 no-offset scenes per merged
               station. Also logs NDSI snow fraction, so a bright-p1 flag can be told from snow.
  B  repair    stations in s2_repair_targets.csv vs the 20260928 backup: every old raw row must be
               bit-equal, or equal _fix(backup row) if it is a repair target; token rows may differ
               ONLY at repair-target dates. Explains the 8 "old rows bit-equal" verify FAILs.
  C  coverage  audit_s2_coverage.py re-run on the post-backfill store (backfill cloud rejections
               count as obtained) vs the pre-backfill csvs/s2_coverage_audit.csv.
  D  drift     per merged station, per-scene patch-mean tokens (l3, l12): each new scene's distance
               to the centroid of OLD scenes of the same calendar month, vs the old scenes' own
               p95 distance. ~5% of new scenes should exceed it; a systematic input error would
               push a station far above.

  --fetch     (soilmoisture)  build the A sample, re-download it -> FRESH/{station}/{date}.npy
  --check     (terramind)     A, B, D -> csvs/backfill_qc_{offset,repair,drift}.csv
  --coverage  (either)        C, after audit_s2_coverage.py has written the post CSV
Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from backfill_merge import LEDGER_DIR, RAW_ROOT, TOK_ROOT, _cats, _dates_int, kept_dates  # noqa: E402

FRESH   = Path("/gpfs/scratch1/shared/pkhanal/s2_backfill_qc")
SAMPLE  = FRESH / "sample.csv"
BACKUP  = Path("/gpfs/work3/0/prjs1968/backfill_backup/20260928")
TARGETS = REPO / "csvs" / "s2_repair_targets.csv"
COV_PRE  = REPO / "csvs" / "s2_coverage_audit.csv"
COV_POST = REPO / "csvs" / "s2_coverage_audit_post_backfill.csv"
FLAGGED = ["ISMN_LABFLUX_Nivolet", "ISMN_SD_DEM_Demokeya"]      # verify "harmonised once" FAILs
NO_PRE2023 = ["Price", "MedBow", "Coldfoot", "SuuRanch", "SwedePeak", "ParleysUpper",
              "Bussolenobosco", "ReynoldsHomestead"]            # §50: 8 train stations
CUT = 20220125                                                  # baseline 04.00 went live


def _merged(only=None):
    """station -> sorted new (kept) dates, for every ledger station that gained rows."""
    out = {}
    for p in sorted(LEDGER_DIR.glob("*.csv")):
        if only and p.stem not in only:
            continue
        d = kept_dates(p.stem)
        if d:
            out[p.stem] = d
    return out


# ── A: fresh re-download ─────────────────────────────────────────────────────

def build_sample(only):
    rng = np.random.default_rng(0)
    rows = []
    for st, new in _merged(only).items():
        led = pd.read_csv(LEDGER_DIR / f"{st}.csv", dtype={"baseline": str})
        led = led[led.date.astype(int).isin(new) & (led.status == "ok")]
        if st in FLAGGED:
            pick = led[(led.date < CUT) | led.needs_offset.astype(bool)]
        else:
            on, off = led[led.needs_offset.astype(bool)], led[~led.needs_offset.astype(bool)]
            pick = pd.concat([on.iloc[rng.permutation(len(on))[:2]],
                              off.iloc[rng.permutation(len(off))[:1]]])
        rows.append(pick[["station", "date", "item_id", "baseline", "needs_offset"]])
    s = pd.concat(rows, ignore_index=True)
    FRESH.mkdir(parents=True, exist_ok=True)
    s.to_csv(SAMPLE, index=False)
    print(f"sample: {len(s)} scenes at {s.station.nunique()} stations "
          f"(offset {int(s.needs_offset.sum())}, no-offset {int((~s.needs_offset).sum())})", flush=True)
    return s


def _fetch_station(args):
    st, g, lat, lon = args
    from backfill_s2_download import _fetch_one
    from download_s2_mpc import center_crop, station_grid
    epsg, bounds, _ = station_grid(lat, lon)
    (FRESH / st).mkdir(parents=True, exist_ok=True)
    n_ok, errs = 0, []
    for r in g.itertuples():
        p = FRESH / st / f"{int(r.date)}.npy"
        if p.exists():
            n_ok += 1
            continue
        for a in range(3):
            try:
                da, _ = _fetch_one(r.item_id, bounds, epsg)
                np.save(p, center_crop(da).fillna(0).clip(-32768, 32767).values.astype(np.int16))
                n_ok += 1
                break
            except Exception as e:                                        # noqa: BLE001
                if a == 2:
                    errs.append(f"{int(r.date)}: {type(e).__name__} {str(e)[:100]}")
    return st, n_ok, errs


def fetch(only, workers):
    s = build_sample(only)
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    from splits_config import station_dir_name
    ll = {station_dir_name(r): (float(r.latitude), float(r.longitude)) for _, r in df.iterrows()}
    tasks = [(st, g, *ll[st]) for st, g in s.groupby("station")]
    bad = 0
    with Pool(workers) as p:
        for st, n, errs in p.imap_unordered(_fetch_station, tasks):
            bad += len(errs)
            for e in errs:
                print(f"  !! {st} {e}", flush=True)
    print(f"fetch: {len(s) - bad}/{len(s)} fresh scenes, {bad} failed", flush=True)


def _offset_station(args):
    st, g = args
    import zarr
    z = zarr.open_group(str(RAW_ROOT / f"{st}.zarr"), mode="r")
    pos = {d: i for i, d in enumerate(_dates_int(z["s2/dates"][:]))}
    rows = []
    for r in g.itertuples():
        d, exp = int(r.date), (1000 if bool(r.needs_offset) else 0)
        rec = dict(station=st, date=d, baseline=r.baseline, expected=exp)
        p = FRESH / st / f"{d}.npy"
        if not p.exists():
            rows.append({**rec, "verdict": "no_fresh"})
            continue
        stored = z["s2/data"][pos[d]].astype(np.int32)
        fresh = np.load(p).astype(np.int32)
        m = (stored != 0) & (fresh != 0)
        diff = (stored - fresh)[m]
        refl = np.where(stored > 0, stored - 1000, 0).astype(np.float32)   # store = +1000 convention
        g3, s11 = refl[2], refl[10]
        ndsi = (g3 - s11) / np.maximum(g3 + s11, 1)
        vis = stored[1:4][stored[1:4] > 0]
        exact = float((diff == exp).mean()) if m.sum() else np.nan
        rec.update(n_px=int(m.sum()), median_diff=float(np.median(diff)) if m.sum() else np.nan,
                   frac_exact=exact, p1_dark=float(np.percentile(vis, 1)) if vis.size else np.nan,
                   snow_frac=round(float(((ndsi > 0.4) & (g3 > 1000)).mean()), 3),
                   verdict=("pass" if m.sum() > 1000 and exact >= 0.999 else
                            "double_offset" if m.sum() > 1000 and abs(np.median(diff) - exp - 1000) <= 1 else
                            "missing_offset" if m.sum() > 1000 and abs(np.median(diff) - exp + 1000) <= 1 else
                            "mismatch"))
        rows.append(rec)
    return rows


def check_offset(workers):
    s = pd.read_csv(SAMPLE, dtype={"baseline": str})
    with Pool(workers) as p:
        res = pd.DataFrame([r for rows in p.map(_offset_station, list(s.groupby("station"))) for r in rows])
    res.to_csv(REPO / "csvs" / "backfill_qc_offset.csv", index=False)
    print("\n== A  offset: stored vs fresh re-download ==")
    print(res.groupby(["expected", "verdict"]).size().to_string())
    for st in FLAGGED:
        f = res[res.station == st]
        if len(f):
            print(f"  {st}: {f.verdict.value_counts().to_dict()}  median p1 {f.p1_dark.median():.0f}  "
                  f"median snow_frac {f.snow_frac.median():.2f}  (p1>=1800 & snow>0.2: "
                  f"{int(((f.p1_dark >= 1800) & (f.snow_frac > 0.2)).sum())}/{int((f.p1_dark >= 1800).sum())})")
    bad = res[res.verdict != "pass"]
    if len(bad):
        print(bad.head(30).to_string(index=False))
    ok = len(bad) == 0
    print(f"A {'PASS' if ok else 'FAIL'}: {len(res) - len(bad)}/{len(res)} scenes exact")
    return ok


# ── B: repaired rows are the only changed rows ───────────────────────────────

def _repair_station(args):
    st, cat, fixes = args
    import zarr
    from backfill_repair import _fix
    rec = dict(station=st, n_targets=len(fixes))
    bdir = BACKUP / "raw" / f"{st}.zarr"
    if not bdir.exists():
        return {**rec, "verdict": "no_backup"}
    b, z = zarr.open_group(str(bdir), mode="r"), zarr.open_group(str(RAW_ROOT / f"{st}.zarr"), mode="r")
    pos = {d: i for i, d in enumerate(_dates_int(z["s2/dates"][:]))}
    raw_bad = []
    for i, d in enumerate(_dates_int(b["s2/dates"][:])):
        old = b["s2/data"][i]
        want = _fix(old, fixes[d]) if d in fixes else old
        if d not in pos or not np.array_equal(z["s2/data"][pos[d]], want):
            raw_bad.append(d)
    bt, t = zarr.open_group(str(BACKUP / "tokens" / cat / st), mode="r"), zarr.open_group(str(TOK_ROOT / cat / st), mode="r")
    tpos = {d: i for i, d in enumerate(_dates_int(t["s2/dates"][:]))}
    changed = set()
    for l in ("l3", "l6", "l9", "l12"):
        o, n = bt[f"s2/{l}"][:], t[f"s2/{l}"][:]
        changed |= {d for i, d in enumerate(_dates_int(bt["s2/dates"][:]))
                    if d not in tpos or not np.array_equal(o[i], n[tpos[d]])}
    tok_bad = sorted(changed - set(fixes))
    return {**rec, "raw_mismatch": len(raw_bad), "tok_changed": len(changed),
            "tok_changed_nontarget": len(tok_bad), "examples": str((raw_bad + tok_bad)[:5]),
            "verdict": "pass" if not raw_bad and not tok_bad else "fail"}


def check_repair(workers, only):
    t = pd.read_csv(TARGETS)
    cats = _cats()
    tasks = [(st, cats[st], dict(zip(g.date.astype(int), g.fix))) for st, g in t.groupby("station")
             if not only or st in only]
    with Pool(workers) as p:
        res = pd.DataFrame(p.map(_repair_station, tasks))
    res.to_csv(REPO / "csvs" / "backfill_qc_repair.csv", index=False)
    print("\n== B  repair: changed rows == repair targets ==")
    print(res.verdict.value_counts().to_string())
    bad = res[res.verdict != "pass"]
    if len(bad):
        print(bad.to_string(index=False))
    ok = len(bad) == 0
    print(f"B {'PASS' if ok else 'FAIL'}: {len(res) - len(bad)}/{len(res)} repair stations")
    return ok


# ── D: token drift, new vs old, month-matched ────────────────────────────────

def _drift_station(args):
    st, cat, new = args
    import zarr
    t = zarr.open_group(str(TOK_ROOT / cat / st), mode="r")
    d = np.array(_dates_int(t["s2/dates"][:]))
    is_new = np.isin(d, new)
    mon = (d // 100) % 100
    rec = dict(station=st, n_old=int((~is_new).sum()), n_new=int(is_new.sum()))
    for l in ("l3", "l12"):
        v = t[f"s2/{l}"][:].astype(np.float32).mean(axis=1)            # [N, 768] patch mean
        n_cmp = n_out = 0
        ratios = []
        for m in range(1, 13):
            o, nw = v[(~is_new) & (mon == m)], v[is_new & (mon == m)]
            if len(o) < 10 or not len(nw):
                continue
            c = o.mean(axis=0)
            do, dn = np.linalg.norm(o - c, axis=1), np.linalg.norm(nw - c, axis=1)
            n_cmp += len(nw)
            n_out += int((dn > np.percentile(do, 95)).sum())
            ratios.append(np.median(dn) / max(np.median(do), 1e-6))
        rec[f"{l}_n_cmp"] = n_cmp
        rec[f"{l}_out_rate"] = round(n_out / n_cmp, 3) if n_cmp else np.nan
        rec[f"{l}_dist_ratio"] = round(float(np.median(ratios)), 3) if ratios else np.nan
    return rec


def check_drift(workers, only):
    cats = _cats()
    tasks = [(st, cats[st], new) for st, new in _merged(only).items()]
    with Pool(workers) as p:
        res = pd.DataFrame(p.map(_drift_station, tasks))
    res.to_csv(REPO / "csvs" / "backfill_qc_drift.csv", index=False)
    print("\n== D  token drift: new scenes vs same-month old scenes (expect ~0.05 outlier rate) ==")
    for l in ("l3", "l12"):
        c = res[res[f"{l}_n_cmp"] > 0]
        pooled = (c[f"{l}_out_rate"] * c[f"{l}_n_cmp"]).sum() / max(c[f"{l}_n_cmp"].sum(), 1)
        print(f"  {l}: {len(c)} stations, {int(c[f'{l}_n_cmp'].sum())} new scenes compared, pooled "
              f"outlier rate {pooled:.3f}, median dist ratio {c[f'{l}_dist_ratio'].median():.2f}")
    flag = res[((res.l3_out_rate > 0.25) | (res.l12_out_rate > 0.25)) & (res.l12_n_cmp >= 5)]
    print(f"  stations with outlier rate > 0.25 (>=5 compared): {len(flag)}")
    if len(flag):
        print(flag.to_string(index=False))
    ok = len(flag) == 0
    print(f"D {'PASS' if ok else 'FAIL (inspect: year/land-cover change also moves tokens)'}")
    return ok


# ── C: coverage before vs after ──────────────────────────────────────────────

def check_coverage():
    pre, post = pd.read_csv(COV_PRE), pd.read_csv(COV_POST)
    print("\n== C  coverage: obtained / catalogue, station-years in window ==")
    w = {}
    for k, r in (("pre", pre), ("post", post)):
        w[k] = r[r.in_window & (r.catalogue > 0)]
        print(f"  {k:4s}: catalogue {int(w[k].catalogue.sum()):,}  obtained {int(w[k].obtained.sum()):,}"
              f"  lost {1 - w[k].obtained.sum() / w[k].catalogue.sum():.2%}  |  "
              + "  ".join(f"<{t}: {int((w[k].ratio < t).sum())} st-yr / {w[k][w[k].ratio < t].station.nunique()} stn"
                          for t in (0.1, 0.5, 0.8)))
    still = w["post"][w["post"].ratio < 0.5].sort_values("ratio")
    if len(still):
        print(f"  post ratio < 0.5 remaining (first 30 of {len(still)}):")
        print(still[["station", "split", "year", "catalogue", "store", "cloud_deleted", "obtained", "ratio"]]
              .head(30).round(3).to_string(index=False))
    pr = post[post.year < 2023]
    print("  the 8 §50 train stations, store scenes before 2023 (pre -> post):")
    ok8 = True
    for k in NO_PRE2023:
        a = pre[pre.station.str.endswith("_" + k) & (pre.year < 2023)].store.sum()
        b = pr[pr.station.str.endswith("_" + k)].store.sum()
        ok8 &= b > 0
        print(f"    {k:18s} {int(a):4d} -> {int(b):4d}")
    ok = ok8 and (w["post"].ratio < 0.1).sum() < (w["pre"].ratio < 0.1).sum()
    print(f"C {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--coverage", action="store_true")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.stations, a.workers)
        sys.exit(0)
    if a.coverage:
        sys.exit(0 if check_coverage() else 1)
    results = {}
    for k, f in (("A", lambda: check_offset(a.workers)), ("B", lambda: check_repair(a.workers, a.stations)),
                 ("D", lambda: check_drift(a.workers, a.stations))):
        try:
            results[k] = f()
        except Exception as e:                                            # noqa: BLE001
            print(f"{k} CRASHED: {type(e).__name__}: {e}", flush=True)
            results[k] = False
    print(f"\nbackfill_qc --check: {results}")
    sys.exit(0 if all(results.values()) else 1)
