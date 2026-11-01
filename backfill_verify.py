"""
backfill_verify.py — §50.5 checks for every backfilled station (terramind, CPU)
================================================================================
Per station, against the phase-0b backup:
  1  raw:    len(dates) == data rows; dates ascending + unique; every backup row bit-equal at
             its new index; every new row equal to its staged TIF
  2  tokens: every s2 layer rows == len(dates); ascending + unique; raw dates == token dates;
             cm ⊇ s2; s2_l{3,6,9}.npy == zarr rows (all rows); backup rows bit-equal
  3  scratch copies (token + raw) identical in dates and shapes to /projects
  4  harmonisation: 1st-percentile dark-band DN of new pre-2022-01-25 scenes (≈1000-1400 if
     harmonised once; ≈2000 double; <900 missing)
Exit non-zero on any failure. Usage: python backfill_verify.py --backup DIR --stations ...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import zarr

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from backfill_merge import RAW_ROOT, TOK_ROOT, STAGE, _cats, _dates_int, kept_dates  # noqa: E402
import rasterio  # noqa: E402
SCR_TOK = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SCR_RAW = Path("/gpfs/scratch1/shared/pkhanal/satellite_zarr")
FAILS = []


def check(ok, st, what, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {st:38s} {what}  {detail}", flush=True)
    if not ok:
        FAILS.append((st, what))


def asc_unique(d):
    return all(a < b for a, b in zip(d, d[1:]))


def one(st, cat, backup):
    new = set(kept_dates(st))
    # 1 raw
    g = zarr.open_group(str(RAW_ROOT / f"{st}.zarr"), mode="r")
    d = _dates_int(g["s2/dates"][:]); a = g["s2/data"]
    b = zarr.open_group(str(backup / "raw" / f"{st}.zarr"), mode="r")
    bd = _dates_int(b["s2/dates"][:]); ba = b["s2/data"]
    pos = {x: i for i, x in enumerate(d)}
    check(len(d) == a.shape[0] and asc_unique(d), st, "raw dates", f"{len(bd)} -> {len(d)}")
    check(set(d) == set(bd) | new, st, "raw = backup ∪ kept")
    check(all(np.array_equal(ba[i], a[pos[x]]) for i, x in enumerate(bd)), st, "raw old rows bit-equal")
    ok = True
    for x in new:
        with rasterio.open(STAGE / st / "S2L2A" / f"{x}.tif") as f:
            ok &= np.array_equal(f.read(), a[pos[x]])
    check(ok, st, "raw new rows == staged TIFs")
    # 2 tokens
    t = zarr.open_group(str(TOK_ROOT / cat / st), mode="r")
    td = _dates_int(t["s2/dates"][:])
    check(td == d, st, "token dates == raw dates")
    check(all(t[f"s2/{l}"].shape[0] == len(td) for l in ("l3", "l6", "l9", "l12")), st, "token layer rows")
    check(set(td) <= set(_dates_int(t["cm/dates"][:])), st, "cm ⊇ s2")
    bt = zarr.open_group(str(backup / "tokens" / cat / st), mode="r")
    btd = _dates_int(bt["s2/dates"][:]); tp = {x: i for i, x in enumerate(td)}
    ok = True
    for l in ("l3", "l6", "l9", "l12"):
        o, n = bt[f"s2/{l}"][:], t[f"s2/{l}"][:]
        ok &= all(np.array_equal(o[i], n[tp[x]]) for i, x in enumerate(btd))
    check(ok, st, "token old rows bit-equal")
    ok = True
    for l in ("l3", "l6", "l9"):
        sd = TOK_ROOT / cat / st
        meta = json.loads((sd / f"s2_{l}.json").read_text())
        mm = np.memmap(sd / f"s2_{l}.npy", dtype=np.float16, mode="r", shape=tuple(meta["shape"]))
        ok &= tuple(meta["shape"]) == t[f"s2/{l}"].shape and np.array_equal(np.asarray(mm), t[f"s2/{l}"][:])
    check(ok, st, "npy memmaps == zarr")
    # 3 scratch copies
    try:
        ts = zarr.open_consolidated(str(SCR_TOK / cat / st), mode="r")
        rs = zarr.open_group(str(SCR_RAW / f"{st}.zarr"), mode="r")
        check(_dates_int(ts["s2/dates"][:]) == td and ts["s2/l12"].shape == t["s2/l12"].shape
              and _dates_int(rs["s2/dates"][:]) == d, st, "scratch copies match /projects")
    except Exception as e:  # noqa: BLE001
        check(False, st, "scratch copies match /projects", repr(e)[:120])
    # 4 harmonisation
    p1 = []
    for x in sorted(new):
        if x < 20220125:
            v = a[pos[x]][1:4]
            pos_v = v[v > 0]
            if pos_v.size > 100:
                p1.append(float(np.percentile(pos_v, 1)))
    if p1:
        med = float(np.median(p1))
        check(900 <= med <= 1600, st, "harmonised once (new pre-2022 p1 dark DN)",
              f"median {med:.0f} over {len(p1)} scenes; >=1800: {sum(v >= 1800 for v in p1)}")


def _one_safe(args):
    st, cat, backup = args
    FAILS.clear()
    try:
        one(st, cat, backup)
    except Exception as e:  # noqa: BLE001
        check(False, st, "crashed", repr(e)[:200])
    return st, list(FAILS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, required=True)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--stations-file", default=None)
    ap.add_argument("--ok-out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    st = list(a.stations or []) + (Path(a.stations_file).read_text().split() if a.stations_file else [])
    cats = _cats()
    from multiprocessing import Pool
    ok, bad = [], []
    with Pool(a.workers) as p:
        for s, f in p.imap_unordered(_one_safe, [(s, cats[s], a.backup) for s in st]):
            (bad.append((s, f)) if f else ok.append(s))
    if a.ok_out:
        Path(a.ok_out).write_text("\n".join(sorted(ok)) + "\n")
    print(f"\nverify: {len(ok)} stations ALL PASS, {len(bad)} with failures")
    for s, f in bad:
        print(f"  FAILED {s}: {f[:4]}")
    sys.exit(0 if ok and not bad else 1)


if __name__ == "__main__":
    main()
