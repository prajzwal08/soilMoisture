"""
backfill_repair.py — §50.8 phase 0d: repair the defective EXISTING S2 scenes, in place by index
================================================================================================
Two defects found by backfill_catalogue.py (§50.7), 169 scenes:
  double_offset  pre-2022-01-25 scene whose catalogue baseline is already >= 04.00 got +1000
                 a second time from the date-based harmonisation  -> −1000 on non-zero pixels
  negative       NaN cast to int16 (−32768, or −31768 after harmonise)  -> 0 on those pixels

Rows are REPLACED at the same index (same date), never inserted, so nothing else moves.

  --list    (terramind)  -> csvs/s2_repair_targets.csv (station, date, fix)
  --raw     (terramind)  satellite_zarr: copy s2 -> s2_merged with the repaired rows, verify
                         that ONLY those rows changed, swap; export the repaired rows as TIFs to
                         s2_repair/{st}/S2L2A/ for the cloud model
  (cloud mask: cloud_masking_inference.py redirected to s2_repair — slurm/backfill_repair_gpu.sh)
  --encode  (terramind GPU) TerraMind for the repaired rows only -> s2_repair/{st}/tokens_repair.npz
  --splice  (terramind, UNDER slurm/backfill_splice_guarded.sh's unlock, --repair mode)
                         zarr_tokens: replace those rows in s2/{l3,l6,l9,l12} and cm (by date),
                         verify every other row bit-equal, swap, regenerate s2_l*.npy, consolidate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from backfill_merge import LAYERS, RAW_ROOT, TOK_ROOT, T_CM, T_TOKENS, _cats, _dates_int, _swap  # noqa: E402

STAGE_R   = Path("/gpfs/scratch1/shared/pkhanal/s2_repair")
STAGE_RCM = Path("/gpfs/scratch1/shared/pkhanal/s2_repair_cm")
TARGETS   = REPO / "csvs" / "s2_repair_targets.csv"


def build_list():
    oc = pd.read_csv(REPO / "csvs" / "s2_store_offset_check.csv")
    dbl = oc[oc.grp.str.startswith("pre_cut_pb>=04")][["station", "date"]].assign(fix="double_offset")
    st = pd.read_csv(REPO / "csvs" / "s2_store_scene_stats.csv", usecols=["station", "date", "n_neg"])
    neg = st[st.n_neg > 0][["station", "date"]].assign(fix="negative")
    t = pd.concat([dbl, neg]).groupby(["station", "date"]).fix.apply("+".join).reset_index()
    t.to_csv(TARGETS, index=False)
    print(f"-> {TARGETS}: {len(t)} scenes at {t.station.nunique()} stations  "
          f"{t.fix.value_counts().to_dict()}")


def _targets(station):
    t = pd.read_csv(TARGETS)
    t = t[t.station == station]
    return dict(zip(t.date.astype(int), t.fix))


def _fix(x: np.ndarray, fix: str) -> np.ndarray:
    y = x.astype(np.int32)
    if "negative" in fix:
        y[y < 0] = 0
    if "double_offset" in fix:
        nz = y != 0
        y[nz] -= 1000
        # A valid pixel must stay valid: 0 is nodata. Native offset data can hold values below
        # 1000 (negative BOA reflectance), so after one −1000 they floor at 1, never at 0.
        y[nz] = np.maximum(y[nz], 1)
    return y.astype(np.int16)


def repair_raw(station: str) -> dict:
    import rasterio
    import zarr
    fixes = _targets(station)
    path = RAW_ROOT / f"{station}.zarr"
    g = zarr.open_group(str(path), mode="a")
    d = _dates_int(g["s2/dates"][:])
    idx = {x: i for i, x in enumerate(d)}
    miss = [x for x in fixes if x not in idx]
    if miss:
        raise ValueError(f"{station}: repair dates not in store: {miss}")
    a = g["s2/data"]
    if "s2_merged" in g:
        del g["s2_merged"]
    mg = g.create_group("s2_merged")
    mg.attrs.update(dict(g["s2"].attrs))
    m = mg.create_dataset("data", shape=a.shape, chunks=a.chunks, dtype=a.dtype,
                          compressor=a.compressor, fill_value=a.fill_value)
    out = STAGE_R / station / "S2L2A"
    out.mkdir(parents=True, exist_ok=True)
    (STAGE_RCM / _cats()[station] / station).mkdir(parents=True, exist_ok=True)   # cloud-mask target
    rows = {idx[x]: f for x, f in fixes.items()}
    for i in range(a.shape[0]):
        x = a[i]
        if i in rows:
            x = _fix(x, rows[i])
            with rasterio.open(out / f"{d[i]}.tif", "w", driver="GTiff", width=x.shape[2],
                               height=x.shape[1], count=x.shape[0], dtype="int16") as f:
                f.write(x)
        m[i] = x
    mg.array("dates", g["s2/dates"][:], chunks=g["s2/dates"].chunks,
             compressor=g["s2/dates"].compressor)
    changed = [i for i in range(a.shape[0]) if not np.array_equal(m[i], a[i])]
    if set(changed) - set(rows):
        raise AssertionError(f"{station}: rows changed outside the repair list: "
                             f"{sorted(set(changed) - set(rows))[:5]}")
    if any(np.asarray(m[i]).min() < 0 for i in rows):
        raise AssertionError(f"{station}: negative value survived the repair")
    _swap(path, "s2")
    return dict(station=station, repaired=len(rows), changed=len(changed))


def encode_repaired(station: str, device: str = "cuda") -> dict:
    import torch
    import zarr
    from retokenize_satellite_zarr import _nn_fill_and_sanitize
    from backfill_merge import _encoder
    fixes = _targets(station)
    g = zarr.open_group(str(RAW_ROOT / f"{station}.zarr"), mode="r")
    d = _dates_int(g["s2/dates"][:])
    idx = [i for i, x in enumerate(d) if x in fixes]
    enc = _encoder(device)
    chunk = np.stack([g["s2/data"][i] for i in idx]).astype(np.float32)
    t = torch.stack([torch.from_numpy(_nn_fill_and_sanitize(c, "S2L2A")) for c in chunk]).to(device)
    with torch.no_grad():
        f = enc(t, "S2L2A")
    np.savez(STAGE_R / station / "tokens_repair.npz",
             dates=np.array([d[i] for i in idx], dtype=np.int64),
             **{lay: f[lay.upper()].half().cpu().numpy() for lay in LAYERS})
    return dict(station=station, encoded=len(idx))


def splice_repair(station: str, cat: str) -> dict:
    import rasterio
    import zarr
    sd = TOK_ROOT / cat / station
    for p in (sd, sd / "s2", sd / "cm"):
        if not os.access(p, os.W_OK):
            raise PermissionError(f"{p} is not writable — run under backfill_splice_guarded.sh")
    z = np.load(STAGE_R / station / "tokens_repair.npz")
    rd = [int(x) for x in z["dates"]]
    g = zarr.open_group(str(sd), mode="a")
    td = _dates_int(g["s2/dates"][:])
    ti = {x: i for i, x in enumerate(td)}
    if any(x not in ti for x in rd):
        raise ValueError(f"{station}: repaired dates missing from the token store")
    for grp in ("s2_merged", "cm_merged"):
        if grp in g:
            del g[grp]
    mg = g.create_group("s2_merged")
    mg.attrs.update(dict(g["s2"].attrs))
    for lay in LAYERS:
        o = g[f"s2/{lay}"][:]
        n = o.copy()
        for k, x in enumerate(rd):
            n[ti[x]] = z[lay][k]
        keep = [i for i in range(len(td)) if td[i] not in set(rd)]
        assert all(np.array_equal(n[i], o[i]) for i in keep), f"{station}: {lay} other rows moved"
        mg.array(lay, n, chunks=(min(T_TOKENS, len(td)),) + o.shape[1:],
                 compressor=g[f"s2/{lay}"].compressor)
    mg.array("dates", g["s2/dates"][:], chunks=(len(td),))
    cd = _dates_int(g["cm/dates"][:])
    ci = {x: i for i, x in enumerate(cd)}
    cm = g["cm/masks"][:]
    cmn = cm.copy()
    for x in rd:
        with rasterio.open(STAGE_RCM / cat / station / "CloudMask" / f"{x}.tif") as f:
            cmn[ci[x]] = f.read(1)[: cm.shape[1], : cm.shape[2]]
    cg = g.create_group("cm_merged")
    cg.attrs.update(dict(g["cm"].attrs))
    cg.array("masks", cmn, chunks=(min(T_CM, len(cd)),) + cm.shape[1:],
             compressor=g["cm/masks"].compressor)
    cg.array("dates", g["cm/dates"][:], chunks=(len(cd),))
    _swap(sd, "s2")
    _swap(sd, "cm")
    for lay in ("l3", "l6", "l9"):
        npy, js = sd / f"s2_{lay}.npy", sd / f"s2_{lay}.json"
        for f in (npy, js):
            if f.exists():
                os.rename(f, f.with_name(f.name + ".prerepair"))
        a = g[f"s2/{lay}"]
        mm = np.memmap(str(npy), dtype=np.float16, mode="w+", shape=a.shape)
        mm[:] = a[:]
        mm.flush()
        del mm
        js.write_text(json.dumps({"shape": list(a.shape), "dtype": "float16"}))
    zarr.consolidate_metadata(str(sd))
    return dict(station=station, rows_replaced=len(rd), cm_replaced=len(rd))


def _run_one(args):
    mode, s, cat = args
    try:
        return s, True, (repair_raw(s) if mode == "raw" else splice_repair(s, cat))
    except Exception as e:                                          # noqa: BLE001
        return s, False, f"{type(e).__name__}: {str(e)[:300]}"


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for f in ("list", "raw", "encode", "splice"):
        m.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--stations-file", default=None)
    ap.add_argument("--ok-out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.list:
        return build_list()
    cats = _cats()
    t = pd.read_csv(TARGETS)
    st = list(a.stations or []) + (Path(a.stations_file).read_text().split() if a.stations_file else [])
    todo = [s for s in (st or sorted(t.station.unique())) if s in set(t.station)]
    ok, bad = [], []
    if a.encode:
        for s in todo:
            try:
                print(f"  {encode_repaired(s)}", flush=True)
                ok.append(s)
            except Exception as e:                                  # noqa: BLE001
                bad.append((s, f"{type(e).__name__}: {str(e)[:300]}"))
    else:
        from multiprocessing import Pool
        mode = "raw" if a.raw else "splice"
        with Pool(a.workers) as p:
            for s, good, r in p.imap_unordered(_run_one, [(mode, s, cats[s]) for s in todo]):
                print(f"  {'OK ' if good else '!! '}{s}: {r}", flush=True)
                (ok.append(s) if good else bad.append((s, r)))
    if a.ok_out:
        Path(a.ok_out).write_text("\n".join(sorted(ok)) + "\n")
    print(f"ok {len(ok)}, FAILED {len(bad)}")
    for s, e in bad:
        print(f"  FAILED {s}: {e}")
    sys.exit(1 if not ok else 0)


def verify_repair(station: str, cat: str, backup: Path) -> list[str]:
    """§50.5-style checks for a repaired station against the 0b backup. Returns failures."""
    import zarr
    fails = []
    fixes = _targets(station)
    g = zarr.open_group(str(RAW_ROOT / f"{station}.zarr"), mode="r")
    b = zarr.open_group(str(backup / "raw" / f"{station}.zarr"), mode="r")
    d, bd = _dates_int(g["s2/dates"][:]), _dates_int(b["s2/dates"][:])
    if d != bd:
        fails.append("raw dates changed")
    p1 = []
    for i, x in enumerate(d):
        new, old = g["s2/data"][i], b["s2/data"][i]
        if x in fixes:
            if new.min() < 0:
                fails.append(f"raw {x}: negative remains")
            if not np.array_equal(new, _fix(old, fixes[x])):
                fails.append(f"raw {x}: not equal to the repair of the backup row")
            if "double_offset" in fixes[x]:
                # The decisive test (check_double_offset.py): the repaired row must equal the
                # freshly downloaded source item wherever both are non-zero. Pixel statistics
                # misled in BOTH directions (a p1 of 1001 was a double offset of DN 1).
                ck = pd.read_csv(REPO / "csvs" / "s2_double_offset_check.csv")
                it = ck[(ck.station == station) & (ck.date == x)].item_id.iloc[0]
                fr = np.load(Path("/gpfs/scratch1/shared/pkhanal/s2_repair_check") / station
                             / f"{x}__{it}.npy").astype(np.int32)
                mk = (new != 0) & (fr != 0)
                p1.append(float(np.median((new.astype(np.int32) - fr)[mk])))
        elif not np.array_equal(new, old):
            fails.append(f"raw {x}: untouched row changed")
    t = zarr.open_group(str(TOK_ROOT / cat / station), mode="r")
    bt = zarr.open_group(str(backup / "tokens" / cat / station), mode="r")
    td = _dates_int(t["s2/dates"][:])
    for lay in LAYERS:
        n, o = t[f"s2/{lay}"][:], bt[f"s2/{lay}"][:]
        for i, x in enumerate(td):
            same = np.array_equal(n[i], o[i])
            if x in fixes and same:
                fails.append(f"token {lay} {x}: repaired row NOT re-encoded")
            if x not in fixes and not same:
                fails.append(f"token {lay} {x}: untouched row changed")
    import rasterio
    cd = _dates_int(t["cm/dates"][:])
    cmz = t["cm/masks"]
    for x in fixes:
        p = STAGE_RCM / cat / station / "CloudMask" / f"{x}.tif"
        if x not in cd or not p.exists():
            fails.append(f"cm {x}: no recomputed mask / no cm row")
            continue
        with rasterio.open(p) as f:
            m = f.read(1)[: cmz.shape[1], : cmz.shape[2]]
        if not np.array_equal(cmz[cd.index(x)], m):
            fails.append(f"cm {x}: row is not the recomputed mask")
    for lay in ("l3", "l6", "l9"):
        meta = json.loads((TOK_ROOT / cat / station / f"s2_{lay}.json").read_text())
        mm = np.memmap(TOK_ROOT / cat / station / f"s2_{lay}.npy", dtype=np.float16, mode="r",
                       shape=tuple(meta["shape"]))
        if not np.array_equal(np.asarray(mm), t[f"s2/{lay}"][:]):
            fails.append(f"s2_{lay}.npy != zarr")
    if p1 and any(v != 0 for v in p1):
        fails.append(f"repaired rows differ from the source item: median deltas {p1}")
    print(f"  [{'PASS' if not fails else 'FAIL'}] {station}: {len(fixes)} repaired"
          f"{f', p1 {np.median(p1):.0f}' if p1 else ''}  {fails[:3]}", flush=True)
    return fails


def main_verify():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, required=True)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--stations-file", default=None)
    ap.add_argument("--ok-out", default=None)
    a, _ = ap.parse_known_args(sys.argv[2:])
    st = list(a.stations or []) + (Path(a.stations_file).read_text().split() if a.stations_file else [])
    cats = _cats()
    per = {s: verify_repair(s, cats[s], a.backup) for s in st}
    bad = [f for v in per.values() for f in v]
    if a.ok_out:
        Path(a.ok_out).write_text("\n".join(sorted(s for s, v in per.items() if not v)) + "\n")
    print("ALL PASS" if not bad else f"FAILED: {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        main_verify()
    else:
        main()
