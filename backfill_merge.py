"""
backfill_merge.py — §50 phases 4-5: merge the kept backfill scenes into the permanent stores
=============================================================================================
Nothing here re-derives an existing row. Every write goes to a SIBLING group, is verified
against the old one, and only then swapped in by directory rename; the old group stays beside
it as `*_prebackfill` until §50.5 passes (backfill_verify.py), then is removed by hand.

  --raw        (terramind, CPU)  satellite_zarr/{st}.zarr:  s2 ∪ kept TIFs -> s2_merged
                                 -> verify -> s2 -> s2_prebackfill, s2_merged -> s2
  --encode     (terramind, GPU)  TerraMind L3/L6/L9/L12 for the NEW dates only, read from the
                                 merged raw store with retokenize's own _nn_fill_and_sanitize
                                 and encoder -> s2_backfill/{st}/tokens_new.npz. No store write.
  --splice     (terramind, CPU, UNDER slurm/backfill_splice_guarded.sh's unlock)
                                 zarr_tokens/{cat}/{st}: s2 ∪ tokens_new -> s2_merged,
                                 cm ∪ Phase-3 masks -> cm_merged, verify, swap, regenerate
                                 s2_l{3,6,9}.{npy,json}, consolidate_metadata.

"Kept" = ledger status ok AND backfill manifest verdict keep AND the TIF is in S2L2A/.
The cm rows for new dates are the Phase 3 SEnSeIv2 masks: the same model and the same
TIF-mask convention the original token-store cm came from (create_token_zarr.write_cloud_mask).
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
from splits_config import ALL_CATEGORIES, category_of, station_dir_name  # noqa: E402

RAW_ROOT   = Path("/projects/prjs1968/satellite_zarr")
TOK_ROOT   = Path("/projects/prjs1968/zarr_tokens")
STAGE      = Path("/gpfs/scratch1/shared/pkhanal/s2_backfill")
STAGE_CM   = Path("/gpfs/scratch1/shared/pkhanal/s2_backfill_cm")
LEDGER_DIR = REPO / "csvs" / "s2_backfill_ledger"
MANIFEST   = REPO / "text" / "s2_backfill_manifest.csv"
LAYERS     = ["l12", "l9", "l6", "l3"]
T_TOKENS, T_CM = 32, 128


def _cats():
    df = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    df["cat"] = df.apply(category_of, axis=1)
    return {station_dir_name(r): r["cat"] for _, r in df[df["cat"].isin(ALL_CATEGORIES)].iterrows()}


def kept_dates(station: str) -> list[int]:
    led = pd.read_csv(LEDGER_DIR / f"{station}.csv")
    per = REPO / "text" / "s2_backfill_manifest" / f"{station}.csv"
    man = pd.read_csv(per if per.exists() else MANIFEST)
    keep = set(man[(man.station == station) & (man.verdict == "keep")].date.astype(int))
    ok = set(led[led.status == "ok"].date.astype(int))
    d = sorted(x for x in keep & ok if (STAGE / station / "S2L2A" / f"{x}.tif").exists())
    return d


def _dates_int(arr) -> list[int]:
    return [int((bytes(x).decode() if isinstance(x, (bytes, np.bytes_)) else str(x))[:8])
            for x in arr]


def _swap(group_dir: Path, name: str):
    """name -> name_prebackfill, name_merged -> name. Refuses to clobber an earlier backup."""
    old, new, bak = group_dir / name, group_dir / f"{name}_merged", group_dir / f"{name}_prebackfill"
    if bak.exists():
        raise FileExistsError(f"{bak} already exists — an earlier backfill was not cleaned up")
    os.rename(old, bak)
    os.rename(new, old)


# ── phase 4: raw store ───────────────────────────────────────────────────────

def merge_raw(station: str) -> dict:
    import rasterio
    import zarr
    new = kept_dates(station)
    rep = dict(station=station, n_new=len(new), status="skip_nothing_kept")
    if not new:
        return rep
    path = RAW_ROOT / f"{station}.zarr"
    g = zarr.open_group(str(path), mode="a")
    old_d = _dates_int(g["s2/dates"][:])
    if set(old_d) & set(new):
        raise ValueError(f"{station}: backfill dates already in the raw store: "
                         f"{sorted(set(old_d) & set(new))[:5]}")
    old_a = g["s2/data"]
    allrows = sorted([(d, "old", i) for i, d in enumerate(old_d)] +
                     [(d, "new", d) for d in new])
    N = len(allrows)
    if "s2_merged" in g:
        del g["s2_merged"]
    mg = g.create_group("s2_merged")
    mg.attrs.update(dict(g["s2"].attrs))
    data = mg.create_dataset("data", shape=(N,) + old_a.shape[1:], chunks=old_a.chunks,
                             dtype=old_a.dtype, compressor=old_a.compressor,
                             fill_value=old_a.fill_value)
    for j, (d, src, key) in enumerate(allrows):
        if src == "old":
            data[j] = old_a[key]
        else:
            with rasterio.open(STAGE / station / "S2L2A" / f"{d}.tif") as f:
                x = f.read()
            if x.shape != old_a.shape[1:] or x.dtype != old_a.dtype:
                raise ValueError(f"{station} {d}: TIF {x.shape} {x.dtype} vs store "
                                 f"{old_a.shape[1:]} {old_a.dtype}")
            data[j] = x
    mg.array("dates", np.array([str(d).encode() for d, *_ in allrows], dtype="|S8"),
             chunks=(N,), compressor=g["s2/dates"].compressor)
    # ── verify before the swap ──
    md = _dates_int(mg["dates"][:])
    assert md == sorted(md) and len(set(md)) == N == data.shape[0], f"{station}: dates"
    for j, (d, src, key) in enumerate(allrows):
        if src == "old" and not np.array_equal(data[j], old_a[key]):
            raise AssertionError(f"{station}: old row {key} ({d}) changed at new index {j}")
    _swap(path, "s2")
    rep.update(status="merged", n_old=len(old_d), n_total=N)
    return rep


# ── phase 5a: encode the new dates only ──────────────────────────────────────

def encode_new(station: str, device: str = "cuda", batch: int = 8) -> dict:
    import torch
    import zarr
    from retokenize_satellite_zarr import _nn_fill_and_sanitize
    new = set(kept_dates(station))
    g = zarr.open_group(str(RAW_ROOT / f"{station}.zarr"), mode="r")
    d_all = _dates_int(g["s2/dates"][:])
    idx = [i for i, d in enumerate(d_all) if d in new]
    if len(idx) != len(new):
        raise ValueError(f"{station}: {len(new)} kept dates but {len(idx)} in the merged raw "
                         f"store — run --raw first")
    enc = _encoder(device)
    out = {lay: [] for lay in LAYERS}
    arr = g["s2/data"]
    for s in range(0, len(idx), batch):
        chunk = np.stack([arr[i] for i in idx[s:s + batch]]).astype(np.float32)
        t = torch.stack([torch.from_numpy(_nn_fill_and_sanitize(c, "S2L2A")) for c in chunk]).to(device)
        with torch.no_grad():
            f = enc(t, "S2L2A")
        for lay in LAYERS:
            out[lay].append(f[lay.upper()].half().cpu().numpy())
    np.savez(STAGE / station / "tokens_new.npz",
             dates=np.array([d_all[i] for i in idx], dtype=np.int64),
             **{lay: np.concatenate(out[lay]) for lay in LAYERS})
    return dict(station=station, encoded=len(idx))


# ── phase 5b: splice into the token store ────────────────────────────────────

def splice_tokens(station: str, cat: str) -> dict:
    import rasterio
    import zarr
    sd = TOK_ROOT / cat / station
    for p in (sd, sd / "s2", sd / "cm"):
        if not os.access(p, os.W_OK):
            raise PermissionError(f"{p} is not writable — run under backfill_splice_guarded.sh")
    z = np.load(STAGE / station / "tokens_new.npz")
    new_d = [int(x) for x in z["dates"]]
    g = zarr.open_group(str(sd), mode="a")
    old_d = _dates_int(g["s2/dates"][:])
    if set(old_d) & set(new_d):
        raise ValueError(f"{station}: new token dates already present")
    rows = sorted([(d, "old", i) for i, d in enumerate(old_d)] +
                  [(d, "new", k) for k, d in enumerate(new_d)])
    N = len(rows)
    for grp in ("s2_merged", "cm_merged"):
        if grp in g:
            del g[grp]
    mg = g.create_group("s2_merged")
    mg.attrs.update(dict(g["s2"].attrs))
    for lay in LAYERS:
        o = g[f"s2/{lay}"][:]
        m = np.empty((N,) + o.shape[1:], dtype=o.dtype)
        for j, (d, src, k) in enumerate(rows):
            m[j] = o[k] if src == "old" else z[lay][k]
        mg.array(lay, m, chunks=(min(T_TOKENS, N),) + o.shape[1:],
                 compressor=g[f"s2/{lay}"].compressor)
        assert all(np.array_equal(m[j], o[k]) for j, (d, s, k) in enumerate(rows) if s == "old")
    mg.array("dates", np.array([str(d) for d, *_ in rows], dtype="U8"), chunks=(N,))

    # cm: old ∪ Phase-3 masks for the new dates, by date
    cm_old_d = _dates_int(g["cm/dates"][:])
    cm_old = g["cm/masks"]
    add = [d for d in new_d if d not in set(cm_old_d)]
    cm_rows = sorted([(d, "old", i) for i, d in enumerate(cm_old_d)] + [(d, "new", d) for d in add])
    cg = g.create_group("cm_merged")
    cg.attrs.update(dict(g["cm"].attrs))
    M = len(cm_rows)
    cmm = np.empty((M,) + cm_old.shape[1:], dtype=cm_old.dtype)
    old_masks = cm_old[:]
    for j, (d, src, k) in enumerate(cm_rows):
        if src == "old":
            cmm[j] = old_masks[k]
        else:
            with rasterio.open(STAGE_CM / cat / station / "CloudMask" / f"{d}.tif") as f:
                cmm[j] = f.read(1)[: cm_old.shape[1], : cm_old.shape[2]]
    cg.array("masks", cmm, chunks=(min(T_CM, M),) + cm_old.shape[1:], compressor=cm_old.compressor)
    cg.array("dates", np.array([str(d) for d, *_ in cm_rows], dtype="U8"), chunks=(M,))
    assert set(d for d, *_ in rows) <= set(d for d, *_ in cm_rows), f"{station}: cm does not cover s2"

    _swap(sd, "s2")
    _swap(sd, "cm")

    # memmaps: old ones are stale by construction (row count changed) — replace them
    for lay in ("l3", "l6", "l9"):
        npy, js = sd / f"s2_{lay}.npy", sd / f"s2_{lay}.json"
        if npy.exists():
            os.rename(npy, sd / f"s2_{lay}.npy.prebackfill")
        if js.exists():
            os.rename(js, sd / f"s2_{lay}.json.prebackfill")
        a = g[f"s2/{lay}"]
        # Raw memmap with a .npy name and a JSON shape sidecar — convert_l369_to_npy.py's
        # format exactly (no numpy header), which is what verify_zarr_store checks.
        mm = np.memmap(str(npy), dtype=np.float16, mode="w+", shape=a.shape)
        mm[:] = a[:]
        mm.flush()
        del mm
        js.write_text(json.dumps({"shape": list(a.shape), "dtype": "float16"}))
    zarr.consolidate_metadata(str(sd))
    return dict(station=station, n_old=len(old_d), n_new=len(new_d), n_total=N, cm_added=len(add))


_ENCODER = None


def _encoder(device):
    global _ENCODER
    if _ENCODER is None:
        from retokenize_satellite_zarr import _load_encoder
        _ENCODER = _load_encoder(device)
    return _ENCODER


def _run_one(args):
    mode, s, cat = args
    try:
        r = merge_raw(s) if mode == "raw" else splice_tokens(s, cat)
        return s, True, r
    except Exception as e:                                          # noqa: BLE001
        return s, False, f"{type(e).__name__}: {str(e)[:300]}"


def read_stations(a) -> list[str]:
    """--stations and/or --stations-file (one per line; the previous step's OK list)."""
    st = list(a.stations or [])
    if a.stations_file:
        st += [x.strip() for x in Path(a.stations_file).read_text().split() if x.strip()]
    return list(dict.fromkeys(st))


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--raw", action="store_true")
    m.add_argument("--encode", action="store_true")
    m.add_argument("--splice", action="store_true")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--stations-file", default=None)
    ap.add_argument("--ok-out", default=None, help="write the stations that succeeded here")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cats = _cats()
    todo = [s for s in read_stations(a) if (LEDGER_DIR / f"{s}.csv").exists()]
    mode = "raw" if a.raw else "encode" if a.encode else "splice"
    ok, bad = [], []
    if mode == "encode":
        # One GPU, one model load for the whole list (the smoke reloaded TerraMind per station).
        for s in todo:
            try:
                print(f"  {encode_new(s)}", flush=True)
                ok.append(s)
            except Exception as e:                                  # noqa: BLE001
                bad.append((s, f"{type(e).__name__}: {str(e)[:300]}"))
    else:
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            for s, good, r in p.imap_unordered(_run_one, [(mode, s, cats[s]) for s in todo]):
                print(f"  {'OK ' if good else '!! '}{s}: {r}", flush=True)
                (ok.append(s) if good else bad.append((s, r)))
    if a.ok_out:
        Path(a.ok_out).write_text("\n".join(sorted(ok)) + "\n")
    print(f"{mode}: ok {len(ok)}, FAILED {len(bad)}")
    for s, e in bad:
        print(f"  FAILED {s}: {e}")
    sys.exit(1 if not ok else 0)       # downstream runs on the OK list; failures are reported


if __name__ == "__main__":
    main()
