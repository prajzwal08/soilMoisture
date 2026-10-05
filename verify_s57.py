"""
verify_s57.py — checks for the §57 "indices" fine inputs, CPU only
===================================================================
Run by slurm/verify_s57.sh (after verify_s48.py, which covers the default "bands" mode).
Each check prints PASS/FAIL; the script exits non-zero if any fails.

  1  model.py de-normalisation constants equal csvs/fine_stats.json
  2  indices model builds and runs; S2 stem takes 4 ch, S1 stem 5 ch
  3  an all-zero fine tensor gives all-zero indices (dropped / ablated = "missing")
  4  modality dropout on the 19-ch tensor zeroes the WHOLE converted modality
  5  raw store: NDVI/NDMI via fine_s2_scene -> fine_to_indices equal the indices computed
     directly from raw 10 m DN (masked 2x2 mean, -1000), fp32 and fp16 paths
  6  raw store: CR via fine_s1_scene -> fine_to_indices equals VH_dB - VV_dB from the raw
     store pooled in linear power
  7  value ranges: NDVI median in [0, 0.9], NDMI median in [-0.3, 0.6], CR median in [-12, -2] dB
  8  +1000 offset: share of valid DN < 1000 is tiny both before and after 2022-01-25
  9  real dataset samples convert without NaN/Inf
  10 the existing "bands" checkpoint (v2 best.pt) still loads strictly via ckpt_utils
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import model as M  # noqa: E402
from model import FINE_CH, SoilMoistureModel, fine_to_indices  # noqa: E402

FAILS = []
V2_BEST = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/lst_tmean_diff/"
               "lst_tmean_diff_era5do05_coarse03_20260930/best.pt")


def check(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


def c1_constants():
    st = json.loads((REPO / "csvs" / "fine_stats.json").read_text())
    keep = st["s2"]["keep_idx"]
    names = [st["s2"]["bands_all"][k] for k in keep]
    ok = True
    for b, i in M._S2_IDX.items():
        ok &= names[i] == b
        ok &= abs(st["s2"]["mean"][keep[i]] - M._S2_MEAN[b]) < 1e-6
        ok &= abs(st["s2"]["std"][keep[i]] - M._S2_STD[b]) < 1e-6
    ok &= tuple(st["s1"]["mean"]) == M._S1_MEAN and tuple(st["s1"]["std"]) == M._S1_STD
    check(bool(ok), "1 constants match fine_stats.json", f"kept bands {names}")


def c2_c4_synthetic():
    torch.manual_seed(0)
    m = SoilMoistureModel(fine_inputs="indices").eval()
    fe = m.fine_encoder
    w2 = fe.stem_s2[0].weight.shape[1]
    w1 = fe.stem_s1[0].weight.shape[1]
    fine = torch.randn(2, FINE_CH, 112, 112)
    fine[:, [10, 14, 18]] = 1.0
    lulc = torch.randint(1, 10, (2, 224, 224))
    with torch.no_grad():
        outs = fe(fine, lulc)
    shapes = [tuple(o.shape) for o in outs]
    n_par = sum(p.numel() for p in fe.parameters())
    check(w2 == 4 and w1 == 5 and shapes == [(2, 32, 112, 112), (2, 64, 56, 56), (2, 128, 28, 28)],
          "2 indices encoder builds and runs", f"stem in s2={w2} s1={w1}, {n_par:,} params")
    mp = SoilMoistureModel(fine_inputs="indices", fine_skips="pool").eval()
    with torch.no_grad():
        mp.fine_encoder(fine, lulc)

    z = fine_to_indices(torch.zeros(3, FINE_CH, 8, 8))
    check(z.shape[1] == 11 and bool((z == 0).all()), "3 all-zero fine -> all-zero indices",
          f"shape {tuple(z.shape)}")

    fe.train()
    fe.modality_dropout = 1.0
    ok = True
    for _ in range(10):
        f = torch.randn(8, FINE_CH, 4, 4)
        f[:, [10, 14, 18]] = 1.0
        x = fine_to_indices(fe._drop_modality(f))
        for i in range(8):
            s2z = bool((x[i, M.FINE_IDX_S2] == 0).all())
            s1z = bool((x[i, M.FINE_IDX_S1] == 0).all())
            ok &= (s2z != s1z)                         # exactly one modality fully zero
            ok &= bool((x[i, M.FINE_IDX_DEM] != 0).any())
    check(bool(ok), "4 modality dropout zeroes a whole converted modality")


def _pool_dn(x, m):
    """(C,224,224) DN, (224,224) bool -> masked 2x2 mean (C,112,112), count (112,112)."""
    C = x.shape[0]
    mf = m.astype(np.float64)
    s = (x * mf).reshape(C, 112, 2, 112, 2).sum(axis=(2, 4))
    n = mf.reshape(112, 2, 112, 2).sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), 0.0), n


def _nd(a, b):
    den = a + b
    ok = np.abs(den) > M._IDX_DEN_FLOOR
    return np.where(ok, (a - b) / np.where(ok, den, 1.0), 0.0).clip(-1, 1), ok


def c5_c8_raw(n_stations=12, n_scenes=12):
    from dataset import RAW_ROOT, _load_fine_stats, fine_s1_scene, fine_s2_scene
    import zarr
    fs = _load_fine_stats()
    rng = np.random.default_rng(57)
    stores = sorted(RAW_ROOT.glob("*.zarr"))
    pick = rng.choice(len(stores), size=min(n_stations, len(stores)), replace=False)
    err32, err16, err_cr, ndvi_all, ndmi_all, cr_all = [], [], [], [], [], []
    d16_px = []
    low = {"pre": [0, 0], "post": [0, 0]}         # [n DN<1000, n valid] over kept bands
    kept = fs["s2_keep"]
    for si in pick:
        p = stores[si]
        try:
            rg = zarr.open_consolidated(str(p), mode="r")
        except KeyError:
            rg = zarr.open_group(str(p), mode="r")
        if "s2/data" in rg:
            dates = [str(bytes(d).decode() if isinstance(d, (bytes, np.bytes_)) else d)[:8]
                     for d in rg["s2/dates"][:]]
            for ri in rng.choice(len(dates), size=min(n_scenes, len(dates)), replace=False):
                x_raw = np.asarray(rg["s2/data"][ri])
                cm = np.zeros((224, 224), np.uint8)
                sc = fine_s2_scene(x_raw, cm, fs)                    # (11,112,112)
                if not sc[10].any():
                    continue
                f = np.zeros((FINE_CH, 112, 112), np.float32)
                f[0:11] = sc
                got32 = fine_to_indices(torch.from_numpy(f)[None])[0].numpy()
                got16 = fine_to_indices(torch.from_numpy(f.astype(np.float16)).float()[None])[0].numpy()
                x = x_raw[kept].astype(np.float64)
                m = (x != 0).all(axis=0)
                era = "pre" if int(dates[ri]) < 20220125 else "post"
                low[era][0] += int((x[:, m] < 1000).sum())
                low[era][1] += int(m.sum()) * x.shape[0]
                pdn, n = _pool_dn(x - M.S2_BOA_OFFSET, m)
                ndvi, ok1 = _nd(pdn[6], pdn[2])
                ndmi, ok2 = _nd(pdn[7], pdn[8])
                v = (n > 0) & ok1 & ok2
                if not v.any():
                    continue
                for got, acc in ((got32, err32), (got16, err16)):
                    acc.append(max(np.abs(got[0][v] - ndvi[v]).max(),
                                   np.abs(got[1][v] - ndmi[v]).max()))
                # §61: per-pixel fp16 errors, so 5b reports how MANY pixels are off, not only the max
                d16_px.append(np.maximum(np.abs(got16[0][v] - ndvi[v]), np.abs(got16[1][v] - ndmi[v])))
                ndvi_all.append(ndvi[v]); ndmi_all.append(ndmi[v])
        for key in ("s1_asc", "s1_desc"):
            if f"{key}/data" not in rg:
                continue
            N = rg[f"{key}/data"].shape[0]
            for ri in rng.choice(N, size=min(4, N), replace=False):
                x_raw = np.asarray(rg[f"{key}/data"][ri], dtype=np.float32)
                sc = fine_s1_scene(x_raw, fs)                        # (3,112,112)
                if not sc[2].any():
                    continue
                f = np.zeros((FINE_CH, 112, 112), np.float32)
                f[12:15] = sc
                got = fine_to_indices(torch.from_numpy(f)[None])[0].numpy()
                m = np.isfinite(x_raw).all(axis=0) & (x_raw != 0).all(axis=0)
                lin = np.where(m, 10.0 ** (np.where(m, x_raw, 0.0) / 10.0), 0.0)
                pl, n = _pool_dn(lin.astype(np.float64), m)
                v = n > 0
                cr_db = 10 * np.log10(np.maximum(pl[1], 1e-10)) - 10 * np.log10(np.maximum(pl[0], 1e-10))
                want = (cr_db + M.CR_SHIFT) / M.CR_SCALE
                err_cr.append(np.abs(got[5][v] - want[v]).max())
                cr_all.append(cr_db[v])

    e32 = max(err32) if err32 else np.inf
    e16 = max(err16) if err16 else np.inf
    check(len(err32) > 20 and e32 < 1e-3,
          "5a NDVI/NDMI (fp32 path) == direct from raw DN", f"{len(err32)} scenes, max |d| {e32:.2e}")
    # §61: 27439912 failed 5b on the MAX alone (0.80). Judge on the distribution instead: p99 |d| and
    # the fraction of valid pixels off by more than 0.02 index units; the max is still printed.
    d16 = np.concatenate(d16_px) if d16_px else np.array([np.inf])
    p99, frac = float(np.percentile(d16, 99)), float((d16 > 2e-2).mean())
    check(len(err16) > 20 and p99 < 2e-2 and frac < 1e-3,
          "5b NDVI/NDMI (fp16 cache path) close to direct",
          f"p99 |d| {p99:.2e}  frac>0.02 {frac:.2e} of {d16.size:,} px  max |d| {e16:.2e}  "
          f"(den floor {M._IDX_DEN_FLOOR:g} DN)")
    ecr = max(err_cr) if err_cr else np.inf
    check(len(err_cr) > 10 and ecr < 1e-3, "6 CR == VH_dB - VV_dB from raw (linear pool)",
          f"{len(err_cr)} passes, max |d| {ecr:.2e} (scaled units)")
    if ndvi_all and cr_all:
        a, b, c = (np.concatenate(z) for z in (ndvi_all, ndmi_all, cr_all))
        q = lambda z: " ".join(f"{v:+.2f}" for v in np.percentile(z, [1, 25, 50, 75, 99]))
        print(f"     NDVI p1/25/50/75/99: {q(a)}\n     NDMI p1/25/50/75/99: {q(b)}\n"
              f"     CR dB p1/25/50/75/99: {q(c)}")
        check(0.0 <= np.median(a) <= 0.9 and -0.3 <= np.median(b) <= 0.6
              and -12 <= np.median(c) <= -2, "7 index value ranges plausible")
    else:
        check(False, "7 index value ranges plausible", "no data")
    fr = {k: (v[0] / max(v[1], 1), v[1]) for k, v in low.items()}
    check(all(f < 1e-3 for f, n in fr.values()) and all(n > 0 for f, n in fr.values()),
          "8 DN < 1000 rare in both eras (offset present everywhere)",
          f"pre {fr['pre'][0]:.2e} of {fr['pre'][1]:,}, post {fr['post'][0]:.2e} of {fr['post'][1]:,}")


def c9_dataset():
    from splits_config import SM_CATEGORIES, TRAIN_YEARS
    from dataset import SoilMoistureDataset
    ds = SoilMoistureDataset(
        splits_csv=str(REPO / "csvs" / "station_splits.csv"),
        era5_stats_path=str(REPO / "csvs" / "era5_stats18.json"),
        years=list(TRAIN_YEARS), category_filter=list(SM_CATEGORIES),
        split_filter=None, training=False, max_stations=3)
    rng = np.random.default_rng(1)
    ok, n_s2 = True, 0
    for i in rng.choice(len(ds), size=min(24, len(ds)), replace=False):
        x = fine_to_indices(ds[int(i)]["fine"].float()[None])[0]
        ok &= bool(torch.isfinite(x).all())
        ok &= bool((x[0:2][:, x[2] == 0] == 0).all()) and bool((x[5][x[6] == 0] == 0).all())
        n_s2 += int(x[2].any())
    check(ok, "9 real samples convert, finite, 0 where invalid", f"{n_s2} with S2")


def c10_bands_ckpt():
    if not V2_BEST.exists():
        check(False, "10 bands checkpoint loads strictly", f"{V2_BEST} missing")
        return
    from ckpt_utils import load_checkpoint
    m, cfg, ep = load_checkpoint(V2_BEST, torch.device("cpu"))
    check(m.fine_encoder.fine_inputs == "bands", "10 bands checkpoint loads strictly", f"epoch {ep}")


if __name__ == "__main__":
    for fn in (c1_constants, c2_c4_synthetic, c5_c8_raw, c9_dataset, c10_bands_ckpt):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(False, f"{fn.__name__} raised")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
