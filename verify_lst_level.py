"""
verify_lst_level.py — checks for the §52 thermal LEVEL term (dT = tile LST - t2m_mean)
=====================================================================================

1. lst_pattern_loss is unchanged by a constant shift of the prediction (the level really is
   invisible to it), and lst_level_loss is not (it has a gradient on the level).
2. lst_level_loss masks NaN targets without NaN gradients; zero targets -> zero loss.
3. lst_level_summary reproduces numpy r / RMSE / bias / skill.
4. era5/date_ints is strictly increasing in every station store (the dataset's dT lookup
   uses searchsorted).
5. SoilMoistureDataset._lst_dT reproduces probe_lst_level_pattern.py's dT_mean for the same
   (station, date) — two independent code paths to the same number.

Exit code 1 on any failure. CPU only; run inside an sbatch job.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import (SoilMoistureDataset, _load_lst22, _load_zarr_era5, _open_zarr)
from model import lst_level_loss, lst_level_stats, lst_level_summary, lst_pattern_loss
from splits_config import category_of, station_dir_name

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def main():
    torch.manual_seed(0)
    sig, sd = 2.7066, 6.0
    B = 8
    obs = 300 + 3 * torch.randn(B, 22, 22)
    obs[:, :5, :] = float("nan")                           # partial scenes
    obs[0] = float("nan")                                  # a no-overpass sample
    dT = torch.randn(B) * 5 + 12
    dT[0] = float("nan")
    pred = torch.randn(B, 1, 22, 22, requires_grad=True)

    print("1. pattern loss blind to the level, level loss is not")
    lp0 = lst_pattern_loss(pred, obs, sig)
    lp1 = lst_pattern_loss(pred + 5.0, obs, sig)
    check("pattern loss invariant to +5 shift", abs(lp0.item() - lp1.item()) < 1e-5,
          f"{lp0.item():.6f} vs {lp1.item():.6f}")
    ll0 = lst_level_loss(pred, obs, dT, sig, sd)
    ll1 = lst_level_loss(pred + 5.0, obs, dT, sig, sd)
    check("level loss changes with +5 shift", abs(ll0.item() - ll1.item()) > 1e-3,
          f"{ll0.item():.4f} vs {ll1.item():.4f}")

    print("2. masking")
    ll0.backward()
    check("level grad finite", bool(torch.isfinite(pred.grad).all()))
    check("no grad into the no-target sample", float(pred.grad[0].abs().sum()) == 0.0)
    # a prediction whose level equals the target exactly -> zero loss
    with torch.no_grad():
        v = torch.isfinite(obs).float()
        exact = torch.zeros(B, 1, 22, 22)
        exact[:, 0] = (torch.nan_to_num(dT, nan=0.0) / sig).view(-1, 1, 1)
    check("exact level -> zero loss", lst_level_loss(exact, obs, dT, sig, sd).item() < 1e-10)
    allnan = torch.full((B,), float("nan"))
    z = lst_level_loss(pred, obs, allnan, sig, sd, return_count=True)
    check("no targets -> 0 loss, n 0", z[0].item() == 0.0 and z[1].item() == 0.0)

    print("3. summary vs numpy")
    s = lst_level_stats(pred.detach(), obs, dT, sig)
    m = lst_level_summary(s.tolist())
    ok = torch.isfinite(dT).numpy()
    vf = torch.isfinite(obs).float()
    p = ((pred.detach()[:, 0] * sig * vf).flatten(1).sum(1) / vf.flatten(1).sum(1).clamp_min(1)
         ).numpy()[ok]
    o = dT.numpy()[ok]
    r_np = np.corrcoef(p, o)[0, 1]
    rmse_np = np.sqrt(np.mean((p - o) ** 2))
    check("r", abs(m["r"] - r_np) < 1e-4, f"{m['r']:.5f} vs {r_np:.5f}")
    check("rmse", abs(m["rmse_K"] - rmse_np) < 1e-3, f"{m['rmse_K']:.4f} vs {rmse_np:.4f}")
    check("bias", abs(m["bias_K"] - (p - o).mean()) < 1e-3)
    check("skill", abs(m["skill"] - (1 - np.mean((p - o) ** 2) / o.var())) < 1e-3)

    print("4. era5/date_ints strictly increasing")
    sp = pd.read_csv("csvs/station_splits.csv")
    sp = sp[sp["has_soil_moisture"].astype(str).str.lower() == "true"]
    bad, n = [], 0
    for r in sp.itertuples():
        row = sp.loc[r.Index]
        zg = _open_zarr(Path(station_dir_name(row)), category_of(row))
        if zg is None:
            continue
        e = _load_zarr_era5(zg)
        if e is None:
            continue
        n += 1
        if not np.all(np.diff(np.asarray(e[1], dtype=np.int64)) > 0):
            bad.append(station_dir_name(row))
    check(f"sorted in {n} stores", not bad, f"unsorted: {bad[:5]}")

    print("5. dataset _lst_dT == probe dT_mean")
    sc = Path("csvs/probe_lst_level_pattern/scenes.csv")
    if not sc.exists():
        check("probe scenes.csv present", False, str(sc))
    else:
        sc = pd.read_csv(sc)
        ds = SoilMoistureDataset.__new__(SoilMoistureDataset)   # only the two caches it reads
        ds._era5_cache = {}
        diffs, nd = [], 0
        for st, g in list(sc.groupby("station"))[:10]:
            row = sp[sp.apply(station_dir_name, axis=1) == st].iloc[0]
            cat = category_of(row)
            ds._era5_cache[st] = _load_zarr_era5(_open_zarr(Path(st), cat))
            idx, arr = _load_lst22(cat, st)
            for d, ref in zip(g["date"], g["dT_mean"]):
                got = ds._lst_dT(st, int(d), np.asarray(arr[idx[int(d)]], np.float32))
                if np.isfinite(ref) or np.isfinite(got):
                    diffs.append(abs(got - ref) if np.isfinite(got) and np.isfinite(ref)
                                 else np.inf)
                    nd += 1
        mx = max(diffs) if diffs else np.inf
        check(f"{nd} scenes over 10 stations agree", mx < 1e-3, f"max |diff| {mx:.2e} K")

    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
