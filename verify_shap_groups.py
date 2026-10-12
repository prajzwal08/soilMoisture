"""§73 verification of shap_groups.py against the real dataset and model (GPU job, no training).

1. Synthetic-game self-test (additive, interaction split, efficiency).
2. Recipe == training dropout: for one batch, each absent recipe must give exactly the tensors
   the dataset produces with that dropout forced to p = 1 (era5, coarse, static).
3. All-on coalition, stacked with other coalitions, reproduces the plain forward (bf16-close).
4. phi_sat == 0 on samples whose window holds no S2/S1 scene and no anchor (dummy property).

Usage: python verify_shap_groups.py --ckpt <path/best.pt> [--n 64]
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ckpt_utils import load_checkpoint
from dataset import SoilMoistureDataset
from eval_predict import ERA5_STATS
from model import SoilMoistureModel
from shap_groups import PLAYERS, apply_absent, self_test, shapley_weight_matrix
from splits_config import SM_CATEGORIES, TRAIN_YEARS
from train import CONFIG, CudaPrefetcher, worker_init_fn

KEYS = {
    "era5": ("era5", "era5_doys"),
    "sat":  ("anchor_l12", "anchor_rel_pos", "anchor_orbit", "anchor_found",
             "s2_pyr", "s2_valid", "s1_pyr", "s1_valid"),
    "dem":  ("dem_pyr", "dem_valid", "fine"),
    "lulc": ("lulc_pyr", "lulc_valid", "lulc"),
    "soil": ("soil_patch", "soil_valid"),
}
FORCE = {"era5": dict(era5_dropout=1.0), "sat": dict(coarse_dropout=1.0),
         "dem": dict(static_dropout=1.0), "lulc": dict(static_dropout=1.0),
         "soil": dict(static_dropout=1.0)}


def build(split, n, **dropout):
    kw = dict(splits_csv=CONFIG["splits_csv"], era5_stats_path=str(ERA5_STATS), years=list(TRAIN_YEARS),
              category_filter=list(SM_CATEGORIES), split_filter=[split], max_stations=2,
              sif_twsa_dropout=0.0, **dropout)
    ds = SoilMoistureDataset(training=bool(dropout), **kw)
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()
    ok = True

    self_test()
    print("1. synthetic self-test: PASS")

    # 2. recipe vs forced training dropout, matched by sample identity (station, year, doy)
    base = build("oos", args.n)
    idx = list(range(min(args.n, len(base))))
    ident = lambda s: (s["station_key"], s["year"], s["doy"])
    one = lambda item: {k: (torch.as_tensor(v).unsqueeze(0) if isinstance(v, (np.ndarray, torch.Tensor,
                        bool, int, float, np.generic)) else [v]) for k, v in item.items()}
    for player in PLAYERS:
        forced = build("oos", args.n, **FORCE[player])
        where = {ident(s): j for j, s in enumerate(forced.samples)}
        mask = (2 ** len(PLAYERS) - 1) & ~(1 << PLAYERS.index(player))
        bad, n_cmp = [], 0
        for i in idx[:16]:
            j = where.get(ident(base.samples[i]))
            if j is None:
                continue
            got, ref = apply_absent(one(base[i]), mask), forced[j]
            n_cmp += 1
            for k in KEYS[player]:
                g, r = got[k][0], torch.as_tensor(ref[k])
                if k == "fine":   # only the DEM channels belong to the dem player; the rest is context
                    g, r = g[17:19], r[17:19]
                if not torch.equal(g.to(r.dtype), r):
                    bad.append((i, k))
        res = "FAIL " + str(bad[:4]) if bad or n_cmp == 0 else "PASS"
        print(f"2. recipe == forced dropout [{player}] on {n_cmp} samples: {res}")
        ok &= (not bad) and n_cmp > 0

    # 3 + 4. model checks on one batch
    model, cfg, _ = load_checkpoint(Path(args.ckpt), torch.device("cpu"))
    device = torch.device("cuda")
    model = model.to(device).eval()
    loader = DataLoader(Subset(base, idx), batch_size=len(idx), shuffle=False, num_workers=4,
                        worker_init_fn=worker_init_fn)
    batch = next(iter(CudaPrefetcher(loader, device)))
    B = batch["era5"].shape[0]
    sr, sc = SoilMoistureModel.STATION_ROW, SoilMoistureModel.STATION_COL
    K, full = len(PLAYERS), 2 ** len(PLAYERS) - 1
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        plain = model(batch)["sm"][:, :, sr, sc].double()
        f = torch.empty(2 ** K, B, plain.shape[1], device=device, dtype=torch.float64)
        for c0 in range(0, 2 ** K, 4):
            masks = list(range(c0, c0 + 4))
            parts = [apply_absent(batch, m) for m in masks]
            st = {k: (torch.cat([q[k] for q in parts], 0) if torch.is_tensor(v) and v.dim() > 0
                      and v.shape[0] == B else v) for k, v in batch.items()}
            f[c0:c0 + 4] = model(st)["sm"][:, :, sr, sc].double().view(4, B, -1)
    d_full = (f[full] - plain).abs().max().item()
    print(f"3. all-on coalition vs plain forward: max |diff| {d_full:.2e} "
          f"({'PASS' if d_full < 5e-3 else 'FAIL'})  [bf16, different batch composition]")
    ok &= d_full < 5e-3
    phi = torch.einsum("km,mbd->kbd", shapley_weight_matrix(K).to(device), f)
    eff = (phi.sum(0) - (f[full] - f[0])).abs().max().item()
    print(f"   efficiency max err {eff:.2e} ({'PASS' if eff < 1e-5 else 'FAIL'})")
    ok &= eff < 1e-5
    empty = (~batch["s2_valid"].bool().any(1)) & (~batch["s1_valid"].bool().any(1)) & (~batch["anchor_found"].bool())
    if empty.any():
        z = phi[PLAYERS.index("sat"), empty].abs().max().item()
        print(f"4. phi_sat on {int(empty.sum())} empty-window samples: max |phi| {z:.2e} "
              f"({'PASS' if z < 1e-6 else 'FAIL'})")
        ok &= z < 1e-6
    else:
        print("4. phi_sat dummy check: no empty-window sample in this batch (not tested)")
    print("mean |phi| per player (0-10 cm) on this batch:",
          {p: round(float(phi[k, :, 0].abs().mean()), 4) for k, p in enumerate(PLAYERS)})
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    main()
