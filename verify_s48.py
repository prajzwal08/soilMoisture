"""
verify_s48.py — pre-registered checks for the §48 build (§46.8 + §48.8), CPU only
==================================================================================
Run by slurm/verify_s48.sh; nothing here trains. Each check prints PASS/FAIL, and the
script exits non-zero if any fails.

  A. model, synthetic batch
     1  output shapes: sm (B,3,112,112), lst (B,1,22,22), z (B,64,112,112)
     2  fine-encoder parameter count (~0.3 M expected for cnn)
     3  step 0 == bottleneck-only: changing the fine input / LULC does not change sm or lst
     4  depth heads start at their own label_mean (map mean within 0.05) and differ
     5  modality dropout zeroes a WHOLE modality — data, valid flag and age — never a part
     6  lst_pattern_loss is invariant to a constant offset of the prediction (alpha = 0)
     7  lst_pattern_loss is 0-with-graph on a sample with no thermal cell
     8  backward of L_sm + lambda*L_lst reaches the fine encoder, head_lst and the trunk,
        and LambdaLST.update sets a finite positive lambda
     9  the "pool" ablation builds and runs
  B. dataset, real stores (the stations prepare_s48_cache.py has cached)
     10 construction admits samples
     11 per-sample tensors: shapes, no NaN in fine, invalid cells exactly 0 after
        normalisation, LULC in [1..10], valid fractions in [0,1], ages in [0,1]
     12 anchor / history never later than day D
     13 a collated real batch runs through the model
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from model import (FINE_CH, FINE_S1, FINE_S2, LULC_PAD, LST_N, SoilMoistureModel,  # noqa: E402
                   lst_pattern_loss, masked_huber_loss)

FAILS = []


def check(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


def synthetic_batch(B=2, g=None):
    g = g or torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    fine = r(B, FINE_CH, 112, 112).half()
    for ch in (10, 14, 18):
        fine[:, ch] = 1.0
    return {
        "s2_pyr": r(B, 60, 4, 768).half(), "s2_rel_pos": torch.randint(0, 365, (B, 60)),
        "s2_valid": torch.ones(B, 60, dtype=torch.bool),
        "s1_pyr": r(B, 40, 4, 768).half(), "s1_rel_pos": torch.randint(0, 365, (B, 40)),
        "s1_valid": torch.ones(B, 40, dtype=torch.bool),
        "s1_orbit": torch.randint(0, 2, (B, 40)),
        "anchor_l12": (4.6 * r(B, 196, 768)).half(),
        "anchor_rel_pos": torch.randint(0, 365, (B,)), "anchor_orbit": torch.randint(0, 3, (B,)),
        "dem_pyr": r(B, 4, 768), "lulc_pyr": r(B, 4, 768),
        "fine": fine, "lulc": torch.randint(1, 10, (B, 224, 224), dtype=torch.uint8),
        "soil_patch": r(B, 21, 74, 74),
        "era5": r(B, 365, 18), "era5_doys": torch.randint(1, 366, (B, 365)),
        "era5_rel_pos": torch.arange(365).expand(B, 365).clone(),
        "sif": r(B, 50, 1), "sif_doys": torch.randint(1, 366, (B, 50)),
        "sif_rel_pos": torch.randint(0, 365, (B, 50)), "sif_valid": torch.ones(B, 50, dtype=torch.bool),
        "twsa": r(B, 12, 1), "twsa_doys": torch.randint(1, 366, (B, 12)),
        "twsa_rel_pos": torch.randint(0, 365, (B, 12)), "twsa_valid": torch.ones(B, 12, dtype=torch.bool),
        "label": torch.tensor([[0.2, 0.25, float("nan")]] * B),
        "lst_obs": 290.0 + 3.0 * r(B, LST_N, LST_N),
    }


def part_a():
    print("\nA. MODEL (synthetic)", flush=True)
    stats = json.loads((REPO / "csvs" / "driver_stats.json").read_text())
    bias = [stats["label_mean"][d] for d in ("0-10", "10-30", "30-100")]
    torch.manual_seed(0)
    m = SoilMoistureModel(head_bias_init=bias).eval()
    b = synthetic_batch()
    with torch.no_grad():
        out = m(b)
    check(tuple(out["sm"].shape) == (2, 3, 112, 112) and tuple(out["lst"].shape) == (2, 1, 22, 22)
          and tuple(out["z"].shape) == (2, 64, 112, 112), "1 output shapes",
          f"sm {tuple(out['sm'].shape)} lst {tuple(out['lst'].shape)} z {tuple(out['z'].shape)}")

    n_fine = sum(p.numel() for p in m.fine_encoder.parameters())
    n_all = sum(p.numel() for p in m.parameters())
    check(1e5 < n_fine < 1e6, "2 fine-encoder parameters", f"{n_fine:,} of {n_all:,}")

    b2 = dict(b)
    b2["fine"] = torch.randn_like(b["fine"].float()).half()
    b2["lulc"] = torch.randint(1, 10, b["lulc"].shape, dtype=torch.uint8)
    with torch.no_grad():
        out2 = m(b2)
    d_sm, d_lst = (out2["sm"] - out["sm"]).abs().max().item(), (out2["lst"] - out["lst"]).abs().max().item()
    check(d_sm < 1e-5 and d_lst < 1e-5, "3 step 0 == bottleneck-only (fine input ignored at init)",
          f"max|dsm|={d_sm:.2e} max|dlst|={d_lst:.2e}")

    means = out["sm"].mean(dim=(0, 2, 3)).tolist()
    close = all(abs(a - b_) < 1e-4 for a, b_ in zip(means, bias))
    check(close and len({round(v, 6) for v in means}) == 3, "4 depth heads start at exactly label_mean",
          "means " + " ".join(f"{v:.4f}" for v in means) + "  vs " + " ".join(f"{v:.4f}" for v in bias))

    fe = m.fine_encoder
    fe.train()
    fe.modality_dropout = 1.0
    ok, seen = True, set()
    for s in range(20):
        torch.manual_seed(s)
        f = fe._drop_modality(torch.ones(4, FINE_CH, 4, 4))
        for i in range(4):
            s2_zero = bool((f[i, FINE_S2] == 0).all())
            s1_zero = bool((f[i, FINE_S1] == 0).all())
            s2_part = bool((f[i, FINE_S2] == 0).any()) and not s2_zero
            s1_part = bool((f[i, FINE_S1] == 0).any()) and not s1_zero
            ok &= (s2_zero != s1_zero) and not s2_part and not s1_part and bool((f[i, 17:] == 1).all())
            seen.add("s2" if s2_zero else "s1")
    fe.modality_dropout = 0.2
    fe.eval()
    check(ok and seen == {"s2", "s1"}, "5 modality dropout zeroes exactly one whole modality",
          f"both modalities dropped over 80 draws: {sorted(seen)}")

    sig = 2.7066
    pred = torch.randn(2, 1, 22, 22)
    obs = 290.0 + 3.0 * torch.randn(2, 22, 22)
    obs[0, :5] = float("nan")
    l1 = lst_pattern_loss(pred, obs, sig)
    l2 = lst_pattern_loss(pred + 7.3, obs, sig)
    l3 = lst_pattern_loss(pred, obs + 11.0, sig)
    check(abs(l1.item() - l2.item()) < 1e-5 and abs(l1.item() - l3.item()) < 1e-4,
          "6 thermal loss ignores the tile level (alpha = 0)",
          f"{l1.item():.6f} / {l2.item():.6f} / {l3.item():.6f}")

    p = torch.randn(2, 1, 22, 22, requires_grad=True)
    l0 = lst_pattern_loss(p, torch.full((2, 22, 22), float("nan")), sig)
    l0.backward()
    check(l0.item() == 0.0 and p.grad is not None, "7 thermal loss on a no-overpass batch is 0 with a graph")

    sys.path.insert(0, str(REPO))
    from train import LambdaLST  # noqa: E402
    m.train()
    out = m(b)
    l_sm = masked_huber_loss(out["sm"], b["label"])
    l_lst = lst_pattern_loss(out["lst"], b["lst_obs"], sig)
    lam = LambdaLST("auto")
    lam.update(l_sm, l_lst, out["z"], ddp_active=False)
    check(lam.value == 0.0 and lam.n_updates == 0 and lam.due(1),
          "8a step 0: dL_sm/dz == 0 (zero-weight heads), so lambda is NOT seeded from it")
    # Emulate the state after the first optimizer step: the heads have left zero.
    with torch.no_grad():
        for h in m.decoder.heads:
            h.weight.normal_(0.0, 1e-3)
    out = m(b)
    l_sm = masked_huber_loss(out["sm"], b["label"])
    l_lst = lst_pattern_loss(out["lst"], b["lst_obs"], sig)
    lam.update(l_sm, l_lst, out["z"], ddp_active=False)
    m.zero_grad()
    (l_sm + lam.value * l_lst).backward()
    gn = lambda mod: sum(float(q.grad.norm()) for q in mod.parameters() if q.grad is not None)  # noqa: E731
    g_skip = float(m.decoder.conv3.net[0].weight.grad[:, 128:].norm())
    check(np.isfinite(lam.value) and lam.value > 0 and gn(m.decoder.head_lst) > 0
          and gn(m.transformer_layers) > 0 and g_skip > 0,
          "8 backward reaches head_lst, trunk and the (zero-init) skip slices; lambda set",
          f"lambda={lam.value:.3e}  |g skip slice|={g_skip:.2e}  |g head_lst|={gn(m.decoder.head_lst):.2e}")

    mp = SoilMoistureModel(head_bias_init=bias, fine_skips="pool").eval()
    with torch.no_grad():
        op = mp(b)
    check(tuple(op["sm"].shape) == (2, 3, 112, 112), "9 pool ablation builds and runs",
          f"fine params {sum(q.numel() for q in mp.fine_encoder.parameters()):,}")


def part_b():
    print("\nB. DATASET (real stores)", flush=True)
    from splits_config import SM_CATEGORIES, TRAIN_YEARS
    from dataset import SoilMoistureDataset
    ds = SoilMoistureDataset(
        splits_csv=str(REPO / "csvs" / "station_splits.csv"),
        era5_stats_path=str(REPO / "csvs" / "era5_stats18.json"),
        years=list(TRAIN_YEARS), category_filter=list(SM_CATEGORIES),
        split_filter=None, training=False, max_stations=3)
    check(len(ds) > 0, "10 dataset admits samples", f"{len(ds)} samples")
    if not len(ds):
        return
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(24, len(ds)), replace=False)
    bad, n_lst, n_s2, n_s1 = [], 0, 0, 0
    for i in idx:
        s = ds[int(i)]
        f = s["fine"].float()
        prob = []
        if tuple(f.shape) != (19, 112, 112):
            prob.append(f"fine {tuple(f.shape)}")
        if torch.isnan(f).any():
            prob.append("NaN in fine")
        for sl, vch in ((slice(0, 10), 10), (slice(12, 14), 14), (slice(17, 18), 18)):
            inval = f[vch] == 0
            if inval.any() and f[sl][:, inval].abs().max() > 0:
                prob.append(f"non-zero data where valid ch {vch} == 0")
        for vch in (10, 14, 18, 11, 15):
            if f[vch].min() < 0 or f[vch].max() > 1.0001:
                prob.append(f"ch {vch} outside [0,1]")
        lu = s["lulc"]
        if lu.min() < 1 or lu.max() > LULC_PAD:
            prob.append(f"lulc range {int(lu.min())}..{int(lu.max())}")
        for k, shp in (("s2_pyr", (60, 4, 768)), ("s1_pyr", (40, 4, 768)),
                       ("anchor_l12", (196, 768)), ("lst_obs", (22, 22)), ("era5", (365, 18))):
            if tuple(s[k].shape) != shp:
                prob.append(f"{k} {tuple(s[k].shape)}")
        if int(s["anchor_rel_pos"]) > 364 or int(s["s2_rel_pos"].max()) > 364:
            prob.append("acquisition after day D")
        n_lst += bool(torch.isfinite(s["lst_obs"]).any())
        n_s2 += bool((f[10] > 0).any())
        n_s1 += bool((f[14] > 0).any())
        if prob:
            bad.append((s["station_key"], s["year"], s["doy"], prob))
    check(not bad, "11 per-sample tensors (shapes, NaN, zero-after-norm, ranges)",
          f"{len(idx)} samples; S2 present {n_s2}, S1 present {n_s1}, LST target {n_lst}"
          + ("" if not bad else f"; first bad: {bad[0]}"))
    check(all("acquisition after day D" not in p for *_, p in bad), "12 nothing after day D")

    from torch.utils.data import default_collate
    batch = default_collate([ds[int(i)] for i in idx[:2]])
    stats = json.loads((REPO / "csvs" / "driver_stats.json").read_text())
    m = SoilMoistureModel(head_bias_init=[stats["label_mean"][d] for d in ("0-10", "10-30", "30-100")]).eval()
    with torch.no_grad():
        out = m(batch)
    ok = all(torch.isfinite(out[k]).all() for k in ("sm", "lst"))
    check(ok, "13 real batch through the model", f"sm station px {out['sm'][:, :, 56, 56].tolist()}")


if __name__ == "__main__":
    for part in (part_a, part_b):
        try:
            part()
        except Exception:
            traceback.print_exc()
            FAILS.append(part.__name__ + " crashed")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
