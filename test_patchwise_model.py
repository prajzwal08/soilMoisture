"""
Regression tests for the patchwise encoder (--arch patchwise, §35.18 / §35.20).

Every check here corresponds to a failure that would NOT crash. That is the whole point: a
patchwise model with the wrong cache indexing, a constructed-but-unused decoder, or a stale
ablation key list all run happily and return plausible soil-moisture numbers.

Synthetic tensors only — no zarr, no GPU, no dataset construction.

DELIBERATELY TINY (d_model=64, 4 heads, 2 layers). Every property tested here — shapes, cache
indexing, gradient coverage, arch gating — is independent of width and depth, while a full
768/12/6 model does forward AND backward on CPU in minutes rather than seconds. A slow test is
a test nobody runs, and this one has to be runnable on a login node.

    python test_patchwise_model.py
"""
import torch
import torch.nn as nn

from dataset import MAX_S1, MAX_S2
from model import SoilMoistureModel, masked_huber_loss

# Small enough for a login node; see the module docstring.
D_MODEL, N_HEADS, N_LAYERS = 64, 4, 2

FAILED = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def fake_batch(B, K, seed=0):
    """A patchwise batch exactly as dataset.py emits it with token_sel set."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)          # noqa: E731
    D = D_MODEL
    ri = lambda hi, s: torch.randint(0, hi, s, generator=g)  # noqa: E731
    return {
        "s2_hist": r(B, MAX_S2, K, D).half(),
        "s2_hist_valid": torch.ones(B, MAX_S2, K, dtype=torch.bool),
        "s2_rel_pos": ri(365, (B, MAX_S2)),
        "s1_hist": r(B, MAX_S1, K, D).half(),
        "s1_hist_valid": torch.ones(B, MAX_S1, K, dtype=torch.bool),
        "s1_rel_pos": ri(365, (B, MAX_S1)),
        "dem_tok": r(B, K, D).half(),
        "lulc_tok": r(B, K, D).half(),
        "token_idx": torch.full((B, K), 105),
        "token_valid": torch.ones(B, K, dtype=torch.bool),
        "soil_patch": r(B, 21, 74, 74),
        "era5": r(B, 365, 19),
        "era5_doys": torch.randint(1, 366, (B, 365), generator=g),
        "sif": r(B, 50, 1), "sif_doys": torch.randint(1, 366, (B, 50), generator=g),
        "sif_rel_pos": ri(365, (B, 50)), "sif_valid": torch.ones(B, 50, dtype=torch.bool),
        "twsa": r(B, 12, 1), "twsa_doys": torch.randint(1, 366, (B, 12), generator=g),
        "twsa_rel_pos": ri(365, (B, 12)), "twsa_valid": torch.ones(B, 12, dtype=torch.bool),
        "label": torch.rand(B, 3, generator=g),
    }


def build(**kw):
    kw.setdefault("driver_layers", 2)
    kw.setdefault("d_model", D_MODEL)
    kw.setdefault("n_heads", N_HEADS)
    kw.setdefault("n_layers", N_LAYERS)
    return SoilMoistureModel(n_depths=3, use_cls_depth=True, **kw)


print("\n── construction ──────────────────────────────────────────────────────")

# The unet path is gone entirely (§35.22) so there is nothing left to gate — but the DDP
# precondition it protected still matters, and is checked under "gradients" below: train.py
# wraps in DDP without find_unused_parameters, so ANY constructed-but-ungradiented parameter
# raises RuntimeError on step 1 of every multi-GPU run.
m = build()
for name in ("decoder", "transformer_layers", "scale_emb", "spatial_row_emb",
             "spatial_modality_emb", "arch"):
    check(f"no unet remnant: {name}", not hasattr(m, name))

try:
    build(use_cls_depth=False)
    check("refuses use_cls_depth=False", False)
except (ValueError, TypeError):
    check("refuses use_cls_depth=False", True)


print("\n── shapes ────────────────────────────────────────────────────────────")

for mode in ("memory", "concat"):
    net = build(driver_mode=mode).eval()
    for K in (1, 4):
        with torch.no_grad():
            out = net(fake_batch(2, K, seed=11))
        check(f"{mode}: K={K} -> (B,K,n_depths)", tuple(out.shape) == (2, K, 3),
              str(tuple(out.shape)))

# The two modes must be interchangeable from the caller's point of view — that is what makes
# --driver-mode an arm rather than a rewrite.
with torch.no_grad():
    a = build(driver_mode="memory").eval()(fake_batch(2, 3, seed=12))
    b = build(driver_mode="concat").eval()(fake_batch(2, 3, seed=12))
check("memory and concat agree on output shape", a.shape == b.shape)


print("\n── the cache is indexed by SAMPLE, not patch ──────────────────────────")

# THE silent bug. kc is (B,431,d) while x is (B*K,106,d); if the expansion uses repeat()
# instead of an interleave-compatible view, every patch reads a DIFFERENT sample's weather.
# Nothing crashes — you just get a mediocre model. Permuting only the drivers across the batch
# must therefore change every prediction.
net = build().eval()
bt = fake_batch(4, 2, seed=13)
with torch.no_grad():
    base = net(bt)
perm = [1, 2, 3, 0]
bt2 = dict(bt)
for k in ("era5", "era5_doys", "sif", "sif_doys", "sif_rel_pos", "sif_valid",
          "twsa", "twsa_doys", "twsa_rel_pos", "twsa_valid", "soil_patch"):
    bt2[k] = bt[k][perm]
with torch.no_grad():
    swapped = net(bt2)
d = (base - swapped).abs().max().item()
check("permuting the weather across the batch changes predictions", d > 1e-4, f"max|d|={d:.6f}")

# And patches within one sample must differ — they carry different history.
with torch.no_grad():
    o = net(fake_batch(2, 4, seed=14))
sp = (o[:, 0, :] - o[:, 1, :]).abs().max().item()
check("patches within a sample are not identical", sp > 1e-5, f"max|d|={sp:.6f}")


print("\n── loss ──────────────────────────────────────────────────────────────")

net = build().eval()
bt = fake_batch(3, 1, seed=15)
with torch.no_grad():
    mu = net(bt)
loss, dsum, dcnt = masked_huber_loss(mu, bt["label"], per_depth=True, return_breakdown=True)
check("masked_huber_loss accepts (B,1,n_depths)", torch.isfinite(loss).item(),
      f"loss={loss.item():.5f}")
check("per-depth counts are right", dcnt.tolist() == [3.0, 3.0, 3.0], str(dcnt.tolist()))

# NaN labels (a depth with no observation) must be skipped, not propagated.
bt["label"][0, 1] = float("nan")
loss2, _, dcnt2 = masked_huber_loss(net(bt), bt["label"], per_depth=True, return_breakdown=True)
check("NaN labels are excluded", torch.isfinite(loss2).item() and dcnt2.tolist() == [3., 2., 3.],
      str(dcnt2.tolist()))

# K>1 is inference-only: supervising several patches needs multi-station labels, which the
# dataset does not emit (§35.19). Silently averaging them onto one label would be wrong.
try:
    masked_huber_loss(torch.randn(2, 4, 3), torch.rand(2, 3))
    check("loss rejects K>1", False)
except ValueError:
    check("loss rejects K>1", True)

# lambda_tv is 0.0 by default so this is dead today, but a run that sets it must not get a
# number computed over the wrong axis.
# A 224x224 map is what the deleted U-Net produced; handing one to this loss means the caller
# is still on the old contract.
try:
    masked_huber_loss(torch.randn(2, 3, 224, 224), torch.rand(2, 3))
    check("loss rejects a 4-D map", False)
except ValueError:
    check("loss rejects a 4-D map", True)


print("\n── gradients (the DDP precondition) ──────────────────────────────────")

net = build().train()
bt = fake_batch(2, 1, seed=16)
loss, *_ = masked_huber_loss(net(bt), bt["label"], per_depth=True, return_breakdown=True)
loss.backward()
missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
check("every parameter receives a gradient", not missing, f"{len(missing)}: {missing[:6]}")

net = build(driver_mode="concat").train()
bt = fake_batch(2, 1, seed=17)
loss, *_ = masked_huber_loss(net(bt), bt["label"], per_depth=True, return_breakdown=True)
loss.backward()
missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
check("concat: every parameter receives a gradient", not missing, f"{len(missing)}: {missing[:6]}")


print("\n── readout ──────────────────────────────────────────────────────────")

# Each depth CLS is its own readout: no shared vector, no FiLM, no patch_cls. Nothing should
# be left over from the modulated design.
net = build()
check("no depth_film", not hasattr(net, "depth_film"))
check("no patch_cls",  not hasattr(net, "patch_cls"))
check("one head per depth", len(net.depth_heads) == 3, str(len(net.depth_heads)))

# The three depths must be able to differ at init -- zero-init depth_tokens plus no positional
# encoding on those slots makes attention permutation-equivariant over them, so all three
# would be numerically identical forever.
dt = net.depth_tokens.detach()
cos = torch.nn.functional.normalize(dt, dim=-1) @ torch.nn.functional.normalize(dt, dim=-1).T
check("depth tokens are not identical at init",
      bool((cos - torch.eye(3)).abs().max() < 0.9), f"max off-diag cos {float((cos-torch.eye(3)).abs().max()):.3f}")

print("\n── entropy diagnostic ────────────────────────────────────────────────")

# §35.20 dropped register standardisation on the explicit condition that this detector works.
# If it silently returns None the run has no way to distinguish "un-pooling does not help"
# from "temporal attention collapsed to a mean".
net = build().eval()
for blk in net.patch_blocks:
    blk.collect_entropy = True
with torch.no_grad():
    net(fake_batch(2, 1, seed=18))
ent = getattr(net, "_last_attn_entropy", None)
check("entropy is collected when armed", ent is not None and ent.numel() == len(net.patch_blocks),
      str(None if ent is None else ent.shape))
if ent is not None:
    import math
    check("entropy is finite and below the uniform bound",
          bool(torch.isfinite(ent).all()) and float(ent.max()) <= math.log(105) + 1e-3,
          f"max={float(ent.max()):.3f} uniform(105)={math.log(105):.3f}")

net2 = build().eval()
with torch.no_grad():
    net2(fake_batch(2, 1, seed=18))
check("entropy is NOT collected by default (SDPA stays enabled)",
      getattr(net2, "_last_attn_entropy", None) is None)


print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
raise SystemExit(1 if FAILED else 0)
