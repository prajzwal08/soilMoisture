"""CPU tests for per-depth loss reporting (runbook §19.3-19.4).

Run:  python test_per_depth_loss.py

No GPU, no data, no DDP — everything here is arithmetic that must hold before a
multi-day H100 job is worth launching.  The reason this file exists is that two
of the paths below cannot be relied on to execute in a smoke run:

  * `_format_depth_line(m=None)` only fires when a depth has training samples but
    no validation samples.  Both smoke subsets happened to carry all three depths
    in val, so the branch never ran.
  * The DDP sum-reduction is only exercised with >1 rank and uneven per-rank depth
    coverage, which a 4-station smoke does not reliably produce.
"""
import math
import torch

from model import masked_huber_loss
from train import _per_depth_mean, _loss_aggregates, _format_depth_line
from dataset import SM_DEPTHS

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# ── 1. The scalar loss must be untouched by return_breakdown ──────────────────
# val_loss drives best.pt, early stopping and the LR scheduler.  If the flag
# perturbed it by even a bit, runs would stop being comparable to the baseline.
print("\n1. scalar loss unchanged")
torch.manual_seed(0)
sm  = torch.rand(16, 3, 224, 224, requires_grad=True)
lab = torch.rand(16, 3)
lab[3:, 2] = float("nan")          # depth 2 sparse
lab[:, 1]  = float("nan")          # depth 1 entirely absent

for per_depth in (True, False):
    a = masked_huber_loss(sm, lab, per_depth=per_depth)
    b, _, _ = masked_huber_loss(sm, lab, per_depth=per_depth, return_breakdown=True)
    check(f"per_depth={per_depth} scalar identical", torch.equal(a, b), f"{a.item():.8f}")


# ── 2. The breakdown must carry no gradient ───────────────────────────────────
# It is a diagnostic.  If it were differentiable it would silently join the
# objective the moment anyone summed it into the loss.
print("\n2. breakdown is detached")
loss, ds, dc = masked_huber_loss(sm, lab, per_depth=True, return_breakdown=True)
check("loss still requires grad", loss.requires_grad)
check("depth_sum detached", not ds.requires_grad and ds.grad_fn is None)
check("depth_cnt detached", not dc.requires_grad and dc.grad_fn is None)

sm.grad = None
masked_huber_loss(sm, lab, per_depth=True).backward()
g_plain = sm.grad.clone()
sm.grad = None
masked_huber_loss(sm, lab, per_depth=True, return_breakdown=True)[0].backward()
check("gradient bit-identical", torch.equal(g_plain, sm.grad))


# ── 3. Absent depth reports nan, never 0.0 ────────────────────────────────────
# 0.0 would read as a perfect fit on a depth that was never observed — the exact
# failure mode that let 30-100 cm regress unnoticed for four epochs.
print("\n3. absent depth -> nan")
got = _per_depth_mean(torch.tensor([0.28, 0.0, 0.043]), torch.tensor([16.0, 0.0, 3.0]))
check("observed depth is finite", math.isfinite(got[SM_DEPTHS[0]]), f"{got[SM_DEPTHS[0]]:.6f}")
check("unobserved depth is nan", math.isnan(got[SM_DEPTHS[1]]))
check("keys match SM_DEPTHS", list(got) == SM_DEPTHS, str(list(got)))

allnan = _per_depth_mean(torch.zeros(3), torch.zeros(3))
check("all depths absent -> all nan", all(math.isnan(v) for v in allnan.values()))


# ── 4. An all-NaN label batch must not crash ──────────────────────────────────
print("\n4. all-NaN batch")
l, s, c = masked_huber_loss(sm, torch.full((16, 3), float("nan")), per_depth=True,
                            return_breakdown=True)
check("loss is 0.0", l.item() == 0.0)
check("counts are zero", c.sum().item() == 0)


# ── 4b. A non-finite prediction at an UNOBSERVED depth must not poison the sum ─
# `elem * valid` looks correct but nan * False is nan, not 0. pred is taken for ALL
# depths while the scalar loss only sees pred[mask], so a nan at a depth with no label
# cannot affect training -- but it would make depth_sum nan, survive all_reduce(SUM) to
# every rank, and silently blank the per-depth diagnostic while train_loss stayed
# healthy. That is precisely the signal the breakdown exists to provide.
print("\n4b. nan at an unobserved depth")
sm_p = torch.rand(4, 3, 224, 224)
sm_p[:, 1, 112, 112] = float("nan")        # nan prediction ...
lab_p = torch.rand(4, 3)
lab_p[:, 1] = float("nan")                 # ... at a depth with no label
l_p, s_p, c_p = masked_huber_loss(sm_p, lab_p, per_depth=True, return_breakdown=True)
check("scalar loss stays finite", torch.isfinite(l_p).item())
check("depth_sum has no nan", bool(torch.isfinite(s_p).all()), str(s_p.tolist()))
check("unobserved depth sums to exactly 0", s_p[1].item() == 0.0)
check("observed depths unaffected", s_p[0].item() > 0 and s_p[2].item() > 0)


# ── 5. DDP sum-reduction equals a single-pass computation ─────────────────────
# Accumulating per-batch SUMS and all_reduce(SUM)ing them must reproduce the
# result of one pooled pass.  Means would not: they cannot be averaged when the
# per-rank sample counts differ, which they always do with sparse deep coverage.
print("\n5. DDP sum-reduce == single pass")
torch.manual_seed(1)
N_RANK, N_BATCH, BS = 4, 3, 8
sm_f  = torch.rand(N_RANK * N_BATCH * BS, 3, 224, 224)
lab_f = torch.rand(N_RANK * N_BATCH * BS, 3)
lab_f[::3, 1] = float("nan")       # depth 1 patchy
lab_f[:40, 2] = float("nan")       # depth 2 missing from the early shards

valid = ~torch.isnan(lab_f)
elem  = torch.nn.functional.huber_loss(
    sm_f[:, :, 112, 112], torch.nan_to_num(lab_f), delta=0.05, reduction="none")
truth = ((elem * valid).sum(0) / valid.sum(0).clamp(min=1)).tolist()

S, C = torch.zeros(3), torch.zeros(3)
for r in range(N_RANK):
    for b in range(N_BATCH):
        i = (r * N_BATCH + b) * BS
        _, d_s, d_c = masked_huber_loss(sm_f[i:i + BS], lab_f[i:i + BS], return_breakdown=True)
        S += d_s
        C += d_c                    # <- all_reduce(SUM) is associative, so this is equivalent
got = _per_depth_mean(S, C)
check("counts sum to total valid", C.sum().item() == valid.sum().item(),
      f"{int(C.sum().item())}")
check("per-depth means match single pass",
      all(abs(got[d] - truth[i]) < 1e-6 for i, d in enumerate(SM_DEPTHS)),
      str([round(v, 8) for v in truth]))


# ── 6. Aggregates ─────────────────────────────────────────────────────────────
# pooled is the cross-run comparable scalar; depth_mean must equal the plain
# average of the printed per-depth lines so the log reconciles on its face.
print("\n6. aggregates")
pooled, dmean = _loss_aggregates(S, C)
check("pooled == Ssum/Scnt", abs(pooled - S.sum().item() / C.sum().item()) < 1e-12,
      f"{pooled:.8f}")
check("depth_mean == mean of per-depth lines",
      abs(dmean - sum(truth) / 3) < 1e-6, f"{dmean:.8f}")
check("pooled != depth_mean under uneven coverage", abs(pooled - dmean) > 1e-9,
      f"pooled={pooled:.6f} depth_mean={dmean:.6f}")

p2, d2 = _loss_aggregates(torch.zeros(3), torch.zeros(3))
check("no data -> both nan", math.isnan(p2) and math.isnan(d2))

# depth_mean ignores absent depths rather than counting them as zero
p3, d3 = _loss_aggregates(torch.tensor([0.4, 0.0, 0.2]), torch.tensor([10.0, 0.0, 5.0]))
check("depth_mean skips absent depth", abs(d3 - (0.04 + 0.04) / 2) < 1e-9, f"{d3:.6f}")


# ── 7. The per-depth print line, including the branch smokes never hit ────────
print("\n7. _format_depth_line")
m = {"MSE": 0.0127, "MAE": 0.0950, "ubRMSE": 0.0913, "bias": 0.0659}
line = _format_depth_line("0-10", 0.026809, 0.003615, m)
check("normal line has all stats", all(k in line for k in ("train_loss", "val_loss", "ubRMSE")))
check("normal line is 6dp", "0.026809" in line and "0.003615" in line, line.strip())

# THE branch that neither smoke exercised: trained, but no val samples.
line_novl = _format_depth_line("30-100", 0.027312, float("nan"), None)
check("m=None does not raise", True)
check("m=None says 'no val samples'", "no val samples" in line_novl, line_novl.strip())
check("m=None still shows train loss", "0.027312" in line_novl)
check("m=None shows nan val", "nan" in line_novl)


# ── 8. No normalisation parameter may land in the weight-decay group ──────────
# Regression guard. The original filter was name-based and missed every BatchNorm2d
# in the decoder, because they sit inside nn.Sequential and get positional names
# (decoder.conv1.net.1.weight) with no "norm" substring. Their biases WERE excluded,
# which is what made it hard to spot. Each BatchNorm gamma is a multiplicative gate
# on a whole decoder feature map, so decaying it attenuates signal rather than
# constraining capacity -- and doubling weight_decay doubled the damage.
print("\n8. optimiser param groups")
import torch.nn as nn
from model import SoilMoistureModel
from train import _split_param_groups, _NORM_TYPES

net = SoilMoistureModel(n_depths=3, d_model=768, n_heads=12, n_layers=6,
                        use_cls_depth=True)
decay, no_decay = _split_param_groups(net, net)

norm_ids = {id(p) for mod in net.modules() if isinstance(mod, _NORM_TYPES)
            for p in mod.parameters(recurse=False)}
decay_ids = {id(p) for p in decay}
leaked = norm_ids & decay_ids
check("no norm-layer param is decayed", not leaked, f"{len(leaked)} leaked")

n_norm_mods = sum(1 for m_ in net.modules() if isinstance(m_, _NORM_TYPES))
check("norm modules were actually found", n_norm_mods > 0, f"{n_norm_mods} modules")
check("BatchNorm2d present (the ones the old filter missed)",
      any(isinstance(m_, nn.BatchNorm2d) for m_ in net.modules()))

bias_leak = [n for n, p in net.named_parameters()
             if p.requires_grad and n.endswith("bias") and id(p) in decay_ids]
check("no bias is decayed", not bias_leak, str(bias_leak[:3]))

# MultiheadAttention's packed QKV bias is `self_attn.in_proj_bias` -- no dot before
# "bias", so a `.bias` suffix test silently starts decaying it. Named explicitly
# because it is the exact mirror of the BatchNorm bug this section exists for.
inproj = [n for n, p in net.named_parameters() if n.endswith("in_proj_bias")]
check("in_proj_bias exists in this model", len(inproj) > 0, f"{len(inproj)} found")
check("in_proj_bias not decayed",
      all(id(p) not in decay_ids for n, p in net.named_parameters()
          if n.endswith("in_proj_bias")))

# Nothing that was protected before may quietly lose protection.
old_nd = {n for n, p in net.named_parameters()
          if p.requires_grad and ("bias" in n or "norm" in n.lower())}
new_nd = {n for n, p in net.named_parameters()
          if p.requires_grad and id(p) not in decay_ids}
check("no regression vs old filter", not (old_nd - new_nd), str(sorted(old_nd - new_nd)[:3]))

dt = [p for n, p in net.named_parameters() if n.endswith("depth_tokens")]
check("depth_tokens excluded from decay", dt and id(dt[0]) not in decay_ids)

total = sum(1 for p in net.parameters() if p.requires_grad)
check("groups partition all trainable params", len(decay) + len(no_decay) == total,
      f"{len(decay)}+{len(no_decay)}=={total}")
check("no param in both groups", not (decay_ids & {id(p) for p in no_decay}))

# The old name-based rule, kept here purely to prove the bug was real.
old_decay = [n for n, p in net.named_parameters()
             if p.requires_grad and "bias" not in n and "norm" not in n.lower()]
old_leak = {n for n, p in net.named_parameters() if n in set(old_decay)
            and id(p) in norm_ids}
check("old name-based filter demonstrably leaked", len(old_leak) > 0,
      f"{len(old_leak)} norm scales, e.g. {sorted(old_leak)[:2]}")


print("\n" + "=" * 66)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("All per-depth loss tests passed.")
