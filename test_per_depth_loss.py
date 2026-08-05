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


print("\n" + "=" * 66)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
    raise SystemExit(1)
print("All per-depth loss tests passed.")
