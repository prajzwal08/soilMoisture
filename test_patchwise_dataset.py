"""Dataset tests for the patchwise loader (§35.8, §35.22).

Covers the silent-wrongness bugs the plan names, none of which crash:
  * token_mask is (T,14,14) not (T,196)   -- wrong-axis indexing yields (T,K,14)
  * padded / NaN slots read back VALID    -- poisonous as a per-patch key mask
  * the empty-history branch broke collate for mixed batches
  * the narrow read must return the SAME tokens the wide read would have

Synthetic tensors only. Runs under slurm/verify_patchwise_refactor.sh, never on a login node.
"""
import numpy as np, torch
from torch.utils.data import default_collate
from dataset import (STATION_TOKEN, N_TOKENS, MAX_S2,
                     _finalise_history, _empty_history, _token_slice, _read_patch_tokens)

FAILED = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra else ''}")
    if not cond: FAILED.append(name)

print("=== unit: constants ===")
check("STATION_TOKEN == 105", STATION_TOKEN == 105, f"got {STATION_TOKEN}")
check("N_TOKENS == 196", N_TOKENS == 196)

print("=== unit: the narrow read returns what the wide read would have ===")
# THE correctness condition for §35.22's I/O change. If these ever diverge, every number the
# model produces is computed on the wrong patch and nothing anywhere crashes.
full = np.random.randn(5, N_TOKENS, 768).astype(np.float16)
for sel in (np.array([STATION_TOKEN]), np.arange(N_TOKENS), np.array([0, 105, 195])):
    tsl  = _token_slice(sel)
    got  = _read_patch_tokens(full, 3, tsl, sel)
    want = full[3][sel]
    check(f"narrow read matches wide, K={len(sel)}", np.array_equal(got, want))
check("contiguous selections use a slice", _token_slice(np.array([STATION_TOKEN])) is not None)
check("non-contiguous falls back", _token_slice(np.array([0, 105, 195])) is None)

print("=== unit: _finalise_history masks padded slots ===")
T, K = 8, 3
sel  = np.array([0, STATION_TOKEN, 195], dtype=np.int64)
# The loaders now hand it an ALREADY-NARROWED (T,K,768) buffer — it no longer slices.
l12  = torch.randn(T, K, 768, dtype=torch.float16)
tm   = torch.ones(T, 14, 14, dtype=torch.bool)      # as the loaders initialise it
doys = torch.zeros(T, dtype=torch.long); doys[:3] = 100   # only 3 real acquisitions
feat, d, va, rp, hv = _finalise_history(l12, tm, doys, torch.zeros(T, dtype=torch.long),
                                        training=False, token_sel=sel, dropout_p=0.0)
check("feat passes through as (T,K,768)", tuple(feat.shape) == (T, K, 768), str(tuple(feat.shape)))
check("hist_valid shape (T,K)", tuple(hv.shape) == (T, K), str(tuple(hv.shape)))
check("padded slots are INVALID", not hv[doys == 0].any(),
      f"{int(hv[doys == 0].sum())} padded slots marked valid")
check("real slots stay valid", bool(hv[:3].all()))

print("=== unit: wrong-axis regression ===")
# token_mask[:, sel] on a (T,14,14) tensor silently yields (T,K,14) -- assert we don't.
check("no (T,K,14) leak", hv.ndim == 2, f"ndim={hv.ndim}")

print("=== unit: _empty_history shape-matches the normal path ===")
ef, ed, ev, er, ehv = _empty_history(MAX_S2, sel)
check("empty feat matches", tuple(ef.shape) == (MAX_S2, K, 768), str(tuple(ef.shape)))
check("empty hist_valid matches", tuple(ehv.shape) == (MAX_S2, K))
check("empty all-invalid", not ehv.any())

print("=== unit: collate a mixed batch (real + empty) ===")
# A station with no in-window acquisition must collate against one that has some, or every
# batch mixing the two raises "stack expects each tensor to be equal size".
a = {"s2_hist": feat, "s2_hist_valid": hv}
b = {"s2_hist": _empty_history(T, sel)[0], "s2_hist_valid": _empty_history(T, sel)[4]}
try:
    batched = default_collate([a, b])
    check("mixed batch collates", tuple(batched["s2_hist"].shape) == (2, T, K, 768),
          str(tuple(batched["s2_hist"].shape)))
except Exception as e:
    check("mixed batch collates", False, f"{type(e).__name__}: {e}")

print()
print("FAILURES:", FAILED if FAILED else "none")
raise SystemExit(1 if FAILED else 0)
