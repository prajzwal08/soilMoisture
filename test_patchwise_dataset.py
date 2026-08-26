"""Stage 2a dataset tests (§35.8 step 4, verification §35.11).

Covers the two paths and the three silent-wrongness bugs the plan names:
  * token_mask is (T,14,14) not (T,196)   -- wrong-axis indexing
  * padded / NaN slots read back VALID    -- poisonous as a per-patch key mask
  * the empty-history branch broke collate for mixed batches
"""
import numpy as np, torch
from torch.utils.data import default_collate
from dataset import (SoilMoistureDataset, STATION_TOKEN, N_TOKENS,
                     MAX_S2, MAX_S1, _finalise_history, _empty_history)

FAILED = []
def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra else ''}")
    if not cond: FAILED.append(name)

print("=== unit: constants ===")
check("STATION_TOKEN == 105", STATION_TOKEN == 105, f"got {STATION_TOKEN}")
check("N_TOKENS == 196", N_TOKENS == 196)

print("=== unit: _finalise_history masks padded slots ===")
T, K = 8, 3
l12  = torch.randn(T, N_TOKENS, 768, dtype=torch.float16)
tm   = torch.ones(T, 14, 14, dtype=torch.bool)      # as the loaders initialise it
doys = torch.zeros(T, dtype=torch.long); doys[:3] = 100   # only 3 real acquisitions
sel  = np.array([0, STATION_TOKEN, 195], dtype=np.int64)
feat, d, va, rp, hv = _finalise_history(l12, tm, doys, torch.zeros(T, dtype=torch.long),
                                        training=False, token_sel=sel, dropout_p=0.0)
check("feat shape (T,K,768)", tuple(feat.shape) == (T, K, 768), str(tuple(feat.shape)))
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
pf, _, _, _, phv = _empty_history(MAX_S2, None)
check("pooled empty is (T,4,768)", tuple(pf.shape) == (MAX_S2, 4, 768))
check("pooled empty hist_valid None", phv is None)

print("=== unit: collate a mixed batch (real + empty) ===")
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
