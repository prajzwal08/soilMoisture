"""§73 exact group-Shapley values (SHAPformer-style) for the §48 single-transformer model.

Players are whole input groups. A player is "absent" when its inputs are set to exactly the
state the TRAINING dropout produces (dataset.py §53 era5 / coarse dropout, §60 static dropout),
so every coalition is in-distribution for a model trained with those dropouts (§64 retrain).
All 2^K coalitions are evaluated (no sampling, no background data), and

    phi = W @ f,     f: (2^K, B, n_depths) coalition outputs,  W: (K, 2^K) fixed weights,

with W[i, S] = +w(|S|-1) if i in S else -w(|S|),  w(s) = s! (K-1-s)! / K!.
Efficiency holds exactly: phi.sum(0) == f[all] - f[none].

Enumeration and weights follow SHAPformer (KIT-IAI, Nat. Commun. 2026,
github.com/KIT-IAI/SHAPformer, src/shap.py); the absent recipes are ours.
Runbook §73.7. Fixed context (never switched off): SIF, TWSA, fine S2/S1 imagery.
"""
from math import factorial

import torch

from model import FINE_DEM, LULC_PAD

PLAYERS = ("era5", "sat", "dem", "lulc", "soil")


# ── absent recipes: each returns NEW tensors for the keys it changes (never in place) ──────
# Each is a copy of the matching training-dropout branch in dataset.py __getitem__.

def _absent_era5(b):
    # dataset.py §53: era5.zero_(); era5_doys.zero_()  (doy 0 = key padding in model.py)
    return {"era5": torch.zeros_like(b["era5"]), "era5_doys": torch.zeros_like(b["era5_doys"])}


def _absent_sat(b):
    # dataset.py §53 coarse dropout: anchor -> select_anchor's not-found return, S2/S1 history 0
    return {
        "anchor_l12":     torch.zeros_like(b["anchor_l12"]),
        "anchor_rel_pos": torch.zeros_like(b["anchor_rel_pos"]),
        "anchor_orbit":   torch.zeros_like(b["anchor_orbit"]),
        "anchor_found":   torch.zeros_like(b["anchor_found"]),
        "s2_pyr":   torch.zeros_like(b["s2_pyr"]),   "s2_valid": torch.zeros_like(b["s2_valid"]),
        "s1_pyr":   torch.zeros_like(b["s1_pyr"]),   "s1_valid": torch.zeros_like(b["s1_valid"]),
    }


def _absent_dem(b, fine):
    # dataset.py §60: dem_pyr 0, dem_valid False, fine[FINE_DEM] = 0 (trunk + fine tied)
    fine[:, FINE_DEM] = 0
    return {"dem_pyr": torch.zeros_like(b["dem_pyr"]), "dem_valid": torch.zeros_like(b["dem_valid"])}


def _absent_lulc(b):
    # dataset.py §60: lulc_pyr 0, lulc_valid False, LULC map all LULC_PAD (fine path)
    return {"lulc_pyr": torch.zeros_like(b["lulc_pyr"]), "lulc_valid": torch.zeros_like(b["lulc_valid"]),
            "lulc": torch.full_like(b["lulc"], LULC_PAD)}


def _absent_soil(b):
    # dataset.py §60: soil_patch 0, soil_valid False
    return {"soil_patch": torch.zeros_like(b["soil_patch"]),
            "soil_valid": torch.zeros_like(b["soil_valid"])}


def apply_absent(batch: dict, present_mask: int, players=PLAYERS) -> dict:
    """Shallow copy of `batch` with every player whose bit is 0 in `present_mask` absent.

    Bit i of present_mask = players[i] present. Unchanged tensors are shared, never modified.
    """
    out = dict(batch)
    absent = [p for i, p in enumerate(players) if not (present_mask >> i) & 1]
    if not absent:
        return out
    fine = None
    for p in absent:
        if p == "era5":
            out.update(_absent_era5(batch))
        elif p == "sat":
            out.update(_absent_sat(batch))
        elif p == "dem":
            fine = batch["fine"].clone() if fine is None else fine
            out.update(_absent_dem(batch, fine))
        elif p == "lulc":
            out.update(_absent_lulc(batch))
        elif p == "soil":
            out.update(_absent_soil(batch))
        else:
            raise KeyError(f"unknown player {p!r}; have {players}")
    if fine is not None:
        out["fine"] = fine
    return out


# ── Shapley weights ─────────────────────────────────────────────────────────────────────

def shapley_weight_matrix(k: int) -> torch.Tensor:
    """W (k, 2^k), float64: phi = W @ f gives the exact Shapley values."""
    w = [factorial(s) * factorial(k - 1 - s) / factorial(k) for s in range(k)]
    W = torch.zeros(k, 2 ** k, dtype=torch.float64)
    for m in range(2 ** k):
        size = bin(m).count("1")
        for i in range(k):
            W[i, m] = w[size - 1] if (m >> i) & 1 else -w[size]
    return W


def self_test(k: int = 5) -> None:
    """Exactness checks on synthetic games with known Shapley values (CPU, no model)."""
    W = shapley_weight_matrix(k)
    full, none = 2 ** k - 1, 0
    # Additive game f(S) = sum_{i in S} a_i  ->  phi_i = a_i exactly.
    a = torch.arange(1, k + 1, dtype=torch.float64)
    f = torch.tensor([sum(a[i] for i in range(k) if (m >> i) & 1) for m in range(2 ** k)],
                     dtype=torch.float64)
    assert torch.allclose(W @ f, a), "additive game failed"
    # Pure pairwise interaction between players 0 and 1 -> split equally, others 0.
    f = torch.tensor([1.0 if (m & 1) and (m & 2) else 0.0 for m in range(2 ** k)], dtype=torch.float64)
    phi = W @ f
    assert torch.allclose(phi[:2], torch.tensor([0.5, 0.5], dtype=torch.float64)) and phi[2:].abs().max() < 1e-12
    # Efficiency on a random game.
    f = torch.randn(2 ** k, dtype=torch.float64)
    assert abs(float((W @ f).sum() - (f[full] - f[none]))) < 1e-10, "efficiency failed"


if __name__ == "__main__":
    self_test()
    print("shap_groups self-test: OK (additive, interaction split, efficiency)")
