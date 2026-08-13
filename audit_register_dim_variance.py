"""§27a.5 — is the register direction a constant offset (pure dilution) or does it carry signal?

§27a.4 found that coordinate 328 spikes in 100% of affected stations for BOTH dem and lulc,
that pooling preserves it, and that after the trained LayerNorm it holds a median 72% of the
DEM vector's squared norm (14x14 window). That measures MAGNITUDE. It does not say whether
the coordinate varies from station to station.

    if dim 328 is near-constant across stations -> it is pure DILUTION. It eats 72% of the
    post-normalisation magnitude while carrying no information that distinguishes one place
    from another, so the discriminative part is squeezed into the remaining ~28%.

    if it varies -> part of what looks like an artefact is real between-site signal, and
    removing it would throw information away.

The decisive test is not the variance of the coordinate but its effect on DISTINGUISHABILITY:

    median cosine similarity between station pairs, post-LayerNorm, computed twice --
    once with every dimension, once with the register dims zeroed. If the register is
    dilution, dropping it makes stations markedly LESS similar to each other, i.e. more
    separable. That is the quantity the model cares about.

CPU only, Pool(64). Never instantiates SoilMoistureDataset.

    python audit_register_dim_variance.py [--workers 64]
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings("ignore")

REPO     = Path(__file__).resolve().parent
TOK_ZARR = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS   = REPO / "csvs" / "station_splits.csv"
CKPT     = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/"
                "phase1_sm_only/cls_depth_star_reg/best.pt")
OUT_JSON = REPO / "csvs" / "register_dim_variance.json"

GRID, D = 14, 768
CATEGORIES = ("sm_only", "sm_and_flux", "flux_only")
MODS = ("dem", "lulc")
WIDTHS = [max(1, GRID * (i + 1) // 8) for i in range(4)]
WIN_PX = [2 * w for w in WIDTHS]
REG_FACTOR = 10.0          # |mean across stations| this many times the median -> register

# NOTE: this detects a DIFFERENT thing from audit_layernorm_compression.py.
#   there  -> dims that spike WITHIN a token (z-score against that token's own spread);
#             found dim 328, the attention-sink coordinate of the extreme token.
#   here   -> dims with a large MEAN ACROSS STATIONS, i.e. a constant offset present in
#             every token of every tile; found dims 87 and 126.
# Both are massive-activation directions; only the second can act as pure dilution, because
# a coordinate that is the same everywhere cannot separate one station from another.

_LN = {}


def station_dir(row: pd.Series) -> str:
    if str(row["source_network"]) == "ISMN":
        return f"ISMN_{row['network']}_{row['station_name']}"
    return f"{row['source_network']}_{row['station_id']}"


def disk_index() -> dict[str, str]:
    idx = {}
    for cat in CATEGORIES:
        d = TOK_ZARR / cat
        if d.is_dir():
            for n in d.iterdir():
                idx.setdefault(n.name, cat)
    return idx


def pyramid_pool(tok: np.ndarray, valid: np.ndarray) -> np.ndarray:
    g = tok.astype(np.float32).reshape(GRID, GRID, -1)
    v = valid.astype(np.float32).reshape(GRID, GRID, 1)
    half = GRID // 2
    out = []
    for w in WIDTHS:
        rs, re = half - w, half + w
        rg, rv = g[rs:re, rs:re, :], v[rs:re, rs:re, :]
        out.append((rg * rv).sum((0, 1)) / np.clip(rv.sum((0, 1)), 1, None))
    return np.stack(out)


def layer_norm(x: np.ndarray, gamma, beta, eps: float = 1e-5) -> np.ndarray:
    """nn.LayerNorm over the last axis, row-wise for a (N, 768) block."""
    mu = x.mean(-1, keepdims=True)
    sd = np.sqrt(x.var(-1, keepdims=True) + eps)
    y = (x - mu) / sd
    if gamma is not None:
        y = y * gamma + (beta if beta is not None else 0.0)
    return y


def load_pooled(task):
    name, cat = task
    try:
        zg = zarr.open_consolidated(str(TOK_ZARR / cat / name))
    except Exception:                                       # noqa: BLE001
        return None
    out = {"station": name}
    for mod in MODS:
        if mod not in zg:
            continue
        t = np.asarray(zg[mod][:], np.float32)
        mk = f"{mod}_token_mask"
        v = (np.asarray(zg[mk][:], bool).reshape(-1) if mk in zg
             else np.ones(GRID * GRID, bool))
        if not v.any():
            v = np.ones(GRID * GRID, bool)
        out[mod] = pyramid_pool(t, v)                       # (4, 768)
    return out


def median_pairwise_cos(x: np.ndarray, n_max: int = 600, seed: int = 0) -> float:
    """Median cosine between rows, on a random subsample to bound the pair count."""
    rng = np.random.default_rng(seed)
    if len(x) > n_max:
        x = x[rng.choice(len(x), n_max, replace=False)]
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    c = xn @ xn.T
    iu = np.triu_indices(len(c), k=1)
    return float(np.median(c[iu]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()

    try:
        import torch
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        k = next(k for k in sorted(sd) if k.endswith("transformer_layers.0.layer.norm1.weight"))
        _LN["gamma"] = sd[k].float().numpy()
        _LN["beta"] = sd[k.replace("weight", "bias")].float().numpy()
        print(f"trained LayerNorm from {k}")
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] no checkpoint LayerNorm ({type(e).__name__}) — identity gamma/beta")
        _LN["gamma"] = _LN["beta"] = None

    disk = disk_index()
    df = pd.read_csv(SPLITS).dropna(subset=["source_network", "network", "station_id"])
    names = [n for n in (station_dir(r) for _, r in df.iterrows()) if n in disk]
    print(f"{len(names)} stations\n")

    with Pool(args.workers) as pool:
        recs = [r for r in pool.map(load_pooled, [(n, disk[n]) for n in names], chunksize=4)
                if r is not None]

    summary = {}
    for mod in MODS:
        have = [r for r in recs if mod in r]
        if not have:
            continue
        P = np.stack([r[mod] for r in have])                 # (S, 4, 768)
        S = len(have)
        print(f"{'='*76}\n{mod.upper()}   {S} stations\n{'='*76}")

        mod_sum = {"n_stations": S}
        for wi, side in enumerate(WIN_PX):
            X = P[:, wi, :]                                  # (S, 768)
            m = X.mean(0)
            reg = np.where(np.abs(m) > REG_FACTOR * np.median(np.abs(m)))[0]
            if reg.size == 0:
                reg = np.array([int(np.argmax(np.abs(m)))])
            other = np.setdiff1d(np.arange(D), reg)

            v = X.var(0)
            cv = float(X[:, reg[0]].std() / max(abs(X[:, reg[0]].mean()), 1e-12))
            cv_other = float(np.median(np.sqrt(v[other]) /
                                       np.maximum(np.abs(m[other]), 1e-12)))
            var_share = float(v[reg].sum() / max(v.sum(), 1e-12))
            mag_share = float((m[reg] ** 2).sum() / max((m ** 2).sum(), 1e-12))

            Y = layer_norm(X, _LN["gamma"], _LN["beta"])
            Yz = Y.copy()
            Yz[:, reg] = 0.0
            cos_all = median_pairwise_cos(Y)
            cos_zero = median_pairwise_cos(Yz)

            if wi == 0:
                print(f"  register dims detected: {reg.tolist()[:8]}"
                      f"{' …' if reg.size > 8 else ''}  (n={reg.size})")
                print(f"\n  {'window':>9} {f'CV(dim{reg[0]})':>11} {'CV(other)':>10} "
                      f"{'var share':>10} {'mag share':>10} "
                      f"{'cos all':>9} {'cos w/o reg':>12}")
            print(f"  {side:>3}x{side:<5} {cv:>11.4f} {cv_other:>10.4f} "
                  f"{var_share:>10.4f} {mag_share:>10.4f} "
                  f"{cos_all:>9.3f} {cos_zero:>12.3f}")

            mod_sum[f"p{side}"] = {
                "register_dims": reg.tolist(), "cv_top_register": cv,
                "cv_other_median": cv_other, "across_station_var_share": var_share,
                "mean_magnitude_share": mag_share,
                "median_cos_postLN_all": cos_all,
                "median_cos_postLN_without_register": cos_zero,
            }
        summary[mod] = mod_sum

        w = mod_sum[f"p{WIN_PX[-1]}"]
        print(f"\n  VERDICT ({WIN_PX[-1]}x{WIN_PX[-1]} window):")
        print(f"    dim {w['register_dims'][0]} carries {100*w['mean_magnitude_share']:.1f}% "
              f"of the mean vector's magnitude but only "
              f"{100*w['across_station_var_share']:.1f}% of the across-station variance")
        print(f"    its coefficient of variation is {w['cv_top_register']:.4f} vs "
              f"{w['cv_other_median']:.4f} for a typical dim "
              f"({w['cv_other_median']/max(w['cv_top_register'],1e-12):.0f}x less variable)")
        print(f"    median pairwise cosine post-LayerNorm: "
              f"{w['median_cos_postLN_all']:.3f} -> "
              f"{w['median_cos_postLN_without_register']:.3f} with the register removed\n")

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
