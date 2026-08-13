"""§27a.4 — do the attention-sink coordinates survive pooling and dominate the LayerNorm?

§27a.3 established that 97.7% of stations have a DEM token whose norm is a median 13x the
tile median, carrying a median 66.7% of the tile's within-tile variance, and that the spike
lives in a handful of COORDINATES (CR200-18: one coord at -1671 against a mean of +1.76).
The raw DEM raster under it is pristine (452.88-454.55 m, 254 unique values, no NaN), so it
is a ViT massive-activation / attention-sink register, not terrain.

Whether that HARMS the model turns on one thing:

    LayerNorm is applied PER TOKEN, over the 768 features of that token. So a sink token
    cannot damage its neighbours -- it only wrecks itself, and there are ~1 of them per
    196. Harmless on its own.

    BUT DEM and LULC are POOLED before the transformer (`_cpu_pyramid_pool`,
    dataset.py:190-220). The pooled vector is a mean over tokens, so it inherits
    sink_value / n_tokens_in_window. Then `TransformerEncoderLayer(norm_first=True)`
    (model.py:405-420) applies LayerNorm to THAT. If the inherited spike still dominates
    the variance, LayerNorm divides every informative coordinate by a hugely inflated
    sigma and the real content is compressed toward zero.

This script measures the compression directly, on the raw tokens and on the four pooled
vectors, using the TRAINED norm1 gamma/beta from the checkpoint rather than an idealised
LayerNorm.

Reported per modality:
    sink_dims          which coordinates spike, and whether the same index recurs across
                       stations (a shared register direction) or is station-specific
    ratio_pooled       |pooled value at the sink dim| / std of the pooled non-sink dims
    compression        std(all dims) / std(non-sink dims)  -- the factor by which LayerNorm
                       shrinks the informative coordinates because of the spike
    post_ln_share      fraction of the post-LayerNorm squared norm sitting in the sink dims
                       -- i.e. how much of what the transformer reads is register, not data

CPU only, Pool(64). Never instantiates SoilMoistureDataset.

    python audit_layernorm_compression.py [--workers 64] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
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
OUT_CSV  = REPO / "csvs" / "layernorm_compression.csv"
OUT_JSON = REPO / "csvs" / "layernorm_compression.json"

GRID, D = 14, 768
CATEGORIES = ("sm_only", "sm_and_flux", "flux_only")
MODS = ("dem", "lulc")
SINK_TOKEN_RATIO = 5.0      # token norm above this multiple of the tile median
SINK_DIM_Z       = 8.0      # coordinate this many robust sigmas from the token median
WIDTHS = [max(1, GRID * (i + 1) // 8) for i in range(4)]        # [1,3,5,7]
WIN_PX = [2 * w for w in WIDTHS]                                # 2,6,10,14 tokens a side

_LN = {}                    # filled in main(), inherited by workers via fork


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


def robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


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


def layer_norm(x: np.ndarray, gamma: np.ndarray | None,
               beta: np.ndarray | None, eps: float = 1e-5) -> np.ndarray:
    """Exactly what nn.LayerNorm(768) does to one vector."""
    mu = x.mean()
    sd = np.sqrt(x.var() + eps)
    y = (x - mu) / sd
    if gamma is not None:
        y = y * gamma + (beta if beta is not None else 0.0)
    return y


def audit_one(task) -> list[dict]:
    name, cat = task
    try:
        zg = zarr.open_consolidated(str(TOK_ZARR / cat / name))
    except Exception:                                          # noqa: BLE001
        return []

    gamma, beta = _LN.get("gamma"), _LN.get("beta")
    recs = []
    for mod in MODS:
        if mod not in zg:
            continue
        t = np.asarray(zg[mod][:], np.float32)                  # (196, 768)
        mk = f"{mod}_token_mask"
        valid = (np.asarray(zg[mk][:], bool).reshape(-1) if mk in zg
                 else np.ones(GRID * GRID, bool))
        if not valid.any():
            valid = np.ones(GRID * GRID, bool)

        nn_ = np.linalg.norm(t, axis=-1)
        med = float(np.median(nn_[valid]))
        sink_tok = np.where(nn_ > SINK_TOKEN_RATIO * med)[0] if med > 0 else np.array([], int)
        if sink_tok.size == 0:
            recs.append({"station": name, "mod": mod, "has_sink": False})
            continue

        # which COORDINATES spike, pooled over all sink tokens in this tile
        dims = set()
        for k in sink_tok:
            x = t[k]
            s = robust_sigma(x)
            if s > 0:
                dims.update(np.where(np.abs(x - np.median(x)) > SINK_DIM_Z * s)[0].tolist())
        sink_dims = np.array(sorted(dims), int)
        if sink_dims.size == 0:
            recs.append({"station": name, "mod": mod, "has_sink": False})
            continue
        other = np.setdiff1d(np.arange(D), sink_dims)

        rec = {
            "station": name, "mod": mod, "has_sink": True,
            "n_sink_tok": int(sink_tok.size), "n_sink_dim": int(sink_dims.size),
            "top_sink_dim": int(sink_dims[np.argmax(np.abs(t[sink_tok[0]][sink_dims]))]),
            "tok_ratio": float(nn_[sink_tok[0]] / med),
        }

        # ── the sink token itself (unpooled path: only this token is affected) ──
        x = t[sink_tok[0]]
        rec["raw_compression"] = float(x.std() / max(x[other].std(), 1e-8))
        y = layer_norm(x, gamma, beta)
        rec["raw_post_ln_share"] = float((y[sink_dims] ** 2).sum() / max((y ** 2).sum(), 1e-12))

        # ── the pooled vectors (what the model actually receives for DEM/LULC) ──
        pooled = pyramid_pool(t, valid)                          # (4, 768)
        for i, side in enumerate(WIN_PX):
            p = pooled[i]
            sd_other = max(float(p[other].std()), 1e-8)
            rec[f"p{side}_ratio"] = float(np.abs(p[sink_dims]).max() / sd_other)
            rec[f"p{side}_compression"] = float(p.std() / sd_other)
            yp = layer_norm(p, gamma, beta)
            rec[f"p{side}_post_ln_share"] = float(
                (yp[sink_dims] ** 2).sum() / max((yp ** 2).sum(), 1e-12))
        recs.append(rec)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()

    # the LayerNorm the tokens actually hit first: transformer_layers[0].norm1
    try:
        import torch
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        gk = [k for k in sd if k.endswith("layer.norm1.weight") or
              k.endswith("layers.0.layer.norm1.weight") or
              (".norm1.weight" in k and "transformer_layers.0" in k)]
        if gk:
            k = sorted(gk)[0]
            _LN["gamma"] = sd[k].float().numpy()
            _LN["beta"] = sd[k.replace("weight", "bias")].float().numpy()
            print(f"using trained LayerNorm from {k}  "
                  f"gamma: mean={_LN['gamma'].mean():.3f} sd={_LN['gamma'].std():.3f}")
        else:
            print("[warn] no norm1 weights found — using identity gamma/beta")
    except Exception as e:                                      # noqa: BLE001
        print(f"[warn] checkpoint unreadable ({type(e).__name__}) — identity gamma/beta")

    disk = disk_index()
    df = pd.read_csv(SPLITS).dropna(subset=["source_network", "network", "station_id"])
    names = [n for n in (station_dir(r) for _, r in df.iterrows()) if n in disk]
    if args.limit:
        names = names[:args.limit]
    print(f"{len(names)} stations\n")

    with Pool(args.workers) as pool:
        out = pool.map(audit_one, [(n, disk[n]) for n in names], chunksize=4)
    res = pd.DataFrame([r for chunk in out for r in chunk])
    res.to_csv(OUT_CSV, index=False)

    summary = {}
    for mod in MODS:
        s = res[(res["mod"] == mod) & (res["has_sink"] == True)]     # noqa: E712
        n_all = int((res["mod"] == mod).sum())
        if s.empty:
            continue
        print(f"{'='*74}\n{mod.upper()}   {len(s)} of {n_all} stations have a sink token\n{'='*74}")
        print(f"  sink coordinates per tile: median {s.n_sink_dim.median():.0f}  "
              f"max {s.n_sink_dim.max():.0f}")
        top = Counter(s.top_sink_dim).most_common(5)
        print(f"  most common sink COORDINATE index: {top}")
        print(f"    -> {'SHARED register direction' if top[0][1] > .3*len(s) else 'station-specific'}"
              f"  ({100*top[0][1]/len(s):.0f}% of stations share the top one)")

        print(f"\n  the sink TOKEN itself (unpooled path — 1 token in 196):")
        print(f"    compression std(all)/std(non-sink) : median {s.raw_compression.median():7.1f}x")
        print(f"    post-LayerNorm share in sink dims  : median {s.raw_post_ln_share.median():7.3f}")

        print(f"\n  the POOLED vectors (what the model receives for {mod.upper()}):")
        print(f"    {'window':>10} {'|sink|/sd_other':>16} {'compression':>13} "
              f"{'post-LN share':>15}")
        for side in WIN_PX:
            print(f"    {side:>4}x{side:<5} "
                  f"{s[f'p{side}_ratio'].median():>16.2f} "
                  f"{s[f'p{side}_compression'].median():>13.2f}x "
                  f"{s[f'p{side}_post_ln_share'].median():>15.3f}")
        summary[mod] = {
            "n_with_sink": len(s), "n_total": n_all,
            "median_n_sink_dim": float(s.n_sink_dim.median()),
            "top_sink_dim": int(top[0][0]), "top_sink_dim_frac": top[0][1] / len(s),
            "raw_compression_median": float(s.raw_compression.median()),
            "raw_post_ln_share_median": float(s.raw_post_ln_share.median()),
            **{f"p{side}_post_ln_share_median": float(s[f"p{side}_post_ln_share"].median())
               for side in WIN_PX},
            **{f"p{side}_compression_median": float(s[f"p{side}_compression"].median())
               for side in WIN_PX},
        }
        print()

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_CSV.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
