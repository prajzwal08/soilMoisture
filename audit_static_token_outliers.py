"""§27a.3 — track down the DEM/LULC tokens that carry an absurd share of a tile's variance.

`plot_token_pca.py` found, on ISMN_TxSON_CR200-18:
    dem  token (r10,c2)  ‖e‖ = 1682.2  vs median 123.1   -> 64.8% of within-tile variance
    lulc token (r8,c3)   ‖e‖ =  999.1  vs median 124.6   -> 35.7%

One token holding two thirds of a tile's DEM embedding variance is almost certainly a
nodata/fill patch rather than terrain. This script answers four questions:

  1. What is in the RAW raster under that token's 16x16 px footprint? (nodata? constant?
     out of range?)
  2. Does `dem_token_mask` / `lulc_token_mask` already flag it? If it does, the pipeline
     handles it and only the figure was wrong. If it does not, everything downstream that
     consumes these embeddings is contaminated.
  3. Is it one tile or systematic? Sweep every station: max/median norm ratio, argmax
     position, and whether the position is repeated across stations.
  4. Is the outlier at the patch border (a padding/resampling artefact) or interior
     (real ground)?

CPU only, Pool(64), no GPU, never instantiates SoilMoistureDataset.

    python audit_static_token_outliers.py [--tile ISMN_TxSON_CR200-18] [--workers 64]
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

REPO      = Path(__file__).resolve().parent
TOK_ZARR  = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ZARR  = Path("/projects/prjs1968/satellite_zarr")
SPLITS    = REPO / "csvs" / "station_splits.csv"
OUT_JSON  = REPO / "csvs" / "static_token_outliers.json"
OUT_CSV   = REPO / "csvs" / "static_token_outliers.csv"

GRID, TOKEN_PX, PATCH_PX = 14, 16, 224
CATEGORIES = ("sm_only", "sm_and_flux", "flux_only")
MODS = ("dem", "lulc")


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


def _footprint(arr: np.ndarray, r: int, c: int) -> np.ndarray:
    return arr[r * TOKEN_PX:(r + 1) * TOKEN_PX, c * TOKEN_PX:(c + 1) * TOKEN_PX]


def audit_one(task) -> dict:
    """-> one record per (station, modality)."""
    name, cat = task
    out = {"station": name}
    try:
        zg = zarr.open_consolidated(str(TOK_ZARR / cat / name))
    except Exception as e:                                    # noqa: BLE001
        return {"station": name, "drop": f"tokens: {type(e).__name__}"}

    try:
        raw = zarr.open_group(str(SAT_ZARR / f"{name}.zarr"), mode="r")
    except Exception:                                          # noqa: BLE001
        raw = None

    for mod in MODS:
        p = f"{mod}_"
        if mod not in zg:
            out[p + "present"] = False
            continue
        t = np.asarray(zg[mod][:], np.float32)                 # (196, 768)
        nn = np.linalg.norm(t, axis=-1)
        med = float(np.median(nn))
        k = int(np.argmax(nn))
        r, c = divmod(k, GRID)

        mk = f"{mod}_token_mask"
        mask = (np.asarray(zg[mk][:], bool).reshape(-1) if mk in zg
                else np.ones(GRID * GRID, bool))

        # share of within-tile variance held by the top token, over VALID tokens only
        valid = mask if mask.any() else np.ones(GRID * GRID, bool)
        xc = t - t[valid].mean(0)
        den = float((xc[valid] ** 2).sum())
        share = float((xc[k] ** 2).sum() / den) if den > 0 else float("nan")

        out.update({
            p + "present": True,
            p + "max_norm": float(nn[k]),
            p + "median_norm": med,
            p + "ratio": float(nn[k] / med) if med > 0 else float("nan"),
            p + "argmax_r": r, p + "argmax_c": c, p + "argmax_i": k,
            p + "var_share": share,
            p + "masked_out": bool(not mask[k]),               # does the pipeline flag it?
            p + "n_masked": int((~mask).sum()),
            p + "border": bool(r in (0, GRID - 1) or c in (0, GRID - 1)),
        })

        # what is actually in the raw raster under that token?
        if raw is not None and f"{mod}/data" in raw:
            try:
                plane = (np.asarray(raw["dem/data"][0], np.float32) if mod == "dem"
                         else np.asarray(raw["lulc/data"][-1], np.float32))
                fp = _footprint(plane, r, c)
                rest = np.delete(plane.reshape(-1), None)
                out.update({
                    p + "fp_min": float(np.nanmin(fp)),
                    p + "fp_max": float(np.nanmax(fp)),
                    p + "fp_mean": float(np.nanmean(fp)),
                    p + "fp_nan": int(np.isnan(fp).sum()),
                    p + "fp_nuniq": int(len(np.unique(fp[~np.isnan(fp)]))),
                    p + "tile_min": float(np.nanmin(plane)),
                    p + "tile_max": float(np.nanmax(plane)),
                    p + "tile_nan": int(np.isnan(rest).sum()),
                })
            except Exception as e:                             # noqa: BLE001
                out[p + "raw_err"] = type(e).__name__
    return out


def deep_dive(name: str, cat: str) -> None:
    """Print the raw raster under the top token, pixel by pixel."""
    print(f"\n{'='*78}\nDEEP DIVE  {name}\n{'='*78}")
    zg = zarr.open_consolidated(str(TOK_ZARR / cat / name))
    raw = zarr.open_group(str(SAT_ZARR / f"{name}.zarr"), mode="r")

    for mod in MODS:
        if mod not in zg:
            continue
        t = np.asarray(zg[mod][:], np.float32)
        nn = np.linalg.norm(t, axis=-1)
        k = int(np.argmax(nn))
        r, c = divmod(k, GRID)
        mk = f"{mod}_token_mask"
        mask = (np.asarray(zg[mk][:], bool).reshape(-1) if mk in zg
                else np.ones(GRID * GRID, bool))

        print(f"\n--- {mod.upper()}  top token (r{r},c{c}) idx={k} ---")
        print(f"  ‖e‖ = {nn[k]:.1f}   median = {np.median(nn):.1f}   "
              f"ratio = {nn[k]/np.median(nn):.1f}x")
        print(f"  token_mask says valid = {bool(mask[k])}   "
              f"({int((~mask).sum())} of 196 masked out in this tile)")
        print(f"  embedding: min={t[k].min():+.2f} max={t[k].max():+.2f} "
              f"mean={t[k].mean():+.3f} n_nonfinite={int((~np.isfinite(t[k])).sum())}")

        plane = (np.asarray(raw["dem/data"][0], np.float32) if mod == "dem"
                 else np.asarray(raw["lulc/data"][-1], np.float32))
        fp = _footprint(plane, r, c)
        print(f"  raw raster over the whole tile: min={np.nanmin(plane):.2f} "
              f"max={np.nanmax(plane):.2f} nan={int(np.isnan(plane).sum())}")
        print(f"  raw raster in this 16x16 footprint [{r*16}:{r*16+16}, "
              f"{c*16}:{c*16+16}]:")
        print(f"    min={np.nanmin(fp):.2f} max={np.nanmax(fp):.2f} "
              f"mean={np.nanmean(fp):.2f} nan={int(np.isnan(fp).sum())} "
              f"unique={len(np.unique(fp[~np.isnan(fp)]))}")
        with np.printoptions(precision=1, suppress=True, linewidth=200):
            print(fp)

        # for comparison, the second-highest token
        k2 = int(np.argsort(nn)[-2])
        r2, c2 = divmod(k2, GRID)
        fp2 = _footprint(plane, r2, c2)
        print(f"  [compare] 2nd-highest token (r{r2},c{c2}) ‖e‖={nn[k2]:.1f}: "
              f"raw min={np.nanmin(fp2):.2f} max={np.nanmax(fp2):.2f} "
              f"nan={int(np.isnan(fp2).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="ISMN_TxSON_CR200-18")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    disk = disk_index()
    df = pd.read_csv(SPLITS).dropna(subset=["source_network", "network", "station_id"])
    names = [n for n in (station_dir(r) for _, r in df.iterrows()) if n in disk]
    if args.limit:
        names = names[:args.limit]
    print(f"{len(names)} stations on disk")

    if args.tile in disk:
        deep_dive(args.tile, disk[args.tile])
    else:
        print(f"[warn] {args.tile} not on disk — skipping deep dive")

    print(f"\n{'='*78}\nSWEEP over {len(names)} stations\n{'='*78}")
    with Pool(args.workers) as pool:
        recs = pool.map(audit_one, [(n, disk[n]) for n in names], chunksize=4)
    res = pd.DataFrame([r for r in recs if "drop" not in r])
    dropped = [r for r in recs if "drop" in r]
    if dropped:
        print(f"[warn] {len(dropped)} stations dropped")

    for mod in MODS:
        p = f"{mod}_"
        if p + "ratio" not in res:
            continue
        s = res[res[p + "present"] == True]                     # noqa: E712
        if s.empty:
            continue
        print(f"\n--- {mod.upper()}  n={len(s)} stations ---")
        print(f"  max/median norm ratio: median={s[p+'ratio'].median():.2f}  "
              f"p90={s[p+'ratio'].quantile(.9):.2f}  max={s[p+'ratio'].max():.1f}")
        big = s[s[p + "ratio"] > 3]
        print(f"  stations with ratio > 3x : {len(big)} of {len(s)} "
              f"({100*len(big)/len(s):.1f}%)")
        print(f"  of those, top token already masked out by the pipeline: "
              f"{int(big[p+'masked_out'].sum())}")
        print(f"  of those, top token on the patch border: "
              f"{int(big[p+'border'].sum())} ({100*big[p+'border'].mean():.0f}%)")
        print(f"  variance share of top token: median={s[p+'var_share'].median():.3f}  "
              f"p90={s[p+'var_share'].quantile(.9):.3f}  max={s[p+'var_share'].max():.3f}")
        vc = s[p + "argmax_i"].value_counts().head(5)
        print(f"  most common argmax token index:\n{vc.to_string()}")
        if p + "fp_nan" in s:
            print(f"  raw footprint NaNs under the top token: "
                  f"{int((s[p+'fp_nan'] > 0).sum())} stations have >0")
            print(f"  raw footprint constant (1 unique value): "
                  f"{int((s[p+'fp_nuniq'] == 1).sum())} stations")

    res.to_csv(OUT_CSV, index=False)
    summary = {
        "n_stations": len(res),
        "n_dropped": len(dropped),
        **{f"{m}_ratio_median": float(res[f"{m}_ratio"].median())
           for m in MODS if f"{m}_ratio" in res},
        **{f"{m}_frac_ratio_gt3": float((res[f"{m}_ratio"] > 3).mean())
           for m in MODS if f"{m}_ratio" in res},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_CSV.name} and {OUT_JSON.name}")


if __name__ == "__main__":
    main()
