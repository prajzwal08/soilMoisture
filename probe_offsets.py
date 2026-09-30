#!/usr/bin/env python
"""
Step-0 feasibility probe (user 2026-09-30): do TerraMind cell embeddings know which spots are
WETTER than their tile?  No deep model — two ridge regressions on per-station summaries.

  target   station 0-10 cm LEVEL (mean over observed days) and AMPLITUDE (SD over observed days)
  stage 1  target ~ TILE features   (ERA5 climatology, tile-mean soil, tile-mean embeddings)
  stage 2  stage-1 residual ~ CELL features  (cell embedding minus tile mean, soil at cell minus
           tile mean) — "is this spot wetter than its tile"

Fit on TRAIN own-centre rows only; alpha chosen by GroupKFold over networks inside train.

Scored three ways:
  A. val / oos stations: R2 of stage 1, and R2 of stage 2 on the stage-1 residual.
  B. WITHIN-TILE PAIRS, every tile that contains >= 2 labelled stations (all networks, not only
     TxSON). Tile features are identical for both stations of a pair, so only stage 2 predicts
     their difference.  Observed difference = mean over COMMON observed days of (sm_i - sm_j),
     >= MIN_COMMON days.  Reported: r(pred diff, obs diff), sign agreement, sd ratio; for all
     pairs and for "clean" pairs (neither station in train), with a tile-bootstrap 95% CI.
  C. zero-anomaly baseline is the null for B (r = 0, sign agreement 0.5).

Embeddings: frozen TerraMind L12, (196,768) per acquisition, averaged per 160 m cell over the
acquisitions where that cell is valid (S2: <=1% cloud/shadow/no-mask pixels, dataset.py's rule;
S1: stored token_mask; DEM/LULC: static token masks).  A station's cell vector is bilinear on the
14x14 grid at its pixel — the centre pixel 112 sits on the corner of four cells, so a centre
station is their average, not an arbitrary one of them.

Read-only.  Writes csvs/probe_offsets/{rows,pairs}.csv and prints the report.
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from pyproj import Transformer

import dataset as D
from splits_config import category_of, station_dir_name

SAT_ZARR    = Path("/projects/prjs1968/satellite_zarr")
OUT_DIR     = Path("csvs/probe_offsets")
RES_M       = 10
GRID, CELL  = 14, 16
MIN_COMMON  = 180          # days both stations of a pair observed
N_PCA       = 16           # components per embedding modality
MODS        = ("s2", "s1", "dem", "lulc")
SPLITS_KEEP = ("train", "val", "oos")


# ── geometry ─────────────────────────────────────────────────────────────────

def tile_geo(d: str):
    p = SAT_ZARR / f"{d}.zarr" / ".zattrs"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    return int(a["epsg"]), [float(v) for v in a["bounds_utm"]]


def pixel(lon, lat, epsg, bounds, _tr={}):
    tr = _tr.get(epsg) or _tr.setdefault(
        epsg, Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True))
    x, y = tr.transform(lon, lat)
    west, _, _, north = bounds
    return (north - y) / RES_M, (x - west) / RES_M          # float row, col


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


# ── per-tile features ────────────────────────────────────────────────────────

def _cell_grid(l12, tm):
    """l12 (N,196,768) f16 memmap/array, tm (N,14,14) bool -> (196,768) f32 mean, (196,) count."""
    tm = tm.reshape(len(tm), -1)
    s = np.zeros((196, 768), np.float64)
    c = tm.sum(0).astype(np.float64)
    for i0 in range(0, len(tm), 32):
        x = np.nan_to_num(np.asarray(l12[i0:i0 + 32], dtype=np.float32))
        s += np.einsum("nk,nkd->kd", tm[i0:i0 + 32].astype(np.float32), x)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (s / c[:, None]).astype(np.float32)
    g[c == 0] = np.nan
    return g, c


def _bilinear(g, row, col):
    """g (196,D) with NaN rows for invalid cells -> (D,) at pixel (row,col); NaN if no support."""
    G = g.reshape(GRID, GRID, -1)
    u = np.clip((row + 0.5) / CELL - 0.5, 0, GRID - 1)
    v = np.clip((col + 0.5) / CELL - 0.5, 0, GRID - 1)
    r0, c0 = int(np.floor(u)), int(np.floor(v))
    r1, c1 = min(r0 + 1, GRID - 1), min(c0 + 1, GRID - 1)
    fu, fv = u - r0, v - c0
    acc, wsum = 0.0, 0.0
    for r, c, w in ((r0, c0, (1 - fu) * (1 - fv)), (r0, c1, (1 - fu) * fv),
                    (r1, c0, fu * (1 - fv)), (r1, c1, fu * fv)):
        if w > 0 and np.isfinite(G[r, c, 0]):
            acc, wsum = acc + w * G[r, c], wsum + w
    return acc / wsum if wsum > 0 else np.full(G.shape[-1], np.nan, np.float32)


def tile_features(task):
    d, cat, positions = task["dir"], task["cat"], task["positions"]   # positions: [(key,row,col)]
    out = {"dir": d, "ok": False}
    zg = D._open_zarr(Path(d), cat)
    cache = D._load_station_cache(D.CACHE_ROOT / cat / d, fine=False)
    if zg is None or cache is None:
        out["why"] = "no store" if zg is None else "no cache"
        return out

    grids = {}
    # S2: dataset.py's per-cell cloud rule on the aligned cloud mask
    if "s2_l12" in cache and "s2_cm" in cache:
        grids["s2"] = _cell_grid(cache["s2_l12"], D._cm_token_mask(np.asarray(cache["s2_cm"])))[0]
    # S1: per-orbit grids from the stored token masks, then averaged where both exist
    s1 = []
    for orb in ("s1_asc", "s1_desc"):
        if f"{orb}_l12" in cache and f"{orb}/token_mask" in zg:
            s1.append(_cell_grid(cache[f"{orb}_l12"], np.asarray(zg[f"{orb}/token_mask"][:]).astype(bool))[0])
    if s1:
        grids["s1"] = np.nanmean(np.stack(s1), 0) if len(s1) > 1 else s1[0]
    for m in ("dem", "lulc"):
        if m in zg:
            g = np.asarray(zg[m][:], dtype=np.float32)
            if f"{m}_token_mask" in zg:
                g[~np.asarray(zg[f"{m}_token_mask"][:]).reshape(-1).astype(bool)] = np.nan
            g[~np.isfinite(g).all(1)] = np.nan
            grids[m] = g

    out["tile"] = {m: np.nanmean(g, 0) for m, g in grids.items() if np.isfinite(g[:, 0]).any()}
    out["cell"] = {k: {m: _bilinear(g, r, c) for m, g in grids.items()} for k, r, c in positions}

    soil = np.asarray(zg["soil"][:], dtype=np.float32) if "soil" in zg else None   # (21,74,74)
    if soil is not None:
        soil[~np.isfinite(soil)] = np.nan
        out["soil_tile"] = np.nanmean(soil.reshape(21, -1), 1)
        sc = soil.shape[-1] / 224.0
        out["soil_cell"] = {k: soil[:, min(int(r * sc), soil.shape[1] - 1),
                                        min(int(c * sc), soil.shape[2] - 1)]
                            for k, r, c in positions}
    era = D._load_zarr_era5(zg)
    out["era5"] = np.nanmean(era[0], 0) if era is not None else None

    lab = D._load_zarr_labels(zg)
    if lab is not None:
        sm, depths, times, qc = lab
        if "0-10" in depths:
            i = depths.index("0-10")
            y = sm[i].astype(np.float64)
            ok = np.isfinite(y) & ((qc[i] == 0) if qc is not None else True)
            out["series"] = pd.Series(np.where(ok, y, np.nan), index=times)
    out["ok"] = True
    return out


# ── regression ───────────────────────────────────────────────────────────────

def ridge_fit(X, y, groups):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    alphas = np.logspace(-2, 5, 29)
    ng = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, ng))
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        err = 0.0
        for tr, te in cv.split(X, y, groups):
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr])
            err += ((m.predict(X[te]) - y[te]) ** 2).sum()
        if err < best:
            best, best_a = err, a
    return make_pipeline(StandardScaler(), Ridge(alpha=best_a)).fit(X, y), best_a, best / len(y)


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(y) > 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tiles", type=int, default=None, help="smoke: first N tiles only")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv("csvs/station_splits.csv")
    sp = sp[sp["split"].isin(SPLITS_KEEP)].copy()
    sp["cat"] = sp.apply(category_of, axis=1)
    sp = sp[sp["cat"].isin(["sm_only", "sm_and_flux"])]
    sp["dir"] = sp.apply(station_dir_name, axis=1)
    sp = sp.drop_duplicates("dir").reset_index(drop=True)
    print(f"stations with SM in train/val/oos: {len(sp)}  "
          f"({sp['split'].value_counts().to_dict()})", flush=True)

    # Which stations fall inside which tile (own centre always; others if inside 224x224)
    geo = {d: tile_geo(d) for d in sp["dir"]}
    lat, lon = sp["latitude"].to_numpy(float), sp["longitude"].to_numpy(float)
    tasks, n_off = [], 0
    for i, r in sp.iterrows():
        g = geo[r["dir"]]
        if g is None:
            continue
        pos = [(r["dir"], 112.0, 112.0)]
        for j in range(len(sp)):
            if j == i or haversine_km(lat[i], lon[i], lat[j], lon[j]) > 1.7:
                continue
            rr, cc = pixel(lon[j], lat[j], *g)
            if 0 <= rr < 224 and 0 <= cc < 224:
                pos.append((sp.at[j, "dir"], rr, cc)); n_off += 1
        tasks.append({"dir": r["dir"], "cat": r["cat"], "positions": pos})
    if args.max_tiles:
        tasks = tasks[:args.max_tiles]
    print(f"tiles: {len(tasks)}  off-centre station readouts: {n_off}  "
          f"tiles with >=2 stations: {sum(len(t['positions']) > 1 for t in tasks)}", flush=True)

    with Pool(args.workers) as pool:
        res = pool.map(tile_features, tasks, chunksize=1)
    bad = [(r["dir"], r.get("why")) for r in res if not r["ok"]]
    print(f"tile features ok: {len(res) - len(bad)}  failed: {len(bad)} {bad[:5]}", flush=True)
    res = {r["dir"]: r for r in res if r["ok"]}
    series = {d: r["series"] for d, r in res.items() if "series" in r}
    meta = sp.set_index("dir")

    # One row per (tile, station-in-tile)
    rows = []
    for t in tasks:
        R = res.get(t["dir"])
        if R is None or R["era5"] is None or "soil_tile" not in R:
            continue
        for k, rr, cc in t["positions"]:
            if k not in series:
                continue
            y = series[k].dropna()
            if len(y) < 365:
                continue
            row = {"tile": t["dir"], "station": k, "centre": k == t["dir"], "row": rr, "col": cc,
                   "split": meta.at[k, "split"], "network": meta.at[k, "network"],
                   "level": float(y.mean()), "amp": float(y.std())}
            for m in MODS:
                tv = R["tile"].get(m)
                cv = R["cell"][k].get(m)
                row[f"T_{m}"] = tv
                row[f"C_{m}"] = (cv - tv) if (tv is not None and cv is not None) else None
            row["T_era5"] = R["era5"]
            row["T_soil"] = R["soil_tile"]
            row["C_soil"] = R["soil_cell"][k] - R["soil_tile"]
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"readout rows: {len(df)}  (own-centre {int(df['centre'].sum())})", flush=True)

    # Feature matrices: PCA per embedding modality, fit on TRAIN own-centre rows
    from sklearn.decomposition import PCA
    fit_mask = (df["centre"] & (df["split"] == "train")).to_numpy()

    def stack(col, dim):
        return np.stack([np.asarray(v, np.float32) if v is not None and np.isfinite(v).all()
                         else np.full(dim, np.nan, np.float32) for v in df[col]])

    def emb_block(col, n_pca):
        """n_pca 0 = all 768 dims, ridge does the shrinkage; else PCA fit on train rows."""
        X = stack(col, 768)
        ok = np.isfinite(X).all(1)
        if n_pca == 0:
            Z = np.where(ok[:, None], X, np.nanmean(X[fit_mask & ok], 0))   # missing -> train mean
            return Z.astype(np.float32), ok
        p = PCA(n_pca, random_state=0).fit(X[fit_mask & ok])
        Z = np.zeros((len(X), n_pca), np.float32)
        Z[ok] = p.transform(X[ok])                   # missing modality -> 0 = train mean
        return Z, ok

    groups = df["network"].to_numpy()
    # raw = main result (ridge alone shrinks weak directions softly); pca16 = comparison only
    for tag, n_pca in (("raw", 0), ("pca16", N_PCA)):
        T_blocks, C_blocks = [stack("T_era5", 18), stack("T_soil", 21)], [stack("C_soil", 21)]
        for m in MODS:
            Zt, okt = emb_block(f"T_{m}", n_pca)
            Zc, okc = emb_block(f"C_{m}", n_pca)
            T_blocks += [Zt, okt[:, None].astype(np.float32)]
            C_blocks += [Zc, okc[:, None].astype(np.float32)]
        XT = np.nan_to_num(np.concatenate(T_blocks, 1))
        XC = np.nan_to_num(np.concatenate(C_blocks, 1))
        print(f"\n################ VARIANT {tag}: features tile {XT.shape[1]}, cell {XC.shape[1]}; "
              f"train fit rows {fit_mask.sum()}", flush=True)

        for tgt in ("level", "amp"):
            y = df[tgt].to_numpy()
            m1, a1, cv1 = ridge_fit(XT[fit_mask], y[fit_mask], groups[fit_mask])
            p1 = m1.predict(XT)
            res1 = y - p1
            m2, a2, cv2 = ridge_fit(XC[fit_mask], res1[fit_mask], groups[fit_mask])
            p2 = m2.predict(XC)
            df[f"{tag}_{tgt}_p1"], df[f"{tag}_{tgt}_p2"] = p1, p2
            print(f"\n=== [{tag}] TARGET {tgt} (0-10 cm)   stage-1 alpha {a1:.3g}   stage-2 alpha {a2:.3g}")
            print(f"  train-CV  stage-1 R2 {1 - cv1 / y[fit_mask].var():.3f}   "
                  f"stage-2 R2 on residual {1 - cv2 / res1[fit_mask].var():.3f}")
            for s in ("val", "oos"):
                mk = (df["centre"] & (df["split"] == s)).to_numpy()
                print(f"  {s:4s} n={mk.sum():4d}  stage-1 R2 {r2(y[mk], p1[mk]):.3f}   "
                      f"stage-2 R2 on residual {r2(res1[mk], p2[mk]):.3f}   "
                      f"level SD {y[mk].std():.4f}  residual SD {res1[mk].std():.4f}")
    
            # B. within-tile pairs, both orderings merged; prediction averaged over the tiles
            pairs = {}
            for tile, g in df.groupby("tile"):
                if len(g) < 2:
                    continue
                recs = g.to_dict("records")
                for a in range(len(recs)):
                    for b in range(a + 1, len(recs)):
                        i, j = sorted((recs[a]["station"], recs[b]["station"]))
                        pi = recs[a] if recs[a]["station"] == i else recs[b]
                        pj = recs[b] if pi is recs[a] else recs[a]
                        pairs.setdefault((i, j), []).append((tile, pi[f"{tag}_{tgt}_p2"] - pj[f"{tag}_{tgt}_p2"]))
            prow = []
            for (i, j), preds in pairs.items():
                si, sj = series[i], series[j]
                both = pd.concat([si, sj], axis=1).dropna()
                if len(both) < MIN_COMMON:
                    continue
                obs = (both.iloc[:, 0].mean() - both.iloc[:, 1].mean() if tgt == "level"
                       else both.iloc[:, 0].std() - both.iloc[:, 1].std())
                prow.append({"a": i, "b": j, "tile": preds[0][0], "n_tiles": len(preds),
                             "n_common": len(both), "obs": obs, "pred": float(np.mean([p for _, p in preds])),
                             "clean": meta.at[i, "split"] != "train" and meta.at[j, "split"] != "train",
                             "network": meta.at[i, "network"]})
            P = pd.DataFrame(prow)
            P.to_csv(OUT_DIR / f"pairs_{tag}_{tgt}.csv", index=False)
            rng = np.random.default_rng(0)
            for lab, sub in (("all", P), ("clean", P[P["clean"]] if len(P) else P)):
                if len(sub) < 3:
                    print(f"  pairs {lab}: n={len(sub)} (too few)")
                    continue
                r = np.corrcoef(sub["obs"], sub["pred"])[0, 1]
                sign = (np.sign(sub["obs"]) == np.sign(sub["pred"])).mean()
                tiles = sub["tile"].unique()
                bs = []
                for _ in range(1000):
                    pick = rng.choice(tiles, len(tiles))
                    bb = pd.concat([sub[sub["tile"] == t] for t in pick])
                    if bb["obs"].std() > 0 and bb["pred"].std() > 0:
                        bs.append(np.corrcoef(bb["obs"], bb["pred"])[0, 1])
                lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
                print(f"  pairs {lab:5s}: n={len(sub):4d} over {len(tiles):3d} tiles, "
                      f"{sub['network'].nunique()} networks   r={r:+.3f} [95% tile-bootstrap {lo:+.3f},{hi:+.3f}]   "
                      f"sign agree {sign:.2f}   sd(pred)/sd(obs) {sub['pred'].std() / sub['obs'].std():.2f}   "
                      f"obs |diff| median {sub['obs'].abs().median():.4f}")
                if lab == "clean":
                    by = sub.groupby("network").apply(
                        lambda g: pd.Series({"n": len(g), "sign": (np.sign(g.obs) == np.sign(g.pred)).mean(),
                                             "r": np.corrcoef(g.obs, g.pred)[0, 1] if len(g) > 2 else np.nan}))
                    print("    by network (clean):\n" + by.sort_values("n", ascending=False).to_string())

    df.drop(columns=[c for c in df.columns if c.startswith(("T_", "C_"))]).to_csv(
        OUT_DIR / "rows.csv", index=False)
    print(f"\nwrote {OUT_DIR}/rows.csv, pairs_{raw,pca16}_{level,amp}.csv", flush=True)


if __name__ == "__main__":
    main()
