"""§27b Scale A part 1 — does the TerraMind embedding space organise itself by wetness?

Asked BEFORE any model is fitted. Take the token of the pixel each station stands on, look at
how the 993 tokens arrange themselves, and ask whether wet stations sit near other wet ones.
No regression, no regularisation choice, no p >> n concern: the structure is there or it is not.

Feature  : the station's OWN token. A station sits at pixel (112,112) of its own patch, which
           is token (7,7) = index 105 -- the 160 m cell it stands in. One 768-vector.
           Dynamic modalities use the multi-year MEAN of that token over acquisitions falling
           inside the station's own label window (§27b.3: yearly rows would be
           pseudo-replication for a per-station target, and an unrestricted window would
           average the embedding over a different period than the soil moisture).
Target   : the station's mean soil moisture, qc==0 days only.
Sweep    : S2 / S1_ASC / S1_DESC x L3 / L6 / L9 / L12, plus DEM / LULC at L12.

Statistics, all in the ORIGINAL 768-d space -- never on the 2-D projection:
  1. k-NN leave-one-out RMSE (cosine).  Non-parametric, essentially nothing fitted, and it
     lands on the same scale as the §20.14 ladder (null 0.0752, best tabular stack 0.0576).
  2. k-means cluster composition: eta^2 of station mean SM across clusters, with an exact
     permutation null.
  3. Neighbourhood purity: median |dSM| to the k-th embedding neighbour vs random pairs.

Every statistic is also computed on WITHIN-NETWORK RESIDUALS (SM minus its network mean,
tokens likewise). That separates "the embedding knows Norway is wet" from "the embedding knows
this field is wet" -- the size of the drop is the result.

Figure: UMAP and PCA of the tokens, coloured by mean SM and, alongside, by Koppen class. The
second panel is the confound made visible. UMAP is fitted UNSUPERVISED; the target is used only
for colour (`fit_transform(X, y=...)` would separate wet from dry by construction).

CPU only, Pool(64). Never instantiates SoilMoistureDataset (it eagerly loads ~16 GB of L12).

    python probe_token_sm_structure.py [--workers 64] [--max-dates 24] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO     = Path(__file__).resolve().parent
TOK_ZARR = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SPLITS   = REPO / "csvs" / "station_splits.csv"
OUT_JSON = REPO / "csvs" / "token_sm_structure.json"
OUT_CSV  = REPO / "csvs" / "token_sm_structure.csv"
FIG_DIR  = REPO / "figures" / "token_sm"

GRID       = 14
CENTRE_TOK = (GRID // 2) * GRID + GRID // 2          # (7,7) -> 105
CATEGORIES = ("sm_only", "sm_and_flux", "flux_only")
LAYERS     = ("l3", "l6", "l9", "l12")
DYNAMIC    = ("s2", "s1_asc", "s1_desc")
STATIC     = ("dem", "lulc")
DEPTHS     = ("0-10", "10-30", "30-100")
KNN_KS     = (1, 5, 10, 20, 50)
KMEANS_KS  = (5, 10, 20, 50)
N_PERM     = 2000
SEED       = 0

# §20.14 baselines for depth 0-10, so every number here is directly comparable
LADDER = {"null": 0.0752, "B1 soil": 0.0621, "B5 +derived": 0.0599,
          "B8 +smap_tb": 0.0576, "network RMS station bias": 0.0618}

assert CENTRE_TOK == 105, CENTRE_TOK


def station_dir(row: pd.Series) -> str:
    if str(row["source_network"]) == "ISMN":
        return f"ISMN_{row['network']}_{row['station_name']}"
    return f"{row['source_network']}_{row['station_id']}"


def disk_index(cats=CATEGORIES) -> dict[str, str]:
    idx = {}
    for cat in cats:
        d = TOK_ZARR / cat
        if d.is_dir():
            for n in d.iterdir():
                idx.setdefault(n.name, cat)
    return idx


def _dstr(a) -> np.ndarray:
    a = np.asarray(a)
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _dint(a) -> np.ndarray:
    """YYYYMMDD strings -> int64. numpy 2 cannot min()/compare <U8 arrays, and integer
    comparison is both safe and faster. Non-numeric entries become -1 and are dropped."""
    out = np.full(len(a), -1, np.int64)
    for i, x in enumerate(_dstr(a)):
        if len(x) == 8 and x.isdigit():
            out[i] = int(x)
    return out


def load_station(task):
    """-> dict with the station's own centre token per modality/layer + mean SM per depth."""
    name, cat, max_dates = task
    try:
        zg = zarr.open_consolidated(str(TOK_ZARR / cat / name))
    except Exception:                                          # noqa: BLE001
        return None

    # ── target: mean SM per depth, qc == 0 only ────────────────────────────
    if "labels/sm" not in zg:
        return None
    sm = zg["labels/sm"][:]
    depths = [str(d) for d in zg["labels/depths"][:]]
    dates = _dint(zg["labels/dates"][:])
    qc = zg["labels/qc"][:] if "labels/qc" in zg else None
    if qc is not None and qc.shape[1] != sm.shape[1]:
        qc = qc[:, -sm.shape[1]:]          # trim_pre2016 trimmed sm/dates but not qc
    y = {}
    for i, d in enumerate(depths):
        if d not in DEPTHS:
            continue
        v = sm[i].astype(np.float64)
        if qc is not None:
            v = np.where(qc[i] == 0, v, np.nan)
        if np.isfinite(v).sum() >= 365:
            y[d] = float(np.nanmean(v))
    if not y:
        return None

    good = dates[dates > 0]
    if good.size == 0:
        return None
    lo, hi = int(good.min()), int(good.max())      # the station's own label window

    out = {"station": name, "y": y, "X": {}}
    for mod in STATIC:
        if mod in zg:
            out["X"][f"{mod}/l12"] = np.asarray(zg[mod][CENTRE_TOK], np.float32)

    for mod in DYNAMIC:
        if f"{mod}/dates" not in zg:
            continue
        ad = _dint(zg[f"{mod}/dates"][:])
        keep = np.where((ad >= lo) & (ad <= hi))[0]            # label window only
        if keep.size == 0:
            continue
        if keep.size > max_dates:                              # evenly spaced subsample
            keep = keep[np.linspace(0, keep.size - 1, max_dates).astype(int)]
        for lay in LAYERS:
            key = f"{mod}/{lay}"
            if key not in zg:
                continue
            arr = zg[key]
            acc = np.zeros(arr.shape[-1], np.float64)
            for i in keep:
                acc += np.asarray(arr[int(i), CENTRE_TOK], np.float64)
            out["X"][key] = (acc / keep.size).astype(np.float32)
    return out if out["X"] else None


# ── statistics, all in the original 768-d space ──────────────────────────────

def knn_loo_rmse(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """Leave-one-out k-NN regression RMSE, cosine distance."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)                             # exclude self
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    pred = y[idx].mean(1)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def cluster_eta2(X: np.ndarray, y: np.ndarray, k: int, seed: int = SEED
                 ) -> tuple[float, float]:
    """eta^2 of y across k-means clusters, plus a permutation p-value."""
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)

    def _eta2(vals):
        gm = vals.mean()
        ssb = sum(((vals[lab == c].mean() - gm) ** 2) * (lab == c).sum()
                  for c in range(k) if (lab == c).any())
        sst = ((vals - gm) ** 2).sum()
        return float(ssb / sst) if sst > 0 else float("nan")

    obs = _eta2(y)
    rng = np.random.default_rng(seed)
    null = np.array([_eta2(rng.permutation(y)) for _ in range(N_PERM)])
    p = float((1 + (null >= obs).sum()) / (1 + N_PERM))
    return obs, p


def neighbour_purity(X: np.ndarray, y: np.ndarray, k: int, seed: int = SEED
                     ) -> tuple[float, float]:
    """Median |dy| to the k-th embedding neighbour, vs a random pair baseline."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    near = float(np.median(np.abs(y[:, None] - y[idx]).mean(1)))
    rng = np.random.default_rng(seed)
    rnd = float(np.median(np.abs(y[:, None] - y[rng.integers(0, len(y), (len(y), k))]).mean(1)))
    return near, rnd


def residualise(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Subtract the per-network mean from both features and target."""
    Xr, yr = X.copy(), y.copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2:
            Xr[m] -= Xr[m].mean(0)
            yr[m] -= yr[m].mean()
        else:
            Xr[m] = np.nan                 # singleton networks carry no within-info
            yr[m] = np.nan
    ok = np.isfinite(yr) & np.isfinite(Xr).all(1)
    Xr, yr = Xr[ok], yr[ok]
    # a dimension that is constant within every network becomes exactly 0 everywhere;
    # StandardScaler leaves it at 0, which is fine, but guard against all-zero rows
    keep_dim = Xr.std(0) > 0
    return (Xr[:, keep_dim] if keep_dim.any() else Xr), yr, ok


def run_block(X: np.ndarray, y: np.ndarray) -> dict:
    Xs = StandardScaler().fit_transform(X)
    res = {"n": int(len(y)), "sd_y": float(y.std()),
           "knn_rmse": {}, "eta2": {}, "purity": {}}
    for k in KNN_KS:
        if k < len(y):
            res["knn_rmse"][str(k)] = knn_loo_rmse(Xs, y, k)
    for k in KMEANS_KS:
        if k < len(y):
            e, p = cluster_eta2(Xs, y, k)
            res["eta2"][str(k)] = {"eta2": e, "p_perm": p}
    near, rnd = neighbour_purity(Xs, y, 10)
    res["purity"] = {"k10_near": near, "k10_random": rnd,
                     "ratio": float(near / rnd) if rnd > 0 else float("nan")}
    return res


# ── figure ───────────────────────────────────────────────────────────────────

def make_figure(X: np.ndarray, y: np.ndarray, kg: np.ndarray, key: str,
                depth: str, out: Path) -> None:
    Xs = StandardScaler().fit_transform(X)
    proj = {"PCA": PCA(n_components=2, random_state=SEED).fit_transform(Xs)}
    try:
        import umap
        proj["UMAP"] = umap.UMAP(n_components=2, random_state=SEED,
                                 n_neighbors=15, min_dist=0.1).fit_transform(Xs)
    except Exception as e:                                     # noqa: BLE001
        print(f"    [warn] UMAP unavailable ({type(e).__name__}) — PCA only")

    names = list(proj)
    fig, axes = plt.subplots(len(names), 2, figsize=(11.5, 5.4 * len(names)),
                             squeeze=False)
    classes = sorted(set(kg))
    cmapc = plt.get_cmap("tab10")
    for r, nm in enumerate(names):
        P = proj[nm]
        sc = axes[r][0].scatter(P[:, 0], P[:, 1], c=y, s=13, cmap="viridis",
                                edgecolors="none")
        fig.colorbar(sc, ax=axes[r][0], label=f"mean SM {depth} (m³/m³)")
        axes[r][0].set_title(f"{nm} — coloured by MEAN SOIL MOISTURE", fontsize=10)
        for i, c in enumerate(classes):
            m = kg == c
            axes[r][1].scatter(P[m, 0], P[m, 1], s=13, color=cmapc(i % 10),
                               label=str(c), edgecolors="none")
        axes[r][1].legend(fontsize=7, markerscale=1.4, loc="best", frameon=False)
        axes[r][1].set_title(f"{nm} — coloured by KÖPPEN CLASS (the confound)", fontsize=10)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle(
        f"{key} — station centre token (index 105), n={len(y)}\n"
        f"UMAP/PCA fitted UNSUPERVISED on the tokens; soil moisture used only for colour. "
        f"All statistics are computed in 768-d, not on these coordinates.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=170)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"    figure -> {out.with_suffix('.png').name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-dates", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--depth", default="0-10")
    ap.add_argument("--fig-keys", default="s2/l3,s2/l12,dem/l12")
    ap.add_argument("--categories", default="sm_only",
                    help="comma-separated subset of sm_only,sm_and_flux,flux_only. "
                         "Default sm_only: the population the model was trained and "
                         "evaluated on (eval manifest category_filter), and the only one "
                         "free of the non-finite S1 tokens found in sm_and_flux.")
    args = ap.parse_args()

    cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    disk = disk_index(cats)
    meta = pd.read_csv(SPLITS).dropna(subset=["source_network", "network", "station_id"])
    meta["dir"] = [station_dir(r) for _, r in meta.iterrows()]
    meta = meta[meta["dir"].isin(disk)].drop_duplicates("dir").set_index("dir")
    names = list(meta.index)[:args.limit] if args.limit else list(meta.index)
    print(f"{len(names)} stations on disk in {cats}; centre token = index {CENTRE_TOK}\n")

    with Pool(args.workers) as pool:
        recs = [r for r in pool.map(
            load_station, [(n, disk[n], args.max_dates) for n in names], chunksize=2)
            if r is not None]
    print(f"loaded {len(recs)} stations with labels\n")

    keys = sorted({k for r in recs for k in r["X"]})
    fig_keys = [k.strip() for k in args.fig_keys.split(",") if k.strip()]
    summary, rows, nonfinite = {}, [], {}

    print(f"depth {args.depth}   baselines: null {LADDER['null']:.4f} | "
          f"B8 tabular {LADDER['B8 +smap_tb']:.4f} | "
          f"network bias {LADDER['network RMS station bias']:.4f}\n")
    hdr = (f"{'modality/layer':>14} {'n':>5} | {'null':>7} {'kNN10':>7} {'skill%':>7} "
           f"{'eta2(20)':>9} {'p':>7} | {'null_w':>7} {'kNN10_w':>8} {'skill%':>7} "
           f"{'eta2':>6} {'p':>7}")
    print(hdr); print("-" * len(hdr))

    for key in keys:
        have = [r for r in recs if key in r["X"] and args.depth in r["y"]]
        if len(have) < 60:
            continue
        X = np.stack([r["X"][key] for r in have]).astype(np.float64)
        y = np.array([r["y"][args.depth] for r in have])
        net = meta.loc[[r["station"] for r in have], "network"].to_numpy()
        kg = meta.loc[[r["station"] for r in have], "kg_macro"].fillna("?").to_numpy()

        # Some stored tokens are non-finite. precompute_terramind.py:427 raises on
        # non-finite at write time, so these came through another path -- record them
        # rather than silently dropping, then continue on the clean rows.
        fin = np.isfinite(X).all(1) & np.isfinite(y)
        n_bad = int((~fin).sum())
        if n_bad:
            bad_names = [have[i]["station"] for i in np.where(~fin)[0]]
            nonfinite[key] = bad_names
            print(f"  [warn] {key}: {n_bad} stations with non-finite tokens dropped "
                  f"(e.g. {bad_names[:2]})")
            X, y, net, kg = X[fin], y[fin], net[fin], kg[fin]
            have = [h for h, f in zip(have, fin) if f]
        if len(y) < 60:
            continue

        glob = run_block(X, y)
        Xr, yr, _ = residualise(X, y, net)
        within = run_block(Xr, yr) if len(yr) >= 60 else None

        e20 = glob["eta2"].get("20", {})
        w20 = (within or {}).get("eta2", {}).get("20", {})
        n0, n1 = glob["sd_y"], (within or {}).get("sd_y", float("nan"))
        k0 = glob["knn_rmse"].get("10", float("nan"))
        k1 = (within or {}).get("knn_rmse", {}).get("10", float("nan"))
        print(f"{key:>14} {len(y):>5} | {n0:>7.4f} {k0:>7.4f} {100*(1-k0/n0):>6.1f}% "
              f"{e20.get('eta2', float('nan')):>9.3f} {e20.get('p_perm', float('nan')):>7.4f} | "
              f"{n1:>7.4f} {k1:>8.4f} {100*(1-k1/n1):>6.1f}% "
              f"{w20.get('eta2', float('nan')):>6.3f} {w20.get('p_perm', float('nan')):>7.4f}")

        summary[key] = {"depth": args.depth, "global": glob, "within_network": within,
                        "sd_y_global": float(y.std()),
                        "sd_y_within": float(yr.std()) if len(yr) else None}
        rows.append({"key": key, "n": len(y), "n_nonfinite_dropped": n_bad,
                     "knn10_global": glob["knn_rmse"].get("10"),
                     "eta2_20_global": e20.get("eta2"), "p_20_global": e20.get("p_perm"),
                     "knn10_within": (within or {}).get("knn_rmse", {}).get("10"),
                     "eta2_20_within": w20.get("eta2"), "p_20_within": w20.get("p_perm"),
                     "null_global": glob["sd_y"],
                     "null_within": (within or {}).get("sd_y"),
                     "purity_ratio_global": glob["purity"]["ratio"]})

        if key in fig_keys:
            make_figure(X, y, kg, key, args.depth,
                        FIG_DIR / f"umap_{key.replace('/', '_')}_{args.depth}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({"ladder_baselines": LADDER, "depth": args.depth,
                                    "nonfinite_tokens": nonfinite,
                                    "results": summary}, indent=2))
    print(f"\nwrote {OUT_CSV.name} and {OUT_JSON.name}")
    print("\nRead kNN10 against the ladder: beating 0.0576 means the embedding carries "
          "SM information\nbeyond the full tabular stack. The WITHIN-NETWORK column is the "
          "one that matters for\nsub-km skill — a large drop there is the expected result, "
          "and its size is the finding.")


if __name__ == "__main__":
    main()
