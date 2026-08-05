"""§20.14 — is per-station MEAN soil moisture predictable from available inputs?

The model's dominant error is a per-station constant OFFSET, not anomaly error. This
probe strips the problem to one number per station and asks whether that number is
predictable at all — the gate between §20.7 (reframe the claim) and §20.8 (fix the model).

Ridge is chosen because it is WEAK: ~600 rows under a real L2 penalty cannot memorise,
so a positive result cannot be an artefact. GBM is the nonlinearity control — ridge
failing while GBM succeeds means the relation is curved, not absent.

CPU only, no GPU, ~1 min. Never instantiates SoilMoistureDataset (that would eagerly
load ~16 GB of L12 tokens); opens zarr per station directly.

    python station_mean_probe.py [--out csvs] [--splits-csv csvs/station_splits.csv]
"""
import argparse
import json
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from dataset import ERA5_VARS, SM_DEPTHS, ZARR_ROOT, _load_zarr_labels, fill_soil_nans  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# The network's own RMS per-station bias at its best epoch — the number this probe is
# compared against. From checkpoints/.../cls_depth_star_reg/val_station_metrics.csv.
NETWORK_RMS_BIAS = {"0-10": 0.0618, "10-30": 0.0611, "30-100": 0.0875}

CATEGORY = "sm_only"
ALPHAS   = np.logspace(-2, 5, 40)


# ── per-station read (worker) ────────────────────────────────────────────────

def _read_station(task):
    """-> dict of raw per-station quantities, or {'drop': reason}."""
    dir_name, path = task
    out = {"dir": dir_name}
    try:
        zg = zarr.open(path, mode="r")
    except Exception as e:                                   # noqa: BLE001
        return {"dir": dir_name, "drop": f"zarr open failed: {type(e).__name__}"}

    if "soil" not in zg.array_keys():
        return {"dir": dir_name, "drop": "no soil array"}
    soil = fill_soil_nans(zg["soil"][:])                     # (21, 74, 74)
    out["soil"] = np.concatenate([soil.mean(axis=(1, 2)), soil.std(axis=(1, 2))]).astype(np.float32)

    if "era5" not in zg.group_keys():
        return {"dir": dir_name, "drop": "no era5 group"}
    era5 = zg["era5/values"][:]                              # (N, 19) RAW units
    out["era5"] = era5.mean(0).astype(np.float32)
    # Seasonal amplitude: spread of the daily series, a cheap proxy for seasonality
    # that needs no date handling (record lengths differ 1096-3653 days).
    out["seasonal"] = np.array([
        era5[:, ERA5_VARS.index("t2m_mean")].std(),
        era5[:, ERA5_VARS.index("tp_sum")].std(),
    ], dtype=np.float32)

    lab = _load_zarr_labels(zg)   # handles the qc/sm length realign (403 of 661 stations)
    if lab is None:
        return {"dir": dir_name, "drop": "no labels"}
    sm_np, depths, _, qc_np = lab
    for d in SM_DEPTHS:
        if d not in depths:
            continue
        j = depths.index(d)
        v = sm_np[j]
        m = (qc_np[j] == 0) if qc_np is not None else np.ones_like(v, dtype=bool)
        m = m & np.isfinite(v)
        if m.sum() > 0:
            out[f"y_{d}"] = float(v[m].mean())
            out[f"n_{d}"] = int(m.sum())
    return out


# ── feature assembly ─────────────────────────────────────────────────────────

def _derived(era5_means):
    """VPD, Hargreaves PET, aridity — from the 19 ERA5 means already in hand."""
    i = ERA5_VARS.index
    t2m = era5_means[:, i("t2m_mean")] - 273.15
    d2m = era5_means[:, i("d2m_mean")] - 273.15
    sat = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))   # noqa: E731  kPa
    vpd = sat(t2m) - sat(d2m)
    dtr = np.maximum(era5_means[:, i("t2m_max")] - era5_means[:, i("t2m_min")], 0.0)
    pet = 0.0023 * (t2m + 17.8) * np.sqrt(dtr)                # Hargreaves, mm/day-ish
    aridity = era5_means[:, i("tp_sum")] / (pet + 1e-6)
    return np.stack([vpd, pet, aridity], axis=1).astype(np.float32)


def build_blocks(meta, soil, era5, seasonal):
    """-> ordered [(name, feature_array), ...] forming the NESTED ladder."""
    elev = np.column_stack([
        meta["elevation_m"].to_numpy(dtype=np.float32),
        pd.get_dummies(meta["elevation_band"].astype(str)).to_numpy(dtype=np.float32),
    ])
    landcover = pd.get_dummies(
        meta[["IGBP", "lc_cci"]].astype(str)).to_numpy(dtype=np.float32)
    # koppen belongs with CLIMATE, not with the static site block: leaving it in B1
    # would leak climate into the "static properties" increment the ladder turns on.
    koppen = pd.get_dummies(meta["koppen_geiger"].astype(str)).to_numpy(dtype=np.float32)
    climate = np.hstack([era5, koppen])
    latlon = meta[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    return [
        ("B1 soil",       soil),
        ("B2 +terrain",   elev),
        ("B3 +landcover", landcover),
        ("B4 +climate",   climate),
        ("B5 +derived",   np.hstack([_derived(era5), seasonal])),
        ("B6 +latlon",    latlon),
    ]


# ── fitting ──────────────────────────────────────────────────────────────────

def fit_block(X, y, is_train, is_val, groups, seed=0):
    """-> dict of scores for one (block, depth). None if too few stations."""
    m_tr = is_train & np.isfinite(y)
    m_va = is_val & np.isfinite(y)
    if m_tr.sum() < 30 or m_va.sum() < 5:
        return None

    scaler = StandardScaler().fit(X[m_tr])
    Xtr, Xva = scaler.transform(X[m_tr]), scaler.transform(X[m_va])
    ytr, yva = y[m_tr], y[m_va]

    rmse = lambda p: float(np.sqrt(np.mean((p - yva) ** 2)))   # noqa: E731

    # GroupKFold on location_group_id: plain K-fold would put neighbouring stations in
    # both fold-train and fold-test and yield an optimistically small alpha.
    n_sp = min(5, len(np.unique(groups[m_tr])))
    folds = list(GroupKFold(n_splits=n_sp).split(Xtr, ytr, groups[m_tr]))
    ridge = RidgeCV(alphas=ALPHAS, cv=folds).fit(Xtr, ytr)

    gbm = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400,
        early_stopping=True, n_iter_no_change=25, validation_fraction=0.2,
        random_state=seed,
    ).fit(Xtr, ytr)

    return {
        "n_train": int(m_tr.sum()), "n_val": int(m_va.sum()), "n_features": int(X.shape[1]),
        "null":  rmse(np.full_like(yva, ytr.mean())),
        "ridge": rmse(ridge.predict(Xva)),
        "gbm":   rmse(gbm.predict(Xva)),
        "ridge_alpha": float(ridge.alpha_),
        "_ridge_coef": ridge.coef_,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-csv", default="csvs/station_splits.csv")
    ap.add_argument("--out",        default="csvs")
    ap.add_argument("--figdir",     default="figures")
    ap.add_argument("--workers",    type=int, default=64)
    args = ap.parse_args()

    # ── station selection: mirrors dataset.py:813-835 ─────────────────────
    splits = pd.read_csv(args.splits_csv)

    def _cat(r):
        sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        fl = str(r.get("has_flux", "False")).lower() == "true"
        return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")

    splits = splits[splits.apply(_cat, axis=1) == CATEGORY]
    splits = splits[splits["split"].isin(["train", "val"])]
    n_before = len(splits)
    splits = splits[splits["soil_patch_ok"].astype(str).str.lower() == "true"]
    print(f"Stations: {n_before} {CATEGORY} train+val; "
          f"{n_before - len(splits)} dropped by soil_patch_ok -> {len(splits)}")
    splits = splits.reset_index(drop=True)

    tasks = []
    for _, r in splits.iterrows():
        dn = (f"ISMN_{r['network']}_{r['station_name']}" if str(r["source_network"]) == "ISMN"
              else f"{r['source_network']}_{r['station_id']}")
        tasks.append((dn, str(ZARR_ROOT / CATEGORY / dn)))

    with Pool(args.workers) as pool:
        recs = pool.map(_read_station, tasks)

    dropped = [r for r in recs if "drop" in r]
    if dropped:                       # no silent caps — say what was skipped and why
        print(f"\nDROPPED {len(dropped)} stations:")
        for reason, n in Counter(r["drop"] for r in dropped).items():
            print(f"   {n:4d}  {reason}")
    keep = np.array(["drop" not in r for r in recs])
    recs = [r for r, k in zip(recs, keep) if k]
    meta = splits[keep].reset_index(drop=True)

    soil     = np.stack([r["soil"] for r in recs])
    era5     = np.stack([r["era5"] for r in recs])
    seasonal = np.stack([r["seasonal"] for r in recs])
    targets  = {d: np.array([r.get(f"y_{d}", np.nan) for r in recs]) for d in SM_DEPTHS}

    is_train = (meta["split"] == "train").to_numpy()
    is_val   = (meta["split"] == "val").to_numpy()
    groups   = meta["location_group_id"].to_numpy()
    print(f"Usable: {len(meta)}  ({is_train.sum()} train / {is_val.sum()} val)")
    for d in SM_DEPTHS:
        y = targets[d]
        print(f"   {d:>7}: {np.isfinite(y[is_train]).sum():3d} train / "
              f"{np.isfinite(y[is_val]).sum():2d} val stations with observed mean")

    blocks = build_blocks(meta, soil, era5, seasonal)

    # ── the nested ladder ─────────────────────────────────────────────────
    results, coefs, X = {}, {}, None
    hdr = f"{'block':<16}{'nfeat':>6} | " + " | ".join(f"{d:^25}" for d in SM_DEPTHS)
    print("\n" + hdr)
    print(f"{'':<22} | " + " | ".join(f"{'ridge':>8}{'gbm':>8}{'vs null':>9}" for _ in SM_DEPTHS))
    print("-" * len(hdr))
    for name, blk in blocks:
        X = blk if X is None else np.hstack([X, blk])
        line = f"{name:<16}{X.shape[1]:>6} | "
        results[name] = {}
        for d in SM_DEPTHS:
            sc = fit_block(X, targets[d], is_train, is_val, groups)
            if sc is None:
                line += f"{'--':>25} | "
                continue
            coefs[(name, d)] = sc.pop("_ridge_coef")
            results[name][d] = sc
            gain = 100 * (1 - min(sc["ridge"], sc["gbm"]) / sc["null"])
            line += f"{sc['ridge']:>8.4f}{sc['gbm']:>8.4f}{gain:>+8.0f}% | "
        print(line)

    print("-" * len(hdr))
    print(f"{'network':<22} | " + " | ".join(
        f"{NETWORK_RMS_BIAS[d]:>8.4f}{'':>8}{'':>9}" for d in SM_DEPTHS))

    # ── ridge coefficients on the full ladder ─────────────────────────────
    full = blocks[-1][0]
    names = (["soil_mean_%02d" % i for i in range(21)] + ["soil_std_%02d" % i for i in range(21)]
             + ["elev"] + [f"elevband_{i}" for i in range(meta["elevation_band"].nunique())]
             + [f"lc_{i}" for i in range(blocks[2][1].shape[1])]
             + list(ERA5_VARS) + [f"koppen_{i}" for i in range(blocks[3][1].shape[1] - 19)]
             + ["vpd", "pet", "aridity", "seas_t2m", "seas_tp", "lat", "lon"])
    print(f"\nTop-10 ridge coefficients ({full}, standardised):")
    for d in SM_DEPTHS:
        c = coefs.get((full, d))
        if c is None:
            continue
        top = np.argsort(np.abs(c))[::-1][:10]
        lbl = [f"{names[i] if i < len(names) else f'f{i}'}={c[i]:+.4f}" for i in top]
        print(f"   {d:>7}: " + "  ".join(lbl))

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    for d in SM_DEPTHS:
        best = min((results[b][d]["ridge"], results[b][d]["gbm"])
                   for b in results if d in results[b])
        best = min(min(x) for x in [best]) if isinstance(best, tuple) else best
        vals = [min(results[b][d]["ridge"], results[b][d]["gbm"]) for b in results if d in results[b]]
        best = min(vals)
        null = results["B1 soil"][d]["null"]
        net  = NETWORK_RMS_BIAS[d]
        if best > null * 0.97:
            v = "NO SIGNAL -> §20.7 Branch A (retarget to anomaly, reframe claim)"
        elif best < net * 0.90:
            v = "SIGNAL THE NETWORK IS DISCARDING -> §20.8 Branch B (post-hoc correction first)"
        else:
            v = "network already extracts what exists -> §20.7; remainder irreducible"
        print(f"  {d:>7}  best={best:.4f}  null={null:.4f}  network={net:.4f}   {v}")
    print("\nNOTE: swvl/ET were not downloaded, so a negative reads 'not learnable from")
    print("      what we have', NOT 'not learnable in principle'. (§20.14.6)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "network_rms_station_bias": NETWORK_RMS_BIAS,
        "n_train": int(is_train.sum()), "n_val": int(is_val.sum()),
        "dropped": [{"dir": r["dir"], "reason": r["drop"]} for r in dropped],
        "ladder": results,
    }
    p = out / "station_mean_probe.json"
    p.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nSaved: {p}")

    _plot(targets, blocks, meta, is_train, is_val, groups, Path(args.figdir))


def _plot(targets, blocks, meta, is_train, is_val, groups, figdir):
    """Predicted vs observed station mean, per depth, on the full ladder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_R, C_G, INK, MUTED, GRID = "#0173B2", "#DE8F05", "#1a1a1a", "#6b6b6b", "#d9d9d9"
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.dpi": 300, "savefig.bbox": "tight",
    })
    X = np.hstack([b for _, b in blocks])
    fig, axes = plt.subplots(1, len(SM_DEPTHS), figsize=(9.5, 3.2))
    for ax, d in zip(axes, SM_DEPTHS):
        y = targets[d]
        m_tr, m_va = is_train & np.isfinite(y), is_val & np.isfinite(y)
        if m_tr.sum() < 30 or m_va.sum() < 5:
            ax.set_axis_off(); continue
        sc = StandardScaler().fit(X[m_tr])
        Xtr, Xva = sc.transform(X[m_tr]), sc.transform(X[m_va])
        n_sp = min(5, len(np.unique(groups[m_tr])))
        folds = list(GroupKFold(n_splits=n_sp).split(Xtr, y[m_tr], groups[m_tr]))
        pr = RidgeCV(alphas=ALPHAS, cv=folds).fit(Xtr, y[m_tr]).predict(Xva)
        pg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                           early_stopping=True, n_iter_no_change=25,
                                           validation_fraction=0.2, random_state=0
                                           ).fit(Xtr, y[m_tr]).predict(Xva)
        ax.scatter(y[m_va], pr, s=18, color=C_R, alpha=0.8, lw=0.5, ec="white", label="Ridge", zorder=3)
        ax.scatter(y[m_va], pg, s=18, color=C_G, alpha=0.8, lw=0.5, ec="white", marker="s",
                   label="GBM", zorder=3)
        lo = float(min(np.nanmin(y[m_va]), pr.min(), pg.min())) - 0.02
        hi = float(max(np.nanmax(y[m_va]), pr.max(), pg.max())) + 0.02
        ax.plot([lo, hi], [lo, hi], ls=":", lw=0.9, color=MUTED, zorder=1)
        ax.axhline(y[m_tr].mean(), ls="--", lw=0.9, color=MUTED, zorder=1)
        ax.annotate("null (train mean)", (lo, y[m_tr].mean()), xytext=(3, 3),
                    textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{d} cm  (n={m_va.sum()})", loc="left", pad=6)
        ax.set_xlabel("Observed station mean SM  (m$^3$/m$^3$)")
        ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Predicted station mean")
    axes[0].legend(frameon=False, loc="upper left")
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = figdir / f"station_mean_probe.{ext}"
        fig.savefig(p); print(f"Saved: {p}")


if __name__ == "__main__":
    main()
