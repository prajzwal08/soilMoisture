"""§23 -- does the predicted soil-moisture field vary in space, or is it a constant?

The model has only ever emitted a 2D field: SoilMoistureModel.forward() (model.py:670)
returns (B, 3, 224, 224).  Every evaluation path indexes the centre pixel (eval_predict.py:75)
and discards the other 50,175, and masked_huber_loss (model.py:751) supervises exactly one of
them.  §16.1 measured output norm-std 0.0065 on an older checkpoint and called it decoder-side;
§16.4 then drove the same statistic to 0.2504 by giving the decoder a dense target, proving the
flatness is a supervision artefact rather than a capacity limit.

This measures it on the FINAL Phase-1 model (cls_depth_star_reg/best.pt, epoch 16), at stations
where the model demonstrably works, across seasons, beside the satellite scene the model actually
consumed.

Figure per station:
    context strip   [ S2 RGB | NDVI | DEM | LULC ]        structure available to paint
    season grid     [ anchor scene | SM x3 absolute || SM x3 per-panel anomaly ]
    metric strip    norm-std per season/depth against the §16.1 and §16.4 reference lines

Stage 1 (--dry-run-selection) is CPU-only and imports no torch, so the station choice can be
reviewed before any GPU time is spent.

    python plot_spatial_heterogeneity.py --dry-run-selection          # login node, seconds
    sbatch slurm/spatial_heterogeneity.sh --station ISMN_RSMN_Iasi --selftest \
           --verify-against eval_output/predictions_oos.parquet
    sbatch slurm/spatial_heterogeneity.sh
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
REPO       = Path(__file__).resolve().parent
EVAL_DIR   = REPO / "eval_output"
SPLITS_CSV = REPO / "csvs" / "station_splits.csv"
ERA5_STATS = REPO / "csvs" / "era5_stats.json"
CKPT_ROOT  = Path("/gpfs/work3/0/prjs1968/checkpoints/soilmoisture/phase1_sm_only")
TOKEN_ROOT = Path("/gpfs/scratch1/shared/pkhanal/zarr")
SAT_ROOT   = Path("/projects/prjs1968/satellite_zarr")
OUT_DIR    = REPO / "figures" / "spatial_heterogeneity"

CATEGORIES = ["sm_only", "sm_and_flux", "flux_only"]
SM_DEPTHS  = ["0-10", "10-30", "30-100"]
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
SEASON_NAMES = {15: "DJF", 105: "MAM", 196: "JJA", 288: "SON"}

# 12-band S2 order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12 (B10 dropped)
S2_RED, S2_GREEN, S2_BLUE, S2_NIR = 3, 2, 1, 7
ORBIT_GROUP  = {0: "s2", 1: "s1_asc", 2: "s1_desc"}
CLOUD_CLASSES = (3, 4, 5)          # thin cloud, thick cloud, shadow (NOT 1 water / 2 snow)

SM_VMIN, SM_VMAX = 0.0, 0.5
STATION_RC = (112, 112)

# Land cover: the zarr stores the REMAPPED TerraMind indices, not raw ESRI values
# (text/modality_bands.txt:59-81).  ESRI v1 "Grass" (3) and "Scrub/Shrub" (6) are
# both merged into Rangeland, which is why e.g. Crossroads reads 100% class 9.
LULC_CLASSES = {0: "No data", 1: "Water", 2: "Trees", 3: "Flooded veg.",
                4: "Crops", 5: "Built area", 6: "Bare ground", 7: "Snow/ice",
                8: "Clouds", 9: "Rangeland"}

IGBP_NAMES = {
    "ENF": "evergreen needleleaf forest", "EBF": "evergreen broadleaf forest",
    "DNF": "deciduous needleleaf forest", "DBF": "deciduous broadleaf forest",
    "MF": "mixed forest", "CSH": "closed shrubland", "OSH": "open shrubland",
    "WSA": "woody savanna", "SAV": "savanna", "GRA": "grassland",
    "WET": "permanent wetland", "CRO": "cropland", "URB": "urban and built-up",
    "CVM": "cropland/natural mosaic", "SNO": "snow and ice",
    "BSV": "barren or sparsely vegetated", "WAT": "water",
}

KOPPEN_NAMES = {
    "Af": "tropical rainforest", "Am": "tropical monsoon", "Aw": "tropical savanna",
    "BWh": "hot desert", "BWk": "cold desert", "BSh": "hot semi-arid",
    "BSk": "cold semi-arid", "Csa": "hot-summer Mediterranean",
    "Csb": "warm-summer Mediterranean", "Cfa": "humid subtropical",
    "Cfb": "temperate oceanic", "Cfc": "subpolar oceanic",
    "Dsa": "hot-summer Mediterranean continental",
    "Dsb": "warm-summer Mediterranean continental",
    "Dfa": "hot-summer humid continental", "Dfb": "warm-summer humid continental",
    "Dfc": "subarctic", "Dwa": "monsoon humid continental", "ET": "tundra",
    "EF": "ice cap",
}

# §16.1 / §16.4 reference values -- runbook §23.0
TIER1_NORM_STD   = 0.0065     # baseline_huber epoch 11, output map
CAPACITY_CEILING = 0.2504     # decoder overfit to a dense synthetic target


# ── small helpers ────────────────────────────────────────────────────────────

def pstretch(a: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Per-channel percentile stretch to [0,1] -- the repo's figure convention."""
    p_lo, p_hi = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p_lo) / max(p_hi - p_lo, 1e-9), 0, 1)


def block_mean(a: np.ndarray, k: int = 16) -> np.ndarray:
    """224x224 -> 14x14 by k x k block averaging (the bottleneck's native grid)."""
    n = a.shape[0] // k
    return a[:n * k, :n * k].reshape(n, k, n, k).mean(axis=(1, 3))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def norm_std(field: np.ndarray) -> float:
    """std / |mean| -- tier1_probe.py:123, reproduced so §16.1's 0.0065 is comparable."""
    f = np.asarray(field, dtype=np.float64)
    m = np.abs(f.mean())
    return float(f.std() / m) if m > 1e-12 else float("nan")


def station_category(station_key: str) -> str | None:
    for cat in CATEGORIES:
        if (TOKEN_ROOT / cat / station_key).exists():
            return cat
    return None


def _dates_of(zg, group: str) -> pd.DatetimeIndex | None:
    key = f"{group}/dates"
    if key not in zg:
        return None
    raw = np.asarray(zg[key][:])
    strs = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw]
    return pd.to_datetime(strs, format="%Y%m%d")


# ── footprint structure (raw satellite zarr) ─────────────────────────────────

def footprint_stats(station_key: str) -> dict:
    """DEM relief and LULC class count for the 224x224 (2240 m) patch.

    The denominator of the whole argument: "the map is flat" only means something
    against "and here is what there was to paint".
    """
    import zarr
    path = SAT_ROOT / f"{station_key}.zarr"
    out = {"dem_relief_m": np.nan, "dem_p2_p98_m": np.nan, "n_lulc": np.nan,
           "lulc_year": None, "sat_zarr": path.exists()}
    if not path.exists():
        return out
    try:
        zg = zarr.open_group(str(path), mode="r")     # raw stores are NOT consolidated
    except Exception as e:                            # noqa: BLE001
        out["error"] = type(e).__name__
        return out

    if "dem/data" in zg:
        dem = np.asarray(zg["dem/data"][0], dtype=np.float64)
        dem = dem[np.isfinite(dem)]
        if dem.size:
            out["dem_relief_m"] = float(dem.max() - dem.min())
            out["dem_p2_p98_m"] = float(np.percentile(dem, 98) - np.percentile(dem, 2))
    if "lulc/data" in zg:
        lulc = np.asarray(zg["lulc/data"][-1])
        vals, counts = np.unique(lulc, return_counts=True)
        frac = counts / counts.sum()
        out["n_lulc"] = int((frac >= 0.01).sum())     # ignore stray single pixels
        if "lulc/years" in zg:
            out["lulc_year"] = int(np.asarray(zg["lulc/years"])[-1])
    return out


# ── §23.1 selection ──────────────────────────────────────────────────────────

def select_stations(split: str, n_stations: int, min_n: int, min_r: float,
                    min_obs_std: float, exclude_igbp: list, control_station: str | None,
                    rank_depth: str = "0-10"):
    """Gated ubRMSE ranking + a heterogeneous pick + a deliberately flat control.

    Ranking on ubRMSE alone selects a lake and two flat deserts (runbook §23.1), i.e.
    sites where a uniform map is arguably correct.  Each gate blocks one rebuttal;
    the control station blocks "the landscape is just flat".
    """
    suf = rank_depth.replace("-", "_")
    ps = pd.read_csv(EVAL_DIR / f"per_station_{split}.csv")
    need = [f"ubRMSE_{suf}", f"n_{suf}", f"R_{suf}", "IGBP"]
    missing = [c for c in need if c not in ps.columns]
    if missing:
        raise SystemExit(f"per_station_{split}.csv lacks {missing}")

    pq = pd.read_parquet(EVAL_DIR / f"predictions_{split}.parquet")
    obs_std = (pq[pq["depth"] == rank_depth]
               .groupby("station_key", observed=True)["obs"].std().rename("obs_std"))
    d = ps.merge(obs_std, on="station_key", how="left")
    d["category"] = [station_category(s) for s in d["station_key"]]

    gates = {
        "n":        d[f"n_{suf}"] >= min_n,
        "R":        d[f"R_{suf}"] >= min_r,
        "obs_std":  d["obs_std"] >= min_obs_std,
        "igbp":     ~d["IGBP"].isin(exclude_igbp),
        "category": d["category"] == "sm_only",
        "ubRMSE":   d[f"ubRMSE_{suf}"].notna(),
    }
    for name, ok in gates.items():
        d[f"gate_{name}"] = ok
    d["gated"] = np.logical_and.reduce(list(gates.values()))

    print(f"[{split}] {len(d)} stations -> gates: "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in gates.items())
          + f"  => {int(d['gated'].sum())} pass all")

    cand = d[d["gated"]].sort_values(f"ubRMSE_{suf}").copy()
    if cand.empty:
        raise SystemExit("no station passed the gates -- loosen --min-n / --min-r / --min-obs-std")

    head = cand.head(max(n_stations * 6, 12)).copy()
    stats = pd.DataFrame([footprint_stats(s) for s in head["station_key"]], index=head.index)
    head = pd.concat([head, stats], axis=1)

    picks, roles = [], []
    best = head.iloc[0]
    picks.append(best["station_key"]); roles.append("best_ubrmse")

    # most heterogeneous footprint, not merely the first that clears the bar --
    # taking the first by ubRMSE rank leaves all picks in one climate zone
    het = head[(head["station_key"] != best["station_key"]) &
               ((head["n_lulc"] >= 3) | (head["dem_relief_m"] >= 30))]
    het = het.sort_values(["n_lulc", "dem_relief_m"], ascending=False)
    if not het.empty:
        picks.append(het.iloc[0]["station_key"]); roles.append("heterogeneous")
    else:
        print("  ! no gated station has >=3 LULC classes or >=30 m relief")

    if control_station:
        if control_station in set(d["station_key"]):
            if control_station not in picks:
                picks.append(control_station); roles.append("flat_control")
        else:
            print(f"  ! control station {control_station} not in {split} -- skipped")

    sel = d[d["station_key"].isin(picks)].copy()
    sel["role"] = [roles[picks.index(s)] for s in sel["station_key"]]
    extra = pd.DataFrame([footprint_stats(s) for s in sel["station_key"]], index=sel.index)
    sel = pd.concat([sel, extra.drop(columns=[c for c in extra.columns
                                              if c in sel.columns])], axis=1)
    sel["split"] = split
    return sel.sort_values("role"), head


def report_selection(sel: pd.DataFrame, shortlist: pd.DataFrame, rank_depth: str):
    suf = rank_depth.replace("-", "_")
    cols = ["station_key", f"ubRMSE_{suf}", f"R_{suf}", f"n_{suf}", "obs_std",
            "IGBP", "koppen_geiger", "n_lulc", "dem_relief_m"]
    print(f"\nshortlist (gated, ranked on ubRMSE_{suf}):")
    print(shortlist[[c for c in cols if c in shortlist.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSELECTED:")
    print(sel[["role"] + [c for c in cols if c in sel.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ── §23.4 the scene the model actually consumed ──────────────────────────────

def cloud_fraction(station_key: str, date: pd.Timestamp) -> float:
    """Fraction of the patch flagged thin cloud / thick cloud / shadow.

    Deliberately NOT mean(cm != 0) (visualize_embeddings.py:118), which counts water
    and snow as cloud and marks every lakeside or winter station permanently cloudy.
    """
    import zarr
    cat = station_category(station_key)
    if cat is None:
        return float("nan")
    try:
        zg = zarr.open(str(TOKEN_ROOT / cat / station_key), mode="r")
        if "cm/masks" not in zg:
            return float("nan")
        dates = _dates_of(zg, "cm")
        hit = np.where(dates == date)[0]
        if not len(hit):
            return float("nan")
        cm = np.asarray(zg["cm/masks"][int(hit[0])])
        return float(np.isin(cm, CLOUD_CLASSES).mean())
    except Exception:                                     # noqa: BLE001
        return float("nan")


def anchor_scene(station_key: str, target_date: datetime,
                 anchor_rel_pos: int, anchor_orbit: int) -> dict:
    """The exact acquisition the model consumed, recovered from the batch.

    dataset.py:503 sets anchor_rel_pos = 364 - (target_date - acq_date).days, and the
    raw store's dates are index-aligned with the token store, so an exact string match
    retrieves the true input scene.  The anchor may be up to 364 days stale
    (select_anchor_zarr, dataset.py:488) -- the lag is reported, never hidden.
    """
    import zarr
    lag = 364 - int(anchor_rel_pos)
    anchor_date = pd.Timestamp(target_date) - pd.Timedelta(days=lag)
    group = ORBIT_GROUP.get(int(anchor_orbit), "s2")
    out = {"kind": group, "date": anchor_date, "lag_days": lag, "img": None,
           "cloud_frac": float("nan"), "exact": False}

    path = SAT_ROOT / f"{station_key}.zarr"
    if not path.exists():
        return out
    zg = zarr.open_group(str(path), mode="r")
    dates = _dates_of(zg, group)
    if dates is None or f"{group}/data" not in zg:
        return out

    hit = np.where(dates == anchor_date)[0]
    if len(hit):
        idx, out["exact"] = int(hit[0]), True
    else:                       # should not happen; fall back and flag it loudly
        idx = int(np.argmin(np.abs((dates - anchor_date).days)))
        out["date_actual"] = dates[idx]
        print(f"    ! anchor {anchor_date.date()} not found in {group}; "
              f"nearest {dates[idx].date()}")

    arr = np.asarray(zg[f"{group}/data"][idx])
    if group == "s2":
        rgb = arr[[S2_RED, S2_GREEN, S2_BLUE]].astype(np.float32)
        out["img"] = np.stack([pstretch(c) for c in rgb], axis=-1)
        out["cloud_frac"] = cloud_fraction(station_key, dates[idx])
    else:
        out["img"] = np.clip(arr[0].astype(np.float32), -20, 0)     # VV dB
    return out


def context_layers(station_key: str, ref_date: datetime) -> dict:
    """RGB / NDVI / DEM / LULC for the same footprint, drawn once per station.

    The RGB here is the least-cloudy scene near ref_date -- context only, explicitly
    NOT the model input (that is anchor_scene()).
    """
    import zarr
    out = {"rgb": None, "ndvi": None, "dem": None, "lulc": None, "rgb_date": None,
           "n_lulc": np.nan}
    path = SAT_ROOT / f"{station_key}.zarr"
    if not path.exists():
        return out
    zg = zarr.open_group(str(path), mode="r")

    dates = _dates_of(zg, "s2")
    if dates is not None and "s2/data" in zg:
        near = np.argsort(np.abs((dates - pd.Timestamp(ref_date)).days))[:8]
        best, best_cf = None, np.inf
        for i in near:                      # prefer a clear scene among the nearest few
            cf = cloud_fraction(station_key, dates[int(i)])
            cf = 0.0 if np.isnan(cf) else cf
            if cf < best_cf:
                best, best_cf = int(i), cf
            if cf < 0.02:
                break
        if best is not None:
            arr = np.asarray(zg["s2/data"][best]).astype(np.float32)
            rgb = arr[[S2_RED, S2_GREEN, S2_BLUE]]
            out["rgb"] = np.stack([pstretch(c) for c in rgb], axis=-1)
            out["rgb_date"] = dates[best]
            out["rgb_cloud"] = best_cf
            nir, red = arr[S2_NIR] / 1e4, arr[S2_RED] / 1e4
            out["ndvi"] = np.where((nir + red) > 1e-6, (nir - red) / (nir + red + 1e-9),
                                   np.nan)
    if "dem/data" in zg:
        out["dem"] = np.asarray(zg["dem/data"][0], dtype=np.float32)
    if "lulc/data" in zg:
        lulc = np.asarray(zg["lulc/data"][-1])
        out["lulc"] = lulc
        vals, counts = np.unique(lulc, return_counts=True)
        out["n_lulc"] = int(((counts / counts.sum()) >= 0.01).sum())
    return out


# ── §23.3 metrics ────────────────────────────────────────────────────────────

def pattern_persistence(seasons: list) -> dict:
    """Is the SAME spatial pattern painted in every season and at every depth?

    A field responding to its inputs should reorganise between January and July.
    A high pairwise correlation between seasonal anomaly maps means the model is
    painting a fixed pattern and only shifting its mean -- the signature of a
    token-grid artefact rather than learned hydrology.  Computed on the 14x14
    bottleneck grid so bilinear-upsample smoothing cannot inflate it.
    """
    out = {}
    for j, depth in enumerate(SM_DEPTHS):
        anoms = [block_mean(r["sm"][j] - r["sm"][j].mean()) for r in seasons]
        pairs = {}
        for a in range(len(anoms)):
            for b in range(a + 1, len(anoms)):
                lab = (f"{SEASON_NAMES.get(seasons[a]['target_doy'], a)}~"
                       f"{SEASON_NAMES.get(seasons[b]['target_doy'], b)}")
                pairs[lab] = safe_corr(anoms[a], anoms[b])
        out[depth] = {"mean_pairwise_r": (float(np.nanmean(list(pairs.values())))
                                          if pairs else float("nan")),
                      "pairs": pairs}
    cd = [safe_corr(block_mean(r["sm"][0] - r["sm"][0].mean()),
                    block_mean(r["sm"][2] - r["sm"][2].mean())) for r in seasons]
    out["cross_depth_0-10_vs_30-100"] = float(np.nanmean(cd)) if cd else float("nan")
    return out


def panel_stats(sm: np.ndarray, dem: np.ndarray | None,
                ndvi: np.ndarray | None) -> dict:
    """Spatial statistics for one (depth, season) map.

    norm_std is tier1_probe's definition so it compares directly with §16.1's 0.0065.
    Correlations are reported at native 224 and at the 14x14 bottleneck grid; the
    latter is the meaningful one -- it removes the bilinear-upsample smoothing.
    No p-values: 50,176 autocorrelated pixels, effective n ~ 196.
    """
    anom = sm - sm.mean()
    p2, p98 = np.percentile(sm, 2), np.percentile(sm, 98)
    d = {
        "mean": float(sm.mean()), "sigma": float(sm.std()), "norm_std": norm_std(sm),
        "delta_p2_p98": float(p98 - p2),
        "min": float(sm.min()), "max": float(sm.max()),
        "centre": float(sm[STATION_RC]),
    }
    sm14 = block_mean(sm)
    up = np.repeat(np.repeat(sm14, 16, axis=0), 16, axis=1)
    d["var_retained_14x14"] = (float(up.var() / sm.var()) if sm.var() > 0 else float("nan"))
    if dem is not None:
        d["r_dem_224"] = safe_corr(anom, dem)
        d["r_dem_14"]  = safe_corr(block_mean(anom), block_mean(dem))
    if ndvi is not None:
        d["r_ndvi_224"] = safe_corr(anom, ndvi)
        d["r_ndvi_14"]  = safe_corr(block_mean(anom), block_mean(np.nan_to_num(ndvi)))
    return d


# ── §23.5 inference ──────────────────────────────────────────────────────────

def choose_year(ds, station_key: str, season_doys: list, max_gap: int,
                year_range=(2016, 2022)) -> int | None:
    """Year filling the most season slots, ties broken by recency.

    Replaces the blind median-year pick (plot_spatial_sm_meeting.py:469), which can
    land on a year covering only summer.
    """
    doys = {}
    for s in ds.samples:
        if s["station_key"] == station_key:
            doys.setdefault(s["year"], []).append(s["doy"])
    cand = {y: v for y, v in doys.items() if year_range[0] <= y <= year_range[1]} or doys
    if not cand:
        return None
    def filled(y):
        a = np.array(sorted(cand[y]))
        return sum(int(np.abs(a - d).min() <= max_gap) for d in season_doys)
    return max(cand, key=lambda y: (filled(y), y))


def infer_year(model, ds, station_key: str, year: int, device, batch_size: int = 16):
    """Centre-pixel prediction + observation for every sample day of the year.

    Gives the temporal half of the figure at ALL THREE depths, including depths the
    station never measured (the parquet only keeps rows with a non-NaN observation,
    eval_predict.py:110, so it cannot show those).
    """
    import torch
    idxs = [i for i, s in enumerate(ds.samples)
            if s["station_key"] == station_key and s["year"] == year]
    if not idxs:
        return None
    idxs.sort(key=lambda i: ds.samples[i]["doy"])

    doys, preds, obs = [], [], []
    for start in range(0, len(idxs), batch_size):
        items = [ds[i] for i in idxs[start:start + batch_size]]
        batch = {}
        for k in items[0]:
            v = items[0][k]
            batch[k] = (torch.stack([it[k] for it in items]).to(device)
                        if isinstance(v, torch.Tensor) else [it[k] for it in items])
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mu = model(batch)
            else:
                mu = model(batch)
        centre = mu[:, :, STATION_RC[0], STATION_RC[1]].float().cpu().numpy()
        preds.append(centre)
        obs.append(np.stack([it["label"].numpy() for it in items]))
        doys.extend(ds.samples[i]["doy"] for i in idxs[start:start + batch_size])
    return {"doy": np.array(doys), "pred": np.concatenate(preds),
            "obs": np.concatenate(obs), "year": year}


def infer_seasons(model, ds, station_key: str, year: int, season_doys: list,
                  device, max_gap: int = 30) -> list:
    """One forward pass per season; keeps the full (3,224,224) field.

    Adapted from plot_spatial_sm_meeting.infer_spatial_for_dates (L163), but also
    returns anchor_rel_pos / anchor_orbit so the scene actually consumed can be shown,
    and the centre pixel so it can be checked against predictions_{split}.parquet.
    """
    import torch
    cands = {ds.samples[i]["doy"]: i for i, s in enumerate(ds.samples)
             if s["station_key"] == station_key and s["year"] == year}
    if not cands:
        return [None] * len(season_doys)
    avail = np.array(sorted(cands))

    results = []
    for doy in season_doys:
        diffs = np.abs(avail - doy)
        if diffs.min() > max_gap:
            results.append(None)
            continue
        best_doy = int(avail[np.argmin(diffs)])
        item = ds[cands[best_doy]]
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]
                 for k, v in item.items()}
        with torch.no_grad():
            if device.type == "cuda":
                # match eval_predict.py:74 so the centre pixel is comparable
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mu = model(batch)
            else:
                mu = model(batch)
        sm = mu[0].float().cpu().numpy()                    # (3, 224, 224)
        results.append({
            "target_doy": doy, "doy": best_doy, "year": year,
            "date": datetime(year, 1, 1) + timedelta(days=best_doy - 1),
            "sm": sm,
            "obs": item["label"].numpy() if "label" in item else None,
            "anchor_rel_pos": int(item["anchor_rel_pos"]),
            "anchor_orbit": int(item["anchor_orbit"]),
        })
    return results


# ── §23.2 figure ─────────────────────────────────────────────────────────────

def station_meta(station_key: str, split: str) -> dict:
    """Identity + §22 skill of the station, for the figure header."""
    import zarr
    m = {"station_key": station_key}
    try:
        ps = pd.read_csv(EVAL_DIR / f"per_station_{split}.csv")
        r = ps[ps["station_key"] == station_key]
        if len(r):
            r = r.iloc[0]
            m.update(lat=r.get("latitude"), lon=r.get("longitude"),
                     igbp=r.get("IGBP"), koppen=r.get("koppen_geiger"),
                     n_years=r.get("n_years"), start=r.get("start_date"),
                     end=r.get("end_date"),
                     ubrmse={d: r.get(f"ubRMSE_{d.replace('-', '_')}")
                             for d in SM_DEPTHS},
                     rr={d: r.get(f"R_{d.replace('-', '_')}") for d in SM_DEPTHS},
                     nobs={d: r.get(f"n_{d.replace('-', '_')}") for d in SM_DEPTHS})
    except Exception:                                             # noqa: BLE001
        pass
    try:
        sp = pd.read_csv(SPLITS_CSV)
        from eval_metrics import _make_key
        k = sp[sp.apply(_make_key, axis=1) == station_key]
        if len(k):
            m["elevation_m"] = float(k.iloc[0].get("elevation_m", np.nan))
            m["network"] = k.iloc[0].get("network")
    except Exception:                                             # noqa: BLE001
        pass
    try:
        a = dict(zarr.open_group(str(SAT_ROOT / f"{station_key}.zarr"), mode="r").attrs)
        m.setdefault("lat", a.get("latitude")); m.setdefault("lon", a.get("longitude"))
        m["epsg"] = a.get("epsg")
        m["network"] = m.get("network") or a.get("network")
    except Exception:                                             # noqa: BLE001
        pass
    return m


def draw_figure(station_key: str, role: str, seasons: list, ctx: dict,
                persist: dict, sigma_time: dict, ts: dict | None, meta: dict,
                out_dir: Path, split: str, epoch: int, run_name: str,
                dpi: int = 300):
    """Spatial and temporal validation of the same station in one figure.

    Rows of maps are seasons on a shared 0-0.5 scale; the panel below is the
    centre-pixel time series for the same year, so every map ties to a point on
    the curve.  The per-panel anomaly block of the first draft is gone -- the
    spatial statistics it carried (sigma, norm-std, spread) are printed on the
    map titles instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    plt.rcParams["text.usetex"] = False
    rows = [s for s in seasons if s is not None]
    n_rows = max(len(rows), 1)

    # row height tracks the panel WIDTH (~12.8/4) so the square maps fill their
    # cells; otherwise equal-aspect imshow leaves a band of white in every row
    h_ctx, h_row, h_ts, h_bot = 3.0, 3.15, 2.5, 3.0
    fig = plt.figure(figsize=(12.8, h_ctx + h_row * n_rows + h_ts + h_bot))
    outer = GridSpec(4, 1, figure=fig,
                     height_ratios=[h_ctx, h_row * n_rows, h_ts, h_bot], hspace=0.24)

    # ── context strip: what structure exists in this footprint ──────────────
    cs = outer[0].subgridspec(1, 4, wspace=0.22)
    lulc_cmap = ListedColormap(plt.get_cmap("tab10").colors[:10])
    panels = [("Sentinel-2 RGB", ctx.get("rgb"), None, None),
              ("NDVI", ctx.get("ndvi"), "RdYlGn", (-0.1, 0.8)),
              ("DEM [m]", ctx.get("dem"), "terrain", None),
              ("Land cover (ESRI v2 → TerraMind idx)", ctx.get("lulc"), None, None)]
    for j, (title, arr, cmap, lim) in enumerate(panels):
        ax = fig.add_subplot(cs[0, j])
        if arr is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8,
                    color="grey", transform=ax.transAxes)
        elif j == 3:
            ax.imshow(arr, cmap=lulc_cmap,
                      norm=BoundaryNorm(np.arange(-0.5, 10.5, 1), 10),
                      interpolation="nearest")
            v, c = np.unique(arr, return_counts=True)
            frac = c / c.sum()
            handles = [Patch(fc=lulc_cmap.colors[int(val)], ec="k", lw=0.3,
                             label=f"{LULC_CLASSES.get(int(val), val)} {f:.0%}")
                       for val, f in sorted(zip(v, frac), key=lambda t: -t[1])
                       if f >= 0.005]
            ax.legend(handles=handles, fontsize=5.5, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            kw = dict(interpolation="nearest")
            if cmap:
                kw["cmap"] = cmap
            if lim:
                kw["vmin"], kw["vmax"] = lim
            im = ax.imshow(arr, **kw)
            if cmap:
                fig.colorbar(im, ax=ax, fraction=0.046,
                             pad=0.02).ax.tick_params(labelsize=6)
        sub = ""
        if j == 2 and ctx.get("dem") is not None:
            sub = (f"\nrelief {np.nanmax(ctx['dem']) - np.nanmin(ctx['dem']):.0f} m, "
                   f"ñ={norm_std(ctx['dem']):.3f}")
        if j == 1 and ctx.get("ndvi") is not None:
            sub = f"\nñ={norm_std(np.nan_to_num(ctx['ndvi'])):.3f}"
        if j == 0 and ctx.get("rgb_date") is not None:
            sub = (f"\n{pd.Timestamp(ctx['rgb_date']).date()} — context only, "
                   "NOT the model input")
        ax.set_title(title + sub, fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(*STATION_RC[::-1], "+", color="cyan", ms=9, mew=1.2)

    # ── season grid: model input + the three depth maps, shared scale ───────
    gs = outer[1].subgridspec(n_rows, 4, wspace=0.08, hspace=0.26)
    for i, r in enumerate(rows):
        a = r["anchor"]
        ax = fig.add_subplot(gs[i, 0])
        if a.get("img") is None:
            ax.text(0.5, 0.5, "no scene", ha="center", va="center", fontsize=8,
                    color="grey", transform=ax.transAxes)
        elif a["kind"] == "s2":
            ax.imshow(a["img"], interpolation="nearest")
        else:
            ax.imshow(a["img"], cmap="gray", vmin=-20, vmax=0, interpolation="nearest")
        cf = a.get("cloud_frac", float("nan"))
        ax.set_title(f"{SEASON_NAMES.get(r['target_doy'], r['target_doy'])}  "
                     f"{r['date'].date()}\nMODEL INPUT: {a['kind'].upper()} "
                     f"{pd.Timestamp(a['date']).date()}, lag {a['lag_days']} d"
                     + (f", cloud {cf:.2f}" if np.isfinite(cf) else ""), fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(*STATION_RC[::-1], "+", color="cyan", ms=9, mew=1.2)

        for j, depth in enumerate(SM_DEPTHS):
            sm, st = r["sm"][j], r["stats"][depth]
            axm = fig.add_subplot(gs[i, 1 + j])
            im = axm.imshow(sm, cmap="YlGnBu", vmin=SM_VMIN, vmax=SM_VMAX,
                            interpolation="nearest")
            axm.set_title(f"{DEPTH_LABELS[depth]}   mean {st['mean']:.3f}\n"
                          f"σ {st['sigma']:.4f} · ñ {st['norm_std']:.3f} · "
                          f"Δ {st['delta_p2_p98']:.3f}",
                          fontsize=6.5, color=DEPTH_COLORS[depth])
            axm.set_xticks([]); axm.set_yticks([])
            axm.plot(*STATION_RC[::-1], "+", color="cyan", ms=7, mew=1.0)
            if j == 2:
                fig.colorbar(im, ax=axm, fraction=0.046, pad=0.02,
                             label="SM [m$^3$/m$^3$]").ax.tick_params(labelsize=6)

    # ── time series of the station pixel, same year ─────────────────────────
    tsg = outer[2].subgridspec(1, 3, wspace=0.16)
    for j, depth in enumerate(SM_DEPTHS):
        ax = fig.add_subplot(tsg[0, j])
        if ts is None:
            ax.text(0.5, 0.5, "no inference sweep", ha="center", va="center",
                    fontsize=8, color="grey", transform=ax.transAxes)
            continue
        dates = pd.to_datetime([f"{ts['year']}{d:03d}" for d in ts["doy"]],
                               format="%Y%j")
        p, o = ts["pred"][:, j].astype(float), ts["obs"][:, j].astype(float)
        pl = p.copy()
        gap = np.append(np.diff(ts["doy"]) > 15, False)     # do not bridge long gaps
        pl[gap] = np.nan
        ax.plot(dates, pl, "-", lw=1.1, color=DEPTH_COLORS[depth], label="predicted")
        fin = np.isfinite(o)
        if fin.any():
            ax.plot(dates[fin], o[fin], ".", ms=2.2, color="black", label="observed")
            e = p[fin] - o[fin]
            ub = float(np.sqrt(np.mean((e - e.mean()) ** 2)))
            rr = (float(np.corrcoef(p[fin], o[fin])[0, 1])
                  if fin.sum() > 2 and o[fin].std() > 0 else float("nan"))
            ax.text(0.02, 0.97, f"ubRMSE {ub:.4f}  r {rr:.2f}  bias {e.mean():+.4f}  "
                    f"n {int(fin.sum())}", transform=ax.transAxes, va="top",
                    fontsize=6, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        else:
            ax.text(0.02, 0.97, "no in-situ sensor at this depth —\nprediction shown, "
                    "unvalidated", transform=ax.transAxes, va="top", fontsize=6,
                    color="#8b1a1a",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        for r in rows:                                       # tie maps to the curve
            ax.axvline(pd.Timestamp(r["date"]), color="grey", lw=0.7, ls=":")
            # season tags go ABOVE the axes -- inside they sit on top of the curve
            ax.annotate(SEASON_NAMES.get(r["target_doy"], ""),
                        xy=(pd.Timestamp(r["date"]), 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.5, color="grey")
        ax.set_ylim(0, SM_VMAX)
        ax.set_ylabel(f"{DEPTH_LABELS[depth]}\nSM [m$^3$/m$^3$]",
                      color=DEPTH_COLORS[depth], fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(alpha=0.3, lw=0.4)
        if j == 0:
            ax.legend(fontsize=6, loc="lower left", ncol=2, framealpha=0.85,
                      borderpad=0.2, handlelength=1.2)
        ax.set_title(f"model at the station pixel (112,112), {ts['year']} — "
                     f"season markers tie each map above to this curve"
                     if j == 1 else "", fontsize=8, pad=12)

    # ── bottom band: heterogeneity vs reference | glossary | summary ────────
    bs = outer[3].subgridspec(1, 3, width_ratios=[1.0, 0.95, 1.15], wspace=0.22)

    ax1 = fig.add_subplot(bs[0, 0])
    xs = np.arange(len(rows))
    for depth in SM_DEPTHS:
        ax1.plot(xs, [r["stats"][depth]["norm_std"] for r in rows], "-o", ms=4,
                 color=DEPTH_COLORS[depth], label=DEPTH_LABELS[depth])
    ax1.axhline(TIER1_NORM_STD, ls="--", lw=1.0, color="grey")
    ax1.text(0.01, TIER1_NORM_STD * 1.15, "§16.1 earlier model 0.0065", fontsize=6,
             color="grey", transform=ax1.get_yaxis_transform())
    ax1.axhline(CAPACITY_CEILING, ls="--", lw=1.0, color="#2e7d32")
    ax1.text(0.01, CAPACITY_CEILING * 1.15, "§16.4 dense-supervision ceiling 0.2504",
             fontsize=6, color="#2e7d32", transform=ax1.get_yaxis_transform())
    ax1.set_yscale("log"); ax1.set_xticks(xs)
    ax1.set_xticklabels([SEASON_NAMES.get(r["target_doy"], str(r["target_doy"]))
                         for r in rows], fontsize=7)
    ax1.set_ylabel("ñ  normalised spatial std", fontsize=7.5)
    ax1.tick_params(labelsize=6); ax1.grid(alpha=0.3, lw=0.4)
    ax1.legend(fontsize=6, frameon=False, ncol=3)

    ax2 = fig.add_subplot(bs[0, 1]); ax2.axis("off")
    ax2.text(0, 1, "SYMBOLS\n"
             "σ       spatial standard deviation of the predicted map [m³/m³]\n"
             "σ_time  temporal std of the station-pixel prediction, this year\n"
             "ñ       normalised spatial std = σ / |mean of map|  [–]\n"
             "Δ       spread of the map, 98th minus 2nd percentile [m³/m³]\n"
             "r       Pearson correlation coefficient  [–]\n"
             "14²     14×14 token grid = model bottleneck, ≈160 m pixels\n"
             "var kept at 14²   share of map variance surviving\n"
             "                  block-averaging to that grid\n"
             "r(season~season)  similarity of the spatial pattern between\n"
             "                  seasons: 1 = one fixed pattern, 0 = unrelated\n"
             "anchor, lag       the satellite acquisition the model actually\n"
             "                  consumed, and its age in days\n"
             "ubRMSE  unbiased RMSE = √(MSE − bias²)  [m³/m³]\n"
             "SM      volumetric soil moisture  [m³/m³]",
             va="top", ha="left", fontsize=6, family="monospace",
             transform=ax2.transAxes)

    ax3 = fig.add_subplot(bs[0, 2]); ax3.axis("off")
    lines = ["SPATIAL SUMMARY", ""]
    for depth in SM_DEPTHS:
        ss = np.nanmean([r["stats"][depth]["sigma"] for r in rows])
        stime = sigma_time.get(depth, float("nan"))
        ratio = ss / stime if np.isfinite(stime) and stime > 0 else float("nan")
        rd = np.nanmean([r["stats"][depth].get("r_dem_14", np.nan) for r in rows])
        rn = np.nanmean([r["stats"][depth].get("r_ndvi_14", np.nan) for r in rows])
        vr = np.nanmean([r["stats"][depth]["var_retained_14x14"] for r in rows])
        pr = (persist or {}).get(depth, {}).get("mean_pairwise_r", float("nan"))
        lines.append(f"{DEPTH_LABELS[depth]:>9s}  σ {ss:.4f}  σ_time {stime:.4f}"
                     f"  σ/σ_time {ratio:.2f}")
        lines.append(f"           r(DEM,14²) {rd:+.2f}   r(NDVI,14²) {rn:+.2f}")
        lines.append(f"           var kept at 14² {vr:.0%}   "
                     f"r(season~season) {pr:+.2f}")
    xd = (persist or {}).get("cross_depth_0-10_vs_30-100", float("nan"))
    lines += ["", f"cross-depth r(0-10, 30-100) {xd:+.2f} — architectural",
              "(star residual, model.py:276), not learned depth structure.", "",
              "HOW TO READ",
              "r(DEM/NDVI) ≈ 0  → the structure is not the terrain or",
              "                   the vegetation of this footprint",
              "r(season~season) → 1  → one pattern, mean-shifted",
              "var kept at 14² ≈ 100%  → no detail below ≈160 m",
              "σ/σ_time ≈ 1  → across 2.2 km the model varies as much",
              "                as it does across the whole year"]
    ax3.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=6,
             family="monospace", transform=ax3.transAxes)

    # ── header and caption ──────────────────────────────────────────────────
    ig = IGBP_NAMES.get(str(meta.get("igbp")), meta.get("igbp"))
    kg = KOPPEN_NAMES.get(str(meta.get("koppen")), meta.get("koppen"))
    lat, lon = meta.get("lat"), meta.get("lon")
    elev = meta.get("elevation_m")
    sk = meta.get("ubrmse", {}) or {}
    skill = "  ".join(f"{DEPTH_LABELS[d]} ubRMSE {sk[d]:.4f}"
                      for d in SM_DEPTHS if sk.get(d) == sk.get(d) and sk.get(d) is not None)
    head2 = (f"{meta.get('network', '?')} network · "
             f"{lat:.4f}°N, {lon:.4f}°E" if isinstance(lat, float) else "")
    if isinstance(elev, float) and np.isfinite(elev):
        head2 += f" · {elev:.0f} m a.s.l."
    head2 += f" · IGBP {meta.get('igbp')} ({ig}) · Köppen {meta.get('koppen')} ({kg})"
    if meta.get("n_years") == meta.get("n_years") and meta.get("n_years") is not None:
        head2 += f" · {meta['n_years']:.0f} yr record"
    fig.suptitle(f"§23  Spatial and temporal validation — {station_key}  [{role}]\n"
                 + head2 + ("\n" + skill if skill else ""),
                 y=1.0, fontsize=10)

    yr = ts["year"] if ts else (rows[0]["year"] if rows else "?")
    caption = (
        f"Predicted soil moisture for {station_key} ({split.upper()} split, checkpoint "
        f"{run_name} epoch {epoch}), year {yr}.  Top: what the 2.24 × 2.24 km footprint "
        f"contains — Sentinel-2 RGB, NDVI, terrain and ESRI land cover (stored as "
        f"TerraMind indices; ESRI v1 'Grass' and 'Scrub/Shrub' are merged into "
        f"Rangeland).  Middle: the acquisition the model actually consumed and the "
        f"predicted 224 × 224 field at each depth, all on one 0–0.5 m³/m³ scale; the "
        f"anchor may pre-date the prediction by up to 364 days, so its lag is stated.  "
        f"Then: the model's own time series at the station pixel for the same year, "
        f"with the four map dates marked.  Bottom: how much the field varies in space, "
        f"against the two reference levels of §16.1 and §16.4, the symbol list, and "
        f"whether the spatial variation corresponds to anything physical.  The cyan "
        f"cross marks the station pixel (112,112) — the only pixel the training loss "
        f"ever sees.")
    fig.text(0.01, -0.015, caption, ha="left", va="top", fontsize=6.5, wrap=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{station_key}_heterogeneity.{ext}", dpi=dpi,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out_dir}/{station_key}_heterogeneity.png")


# ── §23.6 verification ───────────────────────────────────────────────────────

def selftest() -> bool:
    """A synthetic checkerboard must produce the statistics numpy says it does.

    Catches "the panel is flat because imshow got the wrong array" -- the failure mode
    that would make the whole figure a lie in the safe direction.
    """
    x = np.indices((224, 224)).sum(axis=0) // 16 % 2
    sm = 0.20 + 0.05 * x.astype(np.float64)
    st = panel_stats(sm, dem=None, ndvi=None)
    ok = True
    for name, got, want in (("sigma", st["sigma"], sm.std()),
                            ("mean", st["mean"], sm.mean()),
                            ("norm_std", st["norm_std"], sm.std() / abs(sm.mean())),
                            ("delta", st["delta_p2_p98"],
                             np.percentile(sm, 98) - np.percentile(sm, 2))):
        if not np.isclose(got, want, rtol=1e-10, atol=1e-12):
            print(f"  SELFTEST FAIL {name}: {got} != {want}")
            ok = False
    flat = panel_stats(np.full((224, 224), 0.2), None, None)
    if flat["sigma"] != 0.0 or flat["norm_std"] != 0.0:
        print("  SELFTEST FAIL: uniform field did not give sigma 0")
        ok = False
    # a 16x16 checkerboard is exactly representable on the 14x14 grid? no --
    # block_mean over 16x16 blocks collapses it, so var retained must be ~0
    if st["var_retained_14x14"] > 0.05:
        print(f"  SELFTEST FAIL: var_retained {st['var_retained_14x14']:.3f} "
              "should be ~0 for a 16 px checkerboard")
        ok = False
    print(f"  selftest {'PASS' if ok else 'FAIL'}")
    return ok


def verify_against_parquet(records: list, parquet: Path, tol: float) -> bool:
    """Centre pixel of the rendered map vs the value eval_predict.py stored.

    NOT bit-exact: eval_predict.py:74 ran autocast(bf16) at batch 128, this runs batch 1
    and kernel selection is batch-size dependent.  |Δ| <= tol (default 2e-3, ~2.5 bf16
    ulp at 0.2).  Rows with NaN obs were dropped by eval_predict.py:110, so a missing
    key is a skip-with-warning, not a failure.
    """
    df = pd.read_parquet(parquet)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    key = df.set_index(["station_key", "year", "doy", "depth"])["pred"]
    worst, n_ok, n_skip = 0.0, 0, 0
    for r in records:
        for depth, val in r["centre"].items():
            k = (r["station_key"], r["year"], r["doy"], depth)
            if k not in key.index:
                n_skip += 1
                continue
            d = abs(float(key.loc[k]) - val)
            worst = max(worst, d)
            n_ok += 1
    print(f"  verify: {n_ok} matched, {n_skip} absent from parquet (NaN obs), "
          f"max |Δ| = {worst:.2e} (tol {tol:g})")
    if worst > tol:
        print("  VERIFY FAIL -- check checkpoint, sample index, dropout, category_filter")
    return worst <= tol


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",        default="oos")
    ap.add_argument("--run-name",     default="cls_depth_star_reg")
    ap.add_argument("--ckpt",         default="best.pt")
    ap.add_argument("--n-stations",   type=int, default=3)
    ap.add_argument("--station",      nargs="+", default=None)
    ap.add_argument("--rank-depth",   default="0-10", choices=SM_DEPTHS)
    ap.add_argument("--year",         type=int, default=None)
    ap.add_argument("--season-doys",  nargs="+", type=int, default=[15, 105, 196, 288])
    ap.add_argument("--max-gap",      type=int, default=30)
    ap.add_argument("--min-n",        type=int,   default=700)
    ap.add_argument("--min-r",        type=float, default=0.6)
    ap.add_argument("--min-obs-std",  type=float, default=0.03)
    ap.add_argument("--exclude-igbp", nargs="+",  default=["WAT", "SNO"])
    ap.add_argument("--control-station", default="ISMN_SCAN_Crossroads")
    ap.add_argument("--out",          default=str(OUT_DIR))
    ap.add_argument("--dpi",          type=int, default=300)
    ap.add_argument("--dry-run-selection", action="store_true")
    ap.add_argument("--selftest",     action="store_true")
    ap.add_argument("--verify-against", default=None)
    ap.add_argument("--tol",          type=float, default=2e-3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.station:
        sel = pd.DataFrame({"station_key": args.station, "split": args.split,
                            "role": "explicit"})
    else:
        sel, shortlist = select_stations(
            args.split, args.n_stations, args.min_n, args.min_r, args.min_obs_std,
            args.exclude_igbp, args.control_station, args.rank_depth)
        report_selection(sel, shortlist, args.rank_depth)
    sel.to_csv(out_dir / "selection.csv", index=False)
    print(f"\n→ {out_dir/'selection.csv'}")

    if args.dry_run_selection:
        print("\n--dry-run-selection: stopping before any model work.")
        return

    if args.selftest and not selftest():
        raise SystemExit("selftest failed -- not producing figures")

    # heavy imports only past this point
    import tempfile
    import torch
    from ckpt_utils import load_checkpoint
    from dataset import SoilMoistureDataset
    import tier1_probe

    if hasattr(tier1_probe, "patch_open_zarr_no_marker"):
        tier1_probe.patch_open_zarr_no_marker()      # held-out stores lack .complete

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch = load_checkpoint(CKPT_ROOT / args.run_name / args.ckpt, device)
    print(f"Checkpoint {args.run_name}/{args.ckpt} epoch {epoch} on {device}")

    # subset the splits CSV so dataset init only opens the stations we need
    splits_df = pd.read_csv(SPLITS_CSV)
    from eval_metrics import _make_key
    keys = splits_df.apply(_make_key, axis=1)
    wanted = set(sel["station_key"])
    matched = splits_df[keys.isin(wanted)]
    if matched.empty:
        raise SystemExit(f"none of {sorted(wanted)} found in {SPLITS_CSV}")
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    matched.to_csv(tmp.name, index=False); tmp.close()
    print(f"Station filter: {len(matched)} row(s) → temp splits CSV")

    ds = SoilMoistureDataset(
        splits_csv=tmp.name, era5_stats_path=str(ERA5_STATS),
        years=list(range(2016, 2024)),
        category_filter=cfg.get("category_filter", ["sm_only"]),
        split_filter=None, training=False, use_mmap=True)

    # sigma_time: temporal spread of the centre-pixel prediction, for sigma_space/sigma_time
    pq = pd.read_parquet(EVAL_DIR / f"predictions_{args.split}.parquet")
    pq["year"] = pd.to_datetime(pq["date"]).dt.year

    all_metrics, records = {}, []
    for _, row in sel.iterrows():
        station, role = row["station_key"], row.get("role", "")
        print(f"\n[{role}] {station}")
        year = args.year or choose_year(ds, station, args.season_doys, args.max_gap)
        if year is None:
            print("  no samples -- skipping")
            continue
        print(f"  year {year}")

        seasons = infer_seasons(model, ds, station, year, args.season_doys, device,
                                args.max_gap)
        got = [s for s in seasons if s is not None]
        if not got:
            print("  no season within --max-gap -- skipping")
            continue
        print(f"  {len(got)}/{len(args.season_doys)} season slots filled")

        ctx = context_layers(station, got[len(got) // 2]["date"])
        for r in got:
            r["anchor"] = anchor_scene(station, r["date"], r["anchor_rel_pos"],
                                       r["anchor_orbit"])
            r["stats"] = {d: panel_stats(r["sm"][j], ctx.get("dem"), ctx.get("ndvi"))
                          for j, d in enumerate(SM_DEPTHS)}
            print(f"    {SEASON_NAMES.get(r['target_doy'], r['target_doy'])} "
                  f"{r['date'].date()}  anchor {pd.Timestamp(r['anchor']['date']).date()} "
                  f"lag {r['anchor']['lag_days']}d  "
                  + "  ".join(f"{d} ñ={r['stats'][d]['norm_std']:.4f}"
                              for d in SM_DEPTHS))
            records.append({"station_key": station, "year": r["year"], "doy": r["doy"],
                            "centre": {d: r["stats"][d]["centre"] for d in SM_DEPTHS}})

        g = pq[(pq["station_key"] == station) & (pq["year"] == year)]
        sigma_time = {d: float(g[g["depth"] == d]["pred"].std()) for d in SM_DEPTHS}

        persist = pattern_persistence(got)
        print("    pattern persistence r(season~season): "
              + "  ".join(f"{d} {persist[d]['mean_pairwise_r']:+.2f}" for d in SM_DEPTHS)
              + f"   cross-depth {persist['cross_depth_0-10_vs_30-100']:+.2f}")

        ts = infer_year(model, ds, station, year, device)
        if ts is not None:
            print(f"    time series: {len(ts['doy'])} days, "
                  + "  ".join(f"{d} obs {int(np.isfinite(ts['obs'][:, j]).sum())}"
                              for j, d in enumerate(SM_DEPTHS)))
        meta = station_meta(station, args.split)

        draw_figure(station, role, got, ctx, persist, sigma_time, ts, meta,
                    out_dir, args.split, int(epoch), args.run_name, args.dpi)
        all_metrics[station] = {
            "role": role, "year": year, "epoch": int(epoch),
            "run_name": args.run_name, "split": args.split,
            "sigma_time": sigma_time,
            "pattern_persistence": persist,
            "context": {"n_lulc": ctx.get("n_lulc"),
                        "dem_norm_std": (norm_std(ctx["dem"]) if ctx.get("dem") is not None
                                         else None),
                        "ndvi_norm_std": (norm_std(np.nan_to_num(ctx["ndvi"]))
                                          if ctx.get("ndvi") is not None else None)},
            "seasons": [{"season": SEASON_NAMES.get(r["target_doy"], r["target_doy"]),
                         "date": str(r["date"].date()), "doy": r["doy"],
                         "anchor": {"kind": r["anchor"]["kind"],
                                    "date": str(pd.Timestamp(r["anchor"]["date"]).date()),
                                    "lag_days": r["anchor"]["lag_days"],
                                    "exact": bool(r["anchor"]["exact"]),
                                    "cloud_frac": r["anchor"]["cloud_frac"]},
                         "stats": r["stats"]} for r in got],
        }

    meta = {"reference": {"tier1_norm_std": TIER1_NORM_STD,
                          "capacity_ceiling": CAPACITY_CEILING},
            "stations": all_metrics}
    with open(out_dir / "heterogeneity_metrics.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\n→ {out_dir/'heterogeneity_metrics.json'}")

    if args.verify_against:
        verify_against_parquet(records, Path(args.verify_against), args.tol)


if __name__ == "__main__":
    main()
