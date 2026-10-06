"""§70 effective resolution, CPU half: Tests 0, A, B, C on the effres_maps.py output.

    Test 0  observed semivariogram over TxSON station pairs -> the unpredictable floor (nugget)
    Test A  pair-difference skill: does the model get Δ(SM) between two stations right?
            within-tile pairs (same forward pass) and cross-tile pairs (centre pixels), by distance
    Test B  input coarsening: how much does the map move when the within-tile spatial inputs
            are block-averaged at b x 20 m?  (and does station accuracy change?)
    Test C  output coarsening: block-average the plain map around the station pixel; the
            largest block that does not make station accuracy significantly worse is L_C.
            Also: spatial variance retained per block and the radially averaged spectrum of the
            time-anomaly maps (the tile's time-mean map, §56's fixed pattern, removed first).
    Control a synthetic field (Gaussian sigma 160 m) through the Test C machinery.

Truth is held-out stations only: tiles AND stations with split == "oos" (current §47 split).
Writes CSVs + summary.json to <in-dir>/analysis/.

    python effres_analysis.py --in-dir eval_output/effres_txson
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from combine_network import load_observations
from splits_config import category_of, station_dir_name

REPO   = Path(__file__).resolve().parent
SPLITS = REPO / "csvs" / "station_splits.csv"
DEPTHS = ["0-10", "10-30", "30-100"]
MAP_PX, PX_M = 112, 20
BLOCKS = [1, 2, 4, 8, 14, 28, 56, 112]            # Test C, in 20 m map pixels
MIN_DAYS = 60                                       # common days for a pair / station to count
NBOOT = 1000
WITHIN_BINS = [0, 500, 1000, 1600]                  # m; max within-tile separation ~1.6 km
CROSS_BINS  = [0, 1600, 3000, 6000, 12000, 25000, 50000]
VARIO_BINS  = [0, 250, 500, 1000, 1600, 3000, 6000, 12000, 25000, 50000]


# ── helpers ─────────────────────────────────────────────────────────────────────────────────
def hav(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return float(2 * 6_371_000 * np.arcsin(np.sqrt(a)))


def ubrmse(p, o) -> float:
    e = (p - p.mean()) - (o - o.mean())
    return float(np.sqrt((e ** 2).mean()))


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(groups: np.ndarray, stat, rng, nboot=NBOOT):
    """Cluster bootstrap: resample unique `groups` with replacement; stat(mask_rows) -> float."""
    ug = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(nboot):
        pick = rng.choice(ug, len(ug), replace=True)
        v = stat(np.concatenate([rows[g] for g in pick]))
        if np.isfinite(v):
            vals.append(v)
    return (np.nan, np.nan) if len(vals) < nboot // 2 else tuple(np.percentile(vals, [2.5, 97.5]))


def to_date(year, doy):
    return pd.to_datetime([datetime(int(y), 1, 1) + timedelta(days=int(d) - 1)
                           for y, d in zip(year, doy)])


# ── Test 0 ──────────────────────────────────────────────────────────────────────────────────
def test0(obs: pd.DataFrame, coords: dict) -> pd.DataFrame:
    rows = []
    for dep in DEPTHS:
        wide = obs[obs["depth"] == dep].pivot_table(index="date", columns="station", values="obs")
        for a, b in combinations(wide.columns, 2):
            m = wide[[a, b]].dropna()
            if len(m) < MIN_DAYS:
                continue
            xa, xb = m[a].to_numpy(), m[b].to_numpy()
            rows.append(dict(depth=dep, a=a, b=b, dist_m=hav(*coords[a], *coords[b]), n=len(m),
                             g_level=0.5 * (xa.mean() - xb.mean()) ** 2,
                             g_anom=0.5 * (((xa - xa.mean()) - (xb - xb.mean())) ** 2).mean(),
                             var_anom=0.5 * (xa.var() + xb.var())))
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df["dist_m"], VARIO_BINS)
    return df


# ── Test A ──────────────────────────────────────────────────────────────────────────────────
def pair_table(ro: pd.DataFrame, group_col: str, coords: dict) -> pd.DataFrame:
    """Pairs of stations inside each group (tile for within-tile; one group for cross-tile)."""
    rows = []
    for (g, dep), sub in ro.groupby([group_col, "depth"]):
        wp = sub.pivot_table(index="date", columns="station", values="pred")
        wo = sub.pivot_table(index="date", columns="station", values="obs")
        for a, b in combinations(sorted(wo.columns), 2):
            m = pd.concat([wp[a], wp[b], wo[a], wo[b]], axis=1, keys=["pa", "pb", "oa", "ob"]).dropna()
            if len(m) < MIN_DAYS:
                continue
            dp, do = m["pa"] - m["pb"], m["oa"] - m["ob"]
            rows.append(dict(group=g, depth=dep, a=a, b=b, n=len(m),
                             dist_m=hav(*coords[a], *coords[b]),
                             dpred_level=float(dp.mean()), dobs_level=float(do.mean()),
                             r_dyn=corr(dp, do)))
    return pd.DataFrame(rows)


def summarise_pairs(pt: pd.DataFrame, bins, rng) -> pd.DataFrame:
    out = []
    if pt.empty:
        return pd.DataFrame(columns=["depth", "median_dist_m", "r_level_lo", "r_dyn_lo"])
    pt = pt.assign(bin=pd.cut(pt["dist_m"], bins))
    for (dep, bn), s in pt.groupby(["depth", "bin"], observed=True):
        g = s["group"].to_numpy()
        lv_p, lv_o, rd = s["dpred_level"].to_numpy(), s["dobs_level"].to_numpy(), s["r_dyn"].to_numpy()
        r_lev = corr(lv_p, lv_o)
        ci_lev = boot_ci(g, lambda i: corr(lv_p[i], lv_o[i]), rng)
        r_dyn = float(np.nanmedian(rd))
        ci_dyn = boot_ci(g, lambda i: float(np.nanmedian(rd[i])), rng)
        out.append(dict(depth=dep, bin=str(bn), n_pairs=len(s), n_groups=len(np.unique(g)),
                        median_dist_m=float(s["dist_m"].median()),
                        r_level=r_lev, r_level_lo=ci_lev[0], r_level_hi=ci_lev[1],
                        sd_dobs_level=float(lv_o.std()), sd_dpred_level=float(lv_p.std()),
                        r_dyn_median=r_dyn, r_dyn_lo=ci_dyn[0], r_dyn_hi=ci_dyn[1]))
    return pd.DataFrame(out)


# ── Test C (per tile, in a worker) ──────────────────────────────────────────────────────────
_G = {}


def _init(maps_path):
    _G["maps"] = np.load(maps_path, mmap_mode="r")


def block_mean(a: np.ndarray, b: int) -> np.ndarray:
    """(..., 112, 112) -> (..., 112/b, 112/b) block means."""
    n = MAP_PX // b
    return a.reshape(*a.shape[:-2], n, b, n, b).mean(axis=(-3, -1))


def radial_spectrum(a: np.ndarray) -> np.ndarray:
    """(T, 112, 112) anomaly maps -> mean radially averaged power, index k = 1..56 cycles/tile."""
    f = np.abs(np.fft.fft2(a - a.mean(axis=(-2, -1), keepdims=True))) ** 2
    ky, kx = np.meshgrid(np.fft.fftfreq(MAP_PX) * MAP_PX, np.fft.fftfreq(MAP_PX) * MAP_PX,
                         indexing="ij")
    k = np.rint(np.hypot(ky, kx)).astype(int)
    p = f.mean(0)
    return np.array([p[k == i].mean() for i in range(1, MAP_PX // 2 + 1)])


def window_series(maps: np.ndarray, r: int, c: int, b: int) -> np.ndarray:
    """(T, 112, 112) -> (T,) mean over the b x b window centred on (r, c), clipped to the tile."""
    r0 = max(0, min(MAP_PX - b, r - b // 2)); c0 = max(0, min(MAP_PX - b, c - b // 2))
    return maps[:, r0:r0 + b, c0:c0 + b].mean(axis=(-2, -1))


def tile_c(job):
    tile, idx, dates, obs_by_depth = job
    m = np.asarray(_G["maps"][idx], np.float32)                     # (T, 3, 112, 112)
    var_rows, spec_rows, acc_rows = [], [], []
    for d, dep in enumerate(DEPTHS):
        x = m[:, d]
        anom = x - x.mean(0, keepdims=True)                         # §56 fixed pattern removed
        v1 = float(anom.var(axis=(-2, -1)).mean())
        for b in BLOCKS:
            vb = float(block_mean(anom, b).var(axis=(-2, -1)).mean()) if b < MAP_PX else 0.0
            var_rows.append(dict(tile=tile, depth=dep, block=b, L_m=b * PX_M,
                                 var_full=v1, var_blocks=vb,
                                 frac_retained=vb / v1 if v1 > 0 else np.nan,
                                 sd_timemean_map=float(x.mean(0).std())))
        for k, pw in enumerate(radial_spectrum(anom), start=1):
            spec_rows.append(dict(tile=tile, depth=dep, k=k,
                                  wavelength_m=MAP_PX * PX_M / k, power=float(pw)))
        o = obs_by_depth.get(dep)
        if o is None:
            continue
        sel = dates.isin(o.index)
        if sel.sum() < MIN_DAYS:
            continue
        ov = o.loc[dates[sel]].to_numpy()
        for b in BLOCKS:
            p = window_series(x[sel], MAP_PX // 2, MAP_PX // 2, b)
            acc_rows.append(dict(tile=tile, depth=dep, block=b, L_m=b * PX_M, n=int(sel.sum()),
                                 ubRMSE=ubrmse(p, ov), r=corr(p, ov),
                                 bias=float((p - ov).mean())))
    return var_rows, spec_rows, acc_rows


def l_from_accuracy(acc: pd.DataFrame, rng) -> pd.DataFrame:
    """Paired cluster bootstrap over tiles of Δ ubRMSE (block b minus block 1)."""
    out = []
    for dep, s in acc.groupby("depth"):
        w = s.pivot_table(index="tile", columns="block", values="ubRMSE").dropna()
        base = w[1].to_numpy()
        for b in BLOCKS:
            dlt = w[b].to_numpy() - base
            vals = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(NBOOT)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out.append(dict(depth=dep, block=b, L_m=b * PX_M, n_tiles=len(dlt),
                            ubRMSE_mean=float(w[b].mean()), d_ubRMSE=float(dlt.mean()),
                            d_lo=float(lo), d_hi=float(hi), worse=bool(lo > 0)))
    return pd.DataFrame(out)


def largest_ok(df: pd.DataFrame, ok_col: str) -> dict:
    """Per depth: largest block b such that every block <= b is ok."""
    res = {}
    for dep, s in df.sort_values("block").groupby("depth"):
        L = None
        for _, r in s.iterrows():
            if not r[ok_col]:
                break
            L = int(r["L_m"])
        res[dep] = L
    return res


# ── Control ─────────────────────────────────────────────────────────────────────────────────
def control(rng) -> pd.DataFrame:
    """Perfect model of a field smoothed with a Gaussian of sigma 160 m (8 px), plus 20 m
    prediction noise and obs noise. L_C must come out at a few hundred m, not 20 m or 2 km."""
    from scipy.ndimage import gaussian_filter
    rows = []
    for t in range(30):
        T = 300
        f = gaussian_filter(rng.standard_normal((T, MAP_PX, MAP_PX)), sigma=(0, 8, 8), mode="wrap")
        f /= f.std()
        pred = 0.25 + 0.03 * (f + 0.3 * rng.standard_normal(f.shape))
        obs = 0.25 + 0.03 * f[:, MAP_PX // 2, MAP_PX // 2] + 0.005 * rng.standard_normal(T)
        for b in BLOCKS:
            p = window_series(pred, MAP_PX // 2, MAP_PX // 2, b)
            rows.append(dict(tile=f"syn{t}", depth="0-10", block=b, L_m=b * PX_M,
                             ubRMSE=ubrmse(p, obs), r=corr(p, obs)))
    return pd.DataFrame(rows)


# ── main ────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(REPO / "eval_output" / "effres_txson"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.in_dir); out = d / "analysis"; out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = {}

    splits = pd.read_csv(SPLITS)
    splits["key"] = splits.apply(station_dir_name, axis=1)
    cur = dict(zip(splits["key"], splits["split"]))
    cat_map = dict(zip(splits["key"], splits.apply(category_of, axis=1)))
    px = pd.read_csv(d / "readouts_pixels.csv")
    coords = {r["station"]: (r["lat"], r["lon"]) for _, r in px.iterrows()}
    stations = sorted(px["station"].unique())

    obs = load_observations(stations, cat_map)
    obs["date"] = pd.to_datetime(obs["date"])
    print(f"obs: {len(obs):,} rows, {obs['station'].nunique()} stations")

    # Test 0 -- all TxSON stations: observations do not depend on the model's split
    t0 = test0(obs, coords)
    t0.to_csv(out / "test0_pairs.csv", index=False)
    v = t0.groupby(["depth", "bin"], observed=True).agg(
        n_pairs=("n", "size"), g_level=("g_level", "mean"), g_anom=("g_anom", "mean"),
        var_anom=("var_anom", "mean")).reset_index()
    v["g_anom_frac"] = v["g_anom"] / v["var_anom"]
    v.to_csv(out / "test0_variogram.csv", index=False)
    print("\nTest 0 -- observed semivariogram (g_anom_frac = Δ-anomaly variance / anomaly variance)")
    print(v.to_string(index=False))

    # Test A
    ro0 = pd.read_parquet(d / "readouts_L0.parquet")
    ro0["date"] = to_date(ro0["year"], ro0["doy"])
    ro0["tile_split"] = ro0["tile"].map(cur)
    ro0["station_split"] = ro0["station"].map(cur)
    held = ro0[(ro0["tile_split"] == "oos") & (ro0["station_split"] == "oos")]
    held = held.merge(obs, on=["station", "date", "depth"], how="left")
    within = pair_table(held, "tile", coords)
    within.to_csv(out / "testA_within_pairs.csv", index=False)
    sa_w = summarise_pairs(within, WITHIN_BINS, rng)
    sa_w.to_csv(out / "testA_within.csv", index=False)
    cross = pair_table(held[held["is_centre"]].assign(all="all"), "all", coords)
    if not cross.empty:
        cross["group"] = cross["a"]                          # cluster bootstrap on station a
    cross.to_csv(out / "testA_cross_pairs.csv", index=False)
    sa_c = summarise_pairs(cross, CROSS_BINS, rng)
    sa_c.to_csv(out / "testA_cross.csv", index=False)
    print("\nTest A -- within-tile pairs\n" + sa_w.to_string(index=False))
    print("\nTest A -- cross-tile pairs (centre pixels)\n" + sa_c.to_string(index=False))
    la = {}
    for dep, s in pd.concat([sa_w, sa_c]).groupby("depth"):
        sig = s[(s["r_level_lo"] > 0) | (s["r_dyn_lo"] > 0)].sort_values("median_dist_m")
        la[dep] = float(sig["median_dist_m"].iloc[0]) if len(sig) else None
    summary["L_A_m"] = la

    # Test B
    rows_b = []
    base0 = ro0[["idx", "station", "depth", "pred"]].rename(columns={"pred": "pred0"})
    hc = held[held["is_centre"]]
    for f in sorted(d.glob("summary_L*.parquet"), key=lambda p: int(p.stem.split("L")[1])):
        b = int(f.stem.split("L")[1])
        sm = pd.read_parquet(f)
        idx_oos = ro0.loc[ro0["tile_split"] == "oos", "idx"].unique()
        sm = sm[sm["idx"].isin(idx_oos)]
        rb = pd.read_parquet(d / f"readouts_L{b}.parquet").merge(base0, on=["idx", "station", "depth"])
        rb["tile_split"] = rb["tile"].map(cur); rb["station_split"] = rb["station"].map(cur)
        rb = rb[(rb["tile_split"] == "oos") & (rb["station_split"] == "oos")]
        rb["date"] = to_date(rb["year"], rb["doy"])
        for dep in DEPTHS:
            s = sm[sm["depth"] == dep]; r = rb[rb["depth"] == dep]
            spread = r.groupby("idx")["pred0"].transform(lambda x: x - x.mean())
            c = r[r["is_centre"]].merge(obs, on=["station", "date", "depth"]).dropna(subset=["obs"])
            acc = c.groupby("station").apply(
                lambda g: pd.Series(dict(ub_b=ubrmse(g["pred"].to_numpy(), g["obs"].to_numpy()),
                                         ub_0=ubrmse(g["pred0"].to_numpy(), g["obs"].to_numpy())))
                if len(g) >= MIN_DAYS else pd.Series(dict(ub_b=np.nan, ub_0=np.nan)))
            rows_b.append(dict(
                depth=dep, block=b, L_m=b * PX_M,
                S_map=float(np.median(s["rms_diff"] / s["sd_L0"].clip(lower=1e-6))),
                rms_diff_map=float(s["rms_diff"].median()),
                sd_map_L0=float(s["sd_L0"].median()), sd_map_Lb=float(s["sd_Lb"].median()),
                rms_diff_pix=float(np.sqrt(((r["pred"] - r["pred0"]) ** 2).mean())),
                sd_within_tile_pix=float(np.sqrt((spread ** 2).mean())),
                sd_time_pix=float(r.groupby("station")["pred0"].std().mean()),
                ubRMSE_centre_b=float(acc["ub_b"].mean()), ubRMSE_centre_0=float(acc["ub_0"].mean())))
    tb = pd.DataFrame(rows_b)
    if not tb.empty:
        tb["ok"] = tb["S_map"] < 0.1
        tb.to_csv(out / "testB.csv", index=False)
        print("\nTest B -- input coarsening\n" + tb.to_string(index=False))
        summary["L_B_m"] = largest_ok(tb, "ok")

    # Test C
    idx = pd.read_parquet(d / "index_L0.parquet")
    idx["date"] = to_date(idx["year"], idx["doy"])
    jobs = []
    for tile, g in idx.groupby("tile"):
        if cur.get(tile) != "oos":
            continue
        o = obs[obs["station"] == tile]
        obd = {dep: s.set_index("date")["obs"] for dep, s in o.groupby("depth")}
        jobs.append((tile, g["idx"].to_numpy(), pd.DatetimeIndex(g["date"]), obd))
    with Pool(min(args.workers, len(jobs)), initializer=_init,
              initargs=(str(d / "maps_L0.npy"),)) as pool:
        res = pool.map(tile_c, jobs)
    var = pd.DataFrame([r for v_, _, _ in res for r in v_])
    spec = pd.DataFrame([r for _, s_, _ in res for r in s_])
    acc = pd.DataFrame([r for _, _, a_ in res for r in a_])
    var.to_csv(out / "testC_variance_tiles.csv", index=False)
    spec.to_csv(out / "testC_spectrum_tiles.csv", index=False)
    acc.to_csv(out / "testC_accuracy_tiles.csv", index=False)
    vs = var.groupby(["depth", "block", "L_m"])[["frac_retained", "var_full", "sd_timemean_map"]] \
        .median().reset_index()
    vs.to_csv(out / "testC_variance.csv", index=False)
    spec.groupby(["depth", "k", "wavelength_m"])["power"].median().reset_index() \
        .to_csv(out / "testC_spectrum.csv", index=False)
    lc = l_from_accuracy(acc, rng)
    lc["ok"] = ~lc["worse"]
    lc.to_csv(out / "testC_accuracy.csv", index=False)
    print("\nTest C -- spatial anomaly variance retained\n" + vs.to_string(index=False))
    print("\nTest C -- station accuracy vs block size\n" + lc.to_string(index=False))
    summary["L_C_m"] = largest_ok(lc, "ok")

    # Control
    ctl = l_from_accuracy(control(rng), rng)
    ctl["ok"] = ~ctl["worse"]
    ctl.to_csv(out / "control.csv", index=False)
    summary["control_L_C_m (field sigma 160 m)"] = largest_ok(ctl, "ok")
    print("\nControl -- synthetic sigma-160 m field\n" + ctl.to_string(index=False))

    summary["L_eff_m"] = {dep: max([x for x in (summary.get(k, {}).get(dep)
                                               for k in ("L_A_m", "L_B_m", "L_C_m"))
                                    if x is not None], default=None)
                          for dep in DEPTHS}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
