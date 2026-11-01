"""§32.8 — the sufficiency gate: does terrain explain soil moisture where it should?

This is the science go/no-go. No GPU, no model: observed labels against derived terrain.

WHY PAIRS AND NOT A CROSS-STATION SCATTER
A regression of station-mean SM on station HAND across all 993 stations confounds
terrain with climate, soil, land cover and sensor. Colocated pairs difference all of
that out: two stations 200 m apart share rainfall, soil parent material, land cover
class and network protocol, so a difference in their soil moisture that tracks a
difference in their HAND is attributable to terrain in a way a cross-station slope
never is. This is why §26 went looking for colocated pairs in the first place.

WHY WET/DRY AND WHY CLIMATE
Saturation excess is a wet-state mechanism: lateral subsurface flow converges toward
low HAND and keeps those positions wet. It needs water to operate. The first two
networks looked at directly contradict each other, and in exactly the way that
predicts:

    TxSON (semi-arid Texas, 40 stations)  SM 0-10 vs HAND  r = +0.085  (wrong sign)
    HOBE  (humid Denmark,   5 stations)   SM 0-10 vs HAND  r = -0.637  (right sign,
                                          stronger when wet, stronger at 10-30 cm)

So a pooled global null would be the wrong test — it would average a real effect in
humid climates against its absence in arid ones and report nothing. Everything here
is stratified by Koppen macro-climate and split wet/dry.

WHAT WOULD MAKE THIS A PASS
A negative dSM-vs-dHAND slope in humid strata, stronger in the wet state than the dry
state, consistent in sign across pairs rather than driven by a few. A null everywhere
including humid, wet-state, deeper layers is a fail, and per §32.8 that prunes ONE
INPUT — ablation row 2 (Block 3 on, terrain off) isolates the architecture from the
hydrology, so a terrain failure does not kill the per-location rebuild.

CAVEAT CARRIED THROUGHOUT: the §32.6 MERIT gate has NOT yet run, so no station's
accumulation has been validated against MERIT upa. HAND is far less exposed to that
than TWI (it is a local elevation difference, not an upslope integral), but this
analysis is provisional until §32.6 passes.

Outputs
    csvs/colocated_pairs.csv     §32.7 step 7 — persisted, with separations
    csvs/gate_station_table.csv  terrain + SM + climate per station
    csvs/gate_results.json       every statistic quoted
    figures/gate_sm_vs_terrain.png

Usage
    conda activate terramind
    python gate_sm_vs_terrain.py --workers 64
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
import rasterio
from matplotlib.gridspec import GridSpec
from pyproj import Transformer
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

REPO         = Path(__file__).resolve().parent
TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
ZARR_ROOT    = Path("/gpfs/scratch1/shared/pkhanal/zarr")
OUT_DIR      = REPO / "figures"

BANDS       = ["twi", "hand", "acc_cells_mfd", "slope_rad", "valid"]
DEPTHS      = ["0-10", "10-30", "30-100"]
PAIR_CUTOFF_M = 1120.0     # §26/§31.8's definition of colocated
MIN_DAYS      = 180        # a station needs a real record to contribute a mean
MIN_COMMON    = 120        # a pair needs overlapping observations to be differenced


# ── station-level collection, parallel over regions ──────────────────────────

def _open_labels(sid: str):
    import zarr
    for cat in ("sm_only", "sm_and_flux", "flux_only"):
        p = ZARR_ROOT / cat / sid
        if not p.exists():
            continue
        try:
            z = zarr.open_consolidated(str(p), mode="r")
        except Exception:
            try:
                z = zarr.open_group(str(p), mode="r")
            except Exception:
                return None
        return z["labels"] if "labels" in list(z.group_keys()) else None
    return None


def station_series(sid: str) -> dict[str, pd.Series]:
    """
    Observed daily SM per depth, gap-fills excluded (qc == 0 only).

    Defensive throughout: a labels group can exist without an `sm` array at all —
    flux-only stations carry fluxes and no soil moisture — and one such station
    inside a Pool worker takes down the entire run. Returns {} for anything it
    cannot read rather than raising.
    """
    try:
        L = _open_labels(sid)
    except Exception:
        return {}
    if L is None:
        return {}
    keys = set(L.array_keys())
    if not {"sm", "depths", "dates"} <= keys:
        return {}
    try:
        sm = L["sm"][:]
        depths = [str(d) for d in L["depths"][:]]
        dates = pd.to_datetime([str(d) for d in L["dates"][:]], format="%Y%m%d")
        qc = L["qc"][:] if "qc" in keys else None
    except Exception:
        return {}
    if sm.ndim != 2 or sm.shape[0] != len(depths) or sm.shape[1] != len(dates):
        return {}
    out = {}
    for i, d in enumerate(depths):
        v = sm[i].astype(float)
        # qc is sometimes stored over a different window than sm; only apply it when
        # the two align, rather than misaligning them silently
        if qc is not None and qc.shape[1] == v.shape[0]:
            v = np.where(qc[i] == 0, v, np.nan)
        ok = np.isfinite(v)
        if ok.sum() >= MIN_DAYS:
            out[d] = pd.Series(v[ok], index=dates[ok])
    return out


def collect_region(args_t):
    """One region: open its terrain raster once, pull every station inside it."""
    rid, stations = args_t
    p = TERRAIN_ROOT / f"region_{rid:04d}" / "terrain_30m.tif"
    if not p.exists():
        return []
    with rasterio.open(p) as src:
        arr = src.read()
        tr = src.transform
    res = float(tr.a)
    F = {b: arr[i] for i, b in enumerate(BANDS)}
    h, w = F["hand"].shape

    recs = []
    for s in stations:
        col = int((s["laea_x"] - tr.c) / res)
        row = int((tr.f - s["laea_y"]) / res)
        if not (0 <= row < h and 0 <= col < w):
            continue
        rec = {"station_id": s["station_id"], "region_id": rid,
               "hand": float(F["hand"][row, col]),
               "twi": float(F["twi"][row, col]),
               "slope": float(F["slope_rad"][row, col]),
               "acc_cells": float(F["acc_cells_mfd"][row, col])}
        # local terrain contrast inside the 2.24 km tile: if HAND is flat across the
        # tile there is nothing for a per-location model to use even if HAND matters
        half = int(1120 / res)
        r0, r1 = max(row - half, 0), min(row + half + 1, h)
        c0, c1 = max(col - half, 0), min(col + half + 1, w)
        rec["hand_tile_sd"] = float(np.nanstd(F["hand"][r0:r1, c0:c1]))
        rec["twi_tile_sd"] = float(np.nanstd(F["twi"][r0:r1, c0:c1]))

        ser = station_series(s["station_id"])
        if not ser:
            continue
        surf = ser.get("0-10")
        for d in DEPTHS:
            if d in ser:
                rec[f"sm_{d}"] = float(ser[d].mean())
                rec[f"n_{d}"] = int(len(ser[d]))
        if surf is not None:
            q1, q2 = surf.quantile([1 / 3, 2 / 3])
            rec["sm_dry"] = float(surf[surf <= q1].mean())
            rec["sm_wet"] = float(surf[surf >= q2].mean())
            rec["sm_sd"] = float(surf.std())
        recs.append(rec)
    return recs


# ── pairs ────────────────────────────────────────────────────────────────────

def find_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Colocated pairs by ECEF chord distance — antimeridian and poles need no case."""
    to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x, y, z = to_ecef.transform(df["longitude"].to_numpy(),
                                df["latitude"].to_numpy(),
                                np.zeros(len(df)))
    pts = np.column_stack([x, y, z])
    tree = cKDTree(pts)
    rows = []
    for i, j in tree.query_pairs(PAIR_CUTOFF_M):
        rows.append({"station_a": df.iloc[i]["station_id"],
                     "station_b": df.iloc[j]["station_id"],
                     "sep_m": float(np.linalg.norm(pts[i] - pts[j]))})
    return pd.DataFrame(rows).sort_values("sep_m").reset_index(drop=True)


def pair_deltas(pairs: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """
    Per pair: dSM over the COMMON observed dates, and dTerrain.

    Common dates matter. Two stations observed in different years differ in soil
    moisture because the years differed, not because the terrain did; differencing
    station means over disjoint periods would inject that straight into the response.
    """
    idx = st.set_index("station_id")
    cache: dict[str, dict] = {}
    out = []
    for _, p in pairs.iterrows():
        a, b = p["station_a"], p["station_b"]
        if a not in idx.index or b not in idx.index:
            continue
        for s in (a, b):
            if s not in cache:
                cache[s] = station_series(s)
        rec = {"station_a": a, "station_b": b, "sep_m": p["sep_m"],
               "d_hand": idx.loc[a, "hand"] - idx.loc[b, "hand"],
               "d_twi": idx.loc[a, "twi"] - idx.loc[b, "twi"],
               "d_slope": idx.loc[a, "slope"] - idx.loc[b, "slope"],
               "kg_macro": idx.loc[a, "kg_macro"],
               "region_id": idx.loc[a, "region_id"]}
        got = False
        for d in DEPTHS:
            sa, sb = cache[a].get(d), cache[b].get(d)
            if sa is None or sb is None:
                continue
            common = sa.index.intersection(sb.index)
            if len(common) < MIN_COMMON:
                continue
            da, db = sa.loc[common], sb.loc[common]
            rec[f"d_sm_{d}"] = float((da - db).mean())
            rec[f"n_common_{d}"] = int(len(common))
            if d == "0-10":
                mean_state = (da + db) / 2.0
                q1, q2 = mean_state.quantile([1 / 3, 2 / 3])
                rec["d_sm_wet"] = float((da - db)[mean_state >= q2].mean())
                rec["d_sm_dry"] = float((da - db)[mean_state <= q1].mean())
                # sign consistency in time: a real terrain effect holds day to day
                rec["frac_days_same_sign"] = float(
                    np.mean(np.sign(da - db) == np.sign((da - db).mean())))
            got = True
        if got:
            out.append(rec)
    return pd.DataFrame(out)


# ── statistics ───────────────────────────────────────────────────────────────

def corr_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5:
        return {"r": np.nan, "p": np.nan, "n": n, "slope": np.nan}
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    from math import erfc, sqrt
    t = abs(r) * np.sqrt(n - 2) / np.sqrt(max(1 - r * r, 1e-12))
    return {"r": r, "p": float(erfc(t / sqrt(2))), "n": n,
            "slope": float(np.polyfit(x[m], y[m], 1)[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description="§32.8 sufficiency gate")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out-prefix", default="gate")
    args = ap.parse_args()

    sdr = pd.read_csv(REPO / "csvs" / "station_dem_region.csv")
    sp = pd.read_csv(REPO / "csvs" / "station_splits.csv")
    sp["sid"] = sp.apply(
        lambda r: (f"{r.source_network}_{r.network}_{r.station_id}"
                   if r.source_network != r.network else f"{r.network}_{r.station_id}"),
        axis=1)
    meta = sp[["sid", "koppen_geiger", "kg_macro", "IGBP", "igbp_macro",
               "elevation_m", "network", "split"]]

    groups = [(rid, g.to_dict("records"))
              for rid, g in sdr.groupby("region_id")]
    print(f"collecting {len(sdr)} stations over {len(groups)} regions "
          f"with Pool({args.workers})", flush=True)
    with Pool(args.workers) as pool:
        recs = [r for chunk in pool.imap_unordered(collect_region, groups) for r in chunk]
    st = pd.DataFrame(recs)
    st = st.merge(sdr[["station_id", "latitude", "longitude", "laea_x", "laea_y"]],
                  on="station_id", how="left")
    st = st.merge(meta, left_on="station_id", right_on="sid", how="left").drop(columns=["sid"])
    print(f"  {len(st)} stations with terrain AND >= {MIN_DAYS} observed days")

    st.to_csv(REPO / "csvs" / f"{args.out_prefix}_station_table.csv", index=False)

    # ── pairs ────────────────────────────────────────────────────────────────
    pairs = find_pairs(sdr)
    pairs.to_csv(REPO / "csvs" / "colocated_pairs.csv", index=False)
    bins = [(0, 50), (50, 160), (160, 500), (500, 1120)]
    hist = {f"{lo}-{hi}m": int(((pairs.sep_m >= lo) & (pairs.sep_m < hi)).sum())
            for lo, hi in bins}
    n_st = len(set(pairs.station_a) | set(pairs.station_b))
    print(f"  {len(pairs)} colocated pairs across {n_st} stations   {hist}")
    print("  (§31.8 recorded 62 as 13/4/16/29; the difference is that §26 additionally "
          "required overlapping observation periods — enforced below via MIN_COMMON)")

    pd_ = pair_deltas(pairs, st)
    print(f"  {len(pd_)} pairs survive the >= {MIN_COMMON} common-observed-day rule")

    # ── the gate ─────────────────────────────────────────────────────────────
    res: dict = {"n_stations": len(st), "n_pairs_geometric": len(pairs),
                 "n_pairs_usable": len(pd_), "pair_separation_hist": hist,
                 "merit_gate_run": False}

    def block(name, d, xcol, ycol):
        r = corr_p(d[xcol], d[ycol])
        res[name] = r
        star = "  <<<" if (np.isfinite(r["p"]) and r["p"] < 0.05) else ""
        print(f"    {name:<44} r = {r['r']:+.3f}  p = {r['p']:.3f}  n = {r['n']}{star}")
        return r

    print("\n  PAIRWISE GATE — dSM on dTerrain, station identity differenced out")
    for d in DEPTHS:
        c = f"d_sm_{d}"
        if c in pd_ and pd_[c].notna().sum() >= 5:
            block(f"dSM({d}) ~ dHAND", pd_, "d_hand", c)
            block(f"dSM({d}) ~ dTWI", pd_, "d_twi", c)
    if "d_sm_wet" in pd_:
        block("dSM(wet third) ~ dHAND", pd_, "d_hand", "d_sm_wet")
        block("dSM(dry third) ~ dHAND", pd_, "d_hand", "d_sm_dry")

    print("\n  BY KOPPEN MACRO-CLIMATE (dSM 0-10 ~ dHAND)")
    for kg, g in pd_.groupby("kg_macro"):
        if g["d_sm_0-10"].notna().sum() >= 5:
            block(f"kg {kg}", g, "d_hand", "d_sm_0-10")

    print("\n  CROSS-STATION, WITHIN NETWORK (weaker: confounded, shown for context)")
    for kg, g in st.groupby("kg_macro"):
        if len(g) >= 8 and "sm_0-10" in g:
            block(f"cross-station kg {kg}", g, "hand", "sm_0-10")

    # how much terrain contrast is even available inside a tile?
    res["tile_contrast"] = {
        "hand_tile_sd_median": float(st["hand_tile_sd"].median()),
        "twi_tile_sd_median": float(st["twi_tile_sd"].median()),
        "hand_tile_sd_p10": float(st["hand_tile_sd"].quantile(0.10)),
    }
    print(f"\n  within-tile terrain contrast: HAND sd median "
          f"{res['tile_contrast']['hand_tile_sd_median']:.2f} m, "
          f"p10 {res['tile_contrast']['hand_tile_sd_p10']:.2f} m")

    (REPO / "csvs" / f"{args.out_prefix}_results.json").write_text(
        json.dumps(res, indent=2, default=float))
    pd_.to_csv(REPO / "csvs" / f"{args.out_prefix}_pair_deltas.csv", index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5})
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    def sc(ax, d, xcol, ycol, title, colour="#2c7fb8"):
        v = d.dropna(subset=[xcol, ycol])
        ax.scatter(v[xcol], v[ycol], s=34, c=colour, edgecolor="k", linewidth=0.35)
        r = corr_p(v[xcol], v[ycol])
        if np.isfinite(r["r"]):
            xs = np.linspace(v[xcol].min(), v[xcol].max(), 20)
            ax.plot(xs, np.polyval(np.polyfit(v[xcol], v[ycol], 1), xs),
                    color="0.25", ls="--", lw=1.3)
            ax.set_title(f"{title}   r = {r['r']:+.3f}, p = {r['p']:.3f}, n = {r['n']}")
        else:
            ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.axvline(0, color="k", lw=0.6, alpha=0.5)
        ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 0])
    sc(ax, pd_, "d_hand", "d_sm_0-10", "A  all pairs, 0-10 cm")
    ax.set_xlabel("$\\Delta$HAND (m)"); ax.set_ylabel("$\\Delta$SM 0-10 (m$^3$/m$^3$)")

    ax = fig.add_subplot(gs[0, 1])
    for kg, g in pd_.groupby("kg_macro"):
        v = g.dropna(subset=["d_hand", "d_sm_0-10"])
        if len(v) >= 3:
            ax.scatter(v["d_hand"], v["d_sm_0-10"], s=34, label=f"{kg} (n={len(v)})",
                       edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("$\\Delta$HAND (m)"); ax.set_ylabel("$\\Delta$SM 0-10 (m$^3$/m$^3$)")
    ax.set_title("B  by Koppen macro-climate — the TxSON/HOBE contrast, globally")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 0])
    if "d_sm_wet" in pd_:
        v = pd_.dropna(subset=["d_hand", "d_sm_wet", "d_sm_dry"])
        ax.scatter(v["d_hand"], v["d_sm_wet"], s=34, c="#2166ac", edgecolor="k",
                   linewidth=0.35, label="wettest third")
        ax.scatter(v["d_hand"], v["d_sm_dry"], s=34, c="#b2182b", marker="s",
                   edgecolor="k", linewidth=0.35, label="driest third")
        for col, cl in (("d_sm_wet", "#2166ac"), ("d_sm_dry", "#b2182b")):
            if len(v) > 3:
                xs = np.linspace(v["d_hand"].min(), v["d_hand"].max(), 20)
                ax.plot(xs, np.polyval(np.polyfit(v["d_hand"], v[col], 1), xs),
                        color=cl, ls="--", lw=1.3)
        ax.legend(fontsize=7.5)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("$\\Delta$HAND (m)"); ax.set_ylabel("$\\Delta$SM 0-10 (m$^3$/m$^3$)")
    ax.set_title("C  wet vs dry state (§31.5)")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    lines = [f"§32.8 SUFFICIENCY GATE — {len(st)} stations, {len(pd_)} usable pairs", ""]
    for k in ("dSM(0-10) ~ dHAND", "dSM(10-30) ~ dHAND", "dSM(30-100) ~ dHAND",
              "dSM(0-10) ~ dTWI", "dSM(wet third) ~ dHAND", "dSM(dry third) ~ dHAND"):
        if k in res:
            v = res[k]
            lines.append(f"  {k:<26} r={v['r']:+.3f} p={v['p']:.3f} n={v['n']}")
    lines += ["", "by Koppen macro-climate:"]
    for k, v in res.items():
        if isinstance(k, str) and k.startswith("kg "):
            lines.append(f"  {k:<26} r={v['r']:+.3f} p={v['p']:.3f} n={v['n']}")
    lines += ["",
              f"within-tile HAND sd: median "
              f"{res['tile_contrast']['hand_tile_sd_median']:.2f} m",
              "",
              "PROVISIONAL: the §32.6 MERIT gate has not run,",
              "so no station's accumulation is validated yet.",
              "HAND is far less exposed to that than TWI.",
              "",
              "Negative slope = higher above drainage is drier,",
              "which is what saturation excess predicts."]
    ax.text(0.0, 0.99, "\n".join(lines), va="top", ha="left", fontsize=7.6,
            family="monospace", transform=ax.transAxes)

    fig.suptitle("§32.8 sufficiency gate — observed soil moisture against derived terrain",
                 fontsize=11, y=0.975)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.out_prefix}_sm_vs_terrain"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"\nwrote {out}.png, csvs/colocated_pairs.csv, "
          f"csvs/{args.out_prefix}_results.json")


if __name__ == "__main__":
    main()
