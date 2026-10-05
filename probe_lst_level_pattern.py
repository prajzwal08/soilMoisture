"""
probe_lst_level_pattern.py — LST LEVEL (tile mean − T2m) vs the within-tile PATTERN, against SM
===============================================================================================

§52 question: should the thermal head also be supervised on the LEVEL (alpha > 0)? Today it
learns only the centred 22x22 pattern (model.lst_pattern_loss). This probe uses OBSERVED data
only (Landsat lst22 target, ERA5 t2m, in-situ SM) — no model — and asks which of the two
carries soil-moisture information.

Per Landsat scene (>= 10 valid cells, the lst_pattern_stats floor):
    L      = mean LST over valid cells                       (K, the level the loss discards)
    dT     = L − t2m_mean, and L − t2m_max                   (K, level with air temp removed)
    P_stn  = LST(station cell) − L                           (K, the pattern at the station)
    pat_sd = within-tile SD                                   (K)
    sm_*   = observed SM on that day (qc == 0 only; gap-filled days dropped)

PART A — all train+val stations, 2016-2022 (oos is never looked at, so no design leak):
    per-station r(X, SM) for X in {L, dT_mean, dT_max, P_stn}, raw and deseasonalised
    (per-station monthly means removed from both), plus magnitudes and pattern stability.

PART B — TxSON (all 40, includes oos: a data diagnostic, not a model score):
    B1 within the CR200-18 tile: the other TxSON stations that fall inside its 22x22 grid,
       their cell's pattern value vs their SM (per date, and time-mean).
    B2 across the network (~36 km): per-date spatial anomalies of dT and P_stn vs SM.

Station cell: the tile's west/north edge is 1120 m from the station and cells are 100 m, so
the station sits in cell (11, 11) (landsat_target.py).

Outputs: csvs/probe_lst_level_pattern/*.csv, figures/probe_lst_level_pattern/*.png, stdout.
CPU only; run through slurm/probe_lst_level_pattern.sh, never on the login node.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import (ERA5_VARS, SM_DEPTHS, _load_lst22, _load_zarr_era5, _load_zarr_labels,
                     _open_zarr)
from landsat_target import HALF_TILE_M, OUT_N, OUT_RES_M, utm_epsg
from splits_config import TRAIN_YEARS, category_of, station_dir_name

STN_CELL    = int(HALF_TILE_M // OUT_RES_M)      # 11
MIN_CELLS   = 10
MIN_SCENES  = 20        # per station, scenes with both X and SM, for a per-station r
QC_OBSERVED = 0
I_TMEAN, I_TMAX = ERA5_VARS.index("t2m_mean"), ERA5_VARS.index("t2m_max")
SIGNALS = ["L", "dT_mean", "dT_max", "P_stn"]
TXSON_REF = "ISMN_TxSON_CR200-18"

OUT_CSV = Path("csvs/probe_lst_level_pattern")
OUT_FIG = Path("figures/probe_lst_level_pattern")


# ── per station ──────────────────────────────────────────────────────────────

def one(task):
    """Scene table for one station (+ its centred fields if keep_fields). Never raises."""
    d, cat, keep_fields = task
    try:
        lst = _load_lst22(cat, d)
        if lst is None:
            return d, "no_lst22", None, None
        idx, arr = lst
        zg = _open_zarr(Path(d), cat)
        if zg is None:
            return d, "no_zarr", None, None
        era, lab = _load_zarr_era5(zg), _load_zarr_labels(zg, strict=False)
        if era is None or lab is None:
            return d, "no_era5_or_labels", None, None
        vals, dints, _ = era
        erow = {int(x): i for i, x in enumerate(dints)}
        sm, depths, times, qc = lab
        if qc is None:
            return d, "no_qc", None, None                       # fail closed, as dataset.py
        tint = times.year * 10000 + times.month * 100 + times.day
        lrow = {int(t): i for i, t in enumerate(tint)}
        dpos = {str(x): i for i, x in enumerate(depths)}

        recs, fields = [], {}
        for date, i in idx.items():
            if date // 10000 not in TRAIN_YEARS:
                continue
            f = np.asarray(arr[i], dtype=np.float32)
            v = np.isfinite(f)
            if v.sum() < MIN_CELLS:
                continue
            L = float(f[v].mean())
            c = f - L
            r = dict(station=d, date=date, L=L, n_cells=int(v.sum()),
                     pat_sd=float(c[v].std()),
                     P_stn=float(c[STN_CELL, STN_CELL]) if v[STN_CELL, STN_CELL] else np.nan)
            j = erow.get(date)
            r["t2m_mean"] = float(vals[j, I_TMEAN]) if j is not None else np.nan
            r["t2m_max"]  = float(vals[j, I_TMAX])  if j is not None else np.nan
            k = lrow.get(date)
            for dep in SM_DEPTHS:
                di = dpos.get(dep)
                ok = k is not None and di is not None and qc[di, k] == QC_OBSERVED
                r[f"sm_{dep}"] = float(sm[di, k]) if ok else np.nan
            recs.append(r)
            if keep_fields:
                fields[date] = c
        if not recs:
            return d, "no_scenes", None, None
        df = pd.DataFrame(recs)
        df["dT_mean"] = df["L"] - df["t2m_mean"]
        df["dT_max"]  = df["L"] - df["t2m_max"]

        # pattern stability (§29 found +0.967): each scene's centred field vs the station mean
        cube = np.stack([np.asarray(arr[idx[x]], np.float32) - l
                         for x, l in zip(df["date"], df["L"])])
        mp = np.nanmean(cube, axis=0)
        rs = []
        for c in cube:
            m = np.isfinite(c) & np.isfinite(mp)
            if m.sum() >= MIN_CELLS and c[m].std() > 0 and mp[m].std() > 0:
                rs.append(np.corrcoef(c[m], mp[m])[0, 1])
        df["pat_stab_r"] = np.median(rs) if rs else np.nan
        return d, "ok", df, (fields if keep_fields else None)
    except Exception as e:                                     # report, don't kill the pool
        return d, f"error:{type(e).__name__}:{e}", None, None


# ── statistics ───────────────────────────────────────────────────────────────

def deseason(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove each station's monthly mean from every column (months with < 2 scenes dropped)."""
    out = df.copy()
    out["month"] = (out["date"] // 100) % 100
    g = out.groupby(["station", "month"])
    n = g["date"].transform("size")
    for c in cols:
        out[c] = out[c] - g[c].transform("mean")
    return out[n >= 2]


def per_station_r(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    def _r(g):
        m = g[x].notna() & g[y].notna()
        if m.sum() < MIN_SCENES or g.loc[m, x].std() == 0 or g.loc[m, y].std() == 0:
            return np.nan
        return np.corrcoef(g.loc[m, x], g.loc[m, y])[0, 1]
    return df.groupby("station").apply(_r)


def summarise_r(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return dict(n=0)
    return dict(n=len(r), median=r.median(), q25=r.quantile(.25), q75=r.quantile(.75),
                frac_neg=(r < 0).mean(), frac_abs_gt_0p2=(r.abs() > 0.2).mean())


def spatial_anom(df: pd.DataFrame, cols: list[str], min_st: int) -> pd.DataFrame:
    """Per-date across-station anomalies (value − that date's network mean)."""
    out = df.copy()
    n = out.groupby("date")["station"].transform("size")
    out = out[n >= min_st].copy()
    for c in cols:
        out[c] = out[c] - out.groupby("date")[c].transform("mean")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global OUT_CSV, OUT_FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-stations", type=int, default=None, help="smoke test")
    ap.add_argument("--out-suffix", default="",
                    help="write to csvs/probe_lst_level_pattern<suffix>/ (and figures/...) so a "
                         "full run does not overwrite the 2026-09-30 60-station outputs")
    args = ap.parse_args()
    OUT_CSV = OUT_CSV.with_name(OUT_CSV.name + args.out_suffix)
    OUT_FIG = OUT_FIG.with_name(OUT_FIG.name + args.out_suffix)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv("csvs/station_splits.csv")          # pandas: quoted commas (memory)
    sp = sp[sp["has_soil_moisture"].astype(str).str.lower() == "true"].copy()
    sp["dir"] = sp.apply(station_dir_name, axis=1)
    sp["cat"] = sp.apply(category_of, axis=1)
    sp["txson"] = sp["network"].astype(str) == "TxSON"
    use = sp[sp["split"].isin(["train", "val"]) | sp["txson"]]
    if args.max_stations:
        use = pd.concat([use[~use["txson"]].head(args.max_stations), use[use["txson"]]])
    tasks = [(r.dir, r.cat, bool(r.txson)) for r in use.itertuples()]
    print(f"stations: {len(tasks)} ({int(use['txson'].sum())} TxSON)", flush=True)

    with Pool(args.workers) as pool:
        res = pool.map(one, tasks, chunksize=1)
    status = pd.Series({d: s for d, s, _, _ in res})
    print("status:", status.map(lambda s: s.split(":")[0]).value_counts().to_dict())
    for d, s in status[status.str.startswith("error")].items():
        print(f"  {d}: {s}")
    scenes = pd.concat([df for _, s, df, _ in res if s == "ok"], ignore_index=True)
    fields = {d: f for d, s, _, f in res if s == "ok" and f is not None}
    scenes = scenes.merge(sp[["dir", "split", "network", "latitude", "longitude", "txson"]],
                          left_on="station", right_on="dir", how="left").drop(columns="dir")
    scenes.to_csv(OUT_CSV / "scenes.csv", index=False)
    print(f"scenes: {len(scenes)} over {scenes['station'].nunique()} stations; "
          f"t2m missing on {scenes['t2m_mean'].isna().mean():.1%}; observed SM on "
          + ", ".join(f"{d} {scenes[f'sm_{d}'].notna().mean():.1%}" for d in SM_DEPTHS),
          flush=True)
    if scenes["sm_0-10"].notna().sum() == 0:
        raise SystemExit("no observed SM matched any scene -- depth names or qc alignment wrong")

    # ── PART A ───────────────────────────────────────────────────────────────
    A = scenes[scenes["split"].isin(["train", "val"])]
    print("\n=== PART A: train+val stations, 2016-2022 ===")
    mag = A.groupby("station").agg(sd_L=("L", "std"), sd_dT=("dT_mean", "std"),
                                   sd_Pstn=("P_stn", "std"), pat_sd=("pat_sd", "mean"),
                                   pat_stab=("pat_stab_r", "first"), n=("date", "size"))
    Ad = deseason(A, SIGNALS + [f"sm_{d}" for d in SM_DEPTHS])
    mag["sd_dT_des"] = Ad.groupby("station")["dT_mean"].std()
    mag["sd_Pstn_des"] = Ad.groupby("station")["P_stn"].std()
    print("magnitudes, median over stations (K):")
    print(mag.median(numeric_only=True).round(3).to_string())

    rows, rA = [], {}
    for dep in SM_DEPTHS:
        for x in SIGNALS:
            for kind, frame in (("raw", A), ("deseason", Ad)):
                r = per_station_r(frame, x, f"sm_{dep}")
                rA[(dep, x, kind)] = r
                rows.append(dict(depth=dep, signal=x, kind=kind, **summarise_r(r)))
    tabA = pd.DataFrame(rows)
    tabA.to_csv(OUT_CSV / "partA_r_summary.csv", index=False)
    pd.DataFrame({f"{d}|{x}|{k}": r for (d, x, k), r in rA.items()}).to_csv(
        OUT_CSV / "partA_r_per_station.csv")
    mag.to_csv(OUT_CSV / "partA_magnitudes.csv")
    print("\nper-station r(signal, SM), median [IQR], frac<0, frac|r|>0.2:")
    print(tabA.round(3).to_string(index=False))

    # ── PART B ───────────────────────────────────────────────────────────────
    T = scenes[scenes["txson"]].copy()
    print(f"\n=== PART B: TxSON, {T['station'].nunique()} stations ===")
    from pyproj import Transformer
    tx = sp[sp["txson"]].set_index("dir")
    ref = tx.loc[TXSON_REF]
    epsg = utm_epsg(ref.latitude, ref.longitude)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = tr.transform(ref.longitude, ref.latitude)
    west, north = cx - HALF_TILE_M, cy + HALF_TILE_M
    inside = []
    for dname, r in tx.iterrows():
        x, y = tr.transform(r.longitude, r.latitude)
        col, row = int(np.floor((x - west) / OUT_RES_M)), int(np.floor((north - y) / OUT_RES_M))
        if 0 <= col < OUT_N and 0 <= row < OUT_N:
            inside.append((dname, row, col))
    print(f"B1: {len(inside)} TxSON stations inside the {TXSON_REF} 22x22 grid:")
    for dname, row, col in inside:
        print(f"    {dname:28s} cell (row {row:2d}, col {col:2d})")

    b1 = []
    ref_fields = fields.get(TXSON_REF, {})
    smT = T.set_index(["station", "date"])["sm_0-10"]
    for date, c in ref_fields.items():
        for dname, row, col in inside:
            b1.append(dict(date=date, station=dname, pat=c[row, col],
                           sm=smT.get((dname, date), np.nan)))
    b1 = pd.DataFrame(b1)
    b1 = b1[b1["pat"].notna() & b1["sm"].notna()] if len(b1) else b1
    b1.to_csv(OUT_CSV / "partB1_cr200-18_tile.csv", index=False)
    b1_rd = pd.Series(dtype=float)
    if len(b1):
        def _rd(g):
            return (np.corrcoef(g["pat"], g["sm"])[0, 1]
                    if len(g) >= 4 and g["pat"].std() > 0 and g["sm"].std() > 0 else np.nan)
        b1_rd = b1.groupby("date").apply(_rd).dropna()
        b1a = b1.copy()
        b1a["sm"] = b1a["sm"] - b1a.groupby("date")["sm"].transform("mean")
        st_mean = b1a.groupby("station")[["pat", "sm"]].mean()
        r_static = (np.corrcoef(st_mean["pat"], st_mean["sm"])[0, 1]
                    if len(st_mean) >= 3 else np.nan)
        print(f"B1 per-date r(pattern at cell, SM 0-10) across in-tile stations: "
              f"{len(b1_rd)} dates, median {b1_rd.median():+.3f}, frac<0 {(b1_rd < 0).mean():.2f}")
        print(f"B1 time-mean across {len(st_mean)} stations: r = {r_static:+.3f}, "
              f"Spearman {st_mean['pat'].rank().corr(st_mean['sm'].rank()):+.3f}")
        # n ~ 6: one leveraged station can make the whole result. Drop the largest |pattern|.
        lev = st_mean["pat"].abs().idxmax()
        b1x = b1[b1["station"] != lev]
        rdx = b1x.groupby("date").apply(_rd).dropna()
        smx = st_mean.drop(index=lev)
        print(f"B1 without {lev}: per-date median r {rdx.median():+.3f} ({len(rdx)} dates), "
              f"frac<0 {(rdx < 0).mean():.2f}; time-mean r "
              f"{np.corrcoef(smx['pat'], smx['sm'])[0, 1]:+.3f} over {len(smx)}")
        nval = pd.Series({d: np.mean([np.isfinite(c[r_, c_]) for c in ref_fields.values()])
                          for d, r_, c_ in inside})
        print("B1 fraction of scenes with a valid cell at each station:")
        print(nval.round(2).to_string())
        print(st_mean.round(3).to_string())

    Ta = spatial_anom(T, ["dT_mean", "P_stn", "L", "sm_0-10"], min_st=10)
    b2 = []
    for x in ["dT_mean", "P_stn"]:
        def _rd(g, x=x):
            m = g[x].notna() & g["sm_0-10"].notna()
            return (np.corrcoef(g.loc[m, x], g.loc[m, "sm_0-10"])[0, 1] if m.sum() >= 10
                    else np.nan)
        rd = Ta.groupby("date").apply(_rd).dropna()
        b2.append(dict(signal=x, n_dates=len(rd), median=rd.median(),
                       frac_neg=(rd < 0).mean()))
    netm = Ta.groupby("station")[["dT_mean", "P_stn", "sm_0-10"]].mean()
    netm = netm.join(tx[["latitude", "longitude", "split"]])
    for x in ["dT_mean", "P_stn"]:
        m = netm[x].notna() & netm["sm_0-10"].notna()
        rs = np.corrcoef(netm.loc[m, x], netm.loc[m, "sm_0-10"])[0, 1]
        print(f"B2 {x}: per-date spatial r median "
              f"{[b for b in b2 if b['signal'] == x][0]['median']:+.3f}; "
              f"time-mean across {m.sum()} stations r = {rs:+.3f}")
    pd.DataFrame(b2).to_csv(OUT_CSV / "partB2_network_r.csv", index=False)
    netm.to_csv(OUT_CSV / "partB2_network_station_means.csv")

    figures(A, Ad, rA, ref_fields, inside, b1, b1_rd, netm, T)
    print(f"\nwrote {OUT_CSV}/ and {OUT_FIG}/")


# ── figures ──────────────────────────────────────────────────────────────────

def figures(A, Ad, rA, ref_fields, inside, b1, b1_rd, netm, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAMES = {"L": "LST tile mean", "dT_mean": "LST tile − T2m (mean)",
             "dT_max": "LST tile − T2m (max)", "P_stn": "pattern at station cell"}
    INK, MUTED = "#222222", "#777777"

    # A1: per-station r distributions, deseasonalised, 0-10 cm
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, x in zip(axes, SIGNALS):
        r = rA[("0-10", x, "deseason")].dropna()
        ax.hist(r, bins=bins, color="#4a6fa5", edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(r.median() if len(r) else 0, color=INK, lw=2)
        ax.set_title(f"{NAMES[x]}\nmedian r {r.median():+.2f}, n={len(r)}", fontsize=10)
        ax.set_xlabel("per-station r with SM 0-10 cm")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("stations")
    fig.suptitle("Part A — within-station r (monthly climatology removed), train+val 2016-2022",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A1_r_hist_deseason_0-10.png", dpi=150)
    plt.close(fig)

    # A2: magnitudes — how big is each signal, in K
    mag = pd.DataFrame({
        "tile mean L (over time)": A.groupby("station")["L"].std(),
        "LST−T2m (over time)": A.groupby("station")["dT_mean"].std(),
        "LST−T2m, deseasonalised": Ad.groupby("station")["dT_mean"].std(),
        "within-tile pattern SD": A.groupby("station")["pat_sd"].mean(),
        "pattern at station, deseas.": Ad.groupby("station")["P_stn"].std(),
    })
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.boxplot([mag[c].dropna() for c in mag], vert=False, labels=mag.columns,
               showfliers=False, medianprops=dict(color=INK, lw=2))
    ax.set_xlabel("SD per station (K)")
    ax.set_title("Part A — size of each thermal signal", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A2_magnitudes.png", dpi=150)
    plt.close(fig)

    # B1: CR200-18 tile — mean pattern + stations coloured by SM spatial anomaly
    if ref_fields and len(b1):
        mp = np.nanmean(np.stack(list(ref_fields.values())), axis=0)
        b1a = b1.copy()
        b1a["sm"] = b1a["sm"] - b1a.groupby("date")["sm"].transform("mean")
        stm = b1a.groupby("station")[["pat", "sm"]].mean()
        pos = {d: (r, c) for d, r, c in inside}
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8),
                                 gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
        lim = np.nanmax(np.abs(mp))
        im = axes[0].imshow(mp, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=axes[0], label="mean LST pattern (K)", fraction=0.046, pad=0.02)
        smax = np.abs(stm["sm"]).max()
        for d, (r, c) in pos.items():
            if d in stm.index:
                sc = axes[0].scatter(c, r, c=[stm.loc[d, "sm"]], cmap="BrBG", vmin=-smax,
                                     vmax=smax, s=110, edgecolors=INK, linewidths=1.2)
                axes[0].annotate(d.split("_")[-1], (c, r), xytext=(5, 5),
                                 textcoords="offset points", fontsize=8, color=INK)
        fig.colorbar(sc, ax=axes[0], label="station SM 0-10 anomaly (m³/m³)",
                     fraction=0.046, pad=0.12)
        axes[0].set_title(f"{TXSON_REF.split('_')[-1]} tile, 2016-22 mean\n"
                          "(white = no retrieval in any scene)", fontsize=10)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[1].scatter(stm["pat"], stm["sm"], s=60, color="#4a6fa5", edgecolors="white")
        for d, r in stm.iterrows():
            axes[1].annotate(d.split("_")[-1], (r.pat, r.sm), xytext=(4, 4),
                             textcoords="offset points", fontsize=8, color=INK)
        axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].axvline(0, color=MUTED, lw=0.8)
        axes[1].set_xlabel("time-mean pattern at station cell (K)")
        axes[1].set_ylabel("time-mean SM 0-10 anomaly (m³/m³)")
        axes[1].set_title("static: warmer cell ↔ drier station?", fontsize=10)
        axes[2].hist(b1_rd, bins=np.linspace(-1, 1, 21), color="#4a6fa5", edgecolor="white")
        axes[2].axvline(0, color=MUTED, lw=1)
        if len(b1_rd):
            axes[2].axvline(b1_rd.median(), color=INK, lw=2)
        axes[2].set_xlabel("per-date r across in-tile stations")
        axes[2].set_ylabel("dates")
        axes[2].set_title(f"per date, median {b1_rd.median():+.2f} (n={len(b1_rd)})",
                          fontsize=10)
        for ax in axes[1:]:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle("Part B1 — TxSON: does the within-tile LST pattern map the SM field?",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "B1_cr200-18_tile.png", dpi=150)
        plt.close(fig)

    # B2: network — station maps of time-mean spatial anomalies + scatter
    if len(netm):
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
        for ax, col, cmap, lab in ((axes[0], "dT_mean", "RdBu_r", "LST−T2m anomaly (K)"),
                                   (axes[1], "sm_0-10", "BrBG", "SM 0-10 anomaly (m³/m³)")):
            v = netm[col]
            lim = np.nanmax(np.abs(v))
            s = ax.scatter(netm["longitude"], netm["latitude"], c=v, cmap=cmap, vmin=-lim,
                           vmax=lim, s=70, edgecolors=INK, linewidths=0.8)
            fig.colorbar(s, ax=ax, label=lab)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
            ax.set_title(f"time-mean {lab.split(' (')[0]}", fontsize=10)
        for ax, x in ((axes[2], "dT_mean"), (axes[3], "P_stn")):
            m = netm[x].notna() & netm["sm_0-10"].notna()
            r = np.corrcoef(netm.loc[m, x], netm.loc[m, "sm_0-10"])[0, 1]
            ax.scatter(netm.loc[m, x], netm.loc[m, "sm_0-10"], s=50, color="#4a6fa5",
                       edgecolors="white")
            ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_xlabel(f"{NAMES[x]} anomaly (K)")
            ax.set_ylabel("SM 0-10 anomaly (m³/m³)")
            ax.set_title(f"across {m.sum()} stations, r = {r:+.2f}", fontsize=10)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle("Part B2 — TxSON network (~36 km): per-date spatial anomalies, "
                     "averaged over 2016-22", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "B2_network.png", dpi=150)
        plt.close(fig)

    # B3: CR200-18 time series — level vs pattern vs SM, one panel each (no dual axis)
    s = T[T["station"] == TXSON_REF].sort_values("date")
    if len(s):
        t = pd.to_datetime(s["date"].astype(str))
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for ax, col, lab in ((axes[0], "dT_mean", "LST tile − T2m (K)"),
                             (axes[1], "P_stn", "pattern at station (K)"),
                             (axes[2], "sm_0-10", "SM 0-10 (m³/m³)")):
            ax.plot(t, s[col], "o-", ms=3.5, lw=1, color="#4a6fa5")
            ax.set_ylabel(lab)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        axes[0].set_title(f"{TXSON_REF} — Landsat scene days only", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "B3_cr200-18_timeseries.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
