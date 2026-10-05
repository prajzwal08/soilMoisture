#!/usr/bin/env python
"""Deseasonalised LST level / pattern vs SM 0-10 cm, ALL stations in the §52 probe scenes.csv.

The B3 figure (plot_lst_level_pattern_scatter.py) shows ONE station with raw dots, and its
"deseasonalised r" removed monthly means taken over scene days only. Here every plotted value is
an anomaly from a SMOOTH per-station climatology (user's choice, 2026-10-05):

  SM 0-10      climatology from the station's FULL daily record (qc == 0, 2016-2022, every day,
               not just scene days): day-of-year mean, then a 31-day circular running mean.
  LST − T2m,   only exist on scene days, so the climatology is a per-station harmonic fit
  pattern      (mean + annual + semi-annual sin/cos) to that station's scene days.

Anomaly = value − climatology on the scene day. Stations with < MIN_SCENES scenes are dropped
(a 5-parameter fit needs them).

Writes figures/probe_lst_level_pattern/
  C1_deseason_pooled_0-10.{png,pdf}    (a) level anomaly vs SM anomaly, all stations pooled
                                        (b) pattern anomaly vs SM anomaly, pooled
                                        (c) per-station deseasonalised r, level vs pattern
  C2_deseason_per_station_0-10.{png,pdf}  small multiples: level anomaly vs SM anomaly, one panel per station
and csvs/probe_lst_level_pattern/partC_deseason_r_per_station.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENES = Path("csvs/probe_lst_level_pattern/scenes.csv")
OUT_FIG = Path("figures/probe_lst_level_pattern")
OUT_CSV = Path("csvs/probe_lst_level_pattern")
SM = "sm_0-10"
SIGNALS = [("dT_mean", "LST tile − T2m mean, anomaly (K)", "Level"),
           ("P_stn", "Cell − tile mean, anomaly (K)", "Pattern")]
MIN_SCENES = 20          # per-station r needs this many scenes with both values (probe's MIN_SCENES)
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#d9d8d4"
TXSON, OTHER = "#2a78d6", "#d0822c"   # identity: TxSON network vs the rest


CLIM_WINDOW = 31        # days, circular running mean on the SM day-of-year climatology
MIN_DOY_YEARS = 2       # a day-of-year needs observed SM in >= this many years to enter the mean


def doy_of(dates_int):
    """YYYYMMDD ints -> day of year 1..365 (Feb 29 folded onto Feb 28's slot)."""
    t = pd.to_datetime(pd.Series(dates_int).astype(str), format="%Y%m%d")
    doy = t.dt.dayofyear.to_numpy()
    leap_after = t.dt.is_leap_year.to_numpy() & (doy >= 60)
    return np.where(leap_after, doy - 1, doy)


def sm_climatology(task):
    """Full daily SM 0-10 record -> smooth 365-day climatology. Never raises."""
    st, cat = task
    try:
        from dataset import SM_DEPTHS, _load_zarr_labels, _open_zarr
        from splits_config import TRAIN_YEARS
        zg = _open_zarr(Path(st), cat)
        lab = None if zg is None else _load_zarr_labels(zg, strict=False)
        if lab is None or lab[3] is None:
            return st, "no_labels_or_qc", None
        sm, depths, times, qc = lab
        di = {str(x): i for i, x in enumerate(depths)}.get(SM_DEPTHS[0])
        if di is None:
            return st, "no_0-10", None
        tint = (times.year * 10000 + times.month * 100 + times.day).to_numpy()
        keep = np.isin(tint // 10000, TRAIN_YEARS) & (qc[di] == 0) & np.isfinite(sm[di])
        if keep.sum() < 365:
            return st, f"short:{int(keep.sum())}", None
        df = pd.DataFrame({"doy": doy_of(tint[keep]), "year": tint[keep] // 10000, "sm": sm[di][keep]})
        g = df.groupby("doy")
        mean = g["sm"].mean().where(g["year"].nunique() >= MIN_DOY_YEARS).reindex(range(1, 366))
        h = CLIM_WINDOW // 2
        wrap = pd.concat([mean.iloc[-h:], mean, mean.iloc[:h]])
        clim = wrap.rolling(CLIM_WINDOW, center=True, min_periods=CLIM_WINDOW // 2).mean().iloc[h:-h]
        return st, "ok", clim.to_numpy()
    except Exception as e:
        return st, f"error:{type(e).__name__}:{e}", None


def harmonic_anomaly(doy, y):
    """Residual of y on [1, cos, sin, cos2, sin2](2*pi*doy/365.25); NaN where y is NaN."""
    w = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    m = np.isfinite(y)
    out = np.full(len(y), np.nan)
    if m.sum() < MIN_SCENES:
        return out
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    out[m] = y[m] - X[m] @ beta
    return out


def deseason(s, sm_clim):
    """Scene table -> anomalies: SM vs its daily climatology, thermal columns vs harmonic fit."""
    parts = []
    for st, g in s.groupby("station"):
        clim = sm_clim.get(st)
        if clim is None:
            continue
        g = g.copy()
        g["doy"] = doy_of(g["date"].to_numpy())
        g[SM] = g[SM].to_numpy() - clim[g["doy"].to_numpy() - 1]
        for c, _, _ in SIGNALS:
            g[c] = harmonic_anomaly(g["doy"].to_numpy(float), g[c].to_numpy(float))
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def r_of(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 5 or x[m].std() == 0 or y[m].std() == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def style(ax, zero_x=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    if zero_x:
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--txson-only", action="store_true", help="TxSON stations only (outputs get a _TxSON suffix)")
    a = ap.parse_args()
    tag = "_TxSON" if a.txson_only else ""
    from multiprocessing import Pool
    from splits_config import category_of, station_dir_name

    s = pd.read_csv(SCENES)
    if a.txson_only:
        s = s[s["txson"].astype(str).str.lower() == "true"].copy()
    sp = pd.read_csv("csvs/station_splits.csv")          # pandas: quoted commas
    sp["dir"] = sp.apply(station_dir_name, axis=1)
    sp["cat"] = sp.apply(category_of, axis=1)
    cats = sp.set_index("dir")["cat"]
    stations = sorted(s["station"].unique())
    with Pool(16) as pool:
        res = pool.map(sm_climatology, [(st, cats.get(st)) for st in stations], chunksize=1)
    status = pd.Series({st: msg for st, msg, _ in res})
    print("SM climatology:", status.map(lambda m: m.split(":")[0]).value_counts().to_dict())
    for st, msg in status[status != "ok"].items():
        print(f"  {st}: {msg}")
    sm_clim = {st: c for st, msg, c in res if msg == "ok"}

    s = s[s[SM].notna()].copy()                           # scene-day SM is already qc == 0
    d = deseason(s, sm_clim)
    d = d[d[SM].notna()]                                  # doy with no climatology -> dropped
    d["is_txson"] = d["txson"].astype(str).str.lower() == "true"
    print(f"anomalies: {len(d)} scenes, {d['station'].nunique()} stations")

    rows = []
    for st, g in d.groupby("station"):
        row = {"station": st, "network": g["network"].iloc[0], "split": g["split"].iloc[0],
               "txson": bool(g["is_txson"].iloc[0])}
        for c, _, _ in SIGNALS:
            r, n = r_of(g[c], g[SM])
            row[f"r_{c}"], row[f"n_{c}"] = (r if n >= MIN_SCENES else np.nan), n
        rows.append(row)
    per = pd.DataFrame(rows)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT_CSV / f"partC_deseason_r_per_station{tag}.csv", index=False)

    # ---- C1: pooled + per-station r --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, (c, lab, name) in zip(axes[:2], SIGNALS):
        style(ax)
        for flag, col, lbl in ((False, OTHER, "Other networks"), (True, TXSON, "TxSON")):
            g = d[d["is_txson"] == flag]
            if g.empty:
                continue
            ax.scatter(g[SM], g[c], s=7, color=col, alpha=0.35, linewidths=0, label=lbl, rasterized=True)
        r, n = r_of(d[c], d[SM])
        rs = per[f"r_{c}"].dropna()
        m = d[c].notna() & d[SM].notna()
        k, b = np.polyfit(d.loc[m, SM], d.loc[m, c], 1)
        xx = np.linspace(d.loc[m, SM].quantile(.01), d.loc[m, SM].quantile(.99), 2)
        ax.plot(xx, k * xx + b, color=INK, lw=1.5)
        ax.set_title(f"({'ab'[SIGNALS.index((c, lab, name))]}) {name}: pooled r = {r:+.2f} (n = {n})\n"
                     f"per-station median r = {rs.median():+.2f} ({len(rs)} stations)",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Observed SM 0-10 cm, anomaly (m³/m³)", fontsize=9, color=INK)
        ax.set_ylabel(lab, fontsize=9, color=INK)
        lo, hi = d[c].quantile([.005, .995])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    axes[0].legend(frameon=False, fontsize=8, markerscale=2.5, loc="upper right")

    ax = axes[2]
    style(ax, zero_x=False)
    rng = np.random.default_rng(0)
    for i, (c, _, name) in enumerate(SIGNALS):
        p = per.dropna(subset=[f"r_{c}"])
        for flag, col in ((False, OTHER), (True, TXSON)):
            q = p[p["txson"] == flag]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(q)), q[f"r_{c}"], s=16, color=col,
                       alpha=0.8, edgecolors="white", linewidths=0.4)
        med = p[f"r_{c}"].median()
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2)
        ax.annotate(f"{med:+.2f}", (i + 0.3, med), va="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1], [n for _, _, n in SIGNALS])
    ax.set_xlim(-0.6, 1.8)
    ax.set_ylabel("Per-station deseasonalised r with SM 0-10", fontsize=9, color=INK)
    ax.set_title(f"(c) One dot per station (≥ {MIN_SCENES} scenes)", fontsize=9, color=INK, loc="left")
    fig.suptitle(("TxSON only. " if a.txson_only else "") + "Landsat scene days 2016-2022, anomalies from smooth per-station climatologies "
                 "(SM: full daily record; thermal: harmonic fit)",
                 fontsize=10, color=INK)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"C1_deseason_pooled_0-10{tag}.{ext}", dpi=300)
    plt.close(fig)

    # ---- C2: small multiples, level only --------------------------------------------------
    c, lab, _ = SIGNALS[0]
    order = per.sort_values(["txson", f"r_{c}"], na_position="last")["station"].tolist()
    ncol = 8
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.8 * nrow), constrained_layout=True)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    for ax, st in zip(axes.flat, order):
        g = d[d["station"] == st]
        style(ax)
        col = TXSON if g["is_txson"].iloc[0] else OTHER
        ax.scatter(g[SM], g[c], s=6, color=col, alpha=0.7, linewidths=0, rasterized=True)
        r, n = r_of(g[c], g[SM])
        ax.set_title(f"{st.replace('ISMN_', '')[:26]}\nr = {r:+.2f}  n = {n}", fontsize=6.5, color=INK)
        ax.tick_params(labelsize=6)
    fig.supxlabel("Observed SM 0-10 cm, anomaly (m³/m³)", fontsize=9, color=INK)
    fig.supylabel(lab, fontsize=9, color=INK)
    fig.suptitle("Deseasonalised level vs SM per station (orange = other networks, blue = TxSON; "
                 "sorted by r within group)", fontsize=10, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"C2_deseason_per_station_0-10{tag}.{ext}", dpi=200)
    plt.close(fig)

    print(per[["r_dT_mean", "r_P_stn"]].describe().round(3).to_string())
    print(f"wrote {OUT_FIG}/C1_*{tag}, C2_*{tag}  and {OUT_CSV}/partC_deseason_r_per_station{tag}.csv")


if __name__ == "__main__":
    main()
