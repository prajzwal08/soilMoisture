"""§73 group-Shapley figures from eval_predict.py --shap output (shap_<split>.parquet).

Players (K = 5): ERA5, satellite 160 m (S2 + S1 + anchor, one block), DEM, land cover, soil.
Fixed context (never switched off): SIF, TWSA, the 20 m imagery. phi is in m3/m3, per depth,
relative to f_none = the prediction with all five players absent; on every row
f_none + sum(phi) == pred (efficiency, checked again here).

The mock-ups (artifact F7hwYUKM13tD6jkGb5mHdH, K = 4 on §59) are adapted to K = 5:
    A  mean |phi| per player and depth (whiskers = IQR of station means)
    B  share of total |phi| per depth (100 % bars)
    C  |phi_sat| by number of valid scenes in the last 30 days (+ "none in 365 d" bin)
    D  |phi_sat| against days since the last valid scene (median + IQR, one line per depth)
    E  rain-event composite (> 10 mm/d): phi_ERA5 and phi_sat at 0-10 cm, pre-event baseline removed
    F  one station-year: obs / pred / f_none, scene strip, stacked daily phi
    G  waterfalls for two days of the F station (biggest ERA5 wet-up, driest prediction)
    H  phi_ERA5 (0-10 cm) against 7-day ERA5 rain, coloured by days since the last scene
    I  stacked daily phi at 0-10 cm for four stations in four climate groups
    K  share of mean |phi| per player by land-cover group (igbp_macro), one panel per depth
    L  the same by climate group (kg_macro)
J (S2 and S1 as separate players) is NOT drawn: the §74 model was trained with whole-block
satellite dropout, so "S2 off, S1 on" is out of distribution for it.

Usage:
    python plot_shap.py --shap eval_output/<run>_shap/shap_oos.parquet \
        --per-station eval_output/<run>/per_station_oos.csv --out-dir figures/eval/<run>_paper/shap
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style_bw

PLAYERS = ["era5", "sat", "dem", "lulc", "soil"]
LABELS  = {"era5": "ERA5", "sat": "Satellite 160 m", "dem": "DEM", "lulc": "Land cover", "soil": "Soil"}
# palette H (dark blue / light blue / orange) extended by a light orange and a grey for K = 5
COLORS  = {"era5": "#08519C", "sat": "#6BAED6", "dem": "#E6550D", "lulc": "#FDAE6B", "soil": "#636363"}
DEPTHS  = ["0-10", "10-30", "30-100"]
DLAB    = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
DCOL    = {"0-10": "#08519C", "10-30": "#6BAED6", "30-100": "#E6550D"}
PRED_COLOR = "#3776AB"
RAIN_MM, LAG_MIN, LAG_MAX = 10.0, -10, 30
ZARR_PERM = Path("/gpfs/work3/0/prjs1968/zarr_tokens")


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.png", flush=True)


def phi_cols():
    return [f"phi_{p}" for p in PLAYERS]


def last_scene_age(df):
    """Days since the last valid S2 or S1 scene (-1 = none in the 365-d window -> NaN)."""
    a = df[["days_since_s2", "days_since_s1"]].to_numpy(float)
    a[a < 0] = np.nan
    return np.nanmin(np.where(np.isnan(a), np.inf, a), axis=1).astype(float)


# ── A / B ─────────────────────────────────────────────────────────────────────────────────
def fig_A(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True, constrained_layout=True)
    rows = []
    for ax, d in zip(axes, DEPTHS):
        g = df[df["depth"] == d]
        st = g.groupby("station_key", observed=True)[phi_cols()].agg(lambda x: np.abs(x).mean())
        for i, p in enumerate(PLAYERS):
            m = st[f"phi_{p}"].mean()          # mean of station means: same basis as the IQR whisker
            q1, q3 = np.percentile(st[f"phi_{p}"], [25, 75])
            ax.bar(i, m, color=COLORS[p], edgecolor="black", lw=0.8, width=0.7)
            ax.plot([i, i], [q1, q3], color="black", lw=1.4)
            ax.plot([i - 0.12, i + 0.12], [q3, q3], color="black", lw=1.4)
            rows.append(dict(depth=d, player=p, mean_abs_phi=m, station_q1=q1, station_q3=q3,
                             n_rows=len(g), n_stations=len(st)))
        ax.set_xticks(range(len(PLAYERS)), [LABELS[p] for p in PLAYERS], rotation=90)
        ax.text(0.03, 0.97, DLAB[d], transform=ax.transAxes, ha="left", va="top")
        ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
    axes[0].set_ylabel("mean |φ| (m$^3$/m$^3$)")
    save(fig, out, "shap_A_mean_abs_phi")
    t = pd.DataFrame(rows)
    t["share_pct"] = 100 * t["mean_abs_phi"] / t.groupby("depth")["mean_abs_phi"].transform("sum")
    t.to_csv(out / "shap_A_B_table.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


def fig_B(t, out):
    fig, ax = plt.subplots(figsize=(8, 2.6), constrained_layout=True)
    for k, d in enumerate(DEPTHS):
        acc = 0.0
        for p in PLAYERS:
            s = float(t[(t.depth == d) & (t.player == p)]["share_pct"].iloc[0])
            ax.barh(k, s, left=acc, color=COLORS[p], edgecolor="black", lw=0.8, height=0.7)
            if s >= 7:
                ax.text(acc + s / 2, k, f"{s:.0f}%", ha="center", va="center",
                        color="white" if p in ("era5", "dem", "soil") else "black")
            acc += s
    ax.set_yticks(range(3), [DLAB[d] for d in DEPTHS]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("share of total mean |φ| (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORS[p], ec="black") for p in PLAYERS],
              labels=[LABELS[p] for p in PLAYERS], ncol=5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "shap_B_share")


# ── K / L: per ecosystem (igbp_macro) and per climate (kg_macro), user 2026-10-13 ──────────
def fig_group(df, ps, col, name, xlabel, out, min_stations=5):
    """Share of mean |phi| per player, one horizontal 100 % bar per group, one panel per depth.
    Groups with fewer than `min_stations` stations are left out (listed in the csv)."""
    m = df.merge(ps[["station_key", col]], on="station_key", how="left")
    m[col] = m[col].fillna("unknown").astype(str)
    rows = []
    for d in DEPTHS:
        g = m[m["depth"] == d]
        for grp, h in g.groupby(col):
            r = dict(depth=d, group=grp, n_stations=h["station_key"].nunique(), n_rows=len(h))
            for p in PLAYERS:
                r[f"mean_abs_{p}"] = np.abs(h[f"phi_{p}"]).mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    tot = t[[f"mean_abs_{p}" for p in PLAYERS]].sum(1)
    for p in PLAYERS:
        t[f"share_{p}"] = 100 * t[f"mean_abs_{p}"] / tot
    t.to_csv(out / f"{name}_table.csv", index=False)
    keep = (t[t["depth"] == "0-10"].set_index("group")["n_stations"] >= min_stations)
    groups = [g for g in keep.index[keep]]
    groups = sorted(groups, key=lambda g: -float(
        t[(t.depth == "0-10") & (t.group == g)]["n_stations"].iloc[0]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(groups) + 1.8), sharey=True,
                             constrained_layout=True)
    for ax, d in zip(axes, DEPTHS):
        for k, grp in enumerate(groups):
            r = t[(t.depth == d) & (t.group == grp)]
            if not len(r):
                continue
            r = r.iloc[0]; acc = 0.0
            for p in PLAYERS:
                s = r[f"share_{p}"]
                ax.barh(k, s, left=acc, color=COLORS[p], edgecolor="black", lw=0.6, height=0.75)
                if s >= 12:
                    ax.text(acc + s / 2, k, f"{s:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if p in ("era5", "dem", "soil") else "black")
                acc += s
            ax.text(101, k, f"{int(r['n_stations'])}", va="center", fontsize=8)
        ax.set_xlim(0, 100); ax.set_xlabel("share of mean |φ| (%)")
        ax.text(0.0, 1.01, DLAB[d], transform=ax.transAxes, ha="left", va="bottom")
    axes[0].set_yticks(range(len(groups)), groups); axes[0].invert_yaxis()
    axes[0].set_ylabel(xlabel)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORS[p], ec="black") for p in PLAYERS],
               labels=[LABELS[p] for p in PLAYERS], ncol=5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, name)
    print(t[t["group"].isin(groups)].round(3).to_string(index=False))


# ── C / D (satellite staleness) ────────────────────────────────────────────────────────────
def fig_C(df, out):
    n30 = df["n_s2_30d"].to_numpy() + df["n_s1_30d"].to_numpy()
    none = (df["days_since_s2"].to_numpy() < 0) & (df["days_since_s1"].to_numpy() < 0) & \
           (~df["anchor_found"].to_numpy().astype(bool))
    cats = ["none in\n365 d", "0", "1-2", "3-5", "6-9", "10+"]
    b = np.select([none, n30 == 0, n30 <= 2, n30 <= 5, n30 <= 9], [0, 1, 2, 3, 4], 5)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, constrained_layout=True)
    for ax, d in zip(axes, DEPTHS):
        m = (df["depth"] == d).to_numpy()
        data = [np.abs(df.loc[m & (b == i), "phi_sat"].to_numpy()) for i in range(len(cats))]
        pos = [i for i, v in enumerate(data) if len(v)]
        ax.boxplot([data[i] for i in pos], positions=pos, widths=0.6, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor=COLORS["sat"], lw=0.8),
                   medianprops=dict(color="black", lw=1.6))
        for i in pos:
            ax.annotate(f"{len(data[i]):,}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, rotation=90)
        ax.set_xticks(range(len(cats)), cats, rotation=90)
        ax.text(0.03, 0.80, DLAB[d], transform=ax.transAxes, ha="left", va="top")
        ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
        if len(data[0]):
            print(f"  C {d}: 'none in 365 d' max |phi_sat| = {data[0].max():.2e} (must be ~0)")
    axes[0].set_ylabel("|φ$_{sat}$| (m$^3$/m$^3$)")
    fig.supxlabel("valid S2 + S1 scenes in the last 30 days")
    save(fig, out, "shap_C_sat_by_scenes")


def fig_D(df, out):
    age = last_scene_age(df)
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for d in DEPTHS:
        m = (df["depth"] == d).to_numpy() & np.isfinite(age) & (age <= 60)
        if not m.any():
            continue
        a, v = age[m].astype(int), np.abs(df.loc[m, "phi_sat"].to_numpy())
        g = pd.DataFrame({"a": a, "v": v}).groupby("a")["v"]
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        q = q[g.size() >= 50]
        ax.fill_between(q.index, q[0.25], q[0.75], color=DCOL[d], alpha=0.18, lw=0)
        ax.plot(q.index, q[0.5], color=DCOL[d], lw=2.2, label=DLAB[d])
    ax.set_xlabel("days since the last valid S2 or S1 scene")
    ax.set_ylabel("|φ$_{sat}$| (m$^3$/m$^3$)")
    ax.set_xlim(0, 60); ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(lw=0.4, alpha=0.35); ax.set_axisbelow(True)
    save(fig, out, "shap_D_sat_staleness")


# ── rain (E / H) ─────────────────────────────────────────────────────────────────────────
def _rain_one(job):
    key, cat = job
    import dataset as _ds          # already imported by load_rain before the fork: no re-import cost
    try:
        zg = _ds._open_zarr(ZARR_PERM / cat / key, cat)
        if zg is None or _ds.ERA5_ARRAY not in zg:
            return None
        vals, dints = zg[_ds.ERA5_ARRAY][:], zg["era5/date_ints"][:]
        names = [str(v) for v in zg["era5/vars18"][:]]
        tp = np.clip(np.asarray(vals[:, names.index("tp_sum")], float), 0, None) * 1000.0
        dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else str(d) for d in dints],
                               format="%Y%m%d")
        s = pd.Series(tp, index=dates).sort_index()
        return key, s[~s.index.duplicated()]
    except Exception as e:                                   # one bad store must not kill the job
        print(f"  rain: {key} skipped ({e})", flush=True)
        return None


def load_rain(keys, splits_csv, workers):
    from splits_config import category_of, station_dir_name
    sp = pd.read_csv(splits_csv)
    sp["key"] = sp.apply(station_dir_name, axis=1)
    sp["cat"] = sp.apply(category_of, axis=1)
    sp = sp[sp["key"].isin(set(keys))].drop_duplicates("key")
    jobs = list(zip(sp["key"], sp["cat"]))
    import dataset as _ds          # import (torch) ONCE in the parent; forked workers inherit it
    _ds.ZARR_ROOT = ZARR_PERM
    with Pool(workers) as pool:
        res = [r for r in pool.imap_unordered(_rain_one, jobs, chunksize=1) if r is not None]
    print(f"  rain loaded for {len(res)}/{len(set(keys))} stations", flush=True)
    return dict(res)


def fig_E(df, rain, out):
    g0 = df[df["depth"] == "0-10"]
    recs = []
    for key, g in g0.groupby("station_key", observed=True):
        r = rain.get(key)
        if r is None:
            continue
        ser = g.set_index("date")[["phi_era5", "phi_sat"]].sort_index()
        ser = ser[~ser.index.duplicated()]
        ev = r.index[(r > RAIN_MM).to_numpy()]
        ev = ev[(ev >= ser.index.min()) & (ev <= ser.index.max())]
        # decluster: keep an event only if no other > RAIN_MM day fell in the previous 10 days,
        # so the pre-event baseline is not already inside an earlier wet-up
        ev = [e for e in ev
              if not (r[e - pd.Timedelta(days=10): e - pd.Timedelta(days=1)] > RAIN_MM).any()]
        for e in ev:
            idx = pd.date_range(e + pd.Timedelta(days=LAG_MIN), e + pd.Timedelta(days=LAG_MAX))
            w = ser.reindex(idx)
            pre = w.iloc[: -LAG_MIN]                         # lags -10 .. -1
            if pre.notna().sum().min() < 5 or w.iloc[-LAG_MIN:].notna().sum().min() < 15:
                continue
            w = w - pre.mean()
            w["lag"] = np.arange(LAG_MIN, LAG_MAX + 1)
            recs.append(w)
    if not recs:
        print("  E: no events"); return
    c = pd.concat(recs)
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for p in ("era5", "sat"):
        s = c.groupby("lag")[f"phi_{p}"]
        m, se = s.mean(), s.std() / np.sqrt(s.count())
        ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=COLORS[p], alpha=0.2, lw=0)
        ax.plot(m.index, m, color=COLORS[p], lw=2.2, label=LABELS[p])
    ax.axvline(0, color="0.4", ls="--", lw=1); ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(f"days relative to rain > {RAIN_MM:.0f} mm/d (ERA5)")
    ax.set_ylabel("φ change vs. pre-event, 0-10 cm (m$^3$/m$^3$)")
    ax.legend(frameon=False, title=f"{len(recs):,} events (band: ±1.96 SE, events as independent)")
    ax.grid(lw=0.4, alpha=0.35); ax.set_axisbelow(True)
    save(fig, out, "shap_E_rain_composite")


def fig_H(df, rain, out):
    g0 = df[df["depth"] == "0-10"].copy()
    g0["age"] = last_scene_age(g0)
    parts = []
    for key, g in g0.groupby("station_key", observed=True):
        r = rain.get(key)
        if r is None:
            continue
        r7 = r.rolling("7D", min_periods=7).sum()          # calendar days D-6 .. D
        parts.append(pd.DataFrame({"p7": r7.reindex(g["date"]).to_numpy(),
                                   "phi": g["phi_era5"].to_numpy(), "age": g["age"].to_numpy()}))
    if not parts:
        print("  H: no stations with rain"); return
    h = pd.concat(parts).dropna()
    h = h.sample(min(len(h), 40000), random_state=0).sort_values("age", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sc = ax.scatter(h["p7"], h["phi"], c=np.clip(h["age"], 0, 30), cmap="Blues_r", s=4,
                    alpha=0.6, edgecolors="none", vmin=0, vmax=30)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("ERA5 rain, days D-6 to D (mm)")
    ax.set_ylabel("φ$_{ERA5}$, 0-10 cm (m$^3$/m$^3$)")
    ax.set_xlim(0, np.percentile(h["p7"], 99.5))
    fig.colorbar(sc, ax=ax, label="days since last scene (capped 30)")
    save(fig, out, "shap_H_era5_vs_rain")


# ── F / G / I (single stations) ──────────────────────────────────────────────────────────
def _stack(ax, g, width=1.0):
    dates = g["date"].to_numpy()
    up, dn = np.zeros(len(g)), np.zeros(len(g))
    for p in PLAYERS:
        v = g[f"phi_{p}"].to_numpy()
        pos, neg = np.where(v > 0, v, 0), np.where(v < 0, v, 0)
        ax.bar(dates, pos, bottom=up, width=width, color=COLORS[p], lw=0)
        ax.bar(dates, neg, bottom=dn, width=width, color=COLORS[p], lw=0)
        up += pos; dn += neg
    ax.axhline(0, color="black", lw=0.8)


def pick_station_year(df, key):
    g = df[(df["station_key"] == key) & (df["depth"] == "0-10")]
    y = g.groupby("year").size().idxmax()
    return g[g["year"] == y].sort_values("date"), y


def fig_F(df, key, out):
    g, y = pick_station_year(df, key)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[3, 0.6, 3]))
    a = axes[0]
    a.plot(g["date"], g["pred"], color=PRED_COLOR, lw=1.6, label="predicted")
    a.scatter(g["date"], g["obs"], s=4, color="black", label="observed", zorder=3)
    a.plot(g["date"], g["f_none"], color="0.4", ls="--", lw=1.2, label="f(∅)")
    a.set_ylabel("SM 0-10 cm (m$^3$/m$^3$)")
    a.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    s = axes[1]
    for k, (col, lab) in enumerate((("days_since_s2", "S2"), ("days_since_s1", "S1"))):
        dd = g.loc[g[col] == 0, "date"]
        s.vlines(dd, k - 0.35, k + 0.35, color=("#E6550D", "#636363")[k], lw=1.4)
    s.set_yticks([0, 1], ["S2", "S1"]); s.set_ylim(-0.6, 1.6)
    _stack(axes[2], g)
    axes[2].set_ylabel("φ, 0-10 cm (m$^3$/m$^3$)")
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORS[p]) for p in PLAYERS],
                   labels=[LABELS[p] for p in PLAYERS], ncol=5, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[0].text(0.0, 1.13, f"{key}  {y}", transform=axes[0].transAxes, ha="left")
    save(fig, out, "shap_F_station_year")
    return g


def fig_G(g, key, out):
    days = [("wet-up (largest φ$_{ERA5}$)", g.loc[g["phi_era5"].idxmax()]),
            ("driest prediction", g.loc[g["pred"].idxmin()])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for ax, (title, r) in zip(axes, days):
        acc = float(r["f_none"]); y = 0
        ax.plot([acc, acc], [y - 0.35, y + 0.35], color="black", lw=3)
        ax.text(acc, y - 0.5, f"{acc:.3f}", ha="center", va="top", fontsize=9)
        labels = ["f(∅)"]
        for p in PLAYERS:
            y += 1; v = float(r[f"phi_{p}"])
            ax.barh(y, v, left=acc, color=COLORS[p], edgecolor="black", lw=0.8, height=0.7)
            ax.text(acc + v + (0.002 if v >= 0 else -0.002), y, f"{v:+.3f}",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9)
            acc += v; labels.append(LABELS[p])
        y += 1; labels.append("prediction")
        ax.plot([r["pred"], r["pred"]], [y - 0.35, y + 0.35], color=PRED_COLOR, lw=4)
        ax.scatter([r["obs"]], [y], color="black", s=30, zorder=3, label=f"observed {r['obs']:.3f}")
        ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
        ax.set_xlabel("SM 0-10 cm (m$^3$/m$^3$)")
        ax.text(0.0, 1.02, f"{title}: {pd.Timestamp(r['date']).date()}", transform=ax.transAxes)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
        err = abs(float(r["f_none"]) + sum(float(r[f"phi_{p}"]) for p in PLAYERS) - float(r["pred"]))
        print(f"  G {title[:8]}: efficiency |f_none + sum phi - pred| = {err:.2e}")
    save(fig, out, "shap_G_waterfalls")


def fig_I(df, keys, out):
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.3 * len(keys)), sharey=True,
                             constrained_layout=True)
    for ax, (key, lab) in zip(np.atleast_1d(axes), keys):
        g, y = pick_station_year(df, key)
        _stack(ax, g)
        ax.text(0.01, 0.95, f"{lab}: {key} ({y})", transform=ax.transAxes, ha="left", va="top",
                bbox=dict(fc="white", ec="none", pad=1))
    np.atleast_1d(axes)[len(keys) // 2].set_ylabel("φ, 0-10 cm (m$^3$/m$^3$)")
    np.atleast_1d(axes)[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORS[p]) for p in PLAYERS],
                                   labels=[LABELS[p] for p in PLAYERS], ncol=5, frameon=False,
                                   loc="upper center", bbox_to_anchor=(0.5, -0.25))
    save(fig, out, "shap_I_four_stations")


def choose_stations(ps):
    """F: the station with the most 0-10 obs among those within ±20 % of the median ubRMSE.
    I: in each of four Koppen macro groups, the station with the most 0-10 obs."""
    ps = ps.dropna(subset=["ubRMSE_0_10"])
    med = ps["ubRMSE_0_10"].median()
    near = ps[(ps["ubRMSE_0_10"] - med).abs() <= 0.2 * med]
    f_key = near.sort_values("n_0_10", ascending=False)["station_key"].iloc[0]
    groups = ps["kg_macro"].value_counts().index[:4]
    i_keys = [(str(k), ps[ps["kg_macro"] == k].sort_values("n_0_10", ascending=False)["station_key"].iloc[0])
              for k in groups]
    return f_key, [(key, f"Köppen {k}") for k, key in i_keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", required=True)
    ap.add_argument("--per-station", required=True, help="per_station_<split>.csv (station picks)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits-csv", default="csvs/station_splits.csv")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--station", default=None, help="override the F/G station")
    a = ap.parse_args()
    plt.rcParams.update(plot_style_bw.RC_PAPER)
    out = Path(a.out_dir)

    df = pd.read_parquet(a.shap)
    df["depth"] = df["depth"].astype(str)
    df["station_key"] = df["station_key"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    eff = (df["f_none"] + df[phi_cols()].sum(1) - df["pred"]).abs().max()
    print(f"{len(df):,} rows, {df['station_key'].nunique()} stations; "
          f"max |f_none + sum phi - pred| = {eff:.2e}", flush=True)

    t = fig_A(df, out)
    fig_B(t, out)
    ps = pd.read_csv(a.per_station)
    ps["station_key"] = ps["station_key"].astype(str)
    fig_group(df, ps, "igbp_macro", "shap_K_by_ecosystem", "land-cover group (IGBP)", out)
    fig_group(df, ps, "kg_macro", "shap_L_by_climate", "climate group (Köppen)", out)
    fig_C(df, out)
    fig_D(df, out)
    rain = load_rain(df["station_key"].unique(), a.splits_csv, a.workers)
    fig_E(df, rain, out)
    fig_H(df, rain, out)
    f_key, i_keys = choose_stations(ps)
    f_key = a.station or f_key
    print(f"  F/G station: {f_key};  I stations: {i_keys}", flush=True)
    g = fig_F(df, f_key, out)
    fig_G(g, f_key, out)
    fig_I(df, i_keys, out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
