"""§70 effective resolution figure: what spatial scales does the predicted SM map contain?

    (a) radially averaged power spectrum of the daily anomaly maps (tile time-mean map removed)
    (b) fraction of anomaly variance retained after block-averaging at b x 20 m
    (c) semivariogram of the daily anomaly, normalised by the anomaly variance:
        model (random pixel pairs in the map) vs observed (TxSON station pairs, Test 0)

Reads <in-dir>/maps_L0.npy and <in-dir>/analysis/{testC_spectrum,testC_variance,test0_variogram}.csv.

    python plot_effres.py --in-dir eval_output/effres_txson_smoke --out figures/effres/effres_smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style_bw
from plot_style_bw import BLUE, GREEN, RED
from splits_config import station_dir_name

SPLITS = Path(__file__).resolve().parent / "csvs" / "station_splits.csv"

DEPTHS = ["0-10", "10-30", "30-100"]
COLORS = dict(zip(DEPTHS, [BLUE, GREEN, RED]))
LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
MAP_PX, PX_M = 112, 20
TOKEN_M = 160                                         # 14 x 14 token grid on a 2.24 km tile
N_PAIRS = 40000
LAG_EDGES = np.geomspace(20, 1600, 15)


def map_variogram(x: np.ndarray, rng) -> pd.DataFrame:
    """x (T, 112, 112) -> gamma_anom(h) / var_anom over random pixel pairs, binned by distance."""
    anom = x - x.mean(0, keepdims=True)
    r1, c1 = rng.integers(0, MAP_PX, N_PAIRS), rng.integers(0, MAP_PX, N_PAIRS)
    r2, c2 = rng.integers(0, MAP_PX, N_PAIRS), rng.integers(0, MAP_PX, N_PAIRS)
    keep = (r1 != r2) | (c1 != c2)
    r1, c1, r2, c2 = r1[keep], c1[keep], r2[keep], c2[keep]
    a, b = anom[:, r1, c1], anom[:, r2, c2]
    g = 0.5 * ((a - b) ** 2).mean(0)
    v = 0.5 * (a.var(0) + b.var(0))
    df = pd.DataFrame(dict(dist_m=np.hypot(r1 - r2, c1 - c2) * PX_M, g=g, v=v))
    df["bin"] = pd.cut(df["dist_m"], LAG_EDGES)
    out = df.groupby("bin", observed=True).agg(dist_m=("dist_m", "mean"), g=("g", "mean"),
                                               v=("v", "mean"), n=("g", "size")).reset_index()
    out["g_frac"] = out["g"] / out["v"]
    return out


def token_lines(ax):
    ax.axvline(TOKEN_M, color="black", ls="--", lw=1.0, zorder=0)
    ax.axvline(2 * TOKEN_M, color="0.45", ls=":", lw=1.2, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="eval_output/effres_txson_smoke")
    ap.add_argument("--out", default="figures/effres/effres_smoke")
    args = ap.parse_args()
    plot_style_bw.apply(globals(), "paper")
    d = Path(args.in_dir)
    an = d / "analysis"
    spec = pd.read_csv(an / "testC_spectrum.csv")
    var = pd.read_csv(an / "testC_variance.csv")
    obs = pd.read_csv(an / "test0_variogram.csv")

    # model variogram per tile (oos tiles only, as Test C), then mean g and v over tiles per lag bin
    maps = np.load(d / "maps_L0.npy", mmap_mode="r")
    idx = pd.read_parquet(d / "index_L0.parquet")
    splits = pd.read_csv(SPLITS)
    cur = dict(zip(splits.apply(station_dir_name, axis=1), splits["split"]))
    tiles = [(t, g["idx"].to_numpy()) for t, g in idx.groupby("tile") if cur.get(t) == "oos"]
    print(f"model variogram over {len(tiles)} oos tiles")
    rng = np.random.default_rng(0)
    rows = {dep: [] for dep in DEPTHS}
    for t, ii in tiles:
        m = np.asarray(maps[np.sort(ii)], np.float32)                # (T, 3, 112, 112)
        for i, dep in enumerate(DEPTHS):
            rows[dep].append(map_variogram(m[:, i], rng).assign(tile=t))
    vg = {}
    for dep in DEPTHS:
        per = pd.concat(rows[dep])
        per.assign(depth=dep).to_csv(an / f"map_variogram_tiles_{dep}.csv", index=False)
        vg[dep] = per.groupby("bin", observed=True).agg(
            dist_m=("dist_m", "mean"), g=("g", "mean"), v=("v", "mean"),
            n=("n", "sum"), n_tiles=("tile", "nunique")).reset_index()
        vg[dep]["g_frac"] = vg[dep]["g"] / vg[dep]["v"]
        vg[dep].assign(depth=dep).to_csv(an / f"map_variogram_{dep}.csv", index=False)
        print(dep, vg[dep][["dist_m", "g_frac", "n_tiles"]].round(4).to_string(index=False), sep="\n")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axs[0]
    for dep in DEPTHS:
        s = spec[spec["depth"] == dep]
        ax.loglog(s["wavelength_m"], s["power"], color=COLORS[dep], label=LABELS[dep])
    token_lines(ax)
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (m)")
    ax.set_ylabel("Anomaly power (a.u.)")
    ax.text(-0.13, 1.02, "(a)", transform=ax.transAxes, fontsize=13)

    ax = axs[1]
    for dep in DEPTHS:
        s = var[var["depth"] == dep]
        s = s[s["block"] < MAP_PX]
        ax.semilogx(s["L_m"], s["frac_retained"], "o-", color=COLORS[dep], ms=5)
    token_lines(ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Block size (m)")
    ax.set_ylabel("Anomaly variance retained")
    ax.text(-0.13, 1.02, "(b)", transform=ax.transAxes, fontsize=13)

    ax = axs[2]
    for dep in DEPTHS:
        ax.semilogx(vg[dep]["dist_m"], vg[dep]["g_frac"], color=COLORS[dep], lw=1.6)
    o = obs.copy()
    lo_hi = o["bin"].str.strip("(]").str.split(",", expand=True).astype(float)
    o["mid"] = np.sqrt(lo_hi[0].clip(lower=20) * lo_hi[1])
    for dep in DEPTHS:
        od = o[o["depth"] == dep]
        ax.plot(od["mid"], od["g_anom_frac"], "s", color=COLORS[dep], mec="black", ms=7, zorder=3)
    od = o[o["depth"] == "0-10"]
    for _, r in od.iterrows():
        ax.annotate(f"{int(r['n_pairs'])}", (r["mid"], r["g_anom_frac"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=11, color="black")
    token_lines(ax)
    top = max(o["g_anom_frac"].max(), max(v["g_frac"].max() for v in vg.values()))
    ax.set_ylim(0, top * 1.25)
    ax.set_xlabel("Separation (m)")
    ax.set_ylabel("Anomaly semivariance / variance")
    ax.text(-0.13, 1.02, "(c)", transform=ax.transAxes, fontsize=13)

    h = [plt.Line2D([], [], color=COLORS[d], lw=2, label=LABELS[d]) for d in DEPTHS]
    h += [plt.Line2D([], [], color="black", lw=1.6, label="Model map"),
          plt.Line2D([], [], ls="none", marker="s", color="0.6", mec="black", ms=7,
                     label="Observed station pairs"),
          plt.Line2D([], [], color="black", ls="--", lw=1.0, label="Token cell 160 m"),
          plt.Line2D([], [], color="0.45", ls=":", lw=1.2, label="Token Nyquist 320 m")]
    fig.legend(handles=h, loc="outside upper center", ncol=7, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
