#!/usr/bin/env python
"""Where is the OOS station bias large -- land cover, climate, soil -- or is it random?

Per OOS station and depth, `bias` = mean(pred - obs) over observed days (per_station_oos.csv).

Figure (paper style, no titles):
  row 1  land cover (igbp_macro) x climate (Köppen main class): median signed bias per cell
  row 2  soil texture group (0-30 cm, OpenLandMap at the station) x climate: same
  row 3  "is it random?": share of the between-station bias variance explained by each factor
         (R2 of the group-mean model) against its permutation null (2000 label shuffles, 95th pct)
  one column per depth; cells with < MIN_N stations are left blank (count still printed).

Soil texture: centre 3x3 of the 74x74 @30 m soil patch (90 m around the station), clay/sand/silt
at 0-30 cm, USDA class -> Coarse (sand, loamy sand, sandy loam) / Medium (loam, silt loam, silt,
sandy clay loam) / Fine (all clay classes).

Also writes bias_groups_tests.csv: R2, permutation p and Kruskal-Wallis p per factor x depth.
CPU only, seconds.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import kruskal

import plot_style_bw
from eval_metrics import _make_key

RUN     = "baseline_selected_20261005"
TOK     = Path("/gpfs/scratch1/shared/pkhanal/zarr")
DEPTHS  = ["0-10", "10-30", "30-100"]
MIN_N   = 3
N_PERM  = 2000
KG      = {"B": "B Arid", "C": "C Temperate", "D": "D Continental", "E": "E Polar"}
LC_ORDER = ["Forest", "Grass-Crop", "Shrub-Savanna", "Other"]
TX_ORDER = ["Coarse", "Medium", "Fine"]


def usda(sand, silt, clay) -> str:
    t = sand + silt + clay
    sand, silt, clay = 100 * sand / t, 100 * silt / t, 100 * clay / t
    if silt + 1.5 * clay < 15:                                   return "sand"
    if silt + 2 * clay < 30:                                     return "loamy sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):  return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:        return "loam"
    if silt >= 80 and clay < 12:                                 return "silt"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12): return "silt loam"
    if 20 <= clay < 35 and silt < 28 and sand > 45:              return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:                      return "clay loam"
    if 27 <= clay < 40 and sand <= 20:                           return "silty clay loam"
    if clay >= 35 and sand > 45:                                 return "sandy clay"
    if clay >= 40 and silt >= 40:                                return "silty clay"
    return "clay"


def texture_group(cls: str) -> str:
    if cls in ("sand", "loamy sand", "sandy loam"):
        return "Coarse"
    if cls in ("loam", "silt loam", "silt", "sandy clay loam"):
        return "Medium"
    return "Fine"


def station_soil(key: str, cat: str):
    try:
        s = zarr.open_consolidated(str(TOK / cat / key))["soil"][0:3, 36:39, 36:39]
    except Exception:
        return None
    s = np.where(s == 0, np.nan, s.astype(np.float64))
    clay, sand, silt = (np.nanmean(s[i]) for i in range(3))
    if not np.isfinite([clay, sand, silt]).all():
        return None
    return clay, sand, silt


def r2_perm(y: np.ndarray, g: np.ndarray, rng) -> tuple[float, float, float]:
    """R2 of the group-mean model, its permutation p and the null 95th percentile."""
    def r2(lab):
        sst = ((y - y.mean()) ** 2).sum()
        ssw = sum(((y[lab == k] - y[lab == k].mean()) ** 2).sum() for k in np.unique(lab))
        return 1 - ssw / sst
    obs = r2(g)
    null = np.array([r2(rng.permutation(g)) for _ in range(N_PERM)])
    return obs, (1 + (null >= obs).sum()) / (N_PERM + 1), float(np.percentile(null, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-station", default=f"eval_output/{RUN}/per_station_oos.csv")
    ap.add_argument("--out-dir", default=f"figures/eval/{RUN}_paper")
    a = ap.parse_args()
    plt.rcParams.update(plot_style_bw.RC_PAPER)
    rng = np.random.default_rng(0)

    ps = pd.read_csv(a.per_station)
    sp = pd.read_csv("csvs/station_splits.csv")           # pandas: quoted commas in AmeriFlux names
    sp["key"] = sp.apply(_make_key, axis=1)
    sp["cat"] = np.where(sp.has_soil_moisture & sp.has_flux, "sm_and_flux",
                         np.where(sp.has_soil_moisture, "sm_only", "flux_only"))
    cat = dict(zip(sp.key, sp.cat))

    tex, cls = {}, {}
    for k in ps.station_key:
        v = station_soil(k, cat.get(k, "sm_only"))
        if v is not None:
            c = usda(v[1], v[2], v[0])
            cls[k], tex[k] = c, texture_group(c)
    ps["texture"] = ps.station_key.map(tex)
    ps["climate"] = ps.kg_macro.map(KG)
    print(f"soil texture for {ps.texture.notna().sum()}/{len(ps)} stations")
    print(pd.Series(cls).value_counts().to_string())

    # a climate class with fewer than MIN_N stations overall (E: 1) is dropped from the maps
    climates = [KG[c] for c in "BCDE" if (ps.climate == KG[c]).sum() >= MIN_N]
    rows = [("igbp_macro", [c for c in LC_ORDER if c in set(ps.igbp_macro)], "Land cover"),
            ("texture", TX_ORDER, "Soil texture (0-30 cm)")]

    # shared colour scale over every cell that is drawn
    cells = {}
    for col, order, _ in rows:
        for d in DEPTHS:
            b = f"bias_{d.replace('-', '_')}"
            sub = ps[ps[b].notna()]
            m = np.full((len(order), len(climates)), np.nan); n = np.zeros_like(m, int)
            for i, r in enumerate(order):
                for j, c in enumerate(climates):
                    v = sub.loc[(sub[col] == r) & (sub.climate == c), b]
                    n[i, j] = len(v)
                    if len(v) >= MIN_N:
                        m[i, j] = v.median()
            cells[(col, d)] = (m, n)
    vmax = max(np.nanmax(np.abs(m)) for m, _ in cells.values())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    cmap = LinearSegmentedColormap.from_list("dry_wet", [plot_style_bw.BLUE, "white", plot_style_bw.RED])

    fig = plt.figure(figsize=(15.5, 12.0))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1.15, 0.85, 0.9],
                          hspace=0.42, wspace=0.12, left=0.11, right=0.93, top=0.95, bottom=0.07)
    letter = iter("abcdefghijk")
    for ri, (col, order, ylab) in enumerate(rows):
        for dj, d in enumerate(DEPTHS):
            ax = fig.add_subplot(gs[ri, dj])
            m, n = cells[(col, d)]
            im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if n[i, j] == 0:
                        continue
                    txt = (f"{m[i, j]:+.3f}\n{n[i, j]}" if np.isfinite(m[i, j]) else f"–\n{n[i, j]}")
                    ax.text(j, i, txt, ha="center", va="center", linespacing=1.3,
                            color="white" if np.isfinite(m[i, j]) and abs(m[i, j]) > 0.6 * vmax else "black")
            ax.set_xticks(range(len(climates)), [c.replace(" ", "\n") for c in climates])
            ax.set_yticks(range(len(order)), order if dj == 0 else [""] * len(order))
            if dj == 0:
                ax.set_ylabel(ylab)
            ax.text(0.0, 1.02, f"({next(letter)}) {d} cm", transform=ax.transAxes,
                    ha="left", va="bottom")
    cax = fig.add_subplot(gs[0:2, 3])
    fig.colorbar(im, cax=cax, label="median bias, predicted − observed (m$^3$/m$^3$)")

    # ── is it random? ────────────────────────────────────────────────────
    tests = []
    factors = [("igbp_macro", "Land cover"), ("climate", "Climate"), ("texture", "Soil texture")]
    for d in DEPTHS:
        b = f"bias_{d.replace('-', '_')}"
        for col, lab in factors:
            sub = ps[ps[b].notna() & ps[col].notna()]
            keep = sub[col].map(sub[col].value_counts()) >= MIN_N        # groups too small to test
            sub = sub[keep]
            y, g = sub[b].to_numpy(float), sub[col].to_numpy()
            r2, p, null95 = r2_perm(y, g, rng)
            kw = kruskal(*[y[g == k] for k in np.unique(g)]).pvalue
            tests.append(dict(depth=d, factor=lab, n=len(y), groups=len(np.unique(g)),
                              R2=r2, perm_p=p, null95=null95, kruskal_p=kw))
    t = pd.DataFrame(tests)
    print("\n" + t.round(4).to_string(index=False))

    ax = fig.add_subplot(gs[2, 0:3])
    w = 0.26
    cols = [plot_style_bw.BLUE, plot_style_bw.GREEN, plot_style_bw.RED]
    for fi, (_, lab) in enumerate(factors):
        q = t[t.factor == lab].set_index("depth").loc[DEPTHS]
        x = np.arange(len(DEPTHS)) + (fi - 1) * w
        ax.bar(x, 100 * q.R2, w * 0.92, color=cols[fi], edgecolor="black", lw=0.8, label=lab)
        ax.plot(np.c_[x - w * 0.46, x + w * 0.46].T, np.c_[100 * q.null95, 100 * q.null95].T,
                color="black", lw=2.0, ls="--")
        for xi, (r2, p, nl) in zip(x, zip(q.R2, q.perm_p, q.null95)):
            ax.text(xi, 100 * max(r2, nl) + 0.5, f"p = {p:.2f}" if p >= 0.01 else "p < 0.01",
                    ha="center", va="bottom")
    ax.plot([], [], color="black", lw=2.0, ls="--", label="random (95th pct of shuffles)")
    ax.set_xticks(range(len(DEPTHS)), [f"{d} cm" for d in DEPTHS])
    ax.set_ylabel("Bias variance explained (%)")
    ax.set_ylim(0, max(100 * t[["R2", "null95"]].to_numpy().max() + 12, 10))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    ax.text(0.0, 1.02, f"({next(letter)})", transform=ax.transAxes, ha="left", va="bottom")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"bias_groups.{ext}", dpi=600)
    t.to_csv(out / "bias_groups_tests.csv", index=False)
    ps[["station_key", "igbp_macro", "climate", "texture"]].assign(usda=ps.station_key.map(cls)) \
        .to_csv(out / "bias_groups_stations.csv", index=False)
    print(f"wrote {out}/bias_groups.{{png,pdf}}, bias_groups_tests.csv, bias_groups_stations.csv")


if __name__ == "__main__":
    main()
