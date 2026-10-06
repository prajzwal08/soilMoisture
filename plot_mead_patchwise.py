# ============================================================
# Mead: patchwise (pw_stage2a_L3, 160 m) vs the §66 model (20 m map) at Ne1 / Ne2 / south plots,
# all from ONE forward pass on Ne1's tile (runbook §69.7)
# ============================================================
# Inputs: pred/pw_maps_Ne1_2017-2024.npz   (predict_mead_patchwise.py, worktree wt_pw_era5do)
#         pred/maps_AmeriFlux_US-Ne1_2017-2024.npz  (§66 maps), observed Ne1/Ne2 SM,
#         pred/embed_cos_{S2,S1}_<year>_alldates.csv  (plot_mead_embed.py --all-dates: 1 - cos per date)
# Patches: Ne1 (7,7) tok 105 / map px (56,56); Ne2 (7,10) tok 108 / (56,83); south (12,6) tok 174 / (100,52).
# Figures (figures/mead/):
#   mead_pw_series_<year>   per depth: SM at the three patches (patchwise) + observed Ne1/Ne2;
#                           differences: observed Ne1-Ne2, patchwise Ne1-Ne2 / Ne1-south, §66 Ne1-Ne2 / Ne1-south
#   mead_pw_sm_2017-2024    SM at the three patches, patchwise and 20 m model side by side, + observed, ALL dates
#   mead_pw_scatter_<year>  as mead_embed_scatter_south: 1 - cos vs |dSM|, rows S2/S1 x depth, cols season,
#                           Ne1 vs Ne2 and Ne1 vs south, patchwise
#
# Usage:
#   python plot_mead_patchwise.py [--year 2019]
# ============================================================

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import plot_style_bw
from plot_mead_pred import ROOT, FIG_DIR, observed

DEPTHS = ["0-10", "10-30", "30-100"]
TOK    = {"Ne1": 105, "Ne2": 108, "south": 174}
PIX    = {"Ne1": (56, 56), "Ne2": (56, 83), "south": (100, 52)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    y = ap.parse_args().year

    z = np.load(ROOT / "pred" / "pw_maps_Ne1_2017-2024.npz")
    d = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    i = np.where(d.year == y)[0]
    pw = {k: pd.DataFrame(z["maps"][i, t], index=d[i], columns=DEPTHS) for k, t in TOK.items()}
    z = np.load(ROOT / "pred" / "maps_AmeriFlux_US-Ne1_2017-2024.npz")
    dm = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    im = np.where(dm.year == y)[0]
    mm = z["sm"][im].astype(np.float32)
    m66 = {k: pd.DataFrame(mm[:, :, r, c], index=dm[im], columns=DEPTHS) for k, (r, c) in PIX.items()}
    obs = {k: observed(f"AmeriFlux_US-{k}")[0] for k in ("Ne1", "Ne2")}
    days = pw["Ne1"].index
    js_all = (days.month >= 6) & (days.month <= 9)

    print(f"{y}: patchwise {len(days)} days, §66 {len(im)} days")
    for dep in DEPTHS:
        o = (obs["Ne1"][dep] - obs["Ne2"][dep]).reindex(days)
        print(f"\n{dep} cm   mean |.| all / Jun-Sep        corr with observed Ne1-Ne2 (all / Jun-Sep)")
        for lab, s in (("obs        Ne1-Ne2  ", o),
                       ("patchwise  Ne1-Ne2  ", pw["Ne1"][dep] - pw["Ne2"][dep]),
                       ("patchwise  Ne1-south", pw["Ne1"][dep] - pw["south"][dep]),
                       ("§66 map    Ne1-Ne2  ", (m66["Ne1"][dep] - m66["Ne2"][dep]).reindex(days)),
                       ("§66 map    Ne1-south", (m66["Ne1"][dep] - m66["south"][dep]).reindex(days))):
            c = [s[sel].corr(o[sel]) for sel in (np.ones(len(days), bool), js_all)]
            print(f"  {lab}  {s.abs().mean():.4f} / {s[js_all].abs().mean():.4f}      "
                  f"{c[0]:+.2f} / {c[1]:+.2f}" if "obs" not in lab else
                  f"  {lab}  {s.abs().mean():.4f} / {s[js_all].abs().mean():.4f}")
        for k in TOK:
            print(f"  patchwise {k:>5}: mean {pw[k][dep].mean():.3f}  SD over time {pw[k][dep].std():.4f}")

    plot_style_bw.apply(globals(), "paper")
    BL, LB, OR = plot_style_bw.BLUE, plot_style_bw.GREEN, plot_style_bw.RED

    # --- time series ---
    fig, ax = plt.subplots(len(DEPTHS), 2, figsize=(13, 9.5), sharex=True)
    for k, dep in enumerate(DEPTHS):
        a = ax[k, 0]
        for key, col, lab in (("Ne1", BL, "patchwise Ne1"), ("Ne2", LB, "patchwise Ne2"),
                              ("south", OR, "patchwise south plots")):
            a.plot(days, pw[key][dep], "-", color=col, lw=1.1, label=lab)
        for key, mk in (("Ne1", "o"), ("Ne2", "^")):
            s = obs[key][dep].reindex(days)
            a.plot(s.index, s.values, mk, ms=1.8, color="black", alpha=0.6, label=f"observed {key}")
        a.set_ylabel("SM (m$^3$ m$^{-3}$)")
        a.text(0.01, 0.97, f"({'ace'[k]}) {dep} cm", transform=a.transAxes, va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
        a = ax[k, 1]
        a.axhline(0, color="0.6", lw=0.6)
        a.plot(days, (obs["Ne1"][dep] - obs["Ne2"][dep]).reindex(days), "-", color="black", lw=1.0,
               label="observed Ne1 $-$ Ne2")
        a.plot(days, pw["Ne1"][dep] - pw["Ne2"][dep], "-", color=LB, lw=1.1, label="patchwise Ne1 $-$ Ne2")
        a.plot(days, pw["Ne1"][dep] - pw["south"][dep], "-", color=OR, lw=1.1, label="patchwise Ne1 $-$ south")
        a.plot(m66["Ne1"].index, m66["Ne1"][dep] - m66["Ne2"][dep], "--", color=LB, lw=0.8,
               label="20 m model Ne1 $-$ Ne2")
        a.plot(m66["Ne1"].index, m66["Ne1"][dep] - m66["south"][dep], "--", color=OR, lw=0.8,
               label="20 m model Ne1 $-$ south")
        a.set_ylabel("$\\Delta$SM (m$^3$ m$^{-3}$)")
        a.text(0.01, 0.97, f"({'bdf'[k]}) differences, {dep} cm", transform=a.transAxes, va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
    for a in ax.ravel():
        a.axvspan(pd.Timestamp(f"{y}-06-01"), pd.Timestamp(f"{y}-09-30"), color="0.93", lw=0, zorder=0)
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    ax[0, 1].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_pw_series_{y}.{ext}", dpi=300)
    print(f"\nfigure: {FIG_DIR}/mead_pw_series_{y}.png  (grey band = Jun-Sep)")

    # --- full record: SM at the three patches, patchwise and 20 m model, + observed, all dates ---
    z = np.load(ROOT / "pred" / "pw_maps_Ne1_2017-2024.npz")
    da = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    pa = {k: pd.DataFrame(z["maps"][:, t], index=da, columns=DEPTHS) for k, t in TOK.items()}
    z = np.load(ROOT / "pred" / "maps_AmeriFlux_US-Ne1_2017-2024.npz")
    dma = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    ma = {k: pd.DataFrame(z["sm"][:, :, r, c].astype(np.float32), index=dma, columns=DEPTHS)
          for k, (r, c) in PIX.items()}
    fig, ax = plt.subplots(len(DEPTHS), 2, figsize=(16, 9.5), sharex=True, sharey="row")
    for k, dep in enumerate(DEPTHS):
        for j, (src, lab) in enumerate(((pa, "patchwise (160 m)"), (ma, "20 m model"))):
            a = ax[k, j]
            for key, mk in (("Ne1", "o"), ("Ne2", "^")):
                s = obs[key][dep].loc[da[0]:da[-1]]
                a.plot(s.index, s.values, mk, ms=1.2, color="0.25" if key == "Ne1" else "0.6",
                       label=f"observed {key}")
            for key, col in (("Ne1", BL), ("Ne2", LB), ("south", OR)):
                a.plot(src[key].index, src[key][dep], "-", color=col, lw=0.9, label=f"{key}" +
                       (" plots" if key == "south" else ""))
            a.text(0.01, 0.97, f"({'abcdef'[2 * k + j]}) {lab}, {dep} cm", transform=a.transAxes, va="top",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
            if j == 0:
                a.set_ylabel("SM (m$^3$ m$^{-3}$)")
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=5, frameon=False, markerscale=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_pw_sm_{da[0].year}-{da[-1].year}.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_pw_sm_{da[0].year}-{da[-1].year}.png")

    # --- scatter: 1 - cos vs patchwise |dSM| ---
    rows = [(s, b) for s in ("S2", "S1") for b in DEPTHS[:2]]
    fig, ax = plt.subplots(len(rows), 2, figsize=(8, 12))
    for k, (sensor, b) in enumerate(rows):
        r = pd.read_csv(ROOT / "pred" / f"embed_cos_{sensor}_{y}_alldates.csv", index_col=0, parse_dates=True)
        r["pw_ne2"] = (pw["Ne1"][b] - pw["Ne2"][b]).reindex(r.index)
        r["pw_south"] = (pw["Ne1"][b] - pw["south"][b]).reindex(r.index)
        js = (r.index.month >= 6) & (r.index.month <= 9)
        top = max(np.nanmax(r["pw_ne2"].abs()), np.nanmax(r["pw_south"].abs())) * 1.35
        for j, (sel, season) in enumerate(((js, "Jun-Sep"), (~js, "Oct-May"))):
            a = ax[k, j]
            txt = []
            for xc, yc, col, mk, name in (("same_tile", "pw_ne2", LB, "^", "Ne1 vs Ne2"),
                                          ("ne1_south", "pw_south", OR, "s", "Ne1 vs south")):
                x, zz = 1 - r[xc], r[yc].abs()
                a.plot(x[sel], zz[sel], mk, ms=5, color=col, label=name)
                ok = sel & np.isfinite(x.values) & np.isfinite(zz.values)
                n = int(ok.sum())
                txt.append(f"{name} {spearmanr(x.values[ok], zz.values[ok])[0]:+.2f} (n {n})" if n >= 5
                           else f"{name} n/a")
            a.text(0.03, 0.97, f"({'abcdefgh'[2 * k + j]}) {sensor}, {b} cm, {season}, patchwise\nSpearman "
                   + ", ".join(txt), transform=a.transAxes, va="top", fontsize="small",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
            a.set_ylim(-0.002, top)
            if j == 0:
                a.set_ylabel("|$\\Delta$SM| patchwise (m$^3$ m$^{-3}$)")
            a.set_xlabel(f"1 $-$ cosine, {sensor} L12 patches")
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_pw_scatter_{y}.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_pw_scatter_{y}.png")


if __name__ == "__main__":
    main()
