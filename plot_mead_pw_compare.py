# ============================================================
# Mead: patchwise (pw_stage2a_L3, 160 m) vs the §66 20 m model, and both vs observed (runbook §69.7)
# ============================================================
# All days 2017-2024, Ne1 tile, one forward pass per model per day.
#   Ne1 (7,7) tok 105 / map px (56,56); Ne2 (7,10) tok 108 / (56,83); south (12,6) tok 174 / (100,52).
# Figures (figures/mead/):
#   mead_pw_vs_20m    rows depth, cols Ne1 / Ne2 / south: x = 20 m model, y = patchwise, 1:1 line,
#                     r and mean (patchwise - 20 m)
#   mead_models_vs_obs rows depth, cols Ne1 / Ne2: x = observed, y = model, both models,
#                     RMSE / bias / r each (observed days only; frozen soil, ts <= 0, excluded)
#
# Usage:
#   python plot_mead_pw_compare.py
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style_bw
from plot_mead_pred import ROOT, FIG_DIR, observed

DEPTHS = ["0-10", "10-30", "30-100"]
TOK    = {"Ne1": 105, "Ne2": 108, "south": 174}
PIX    = {"Ne1": (56, 56), "Ne2": (56, 83), "south": (100, 52)}


def main():
    z = np.load(ROOT / "pred" / "pw_maps_Ne1_2017-2024.npz")
    dp = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    pw = {k: pd.DataFrame(z["maps"][:, t], index=dp, columns=DEPTHS) for k, t in TOK.items()}
    z = np.load(ROOT / "pred" / "maps_AmeriFlux_US-Ne1_2017-2024.npz")
    dm = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    m20 = {k: pd.DataFrame(z["sm"][:, :, r, c].astype(np.float32), index=dm, columns=DEPTHS)
           for k, (r, c) in PIX.items()}
    obs, ts = {}, {}
    for k in ("Ne1", "Ne2"):
        obs[k], ts[k] = observed(f"AmeriFlux_US-{k}")

    plot_style_bw.apply(globals(), "paper")
    BL, OR = plot_style_bw.BLUE, plot_style_bw.RED

    # --- patchwise vs 20 m model ---
    fig, ax = plt.subplots(3, 3, figsize=(11, 11))
    print("patchwise vs 20 m model, all days 2017-2024")
    for k, dep in enumerate(DEPTHS):
        for j, key in enumerate(TOK):
            a = ax[k, j]
            x, y = m20[key][dep].align(pw[key][dep], join="inner")
            js = (x.index.month >= 6) & (x.index.month <= 9)
            a.plot(x[~js], y[~js], ".", ms=2, color="0.6", alpha=0.5, label="Oct-May")
            a.plot(x[js], y[js], ".", ms=2, color=BL, alpha=0.6, label="Jun-Sep")
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            a.plot([lo, hi], [lo, hi], "-", color="black", lw=0.7)
            r, b = x.corr(y), (y - x).mean()
            print(f"  {key:>5} {dep:>6}: r {r:+.2f}  mean(patchwise - 20 m) {b:+.3f}")
            a.text(0.03, 0.97, f"({'abcdefghi'[3 * k + j]}) {key}, {dep} cm\nr {r:+.2f}, "
                   f"patchwise $-$ 20 m {b:+.3f}", transform=a.transAxes, va="top", fontsize="small",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
            if k == 2:
                a.set_xlabel("20 m model SM (m$^3$ m$^{-3}$)")
            if j == 0:
                a.set_ylabel("patchwise SM (m$^3$ m$^{-3}$)")
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, markerscale=5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_pw_vs_20m.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_pw_vs_20m.png")

    # --- both models vs observed ---
    fig, ax = plt.subplots(3, 2, figsize=(8.5, 11))
    print("\nmodels vs observed, observed days, ts > 0")
    for k, dep in enumerate(DEPTHS):
        for j, key in enumerate(("Ne1", "Ne2")):
            a = ax[k, j]
            o = obs[key][dep]
            o = o[(ts[key].reindex(o.index) > 0) if ts[key] is not None else o.notna()].dropna()
            lo, hi = o.min(), o.max()
            txt = []
            for src, col, lab, mk in ((m20, OR, "20 m model", "."), (pw, BL, "patchwise", ".")):
                x, y = o.align(src[key][dep], join="inner")
                a.plot(x, y, mk, ms=2, color=col, alpha=0.4, label=lab)
                e = y - x
                txt.append(f"{lab}: RMSE {np.sqrt((e ** 2).mean()):.3f}, bias {e.mean():+.3f}, r {x.corr(y):.2f}")
                print(f"  {key} {dep:>6} {lab:>10}: n {len(x)}  {txt[-1].split(': ')[1]}")
                lo, hi = min(lo, y.min()), max(hi, y.max())
            a.plot([lo, hi], [lo, hi], "-", color="black", lw=0.7)
            a.text(0.03, 0.97, f"({'abcdef'[2 * k + j]}) {key}, {dep} cm\n" + "\n".join(txt),
                   transform=a.transAxes, va="top", fontsize="x-small",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
            if k == 2:
                a.set_xlabel("observed SM (m$^3$ m$^{-3}$)")
            if j == 0:
                a.set_ylabel("model SM (m$^3$ m$^{-3}$)")
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, markerscale=5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_models_vs_obs.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_models_vs_obs.png")


if __name__ == "__main__":
    main()
