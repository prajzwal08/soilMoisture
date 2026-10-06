# ============================================================
# Mead: is the off-centre SM map a fixed decoder pattern?  Tests 1 + 2 (runbook §69.6)
# ============================================================
# Input: maps_AmeriFlux_US-Ne{1,2}_2017-2024.npz (frozen §66 model, every day, (3,112,112) at 20 m).
#
# Test 1 -- same ground, two positions (overlap). Ne2's tile is Ne1's shifted 546 m east and
#   6 m south (bounds_utm) = 27.3 / 0.3 map px -> Ne1 (r, c) is the same ground as Ne2 (r, c-27),
#   c = 27..111 (85 columns). Per depth:
#   corr_ground   = spatial corr of the two tiles' MEAN maps over the SAME ground
#   corr_position = spatial corr of the two tiles' MEAN maps at the SAME map position (full tile)
#   If the map follows the ground, corr_ground > corr_position; a fixed pattern gives the reverse.
#   Also per day: spatial corr of the two tiles over the same ground (daily anomalies).
# Test 2 -- static vs moving. Per tile and depth: M = temporal mean map; each day's spatial
#   anomaly A_t = map_t - its spatial mean. corr(A_t, M - mean M) per day; static share =
#   var_space(M) / mean_t var_space(A_t). Near 1 everywhere = the spatial structure never changes.
#
# Writes figures/mead/mead_map_pattern.png and prints the numbers.
#
# Usage:
#   python analyse_mead_map_pattern.py
# ============================================================

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style_bw
from plot_mead_pred import ROOT, FIG_DIR

RAW    = Path("/projects/prjs1968/satellite_zarr")
TILES  = {"Ne1": "AmeriFlux_US-Ne1", "Ne2": "AmeriFlux_US-Ne2"}
DEPTHS = ["0-10", "10-30", "30-100"]
PX     = 20.0


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def daily_corr(x, y):
    """Per-day spatial corr of (days, H, W) arrays."""
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))


def main():
    maps, dates = {}, {}
    for k, st in TILES.items():
        z = np.load(ROOT / "pred" / f"maps_{st}_2017-2024.npz")
        maps[k], dates[k] = z["sm"].astype(np.float32), pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    assert dates["Ne1"].equals(dates["Ne2"]), "tile dates differ"
    d = dates["Ne1"]
    b = {k: json.loads((RAW / f"{st}.zarr" / ".zattrs").read_text())["bounds_utm"] for k, st in TILES.items()}
    dx, dy = (b["Ne2"][0] - b["Ne1"][0]) / PX, (b["Ne1"][3] - b["Ne2"][3]) / PX
    sh = int(round(dx))
    print(f"{len(d)} days {d[0].date()}..{d[-1].date()}; Ne2 tile offset {dx:.2f} px east, {dy:.2f} px south "
          f"-> column shift {sh}, residual {dx - sh:+.2f} / {dy:+.2f} px")

    res, fig_rows = {}, []
    for k, dep in enumerate(DEPTHS):
        a1, a2 = maps["Ne1"][:, k], maps["Ne2"][:, k]
        m1, m2 = a1.mean(0), a2.mean(0)
        g1, g2 = m1[:, sh:], m2[:, :112 - sh]                      # same ground
        c_ground, c_pos = corr(g1, g2), corr(m1, m2)
        dd = daily_corr(a1[:, :, sh:], a2[:, :, :112 - sh])
        diff = g1 - g2
        out = {"corr_ground": c_ground, "corr_position": c_pos,
               "daily_ground_corr_median": float(np.median(dd)),
               "mean_diff": float(diff.mean()), "sd_diff_space": float(diff.std()),
               "sd_mean_map_space": float(np.mean([g1.std(), g2.std()]))}
        print(f"\n{dep} cm  TEST 1 overlap ({g1.shape[1]} cols):")
        print(f"  mean-map corr, SAME GROUND {c_ground:+.3f}   SAME POSITION {c_pos:+.3f}")
        print(f"  daily same-ground corr: median {np.median(dd):+.3f}  5-95% {np.percentile(dd, 5):+.3f}..{np.percentile(dd, 95):+.3f}")
        print(f"  same-ground diff of mean maps: mean {diff.mean():+.4f}, spatial SD {diff.std():.4f} "
              f"(vs mean-map spatial SD {out['sd_mean_map_space']:.4f})")
        print(f"  TEST 2 static vs moving:")
        for t, arr, m in (("Ne1", a1, m1), ("Ne2", a2, m2)):
            dc = daily_corr(arr, np.broadcast_to(m, arr.shape))
            an = arr - arr.mean((1, 2), keepdims=True)
            share = float(m.var() / an.var((1, 2)).mean())
            out[f"static_corr_median_{t}"], out[f"static_share_{t}"] = float(np.median(dc)), share
            out[f"_dc_{t}"] = dc
            print(f"    {t} tile: day-vs-mean corr median {np.median(dc):+.3f}  5% {np.percentile(dc, 5):+.3f}  "
                  f"min {dc.min():+.3f};  static share of spatial variance {share:.3f}")
        res[dep] = out
        fig_rows.append((dep, m1, m2, diff))

    plot_style_bw.apply(globals(), "paper")
    fig, ax = plt.subplots(len(DEPTHS), 4, figsize=(15, 10.5),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    for k, (dep, m1, m2, diff) in enumerate(fig_rows):
        lo, hi = np.percentile(np.r_[m1.ravel(), m2.ravel()], [2, 98])
        for j, (img, lab) in enumerate(((m1, "Ne1 tile, mean map"), (m2, "Ne2 tile, mean map"))):
            im = ax[k, j].imshow(img, vmin=lo, vmax=hi, cmap="YlGnBu", interpolation="nearest")
            ax[k, j].plot(56, 56, "o", ms=5, mfc="white", mec="black")
            ax[k, j].text(0.02, 0.98, f"({'abcdefghijkl'[4 * k + j]}) {lab}, {dep} cm", transform=ax[k, j].transAxes,
                          va="top", fontsize="x-small", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        fig.colorbar(im, ax=ax[k, :2], fraction=0.02, pad=0.01, label="SM (m$^3$ m$^{-3}$)")
        v = np.abs(diff).max()
        im = ax[k, 2].imshow(diff, vmin=-v, vmax=v, cmap="RdBu", interpolation="nearest")
        ax[k, 2].text(0.02, 0.98, f"({'abcdefghijkl'[4 * k + 2]}) Ne1 $-$ Ne2, same ground\n"
                      f"corr ground {res[dep]['corr_ground']:+.2f}, position {res[dep]['corr_position']:+.2f}",
                      transform=ax[k, 2].transAxes, va="top", fontsize="x-small",
                      bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        fig.colorbar(im, ax=ax[k, 2], fraction=0.046, pad=0.02, label="$\\Delta$SM")
        for a in ax[k, :3]:
            a.set_xticks([])
            a.set_yticks([])
        a = ax[k, 3]
        for t, col in (("Ne1", plot_style_bw.BLUE), ("Ne2", plot_style_bw.GREEN)):
            a.plot(d, res[dep][f"_dc_{t}"], "-", lw=0.6, color=col, label=f"{t} tile")
        a.set_ylim(-1, 1.02)
        a.axhline(0, color="0.6", lw=0.6)
        a.set_ylabel("corr(day, mean map)")
        a.text(0.02, 0.04, f"({'abcdefghijkl'[4 * k + 3]}) static test, {dep} cm", transform=a.transAxes,
               fontsize="x-small", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax[0, 3].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_map_pattern.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nfigure: {FIG_DIR}/mead_map_pattern.png")
    json.dump({dep: {k: v for k, v in r.items() if not k.startswith("_")} for dep, r in res.items()},
              open(ROOT / "pred" / "map_pattern_tests.json", "w"), indent=1)


if __name__ == "__main__":
    main()
