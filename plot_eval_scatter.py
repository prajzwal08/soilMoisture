"""
Scatter diagnostics for the held-out splits (§22, Phase 3).

CPU only -- reads eval_output/predictions_{split}.parquet.  Seconds to run,
so styling and binning can be iterated freely.

Figures (PNG + PDF, dpi 300, house style §13.3):
    scatter_pred_obs          3x3 -- every observation vs its prediction, 1:1 line
    scatter_station_mean      3x3 -- one dot per station: mean pred vs mean obs
                                     (isolates the §20.1 absolute-level failure)
    scatter_station_metrics   per-station metric distributions across splits
    scatter_ubrmse_vs_offset  dynamics error vs level error (MSE ~ ubRMSE^2 + bias^2)
    oot_error_vs_doy          §22.7 diagnostic -- OOT error through 2023, OOST as control

Usage:
    python plot_eval_scatter.py [--in-dir eval_output] [--out-dir figures/eval]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scienceplots        # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})

from eval_metrics import SM_DEPTHS, metrics_from_arrays, per_station_metrics

# §13.3 house style
DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
SPLIT_COLORS = {"oos": "#1a6faf", "oot": "#e8851a", "oost": "#9b59b6", "val": "#7f8c8d"}
SPLIT_LABELS = {"oos": "OOS (novel stations, 2016-2022)",
                "oot": "OOT (seen stations, 2023)",
                "oost": "OOST (novel stations, 2023)",
                "val": "val (internal)"}
HELD_OUT = ["oos", "oot", "oost"]
SM_LIM   = (0.0, 0.62)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir/name}.png")


def load_predictions(in_dir: Path) -> dict:
    out = {}
    for split in HELD_OUT:
        p = in_dir / f"predictions_{split}.parquet"
        if p.exists():
            out[split] = pd.read_parquet(p)
    if not out:
        raise SystemExit(f"No prediction parquets in {in_dir}")
    return out


# ── 1. Predicted vs observed, every sample ────────────────────────────────────

def fig_pred_obs(preds: dict, out_dir: Path):
    splits = [s for s in HELD_OUT if s in preds]
    fig, axes = plt.subplots(len(splits), len(SM_DEPTHS),
                             figsize=(9.0, 3.0 * len(splits)),
                             constrained_layout=True, squeeze=False)

    for i, split in enumerate(splits):
        df = preds[split]
        for j, depth in enumerate(SM_DEPTHS):
            ax = axes[i][j]
            g = df[df["depth"] == depth]
            if g.empty:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                continue

            p = g["pred"].to_numpy(np.float64)
            t = g["obs"].to_numpy(np.float64)
            hb = ax.hexbin(t, p, gridsize=55, extent=(*SM_LIM, *SM_LIM),
                           bins="log", mincnt=1, cmap="viridis", linewidths=0)
            ax.plot(SM_LIM, SM_LIM, "k--", lw=0.8, zorder=3)

            m = metrics_from_arrays(p, t)
            ax.text(0.03, 0.97,
                    f"RMSE {m['RMSE']:.3f}\nubRMSE {m['ubRMSE']:.3f}\n"
                    f"$r^2$ {m['R2_pearson']:.2f}\nNSE {m['NSE']:+.2f}\n"
                    f"bias {m['bias']:+.3f}\nn {m['n']:,}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

            ax.set_xlim(SM_LIM); ax.set_ylim(SM_LIM); ax.set_aspect("equal")
            if i == 0:
                ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            if j == 0:
                ax.set_ylabel(f"{split.upper()}\npredicted SM (m$^3$/m$^3$)")
            if i == len(splits) - 1:
                ax.set_xlabel("observed SM (m$^3$/m$^3$)")

    fig.colorbar(hb, ax=axes[:, -1].tolist(), label="samples per bin (log)",
                 shrink=0.6)
    fig.suptitle("Predicted vs observed soil moisture -- held-out splits", y=1.01)
    save(fig, out_dir, "scatter_pred_obs")


# ── 2. Station-mean predicted vs observed ─────────────────────────────────────

def fig_station_mean(preds: dict, out_dir: Path):
    """One dot per station.  This is where the §20.1 level failure shows."""
    splits = [s for s in HELD_OUT if s in preds]
    fig, axes = plt.subplots(len(splits), len(SM_DEPTHS),
                             figsize=(9.0, 3.0 * len(splits)),
                             constrained_layout=True, squeeze=False)

    for i, split in enumerate(splits):
        df = preds[split]
        for j, depth in enumerate(SM_DEPTHS):
            ax = axes[i][j]
            g = df[df["depth"] == depth]
            if g.empty:
                ax.set_axis_off(); continue

            st = g.groupby("station_key", observed=True)[["pred", "obs"]].mean()
            ax.scatter(st["obs"], st["pred"], s=14, alpha=0.75,
                       c=SPLIT_COLORS[split], edgecolors="k", linewidths=0.3)
            ax.plot(SM_LIM, SM_LIM, "k--", lw=0.8)

            resid = st["pred"] - st["obs"]
            rms_off = float(np.sqrt(np.mean(resid ** 2)))
            r = (float(np.corrcoef(st["obs"], st["pred"])[0, 1])
                 if len(st) > 2 and st["obs"].std() > 0 and st["pred"].std() > 0
                 else np.nan)
            ax.text(0.03, 0.97,
                    f"RMS offset {rms_off:.3f}\n$r$ {r:.2f}\n{len(st)} stations",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

            ax.set_xlim(SM_LIM); ax.set_ylim(SM_LIM); ax.set_aspect("equal")
            if i == 0:
                ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            if j == 0:
                ax.set_ylabel(f"{split.upper()}\nmean predicted (m$^3$/m$^3$)")
            if i == len(splits) - 1:
                ax.set_xlabel("mean observed (m$^3$/m$^3$)")

    fig.suptitle("Station-mean predicted vs observed -- absolute level only", y=1.01)
    save(fig, out_dir, "scatter_station_mean")


# ── 3. Per-station metric distributions ───────────────────────────────────────

def fig_station_metrics(ps_all: pd.DataFrame, out_dir: Path):
    metrics = [("ubRMSE", "ubRMSE (m$^3$/m$^3$)", None),
               ("R2_pearson", "$r^2$", (0, 1)),
               ("NSE_anom", "NSE (anomaly)", (-1, 1))]
    splits = [s for s in HELD_OUT if s in ps_all["eval_split"].unique()]

    fig, axes = plt.subplots(len(metrics), len(SM_DEPTHS),
                             figsize=(9.0, 2.7 * len(metrics)),
                             constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)

    for i, (metric, label, ylim) in enumerate(metrics):
        for j, depth in enumerate(SM_DEPTHS):
            ax = axes[i][j]
            data = []
            for k, split in enumerate(splits):
                v = ps_all[(ps_all["eval_split"] == split) &
                           (ps_all["depth"] == depth)][metric].dropna().to_numpy()
                data.append(v)
                if len(v):
                    x = k + rng.uniform(-0.13, 0.13, len(v))
                    ax.scatter(x, v, s=7, alpha=0.5, c=SPLIT_COLORS[split],
                               edgecolors="none", zorder=2)
            bp = ax.boxplot(data, positions=range(len(splits)), widths=0.5,
                            showfliers=False, zorder=3,
                            medianprops=dict(color="k", lw=1.2),
                            boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7),
                            capprops=dict(lw=0.7))
            for patch, split in zip(bp["boxes"], splits):
                patch.set_alpha(0.9)

            ax.set_xticks(range(len(splits)))
            ax.set_xticklabels([s.upper() for s in splits], fontsize=7)
            if ylim:
                ax.set_ylim(*ylim)
            if metric == "NSE_anom":
                ax.axhline(0, color="grey", lw=0.6, ls=":")
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            ax.text(0.98, 0.03, "\n".join(f"n={len(d)}" for d in [data[0]]),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                    color="grey")

    fig.suptitle("Per-station metric distributions (one dot = one station)", y=1.01)
    save(fig, out_dir, "scatter_station_metrics")


# ── 4. Dynamics error vs level error ──────────────────────────────────────────

def fig_ubrmse_vs_offset(ps_all: pd.DataFrame, out_dir: Path):
    """MSE ~ ubRMSE^2 + bias^2 (§20.1). Points below the diagonal are
    level-limited: the model tracks the dynamics but sits at the wrong level."""
    fig, axes = plt.subplots(1, len(SM_DEPTHS), figsize=(9.0, 3.1),
                             constrained_layout=True, squeeze=False)
    for j, depth in enumerate(SM_DEPTHS):
        ax = axes[0][j]
        for split in [s for s in HELD_OUT if s in ps_all["eval_split"].unique()]:
            g = ps_all[(ps_all["eval_split"] == split) & (ps_all["depth"] == depth)]
            if g.empty:
                continue
            ax.scatter(g["ubRMSE"], g["bias"].abs(), s=13, alpha=0.65,
                       c=SPLIT_COLORS[split], edgecolors="none",
                       label=split.upper())
        lim = (0, 0.2)
        ax.plot(lim, lim, "k--", lw=0.8, zorder=1)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xlabel("ubRMSE (dynamics error)")
        ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
        if j == 0:
            ax.set_ylabel("|per-station bias| (level error)")
            ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.text(0.5, 0.02, "above line: level-limited", transform=ax.transAxes,
                ha="center", fontsize=6, color="grey")

    fig.suptitle("Dynamics error vs absolute-level error, per station", y=1.03)
    save(fig, out_dir, "scatter_ubrmse_vs_offset")


# ── 5. §22.7 diagnostic: OOT error through 2023 ───────────────────────────────

def fig_oot_error_vs_doy(preds: dict, out_dir: Path, n_bins: int = 24):
    """OOT seen-context fraction falls 100% -> 0% across 2023 (§22.3).

    Rising OOT with flat OOST  => reliance on memorised input context.
    Both tracing the same shape => seasonality; the diagnostic says nothing.
    Errors are per-station-standardised first, so stations entering or leaving
    the record mid-year cannot masquerade as a trend.
    """
    if "oot" not in preds:
        print("  (skipping oot_error_vs_doy -- no OOT predictions)")
        return

    fig, axes = plt.subplots(1, len(SM_DEPTHS), figsize=(9.0, 3.0),
                             constrained_layout=True, squeeze=False)
    edges   = np.linspace(0, 366, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    for j, depth in enumerate(SM_DEPTHS):
        ax = axes[0][j]
        for split in ("oot", "oost"):
            if split not in preds:
                continue
            g = preds[split]
            g = g[g["depth"] == depth].copy()
            if g.empty:
                continue
            g["abserr"] = (g["pred"] - g["obs"]).abs()
            # standardise within station so composition changes cannot fake a trend
            g["z"] = g.groupby("station_key", observed=True)["abserr"].transform(
                lambda s: s - s.mean())
            idx = np.digitize(g["doy"].to_numpy(), edges) - 1
            idx = np.clip(idx, 0, n_bins - 1)
            mean = np.array([g["z"].to_numpy()[idx == b].mean() if (idx == b).any()
                             else np.nan for b in range(n_bins)])
            se = np.array([
                (g["z"].to_numpy()[idx == b].std(ddof=1) /
                 max(np.sqrt((idx == b).sum()), 1)) if (idx == b).sum() > 1 else np.nan
                for b in range(n_bins)])
            ax.plot(centers, mean, "-o", ms=2.5, lw=1.1,
                    color=SPLIT_COLORS[split], label=split.upper())
            ax.fill_between(centers, mean - se, mean + se, alpha=0.18,
                            color=SPLIT_COLORS[split], lw=0)

        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlim(0, 366)
        ax.set_xlabel("day of year, 2023")
        ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
        if j == 0:
            ax.set_ylabel("|error| anomaly (m$^3$/m$^3$)\nper-station mean removed")
            ax.legend(fontsize=6, frameon=False)

    fig.suptitle("§22.7  OOT error through 2023 (OOST = seasonality control)", y=1.04)
    save(fig, out_dir, "oot_error_vs_doy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default="eval_output")
    p.add_argument("--out-dir", default="figures/eval")
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    preds = load_predictions(in_dir)
    print(f"Loaded splits: {', '.join(preds)}")

    ps_all = []
    for split, df in preds.items():
        ps = per_station_metrics(df)
        ps["eval_split"] = split
        ps_all.append(ps)
    ps_all = pd.concat(ps_all, ignore_index=True)

    fig_pred_obs(preds, out_dir)
    fig_station_mean(preds, out_dir)
    fig_station_metrics(ps_all, out_dir)
    fig_ubrmse_vs_offset(ps_all, out_dir)
    fig_oot_error_vs_doy(preds, out_dir)
    print(f"\nAll figures in {out_dir}/")


if __name__ == "__main__":
    main()
