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
    oot_error_vs_doy          §22.7 diagnostic -- OOT error by day-of-year (2023-2025 pooled), OOST as control

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

from eval_metrics import SM_DEPTHS, clip_pred, metrics_from_arrays, per_station_metrics

# §13.3 house style
DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
SPLIT_COLORS = {"oos": "#1a6faf", "oot": "#e8851a", "oost": "#9b59b6", "val": "#7f8c8d"}
SPLIT_LABELS = {"oos": "OOS (novel stations, 2016-2022)",
                "oot": "OOT (seen stations, 2023-2025)",
                "oost": "OOST (novel stations, 2023-2025)",
                "val": "val (internal)"}
HELD_OUT = ["oos", "oot", "oost"]
HEX_CMAP    = "viridis"     # these four are replaced by plot_style_bw.apply under --style bw
SPLIT_MARKER = {}
SPLIT_LS    = {}
BW          = False
FS           = 1.0         # annotation font-size multiplier (paper style raises it)
DPI          = 300
PAPER        = False       # paper style: no in-figure titles/descriptions (the caption carries them)
CS           = 1.0         # extra multiplier for count / median annotations (paper style)
BOX_ALPHA    = 0.55        # box fill opacity (bw + paper: 1.0, solid)
XROT         = None        # category tick rotation override (paper style: 90)
SM_LIM   = (0.0, 0.62)


def _afs(base: float) -> float:
    """Annotation font size: base*FS, never below the tick-label size in paper style."""
    if not PAPER:
        return base * FS
    ts = plt.rcParams.get("xtick.labelsize", 11)
    ts = ts if isinstance(ts, (int, float)) else 11
    return max(base * FS, ts)


def _note_color() -> str:
    """Secondary text/guide colour: grey in the colour style, black in paper style."""
    return "black" if PAPER else "grey"


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir/name}.png")


def load_predictions(in_dir: Path) -> dict:
    out = {}
    for split in HELD_OUT:
        p = in_dir / f"predictions_{split}.parquet"
        if p.exists():
            out[split] = clip_pred(pd.read_parquet(p))
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
                        transform=ax.transAxes, fontsize=_afs(8), color=_note_color())
                continue

            p = g["pred"].to_numpy(np.float64)
            t = g["obs"].to_numpy(np.float64)
            hb = ax.hexbin(t, p, gridsize=55, extent=(*SM_LIM, *SM_LIM),
                           bins="log", mincnt=1, cmap=HEX_CMAP, linewidths=0)
            ax.plot(SM_LIM, SM_LIM, "k--", lw=0.8, zorder=3)

            m = metrics_from_arrays(p, t)
            if PAPER:
                # short box in the lower-right corner (below the 1:1 line at high
                # obs, where pred-vs-obs scatter is sparse), semi-transparent
                ax.text(0.97, 0.03,
                        f"ubRMSE {m['ubRMSE']:.3f}\n$r^2$ {m['R2_pearson']:.2f}\n"
                        f"bias {m['bias']:+.3f}\nn {m['n']:,}",
                        transform=ax.transAxes, va="bottom", ha="right",
                        fontsize=_afs(6), zorder=4,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            else:
                ax.text(0.03, 0.97,
                        f"RMSE {m['RMSE']:.3f}\nubRMSE {m['ubRMSE']:.3f}\n"
                        f"$r^2$ {m['R2_pearson']:.2f}\nNSE {m['NSE']:+.2f}\n"
                        f"bias {m['bias']:+.3f}\nn {m['n']:,}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=6 * FS,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

            ax.set_xlim(SM_LIM); ax.set_ylim(SM_LIM); ax.set_aspect("equal")
            if i == 0:
                ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            if j == 0:
                ax.set_ylabel(f"{split.upper()}\npredicted SM (m$^3$/m$^3$)")
            if i == len(splits) - 1:
                ax.set_xlabel("observed SM (m$^3$/m$^3$)")

    fig.colorbar(hb, ax=axes[:, -1].tolist(), label="samples per bin (log)",
                 shrink=1.0 if PAPER else 0.6)   # paper: span the full column height
    if not PAPER:
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
                    transform=ax.transAxes, va="top", ha="left", fontsize=_afs(6),
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

            ax.set_xlim(SM_LIM); ax.set_ylim(SM_LIM); ax.set_aspect("equal")
            if i == 0:
                ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            if j == 0:
                ax.set_ylabel(f"{split.upper()}\nmean predicted (m$^3$/m$^3$)")
            if i == len(splits) - 1:
                ax.set_xlabel("mean observed (m$^3$/m$^3$)")

    if not PAPER:
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
        filled = []                                   # axes that received data
        for j, depth in enumerate(SM_DEPTHS):
            ax = axes[i][j]
            data = []
            for k, split in enumerate(splits):
                v = ps_all[(ps_all["eval_split"] == split) &
                           (ps_all["depth"] == depth)][metric].dropna().to_numpy()
                data.append(v)
                if len(v):
                    x = k + rng.uniform(-0.13, 0.13, len(v))
                    if ylim:
                        # fixed limits: draw out-of-range stations AT the limit with a
                        # triangle pointing off-axis, so clipping is never silent
                        lo, hi = ylim
                        inr = (v >= lo) & (v <= hi)
                        ax.scatter(x[inr], v[inr], s=7, alpha=0.5, c=SPLIT_COLORS[split],
                                   edgecolors="none", zorder=2)
                        for sel, yv, mk in ((v < lo, lo, "v"), (v > hi, hi, "^")):
                            if sel.any():
                                ax.scatter(x[sel], np.full(sel.sum(), yv), s=16,
                                           marker=mk, c=SPLIT_COLORS[split],
                                           edgecolors="k", linewidths=0.3,
                                           zorder=4, clip_on=False)
                    else:
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
            # station count per split lives in the tick label (e.g. "OOS\n221")
            ax.set_xticklabels([f"{s.upper()}\n{len(d)}" for s, d in zip(splits, data)],
                               fontsize=_afs(7))
            if ylim:
                ax.set_ylim(*ylim)
            if metric == "NSE_anom":
                ax.axhline(0, color=_note_color(), lw=0.6, ls=":")
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            if any(len(d) for d in data):
                filled.append(ax)

        if not ylim and filled:
            # free-scale rows (ubRMSE): one shared y range across depths
            los, his = zip(*(a.get_ylim() for a in filled))
            for a in axes[i]:
                a.set_ylim(min(los), max(his))

    if not PAPER:
        fig.suptitle("Per-station metric distributions (one dot = one station)", y=1.01)
    save(fig, out_dir, "scatter_station_metrics")


# ── 4. Dynamics error vs level error ──────────────────────────────────────────

def fig_ubrmse_vs_offset(ps_all: pd.DataFrame, out_dir: Path):
    """MSE ~ ubRMSE^2 + bias^2 (§20.1). Points ABOVE the diagonal (|bias| > ubRMSE)
    are level-limited: the model tracks the dynamics but sits at the wrong level."""
    fig, axes = plt.subplots(1, len(SM_DEPTHS), figsize=(9.0, 3.1),
                             constrained_layout=True, squeeze=False)
    splits = [s for s in HELD_OUT if s in ps_all["eval_split"].unique()]
    # one data-driven limit for both axes and all panels: 99.5th pct of
    # max(ubRMSE, |bias|) plus 10% headroom (the old fixed 0-0.20 clipped points)
    held = ps_all[ps_all["eval_split"].isin(splits)]
    span = np.fmax(held["ubRMSE"].to_numpy(np.float64),
                   held["bias"].abs().to_numpy(np.float64))
    span = span[np.isfinite(span)]
    lim = (0.0, float(np.percentile(span, 99.5)) * 1.1 if len(span) else 0.2)
    for j, depth in enumerate(SM_DEPTHS):
        ax = axes[0][j]
        for split in splits:
            g = ps_all[(ps_all["eval_split"] == split) & (ps_all["depth"] == depth)]
            if g.empty:
                continue
            ax.scatter(g["ubRMSE"], g["bias"].abs(), s=13, alpha=0.65,
                       c=SPLIT_COLORS[split], edgecolors="k" if BW else "none",
                       linewidths=0.3, marker=SPLIT_MARKER.get(split, "o"),
                       label=split.upper())
        ax.plot(lim, lim, "k--", lw=0.8, zorder=1)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xlabel("ubRMSE (dynamics error)")
        ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
        # upper-left = above the diagonal = the (sparse) level-limited region
        ax.text(0.04, 0.96, "above line:\nlevel-limited", transform=ax.transAxes, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
                ha="left", va="top", fontsize=_afs(6), color=_note_color())
        if j == 0:
            ax.set_ylabel("|per-station bias| (level error)")
            # "best" avoids the points and the note above (drawn first)
            if not PAPER:
                ax.legend(fontsize=_afs(6), frameon=False, loc="best")
    if PAPER:                           # one legend above the panels, never on the points
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(l),
                   fontsize=_afs(6), frameon=False, markerscale=1.5)

    if not PAPER:
        fig.suptitle("Dynamics error vs absolute-level error, per station", y=1.03)
    save(fig, out_dir, "scatter_ubrmse_vs_offset")


# ── 5. §22.7 diagnostic: OOT error by day-of-year ───────────────────────────────

def fig_oot_error_vs_doy(preds: dict, out_dir: Path, n_bins: int = 24):
    """OOT seen-context fraction falls 100% -> 0% across 2023 (§22.3).

    Rising OOT with flat OOST  => reliance on memorised input context.
    Both tracing the same shape => seasonality; the diagnostic says nothing.
    Errors are per-station-standardised first, so stations entering or leaving
    the record mid-year cannot masquerade as a trend.

    NOTE (§47): OOT/OOST now span 2023-2025 (splits_config.OOT_YEARS). Samples are
    binned by day-of-year ONLY, so all OOT years are pooled into one seasonal
    cycle -- the x axis is not calendar 2023, and the 2023 context decay is
    diluted by the 2024-2025 samples (input windows wholly past the cut).
    """
    if "oot" not in preds:
        print("  (skipping oot_error_vs_doy -- no OOT predictions)")
        return

    yrs = sorted({int(y) for s in ("oot", "oost") if s in preds
                  and "year" in preds[s] for y in pd.unique(preds[s]["year"])})
    if not yrs:
        xlabel = "Day of year (OOT years pooled)"
    elif yrs[0] == yrs[-1]:
        xlabel = f"Day of year ({yrs[0]})"
    else:
        xlabel = f"Day of year ({yrs[0]}-{yrs[-1]} pooled)"

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
            ax.plot(centers, mean, ms=2.5, lw=1.1,
                    color=SPLIT_COLORS[split], ls=SPLIT_LS.get(split, "-"),
                    marker=SPLIT_MARKER.get(split, "o"), label=split.upper())
            ax.fill_between(centers, mean - se, mean + se, alpha=0.18,
                            color=SPLIT_COLORS[split], lw=0)

        ax.axhline(0, color=_note_color(), lw=0.6, ls=":")
        ax.set_xlim(0, 366)
        ax.set_xlabel(xlabel)
        ax.set_title(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
        if j == 0:
            ax.set_ylabel("|error| anomaly (m$^3$/m$^3$)\nper-station mean removed")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if PAPER and handles:
        # one figure legend above the panels: never covers the OOST trough
        # anchored just above the figure top (above the depth titles); bbox_inches
        # ="tight" in save() keeps it in the file (works on any matplotlib >= 3.x)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   ncol=len(labels), fontsize=_afs(6), frameon=False)
    elif handles:
        axes[0][0].legend(fontsize=_afs(6), frameon=False, loc="best")

    if not PAPER:
        fig.suptitle("§22.7  OOT error, day-of-year (OOST = seasonality control)", y=1.04)
    save(fig, out_dir, "oot_error_vs_doy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",  default="eval_output")
    p.add_argument("--out-dir", default="figures/eval")
    p.add_argument("--style", choices=["color", "bw", "paper"], default="color",
                   help="bw = black-and-white; paper = blue/green/red, Times (plot_style_bw.py)")
    args = p.parse_args()
    if args.style != "color":
        import plot_style_bw
        plot_style_bw.apply(globals(), args.style)

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
