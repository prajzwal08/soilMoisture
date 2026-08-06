"""
Predicted vs observed time series for the best and worst stations per split (§22).

CPU only -- everything comes from eval_output/predictions_{split}.parquet, so
re-plotting all 60 figures takes seconds.  (Contrast plot_timeseries_meeting.py,
which rebuilds the dataset and re-runs GPU inference for every station.)

One figure per station: 3 stacked panels, one per depth.
    observed = black dots, predicted = coloured line (depth colour, §13.3)

Outputs:
    eval_output/timeseries/{split}/{BEST|WORST}_{NN}_{station}.png
    eval_output/timeseries/contact_{split}.pdf     -- multi-page contact sheet

Usage:
    python plot_eval_timeseries.py [--n 10] [--splits oos oot oost] [--min-n 100]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

try:
    import scienceplots        # noqa: F401
    plt.style.use(["science", "nature"])
except ImportError:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})

from eval_metrics import SM_DEPTHS, metrics_from_arrays, _make_key

DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
SPLITS_CSV   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
RANK_DEPTH   = "0-10"
GAP_DAYS     = 15      # break the prediction line across gaps longer than this


def select_stations(df: pd.DataFrame, n: int, min_n: int, rank_metric: str):
    """Best-n and worst-n by rank_metric at 0-10 cm.

    The min_n guard matters: n ranges from 5 to ~2500 days across stations, so
    an unguarded ranking selects short records rather than well-modelled ones.
    Falls back to a lower threshold rather than silently returning too few.
    """
    g = df[df["depth"] == RANK_DEPTH]
    rows = []
    for station, s in g.groupby("station_key", observed=True):
        m = metrics_from_arrays(s["pred"].to_numpy(np.float64),
                                s["obs"].to_numpy(np.float64))
        rows.append({"station_key": station, "n": m["n"], rank_metric: m[rank_metric]})
    stats = pd.DataFrame(rows).dropna(subset=[rank_metric])

    eligible = stats[stats["n"] >= min_n]
    if len(eligible) < 2 * n:
        relaxed = int(stats["n"].quantile(0.5)) if len(stats) else 0
        print(f"    only {len(eligible)} stations with n >= {min_n}; "
              f"relaxing to n >= {relaxed}")
        eligible = stats[stats["n"] >= relaxed]
    if eligible.empty:
        return []

    eligible = eligible.sort_values(rank_metric)
    best  = eligible.head(n).assign(rank="BEST")
    worst = eligible.tail(n).iloc[::-1].assign(rank="WORST")
    sel   = pd.concat([best, worst], ignore_index=True)
    sel["rank_idx"] = sel.groupby("rank").cumcount() + 1
    return sel.to_dict("records")


def plot_station(df: pd.DataFrame, info: dict, meta: dict, split: str,
                 n_total: int) -> plt.Figure:
    """3 stacked depth panels for one station."""
    station = info["station_key"]
    fig, axes = plt.subplots(len(SM_DEPTHS), 1, figsize=(7.4, 5.6),
                             sharex=True, constrained_layout=True)

    for ax, depth in zip(axes, SM_DEPTHS):
        g = df[(df["station_key"] == station) & (df["depth"] == depth)]
        g = g.sort_values("date")
        if g.empty:
            ax.text(0.5, 0.5, f"no data at {DEPTH_LABELS[depth]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="grey")
            ax.set_ylabel(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            ax.set_yticks([])
            continue

        # Break the line across data gaps -- otherwise matplotlib draws a
        # straight segment across months of missing record, which reads as a
        # confident flat prediction that was never made.
        dates = g["date"].to_numpy()
        pred  = g["pred"].to_numpy(np.float64).copy()
        gap   = np.diff(dates).astype("timedelta64[D]").astype(int) > GAP_DAYS
        pred[np.append(gap, False)] = np.nan

        ax.plot(dates, pred, "-", lw=0.9, color=DEPTH_COLORS[depth],
                label="predicted", zorder=2)
        ax.plot(g["date"], g["obs"], ".", ms=1.9, color="black",
                label="observed", zorder=3)

        m = metrics_from_arrays(g["pred"].to_numpy(np.float64),
                                g["obs"].to_numpy(np.float64))
        ax.text(0.005, 0.96,
                f"ubRMSE {m['ubRMSE']:.3f}   RMSE {m['RMSE']:.3f}   "
                f"$r^2$ {m['R2_pearson']:.2f}   NSE {m['NSE']:+.2f}   "
                f"bias {m['bias']:+.3f}   n {m['n']}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

        # OOS stations continue into 2023 as OOST -- mark the boundary
        if split == "oos" and g["date"].max() >= pd.Timestamp("2023-01-01"):
            ax.axvspan(pd.Timestamp("2023-01-01"), g["date"].max(),
                       color="#9b59b6", alpha=0.08, lw=0, zorder=0)
            ax.axvline(pd.Timestamp("2023-01-01"), color="#9b59b6",
                       lw=0.8, ls="--", zorder=1)

        ax.set_ylabel(f"{DEPTH_LABELS[depth]}\nSM (m$^3$/m$^3$)",
                      color=DEPTH_COLORS[depth])
        ax.set_ylim(0, max(0.55, float(g[["pred", "obs"]].to_numpy().max()) * 1.1))
        ax.margins(x=0.01)

    axes[0].legend(fontsize=6, frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("date")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        mdates.AutoDateLocator()))

    bits = [f"{station}", f"[{split.upper()} {info['rank']} "
                          f"{info['rank_idx']}/{n_total}]"]
    if meta:
        loc = f"{meta.get('latitude', float('nan')):.2f}, " \
              f"{meta.get('longitude', float('nan')):.2f}"
        bits.append(f"{meta.get('IGBP', '?')} | {meta.get('koppen_geiger', '?')} "
                    f"| {loc}")
    fig.suptitle("   ".join(bits), fontsize=8)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir",      default="eval_output")
    p.add_argument("--out-dir",     default="eval_output/timeseries")
    p.add_argument("--splits",      nargs="+", default=["oos", "oot", "oost"])
    p.add_argument("--n",           type=int, default=10)
    p.add_argument("--min-n",       type=int, default=100)
    p.add_argument("--rank-metric", default="ubRMSE",
                   choices=["ubRMSE", "RMSE", "MAE", "NSE", "R2_pearson"])
    p.add_argument("--per-page",    type=int, default=6)
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    meta_df = pd.read_csv(SPLITS_CSV)
    meta_df["station_key"] = meta_df.apply(_make_key, axis=1)
    meta_map = meta_df.set_index("station_key").to_dict("index")

    for split in args.splits:
        path = in_dir / f"predictions_{split}.parquet"
        if not path.exists():
            print(f"[{split}] no parquet -- skipping")
            continue

        df = pd.read_parquet(path)

        # An OOS station continues into 2023 as OOST; show both in one figure
        # so the temporal extrapolation is visible on the same axes.
        if split == "oos":
            oost_path = in_dir / "predictions_oost.parquet"
            if oost_path.exists():
                df = pd.concat([df, pd.read_parquet(oost_path)], ignore_index=True)

        print(f"\n[{split}] {len(df):,} rows | "
              f"{df['station_key'].nunique()} stations")
        selected = select_stations(df, args.n, args.min_n, args.rank_metric)
        if not selected:
            print("    no station qualified -- skipping")
            continue

        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"contact_{split}.pdf"
        figs = []

        for info in selected:
            fig = plot_station(df, info, meta_map.get(info["station_key"], {}),
                               split, args.n)
            name = (f"{info['rank']}_{info['rank_idx']:02d}_"
                    f"{info['station_key']}.png")
            fig.savefig(split_dir / name, dpi=300, bbox_inches="tight")
            figs.append(fig)
            print(f"    {info['rank']:>5s} {info['rank_idx']:>2d}  "
                  f"{info['station_key']:<45s} "
                  f"{args.rank_metric}={info[args.rank_metric]:.4f}  "
                  f"n={info['n']}")

        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        for fig in figs:
            plt.close(fig)
        print(f"    → {split_dir}/  ({len(figs)} figures)")
        print(f"    → {pdf_path}")


if __name__ == "__main__":
    main()
