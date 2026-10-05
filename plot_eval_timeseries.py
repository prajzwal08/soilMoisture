"""
Predicted vs observed time series for the best and worst stations per split (§22).

CPU only -- everything comes from eval_output/predictions_{split}.parquet, so
re-plotting all 60 figures takes seconds.  (Contrast plot_timeseries_meeting.py,
which rebuilds the dataset and re-runs GPU inference for every station.)

One figure per station: 3 stacked panels, one per depth.
    observed = black dots, predicted = coloured line (depth colour, §13.3)

Three selection modes:
    --select extremes  best-n and worst-n by ubRMSE at 0-10 cm (default)
    --select median    n random stations either side of the split median --
                       typical cases rather than tails, all three depths present
    --select named     exactly the stations given by --stations, in that order,
                       with no ranking and no min-n filter

Outputs:
    eval_output/timeseries/{split}/{BEST|WORST}_{NN}_{station}.png
    eval_output/timeseries/contact_{split}.pdf     -- multi-page contact sheet
    eval_output/timeseries/median_sample/{split}/{BELOW|ABOVE}_{NN}_{station}.png
    eval_output/timeseries/median_sample/selected_median_sample.csv
    eval_output/timeseries/named/{split}/NAMED_{NN}_{station}.png
    eval_output/timeseries/named/selected_named.csv

Usage:
    python plot_eval_timeseries.py [--n 10] [--splits oos oot oost] [--min-n 100]
    python plot_eval_timeseries.py --select median --n-each 2 [--seed 0]
    python plot_eval_timeseries.py --select named --splits val \\
        --stations ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25
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

from eval_metrics import SM_DEPTHS, clip_pred, metrics_from_arrays, _make_key

DEPTH_COLORS = {"0-10": "#e74c3c", "10-30": "#2980b9", "30-100": "#27ae60"}
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
SPLITS_CSV   = Path("/gpfs/work3/0/prjs1968/soilMoisture/csvs/station_splits.csv")
RANK_DEPTH   = "0-10"
GAP_DAYS     = 15      # break the prediction line across gaps longer than this
# replaced by plot_style_bw.apply under --style bw (None = depth colour / the defaults below)
PRED_COLOR   = None
OBS_COLOR    = "black"
OOT_SHADE    = "#9b59b6"
BW           = False
FS           = 1.0         # annotation font-size multiplier (paper style raises it)
DPI          = 300
PAPER        = False       # paper style: no in-figure titles/descriptions (the caption carries them)
CS           = 1.0         # extra multiplier for count / median annotations (paper style)
BOX_ALPHA    = 0.55        # box fill opacity (bw + paper: 1.0, solid)
LINE_DF      = None        # every-day predictions for the line (--line-dir); None = observed days only
XROT         = None        # category tick rotation override (paper style: 90)


def select_stations(df: pd.DataFrame, n: int, min_n: int, rank_metric: str,
                    rank_depth: str = RANK_DEPTH):
    """Best-n and worst-n by rank_metric at rank_depth.

    The min_n guard matters: n ranges from 5 to ~2500 days across stations, so
    an unguarded ranking selects short records rather than well-modelled ones.
    Falls back to a lower threshold rather than silently returning too few.
    """
    g = df[df["depth"] == rank_depth]
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


def select_around_median(df: pd.DataFrame, n_each: int, min_n: int,
                         rank_metric: str, rank_depth: str, seed: int):
    """n_each random stations below and above the split median of rank_metric.

    A station has one ubRMSE per depth, not one overall, so the draw is ranked
    on rank_depth and restricted to stations that carry all three depths --
    otherwise a "below median" pick can be missing the panels it was chosen to
    illustrate.  Random rather than extreme: these are meant to be typical of
    each side, not the tails that select_stations() already covers.
    """
    per_depth, counts = {}, {}
    for depth in SM_DEPTHS:
        g = df[df["depth"] == depth]
        rows = []
        for station, s in g.groupby("station_key", observed=True):
            m = metrics_from_arrays(s["pred"].to_numpy(np.float64),
                                    s["obs"].to_numpy(np.float64))
            rows.append({"station_key": station, "n": m["n"],
                         rank_metric: m[rank_metric]})
        d = pd.DataFrame(rows)
        d = d[(d["n"] >= min_n)].dropna(subset=[rank_metric])
        per_depth[depth] = d.set_index("station_key")[rank_metric]
        counts[depth] = d.set_index("station_key")["n"]

    common = set.intersection(*(set(v.index) for v in per_depth.values()))
    stats = per_depth[rank_depth].loc[sorted(common)]
    if len(stats) < 2 * n_each:
        print(f"    only {len(stats)} stations with all depths and n >= {min_n}")
        if stats.empty:
            return []

    med = float(stats.median())
    rng = np.random.default_rng(seed)
    out = []
    for side, pool in (("BELOW", stats[stats < med]), ("ABOVE", stats[stats > med])):
        take = min(n_each, len(pool))
        if take < n_each:
            print(f"    {side} median: only {len(pool)} stations available")
        picks = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        for i, station in enumerate(sorted(picks), start=1):
            rec = {"station_key": station, "rank": side, "rank_idx": i,
                   rank_metric: float(per_depth[rank_depth][station]),
                   "n": int(counts[rank_depth][station]),
                   "split_median": med}
            for depth in SM_DEPTHS:
                rec[f"{rank_metric}_{depth}"] = float(per_depth[depth][station])
                rec[f"n_{depth}"] = int(counts[depth][station])
            out.append(rec)
    return out


def select_named(df: pd.DataFrame, stations: list[str], min_n: int,
                 rank_metric: str, rank_depth: str):
    """Exactly the stations asked for, in the order given.

    No ranking and no min_n filter: the caller named these, so a short record is
    still the record they wanted to see.  A station absent from the split is
    reported by name rather than silently dropped -- "I asked for six and got
    four" must be visible in the log, not inferred from the file count.
    """
    g = df[df["depth"] == rank_depth]
    present = set(df["station_key"].unique())
    out = []
    for i, station in enumerate(stations, start=1):
        if station not in present:
            print(f"    NOT IN SPLIT: {station}")
            continue
        s = g[g["station_key"] == station]
        m = (metrics_from_arrays(s["pred"].to_numpy(np.float64),
                                 s["obs"].to_numpy(np.float64))
             if not s.empty else {"n": 0, rank_metric: float("nan")})
        if m["n"] < min_n:
            print(f"    n={m['n']} < {min_n} at {rank_depth} (plotting anyway): "
                  f"{station}")
        out.append({"station_key": station, "rank": "NAMED", "rank_idx": i,
                    rank_metric: float(m[rank_metric]), "n": int(m["n"])})
    return out


def _clean_station_name(station: str) -> str:
    """'ISMN_TxSON_CR200-18' -> 'TxSON CR200-18' (source prefix dropped, network kept)."""
    for prefix in ("ISMN_", "ICOS_", "AmeriFlux_", "FLUXNET_"):
        if station.startswith(prefix):
            station = station[len(prefix):]
            break
    return station.replace("_", " ")


def _paper_ylim(vals: np.ndarray) -> tuple[float, float]:
    """Data-driven y range: floor at 0 (or just below the minimum), 12 % headroom."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.5
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))   # one bad reading cannot stretch the axis
    pad = 0.05 * max(hi - lo, 0.02)
    ylo = min(0.0, lo) - pad * 0.6            # always just below 0: zeros stay visible, floors consistent
    return ylo, hi * 1.12 + 0.005


def plot_station(df: pd.DataFrame, info: dict, meta: dict, split: str,
                 n_total: int):
    """Stacked depth panels for one station; None if the station has no data.

    Colour style: always 3 panels (empty ones say so).  PAPER: only the depths
    with data, legend + metrics above the axes, data-driven y limits.
    """
    station = info["station_key"]
    st = df[df["station_key"] == station]
    present = [d for d in SM_DEPTHS if (st["depth"] == d).any()]
    if not present:
        print(f"    no data at any depth -- skipping {station}")
        return None
    depths = present if PAPER else list(SM_DEPTHS)
    nrow = len(depths)
    # paper: ~1.9 in per panel + room for the legend row and title
    figsize = (7.4, 1.9 * nrow + 1.0) if PAPER else (7.4, 5.6)
    fig, axes = plt.subplots(nrow, 1, figsize=figsize, sharex=True,
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    tick_fs = plt.rcParams["xtick.labelsize"]
    if isinstance(tick_fs, str):
        tick_fs = plt.rcParams["font.size"]
    ann_fs = max(6 * FS, float(tick_fs)) if PAPER else 6 * FS
    text_grey = "black" if PAPER else "grey"
    shaded = False

    for ax, depth in zip(axes, depths):
        g = st[st["depth"] == depth].sort_values("date")
        if g.empty:          # colour style only -- PAPER never draws empty panels
            ax.text(0.5, 0.5, f"no data at {DEPTH_LABELS[depth]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8 * FS, color=text_grey)
            ax.set_ylabel(DEPTH_LABELS[depth], color=DEPTH_COLORS[depth])
            ax.set_yticks([])
            continue

        # Prediction line: from the every-day predictions (--line-dir, eval_predict.py
        # --keep-unobserved) when available, so the model output continues through gaps in
        # the observed record (user 2026-10-05). Dots + metrics stay on observed days only.
        gl = g
        if LINE_DF is not None:
            ll = LINE_DF[(LINE_DF["station_key"] == station) & (LINE_DF["depth"] == depth)]
            if len(ll):
                gl = ll.sort_values("date")
        g = g[g["obs"].notna()]

        # Break the line across data gaps -- otherwise matplotlib draws a
        # straight segment across months of missing record, which reads as a
        # confident flat prediction that was never made.
        dates = gl["date"].to_numpy()
        pred  = gl["pred"].to_numpy(np.float64).copy()
        gap   = np.diff(dates).astype("timedelta64[D]").astype(int) > GAP_DAYS
        pred[np.append(gap, False)] = np.nan

        # BW: observed grey dots UNDER a black prediction line, so the line stays readable
        # PAPER: solid, fully opaque navy line drawn ABOVE the observed dots
        line_on_top = BW or PAPER
        ax.plot(dates, pred, "-", lw=1.1 if PAPER else 0.9, color=PRED_COLOR or DEPTH_COLORS[depth],
                alpha=1.0, solid_capstyle="butt",
                label="predicted", zorder=3 if line_on_top else 2)
        ax.plot(g["date"], g["obs"], ".", ms=1.9, color=OBS_COLOR,
                label="observed", zorder=2 if line_on_top else 3)

        m = metrics_from_arrays(g["pred"].to_numpy(np.float64),
                                g["obs"].to_numpy(np.float64))
        sep = "  " if PAPER else "   "     # paper: larger font, keep it inside the width
        mtxt = sep.join([f"ubRMSE {m['ubRMSE']:.3f}", f"RMSE {m['RMSE']:.3f}",
                         f"$r^2$ {m['R2_pearson']:.2f}", f"NSE {m['NSE']:+.2f}",
                         f"bias {m['bias']:+.3f}", f"n {m['n']}"])
        if PAPER:
            # above the panel, clear of the data (wet stations reach the top)
            ax.text(1.0, 1.01, mtxt, transform=ax.transAxes, va="bottom",
                    ha="right", fontsize=ann_fs, color="black")
        else:
            ax.text(0.005, 0.96, mtxt, transform=ax.transAxes, va="top",
                    ha="left", fontsize=6 * FS,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))

        # OOS stations continue into 2023 as OOST -- mark the boundary
        if split == "oos" and g["date"].max() >= pd.Timestamp("2023-01-01"):
            ax.axvspan(pd.Timestamp("2023-01-01"), g["date"].max(),
                       color=OOT_SHADE, alpha=0.12 if PAPER else 0.08,
                       lw=0, zorder=0)
            ax.axvline(pd.Timestamp("2023-01-01"), color=OOT_SHADE,
                       lw=0.8, ls="--", zorder=1)
            shaded = True

        if PAPER:
            ax.set_ylabel(DEPTH_LABELS[depth], color="black")
            ax.set_ylim(*_paper_ylim(np.concatenate([gl["pred"].to_numpy(np.float64),
                                                     g["obs"].to_numpy(np.float64)])))
        else:
            ax.set_ylabel(f"{DEPTH_LABELS[depth]}\nSM (m$^3$/m$^3$)",
                          color=DEPTH_COLORS[depth])
            ax.set_ylim(0, max(0.55, float(g[["pred", "obs"]].to_numpy().max()) * 1.1))
        ax.margins(x=0.01)

    if PAPER:
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Patch
        from matplotlib.transforms import offset_copy
        handles, labels = axes[0].get_legend_handles_labels()
        if shaded:
            handles.append(Patch(facecolor=to_rgba(OOT_SHADE, 0.12),
                                 edgecolor=OOT_SHADE, ls="--", lw=0.8))
            labels.append("2023–2025 (OOST)")
        # in the figure margin above the top panel, never over the data:
        # anchored at the top-left of the axes, lifted (in points) past the
        # metrics line that sits just above the axes
        lift = offset_copy(axes[0].transAxes, fig=fig, y=1.8 * ann_fs, units="points")
        axes[0].legend(handles, labels, fontsize=ann_fs, frameon=False,
                       loc="lower left", bbox_to_anchor=(0.0, 1.0),
                       bbox_transform=lift,
                       ncol=3, markerscale=4, handlelength=1.6,
                       borderaxespad=0.0, columnspacing=1.2)
        fig.supylabel("Soil moisture (m$^3$/m$^3$)", fontsize=plt.rcParams["axes.labelsize"],
                      fontweight="bold")
    else:
        axes[0].legend(fontsize=6 * FS, frameon=False, loc="upper right", ncol=2)
        axes[-1].set_xlabel("date")
    loc = mdates.AutoDateLocator()       # ONE instance for locator and formatter
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    if PAPER:
        title = _clean_station_name(station)
        if meta:
            lat = meta.get("latitude", float("nan"))
            lon = meta.get("longitude", float("nan"))
            title += (f" — {meta.get('IGBP', '?')} | {meta.get('koppen_geiger', '?')}"
                      f" | {lat:.2f}°, {lon:.2f}°")
        fig.suptitle(title, fontsize=plt.rcParams["figure.titlesize"],
                     fontweight="bold")
    else:
        bits = [f"{station}", f"[{split.upper()} {info['rank']} "
                              f"{info['rank_idx']}/{n_total}]"]
        if meta:
            loc_s = f"{meta.get('latitude', float('nan')):.2f}, " \
                    f"{meta.get('longitude', float('nan')):.2f}"
            bits.append(f"{meta.get('IGBP', '?')} | {meta.get('koppen_geiger', '?')} "
                        f"| {loc_s}")
        fig.suptitle("   ".join(bits), fontsize=8 * FS)
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
    p.add_argument("--select",      default="extremes",
                   choices=["extremes", "median", "named"],
                   help="extremes: best-n and worst-n; "
                        "median: n random stations either side of the median; "
                        "named: exactly the --stations given, in that order")
    p.add_argument("--stations",    nargs="+", default=None,
                   help="--select named: station_key values to plot, e.g. "
                        "ISMN_TxSON_CR200-18 ISMN_TxSON_CR200-25")
    p.add_argument("--n-each",      type=int, default=2,
                   help="--select median: stations per side per split")
    p.add_argument("--rank-depth",  default=RANK_DEPTH, choices=SM_DEPTHS,
                   help="depth whose metric decides best/worst or below/above")
    p.add_argument("--seed",        type=int, default=0,
                   help="--select median: seed for the random draw")
    p.add_argument("--style", choices=["color", "bw", "paper"], default="color",
                   help="bw = black-and-white; paper = blue/green/red, Times (plot_style_bw.py)")
    p.add_argument("--line-dir", default=None,
                   help="dir with every-day predictions (eval_predict.py --keep-unobserved); the "
                        "prediction LINE is drawn from it so it continues through gaps in the "
                        "observed record. Dots, metrics and ranking still use --in-dir.")
    args = p.parse_args()
    if args.style != "color":
        import plot_style_bw
        plot_style_bw.apply(globals(), args.style)
    if args.select == "named" and not args.stations:
        p.error("--select named requires --stations")

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    meta_df = pd.read_csv(SPLITS_CSV)
    meta_df["station_key"] = meta_df.apply(_make_key, axis=1)
    meta_map = meta_df.set_index("station_key").to_dict("index")
    picked = []

    for split in args.splits:
        path = in_dir / f"predictions_{split}.parquet"
        if not path.exists():
            print(f"[{split}] no parquet -- skipping")
            continue

        df = clip_pred(pd.read_parquet(path))
        df_rank = df        # rank on this split alone, before OOST is merged in
        global LINE_DF
        LINE_DF = None
        if args.line_dir:
            lines = [Path(args.line_dir) / f"predictions_{s}.parquet"
                     for s in ([split, "oost"] if split == "oos" else [split])]
            lines = [p_ for p_ in lines if p_.exists()]
            if lines:
                LINE_DF = clip_pred(pd.concat([pd.read_parquet(p_) for p_ in lines],
                                              ignore_index=True))
                print(f"    line from every-day predictions: {', '.join(p_.name for p_ in lines)} "
                      f"({len(LINE_DF):,} rows)")

        # An OOS station continues into 2023 as OOST; show both in one figure
        # so the temporal extrapolation is visible on the same axes.
        if split == "oos":
            oost_path = in_dir / "predictions_oost.parquet"
            if oost_path.exists():
                df = pd.concat([df, clip_pred(pd.read_parquet(oost_path))], ignore_index=True)

        print(f"\n[{split}] {len(df):,} rows | "
              f"{df['station_key'].nunique()} stations")
        if args.select == "median":
            selected = select_around_median(df_rank, args.n_each, args.min_n,
                                            args.rank_metric, args.rank_depth,
                                            args.seed)
            n_total, split_dir = args.n_each, out_dir / "median_sample" / split
            pdf_path = out_dir / "median_sample" / f"contact_median_{split}.pdf"
        elif args.select == "named":
            selected = select_named(df_rank, args.stations, args.min_n,
                                    args.rank_metric, args.rank_depth)
            n_total, split_dir = len(args.stations), out_dir / "named" / split
            pdf_path = out_dir / "named" / f"contact_named_{split}.pdf"
        else:
            selected = select_stations(df, args.n, args.min_n, args.rank_metric,
                                       args.rank_depth)
            n_total, split_dir = args.n, out_dir / split
            pdf_path = out_dir / f"contact_{split}.pdf"
        if not selected:
            print("    no station qualified -- skipping")
            continue

        split_dir.mkdir(parents=True, exist_ok=True)
        figs = []

        for info in selected:
            fig = plot_station(df, info, meta_map.get(info["station_key"], {}),
                               split, n_total)
            if fig is None:          # no depth has data
                continue
            name =(f"{info['rank']}_{info['rank_idx']:02d}_"
                    f"{info['station_key']}.png")
            fig.savefig(split_dir / name, dpi=DPI, bbox_inches="tight")
            figs.append(fig)
            print(f"    {info['rank']:>5s} {info['rank_idx']:>2d}  "
                  f"{info['station_key']:<45s} "
                  f"{args.rank_metric}={info[args.rank_metric]:.4f}  "
                  f"n={info['n']}")
        picked.extend({"split": split, **info} for info in selected)

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        for fig in figs:
            plt.close(fig)
        print(f"    → {split_dir}/  ({len(figs)} figures)")
        print(f"    → {pdf_path}")

    if picked and args.select == "median":
        csv_path = out_dir / "median_sample" / "selected_median_sample.csv"
        pd.DataFrame(picked).to_csv(csv_path, index=False)
        print(f"\n→ {csv_path}  ({len(picked)} stations)")
    if picked and args.select == "named":
        csv_path = out_dir / "named" / "selected_named.csv"
        pd.DataFrame(picked).to_csv(csv_path, index=False)
        print(f"\n→ {csv_path}  ({len(picked)} of "
              f"{len(args.stations) * len(args.splits)} requested station-splits)")


if __name__ == "__main__":
    main()
