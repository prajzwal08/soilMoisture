"""§22.13 -- SoMo.ml Fig. 5 style distributional comparison (figures/sample_fig).

Three rows (our depths, their Layer1/2/3) x two columns:

    a) between stations   station-mean predicted vs observed, one dot per station,
                          n + Pearson r, 1:1 dotted, red OLS line.  Their panel
                          averages over pixels; a station is our pixel.
    b) all time series    density histograms of every daily value, observed grey
                          vs predicted blue, shared bins.

The published figure has a third panel splitting the distribution by continent.
It is dropped here: the station pool is overwhelmingly North American, so a
by-region panel would carry no information for us.

Note the comparison is not like-for-like with the published figure: SoMo.ml Fig. 5
is drawn on TRAINING grid cells, so it reports r = 0.92-0.98.  These are held-out
splits, which is the point -- panel (a) is the station-level transfer failure of
§22.10 drawn in their layout.

CPU only.  Needs pyarrow -> run under the terramind env.

    python plot_eval_somo_panel.py [--splits oos oot oost]
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
plt.rcParams["text.usetex"] = False     # sfmath.sty is absent on the login nodes

SM_DEPTHS    = ["0-10", "10-30", "30-100"]
DEPTH_LABELS = {"0-10": "0-10 cm", "10-30": "10-30 cm", "30-100": "30-100 cm"}
OBS_COLOR    = "#7f7f7f"        # grey  -- in situ, as in the original
PRED_COLOR   = "#7b7fd4"        # blue  -- model
FIT_COLOR    = "#d62728"
SM_LIM       = (0.0, 0.60)      # our SM rarely exceeds 0.55; 0-1 would waste the axes


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_dir/name}.png")


def figure_for_split(df: pd.DataFrame, split: str, out_dir: Path):
    fig, axes = plt.subplots(len(SM_DEPTHS), 2, figsize=(6.8, 8.6),
                             constrained_layout=True)
    bins = np.linspace(*SM_LIM, 61)
    rows = []

    for i, depth in enumerate(SM_DEPTHS):
        g = df[df["depth"] == depth]
        ax_a, ax_b = axes[i]

        # ── a) station means ────────────────────────────────────────────────
        st = g.groupby("station_key", observed=True)[["pred", "obs"]].mean()
        if len(st) > 2:
            ax_a.scatter(st["obs"], st["pred"], s=9, c="#2b3a8f", alpha=0.75,
                         edgecolors="none")
            r = float(np.corrcoef(st["obs"], st["pred"])[0, 1])
            slope, intercept = np.polyfit(st["obs"], st["pred"], 1)
            xs = np.array(SM_LIM)
            ax_a.plot(xs, slope * xs + intercept, color=FIT_COLOR, lw=1.4)
            ax_a.text(0.05, 0.95, f"n={len(st)}\nr={r:.2f}", transform=ax_a.transAxes,
                      va="top", ha="left", fontsize=8)
            rows.append(dict(split=split, depth=depth, n_stations=len(st), r=r,
                             slope=float(slope), intercept=float(intercept),
                             mean_obs=float(st["obs"].mean()),
                             mean_pred=float(st["pred"].mean()),
                             sd_obs_daily=float(g["obs"].std()),
                             sd_pred_daily=float(g["pred"].std())))
        ax_a.plot(SM_LIM, SM_LIM, "k:", lw=1.0)
        ax_a.set_xlim(*SM_LIM); ax_a.set_ylim(*SM_LIM); ax_a.set_aspect("equal")
        ax_a.set_ylabel("model [m$^3$/m$^3$]")
        ax_a.annotate(DEPTH_LABELS[depth], xy=(-0.30, 0.5),
                      xycoords="axes fraction", rotation=90, ha="center",
                      va="center", fontweight="bold", fontsize=10)

        # ── b) marginal distributions ───────────────────────────────────────
        for vals, color, label, alpha in ((g["obs"], OBS_COLOR, "in situ", 0.75),
                                          (g["pred"], PRED_COLOR, "model", 0.65)):
            h, _ = np.histogram(vals.to_numpy(np.float64), bins=bins, density=True)
            ax_b.bar(bins[:-1], h, width=np.diff(bins), align="edge",
                     color=color, alpha=alpha, label=label, lw=0)
        ax_b.set_xlim(*SM_LIM)
        ax_b.set_ylabel("Density")
        ax_b.legend(fontsize=7, frameon=True, framealpha=0.9)

        if i == 0:
            ax_a.set_title("a) between stations", fontsize=10)
            ax_b.set_title("b) all time series", fontsize=10)
        if i == len(SM_DEPTHS) - 1:
            ax_a.set_xlabel("in situ [m$^3$/m$^3$]")
            ax_b.set_xlabel("Soil Moist. [m$^3$/m$^3$]")

    fig.suptitle(f"{split.upper()} -- model vs in situ soil moisture "
                 f"(held-out stations)", y=1.02)
    save(fig, out_dir, f"somo_panel_{split}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="eval_output")
    ap.add_argument("--out-dir", default="figures/eval")
    ap.add_argument("--splits",  nargs="+", default=["oos", "oot", "oost"])
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    summary = []
    for split in args.splits:
        path = in_dir / f"predictions_{split}.parquet"
        if not path.exists():
            print(f"[{split}] no parquet -- skipping")
            continue
        df = pd.read_parquet(path)
        print(f"[{split}] {len(df):,} rows | {df['station_key'].nunique()} stations")
        summary += figure_for_split(df, split, out_dir)

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(out_dir / "somo_panel_summary.csv", index=False)
        print("\n" + s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n→ {out_dir}/somo_panel_summary.csv")


if __name__ == "__main__":
    main()
