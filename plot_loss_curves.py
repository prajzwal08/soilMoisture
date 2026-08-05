"""Publication-quality train/val loss curves from a training log.

Usage:
    python plot_loss_curves.py --log logs/train_25235976.out --run cls_depth_star_reg
    → figures/loss_curves_{run}.{png,pdf}

Single y-axis (log) rather than a dual axis: train and val differ by ~7x and a
second scale would fabricate crossings that are not in the data.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colourblind-safe pair, validated (CVD ΔE 26.2 protan / 29.3 tritan, normal 33.5).
C_TRAIN = "#0173B2"
C_VAL   = "#DE8F05"
INK     = "#1a1a1a"
MUTED   = "#6b6b6b"
GRID    = "#d9d9d9"

EPOCH_RE = re.compile(r"^Epoch (\d+)\s+\|\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)")


def parse(log_path):
    epochs, train, val = [], [], []
    for line in Path(log_path).read_text().splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    if not epochs:
        raise SystemExit(f"No epoch lines found in {log_path}")
    return epochs, train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train_25235976.out")
    ap.add_argument("--run", default="cls_depth_star_reg")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    ep, tr, va = parse(args.log)
    ratio = [v / t for v, t in zip(va, tr)]
    best_i = min(range(len(va)), key=lambda i: va[i])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ── (a) loss curves, single log axis ──────────────────────────────
    ax1.plot(ep, tr, "-o", color=C_TRAIN, lw=1.8, ms=5, mec="white", mew=0.8,
             label="Train", zorder=3)
    ax1.plot(ep, va, "-s", color=C_VAL, lw=1.8, ms=5, mec="white", mew=0.8,
             label="Validation", zorder=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber loss (per-depth mean)")
    ax1.set_title("(a) Train vs. validation loss", loc="left", pad=8)

    ax1.plot(ep[best_i], va[best_i], "o", ms=10, mfc="none", mec=C_VAL, mew=1.4, zorder=4)
    ax1.annotate(f"best {va[best_i]:.6f}", (ep[best_i], va[best_i]),
                 textcoords="offset points", xytext=(-8, -16), ha="right",
                 fontsize=7, color=MUTED)

    # ── (b) generalisation gap ────────────────────────────────────────
    ax2.plot(ep, ratio, "-o", color=INK, lw=1.8, ms=5, mec="white", mew=0.8, zorder=3)
    ax2.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Validation / train loss")
    ax2.set_title("(b) Generalisation gap", loc="left", pad=8)
    for x, y in zip(ep, ratio):
        ax2.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color=MUTED)
    ax2.set_ylim(0, max(ratio) * 1.28)

    for ax in (ax1, ax2):
        ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(ep)
        ax.set_xlim(ep[0] - 0.35, ep[-1] + 0.35)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax1.legend(frameon=False, loc="center left", handlelength=1.6)
    fig.subplots_adjust(wspace=0.32)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        p = outdir / f"loss_curves_{args.run}.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")

    print("\nepoch  train      val        ratio")
    for e, t, v, r in zip(ep, tr, va, ratio):
        print(f"{e:5d}  {t:.6f}  {v:.6f}  {r:.1f}x")


if __name__ == "__main__":
    main()
