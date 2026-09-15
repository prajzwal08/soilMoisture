"""Publication-quality train/val loss curves from a training log.

Usage:
    python plot_loss_curves.py --log logs/train_26083217.out --run pw_stage2a_L3
    → figures/loss_curves_{run}.{png,pdf}

Panel (a) plots the *per-batch* train loss (rank-0, logged every `log_every`
batches by train_epoch()) against the per-epoch validation loss.  The epoch-only
view compressed the interesting part — where train detaches from val, which on
pw_stage2a_L3 happens inside the first two epochs — into two markers.

Single y-axis (log) rather than a dual axis: train and val differ by up to ~40x
and a second scale would fabricate crossings that are not in the data.
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
# Accepts both the old `.4f` and the current `.3e` batch-loss format.
BATCH_RE = re.compile(r"^\s+batch (\d+)\s+loss=([\d.eE+-]+)")
# The metric that actually drives best.pt, early stopping and ReduceLROnPlateau.
# val_loss is a mean-of-batch-means and its minimum is a DIFFERENT epoch -- ringing
# that one would label the wrong checkpoint as "best".
SELECT_RE = re.compile(r"^\s+SELECT\s+val_ubrmse_depth_mean=([\d.]+)")

MEDIAN_WINDOW = 9   # logged points, i.e. ~9 * log_every real batches


def rolling_median(y, w):
    """Centred rolling median, shrinking window at the edges. Plain-python so the
    script keeps its numpy-only-via-matplotlib dependency footprint."""
    half, n = w // 2, len(y)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = sorted(y[lo:hi])
        m = len(win)
        out.append(win[m // 2] if m % 2 else 0.5 * (win[m // 2 - 1] + win[m // 2]))
    return out


def parse(log_path):
    """Single pass. Batch lines are emitted *before* the `Epoch NNN |` summary of the
    epoch they belong to, so buffer them and flush on each epoch line. Segmenting by a
    reset of the batch counter would be wrong: `skip_batches` (mid-epoch resume) makes
    that counter non-monotonic within an epoch on a resumed run."""
    epochs, train, val, select = [], [], [], []
    batch_x, batch_loss = [], []
    pending = []          # (batch_number, loss) awaiting their epoch line
    fixed_point = True    # every batch loss printed in `.4f` -> quantised to 1e-4

    for line in Path(log_path).read_text().splitlines():
        mb = BATCH_RE.match(line)
        if mb:
            raw = mb.group(2)
            if "e" in raw or "E" in raw:
                fixed_point = False
            pending.append((int(mb.group(1)), float(raw)))
            continue
        me = EPOCH_RE.match(line)
        if me:
            e = int(me.group(1))
            epochs.append(e)
            train.append(float(me.group(2)))
            val.append(float(me.group(3)))
            if pending:
                bmax = max(b for b, _ in pending)
                for b, l in pending:
                    batch_x.append((e - 1) + b / bmax)
                    batch_loss.append(l)
                pending = []
            continue
        ms = SELECT_RE.match(line)
        if ms and len(select) < len(epochs):
            # SELECT follows its own `Epoch NNN |` line, so it belongs to epochs[-1].
            select.append(float(ms.group(1)))

    orphans = len(pending)   # batch lines after the last completed epoch
    if not epochs:
        raise SystemExit(f"No epoch lines found in {log_path}")
    if len(select) != len(epochs):
        select = []          # older logs predate the SELECT line -- fall back
    # `.4f` cannot resolve below 5e-5 and prints 1e-4 as the smallest nonzero value.
    floor = 1e-4 if (fixed_point and batch_loss) else None
    return epochs, train, val, select, batch_x, batch_loss, orphans, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train_26083217.out")
    ap.add_argument("--run", default="pw_stage2a_L3")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--clip-warmup", type=int, default=1,
                    help="drop the first N logged batch points (batch 0001 is logged "
                         "at wu=0.00, an order of magnitude above the rest, and "
                         "stretches the log axis)")
    args = ap.parse_args()

    ep, tr, va, sel, bx, bl, orphans, floor = parse(args.log)
    n_batch_raw = len(bx)
    if args.clip_warmup > 0:
        bx, bl = bx[args.clip_warmup:], bl[args.clip_warmup:]
    ratio = [v / t for v, t in zip(va, tr)]
    # Ring the checkpoint the RUN actually kept.  On pw_stage2a_L3 the val_loss minimum
    # is epoch 14 while best.pt is epoch 2 -- ranking on the wrong one mislabels the plot.
    if sel:
        best_i   = min(range(len(sel)), key=lambda i: sel[i])
        best_txt = f"best.pt  ep{ep[best_i]}  ubRMSE {sel[best_i]:.6f}"
    else:
        best_i   = min(range(len(va)), key=lambda i: va[i])
        best_txt = f"min val_loss {va[best_i]:.6f}"
        print("NOTE: no `SELECT val_ubrmse_depth_mean=` lines in this log -- ringing the "
              "val_loss minimum instead, which is NOT necessarily the saved checkpoint.")

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9))

    # ── (a) per-batch train vs per-epoch val, single log axis ─────────
    # Epoch separators first, so they sit under everything.
    for e in ep[:-1]:
        ax1.axvline(e, color=GRID, lw=0.4, zorder=0)

    # Shade the region the LOG cannot resolve.  With the old `loss={_l:.4f}` format the
    # smallest printable nonzero value is 1e-4, so once the run converges past that the
    # per-batch trace is a print artifact -- it flattens at the floor while the epoch
    # means keep falling below it.  Say so on the figure instead of letting it read as
    # a real plateau.  Runs logged with `.3e` have no floor and get no band.
    if floor is not None and bx:
        ax1.axhspan(1e-9, floor, color=MUTED, alpha=0.09, lw=0, zorder=1)
        ax1.axhline(floor, color=MUTED, lw=0.6, ls="--", zorder=1)
        ax1.annotate("below log print precision", (ep[-1], floor),
                     textcoords="offset points", xytext=(-3, 3), ha="right",
                     fontsize=6, color=MUTED, zorder=6)

    if bx:
        ax1.plot(bx, bl, "-", color=C_TRAIN, lw=0.7, alpha=0.35, zorder=2,
                 label="Train (per batch)")
        ax1.plot(bx, rolling_median(bl, MEDIAN_WINDOW), "-", color=C_TRAIN,
                 lw=1.6, zorder=3, label=f"Train (median, {MEDIAN_WINDOW} pts)")
    # Epoch-mean train loss: the number the run itself reported, as a check that the
    # batch cloud lands where it should.
    ax1.plot(ep, tr, "o", color=C_TRAIN, ms=3.2, mfc="white", mew=0.9, zorder=4,
             label="Train (epoch mean)")
    ax1.plot(ep, va, "-s", color=C_VAL, lw=1.8, ms=4.5, mec="white", mew=0.8,
             label="Validation (per epoch)", zorder=5)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber loss (per-depth mean)")
    ax1.set_title("(a) Per-batch train vs. validation loss", loc="left", pad=8)

    ax1.plot(ep[best_i], va[best_i], "o", ms=10, mfc="none", mec=C_VAL, mew=1.4, zorder=6)
    ax1.annotate(best_txt, (ep[best_i], va[best_i]),
                 textcoords="offset points", xytext=(8, 10), ha="left",
                 fontsize=7, color=MUTED)

    ax1.set_xlim(0, ep[-1] + 0.35)
    ax1.set_xticks(ep)
    # 22 labels collide at this width — label every other tick.
    ax1.set_xticklabels([str(e) if i % 2 == 0 else "" for i, e in enumerate(ep)])
    ax1.legend(frameon=False, loc="lower left", handlelength=1.6, labelspacing=0.3)

    # ── (b) generalisation gap ────────────────────────────────────────
    ax2.plot(ep, ratio, "-o", color=INK, lw=1.8, ms=4.5, mec="white", mew=0.8, zorder=3)
    ax2.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Validation / train loss")
    ax2.set_title("(b) Generalisation gap", loc="left", pad=8)
    ax2.set_ylim(0, max(ratio) * 1.28)
    ax2.set_xticks(ep)
    ax2.set_xticklabels([str(e) if i % 2 == 0 else "" for i, e in enumerate(ep)])
    ax2.set_xlim(ep[0] - 0.35, ep[-1] + 0.35)
    # Annotating all 22 would be unreadable; label the ends and the best epoch.
    for i in {0, best_i, len(ep) - 1}:
        ax2.annotate(f"{ratio[i]:.0f}x", (ep[i], ratio[i]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color=MUTED)

    for ax in (ax1, ax2):
        ax.grid(True, which="major", axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.subplots_adjust(wspace=0.32)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        p = outdir / f"loss_curves_{args.run}.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")

    print(f"\nbatch points: {n_batch_raw} parsed, {args.clip_warmup} clipped as warmup, "
          f"{len(bx)} plotted across {len(ep)} epochs "
          f"(~{len(bx) / max(len(ep), 1):.0f} per epoch)")
    if orphans:
        print(f"WARNING: {orphans} batch lines followed the last completed epoch "
              f"(interrupted epoch) and were dropped")
    if floor is not None:
        n_at_floor = sum(1 for l in bl if l <= floor)
        print(f"WARNING: this log used the `.4f` batch-loss format -- {n_at_floor} of "
              f"{len(bl)} plotted points sit at or below {floor:g}, the smallest value "
              f"it can print. Below the dashed line panel (a) shows print precision, "
              f"not the loss. (train.py now logs `.3e`.)")

    print(f"\nepoch  train      val        ratio   "
          f"{'ubRMSE(select)' if sel else ''}")
    for i, (e, t, v, r) in enumerate(zip(ep, tr, va, ratio)):
        s = f"  {sel[i]:.6f}" + ("  <-- best.pt" if i == best_i else "") if sel else ""
        print(f"{e:5d}  {t:.6f}  {v:.6f}  {r:.1f}x{s}")


if __name__ == "__main__":
    main()
