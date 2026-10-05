#!/usr/bin/env python
"""Depth sweep (§59/§62/§63 + the 6-layer no-LST run): figure + table from the training logs.

Parses each run's log for the per-epoch train/val loss, SELECT (val ubRMSE depth mean) and the
per-depth ubRMSE / r, keeps epochs 1-10, and writes
  eval_output/depth_sweep/depth_sweep.{pdf,png}   (a) SELECT vs epoch  (b) val/train gap vs epoch
                                                   (c) ep10 SELECT vs number of layers
  eval_output/depth_sweep/depth_sweep.{csv,md}    ep10 row per depth
A run whose log is missing or has fewer than 10 epochs is reported and plotted with what it has.

Gap = val_loss / train_loss from the "Epoch NNN" line (the same ratio train.py prints as "gap").
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = Path("logs/lst_tmean_diff")
OUT_DIR = Path("eval_output/depth_sweep")
MAX_EP = 10
DEPTHS = ["0-10", "10-30", "30-100"]

# (layers, log glob) — 6L = 27417115 (warmup 1000, older code; counted with the rest by the user's call)
RUNS = [
    (1, "train_nolst_L1_wu200_*.out"),
    (2, "train_nolst_L2_wu200_27611631.out"),
    (3, "train_nolst_L3_wu200_27575211.out"),
    (6, "train_nolst_27417115.out"),
]
# Ordinal one-hue blue ramp (validated: dataviz validate_palette.js --ordinal, light): more layers = darker
COLORS = {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 6: "#0d366b"}
MARKERS = {1: "o", 2: "s", 3: "^", 6: "D"}

RE_EPOCH = re.compile(r"^Epoch (\d+)\s+\|\s+train_loss=([\d.eE+-]+)\s+val_loss=([\d.eE+-]+)")
RE_DEPTH = re.compile(r"^\s+(0-10|10-30|30-100)\s+train_loss=.*ubRMSE=([\d.]+).*\br=([-\d.]+)")
RE_SELECT = re.compile(r"SELECT\s+val_ubrmse_depth_mean=([\d.]+)")
RE_PARAMS = re.compile(r"Trainable parameters: ([\d,]+)")


def parse(path):
    rows, cur, params = [], None, None
    for line in path.read_text(errors="replace").splitlines():
        if params is None and (m := RE_PARAMS.search(line)):
            params = int(m.group(1).replace(",", ""))
        if m := RE_EPOCH.match(line):
            cur = {"epoch": int(m.group(1)), "train_loss": float(m.group(2)), "val_loss": float(m.group(3))}
        elif cur is not None and (m := RE_DEPTH.match(line)):
            cur[f"ub_{m.group(1)}"] = float(m.group(2))
            cur[f"r_{m.group(1)}"] = float(m.group(3))
        elif cur is not None and (m := RE_SELECT.search(line)):
            cur["select"] = float(m.group(1))
            rows.append(cur)
            cur = None
    df = pd.DataFrame(rows).drop_duplicates("epoch", keep="last").sort_values("epoch")
    df = df[df.epoch <= MAX_EP].copy()
    df["gap"] = df.val_loss / df.train_loss
    return df, params


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves, table = {}, []
    for n, pattern in RUNS:
        hits = sorted(LOG_DIR.glob(pattern))
        if not hits:
            print(f"[skip] {n}L: no log matching {pattern}")
            continue
        df, params = parse(hits[-1])
        if df.empty:
            print(f"[skip] {n}L: no completed epochs in {hits[-1].name}")
            continue
        if df.epoch.max() < MAX_EP:
            print(f"[warn] {n}L: only {df.epoch.max()} epochs in {hits[-1].name}")
        curves[n] = df
        last, best = df.iloc[-1], df.loc[df.select.idxmin()]
        row = {"layers": n, "params_M": round(params / 1e6, 1) if params else None, "log": hits[-1].name,
               "epoch": int(last.epoch), "SELECT": round(last.select, 4),
               "best_le10": round(best.select, 4), "best_ep": int(best.epoch)}
        row.update({f"ubRMSE {d}": round(last[f"ub_{d}"], 4) for d in DEPTHS})
        row.update({f"r {d}": round(last[f"r_{d}"], 3) for d in DEPTHS})
        row["gap"] = round(last.gap, 2)
        table.append(row)

    tab = pd.DataFrame(table)
    tab.to_csv(OUT_DIR / "depth_sweep.csv", index=False)
    try:
        md = tab.drop(columns="log").to_markdown(index=False)
    except ImportError:  # tabulate missing
        md = tab.drop(columns="log").to_string(index=False)
    (OUT_DIR / "depth_sweep.md").write_text(md + "\n")
    print(tab.drop(columns="log").to_string(index=False))

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for n, df in curves.items():
        kw = dict(color=COLORS[n], marker=MARKERS[n], ms=4, lw=1.5, label=f"{n} layer" + ("s" if n > 1 else ""))
        axes[0].plot(df.epoch, df.select, **kw)
        axes[1].plot(df.epoch, df.gap, **kw)
    axes[0].set(xlabel="Epoch", ylabel="Val ubRMSE, depth mean (m³/m³)", title="(a) Validation skill")
    axes[1].set(xlabel="Epoch", ylabel="Val loss / train loss", title="(b) Generalisation gap")
    axes[1].axhline(1, color="0.6", lw=0.8, ls=":")
    for ax in axes[:2]:
        ax.set_xticks(range(1, MAX_EP + 1))
        ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    xs = list(range(len(tab)))
    for i, r in tab.iterrows():
        ax.plot(xs[i], r.SELECT, marker=MARKERS[r.layers], color=COLORS[r.layers], ms=7, ls="none")
        ax.annotate(f"{r.SELECT:.4f}\n{r.params_M} M", (xs[i], r.SELECT), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="0.25")
    ax.plot(xs, tab.SELECT, color="0.7", lw=0.8, zorder=0)
    ax.set_xticks(xs, [str(n) for n in tab.layers])
    ax.set_xlim(-0.5, len(xs) - 0.5)
    lo, hi = tab.SELECT.min(), tab.SELECT.max()
    pad = max(hi - lo, 1e-3) * 0.6
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.set(xlabel="Transformer layers", ylabel="Val ubRMSE at epoch 10 (m³/m³)", title="(c) Epoch-10 skill vs depth")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"depth_sweep.{ext}", dpi=300)
    print(f"wrote {OUT_DIR}/depth_sweep.{{pdf,png,csv,md}}")


if __name__ == "__main__":
    main()
