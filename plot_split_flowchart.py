#!/usr/bin/env python
"""Paper figure: how stations are assigned to Train / Validation / OOS, and OOT / OOST in time.

Steps follow create_evaluation_splits.py + runbook §47 (constants from splits_config.py):
  inventory -> soil-moisture stations -> location groups (< COLOC_THRESHOLD_KM) ->
  stratified draw (climate x land cover x elevation band): OOS_FRACTION -> OOS, VAL_FRACTION -> Val ->
  three demotion rules (tile < 1120 m, Netherlands, < MIN_PRE_CUT_DAYS before the cut) -> OOS ->
  validation top-up drawn from OOS -> final Train / Validation / OOS ->
  time cut OOT_CUT_DATE: 2016-2022 vs 2023-2025 (>= MIN_POST_CUT_DAYS after the cut for OOT / OOST).

Counts are read from csvs/station_splits.csv and the eval per_station files, not typed in.
Paper style; CPU, seconds.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plot_style_bw
from splits_config import (COLOC_THRESHOLD_KM, MIN_POST_CUT_DAYS, MIN_PRE_CUT_DAYS, OOS_FRACTION,
                           OOT_CUT_DATE, VAL_FRACTION)

RUN = "baseline_selected_20261005"
BLUE, LBLUE, ORANGE, GREY = plot_style_bw.BLUE, plot_style_bw.GREEN, plot_style_bw.RED, "0.45"


def box(ax, x, y, w, h, text, fc="white", tc="black", lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec="black", lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", color=tc, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.3, color="black", zorder=1, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"eval_output/{RUN}")
    ap.add_argument("--out-dir", default=f"figures/eval/{RUN}_paper")
    a = ap.parse_args()
    plt.rcParams.update(plot_style_bw.RC_PAPER)

    sp = pd.read_csv("csvs/station_splits.csv")          # pandas: quoted commas in AmeriFlux names
    n_inv = len(sp)
    sm = sp[sp.has_soil_moisture.astype(bool)]
    n_sm, n_dup = len(sm), int((sm.split == "duplicate").sum())
    n = sm.split.value_counts()
    ev = {s: pd.read_csv(Path(a.eval_dir) / f"per_station_{s}.csv") for s in ("oos", "oot", "oost")}
    oot_tr, oot_va = int((ev["oot"].split == "train").sum()), int((ev["oot"].split == "val").sum())
    cut = f"{str(OOT_CUT_DATE)[:4]}-{str(OOT_CUT_DATE)[4:6]}-{str(OOT_CUT_DATE)[6:]}"
    print(f"inventory {n_inv}, SM {n_sm}, duplicate {n_dup}, splits {n.to_dict()}, "
          f"OOT {oot_tr}+{oot_va}, OOST {len(ev['oost'])}, OOS eval {len(ev['oos'])}")

    fig, ax = plt.subplots(figsize=(13.0, 14.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, H = 0.62, 0.062
    steps = [
        f"Station inventory: {n_inv} stations\n(ISMN, ICOS, AmeriFlux)",
        f"Keep stations with soil-moisture observations: {n_sm - n_dup}\n"
        f"(drop {n_inv - n_sm} flux-only sites and {n_dup} duplicate record)",
        f"Group stations closer than {COLOC_THRESHOLD_KM:g} km into one location group\n"
        "(nearby sensors always end up in the same split)",
        "Stratify groups by climate × land cover × elevation band;\n"
        f"draw {OOS_FRACTION:.0%} of groups → OOS, {VAL_FRACTION:.0%} → Validation, rest → Train",
        "Move Train / Validation stations to OOS if they\n"
        "(1) share a 2.24 km image tile with another station (< 1120 m)\n"
        "(2) are in the Netherlands (held out for the national map)\n"
        f"(3) have < {MIN_PRE_CUT_DAYS} observed days before {cut}",
        "Top up Validation to ~10 % with stratified groups drawn from OOS\n"
        "(never from Train, so training data is unchanged)",
    ]
    ys = [0.955, 0.86, 0.765, 0.67, 0.555, 0.44]
    hs = [H, H, H, H, 0.088, H]
    for i, (t, y, h) in enumerate(zip(steps, ys, hs)):
        box(ax, 0.5, y, W, h, t)
        if i:
            arrow(ax, 0.5, ys[i - 1] - hs[i - 1] / 2, 0.5, y + h / 2)

    # final station split
    yS, wS, hS = 0.315, 0.27, 0.07
    xs = [0.17, 0.5, 0.83]
    final = [("Train", n.get("train", 0), GREY, "white"), ("Validation", n.get("val", 0), LBLUE, "black"),
             ("OOS", n.get("oos", 0), BLUE, "white")]
    for x, (lab, k, fc, tc) in zip(xs, final):
        arrow(ax, 0.5, ys[-1] - hs[-1] / 2, x, yS + hS / 2)
        box(ax, x, yS, wS, hS, f"{lab} stations\n{k}", fc=fc, tc=tc, lw=1.5)

    # time split
    yT, hT, wT = 0.13, 0.115, 0.125
    pre = ["Training\n2016–2022", "Model selection\n(best epoch)\n2016–2022",
           f"OOS\nnew stations\n2016–2022\n{len(ev['oos'])} evaluated"]
    post = [f"OOT\nseen stations,\nnew years\n2023–2025\n{oot_tr} stations",
            f"OOT\nseen stations,\nnew years\n2023–2025\n{oot_va} stations",
            f"OOST\nnew stations,\nnew years\n2023–2025\n{len(ev['oost'])} stations"]
    cpre = [(GREY, "white"), (LBLUE, "black"), (BLUE, "white")]
    cpost = [(LBLUE, "black"), (LBLUE, "black"), (ORANGE, "white")]
    for x, p0, p1, c0, c1 in zip(xs, pre, post, cpre, cpost):
        for dx, t, (fc, tc) in [(-0.072, p0, c0), (0.072, p1, c1)]:
            arrow(ax, x, yS - hS / 2, x + dx, yT + hT / 2)
            box(ax, x + dx, yT, wT, hT, t, fc=fc, tc=tc)
    ax.text(0.5, 0.035, f"Bottom row: each set split in time at {cut}. OOT and OOST only use stations with at least "
            f"{MIN_POST_CUT_DAYS} observed days after the cut.", ha="center", va="center")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"split_flowchart.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {out}/split_flowchart.{{png,pdf}}")


if __name__ == "__main__":
    main()
