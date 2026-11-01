#!/usr/bin/env python
"""Paper figure: how stations are assigned to Train / Validation / OOS, and OOT / OOST in time.

(a) daily soil-moisture QC per station (preprocessing_*.py), (b) split assignment.
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


def line(ax, xs, ys):
    ax.plot(xs, ys, color="black", lw=1.3, zorder=1, solid_capstyle="butt")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22,
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

    fig, ax = plt.subplots(figsize=(19.0, 14.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # (a) quality control, per station (preprocessing_ISMN_soilMoisture.py; ICOS / AmeriFlux use the
    #     same pipeline with their own QC flag, preprocessing_icos.py / preprocessing_ameriflux.py)
    XL, WL, HL = 0.195, 0.37, 0.048
    qc = [
        "Raw sub-daily soil moisture\n(ISMN hourly; ICOS and AmeriFlux half-hourly)",
        "Keep good readings only\n(ISMN flag G or D*; ICOS / AmeriFlux QC = 0, measured)",
        "Average sensors at the same depth; daily mean,\nkept only if at least 6 valid readings that day",
        "Average into layers 0–10, 10–30, 30–100 cm\n(sensors deeper than 100 cm dropped)",
        "Keep the longest 0–10 cm record (gaps of up to 7 days bridged);\n"
        "deeper layers kept only if at least 95 % complete in that window",
        "Keep stations with at least 3 years (1095 days) of valid daily data",
        "Fill remaining gaps with the same calendar-day mean, flagged;\n"
        "the model is trained and evaluated on observed days only",
        f"{n_sm - n_dup} stations with daily soil moisture",
    ]
    yl = [0.958 - i * 0.073 for i in range(len(qc))]
    for i, (t, y) in enumerate(zip(qc, yl)):
        box(ax, XL, y, WL, HL, t, lw=1.5 if i == len(qc) - 1 else 1.2)
        if i:
            arrow(ax, XL, yl[i - 1] - HL / 2, XL, y + HL / 2)
    ax.text(XL - WL / 2, 0.993, "(a)   Daily soil moisture, each station", ha="left", va="bottom")

    # (b) assigning stations to sets (create_evaluation_splits.py + runbook §47)
    XR, WR = 0.71, 0.52
    steps = [
        f"Group stations closer than {COLOC_THRESHOLD_KM:g} km into one location group\n"
        "(nearby sensors always end up in the same set)",
        "In each climate × land cover cell, draw at random\n"
        f"{OOS_FRACTION:.0%} of the groups → OOS, then {VAL_FRACTION:.0%} of the rest → Validation;\n"
        "the remaining groups → Train (cells with < 3 groups → Train)",
        "Move Train / Validation stations to OOS if they\n"
        "(1) share a 2.24 km image tile with another station (< 1120 m)\n"
        "(2) are in the Netherlands (held out for the national map)\n"
        f"(3) have < {MIN_PRE_CUT_DAYS} observed days before {cut}",
        "Top up Validation to ~10 % with groups drawn from OOS, stratified by\n"
        "climate × land cover × elevation band (never from Train)",
    ]
    ys = [0.955, 0.835, 0.685, 0.535]
    hs = [0.054, 0.078, 0.098, 0.062]
    for i, (t, y, h) in enumerate(zip(steps, ys, hs)):
        box(ax, XR, y, WR, h, t)
        if i:
            arrow(ax, XR, ys[i - 1] - hs[i - 1] / 2, XR, y + h / 2)
    ax.text(XR - WR / 2, 0.993, "(b)   Assigning stations to sets", ha="left", va="bottom")
    # (a) feeds (b): right from the last QC box, up the gap, into the first split box
    xm = (XL + WL / 2 + XR - WR / 2) / 2
    line(ax, [XL + WL / 2, xm, xm], [yl[-1], yl[-1], ys[0]])
    arrow(ax, xm, ys[0], XR - WR / 2, ys[0])
    ys, hs = [ys[-1]], [hs[-1]]                # the station-split row hangs off the last (b) box
    xsrc = XR

    # final station split
    yS, wS, hS = 0.315, 0.27, 0.07
    xs = [0.17, 0.5, 0.83]
    final = [("Train", n.get("train", 0), GREY, "white"), ("Validation", n.get("val", 0), LBLUE, "black"),
             ("OOS", n.get("oos", 0), BLUE, "white")]
    YB1 = 0.395                                         # horizontal bus under (b)
    line(ax, [xsrc, xsrc], [ys[-1] - hs[-1] / 2, YB1])
    line(ax, [xs[0], xs[-1]], [YB1, YB1])
    for x, (lab, k, fc, tc) in zip(xs, final):
        arrow(ax, x, YB1, x, yS + hS / 2)                 # straight down from the bus
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
    YB2 = 0.235
    for x, p0, p1, c0, c1 in zip(xs, pre, post, cpre, cpost):
        line(ax, [x, x], [yS - hS / 2, YB2]); line(ax, [x - 0.072, x + 0.072], [YB2, YB2])
        for dx, t, (fc, tc) in [(-0.072, p0, c0), (0.072, p1, c1)]:
            arrow(ax, x + dx, YB2, x + dx, yT + hT / 2)        # straight down from the small bus
            box(ax, x + dx, yT, wT, hT, t, fc=fc, tc=tc)
    ax.text(0.5, 0.035, f"Bottom row: each set split in time at {cut}. OOT and OOST only use stations with at least "
            f"{MIN_POST_CUT_DAYS} observed days after the cut.", ha="center", va="center")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"split_flowchart.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {out}/split_flowchart.{{png,pdf}}")


if __name__ == "__main__":
    main()
