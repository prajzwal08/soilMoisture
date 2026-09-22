#!/usr/bin/env python
"""Step 2b: the observed soil-moisture record of every station in a tile, on its own.

The map figure cannot answer the question this one exists for: DO THESE CO-LOCATED
PROBES ACTUALLY DISAGREE, AND WHEN?  That is the precondition for any within-tile
spatial claim, and outside TxSON's six (plot_txson_six.py) it has never been drawn.

One panel per depth, every member overplotted, plus a between-station spread trace --
the SD across members per day, which is the quantity a spatial model is trying to
reproduce and the honest scale against which 34.4's "between-station spread 15-19% of
observed, target > 35%" is judged.

OBSERVED AND GAP-FILLED ARE DRAWN DIFFERENTLY and that is not cosmetic.  The fill is a
month-day climatology (gapfill_by_monthday_mean_with_feb29_fallback), so a filled
station can look perfectly well-behaved while carrying no information at all.

Env: terramind.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_gra_thermal import (REPO, OUT_DIR, load_bundle, member_labels,
                              pick_summer_dates, _as_str)
from dataset import SM_DEPTHS

log = logging.getLogger("gra_sm_ts")


def draw(cl, mem, args) -> bool:
    cid = cl.cluster_id
    inside = mem[mem.in_tile == 1].reset_index(drop=True)
    if inside.empty:
        return False
    labels = {r.station_id: member_labels(r.folder, r.category)
              for _, r in inside.iterrows()}
    if not any(labels.values()):
        log.warning("%s: no soil-moisture record for any member -- skipped", cid)
        return False

    # the same dates the map figure marks, so the two read together
    z, _ = load_bundle(cl.rep_folder)
    marks = []
    if z is not None:
        marks = [_as_str(z["day_utc"][i])[:10]
                 for _, i in pick_summer_dates(z, args.months, args.min_valid_frac)]

    depths = [d for d in SM_DEPTHS
              if any(d in v and not v[d].empty for v in labels.values())]
    if not depths:
        log.warning("%s: no depth has data -- skipped", cid)
        return False

    nrow = len(depths) + 1
    fig, axes = plt.subplots(nrow, 1, figsize=(13.5, 2.4 * nrow), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")

    fill_pct = {}
    for k, dep in enumerate(depths):
        ax = axes[k]
        wide = {}
        for j, (_, m) in enumerate(inside.iterrows()):
            d = labels.get(m.station_id, {}).get(dep)
            if d is None or d.empty:
                continue
            col = cmap(j % 10)
            obs, fil = d[d.qc == 0], d[d.qc == 1]
            ax.plot(obs.date, obs.sm, lw=0.9, color=col,
                    label=str(m.station_id) if k == 0 else None)
            if not fil.empty:
                ax.plot(fil.date, fil.sm, lw=0.8, color=col, alpha=0.30, ls=":")
            fill_pct.setdefault(m.station_id, {})[dep] = float((d.qc == 1).mean())
            wide[m.station_id] = d.set_index("date")["sm"].where(d.set_index("date")["qc"] == 0)
        ax.set_ylabel(f"{dep} cm\n(m3/m3)", fontsize=9)
        ax.grid(alpha=.25, lw=.5)
        ax.tick_params(labelsize=8)
        for dt in marks:
            ax.axvline(pd.Timestamp(dt), color="#444", lw=0.8, ls="--", alpha=.65, zorder=1)
        if k == 0 and wide:
            ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper left")
        # surface depth carries the spread trace
        if dep == depths[0] and len(wide) >= 2:
            W = pd.DataFrame(wide)
            sd = W.std(axis=1, ddof=0)
            n_ok = W.notna().sum(axis=1)
            sd = sd.where(n_ok >= 2)
            ax2 = axes[-1]
            ax2.plot(sd.index, sd.values, lw=0.8, color="#B03A2E")
            ax2.set_ylabel("between-station\nSD (m3/m3)", fontsize=9)
            ax2.grid(alpha=.25, lw=.5)
            ax2.tick_params(labelsize=8)
            for dt in marks:
                ax2.axvline(pd.Timestamp(dt), color="#444", lw=0.8, ls="--", alpha=.65)
            med = float(np.nanmedian(sd.values)) if np.isfinite(sd.values).any() else np.nan
            obs_sd = float(np.nanmedian(W.std(axis=0, ddof=0)))
            pct = 100 * med / obs_sd if obs_sd and np.isfinite(obs_sd) else np.nan
            ax2.set_title(
                f"median between-station SD {med:.4f} m3/m3 "
                f"= {pct:.0f}% of the median within-station temporal SD "
                f"({obs_sd:.4f})   [34.4 target for the model: > 35%]",
                fontsize=8.5, pad=4)

    if len(depths) + 1 == nrow and len(inside) < 2:
        axes[-1].text(.5, .5, "only one station in this window -- no spread to show",
                      ha="center", va="center", transform=axes[-1].transAxes, color="#888")
    axes[-1].set_xlabel("date", fontsize=9)

    worst = sorted(((max(v.values()), s) for s, v in fill_pct.items()), reverse=True)[:3]
    tag = "   ".join(f"{s} {100*f:.0f}% filled" for f, s in worst)
    fig.suptitle(
        f"{cid}   {cl.n_stations} station(s), extent {cl.extent_km:.2f} km   "
        f"observed soil moisture\n"
        f"solid = observed (qc 0), dotted = gap-filled month-day climatology (qc 1)"
        + (f"   |   most-filled: {tag}" if tag else ""),
        fontsize=10.5, y=0.998)

    out = OUT_DIR / f"sm_timeseries_{cid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("%s: wrote %s", cid, out)
    for s, v in sorted(fill_pct.items()):
        log.info("    %-26s gap-filled %s", s,
                 "  ".join(f"{d}:{100*f:.0f}%" for d, f in v.items()))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default=str(REPO / "csvs" / "gra_thermal_clusters.csv"))
    ap.add_argument("--members",  default=str(REPO / "csvs" / "gra_thermal_members.csv"))
    ap.add_argument("--cluster", default="")
    ap.add_argument("--months", default="5,6,7,8,9",
                    help="growing season. JJAS alone yields only 2 columns at "
                         "most TxSON clusters -- not because of cloud but because "
                         "whole months are off-swath (best valid fraction 0.00)")
    ap.add_argument("--min-valid-frac", type=float, default=0.75)
    ap.add_argument("--min-members", type=int, default=2)
    args = ap.parse_args()
    args.months = [int(x) for x in args.months.split(",") if x.strip()]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    C = pd.read_csv(args.clusters)
    M = pd.read_csv(args.members)
    if args.cluster:
        C = C[C.cluster_id == args.cluster]
    C = C[C.n_stations >= args.min_members]
    log.info("rendering %d cluster(s)", len(C))

    ok = 0
    for _, cl in C.iterrows():
        ok += bool(draw(cl, M[M.cluster_id == cl.cluster_id], args))
    log.info("")
    log.info("%d/%d SM time-series figures written to %s", ok, len(C), OUT_DIR)


if __name__ == "__main__":
    main()
