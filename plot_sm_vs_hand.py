"""Soil moisture against HAND at TxSON — the look before §32.8's statistics.

HAND is the terrain field that survived §32.9.4: dHAND retains r = 0.99 under a 0.2 m
DEM perturbation where dTWI retains 0.20. So HAND is the one that CAN carry the
sufficiency gate. Whether it SHOULD is a different question, and this figure asks it
four ways, because a single cross-station scatter can hide or manufacture a result.

  A  station-mean SM vs HAND, observed and predicted on the same axes.
     The contrast matters: the model's predicted level is its own object (§31), so a
     terrain relationship visible only in the prediction is a model artefact.
  B  dSM vs dHAND over every station pair, with close pairs marked.
     This is the closest thing here to §32.8's actual gate, which regresses dSM on
     dHAND between colocated pairs. Pairing differences out the between-station
     confounds (soil, land cover, sensor) that panel A cannot control.
  C  wet vs dry, per §31.5. Saturation-excess is a WET-state mechanism: if HAND does
     anything, it should do it when the landscape is wet and vanish when it is dry.
     A null pooled over both states would hide that.
  D  by depth. The largest risk in the whole terrain arm is that these are
     saturation-excess concepts while the labels are the top 10 cm (§32.8, and §29
     measured -0.077 on the same shape of argument). If HAND reaches 30-100 cm but
     not 0-10, that is a finding, not a failure.

Usage
-----
    python plot_sm_vs_hand.py
    python plot_sm_vs_hand.py --region-id 2 --close-km 5
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

REPO         = Path(__file__).resolve().parent
TERRAIN_ROOT = Path("/gpfs/work3/0/prjs1968/data/terrain")
TS_PARQUET   = REPO / "eval_output" / "txson_timeseries.parquet"
ZARR_ROOT    = Path("/gpfs/scratch1/shared/pkhanal/zarr")
OUT_DIR      = REPO / "figures"
BANDS = ["twi", "hand", "acc_cells_mfd", "slope_rad", "valid"]


def observed_sm(station_ids) -> pd.DataFrame:
    """
    Station-mean OBSERVED soil moisture per depth, straight from the label zarr.

    Read from labels/ rather than from an evaluation parquet so the analysis is
    model-independent and works for any network, not just the one that happened to
    be evaluated. QC is honoured: 0 = observed, 1 = gap-filled. Gap-filled days are
    climatological fills, so including them would pull every station toward the same
    seasonal mean and shrink exactly the between-station contrast being tested.
    """
    import zarr
    rows = []
    for sid in station_ids:
        for cat in ("sm_only", "sm_and_flux", "flux_only"):
            p = ZARR_ROOT / cat / sid
            if not p.exists():
                continue
            try:
                z = zarr.open_consolidated(str(p), mode="r")
            except Exception:
                try:
                    z = zarr.open_group(str(p), mode="r")
                except Exception:
                    break
            if "labels" not in list(z.group_keys()):
                break
            L = z["labels"]
            sm = L["sm"][:]
            depths = [str(d) for d in L["depths"][:]]
            dates = pd.to_datetime([str(d) for d in L["dates"][:]], format="%Y%m%d")
            qc = L["qc"][:] if "qc" in list(L.array_keys()) else None
            for i, d in enumerate(depths):
                v = sm[i].astype(float)
                # qc can be stored over a different window than sm; only trust it
                # when the two line up, and say so rather than silently misaligning
                if qc is not None and qc.shape[1] == v.shape[0]:
                    v = np.where(qc[i] == 0, v, np.nan)
                ok = np.isfinite(v)
                if ok.sum() < 30:
                    continue
                rows.append(pd.DataFrame({"station": sid, "depth": d,
                                          "date": dates[ok], "obs": v[ok]}))
            break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["station", "depth", "date", "obs"])


def terrain_at_stations(rid: int, st: pd.DataFrame) -> pd.DataFrame:
    with rasterio.open(TERRAIN_ROOT / f"region_{rid:04d}" / "terrain_30m.tif") as src:
        arr = src.read()
        tr = src.transform
    res = float(tr.a)
    F = {b: arr[i] for i, b in enumerate(BANDS)}
    for name, key in (("hand_st", "hand"), ("twi_st", "twi"), ("slope_st", "slope_rad")):
        st[name] = [F[key][int((tr.f - y) / res), int((x - tr.c) / res)]
                    for x, y in zip(st["laea_x"], st["laea_y"])]
    return st


def fit_line(ax, x, y, colour, label):
    """Least squares fit plus Pearson r and its two-sided p, printed in the legend."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 4:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    # t = r sqrt(n-2)/sqrt(1-r^2); normal approximation is enough at n=40 to say
    # whether this is anywhere near significant
    t = abs(r) * np.sqrt(max(len(x) - 2, 1)) / np.sqrt(max(1 - r * r, 1e-12))
    from math import erfc, sqrt
    p = erfc(t / sqrt(2))
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=colour, lw=1.4, ls="--",
            label=f"{label}: r = {r:+.3f}, p ≈ {p:.2f}, n = {len(x)}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="Soil moisture vs HAND at TxSON")
    ap.add_argument("--region-id", type=int, default=2)
    ap.add_argument("--network", default=None,
                    help="Substring of station_id, e.g. HOBE or TxSON. Selects the "
                         "region containing those stations and labels the figure.")
    ap.add_argument("--depth", default=None,
                    help="Depth to analyse; default is the shallowest present.")
    ap.add_argument("--close-km", type=float, default=5.0,
                    help="Pairs closer than this are marked: they share climate and "
                         "are what §32.8's gate is actually built on.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    all_st = pd.read_csv(REPO / "csvs" / "station_dem_region.csv")
    if args.network:
        sel = all_st[all_st["station_id"].str.contains(args.network, case=False)]
        if sel.empty:
            raise SystemExit(f"no stations matching {args.network!r}")
        rid = int(sel["region_id"].mode().iloc[0])
        st = sel[sel["region_id"] == rid].reset_index(drop=True)
        label = args.network
    else:
        rid = args.region_id
        st = all_st[all_st["region_id"] == rid].reset_index(drop=True)
        label = f"region {rid}"
    st = terrain_at_stations(rid, st)

    ts = observed_sm(st["station_id"])
    if ts.empty:
        raise SystemExit("no observed labels found for these stations")

    depths = [d for d in ["0-10", "10-30", "30-100"] if d in set(ts["depth"])]
    surf = args.depth or depths[0]

    piv = ts.groupby(["station", "depth"])["obs"].mean().unstack()
    st = st.merge(piv.add_prefix("obs_"), left_on="station_id", right_index=True, how="left")

    # the model prediction is an optional overlay: only TxSON has been evaluated
    st["pred_" + surf] = np.nan
    if TS_PARQUET.exists():
        pr = pd.read_parquet(TS_PARQUET)
        pr = pr[pr["is_centre"]].groupby(["station", "depth"])["pred"].mean().unstack()
        if surf in pr:
            st = st.drop(columns=["pred_" + surf]).merge(
                pr[[surf]].add_prefix("pred_"), left_on="station_id",
                right_index=True, how="left")

    # wet / dry by the network's own median: the wettest and driest thirds of days
    daily = ts[ts["depth"] == surf].groupby("date")["obs"].mean()
    q1, q2 = daily.quantile([1 / 3, 2 / 3])
    dry_days = set(daily[daily <= q1].index)
    wet_days = set(daily[daily >= q2].index)
    sub = ts[ts["depth"] == surf]
    st = st.merge(sub[sub["date"].isin(wet_days)].groupby("station")["obs"]
                  .mean().rename("obs_wet"), left_on="station_id", right_index=True, how="left")
    st = st.merge(sub[sub["date"].isin(dry_days)].groupby("station")["obs"]
                  .mean().rename("obs_dry"), left_on="station_id", right_index=True, how="left")

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5})
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # ── A: observed and predicted level vs HAND ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(st["hand_st"], st[f"obs_{surf}"], s=44, c="#2c7fb8",
               edgecolor="k", linewidth=0.4, label="observed", zorder=3)
    ax.scatter(st["hand_st"], st[f"pred_{surf}"], s=32, c="#fdae61", marker="^",
               edgecolor="k", linewidth=0.3, label="model prediction", zorder=3)
    r_obs = fit_line(ax, st["hand_st"], st[f"obs_{surf}"], "#2c7fb8", "observed")
    r_pre = fit_line(ax, st["hand_st"], st[f"pred_{surf}"], "#d95f02", "predicted")
    ax.set_xlabel("HAND at station (m)")
    ax.set_ylabel(f"mean SM {surf} cm (m$^3$/m$^3$)")
    ax.set_title("A  station level vs HAND — observed and predicted")
    ax.legend(fontsize=7.2, loc="best")
    ax.grid(alpha=0.25)

    # ── B: pairwise differences — closest to the actual gate ─────────────────
    ax = fig.add_subplot(gs[0, 1])
    x, y, sep = [], [], []
    v = st.dropna(subset=["hand_st", f"obs_{surf}"]).reset_index(drop=True)
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            x.append(v.loc[i, "hand_st"] - v.loc[j, "hand_st"])
            y.append(v.loc[i, f"obs_{surf}"] - v.loc[j, f"obs_{surf}"])
            sep.append(np.hypot(v.loc[i, "laea_x"] - v.loc[j, "laea_x"],
                                v.loc[i, "laea_y"] - v.loc[j, "laea_y"]) / 1000.0)
    x, y, sep = np.array(x), np.array(y), np.array(sep)
    near = sep <= args.close_km
    ax.scatter(x[~near], y[~near], s=10, c="0.72", edgecolor="none",
               label=f"all pairs (n = {len(x)})", zorder=2)
    ax.scatter(x[near], y[near], s=26, c="#d7191c", edgecolor="k", linewidth=0.3,
               label=f"< {args.close_km:g} km apart (n = {int(near.sum())})", zorder=3)
    fit_line(ax, x, y, "0.35", "all pairs")
    if near.sum() > 4:
        fit_line(ax, x[near], y[near], "#d7191c", f"< {args.close_km:g} km")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("$\\Delta$HAND between stations (m)")
    ax.set_ylabel(f"$\\Delta$ mean SM {surf} (m$^3$/m$^3$)")
    ax.set_title("B  pairwise — differences out the between-station confounds")
    ax.legend(fontsize=7.2, loc="best")
    ax.grid(alpha=0.25)
    # 780 pairs come from 40 stations, so they are nowhere near independent: each
    # station appears in 39 of them. The printed p is the nominal one and is far too
    # small. The honest degrees of freedom are closer to the station count, which is
    # why §32.8 uses station fixed effects rather than a pooled pair regression.
    ax.text(0.02, 0.02,
            f"p is NOMINAL: {len(x)} pairs from {len(v)} stations are not\n"
            f"independent (each station is in {len(v)-1}). True d.f. ~ n stations.",
            transform=ax.transAxes, fontsize=6.8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec="0.7", lw=0.5))

    # ── C: wet vs dry ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(st["hand_st"], st["obs_wet"], s=42, c="#2166ac", edgecolor="k",
               linewidth=0.4, label="wettest third of days", zorder=3)
    ax.scatter(st["hand_st"], st["obs_dry"], s=42, c="#b2182b", edgecolor="k",
               linewidth=0.4, marker="s", label="driest third of days", zorder=3)
    r_wet = fit_line(ax, st["hand_st"], st["obs_wet"], "#2166ac", "wet")
    r_dry = fit_line(ax, st["hand_st"], st["obs_dry"], "#b2182b", "dry")
    ax.set_xlabel("HAND at station (m)")
    ax.set_ylabel(f"mean SM {surf} (m$^3$/m$^3$)")
    ax.set_title("C  wet vs dry (§31.5) — saturation excess is a wet-state mechanism")
    ax.legend(fontsize=7.2, loc="best")
    ax.grid(alpha=0.25)

    # ── D: by depth ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    cols = {"0-10": "#1b9e77", "10-30": "#7570b3", "30-100": "#e7298a"}
    rs = {}
    for d in depths:
        col = f"obs_{d}"
        if col not in st or st[col].notna().sum() < 5:
            continue
        z = st[[col, "hand_st"]].dropna()
        # standardised so three depths with different means share one axis
        zz = (z[col] - z[col].mean()) / z[col].std()
        ax.scatter(z["hand_st"], zz, s=34, c=cols.get(d, "0.5"), edgecolor="k",
                   linewidth=0.3, label=None, zorder=3)
        rs[d] = fit_line(ax, z["hand_st"], zz, cols.get(d, "0.5"), f"{d} cm")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("HAND at station (m)")
    ax.set_ylabel("standardised mean SM (z)")
    ax.set_title("D  by depth — does HAND reach the surface, or only deeper?")
    ax.legend(fontsize=7.2, loc="best")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Soil moisture vs HAND — {label} (region {rid}, {len(st)} stations, "
        f"observed labels)   "
        f"HAND is the terrain field that survived the stability test (§32.9.4)",
        fontsize=11, y=0.975)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (OUT_DIR / f"sm_vs_hand_{label.lower().replace(' ', '')}")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"wrote {out}.png / .pdf\n")

    print(f"A  observed {surf} vs HAND      r = {r_obs:+.4f}")
    print(f"A  predicted {surf} vs HAND     r = {r_pre:+.4f}")
    print(f"B  dSM vs dHAND, all pairs      r = {np.corrcoef(x, y)[0,1]:+.4f}  n = {len(x)}")
    if near.sum() > 4:
        print(f"B  dSM vs dHAND, < {args.close_km:g} km        "
              f"r = {np.corrcoef(x[near], y[near])[0,1]:+.4f}  n = {int(near.sum())}")
    print(f"C  wet third vs HAND            r = {r_wet:+.4f}")
    print(f"C  dry third vs HAND            r = {r_dry:+.4f}")
    for d, r in rs.items():
        print(f"D  {d:>6} cm vs HAND           r = {r:+.4f}")


if __name__ == "__main__":
    main()
