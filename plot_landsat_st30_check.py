#!/usr/bin/env python
"""Eyeball the Landsat ST bundles: do the maps look like a landscape, or like a bug?

Range checks pass on data that is spatially wrong.  verify_landsat_st.py proves the numbers are
plausible; this proves the PICTURES are.  Four things it is actually trying to catch:

1.  GEOLOCATION.  The station should sit where meta says it does.  Its pixel is marked on every
    map; if the LST pattern is offset from the terrain, or the marker lands off-tile, the grid
    construction is wrong.

2.  THE REPROJECTION.  30% of scenes arrive in a neighbouring UTM zone and are warped 30->30
    onto the station grid.  If that warp is sound, the median LST map built from IN-ZONE scenes
    and the one built from REPROJECTED scenes must agree.  If it is broken they will be shifted
    or rotated against each other.  This is the decisive panel, and it is measured (correlation
    and mean offset), not just drawn.  Only stations with both populations get it -- at
    Cascade#2 every scene is reprojected, so there is nothing to compare against.

3.  THE QC MASK.  A cloudy scene is drawn raw next to its decoded clear mask.  The mask should
    trace the cloud, not a grid artefact or the tile edge.

4.  THE STATIC HOLE.  §29.15 found 18.2% of the TxSON tile has no ST retrieval, and that the
    hole contained the wettest station.  The per-pixel clear-observation count shows whether
    each station has such a hole and where it is.

Runs in `soilmoisture` so it can import the REFERENCE decoder from download_landsat_st30 rather
than keeping a second copy that can drift.  It downloads nothing.

OUTPUT  fig/landsat_st30_check/_summary.png  -- always, an aggregate over every bundle
        fig/landsat_st30_check/{station}.png  -- only with --stations or --all-stations.
        Per-station figures are OPT-IN: a full run writes 993 PNGs / ~185 MB and buries the
        three aggregate plots that are what actually get looked at.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/gpfs/work3/0/prjs1968/soilMoisture")
from download_landsat_st30 import DATA_ROOT, qa_decode   # the reference decoder

warnings.filterwarnings("ignore", category=RuntimeWarning)

FIG = Path("/gpfs/work3/0/prjs1968/soilMoisture/fig/landsat_st30_check")
LST_CMAP, CNT_CMAP = "inferno", "viridis"


def _im(ax, a, title, cmap=LST_CMAP, cen=None, **kw):
    m = ax.imshow(a, cmap=cmap, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    if cen is not None:
        ax.plot(cen[0], cen[1], "c+", ms=11, mew=1.6)
    plt.colorbar(m, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    return m


def masked_median(cube, mask):
    out = np.where(mask, cube, np.nan)
    return np.nanmedian(out, axis=0), np.nanstd(out, axis=0), mask.sum(axis=0)


def station_figure(path: Path, save: bool = True) -> dict:
    """Per-station diagnostics.  `save` writes the six-panel PNG; the returned statistics feed
    the summary either way, so the aggregate is always computed over EVERY bundle even when only
    a handful of figures are written."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"][0]))
    name = meta["station_id"]
    lst, qap = z["lst30"], z["qa_pixel30"]
    clear, water = qa_decode(qap.astype("float64"))
    ok = clear & np.isfinite(lst)
    cen = meta.get("centre_px", [38, 38])

    med, sd, cnt = masked_median(lst, ok)
    cf = z["clear_frac"]
    repro = z["reprojected"].astype(bool)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    fig.suptitle(f"{name}   ({meta['category']}, EPSG:{meta['epsg']}, "
                 f"N={lst.shape[0]} scenes, {int(repro.sum())} reprojected)", fontsize=11)

    _im(axes[0, 0], med, "median clear LST (K)\nthe static pattern (§29.15)", cen=cen)
    _im(axes[0, 1], sd, "temporal SD of clear LST (K)", cmap="magma", cen=cen)
    _im(axes[0, 2], cnt, f"clear observations per pixel\n(max {int(cnt.max())})",
        cmap=CNT_CMAP, cen=cen)
    _im(axes[0, 3], z["emis30"], "emis30 (static)\n§41.5 falsification input",
        cmap="cividis", cen=cen)

    # clearest and cloudiest scenes, side by side with the decoded mask
    i_clear, i_cloud = int(np.argmax(cf)), int(np.argmin(cf))
    _im(axes[1, 0], lst[i_clear], f"clearest scene {z['dates'][i_clear]}\n"
                                 f"clear_frac={cf[i_clear]:.2f}", cen=cen)
    _im(axes[1, 1], lst[i_cloud], f"cloudiest scene {z['dates'][i_cloud]}\n"
                                 f"clear_frac={cf[i_cloud]:.2f}  (RAW, unmasked)", cen=cen)
    _im(axes[1, 2], clear[i_cloud].astype(float), "decoded clear mask, same scene\n"
        "should trace the cloud, not the tile edge", cmap="Greys_r", vmin=0, vmax=1, cen=cen)

    # THE REPROJECTION TEST
    ax = axes[1, 3]
    out = {"station": name, "n": int(lst.shape[0]), "n_repro": int(repro.sum())}
    if repro.any() and (~repro).any() and ok[repro].any() and ok[~repro].any():
        m_in, _, c_in = masked_median(lst[~repro], ok[~repro])
        m_rp, _, c_rp = masked_median(lst[repro], ok[repro])
        good = np.isfinite(m_in) & np.isfinite(m_rp) & (c_in >= 3) & (c_rp >= 3)
        if good.sum() > 40:
            a, b = m_in[good] - m_in[good].mean(), m_rp[good] - m_rp[good].mean()
            r = float(np.corrcoef(a, b)[0, 1])
            bias = float(np.nanmean(m_rp[good] - m_in[good]))
            out.update(repro_r=round(r, 4), repro_bias_k=round(bias, 3),
                       repro_npx=int(good.sum()))
            d = np.where(good, m_rp - m_in, np.nan)
            v = np.nanpercentile(np.abs(d), 98) or 1.0
            _im(ax, d, f"REPROJECTED minus IN-ZONE median (K)\n"
                       f"pattern r={r:+.3f}   mean offset={bias:+.2f} K",
                cmap="RdBu_r", vmin=-v, vmax=v, cen=cen)
        else:
            ax.text(.5, .5, "too few co-valid pixels", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
    else:
        why = "all scenes reprojected" if repro.all() else "no reprojected scenes"
        ax.text(.5, .5, f"reprojection test N/A\n({why})", ha="center", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("reprojection check", fontsize=8)

    if save:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / f"{name.replace(chr(47), chr(95))}.png", dpi=115)
    plt.close(fig)

    out.update(
        lst_med=round(float(np.nanmedian(med)), 2),
        hole_frac=round(float((cnt == 0).mean()), 4),
        emis_med=round(float(np.nanmedian(z["emis30"])), 4),
        centre_lst=round(float(np.nanmedian(lst[:, int(cen[1]), int(cen[0])][ok[:, int(cen[1]), int(cen[0])]])), 2)
        if ok[:, int(cen[1]), int(cen[0])].any() else np.nan,
        # §41.5 preview: is the static LST pattern just the emissivity pattern?
        r_lst_emis=_corr(med, z["emis30"]),
        dates=z["dates"], tile_mean=np.array([np.nanmean(np.where(ok[i], lst[i], np.nan))
                                              for i in range(lst.shape[0])]),
    )
    return out


def _corr(a, b):
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 40 or np.nanstd(b[g]) == 0:
        return np.nan
    return round(float(np.corrcoef(a[g], b[g])[0, 1]), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default="",
                    help="comma-separated station_ids to write per-station figures for")
    ap.add_argument("--all-stations", action="store_true",
                    help="write a figure for EVERY station -- 993 PNGs, ~185 MB. Off by "
                         "default: the aggregates are what get looked at, and a full run "
                         "buries them.")
    args = ap.parse_args()

    bundles = sorted(DATA_ROOT.glob("*/*/LANDSAT_ST/*_st30_*.npz"))
    print(f"{len(bundles)} bundles")
    if not bundles:
        raise SystemExit("no bundles")

    want = {s.strip() for s in args.stations.split(",") if s.strip()}

    def _save(p: Path) -> bool:
        if args.all_stations:
            return True
        return bool(want) and p.parent.parent.name.split("_")[-1] in want

    # every bundle is still READ -- the summary is an aggregate over all 993 either way.
    res = [station_figure(p, save=_save(p)) for p in bundles]
    n_fig = sum(1 for p in bundles if _save(p))
    print(f"per-station figures written: {n_fig}"
          f"{'  (use --stations or --all-stations for more)' if not n_fig else ''}")

    # ---- summary: seasonality is the cheapest sanity check there is
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in res:
        d = np.array([int(x[4:6]) for x in r["dates"]])
        axes[0].plot(d + np.random.uniform(-.2, .2, d.size), r["tile_mean"], ".",
                     ms=3, alpha=.45, label=r["station"])
    axes[0].set_xlabel("month"); axes[0].set_ylabel("tile-mean clear LST (K)")
    axes[0].set_title("Seasonal cycle -- if this is flat, something is wrong", fontsize=10)
    axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=.3)

    st = [r["station"] for r in res]
    axes[1].barh(st, [r.get("r_lst_emis", np.nan) for r in res], color="tab:purple")
    axes[1].set_xlabel("corr(median LST map, emis map)")
    axes[1].set_title("§41.5 preview: is the static pattern just emissivity?", fontsize=10)
    axes[1].axvline(0, color="k", lw=.8); axes[1].grid(alpha=.3, axis="x")
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "_summary.png", dpi=115)
    plt.close(fig)

    print(f"\n{'station':<22}{'N':>5}{'repro':>7}{'LSTmed':>8}{'centre':>8}"
          f"{'hole':>7}{'emis':>7}{'r(LST,emis)':>13}{'repro_r':>9}{'bias_K':>8}")
    for r in res:
        print(f"{r['station']:<22}{r['n']:>5}{r['n_repro']:>7}{r['lst_med']:>8.2f}"
              f"{r['centre_lst']:>8.2f}{r['hole_frac']:>7.3f}{r['emis_med']:>7.3f}"
              f"{str(r.get('r_lst_emis','-')):>13}{str(r.get('repro_r','-')):>9}"
              f"{str(r.get('repro_bias_k','-')):>8}")
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
