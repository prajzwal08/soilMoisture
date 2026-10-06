# ============================================================
# Mead: are the Ne1 and Ne2 TerraMind embeddings different, and does that
# difference explain the soil-moisture difference?  (runbook §69.5)
# ============================================================
# The model reads L12 tokens (14 x 14 grid of 160 m patches per 2.24 km tile):
#   pooled pyramid, level 0 = centre 2x2 tokens of each station's OWN tile  -> "tile centres"
#   the anchor = full 14 x 14 grid; inside Ne1's tile Ne2 sits 3.4 tokens east -> "same tile"
# Per 2019 acquisition (S2 and S1 ascending), cosine similarity of the mean 2x2 token
# vector at Ne1 vs at Ne2:
#   same tile  : Ne1 tile, rows 6-7, cols 6-7 (Ne1) vs cols 9-10 (Ne2)
#   tile centre: Ne1 tile rows/cols 6-7 vs Ne2 tile rows/cols 6-7 (what the pyramid sees)
#   reference  : Ne1 tile, Ne1 block vs cols 3-4 (a neighbour 3 tokens west, same distance)
# Blocks with any invalid token (cloud for S2, token_mask for S1) -> NaN.
# Then: Spearman of (1 - cos) with |Ne1 - Ne2| SM on the acquisition day, observed and model.
# Figures: mead_embed_cos_<year>  time series (S2, S1, |dSM| obs/model, P)
#          mead_embed_scatter_<year> (1 - cos) vs |dSM|, observed and model
#          mead_embed_matrix_<year>  date x date cosine (Ne1-Ne1, Ne2-Ne2, Ne1-Ne2) per sensor
# Dates: S2 = clear scenes from plot_mead_ndvi_sm.py (cloud >= 0.99 clear, no snow; run it first),
#        S1 = soil unfrozen (ts > 0) at both sites.
#
# Usage:
#   python plot_mead_embed.py [--year 2019]
# ============================================================

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import spearmanr

import plot_style_bw
from dataset import ZARR_ROOT, _cm_token_mask
from plot_mead_pred import ROOT, FIG_DIR, PRED, PRED_COLOR, observed, predicted
from plot_mead_precip import load as load_precip

ST    = {"Ne1": "AmeriFlux_US-Ne1", "Ne2": "AmeriFlux_US-Ne2"}
ROWS  = slice(6, 8)
BLK   = {"ne1": slice(6, 8), "ne2": slice(9, 11), "ref": slice(3, 5)}
BINS  = ["0-10", "10-30"]


def tokens(st, sensor, year):
    """dates, (N,14,14,768) float32 L12, (N,14,14) valid for one station tile and year."""
    g = zarr.open_group(str(ZARR_ROOT / "flux_only" / st), mode="r")
    grp = "s2" if sensor == "S2" else "s1_asc"
    d = pd.to_datetime([str(x)[:8] for x in g[f"{grp}/dates"][:]], format="%Y%m%d")
    i = np.where(d.year == year)[0]
    t = np.asarray(g[f"{grp}/l12"].get_orthogonal_selection((i, slice(None), slice(None))),
                   dtype=np.float32).reshape(len(i), 14, 14, -1)
    if sensor == "S1":
        v = np.asarray(g["s1_asc/token_mask"].get_orthogonal_selection(
            (i, slice(None), slice(None)))).astype(bool)
    else:
        cd = pd.to_datetime([str(x)[:8] for x in g["cm/dates"][:]], format="%Y%m%d")
        row = {x: k for k, x in enumerate(cd)}
        cm = np.asarray(g["cm/masks"][:])
        v = np.zeros((len(i), 14, 14), bool)
        for k, x in enumerate(d[i]):
            if x in row:
                v[k] = _cm_token_mask(cm[row[x]][None])[0]
    v &= np.isfinite(t).all(-1)
    return d[i], t, v


def block(t, v, cols):
    """Mean token over a 2x2 block; NaN where any token in it is invalid."""
    m = t[:, ROWS, cols].mean(axis=(1, 2))
    ok = v[:, ROWS, cols].all(axis=(1, 2))
    m[~ok] = np.nan
    return m


def cos(a, b):
    num = (a * b).sum(-1)
    return num / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    y = args.year
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PRED)
    df["date"] = pd.to_datetime(df["date"])
    obs = {k: observed(v)[0] for k, v in ST.items()}
    tsoil = {k: observed(v)[1] for k, v in ST.items()}
    f = ROOT / "pred" / f"ndvi_sm_diff_{y}.csv"           # written by plot_mead_ndvi_sm.py
    s2_ok = pd.DatetimeIndex(pd.read_csv(f, index_col=0, parse_dates=True).index)
    own = {k: predicted(df, v, True) for k, v in ST.items()}
    w = df[df["tile"] == ST["Ne1"]].pivot_table(index=["date", "depth"], columns="station", values="pred")
    same = (w[ST["Ne1"]] - w[ST["Ne2"]]).unstack("depth")
    p, _ = load_precip()
    irr = pd.read_csv(ROOT / "bif" / "US-Ne2_irrigation_events.csv", parse_dates=["date_start", "date_end"])
    irr = irr[(irr["date_start"].dt.year == y) & ~irr["cancelled"].fillna(False).astype(bool)]

    res, emb = {}, {}
    for sensor in ("S2", "S1"):
        d1, t1, v1 = tokens(ST["Ne1"], sensor, y)
        d2, t2, v2 = tokens(ST["Ne2"], sensor, y)
        common = d1.intersection(d2)
        i1 = [d1.get_loc(x) for x in common]
        i2 = [d2.get_loc(x) for x in common]
        a_ne1 = block(t1, v1, BLK["ne1"])[i1]
        a_ne2 = block(t1, v1, BLK["ne2"])[i1]
        a_ref = block(t1, v1, BLK["ref"])[i1]
        b_ne2 = block(t2, v2, BLK["ne1"])[i2]                 # Ne2 at its own tile centre
        r = pd.DataFrame({"same_tile": cos(a_ne1, a_ne2), "tile_centre": cos(a_ne1, b_ne2),
                          "ref_west": cos(a_ne1, a_ref)}, index=common)
        if sensor == "S2":                               # clear: cloud-free >= 0.99, no snow
            keep = common.isin(s2_ok)
        else:                                            # S1: drop frozen soil (ts <= 0 at either site)
            keep = np.ones(len(common), bool)
            for k in ST:
                keep &= (tsoil[k].reindex(common) > 0).values
        print(f"\n{sensor} {y}: keep {int(keep.sum())} of {len(common)} shared dates "
              f"({'clear S2 from ' + f.name if sensor == 'S2' else 'unfrozen soil at both sites'})")
        r = r[keep]
        common, a_ne1, b_ne2 = common[keep], a_ne1[keep], b_ne2[keep]
        for b in BINS:
            r[f"obs_{b}"] = (obs["Ne1"][b] - obs["Ne2"][b]).reindex(common).values
            r[f"own_{b}"] = (own["Ne1"][b] - own["Ne2"][b]).reindex(common).values
            r[f"same_{b}"] = same[b].reindex(common).values
        res[sensor] = r
        emb[sensor] = (common, a_ne1, b_ne2)
        print(f"\n{sensor} {y}: {len(common)} shared dates, "
              f"{int(r['same_tile'].notna().sum())} with both blocks valid")
        print(r[["same_tile", "tile_centre", "ref_west"]].describe().loc[["count", "mean", "min", "max"]]
              .to_string(float_format=lambda v: f"{v:.4f}"))
        js = (r.index.month >= 6) & (r.index.month <= 9)
        print(f"  Jun-Sep mean cos: same tile {r.loc[js, 'same_tile'].mean():.4f}  "
              f"tile centre {r.loc[js, 'tile_centre'].mean():.4f}  ref west {r.loc[js, 'ref_west'].mean():.4f}")
        print(f"  Spearman of (1 - cos) with |Ne1 - Ne2| SM on acquisition days (all {y} / Jun-Sep):")
        for c in ("same_tile", "tile_centre"):
            for b in BINS:
                for lab, col in (("obs", f"obs_{b}"), ("model own", f"own_{b}"), ("model same", f"same_{b}")):
                    out = []
                    for sel in (np.ones(len(r), bool), js):
                        x = (1 - r.loc[sel, c]).values
                        z = r.loc[sel, col].abs().values
                        ok = np.isfinite(x) & np.isfinite(z)
                        out.append(f"{spearmanr(x[ok], z[ok])[0]:+.2f} (n {ok.sum()})" if ok.sum() >= 5
                                   else "   n/a")
                    print(f"    {c:>11} vs {lab:>10} {b:>5}: {out[0]}   {out[1]}")
        r.to_csv(ROOT / "pred" / f"embed_cos_{sensor}_{y}.csv")

    # --- time series ---
    plot_style_bw.apply(globals(), "paper")
    lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    fig, ax = plt.subplots(5, 1, figsize=(9, 12), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.7]})
    for k, sensor in enumerate(("S2", "S1")):
        a = ax[k]
        r = res[sensor]
        for c, mk, col, lab in (("same_tile", "o", PRED_COLOR, "Ne1 vs Ne2, same tile"),
                                ("tile_centre", "^", plot_style_bw.GREEN, "Ne1 vs Ne2, tile centres"),
                                ("ref_west", "x", "0.5", "Ne1 vs 480 m west (reference)")):
            s = r[c].dropna()
            a.plot(s.index, s.values, "-" + mk, ms=3.5, lw=0.8, color=col, label=lab)
        a.text(0.01, 0.95, f"({'ab'[k]}) {sensor} L12 cosine similarity", transform=a.transAxes, va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
        a.set_ylabel("Cosine similarity")
    for k, b in enumerate(BINS, start=2):
        a = ax[k]
        o = (obs["Ne1"][b] - obs["Ne2"][b]).loc[lo:hi].abs()
        a.plot(o.index, o.values, "-", color="black", lw=1.2, label="|Ne1 $-$ Ne2| observed")
        m = (own["Ne1"][b] - own["Ne2"][b]).loc[lo:hi].abs()
        a.plot(m.index, m.values, "-", color=PRED_COLOR, lw=1.2, label="|Ne1 $-$ Ne2| model")
        a.text(0.01, 0.95, f"({'cd'[k - 2]}) {b} cm", transform=a.transAxes, va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
        a.set_ylabel("|$\\Delta$SM| (m$^3$ m$^{-3}$)")
    a = ax[4]
    e = p["ERA5"].loc[lo:hi]
    a.bar(e.index, e.values, width=1.0, color="0.6", label="ERA5-Land P")
    g2 = p["US-Ne2"].loc[lo:hi]
    g2 = g2[g2 >= 0.2]
    a.plot(g2.index, g2.values, "^", ms=3.5, color=plot_style_bw.GREEN, label="Ne2 gauge")
    a.text(0.01, 0.95, "(e) precipitation", transform=a.transAxes, va="top",
           bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.5))
    a.set_ylabel("P (mm d$^{-1}$)")
    for a in ax:
        for rr in irr.itertuples():
            a.axvspan(rr.date_start, rr.date_end + pd.Timedelta("1D"), color="0.88", lw=0, zorder=0)
    a.set_xlim(lo, hi + pd.Timedelta("1D"))
    h, l = [], []
    for a in (ax[0], ax[2], ax[4]):
        hh, ll = a.get_legend_handles_labels()
        h, l = h + hh, l + ll
    h.append(plt.Rectangle((0, 0), 1, 1, color="0.88"))
    l.append("Ne2 irrigation logged")
    ax[0].legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False)
    fig.align_ylabels(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_embed_cos_{y}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_embed_cos_{y}.png")

    # --- scatter: embedding difference vs SM difference (filtered dates) ---
    fig, ax = plt.subplots(2, 2, figsize=(8, 7.5))
    for k, sensor in enumerate(("S2", "S1")):
        r = res[sensor]
        js = (r.index.month >= 6) & (r.index.month <= 9)
        for j, b in enumerate(BINS):
            a = ax[k, j]
            x = 1 - r["tile_centre"]
            txt = []
            for lab, col, mk, name in (("obs", "black", "o", "observed"),
                                       ("own", PRED_COLOR, "^", "model")):
                z = r[f"{lab}_{b}"].abs()
                for sel, fill in ((js, col), (~js, "white")):
                    a.plot(x[sel], z[sel], mk, ms=5, mfc=fill, mec=col, mew=0.9,
                           label=f"{name}, {'Jun-Sep' if fill == col else 'Oct-May'}")
                ok = np.isfinite(x.values) & np.isfinite(z.values)
                txt.append(f"{name} {spearmanr(x.values[ok], z.values[ok])[0]:+.2f}")
            a.text(0.03, 0.97, f"({'abcd'[2 * k + j]}) {sensor}, {b} cm  (n {len(r)})\nSpearman "
                   + ", ".join(txt), transform=a.transAxes, va="top", fontsize="small",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
            if j == 0:
                a.set_ylabel("|$\\Delta$SM| Ne1 $-$ Ne2 (m$^3$ m$^{-3}$)")
            a.set_xlabel(f"1 $-$ cosine, {sensor} L12 (tile centres)")
    ax[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_embed_scatter_{y}.{ext}", dpi=600)
    print(f"figure: {FIG_DIR}/mead_embed_scatter_{y}.png")

    # --- date x date matrices ---
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for k, sensor in enumerate(("S2", "S1")):
        dd, a1, b2 = emb[sensor]
        ok = np.isfinite(a1).all(1) & np.isfinite(b2).all(1)
        dd, a1, b2 = dd[ok], a1[ok], b2[ok]
        n1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
        n2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
        for j, (m, lab) in enumerate(((n1 @ n1.T, "Ne1 vs Ne1"), (n2 @ n2.T, "Ne2 vs Ne2"),
                                      (n1 @ n2.T, "Ne1 (rows) vs Ne2 (cols)"))):
            a = ax[k, j]
            im = a.imshow(m, vmin=0.5, vmax=1.0, cmap="viridis")
            tk = np.arange(0, len(dd), max(1, len(dd) // 8))
            a.set_xticks(tk)
            a.set_xticklabels([f"{dd[t]:%d %b}" for t in tk], rotation=90, fontsize="small")
            a.set_yticks(tk)
            a.set_yticklabels([f"{dd[t]:%d %b}" for t in tk], fontsize="small")
            a.text(0.02, 0.98, f"({'abcdef'[3 * k + j]}) {sensor}, {lab}", transform=a.transAxes,
                   va="top", fontsize="small",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.colorbar(im, cax=fig.add_axes([0.93, 0.25, 0.015, 0.5]), label="Cosine similarity")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mead_embed_matrix_{y}.{ext}", dpi=300)
    print(f"figure: {FIG_DIR}/mead_embed_matrix_{y}.png  (tile-centre 2x2 blocks, valid dates only)")


if __name__ == "__main__":
    main()
