"""
measure_token_scale.py — how big ARE the frozen TerraMind tokens? (§35.25)
==========================================================================

§35.24 added an input LayerNorm on the frozen L12 features (`model.py` s2_norm /
s1_norm / dem_norm / lulc_norm) on an argument that was never measured:

    Every positional and modality embedding is now initialised at std 0.02
    (EMB_INIT_STD, the ViT/BERT convention).  A token is annotated by ADDING that
    code to it.  If the token's own per-element spread is ~50, the annotation is a
    0.04% perturbation and the model cannot tell a 3-day-old acquisition from a
    300-day-old one.  If the spread is already ~1, the annotation is already ~2%
    and the input LayerNorm buys nothing.

That is a single number per modality, and nobody has measured it.  This script does.

THE DECISIVE NUMBER is `per_elem_std`: the standard deviation across the 768 features
INSIDE one token.  That is exactly the quantity LayerNorm divides by, so:

    tag_share_raw  = EMB_INIT_STD / per_elem_std      <- what the tag is worth today
    tag_share_norm = EMB_INIT_STD / 1.0  = 0.02       <- what it is worth after LayerNorm

If tag_share_raw is already ~0.02, the input LayerNorm is a no-op and should be
turned off (it costs a little and risks the second question below).

THE SECOND QUESTION is what LayerNorm destroys.  It deletes each token's magnitude
and keeps only its direction.  For the patchwise hypothesis (§34) the thing that
matters is WITHIN-TILE variation across the 196 patches — that is the signal the
whole architecture exists to find.  So for each tile-acquisition we split the
within-tile variance into the part carried by magnitude and the part carried by
direction:

    T_k              token of patch k                     (196, 768)
    n_k = ||T_k||    its magnitude
    U_k = T_k / n_k  its direction (unit vector)

    var_total  = mean_k || T_k - mean(T) ||^2
    var_mag    = variance you keep if every patch shares the MEAN direction and
                 differs only in magnitude          -> n_k * u_bar
    var_dir    = variance you keep if every patch shares the MEAN magnitude and
                 differs only in direction          -> n_bar * U_k

If var_mag / var_total is large, input LayerNorm is throwing away a real fraction of
the within-tile signal and should not be on by default.  If it is small, LayerNorm
costs nothing and the first number decides.

Everything is ALSO reported with the register dimensions removed.  §27a/§35.3
established that a handful of shared coordinates carry ~88% of the magnitude
(csvs/register_across_modalities.json), so the raw norms are mostly register and the
interesting question is what the content does underneath them.

Usage (SLURM only — nothing runs on the login node):
    sbatch slurm/token_scale.sh
Output:
    csvs/token_scale.json   + a printed table
"""

from __future__ import annotations

import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Import the real constants rather than restating them, so this measures exactly what
# the model consumes.
from dataset import ZARR_ROOT, STATION_TOKEN, N_TOKENS          # noqa: E402
from model import EMB_INIT_STD                                   # noqa: E402

SPLITS_CSV      = Path("csvs/station_splits.csv")
OUT_JSON        = Path("csvs/token_scale.json")

# Store layout is NOT uniform: the temporal modalities nest the layer
# ({mod}/l12, shape (N, 196, 768)) while the statics are top-level arrays
# (dem, lulc, shape (196, 768)). The first version of this script assumed
# "{mod}/l12" for all five and silently reported nothing for DEM and LULC —
# they were absent from the output table rather than flagged, which is the
# same fail-quiet pattern §35.24 spent the day removing.
MODALITY_KEYS = {
    "s2":      "s2/l12",
    "s1_asc":  "s1_asc/l12",
    "s1_desc": "s1_desc/l12",
    "dem":     "dem",
    "lulc":    "lulc",
}
MODALITIES = list(MODALITY_KEYS)
N_STATIONS      = 120      # sampled stations; the statistic is per-token, so this is plenty
N_ACQ_PER_STA   = 4        # temporal modalities: acquisitions sampled per station
N_REGISTER_DIMS = 6        # §35.3 reports ~6 dims carrying the bulk of the magnitude
SEED            = 0


# ── one station ───────────────────────────────────────────────────────────────

def _tile_stats(T: np.ndarray) -> dict:
    """Within-tile magnitude-vs-direction split for one (P, 768) tile-acquisition."""
    T = T.astype(np.float64)
    n = np.linalg.norm(T, axis=1)                       # (P,)
    keep = n > 1e-9
    if keep.sum() < 8:
        return {}
    T, n = T[keep], n[keep]

    T_bar   = T.mean(0)
    var_tot = float(((T - T_bar) ** 2).sum(1).mean())
    if var_tot <= 0:
        return {}

    U      = T / n[:, None]                             # unit directions
    u_bar  = U.mean(0)
    u_bar /= max(np.linalg.norm(u_bar), 1e-12)
    n_bar  = float(n.mean())

    # magnitude-only reconstruction: common direction, per-patch magnitude
    M_mag   = n[:, None] * u_bar[None, :]
    var_mag = float(((M_mag - M_mag.mean(0)) ** 2).sum(1).mean())

    # direction-only reconstruction: common magnitude, per-patch direction
    M_dir   = n_bar * U
    var_dir = float(((M_dir - M_dir.mean(0)) ** 2).sum(1).mean())

    return {
        "var_total":     var_tot,
        "frac_mag":      var_mag / var_tot,
        "frac_dir":      var_dir / var_tot,
        "norm_cv":       float(n.std() / max(n.mean(), 1e-12)),
        "mean_cos_to_bar": float((U @ u_bar).mean()),
    }


def _scan_station(args) -> dict | None:
    category, station = args
    path = ZARR_ROOT / category / station
    if not (path / ".complete").exists():
        return None
    try:
        try:
            zg = zarr.open_consolidated(str(path), mode="r")
        except KeyError:
            zg = zarr.open_group(str(path), mode="r")
    except Exception:
        return None

    rng = np.random.default_rng(abs(hash(station)) % (2**32))
    out: dict[str, dict] = {}

    for mod, key in MODALITY_KEYS.items():
        if key not in zg:
            continue
        arr = zg[key]
        try:
            if arr.ndim == 3:                                   # (N, 196, 768) temporal
                N = arr.shape[0]
                if N == 0:
                    continue
                idx = rng.choice(N, size=min(N_ACQ_PER_STA, N), replace=False)
                tiles = [np.asarray(arr[int(i)]) for i in sorted(idx)]
            elif arr.ndim == 2:                                 # (196, 768) static
                tiles = [np.asarray(arr[:])]
            else:
                continue
        except Exception:
            continue

        tiles = [t for t in tiles if t.shape[-1] == 768 and t.shape[0] == N_TOKENS]
        if not tiles:
            continue

        stacked = np.concatenate([t.astype(np.float64) for t in tiles], axis=0)  # (n, 768)
        finite  = np.isfinite(stacked).all(1)
        stacked = stacked[finite]
        if stacked.shape[0] < 8:
            continue

        # Register dims for THIS station+modality: the coordinates with the largest
        # variance across tokens. §27a/§35.3 established these are a SHARED direction, so
        # a per-station top-6 recovers the same handful; the parent checks that they agree
        # across stations rather than assuming it.
        var_per_dim = stacked.var(axis=0)
        reg_dims    = np.argsort(var_per_dim)[::-1][:N_REGISTER_DIMS]

        def _strip(a):
            b = a.astype(np.float64).copy()
            b[..., reg_dims] = 0.0
            return b

        stripped = _strip(stacked)

        rec = {
            # THE decisive number: spread across the 768 features inside one token.
            # This is exactly what LayerNorm divides by.
            "per_elem_std": np.std(stacked, axis=1).tolist(),
            # ...and the same with the register coordinates removed, i.e. the spread of
            # the CONTENT. LayerNorm divides content by the register-inflated sigma, which
            # is §27a.4's compression concern measured directly.
            "per_elem_std_noreg": np.std(stripped, axis=1).tolist(),
            "l2_norm":      np.linalg.norm(stacked, axis=1).tolist(),
            "abs_mean":     np.abs(stacked).mean(axis=1).tolist(),
            "sumsq_per_dim": (stacked ** 2).sum(axis=0).tolist(),
            "reg_dims":      reg_dims.tolist(),
            "n_tokens":      int(stacked.shape[0]),
            # station token only — the one training actually supervises
            "station_tok_std": float(np.std(tiles[0][STATION_TOKEN].astype(np.float64))),
            # The magnitude-vs-direction split, computed twice. If the raw split says
            # magnitude carries a large share but the register-stripped one says it does
            # not, then what LayerNorm deletes is register variation, not content — and
            # deleting it is a gain, not a loss.
            "tiles":       [_tile_stats(t) for t in tiles],
            "tiles_noreg": [_tile_stats(_strip(t)) for t in tiles],
        }
        out[mod] = rec

    return {"station": station, "category": category, "mods": out} if out else None


# ── aggregation ───────────────────────────────────────────────────────────────

def _agg(records: list[dict]) -> dict:
    result = {}
    for mod in MODALITIES:
        per_elem, per_elem_nr, l2, absm, sta_tok = [], [], [], [], []
        sumsq = np.zeros(768, dtype=np.float64)
        ntok  = 0
        tiles, tiles_nr = [], []
        dim_votes: dict[int, int] = {}
        for r in records:
            m = r["mods"].get(mod)
            if not m:
                continue
            per_elem    += m["per_elem_std"]
            per_elem_nr += m["per_elem_std_noreg"]
            l2          += m["l2_norm"]
            absm        += m["abs_mean"]
            sta_tok.append(m["station_tok_std"])
            sumsq += np.asarray(m["sumsq_per_dim"], dtype=np.float64)
            ntok  += m["n_tokens"]
            tiles    += [t for t in m["tiles"] if t]
            tiles_nr += [t for t in m["tiles_noreg"] if t]
            for d in m["reg_dims"]:
                dim_votes[int(d)] = dim_votes.get(int(d), 0) + 1
        if not per_elem:
            continue

        per_elem    = np.asarray(per_elem)
        per_elem_nr = np.asarray(per_elem_nr)
        med_std     = float(np.median(per_elem))
        reg_dims    = np.argsort(sumsq)[::-1][:N_REGISTER_DIMS]
        reg_share   = float(sumsq[reg_dims].sum() / max(sumsq.sum(), 1e-12))
        # How shared is the register direction? Fraction of stations whose own top-6
        # includes the globally top-1 dim. Near 1.0 means one direction for the corpus.
        n_sta_mod   = max(sum(1 for r in records if mod in r["mods"]), 1)
        top1_agree  = dim_votes.get(int(reg_dims[0]), 0) / n_sta_mod

        result[mod] = {
            "n_stations":        sum(1 for r in records if mod in r["mods"]),
            "n_tokens":          ntok,
            "per_elem_std_median": med_std,
            "per_elem_std_p10":  float(np.percentile(per_elem, 10)),
            "per_elem_std_p90":  float(np.percentile(per_elem, 90)),
            "l2_norm_median":    float(np.median(l2)),
            "abs_mean_median":   float(np.median(absm)),
            "station_token_std_median": float(np.median(sta_tok)) if sta_tok else None,
            # what a 0.02 annotation is worth against this token, today
            "tag_share_raw":     float(EMB_INIT_STD / max(med_std, 1e-12)),
            "tag_share_after_layernorm": float(EMB_INIT_STD),
            "layernorm_gain":    float(max(med_std, 1e-12) / 1.0),
            "per_elem_std_noreg_median": float(np.median(per_elem_nr)),
            "register_dims":     reg_dims.tolist(),
            "register_share_of_sumsq": reg_share,
            "register_top1_station_agreement": top1_agree,
            # within-tile magnitude vs direction — what LayerNorm would discard
            "within_tile_frac_magnitude": float(np.median([t["frac_mag"] for t in tiles]))
                                          if tiles else None,
            "within_tile_frac_direction": float(np.median([t["frac_dir"] for t in tiles]))
                                          if tiles else None,
            "within_tile_norm_cv":        float(np.median([t["norm_cv"] for t in tiles]))
                                          if tiles else None,
            # the same split with the register coordinates zeroed — the decisive one
            "within_tile_frac_magnitude_noreg": float(np.median([t["frac_mag"]
                                                                 for t in tiles_nr]))
                                                if tiles_nr else None,
            "within_tile_frac_direction_noreg": float(np.median([t["frac_dir"]
                                                                 for t in tiles_nr]))
                                                if tiles_nr else None,
            "n_tiles_scored":             len(tiles),
        }
    return result


def main() -> int:
    if not SPLITS_CSV.exists():
        print(f"FATAL: {SPLITS_CSV} not found", file=sys.stderr)
        return 2
    splits = pd.read_csv(SPLITS_CSV)

    # Category and directory name are taken verbatim from dataset.py's own logic
    # (SoilMoistureDataset.__init__), not re-guessed — the first version of this script
    # invented a "station" column that station_splits.csv does not have and died on the
    # first row. The store's directory names are ISMN_{network}_{station_name} for ISMN
    # and {source_network}_{station_id} for everything else.
    def _cat(r):
        sm = str(r.get("has_soil_moisture", "False")).lower() == "true"
        fl = str(r.get("has_flux",          "False")).lower() == "true"
        return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")

    def _dir_name(r):
        if str(r["source_network"]) == "ISMN":
            return f"ISMN_{r['network']}_{r['station_name']}"
        return f"{r['source_network']}_{r['station_id']}"

    jobs = []
    for _, row in splits.iterrows():
        jobs.append((_cat(row), _dir_name(row)))

    rng = np.random.default_rng(SEED)
    if len(jobs) > N_STATIONS:
        pick = rng.choice(len(jobs), size=N_STATIONS, replace=False)
        jobs = [jobs[int(i)] for i in sorted(pick)]

    print(f"scanning {len(jobs)} stations with Pool(64) ...", flush=True)
    records, failed = [], 0
    with Pool(64) as pool:
        for i, rec in enumerate(pool.imap_unordered(_scan_station, jobs, chunksize=1)):
            if rec is None:
                failed += 1
            else:
                records.append(rec)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(jobs)}  ok={len(records)} skipped={failed}", flush=True)

    if not records:
        print("FATAL: no station yielded tokens — is the zarr store on scratch?",
              file=sys.stderr)
        return 2

    agg = _agg(records)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"emb_init_std": EMB_INIT_STD,
                   "n_stations_scanned": len(records),
                   "n_stations_skipped": failed,
                   "modalities": agg}, f, indent=2)

    # ── the table that answers the question ──────────────────────────────────
    print()
    print("=" * 100)
    print(f"TOKEN SCALE — EMB_INIT_STD = {EMB_INIT_STD}   ({len(records)} stations, "
          f"{failed} skipped)")
    print("=" * 100)
    hdr = (f"{'modality':<9} {'per-elem std':>12} {'(no reg)':>9} {'L2':>7} "
           f"{'tag share':>10} {'reg share':>10} {'agree':>6} "
           f"{'tile mag':>9} {'mag-noreg':>10}")
    print(hdr)
    print("-" * len(hdr))
    def _pct(v, width):
        return f"{'n/a':>{width}}" if v is None else f"{v * 100:>{width - 1}.1f}%"

    for mod, s in agg.items():
        print(f"{mod:<9} {s['per_elem_std_median']:>12.3f} "
              f"{s['per_elem_std_noreg_median']:>9.3f} {s['l2_norm_median']:>7.1f} "
              f"{s['tag_share_raw'] * 100:>9.2f}% "
              f"{_pct(s['register_share_of_sumsq'], 10)} "
              f"{_pct(s['register_top1_station_agreement'], 6)} "
              f"{_pct(s['within_tile_frac_magnitude'], 9)} "
              f"{_pct(s['within_tile_frac_magnitude_noreg'], 10)}")
    print()
    print("HOW TO READ THIS")
    print("  tag share      what a 0.02 positional/modality code is worth against the raw")
    print("                 token.  After input LayerNorm it becomes 2.00% by construction.")
    print("                 Near 2% already -> the input LayerNorms are a no-op and")
    print("                 model.py's s2_norm/s1_norm/dem_norm/lulc_norm should come out.")
    print("                 Near 0% -> staleness is invisible without them.")
    print("  tile mag       share of WITHIN-TILE variance carried by token magnitude — the")
    print("                 part input LayerNorm deletes.  Large means it is discarding")
    print("                 signal the patchwise hypothesis (§34) exists to find.")
    print("  mag-noreg      THE DECISIVE ONE: the same share with the register coordinates")
    print("                 zeroed.  If tile-mag is large but mag-noreg is small, what")
    print("                 LayerNorm deletes is register variation, not content — and")
    print("                 deleting it is a gain.  If both are large, it is deleting real")
    print("                 within-tile content and must not be on by default.")
    print("  reg share      share of summed square in the top-6 dims (§27a registers).")
    print("  agree          fraction of stations whose own top-6 contains the global top-1")
    print("                 dim — near 100% means one shared register direction.")
    print("=" * 100)
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
