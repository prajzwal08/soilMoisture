"""
For every S1 acquisition whose stored TerraMind tokens are fully NaN (found by
the full zarr scan, job 23706813), check the RAW satellite_zarr patch against
the intended filter criteria:

  Per 16x16 token-patch (196 per 224x224 image):
    nodata_frac < 1%   -> nearest-neighbour fillable
    nodata_frac >= 1%  -> "masked" token

  Per image:
    masked_token_frac > 50%  -> should be DISCARDED entirely
    masked_token_frac <= 50% -> should be KEPT, with a per-token mask for training

This tells us, for each bad acquisition, whether the NaN tokens arose from an
image that should have been discarded (>50% masked) or one that should have
been kept-but-masked (<=50%) -- i.e. whether the bug is in the
discard/keep decision or in the sanitize-before-encode step.

Usage:
    conda run -n terramind python check_bad_s1_filter_criteria.py
"""

from multiprocessing import Pool

import numpy as np
import zarr

ZARR_ROOT     = "/gpfs/scratch1/shared/pkhanal/zarr/sm_only"
SAT_ZARR_ROOT = "/gpfs/work3/0/prjs1968/satellite_zarr"

TOKEN_SIZE = 16
N_SIDE     = 14
PATCH_NAN_THRESH = 0.01
IMAGE_MASKED_THRESH = 0.50

# (station_dir, orbit_key) pairs known to have fully-NaN token acquisitions
# (from logs/scan_nan_23706813.out)
AFFECTED = [
    ("ISMN_Berlin_PSA4Heerstrasse", "s1_desc"),
    ("ISMN_COSMOS-UK_BickleyHall", "s1_desc"),
    ("ISMN_COSMOS-UK_HartwoodHome", "s1_desc"),
    ("ISMN_COSMOS-UK_Sheepdrove", "s1_desc"),
    ("ISMN_FMI_SOD012", "s1_desc"),
    ("ISMN_FMI_SOD130", "s1_asc"),
    ("ISMN_FMI_SOD130", "s1_desc"),
    ("ISMN_REMEDHUS_LasVacas", "s1_desc"),
    ("ISMN_ROMPS_GolubinGlacier", "s1_desc"),
    ("ISMN_RSMN_Alexandria", "s1_desc"),
    ("ISMN_SCAN_LyeBrook", "s1_asc"),
    ("ISMN_SCAN_MonoclineRidge", "s1_desc"),
    ("ISMN_SCAN_WillowWells", "s1_asc"),
    ("ISMN_SMOSMANIA_Mouthoumet", "s1_desc"),
    ("ISMN_SNOTEL_AtlantaSummit", "s1_desc"),
    ("ISMN_SNOTEL_BearCreek", "s1_desc"),
    ("ISMN_SNOTEL_BlackFlat-U.M.Ck", "s1_asc"),
    ("ISMN_SNOTEL_ForestdaleCreek", "s1_desc"),
    ("ISMN_SNOTEL_Hopewell", "s1_desc"),
    ("ISMN_SNOTEL_HuntingtonHorse", "s1_asc"),
    ("ISMN_SNOTEL_JackwhackerGulch", "s1_asc"),
    ("ISMN_SNOTEL_LostCreekResv", "s1_asc"),
    ("ISMN_SNOTEL_MonitorPass", "s1_desc"),
    ("ISMN_SNOTEL_PalisadesTahoe", "s1_asc"),
    ("ISMN_SNOTEL_PorphyryCreek", "s1_desc"),
    ("ISMN_SNOTEL_RainyPass", "s1_asc"),
    ("ISMN_SNOTEL_CrowCreek", "s1_asc"),
]


def _patch_masked_frac(arr: np.ndarray) -> tuple[float, int]:
    """arr: (2,224,224). Returns (overall_nodata_frac, n_masked_tokens of 196)."""
    nodata_2d = np.isnan(arr).any(axis=0)
    overall = float(nodata_2d.mean())
    n_masked = 0
    for i in range(N_SIDE):
        for j in range(N_SIDE):
            rs, re = i * TOKEN_SIZE, (i + 1) * TOKEN_SIZE
            cs, ce = j * TOKEN_SIZE, (j + 1) * TOKEN_SIZE
            frac = nodata_2d[rs:re, cs:ce].mean()
            if frac >= PATCH_NAN_THRESH:
                n_masked += 1
    return overall, n_masked


def _process(pair):
    station, orbit_key = pair
    out_lines = []
    try:
        tg = zarr.open_consolidated(f"{ZARR_ROOT}/{station}", mode="r")
    except Exception:
        tg = zarr.open_group(f"{ZARR_ROOT}/{station}", mode="r")
    rg = zarr.open_group(f"{SAT_ZARR_ROOT}/{station}.zarr", mode="r")

    t_dates = [str(d) for d in tg[f"{orbit_key}/dates"][:]]
    r_dates = [d.decode() if isinstance(d, bytes) else str(d) for d in rg[f"{orbit_key}/dates"][:]]
    assert t_dates == r_dates, f"{station} {orbit_key}: date mismatch"

    l3 = np.asarray(tg[f"{orbit_key}/l3"][:])
    nan_per_acq = np.isnan(l3).reshape(l3.shape[0], -1).mean(axis=1)
    bad_idx = np.where(nan_per_acq > 0.99)[0]

    raw = np.asarray(rg[f"{orbit_key}/data"][:])  # (N,2,224,224)
    for i in bad_idx:
        overall, n_masked = _patch_masked_frac(raw[i].astype(np.float32))
        masked_frac = n_masked / (N_SIDE * N_SIDE)
        verdict = "DISCARD(>50%)" if masked_frac > IMAGE_MASKED_THRESH else "KEEP+MASK(<=50%)"
        out_lines.append(
            f"{station} {orbit_key} {t_dates[i]}: overall_nodata={overall:.3%} "
            f"masked_tokens={n_masked}/196 ({masked_frac:.1%}) -> {verdict}"
        )
    return out_lines


def main():
    with Pool(64) as pool:
        results = pool.map(_process, AFFECTED)

    all_lines = [line for lines in results for line in lines]
    n_discard = sum(1 for l in all_lines if "DISCARD" in l)
    n_keep    = sum(1 for l in all_lines if "KEEP" in l)

    print(f"Total bad acquisitions checked: {len(all_lines)}")
    print(f"  -> should DISCARD (>50% masked):  {n_discard}")
    print(f"  -> should KEEP+MASK (<=50%):       {n_keep}")
    print()
    for line in all_lines:
        print(line)


if __name__ == "__main__":
    main()
