"""
Re-encode the fully-NaN S1 TerraMind token acquisitions (27 station-orbit pairs,
26 stations, 571 acquisitions -- found by job 23706813, confirmed stale by job
23708117) and overwrite them in-place in the existing zarr token store.

For each (station, orbit_key) pair:
  1. Open the token zarr (read-only) and find acquisitions whose stored l3
     tokens are fully NaN (nan_frac > 0.99).
  2. Load the corresponding raw patches from satellite_zarr.
  3. Sanitize via _nn_fill_and_sanitize(..., "S1RTC") (current code, confirmed
     NaN-free for these exact acquisitions by job 23708117).
  4. Batch through TerraMindEncoder (CPU) to get L3/L6/L9/L12.
  5. Overwrite l3/l6/l9/l12 at the bad indices in-place (zarr vindex) and
     re-consolidate metadata.

Usage:
    conda run -n terramind python fix_bad_s1_tokens.py [--execute]
"""

import argparse

import numpy as np
import torch
import zarr

from precompute_terramind import TerraMindEncoder, _nn_fill_and_sanitize

ZARR_ROOT     = "/gpfs/scratch1/shared/pkhanal/zarr/sm_only"
SAT_ZARR_ROOT = "/gpfs/work3/0/prjs1968/satellite_zarr"

BATCH_SIZE = 16
LAYERS     = ["l3", "l6", "l9", "l12"]

# (station_dir, orbit_key) pairs with fully-NaN token acquisitions
# (from logs/scan_nan_23706813.out, confirmed by job 23707948 / 23708117)
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


def _find_bad_indices(station, orbit_key):
    store = zarr.DirectoryStore(f"{ZARR_ROOT}/{station}")
    tg = zarr.open_group(store=store, mode="r")
    l3 = np.asarray(tg[f"{orbit_key}/l3"][:])
    nan_frac = np.isnan(l3).reshape(l3.shape[0], -1).mean(axis=1)
    return np.where(nan_frac > 0.99)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    jobs = []  # (station, orbit_key, idx, sanitized_patch)
    total_bad = 0
    for station, orbit_key in AFFECTED:
        bad_idx = _find_bad_indices(station, orbit_key)
        total_bad += len(bad_idx)
        print(f"{station} {orbit_key}: {len(bad_idx)} bad acquisitions")
        if not args.execute:
            continue
        rg = zarr.open_group(f"{SAT_ZARR_ROOT}/{station}.zarr", mode="r")
        for idx in bad_idx:
            raw = np.asarray(rg[f"{orbit_key}/data"][int(idx)]).astype(np.float32)
            sanitized = _nn_fill_and_sanitize(raw.copy(), "S1RTC")
            jobs.append((station, orbit_key, int(idx), sanitized))

    print(f"\nTotal bad acquisitions: {total_bad}")
    if not args.execute:
        print("Dry run -- pass --execute to re-encode and overwrite.")
        return

    print("Loading TerraMindEncoder (CPU)...")
    encoder = TerraMindEncoder(frozen=True).eval()

    results: dict = {}  # (station, orbit_key) -> list of (idx, {layer: tok})
    for start in range(0, len(jobs), BATCH_SIZE):
        chunk = jobs[start:start + BATCH_SIZE]
        batch_t = torch.from_numpy(np.stack([c[3] for c in chunk]))
        with torch.no_grad():
            feats = encoder(batch_t, "S1RTC")
        for i, (station, orbit_key, idx, _) in enumerate(chunk):
            toks = {lay: feats[lay.upper()][i].half().numpy() for lay in LAYERS}
            results.setdefault((station, orbit_key), []).append((idx, toks))
        print(f"  encoded {start + len(chunk)}/{len(jobs)}", flush=True)

    print("\nOverwriting stale tokens in-place...")
    for (station, orbit_key), items in results.items():
        store = zarr.DirectoryStore(f"{ZARR_ROOT}/{station}")
        root = zarr.open_group(store=store, mode="a")
        idxs = np.array([i for i, _ in items])
        for lay in LAYERS:
            arr = root[f"{orbit_key}/{lay}"]
            new_vals = np.stack([toks[lay] for _, toks in items]).astype(arr.dtype)
            arr.oindex[idxs] = new_vals
        zarr.consolidate_metadata(store)
        print(f"  {station}/{orbit_key}: overwrote {len(items)} acquisitions")

    print("\nVerifying...")
    n_remaining = 0
    for station, orbit_key in AFFECTED:
        store = zarr.DirectoryStore(f"{ZARR_ROOT}/{station}")
        root = zarr.open_group(store=store, mode="r")
        l3 = np.asarray(root[f"{orbit_key}/l3"][:])
        nan_frac = np.isnan(l3).reshape(l3.shape[0], -1).mean(axis=1)
        bad = np.where(nan_frac > 0.99)[0]
        n_remaining += len(bad)
        if len(bad):
            print(f"  STILL BAD: {station}/{orbit_key}: {len(bad)} acquisitions")

    print(f"\nRemaining fully-NaN acquisitions: {n_remaining}")
    print("PASS" if n_remaining == 0 else "FAIL")


if __name__ == "__main__":
    main()
