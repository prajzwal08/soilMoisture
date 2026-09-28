"""
§47 — the single source of truth for split geometry, split categories and the temporal cut.

Before this module the cut date lived in two unlinked places (`create_evaluation_splits.py:27`
and `train.py:280`'s year list) and the sm_only/sm_and_flux/flux_only derivation lived in five.
Moving the cut in one file silently made OOT either contaminated or empty, with no error —
§44.6. Everything that needs to know where train stops and the holdouts begin imports it here.

Nothing in this module reads a file or has side effects; it is constants plus three pure
helpers, so it is safe to import from a SLURM worker, a notebook or the login node.
"""

from __future__ import annotations

# ── temporal ────────────────────────────────────────────────────────────────────
TRAIN_YEARS       = list(range(2016, 2023))   # 2016-2022 inclusive
OOT_YEARS         = [2023, 2024, 2025]        # §47: inputs reach 2025, so the holdout does
OOT_CUT_DATE      = 20230101                  # YYYYMMDD; first day NOT seen in training

# A station needs a real record on BOTH sides of the cut to mean what its split says.
MIN_PRE_CUT_DAYS  = 365   # else it is not a "seen" station — demote to oos (§47.3 rule 3)
MIN_POST_CUT_DAYS = 365   # else its OOT metric is a seasonal fragment (§47.5)

# ── tile geometry ───────────────────────────────────────────────────────────────
# Every station gets its OWN tile centred on itself (download_s2_mpc.py:57-58,
# download_s1_lulc_mpc.py:82-83), so "B shares A's tile" is exactly d < HALF_TILE_M.
# §39.3's 2.24 km is *tile overlap* — a looser, different relation. Do not substitute it.
TILE_M       = 2240.0
PATCH_M      = 160.0      # one TerraMind token: 16 px x 10 m (precompute_terramind.py:32)
HALF_TILE_M  = TILE_M / 2.0
DUP_M        = 50.0       # closer than this across networks = one physical site, ingested twice

# ── holdout populations ─────────────────────────────────────────────────────────
# TWENTE is the whole of the Dutch inventory: RAAM was never downloaded and ICOS NL-Loo /
# NL-Hor are absent (§44.5). The product is a 10 m map of the Netherlands, so no Dutch
# station may train.
NL_NETWORKS = frozenset({"TWENTE"})

# flux_only level-1 files carry no `soil_moisture` variable at all (§47.7), so they cannot
# supply a target however the filter is written.
SM_CATEGORIES  = ["sm_only", "sm_and_flux"]
ALL_CATEGORIES = ["sm_only", "sm_and_flux", "flux_only"]

# ── split assignment (create_evaluation_splits.py) ──────────────────────────────
RANDOM_SEED        = 42
COLOC_THRESHOLD_KM = 3.0
OOS_FRACTION       = 0.20
VAL_FRACTION       = 0.10
ABLATION_FRACTION  = 0.20
MIN_CELL_SIZE      = 3
OOT_MIN_PRE_YEARS  = 1
ELEV_IMBALANCE_TOL = 0.10
FLUX_SM_OOS_TARGET = 15

# ── invariants, checked at import ───────────────────────────────────────────────
assert not (set(TRAIN_YEARS) & set(OOT_YEARS)), "train years and OOT years overlap"
assert max(TRAIN_YEARS) < OOT_CUT_DATE // 10000, "TRAIN_YEARS runs past OOT_CUT_DATE"
assert min(OOT_YEARS) == OOT_CUT_DATE // 10000, "OOT_YEARS does not start at the cut"


def _truthy(v) -> bool:
    """CSV booleans arrive as 'True'/'False' strings or real bools depending on the reader."""
    return str(v).strip().lower() == "true"


def category_of(row) -> str:
    """sm_only | sm_and_flux | flux_only, from the has_soil_moisture / has_flux columns.

    A station with BOTH is `sm_and_flux`, which is why `category_filter=["sm_only"]` used to
    discard 48 stations that carry perfectly good soil moisture (§47.1 item 1).
    """
    sm = _truthy(row.get("has_soil_moisture", False))
    fl = _truthy(row.get("has_flux", False))
    return "sm_and_flux" if (sm and fl) else ("sm_only" if sm else "flux_only")


def station_dir_name(row) -> str:
    """The on-disk directory / level-1 stem for a station_splits.csv row.

    ISMN keys on `station_name`, everything else on `station_id` — the convention baked into
    the zarr store, `csvs/station_duration_audit.csv:file` and `csvs/colocated_pairs.csv`.
    """
    if str(row["source_network"]) == "ISMN":
        return f"ISMN_{row['network']}_{row['station_name']}"
    return f"{row['source_network']}_{row['station_id']}"


def station_key(row) -> str:
    """Globally unique station key. `station_id` alone is NOT unique across source networks,
    which is the latent defect in §35.29's `same_patch_pair` / `tile_pair_eval` columns."""
    return f"{row['source_network']}|{row['network']}|{row['station_id']}"
