"""
Stratified Environmental Split (SES) framework.

Creates OOS / OOT / OOST evaluation splits for the station inventory,
respecting co-location (same S2 patch) and environmental stratification.

Output: csvs/station_splits.csv

Run with:
  /home/khanalp/miniforge3/envs/soilmoisture/bin/python create_evaluation_splits.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

INVENTORY_CSV = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_inventory.csv")
OUT_CSV       = Path("/home/khanalp/code/PhD/soilMoisture/csvs/station_splits.csv")

RANDOM_SEED        = 42
COLOC_THRESHOLD_KM = 3.0
OOS_FRACTION       = 0.20
VAL_FRACTION       = 0.10
ABLATION_FRACTION  = 0.20   # ~20% of train → ablation_train; ~20% of OOS → ablation_oos
MIN_CELL_SIZE      = 3        # cells with fewer groups → all go to train
OOT_CUT_DATE      = 20230101  # YYYYMMDD integer
OOT_MIN_PRE_YEARS  = 1        # ≥ 1 year of pre-2023 data required for OOT
ELEV_IMBALANCE_TOL = 0.10     # 10 pp tolerance before elevation swap

FLUX_SM_OOS_TARGET = 15       # ~15 flux+SM sites forced into OOS (~30% of 49)

# IGBP macro-group mapping
IGBP_MACRO = {
    "ENF": "Forest",  "EBF": "Forest",  "DNF": "Forest",
    "DBF": "Forest",  "MF":  "Forest",
    "SAV": "Shrub-Savanna", "OSH": "Shrub-Savanna",
    "CSH": "Shrub-Savanna", "WSA": "Shrub-Savanna",
    "GRA": "Grass-Crop", "CRO": "Grass-Crop",
}  # anything else → "Other"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)


# ---------------------------------------------------------------------------
# Step 1: Build location groups
# ---------------------------------------------------------------------------
def build_location_groups(df: pd.DataFrame) -> pd.Series:
    """Return Series of location_group_id (int) indexed like df."""
    lats = df["latitude"].values
    lons = df["longitude"].values
    n = len(df)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d < COLOC_THRESHOLD_KM:
                uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    # Re-label roots as compact integers
    unique_roots = {r: gid for gid, r in enumerate(sorted(set(roots)))}
    group_ids = [unique_roots[r] for r in roots]
    return pd.Series(group_ids, index=df.index, name="location_group_id")


# ---------------------------------------------------------------------------
# Step 2: Reserve flux+SM joint-eval sites
# ---------------------------------------------------------------------------
def reserve_flux_sm_oos(df: pd.DataFrame, rng: np.random.Generator) -> set:
    """Return set of indices (from df) to force into OOS as joint-eval sites."""
    flux_sm = df[df["has_soil_moisture"] & df["has_flux"]].copy()
    flux_sm["igbp_macro"] = flux_sm["IGBP"].map(IGBP_MACRO).fillna("Other")

    selected_idx = []
    # Proportional stratified sample across IGBP macro-groups
    counts = flux_sm["igbp_macro"].value_counts()
    total  = len(flux_sm)
    per_macro = {}
    remaining = FLUX_SM_OOS_TARGET
    macros = counts.index.tolist()

    for macro in macros:
        quota = max(1, round(FLUX_SM_OOS_TARGET * counts[macro] / total))
        per_macro[macro] = quota

    # Adjust to hit target
    while sum(per_macro.values()) > FLUX_SM_OOS_TARGET:
        biggest = max(per_macro, key=per_macro.get)
        per_macro[biggest] -= 1
    while sum(per_macro.values()) < FLUX_SM_OOS_TARGET:
        # Add to largest macro
        biggest = max(per_macro, key=lambda k: counts[k] - per_macro[k])
        per_macro[biggest] += 1

    for macro, quota in per_macro.items():
        pool = flux_sm[flux_sm["igbp_macro"] == macro].index.tolist()
        quota = min(quota, len(pool))
        chosen = rng.choice(pool, size=quota, replace=False)
        selected_idx.extend(chosen.tolist())

    print(f"Flux+SM joint-eval sites reserved: {len(selected_idx)}")
    for idx in selected_idx:
        r = df.loc[idx]
        print(f"  {r['source_network']:12s} {r['station_id']:25s}  IGBP={r['IGBP']}")
    return set(selected_idx)


# ---------------------------------------------------------------------------
# Step 3: Temporal eligibility
# ---------------------------------------------------------------------------
def classify_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_date"] = df["start_date"].astype(int)
    df["end_date"]   = df["end_date"].astype(int)
    # OOT requires:
    #   - end_date ≥ 2023 (has test data in OOT window)
    #   - start_date ≤ 2021-12-31 (≥1 yr pre-2023 for context window)
    #   - n_years ≥ 3 (consistent with ICOS/AmeriFlux MIN_VALID_DAYS=1095)
    df["oot_candidate"] = (
        (df["end_date"] >= OOT_CUT_DATE) &
        (df["start_date"] <= int(f"{2023 - OOT_MIN_PRE_YEARS}1231")) &
        (df["n_years"] >= 3)
    )
    return df


# ---------------------------------------------------------------------------
# Step 4 + 5: Environmental strata and OOS selection
# ---------------------------------------------------------------------------
def select_oos(df: pd.DataFrame, reserved_oos: set, rng: np.random.Generator) -> set:
    """Return set of indices to assign to OOS (excluding reserved_oos already decided)."""
    df = df.copy()
    df["kg_macro"]   = df["koppen_geiger"].str[0].fillna("?")
    df["igbp_macro"] = df["IGBP"].map(IGBP_MACRO).fillna("Other")

    # Work at location-group level
    # Build group representative: use the row with lowest index per group
    group_rep = (
        df.groupby("location_group_id")
        .apply(lambda g: g.index[0])
        .rename("rep_idx")
    )
    group_df = df.loc[group_rep.values].copy()
    group_df["group_id"] = group_rep.values  # group representative index

    # Exclude groups that already have a reserved flux+SM site
    reserved_groups = set(df.loc[list(reserved_oos), "location_group_id"])
    eligible = group_df[~group_df["location_group_id"].isin(reserved_groups)]

    oos_indices = set()

    for (kg, igbp), cell in eligible.groupby(["kg_macro", "igbp_macro"]):
        n_groups = len(cell)
        if n_groups < MIN_CELL_SIZE:
            continue  # too few → all to train
        n_oos = max(1, round(n_groups * OOS_FRACTION))
        chosen_reps = rng.choice(cell.index.tolist(), size=n_oos, replace=False)
        # Map representative indices back to all stations in those groups
        chosen_group_ids = df.loc[chosen_reps, "location_group_id"].values
        for gid in chosen_group_ids:
            oos_indices.update(df[df["location_group_id"] == gid].index.tolist())

    return oos_indices


# ---------------------------------------------------------------------------
# Step 5b: Validation set selection (from remaining training stations)
# ---------------------------------------------------------------------------
def select_val(df: pd.DataFrame, oos_idx: set, rng: np.random.Generator) -> set:
    """Sample ~10% of training groups per KG×IGBP cell as validation set."""
    df = df.copy()
    df["kg_macro"]   = df["koppen_geiger"].str[0].fillna("?")
    df["igbp_macro"] = df["IGBP"].map(IGBP_MACRO).fillna("Other")

    train_df = df[~df.index.isin(oos_idx)]

    group_rep = (
        train_df.groupby("location_group_id")
        .apply(lambda g: g.index[0])
        .rename("rep_idx")
    )
    group_df = train_df.loc[group_rep.values].copy()

    val_indices = set()
    for (kg, igbp), cell in group_df.groupby(["kg_macro", "igbp_macro"]):
        n_groups = len(cell)
        if n_groups < MIN_CELL_SIZE:
            continue
        n_val = max(1, round(n_groups * VAL_FRACTION))
        chosen_reps = rng.choice(cell.index.tolist(), size=n_val, replace=False)
        chosen_gids = train_df.loc[chosen_reps, "location_group_id"].values
        for gid in chosen_gids:
            val_indices.update(train_df[train_df["location_group_id"] == gid].index.tolist())

    return val_indices


# ---------------------------------------------------------------------------
# Step 5c: Ablation subset selection (miniature representative dataset)
# ---------------------------------------------------------------------------
def select_ablation(df: pd.DataFrame, rng: np.random.Generator) -> tuple[set, set]:
    """Return (ablation_train_idx, ablation_oos_idx) — ~20% of each, stratified."""
    df = df.copy()
    df["kg_macro"]   = df["koppen_geiger"].str[0].fillna("?")
    df["igbp_macro"] = df["IGBP"].map(IGBP_MACRO).fillna("Other")

    abl_train, abl_oos = set(), set()

    for split_name, split_mask, target_set in [
        ("train", df["split"] == "train", abl_train),
        ("oos",   df["split"] == "oos",   abl_oos),
    ]:
        split_df = df[split_mask]
        group_rep = (
            split_df.groupby("location_group_id")
            .apply(lambda g: g.index[0])
            .rename("rep_idx")
        )
        group_df = split_df.loc[group_rep.values].copy()

        for (kg, igbp), cell in group_df.groupby(["kg_macro", "igbp_macro"]):
            n_groups = len(cell)
            if n_groups < MIN_CELL_SIZE:
                # too few — include all in ablation for this cell
                chosen_reps = cell.index.tolist()
            else:
                n_abl = max(1, round(n_groups * ABLATION_FRACTION))
                chosen_reps = rng.choice(cell.index.tolist(), size=n_abl, replace=False).tolist()
            for rep in chosen_reps:
                gid = split_df.loc[rep, "location_group_id"]
                target_set.update(split_df[split_df["location_group_id"] == gid].index.tolist())

    return abl_train, abl_oos


# ---------------------------------------------------------------------------
# Step 6: Elevation balance check
# ---------------------------------------------------------------------------
def elevation_balance_check(df: pd.DataFrame, oos_idx: set) -> set:
    """Swap whole location groups to keep OOS elevation distribution balanced."""
    def band(e):
        if e < 500:   return "Low"
        if e < 1500:  return "Mid"
        return "High"

    df = df.copy()
    df["elev_band"] = df["elevation_m"].apply(band)
    df["in_oos"]    = df.index.isin(oos_idx)
    oos_idx = set(oos_idx)

    # Representative row per group (for swap candidate selection)
    group_rep = df.groupby("location_group_id").apply(lambda g: g.index[0])

    total     = len(df)
    total_oos = len(oos_idx)

    for b in ["Low", "Mid", "High"]:
        n_band     = (df["elev_band"] == b).sum()
        n_oos_band = ((df["elev_band"] == b) & df["in_oos"]).sum()
        if n_band == 0:
            continue
        frac_overall = n_band / total
        frac_oos     = n_oos_band / max(total_oos, 1)
        gap = frac_overall - frac_oos

        if gap > ELEV_IMBALANCE_TOL:
            # Under-represented — move a train group of this band into OOS
            cand_groups = (
                df[(df["elev_band"] == b) & (~df["in_oos"])]
                ["location_group_id"].unique()
            )
            if len(cand_groups):
                gid = cand_groups[0]
                members = df[df["location_group_id"] == gid].index.tolist()
                oos_idx.update(members)
                print(f"  Elevation swap +OOS: group {gid} "
                      f"({df.loc[members[0],'station_id']}, band={b}, gap={gap:.2f})")

        elif gap < -ELEV_IMBALANCE_TOL:
            # Over-represented — move an OOS group of this band to train
            cand_groups = (
                df[(df["elev_band"] == b) & df["in_oos"]]
                ["location_group_id"].unique()
            )
            if len(cand_groups):
                gid = cand_groups[0]
                members = df[df["location_group_id"] == gid].index.tolist()
                for m in members:
                    oos_idx.discard(m)
                print(f"  Elevation swap -OOS: group {gid} "
                      f"({df.loc[members[0],'station_id']}, band={b}, gap={gap:.2f})")

    return oos_idx


# ---------------------------------------------------------------------------
# Step 7: Assign final labels and write
# ---------------------------------------------------------------------------
def assign_labels(
    df: pd.DataFrame, oos_idx: set, val_idx: set, joint_eval_idx: set,
    ablation_train_idx: set, ablation_oos_idx: set,
) -> pd.DataFrame:
    df = df.copy()

    df["split"] = "train"
    df.loc[df.index.isin(oos_idx), "split"] = "oos"
    df.loc[df.index.isin(val_idx),  "split"] = "val"

    df["oot_eligible"]    = (df["split"] == "train") & df["oot_candidate"]
    df["oost_eligible"]   = (df["split"] == "oos") & (df["end_date"] >= OOT_CUT_DATE)
    df["joint_eval"]      = df.index.isin(joint_eval_idx)
    # flux-only OOS sites: LE-only evaluation at unseen locations (no SM co-supervision)
    df["flux_only_eval"]  = (
        (df["split"] == "oos") &
        df["has_flux"] &
        ~df["has_soil_moisture"]
    )
    # ablation subset — miniature representative version of train and OOS
    df["ablation_train"]  = df.index.isin(ablation_train_idx)
    df["ablation_oos"]    = df.index.isin(ablation_oos_idx)

    df["kg_macro"]      = df["koppen_geiger"].str[0].fillna("?")
    df["igbp_macro"]    = df["IGBP"].map(IGBP_MACRO).fillna("Other")

    def elev_band(e):
        if e < 500:   return "Low"
        if e < 1500:  return "Mid"
        return "High"
    df["elevation_band"] = df["elevation_m"].apply(elev_band)

    return df.drop(columns=["oot_candidate"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(INVENTORY_CSV)
    print(f"Loaded {len(df)} stations from {INVENTORY_CSV.name}")

    # Step 1
    print("\n--- Step 1: Building location groups ---")
    df["location_group_id"] = build_location_groups(df)
    n_groups  = df["location_group_id"].nunique()
    n_coloc   = (df.groupby("location_group_id").size() > 1).sum()
    print(f"  {n_groups} location groups ({n_coloc} groups with ≥2 co-located stations)")

    # Cross-network co-locations
    cross = []
    for gid, g in df.groupby("location_group_id"):
        if g["source_network"].nunique() > 1:
            cross.append(gid)
    print(f"  Cross-network co-located groups: {len(cross)}")
    for gid in cross[:8]:
        g = df[df["location_group_id"] == gid]
        pairs = ", ".join(f"{r.source_network}/{r.station_id}" for _, r in g.iterrows())
        print(f"    group {gid}: {pairs}")

    # Step 2
    print("\n--- Step 2: Reserving flux+SM joint-eval sites ---")
    joint_eval_idx = reserve_flux_sm_oos(df, rng)

    # Step 3
    print("\n--- Step 3: Classifying temporal eligibility ---")
    df = classify_temporal(df)
    print(f"  OOT-candidate stations (train pool): {df['oot_candidate'].sum()}")

    # Expand joint-eval indices to full location groups (group integrity)
    joint_eval_groups = set(df.loc[list(joint_eval_idx), "location_group_id"])
    joint_eval_full   = set(df[df["location_group_id"].isin(joint_eval_groups)].index.tolist())

    # Step 4 + 5
    print("\n--- Steps 4-5: Environmental stratification + OOS selection ---")
    oos_idx = select_oos(df, joint_eval_full, rng)
    oos_idx |= joint_eval_full  # add reserved flux+SM sites (full groups)
    print(f"  OOS stations selected: {len(oos_idx)}")

    # Step 5b
    print("\n--- Step 5b: Validation set selection ---")
    val_idx = select_val(df, oos_idx, rng)
    print(f"  Validation stations selected: {len(val_idx)}")

    # Step 6
    print("\n--- Step 6: Elevation balance check ---")
    oos_idx = elevation_balance_check(df, oos_idx)

    # Step 7
    print("\n--- Step 7: Assigning final labels ---")
    # Placeholder empty sets — ablation added after split column exists
    df = assign_labels(df, oos_idx, val_idx, joint_eval_idx, set(), set())

    # Verify no group is split
    for gid, g in df.groupby("location_group_id"):
        assert g["split"].nunique() == 1, f"Group {gid} split across train/oos/val!"
    print("  Group integrity check passed.")

    # Step 5c — ablation subset (needs split column to exist first)
    print("\n--- Step 5c: Ablation subset selection ---")
    ablation_train_idx, ablation_oos_idx = select_ablation(df, rng)
    df["ablation_train"] = df.index.isin(ablation_train_idx)
    df["ablation_oos"]   = df.index.isin(ablation_oos_idx)
    print(f"  ablation_train: {df['ablation_train'].sum()}  (~{100*df['ablation_train'].sum()/(df['split']=='train').sum():.0f}% of train)")
    print(f"  ablation_oos:   {df['ablation_oos'].sum()}  (~{100*df['ablation_oos'].sum()/(df['split']=='oos').sum():.0f}% of OOS)")

    # Summary
    print("\n=== SPLIT SUMMARY ===")
    print(f"  train:          {(df['split']=='train').sum():5d}")
    print(f"  val:            {(df['split']=='val').sum():5d}")
    print(f"  oot_eligible:   {df['oot_eligible'].sum():5d}  (train, 2023+ data, ≥1yr pre-2023, n_years≥3)")
    print(f"  oos:            {(df['split']=='oos').sum():5d}")
    print(f"  oost_eligible:  {df['oost_eligible'].sum():5d}  (oos, 2023+ data)")
    print(f"  joint_eval:     {df['joint_eval'].sum():5d}  (oos flux+SM  — SM+LE evaluation)")
    print(f"  flux_only_eval: {df['flux_only_eval'].sum():5d}  (oos flux-only — LE-only evaluation)")
    print(f"\n  ablation_train: {df['ablation_train'].sum():5d}  (~20% of train, stratified)")
    print(f"  ablation_oos:   {df['ablation_oos'].sum():5d}  (~20% of OOS, stratified)")
    abl_flux_sm = (df['ablation_train'] & df['has_flux'] & df['has_soil_moisture']).sum()
    abl_flux_only = (df['ablation_train'] & df['has_flux'] & ~df['has_soil_moisture']).sum()
    print(f"  ablation flux+SM in train:   {abl_flux_sm}")
    print(f"  ablation flux-only in train: {abl_flux_only}")

    print("\n--- OOS by network ---")
    print(df[df["split"]=="oos"]["source_network"].value_counts().to_string())

    print("\n--- OOS by KG macro-zone ---")
    print(df[df["split"]=="oos"]["kg_macro"].value_counts().sort_index().to_string())

    print("\n--- OOS by IGBP macro-group ---")
    print(df[df["split"]=="oos"]["igbp_macro"].value_counts().to_string())

    print("\n--- Elevation distribution: OOS vs train ---")
    for band in ["Low", "Mid", "High"]:
        n_tr  = ((df["split"]=="train") & (df["elevation_band"]==band)).sum()
        n_oos = ((df["split"]=="oos")   & (df["elevation_band"]==band)).sum()
        frac_tr  = n_tr  / (df["split"]=="train").sum()
        frac_oos = n_oos / (df["split"]=="oos").sum()
        print(f"  {band:4s}:  train {frac_tr:.1%}  oos {frac_oos:.1%}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
