"""§24 -- modality shuffling: is the satellite branch doing anything?

Keeps the trained checkpoint, the temporal transformer and every non-ablated input
exactly as they are, and swaps ONE modality between samples.  Whatever skill survives
was never coming from that modality.

Why shuffle and not zero: zeroing moves the input off the training distribution, so a
collapse would show the model dislikes zeros rather than that it uses the modality.
Shuffling holds every marginal distribution fixed -- same statistics, magnitudes,
sparsity -- and destroys only the CORRESPONDENCE between the modality and the station
being predicted.  That is what makes a null result interpretable.

Two donor rules (runbook §24.1):
    cross_station   different site, same season (±season_window days)
                    -> destroys station identity, keeps seasonality
    within_station  same site, different season (>min_doy_gap days apart)
                    -> destroys temporal state, keeps station identity

Used by eval_predict.py via --ablate / --ablate-mode / --seed.
"""
import numpy as np
import torch

# Keys as returned by SoilMoistureDataset.__getitem__ (dataset.py:1083-1125).
# A modality MOVES AS A WHOLE: permuting s2_pyr without s2_rel_pos hands the model
# tokens whose declared time offsets belong to a different acquisition, which tests
# incoherence rather than absence.
# --arch patchwise emits DIFFERENT keys and pops the pooled ones (dataset.py:1247-1254), so the
# pooled names below are absent there. Both sets are listed and the swap asserts that at least
# one key was actually present — see AblationDataset.__getitem__.
MODALITY_KEYS = {
    "s2":   ["s2_hist", "s2_hist_valid", "s2_doys", "s2_valid", "s2_rel_pos"],
    "s1":   ["s1_hist", "s1_hist_valid", "s1_doys", "s1_valid", "s1_rel_pos"],
    "dem":  ["dem_tok"],
    "lulc": ["lulc_tok"],
    # positive control: we are confident ERA5 forcing matters. If shuffling THIS changes
    # nothing, the harness never reached the model -- stop and debug (§24.5).
    "era5": ["era5", "era5_doys"],
}
MODALITY_KEYS["sat"] = (MODALITY_KEYS["s2"] + MODALITY_KEYS["s1"]
                        + MODALITY_KEYS["dem"] + MODALITY_KEYS["lulc"])

# ── The FROZEN U-NET ARM uses different key names ────────────────────────────────────────
# The map above was rewritten for the patchwise arm and its comment claiming "both sets are
# listed" is stale: dataset_unet.py emits s2_pyr/dem_pyr/lulc_pyr (:1086, :1098, :1099), never
# s2_hist/dem_tok/lulc_tok. Running the patchwise map against the U-Net arm fails two ways, and
# only one of them is loud:
#     dem, lulc  -> no key matches       -> KeyError below.                   SAFE.
#     s2, s1, sat-> s2_doys/s2_valid/s2_rel_pos exist on BOTH arms, so n_swapped > 0 and the
#                   guard passes while s2_pyr -- the actual tokens -- never moves. The model
#                   then gets a donor's timestamps stapled to its own imagery: INCOHERENCE,
#                   not absence, which §24.2 says makes a shuffle result uninterpretable.
# That silent case is why this map exists and why the U-Net arm is checked strictly (every
# listed key must be present, not merely one).
MODALITY_KEYS_UNET = {
    "s2":     ["s2_pyr", "s2_doys", "s2_valid", "s2_rel_pos"],
    "s1":     ["s1_pyr", "s1_doys", "s1_valid", "s1_rel_pos"],
    "dem":    ["dem_pyr"],
    "lulc":   ["lulc_pyr"],
    "era5":   ["era5", "era5_doys"],
    # Never tested before §24.13. Both are fed RAW (unnormalised) in this arm -- there is no
    # _sif_mean/_twsa_mean anywhere in dataset_unet.py; the z-scoring at dataset.py:1030-1033
    # was added later, for the patchwise arm.
    "sif":    ["sif",  "sif_doys",  "sif_rel_pos",  "sif_valid"],
    "twsa":   ["twsa", "twsa_doys", "twsa_rel_pos", "twsa_valid"],
    # Only definable on this arm -- the patchwise trunk has no decoder skips. This is the
    # `--ablate anchor` §24.11 caveat 1 asked for.
    "anchor": ["anchor_l3", "anchor_l6", "anchor_l9", "anchor_l12",
               "anchor_rel_pos", "anchor_orbit"],
    "soil":   ["soil_patch"],
}
MODALITY_KEYS_UNET["sat"] = (MODALITY_KEYS_UNET["s2"] + MODALITY_KEYS_UNET["s1"]
                             + MODALITY_KEYS_UNET["dem"] + MODALITY_KEYS_UNET["lulc"]
                             + MODALITY_KEYS_UNET["anchor"])

# ── The §48 / §59 (s48) model: the final baseline_selected_20261005 ────────────────────────
# Keys from dataset.py SoilMoistureDataset.__getitem__ (the return dict). A tuple
# (key, start, stop) swaps only those CHANNELS of a stacked tensor: the 20 m `fine` patch
# holds S2 (0:12) | S1 (12:17) | DEM (17:19) (model.py FINE_S2/FINE_S1/FINE_DEM, raw-band
# fine inputs), so "imagery" and "DEM" can be ablated separately. Checked STRICTLY: every
# listed key must exist (a partial swap is incoherence, not absence — see below).
_S2_160   = ["s2_pyr", "s2_doys", "s2_valid", "s2_rel_pos"]
_S1_160   = ["s1_pyr", "s1_doys", "s1_valid", "s1_rel_pos", "s1_orbit"]
_ANCHOR   = ["anchor_l12", "anchor_rel_pos", "anchor_orbit", "anchor_found"]
_FINE_IMG = [("fine", 0, 17)]                       # 20 m S2 + S1 channels, DEM channel kept
MODALITY_KEYS_S48 = {
    "era5":   ["era5", "era5_doys", "era5_rel_pos"],
    "s2":     _S2_160,                              # 160 m S2 history (anchor untouched)
    "s1":     _S1_160,                              # 160 m S1 history (anchor untouched)
    "sat160": _S2_160 + _S1_160 + _ANCHOR,          # every 160 m satellite token
    "fine":   _FINE_IMG,                            # 20 m imagery
    "sat":    _S2_160 + _S1_160 + _ANCHOR + _FINE_IMG,   # all satellite, 160 m + 20 m
    "dem":    ["dem_pyr", "dem_valid", ("fine", 17, 19)],
    "lulc":   ["lulc_pyr", "lulc_valid", "lulc"],   # 160 m tokens + the 10 m class map
    "soil":   ["soil_patch", "soil_valid"],
    "sif":    ["sif", "sif_doys", "sif_rel_pos", "sif_valid"],
    "twsa":   ["twsa", "twsa_doys", "twsa_rel_pos", "twsa_valid"],
}

KEY_MAPS = {"patchwise": MODALITY_KEYS, "unet": MODALITY_KEYS_UNET, "s48": MODALITY_KEYS_S48}
_STRICT_ARMS = {"unet", "s48"}

# Every modality must match at least one key in every sample. Silence means a stale key list,
# which is a SILENT NO-OP ablation -- see AblationDataset.__getitem__.
_OPTIONAL_MODALITIES: set[str] = set()

MODALITIES = sorted(set(MODALITY_KEYS) | set(MODALITY_KEYS_UNET) | set(MODALITY_KEYS_S48))


def build_donor_map(samples, mode: str, seed: int = 0, season_window: int = 15,
                    min_doy_gap: int = 60, max_tries: int = 40):
    """-> (donor_idx array, stats dict).

    Built from sample METADATA only (station_key, doy) -- no token reads, so this is
    milliseconds even for 800k samples.  Rejection sampling keeps it O(1) per sample
    instead of materialising a candidate list per row.

    Samples with no valid donor fall back to themselves; the count is REPORTED, never
    silent, because a large fallback fraction would quietly weaken the ablation.
    """
    rng = np.random.default_rng(seed)
    n = len(samples)
    doys = np.fromiter((s["doy"] for s in samples), dtype=np.int32, count=n)
    stations = np.array([s["station_key"] for s in samples])

    donor = np.arange(n, dtype=np.int64)
    n_fallback = 0

    if mode == "cross_station":
        buckets = {}
        for i, d in enumerate(doys):
            buckets.setdefault(int(d) // season_window, []).append(i)
        buckets = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
        for i in range(n):
            pool = buckets[int(doys[i]) // season_window]
            if len(pool) < 2:
                n_fallback += 1
                continue
            for _ in range(max_tries):
                j = int(pool[rng.integers(len(pool))])
                if stations[j] != stations[i]:
                    donor[i] = j
                    break
            else:
                n_fallback += 1
    elif mode == "within_station":
        by_station = {}
        for i, k in enumerate(stations):
            by_station.setdefault(k, []).append(i)
        by_station = {k: np.asarray(v, dtype=np.int64) for k, v in by_station.items()}
        for i in range(n):
            pool = by_station[stations[i]]
            if len(pool) < 2:
                n_fallback += 1
                continue
            for _ in range(max_tries):
                j = int(pool[rng.integers(len(pool))])
                if abs(int(doys[j]) - int(doys[i])) > min_doy_gap:
                    donor[i] = j
                    break
            else:
                n_fallback += 1
    else:
        raise ValueError(f"unknown mode {mode!r}")

    changed = donor != np.arange(n)
    stats = {
        "mode": mode, "seed": seed, "n_samples": n,
        "n_fallback": int(n_fallback),
        "frac_donor_assigned": float(changed.mean()),
        "frac_donor_diff_station": float((stations[donor] != stations).mean()),
        "median_abs_doy_gap": float(np.median(np.abs(doys[donor].astype(int)
                                                    - doys.astype(int)))),
    }
    return donor, stats


class AblationDataset(torch.utils.data.Dataset):
    """Wraps SoilMoistureDataset; replaces one modality with another sample's.

    NOTE this is deliberately NOT a batch-dimension permutation.  eval_predict.py
    iterates a non-shuffled loader, so a batch is typically consecutive days from one
    station -- permuting inside it would silently produce a within-station date shuffle
    while the run is labelled cross_station (runbook §24.3).
    """

    def __init__(self, base, modality: str, mode: str, seed: int = 0,
                 arm: str = "patchwise", **kw):
        if arm not in KEY_MAPS:
            raise ValueError(f"arm must be one of {sorted(KEY_MAPS)}, got {arm!r}")
        keys_map = KEY_MAPS[arm]
        if modality not in keys_map:
            raise ValueError(f"modality {modality!r} is not defined for arm {arm!r}; "
                             f"available: {sorted(keys_map)}")
        self.base = base
        self.arm = arm
        self.modality = modality
        self.keys = keys_map[modality]
        self.donor, self.stats = build_donor_map(base.samples, mode, seed, **kw)
        self.stats["modality"] = modality
        self.stats["arm"] = arm
        self.stats["keys"] = list(self.keys)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        j = int(self.donor[i])
        if j == i:
            return item                       # fallback: nothing to swap
        d = self.base[j]
        n_swapped = 0
        for k in self.keys:
            if isinstance(k, tuple):              # channel slice of a stacked tensor
                name, a, b = k
                if name in d and name in item:
                    t = item[name].clone()        # FRESH tensor: never edit a cached array
                    t[a:b] = d[name][a:b]
                    item[name] = t
                    n_swapped += 1
            elif k in d:
                item[k] = d[k]
                n_swapped += 1
        if self.arm in _STRICT_ARMS and n_swapped != len(self.keys):
            # STRICT on the U-Net arm. `n_swapped == 0` catches a wholly stale key list but not
            # a PARTIALLY stale one, and the partial case is the dangerous one: with the
            # patchwise map, `s2` matches s2_doys/s2_valid/s2_rel_pos on this arm and passes the
            # loose guard while `s2_pyr` -- the tokens themselves -- never moves. The model then
            # sees a donor's timestamps on its own imagery, which is incoherence rather than
            # absence and is uninterpretable (§24.2). A modality MOVES AS A WHOLE or not at all.
            missing = [k for k in self.keys if (k[0] if isinstance(k, tuple) else k) not in d]
            raise KeyError(
                f"ablation '{self.modality}' (arm={self.arm}) swapped {n_swapped} of "
                f"{len(self.keys)} keys; missing {missing}. A partial swap is incoherence, "
                f"not absence — refusing. Sample keys: {sorted(d)[:12]}..."
            )
        if n_swapped == 0 and self.modality not in _OPTIONAL_MODALITIES:
            # This used to be a bare `if k in d` with no else, which made a stale key list a
            # SILENT NO-OP: report() still printed a healthy donor fraction while nothing was
            # ablated, and the run came back "this modality does not matter". §35.9's arms are
            # the instrument for the whole patchwise hypothesis, so this must be fatal.
            raise KeyError(
                f"ablation '{self.modality}' matched none of {self.keys} in the sample. "
                f"Sample keys: {sorted(d)[:12]}... MODALITY_KEYS is stale for this arch."
            )
        return item

    def report(self) -> str:
        s = self.stats
        return (f"  ablation: {s['modality']} ({len(s['keys'])} keys) "
                f"mode={s['mode']} seed={s['seed']}\n"
                f"    donors assigned {s['frac_donor_assigned']:.1%}, "
                f"fallback {s['n_fallback']}, "
                f"donor from a different station {s['frac_donor_diff_station']:.1%}, "
                f"median |Δdoy| {s['median_abs_doy_gap']:.0f} d")
