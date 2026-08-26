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

# Every modality must match at least one key in every sample. Silence means a stale key list,
# which is a SILENT NO-OP ablation -- see AblationDataset.__getitem__.
_OPTIONAL_MODALITIES: set[str] = set()

MODALITIES = sorted(MODALITY_KEYS)


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

    def __init__(self, base, modality: str, mode: str, seed: int = 0, **kw):
        if modality not in MODALITY_KEYS:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        self.base = base
        self.modality = modality
        self.keys = MODALITY_KEYS[modality]
        self.donor, self.stats = build_donor_map(base.samples, mode, seed, **kw)
        self.stats["modality"] = modality
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
            if k in d:
                item[k] = d[k]
                n_swapped += 1
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
