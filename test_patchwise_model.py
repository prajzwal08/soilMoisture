"""
Regression tests for the patchwise model (§35.18 / §35.20 / §35.24).

WHAT THIS FILE IS FOR
---------------------
Every check here corresponds to a failure that would NOT crash. A patchwise model with a
flipped `~`, a permuted `depth_heads`, a mis-indexed driver cache or an entropy detector that
spans the wrong columns all run happily and return plausible soil-moisture numbers.

The previous version of this file asserted shapes and no-crash and nothing else: every test
batch set every validity tensor all-True and `era5_doys` was never 0, so no test batch
contained a single padded token, and deleting a `~` anywhere in the masking chain passed the
whole suite. `text/patchwise_math.md` §12 records that. The masking chain and the depth
ordering are correct; this file is what defends them.

The masking chain under test spans three different conventions and the polarity flips twice:

    dataset.py        *_valid / *_hist_valid / dem_valid   True = VALID
    model.py pads     `~valid`                             True = IGNORE  (nn.MHA convention)
    model.py _cross   `(~mem_pad)` as an SDPA bool mask     True = PARTICIPATE

Test strategy for that: put garbage in the slots that are supposed to be ignored and assert
the output does not move. A flipped `~` at ANY link makes the garbage visible.

DELIBERATELY TINY (d_model=64, 4 heads, 2 layers). Everything asserted here — polarity,
ordering, cache indexing, detector arithmetic — is independent of width and depth, and a full
768/12/6 model does forward AND backward on CPU in minutes rather than seconds.

Synthetic tensors only: no zarr, no dataset construction, no GPU by default.
Anything needing a GPU is marked `@pytest.mark.gpu` and is deselected by `-m "not gpu"`.

    sbatch slurm/run_tests.sh          # the only supported way to run this
"""
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MAX_S1, MAX_S2, MAX_SIF, MAX_TWSA, SM_DEPTHS   # noqa: E402
from model import (                                                # noqa: E402
    DOY_MAX_HARMONIC,
    EMB_INIT_STD,
    HIST_EMB_INIT_STD,
    SoilMoistureModel,
    _row_entropy,
    circular_doy_pe,
    masked_huber_loss,
)

# Small enough for a login-node-sized CPU budget; see the module docstring.
D_MODEL, N_HEADS, N_LAYERS, N_DEPTHS = 64, 4, 2, 3

# Nyquist for daily sampling of an annual cycle. Anything above this is a reflected copy of a
# harmonic already present, which is what the old linear k = 1 … 384 ramp spent half its
# channels on.
NYQUIST_HARMONIC = 365.25 / 2.0


# ── batch construction ───────────────────────────────────────────────────────

def build(**kw):
    """A tiny SoilMoistureModel with the production defaults everywhere else."""
    kw.setdefault("driver_layers", 2)
    kw.setdefault("d_model", D_MODEL)
    kw.setdefault("n_heads", N_HEADS)
    kw.setdefault("n_layers", N_LAYERS)
    return SoilMoistureModel(n_depths=N_DEPTHS, use_cls_depth=True, **kw)


def fake_batch(B, K, seed=0, D=D_MODEL,
               s2_valid_frac=1.0, s1_valid_frac=1.0,
               era5_valid_frac=1.0, sif_valid=True, twsa_valid=True,
               dem_valid=True, lulc_valid=True):
    """A patchwise batch exactly as dataset.py emits it, with real padding available.

    `*_valid_frac` keeps the LEADING fraction of slots valid and pads the rest, which is the
    shape the loaders actually produce (they compact oldest-first and pad at the tail).
    Validity is uniform across the K patches so that `*_rel_pos` — which is per (B, T) and
    shared by every patch — can be scrambled in the padded slots without touching a live one.
    """
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g)

    def ri(hi, s):
        return torch.randint(0, hi, s, generator=g)

    def head_mask(T, frac):
        n = max(1, int(round(T * frac)))
        m = torch.zeros(B, T, dtype=torch.bool)
        m[:, :n] = True
        return m

    s2_ok = head_mask(MAX_S2, s2_valid_frac)          # (B, T)
    s1_ok = head_mask(MAX_S1, s1_valid_frac)
    era5_ok = head_mask(365, era5_valid_frac)

    era5_doys = torch.randint(1, 366, (B, 365), generator=g)
    era5_doys[~era5_ok] = 0                            # 0 IS the ERA5 padding signal

    return {
        # per-patch satellite history
        "s2_hist":        r(B, MAX_S2, K, D).half(),
        "s2_hist_valid":  s2_ok[:, :, None].expand(B, MAX_S2, K).contiguous(),
        "s2_rel_pos":     ri(365, (B, MAX_S2)),
        "s1_hist":        r(B, MAX_S1, K, D).half(),
        "s1_hist_valid":  s1_ok[:, :, None].expand(B, MAX_S1, K).contiguous(),
        "s1_rel_pos":     ri(365, (B, MAX_S1)),
        "s1_orbit":       ri(2, (B, MAX_S1)),
        # per-patch statics
        "dem_tok":        r(B, K, D).half(),
        "lulc_tok":       r(B, K, D).half(),
        "dem_valid":      torch.full((B, K), bool(dem_valid)),
        "lulc_valid":     torch.full((B, K), bool(lulc_valid)),
        "token_idx":      torch.full((B, K), 105),
        # tile-level drivers
        "soil_patch":     r(B, 21, 74, 74),
        "era5":           r(B, 365, 19),
        "era5_doys":      era5_doys,
        "era5_rel_pos":   ri(365, (B, 365)),
        "sif":            r(B, MAX_SIF, 1),
        "sif_doys":       torch.randint(1, 366, (B, MAX_SIF), generator=g),
        "sif_rel_pos":    ri(365, (B, MAX_SIF)),
        "sif_valid":      torch.full((B, MAX_SIF), bool(sif_valid)),
        "twsa":           r(B, MAX_TWSA, 1),
        "twsa_doys":      torch.randint(1, 366, (B, MAX_TWSA), generator=g),
        "twsa_rel_pos":   ri(365, (B, MAX_TWSA)),
        "twsa_valid":     torch.full((B, MAX_TWSA), bool(twsa_valid)),
        "label":          torch.rand(B, N_DEPTHS, generator=g),
    }


def scramble_ignored(batch, seed=999):
    """Replace the CONTENT of every slot the masks declare invalid with loud garbage.

    Values only — never the mask itself, and never `era5_doys`, which IS the ERA5 mask.
    If the model output moves, something the masks said to ignore was read.
    """
    g = torch.Generator().manual_seed(seed)
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}

    def loud(shape, dtype):
        return (torch.randn(shape, generator=g) * 37.0).to(dtype)

    for key in ("s2", "s1"):
        inv = ~b[f"{key}_hist_valid"]                             # (B, T, K)
        h = b[f"{key}_hist"]
        b[f"{key}_hist"] = torch.where(inv.unsqueeze(-1), loud(h.shape, h.dtype), h)
        # rel_pos / orbit are per (B, T) and shared by all K patches, so only touch slots
        # that are invalid for EVERY patch — otherwise a live patch's token changes.
        all_inv = inv.all(dim=2)                                  # (B, T)
        rp = b[f"{key}_rel_pos"]
        b[f"{key}_rel_pos"] = torch.where(all_inv, torch.randint_like(rp, 0, 365), rp)
        if key == "s1":
            ob = b["s1_orbit"]
            b["s1_orbit"] = torch.where(all_inv, 1 - ob, ob)

    era5_inv = b["era5_doys"] == 0                                # (B, 365)
    b["era5"] = torch.where(era5_inv.unsqueeze(-1),
                            loud(b["era5"].shape, b["era5"].dtype), b["era5"])
    erp = b["era5_rel_pos"]
    b["era5_rel_pos"] = torch.where(era5_inv, torch.randint_like(erp, 0, 365), erp)

    for key in ("sif", "twsa"):
        inv = ~b[f"{key}_valid"]                                  # (B, T)
        v = b[key]
        b[key] = torch.where(inv.unsqueeze(-1), loud(v.shape, v.dtype), v)
        for suffix, hi in (("doys", 366), ("rel_pos", 365)):
            t = b[f"{key}_{suffix}"]
            b[f"{key}_{suffix}"] = torch.where(inv, torch.randint_like(t, 1, hi), t)

    for key in ("dem", "lulc"):
        inv = ~b[f"{key}_valid"]                                  # (B, K)
        t = b[f"{key}_tok"]
        b[f"{key}_tok"] = torch.where(inv.unsqueeze(-1), loud(t.shape, t.dtype), t)

    return b


def fwd(net, batch):
    with torch.no_grad():
        return net(batch)


# ═════════════════════════════════════════════════════════════════════════════
# 1. MASK POLARITY — both conventions, every link in the chain
# ═════════════════════════════════════════════════════════════════════════════
#
# Catches: a flipped or deleted `~` at dataset(True=valid) -> model `~pads`
# (True=ignore, nn.MultiheadAttention) -> `(~mem_pad)` (True=participate, SDPA).
# With the polarity inverted the model reads exactly the slots it was told to drop, which is
# padded zeros and cloud — a plausible-looking, slightly worse model and no traceback.

MASK_CASES = [
    ("s2_history",  dict(s2_valid_frac=0.5)),
    ("s1_history",  dict(s1_valid_frac=0.5)),
    ("era5_doys0",  dict(era5_valid_frac=0.5)),
    ("sif",         dict(sif_valid=False)),
    ("twsa",        dict(twsa_valid=False)),
    ("dem",         dict(dem_valid=False)),
    ("lulc",        dict(lulc_valid=False)),
    ("all_at_once", dict(s2_valid_frac=0.5, s1_valid_frac=0.5, era5_valid_frac=0.5,
                         sif_valid=False, twsa_valid=False,
                         dem_valid=False, lulc_valid=False)),
]


@pytest.mark.parametrize("mode", ["memory", "concat"])
@pytest.mark.parametrize("name,kw", MASK_CASES, ids=[c[0] for c in MASK_CASES])
def test_invalid_slots_cannot_influence_the_output(mode, name, kw):
    net = build(driver_mode=mode).eval()
    base_batch = fake_batch(2, 3, seed=101, **kw)

    # sanity: the case must actually contain padding, or the test proves nothing. This is
    # precisely the hole in the old suite.
    n_pad = (
        (~base_batch["s2_hist_valid"]).sum()
        + (~base_batch["s1_hist_valid"]).sum()
        + (base_batch["era5_doys"] == 0).sum()
        + (~base_batch["sif_valid"]).sum()
        + (~base_batch["twsa_valid"]).sum()
        + (~base_batch["dem_valid"]).sum()
        + (~base_batch["lulc_valid"]).sum()
    )
    assert n_pad > 0, f"{name}: batch has no padded slot — the test would be vacuous"

    base = fwd(net, base_batch)
    for seed in (1, 2):
        got = fwd(net, scramble_ignored(base_batch, seed=seed))
        delta = (base - got).abs().max().item()
        assert delta < 1e-5, (
            f"{mode}/{name}: scrambling MASKED slots moved the output by {delta:.3e}. "
            f"The masking chain reads slots it was told to ignore — check the `~` in "
            f"_build_patch_seq / _build_driver_tokens and the `(~mem_pad)` in _cross."
        )


@pytest.mark.parametrize("mode", ["memory", "concat"])
def test_valid_slots_DO_influence_the_output(mode):
    """The converse. A mask that ignores everything also passes the test above."""
    net = build(driver_mode=mode).eval()
    b = fake_batch(2, 3, seed=102, s2_valid_frac=0.5, s1_valid_frac=0.5,
                   era5_valid_frac=0.5)
    base = fwd(net, b)

    for key, sel in (("s2_hist",  b["s2_hist_valid"]),
                     ("s1_hist",  b["s1_hist_valid"]),
                     ("dem_tok",  b["dem_valid"]),
                     ("lulc_tok", b["lulc_valid"])):
        b2 = dict(b)
        t = b[key].clone()
        m = sel.unsqueeze(-1) if sel.ndim == t.ndim - 1 else sel
        b2[key] = torch.where(m, t + 3.0, t)
        d = (base - fwd(net, b2)).abs().max().item()
        assert d > 1e-6, f"{mode}: perturbing VALID {key} did not change the output ({d:.3e})"

    b3 = dict(b)
    e = b["era5"].clone()
    b3["era5"] = torch.where((b["era5_doys"] > 0).unsqueeze(-1), e + 3.0, e)
    d = (base - fwd(net, b3)).abs().max().item()
    assert d > 1e-6, f"{mode}: perturbing VALID era5 rows did not change the output"


def test_era5_padding_is_signalled_by_doy_zero_only():
    """`era5_doys == 0` is the whole ERA5 mask — there is no separate era5_valid tensor.

    So flipping a slot's doy from 0 to non-zero must ADMIT it (output changes), and the
    staleness embedding of a padded row must be inert.
    """
    net = build().eval()
    b = fake_batch(1, 1, seed=103, era5_valid_frac=0.5)
    base = fwd(net, b)

    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    pad_idx = (b["era5_doys"][0] == 0).nonzero()[0, 0].item()
    b2["era5_doys"][0, pad_idx] = 200
    assert (base - fwd(net, b2)).abs().max().item() > 1e-6, (
        "un-padding an ERA5 slot did not change the output — `era5_doys == 0` is not "
        "reaching the driver pad mask"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. DEPTH ORDERING
# ═════════════════════════════════════════════════════════════════════════════
#
# Catches: a permutation between SM_DEPTHS, the dataset's label column, the model's output
# column and train.py's metric index. Silent, and it would report the 30-100 cm model's error
# under the 0-10 cm label. The dataset half of this is in test_patchwise_dataset.py.

def test_sm_depths_order_is_the_contract():
    assert SM_DEPTHS == ["0-10", "10-30", "30-100"], (
        "SM_DEPTHS defines the label column order, the model's output column order and "
        "train.py's per-depth metric index. Changing it silently re-labels every reported "
        "number."
    )


@pytest.mark.parametrize("per_depth", [False, True])
@pytest.mark.parametrize("d", range(N_DEPTHS))
def test_only_the_observed_depth_trains_its_head(d, per_depth):
    net = build().eval()
    b = fake_batch(2, 1, seed=110 + d)
    b["label"] = torch.full((2, N_DEPTHS), float("nan"))
    b["label"][:, d] = 0.3

    net.zero_grad(set_to_none=True)
    kw = dict(per_depth=True, depth_weights=torch.ones(N_DEPTHS)) if per_depth else {}
    masked_huber_loss(net(b), b["label"], **kw).backward()

    g = net.depth_heads[d].weight.grad
    assert g is not None and g.abs().max() > 0, (
        f"label [{'nan,' * d}x] gave depth_heads[{d}] no gradient — the model's output "
        f"column {d} is not the column the label occupies"
    )
    for j in range(N_DEPTHS):
        if j == d:
            continue
        gj = net.depth_heads[j].weight.grad
        assert gj is None or gj.abs().max() == 0, (
            f"a label observed only at depth {d} ({SM_DEPTHS[d]}) sent gradient into "
            f"depth_heads[{j}] ({SM_DEPTHS[j]}) — the heads are permuted against the label"
        )


def test_each_head_reads_its_own_depth_cls_row():
    """depth_heads[i] must read depth_ctx[:, i, :] and no other row.

    Zeroing head i's weight must change output column i only.
    """
    net = build().eval()
    b = fake_batch(2, 2, seed=120)
    base = fwd(net, b)
    for i in range(N_DEPTHS):
        net2 = build().eval()
        net2.load_state_dict(net.state_dict())
        with torch.no_grad():
            net2.depth_heads[i].weight.zero_()
        got = fwd(net2, b)
        moved = (base - got).abs().amax(dim=(0, 1))            # (n_depths,)
        assert moved[i] > 1e-6, f"zeroing depth_heads[{i}] did not move output column {i}"
        for j in range(N_DEPTHS):
            if j != i:
                assert moved[j] < 1e-7, (
                    f"zeroing depth_heads[{i}] moved output column {j} — the readout is "
                    f"not one head per column"
                )


# ═════════════════════════════════════════════════════════════════════════════
# 3. THE CACHE IS PER DAY, NOT PER PATCH
# ═════════════════════════════════════════════════════════════════════════════
#
# patchwise_math.md §5: "Read the index letters: `d` (day) appears on both Q and Kc; `p`
# (patch) appears on Q only. That asymmetry IS the sharing. This is where a silent bug will
# live... It must be asserted in test_patchwise_model.py."
#
# A stray repeat_interleave on the wrong axis makes every patch read some other day's
# weather. Nothing crashes; you get a mediocre model and no signal.

def _uniform_patch_batch(B, K, seed):
    """Every (b, k) patch carries IDENTICAL satellite/static content; drivers differ per b."""
    b = fake_batch(B, K, seed=seed)
    one_s2 = b["s2_hist"][:1, :, :1]                              # (1, T, 1, D)
    one_s1 = b["s1_hist"][:1, :, :1]
    b["s2_hist"] = one_s2.expand(B, MAX_S2, K, D_MODEL).contiguous()
    b["s1_hist"] = one_s1.expand(B, MAX_S1, K, D_MODEL).contiguous()
    b["s2_rel_pos"] = b["s2_rel_pos"][:1].expand(B, MAX_S2).contiguous()
    b["s1_rel_pos"] = b["s1_rel_pos"][:1].expand(B, MAX_S1).contiguous()
    b["s1_orbit"] = b["s1_orbit"][:1].expand(B, MAX_S1).contiguous()
    b["dem_tok"] = b["dem_tok"][:1, :1].expand(B, K, D_MODEL).contiguous()
    b["lulc_tok"] = b["lulc_tok"][:1, :1].expand(B, K, D_MODEL).contiguous()
    # drivers stay per-sample-distinct, and made obviously so
    b["era5"] = b["era5"] + torch.arange(B).float()[:, None, None] * 5.0
    b["soil_patch"] = b["soil_patch"] + torch.arange(B).float()[:, None, None, None] * 5.0
    return b


@pytest.mark.parametrize("mode", ["memory", "concat"])
def test_every_patch_reads_its_own_samples_memory(mode):
    B, K = 3, 4
    net = build(driver_mode=mode).eval()
    b = _uniform_patch_batch(B, K, seed=130)
    out = fwd(net, b)                                             # (B, K, n_depths)

    # Patch content is identical within AND across samples, so any variation across k means
    # patch k read something patch 0 did not — i.e. the memory was indexed by patch.
    spread_k = (out - out[:, :1, :]).abs().max().item()
    assert spread_k < 1e-5, (
        f"{mode}: identical patches produced different predictions (max spread {spread_k:.3e}) "
        f"— the driver cache is being indexed on the patch axis. See patchwise_math.md §5."
    )
    # And the drivers must still separate the samples, or the test above is vacuous.
    spread_b = (out - out[:1]).abs().max().item()
    assert spread_b > 1e-5, (
        f"{mode}: samples with different weather gave identical predictions "
        f"({spread_b:.3e}) — the drivers are not reaching the readout at all"
    )


@pytest.mark.parametrize("mode", ["memory", "concat"])
def test_batched_forward_equals_per_sample_forward(mode):
    """B=1 cannot mis-index a batch axis, so the single-sample forward is ground truth."""
    B, K = 3, 4
    net = build(driver_mode=mode).eval()
    b = fake_batch(B, K, seed=131, s2_valid_frac=0.6, era5_valid_frac=0.7)
    batched = fwd(net, b)
    for i in range(B):
        single = {k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in b.items()}
        d = (batched[i] - fwd(net, single)[0]).abs().max().item()
        assert d < 1e-4, (
            f"{mode}: sample {i} of a batch of {B} differs from its own single-sample "
            f"forward by {d:.3e} — samples are leaking into each other"
        )


def test_permuting_only_the_weather_changes_every_prediction():
    net = build().eval()
    b = fake_batch(4, 2, seed=132)
    base = fwd(net, b)
    perm = [1, 2, 3, 0]
    b2 = dict(b)
    for k in ("era5", "era5_doys", "era5_rel_pos", "soil_patch",
              "sif", "sif_doys", "sif_rel_pos", "sif_valid",
              "twsa", "twsa_doys", "twsa_rel_pos", "twsa_valid"):
        b2[k] = b[k][perm]
    d = (base - fwd(net, b2)).abs().max().item()
    assert d > 1e-4, f"permuting the weather across the batch changed nothing (max|d|={d:.3e})"


# ═════════════════════════════════════════════════════════════════════════════
# 4. K = 196, END TO END
# ═════════════════════════════════════════════════════════════════════════════
#
# The inference path. It has never been executed; training runs K=1 only.

@pytest.mark.parametrize("mode", ["memory", "concat"])
def test_full_map_forward_K196(mode, capsys):
    from dataset import N_TOKENS, TOKEN_GRID
    assert N_TOKENS == TOKEN_GRID * TOKEN_GRID == 196

    net = build(driver_mode=mode).eval()
    b = fake_batch(1, N_TOKENS, seed=140, s2_valid_frac=0.6, era5_valid_frac=0.8)
    out = fwd(net, b)
    assert tuple(out.shape) == (1, N_TOKENS, N_DEPTHS), tuple(out.shape)
    assert torch.isfinite(out).all(), "K=196 forward produced non-finite predictions"

    m = out.reshape(1, TOKEN_GRID, TOKEN_GRID, N_DEPTHS)
    assert tuple(m.shape) == (1, 14, 14, N_DEPTHS)
    # row-major: map[0, r, c] must be patch r*14 + c
    assert torch.equal(m[0, 3, 7], out[0, 3 * TOKEN_GRID + 7])

    # Across-patch SD must be FINITE. Whether it is LARGE is the scientific question
    # (§35.20 step 1), not a test invariant — so it is reported, never asserted.
    sd = out[0].std(dim=0)
    assert torch.isfinite(sd).all(), f"across-patch SD is not finite: {sd.tolist()}"
    with capsys.disabled():
        print(f"\n  [{mode}] K=196 across-patch SD per depth (untrained weights, "
              f"reported not asserted): {[round(float(v), 6) for v in sd]}")


def test_training_loss_refuses_K_gt_1():
    """K>1 is inference-only: supervising several patches needs multi-station labels, which
    dataset.py does not emit (§35.19). Silently averaging them onto one label would be wrong."""
    with pytest.raises(ValueError, match="K=1"):
        masked_huber_loss(torch.randn(2, 196, N_DEPTHS), torch.rand(2, N_DEPTHS))
    with pytest.raises(ValueError, match="K=1"):
        masked_huber_loss(torch.randn(2, 4, N_DEPTHS), torch.rand(2, N_DEPTHS))
    # a 224x224 map is the deleted U-Net's contract
    with pytest.raises(ValueError):
        masked_huber_loss(torch.randn(2, 3, 224, 224), torch.rand(2, N_DEPTHS))
    # K=1 is accepted
    assert torch.isfinite(
        masked_huber_loss(torch.randn(2, 1, N_DEPTHS), torch.rand(2, N_DEPTHS)))


# ═════════════════════════════════════════════════════════════════════════════
# 5. THE COLLAPSE DETECTORS
# ═════════════════════════════════════════════════════════════════════════════
#
# Contract: _last_attn_entropy is (n_layers, 3) = [sum_entropy_nats, sum_ratio, count] over
# HISTORY keys only, renormalised over that slice, with a PER-SAMPLE log(n_valid) reference.
# §35.20 stakes the deferral of register standardisation on this detector working.

def _attn(N, h, L, n_readout, hist_start, hist_end, pad, kind="uniform",
          prefix_mass=0.0):
    """Hand-built attention weights: rows are proper distributions over all L keys."""
    w = torch.zeros(N, h, L, L)
    for n in range(N):
        live = (~pad[n, hist_start:hist_end]).nonzero().flatten()
        if live.numel() == 0:
            continue
        hist_mass = 1.0 - prefix_mass
        if kind == "uniform":
            vals = torch.full((live.numel(),), hist_mass / live.numel())
        elif kind == "peaked":
            vals = torch.full((live.numel(),), hist_mass * 1e-6 / max(live.numel() - 1, 1))
            vals[0] = hist_mass * (1.0 - 1e-6)
        else:
            raise ValueError(kind)
        w[n, :, :n_readout, hist_start + live] = vals
        if prefix_mass > 0 and hist_start > 0:
            w[n, :, :n_readout, :hist_start] = prefix_mass / hist_start
    return w


def test_row_entropy_uniform_gives_ratio_one_whatever_n_valid():
    """THE point of the §35.24 rewrite. The old fixed log(100) reference scored a fully
    collapsed 36-slot row at ~3.58/4.605 = 0.78 and called it healthy."""
    N, h, L = 2, 3, 12
    n_readout, hist_start, hist_end = 3, 5, 12          # 7 history columns
    pad = torch.zeros(N, L, dtype=torch.bool)
    pad[0, 9:12] = True                                  # sample 0: 4 valid, sample 1: 7

    w = _attn(N, h, L, n_readout, hist_start, hist_end, pad, "uniform")
    ent_sum, ratio_sum, count = _row_entropy(w, pad, n_readout, hist_start, hist_end)

    n_triples = N * h * n_readout
    assert count.item() == pytest.approx(n_triples)
    assert ratio_sum.item() == pytest.approx(n_triples, abs=1e-3), (
        "uniform (== collapsed) attention must score ratio 1.0 per row regardless of how "
        "many history slots are valid"
    )
    expect_nats = h * n_readout * (math.log(4) + math.log(7))
    assert ent_sum.item() == pytest.approx(expect_nats, rel=1e-4)


def test_row_entropy_is_invariant_to_the_number_of_valid_slots():
    N, h, L = 1, 2, 60
    n_readout, hist_start, hist_end = 3, 5, 60
    ratios = {}
    for n_valid in (2, 5, 20, 55):
        pad = torch.ones(N, L, dtype=torch.bool)
        pad[:, :hist_start] = False
        pad[:, hist_start:hist_start + n_valid] = False
        w = _attn(N, h, L, n_readout, hist_start, hist_end, pad, "uniform")
        _, ratio_sum, count = _row_entropy(w, pad, n_readout, hist_start, hist_end)
        ratios[n_valid] = ratio_sum.item() / count.item()
    assert all(abs(v - 1.0) < 1e-3 for v in ratios.values()), ratios


def test_row_entropy_peaked_is_well_below_one():
    N, h, L = 2, 3, 40
    n_readout, hist_start, hist_end = 3, 5, 40
    pad = torch.zeros(N, L, dtype=torch.bool)
    w = _attn(N, h, L, n_readout, hist_start, hist_end, pad, "peaked")
    _, ratio_sum, count = _row_entropy(w, pad, n_readout, hist_start, hist_end)
    mean_ratio = ratio_sum.item() / count.item()
    assert mean_ratio < 0.2, f"a sharply-peaked row scored {mean_ratio:.3f}; expected ~0"


def test_row_entropy_renormalises_over_history_only():
    """Mass leaking onto the 5-token depth-CLS/dem/lulc prefix must not change the statistic."""
    N, h, L = 2, 3, 30
    n_readout, hist_start, hist_end = 3, 5, 30
    pad = torch.zeros(N, L, dtype=torch.bool)
    a = _row_entropy(_attn(N, h, L, n_readout, hist_start, hist_end, pad, "uniform", 0.0),
                     pad, n_readout, hist_start, hist_end)
    b = _row_entropy(_attn(N, h, L, n_readout, hist_start, hist_end, pad, "uniform", 0.6),
                     pad, n_readout, hist_start, hist_end)
    assert torch.allclose(a, b, atol=1e-4), (a.tolist(), b.tolist())


def test_row_entropy_drops_rows_with_fewer_than_two_valid_slots():
    N, h, L = 3, 2, 20
    n_readout, hist_start, hist_end = 3, 5, 20
    pad = torch.zeros(N, L, dtype=torch.bool)
    pad[2, hist_start + 1:] = True                       # sample 2 has exactly 1 valid slot
    w = _attn(N, h, L, n_readout, hist_start, hist_end, pad, "uniform")
    _, _, count = _row_entropy(w, pad, n_readout, hist_start, hist_end)
    assert count.item() == pytest.approx(2 * h * n_readout), (
        "log(1) = 0 would divide by zero; rows with <2 valid history slots must be dropped "
        "from BOTH the sum and the count"
    )


def test_row_entropy_excludes_columns_outside_hist_start_hist_end():
    """The bound at BOTH ends. hist_start excludes the prefix; hist_end excludes concat's
    431 driver columns."""
    N, h, L = 1, 2, 40
    n_readout, hist_start, hist_end = 3, 5, 20
    pad = torch.zeros(N, L, dtype=torch.bool)
    w = _attn(N, h, L, n_readout, hist_start, hist_end, pad, "uniform")
    ref = _row_entropy(w, pad, n_readout, hist_start, hist_end)
    w2 = w.clone()
    w2[:, :, :n_readout, hist_end:] = 0.25               # loud content past the bound
    w2[:, :, :n_readout, :hist_start] = 0.25
    got = _row_entropy(w2, pad, n_readout, hist_start, hist_end)
    assert torch.allclose(ref, got, atol=1e-5), (ref.tolist(), got.tolist())


def test_model_emits_the_contract_shaped_entropy():
    net = build().eval()
    for blk in net.patch_blocks:
        blk.collect_entropy = True
    B, K = 2, 2
    fwd(net, fake_batch(B, K, seed=150, s2_valid_frac=0.5, s1_valid_frac=0.5))

    ent = net._last_attn_entropy
    assert ent is not None, "detector armed but nothing was stashed"
    assert tuple(ent.shape) == (N_LAYERS, 3), tuple(ent.shape)
    assert ent.dtype == torch.float32
    assert torch.isfinite(ent).all()
    expect = B * K * N_HEADS * N_DEPTHS
    for layer in range(N_LAYERS):
        assert ent[layer, 2].item() == pytest.approx(expect), (
            f"layer {layer} count {ent[layer, 2].item()} != n_samples({B * K}) x "
            f"n_heads({N_HEADS}) x n_readout({N_DEPTHS}) = {expect}"
        )
        mean_ratio = ent[layer, 1].item() / ent[layer, 2].item()
        assert 0.0 <= mean_ratio <= 1.0 + 1e-4, mean_ratio


def test_entropy_is_off_by_default():
    """Collecting weights forces the math kernel and gives up SDPA — it must be opt-in."""
    net = build().eval()
    fwd(net, fake_batch(2, 1, seed=151))
    assert getattr(net, "_last_attn_entropy", None) is None


def test_entropy_count_is_identical_in_both_driver_modes():
    """The hist_end bound. In concat mode the sequence continues into 431 driver tokens after
    the 105 patch ones; an open-ended `hist_start:` slice folds the weather into the history
    entropy and the two --driver-mode arms stop being comparable — which is the one comparison
    those arms exist for."""
    B, K = 2, 2
    b = fake_batch(B, K, seed=152, s2_valid_frac=0.5, s1_valid_frac=0.5,
                   era5_valid_frac=0.5)
    counts = {}
    for mode in ("memory", "concat"):
        net = build(driver_mode=mode).eval()
        for blk in net.patch_blocks:
            blk.collect_entropy = True
        fwd(net, b)
        counts[mode] = net._last_attn_entropy[:, 2].tolist()
    assert counts["memory"] == counts["concat"], (
        f"entropy count differs by driver mode: {counts}. The detector is spanning the "
        f"driver columns in concat mode — restore the hist_end bound in _row_entropy."
    )


def test_depth_ctx_diagnostic_is_produced():
    """The producer was dead for several sessions after the U-Net strip: train.py read a
    getattr default and logged nothing."""
    B, K = 2, 3
    net = build().eval()
    fwd(net, fake_batch(B, K, seed=153))
    ctx = getattr(net, "_last_depth_ctx", None)
    n = getattr(net, "_last_depth_ctx_n", None)
    assert ctx is not None and n is not None
    assert tuple(ctx.shape) == (N_DEPTHS, net.d_model), tuple(ctx.shape)
    assert ctx.dtype == torch.float32 and torch.isfinite(ctx).all()
    assert n == B * K, f"_last_depth_ctx_n={n} but the batch had {B * K} (sample, patch) rows"
    assert not ctx.requires_grad, "_last_depth_ctx must be detached"


def test_depth_ctx_is_a_sum_not_a_mean():
    """train.py all_reduce(SUM)s it, which is only correct on sums."""
    net = build().eval()
    b1 = fake_batch(1, 1, seed=154)
    fwd(net, b1)
    one = net._last_depth_ctx.clone()
    b2 = {k: (torch.cat([v, v]) if torch.is_tensor(v) else v) for k, v in b1.items()}
    fwd(net, b2)
    two = net._last_depth_ctx
    assert torch.allclose(two, one * 2, atol=1e-4), (
        "doubling the batch did not double _last_depth_ctx — it is a mean, but the contract "
        "and train.py's all_reduce(SUM) require a sum"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 6. S1 ORBIT TAGGING (model half; the loader half is in test_patchwise_dataset.py)
# ═════════════════════════════════════════════════════════════════════════════
#
# RTC backscatter differs systematically between ascending and descending passes by an amount
# comparable to the moisture signal. With one shared S1 tag an orbit switch is
# indistinguishable from a wetting event and that variance is attributed to soil moisture.

def test_hist_modality_embedding_is_three_way():
    net = build()
    assert net.hist_modality_emb.num_embeddings == 3, (
        "hist_modality_emb must be 3-way: 0 = S2, 1 = S1 ascending, 2 = S1 descending"
    )
    w = net.hist_modality_emb.weight.detach()
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert (w[i] - w[j]).abs().max() > 0, f"modality rows {i} and {j} are identical"


def test_orbit_changes_the_prediction():
    net = build().eval()
    b = fake_batch(2, 2, seed=160)
    b["s1_orbit"] = torch.zeros_like(b["s1_orbit"])                 # all ASC
    asc = fwd(net, b)
    b2 = dict(b)
    b2["s1_orbit"] = torch.ones_like(b["s1_orbit"])                 # all DESC
    d = (asc - fwd(net, b2)).abs().max().item()
    assert d > 1e-6, (
        f"identical acquisitions on different orbits produced identical tokens ({d:.3e}) — "
        f"s1_orbit is not reaching hist_modality_emb"
    )


def test_orbit_only_matters_on_valid_slots():
    net = build().eval()
    b = fake_batch(2, 2, seed=161, s1_valid_frac=0.5)
    base = fwd(net, b)
    b2 = dict(b)
    ob = b["s1_orbit"].clone()
    inv = (~b["s1_hist_valid"]).all(dim=2)                          # (B, T)
    b2["s1_orbit"] = torch.where(inv, 1 - ob, ob)
    assert (base - fwd(net, b2)).abs().max().item() < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# 7. CONSTRUCTION, CHECKPOINTS AND THE FROZEN-FEATURE LAYERNORMS
# ═════════════════════════════════════════════════════════════════════════════

def test_no_unet_remnants():
    m = build()
    for name in ("decoder", "transformer_layers", "scale_emb", "spatial_row_emb",
                 "spatial_modality_emb", "arch", "depth_film", "patch_cls"):
        assert not hasattr(m, name), f"unet/FiLM remnant still constructed: {name}"


def test_use_cls_depth_false_is_refused():
    with pytest.raises((ValueError, TypeError)):
        build(use_cls_depth=False)


def test_bad_driver_mode_is_refused():
    with pytest.raises(ValueError):
        build(driver_mode="perceiver")


def test_concat_does_not_build_T1():
    """T1's justification is that it restores the 431x431 'weather reads weather' block, which
    concat already has inside its joint stack. Running it in concat mode gave that arm the
    block twice.

    §35.26 separated the two decisions that used to ride on one conditional. `driver_enc` is
    still memory-only — that is the architecture question the arm exists to answer. But
    `driver_norm` is now built in BOTH modes: making the norm memory-only meant the arms
    differed in contextualisation AND in normalisation, so a difference in result could not be
    attributed to either. See test_driver_norm_applies_in_both_modes."""
    mem = build(driver_mode="memory")
    cat = build(driver_mode="concat")

    assert len(mem.driver_enc) == 2 and isinstance(mem.driver_norm, nn.LayerNorm)
    assert isinstance(cat.driver_enc, nn.ModuleList) and len(cat.driver_enc) == 0
    assert isinstance(cat.driver_norm, nn.LayerNorm), (
        "driver_norm must exist in concat mode too (§35.26) — otherwise the two "
        "--driver-mode arms differ in normalisation as well as contextualisation"
    )

    n_mem = sum(p.numel() for p in mem.parameters())
    n_cat = sum(p.numel() for p in cat.parameters())
    assert n_cat < n_mem, (
        f"concat ({n_cat:,}) should have materially fewer parameters than memory "
        f"({n_mem:,}): it builds no T1 and no cross-attention projections"
    )
    for blk in cat.patch_blocks:
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "norm_cross"):
            assert not hasattr(blk, attr), f"concat block still built {attr}"


def test_head_bias_init_sets_the_biases():
    """Labels are raw m3/m3 (~0.25); a default U(+-0.036) bias opens the loss in Huber's
    LINEAR regime, whose gradient is a constant +-delta carrying no information about the size
    of the error."""
    vals = [0.21, 0.27, 0.33]
    net = build(head_bias_init=vals)
    for i, v in enumerate(vals):
        assert net.depth_heads[i].bias.item() == pytest.approx(v), (
            f"depth_heads[{i}].bias = {net.depth_heads[i].bias.item()}, expected {v} "
            f"({SM_DEPTHS[i]})"
        )
    # and the untouched default is nowhere near the data mean. PyTorch inits Linear bias to
    # U(+-1/sqrt(fan_in)), so the bound is d_model-dependent — at the production d_model=768
    # that is +-0.036, but this tiny model has d_model=64 and so +-0.125. Hard-coding 0.05
    # made the test fail on the model it actually builds, roughly 60% of the time.
    default_bias = abs(build().depth_heads[0].bias.item())
    assert default_bias <= 1.0 / math.sqrt(D_MODEL) + 1e-6
    assert default_bias < 0.20                       # nowhere near a mean SM of ~0.25

    with pytest.raises(ValueError):
        build(head_bias_init=[0.2, 0.3])


def test_doy_pe_buffer_is_not_in_the_state_dict():
    """Registered persistent=False so it does not bloat the state_dict or break older
    checkpoints."""
    net = build()
    sd = net.state_dict()
    assert "doy_pe" not in sd, "doy_pe leaked into the state_dict (persistent=False?)"
    assert net.doy_pe.shape == (367, D_MODEL)
    # an older checkpoint — one that never held doy_pe — must still load strictly
    net2 = build()
    net2.load_state_dict(sd)                        # strict=True by default
    assert (net2.doy_pe - net.doy_pe).abs().max() == 0


def _scale_frozen(b, factor=64.0):
    """Multiply every frozen TerraMind feature by an exact power of two (no fp16 rounding)."""
    b2 = dict(b)
    for k in ("s2_hist", "s1_hist", "dem_tok", "lulc_tok"):
        b2[k] = b[k] * factor
    return b2


def test_input_norm_is_off_by_default_and_magnitude_survives():
    """§35.26. The input LayerNorms are OFF by default, so a frozen token keeps its own
    magnitude — 9.3% of S2's TEMPORAL variance rides there and does not collapse when the
    register dims are stripped (§35.25), and the frozen pooled baseline does not normalise
    either. If someone flips the default back on, this fails."""
    net = build().eval()
    for name in ("s2_norm", "s1_norm", "dem_norm", "lulc_norm"):
        assert isinstance(getattr(net, name), nn.Identity), (
            f"{name} is not nn.Identity — the input norm defaulted back on"
        )

    b    = fake_batch(2, 2, seed=170)
    base = fwd(net, b)
    d    = (base - fwd(net, _scale_frozen(b))).abs().max().item()
    assert d > 1e-3, (
        f"scaling the frozen features by 64 moved the output only {d:.3e} — magnitude is "
        f"being discarded somewhere even with the input norms off"
    )


def test_input_norm_when_enabled_makes_the_model_scale_invariant():
    """The flag's other state. LayerNorm is scale-invariant, so with it on, multiplying every
    frozen feature by 64 must not move the output at all — which is exactly the property
    §35.26 decided to give up by default."""
    net = build(use_input_norm=True).eval()
    for name in ("s2_norm", "s1_norm", "dem_norm", "lulc_norm"):
        assert isinstance(getattr(net, name), nn.LayerNorm), f"{name} missing"

    b    = fake_batch(2, 2, seed=170)
    base = fwd(net, b)
    d    = (base - fwd(net, _scale_frozen(b))).abs().max().item()
    assert d < 1e-4, (
        f"with --input-norm, scaling the frozen features by 64 moved the output by {d:.3e} "
        f"— one of the four norms is not being applied"
    )


def test_staleness_tables_are_split_by_stream():
    """§35.26's actual fix. One shared rel_pos table had to serve driver content at std 0.22
    and frozen-token content at std 4.65 — a 21x spread no single init can suit. Two tables,
    two scales: small for drivers, full-scale for history."""
    net = build()
    assert net.rel_pos_emb.weight.shape == net.rel_pos_emb_hist.weight.shape
    drv  = net.rel_pos_emb.weight.std().item()
    hist = net.rel_pos_emb_hist.weight.std().item()
    assert drv < 0.05, f"driver staleness table init {drv:.4f}, expected ~{EMB_INIT_STD}"
    assert hist > 0.5, f"history staleness table init {hist:.4f}, expected ~{HIST_EMB_INIT_STD}"
    assert hist > 10 * drv, (
        "the two staleness tables collapsed back to a common scale — that is the bug "
        "§35.26 split them to fix"
    )


def test_driver_norm_applies_in_both_modes():
    """§35.26. driver_enc stays memory-only (§4.3: concat already has the 431x431 block in
    its joint stack), but the NORM must not be, or the two --driver-mode arms differ both in
    contextualisation and in normalisation and a difference in result cannot be attributed."""
    for mode in ("memory", "concat"):
        net = build(driver_mode=mode)
        assert isinstance(net.driver_norm, nn.LayerNorm), (
            f"driver_norm is {type(net.driver_norm).__name__} in {mode} mode"
        )
    assert len(build(driver_mode="concat").driver_enc) == 0
    assert len(build(driver_mode="memory").driver_enc) == 2


def test_every_parameter_receives_a_gradient():
    """train.py wraps in DDP WITHOUT find_unused_parameters, so any constructed-but-
    ungradiented parameter raises RuntimeError on step 1 of every multi-GPU run."""
    for mode in ("memory", "concat"):
        net = build(driver_mode=mode).train()
        b = fake_batch(2, 1, seed=180, s2_valid_frac=0.6, era5_valid_frac=0.8)
        loss, *_ = masked_huber_loss(net(b), b["label"], per_depth=True,
                                     depth_weights=torch.ones(N_DEPTHS),
                                     return_breakdown=True)
        loss.backward()
        missing = [n for n, p in net.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert not missing, f"{mode}: {len(missing)} ungradiented parameters: {missing[:6]}"


def test_depth_tokens_are_not_identical_at_init():
    """Zero-init plus no positional encoding on those slots makes attention
    permutation-equivariant over them: all three depths would be numerically identical."""
    dt = build().depth_tokens.detach()
    cos = nn.functional.normalize(dt, dim=-1) @ nn.functional.normalize(dt, dim=-1).T
    off = (cos - torch.eye(N_DEPTHS)).abs().max().item()
    assert off < 0.9, f"depth tokens are near-collinear at init (max off-diag cos {off:.3f})"


# ═════════════════════════════════════════════════════════════════════════════
# 8. LOSS INVARIANTS
# ═════════════════════════════════════════════════════════════════════════════

def test_uniform_depth_weights_equal_the_pooled_branch():
    """`w_d = 1 reduces exactly to the pooled branch` — the docstring's claim, tested."""
    torch.manual_seed(0)
    for trial in range(5):
        pred = torch.randn(6, 1, N_DEPTHS)
        label = torch.rand(6, N_DEPTHS)
        label[label < 0.25] = float("nan")               # some depths unobserved
        if torch.isnan(label).all():
            continue
        pooled = masked_huber_loss(pred, label, per_depth=False)
        for w in (None, torch.ones(N_DEPTHS), torch.full((N_DEPTHS,), 3.0)):
            weighted = masked_huber_loss(pred, label, per_depth=True, depth_weights=w)
            assert weighted.item() == pytest.approx(pooled.item(), abs=1e-6), (
                f"trial {trial}, weights={w}: per_depth={weighted.item():.8f} vs "
                f"pooled={pooled.item():.8f}"
            )


def test_non_uniform_depth_weights_do_something():
    pred = torch.zeros(4, 1, N_DEPTHS)
    label = torch.tensor([[0.1, 0.5, float("nan")]] * 4)
    a = masked_huber_loss(pred, label, per_depth=True, depth_weights=torch.ones(N_DEPTHS))
    b = masked_huber_loss(pred, label, per_depth=True,
                          depth_weights=torch.tensor([1.0, 9.0, 1.0]))
    assert b.item() > a.item()


def test_breakdown_is_a_sum_and_the_scalar_is_unchanged_by_the_flag():
    pred = torch.randn(5, 1, N_DEPTHS)
    label = torch.rand(5, N_DEPTHS)
    label[0, 1] = float("nan")
    label[3, 1] = float("nan")
    label[2, 2] = float("nan")
    plain = masked_huber_loss(pred, label, per_depth=True, depth_weights=torch.ones(N_DEPTHS))
    loss, dsum, dcnt = masked_huber_loss(pred, label, per_depth=True,
                                         depth_weights=torch.ones(N_DEPTHS),
                                         return_breakdown=True)
    assert loss.item() == plain.item(), "return_breakdown changed the scalar loss"
    assert dcnt.tolist() == [5.0, 3.0, 4.0], dcnt.tolist()
    assert torch.isfinite(dsum).all()
    assert not dsum.requires_grad and not dcnt.requires_grad


def test_breakdown_stays_finite_when_an_unobserved_depth_predicts_nan():
    """A non-finite prediction at a depth with NO label cannot affect training, but it would
    make depth_sum nan, survive all_reduce(SUM) to every rank and silently kill the very
    diagnostic the breakdown exists to provide."""
    pred = torch.randn(3, 1, N_DEPTHS)
    pred[:, 0, 2] = float("nan")
    label = torch.rand(3, N_DEPTHS)
    label[:, 2] = float("nan")
    loss, dsum, dcnt = masked_huber_loss(pred, label, return_breakdown=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(dsum).all(), dsum.tolist()
    assert dcnt.tolist() == [3.0, 3.0, 0.0]


def test_all_nan_label_gives_a_finite_zero_loss_with_gradient_path():
    pred = torch.randn(3, 1, N_DEPTHS, requires_grad=True)
    label = torch.full((3, N_DEPTHS), float("nan"))
    for kw in ({}, dict(per_depth=True, depth_weights=torch.ones(N_DEPTHS))):
        loss = masked_huber_loss(pred, label, **kw)
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()


# ═════════════════════════════════════════════════════════════════════════════
# 9. CIRCULAR DAY-OF-YEAR ENCODING
# ═════════════════════════════════════════════════════════════════════════════

def _cos(a, b):
    return float(nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))


def test_doy_pe_is_periodic_across_the_year_boundary():
    pe = circular_doy_pe(torch.arange(0, 400), dim=256)
    near = _cos(pe[1], pe[366])
    far = _cos(pe[1], pe[183])
    assert near > 0.99, f"DOY 1 vs DOY 366 cosine {near:.4f} — the year boundary is a seam"
    assert near > far + 0.1, f"near={near:.4f} not clearly above half-year far={far:.4f}"


def test_doy_pe_has_no_harmonic_above_nyquist():
    assert DOY_MAX_HARMONIC <= NYQUIST_HARMONIC, (
        f"DOY_MAX_HARMONIC={DOY_MAX_HARMONIC} exceeds the Nyquist limit "
        f"{NYQUIST_HARMONIC:.1f} for daily sampling; everything above it is a reflected copy "
        f"of a harmonic already present"
    )
    # empirical: the spectrum of the code over one year must die well before Nyquist.
    pe = circular_doy_pe(torch.arange(1, 366), dim=256)          # (365, 256)
    power = torch.fft.rfft(pe - pe.mean(0, keepdim=True), dim=0).abs().pow(2).sum(1)
    total = power.sum()
    high = power[60:].sum() / total
    assert high < 0.02, (
        f"{100 * float(high):.2f}% of the DOY code's power sits above harmonic 60 — the "
        f"linear k = 1 … dim/2 ramp is back and half the channels are aliased duplicates"
    )


def test_doy_pe_lands_at_the_embedding_init_scale():
    """Every positional / modality term is initialised at EMB_INIT_STD; the DOY code has to
    match or the driver token is ~98% calendar and ~2% weather."""
    pe = circular_doy_pe(torch.arange(1, 366), dim=256)
    s = float(pe.std())
    assert 0.4 * EMB_INIT_STD < s < 2.5 * EMB_INIT_STD, (
        f"DOY code std {s:.4f} against EMB_INIT_STD {EMB_INIT_STD}"
    )


def test_annotation_tables_are_initialised_per_stream():
    """§35.26. TWO scales, because the annotations serve two streams whose content differs in
    magnitude by ~21x (measured, §35.25):

        driver content   era5_mlp / sif_mlp / twsa_mlp output      std 0.22
        history content  raw frozen TerraMind L12 token            std 4.65

    §35.24 set all seven tables to 0.02. That fixed the drivers — nn.Embedding's default
    normal_(0, 1) had made a driver token ~450% annotation — and BROKE the history, dropping
    staleness there to 0.43% from a perfectly healthy ~21%. The input LayerNorm was then added
    to paper over the damage. Splitting the tables is the fix; this test is what stops them
    silently collapsing back to one scale."""
    net = build()

    for name in ("soil_modality_emb", "era5_modality_emb", "sif_modality_emb",
                 "twsa_modality_emb", "rel_pos_emb"):
        s = float(getattr(net, name).weight.detach().std())
        assert s < 5 * EMB_INIT_STD, (
            f"DRIVER table {name} initialised at std {s:.3f}; against std-0.22 content that "
            f"makes the token mostly calendar and ~nothing weather"
        )

    for name in ("static_modality_emb", "hist_modality_emb", "rel_pos_emb_hist"):
        s = float(getattr(net, name).weight.detach().std())
        assert s > 0.3 * HIST_EMB_INIT_STD, (
            f"HISTORY table {name} initialised at std {s:.3f}; against std-4.65 frozen "
            f"TerraMind content a 0.02 annotation is 0.43% and staleness is invisible"
        )

    assert net.rel_pos_emb.num_embeddings == 365
    assert net.rel_pos_emb_hist.num_embeddings == 365


# ═════════════════════════════════════════════════════════════════════════════
# 10. GPU (deselected by default with -m "not gpu")
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_forward_matches_cpu():
    torch.manual_seed(0)
    net = build().eval()
    b = fake_batch(2, 2, seed=190, s2_valid_frac=0.5, era5_valid_frac=0.7)
    cpu_out = fwd(net, b)
    net_cuda = build().eval().cuda()
    net_cuda.load_state_dict(net.state_dict())
    gpu_out = fwd(net_cuda, {k: (v.cuda() if torch.is_tensor(v) else v)
                             for k, v in b.items()}).cpu()
    assert (cpu_out - gpu_out).abs().max().item() < 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_masking_holds_under_bf16_autocast():
    net = build().cuda().eval()
    b = fake_batch(2, 2, seed=191, s2_valid_frac=0.5, era5_valid_frac=0.5)
    cu = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = net(cu).float()
        got = net({k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in scramble_ignored(b).items()}).float()
    assert (base - got).abs().max().item() < 1e-2
