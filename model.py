"""
SoilMoistureModel — patchwise, two transformers, no decoder
===========================================================
§34 / §35.18. Derivation with every dimension: text/patchwise_math.md.

The pooled U-Net-temporal baseline this replaced is frozen in model_unet.py (with
dataset_unet.py / train_unet.py / ckpt_utils_unet.py); tag `baseline-unet-temporal`.
Nothing here branches on it.

  T1  DriverMemoryEncoder      431 tokens, driver_layers deep, runs ONCE per sample
        era5 365 + sif 50 + twsa 12 + soil 4, each + circular_doy_pe + rel_pos_emb + a
        modality tag; then self-attention so the driver days can see each other.
        Tile-level, so it carries NO patch index — which is what makes the cache exact.
        -> m (B, 431, 768), then Kc_l = m.Wk_c(l), Vc_l = m.Wv_c(l) for each T2 layer.

  T2  PatchwiseBlock x n_layers   105 tokens, runs K times (K folded into the batch)
        [ depth_CLS x3 | dem_k | lulc_k | hist_k x100 ]
        per layer:  SelfAttn(105)  ->  CrossAttn(Q=105, K/V=cached 431)  ->  FFN
        Statics are a PREFIX so temporal attention can condition drydown on cover/terrain.
        History carries staleness + modality only: no scale_emb (it indexed pyramid levels)
        and no absolute DOY (ERA5 is the seasonal anchor).

  Readout: each depth CLS token attends with its OWN query, and its output row IS that
        depth's prediction -> Linear(768,1) per depth -> (B, K, n_depths) at 160 m.
        196 patches = a 14x14 map. STEP 1 has NO decoder, and there is no star residual.

  Weight sharing across patches is the mechanism, not an optimisation: supervising the
  station's single token teaches the mapping at all 196 (§34.4). Training runs K=1
  (token 105), inference K=196, same checkpoint.

  --driver-mode concat puts all 536 tokens (105 + 431) in one self-attention stack instead,
  and does NOT build T1 at all. Kept as an option because it cannot be retrofitted
  (§3.4 of the maths doc), but not run.

  Loss: Huber on (B, K, n_depths) against the ISMN label, NaN depths masked. There is no
  map and nothing to index.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth: drop the entire layer residual with probability drop_prob."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        noise = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep).div_(keep)
        return x * noise


class DropPathTransformerLayer(nn.Module):
    """nn.TransformerEncoderLayer wrapped with stochastic depth on the combined residual."""
    def __init__(self, layer: nn.TransformerEncoderLayer, drop_prob: float = 0.0):
        super().__init__()
        self.layer = layer
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        y = self.layer(x, src_key_padding_mask=src_key_padding_mask)
        if self.drop_path.drop_prob > 0.0:
            return x + self.drop_path(y - x)
        return y


# ── Positional encoding ──────────────────────────────────────────────────────

# Harmonic ceiling for circular_doy_pe. Daily sampling puts the Nyquist limit at k = 182
# (= 365.25 / 2); every harmonic above it is a reflected copy of one below, and k and
# k + 365 are near-degenerate, so the old linear ramp k = 1 … 384 spent more than half its
# channels on aliased duplicates of harmonics it already had. 26 is ~2-week resolution,
# comfortably inside Nyquist, and is where the seasonal signal actually lives.
DOY_MAX_HARMONIC = 26

# Every positional / modality embedding is initialised at this scale. nn.Embedding defaults
# to normal_(0, 1), which put rel_pos_emb, the modality tags and the DOY code into the
# residual stream at sigma ~1.0 against an era5_mlp output of sigma ~0.22 — i.e. the driver
# token entering T1 was ~98% calendar and ~2% weather, and driver_norm then normalised that
# mixture. 0.02 is the ViT/BERT convention (and is already what depth_tokens uses), which
# puts content back in charge at initialisation.
EMB_INIT_STD = 0.02


def circular_doy_pe(doys: torch.Tensor, dim: int = 768,
                    scale: float = EMB_INIT_STD) -> torch.Tensor:
    """
    Circular positional encoding for day-of-year. Periodic at 365.25 days so
    DOY 365 and DOY 1 share similar representations (no year-boundary seam).

    Harmonics are geometrically spaced INTEGERS in [1, DOY_MAX_HARMONIC]: integer so the
    code stays exactly periodic at 365.25 days, geometric so the low frequencies get most
    of the channels, capped so nothing aliases.

    doys : (N,) long tensor of day-of-year values [1, 365]
    returns (N, dim) float, per-channel std ≈ `scale`
    """
    device = doys.device
    base   = 2.0 * math.pi / 365.25
    k      = torch.round(torch.exp(torch.linspace(
        0.0, math.log(DOY_MAX_HARMONIC), dim // 2, device=device))).float()   # (dim//2,)
    angles = doys.float().unsqueeze(1) * base * k                     # (N, dim//2)
    pe     = torch.zeros(len(doys), dim, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    # A sin/cos pair has RMS 1/sqrt(2) over uniformly distributed DOYs; rescale so the code
    # lands at `scale`, matching every other positional term (see EMB_INIT_STD).
    return pe * (scale * math.sqrt(2.0))                               # (N, dim)


class PatchwiseBlock(nn.Module):
    """
    One layer of the patch decoder (transformer 2 of the two-transformer design, §35.18).

        x = x + SelfAttn (LN(x))                 over the patch's own 105 tokens
        x = x + CrossAttn(LN(x), memory)         memory mode only — reads the 431 driver tokens
        x = x + FFN      (LN(x))

    Cross-attention is written out by hand rather than with nn.MultiheadAttention, and that is
    the whole point of the design. MHA runs `in_proj` on whatever it is handed, so passing an
    .expand()ed memory would re-project the same 431 driver tokens once per patch — exactly the
    duplication the cache exists to remove. Here `k_proj`/`v_proj` are called by the parent, ONCE
    per sample, and this block receives the projected (Kc, Vc) directly.

    The queries are reshaped to (B, h, K*L, dh) rather than expanding the memory to (B*K, ...).
    All K patches of a sample share one memory, so folding K into the query length is exact and
    allocates nothing; expanding the memory would cost B*K*h*M*dh (≈2 GB at K=196).
    """

    def __init__(self, d_model: int, n_heads: int, driver_mode: str = "memory",
                 dropout: float = 0.1, drop_path: float = 0.0, n_readout: int = 3,
                 hist_start: int = 5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim    = d_model // n_heads
        self.driver_mode = driver_mode
        self.n_readout   = n_readout
        # First HISTORY column of the sequence: everything before it is the depth CLS prefix
        # plus dem/lulc. Only the history columns are meaningful to the collapse detector.
        self.hist_start  = hist_start
        self.attn_drop   = dropout

        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)

        if driver_mode == "memory":
            self.norm_cross = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)   # called by the PARENT, once per sample
            self.v_proj = nn.Linear(d_model, d_model)   # ditto — never inside the patch loop
            self.o_proj = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        # Residual dropout on all three sub-blocks. Without it the only dropout in T2 was
        # inside MultiheadAttention's weights and mid-FFN, so the 6-layer, ~60 M-parameter
        # half of the model was materially LESS regularised than the 2-layer T1, which gets
        # full residual dropout from nn.TransformerEncoderLayer.
        self.resid_drop = nn.Dropout(dropout)

        # One DropPath draw per residual branch — and this block has THREE branches (self,
        # cross, FFN) where a standard ViT block has two. `drop_path` is a per-LAYER rate
        # from the dpr schedule, so the per-branch rate must be deflated or effective
        # survival is (1-p)^3, nearly 3x the intended drop at the deepest layer.
        branch_p = 1.0 - (1.0 - drop_path) ** (1.0 / 3.0)
        self.drop_path = DropPath(branch_p)

        # Set by the parent when train.py asks for a diagnostic pass. Off by default: collecting
        # weights forces the math kernel and gives up SDPA.
        self.collect_entropy = False
        self.last_entropy: torch.Tensor | None = None

    def _cross(self, x, kc, vc, mem_pad, B, K):
        """
        x       (B*K, L, d)      queries, patch tokens
        kc, vc  (B, M, d)        ALREADY projected by the parent — one copy per sample
        mem_pad (B, M) bool      True = ignore
        """
        N, L, d = x.shape
        h, dh   = self.n_heads, self.head_dim
        M       = kc.shape[1]

        q = self.q_proj(x)                                       # (B*K, L, d)
        # (B*K, L, d) -> (B, K*L, h, dh) -> (B, h, K*L, dh); no copy of the memory anywhere.
        q = q.reshape(B, K * L, h, dh).transpose(1, 2)           # (B, h, K*L, dh)
        k = kc.reshape(B, M, h, dh).transpose(1, 2)              # (B, h, M,   dh)
        v = vc.reshape(B, M, h, dh).transpose(1, 2)

        attn_mask = None
        if mem_pad is not None:
            # SDPA bool mask: True = participate. (B,1,1,M) broadcasts over heads and queries.
            attn_mask = (~mem_pad)[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )                                                        # (B, h, K*L, dh)
        out = out.transpose(1, 2).reshape(N, L, d)
        return self.o_proj(out)

    def forward(self, x, kc=None, vc=None, mem_pad=None, self_pad=None, B=None, K=None,
                hist_end=None):
        h = self.norm_self(x)
        if self.collect_entropy:
            # average_attn_weights=False: head-averaging BEFORE the entropy is what made the
            # detector blind. Twelve heads each sharply peaked on a different handful of
            # tokens average to something numerically indistinguishable from uniform.
            a, w = self.self_attn(h, h, h, key_padding_mask=self_pad,
                                  need_weights=True, average_attn_weights=False)
            self.last_entropy = _row_entropy(
                w, self_pad, self.n_readout, self.hist_start,
                hist_end if hist_end is not None else x.shape[1]).detach()
        else:
            a, _ = self.self_attn(h, h, h, key_padding_mask=self_pad, need_weights=False)
        x = x + self.drop_path(self.resid_drop(a))

        if self.driver_mode == "memory":
            x = x + self.drop_path(self.resid_drop(
                self._cross(self.norm_cross(x), kc, vc, mem_pad, B, K)))

        x = x + self.drop_path(self.resid_drop(self.ffn(self.norm_ffn(x))))
        return x


def _row_entropy(w: torch.Tensor, pad: torch.Tensor | None,
                 n_readout: int, hist_start: int, hist_end: int) -> torch.Tensor:
    """
    Collapse statistic for the readout rows — accumulated, not sampled.

    This is the detector §35.20 step 1 relies on, and it is the SOLE evidence separating
    "un-pooling genuinely does not help" from "attention collapsed". Register-dominated keys
    make q.k near-constant, which shows up here as entropy pinned at the uniform value — and
    NOT in the map SD, which is why it has to be logged explicitly.

    w           (N, h, L, L)  attention weights, PER HEAD (not head-averaged)
    pad         (N, L)        True = ignored key
    n_readout                 leading rows that are readouts (the depth CLS tokens)
    hist_start                first history column; the depth-CLS/dem/lulc prefix is excluded
    hist_end                  one past the last history column. In CONCAT mode the sequence
                              continues into 431 driver tokens after the 105 patch ones, and
                              an open-ended slice would silently fold the weather into the
                              history entropy — making the two driver modes' numbers
                              incomparable, which is the one comparison the arm exists for.

    Returns (3,) float32:  [sum_entropy_nats, sum_ratio, count]  over (sample, head,
    readout-row) triples, for the caller to accumulate over the epoch and all_reduce(SUM).
    `ratio` is entropy / log(n_valid_hist), so 1.0 is exactly uniform — collapsed —
    whatever that sample's valid-slot count happened to be.

    Three defects this replaces, all of which made a collapsed run readable as healthy:
    the weights were head-averaged first; the row spanned the 5 non-history prefix columns;
    and the result was compared against a fixed log(100) although the median station-year
    carries ~36 of 60 S2 slots, so a fully collapsed row scored ~4.0 against a 4.605
    "collapse" threshold and passed.
    """
    rows = w[:, :, :n_readout, hist_start:hist_end]                  # (N, h, R, H)
    if pad is not None:
        keep    = (~pad[:, hist_start:hist_end])[:, None, None, :]   # (N, 1, 1, H)
        rows    = rows * keep
        n_valid = keep.reshape(rows.shape[0], -1).sum(-1)            # (N,)
    else:
        n_valid = torch.full((rows.shape[0],), rows.shape[-1],
                             device=rows.device, dtype=torch.long)

    # Renormalise over the history columns alone. The row is a softmax over ALL L keys, so
    # without this step the entropy is partly a measure of how much mass leaked to the
    # prefix rather than of how sharply the history is being read.
    p   = rows / rows.sum(-1, keepdim=True).clamp_min(1e-9)
    ent = -(p.clamp_min(1e-9).log() * p).sum(-1)                     # (N, h, R)

    # A sample with <2 valid history slots has no meaningful entropy and log(1) = 0 would
    # divide by zero; drop those from both the sum and the count.
    ok    = (n_valid >= 2)[:, None, None].expand_as(ent)
    ref   = n_valid.clamp_min(2).float().log()[:, None, None]        # (N, 1, 1)
    ent   = torch.where(ok, ent, torch.zeros_like(ent))
    ratio = torch.where(ok, ent / ref, torch.zeros_like(ent))
    return torch.stack([ent.sum().float(), ratio.sum().float(), ok.sum().float()])


# ── Soil encoder ─────────────────────────────────────────────────────────────

class SoilEncoder(nn.Module):
    """
    Lightweight depthwise-separable CNN + 4-scale spatial pyramid.

    Input : (B, 21, 74, 74) float32 — NaN-free (pre-filled by dataset)
    Output: (B,  4, 768)    float32 — 4 static soil tokens
    ~211 K parameters

    Architecture (from architecture.md §4d):
      Block 1: DWConv(21, 3×3) → PWConv(21→32) → BN → GELU  # (B,32,74,74)
      Block 2: DWConv(32, 3×3, s=2) → PWConv(32→64) → BN → GELU  # (B,64,37,37)
      Pyramid: centre 1×1 / 3×3 / 7×7 / full 37×37 → mean → Linear(64→768)
    """
    IN_CH  = 21
    MID_CH = 32
    OUT_CH = 64

    def __init__(self, d_model: int = 768):
        super().__init__()
        c = self.OUT_CH
        self.block1 = nn.Sequential(
            nn.Conv2d(self.IN_CH,  self.IN_CH,  3, padding=1, groups=self.IN_CH,  bias=False),
            nn.Conv2d(self.IN_CH,  self.MID_CH, 1, bias=False),
            nn.BatchNorm2d(self.MID_CH),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.MID_CH, self.MID_CH, 3, stride=2, padding=1, groups=self.MID_CH, bias=False),
            nn.Conv2d(self.MID_CH, c,           1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.proj = nn.ModuleList([nn.Linear(c, d_model) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.block1(x)                                         # (B, 32, 74, 74)
        x  = self.block2(x)                                         # (B, 64, 37, 37)
        cy = cx = 18                                                 # centre of 37×37
        # Scales: input is 30 m/px, but block2 has stride 2 → cells here are 60 m, and each
        # cell has a 5-input-px receptive field. So a k×k window spans k×60 m and sees
        # (5 + 2(k-1))×30 m of input. (Earlier comments assumed 30 m cells — 2× too small.)
        t0 = x[:, :, cy:cy+1,   cx:cx+1  ].mean(dim=(-2, -1))     # 1×1   win 60 m,  RF 150 m
        t1 = x[:, :, cy-1:cy+2, cx-1:cx+2].mean(dim=(-2, -1))     # 3×3   win 180 m, RF 270 m
        t2 = x[:, :, cy-3:cy+4, cx-3:cx+4].mean(dim=(-2, -1))     # 7×7   win 420 m, RF 510 m
        t3 = x.mean(dim=(-2, -1))                                   # 37×37 win 2.22 km = full patch
        return torch.stack(
            [self.proj[i](t) for i, t in enumerate([t0, t1, t2, t3])], dim=1
        )                                                            # (B, 4, 768)


# ── Full model ───────────────────────────────────────────────────────────────

class SoilMoistureModel(nn.Module):
    """
    Args:
        n_depths      : number of SM depth bins (3)
        d_model       : token dimension (768)
        n_heads       : attention heads (12)
        n_layers      : T2 patch-decoder layers (6)
        driver_layers : T1 weather-encoder depth (2). A DEPTH, not a repeat count — T1 runs
                        once per sample whatever its depth (§35.19).
        driver_mode   : "memory" (read-only cross-attended drivers, K/V cached once per
                        sample) or "concat" (all 536 = 105 + 431 in one self-attention
                        stack; T1 is not built at all in that mode).
        head_bias_init: per-depth initial head bias in m3/m3, SM_DEPTHS order. train.py
                        passes the train-set means from csvs/driver_stats.json.
    """

    def __init__(
        self,
        n_depths:      int   = 3,
        d_model:       int   = 768,
        n_heads:       int   = 12,
        n_layers:      int   = 6,
        drop_path_rate: float = 0.1,
        use_cls_depth:  bool  = True,
        driver_mode:    str   = "memory",
        driver_layers:  int   = 2,
        head_bias_init: list[float] | None = None,
    ):
        super().__init__()
        if driver_mode not in ("memory", "concat"):
            raise ValueError(f"driver_mode must be 'memory' or 'concat', got {driver_mode!r}")
        self.d_model       = d_model
        self.n_depths      = n_depths
        self.use_cls_depth = use_cls_depth
        self.driver_mode   = driver_mode
        self.driver_layers = driver_layers

        # ── Encoders ──────────────────────────────────────────────────
        self.soil_encoder = SoilEncoder(d_model=d_model)

        self.era5_mlp = nn.Sequential(
            nn.Linear(19, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model),
        )

        # SIF and TWSA MLP encoders (scalar → token)
        self.sif_mlp  = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))
        self.twsa_mlp = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))

        # ── Modality type embeddings ───────────────────────────────────
        # Soil tokens
        self.soil_modality_emb = nn.Embedding(1, d_model)
        # Static (DEM=0, LULC=1)
        self.static_modality_emb = nn.Embedding(2, d_model)
        # Satellite history: 0 = S2, 1 = S1 ascending, 2 = S1 descending.
        #
        # THREE, not two. dataset.py merges the two S1 orbits into one date-sorted list, and
        # RTC backscatter differs systematically between ascending and descending (incidence
        # angle, look azimuth, local geometry) by an amount comparable to the moisture signal.
        # With a single shared S1 tag an orbit switch is indistinguishable from a wetting
        # event, so that variance gets attributed to soil moisture.
        self.hist_modality_emb = nn.Embedding(3, d_model)
        # ERA5 temporal tokens
        self.era5_modality_emb = nn.Embedding(1, d_model)
        # Optional sparse modalities
        self.sif_modality_emb  = nn.Embedding(1, d_model)
        self.twsa_modality_emb = nn.Embedding(1, d_model)

        # Learned relative position embedding: staleness within the 365-day window.
        # Indexed by the dataset's `*_rel_pos` (364 = the target day), NEVER by slot index —
        # see the note in _build_driver_tokens.
        self.rel_pos_emb = nn.Embedding(365, d_model)

        # Every one of the seven tables above is a learned *input*, not a weight matrix, and
        # they are summed into content that leaves its MLP at sigma ~0.22. nn.Embedding's
        # default normal_(0, 1) therefore made the token ~98% annotation and ~2% content.
        # trunc_normal_(0.02) is the ViT/BERT convention and what depth_tokens already used.
        for emb in (self.soil_modality_emb, self.static_modality_emb, self.hist_modality_emb,
                    self.era5_modality_emb, self.sif_modality_emb, self.twsa_modality_emb,
                    self.rel_pos_emb):
            nn.init.trunc_normal_(emb.weight, std=EMB_INIT_STD)

        # Day-of-year code, precomputed. It is a pure function of one integer in [0, 366], so
        # recomputing (B·365, 768) sin/cos every forward was tens of MB of allocation and a
        # transcendental pass per step for a 367-row lookup table. Registered as a buffer so
        # it follows .to(device) and lands in the checkpoint's device placement, but with
        # persistent=False so it does not bloat the state_dict or break older checkpoints.
        self.register_buffer("doy_pe",
                             circular_doy_pe(torch.arange(367), d_model),
                             persistent=False)

        # Input normalisation for the FROZEN TerraMind features.
        #
        # These arrive at whatever scale the upstream encoder produced — and per §35.3's
        # register audit that scale is large and dominated by a handful of register
        # dimensions. They were being added straight to rel_pos_emb / hist_modality_emb and
        # carried at raw magnitude through the residual stream into the FFN and the readout;
        # only the attention *input* was ever normalised, by norm_self.
        #
        # This is required for EMB_INIT_STD to mean anything on the satellite tokens: a
        # positional code at std 0.02 against an unnormalised frozen feature is invisible,
        # which would trade the driver-token problem for the same problem on the history.
        # One LayerNorm per stream, not one shared, so each sensor keeps its own gain.
        self.s2_norm   = nn.LayerNorm(d_model)
        self.s1_norm   = nn.LayerNorm(d_model)
        self.dem_norm  = nn.LayerNorm(d_model)
        self.lulc_norm = nn.LayerNorm(d_model)

        # Stochastic-depth schedule, shared by T1 and T2.
        dpr = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]

        # ── Depth-specific CLS tokens (one per depth, attend across all tokens) ──
        if use_cls_depth:
            self.depth_tokens = nn.Parameter(torch.zeros(n_depths, d_model))
            # Zero-init would make all depth queries numerically identical, and no positional
            # encoding is added to these slots — attention is permutation-equivariant over
            # them, so depth_ctx[:,0,:] == depth_ctx[:,1,:] == ... exactly at step 0. Random
            # init gives each depth a distinct query from the first step.
            nn.init.trunc_normal_(self.depth_tokens, std=0.02)

        # ── Two transformers (§35.18, text/patchwise_math.md) ──
        #
        #   T1 driver_enc   431 tokens, driver_layers deep, runs ONCE per sample
        #   T2 patch_blocks 105 tokens, n_layers deep, runs K times (K folded into batch)
        #
        # STEP 1 has no decoder at all (§34.4): the prediction is the token head at 160 m.
        if not use_cls_depth:
            raise ValueError("use_cls_depth is required: the per-patch sequence carries "
                             "the depth CLS tokens as a prefix.")

        # T1 — the weather encoder. Depth here processes only tile-constant drivers, so it
        # is deliberately shallower than T2 (§35.19: 6 would take the model to 100.4 M,
        # 2.0x the 50.35 M baseline, confounding capacity with architecture).
        #
        # MEMORY MODE ONLY. T1's justification (§4.3 of the maths doc) is that it restores the
        # 431x431 "weather reads weather" block, which the memory design would otherwise lose
        # — concat already has that block inside its joint stack. Running T1 unconditionally
        # gave concat the block twice, made its sequence 536 rather than the 537 the doc
        # derives, and fed it a LayerNorm'd memory against raw patch tokens, biasing the very
        # softmax-budget competition §7/§8 says the arm exists to measure.
        #
        # T1's stochastic-depth rates are scaled against T2's depth, not against T1's own.
        # Normalising by (driver_layers - 1) put HALF of a 2-layer encoder at the maximum
        # drop rate — ddpr = [0.0, 0.1] — which is a great deal of stochastic depth for two
        # layers. Against n_layers the same schedule gives [0.0, 0.02].
        ddpr = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(driver_layers)]
        self.driver_enc = nn.ModuleList([
            DropPathTransformerLayer(
                nn.TransformerEncoderLayer(
                    d_model         = d_model,
                    nhead           = n_heads,
                    dim_feedforward = d_model * 4,
                    dropout         = 0.1,
                    batch_first     = True,
                    norm_first      = True,
                ),
                drop_prob = ddpr[i],
            )
            for i in range(driver_layers)
        ]) if driver_mode == "memory" else nn.ModuleList()
        self.driver_norm = nn.LayerNorm(d_model) if driver_mode == "memory" else nn.Identity()

        # T2 — the patch decoder. hist_start = n_depths depth-CLS rows + dem + lulc.
        self.hist_start = n_depths + 2
        self.patch_blocks = nn.ModuleList([
            PatchwiseBlock(d_model, n_heads, driver_mode=driver_mode,
                           dropout=0.1, drop_path=dpr[i], n_readout=n_depths,
                           hist_start=self.hist_start)
            for i in range(n_layers)
        ])
        self.patch_norm = nn.LayerNorm(d_model)

        # Independent depth heads, one per depth, reading that depth's OWN CLS row.
        #
        # NOT §18.4's star residual (`depth_d = base + offset_d`), which was a sample-efficiency
        # bias rather than a data necessity (§35.8). And no FiLM either: an earlier draft ran
        # head_i(FiLM1d_i(patch_cls, depth_ctx_i)), but FiLM earned its place in the U-Net by
        # broadcasting one context vector across a (B,C,H,W) map — here both operands are
        # (N, 768) vectors from the same transformer, so modulation buys nothing a direct
        # readout does not already have. It also cost 3 x 1.18 M parameters and, being
        # identity-initialised, started all three depths reading the identical vector.
        #
        # Each depth CLS is a full readout over all 105 tokens with its own learned query, so
        # a separate patch CLS was strictly redundant with them and is gone too.
        self.depth_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_depths)])

        # Initialise each head's bias to that depth's TRAIN-SET mean soil moisture.
        #
        # Labels are raw m3/m3 (~0.25 typical), so a default bias of U(±0.036) starts every
        # prediction ~0.2 away from the truth — far outside Huber's delta=0.05, which means
        # the loss opens in its LINEAR regime with a constant +/-delta gradient carrying no
        # information about how wrong the prediction is. The first epochs then go on walking
        # three scalars to the data mean while the collapse diagnostics report on an
        # attention pattern that has not started training yet.
        if head_bias_init is not None:
            if len(head_bias_init) != n_depths:
                raise ValueError(f"head_bias_init needs {n_depths} values, "
                                 f"got {len(head_bias_init)}")
            with torch.no_grad():
                for i, b in enumerate(head_bias_init):
                    self.depth_heads[i].bias.fill_(float(b))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_driver_tokens(self, batch: dict):
        """
        TRANSFORMER 1's input: the 431 tile-level driver tokens.

            era5 365 + sif 50 + twsa 12 + soil 4 = 431

        Built the way the pooled baseline's `_build_sequence` did (it lives in model_unet.py
        now, tag `baseline-unet-temporal`) — same MLPs, same DOY code, same rel_pos_emb, same
        modality embeddings, same padding rules, so the patchwise arm stays comparable to it.
        Two deliberate departures, both from §35.24: the DOY code is a precomputed table
        rather than a per-forward sin/cos pass, and ERA5's rel_pos comes from the dataset's
        real row dates instead of the slot index.

        These tokens carry NO patch index: ERA5 is 9 km and the tile is 2.24 km, so one grid cell
        covers the whole tile. That is what makes the K/V cache exact (text/patchwise_math.md §2.4).

        Returns (m_raw (B, 431, 768), pad (B, 431) bool  True = ignore).
        """
        # From the PARAMETERS, not from the batch. Taking it from batch["era5"] made every
        # subsequent .to(device) a no-op moving a tensor to where it already was — so a batch
        # that arrived on CPU would not have been moved to the GPU, it would have failed deep
        # inside era5_mlp on a CPU-vs-CUDA matmul. Now the .to() calls are real.
        device = next(self.parameters()).device
        era5   = batch["era5"].to(device)
        B      = era5.shape[0]
        toks, pads = [], []

        # ── soil (4) ───────────────────────────────────────────────────────
        soil_tok = self.soil_encoder(batch["soil_patch"].to(device))          # (B, 4, 768)
        soil_tok = soil_tok + self.soil_modality_emb(
            torch.zeros(1, dtype=torch.long, device=device))
        toks.append(soil_tok)
        pads.append(torch.zeros(B, soil_tok.shape[1], device=device, dtype=torch.bool))

        # ── ERA5 (365) ─────────────────────────────────────────────────────
        era5_doys = batch["era5_doys"].to(device)
        era5_tok  = self.era5_mlp(era5)
        # rel_pos comes from the dataset's REAL row dates, never from the slot index.
        # load_era5_rolling right-aligns a *compacted* window, so slot index equals staleness
        # only when the 365-day window happens to have no interior missing day — and the
        # admission guard used to be year-granular, which let a months-old row sit at slot
        # 364 and be embedded as "today's weather". S2/S1/SIF/TWSA all carry a real
        # per-observation rel_pos; ERA5 was the only modality that did not.
        era5_rel = batch["era5_rel_pos"].to(device).reshape(-1).clamp(0, 364)
        era5_tok  = (era5_tok
                     + self.doy_pe[era5_doys.reshape(-1).clamp(0, 366)
                                   ].reshape(B, 365, self.d_model)
                     + self.rel_pos_emb(era5_rel).reshape(B, 365, self.d_model)
                     + self.era5_modality_emb(torch.zeros(1, dtype=torch.long, device=device)))
        toks.append(era5_tok)
        pads.append((era5_doys == 0).to(device))

        # ── SIF (50) and TWSA (12), both sparse ────────────────────────────
        # No `if vals.shape[1] == 0: continue` guard. MAX_SIF/MAX_TWSA are module constants so
        # the branch was unreachable, and taking it would have dropped sif_mlp/twsa_mlp out of
        # the autograd graph entirely — a hard DDP error, since train.py builds DDP without
        # find_unused_parameters. Empty slots are handled by the `valid` mask, as elsewhere.
        for key, mlp, mod_emb in (
            ("sif",  self.sif_mlp,  self.sif_modality_emb),
            ("twsa", self.twsa_mlp, self.twsa_modality_emb),
        ):
            vals    = batch[key].to(device)
            doys    = batch[f"{key}_doys"].to(device)
            rel_pos = batch[f"{key}_rel_pos"].to(device)
            valid   = batch[f"{key}_valid"].to(device)
            tok = (mlp(vals.float())
                   + self.doy_pe[doys.reshape(-1).clamp(0, 366)
                                 ].reshape(B, -1, self.d_model)
                   + self.rel_pos_emb(rel_pos.reshape(-1).clamp(0, 364)
                                      ).reshape(B, -1, self.d_model)
                   + mod_emb(torch.zeros(1, dtype=torch.long, device=device)))
            toks.append(tok)
            pads.append(~valid)

        return torch.cat(toks, dim=1), torch.cat(pads, dim=1)

    def _build_patch_seq(self, batch: dict):
        """
        TRANSFORMER 2's input, per patch k:

            [ depth_CLS x3 | dem_k | lulc_k | hist_k x100 ]   = 105 tokens

        The depth CLS tokens ARE the readouts — each attends with its own query and its output
        is that depth's prediction. There is no separate patch CLS.

        Statics enter as a PREFIX, not appended to the summary, so temporal attention can
        condition drydown on cover and terrain (§34.3). History carries staleness and modality
        only — no scale_emb (it indexed pyramid levels, meaningless un-pooled) and no absolute
        DOY (ERA5 is the seasonal anchor; the driver tokens carry the season, the history
        carries only how long ago it was observed).

        Returns (x (B, K, 105, 768), pad (B, K, 105) bool  True = ignore).
        """
        device = next(self.parameters()).device
        d      = self.d_model
        # Frozen TerraMind features are LayerNorm'd on the way in — see the tm-norm comment
        # in __init__. Without it the raw register-dominated magnitude flows through the
        # residual stream to the FFN and the readout, and swamps the positional codes.
        dem    = self.dem_norm(batch["dem_tok"].to(device).float())        # (B, K, 768)
        B, K   = dem.shape[:2]

        blocks, pads = [], []

        # depth CLS prefix — (n_depths, d) shared across patches
        blocks.append(self.depth_tokens.view(1, 1, self.n_depths, d).expand(B, K, -1, -1))
        pads.append(torch.zeros(B, K, self.n_depths, device=device, dtype=torch.bool))

        # per-patch statics.
        #
        # dem_valid / lulc_valid come from the nodata masks that the dataset already computed
        # and used to drop on the floor, leaving model.py to hardcode "statics are always
        # valid". A tile with DEM void-fill, or a station with no DEM in the zarr at all
        # (which the dataset padded with an all-zero token), was feeding a fabricated
        # elevation embedding into the PREFIX of every patch sequence — and terrain is the
        # §34.3 mechanism the architecture rests on.
        static_w = self.static_modality_emb.weight                          # (2, d)
        blocks.append((dem + static_w[0]).unsqueeze(2))                     # (B, K, 1, d)
        blocks.append((self.lulc_norm(batch["lulc_tok"].to(device).float())
                       + static_w[1]).unsqueeze(2))
        pads.append(torch.stack([
            ~batch["dem_valid"].to(device).bool(),
            ~batch["lulc_valid"].to(device).bool(),
        ], dim=-1))                                                         # (B, K, 2)

        # per-patch history: S2 then S1
        for hist_idx, (key, tok_norm) in enumerate((("s2", self.s2_norm),
                                                    ("s1", self.s1_norm))):
            h = batch[f"{key}_hist"].to(device).float()                     # (B, T, K, 768)
            h = tok_norm(h)                                                 # frozen-feature LN
            h = h.permute(0, 2, 1, 3)                                       # (B, K, T, 768)
            rel = self.rel_pos_emb(
                batch[f"{key}_rel_pos"].to(device).reshape(-1).clamp(0, 364)
            ).reshape(B, 1, -1, d)                                          # (B, 1, T, d)
            if key == "s2":
                mod = self.hist_modality_emb(
                    torch.zeros(1, dtype=torch.long, device=device)).view(1, 1, 1, d)
            else:
                # 1 = S1 ascending, 2 = S1 descending; see the hist_modality_emb comment.
                orb = batch["s1_orbit"].to(device).long().clamp(0, 1)       # (B, T)
                mod = self.hist_modality_emb(orb + 1).unsqueeze(1)          # (B, 1, T, d)
            blocks.append(h + rel + mod)
            # dataset.py's _finalise_history already ANDs the token mask with (doys > 0), so
            # this covers padded slots, NaN-skipped acquisitions, and — since §35.24 made the
            # cloud mask fail closed — acquisitions with no cloud-mask entry alike.
            pads.append(~batch[f"{key}_hist_valid"].to(device).permute(0, 2, 1))

        return torch.cat(blocks, dim=2), torch.cat(pads, dim=2)

    def _forward_patchwise(self, batch: dict) -> torch.Tensor:
        """Returns (B, K, n_depths) — one soil-moisture value per depth per patch, at 160 m."""
        # ── T1: encode the weather ONCE, then project the cache ONCE ───────
        m, mem_pad = self._build_driver_tokens(batch)
        # T1 runs in memory mode only — see the driver_enc comment in __init__. In concat mode
        # the raw driver tokens join the joint self-attention stack, which is what §3 derives.
        for layer in self.driver_enc:
            m = layer(m, src_key_padding_mask=mem_pad)
        m = self.driver_norm(m)                                             # (B, 431, 768)

        # k_proj/v_proj live on the blocks but are called HERE, outside the patch loop. This is
        # the entire cost argument: 2*431*d^2 once per sample instead of once per patch.
        kv = ([(blk.k_proj(m), blk.v_proj(m)) for blk in self.patch_blocks]
              if self.driver_mode == "memory" else None)

        # ── T2: run every patch against that cache ─────────────────────────
        x, pad = self._build_patch_seq(batch)                               # (B,K,105,d)
        B, K, L, d = x.shape
        # One past the last HISTORY column, captured BEFORE concat mode appends the 431 driver
        # tokens. The entropy detector must span the same columns in both driver modes or the
        # two arms' numbers cannot be compared — which is the only reason the arms exist.
        hist_end = L
        x   = x.reshape(B * K, L, d)
        pad = pad.reshape(B * K, L)

        if self.driver_mode == "concat":
            # No cache and no cross-attention: the memory joins the self-attention sequence, so
            # every patch carries its own copy of all 431 driver tokens (T = 105 + 431 = 536).
            mem = m.unsqueeze(1).expand(B, K, -1, -1).reshape(B * K, -1, d)
            x   = torch.cat([x, mem], dim=1)
            pad = torch.cat([pad, mem_pad.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)], 1)

        ent = []
        for i, blk in enumerate(self.patch_blocks):
            kc, vc = kv[i] if kv is not None else (None, None)
            x = blk(x, kc, vc, mem_pad, pad, B, K, hist_end)
            if blk.collect_entropy and blk.last_entropy is not None:
                ent.append(blk.last_entropy)
        # Diagnostic stash, read by train.py. §35.20 made this the SOLE detector for
        # register-driven attention collapse, so it is emitted as SUMS and a COUNT per layer —
        # (n_layers, 3) = [sum_entropy_nats, sum_ratio, count] — for the caller to accumulate
        # over the whole val epoch and all_reduce(SUM). It used to be overwritten on every
        # forward, so what actually reached W&B was one batch on rank 0.
        #
        # `ent` collects only blocks that were ARMED, so arming a subset would return a
        # shorter tensor that train.py's per-layer logging would mislabel — layer 4's number
        # reported under layer 0's name, silently. Arm all or arm none.
        if ent and len(ent) != len(self.patch_blocks):
            raise RuntimeError(
                f"collect_entropy was set on {len(ent)} of {len(self.patch_blocks)} patch "
                f"blocks. The (n_layers, 3) diagnostic contract requires all or none — a "
                f"partial arm produces a tensor whose rows do not correspond to layer index."
            )
        self._last_attn_entropy = torch.stack(ent) if ent else None

        x = self.patch_norm(x)
        # The depth CLS rows are the first n_depths tokens in BOTH driver modes (concat appends
        # the memory after the patch tokens, so the prefix is untouched).
        depth_ctx = x[:, :self.n_depths, :]                       # (B*K, n_depths, d)

        # Collapse diagnostic for the depth heads, as a SUM plus its count so train.py can
        # accumulate across the epoch. The OUTPUT cosine is the one that matters: the input
        # depth_tokens can stay near-orthogonal (they are excluded from weight decay, so they
        # will) while these three collapse to the same vector, which is exactly use_cls_depth
        # being inert. The producer was lost with the U-Net strip and train.py has been
        # reading a getattr default ever since, logging nothing.
        self._last_depth_ctx   = depth_ctx.detach().float().sum(0)          # (n_depths, d)
        self._last_depth_ctx_n = depth_ctx.shape[0]

        # Readout in fp32, outside autocast. Under bf16 the head's output carries ~0.2%
        # relative precision — ~1e-3 absolute at SM = 0.5 — while epoch-to-epoch checkpoint
        # decisions are made on val differences of order 1e-5. Three Linear(768, 1) layers
        # cost nothing to run unautocast, and it takes the quantisation out of every number
        # that ends up in a table.
        with torch.autocast(device_type=depth_ctx.device.type, enabled=False):
            dc  = depth_ctx.float()
            out = torch.cat([
                self.depth_heads[i](dc[:, i, :])
                for i in range(self.n_depths)
            ], dim=-1)                                            # (B*K, n_depths)
        return out.reshape(B, K, self.n_depths)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns mu (B, K, n_depths) at 160 m. K=1 in training, 196 for a full map."""
        return self._forward_patchwise(batch)


# ── Loss ─────────────────────────────────────────────────────────────────────

def masked_huber_loss(
    pred_k:      torch.Tensor,   # (B, K, n_depths) — the model output; K=1 in training
    label:       torch.Tensor,   # (B, n_depths) — NaN where depth absent
    delta:       float = 0.05,
    per_depth:   bool  = False,
    depth_weights: torch.Tensor | None = None,
    return_breakdown: bool = False,
):
    """Huber loss on the supervised patch, ignoring depths with no observation.

    There is no station-pixel index. The U-Net emitted a 224x224 map and something had to pick
    the supervised pixel; this model emits (B, K, n_depths) where the value IS the prediction,
    and the dataset already selected patch 105 (§35.20).

    return_breakdown=True additionally returns (depth_sum, depth_cnt), both
    (n_depths,) float32, detached, on-device:
        depth_sum[d] = Σ Huber over samples in this batch that observed depth d
        depth_cnt[d] = number of those samples
    Raw SUMS, not means — the caller accumulates them over the epoch and
    all_reduce(SUM)s across ranks, which is only correct on sums.

    The returned scalar `loss` is byte-identical with and without the flag, so
    train/val loss stays comparable across runs.

    `depth_weights` (n_depths,) sets the fixed per-depth weight used when per_depth=True —
    inverse per-depth frequency over the training set. None means uniform, which reduces
    exactly to the pooled branch. It must NOT be derived from the batch; see the comment
    on the per_depth branch for why.

    NOTE: mean(depth_sum / depth_cnt) still does not equal the scalar loss — the breakdown
    is sample-weighted over the epoch while the scalar is w_d-weighted. Deep coverage is
    sparse (43 val stations at 30-100 cm vs 74 at 0-10), which is what w_d compensates for.
    Two different quantities on purpose; see training_runbook.md §19.3.
    """
    if pred_k.ndim != 3:
        raise ValueError(f"expected (B, K, n_depths), got {tuple(pred_k.shape)}")
    if pred_k.shape[1] != 1:
        raise ValueError(
            f"training loss expects K=1, got K={pred_k.shape[1]}. token_sel='all' is for "
            "inference only; supervising several patches needs multi-station labels, which "
            "dataset.py does not emit yet (§35.19)."
        )
    pred = pred_k[:, 0, :]                                             # (B, n_depths)

    if return_breakdown:
        # Branch-free so no data-dependent control flow is introduced: the
        # `if mask_d.any():` pattern below would cost one GPU sync per depth
        # per batch. nan_to_num keeps NaN out of the autograd-free arithmetic;
        # the `valid` mask zeroes those entries out anyway.
        valid     = ~torch.isnan(label)                                # (B, D) bool
        lab       = torch.nan_to_num(label, nan=0.0)
        elem      = F.huber_loss(pred.detach(), lab, delta=delta, reduction="none")
        # torch.where, NOT `elem * valid`: nan * False is nan, not 0.  `pred` is taken
        # for ALL depths while the scalar loss only ever sees pred[mask], so a non-finite
        # prediction at a depth with no label cannot affect training — but it would make
        # depth_sum nan, survive all_reduce(SUM) to every rank, and silently turn the
        # per-depth diagnostic into nan while train_loss still looked healthy.  That is
        # precisely the signal this breakdown exists to provide.
        depth_sum = torch.where(valid, elem, elem.new_zeros(())).sum(0).float()   # (D,)
        depth_cnt = valid.sum(0).float()                               # (D,)

    if per_depth:
        # Equal gradient weight per depth, WITHOUT letting batch composition set it.
        #
        # The old form was mean-over-depths of (mean over that batch's valid samples), which
        # gives every sample a weight of 1/n_d(batch): a batch holding 120 surface labels and
        # 2 at 30-100 cm handed each deep sample 20x the per-sample gradient of a surface
        # one, and 40x if it held only 1. The effective epoch objective was therefore an
        # E_batch[1/n_d]-weighted thing that is not a fixed function of the dataset and does
        # not survive a change of batch size.
        #
        # Here each (sample, depth) pair carries a FIXED weight w_d supplied by the caller —
        # inverse per-depth frequency over the whole training set — and the loss is the
        # weighted mean over valid pairs. Nothing depends on how the batch was drawn.
        # w_d = 1 reduces exactly to the pooled branch below.
        mask = ~torch.isnan(label)                                     # (B, D)
        if depth_weights is None:
            w = torch.ones_like(label)
        else:
            w = depth_weights.to(device=label.device, dtype=label.dtype).expand_as(label)
        wm   = torch.where(mask, w, torch.zeros_like(w))               # (B, D)
        elem = F.huber_loss(pred, torch.nan_to_num(label, nan=0.0),
                            delta=delta, reduction="none")             # (B, D)
        denom = wm.sum()
        # `elem * wm` is safe where `wm == 0`: label was nan_to_num'd, so elem is finite there
        # and the zero weight removes it from both the numerator and the denominator — a
        # depth with no label in this batch contributes no gradient to its head.
        loss = ((elem * wm).sum() / denom) if denom > 0 else pred.sum() * 0.0
    else:
        # Default: pool all valid (batch × depth) pairs into one mean — preserves
        # backward compatibility with baseline runs.
        mask = ~torch.isnan(label)
        loss = (F.huber_loss(pred[mask], label[mask], delta=delta, reduction="mean")
                if mask.any() else pred.sum() * 0.0)

    if return_breakdown:
        return loss, depth_sum, depth_cnt
    return loss

