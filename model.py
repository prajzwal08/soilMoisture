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

  --driver-mode concat puts all 537 tokens in one self-attention stack instead. Kept as an
  option because it cannot be retrofitted (§3.4 of the maths doc), but not run.

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

def circular_doy_pe(doys: torch.Tensor, dim: int = 768) -> torch.Tensor:
    """
    Circular positional encoding for day-of-year. Periodic at 365.25 days so
    DOY 365 and DOY 1 share similar representations (no year-boundary seam).
    Uses harmonics sin/cos(k * 2π * DOY / 365.25) for k = 1 … dim//2.
    doys : (N,) long tensor of day-of-year values [1, 365]
    returns (N, dim) float
    """
    device = doys.device
    base   = 2.0 * math.pi / 365.25
    k      = torch.arange(1, dim // 2 + 1, device=device).float()     # (dim//2,)
    angles = doys.float().unsqueeze(1) * base * k                     # (N, dim//2)
    pe     = torch.zeros(len(doys), dim, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe                                                          # (N, dim)


class PatchwiseBlock(nn.Module):
    """
    One layer of the patch decoder (transformer 2 of the two-transformer design, §35.18).

        x = x + SelfAttn (LN(x))                 over the patch's own 106 tokens
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
                 dropout: float = 0.1, drop_path: float = 0.0, n_readout: int = 3):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim    = d_model // n_heads
        self.driver_mode = driver_mode
        self.n_readout   = n_readout

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
        self.drop_path = DropPath(drop_path)

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

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)   # (B, h, K*L, dh)
        out = out.transpose(1, 2).reshape(N, L, d)
        return self.o_proj(out)

    def forward(self, x, kc=None, vc=None, mem_pad=None, self_pad=None, B=None, K=None):
        h = self.norm_self(x)
        if self.collect_entropy:
            a, w = self.self_attn(h, h, h, key_padding_mask=self_pad,
                                  need_weights=True, average_attn_weights=True)
            self.last_entropy = _row_entropy(w, self_pad, self.n_readout).detach()
        else:
            a, _ = self.self_attn(h, h, h, key_padding_mask=self_pad, need_weights=False)
        x = x + self.drop_path(a)

        if self.driver_mode == "memory":
            x = x + self.drop_path(self._cross(self.norm_cross(x), kc, vc, mem_pad, B, K))

        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


def _row_entropy(w: torch.Tensor, pad: torch.Tensor | None, n_readout: int) -> torch.Tensor:
    """
    Mean Shannon entropy (nats) of the readout rows of an attention matrix.

    This is the detector §35.20 step 1 relies on. Register-dominated keys make q.k near-constant,
    which shows up here as entropy pinned at the uniform value log(n_valid) — and NOT in the map
    SD, which is why it has to be logged explicitly. Uniform over 100 history tokens = 4.605 nats.

    w         (N, L, L) attention weights, already head-averaged
    pad       (N, L)    True = ignored key
    n_readout           how many leading rows are readouts (the depth CLS tokens)
    """
    rows = w[:, :n_readout, :]                          # the depth CLS rows: (N, n_readout, L)
    if pad is not None:
        rows = rows.masked_fill(pad[:, None, :], 0.0)
    ent = -(rows.clamp_min(1e-9).log() * rows).sum(-1)  # (N, n_readout)
    return ent.mean()


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
                        sample) or "concat" (all 537 in one self-attention stack).
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
        # Satellite history (S2hist=0, S1hist=1) — distinguishes S2 from S1 in sequence
        self.hist_modality_emb = nn.Embedding(2, d_model)
        # ERA5 temporal tokens
        self.era5_modality_emb = nn.Embedding(1, d_model)
        # Optional sparse modalities
        self.sif_modality_emb  = nn.Embedding(1, d_model)
        self.twsa_modality_emb = nn.Embedding(1, d_model)

        # Learned relative position embedding: position within 365-day window
        self.rel_pos_emb = nn.Embedding(365, d_model)

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
        #   T2 patch_blocks 106 tokens, n_layers deep, runs K times (K folded into batch)
        #
        # STEP 1 has no decoder at all (§34.4): the prediction is the token head at 160 m.
        if not use_cls_depth:
            raise ValueError("use_cls_depth is required: the per-patch sequence carries "
                             "the depth CLS tokens as a prefix.")

        # T1 — the weather encoder. Depth here processes only tile-constant drivers, so it
        # is deliberately shallower than T2 (§35.19: 6 would take the model to 100.4 M,
        # 2.0x the 50.35 M baseline, confounding capacity with architecture).
        ddpr = [drop_path_rate * i / max(driver_layers - 1, 1) for i in range(driver_layers)]
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
        ])
        self.driver_norm = nn.LayerNorm(d_model)

        # T2 — the patch decoder.
        self.patch_blocks = nn.ModuleList([
            PatchwiseBlock(d_model, n_heads, driver_mode=driver_mode,
                           dropout=0.1, drop_path=dpr[i], n_readout=n_depths)
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

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_driver_tokens(self, batch: dict):
        """
        TRANSFORMER 1's input: the 431 tile-level driver tokens.

            era5 365 + sif 50 + twsa 12 + soil 4 = 431

        Built exactly as _build_sequence does (`model.py` ERA5/SIF/TWSA blocks) — same MLPs, same
        circular_doy_pe, same rel_pos_emb, same modality embeddings, same padding rules. Copied
        rather than reinvented so the patchwise arm stays comparable to the baseline.

        These tokens carry NO patch index: ERA5 is 9 km and the tile is 2.24 km, so one grid cell
        covers the whole tile. That is what makes the K/V cache exact (text/patchwise_math.md §2.4).

        Returns (m_raw (B, 431, 768), pad (B, 431) bool  True = ignore).
        """
        device = batch["era5"].device
        B      = batch["era5"].shape[0]
        toks, pads = [], []

        # ── soil (4) ───────────────────────────────────────────────────────
        soil_tok = self.soil_encoder(batch["soil_patch"].to(device))          # (B, 4, 768)
        soil_tok = soil_tok + self.soil_modality_emb(
            torch.zeros(1, dtype=torch.long, device=device))
        toks.append(soil_tok)
        pads.append(torch.zeros(B, soil_tok.shape[1], device=device, dtype=torch.bool))

        # ── ERA5 (365) ─────────────────────────────────────────────────────
        era5_doys = batch["era5_doys"]
        era5_tok  = self.era5_mlp(batch["era5"])
        era5_tok  = (era5_tok
                     + circular_doy_pe(era5_doys.reshape(-1), self.d_model
                                       ).reshape(B, 365, self.d_model)
                     + self.rel_pos_emb(torch.arange(365, device=device)).unsqueeze(0)
                     + self.era5_modality_emb(torch.zeros(1, dtype=torch.long, device=device)))
        toks.append(era5_tok)
        pads.append((era5_doys == 0).to(device))

        # ── SIF (50) and TWSA (12), both sparse ────────────────────────────
        for key, mlp, mod_emb in (
            ("sif",  self.sif_mlp,  self.sif_modality_emb),
            ("twsa", self.twsa_mlp, self.twsa_modality_emb),
        ):
            vals = batch[key].to(device)
            if vals.shape[1] == 0:
                continue
            doys    = batch[f"{key}_doys"].to(device)
            rel_pos = batch[f"{key}_rel_pos"].to(device)
            valid   = batch[f"{key}_valid"].to(device)
            tok = (mlp(vals.float())
                   + circular_doy_pe(doys.reshape(-1), self.d_model
                                     ).reshape(B, -1, self.d_model)
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
        DOY (ERA5 is the seasonal anchor; see the comment in _build_sequence).

        Returns (x (B, K, 105, 768), pad (B, K, 105) bool  True = ignore).
        """
        device = batch["era5"].device
        d      = self.d_model
        dem    = batch["dem_tok"].to(device).float()                       # (B, K, 768)
        B, K   = dem.shape[:2]

        blocks, pads = [], []

        # depth CLS prefix — (n_depths, d) shared across patches
        blocks.append(self.depth_tokens.view(1, 1, self.n_depths, d).expand(B, K, -1, -1))
        pads.append(torch.zeros(B, K, self.n_depths, device=device, dtype=torch.bool))

        # per-patch statics
        static_w = self.static_modality_emb.weight                          # (2, d)
        blocks.append((dem + static_w[0]).unsqueeze(2))                     # (B, K, 1, d)
        blocks.append((batch["lulc_tok"].to(device).float() + static_w[1]).unsqueeze(2))
        pads.append(torch.zeros(B, K, 2, device=device, dtype=torch.bool))

        # per-patch history: S2 then S1
        for hist_idx, key in enumerate(("s2", "s1")):
            h = batch[f"{key}_hist"].to(device).float()                     # (B, T, K, 768)
            h = h.permute(0, 2, 1, 3)                                       # (B, K, T, 768)
            rel = self.rel_pos_emb(
                batch[f"{key}_rel_pos"].to(device).reshape(-1).clamp(0, 364)
            ).reshape(B, 1, -1, d)                                          # (B, 1, T, d)
            mod = self.hist_modality_emb(
                torch.tensor(hist_idx, dtype=torch.long, device=device))    # (d,)
            blocks.append(h + rel + mod)
            # dataset.py:265 already ANDs the token mask with (doys > 0), so this covers padded
            # slots, NaN-skipped acquisitions and dates with no cloud-mask entry alike.
            pads.append(~batch[f"{key}_hist_valid"].to(device).permute(0, 2, 1))

        return torch.cat(blocks, dim=2), torch.cat(pads, dim=2)

    def _forward_patchwise(self, batch: dict) -> torch.Tensor:
        """Returns (B, K, n_depths) — one soil-moisture value per depth per patch, at 160 m."""
        # ── T1: encode the weather ONCE, then project the cache ONCE ───────
        m, mem_pad = self._build_driver_tokens(batch)
        for layer in self.driver_enc:
            m = layer(m, src_key_padding_mask=mem_pad)
        m = self.driver_norm(m)                                             # (B, 431, 768)

        # k_proj/v_proj live on the blocks but are called HERE, outside the patch loop. This is
        # the entire cost argument: 2*431*d^2 once per sample instead of once per patch.
        kv = ([(blk.k_proj(m), blk.v_proj(m)) for blk in self.patch_blocks]
              if self.driver_mode == "memory" else None)

        # ── T2: run every patch against that cache ─────────────────────────
        x, pad = self._build_patch_seq(batch)                               # (B,K,106,d)
        B, K, L, d = x.shape
        x   = x.reshape(B * K, L, d)
        pad = pad.reshape(B * K, L)

        if self.driver_mode == "concat":
            # No cache and no cross-attention: the memory joins the self-attention sequence, so
            # every patch carries its own copy of all 431 driver tokens (T = 537).
            mem = m.unsqueeze(1).expand(B, K, -1, -1).reshape(B * K, -1, d)
            x   = torch.cat([x, mem], dim=1)
            pad = torch.cat([pad, mem_pad.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)], 1)

        ent = []
        for i, blk in enumerate(self.patch_blocks):
            kc, vc = kv[i] if kv is not None else (None, None)
            x = blk(x, kc, vc, mem_pad, pad, B, K)
            if blk.collect_entropy and blk.last_entropy is not None:
                ent.append(blk.last_entropy)
        # Diagnostic stash, read by train.py. §35.20 made this the SOLE detector for
        # register-driven attention collapse; uniform over 100 history tokens is 4.605 nats.
        self._last_attn_entropy = torch.stack(ent) if ent else None

        x = self.patch_norm(x)
        # The depth CLS rows are the first n_depths tokens in BOTH driver modes (concat appends
        # the memory after the patch tokens, so the prefix is untouched).
        depth_ctx = x[:, :self.n_depths, :]                       # (B*K, n_depths, d)

        out = torch.cat([
            self.depth_heads[i](depth_ctx[:, i, :])
            for i in range(self.n_depths)
        ], dim=-1)                                                # (B*K, n_depths)
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

    NOTE: mean(depth_sum / depth_cnt) does NOT equal the scalar loss. The scalar
    is a mean-of-batch-means-of-depth-means; the breakdown is sample-weighted
    over the whole epoch. Sample-weighting is deliberate — deep coverage is
    sparse (43 val stations at 30-100 cm vs 74 at 0-10), so batch-means would
    over-weight batches that happen to hold two deep samples. Two different
    quantities on purpose; see training_runbook.md §19.3.
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
        # Equal gradient weight per depth: compute Huber separately for each depth,
        # then mean across depths that have at least one valid observation.
        # Prevents the high-variance surface layer from dominating deeper layers.
        depth_losses = []
        for d in range(pred.shape[1]):
            mask_d = ~torch.isnan(label[:, d])
            if mask_d.any():
                depth_losses.append(
                    F.huber_loss(pred[mask_d, d], label[mask_d, d], delta=delta, reduction="mean")
                )
        loss = torch.stack(depth_losses).mean() if depth_losses else pred.sum() * 0.0
    else:
        # Default: pool all valid (batch × depth) pairs into one mean — preserves
        # backward compatibility with baseline runs.
        mask = ~torch.isnan(label)
        loss = (F.huber_loss(pred[mask], label[mask], delta=delta, reduction="mean")
                if mask.any() else pred.sum() * 0.0)

    if return_breakdown:
        return loss, depth_sum, depth_cnt
    return loss

