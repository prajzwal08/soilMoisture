"""
SoilMoistureModel
=================
Full architecture:

  Pre-computed TerraMind features (loaded from disk, see precompute_terramind.py):
    S2:   L12 (MAX_S2, 196, 768) fp16 → s2_pyramid_attn   → 4×768 tokens per acquisition
    S1:   L12 (MAX_S1, 196, 768) fp16 → s1_pyramid_attn   → 4×768 tokens per acquisition
    DEM:  L12 (196, 768) fp16         → dem_pyramid_attn  → 4×768 static tokens
    LULC: L12 (196, 768) fp16         → lulc_pyramid_attn → 4×768 static tokens
    Skip: L3/L6/L9 (196, 768) fp16 for most-recent acquisition → U-Net decoder

  ERA5 MLP  :  (B, 365, 19) → (B, 365, 768)

  Sequence (per station-year):
    [DEM pyramid × 4]                        ← static prefix (pre-computed)
    [S2 pyramid tokens × N_s2 × 4]           ← from stored L12 + s2_pyramid_attn
    [S1 pyramid tokens × N_s1 × 4]           ← from stored L12 + s1_pyramid_attn
    [ERA5 tokens × 365]                      ← + DoY PE
    → Temporal Transformer (6L, 768D, 12H, bidirectional)

  Target spatial tokens:
    Most-recent S2 or S1 L12 (196×768) from stored features — no TerraMind pass.

  Bottleneck: transformer output at target DoY → reshape (B, 768, 14, 14)

  U-Net Decoder  (skip connections from TerraMind L3, L6, L9)
    14×14 → 28×28 → 56×56 → 112×112 → 224×224
    → SM map (B, n_depths, 224, 224)

  Loss: Huber masked to station pixel (centre: row=col=112 in 224×224 output)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Positional encoding ──────────────────────────────────────────────────────

def sinusoidal_pe(doys: torch.Tensor, dim: int = 768) -> torch.Tensor:
    """
    doys : (N,) long tensor of day-of-year values [1, 365]
    returns (N, dim) float positional encoding
    """
    device = doys.device
    pe     = torch.zeros(len(doys), dim, device=device)
    pos    = doys.float().unsqueeze(1)
    div    = torch.exp(
        torch.arange(0, dim, 2, device=device).float()
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                                           # (N, dim)


# ── Spatial pyramid pooling ──────────────────────────────────────────────────

def spatial_pyramid_pool(tokens: torch.Tensor,
                         token_valid: torch.Tensor = None,
                         attn: nn.Module = None) -> torch.Tensor:
    """
    tokens      : (B, 196, 768) — TerraMind L12 output for one acquisition
    token_valid : (B, 196) bool — True = clear/valid token; None = all valid
    attn        : nn.Linear(768, 1) for attention-weighted pooling; None = masked mean

    Attention mode (S2/S1): learned scores per token; masked tokens → -inf before softmax.
    Mean mode (DEM): masked average, ignoring invalid tokens.

    returns: (B, 4, 768)
               scale 0: centre ~1×1  (~160 m)
               scale 1: inner  ~3×3  (~480 m)
               scale 2: inner  ~7×7  (~1.12 km)
               scale 3: full  14×14  (~2.24 km)
    """
    B = tokens.shape[0]
    g = tokens.reshape(B, 14, 14, 768)

    if attn is not None:
        scores = attn(tokens).reshape(B, 14, 14, 1)
        if token_valid is not None:
            cloud = ~token_valid.reshape(B, 14, 14).unsqueeze(-1)
            scores = scores.masked_fill(cloud, float("-inf"))

        def _pool(rs, re, cs, ce):
            wg = g[:, rs:re, cs:ce, :]
            ws = scores[:, rs:re, cs:ce, :]
            w  = F.softmax(ws.reshape(B, -1), dim=1)
            w  = torch.nan_to_num(w, nan=0.0)
            return (wg * w.reshape(B, re-rs, ce-cs, 1)).sum(dim=(1, 2))
    else:
        v = (token_valid.reshape(B, 14, 14).to(tokens.dtype).unsqueeze(-1)
             if token_valid is not None
             else torch.ones(B, 14, 14, 1, device=tokens.device, dtype=tokens.dtype))

        def _pool(rs, re, cs, ce):
            rg = g[:, rs:re, cs:ce, :]
            rv = v[:, rs:re, cs:ce, :]
            return (rg * rv).sum(dim=(1, 2)) / rv.sum(dim=(1, 2)).clamp(min=1)

    t0 = _pool(6, 8,  6, 8)
    t1 = _pool(4, 10, 4, 10)
    t2 = _pool(2, 12, 2, 12)
    t3 = _pool(0, 14, 0, 14)
    return torch.stack([t0, t1, t2, t3], dim=1)                        # (B, 4, 768)


# ── U-Net decoder ────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetDecoder(nn.Module):
    """
    4× upsampling: 14×14 → 28×28 → 56×56 → 112×112 → 224×224

    Skip connections from TerraMind L9, L6, L3 of the most-recent acquisition.
    """

    def __init__(
        self,
        in_ch:               int   = 768,
        skip_ch:             int   = 768,
        dec_ch:              tuple = (512, 256, 128, 64),
        n_depths:            int   = 3,
        predict_uncertainty: bool  = True,
    ):
        super().__init__()
        c = dec_ch
        self.n_depths            = n_depths
        self.predict_uncertainty = predict_uncertainty

        self.bottle_proj = nn.Conv2d(in_ch, c[0], 1)
        self.skip_proj   = nn.ModuleList([
            nn.Conv2d(skip_ch, c[0], 1),   # L9
            nn.Conv2d(skip_ch, c[1], 1),   # L6
            nn.Conv2d(skip_ch, c[2], 1),   # L3
        ])

        self.up1   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = _ConvBlock(c[0] + c[0], c[1])

        self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = _ConvBlock(c[1] + c[1], c[2])

        self.up3   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = _ConvBlock(c[2] + c[2], c[3])

        self.up4   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv4 = _ConvBlock(c[3], c[3])

        # When uncertainty=True: output 2×n_depths (mu then log_var per depth).
        # The log_var bias is initialised to 0 so σ² ≈ 1 at the start of training.
        out_ch    = 2 * n_depths if predict_uncertainty else n_depths
        self.head = nn.Conv2d(c[3], out_ch, 1)
        if predict_uncertainty:
            nn.init.zeros_(self.head.bias)

    def forward(self, bottleneck, skip_L9, skip_L6, skip_L3):
        x   = self.bottle_proj(bottleneck)
        s9  = self.skip_proj[0](skip_L9)
        s6  = self.skip_proj[1](skip_L6)
        s3  = self.skip_proj[2](skip_L3)

        x = self.up1(x)
        x = self.conv1(torch.cat([x, F.interpolate(s9, x.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, F.interpolate(s6, x.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, F.interpolate(s3, x.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        x = self.up4(x)
        x = self.conv4(x)
        raw = self.head(x)                                              # (B, out_ch, 224, 224)

        if self.predict_uncertainty:
            mu      = raw[:, :self.n_depths]
            # Clamp here so gradients flow correctly at all values.
            # Without this, extreme raw values get clamped only inside the loss,
            # where PyTorch's clamp has zero gradient outside bounds — dead neurons.
            log_var = raw[:, self.n_depths:].clamp(-6.0, 6.0)
            return mu, log_var
        return raw, None                                                # None signals no uncertainty


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
        t0 = x[:, :, cy:cy+1,   cx:cx+1  ].mean(dim=(-2, -1))     # 1×1  ~30 m
        t1 = x[:, :, cy-1:cy+2, cx-1:cx+2].mean(dim=(-2, -1))     # 3×3  ~90 m
        t2 = x[:, :, cy-3:cy+4, cx-3:cx+4].mean(dim=(-2, -1))     # 7×7  ~210 m
        t3 = x.mean(dim=(-2, -1))                                   # 37×37 ~1.1 km
        return torch.stack(
            [self.proj[i](t) for i, t in enumerate([t0, t1, t2, t3])], dim=1
        )                                                            # (B, 4, 768)


# ── Full model ───────────────────────────────────────────────────────────────

class SoilMoistureModel(nn.Module):
    """
    Args:
        n_depths  : number of SM depth bins (default 4)
        d_model   : transformer / token dimension (default 768)
        n_heads   : attention heads (default 12)
        n_layers  : transformer layers (default 6)
    """

    STATION_ROW = 112
    STATION_COL = 112

    def __init__(
        self,
        n_depths:            int  = 3,
        d_model:             int  = 768,
        n_heads:             int  = 12,
        n_layers:            int  = 6,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        self.d_model             = d_model
        self.n_depths            = n_depths
        self.predict_uncertainty = predict_uncertainty

        # ── Encoders ──────────────────────────────────────────────────
        self.soil_encoder = SoilEncoder(d_model=d_model)

        self.era5_mlp = nn.Sequential(
            nn.Linear(19, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )

        # SIF and TWSA MLP encoders (scalar → token)
        self.sif_mlp  = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, d_model))
        self.twsa_mlp = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Linear(256, d_model))

        # ── Modality type embeddings ───────────────────────────────────
        # Soil tokens
        self.soil_modality_emb = nn.Embedding(1, d_model)
        # Static (DEM=0, LULC=1)
        self.static_modality_emb = nn.Embedding(2, d_model)
        # Target-day spatial (S2=0, S1=1)
        self.spatial_modality_emb = nn.Embedding(2, d_model)
        # Satellite history (S2hist=0, S1hist=1) — distinguishes S2 from S1 in sequence
        self.hist_modality_emb = nn.Embedding(2, d_model)
        # ERA5 temporal tokens
        self.era5_modality_emb = nn.Embedding(1, d_model)
        # Optional sparse modalities
        self.sif_modality_emb  = nn.Embedding(1, d_model)
        self.twsa_modality_emb = nn.Embedding(1, d_model)

        # Learned scale embedding: 4 pyramid levels
        self.scale_emb = nn.Embedding(4, d_model)

        # Learned relative position embedding: position within 365-day window
        self.rel_pos_emb = nn.Embedding(365, d_model)

        # Modality-specific learned pyramid attention scorers
        self.s2_pyramid_attn   = nn.Linear(d_model, 1)
        self.s1_pyramid_attn   = nn.Linear(d_model, 1)
        self.dem_pyramid_attn  = nn.Linear(d_model, 1)
        self.lulc_pyramid_attn = nn.Linear(d_model, 1)

        # Target-day spatial tokens: 2D spatial PE
        self.spatial_row_emb = nn.Embedding(14, d_model)
        self.spatial_col_emb = nn.Embedding(14, d_model)

        # ── Temporal transformer ──────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = 0.1,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── U-Net decoder ─────────────────────────────────────────────
        self.decoder = UNetDecoder(
            in_ch               = d_model,
            skip_ch             = d_model,
            dec_ch              = (512, 256, 128, 64),
            n_depths            = n_depths,
            predict_uncertainty = predict_uncertainty,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _pyramid_from_l12(self,
                          l12:        torch.Tensor,
                          valid:      torch.Tensor,
                          token_mask: torch.Tensor | None,
                          attn:       nn.Linear) -> torch.Tensor:
        """
        Compute pyramid tokens from pre-stored L12 features.

        l12        : (B, MAX_ACQ, 196, 768) fp16 — loaded from disk
        valid      : (B, MAX_ACQ) bool
        token_mask : (B, MAX_ACQ, 14, 14) bool | None — True=clear (S2 only)
        attn       : modality-specific nn.Linear(768, 1) scorer

        Returns: (B, MAX_ACQ, 4, 768)
        """
        B, MAX_ACQ = valid.shape
        device     = l12.device

        pyramid    = torch.zeros(B, MAX_ACQ, 4, self.d_model, device=device)

        flat_l12   = l12.reshape(B * MAX_ACQ, 196, self.d_model).float()
        flat_valid = valid.reshape(B * MAX_ACQ)

        valid_idx  = flat_valid.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return pyramid

        valid_l12 = flat_l12[valid_idx]                                # (N, 196, 768)

        if token_mask is not None:
            flat_tm  = token_mask.reshape(B * MAX_ACQ, 196)
            valid_tm = flat_tm[valid_idx]
        else:
            valid_tm = None

        pyr = spatial_pyramid_pool(valid_l12, valid_tm, attn=attn)     # (N, 4, 768)

        batch_idx = valid_idx // MAX_ACQ
        acq_idx   = valid_idx %  MAX_ACQ
        pyramid[batch_idx, acq_idx] = pyr

        return pyramid                                                   # (B, MAX_ACQ, 4, 768)

    def _static_pyramid(self, l12: torch.Tensor, attn: nn.Linear) -> torch.Tensor:
        """l12: (B, 196, 768) fp16 → (B, 4, 768) fp32 pyramid."""
        return spatial_pyramid_pool(l12.float(), token_valid=None, attn=attn)

    def _get_target_spatial_tokens(self, batch: dict, B: int,
                                   device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the most recent valid acquisition ≤ T per sample and return its
        stored L12 features as target spatial tokens.

        Returns:
            spatial_tokens : (B, 196, 768)
            use_s1         : (B,) bool — True where S1 is most recent
        """
        # Use rel_pos (0=oldest, 364=today) not DOY — DOY breaks at year boundaries
        # (e.g. Dec DOY~350 > Jun DOY~180, so DOY wrongly picks a 6-month-old image)
        s2_rel   = batch["s2_rel_pos"].to(device)
        s1_rel   = batch["s1_rel_pos"].to(device)
        s2_valid = batch["s2_valid"].to(device)
        s1_valid = batch["s1_valid"].to(device)

        s2_latest = (s2_rel * s2_valid.long()).max(dim=1)
        s1_latest = (s1_rel * s1_valid.long()).max(dim=1)
        use_s1    = s1_latest.values > s2_latest.values                # (B,) bool

        spatial_tokens = torch.zeros(B, 196, self.d_model, device=device)

        s2_mask = ~use_s1
        if s2_mask.any():
            idx      = s2_latest.indices[s2_mask]
            l12_s2   = batch["s2_l12"][s2_mask].float().to(device)    # (n, MAX_S2, 196, 768)
            spatial_tokens[s2_mask] = l12_s2[
                torch.arange(s2_mask.sum(), device=device), idx
            ]

        s1_mask = use_s1
        if s1_mask.any():
            idx      = s1_latest.indices[s1_mask]
            l12_s1   = batch["s1_l12"][s1_mask].float().to(device)
            spatial_tokens[s1_mask] = l12_s1[
                torch.arange(s1_mask.sum(), device=device), idx
            ]

        # 2D spatial positional encoding
        rows       = torch.arange(14, device=device)
        cols       = torch.arange(14, device=device)
        spatial_pe = (self.spatial_row_emb(rows).unsqueeze(1) +
                      self.spatial_col_emb(cols).unsqueeze(0)).reshape(196, self.d_model)
        spatial_tokens = spatial_tokens + spatial_pe.unsqueeze(0)

        # Modality embedding
        mod_emb        = self.spatial_modality_emb(use_s1.long())      # (B, 768)
        spatial_tokens = spatial_tokens + mod_emb.unsqueeze(1)

        # Staleness embedding: tells the U-Net decoder how old the anchor image is.
        # Without this the decoder is blind to whether the spatial texture is
        # from yesterday or 6 months ago — critical when ERA5 shows rain since capture.
        anchor_rel_pos = torch.where(use_s1,
                                     s1_latest.values,
                                     s2_latest.values).clamp(0, 364)   # (B,)
        staleness_emb  = self.rel_pos_emb(anchor_rel_pos)              # (B, 768)
        spatial_tokens = spatial_tokens + staleness_emb.unsqueeze(1)

        return spatial_tokens, use_s1

    def _get_skip_connections(self, batch: dict, B: int,
                              device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape preloaded L3/L6/L9 skip features to (B, 768, 14, 14) spatial maps.
        Features are precomputed by precompute_terramind.py and loaded by the dataset.
        """
        def to_spatial(key: str) -> torch.Tensor:
            return (batch[key].float().to(device)          # (B, 196, 768)
                    .reshape(B, 14, 14, self.d_model)
                    .permute(0, 3, 1, 2))                  # (B, 768, 14, 14)

        return to_spatial("skip_l3"), to_spatial("skip_l6"), to_spatial("skip_l9")

    def _build_sequence(self, batch: dict, dem_pyr, lulc_pyr, soil_tok,
                        s2_pyr, s2_doys, s2_valid,
                        s1_pyr, s1_doys, s1_valid,
                        spatial_tokens):
        """
        Assemble the full token sequence:
            [DEM×4 | LULC×4 | Soil×4 | target_spatial×196 | satellite_history | era5×365]

        Returns:
            seq           : (B, T, 768)
            key_mask      : (B, T)  True = ignore
            spatial_start : int     index of first spatial token (= 12)
        """
        B      = batch["era5"].shape[0]
        device = batch["era5"].device
        tokens = []
        is_pad = []

        scale_e    = self.scale_emb(torch.arange(4, device=device))        # (4, 768)
        static_mod = self.static_modality_emb.weight                       # (2, 768)

        # ── Static prefix: DEM ─────────────────────────────────────────
        dem_tok = dem_pyr + scale_e + static_mod[0]
        tokens.append(dem_tok)                                             # (B, 4, 768)
        is_pad.append(torch.zeros(B, 4, device=device, dtype=torch.bool))

        # ── Static prefix: LULC ────────────────────────────────────────
        lulc_tok = lulc_pyr + scale_e + static_mod[1]
        tokens.append(lulc_tok)                                            # (B, 4, 768)
        is_pad.append(torch.zeros(B, 4, device=device, dtype=torch.bool))

        # ── Static prefix: Soil ────────────────────────────────────────
        soil_mod_e = self.soil_modality_emb(
            torch.zeros(1, dtype=torch.long, device=device)
        )                                                                  # (1, 768)
        soil_tokens = soil_tok + scale_e.unsqueeze(0) + soil_mod_e.unsqueeze(0)
        tokens.append(soil_tokens)                                         # (B, 4, 768)
        is_pad.append(torch.zeros(B, 4, device=device, dtype=torch.bool))

        # ── Target-day spatial tokens ──────────────────────────────────
        spatial_start = sum(t.shape[1] for t in tokens)                   # = 12
        tokens.append(spatial_tokens)                                  # (B, 196, 768)
        is_pad.append(torch.zeros(B, 196, device=device, dtype=torch.bool))

        # ── Satellite history tokens (S2 then S1) ─────────────────────
        for hist_idx, (pyr, doys, valid, rel_pos) in enumerate([
            (s2_pyr, s2_doys, s2_valid, batch["s2_rel_pos"]),
            (s1_pyr, s1_doys, s1_valid, batch["s1_rel_pos"]),
        ]):
            MAX_ACQ = pyr.shape[1]

            rp_flat   = rel_pos.reshape(-1).clamp(0, 364)
            rp_flat   = self.rel_pos_emb(rp_flat)
            rp        = rp_flat.reshape(B, MAX_ACQ, 1, self.d_model)

            # Modality type embedding distinguishes S2 history from S1 history
            hist_mod  = self.hist_modality_emb(
                torch.tensor(hist_idx, dtype=torch.long, device=device)
            )                                                          # (768,)

            # Satellite history uses only relative position (staleness) — no absolute DOY.
            # ERA5 is the seasonal anchor; satellite tokens only need to know how old they are.
            sat_tok   = pyr + rp + scale_e.unsqueeze(0).unsqueeze(0) + hist_mod
            sat_tok   = sat_tok.reshape(B, MAX_ACQ * 4, self.d_model)

            pad_acq   = ~valid
            pad_tok   = pad_acq.unsqueeze(-1).expand(-1, -1, 4).reshape(B, MAX_ACQ * 4)

            tokens.append(sat_tok)
            is_pad.append(pad_tok)

        # ── ERA5 tokens ────────────────────────────────────────────────
        era5_raw = batch["era5"]
        era5_tok = self.era5_mlp(era5_raw)

        era5_doys = batch["era5_doys"]
        flat_doys = era5_doys.reshape(-1)
        era5_pe   = sinusoidal_pe(flat_doys, self.d_model).reshape(B, 365, self.d_model)

        era5_rel  = torch.arange(365, device=device)
        era5_rp   = self.rel_pos_emb(era5_rel).unsqueeze(0)

        era5_mod  = self.era5_modality_emb(
            torch.zeros(1, dtype=torch.long, device=device)
        )                                                              # (1, 768)

        era5_tok  = era5_tok + era5_pe + era5_rp + era5_mod

        # ERA5 is right-aligned: slot 364 = target day, earlier slots = past days.
        # There is no future data in the array — mask only zero-padded slots
        # (era5_doys == 0 means no data was available for that slot).
        era5_pad    = (era5_doys == 0).to(device)                      # (B, 365)

        tokens.append(era5_tok)
        is_pad.append(era5_pad)

        # ── SIF tokens (optional sparse) ──────────────────────────────
        sif_vals  = batch["sif"].to(device)                           # (B, MAX_SIF, 1)
        sif_doys  = batch["sif_doys"].to(device)                      # (B, MAX_SIF)
        sif_valid = batch["sif_valid"].to(device)                     # (B, MAX_SIF)
        if sif_vals.shape[1] > 0:
            sif_tok  = self.sif_mlp(sif_vals.float())                 # (B, MAX_SIF, 768)
            sif_pe   = sinusoidal_pe(sif_doys.reshape(-1), self.d_model
                                     ).reshape(B, -1, self.d_model)
            sif_rp   = self.rel_pos_emb(sif_doys.reshape(-1).clamp(0, 364)
                                        ).reshape(B, -1, self.d_model)
            sif_mod  = self.sif_modality_emb(
                torch.zeros(1, dtype=torch.long, device=device)
            )
            sif_tok  = sif_tok + sif_pe + sif_rp + sif_mod
            tokens.append(sif_tok)
            is_pad.append(~sif_valid)

        # ── TWSA tokens (optional sparse) ─────────────────────────────
        twsa_vals  = batch["twsa"].to(device)                         # (B, MAX_TWSA, 1)
        twsa_doys  = batch["twsa_doys"].to(device)
        twsa_valid = batch["twsa_valid"].to(device)
        if twsa_vals.shape[1] > 0:
            twsa_tok  = self.twsa_mlp(twsa_vals.float())              # (B, MAX_TWSA, 768)
            twsa_pe   = sinusoidal_pe(twsa_doys.reshape(-1), self.d_model
                                      ).reshape(B, -1, self.d_model)
            twsa_rp   = self.rel_pos_emb(twsa_doys.reshape(-1).clamp(0, 364)
                                         ).reshape(B, -1, self.d_model)
            twsa_mod  = self.twsa_modality_emb(
                torch.zeros(1, dtype=torch.long, device=device)
            )
            twsa_tok  = twsa_tok + twsa_pe + twsa_rp + twsa_mod
            tokens.append(twsa_tok)
            is_pad.append(~twsa_valid)

        seq      = torch.cat(tokens, dim=1)
        key_mask = torch.cat(is_pad,  dim=1)
        return seq, key_mask, spatial_start

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns (mu, log_var):
          mu      : (B, n_depths, 224, 224)
          log_var : (B, n_depths, 224, 224) if predict_uncertainty else None
        """
        B      = batch["era5"].shape[0]
        device = batch["era5"].device

        # ── 1. Pyramid tokens from pre-stored L12 (no TerraMind) ──────
        s2_pyr = self._pyramid_from_l12(
            batch["s2_l12"].to(device),
            batch["s2_valid"].to(device),
            batch["s2_token_mask"].to(device),
            self.s2_pyramid_attn,
        )
        s1_pyr = self._pyramid_from_l12(
            batch["s1_l12"].to(device),
            batch["s1_valid"].to(device),
            token_mask = None,
            attn       = self.s1_pyramid_attn,
        )

        # ── 2. Static pyramid tokens (DEM + LULC) ─────────────────────
        dem_pyr  = self._static_pyramid(batch["dem_l12"].to(device),  self.dem_pyramid_attn)
        lulc_pyr = self._static_pyramid(batch["lulc_l12"].to(device), self.lulc_pyramid_attn)

        # ── 2b. Soil tokens ────────────────────────────────────────────
        soil_tok = self.soil_encoder(batch["soil_patch"].to(device))   # (B, 4, 768)

        # ── 3. Target spatial tokens from stored L12 ──────────────────
        spatial_tokens, _ = self._get_target_spatial_tokens(batch, B, device)

        # ── 4. Skip connections from precomputed features ─────────────
        skip_L3, skip_L6, skip_L9 = self._get_skip_connections(batch, B, device)

        # ── 5. Build sequence and run transformer ──────────────────────
        seq, key_mask, spatial_start = self._build_sequence(
            batch,
            dem_pyr,
            lulc_pyr,
            soil_tok,
            s2_pyr, batch["s2_doys"].to(device), batch["s2_valid"].to(device),
            s1_pyr, batch["s1_doys"].to(device), batch["s1_valid"].to(device),
            spatial_tokens,
        )

        ctx = self.transformer(seq, src_key_padding_mask=key_mask)    # (B, T, 768)

        # ── 6. Extract spatially-structured bottleneck ─────────────────
        spatial_ctx = ctx[:, spatial_start : spatial_start + 196, :]  # (B, 196, 768)
        bottleneck  = spatial_ctx.reshape(B, 14, 14, self.d_model).permute(0, 3, 1, 2)
                                                                       # (B, 768, 14, 14)

        # ── 7. U-Net decoder → SM map (+ optional log_var) ────────────
        return self.decoder(bottleneck, skip_L9, skip_L6, skip_L3)


# ── Loss ─────────────────────────────────────────────────────────────────────

def masked_huber_loss(
    sm_map:      torch.Tensor,   # (B, n_depths, 224, 224)
    label:       torch.Tensor,   # (B, n_depths) — NaN where depth absent
    station_row: int   = SoilMoistureModel.STATION_ROW,
    station_col: int   = SoilMoistureModel.STATION_COL,
    delta:       float = 0.05,
) -> torch.Tensor:
    pred = sm_map[:, :, station_row, station_col]                      # (B, n_depths)
    mask = ~torch.isnan(label)

    if not mask.any():
        return pred.sum() * 0.0

    return F.huber_loss(pred[mask], label[mask], delta=delta, reduction="mean")


def masked_nll_loss(
    mu:          torch.Tensor,   # (B, n_depths, 224, 224)
    log_var:     torch.Tensor,   # (B, n_depths, 224, 224)
    label:       torch.Tensor,   # (B, n_depths) — NaN where depth absent
    station_row: int   = SoilMoistureModel.STATION_ROW,
    station_col: int   = SoilMoistureModel.STATION_COL,
    log_var_min: float = -6.0,   # clamp prevents σ² → 0 collapse
    log_var_max: float =  6.0,   # clamp prevents σ² → ∞ trivial solution
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood at the station pixel.
    NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
    log_var is clamped to [log_var_min, log_var_max] for numerical stability.
    """
    mu_pt  = mu[:, :, station_row, station_col]                        # (B, n_depths)
    lv_pt  = log_var[:, :, station_row, station_col]                   # (B, n_depths)
    mask   = ~torch.isnan(label)

    if not mask.any():
        return mu_pt.sum() * 0.0

    mu_m  = mu_pt[mask]
    lv_m  = lv_pt[mask].clamp(log_var_min, log_var_max)
    y_m   = label[mask]

    nll = 0.5 * lv_m + 0.5 * (y_m - mu_m) ** 2 / lv_m.exp()
    return nll.mean()
