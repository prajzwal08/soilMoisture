"""
SoilMoistureModel
=================
Full architecture:

  Pre-computed TerraMind features (loaded from disk, see precompute_terramind.py):
    S2:   L12 (MAX_S2, 196, 768) fp16 → _cpu_pyramid_pool (dataset worker) → (MAX_S2, 4, 768)
    S1:   L12 (MAX_S1, 196, 768) fp16 → _cpu_pyramid_pool (dataset worker) → (MAX_S1, 4, 768)
    DEM:  L12 (196, 768) fp16         → _cpu_pyramid_pool (dataset worker) → (4, 768)
    LULC: L12 (196, 768) fp16         → _cpu_pyramid_pool (dataset worker) → (4, 768)
    Skip: L3/L6/L9 (196, 768) fp16 for most-recent acquisition → U-Net decoder

  ERA5 MLP  :  (B, 365, 19) → (B, 365, 768)

  Sequence (per station-year):
    [DEM pyramid × 4]                        ← static prefix (pre-computed)
    [S2 pyramid tokens × N_s2 × 4]           ← pre-pooled in dataset worker (CPU)
    [S1 pyramid tokens × N_s1 × 4]           ← pre-pooled in dataset worker (CPU)
    [ERA5 tokens × 365]                      ← + circular DoY PE
    → Temporal Transformer (6L, 768D, 12H, bidirectional)

  Target spatial tokens:
    Most-recent S2 or S1 L12 (196×768) from stored features — no TerraMind pass.

  Bottleneck: transformer output at target DoY → reshape (B, 768, 14, 14)

  Temporal context: mean-pool of all non-spatial transformer output tokens → (B, 768)
    → FiLM-modulates each U-Net skip connection with weather/history context

  U-Net Decoder  (FiLM-modulated skip connections from TerraMind L3, L6, L9)
    14×14 → 28×28 → 56×56 → 112×112 → 224×224
    → SM map (B, n_depths, 224, 224)

  Loss: Huber masked to station pixel (centre: row=col=112 in 224×224 output)
       + λ_tv * Total Variation on the full SM map (spatial smoothness)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ── Spatial pyramid pooling ──────────────────────────────────────────────────

def spatial_pyramid_pool(tokens: torch.Tensor,
                         token_valid: torch.Tensor = None,
                         attn: nn.Module = None) -> torch.Tensor:
    """
    tokens      : (B, N_tok, D) — TerraMind L12 output for one acquisition
    token_valid : (B, N_tok) bool — True = clear/valid token; None = all valid
    attn        : nn.Linear(D, 1) for attention-weighted pooling; None = masked mean

    Attention mode (S2/S1): learned scores per token; masked tokens → -inf before softmax.
    Mean mode (DEM): masked average, ignoring invalid tokens.

    Grid size is derived from N_tok (must be a perfect square, e.g. 196 → 14×14).

    returns: (B, 4, D)
               scale 0: centre ~1×1  (~160 m)
               scale 1: inner  ~3×3  (~480 m)
               scale 2: inner  ~7×7  (~1.12 km)
               scale 3: full  14×14  (~2.24 km)
    """
    B, N_tok, D = tokens.shape
    G = int(N_tok ** 0.5)                                              # grid side (14 for 196 tokens)
    g = tokens.reshape(B, G, G, D)

    if attn is not None:
        scores = attn(tokens).reshape(B, G, G, 1)
        if token_valid is not None:
            cloud = ~token_valid.reshape(B, G, G).unsqueeze(-1)
            scores = scores.masked_fill(cloud, -1e4)   # -inf → NaN gradient when all-clouded

        def _pool(rs, re, cs, ce):
            wg = g[:, rs:re, cs:ce, :]
            ws = scores[:, rs:re, cs:ce, :]
            w  = F.softmax(ws.reshape(B, -1), dim=1)
            return (wg * w.reshape(B, re-rs, ce-cs, 1)).sum(dim=(1, 2))
    else:
        v = (token_valid.reshape(B, G, G).to(tokens.dtype).unsqueeze(-1)
             if token_valid is not None
             else torch.ones(B, G, G, 1, device=tokens.device, dtype=tokens.dtype))

        def _pool(rs, re, cs, ce):
            rg = g[:, rs:re, cs:ce, :]
            rv = v[:, rs:re, cs:ce, :]
            return (rg * rv).sum(dim=(1, 2)) / rv.sum(dim=(1, 2)).clamp(min=1)

    # Four nested scales centred on the grid; widths computed from G so the
    # function works for any square token grid, not just 14×14.
    half   = G // 2
    n_sc   = 4
    widths = [max(1, G * (i + 1) // (n_sc * 2)) for i in range(n_sc)]
    pools  = [_pool(half - w, half + w, half - w, half + w) for w in widths]
    return torch.stack(pools, dim=1)                                   # (B, 4, D)


# ── U-Net decoder ────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: modulate a spatial feature map with a
    context vector via learned scale and shift. Initialised as identity
    (scale=1, shift=0) so training starts from the unmodulated baseline."""

    def __init__(self, d_context: int, n_channels: int):
        super().__init__()
        self.proj = nn.Linear(d_context, 2 * n_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:n_channels])    # scale → 1 at init
        nn.init.zeros_(self.proj.bias[n_channels:])   # shift → 0 at init

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  context: (B, d_context)
        params = self.proj(context)                               # (B, 2C)
        C      = x.shape[1]
        scale  = params[:, :C].unsqueeze(-1).unsqueeze(-1)       # (B, C, 1, 1)
        shift  = params[:, C:].unsqueeze(-1).unsqueeze(-1)
        return scale * x + shift


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

    Skip connections from TerraMind L9, L6, L3 of the most-recent acquisition,
    each FiLM-modulated by the Transformer temporal context vector so the
    high-resolution decoder layers are aware of recent weather and history.
    """

    def __init__(
        self,
        in_ch:     int   = 768,
        skip_ch:   int   = 768,
        dec_ch:    tuple = (512, 256, 128, 64),
        n_depths:  int   = 3,
        d_context: int   = 768,
    ):
        super().__init__()
        c = dec_ch
        self.n_depths = n_depths

        self.bottle_proj = nn.Conv2d(in_ch, c[0], 1)
        self.skip_proj   = nn.ModuleList([
            nn.Conv2d(skip_ch, c[0], 1),   # L9
            nn.Conv2d(skip_ch, c[1], 1),   # L6
            nn.Conv2d(skip_ch, c[2], 1),   # L3
        ])

        # FiLM layers: inject temporal context into each skip before concatenation
        self.film_s9 = FiLMLayer(d_context, c[0])
        self.film_s6 = FiLMLayer(d_context, c[1])
        self.film_s3 = FiLMLayer(d_context, c[2])

        self.up1   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = _ConvBlock(c[0] + c[0], c[1])

        self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = _ConvBlock(c[1] + c[1], c[2])

        self.up3   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = _ConvBlock(c[2] + c[2], c[3])

        self.up4   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv4 = _ConvBlock(c[3], c[3])

        self.head = nn.Conv2d(c[3], n_depths, 1)

    def forward(self, bottleneck, skip_L9, skip_L6, skip_L3, context):
        # context: (B, d_context) — temporal summary from the Transformer
        x   = self.bottle_proj(bottleneck)
        s9  = self.film_s9(self.skip_proj[0](skip_L9), context)
        s6  = self.film_s6(self.skip_proj[1](skip_L6), context)
        s3  = self.film_s3(self.skip_proj[2](skip_L3), context)

        x = self.up1(x)
        x = self.conv1(torch.cat([x, F.interpolate(s9, x.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, F.interpolate(s6, x.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, F.interpolate(s3, x.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        x = self.up4(x)
        x = self.conv4(x)
        return self.head(x)                                             # (B, n_depths, 224, 224)


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
        n_depths: int = 3,
        d_model:  int = 768,
        n_heads:  int = 12,
        n_layers: int = 6,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_depths = n_depths

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
        # Target-day spatial (S2=0, S1_asc=1, S1_desc=2)
        self.spatial_modality_emb = nn.Embedding(3, d_model)
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

        # All pyramid pooling (S2, S1, DEM, LULC) uses static masked-mean in dataset workers.

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
            in_ch     = d_model,
            skip_ch   = d_model,
            dec_ch    = (512, 256, 128, 64),
            n_depths  = n_depths,
            d_context = d_model,
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

        flat_l12   = l12.reshape(B * MAX_ACQ, 196, self.d_model)
        flat_valid = valid.reshape(B * MAX_ACQ)                        # (B*MAX_ACQ,) bool
        flat_tm    = token_mask.reshape(B * MAX_ACQ, 196) if token_mask is not None else None

        # Run on all rows including padded ones. Avoids .nonzero() which forces a
        # GPU↔CPU sync and stalls the async pipeline, causing DDP rank imbalance.
        # INVARIANT: dataset zero-inits l12 for padded slots (torch.zeros in
        # load_s2/s1_rolling_zarr), so pooled output for padded rows is zero before
        # the flat_valid multiply. Do not replace torch.zeros with torch.empty there.
        pyr = spatial_pyramid_pool(flat_l12, flat_tm, attn=attn)       # (B*MAX_ACQ, 4, D)
        pyr = pyr * flat_valid[:, None, None]                          # zero padded slots

        return pyr.reshape(B, MAX_ACQ, 4, self.d_model)                # (B, MAX_ACQ, 4, 768)

    def _static_pyramid(self, l12: torch.Tensor, attn: nn.Linear,
                        token_valid: torch.Tensor | None = None) -> torch.Tensor:
        """l12: (B, 196, 768) fp16 → (B, 4, 768) fp32 pyramid.
        token_valid: (B, 14, 14) bool | None — True = valid patch."""
        B  = l12.shape[0]
        tv = token_valid.reshape(B, 196) if token_valid is not None else None
        return spatial_pyramid_pool(l12, token_valid=tv, attn=attn)

    def _get_target_spatial_tokens(self, batch: dict, B: int,
                                   device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build target spatial tokens from the pre-selected anchor acquisition
        (chosen in dataset.py by select_anchor_zarr: most recent fully-clear,
        falling back to most recent regardless).

        Returns:
            spatial_tokens : (B, 196, 768)
            use_s1         : (B,) bool
        """
        spatial_tokens = batch["anchor_l12"].to(device)                 # (B, 196, 768)
        anchor_orbit   = batch["anchor_orbit"].to(device)              # (B,) long: 0=S2,1=S1_asc,2=S1_desc

        # 2D spatial positional encoding
        rows       = torch.arange(14, device=device)
        cols       = torch.arange(14, device=device)
        spatial_pe = (self.spatial_row_emb(rows).unsqueeze(1) +
                      self.spatial_col_emb(cols).unsqueeze(0)).reshape(196, self.d_model)
        spatial_tokens = spatial_tokens + spatial_pe.unsqueeze(0)

        # Modality embedding: distinct learned vector for S2, S1-asc, S1-desc
        mod_emb        = self.spatial_modality_emb(anchor_orbit)       # (B, 768)
        spatial_tokens = spatial_tokens + mod_emb.unsqueeze(1)

        # Staleness embedding: how old is the anchor relative to target day
        anchor_rp      = batch["anchor_rel_pos"].to(device).clamp(0, 364)  # (B,)
        staleness_emb  = self.rel_pos_emb(anchor_rp)                   # (B, 768)
        spatial_tokens = spatial_tokens + staleness_emb.unsqueeze(1)

        use_s1 = anchor_orbit > 0
        return spatial_tokens, use_s1

    def _get_skip_connections(self, batch: dict, B: int,
                              device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape preloaded L3/L6/L9 skip features to (B, 768, 14, 14) spatial maps.
        Features are precomputed by precompute_terramind.py and loaded by the dataset.
        """
        def to_spatial(key: str) -> torch.Tensor:
            return (batch[key].to(device)                  # (B, 196, 768)
                    .reshape(B, 14, 14, self.d_model)
                    .permute(0, 3, 1, 2))                  # (B, 768, 14, 14)

        return to_spatial("anchor_l3"), to_spatial("anchor_l6"), to_spatial("anchor_l9")

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
        era5_pe   = circular_doy_pe(flat_doys, self.d_model).reshape(B, 365, self.d_model)

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
            sif_rel_pos = batch["sif_rel_pos"].to(device)             # (B, MAX_SIF)
            sif_tok  = self.sif_mlp(sif_vals.float())                 # (B, MAX_SIF, 768)
            sif_pe   = circular_doy_pe(sif_doys.reshape(-1), self.d_model
                                       ).reshape(B, -1, self.d_model)
            sif_rp   = self.rel_pos_emb(sif_rel_pos.reshape(-1).clamp(0, 364)
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
            twsa_rel_pos = batch["twsa_rel_pos"].to(device)           # (B, MAX_TWSA)
            twsa_tok  = self.twsa_mlp(twsa_vals.float())              # (B, MAX_TWSA, 768)
            twsa_pe   = circular_doy_pe(twsa_doys.reshape(-1), self.d_model
                                        ).reshape(B, -1, self.d_model)
            twsa_rp   = self.rel_pos_emb(twsa_rel_pos.reshape(-1).clamp(0, 364)
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

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Returns mu: (B, n_depths, 224, 224) predicted soil moisture map.
        """
        B      = batch["era5"].shape[0]
        device = batch["era5"].device

        # ── 1-3. All pyramid tokens — pre-pooled in dataset workers (CPU) ───
        # _cpu_pyramid_pool() runs static masked-mean in __getitem__ for all four
        # modalities; no L12 tensors cross the DataLoader IPC barrier.
        s2_pyr   = batch["s2_pyr"].to(device)    # (B, MAX_S2, 4, 768) fp32
        s1_pyr   = batch["s1_pyr"].to(device)    # (B, MAX_S1, 4, 768) fp32
        dem_pyr  = batch["dem_pyr"].to(device)   # (B, 4, 768) fp32
        lulc_pyr = batch["lulc_pyr"].to(device)  # (B, 4, 768) fp32

        # ── 3b. Soil tokens ────────────────────────────────────────────
        soil_tok = self.soil_encoder(batch["soil_patch"].to(device))   # (B, 4, 768)

        # ── 4. Target spatial tokens from stored L12 ──────────────────
        spatial_tokens, _ = self._get_target_spatial_tokens(batch, B, device)

        # ── 5. Skip connections from precomputed features ─────────────
        skip_L3, skip_L6, skip_L9 = self._get_skip_connections(batch, B, device)

        # ── 6. Build sequence and run transformer ──────────────────────
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

        # ── 7. Extract spatially-structured bottleneck ─────────────────
        spatial_ctx = ctx[:, spatial_start : spatial_start + 196, :]  # (B, 196, 768)
        bottleneck  = spatial_ctx.reshape(B, 14, 14, self.d_model).permute(0, 3, 1, 2)
                                                                       # (B, 768, 14, 14)

        # ── 8. Temporal context for FiLM: mean-pool non-spatial tokens ─
        # Excludes the spatial tokens (used as bottleneck) and padding tokens.
        ctx_mask  = (~key_mask).clone()                                # True = valid
        ctx_mask[:, spatial_start:spatial_start + 196] = False        # exclude spatial
        ctx_float = ctx_mask.float().unsqueeze(-1)                     # (B, T, 1)
        context   = (ctx * ctx_float).sum(1) / ctx_float.sum(1).clamp(min=1)  # (B, 768)

        # ── 9. U-Net decoder → SM map (+ optional var) ────────────────
        return self.decoder(bottleneck, skip_L9, skip_L6, skip_L3, context)


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


def total_variation_loss(sm_map: torch.Tensor) -> torch.Tensor:
    """
    Isotropic Total Variation loss encouraging spatial smoothness in the SM map.
    Penalises |∂SM/∂row| + |∂SM/∂col| averaged over all pixels, depths, and samples.
    sm_map : (B, n_depths, H, W)
    """
    diff_h = (sm_map[:, :, 1:, :] - sm_map[:, :, :-1, :]).abs().mean()
    diff_w = (sm_map[:, :, :, 1:] - sm_map[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w
