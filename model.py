"""
SoilMoistureModel
=================
Full architecture:

  TerraMind (frozen by default)
    ├─ S2L2A / S1RTC / DEM patches → L3, L6, L9, L12 features
    └─ L12 → spatial pyramid → 4 × 768 tokens per acquisition

  Pre-computed L12 (loaded from disk, see precompute_terramind.py):
    S2: (MAX_S2, 196, 768) fp16  per sample
    S1: (MAX_S1, 196, 768) fp16  per sample
    → s2_pyramid_attn / s1_pyramid_attn (learned, modality-specific) → 4×768 tokens

  ERA5 MLP  :  (B, 365, 19) → (B, 365, 768)

  Sequence (per station-year):
    [DEM pyramid × 4]                        ← static prefix (pre-computed)
    [S2 pyramid tokens × N_s2 × 4]           ← from stored L12 + s2_pyramid_attn
    [S1 pyramid tokens × N_s1 × 4]           ← from stored L12 + s1_pyramid_attn
    [ERA5 tokens × 365]                      ← + DoY PE
    → Temporal Transformer (6L, 768D, 12H, bidirectional)

  Target spatial tokens:
    Most-recent S2 or S1 L12 (196×768) from stored features — no TerraMind pass.

  Skip connections (L3, L6, L9):
    1 TerraMind pass per sample on the most-recent raw patch only.

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
from terratorch import BACKBONE_REGISTRY

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


# ── TerraMind wrapper ────────────────────────────────────────────────────────

class TerraMindEncoder(nn.Module):
    """
    Wraps TerraMind Base and exposes intermediate layer outputs via hooks.

    Intermediate outputs extracted:
        L3  (after block 2)  — fine / low-level features
        L6  (after block 5)  — mid-level features
        L9  (after block 8)  — deep features
        L12 (after block 11) — final semantic features

    All outputs are (B, 196, 768).

    frozen=True (default): TerraMind weights are not updated.
    """

    HOOK_LAYERS = {"L3": 2, "L6": 5, "L9": 8, "L12": 11}

    MODALITY_MAP = {
        "S2L2A": "untok_sen2l2a@224",
        "S1RTC": "untok_sen1rtc@224",
        "DEM"  : "untok_dem@224",
        "LULC" : "untok_lulc@224",
    }

    def __init__(self, frozen: bool = True):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_base",
            pretrained=True,
            modalities=list(self.MODALITY_MAP.values()),
        )
        self.frozen = frozen
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self._feats   = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, idx in self.HOOK_LAYERS.items():
            handle = self.backbone.encoder[idx].register_forward_hook(
                self._make_hook(name)
            )
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self._feats[name] = output if output.dim() == 3 else output[0]
        return hook

    def forward(self, patch: torch.Tensor, modality: str) -> dict:
        """
        patch    : (B, C, 224, 224) float32
        modality : one of 'S2L2A', 'S1RTC', 'DEM'
        returns  : dict  L3/L6/L9/L12 → (B, 196, 768)
        """
        self._feats = {}
        tm_key = self.MODALITY_MAP[modality]
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            _ = self.backbone({tm_key: patch})
        return {k: v.clone() for k, v in self._feats.items()}

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


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
        in_ch:   int   = 768,
        skip_ch: int   = 768,
        dec_ch:  tuple = (512, 256, 128, 64),
        n_depths: int  = 4,
    ):
        super().__init__()
        c = dec_ch

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

        self.head  = nn.Conv2d(c[3], n_depths, 1)

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
        n_depths: int = 4,
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

        # Learned modality embedding for soil tokens (index 9 per architecture.md)
        self.soil_modality_emb = nn.Embedding(1, d_model)

        # Learned scale embedding: 4 pyramid levels
        self.scale_emb = nn.Embedding(4, d_model)

        # Learned relative position embedding: position within 365-day window
        self.rel_pos_emb = nn.Embedding(365, d_model)

        # Modality-specific learned pyramid attention scorers
        self.s2_pyramid_attn = nn.Linear(d_model, 1)   # for S2 acquisitions
        self.s1_pyramid_attn = nn.Linear(d_model, 1)   # for S1 acquisitions

        # Target-day spatial tokens: 2D spatial PE + modality embedding
        self.spatial_row_emb      = nn.Embedding(14, d_model)
        self.spatial_col_emb      = nn.Embedding(14, d_model)
        self.spatial_modality_emb = nn.Embedding(2, d_model)  # 0=S2, 1=S1

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
            in_ch    = d_model,
            skip_ch  = d_model,
            dec_ch   = (512, 256, 128, 64),
            n_depths = n_depths,
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

    def _get_target_spatial_tokens(self, batch: dict, B: int,
                                   device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select the most recent valid acquisition ≤ T per sample and return its
        stored L12 features as target spatial tokens.

        Returns:
            spatial_tokens : (B, 196, 768)
            use_s1         : (B,) bool — True where S1 is most recent
        """
        s2_doys  = batch["s2_doys"].float()
        s1_doys  = batch["s1_doys"].float()
        s2_valid = batch["s2_valid"]
        s1_valid = batch["s1_valid"]

        s2_latest = (s2_doys * s2_valid.float()).max(dim=1)
        s1_latest = (s1_doys * s1_valid.float()).max(dim=1)
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

    def _build_sequence(self, batch: dict, dem_pyr, soil_tok,
                        s2_pyr, s2_doys, s2_valid,
                        s1_pyr, s1_doys, s1_valid,
                        spatial_tokens):
        """
        Assemble the full token sequence:
            [static_prefix | target_spatial | satellite_history | era5_tokens]

        Returns:
            seq           : (B, T, 768)
            key_mask      : (B, T)  True = ignore
            spatial_start : int
        """
        B      = batch["era5"].shape[0]
        device = batch["era5"].device
        tokens = []
        is_pad = []

        # ── Static prefix: DEM ─────────────────────────────────────────
        scale_e = self.scale_emb(torch.arange(4, device=device))      # (4, 768)
        dem_tok = dem_pyr + scale_e.unsqueeze(0)
        tokens.append(dem_tok)                                         # (B, 4, 768)
        is_pad.append(torch.zeros(B, 4, device=device, dtype=torch.bool))

        # ── Static prefix: Soil ────────────────────────────────────────
        soil_mod_e = self.soil_modality_emb(
            torch.zeros(1, dtype=torch.long, device=device)
        )                                                              # (1, 768)
        soil_tokens = soil_tok + scale_e.unsqueeze(0) + soil_mod_e.unsqueeze(0)
        tokens.append(soil_tokens)                                     # (B, 4, 768)
        is_pad.append(torch.zeros(B, 4, device=device, dtype=torch.bool))

        # ── Target-day spatial tokens ──────────────────────────────────
        spatial_start = sum(t.shape[1] for t in tokens)               # = 4
        tokens.append(spatial_tokens)                                  # (B, 196, 768)
        is_pad.append(torch.zeros(B, 196, device=device, dtype=torch.bool))

        # ── Satellite tokens ───────────────────────────────────────────
        for pyr, doys, valid, rel_pos in [
            (s2_pyr, s2_doys, s2_valid, batch["s2_rel_pos"]),
            (s1_pyr, s1_doys, s1_valid, batch["s1_rel_pos"]),
        ]:
            MAX_ACQ = pyr.shape[1]

            flat_doys = doys.reshape(-1)
            pe_flat   = sinusoidal_pe(flat_doys, self.d_model)
            pe        = pe_flat.reshape(B, MAX_ACQ, 1, self.d_model)

            rp_flat   = rel_pos.reshape(-1).clamp(0, 364)
            rp_flat   = self.rel_pos_emb(rp_flat)
            rp        = rp_flat.reshape(B, MAX_ACQ, 1, self.d_model)

            sat_tok   = pyr + pe + rp + scale_e.unsqueeze(0).unsqueeze(0)
            sat_tok   = sat_tok.reshape(B, MAX_ACQ * 4, self.d_model)

            pad_acq   = ~valid
            pad_tok   = pad_acq.unsqueeze(-1).expand(-1, -1, 4).reshape(B, MAX_ACQ * 4)

            tokens.append(sat_tok)
            is_pad.append(pad_tok)

        # ── ERA5 tokens ────────────────────────────────────────────────
        era5_raw  = batch["era5"]
        era5_tok  = self.era5_mlp(era5_raw)

        era5_doys = batch["era5_doys"]
        flat_doys = era5_doys.reshape(-1)
        era5_pe   = sinusoidal_pe(flat_doys, self.d_model).reshape(B, 365, self.d_model)

        era5_rel  = torch.arange(365, device=device)
        era5_rp   = self.rel_pos_emb(era5_rel).unsqueeze(0)

        era5_tok  = era5_tok + era5_pe + era5_rp

        target_doys = batch["target_doy"]
        day_idx     = torch.arange(365, device=device).unsqueeze(0)
        era5_pad    = day_idx >= target_doys.unsqueeze(1)

        tokens.append(era5_tok)
        is_pad.append(era5_pad)

        seq      = torch.cat(tokens, dim=1)
        key_mask = torch.cat(is_pad,  dim=1)
        return seq, key_mask, spatial_start

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Returns sm_map : (B, n_depths, 224, 224)
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

        # ── 2. DEM pyramid from batch (pre-computed) ───────────────────
        dem_pyr = batch["dem_pyramid"].to(device)                      # (B, 4, 768)

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

        # ── 7. U-Net decoder → SM map ──────────────────────────────────
        return self.decoder(bottleneck, skip_L9, skip_L6, skip_L3)    # (B, n_depths, 224, 224)


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
