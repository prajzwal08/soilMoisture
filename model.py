"""
SoilMoistureModel — temporal trunk + fine CNN encoder + U-Net decoder, two heads
================================================================================
§48 (amending §46). Runbook: text/training_runbook.md §46, §48, §49.

This file REPLACED the patchwise arm (§34/§35.18); that code lives at tags `pw_stage2a-ep9` and
`pre-s1-decoder-aux`. The pooled U-Net baseline this trunk is ported from is frozen in
model_unet.py (tag `baseline-unet-temporal`). The trunk keeps the canonical fixes the patchwise
arm accumulated: 18 ERA5 columns (`era5/values18`), ERA5 staleness from real row dates, the
precomputed DOY table, and the driver/history split of annotation scales (§35.24-26).

  TRUNK  one temporal transformer, runs once per sample
    [ depth_CLS x3 | DEM pyr x4 | LULC pyr x4 | soil x4 | anchor L12 x196 |
      S2 pyr x MAX_S2*4 | S1 pyr x MAX_S1*4 | ERA5 x365 | SIF x50 | TWSA x12 ]
    -> bottleneck: the 196 anchor rows, (B, 768, 14, 14) at 160 m
    -> context:    mean of the valid non-spatial, non-CLS rows, (B, 768)
    -> depth_ctx:  the 3 CLS rows, (B, 3, 768)

  FINE ENCODER (§48.1) — light CNN on the most recent imagery on or before day D
    fine (B, 19, 112, 112) @ 20 m   S2 12 | S1 5 | DEM 2      (channel map: FINE_* below)
    lulc (B, 224, 224) long @ 10 m  class index, LULC_PAD = nodata
    stems -> 32 ch @112 -> E1 32 @112 (20 m), E2 64 @56 (40 m), E3 128 @28 (80 m)

  DECODER — coarse context up, measured detail down (§46.4)
    bottle 512 @14 -> up -> conv1(512+128) @28 -> up -> conv2(256+64) @56
                   -> up -> conv3(128+32) @112 = z (B, 64, 112, 112)
    each skip FiLM-modulated by `context`; the skip slices of conv{1,2,3}'s first conv are
    zero-initialised, so step 0 is exactly the bottleneck-only decoder.

  HEADS on z
    SM   3 disconnected per-depth heads, each FiLM'd by its own CLS row, bias = label_mean
         -> (B, 3, 112, 112); supervised at the station pixel (56, 56)
    LST  1x1 conv -> pixels 0..109 -> avg_pool 5 -> (B, 1, 22, 22) @ 100 m, in units of
         sigma_ST, PATTERN ONLY: its level is unconstrained and is never Kelvin (§48.9 item 4)

  GroupNorm everywhere a BatchNorm used to be (§48.2 item 8).
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

# Initialisation scale for the positional / modality annotations.
#
# TWO scales, not one, because the annotations serve two streams whose content differs in
# magnitude by ~21x (§35.25, measured):
#
#     driver content   era5_mlp / sif_mlp / twsa_mlp output      std 0.22
#     history content  raw frozen TerraMind L12 token            std 4.65
#
# A single shared table cannot suit both. At std 1.0 the annotation was ~450% of a driver
# token (the §35.24 bug — the driver token was mostly calendar) and a sensible ~21% of a
# history token. The pooled pyramid tokens and the anchor tokens here are frozen TerraMind
# features, so they take the HISTORY scale.
EMB_INIT_STD      = 0.02    # annotations on DRIVER tokens (era5, sif, twsa, soil)
HIST_EMB_INIT_STD = 1.0     # annotations on FROZEN TerraMind tokens (pyramids, anchor)


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


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with at most 8 groups and at least 4 channels per group.

    Per-sample statistics, identical in train and eval: BatchNorm's batch statistics are noisy
    at a few samples per GPU, are not synced across DDP ranks, and its eval-time running
    averages would be accumulated on batches that contain modality-dropped samples (§48.2
    item 8).
    """
    return nn.GroupNorm(max(1, min(8, c // 4)), c)


# ── Fine imagery layout (§48.3 + §48.9 item 1) ───────────────────────────────
#
# `fine` is (19, 112, 112) at 20 m, built by dataset.py in §46.3's worker order: normalised
# with TerraMind's constants, then invalid pixels zeroed. The valid flags are channels, so
# the network always sees "missing" explicitly rather than a plausible-looking zero.

FINE_S2    = slice(0, 12)    # 10 bands (B01/B09 dropped) | s2_valid | s2_age
FINE_S1    = slice(12, 17)   # VV | VH | s1_valid | s1_age | orbit (0 asc, 1 desc)
FINE_DEM   = slice(17, 19)   # DEM | dem_valid
FINE_CH    = 19
FINE_VALID = {"s2": 10, "s1": 14, "dem": 18}     # absolute channel index of each flag

LULC_N_CLASSES = 10          # TerraMind LULC indices 0..9
LULC_PAD       = 10          # nodata; maps to a fixed zero embedding
LULC_EMB_DIM   = 8

ENC_CH = (32, 64, 128)       # E1 @112 (20 m), E2 @56 (40 m), E3 @28 (80 m)


def _masked_avg_pool(x: torch.Tensor, m: torch.Tensor, k: int) -> torch.Tensor:
    """pool(x*m) / pool(m): an unbiased mean over valid pixels, 0 where none are valid."""
    num = F.avg_pool2d(x * m, k)
    den = F.avg_pool2d(m, k)
    return num / den.clamp_min(1e-6) * (den > 0)


class _ConvBlock(nn.Module):
    """2 x (3x3 conv -> GroupNorm -> ReLU). `stride` applies to the first conv only.

    The Sequential is named `.net` — the zero-init of the skip slices addresses
    `conv{i}.net[0].weight` (§46.7 trap 4).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _stem(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                         _gn(out_ch), nn.ReLU(inplace=True))


class FineEncoder(nn.Module):
    """
    Light CNN on the most recent imagery (§48.1): one stem per modality, then three levels.

        S2  12 -> 16 | S1 5 -> 8 | DEM 2 -> 4 | LULC emb 8 @10 m -> 2x2 mean -> 4   = 32 @112
        E1 32 @112   E2 64 @56   E3 128 @28                                  ~0.3 M params

    LULC is embedded at its native 10 m and THEN averaged to 20 m. A 20 m cell of 3 crop and
    1 forest pixel becomes 0.75*e_crop + 0.25*e_forest — algebraically the one-hot area
    fraction times a learned matrix, so mixed pixels survive (§48.2 item 3). The mean is over
    valid pixels only; LULC_PAD has a fixed zero vector.

    fine_skips="pool" is §46's parameter-free alternative kept as the ablation: masked average
    pooling of the 19 raw channels + the LULC embedding to each scale, then a 1x1 conv to the
    same widths. It sees one pixel's values, never a neighbourhood.

    Modality dropout (§48.2 item 9), training only: with probability `modality_dropout` per
    sample, zero ALL of S2's or S1's channels — data, valid flag and age together — so the
    network sees "missing", never "valid at the mean". It is applied here, on the device,
    so train and eval batches from the dataset are identical.
    """

    def __init__(self, fine_skips: str = "cnn", modality_dropout: float = 0.2):
        super().__init__()
        if fine_skips not in ("cnn", "pool"):
            raise ValueError(f"fine_skips must be 'cnn' or 'pool', got {fine_skips!r}")
        self.fine_skips       = fine_skips
        self.modality_dropout = modality_dropout

        self.lulc_emb = nn.Embedding(LULC_N_CLASSES + 1, LULC_EMB_DIM, padding_idx=LULC_PAD)

        if fine_skips == "cnn":
            self.stem_s2   = _stem(FINE_S2.stop - FINE_S2.start, 16)
            self.stem_s1   = _stem(FINE_S1.stop - FINE_S1.start, 8)
            self.stem_dem  = _stem(FINE_DEM.stop - FINE_DEM.start, 4)
            self.stem_lulc = _stem(LULC_EMB_DIM, 4)
            self.enc1 = _ConvBlock(32, ENC_CH[0])
            self.enc2 = _ConvBlock(ENC_CH[0], ENC_CH[1], stride=2)
            self.enc3 = _ConvBlock(ENC_CH[1], ENC_CH[2], stride=2)
        else:
            n_in = FINE_CH + LULC_EMB_DIM
            self.pool_proj = nn.ModuleList([nn.Conv2d(n_in, c, 1) for c in ENC_CH])

    def _drop_modality(self, fine: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return fine
        B = fine.shape[0]
        drop  = torch.rand(B, device=fine.device) < self.modality_dropout      # (B,)
        which = torch.rand(B, device=fine.device) < 0.5                        # True = S2
        keep  = torch.ones(B, FINE_CH, 1, 1, device=fine.device, dtype=fine.dtype)
        keep[:, FINE_S2] = (~(drop & which)).to(fine.dtype).view(B, 1, 1, 1)
        keep[:, FINE_S1] = (~(drop & ~which)).to(fine.dtype).view(B, 1, 1, 1)
        return fine * keep

    def _lulc_20m(self, lulc: torch.Tensor) -> torch.Tensor:
        """(B, 224, 224) long @ 10 m -> (B, 8, 112, 112) float @ 20 m, masked 2x2 mean."""
        e = self.lulc_emb(lulc).permute(0, 3, 1, 2)                    # (B, 8, 224, 224)
        m = (lulc != LULC_PAD).unsqueeze(1).to(e.dtype)                # (B, 1, 224, 224)
        return _masked_avg_pool(e, m, 2)

    def forward(self, fine: torch.Tensor, lulc: torch.Tensor):
        """Returns [E1 (B,32,112,112), E2 (B,64,56,56), E3 (B,128,28,28)]."""
        fine = self._drop_modality(fine.float())
        lulc = self._lulc_20m(lulc.long())

        if self.fine_skips == "cnn":
            x = torch.cat([
                self.stem_s2(fine[:, FINE_S2]),
                self.stem_s1(fine[:, FINE_S1]),
                self.stem_dem(fine[:, FINE_DEM]),
                self.stem_lulc(lulc),
            ], dim=1)                                                  # (B, 32, 112, 112)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            return [e1, e2, e3]

        # pool: each modality is averaged over its OWN valid pixels, so a 40 m cell with one
        # cloudy 20 m pixel is the mean of the other three, not diluted toward zero.
        x = torch.cat([fine, lulc], dim=1)                             # (B, 27, 112, 112)
        m = torch.ones_like(x[:, :1]).expand_as(x).clone()
        for name, sl in (("s2", FINE_S2), ("s1", FINE_S1), ("dem", FINE_DEM)):
            m[:, sl] = fine[:, FINE_VALID[name]:FINE_VALID[name] + 1]
        m[:, FINE_CH:] = (lulc.abs().sum(1, keepdim=True) > 0).to(x.dtype)
        outs = []
        for k, proj in zip((1, 2, 4), self.pool_proj):
            outs.append(proj(x if k == 1 else _masked_avg_pool(x, m, k)))
        return outs


# ── Decoder ──────────────────────────────────────────────────────────────────

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


LST_POOL = 5                 # 112 @ 20 m -> 22 @ 100 m over pixels 0..109 (landsat_target.py)
LST_N    = 22


class UNetDecoder(nn.Module):
    """
    14 -> 28 -> 56 -> 112. No up4 / conv4: nothing supervises 10 m (§46.4).

    The skips are the FINE encoder's features, not TerraMind L9/L6/L3 — tokens carry nothing
    below 160 m (§34.9). The first conv of each stage sees [upsampled path | skip]; the skip
    columns of its weight start at zero, so at step 0 the decoder IS the bottleneck-only
    decoder and the encoder is brought in by gradient, not by initialisation (§48.4).
    """

    def __init__(
        self,
        in_ch:     int   = 768,
        dec_ch:    tuple = (512, 256, 128, 64),
        n_depths:  int   = 3,
        d_context: int   = 768,
        head_bias_init: list[float] | None = None,
    ):
        super().__init__()
        c = dec_ch
        self.n_depths = n_depths

        self.bottle_proj = nn.Conv2d(in_ch, c[0], 1)
        self.film_skip   = nn.ModuleList([FiLMLayer(d_context, e) for e in ENC_CH[::-1]])

        self.up    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = _ConvBlock(c[0] + ENC_CH[2], c[1], dropout=0.15)   # @28
        self.conv2 = _ConvBlock(c[1] + ENC_CH[1], c[2], dropout=0.15)   # @56
        self.conv3 = _ConvBlock(c[2] + ENC_CH[0], c[3], dropout=0.15)   # @112
        with torch.no_grad():
            for conv, c_path in ((self.conv1, c[0]), (self.conv2, c[1]), (self.conv3, c[2])):
                conv.net[0].weight[:, c_path:].zero_()

        self.pre_head_drop = nn.Dropout(0.1)

        # Three DISCONNECTED depth heads (§46.1 rows 5-9): no star residual, so each predicts
        # absolutely and each bias starts at that depth's train-set mean. Zero-init would mean
        # "predict zero moisture", far outside the data.
        self.depth_film = nn.ModuleList([FiLMLayer(d_context, c[3]) for _ in range(n_depths)])
        self.heads      = nn.ModuleList([nn.Conv2d(c[3], 1, 1) for _ in range(n_depths)])
        # Zero WEIGHTS, bias = label_mean: each depth opens at exactly its own mean. With the
        # default init the 64 post-ReLU channels add +/-0.4 m3/m3 on top of the bias (measured
        # by verify_s48.py check 4: 0.05 / 0.56 / 0.56 against 0.17 / 0.19 / 0.19), which
        # opens training in Huber's linear regime — the §35.24 defect the bias init exists to
        # remove. The weights still get a gradient on step 1, since the map they read is not 0.
        for h in self.heads:
            nn.init.zeros_(h.weight)
        if head_bias_init is not None:
            if len(head_bias_init) != n_depths:
                raise ValueError(f"head_bias_init needs {n_depths} values, "
                                 f"got {len(head_bias_init)}")
            with torch.no_grad():
                for h, b in zip(self.heads, head_bias_init):
                    h.bias.fill_(float(b))

        # Thermal head: OUTSIDE the per-depth branch — Kelvin is not a depth. Reads z directly.
        self.head_lst = nn.Conv2d(c[3], 1, 1)

    def forward(self, bottleneck, skips, context, depth_ctx):
        # skips: [E1 @112, E2 @56, E3 @28]; context (B, d); depth_ctx (B, n_depths, d)
        e1, e2, e3 = skips
        x = self.bottle_proj(bottleneck)                                        # (B,512,14,14)
        x = self.conv1(torch.cat([self.up(x), self.film_skip[0](e3, context)], 1))  # @28
        x = self.conv2(torch.cat([self.up(x), self.film_skip[1](e2, context)], 1))  # @56
        z = self.conv3(torch.cat([self.up(x), self.film_skip[2](e1, context)], 1))  # @112
        x = self.pre_head_drop(z)

        # Readouts in fp32, outside autocast: epoch-to-epoch checkpoint decisions are made on
        # val differences of order 1e-5 m3/m3, below bf16's ~1e-3 absolute at SM = 0.5.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf, dcf = x.float(), depth_ctx.float()
            sm = torch.cat([self.heads[d](self.depth_film[d](xf, dcf[:, d, :]))
                            for d in range(self.n_depths)], dim=1)             # (B,3,112,112)
            lst_map = self.head_lst(xf)                                         # (B,1,112,112)
            n = LST_POOL * LST_N
            lst = F.avg_pool2d(lst_map[:, :, :n, :n], LST_POOL)                 # (B,1,22,22)
        return sm, lst, z


# ── Soil encoder ─────────────────────────────────────────────────────────────

class SoilEncoder(nn.Module):
    """
    Lightweight depthwise-separable CNN + 4-scale spatial pyramid.

    Input : (B, 21, 74, 74) float32 — NaN-free (pre-filled by dataset)
    Output: (B,  4, 768)    float32 — 4 static soil tokens

    Architecture (from architecture.md §4d), GroupNorm in place of BatchNorm (§48):
      Block 1: DWConv(21, 3×3) → PWConv(21→32) → GN → GELU  # (B,32,74,74)
      Block 2: DWConv(32, 3×3, s=2) → PWConv(32→64) → GN → GELU  # (B,64,37,37)
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
            _gn(self.MID_CH),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.MID_CH, self.MID_CH, 3, stride=2, padding=1, groups=self.MID_CH, bias=False),
            nn.Conv2d(self.MID_CH, c,           1, bias=False),
            _gn(c),
            nn.GELU(),
        )
        self.proj = nn.ModuleList([nn.Linear(c, d_model) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.block1(x)                                         # (B, 32, 74, 74)
        x  = self.block2(x)                                         # (B, 64, 37, 37)
        cy = cx = 18                                                 # centre of 37×37
        # Scales: input is 30 m/px, but block2 has stride 2 → cells here are 60 m, and each
        # cell has a 5-input-px receptive field. So a k×k window spans k×60 m and sees
        # (5 + 2(k-1))×30 m of input.
        t0 = x[:, :, cy:cy+1,   cx:cx+1  ].mean(dim=(-2, -1))     # 1×1   win 60 m,  RF 150 m
        t1 = x[:, :, cy-1:cy+2, cx-1:cx+2].mean(dim=(-2, -1))     # 3×3   win 180 m, RF 270 m
        t2 = x[:, :, cy-3:cy+4, cx-3:cx+4].mean(dim=(-2, -1))     # 7×7   win 420 m, RF 510 m
        t3 = x.mean(dim=(-2, -1))                                   # 37×37 win 2.22 km = full patch
        return torch.stack(
            [self.proj[i](t) for i, t in enumerate([t0, t1, t2, t3])], dim=1
        )                                                            # (B, 4, 768)


# ── Full model ───────────────────────────────────────────────────────────────

N_ERA5 = 18                  # era5/values18: skt dropped, ssrd_sum/strd_sum added (§43.12)


class SoilMoistureModel(nn.Module):
    """
    Args:
        n_depths        : SM depth bins (3), SM_DEPTHS order
        d_model         : token dimension (768)
        n_heads         : attention heads (12)
        n_layers        : trunk transformer layers (6)
        head_bias_init  : per-depth initial SM head bias in m3/m3 — train.py passes
                          driver_stats.json's `label_mean`
        fine_skips      : "cnn" (§48, default) or "pool" (§46's masked pool + 1x1, the ablation)
        modality_dropout: per-sample probability of zeroing S2 or S1 in the fine path (train)

    forward(batch) -> dict
        sm   (B, 3, 112, 112)  soil moisture map @ 20 m; station pixel (56, 56)
        lst  (B, 1, 22, 22)    thermal pattern @ 100 m, units of sigma_ST, level meaningless
        z    (B, 64, 112, 112) the shared map both heads read — train.py takes the gradient
                               norms for lambda here, never at parameter leaves (§46.5 item 28)
    """

    STATION_ROW = 56
    STATION_COL = 56

    def __init__(
        self,
        n_depths:         int   = 3,
        d_model:          int   = 768,
        n_heads:          int   = 12,
        n_layers:         int   = 6,
        drop_path_rate:   float = 0.1,
        head_bias_init:   list[float] | None = None,
        fine_skips:       str   = "cnn",
        modality_dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_depths = n_depths
        # Kept as an attribute: train.py's inert-CLS diagnostic and checkpoints read it. The
        # disconnected per-depth heads require the CLS rows, so it is not optional here.
        self.use_cls_depth = True

        # ── Driver encoders ───────────────────────────────────────────
        self.soil_encoder = SoilEncoder(d_model=d_model)
        self.era5_mlp = nn.Sequential(
            nn.Linear(N_ERA5, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))
        self.sif_mlp  = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))
        self.twsa_mlp = nn.Sequential(nn.Linear(1, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d_model))

        # ── Annotations ───────────────────────────────────────────────
        # Driver side (small, driver content is small)
        self.soil_modality_emb = nn.Embedding(1, d_model)
        self.era5_modality_emb = nn.Embedding(1, d_model)
        self.sif_modality_emb  = nn.Embedding(1, d_model)
        self.twsa_modality_emb = nn.Embedding(1, d_model)
        self.rel_pos_emb       = nn.Embedding(365, d_model)   # DRIVERS: era5, sif, twsa
        # History side (full scale, frozen TerraMind content is large)
        self.static_modality_emb  = nn.Embedding(2, d_model)  # DEM=0, LULC=1
        self.spatial_modality_emb = nn.Embedding(3, d_model)  # anchor: S2=0, S1 asc=1, desc=2
        # Satellite history: 0 = S2, 1 = S1 ascending, 2 = S1 descending. THREE, not two:
        # asc/desc backscatter differs by an amount comparable to the moisture signal, so a
        # shared S1 tag makes an orbit switch indistinguishable from a wetting event.
        self.hist_modality_emb = nn.Embedding(3, d_model)
        self.scale_emb         = nn.Embedding(4, d_model)     # pyramid level
        self.rel_pos_emb_hist  = nn.Embedding(365, d_model)   # HISTORY + anchor staleness
        self.spatial_row_emb   = nn.Embedding(14, d_model)
        self.spatial_col_emb   = nn.Embedding(14, d_model)

        for emb in (self.soil_modality_emb, self.era5_modality_emb,
                    self.sif_modality_emb, self.twsa_modality_emb, self.rel_pos_emb):
            nn.init.trunc_normal_(emb.weight, std=EMB_INIT_STD)
        for emb in (self.static_modality_emb, self.spatial_modality_emb, self.hist_modality_emb,
                    self.scale_emb, self.rel_pos_emb_hist,
                    self.spatial_row_emb, self.spatial_col_emb):
            nn.init.trunc_normal_(emb.weight, std=HIST_EMB_INIT_STD)

        # DOY code as a 367-row table (a pure function of one integer); persistent=False so it
        # follows .to(device) without entering the state_dict.
        self.register_buffer("doy_pe", circular_doy_pe(torch.arange(367), d_model),
                             persistent=False)

        # ── Depth CLS tokens ──────────────────────────────────────────
        # trunc_normal, not zero: with no positional code on these slots, zero-init makes all
        # three depth queries identical, and attention is permutation-equivariant over them.
        self.depth_tokens = nn.Parameter(torch.zeros(n_depths, d_model))
        nn.init.trunc_normal_(self.depth_tokens, std=0.02)

        # ── Temporal transformer ──────────────────────────────────────
        dpr = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.transformer_layers = nn.ModuleList([
            DropPathTransformerLayer(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True, norm_first=True),
                drop_prob=dpr[i])
            for i in range(n_layers)
        ])
        self.transformer_norm = nn.LayerNorm(d_model)

        # ── Fine path + decoder ───────────────────────────────────────
        self.fine_encoder = FineEncoder(fine_skips=fine_skips,
                                        modality_dropout=modality_dropout)
        self.decoder = UNetDecoder(in_ch=d_model, n_depths=n_depths, d_context=d_model,
                                   head_bias_init=head_bias_init)

    # ── Sequence ─────────────────────────────────────────────────────────────

    def _anchor_tokens(self, batch: dict, device) -> torch.Tensor:
        """Target-day spatial tokens: the anchor's L12 (B, 196, 768) + 2-D PE + sensor + age.

        The anchor is chosen by dataset.py's select_anchor_zarr: the most recent fully-clear
        acquisition on or before day D, falling back to the most recent regardless.
        """
        tok   = batch["anchor_l12"].to(device).float()                        # (B, 196, 768)
        rows  = torch.arange(14, device=device)
        pe    = (self.spatial_row_emb(rows).unsqueeze(1) +
                 self.spatial_col_emb(rows).unsqueeze(0)).reshape(196, self.d_model)
        orbit = batch["anchor_orbit"].to(device).long().clamp(0, 2)            # (B,)
        age   = batch["anchor_rel_pos"].to(device).long().clamp(0, 364)        # (B,)
        return (tok + pe.unsqueeze(0)
                + self.spatial_modality_emb(orbit).unsqueeze(1)
                + self.rel_pos_emb_hist(age).unsqueeze(1))

    def _build_sequence(self, batch: dict):
        """
        [ CLS x3 | DEM x4 | LULC x4 | soil x4 | anchor x196 | S2 x MAX_S2*4 | S1 x MAX_S1*4 |
          ERA5 x365 | SIF | TWSA ]

        Returns (seq (B, T, d), pad (B, T) True = ignore, spatial_start int).
        """
        device = next(self.parameters()).device
        d      = self.d_model
        era5   = batch["era5"].to(device).float()
        B      = era5.shape[0]
        toks, pads = [], []

        def _nopad(n):
            return torch.zeros(B, n, device=device, dtype=torch.bool)

        # depth CLS prefix
        toks.append(self.depth_tokens.unsqueeze(0).expand(B, -1, -1))
        pads.append(_nopad(self.n_depths))

        scale_e  = self.scale_emb.weight                                       # (4, d)
        static_w = self.static_modality_emb.weight                             # (2, d)

        # statics: DEM and LULC pyramids (pooled frozen TerraMind L12), then soil
        for i, key in enumerate(("dem_pyr", "lulc_pyr")):
            toks.append(batch[key].to(device).float() + scale_e + static_w[i])
            pads.append(_nopad(4))
        soil = self.soil_encoder(batch["soil_patch"].to(device).float())
        toks.append(soil + self.soil_modality_emb.weight)
        pads.append(_nopad(4))

        # anchor spatial tokens
        spatial_start = sum(t.shape[1] for t in toks)
        toks.append(self._anchor_tokens(batch, device))
        pads.append(_nopad(196))

        # satellite history: pooled pyramids + staleness + level + sensor/orbit
        for key in ("s2", "s1"):
            pyr   = batch[f"{key}_pyr"].to(device).float()                    # (B, T, 4, d)
            T     = pyr.shape[1]
            rel   = self.rel_pos_emb_hist(
                batch[f"{key}_rel_pos"].to(device).long().reshape(-1).clamp(0, 364)
            ).reshape(B, T, 1, d)
            if key == "s2":
                mod = self.hist_modality_emb.weight[0].view(1, 1, 1, d)
            else:
                orb = batch["s1_orbit"].to(device).long().clamp(0, 1)          # (B, T)
                mod = self.hist_modality_emb(orb + 1).unsqueeze(2)             # (B, T, 1, d)
            toks.append((pyr + rel + scale_e.view(1, 1, 4, d) + mod).reshape(B, T * 4, d))
            valid = batch[f"{key}_valid"].to(device).bool()                    # (B, T)
            pads.append((~valid).unsqueeze(-1).expand(-1, -1, 4).reshape(B, T * 4))

        # ERA5: staleness from the dataset's REAL row dates, never the slot index
        era5_doys = batch["era5_doys"].to(device).long()
        era5_rel  = batch["era5_rel_pos"].to(device).long().reshape(-1).clamp(0, 364)
        toks.append(self.era5_mlp(era5)
                    + self.doy_pe[era5_doys.reshape(-1).clamp(0, 366)].reshape(B, -1, d)
                    + self.rel_pos_emb(era5_rel).reshape(B, -1, d)
                    + self.era5_modality_emb.weight)
        pads.append(era5_doys == 0)

        # SIF and TWSA, both sparse; empty slots handled by the valid mask, never by a branch
        # that would drop an MLP out of the graph (DDP runs without find_unused_parameters).
        for key, mlp, mod_emb in (("sif",  self.sif_mlp,  self.sif_modality_emb),
                                  ("twsa", self.twsa_mlp, self.twsa_modality_emb)):
            vals = batch[key].to(device).float()
            doys = batch[f"{key}_doys"].to(device).long()
            rel  = batch[f"{key}_rel_pos"].to(device).long()
            toks.append(mlp(vals)
                        + self.doy_pe[doys.reshape(-1).clamp(0, 366)].reshape(B, -1, d)
                        + self.rel_pos_emb(rel.reshape(-1).clamp(0, 364)).reshape(B, -1, d)
                        + mod_emb.weight)
            pads.append(~batch[f"{key}_valid"].to(device).bool())

        return torch.cat(toks, 1), torch.cat(pads, 1), spatial_start

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        device = next(self.parameters()).device
        seq, pad, sp = self._build_sequence(batch)
        B = seq.shape[0]

        x = seq
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=pad)
        ctx = self.transformer_norm(x)                                         # (B, T, d)

        depth_ctx = ctx[:, :self.n_depths, :]                                  # (B, 3, d)
        # Collapse diagnostic, as a SUM plus count for epoch accumulation: the OUTPUT cosine
        # is what matters — the CLS parameters can stay near-orthogonal while six layers drive
        # all three rows to the same content, which is use_cls_depth being inert.
        self._last_depth_ctx   = depth_ctx.detach().float().sum(0)            # (3, d)
        self._last_depth_ctx_n = B

        bottleneck = ctx[:, sp:sp + 196, :].reshape(B, 14, 14, self.d_model).permute(0, 3, 1, 2)

        # FiLM context: mean of valid rows, excluding the CLS prefix and the spatial block
        keep = (~pad).clone()
        keep[:, :self.n_depths] = False
        keep[:, sp:sp + 196]    = False
        kf      = keep.unsqueeze(-1).to(ctx.dtype)
        context = (ctx * kf).sum(1) / kf.sum(1).clamp_min(1.0)                # (B, d)

        skips = self.fine_encoder(batch["fine"].to(device), batch["lulc"].to(device))
        sm, lst, z = self.decoder(bottleneck, skips, context, depth_ctx)
        return {"sm": sm, "lst": lst, "z": z}


# ── Losses ───────────────────────────────────────────────────────────────────

def masked_huber_loss(
    sm_map:        torch.Tensor,   # (B, n_depths, 112, 112)
    label:         torch.Tensor,   # (B, n_depths) — NaN where depth absent
    station_row:   int   = SoilMoistureModel.STATION_ROW,
    station_col:   int   = SoilMoistureModel.STATION_COL,
    delta:         float = 0.05,
    per_depth:     bool  = False,
    depth_weights: torch.Tensor | None = None,
    return_breakdown: bool = False,
):
    """Huber loss at the station pixel, ignoring depths with no observation.

    return_breakdown=True additionally returns (depth_sum, depth_cnt), both (n_depths,)
    float32, detached, on-device — raw SUMS for the caller to accumulate over the epoch and
    all_reduce(SUM) across ranks, which is only correct on sums. The scalar `loss` is
    byte-identical with and without the flag.

    per_depth=True weights each valid (sample, depth) pair by a FIXED w_d supplied by the
    caller (inverse per-depth frequency over the training set), never by batch composition:
    a batch-mean-per-depth form hands each deep sample 1/n_d(batch) of the gradient, which
    is not a function of the dataset. w_d = 1 reduces exactly to the pooled branch.
    See training_runbook.md §19.3 for why the breakdown and the scalar differ on purpose.
    """
    pred = sm_map[:, :, station_row, station_col]                      # (B, n_depths)

    if return_breakdown:
        valid     = ~torch.isnan(label)
        lab       = torch.nan_to_num(label, nan=0.0)
        elem      = F.huber_loss(pred.detach(), lab, delta=delta, reduction="none")
        # torch.where, NOT `elem * valid`: nan * False is nan, and a non-finite prediction at
        # an unlabelled depth would otherwise turn the per-depth diagnostic nan on every rank.
        depth_sum = torch.where(valid, elem, elem.new_zeros(())).sum(0).float()
        depth_cnt = valid.sum(0).float()

    mask = ~torch.isnan(label)
    if per_depth:
        w = (torch.ones_like(label) if depth_weights is None else
             depth_weights.to(device=label.device, dtype=label.dtype).expand_as(label))
        wm    = torch.where(mask, w, torch.zeros_like(w))
        elem  = F.huber_loss(pred, torch.nan_to_num(label, nan=0.0), delta=delta,
                             reduction="none")
        denom = wm.sum()
        loss  = ((elem * wm).sum() / denom) if denom > 0 else pred.sum() * 0.0
    else:
        loss = (F.huber_loss(pred[mask], label[mask], delta=delta, reduction="mean")
                if mask.any() else pred.sum() * 0.0)

    if return_breakdown:
        return loss, depth_sum, depth_cnt
    return loss


def lst_pattern_loss(
    lst_pred:  torch.Tensor,   # (B, 1, 22, 22) model output, units of sigma_ST
    lst_obs:   torch.Tensor,   # (B, 22, 22) Kelvin, NaN where no retrieval / no overpass
    sigma_st:  float,
    delta:     float = 1.0,
    min_cells: int   = 2,
    return_count: bool = False,
):
    """Pattern-only thermal loss (§46.5 item 27, §48.9 item 4, alpha = 0).

    Both fields are centred over the SAME valid cells, so any tile-level error cancels
    exactly and the model cannot score by knowing "hot day". The observed anomaly is divided
    by sigma_ST (lst_stats.json, 2.7066 K) and the head predicts in those units, so delta=1.0
    means +/- 1 sigma_ST: quadratic within normal within-tile spread, linear beyond (§49.5).

    A sample on a non-overpass day has no valid cell and contributes nothing — the sample-level
    mask falls out of the cell mask. Samples with fewer than `min_cells` valid cells are
    dropped (a single cell has no pattern), matching compute_lst_stats.py.

    Returns the mean over all valid cells in the batch (0-graph-connected if none), and with
    return_count=True also the number of cells as a detached float.
    """
    pred  = lst_pred[:, 0].float()                                     # (B, 22, 22)
    valid = torch.isfinite(lst_obs)
    n     = valid.flatten(1).sum(1)                                    # (B,)
    valid = valid & (n >= min_cells).view(-1, 1, 1)
    vf    = valid.to(pred.dtype)
    cnt   = vf.flatten(1).sum(1).clamp_min(1.0).view(-1, 1, 1)

    obs   = torch.nan_to_num(lst_obs.float(), nan=0.0) / sigma_st
    obs_c = obs  - (obs  * vf).flatten(1).sum(1).view(-1, 1, 1) / cnt
    prd_c = pred - (pred * vf).flatten(1).sum(1).view(-1, 1, 1) / cnt

    elem  = F.huber_loss(prd_c, obs_c, delta=delta, reduction="none")
    total = vf.sum()
    loss  = (torch.where(valid, elem, elem.new_zeros(())).sum() / total
             if total > 0 else pred.sum() * 0.0)
    if return_count:
        return loss, total.detach()
    return loss
