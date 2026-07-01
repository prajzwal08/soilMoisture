# Multimodal Soil Moisture & Flux Model — Architecture Blueprint

> Last updated: 2026-05-13

---

## 0. Scientific Description

We present a multimodal neural land surface model that predicts daily soil moisture profiles and latent heat flux (LE) at 160 m spatial resolution, trained jointly on point observations from the International Soil Moisture Network (ISMN) and FLUXNET eddy-covariance towers. The model is designed for near-real-time estimation: given all observations available up to a target day T, it produces a spatially continuous output without any location-specific parameters, enabling direct generalisation to unobserved regions.

**Input encoding.** The model ingests observations across a range of spatial and temporal resolutions. High-resolution surface state is captured through Sentinel-2 Level-2A multispectral imagery (12 bands, ~10 m) and Sentinel-1 RTC synthetic aperture radar backscatter (VH/VV, ~10 m), both acquired within a rolling 365-day window ending at T. All Sentinel-2 acquisitions are cloud-masked at the 16×16 pixel token level using a pre-trained CloudSEN12 model applied offline. Static landscape context is provided by the Copernicus GLO-30 Digital Elevation Model, ESA WorldCover land use/land cover map, and soil physical and chemical properties from the OpenLandMap-soildb dataset at 30 m resolution. Meteorological forcing is drawn from the ERA5-Land reanalysis at ~9 km resolution, providing 19 daily variables (temperature, dewpoint, skin temperature, wind components, surface pressure, and total precipitation, each as daily mean, minimum, and maximum where applicable) for all 365 days in the rolling window. Two coarse-scale auxiliary inputs are optionally incorporated: Solar-Induced Fluorescence from TROPOMI (~3.5 km, sparse daily) and Terrestrial Water Storage Anomalies from GRACE (~300 km, monthly).

**Feature extraction.** Sentinel-1, Sentinel-2, DEM, and LULC imagery are processed through TerraMind Base, a frozen Vision Transformer pre-trained on multi-modal Earth observation data (12 layers, 768-dim, 12 heads). TerraMind produces a 14×14 grid of 768-dimensional spatial tokens per acquisition (196 tokens, each representing a 160 m patch). Intermediate features at layers 3, 6, 9, and 12 (L3, L6, L9, L12) are extracted via forward hooks. For each historical S2 or S1 acquisition, the 196 L12 tokens are compressed into four summary tokens through a Latent Cross-Attention bottleneck: four learnable query vectors attend over all 196 spatial tokens as keys and values, with cloud-contaminated tokens suppressed via a key padding mask. This content-adaptive compression replaces the fixed geometric spatial pyramid, allowing the model to dynamically focus on informationally rich, cloud-free regions regardless of their spatial position. Static DEM and LULC tokens are produced analogously without cloud masking. Soil physical properties are encoded through a lightweight depthwise-separable convolutional network applied to the 74×74 pixel (30 m) soil patch, followed by a four-level spatial pyramid producing four 768-dimensional static tokens representing scales from 30 m to 1.1 km. ERA5-Land variables are first normalised using a Yeo-Johnson Power Transform fitted on the training split to correct for scale differences spanning eight orders of magnitude and distributional skewness in precipitation, and then projected to 768 dimensions through a two-layer MLP. SIF and TWSA observations are each encoded through separate single-variable MLPs with sparse token injection on observation days only, with mid-month day-of-year assignment for TWSA.

**Temporal Transformer.** All encoded tokens are assembled into a single sequence of approximately 680–730 tokens and processed by a six-layer bidirectional Transformer encoder (768-dim, 12 heads, pre-norm, MLP ratio 4). Causality is enforced by masking ERA5 tokens for days after T via key padding mask, while all visible tokens attend freely to one another without autoregressive masking. Three positional signals are additively combined per token: a sinusoidal encoding of the absolute calendar day-of-year for season awareness; a learned relative position embedding (indices 0 to 364) encoding the token's temporal order within the rolling window and resolving the year-boundary ambiguity; and a unified learned modality type embedding (10 entries) that distinguishes all token types, including ERA5, SIF, TWSA, historical S2 and S1 compressed tokens, target-day spatial tokens, DEM, LULC, and soil. Satellite history tokens additionally receive a learned scale embedding distinguishing the four latent query levels. Modality dropout is applied during training (50% for SIF and TWSA; 20% for S1 or S2 individually; 10% for S1 and S2 together) to enable inference-time ablations and graceful degradation when modalities are unavailable.

**Soil moisture prediction.** The target day T is represented by 196 uncompressed L12 tokens from the most recent cloud-free Sentinel-2 or Sentinel-1 acquisition at or before T, augmented with learned two-dimensional spatial positional encodings and a modality embedding. The Transformer output at these 196 spatial positions is reshaped to a 14×14×768 spatially structured bottleneck and decoded to a 224×224 soil moisture map through a four-stage U-Net decoder with bilinear upsampling and convolutional refinement blocks. Skip connections from L3, L6, and L9 of the most-recent acquisition are incorporated at each upsampling stage to recover fine spatial detail. To mitigate temporal ghosting from stale skip connections, a learned staleness gate applies a sigmoid decay $\text{gate} = \sigma(-\Delta t / \tau)$ to all skip contributions, where $\Delta t$ is the number of days since the most recent acquisition and $\tau$ is a learned scalar; fresh imagery is fully trusted while stale imagery progressively yields to the temporally-informed bottleneck. The decoder outputs predictions for four depth bins (0–10, 10–20, 20–40, 40–100 cm) simultaneously through a final 1×1 convolution, and is supervised at the station pixel (row 112, column 112) via a masked Huber loss applied only to depth bins with valid observations.

**Flux prediction.** Latent heat flux (LE) estimation employs a physically motivated Gaussian footprint aggregation module. The footprint head is a multilayer perceptron that receives the mean-pooled transformer spatial output tokens, the mean-pooled DEM static tokens encoding local terrain, and ERA5 wind speed and direction for day T, and predicts three parameters of a 2D Gaussian ellipse: the upwind peak distance $d_\text{peak}$, the crosswind spread $\sigma_\text{cross}$, and the along-wind spread $\sigma_\text{along}$. The ERA5 wind direction rotates the ellipse on the 14×14 spatial grid. DEM tokens are included to allow the footprint MLP to compensate for local topographic deflection of the 9 km ERA5 wind vector. The resulting normalised Gaussian weights aggregate the 196 transformer spatial output tokens into a single 768-dimensional footprint-weighted flux token, which is passed to a single linear head (768→1) predicting LE only.

**Training.** Training proceeds in two phases. In Phase 1, the model is trained on ISMN stations only using the soil moisture Huber loss until convergence, establishing a robust spatiotemporal backbone. In Phase 2, FLUXNET tower observations are incorporated jointly with a Huber loss on LE; the backbone and soil moisture head are fine-tuned at a reduced learning rate (1×10⁻⁵) while the LE head is trained from scratch at the full learning rate (1×10⁻⁴). Missing labels are masked per station: ISMN stations contribute only the soil moisture loss and FLUXNET stations contribute only the LE loss. Models are evaluated on three generalisation splits — out-of-space (held-out station networks), out-of-time (held-out years), and out-of-space-and-time — against SMAP and ESA CCI baselines using MSE, MAE, and unbiased RMSE.

---

## 1. System Overview

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                                       INPUTS                                              ║
║                                                                                           ║
║ ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────┐ ┌──────────┐ ┌───────┐ ┌──────┐ ┌───────┐  ║
║ │ S2-L2A  │ │ S1-RTC  │ │  DEM  │ │ LULC  │ │ERA5-Land │ │  SIF  │ │ TWSA │ │OpenLM │  ║
║ │ up to T │ │ up to T │ │static │ │static │ │days 1→T  │ │sparse │ │monthl│ │ 30 m  │  ║
║ │  ~10m   │ │  ~10m   │ │  10m  │ │  10m  │ │  ~9 km   │ │3.5 km │ │300km │ │ soil  │  ║
║ └────┬────┘ └────┬────┘ └───┬───┘ └───┬───┘ └────┬─────┘ └───┬───┘ └──┬───┘ └───┬───┘  ║
╚══════╪═══════════╪══════════╪══════════╪══════════╪═══════════╪════════╪══════════╪══════╝
       └─────┬─────────────────┘          │          │           │        │          │
             │  (cloud-masked, DoY ≤ T)   │          │           │        │          │
             ▼                            ▼          ▼           ▼        ▼          ▼
  ┌──────────────────────┐   ┌────────────────┐ ┌────────┐ ┌────────────────────┐ ┌──────┐
  │  TerraMind Base      │   │  TerraMind     │ │ ERA5   │ │ SIF MLP            │ │ Soil │
  │  (frozen, 12 layers) │   │  (frozen)      │ │  MLP   │ │ TWSA MLP           │ │ CNN  │
  │  14×14 tokens output │   │  DEM + LULC    │ │ +YJ PT │ │ (separate, sparse) │ │      │
  └──────────────────────┘   └────────────────┘ └────────┘ └────────────────────┘ └──────┘
             │                       │                │               │               │
      skip L3,L6,L9             4×768 static      365×768       sparse tokens    4×768 static
      + L12 → latent CA          tokens each        tokens                         tokens
             │                       │                │               │               │
             └───────────────────────┴────────────────┴───────────────┴───────────────┘
                                                      │
                              ┌───────────────────────────────────────────┐
                              │           Temporal Transformer             │
                              │  6 layers · 768-dim · 12 heads            │
                              │  Bidirectional · Causal via key_mask       │
                              │  Unified modality type embedding (all toks)│
                              └───────────────────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┤
                       │                              │
                       ▼                              ▼
        ┌──────────────────────────┐   ┌──────────────────────────────┐
        │   SM Head (ISMN only)    │   │  Flux Heads (FLUXNET only)   │
        │   U-Net Decoder          │   │  Gaussian Footprint Head     │
        │   → SM map 224×224       │   │  → LE Head (768→1)           │
        │   Huber loss @ station   │   │  → Huber loss on LE          │
        └──────────────────────────┘   └──────────────────────────────┘
```

---

## 2. Primary Use Case

**Near-real-time soil moisture estimation**: given all observations available up to today (day T), produce a SM map for day T. No future data used. No station-specific parameters — the model generalises from ISMN/FLUXNET training stations to any location at inference.

| Input | Source | Notes |
|---|---|---|
| S2-L2A (cloud-masked, DoY ≤ T) | CDSE / OpenEO | ~5-day revisit; cloud mask from CloudSEN12 |
| S1-RTC (DoY ≤ T) | CDSE | ~6-day revisit; cloud-penetrating |
| DEM | Copernicus GLO-30 | Static, global |
| LULC | ESA WorldCover | Static, global |
| OpenLandMap-soildb | world.soils.app / Zenodo | Static 30m; 5-year composites |
| ERA5-Land days 1→T | CDS / ECMWF | Always complete; last ~5 days: ECMWF HRES |
| SIF (TROPOMI) | Copernicus | Sparse daily, cloud-permitting, ~3.5 km |
| TWSA (GRACE) | NASA / GFZ | Monthly anomaly, ~300 km |

---

## 3. TerraMind Encoder (frozen)

```
TerraMind Base — ViT-Base: 12 layers, 768-dim, 12 heads. Frozen throughout training.

Input per acquisition:
  S2-L2A : 224×224 @ 10m  (12 spectral bands, cloud-masked)
  S1-RTC : 224×224 @ 10m  (2 channels: VH, VV)
  DEM    : 224×224 @ 10m  (static)
  LULC   : 224×224 @ 10m  (static)

  Patch size  : 16×16 px = 160 m per token
  Token grid  : 14×14 = 196 tokens per acquisition
  Token dim   : 768

Hook extraction (forward hooks on encoder blocks):
  L3  (block 2)  → (B, 196, 768)  fine/low-level    → U-Net skip connection
  L6  (block 5)  → (B, 196, 768)  mid-level          → U-Net skip connection
  L9  (block 8)  → (B, 196, 768)  deep               → U-Net skip connection
  L12 (block 11) → (B, 196, 768)  semantic           → Latent CA bottleneck
```

### 3a. Latent Cross-Attention Bottleneck (historical S2/S1)

Replaces spatial pyramid pooling. Applied per acquisition to L12 tokens.

```
LatentCrossAttention:
  queries   : nn.Parameter(4, 768)  initialised N(0, 0.02)
  cross_attn: nn.MultiheadAttention(d_model=768, n_heads=8, batch_first=True)

  Q = queries.expand(B, 4, 768)          # (B, 4, 768)
  K = V = L12 tokens                     # (B, 196, 768)
  key_padding_mask = ~cloud_mask         # True = ignore (PyTorch convention)
                                         # cloud_mask: True = cloud-free token
  → output: (B, 4, 768)

Two separate modules:
  s2_bottleneck — with cloud_mask (B, 196) from CloudSEN12
  s1_bottleneck — cloud_mask=None (radar, cloud-penetrating)

Post-compression embeddings added (all additive):
  + sinusoidal DoY PE       (absolute calendar day, season)
  + learned rel-pos emb     nn.Embedding(365, 768)  (0=oldest, 364=today)
  + learned scale emb       nn.Embedding(4, 768)    (distinguishes 4 latent queries)
  + learned modality type   nn.Embedding(10, 768)   (index 3=S2, 4=S1)

→ 4 × 768 tokens per acquisition
```

### 3b. Static Tokens — DEM and LULC

Pre-computed at data-loading time (saved as `dem_pyramid.pt`, `lulc_pyramid.pt`).

```
TerraMind L12 (196 × 768)
  → LatentCrossAttention (no cloud mask)
  → 4 × 768 static tokens
  + learned scale emb    nn.Embedding(4, 768)
  + learned modality type nn.Embedding(10, 768)  (index 7=DEM, 8=LULC)

No temporal PE. Always visible to all transformer layers.
```

### 3c. Target-Day Spatial Tokens (196 uncompressed)

Most recent cloud-free acquisition ≤ T — S2 if more recent than S1, otherwise S1.

```
TerraMind L12 → all 196 tokens kept (no compression)
  + 2D spatial PE: learned row emb nn.Embedding(14, 768)
                 + learned col emb nn.Embedding(14, 768)
  + learned modality type nn.Embedding(10, 768)  (index 5=S2, 6=S1)

→ 196 × 768 tokens
Transformer output at these 196 positions → reshape (B, 14, 14, 768)
→ spatially-structured bottleneck for U-Net decoder
```

### 3d. Skip Connections for U-Net

L3/L6/L9 from the most-recent acquisition ≤ T feed the U-Net decoder as skip connections.

**Pre-computation (run once, before training or inference):**
Every TerraMind forward pass saves all four layers:
```
{stem}_L12.pt   (196, 768) fp16   ← history tokens for all acquisitions
{stem}_L9.pt    (196, 768) fp16   ← skip connection (most-recent only)
{stem}_L6.pt    (196, 768) fp16   ← skip connection (most-recent only)
{stem}_L3.pt    (196, 768) fp16   ← skip connection (most-recent only)
```
One forward pass → hooks capture all four → save all four. No extra compute.

**During training:**
L3/L6/L9 are computed **live** from the raw patch (1 TerraMind pass per sample).
Only 1 pass needed — acceptable cost. Raw patch already loaded in the batch.

**During map inference:**
No TerraMind runs at all. Load pre-cached L12 for all history tokens,
load pre-cached L3/L6/L9 of the most-recent acquisition as skip connections.

The transformer encodes image staleness implicitly via the rel-pos embedding.
No separate `days_ago` signal needed in the decoder.

---

## 4. Encoders

### 4a. ERA5-Land Meteo Encoder

19 daily variables: `t2m_mean/min/max`, `d2m_mean/min/max`, `skt_mean/min/max`,
`u10_mean/min/max`, `v10_mean/min/max`, `sp_mean/min/max`, `tp_sum`.

```
Preprocessing — Yeo-Johnson Power Transform (fit once on training split):
  sklearn PowerTransformer(method='yeo-johnson').fit(era5_train)
  Stored: lambdas_ (19,), mean_ (19,), scale_ (19,)
  Path:   /home/khanalp/data/soilmoisture/era5_pt_params/
  Applied uniformly to all 19 variables at dataset load time.
  Handles 8-orders-of-magnitude scale difference + precipitation skewness.

MLP: 19 → 256 (GELU) → 768    (weights learned during training)
  → 1 × 768 token per day
  + sinusoidal DoY PE
  + learned rel-pos emb  nn.Embedding(365, 768)
  + learned modality type nn.Embedding(10, 768)  (index 0)

→ 365 tokens; days after T: key_padding_mask=True (invisible to attention)
```

### 4b. SIF Encoder (TROPOMI, ~3.5 km)

```
Input: 1 scalar per valid observation day
MLP: 1 → 256 (GELU) → 768
  → 1 token per observation (sparse — only days with valid retrievals)
  + sinusoidal DoY PE
  + learned rel-pos emb
  + learned modality type  nn.Embedding(10, 768)  (index 1)

Modality dropout p=0.5 during training → clean ablations at inference
```

### 4c. TWSA Encoder (GRACE, ~300 km)

```
Input: 1 scalar per month (terrestrial water storage anomaly)
MLP: 1 → 256 (GELU) → 768
  → 1 token per month (up to 12 per year)
  + sinusoidal DoY PE using mid-month DoY (Jan=15, Feb=46, Mar=74, ...)
  + learned rel-pos emb
  + learned modality type  nn.Embedding(10, 768)  (index 2)

Modality dropout p=0.5 during training → clean ablations at inference
```

### 4d. Soil Properties Encoder (OpenLandMap-soildb, 30 m)

Replaces SoilGrids 250 m point MLP. At 30 m, a 2.24 km chip contains ~74×74 pixels
with real field-scale spatial heterogeneity.

```
Data: OpenLandMap-soildb — 21 channels per pixel (3 depths × 7 properties)
  Depths: 0–30 cm, 30–60 cm, 60–100 cm
  Per depth: clay%, sand%, silt%, SOC content, SOC density, bulk density, pH
  Single composite: 2020–2022 (only composite with full 7-property coverage;
    bd/clay/sand/silt not available in earlier periods)
  Input patch: (B, 21, 74, 74)   — static, same patch for every sample
```

**Station filtering — soil_patch_ok flag:**
19 stations (1.8% of 1048) have 100% NaN across the entire patch due to geographic
coverage gaps in the source COG (Hawaiian volcanic islands, Greenland ice sheet,
Tibetan Plateau permafrost, Alaskan tundra). These are excluded from training via
the `soil_patch_ok` column in `station_splits.csv`. No special model code path needed.
1029 stations remain.

**Partial NaN handling — nearest-neighbour fill:**
357 stations have some NaN pixels (edge effects where the patch bbox overhangs the
COG boundary — typically <10% of pixels). These are filled at dataset load time using
nearest-neighbour propagation (scipy.ndimage.distance_transform_edt): each NaN pixel
is replaced by the value of its closest valid pixel. This is geophysically sound —
soil properties are spatially smooth. The CNN never receives NaN inputs.

```python
# Applied per station at dataset load time (once, not per training step)
def fill_soil_nans(patch: np.ndarray) -> np.ndarray:
    # patch: (21, 74, 74)
    from scipy.ndimage import distance_transform_edt
    out = patch.copy()
    for c in range(patch.shape[0]):
        mask = np.isnan(out[c])
        if mask.any():
            _, idx = distance_transform_edt(mask, return_indices=True)
            out[c] = out[c][tuple(idx)]
    return out
```

```
Lightweight CNN (weights learned during training):
  Block 1 — feature extraction (stride 1):
    DWConv(21, 3×3) → PWConv(21→32) → BN → GELU    # (B, 32, 74, 74)
  Block 2 — spatial compression (stride 2):
    DWConv(32, 3×3, s=2) → PWConv(32→64) → BN → GELU  # (B, 64, 37, 37)

Spatial pyramid — 4 centre-crop scales → 4 × 768 static tokens:
  centre 1×1  → mean → Linear(64→768) + scale_emb[0]  # ~30 m   station pixel
  centre 3×3  → mean → Linear(64→768) + scale_emb[1]  # ~90 m   neighbourhood
  centre 7×7  → mean → Linear(64→768) + scale_emb[2]  # ~210 m  field scale
  full  37×37 → mean → Linear(64→768) + scale_emb[3]  # ~1.1 km landscape

  + learned modality type  nn.Embedding(10, 768)  (index 9)
  → 4 × 768 static soil tokens
  Parameter count: ~211 K  (input channels 7→21, all other dims unchanged)
```

**Design decisions considered and rejected:**
- Patch embed + Latent Cross-Attention with NaN masking: architecturally cleaner but
  adds complexity without benefit since pre-fill already resolves all partial-NaN cases.
- PVT v2 B0 / MMST-ViT: designed for temporal multi-modal fusion on natural images;
  3.7M params, poor ImageNet→soil transfer, no NaN handling advantage over pre-fill.
- Learned missing_soil_emb for 19 fully-NaN stations: valid but unnecessary since
  those stations are removed — they are scientifically unusual terrain where soil
  moisture prediction from soil properties is questionable regardless.

---

## 5. Unified Modality Type Embedding

One `nn.Embedding(10, 768)` added to **every** token in the sequence.
Tells the transformer what kind of signal each token carries.

| Index | Modality |
|---|---|
| 0 | ERA5-Land daily |
| 1 | SIF (TROPOMI) |
| 2 | TWSA (GRACE) |
| 3 | S2 history (compressed, 4 latent tokens) |
| 4 | S1 history (compressed, 4 latent tokens) |
| 5 | Target-day S2 (196 uncompressed spatial) |
| 6 | Target-day S1 (196 uncompressed spatial) |
| 7 | DEM (static, 4 latent tokens) |
| 8 | LULC (static, 4 latent tokens) |
| 9 | Soil (OpenLandMap CNN, 4 static tokens) |

---

## 6. Temporal Transformer — Sequence Composition

```
Full token sequence per sample (target day T):

  ┌──────────────────────────────────────────────────────────────────────┐
  │  STATIC PREFIX  (no temporal PE, always visible)                      │
  │                                                                       │
  │  DEM   → latent CA → 4 × 768  + scale emb + modality emb (7)        │
  │  LULC  → latent CA → 4 × 768  + scale emb + modality emb (8)        │
  │  Soil  → CNN+pyr  → 4 × 768  + scale emb + modality emb (9)        │
  │                                              subtotal: 12 tokens     │
  ├──────────────────────────────────────────────────────────────────────┤
  │  TARGET-DAY SPATIAL TOKENS  (196 uncompressed)                        │
  │                                                                       │
  │  Most recent valid image ≤ T (cloud-free S2 or S1, whichever later) │
  │  TerraMind L12 → 196 tokens + 2D spatial PE + modality emb (5 or 6) │
  │                                              subtotal: 196 tokens    │
  │  Transformer output here → (B, 768, 14, 14) U-Net bottleneck        │
  ├──────────────────────────────────────────────────────────────────────┤
  │  HISTORICAL SATELLITE TOKENS  (DoY ≤ T, compressed)                  │
  │                                                                       │
  │  S2: N_s2 acq × 4 tokens  + DoY PE + rel-pos + scale emb + type (3)│
  │  S1: N_s1 acq × 4 tokens  + DoY PE + rel-pos + scale emb + type (4)│
  │                                              subtotal: variable      │
  ├──────────────────────────────────────────────────────────────────────┤
  │  DENSE DAILY TOKENS                                                   │
  │                                                                       │
  │  ERA5: 365 tokens  (days > T: key_padding_mask=True)                 │
  │        + DoY PE + rel-pos + modality type (0)                        │
  │                                              subtotal: 365 (T active)│
  ├──────────────────────────────────────────────────────────────────────┤
  │  SPARSE TOKENS  (modality dropout p=0.5 during training)             │
  │                                                                       │
  │  SIF  : 0–50 tokens/year  + DoY PE + rel-pos + type (1)             │
  │  TWSA : ~12 tokens/year   + mid-month DoY PE + rel-pos + type (2)   │
  └──────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │                    Temporal Transformer                               │
  │                                                                       │
  │  Layers    : 6                                                        │
  │  Dim       : 768                                                      │
  │  Heads     : 12                                                       │
  │  MLP ratio : 4  (hidden = 3072)                                       │
  │  Norm      : Pre-norm (norm_first=True)                               │
  │  Dropout   : 0.1                                                      │
  │                                                                       │
  │  Attention : Bidirectional within visible context.                    │
  │              Future ERA5 days suppressed via key_padding_mask —       │
  │              NOT a causal attention mask. All visible tokens          │
  │              attend freely to each other.                             │
  │                                                                       │
  │  Per-token positional encoding:                                       │
  │    (1) Sinusoidal absolute DoY — season awareness                     │
  │    (2) Learned rel-pos emb nn.Embedding(365, 768) — order in window  │
  │    (3) Learned modality type emb nn.Embedding(10, 768)               │
  │    (4) Learned scale emb nn.Embedding(4, 768) — spatial/latent tokens│
  └──────────────────────────────────────────────────────────────────────┘
```

### Token Count Summary

| Token type | Count | Notes |
|---|---|---|
| DEM static | 4 | Scale emb + modality type (7) |
| LULC static | 4 | Scale emb + modality type (8) |
| Soil static | 4 | Scale emb + modality type (9) |
| Target-day spatial | 196 | 2D spatial PE + modality type (5 or 6) |
| S2 history | N_s2 × 4 | DoY PE + rel-pos + scale emb + type (3) |
| S1 history | N_s1 × 4 | DoY PE + rel-pos + scale emb + type (4) |
| ERA5 | 365 (T active) | DoY PE + rel-pos + type (0) |
| SIF | 0–50 | DoY PE + rel-pos + type (1) |
| TWSA | ~12 | Mid-month DoY PE + rel-pos + type (2) |
| **Total** | **~680–730** | |

---

## 7. SM Prediction Head — U-Net Decoder

```
Transformer output at 196 target-day spatial positions:
  (B, 196, 768) → reshape → (B, 14, 14, 768) → permute → (B, 768, 14, 14)

  ┌─────────────────────────────────────────────────────┐
  │                   UNetDecoder                        │
  │                                                      │
  │  14×14×768  → Conv+BN+ReLU → 14×14×512             │
  │     ↓ Upsample ×2                                   │
  │  28×28×512  + L9 skip → Conv → 28×28×256           │
  │     ↓ Upsample ×2                                   │
  │  56×56×256  + L6 skip → Conv → 56×56×128           │
  │     ↓ Upsample ×2                                   │
  │ 112×112×128 + L3 skip → Conv → 112×112×64          │
  │     ↓ Upsample ×2                                   │
  │ 224×224×64            → Conv → 224×224×32           │
  │     ↓ Conv2d(32 → n_depths)                         │
  │  SM map: (B, n_depths, 224, 224) @ 10 m            │
  └─────────────────────────────────────────────────────┘

Skip connections L3/L6/L9: one TerraMind pass on the most-recent raw patch.

Staleness gate — learned confidence decay on skip connections:
  days_ago = target_doy − most_recent_acq_doy   # (B,) already in batch
  gate     = sigmoid(−days_ago / τ)             # τ: learned scalar parameter
  skip_L9  = skip_L9 * gate.view(B, 1, 1, 1)   # fresh→1, stale→0
  skip_L6  = skip_L6 * gate.view(B, 1, 1, 1)
  skip_L3  = skip_L3 * gate.view(B, 1, 1, 1)

Fresh image (days_ago=0): gate≈1, skip connections fully trusted.
Stale image (days_ago=30+): gate→0, decoder relies on temporal bottleneck only.
Prevents "ghosting" where SM map reflects surface state from weeks ago.

Loss:
  pred = sm_map[:, :, row=112, col=112]    # (B, n_depths) at station pixel
  mask = ~isnan(label)
  L_SM = Huber(pred[mask], label[mask])

Depth bins: 0–10 cm · 10–20 cm · 20–40 cm · 40–100 cm
```

---

## 8. Flux Prediction Heads

### 8a. Gaussian Footprint Head

Aggregates the 196 transformer spatial output tokens into one footprint-weighted
flux token using a learned parameterized 2D Gaussian ellipse.
No Kljun FFP precomputation — end-to-end supervised by tower flux measurements only.

```
Inputs:
  spatial_tokens : (B, 196, 768)  transformer output at 196 spatial positions
  dem_tokens     : (B, 4, 768)    DEM static tokens (local terrain context)
  wind_speed     : (B,)  ERA5 scalar for target day T
  wind_dir       : (B,)  ERA5 scalar for target day T (degrees)

Footprint MLP:
  mean_tok = spatial_tokens.mean(dim=1)        # (B, 768) global context
  dem_mean = dem_tokens.mean(dim=1)            # (B, 768) terrain context
  concat   = [mean_tok, dem_mean, wind_speed, wind_dir]
  → MLP → 3 scalars: d_peak, σ_cross, σ_along

DEM tokens inform the MLP about local topography (hills, valleys, tree lines)
that can deflect wind from the coarse 9km ERA5 vector.

2D Gaussian on 14×14 grid:
  Centre at tower pixel (row=7, col=7)
  Rotate ellipse by ERA5 wind direction
  Normalise weights to sum=1  →  (B, 196) footprint weights

Flux token:
  flux_tok = (weights.unsqueeze(-1) * spatial_tokens).sum(dim=1)  # (B, 768)

Gaussian parameters are interpretable and publishable.
No external footprint model required at training or inference.
```

### 8b. LE Head

```
flux_tok (768-dim)
        ▼
┌─────────────────┐
│    LE Head      │
│ Linear(768→1)   │
│ Output: LE      │
│ FLUXNET only    │
└─────────────────┘
```

Single output: latent heat flux (LE, W m⁻²). NEE, H, and NETRAD are not predicted.

---

## 9. Training Strategy

### Phase 1 — SM only (ISMN)

```
Loss:
  L = λ_SM · L_SM
  L_SM = Huber( pred_sm[:, :, 112, 112][mask], target_sm[mask] )
         masked per depth bin (NaN where depth absent)

Train until convergence. Builds solid backbone before adding flux complexity.
```

### Phase 2 — Joint fine-tuning (ISMN + FLUXNET)

```
Loss:
  L = λ_SM · L_SM    [Huber, masked per depth  — ISMN stations]
    + λ_LE · L_LE    [Huber on LE              — FLUXNET stations]

  L_LE = Huber( pred_LE, target_LE )

Missing labels masked per station:
  ISMN stations    → L_SM only
  FLUXNET stations → L_LE only

Learning rates:
  Backbone + SM head : low LR (1e-5) — preserve SM performance
  LE head            : full LR (1e-4) — train from scratch

Metrics: MSE · MAE · ubRMSE per variable
```

### Causal enforcement

ERA5 days after T: `key_padding_mask=True`. S2/S1 acquisitions after T: not loaded.
Static tokens always visible. Training and inference are identical — no mismatch.

### Modality dropout

| Branch | Dropout prob |
|---|---|
| SIF | 50% |
| TWSA | 50% |
| S2 only | 20% |
| S1 only | 20% |
| S1 + S2 together | 10% |

At inference: simply omit tokens for any dropped modality — no retraining needed.

### Step 2 — Disaggregation Module (transformer frozen)

```
Input:  footprint-level flux (Step 1) + Landsat LST (160m) + S2 NIRv (160m)
Output: 160m daily flux map
Loss:   MSE(spatial pattern vs LST/NIRv) + λ_cons · conservation constraint
        (area-weighted mean of 160m predictions = footprint-level flux)
```

### Step 3 — Forecasting Extension

```
Lead times: T+1, T+3, T+7, T+14 (direct multi-step, no autoregression)
ERA5 beyond T:
  Training:  ECMWF reforecast (preferred) or noisy ERA5 + Gaussian noise ∝ lead time
  Inference: ECMWF HRES (T+1→T+10), ECMWF ENS (T+10→T+15)
  Same ERA5 MLP encoder — no architecture change needed
S1/S2 beyond T: not available → carry forward last available L3/L6/L9 skip features
```

---

## 10. Evaluation Splits

| Split | Strategy | Tests |
|---|---|---|
| OOS | Hold out ~20% stations by network/region | Spatial generalisation |
| OOT | Train early years, test later years (≥3 years) | Temporal generalisation |
| OOST | OOS stations on later period only | Both combined |

Baselines: SMAP · ESA CCI · persistence · climatology

---

## 11. Spatial Inference — Netherlands Maps

```
~41,500 km² → ~8,300 chips of 2.24 km × 2.24 km (224×224 px @ 10 m)

Per chip, target day T:
  1. TerraMind (frozen): one pass per S1/S2/DEM/LULC acquisition ≤ T
       L12 → latent CA → 4 × 768 compressed tokens
       L3/L6/L9 skip features from most-recent raw patch
  2. ERA5 MLP (Yeo-Johnson preprocessed): days 1→T visible, T+1→365 masked
  3. SIF/TWSA MLPs: sparse tokens in rolling window
  4. Soil CNN: one pass on 74×74 patch → 4 × 768 static tokens
  5. Temporal transformer → (B, 768, 14, 14) bottleneck
  6. U-Net decoder + skip connections → (B, n_depths, 224, 224) SM map

Stitch chips → continuous SM map @ 10 m for day T
Validity: OOS evaluation directly tests spatial generalisation
```

---

## 12. Key Design Decisions

| Component | Decision | Reason |
|---|---|---|
| Spatial pooling (S2/S1 history) | Latent cross-attention: 4 learnable queries × 196 L12 tokens → 4 × 768 | Content-adaptive; globally cloud-aware via key_padding_mask; replaces fixed geometric pyramid |
| Cloud masking | CloudSEN12 UNetMobV2_V2 offline → per-token (16×16 px) mask; cloudy: key_padding_mask=True | Token-level; any clear token anywhere contributes to all 4 latent queries |
| Soil encoder | OpenLandMap-soildb 30m CNN (DWConv+PWConv) + spatial pyramid → 4 × 768; 21 channels (3 depths × 7 props); NaN pre-filled via nearest-neighbour; 19 fully-NaN stations excluded | Real 74×74 spatial heterogeneity; replaces SoilGrids 250m point MLP; CNN input corrected from 7→21 ch |
| ERA5 normalization | Yeo-Johnson Power Transform, fit on training split, stored, applied at dataset load | Uniform treatment; handles 8-orders-of-magnitude scale difference and precip skewness |
| ERA5 token count | Full 365 tokens (no Perceiver compression) | 670-token sequence is manageable; compression risks losing fine-grained daily patterns |
| Modality type embedding | Unified nn.Embedding(10, 768) on every token including static | Transformer always knows what it attends to; replaces separate spatial_modality_emb |
| Target-day spatial | 196 uncompressed L12 tokens from most recent valid image ≤ T | Spatially-structured U-Net bottleneck; type emb distinguishes S2 vs S1 |
| Flux footprint | Parameterized 2D Gaussian (d_peak, σ_cross, σ_along) predicted by MLP from spatial tokens + ERA5 wind; ERA5 wind direction rotates ellipse; single LE head (768→1) | No Kljun precomputation; end-to-end supervised; LE only (NEE/H/NETRAD dropped) |
| Temporal attention | Bidirectional transformer; future ERA5 days: key_padding_mask=True | Full bidirectional attention within visible context; simpler than causal mask |
| Positional encoding | Sinusoidal DoY + learned rel-pos emb (0→364) + modality type emb | DoY: season; rel-pos: monotonic order at year boundary; type: temporal scale |
| U-Net skip connections | L3/L6/L9 from one TerraMind pass on most-recent raw patch; no days_ago emb | Transformer encodes staleness via rel-pos; decoder handles spatial upsampling only |
| SIF/TWSA | Separate MLPs; sparse token injection; modality dropout p=0.5; mid-month DoY for TWSA | Clean ablations; type emb distinguishes temporal scales |
| Training | Joint ISMN + FLUXNET; missing labels masked per station | Unified land surface model |
| Forecasting | Direct multi-step T+1/T+3/T+7/T+14; NWP replaces ERA5 after T | No error accumulation; natural causal extension; same ERA5 MLP |

---

## 13. Data Paths

| Data | Path |
|---|---|
| ISMN processed NetCDF | `/home/khanalp/data/soilmoisture/level1/` |
| ISMN station metadata | `/home/khanalp/data/soilmoisture/level1/station_metadata.csv` |
| Satellite patches + L12 features | `/home/khanalp/data/satellite/{network}_{station}/` |
| ERA5 Yeo-Johnson params | `/home/khanalp/data/soilmoisture/era5_pt_params/` |
| AmeriFlux FLUXNET | `/home/khanalp/data/ameriflux_raw/` |

---

## 14. Future Work

Items identified but deferred — implement if evaluation reveals specific weaknesses:

| Item | Trigger condition |
|---|---|
| Unfreeze TerraMind LayerNorms | Root zone or S1 backscatter features underperform |
| LoRA on TerraMind last 2–3 layers | LayerNorm unfreezing insufficient for hydrology signals |
| Infiltration latent state: cumulative ERA5 (precip − ET) as additional input feature | Deeper depth bins (20–40, 40–100 cm) consistently underperform surface |
| Cross-modality attention for TWSA/SIF | Post-training attention analysis shows TWSA/SIF weights near-zero |
| Depth-conditioned skip connection gating | Large performance gap between surface and root zone in OOS evaluation |

---

## 15. Key References

- TerraMind: [arXiv:2504.11171](https://arxiv.org/abs/2504.11171) · [GitHub](https://github.com/IBM/terramind)
- Contextformer (Benson et al., CVPR 2024): [arXiv:2303.16198](https://arxiv.org/abs/2303.16198)
- OpenLandMap-soildb: [essd.copernicus.org/articles/18/989/2026](https://essd.copernicus.org/articles/18/989/2026)
- CloudSEN12: [cloudsen12.github.io](https://cloudsen12.github.io)
- ISMN: [ismn.earth](https://ismn.earth)
- AmeriFlux: [ameriflux.lbl.gov](https://ameriflux.lbl.gov)
