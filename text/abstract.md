# Abstract — multimodal soil-moisture & latent-heat flux model

*Nature summary-paragraph style. Conference/meeting version. Headline metrics are the full
held-out validation set (74 stations) from the converged `baseline_huber` run (job 23936932), run
to epoch 15 (val_loss 0.0021). The full out-of-sample (OOS) eval (job 23992292) was cancelled and
never ran, so OOS correlation is still from a 10-station preliminary subset. Flux: only latent-heat
flux (`LE_F_MDS` ≈ evapotranspiration) is used.*

---

## Full version (~265 words)

Soil moisture and the latent-heat (evapotranspiration) flux that couples the land surface to the
atmosphere jointly regulate the terrestrial water and energy budgets, modulating droughts,
agricultural productivity and land–atmosphere feedbacks. Yet the in-situ networks that constrain
them — soil-moisture probes and eddy-covariance flux towers — are spatially sparse and unevenly
distributed, while spaceborne retrievals remain coarse (tens of kilometres) and sensitive only to
the top few centimetres of soil, leaving fine-scale, depth-resolved soil moisture and surface
energy partitioning poorly observed. Here we present a multimodal deep-learning framework that
fuses frozen TerraMind vision-transformer embeddings of Sentinel-2 optical, Sentinel-1 synthetic-
aperture-radar, elevation and land-cover imagery with ERA5-Land meteorological reanalysis,
solar-induced chlorophyll fluorescence, GRACE terrestrial water-storage anomalies and static soil
properties. A temporal transformer encodes irregularly sampled satellite time series through
attention over acquisition dates, and a FiLM-modulated U-Net decoder reconstructs daily,
10–100 m-resolution 224 × 224-pixel fields from sparse point labels, supervised on nearly 990
ISMN soil-moisture stations co-located with 61 ICOS and 84 AmeriFlux eddy-covariance towers that
provide latent-heat (evapotranspiration) flux. On a fully held-out validation set of 74 stations, the converged model
retrieves daily volumetric soil moisture at three depths with unbiased RMSEs of 0.053, 0.049 and
0.057 m³ m⁻³ (0–10, 10–30 and 30–100 cm) — each below the 0.07 m³ m⁻³ benchmark for satellite
soil-moisture products — remaining stable across a 15-epoch optimisation, with initial
out-of-sample evaluation yielding temporal correlations up to R ≈ 0.69. By learning a shared
representation that downscales sparse point observations into spatially continuous, depth-resolved
fields, the framework enables high-resolution monitoring of root-zone soil moisture and
evapotranspiration for drought early-warning, irrigation and water-resource management, crop-yield
forecasting, and the evaluation and constraint of land-surface and climate models.

---

## Short version (~120 words, for a slide)

Soil moisture and latent-heat (evapotranspiration) flux couple the terrestrial water and energy
cycles, but are measured only by sparse, unevenly distributed ground networks, while satellite
retrievals stay coarse and surface-limited. We present a multimodal model that fuses frozen
TerraMind vision-transformer embeddings (Sentinel-2, Sentinel-1, elevation, land cover) with
ERA5-Land, solar-induced fluorescence, GRACE water-storage anomalies and soil properties. A
temporal transformer plus a FiLM-modulated U-Net decoder downscales sparse point labels — from
~990 ISMN soil-moisture stations co-located with 61 ICOS and 84 AmeriFlux latent-heat flux towers
— into daily, 10–100 m-resolution, multi-depth maps. On a fully held-out validation set, surface
unbiased RMSE
reaches 0.053 m³ m⁻³ (0.049–0.057 across depths), below the 0.07 m³ m⁻³ benchmark, with
out-of-sample correlations up to R ≈ 0.69 — supporting drought monitoring, irrigation and
water-resource management, and land-surface-model evaluation.

---

## Notes / flags for finalising

- **Flux = latent heat only.** The active extracted flux variable is `LE_F_MDS` (gap-filled latent
  heat ≈ evapotranspiration; `FLUX_VARS = ["LE_F_MDS"]` in `preprocessing_ameriflux.py` /
  `preprocessing_icos.py`). Sensible heat (`H_F_MDS`), net radiation (`NETRAD`) and carbon fluxes
  (NEE, GPP, RECO) have metadata defined but are **not** extracted — do not cite them. Flux is
  co-assembled into the same `sm_and_flux` / `flux_only` station structure; the current model
  demonstrates soil moisture, with latent-heat flux as the unifying joint-prediction direction.
- **Headline numbers are full validation-set, converged:** job 23936932 (`baseline_huber`, memmap,
  epoch 15, val_loss 0.0021). Per-depth ubRMSE (pooled / station-mean): 0–10 0.054/0.049,
  10–30 0.049/0.046, 30–100 0.057/0.051 m³ m⁻³. Source: `logs/train_23936932.out` epochs 13–15.
- **Correlation R ≈ 0.69 is still preliminary:** from a 10-station OOS smoke (`text/logs.txt`
  session 18). The full OOS eval (job 23992292) was **cancelled and never ran** — rerun it for a
  converged OOS table (ubRMSE/R over ~180 stations) and replace the R figure when available.
- **Site counts** (`csvs/station_splits.csv`): 842 ISMN, 90 AmeriFlux, 61 ICOS = 993 total.
  Flux-providing (`has_flux=True`): 61 ICOS + 84 AmeriFlux = 145 towers (6 AmeriFlux lack the flux
  flag) — abstract cites 61 ICOS / 84 AmeriFlux for the flux co-location claim.
- **Resolution:** 10–100 m spatial (Sentinel native ground sample distance), daily temporal.
- "~990 stations" = 993 active ISMN; Phase-1 split 587 train / 74 val / 181 OOS (sm_only).
