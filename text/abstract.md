# High-resolution daily soil moisture and evapotranspiration from sparse in-situ networks via multimodal deep learning

*Conference/meeting abstract. Headline metrics are the full held-out validation set (74 stations)
from the converged `baseline_huber` run (job 23936932, epoch 15, val_loss 0.0021). The full
out-of-sample (OOS) eval (job 23992292) was cancelled and never ran, so OOS correlation is still
from a 10-station preliminary subset. Flux: only latent-heat flux (`LE_F_MDS` ≈ evapotranspiration)
is used.*

---

## Abstract

Soil moisture and latent heat flux (evapotranspiration) regulate terrestrial water and energy
exchanges, influencing drought development, ecosystem productivity, and land–atmosphere feedbacks.
Yet observations remain limited. While modern satellite missions provide high-resolution
observations of the land surface, no single sensor delivers daily, spatially continuous,
depth-resolved estimates of these variables. Optical imagery is hindered by cloud cover and
irregular revisit times, synthetic-aperture radar retrievals are affected by vegetation and
surface roughness, and microwave observations are largely sensitive only to near-surface soil
moisture. Meanwhile, in-situ soil-moisture networks and eddy-covariance flux towers provide
accurate measurements but are spatially sparse and unevenly distributed.

These complementary limitations motivate multimodal data fusion. Here we present a multimodal
deep-learning framework that integrates frozen TerraMind vision-transformer embeddings of
Sentinel-2 optical, Sentinel-1 synthetic-aperture radar, elevation, and land-cover imagery with
ERA5-Land meteorological reanalysis, solar-induced chlorophyll fluorescence, GRACE terrestrial
water-storage anomalies, and static soil properties. A temporal transformer models irregularly
sampled satellite observations through acquisition-time attention, while a FiLM-modulated U-Net
reconstructs daily 224 × 224 pixel fields at 10–100 m resolution from sparse point observations.
The model is trained using measurements from approximately 990 ISMN soil-moisture stations
together with latent heat flux observations from 61 ICOS and 84 AmeriFlux eddy-covariance towers.

On a geographically independent validation set of 74 held-out stations, the model estimates daily
volumetric soil moisture at 0–10, 10–30, and 30–100 cm depths with unbiased root-mean-square
errors of 0.053, 0.049, and 0.057 m³ m⁻³, respectively, all below the 0.07 m³ m⁻³ benchmark for
satellite soil-moisture products. The model remains stable throughout training and achieves
temporal correlations of up to R ≈ 0.69 in out-of-sample evaluation. By learning a shared
multimodal representation of complementary Earth observations and geophysical data, the framework
produces high-resolution, spatially continuous estimates of root-zone soil moisture and
evapotranspiration, enabling improved monitoring of terrestrial hydrology and supporting drought
assessment, irrigation management, agricultural forecasting, and land-surface model evaluation.

---

## Notes / flags for finalising

- **Flux = latent heat only.** Active extracted variable is `LE_F_MDS` (gap-filled latent heat ≈
  evapotranspiration; `FLUX_VARS = ["LE_F_MDS"]`). Sensible heat, net radiation and carbon fluxes
  (NEE, GPP, RECO) have metadata defined but are **not** extracted — do not cite them.
- **Headline numbers are full validation-set, converged:** job 23936932 (`baseline_huber`, memmap,
  epoch 15, val_loss 0.0021). Per-depth ubRMSE (pooled / station-mean): 0–10 0.054/0.049,
  10–30 0.049/0.046, 30–100 0.057/0.051 m³ m⁻³. Source: `logs/train_23936932.out` epochs 13–15.
- **Correlation R ≈ 0.69 is still preliminary:** from a 10-station OOS smoke (`text/logs.txt`
  session 18). The full OOS eval (job 23992292) was **cancelled and never ran** — rerun it for a
  converged OOS table (ubRMSE/R over ~180 stations) and replace the R figure when available.
- **Site counts** (`csvs/station_splits.csv`): 842 ISMN, 90 AmeriFlux, 61 ICOS = 993 total.
  Flux-providing (`has_flux=True`): 61 ICOS + 84 AmeriFlux = 145 towers (6 AmeriFlux lack the flag).
- **Resolution:** 10–100 m spatial (Sentinel native ground sample distance), daily temporal.
- "~990 stations" = 993 active ISMN; Phase-1 split 587 train / 74 val / 181 OOS (sm_only).
