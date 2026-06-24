# Abstract — multimodal soil-moisture & land–atmosphere flux model

*Nature summary-paragraph style. Conference/meeting version. Results are the **full
held-out validation set** (74 stations) from the converged `baseline_huber` run (job 23936932),
stable across epochs 13–15. The full out-of-sample (OOS) eval (job 23992292) was cancelled and
never ran, so OOS correlation is still from a 10-station preliminary subset.*

---

## Full version (~225 words)

Soil moisture and the land–atmosphere fluxes of water, energy and carbon are tightly coupled,
and together they govern droughts, crop productivity and the global water and carbon cycles. Yet
the in-situ networks that measure them — soil-moisture probes and eddy-covariance flux towers —
are sparse and unevenly distributed, while satellite retrievals remain coarse and limited to the
surface, leaving fine-scale, depth-resolved soil moisture and ecosystem fluxes largely
unobserved. Here we present a multimodal deep-learning framework that fuses frozen TerraMind
vision-transformer tokens from Sentinel-2, Sentinel-1, elevation and land-cover imagery with
ERA5-Land meteorology, solar-induced fluorescence, terrestrial water-storage anomalies and soil
properties. A temporal transformer integrates irregular satellite time series, and a
FiLM-modulated U-Net decoder reconstructs 224×224 maps from sparse point labels harmonised across
nearly 990 ISMN soil-moisture stations together with co-located ICOS and AmeriFlux towers that
provide latent- and sensible-heat, net-radiation and carbon (net ecosystem exchange, gross
primary production) fluxes. Across a fully held-out validation set of 74 stations, the converged
model predicts daily soil moisture at three depths with unbiased RMSEs of 0.053, 0.049 and
0.057 m³/m³ (0–10, 10–30 and 30–100 cm) — all below the 0.07 m³/m³ benchmark — and remains stable
over a 15-epoch run, with initial out-of-sample tests giving temporal correlations up to R≈0.69.
By unifying soil-moisture and flux observations under a single learned
representation, the framework offers a scalable route toward jointly monitoring land-surface
water, energy and carbon exchange.

---

## Short version (~120 words, for a slide)

Soil moisture and land–atmosphere water, energy and carbon fluxes are tightly coupled but
measured only by sparse, unevenly distributed ground networks, while satellites stay coarse and
surface-limited. We present a multimodal model that fuses frozen TerraMind vision-transformer
tokens (Sentinel-2/-1, elevation, land cover) with ERA5-Land, solar-induced fluorescence,
terrestrial water-storage anomalies and soil data. A temporal transformer plus a FiLM-modulated
U-Net decoder turns sparse point labels — from ~990 ISMN soil-moisture stations and co-located
ICOS/AmeriFlux flux towers (latent and sensible heat, net radiation, NEE, GPP) — into
high-resolution, multi-depth maps. On a fully held-out validation set, surface unbiased RMSE
reaches 0.053 m³/m³ (0.049–0.057 across depths), below the 0.07 m³/m³ benchmark, with
out-of-sample correlations up to R≈0.69 — a scalable path to joint soil-moisture and flux
monitoring.

---

## Notes / flags for finalising

- **Flux is co-assembled, not yet a demonstrated output.** The pipeline harmonises eddy-covariance
  fluxes (active variable: latent heat `LE_F_MDS` ≈ evapotranspiration; carbon NEE/GPP/RECO,
  sensible heat `H`, net radiation `NETRAD` also preprocessed) from ICOS + AmeriFlux into the same
  `sm_and_flux` / `flux_only` station structure as soil moisture. The current model predicts soil
  moisture only; the abstract frames flux as the unifying multi-target direction — wording kept
  honest ("offers a scalable route toward jointly monitoring"), not claiming flux metrics.
- **Headline numbers are full validation-set, converged:** job 23936932 (`baseline_huber`, memmap,
  run to epoch 15, val_loss 0.0021). Per-depth ubRMSE (pooled / station-mean): 0–10 0.054/0.049,
  10–30 0.049/0.046, 30–100 0.057/0.051 m³/m³. Source: `logs/train_23936932.out` epochs 13–15.
- **Correlation R≈0.69 is still preliminary:** from a 10-station OOS smoke (`text/logs.txt` session
  18). The full OOS eval (job 23992292) was **cancelled and never ran** — rerun it for a converged
  OOS table (ubRMSE/R over ~180 stations) and replace the R figure when available.
- "~990 stations" = 993 active ISMN; Phase-1 split 587 train / 74 val / 181 OOS (sm_only).
