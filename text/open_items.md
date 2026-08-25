# Open items — things to fix before §33 is built

Collected 2026-08-25. Nothing here is done. Each entry says **where**, **what**, and **why it
bites**. Ordered by whether it can silently corrupt a result.

---

## A. Code bugs — these corrupt results silently

**A1. All-zero anchor is indistinguishable from a real S2 anchor.**
`dataset.py:483-486` returns an all-zero anchor with `orbit_id = 0` when no acquisition is
available. `model.py:498,508` reads `orbit_id = 0` as "S2" and applies the S2
`spatial_modality_emb`. So a *missing* anchor is told it is a clear Sentinel-2 image.
`bottleneck` is built entirely from these 196 anchor tokens, so this poisons the one
spatially-resolved object in the sequence. Logged in §32, still open.
**Fix:** a sentinel modality id (or a separate `anchor_valid` flag) for the absent case.

**A2. `csvs/era5_stats.json` may be contaminated.**
If the normalisation stats were computed over all 993 stations rather than the train split,
val/test statistics are already in the baseline — and every §33 number would inherit it.
**Fix:** audit how it was built; recompute train-split-only if needed. §33.12(g)(1).

**A3. The station pixel has no centre cell.**
`masked_huber_loss` (`model.py:751`) hardcodes `(112,112)` in a 224 grid. 224 is even, so the
station falls *between* 111 and 112 — a 5 m offset at 10 m, 10 m at 20 m if the SM head moves
to `up3`. Same problem at 112×112 (between 55 and 56).
**Fix:** choose deliberately — one cell, or average the two — and take a per-sample cell index
rather than a hardcoded constant. §33.11 already flags the signature change.

---

## B. §33 S1 spec gaps — must be resolved before `build_s1_interaction.py`

**B1. Notation clash between §33.4 and §33.6/§33.12.** THE ONE MOST LIKELY TO CAUSE A BUG.
- §33.4: `p` = pixel, `k` = pass, `r_p` = permanent per-pixel term, `c_k` = per-pass term.
- §33.6/§33.12: `c_VV` = the permanent ℓ×ℓ **map**, `w` = the per-pass **scalar**.

The roles of `c` and `r` are **swapped** between the two conventions. `s1processing.md` §9.1
uses `p` for passes and `k` for cells, a third variant.
**Fix:** pick one convention, rewrite both documents, before any code is written.

**B2. `MAX_AGE` has no value anywhere.** `valid` is defined entirely by it
(`s1processing.md` §9.9) and no document assigns a number.
**Fix:** measure it — §33.13 specifies the constructed-lag design as gate 1's second axis.

**B3. `age` scaling unspecified.** Raw days would range 0–30+ sitting in the same tensor as
standardised dB. Needs a divisor or a clamp.

**B4. `age` is undefined when `valid = 0`.** §33.12's channel table says "0 ⇒ the four above
are 0", which reads literally as `d_CR, w_VV, w_CR, age` but plainly intends
`d_VV, d_CR, w_VV, w_CR`. Either way `age` needs an explicit invalid-case value — an undefined
or huge age paired with `valid = 0` will quietly undo the zero-init.

**B5. No tile-level valid-fraction floor on `p*`.** §9.10's floor is **per cell**; nothing
rejects a pass that is, say, 80% masked. Such a pass is still selected as `p*` and still gets
`valid = 1` while carrying almost no information.
**Fix:** a tile-level floor. §33.13 notes stale and mostly-masked are the same kind of
degradation and should be scored on the same curve.

**B6. Group selection when several groups have candidate passes.** A station has ~4 groups
(ASC/DESC × relative orbit). "Most recent across all groups" and "a fixed preferred group" are
different features; the second is more stable because `c` differs per group. Undecided.

**B7. `c_k` temporal look-ahead — OPEN, blocks any published number.** §33.12(h). Not label
leakage, but the input at `t` depends on passes after `t`. The **chronological split-half**
(§33.12(g)(5)) settles it with a number: compare `ρ_chrono` against the random split-half
ceiling of 0.998.

---

## C. Thermal target — Landsat arm

**C1. Gate 6 unrun.** Lag-1 ACF of `d_LST` between consecutive passes vs a location-shuffled
control. Landsat ST retrieval noise is ~1–2 K and `d_LST` may not exceed it. Can kill the arm.

**C2. Static-map sufficiency gate not yet a gate.** Δ`r_cell` vs Δ(mean SM) across
`csvs/colocated_pairs.csv`, reusing §32.10's machinery — the same test that killed terrain.
Decides whether the static thermal map is worth supervising on at all. Should become §33.9
gate 7.

**C3. No `n` floor for WRS-2 path/row groups.** S1 has gate 5 (n ≥ 20 floor, n ≥ 50
comfortable, with a measured RMSE curve). The LST side has no equivalent, and clear-scene
count is set by **climate, not effort** — cloudy stations may never reach a usable `n`.
**And those are the wet ones**, so thermal supervision is systematically thinnest where soil
moisture is highest. Needs a per-station clear-scene coverage map **before** the pull is
trusted.

**C4. Landsat 9 joined in 2021.** Same pre/post temporal-balance problem as `c_k`
(§33.12(g)(6)). Also means the "~6% of samples carry an LST gradient" figure is really ~3%
before 2022 and ~12% after, so the LST term's effective weight drifts with era.

**C5. The retrieval mask is spatially biased, not random.** §29 found 18.2% of the tile had no
ST retrieval and **that region contained the wettest station**. The double-centring therefore
runs over a biased subset of cells.

**C6. ±1 day pairing tolerance is assumed, never validated.** §33.11 pairs a clear scene to a
label date within ±1 day. Thermal state moves fast.

**C7. The global pull does not exist.** `download_landsat_st_mpc.py` is single-tile
(`--tile ISMN_TxSON_CR200-18`, `csvs/landsat_st_download_log.csv` has 494 rows for that one
station). §33.11 lists `download_landsat_st_global.py` as new; it is really a station loop /
array job over the existing script.

---

## D. Thermal target — ECOSTRESS / DTR arm (§29 Phase B, untouched)

Landsat C2 L2 ST is **daytime only** and can never give a diurnal range (§29.2). ECOSTRESS
gives day *and* night, unlocking DTR — the thermal-inertia proxy, and per §29.2 "arguably the
strongest LST-based soil-moisture signal". §29.13's negative (within-station r = −0.077) is
scoped to **daytime instantaneous** LST and does **not** refute thermal inertia. §30.6 and §32
both record Phase B as the only arm that could still recover a direct LST↔SM link.

**D1. The CMR census is unrun** — §29.3 ran the Landsat half live on 2026-08-13; the ECOSTRESS
half is still the first thing. Metadata only, no credentials, no downloads, minutes.
**Ask it for paired clear day/night overpasses per station per year**, not granule counts. ISS
precession means day and night passes are usually *not* on the same date, so pairing — not
granule count — is the binding constraint. This is the go/no-go for DTR.

**D2. `download_ecostress_lste.py` was specified in §29.6 and never built.** LP DAAC via
`earthaccess`, needs Earthdata credentials. §29 costed it at 6–12 h — the long pole.

**D3. The grouping recipe does not transfer.** Landsat groups by WRS-2 path/row because
different paths image at different local times. ECOSTRESS has no fixed path — the ISS
precesses, so overpass time drifts continuously and viewing geometry varies. Grouping must be
by **local solar time bin** (e.g. day 10:00–14:00, night 00:00–04:00), and that binning is
undecided.

**D4. `--night-assign same` vs `prev` undecided** (§29.10). A 02:00 overpass arguably reflects
the previous day's drydown.

**D5. ECOSTRESS starts 2018-07.** Zero coverage of 2016–2017, ~63% of the TxSON label window.
Nothing before mid-2018 gets thermal supervision from this arm.

**D6. Target grid.** With the SM head at `up3` (112×112 @ 20 m):

```
avg_pool2d(7,7) -> 16x16 @ EXACTLY 140 m     112 = 7 x 16, no crop
ECOSTRESS 70 m  -> 2x2 average -> 140 m      exact
```

Both sides partition exactly and the 2×2 target aggregation halves retrieval noise — which is
what gate 6 is worried about. 70 m is a *grid*, not a resolution: the native footprint is
~38 × 69 m at nadir and grows substantially off-nadir, so 140 m declines to claim resolution
the instrument does not have.

**D7. Two-channel head recovers supervision density.** Output **day and night as separate
channels** rather than one DTR channel. Supervise day where a day pass exists, night where a
night pass exists, and add the DTR constraint only when both do — instead of discarding every
unpaired pass. Costs one extra output channel.

**D8. Scale mismatch is real and unfixable at the sensor.** A 140 m cell is ~2 ha; in
fragmented landscapes it straddles two or three management units and their thermal signatures
average out. Mixing **only attenuates, never manufactures** contrast, so a gate 6 *pass* at
140 m is strong while a *fail* is ambiguous between "no signal" and "wrong scale".
**Stratify the gate 6 result by parcel-size regime before believing a negative.**

---

## E. Decisions taken 2026-08-25, recorded so they are not relitigated

- **S2 is not to be used.** User decision.
- **Therefore `up4` has no measured input.** Without S2's 10 m bands, the only 10 m sources are
  static WorldCover LULC and S1 gridded at 10 m — the latter one look per cell (ENL ≈ 4.4,
  ~2.7 dB speckle), far below the `d` signal. This independently re-derives §33.6's reason for
  dropping `up4` ("the only stage with no measured input"). TerraMind cannot fill it: all four
  skips are 14×14, so `skip_L3` at 224 is a 16× interpolation carrying nothing new.
  → SM head at `up3`, 112×112 @ 20 m; thermal target at 140 m per D6.
- **FiLM stays for now.** `context` (`model.py:743`) reaches the decoder *only* through
  `film_s9/s6/s3` (`model.py:253-255`), so deleting FiLM would sever the direct path for the
  non-spatial branch. It is a shortcut rather than the sole path — the 196 spatial tokens
  already attended to those same tokens — so "FiLM vs `w`" is a legitimate ablation, deferred
  until §33 has a verdict. Changing it now would make the "zero the seven channels" ablation
  uninterpretable.

---

## F. Deferred architecture ideas — after §33 has a verdict, not before

**F1. Learned CLS pooling instead of the masked mean.** `context` (`model.py:736-743`) is a
flat mean over the non-spatial tokens, so a token from 60 days ago weighs the same as
yesterday's — a drydown and a wetting sequence with equal averages give identical `context`.
Appending a learned CLS token and reading its output lets attention choose the weighting. Two
lines, one extra token. Moves the baseline, so not now.

**F2. Cross-attention from decoder positions to the temporal tokens.** Would let each location
integrate the same weather differently — the *response × wetness* behaviour §33.6 currently
hopes a conv approximates from a broadcast `w`. **Only works because of §33**: without `d` at
that cell the query is an upsampled 160 m token and every position asks the same question.
Costs 12,544 queries at 112×112 per stage. Pointless if gate 1 fails.

**F3. Skip-connection ablation** (already deferred): raw-image CNN skips vs TerraMind
L3/L6/L9. Related to F1/F2 — all three are "what should the decoder actually receive".

**F4. Per-location processor (§30/§31, both unbuilt).** The structural gap they were designed
for is still real: `model.py:538` shows DEM, LULC, soil and the entire satellite history are
`_cpu_pyramid_pool`'d to 4 tokens each, so **there is no per-patch time series anywhere** — a
patch has appearance from exactly one anchor date and knows only the *tile's* history, not its
own. §33 attacks the same gap from the decoder side instead.
