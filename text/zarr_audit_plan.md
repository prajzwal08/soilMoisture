# Comprehensive Zarr Modality & Coverage Audit — Plan

**Date:** 2026-06-10
**Status:** 993/993 stations have `.complete` zarr stores. Before
`check_dataset.sh` / training, run a full per-modality + per-date audit.

## Why

`.complete` only means `create_token_zarr.py` / `retokenize_satellite_zarr.py`
ran without raising — it does not guarantee every modality is present or that
every year in a station's date range has data. Past recoveries (Groups A-D)
showed `.complete` can exist with broken/missing modalities. Two specific
checks requested:

1. For every S2L2A acquisition date, does a matching cloud-mask entry exist?
2. For each station's `[start_date, end_date]` window (from
   `csvs/station_splits.csv`), does each modality have data covering every
   year in that range (no all-zero years)?

## Schema reference (confirmed via live inspection of zarr stores)

Per-station zarr group at `ZARR_ROOT/{category}/{dir_name}/`
(`ZARR_ROOT = /gpfs/scratch1/shared/pkhanal/zarr`):

- `s2/{l12,l9,l6,l3,dates}` — always expected
- `s1_asc/{l12,l9,l6,l3,dates}`, `s1_desc/{...}` — at least one expected; both
  optional individually
- `cm/{masks,dates}` — always expected
- `dem`, `lulc`, `soil` — static, always expected
- `era5/{values,date_ints,doys}` — always expected
- `sif/{values,date_ints,doys}` — optional (informational only)
- `twsa/{lwe,lwe_uncertainty,date_ints,doys}` — optional (informational only)
- `labels/sm` (+`qc`,`depths`,`dates`) — soil moisture labels
- `labels/le` (+`le_qc`,`dates_flux`) — flux (latent heat) labels

`dir_name`/`category` derivation matches `dataset.py` /
`retokenize_satellite_zarr.py`: ISMN → `ISMN_{network}_{station_name}`,
others → `{source_network}_{station_id}`; category = sm_and_flux / sm_only /
flux_only based on `has_soil_moisture` / `has_flux`.

### Per-category label requirements

| category      | `labels/sm` required | `labels/le` required |
|----------------|:---:|:---:|
| `sm_only`      | ✅  | —   |
| `sm_and_flux`  | ✅  | ✅  |
| `flux_only`    | —   | ✅  |

## New script: `audit_zarr_complete.py`

Modeled after `audit_pretrain.py` (multiprocessing.Pool over
`station_splits.csv` rows), but reading from zarr (`zarr.open_consolidated`)
instead of `.pt`/`.nc`.

### Per-station checks (one row of output per station)

1. **Existence flags**: `has_s2`, `has_s1_asc`, `has_s1_desc`, `has_cm`,
   `has_dem`, `has_lulc`, `has_soil`, `has_era5`, `has_sif`, `has_twsa`,
   `has_labels_sm`, `has_labels_le`. For s2/s1/cm verify ALL 4 layers
   (`l12,l9,l6,l3`) + `dates` are present, not just the group.
   `has_labels_sm`/`has_labels_le` are checked against the per-category
   requirement table above — e.g. a `sm_and_flux` station missing either
   `labels/sm` or `labels/le` is CRITICAL.

2. **Cloud-mask ↔ S2 alignment**: `n_s2_dates`, `n_cm_dates`,
   `n_s2_missing_cm` (S2 dates with no matching cm date), `cm_coverage_pct`.
   Flag `WARN` if `cm_coverage_pct < 95%`.

3. **Per-year coverage within `[start_date, end_date]`**: for each modality
   with a date array (`s2`, `s1_asc`, `s1_desc`, `cm`, `era5`, `labels_sm`,
   `labels_le`, `sif`, `twsa` — only the label variants required for that
   station's category are checked), compute the set of years present and list
   any year in `range(start_year, end_year+1)` with zero entries →
   `{modality}_missing_years` column (comma-separated, empty if none).
   SIF/TWSA gaps recorded but not counted toward WARN/CRITICAL status
   (informational).

4. **Status rollup**:
   - `CRITICAL`: zarr group fails to open / `.complete` missing / any required
     modality (s2, cm, dem, lulc, soil, era5, and the category-required labels
     above) entirely missing, or s2 AND s1 both entirely missing.
   - `WARN`: cm_coverage_pct < 95%, OR any required modality has a
     missing-year gap within the station's date range.
   - `OK`: everything else.
   - `flags`: semicolon-joined human-readable list of all triggered issues.

### Outputs

- `csvs/audit_zarr_complete.csv` — full per-station detail (993 rows)
- `text/audit_zarr_complete_summary.txt` — aggregate report:
  - counts by status (OK/WARN/CRITICAL)
  - per-modality presence counts (# stations missing each modality)
  - CM coverage distribution (mean/median/min, # stations < 95%)
  - # stations with missing-year gaps per modality, with station list for
    CRITICAL ones
  - SIF/TWSA presence stats (informational section)

## Execution

- New SLURM script `slurm/audit_zarr_complete.sh`:
  - `--partition=rome`, `--cpus-per-task=64`, `--mem=128G`, `--time=1:00:00`
    (CPU-only, metadata + small-array reads — matches `group_a_zarr.sh`
    pattern)
  - `--mail-type=BEGIN,END,FAIL --mail-user=ktm.prajwalkhanal@gmail.com`
    (required)
  - `conda run --no-capture-output -n sensei python audit_zarr_complete.py --workers 64`
- Submit via `sbatch slurm/audit_zarr_complete.sh`, monitor log in `logs/`.

## Verification

- Run the SLURM job, confirm it completes with 993 rows in
  `csvs/audit_zarr_complete.csv` and exit code 0.
- Read `text/audit_zarr_complete_summary.txt` and report status breakdown.
- If CRITICAL/WARN stations are found, surface the list so follow-up fixes can
  be planned before `check_dataset.sh`/training.
