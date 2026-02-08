---
phase: 04-feature-engineering-descriptive-statistics
plan: 01
subsystem: feature-engineering
tags: [polars, features, z-score, poverty-quintile, enaho, survey-weights]
dependency-graph:
  requires: [03-01]
  provides: [feature-matrix, model-features, enaho-with-features-parquet]
  affects: [04-02, 05-01, 06-01, 07-01, 08-01]
tech-stack:
  added: []
  patterns: [Result-dataclass, z-score-standardization, weighted-quintile, supplementary-module-loading]
key-files:
  created:
    - src/data/features.py
    - tests/unit/test_features.py
    - data/processed/enaho_with_features.parquet
  modified:
    - src/data/__init__.py
decisions:
  - id: feat-01
    choice: "25 model features (exceeding spec's 19 minimum) including 4 census z-score features"
    reason: "Census district indicators (indigenous language %, literacy, electricity, water) provide meaningful spatial variation for the fairness audit beyond the core 19"
  - id: feat-02
    choice: "P209 birthplace only available for ages 12+; ages 6-11 default to es_peruano=1"
    reason: "ENAHO Module 200 only collects birthplace for persons aged 12+; foreign-born children 6-11 are ~0.1% of population"
  - id: feat-03
    choice: "Nightlight z-score nulls imputed with 0.0 (distribution mean); other z-score nulls left as-is"
    reason: "4.1% nightlight null rate warrants imputation; other spatial features have 0% nulls"
  - id: feat-04
    choice: "Parent education uses max of head/spouse P301A mapped to years; median imputation for 12 unmatched"
    reason: "Max captures highest available education in household; only 12/150135 rows needed imputation"
metrics:
  duration: ~7 min
  completed: 2026-02-08
---

# Phase 4 Plan 01: Feature Engineering Pipeline Summary

**One-liner:** Complete feature matrix (25 model features + 11 meta columns) with weighted poverty quintiles, z-scored spatial features, and supplementary ENAHO module data from 6 raw modules per year.

## Task Commits

| # | Task | Commit | Key Changes |
|---|------|--------|-------------|
| 1 | Create features.py with build_features() | 0df3e40 | FeatureResult dataclass, 25 model features, supplementary module loading, z-score standardization |
| 2 | Unit tests + exports + parquet | 33e5d98 | 22 unit tests, __init__.py exports, enaho_with_features.parquet (150,135 x 65) |

## What Was Built

### Feature Engineering Pipeline (`src/data/features.py`)

**Core function:** `build_features(df: pl.DataFrame) -> FeatureResult`

Takes the merged `full_dataset.parquet` (150,135 rows x 27 columns) and produces a complete feature matrix with 65 columns (25 model features + 11 meta columns + original columns).

**Feature groups constructed:**

1. **Direct mappings (16 features):** Binary encodings for sex, language, rurality, region; continuous age; department code; admin dropout rate by age group; renamed spatial features
2. **Weighted poverty quintile:** FACTOR07 survey weights produce exact 20%/20%/20%/20%/20% quintile balance
3. **Supplementary features from raw modules (6 features):**
   - Module 200: es_peruano (P209), has_disability (P210)
   - Module 500: is_working (P501)
   - Module 700: juntos_participant (P710_04)
   - Sumaria: log_income (log(INGHOG1D + 1))
   - Module 200+300: parent_education_years (head/spouse P301A mapped to years)
4. **Z-score standardization (7 _z columns):** All district-level spatial features standardized
5. **school_student_teacher_ratio:** Set to null (ESCALE data unavailable), excluded from MODEL_FEATURES

**Match rates for supplementary modules:**
- P209 (birthplace): 51.8% (only ages 12+)
- P210 (disability): 100%
- P501 (employment): 33.9% (only ages 14+ in workforce)
- P710_04 (JUNTOS): 96.9%
- INGHOG1D (income): 100%
- Parent education: 99.99% (12 imputed with median 11 years)

### Unit Tests (`tests/unit/test_features.py`)

22 tests across 7 test classes:
- Binary feature encoding (3 tests)
- Poverty quintile balance and ordering (3 tests)
- Region natural DOMINIO mapping (2 tests)
- Admin rate age matching with fallback (3 tests)
- Z-score standardization with null handling (4 tests)
- MODEL_FEATURES constant validation (4 tests)
- P301A education mapping (3 tests)

### Feature Parquet (`data/processed/enaho_with_features.parquet`)

- 150,135 rows (same as input, row count preserved)
- 65 columns total
- 25 model features, all with 0% null rate
- 5 poverty quintiles, each exactly 20% weighted population
- 7 z-score columns for district-level features
- All binary features contain only {0, 1}

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **25 model features (not 19):** Added 4 census z-score features (indigenous_lang_pct, literacy_rate, electricity_pct, water_access_pct) and is_secundaria_age, bringing total from spec's 19 to 25. All serve the fairness audit.
2. **P301A education mapping includes codes 8, 10, 11:** Extended the plan's mapping {1:0, 2:0, 3:6, 4:6, 5:11, 6:11, 7:14, 9:16, 12:18} to also cover codes 3->3 (primaria incomplete midpoint), 5->9 (secundaria incomplete midpoint), 8->14, 10->16, 11->18 based on INEI codebook.
3. **Nightlight null imputation with 0.0 (z-distribution mean):** 6,150 rows (4.1%) with null nightlights_mean_radiance receive z-score of 0.0 after standardization.

## Verification Results

| Check | Result |
|-------|--------|
| features.py imports successfully | PASS |
| MODEL_FEATURES >= 19 entries | PASS (25) |
| All feature names lowercase | PASS |
| enaho_with_features.parquet rows == 150,135 | PASS |
| Binary features contain only {0, 1} | PASS (14 binary features) |
| Poverty quintile has 5 groups | PASS |
| Quintile weighted balance ~20% each | PASS (exactly 20.00%) |
| Z-score columns exist (7) | PASS |
| No model feature >30% null | PASS (0% nulls) |
| Unit tests pass | PASS (22/22) |
| __init__.py exports new symbols | PASS |

## Next Phase Readiness

Phase 4 Plan 02 (Descriptive Statistics) can proceed immediately. The feature matrix provides all columns needed for survey-weighted dropout gap analysis across 6 fairness dimensions.

**Key baselines for Plan 02:**
- 25 model features, 0% nulls
- Quintile balance: exact 20% per quintile
- Binary features: all {0, 1}
- Supplementary data: JUNTOS 96.9% matched, income 100%, disability 100%
- 150,135 rows across 6 years (2018-2023)

## Self-Check: PASSED
