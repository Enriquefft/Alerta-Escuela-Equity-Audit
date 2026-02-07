# Roadmap: Alerta Escuela Equity Audit

## Overview

This roadmap delivers the first independent equity audit of Peru's Alerta Escuela dropout prediction system. The pipeline progresses from environment setup through ENAHO data ingestion, multi-year harmonization, spatial enrichment, feature engineering, model training (LogReg/LightGBM/XGBoost), fairness analysis across 6 protected dimensions with 3 intersections, SHAP interpretability, admin data cross-validation, and finally distillation of 5-7 bilingual media-ready findings exported as JSON for the M4 scrollytelling site. The spec (specs.md) is the single source of truth for all implementation details.

## Phases

**Phase Numbering:**
- Integer phases (0-11): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 0: Environment Setup** - Nix flake, pyproject.toml, download script, directory structure, .gitignore
- [ ] **Phase 1: ENAHO Single-Year Loader** - Load 2023 ENAHO with delimiter detection, dropout target, UBIGEO padding
- [ ] **Phase 2: Multi-Year Loader + Harmonization** - Load 7 years (2018-2024), P300A mother tongue harmonization
- [ ] **Phase 3: Spatial + Supplementary Data Merges** - Admin dropout rates, Census 2017, VIIRS nightlights, LEFT JOIN on UBIGEO
- [ ] **Phase 4: Feature Engineering + Descriptive Statistics** - 19+ model features, survey-weighted descriptive gaps, first export
- [ ] **Phase 5: Baseline Model + Temporal Splits** - Temporal split discipline, logistic regression baseline, evaluation patterns
- [ ] **Phase 6: LightGBM + XGBoost** - Optuna-tuned LightGBM, XGBoost comparison, algorithm-independence check
- [ ] **Phase 7: Calibration + ONNX Export + Final Test** - Calibrate best model, ONNX export, test set (2024) touched once
- [ ] **Phase 8: Subgroup Fairness Metrics** - fairlearn MetricFrame across 6 dimensions + 3 intersections
- [ ] **Phase 9: SHAP Interpretability Analysis** - Global/regional/interaction SHAP, ES_PERUANO + ES_MUJER quantification
- [ ] **Phase 10: Cross-Validation with Admin Data** - District-level prediction vs admin rates, spatial error patterns
- [ ] **Phase 11: Findings Distillation + Final Exports** - 5-7 bilingual findings, all 7 JSON exports, export README

## Phase Details

### Phase 0: Environment Setup
**Goal**: Developer can clone the repo, enter nix develop, and have a fully working Python 3.12 environment with all ML dependencies resolved
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05
**Success Criteria** (what must be TRUE):
  1. Running `nix develop` provides a shell with Python 3.12, uv, and system dependencies (OpenMP, cmake)
  2. Running `uv sync` installs all Python packages (polars, lightgbm, xgboost, fairlearn, shap, optuna, onnxmltools) without errors
  3. The directory structure matches spec Section 3 exactly (src/data/, src/models/, src/fairness/, tests/gates/, tests/unit/, data/raw/, data/processed/, data/exports/)
  4. Running `python src/data/download.py` fetches ENAHO modules and admin dropout CSVs into data/raw/
  5. data/raw/ and data/processed/ are gitignored; data/exports/ is tracked
**Plans**: 2 plans

Plans:
- [x] 00-01-PLAN.md -- Nix flake + .envrc + pyproject.toml + uv environment (Wave 1)
- [x] 00-02-PLAN.md -- Directory scaffolding + download script relocation + .gitignore + notebooks + justfile (Wave 1)

### Phase 1: ENAHO Single-Year Loader
**Goal**: A single year of ENAHO survey data loads correctly with proper delimiter detection, UBIGEO padding, dropout target construction, and school-age filtering
**Depends on**: Phase 0
**Requirements**: DATA-01, DATA-02, TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. `load_enaho_year(2023)` returns a polars DataFrame with ~25,000 school-age rows (ages 6-17), ~3,500 unweighted dropouts, and ~14% FACTOR07-weighted dropout rate
  2. All UBIGEO values are exactly 6 characters long with no leading-zero loss
  3. Gate test 1.1 passes all assertions (row count, dropout count, weighted rate, null checks, UBIGEO length)
  4. 10 random dropout rows print for human inspection and look like real student records
**Human Gate**: Yes -- review 10 printed dropout rows before proceeding
**Plans**: TBD

Plans:
- [ ] 01-01: ENAHO single-year loader + utils + gate test 1.1 + unit tests

### Phase 2: Multi-Year Loader + Harmonization
**Goal**: All 7 years of ENAHO data (2018-2024) stack into one consistent dataset with P300A mother tongue harmonization preserving both cross-year and disaggregated codes
**Depends on**: Phase 1
**Requirements**: DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. `load_all_years()` returns a pooled polars DataFrame with ~150K-180K rows, ~20K+ unweighted dropouts, and a `year` column spanning 2018-2024
  2. `p300a_harmonized` column collapses codes 10-15 back to code 3; `p300a_original` preserves disaggregated codes for 2020+ analysis
  3. Sum of codes 3+10+11+12+13+14+15 is stable across years (within 30% of each other) -- confirming harmonization not masking real population shifts
  4. Gate test 1.2 passes all assertions (year coverage, column consistency, harmonization stability, pooled counts)
**Human Gate**: No
**Plans**: TBD

Plans:
- [ ] 02-01: Multi-year loader + P300A harmonization + gate test 1.2

### Phase 3: Spatial + Supplementary Data Merges
**Goal**: ENAHO microdata is enriched with district-level admin dropout rates, Census 2017 indicators, and VIIRS nightlights via LEFT JOIN on UBIGEO without losing or duplicating any ENAHO rows
**Depends on**: Phase 2
**Requirements**: DATA-05, DATA-06, DATA-07, DATA-08
**Success Criteria** (what must be TRUE):
  1. Admin dropout rates load with ~1,890 districts for primaria (~0.93% mean) and ~1,846 for secundaria (~2.05% mean), all UBIGEO zero-padded
  2. `full_dataset.parquet` has the same row count as the input ENAHO DataFrame (no rows gained or lost from merges)
  3. ENAHO-to-admin merge rate exceeds 85%; Census merge rate exceeds 90%; nightlights coverage exceeds 85%
  4. Gate tests 1.3 and 1.4 pass all assertions (UBIGEO integrity, merge rates, no duplicates, null column report)
  5. Spot-check: Lima districts show low dropout rates, Amazonas districts show high -- directionally correct
**Human Gate**: Yes -- review spot-checked districts and columns with >10% nulls
**Plans**: TBD

Plans:
- [ ] 03-01: Admin, Census, nightlights loaders + spatial merge + gate tests 1.3/1.4

### Phase 4: Feature Engineering + Descriptive Statistics
**Goal**: All 19+ model features are engineered per spec Section 5 and survey-weighted descriptive statistics quantify dropout gaps across all 6 fairness dimensions, producing the first export JSON
**Depends on**: Phase 3
**Requirements**: DATA-09, DATA-10, DESC-01, DESC-02, DESC-03, DESC-04, DESC-05, DESC-06
**Success Criteria** (what must be TRUE):
  1. Feature matrix contains all 19+ model features with exact column names from spec; all binary features are {0, 1}; poverty quintiles have 5 groups with roughly equal weighted populations
  2. Survey-weighted Awajun dropout rate exceeds 18% for 2020+ years; Castellano rate is between 10-18% -- confirming the language equity gap
  3. `data/exports/descriptive_tables.json` exists, is valid JSON, and matches the M4 schema (Section 11.6 of spec) with breakdowns by language, sex, rural, region, poverty quintile, heatmap, and temporal trend
  4. `enaho_with_features.parquet` saved with complete feature matrix
  5. Gate test 1.5 passes all assertions (feature count, binary validation, quintile balance, dropout rates, correlation check, export validation)
**Human Gate**: Yes -- review weighted dropout rates by subgroup, 7 visualizations, and Awajun gap
**Plans**: TBD

Plans:
- [ ] 04-01: Feature engineering pipeline (all 19+ features)
- [ ] 04-02: Descriptive statistics + descriptive_tables.json export + gate test 1.5

### Phase 5: Baseline Model + Temporal Splits
**Goal**: Temporal split discipline is established and a logistic regression baseline validates the modeling pipeline with survey-weighted evaluation, setting patterns for gradient boosting phases
**Depends on**: Phase 4
**Requirements**: MODL-01, MODL-02, MODL-06, MODL-08, MODL-09
**Success Criteria** (what must be TRUE):
  1. Temporal splits are correct: train years max=2022, validate year=2023, test year=2024, with zero overlap between splits
  2. Logistic regression converges and achieves PR-AUC > 0.14 on validation (beating random baseline given ~14% dropout rate)
  3. Weighted metrics differ from unweighted metrics (asserting FACTOR07 is actually applied)
  4. `data/exports/model_results.json` exists with `logistic_regression` entry including validate_2023 metrics and threshold analysis at 0.3/0.4/0.5/0.6/0.7
  5. Gate test 2.1 passes; LR coefficients printed for human review showing sensible odds ratios
**Human Gate**: Yes -- review LR coefficients (poverty increases risk, urban decreases risk, age effects)
**Plans**: TBD

Plans:
- [ ] 05-01: Temporal splits + logistic regression + evaluate + gate test 2.1

### Phase 6: LightGBM + XGBoost
**Goal**: The primary LightGBM model (matching Alerta Escuela's algorithm) and XGBoost comparison model are trained and Optuna-tuned, establishing algorithm-independence of fairness findings
**Depends on**: Phase 5
**Requirements**: MODL-03, MODL-04
**Success Criteria** (what must be TRUE):
  1. LightGBM achieves higher validation PR-AUC than logistic regression baseline
  2. XGBoost validation PR-AUC is within 5% of LightGBM (confirming fairness findings are algorithm-independent)
  3. No single feature accounts for more than 50% of LightGBM importance (model uses diverse signal)
  4. `model_results.json` updated with lightgbm and xgboost entries including Optuna best hyperparameters, validation metrics, and threshold analysis
  5. Gate test 2.2 passes; top-10 feature importances printed for human review
**Human Gate**: Yes -- review feature importances (age, poverty, rurality should be top-5)
**Plans**: TBD

Plans:
- [ ] 06-01: LightGBM Optuna tuning + XGBoost training + gate test 2.2

### Phase 7: Calibration + ONNX Export + Final Test
**Goal**: The best model is calibrated, exported to ONNX for browser inference, and evaluated on the 2024 test set exactly once -- the only time test data is touched
**Depends on**: Phase 6
**Requirements**: MODL-05, MODL-07, EXPO-01
**Success Criteria** (what must be TRUE):
  1. Calibrated model has lower Brier score than uncalibrated on validation set
  2. Test set (2024) PR-AUC is within 0.07 of validation PR-AUC (no extreme overfitting)
  3. ONNX file exists at `data/exports/onnx/lightgbm_dropout.onnx`, is under 50MB, and predictions match Python model within 1e-4 absolute difference on 100 random samples
  4. `model_results.json` updated with `test_2024_final` and `test_2024_calibrated` entries plus Alerta Escuela comparison
  5. Gate test 2.3 passes; comparison table with Alerta Escuela published metrics printed for human review
**Human Gate**: Yes -- review calibration plot and metrics vs Alerta Escuela comparison
**Plans**: TBD

Plans:
- [ ] 07-01: Calibration + ONNX export + final test evaluation + gate test 2.3

### Phase 8: Subgroup Fairness Metrics
**Goal**: Comprehensive fairness metrics are computed across all 6 protected dimensions and 3 intersections, quantifying where the model systematically fails different student populations
**Depends on**: Phase 7
**Requirements**: FAIR-01, FAIR-02, FAIR-03, FAIR-04, FAIR-05, FAIR-06
**Success Criteria** (what must be TRUE):
  1. TPR, FPR, FNR, precision, and PR-AUC computed per subgroup across all 6 dimensions (language, sex, geography, region, poverty, nationality) using fairlearn MetricFrame with FACTOR07 survey weights
  2. Calibration per group computed: actual dropout rate among predicted-high-risk (>0.7 proba) differs meaningfully across groups
  3. Three intersectional analyses completed (language x rurality, sex x poverty, language x region) with groups having <50 unweighted observations flagged
  4. `data/exports/fairness_metrics.json` exists, is valid JSON, matches M4 schema (Section 11.2 of spec) including gaps (equalized odds, predictive parity)
  5. Gate test 3.1 passes; FNR by language group and calibration table printed for human review
**Human Gate**: Yes -- review FNR gap direction, calibration by group, intersectional findings
**Plans**: TBD

Plans:
- [ ] 08-01: Subgroup fairness metrics (6 dimensions + 3 intersections) + gate test 3.1

### Phase 9: SHAP Interpretability Analysis
**Goal**: Global, regional, and interaction SHAP values quantify each feature's contribution to dropout predictions, with specific attention to ES_PERUANO (nationality) and ES_MUJER (gender) effects
**Depends on**: Phase 7
**Requirements**: SHAP-01, SHAP-02, SHAP-03, SHAP-04, SHAP-05, SHAP-06
**Success Criteria** (what must be TRUE):
  1. Global SHAP values computed on 2024 test set; top-5 SHAP features overlap with top-5 LR coefficient magnitudes (at least 3 in common)
  2. Regional SHAP computed separately for Costa, Sierra, Selva -- revealing where features like mother tongue matter more
  3. ES_PERUANO and ES_MUJER average SHAP magnitudes quantified specifically; 10 representative student profiles exported with feature values + SHAP values + predicted probability
  4. `data/exports/shap_values.json` exists, matches M4 schema (Section 11.3 of spec) with global_importance, regional, and profiles sections
  5. Gate test 3.2 passes; top-5 global SHAP features and ES_PERUANO/ES_MUJER magnitudes printed for human review
**Human Gate**: Yes -- review feature intuition, nationality magnitude, gender in secundaria, regional differences
**Plans**: TBD

Plans:
- [ ] 09-01: Global + regional + interaction SHAP + 10 profiles + gate test 3.2

### Phase 10: Cross-Validation with Admin Data
**Goal**: Model predictions aggregated to district level correlate positively with administrative dropout rates, and prediction error patterns by indigenous language prevalence reveal spatial equity gaps
**Depends on**: Phase 8
**Requirements**: XVAL-01, XVAL-02, XVAL-03
**Success Criteria** (what must be TRUE):
  1. Pearson correlation between district-level model predictions and admin dropout rates is positive and statistically significant (p < 0.05)
  2. Mean absolute prediction error quantified separately for high-indigenous (>50%) vs low-indigenous (<10%) districts
  3. `data/exports/choropleth.json` exists with >1,500 districts, each having ubigeo, predicted_dropout_rate, admin_dropout_rate, model_error, indigenous_language_pct, and poverty_index
  4. Gate test 3.3 passes; correlation, p-value, and mean error by indigenous % group printed for human review
**Human Gate**: Yes -- review whether indigenous-majority districts have higher error, spatial patterns
**Plans**: TBD

Plans:
- [ ] 10-01: District-level cross-validation + choropleth export + gate test 3.3

### Phase 11: Findings Distillation + Final Exports
**Goal**: All analysis is synthesized into 5-7 bilingual media-ready findings ordered by impact, and all 7 JSON export files are validated against M4 site contracts
**Depends on**: Phases 8, 9, 10
**Requirements**: EXPO-02, EXPO-03, EXPO-04, EXPO-05
**Success Criteria** (what must be TRUE):
  1. `data/exports/findings.json` contains 5-7 findings, each with id, stat, headline_es, headline_en, explanation_es, explanation_en, metric_source, visualization_type, data_key, and severity -- ordered by impact (most striking first)
  2. Every metric_source path in findings.json resolves to an actual value in the referenced export file
  3. All 7 export files present in data/exports/ (findings.json, fairness_metrics.json, shap_values.json, choropleth.json, model_results.json, descriptive_tables.json, onnx/lightgbm_dropout.onnx) plus README.md
  4. `data/exports/README.md` documents each file's purpose, schema, M4 site component mapping, and data provenance chain
  5. Gate test 3.4 passes; all finding headlines printed for human review; Spanish translations are natural for Peruvian audience
**Human Gate**: Yes -- review finding impact ordering, readability, Spanish quality; tag v1.0-analysis-complete
**Plans**: TBD

Plans:
- [ ] 11-01: Findings distillation + final export validation + gate test 3.4

## Progress

**Execution Order:**
Phases execute in numeric order: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11

Note: Phases 8 and 9 are independent (both depend on Phase 7); Phase 10 depends on Phase 8. Phase 11 depends on Phases 8, 9, and 10.

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 0. Environment Setup | 2/2 | Complete | 2026-02-07 |
| 1. ENAHO Single-Year Loader | 0/1 | Not started | - |
| 2. Multi-Year Loader + Harmonization | 0/1 | Not started | - |
| 3. Spatial + Supplementary Data Merges | 0/1 | Not started | - |
| 4. Feature Engineering + Descriptive Statistics | 0/2 | Not started | - |
| 5. Baseline Model + Temporal Splits | 0/1 | Not started | - |
| 6. LightGBM + XGBoost | 0/1 | Not started | - |
| 7. Calibration + ONNX Export + Final Test | 0/1 | Not started | - |
| 8. Subgroup Fairness Metrics | 0/1 | Not started | - |
| 9. SHAP Interpretability Analysis | 0/1 | Not started | - |
| 10. Cross-Validation with Admin Data | 0/1 | Not started | - |
| 11. Findings Distillation + Final Exports | 0/1 | Not started | - |
