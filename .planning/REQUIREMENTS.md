# Requirements: Alerta Escuela Equity Audit

**Defined:** 2026-02-07
**Core Value:** The fairness audit is the product. Models exist to be audited, not to achieve SOTA.

## v1 Requirements

### Environment

- [x] **ENV-01**: Nix flake provides Python 3.12, uv, and system dependencies (OpenMP, cmake for LightGBM)
- [x] **ENV-02**: pyproject.toml declares all Python dependencies with version constraints per spec
- [x] **ENV-03**: download.py fetches ENAHO modules (02, 03, 05, 34, 37) for 2018–2024 and admin dropout CSVs
- [x] **ENV-04**: Directory structure matches spec Section 3 exactly (src/data/, src/models/, src/fairness/, etc.)
- [x] **ENV-05**: .gitignore excludes data/raw/, data/processed/, __pycache__/, .ipynb_checkpoints/

### Data Pipeline

- [ ] **DATA-01**: Load single-year ENAHO with auto-detected delimiter (pipe <=2019, comma >=2020)
- [ ] **DATA-02**: Construct dropout target as (P303==1 & P306==2) with FACTOR07 survey weight preserved
- [ ] **DATA-03**: Load all 7 years (2018–2024) with consistent schema and year column
- [ ] **DATA-04**: Harmonize P300A mother tongue codes (collapse 10–15 -> 3 for cross-year; preserve originals)
- [ ] **DATA-05**: Load and zero-pad district admin dropout rates from datosabiertos
- [ ] **DATA-06**: Load Census 2017 district-level enrichment data
- [ ] **DATA-07**: Load VIIRS nightlights district-level economic proxy
- [ ] **DATA-08**: Merge all sources via LEFT JOIN on UBIGEO preserving ENAHO row count
- [ ] **DATA-09**: Engineer all 19+ model features per spec Section 5 (exact column names)
- [ ] **DATA-10**: Save intermediate datasets as parquet (enaho_merged, enaho_with_features, full_dataset)

### Descriptive Analysis

- [ ] **DESC-01**: Compute survey-weighted dropout rates by mother tongue (harmonized + Awajun disaggregated)
- [ ] **DESC-02**: Compute survey-weighted dropout rates by sex x education level
- [ ] **DESC-03**: Compute survey-weighted dropout rates by rural/urban, region, poverty quintile
- [ ] **DESC-04**: Generate heatmap data for language x rurality interaction
- [ ] **DESC-05**: Generate choropleth prep data (district-level weighted dropout rates)
- [ ] **DESC-06**: Export descriptive tables to data/exports/descriptive_tables.json matching M4 schema

### Modeling

- [ ] **MODL-01**: Create temporal splits: train 2018–2022, validate 2023, test 2024 with no overlap
- [ ] **MODL-02**: Train logistic regression baseline with class_weight='balanced' and survey-weighted eval
- [ ] **MODL-03**: Train LightGBM with Optuna hyperparameter tuning (50 trials on validation PR-AUC)
- [ ] **MODL-04**: Train XGBoost for algorithm-independence comparison
- [ ] **MODL-05**: Calibrate best model (isotonic vs sigmoid, keep lower Brier) on validation set
- [ ] **MODL-06**: Tune threshold per model using F1 on validation; report at 0.3, 0.4, 0.5, 0.6, 0.7
- [ ] **MODL-07**: Evaluate calibrated model on test set (2024) exactly once -- final evaluation
- [ ] **MODL-08**: Export model results to data/exports/model_results.json matching M4 schema
- [ ] **MODL-09**: All metrics survey-weighted via FACTOR07; also compute unweighted and assert they differ

### Fairness Audit

- [ ] **FAIR-01**: Compute TPR, FPR, FNR, precision, PR-AUC per subgroup across all 6 dimensions
- [ ] **FAIR-02**: Compute calibration per group (actual dropout rate among predicted-high-risk >0.7)
- [ ] **FAIR-03**: Analyze 3 intersections: language x rurality, sex x poverty, language x region
- [ ] **FAIR-04**: Flag intersectional groups with <50 unweighted observations
- [ ] **FAIR-05**: Compute equalized odds gap and predictive parity gap per dimension
- [ ] **FAIR-06**: Export fairness metrics to data/exports/fairness_metrics.json matching M4 schema

### SHAP Analysis

- [ ] **SHAP-01**: Compute global SHAP values using TreeExplainer on test set (2024)
- [ ] **SHAP-02**: Compute regional SHAP (Costa/Sierra/Selva separately) and compare feature rankings
- [ ] **SHAP-03**: Compute SHAP interaction values (subsample to 1000 if needed)
- [ ] **SHAP-04**: Quantify ES_PERUANO and ES_MUJER SHAP magnitudes specifically
- [ ] **SHAP-05**: Generate 10 representative student profiles with SHAP values (diverse regions/languages)
- [ ] **SHAP-06**: Export SHAP values to data/exports/shap_values.json matching M4 schema

### Cross-Validation

- [ ] **XVAL-01**: Aggregate model predictions to district level and correlate with admin dropout rates
- [ ] **XVAL-02**: Compare prediction error between high-indigenous (>50%) and low-indigenous (<10%) districts
- [ ] **XVAL-03**: Export choropleth data to data/exports/choropleth.json matching M4 schema

### Export

- [ ] **EXPO-01**: Export LightGBM as ONNX with prediction validation (max diff < 1e-4)
- [ ] **EXPO-02**: Distill 5–7 media-ready findings with Spanish and English translations
- [ ] **EXPO-03**: All 7 export JSON files present matching M4 site contracts (Section 11 of specs.md)
- [ ] **EXPO-04**: Export README.md documenting each file's schema and provenance
- [ ] **EXPO-05**: Findings ordered by impact (most striking first)

### Testing

- [ ] **TEST-01**: Gate tests for each phase validate outputs before proceeding (12 gate test files)
- [ ] **TEST-02**: Unit tests for ENAHO loader, harmonization, UBIGEO padding
- [ ] **TEST-03**: Human gate reviews at Phases 1, 4, 5, 6, 7, 8, 9, 10, 11

## v2 Requirements

### Notifications & Monitoring

- **V2-01**: Automated pipeline re-run when new ENAHO year released
- **V2-02**: Dashboard for exploring fairness metrics interactively

### Extended Analysis

- **V2-03**: Causal analysis (mediation/moderation) of language -> dropout pathway
- **V2-04**: Bias mitigation experiments (reweighting, constraint optimization)
- **V2-05**: Comparison with other Latin American education prediction systems

### Data Extensions

- **V2-06**: Censo Escolar school-level features (student-teacher ratio, infrastructure)
- **V2-07**: Panel tracking across ENAHO waves (if feasible)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Replicating Alerta Escuela's exact model | No SIAGIE access -- we use ENAHO proxy |
| Web application / M4 site | Separate repo -- we only produce export JSON |
| Real-time prediction or deployment | Audit, not production system |
| Deep learning (TabNet, neural networks) | Spec-locked to 3 models: LogReg, LightGBM, XGBoost |
| Experiment tracking (MLflow, DVC, W&B) | Export metrics as JSON instead |
| Geospatial processing (geopandas, GDAL) | All geo data is pre-aggregated CSV/JSON |
| Bias mitigation / debiasing | Audit scope only -- document gaps, don't fix them |
| Causal inference | Observational audit, not causal study |
| pandas for data processing | Polars-first; pandas only at sklearn/fairlearn/shap boundary |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 0 | Complete |
| ENV-02 | Phase 0 | Complete |
| ENV-03 | Phase 0 | Complete |
| ENV-04 | Phase 0 | Complete |
| ENV-05 | Phase 0 | Complete |
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| DATA-05 | Phase 3 | Pending |
| DATA-06 | Phase 3 | Pending |
| DATA-07 | Phase 3 | Pending |
| DATA-08 | Phase 3 | Pending |
| DATA-09 | Phase 4 | Pending |
| DATA-10 | Phase 4 | Pending |
| DESC-01 | Phase 4 | Pending |
| DESC-02 | Phase 4 | Pending |
| DESC-03 | Phase 4 | Pending |
| DESC-04 | Phase 4 | Pending |
| DESC-05 | Phase 4 | Pending |
| DESC-06 | Phase 4 | Pending |
| MODL-01 | Phase 5 | Pending |
| MODL-02 | Phase 5 | Pending |
| MODL-03 | Phase 6 | Pending |
| MODL-04 | Phase 6 | Pending |
| MODL-05 | Phase 7 | Pending |
| MODL-06 | Phase 5 | Pending |
| MODL-07 | Phase 7 | Pending |
| MODL-08 | Phase 7 | Pending |
| MODL-09 | Phase 5 | Pending |
| FAIR-01 | Phase 8 | Pending |
| FAIR-02 | Phase 8 | Pending |
| FAIR-03 | Phase 8 | Pending |
| FAIR-04 | Phase 8 | Pending |
| FAIR-05 | Phase 8 | Pending |
| FAIR-06 | Phase 8 | Pending |
| SHAP-01 | Phase 9 | Pending |
| SHAP-02 | Phase 9 | Pending |
| SHAP-03 | Phase 9 | Pending |
| SHAP-04 | Phase 9 | Pending |
| SHAP-05 | Phase 9 | Pending |
| SHAP-06 | Phase 9 | Pending |
| XVAL-01 | Phase 10 | Pending |
| XVAL-02 | Phase 10 | Pending |
| XVAL-03 | Phase 10 | Pending |
| EXPO-01 | Phase 7 | Pending |
| EXPO-02 | Phase 11 | Pending |
| EXPO-03 | Phase 11 | Pending |
| EXPO-04 | Phase 11 | Pending |
| EXPO-05 | Phase 11 | Pending |
| TEST-01 | Phase 1 | Complete |
| TEST-02 | Phase 1 | Complete |
| TEST-03 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 53 total
- Mapped to phases: 53
- Unmapped: 0

---
*Requirements defined: 2026-02-07*
*Last updated: 2026-02-07 after roadmap creation*
