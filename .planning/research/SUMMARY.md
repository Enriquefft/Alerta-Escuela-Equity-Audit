# Project Research Summary

**Project:** Alerta Escuela Equity Audit
**Domain:** ML fairness audit pipeline (survey microdata, binary classification, equity analysis)
**Researched:** 2026-02-07
**Confidence:** HIGH

## Executive Summary

This project is an independent equity audit of Peru's Alerta Escuela dropout prediction system. The audit uses ENAHO survey microdata (2018-2024, ~180K students) to replicate Alerta Escuela's LightGBM approach, evaluate fairness across six protected dimensions (language, sex, geography, region, poverty, nationality), and produce bilingual media-ready findings for a Next.js scrollytelling site. Research confirms this is a specialized ML fairness pipeline requiring survey-weighted metrics throughout, strict temporal validation (train 2018-2022, validate 2023, test 2024), and robust handling of government data inconsistencies.

The recommended approach uses polars for data processing (30-50x faster than pandas for CSV operations), LightGBM + XGBoost for modeling with Optuna hyperparameter tuning, fairlearn for subgroup metrics with survey weights, and SHAP TreeExplainer for feature attribution. The architecture follows a gate-guarded pipeline pattern with immutable intermediate artifacts (parquet files) at each stage, enabling reproducibility and incremental execution. The spec-locked Python 3.12 stack has been validated via live `uv pip compile` resolution, confirming all packages resolve compatibly.

Critical risks center on methodological discipline rather than technical complexity. Survey weights (FACTOR07) must be passed to every metric computation — omission invalidates the entire audit. UBIGEO zero-padding and P300A mother tongue harmonization (2020 coding change) are data integrity issues that create silent merge failures or fake demographic shifts if mishandled. Temporal data leakage (touching 2024 test data before Phase 7) destroys audit credibility. These pitfalls are preventable through gate tests and defensive assertions embedded throughout the pipeline.

## Key Findings

### Recommended Stack

**Confidence:** HIGH — all versions verified via live `uv pip compile` on this machine (2026-02-07).

The stack is fully specified and validated. Python 3.12 with polars 1.38.1 for data processing, scikit-learn 1.8.0 + LightGBM 4.6.0 + XGBoost 3.1.3 for modeling, fairlearn 0.13.0 for fairness metrics, SHAP 0.50.0 for interpretability, Optuna 4.7.0 for hyperparameter tuning, and onnxmltools 1.16.0 for ONNX export. All dependencies resolve without conflicts in Python 3.12.

**Core technologies:**
- **polars (>=1.38.1):** Primary data processing — 30-50x faster than pandas for CSV/parquet operations on ~180K rows, expression-based API prevents mutation bugs
- **LightGBM (>=4.6.0):** Primary model — matches Alerta Escuela's algorithm, sklearn API for Optuna integration, callbacks-based early stopping in v4.x
- **fairlearn (>=0.13.0):** Fairness metrics — MetricFrame with sample_params for survey-weighted subgroup metrics, narwhals backend makes it pandas-agnostic
- **SHAP (>=0.50.0):** Interpretability — TreeExplainer for LightGBM feature attribution, interaction values for compound disadvantage analysis
- **Nix + uv:** Environment management — Nix provides system deps (gcc, openblas, libgomp for LightGBM OpenMP), uv manages Python packages (already installed system-wide via Nix)

**Critical dependencies:**
- **pyarrow (>=19.0):** Required for polars `.to_pandas()` conversion at sklearn boundary — must be in dependencies
- **pandas (>=3.0.0):** Only used at sklearn/fairlearn/shap boundaries via polars `.to_pandas()` — pandas 3.0 is verified compatible with all stack packages
- **Nix flake:** Use Manim-ML pattern (Nix provides system deps, uv builds wheels) rather than uv2nix pattern (too complex for compiled ML packages)

**Key API patterns documented:**
- Polars: `str.pad_start(6, "0")` for UBIGEO zero-padding (NOT `str.zfill`, which doesn't exist in polars)
- Polars CSV: `separator` parameter (NOT `sep`)
- LightGBM 4.x: `callbacks=[lgb.early_stopping(50)]` (NOT deprecated `early_stopping_rounds` parameter)
- fairlearn MetricFrame: nested `sample_params` dict for weighted metrics per metric name
- SHAP 0.50.0: may return list or ndarray for binary classification depending on version — defensive code needed

### Expected Features

**Confidence:** MEDIUM — based on ML fairness audit best practices, peer-reviewed literature, and spec requirements.

**Must have (table stakes):**
- **Survey-weighted metrics throughout:** Every metric (descriptive, model eval, fairness) must pass FACTOR07 expansion factors — unweighted metrics describe the sample, not Peru's population (methodological error)
- **Subgroup fairness metrics (6 dimensions):** Language, sex, geography, region, poverty, nationality with FNR/FPR/TPR/precision/PR-AUC per group via fairlearn MetricFrame
- **Intersectional analysis (3 intersections):** Language x rurality, sex x poverty, language x region to capture compounding disadvantage
- **SHAP-based feature attribution:** Quantify marginal contribution of each feature (e.g., "How much does being non-Peruvian independently increase predicted risk?")
- **Temporal train/validate/test split:** Train on 2018-2022, validate on 2023, test on 2024 (touched once) — prevents temporal leakage
- **PR-AUC as primary metric:** Sensitive to minority class (14% dropout rate), unlike ROC-AUC which is inflated by true negatives
- **Bilingual findings (Spanish + English):** Natural Spanish for Peruvian journalists, English for international audience
- **JSON export contracts for M4 site:** Seven schema-documented exports (findings, fairness_metrics, shap_values, choropleth, model_results, descriptive_tables, onnx) — committed to git as single source of truth

**Should have (competitive advantage):**
- **Cross-validation against admin district data:** ENAHO predictions aggregated to district level vs MINEDU admin dropout rates — reveals geographic error patterns
- **Regional SHAP decomposition (Costa/Sierra/Selva):** Peru's three natural regions have different dropout dynamics — regional SHAP reveals where features matter differently
- **P300A harmonization (2020 structural break):** INEI disaggregated code 3 into 6 indigenous languages in 2020 — harmonization enables cross-year analysis without artifactual trends
- **SHAP interaction effects:** Quantify super-additive risk (poverty x language worse than sum) — evidence for "intersectional harm" beyond group-level metrics
- **10 representative student profiles with SHAP waterfall:** Concrete stories for media (e.g., "14-year-old Awajun girl in rural Loreto") bridge statistics to human narratives
- **Threshold sensitivity analysis:** Metrics at 0.3, 0.4, 0.5, 0.6, 0.7 — shows whether any threshold achieves acceptable fairness
- **Choropleth-ready spatial export:** District-level prediction error colored by indigenous language prevalence — most powerful single visualization
- **ONNX model export for browser inference:** Enables interactive prediction demo on scrollytelling site

**Defer (v2+ or out of scope):**
- **Bias mitigation / debiased model:** This is an audit, not a model improvement project — conflates auditor with developer role
- **Causal fairness analysis:** Requires causal DAG and assumptions beyond observational survey data
- **Deep learning models (TabNet, neural nets):** Spec locks to LogReg, LightGBM, XGBoost — neural nets make SHAP intractable
- **Experiment tracking (MLflow, W&B, DVC):** Overkill for 3 models with one Optuna sweep each — JSON exports are sufficient reproducibility
- **Real-time dashboard / Streamlit app:** Downstream consumer is Next.js scrollytelling site, not a dashboard
- **Geospatial processing (geopandas, GDAL):** Heavy dependencies for data production — map rendering is site's job, not audit's

### Architecture Approach

**Confidence:** HIGH — architecture patterns are well-established for batch ML pipelines.

The pipeline follows a five-layer architecture with gate-guarded stages: (1) Data Ingestion (ENAHO, admin, census, nightlights loaders), (2) Harmonization + Merge (LEFT JOINs on UBIGEO, P300A harmonization, feature engineering), (3) Model Training (temporal splits, LogReg/LightGBM/XGBoost training, calibration, ONNX export), (4) Fairness Analysis (MetricFrame across 6 dimensions + 3 intersections, SHAP global/regional/interaction, admin cross-validation), and (5) Distillation + Export (synthesize all JSONs into media-ready findings).

**Major components:**
1. **`src/data/`** — One file per data source (enaho.py, admin.py, census.py, nightlights.py) creates clean responsibility boundaries; features.py orchestrates merges
2. **`src/models/`** — Separates train.py (produces models), evaluate.py (measures), calibrate.py (post-processes), export_onnx.py (serializes) to enforce "test set touched once" discipline
3. **`src/fairness/`** — Mirrors modeling layer for the actual product: metrics.py (fairlearn subgroup metrics), shap_analysis.py (interpretability), distill.py (journalist-facing synthesis)
4. **`tests/gates/`** — Numbered gate tests (1.1, 1.2, ..., 3.4) validate pipeline stage outputs against invariants before proceeding
5. **Polars-first with boundary conversion** — Pure polars for all data wrangling, `.to_pandas()` only at sklearn/fairlearn/shap boundaries, explicit conversion prevents type confusion

**Key architectural patterns:**
- **Gate-guarded pipeline stages:** Each stage writes parquet/JSON, gate test validates, next stage proceeds — catches data quality issues early
- **Survey-weight-first metric functions:** Every function requires `sample_weight` parameter, asserts weighted != unweighted to catch silent omission
- **Immutable intermediate artifacts:** Stages never modify inputs, full reproducibility via delete-and-rerun from raw
- **Temporal split isolation:** 2024 test data excluded from all preprocessing (StandardScaler, quintile breaks) until Phase 7

**Data flow summary:**
```
Raw CSVs → enaho_merged.parquet → full_dataset.parquet → enaho_with_features.parquet
  → [polars-to-pandas boundary] → sklearn training → calibrated models + predictions
  → [polars-to-pandas boundary] → fairlearn MetricFrame + SHAP → all JSON exports
  → distill.py → findings.json (bilingual, media-ready)
```

### Critical Pitfalls

**Confidence:** HIGH — most pitfalls verified via official docs, GitHub issues, or academic literature.

1. **Survey weights omitted from metrics (Pitfall 4):** All reported numbers are wrong if FACTOR07 is not passed to every metric computation. ENAHO oversamples rural areas; unweighted dropout rates are biased upward. Prevention: assert `weighted != unweighted` after every metric computation, make `sample_weight` a required parameter.

2. **UBIGEO leading zero loss creates silent merge failures (Pitfall 2):** Departments 01-09 lose leading zero when parsed as integer, causing 36% of districts to fail spatial merges silently. Prevention: force string type and pad at load time (`pl.col('UBIGEO').cast(pl.Utf8).str.pad_start(6, "0")`), assert `ubigeo.str.len_chars().min() == 6` after every merge.

3. **P300A harmonization failure creates fake demographic shifts (Pitfall 3):** INEI disaggregated code 3 ("otra lengua nativa") into 6 specific indigenous languages in 2020. Without harmonization, code 3 appears to drop 80% in 2020. Prevention: implement `harmonize_p300a()` creating both `p300a_harmonized` (cross-year) and `p300a_original` (2020+ disaggregated) columns.

4. **Temporal data leakage — using 2024 test data before Phase 7 (Pitfall 7):** Computing feature engineering parameters (income quintiles, StandardScaler) on full dataset including 2024 leaks distributional information. Prevention: `create_temporal_splits()` must be called before any model-related computation, assert `X_train['year'].max() == 2022` throughout.

5. **fairlearn MetricFrame sample_params API silently drops weights (Pitfall 5):** Multi-metric usage requires nested dict `{'metric_name': {'sample_weight': w}}` not flat dict. Flat dict is silently ignored. Prevention: always use nested dict format, verify `mf.overall` differs from unweighted manual computation.

6. **ENAHO delimiter mismatch silently loads garbage (Pitfall 1):** INEI changed delimiter from pipe `|` (2018-2019) to comma `,` (2020-2024). Wrong delimiter parses entire row as single column. Prevention: hardcode `separator = '|' if year <= 2019 else ','`, assert DataFrame has >=20 columns after load.

7. **SHAP TreeExplainer output shape varies by version (Pitfall 6):** SHAP <0.45 returns list `[neg_class, pos_class]`, SHAP >=0.45 returns ndarray. Spec requires >=0.45 but code examples use old `[1]` indexing. Prevention: defensive code handling both formats, assert `shap_vals.shape == (n_samples, n_features)`.

8. **Polars-to-pandas conversion corrupts data types (Pitfall 9):** Nullable Int64 becomes float64, categorical loses encoding, column order not guaranteed. Prevention: fill nulls before conversion, explicitly select and order columns, assert column order matches `MODEL_FEATURES`.

## Implications for Roadmap

Based on research, the pipeline naturally decomposes into 11 phases following the data flow from raw CSVs to media-ready findings. Phase ordering is constrained by hard dependencies: data before models, baseline before boost, training before fairness, all exports before distillation.

### Phase 1: Data Ingestion Foundation
**Rationale:** Must load and validate ENAHO data before any other work. Single-year loader validates delimiter detection, UBIGEO padding, column normalization patterns before scaling to multi-year.
**Delivers:** `src/data/enaho.py` with `load_enaho_year()`, `src/utils.py` with UBIGEO padding, gate tests validating row counts and dropout rates.
**Addresses:** Pitfall 1 (delimiter mismatch), Pitfall 2 (UBIGEO zero loss), Pitfall 15 (column name variations).
**Research flag:** Standard patterns — skip phase-specific research.

### Phase 2: Multi-Year Pipeline + Harmonization
**Rationale:** Cross-year analysis requires P300A harmonization before proceeding. Both harmonized and original columns needed for different analyses.
**Delivers:** Multi-year loader, `harmonize_p300a()`, `enaho_merged.parquet` (7 years, ~180K rows).
**Addresses:** Pitfall 3 (P300A coding change 2020), creates foundation for language-disaggregated fairness analysis.
**Research flag:** Standard patterns — P300A harmonization is fully specified.

### Phase 3: Spatial Data Integration
**Rationale:** Admin/census/nightlights enrichment depends on ENAHO foundation but can happen before feature engineering.
**Delivers:** Admin, census, nightlights loaders, spatial merge in `features.py`, `full_dataset.parquet`.
**Addresses:** Pitfall 2 (UBIGEO in all sources), Pitfall 14 (cartesian join from duplicates).
**Research flag:** Standard patterns — LEFT JOIN on UBIGEO is straightforward.

### Phase 4: Feature Engineering + Descriptive Analysis
**Rationale:** All modeling depends on features. Descriptive gaps (weighted crosstabs) are model-independent findings that inform modeling phase.
**Delivers:** Feature engineering pipeline, survey-weighted descriptive statistics notebook, `enaho_with_features.parquet`.
**Addresses:** Establishes survey weight discipline (Pitfall 4), produces first findings (language/sex/region dropout rates).
**Research flag:** Standard patterns — weighted aggregations in polars are well-documented.

### Phase 5: Baseline Model (Logistic Regression)
**Rationale:** Validates temporal split + training pipeline works before expensive LightGBM Optuna tuning. Linear model provides interpretable coefficient baseline for SHAP comparison.
**Delivers:** Temporal splits (train/val/test), LogisticRegression with StandardScaler, survey-weighted evaluation metrics.
**Addresses:** Pitfall 7 (temporal leakage), Pitfall 9 (polars-to-pandas conversion), establishes evaluation patterns.
**Research flag:** Standard patterns — sklearn Pipeline with sample_weight is well-documented.

### Phase 6: Gradient Boosting Models (LightGBM + XGBoost)
**Rationale:** Primary models for the audit. LightGBM matches Alerta Escuela's algorithm, XGBoost validates algorithm-independence of fairness findings. Optuna tunes both.
**Delivers:** Trained LightGBM and XGBoost models, Optuna hyperparameter search, validation set performance comparison.
**Addresses:** Pitfall 11 (class imbalance + survey weights interaction), establishes best model for Phase 7.
**Research flag:** Standard patterns — LightGBM sklearn API + Optuna integration well-documented.

### Phase 7: Calibration + ONNX Export + Final Test Evaluation
**Rationale:** First (and only) time test set is used. Calibration improves Brier score for probabilistic interpretation. ONNX export enables browser inference on M4 site.
**Delivers:** Calibrated LightGBM model, ONNX export with validation, test set (2024) evaluation metrics.
**Addresses:** Pitfall 8 (ONNX float32 accumulation), Pitfall 13 (ONNX output indexing), Pitfall 18 (calibration on correct set).
**Research flag:** Moderate — ONNX conversion + validation may need debugging if tolerance fails.

### Phase 8: Subgroup Fairness Metrics
**Rationale:** Core audit output. Depends on calibrated model + test predictions from Phase 7.
**Delivers:** fairlearn MetricFrame across 6 dimensions (language, sex, geography, region, poverty, nationality) + 3 intersections, calibration by group, `fairness_metrics.json`.
**Addresses:** Pitfall 5 (MetricFrame sample_params nesting), Pitfall 10 (small sample intersections).
**Research flag:** Moderate — fairlearn MetricFrame with sample_params needs careful testing; API is well-documented but easy to misuse.

### Phase 9: SHAP Interpretability Analysis
**Rationale:** Quantifies feature contributions, answers "How much does nationality independently increase risk?" Runs in parallel to Phase 8 (both consume same test set).
**Delivers:** Global SHAP, regional SHAP (Costa/Sierra/Selva), interaction values, 10 representative profiles, `shap_values.json`.
**Addresses:** Pitfall 6 (SHAP output shape), Pitfall 12 (SHAP on correct dataset), Pitfall 17 (model_output setting).
**Research flag:** Low-moderate — SHAP TreeExplainer is well-documented, but version-specific behavior needs defensive coding.

### Phase 10: Admin Data Cross-Validation
**Rationale:** Validates model against external ground truth. Depends on Phase 8 producing district-aggregated predictions.
**Delivers:** District-level prediction error correlation with admin rates, spatial patterns by indigenous language prevalence, `choropleth.json`.
**Addresses:** Cross-validation differentiator feature, spatial visualization for M4 site.
**Research flag:** Standard patterns — aggregation + correlation straightforward.

### Phase 11: Findings Distillation + Final Exports
**Rationale:** Synthesizes all prior analyses into 5-7 journalist-readable findings. Depends on all Phase 8-10 exports existing.
**Delivers:** `findings.json` (bilingual), `README.md` (export schemas), all 7 JSON exports committed.
**Addresses:** Bilingual media-ready output (table stakes feature), export contracts for M4 site.
**Research flag:** Standard patterns — JSON export + bilingual text writing.

### Phase Ordering Rationale

**Hard dependencies:**
- Phases 1-4 are strictly sequential (data foundation → harmonization → merge → features)
- Phase 5 must precede 6 (baseline validates pipeline before expensive boosting)
- Phases 5-7 must precede 8-10 (fairness analysis requires trained models)
- Phase 11 must be last (distillation reads all prior exports)

**Parallelization opportunities:**
- Phases 8 (fairness) and 9 (SHAP) are independent given Phase 7 outputs
- Phase 10 depends on Phase 8 district aggregations, so must follow 8

**Pitfall mitigation through ordering:**
- Phase 1 establishes UBIGEO/delimiter patterns before merge complexity
- Phase 2 harmonizes P300A before any language-based analysis
- Phase 5 validates temporal split discipline before expensive tuning
- Phase 7 isolates test set usage to single gate-tested phase

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 7 (ONNX Export):** ONNX float32 tolerance may require debugging; sklearn-onnx conversion options may need tuning; browser onnxruntime-web compatibility uncertain
- **Phase 8 (Fairness Metrics):** fairlearn sample_params API is subtle; intersectional sample sizes need careful handling; confidence intervals for small groups may need bootstrap implementation

**Phases with standard patterns (skip research-phase):**
- **Phases 1-4:** Polars data loading + feature engineering well-documented; spatial merges straightforward
- **Phases 5-6:** sklearn + LightGBM + Optuna integration widely documented
- **Phase 9:** SHAP TreeExplainer API stable despite version differences
- **Phases 10-11:** Aggregation + JSON export are standard Python patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified via live `uv pip compile` on this machine (2026-02-07); API patterns verified or extracted from official docs |
| Features | MEDIUM | Based on ML fairness audit literature (Gender Shades, ProPublica COMPAS, NIST AI RMF) and spec requirements; WebSearch unavailable to verify latest community practices |
| Architecture | HIGH | Gate-guarded pipeline with immutable artifacts is well-established pattern; polars-pandas boundary explicitly documented in spec |
| Pitfalls | HIGH | Most pitfalls verified via official docs, GitHub issues, or peer-reviewed literature; some (e.g., LightGBM `is_unbalance` + survey weights interaction) are medium confidence due to sparse documentation |

**Overall confidence:** HIGH

The project is well-specified with authoritative docs (specs.md) as primary source. Stack is fully validated and compatible. Architecture patterns are standard for batch ML pipelines. The main risks are methodological discipline (survey weights, temporal leakage, harmonization) rather than technical unknowns — all are preventable via gate tests.

### Gaps to Address

**Known gaps requiring validation during implementation:**

1. **fairlearn 0.13.0 sample_params API:** Spec examples show nested dict pattern, but fairlearn docs should be verified at implementation time to confirm exact parameter names and behavior. Training knowledge through May 2025 suggests API is stable, but version-specific testing needed.

2. **SHAP 0.50.0 binary classification output format:** Research identifies that SHAP <0.45 returns list, >=0.45 returns ndarray. Spec requires >=0.45, but exact behavior with LightGBM sklearn API (LGBMClassifier) vs native Booster needs runtime verification. Defensive code handling both formats recommended.

3. **LightGBM `is_unbalance` + survey weights interaction:** Documentation is sparse on how these two weighting mechanisms interact. Research recommends using survey weights alone (set `is_unbalance=False`), but this should be validated by comparing calibration curves with both strategies during Phase 5-6.

4. **ONNX float32 accumulation tolerance:** Research indicates 500 trees may exceed 1e-5 tolerance. Phase 7 should test with actual trained model; tolerance may need relaxing to 1e-4 or using double-precision tree summation if available in onnxmltools.

5. **pandas 3.0 breaking changes:** Research confirms pandas 3.0 resolves with all stack packages via narwhals backend in fairlearn 0.13.0, but should verify no deprecated API usage in transitive dependencies (statsmodels, fairlearn) at runtime.

6. **Intersectional sample sizes:** Spec specifies language x rurality, sex x poverty, language x region. Research flags that disaggregated indigenous language codes (Awajun, Shawi, etc.) have small samples even without intersections. Phase 8 must compute unweighted sample sizes per intersection and flag groups with n < 50.

**How to handle gaps:**
- **Phases 5-8:** Write defensive code with explicit type/shape assertions to catch API mismatches early
- **Phase 7:** Budget extra time for ONNX debugging; prepare fallback strategies (reduced tree count, relaxed tolerance)
- **Phase 8:** Implement minimum sample size checks; flag (don't suppress) small groups in exports with caveats
- **All phases:** Run gate tests after each phase to validate assumptions before proceeding

## Sources

### Primary (HIGH confidence)
- **specs.md** — Authoritative project specification covering stack, features, architecture, pitfalls, and export contracts
- **STACK.md** — Live `uv pip compile` verification on 2026-02-07; actual polars 1.38.1 API testing via `uv run`
- **Official documentation:**
  - [fairlearn 0.14 MetricFrame](https://fairlearn.org/main/user_guide/assessment/advanced_metricframe.html) — sample_params nesting
  - [SHAP release notes v0.45](https://shap.readthedocs.io/en/latest/release_notes.html) — output shape breaking change
  - [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — `is_unbalance` and callbacks API
  - [polars to_pandas](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.to_pandas.html) — type conversion behavior
  - [sklearn-onnx LightGBM](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm_reg.html) — float32 accumulation

### Secondary (MEDIUM confidence)
- **GitHub issues:**
  - [SHAP #526](https://github.com/shap/shap/issues/526) — TreeExplainer binary classification output
  - [polars #8204](https://github.com/pola-rs/polars/issues/8204) — Int64-to-float64 with nulls
  - [LightGBM #6807](https://github.com/microsoft/LightGBM/issues/6807) — class/sample weights interaction (sparse discussion)
  - [onnxmltools #150](https://github.com/onnx/onnxmltools/issues/150) — ONNX prediction divergence
- **Academic literature:**
  - [Survey Weights in ML (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0280387) — 5pp F1 overestimation
  - [Sample Size for Fairness Audits (arXiv)](https://arxiv.org/html/2312.04745) — minimum samples for intersectional analysis
- **Community resources:**
  - [PeruData/ENAHO](https://github.com/PeruData/ENAHO) — delimiter and UBIGEO handling patterns
  - [UBIGEO Peru Repository](https://github.com/ernestorivero/Ubigeo-Peru) — coding standards

### Tertiary (LOW confidence, needs validation)
- **LightGBM `is_unbalance` + survey weights:** No definitive guidance found; recommendation based on statistical reasoning
- **ONNX onnxruntime-web browser compatibility:** Not verified; assumption based on ONNX Runtime documentation

---
*Research completed: 2026-02-07*
*Ready for roadmap: yes*
