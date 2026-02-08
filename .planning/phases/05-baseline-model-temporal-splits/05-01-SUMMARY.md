---
phase: 05-baseline-model-temporal-splits
plan: 01
subsystem: models
tags: [sklearn, statsmodels, logistic-regression, survey-weights, temporal-splits, threshold-analysis]

# Dependency graph
requires:
  - phase: 04-feature-engineering-descriptive-statistics
    provides: "enaho_with_features.parquet with 25 model features and 150,135 rows"
provides:
  - "Temporal split functions (train=2018-2021, val=2022, test=2023)"
  - "Survey-weighted evaluation pipeline (PR-AUC, ROC-AUC, F1, Brier, log-loss)"
  - "Threshold analysis at 5 fixed points + optimal"
  - "model_results.json with LR entry (metrics, coefficients, threshold analysis)"
  - "predictions_lr.parquet with per-row predictions for fairness/SHAP"
  - "Persisted sklearn LR model (model_lr.joblib)"
  - "PR curve visualization"
affects: [06-lightgbm-xgboost, 07-calibration-onnx-export, 08-subgroup-fairness-metrics, 09-shap-interpretability]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual-model: sklearn LR for prediction/persistence + statsmodels GLM for inference"
    - "Polars-to-numpy conversion at sklearn boundary with Boolean->Int8 cast"
    - "compute_metrics() with sample_weight for weighted/unweighted comparison"
    - "Threshold analysis from precision_recall_curve output"
    - "PrecisionRecallDisplay.from_predictions() with sample_weight"

key-files:
  created:
    - "src/models/baseline.py"
    - "tests/gates/test_gate_2_1.py"
    - "data/exports/model_results.json"
    - "data/exports/figures/pr_curve_lr.png"
    - "data/processed/predictions_lr.parquet"
    - "data/processed/model_lr.joblib"
  modified: []

key-decisions:
  - "Max weighted F1 as threshold optimization target (optimal=0.5168)"
  - "class_weight='balanced' + FACTOR07 sample_weight (multiplicative interaction documented)"
  - "freq_weights p-values all 0.0 — documented in metadata, use odds ratios for interpretation"
  - "poverty_quintile negative sign due to multicollinearity with poverty_index_z (expected)"
  - "rural near-zero due to correlation with spatial features (expected)"

patterns-established:
  - "Temporal split by year with zero-overlap assertions"
  - "compute_metrics() reusable for LightGBM/XGBoost in Phase 6"
  - "model_results.json model-keyed structure (add lightgbm/xgboost entries later)"
  - "predictions parquet with ID columns + prob/pred for downstream phases"

# Metrics
duration: ~5 min
completed: 2026-02-08
---

# Phase 5 Plan 1: Baseline Model + Temporal Splits Summary

**Logistic regression baseline with temporal splits (2018-2021/2022/2023), survey-weighted evaluation (PR-AUC=0.2103), threshold analysis, and statsmodels coefficient inference for equity feature review**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-08
- **Completed:** 2026-02-08T08:07:40Z
- **Tasks:** 3 (2 auto + 1 human-verify checkpoint)
- **Files created:** 6

## Accomplishments
- Temporal splits with zero overlap: train=98,023 (2018-2021), val=26,477 (2022), test=25,635 (2023)
- LR converges in 307 iterations, val PR-AUC (weighted) = 0.2103, test PR-AUC = 0.1927
- All indigenous language coefficients positive (Quechua OR=1.60, Aimara OR=1.41, Other indigenous OR=2.20)
- Threshold analysis at 5 fixed + optimal (0.5168, max weighted F1=0.2877)
- model_results.json with full metrics, 26 coefficients, threshold analysis
- 52,112 prediction rows saved for Phase 8 fairness and Phase 9 SHAP

## Task Commits

Each task was committed atomically:

1. **Task 1: Create baseline.py with temporal splits, LR training, evaluation** - `b5f0501` (feat)
2. **Task 2: Create gate test 2.1** - `d93481c` (test)
3. **Task 3: Human review of LR coefficients** - approved (no commit)

## Files Created/Modified
- `src/models/baseline.py` - Full baseline pipeline: splits, training, evaluation, threshold analysis, JSON export
- `tests/gates/test_gate_2_1.py` - 11 gate tests validating all pipeline outputs
- `data/exports/model_results.json` - Model metrics, coefficients, threshold analysis (M4 schema)
- `data/exports/figures/pr_curve_lr.png` - Precision-recall curve with threshold markers (72.6 KB)
- `data/processed/predictions_lr.parquet` - 52,112 per-row predictions (val + test)
- `data/processed/model_lr.joblib` - Persisted sklearn LogisticRegression model

## Decisions Made
- Max weighted F1 as threshold optimization target — F1 balances precision/recall for equity audit
- poverty_quintile negative sign accepted: multicollinearity with poverty_index_z (continuous version correctly positive)
- rural near-zero accepted: effect absorbed by correlated spatial features
- All p-values = 0.0 due to freq_weights inflating effective n to ~25M — documented in metadata

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- model_results.json ready for LightGBM/XGBoost entries in Phase 6
- predictions_lr.parquet ready for Phase 8 fairness analysis
- model_lr.joblib available for Phase 7 comparison
- compute_metrics() and temporal split patterns established for reuse

---
*Phase: 05-baseline-model-temporal-splits*
*Completed: 2026-02-08*
