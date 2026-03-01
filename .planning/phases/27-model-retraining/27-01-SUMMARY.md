---
phase: 27-model-retraining
plan: 01
subsystem: models
tags: [lightgbm, xgboost, optuna, platt-scaling, onnx, calibration]

requires:
  - phase: 26-feature-engineering
    provides: "31-feature matrix (enaho_with_features.parquet)"
provides:
  - "Retrained LightGBM model (31 features, val PR-AUC 0.2908)"
  - "Retrained XGBoost model (31 features, val PR-AUC 0.2870)"
  - "Re-calibrated LightGBM with updated Platt A/B coefficients"
  - "ONNX export of retrained LightGBM (max diff 1.38e-07)"
  - "Updated model_results.json with all new metrics"
affects: [27-02, fairness, shap, findings]

tech-stack:
  added: []
  patterns: ["dynamic MODEL_FEATURES import enables seamless feature expansion"]

key-files:
  created: []
  modified:
    - data/exports/model_results.json
    - data/exports/onnx/lightgbm_dropout.onnx
    - data/exports/figures/pr_curve_lgbm.png
    - data/exports/figures/pr_curve_xgb.png
    - data/exports/figures/calibration_plot.png
    - data/processed/model_lgbm.joblib
    - data/processed/model_xgb.joblib
    - data/processed/model_lgbm_calibrated.joblib
    - data/processed/predictions_lgbm.parquet
    - data/processed/predictions_xgb.parquet
    - data/processed/predictions_lgbm_calibrated.parquet

key-decisions:
  - "No code changes needed -- MODEL_FEATURES import worked seamlessly for 25-to-31 feature expansion"
  - "New Platt parameters: A=-8.156711, B=5.069181 (steeper sigmoid, shifted intercept)"
  - "LightGBM val PR-AUC improved from 0.2616 to 0.2908 (+11.2%) with 6 additional features"

patterns-established:
  - "Pipeline re-execution pattern: run existing scripts with updated features, no code changes"

requirements-completed: [MODEL-01, MODEL-03]

duration: 7min
completed: 2026-03-01
---

# Phase 27 Plan 01: Model Retraining Summary

**LightGBM and XGBoost retrained with 31-feature matrix (was 25); Platt calibration updated; ONNX re-exported; val PR-AUC improved 11.2%**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-01T17:19:52Z
- **Completed:** 2026-03-01T17:26:47Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- LightGBM retrained with 31 features: val PR-AUC 0.2908 (was 0.2616, +11.2% improvement)
- XGBoost retrained with 31 features: val PR-AUC 0.2870 (was 0.2626, +9.3% improvement)
- Algorithm-independence ratio 0.9870 (PASS, threshold >= 0.95)
- Platt calibration re-applied: A=-8.156711, B=5.069181 (was A=-6.236085, B=4.442308)
- Brier score improvement: 34.0% (0.1710 -> 0.1129)
- ONNX re-exported: 0.09 MB, max diff 1.38e-07
- Val-test PR-AUC gap: 0.0336 (within 0.07 threshold)
- Gate test 2.3: all 11 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Retrain LightGBM + XGBoost with 31-feature matrix** - `ad2a59b` (feat)
2. **Task 2: Re-calibrate LightGBM and re-export ONNX** - `8a1c942` (feat)

## Files Created/Modified
- `data/exports/model_results.json` - Updated lightgbm, xgboost, test_2023_calibrated entries (31 features)
- `data/exports/onnx/lightgbm_dropout.onnx` - ONNX export of retrained LightGBM (31 inputs)
- `data/exports/figures/pr_curve_lgbm.png` - LightGBM PR curve (updated)
- `data/exports/figures/pr_curve_xgb.png` - XGBoost PR curve (updated)
- `data/exports/figures/calibration_plot.png` - Calibration reliability diagram (updated)
- `data/processed/model_lgbm.joblib` - Retrained LightGBM model (gitignored)
- `data/processed/model_xgb.joblib` - Retrained XGBoost model (gitignored)
- `data/processed/model_lgbm_calibrated.joblib` - Re-calibrated LightGBM (gitignored)
- `data/processed/predictions_*.parquet` - Updated prediction parquets (gitignored)

## Key Metrics (Before -> After)

| Metric | Before (25 features) | After (31 features) | Change |
|--------|---------------------|---------------------|--------|
| LightGBM val PR-AUC (W) | 0.2616 | 0.2908 | +11.2% |
| XGBoost val PR-AUC (W) | 0.2626 | 0.2870 | +9.3% |
| Platt A | -6.236085 | -8.156711 | steeper |
| Platt B | 4.442308 | 5.069181 | shifted |
| Brier (uncal -> cal) | 0.186->0.116 | 0.171->0.113 | similar improvement |
| ONNX max diff | 9.96e-08 | 1.38e-07 | both well under 1e-4 |
| Test 2023 PR-AUC | 0.2378 | 0.2572 | +8.2% |
| Test FNR | 62.4% | 64.8% | +2.4pp (threshold shift) |
| Algorithm-independence | 1.0006 | 0.9870 | both PASS |

## Decisions Made
- No code changes needed -- the MODEL_FEATURES dynamic import pattern worked seamlessly for the 25-to-31 feature expansion
- New Platt parameters (A=-8.156711, B=5.069181) replace previous values; model_results.json is authoritative
- LightGBM best iteration: 66 (was 79 with 25 features); XGBoost best iteration: 53 (was 50)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - both pipeline scripts ran without errors or code changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All model artifacts regenerated with 31 features
- Gate test 2.3 passes (11/11)
- Ready for 27-02: fairness re-analysis, SHAP re-run, and findings update
- Note: Gate tests 3.1 (fairness) and 3.2 (SHAP) will need re-running after 27-02

## Self-Check: PASSED

All artifacts verified present. Both commits (ad2a59b, 8a1c942) confirmed in git log. Gate test 2.3: 11/11 pass.

---
*Phase: 27-model-retraining*
*Completed: 2026-03-01*
