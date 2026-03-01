---
phase: 27-model-retraining
plan: 02
subsystem: models
tags: [logistic-regression, random-forest, mlp, optuna, standardscaler, permutation-importance]

requires:
  - phase: 27-model-retraining
    plan: 01
    provides: "Retrained LightGBM/XGBoost with 31 features in model_results.json"
  - phase: 26-feature-engineering
    provides: "31-feature matrix (enaho_with_features.parquet)"
provides:
  - "Retrained LR model (31 features, val PR-AUC 0.2404)"
  - "Retrained RF model (31 features, val PR-AUC 0.2858)"
  - "Retrained MLP model (31 features, val PR-AUC 0.2677)"
  - "Before/after PR-AUC comparison (prauc_before_after.json) for all 5 model families"
  - "Updated gate tests 2.1/2.2 for 31-feature assertions"
affects: [28-fairness-reanalysis, findings, paper]

tech-stack:
  added: []
  patterns: ["retrain_all.py orchestration script preserves model_results.json entries across baseline.py overwrites"]

key-files:
  created:
    - data/exports/prauc_before_after.json
    - scripts/retrain_all.py
  modified:
    - data/exports/model_results.json
    - data/exports/figures/pr_curve_lr.png
    - data/exports/figures/pr_curve_rf.png
    - data/exports/figures/pr_curve_mlp.png
    - tests/gates/test_gate_2_1.py
    - tests/gates/test_gate_2_2.py

key-decisions:
  - "All 5 models improved with 6 new features: LR +14.3%, LightGBM +11.4%, XGBoost +9.9%, RF +9.4%, MLP +12.5%"
  - "Created retrain_all.py orchestration script to handle baseline.py overwrite safely"
  - "Gate test n_features assertions updated from 25 to 31; LR coefficient count from 26 to 32"

patterns-established:
  - "retrain_all.py: save-run-merge pattern for baseline.py which overwrites model_results.json"

requirements-completed: [MODEL-02, MODEL-04]

duration: 80min
completed: 2026-03-01
---

# Phase 27 Plan 02: LR/RF/MLP Retraining Summary

**LR, RF, MLP retrained with 31-feature matrix; all 5 model families improved 9-14% PR-AUC; before/after comparison confirms new features add predictive signal**

## Performance

- **Duration:** ~80 min (MLP Optuna 50 trials + permutation importance dominated)
- **Started:** 2026-03-01T17:30:00Z
- **Completed:** 2026-03-01T19:00:44Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- LR retrained with 31 features: val PR-AUC 0.2404 (was 0.2103, +14.3% improvement)
- RF retrained with 31 features via Optuna (50 trials): val PR-AUC 0.2858 (was 0.2613, +9.4%)
- MLP retrained with 31 features via Optuna (50 trials): val PR-AUC 0.2677 (was 0.2380, +12.5%)
- prauc_before_after.json created with 5-model before/after comparison
- Gate tests 2.1, 2.2, 2.3, 2.4 all pass (47/47 tests)
- Created retrain_all.py orchestration script for safe multi-model retraining

## Task Commits

Each task was committed atomically:

1. **Task 1: Retrain LR, RF, MLP with 31-feature matrix** - `71cb384` (feat)
2. **Task 2: Before/after PR-AUC comparison and gate test updates** - `9d7f7f9` (feat)

## Files Created/Modified
- `data/exports/model_results.json` - Updated logistic_regression, random_forest, mlp entries (31 features)
- `data/exports/prauc_before_after.json` - 5-model before/after PR-AUC comparison
- `data/exports/figures/pr_curve_lr.png` - LR PR curve (updated)
- `data/exports/figures/pr_curve_rf.png` - RF PR curve (updated)
- `data/exports/figures/pr_curve_mlp.png` - MLP PR curve (updated)
- `scripts/retrain_all.py` - Orchestration script for safe multi-model retraining
- `tests/gates/test_gate_2_1.py` - Updated n_features=31, coefficients=32
- `tests/gates/test_gate_2_2.py` - Updated n_features=31, importances count=31

## Key Metrics (Before -> After)

| Model | 25-feat PR-AUC | 31-feat PR-AUC | Change |
|-------|---------------|---------------|--------|
| Logistic Regression | 0.2103 | 0.2404 | +14.3% |
| LightGBM | 0.2611 | 0.2908 | +11.4% |
| XGBoost | 0.2612 | 0.2870 | +9.9% |
| Random Forest | 0.2613 | 0.2858 | +9.4% |
| MLP | 0.2380 | 0.2677 | +12.5% |

## Decisions Made
- All 5 models improved with the 6 new features (overage, interactions), confirming predictive signal -- a positive finding for the v4 experiment
- LR showed largest relative improvement (+14.3%), suggesting new features add linear signal
- Created orchestration script rather than running models manually to avoid partial-state risk from baseline.py overwriting model_results.json

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all three training pipelines ran without code changes. MLP training took ~60 min wall time due to Optuna + permutation importance with 10 repeats on 31 features.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 5 model families retrained with 31 features
- model_results.json has all 6 keys with n_features=31
- Gate tests 2.1-2.4 all pass (47/47)
- prauc_before_after.json ready for Phase 29 interpretation
- Ready for Phase 28: fairness re-analysis with updated model predictions

## Self-Check: PASSED

All artifacts verified present. Both commits (71cb384, 9d7f7f9) confirmed in git log. Gate tests 2.1-2.4: 47/47 pass.
