---
phase: 28-fairness-re-analysis
plan: 01
subsystem: fairness
tags: [fairlearn, bootstrap, permutation-test, survey-weighted, fnr, fpr]

# Dependency graph
requires:
  - phase: 27-model-retraining
    provides: "Retrained 31-feature models with new predictions and thresholds"
provides:
  - "Updated fairness metrics for all 5 models (lgbm_calibrated, lr, rf, mlp, xgb)"
  - "v1 backups of old 25-feature fairness metrics for before/after comparison"
  - "LR and XGBoost fairness exports (first time generated)"
affects: [28-02-comparison, paper-revision, findings]

# Tech tracking
tech-stack:
  added: []
  patterns: [fairness-pipeline-rerun, v1-backup-pattern]

key-files:
  created:
    - data/exports/fairness_metrics_v1.json
    - data/exports/fairness_metrics_rf_v1.json
    - data/exports/fairness_metrics_mlp_v1.json
    - data/exports/fairness_metrics_lr.json
    - data/exports/fairness_metrics_xgb.json
  modified:
    - data/exports/fairness_metrics.json
    - data/exports/fairness_metrics_rf.json
    - data/exports/fairness_metrics_mlp.json
    - src/fairness/metrics.py

key-decisions:
  - "Updated calibration_note from 0.431 to 0.476 max calibrated prob (new Platt params)"
  - "Patched JSON directly instead of re-running 30-min pipeline for single string change"

patterns-established:
  - "v1-backup pattern: copy old exports as _v1.json before overwriting for before/after comparison"

requirements-completed: [FAIR-01, FAIR-03]

# Metrics
duration: 124min
completed: 2026-03-01
---

# Phase 28 Plan 01: Fairness Re-analysis Summary

**Full fairness pipeline re-run on all 5 retrained 31-feature models with survey-weighted FNR/TPR/FPR across 7 dimensions and 3 intersections**

## Performance

- **Duration:** 124 min (dominated by bootstrap/permutation computation ~30 min per model)
- **Started:** 2026-03-01T19:38:18Z
- **Completed:** 2026-03-01T21:42:00Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments
- Backed up 3 existing fairness exports as _v1 variants for Plan 02 before/after comparison
- Re-ran full fairness pipeline (7 dims, 3 intersections, 1000 bootstrap, 5000 permutations) on all 5 models
- Generated first-ever fairness exports for LR and XGBoost models
- Updated calibration_note to reflect new Platt scaling max prob (0.431 -> 0.476)
- All 20 gate tests pass on updated fairness_metrics.json

## Key Fairness Results (31-feature models)

| Model | Threshold | Castellano FNR | Other Indigenous FNR |
|-------|-----------|----------------|---------------------|
| lgbm_calibrated | 0.185024 | 0.6627 | 0.347 |
| lr | 0.540615 | 0.6082 | -- |
| rf | 0.513856 | 0.6071 | -- |
| mlp | 0.200005 | 0.6191 | -- |
| xgb | 0.529393 | 0.6489 | -- |

Castellano FNR remains consistently the highest across all 5 models (0.607-0.663), confirming algorithm-independence of the "invisibility bias" finding.

## Task Commits

Each task was committed atomically:

1. **Task 1: Back up old fairness exports and clear stale caches** - `9c3617e` (chore)
2. **Task 2: Re-run fairness pipeline for all 5 models** - `fd9bc27` (feat)
3. **Task 3: Run gate tests** - verification only, no commit needed (20/20 pass)

## Files Created/Modified
- `data/exports/fairness_metrics_v1.json` - Backup of old lgbm_calibrated fairness (threshold 0.167268)
- `data/exports/fairness_metrics_rf_v1.json` - Backup of old RF fairness
- `data/exports/fairness_metrics_mlp_v1.json` - Backup of old MLP fairness
- `data/exports/fairness_metrics.json` - Updated lgbm_calibrated fairness (threshold 0.185024)
- `data/exports/fairness_metrics_lr.json` - NEW: LR fairness metrics
- `data/exports/fairness_metrics_rf.json` - Updated RF fairness
- `data/exports/fairness_metrics_mlp.json` - Updated MLP fairness
- `data/exports/fairness_metrics_xgb.json` - NEW: XGBoost fairness metrics
- `src/fairness/metrics.py` - Updated calibration_note string (0.431 -> 0.476)

## Decisions Made
- Updated calibration_note from "max at 0.431" to "max at 0.476" to reflect new Platt parameters (A=-8.156711, B=5.069181)
- Patched the fairness_metrics.json directly for the calibration_note string rather than re-running the 30-minute pipeline

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated stale calibration_note hardcoded string**
- **Found during:** Task 2 (fairness pipeline re-run)
- **Issue:** `src/fairness/metrics.py` line 931 had hardcoded "Calibrated probs max at 0.431" but new Platt scaling produces max 0.476
- **Fix:** Updated string to "0.476" in source code and patched the already-generated JSON
- **Files modified:** src/fairness/metrics.py, data/exports/fairness_metrics.json
- **Verification:** Confirmed max calibrated prob = 0.4761 from predictions parquet
- **Committed in:** fd9bc27 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Essential for accuracy of calibration documentation. No scope creep.

## Issues Encountered
- Pipeline took ~30 min per model (total ~120 min) due to bootstrap/permutation computation on 31-feature models. Expected range was 5-8 min per model in the plan.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 fairness JSONs ready for Plan 02 before/after comparison
- v1 backups preserved for delta analysis
- Gate test 3.1 passes (20/20) confirming structural validity

---
*Phase: 28-fairness-re-analysis*
*Completed: 2026-03-01*
