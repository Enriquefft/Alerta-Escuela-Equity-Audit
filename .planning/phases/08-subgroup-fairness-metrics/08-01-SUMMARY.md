---
phase: 08-subgroup-fairness-metrics
plan: 01
subsystem: fairness
tags: [fairlearn, metricframe, fnr, fpr, calibration, equalized-odds, intersectional, survey-weights]

# Dependency graph
requires:
  - phase: 07-calibration-onnx-export-final-test
    provides: "Calibrated predictions parquet, model_results.json with threshold"
  - phase: 04-feature-engineering-descriptive-statistics
    provides: "enaho_with_features.parquet with sensitive feature columns"
provides:
  - "Fairness metrics across 7 dimensions + 3 intersections (fairness_metrics.json)"
  - "Per-group TPR/FPR/FNR/precision/PR-AUC with FACTOR07 survey weights"
  - "Calibration-by-group (actual dropout rate among high-risk >0.7 uncalibrated)"
  - "Equalized odds gaps, predictive parity gaps per dimension"
  - "Intersectional analysis: language x rurality, sex x poverty, language x region"
affects: [10-cross-validation-admin-data, 11-findings-distillation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-MetricFrame pattern: separate binary (TPR/FPR/FNR/precision) and probability (PR-AUC) instances"
    - "Custom weighted_fpr function (sklearn lacks fpr_score)"
    - "sample_params nested dict: {'metric_name': {'sample_weight': weights}} per metric"
    - "Uncalibrated >0.7 for high-risk calibration (calibrated max=0.431)"

key-files:
  created:
    - src/fairness/metrics.py
    - tests/gates/test_gate_3_1.py
    - data/exports/fairness_metrics.json

key-decisions:
  - "Two-MetricFrame pattern for mixed binary/proba metrics (fairlearn 0.13.0 requirement)"
  - "Uncalibrated probs for >0.7 high-risk analysis (calibrated max 0.431, 0 obs above 0.7)"
  - "7 dimensions (6 + disaggregated language) with different min_sample thresholds"
  - "Nationality dimension included but flagged (n=27) -- user requested minimal mention"
  - "FNR-FPR trade-off finding: indigenous groups have low FNR but high FPR (surveillance bias vs invisibility bias)"

patterns-established:
  - "fairlearn MetricFrame with sample_params for survey-weighted subgroup analysis"
  - "Intersectional analysis via pandas DataFrame as sensitive_features argument"

# Metrics
duration: ~8min
completed: 2026-02-08
---

# Plan 08-01: Subgroup Fairness Metrics Summary

**Comprehensive fairness audit across 7 dimensions + 3 intersections revealing FNR-FPR trade-off: model catches indigenous dropouts (FNR=0.227) but over-flags them (FPR=0.537), while missing castellano/urban dropouts (FNR=0.639) with fewer false alarms (FPR=0.160)**

## Performance

- **Duration:** ~8 min
- **Tasks:** 3 (2 auto + 1 human checkpoint)
- **Files created:** 3
- **Gate tests:** 12/12 pass, 166 total regression pass

## Accomplishments

- Fairness metrics computed for 7 dimensions (language, language_disaggregated, sex, geography, region, poverty, nationality) + 3 intersections
- All metrics survey-weighted with FACTOR07 via fairlearn MetricFrame
- Calibration-by-group computed using uncalibrated >0.7 threshold (702 test observations)
- Equalized odds gaps, predictive parity gaps, max FNR gaps per dimension
- Small sample groups correctly flagged (aimara=76, non_peruvian=27, multiple intersections <50)
- fairness_metrics.json (27.6 KB) matches M4 schema
- Gate test 3.1: 12/12 passed

## Key Equity Findings

- **FNR-FPR trade-off by language:** other_indigenous FNR=0.227/FPR=0.537 vs castellano FNR=0.639/FPR=0.160 -- classic equalized odds violation
- **Interpretation:** Indigenous students face surveillance bias (over-flagged), castellano/urban face invisibility bias (dropouts missed)
- **Geography:** Urban FNR=0.653 > Rural FNR=0.536 (gap 0.117)
- **Calibration:** Selva 28.1% vs Sierra 38.9% actual dropout among high-risk -- "high risk" means different things by region
- **Worst intersection:** other_indigenous_urban FNR=0.753 (n=89) -- indigenous speakers in urban settings most missed
- **Sex gap minimal:** FNR gap only 0.026 between male and female

## Task Commits

1. **Task 1: Implement fairness metrics pipeline** -- `319ee69` (feat)
2. **Task 2: Write gate test 3.1** -- `00a853e` (test)
3. **Task 3: Human review** -- approved with notes:
   - Selva calibration gap is a real finding
   - other_indigenous_urban should be reported with sample size caveat
   - Nationality dimension barely mentioned, excluded from most results
   - FNR-FPR trade-off framing (surveillance vs invisibility bias) for Phase 11

## Files Created/Modified

- `src/fairness/metrics.py` -- Full fairness metrics pipeline (7 dims + 3 intersections)
- `tests/gates/test_gate_3_1.py` -- 12 gate test assertions + human review print block
- `data/exports/fairness_metrics.json` -- M4-schema-compliant fairness metrics (27.6 KB)

## Decisions Made

- Two-MetricFrame pattern for binary (y_pred) vs proba (y_pred_proba) metrics
- Uncalibrated >0.7 for calibration-by-group (calibrated probs max at 0.431)
- 7 dimensions (added language_disaggregated with 50-obs threshold)
- Nationality included but flagged as unreliable (n=27)

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Phase 11 Framing Notes (from human review)

1. **Core finding:** FNR-FPR trade-off = "surveillance bias" (indigenous over-flagged) vs "invisibility bias" (castellano/urban missed). Not the typical "model disadvantages minorities" narrative.
2. **Selva calibration gap:** Worth highlighting -- "high risk" predicts 28.1% dropout in Selva vs 38.9% in Sierra.
3. **other_indigenous_urban (FNR=0.753, n=89):** Report with sample size caveat -- indigenous speakers in urban settings are the most invisible group.
4. **Nationality:** Barely mention, n=27 is too small for conclusions. Exclude from most analyses.

---
*Phase: 08-subgroup-fairness-metrics*
*Completed: 2026-02-08*
