---
phase: 28-fairness-re-analysis
plan: 02
subsystem: fairness
tags: [fnr, fairness, cross-architecture, comparison, invisibility-bias]

# Dependency graph
requires:
  - phase: 28-01
    provides: fairness_metrics_v1.json (old 25-feature) and fairness_metrics.json (new 31-feature) plus per-model variants
provides:
  - fairness_comparison.json with before/after FNR, scenario classification, cross-architecture consistency
  - 4 new gate tests for FAIR-02 and FAIR-03 validation
affects: [29-paper-update]

# Tech tracking
tech-stack:
  added: []
  patterns: [scenario-classification-thresholds, cross-architecture-rank-order-analysis]

key-files:
  created:
    - data/exports/fairness_comparison.json
  modified:
    - tests/gates/test_gate_3_1.py

key-decisions:
  - "Castellano FNR disparity classified as 'persist' -- gap 0.6531 > 0.20 with rank order preserved across all 5 models"
  - "Cross-architecture FNR rank order identical in all 5 models: castellano > quechua > other_indigenous"
  - "Castellano FNR slightly increased from 0.633 to 0.663 (+4.7%) with 31-feature model"

patterns-established:
  - "Scenario classification: persist (gap > 0.20 + rank preserved), narrow (gap shrinks > 20%), disappear (gap < 0.05 or reversal)"

requirements-completed: [FAIR-02, FAIR-03]

# Metrics
duration: 3min
completed: 2026-03-01
---

# Phase 28 Plan 02: Fairness Comparison Export Summary

**Before/after FNR comparison showing castellano invisibility bias persists (FNR 0.633->0.663, gap 0.65) with identical rank order across all 5 model architectures**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-01T21:45:12Z
- **Completed:** 2026-03-01T21:48:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Built fairness_comparison.json with before/after FNR for all language groups (v1 25-feature vs v2 31-feature)
- Classified FNR disparity scenario as "persist" -- max gap 0.6531 > 0.20, rank order preserved
- Cross-architecture analysis confirms castellano is worst FNR group in all 5 models (LightGBM, LR, RF, MLP, XGBoost) with identical rank order
- Added 4 gate tests for FAIR-02 and FAIR-03; all 24 gate_3_1 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Build fairness_comparison.json** - `1ecf518` (feat)
2. **Task 2: Add gate tests for FAIR-02 and FAIR-03** - `2e09270` (test)

## Files Created/Modified
- `data/exports/fairness_comparison.json` - Before/after FNR comparison with scenario classification and cross-architecture consistency analysis
- `tests/gates/test_gate_3_1.py` - 4 new tests validating comparison export and cross-architecture checks (24 total)

## Decisions Made
- Castellano FNR disparity classified as "persist" -- the 31-feature model did not reduce the gap (actually increased castellano FNR slightly from 0.633 to 0.663)
- Cross-architecture FNR rank order is identical across all 5 models: castellano > quechua > other_indigenous
- Max FNR gap includes flagged groups (unknown, foreign, aimara) in JSON field but scenario classification uses non-flagged group rank order

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- fairness_comparison.json provides the core analytical output for Phase 28 and paper update
- Key finding: castellano invisibility bias persists despite model improvement, confirmed across all architectures
- One pre-existing test failure in test_gate_3_2 (feature_names_match) unrelated to this plan

---
*Phase: 28-fairness-re-analysis*
*Completed: 2026-03-01*
