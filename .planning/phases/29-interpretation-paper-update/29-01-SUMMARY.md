---
phase: 29-interpretation-paper-update
plan: 01
subsystem: paper
tags: [latex, tables, prauc, fnr, robustness]

# Dependency graph
requires:
  - phase: 27-model-retraining
    provides: "v2 (31-feature) model results"
  - phase: 28-fairness-reanalysis
    provides: "v2 fairness metrics and cross-architecture FNR comparison"
provides:
  - "Table 04 with v1/v2 PR-AUC comparison columns"
  - "Table 10 with v1/v2 FNR rows per language group"
  - "Table 12 dedicated before/after PR-AUC comparison"
affects: [29-02, 29-03, paper-narrative]

# Tech tracking
tech-stack:
  added: []
  patterns: ["multirow LaTeX table layout for v1/v2 comparison"]

key-files:
  created:
    - "paper/tables/table_12_prauc_comparison.tex"
  modified:
    - "paper/tables/table_04_models.tex"
    - "paper/tables/table_10_crossmodel_fnr.tex"

key-decisions:
  - "Used multirow layout in Table 10 for v1/v2 FNR comparison (clearer than 10-column approach)"
  - "Aimara shown v1 only in Table 10 due to small-sample instability"
  - "Table 04 keeps test/Brier/BSS as v1 values since v1 is primary model"

patterns-established:
  - "v1/v2 comparison pattern: show both feature-set results side-by-side in tables"

requirements-completed: [PAPER-02]

# Metrics
duration: 1min
completed: 2026-03-01
---

# Phase 29 Plan 01: Table Updates Summary

**Three paper tables updated/created with v1 (25f) vs v2 (31f) PR-AUC and FNR comparisons across all 5 models**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-01T22:37:20Z
- **Completed:** 2026-03-01T22:38:24Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Table 04 now shows both 25-feature and 31-feature PR-AUC columns for all models
- Table 10 restructured with multirow layout showing v1/v2 FNR per language group across 5 models
- New Table 12 provides dedicated before/after PR-AUC comparison with absolute and percentage changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Update Table 04 and create Table 12** - `600f931` (feat)
2. **Task 2: Update Table 10 with v2 FNR** - `34c9c5c` (feat)

## Files Created/Modified
- `paper/tables/table_04_models.tex` - Model comparison with v2 PR-AUC column added
- `paper/tables/table_10_crossmodel_fnr.tex` - Cross-model FNR with v1/v2 rows per language group
- `paper/tables/table_12_prauc_comparison.tex` - New before/after PR-AUC comparison table

## Decisions Made
- Used multirow layout (Option A) for Table 10 rather than doubling columns to 10 — more readable
- Aimara excluded from v2 rows in Table 10 due to small-sample instability (n=76)
- Table 04 test metrics and Brier/BSS remain v1 values since v1 is the primary model; v2 column is robustness check

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All three table .tex files ready for inclusion in main.tex
- Tables provide data backbone for narrative text updates in 29-02

---
*Phase: 29-interpretation-paper-update*
*Completed: 2026-03-01*
