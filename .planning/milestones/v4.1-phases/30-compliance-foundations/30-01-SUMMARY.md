---
phase: 30-compliance-foundations
plan: 01
subsystem: fairness
tags: [threshold-sweep, fnr, fairness, latex, jedm]

# Dependency graph
requires:
  - phase: 28-fairness-reanalysis
    provides: Fairness metrics, v1/v2 Platt parameters, calibrated predictions
provides:
  - Threshold sweep analysis verifying FNR rank-order across classification thresholds
  - LaTeX Table 13 (appendix) showing FNR by language group at each threshold
  - Updated Methods Section 4.4 with threshold-invariance finding
  - JEDM-compliant AI declaration with section-specific Claude Code (Anthropic) usage
affects: [31-framing-strengthening, 32-final-polish]

# Tech tracking
tech-stack:
  added: []
  patterns: [platt-calibration-sweep, rank-order-verification]

key-files:
  created:
    - src/fairness/threshold_sweep.py
    - data/exports/threshold_sweep.json
    - paper/tables/table_13_threshold_sweep.tex
  modified:
    - paper/main.tex

key-decisions:
  - "Castellano has highest FNR at all operationally meaningful thresholds (0.05-0.20), not fully invariant across all thresholds"
  - "At thresholds 0.25+ other_indigenous overtakes castellano due to probability ceiling compression -- reported honestly"
  - "Both v1 and v2 calibration parameters applied to same raw probs (v2 model) since v1 model no longer exists"

patterns-established:
  - "Threshold sweep: sweep calibrated probs across 0.05-0.95, exclude degenerate range above max calibrated prob"

requirements-completed: [COMP-01, COMP-02]

# Metrics
duration: 7min
completed: 2026-03-02
---

# Phase 30 Plan 01: Compliance Foundations Summary

**Threshold sweep confirms castellano highest FNR at all operational thresholds (0.05-0.20); JEDM AI declaration updated with section-specific Claude Code usage**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-02T03:48:47Z
- **Completed:** 2026-03-02T03:55:47Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Verified castellano speakers have highest FNR at all operationally meaningful thresholds (0.05-0.20), including optimal threshold 0.185
- Generated threshold sweep JSON with 19+ thresholds x 4 language groups x 2 calibration models
- Created LaTeX Table 13 for appendix with FNR rank superscripts and optimal threshold row
- Rewrote AI declaration to JEDM section-specific format naming Sections 3, 4, 5, 6, and Appendix A

## Task Commits

Each task was committed atomically:

1. **Task 1: Create threshold sweep script and generate outputs** - `5b365c0` (feat)
2. **Task 2: Update paper Methods 4.4 and AI declaration** - `cfbd527` (feat)

## Files Created/Modified
- `src/fairness/threshold_sweep.py` - Threshold sweep analysis (435 lines), sweeps FNR across 19 thresholds for v1/v2 calibrations
- `data/exports/threshold_sweep.json` - FNR by language group at each threshold, rank invariance metadata (17.6 KB)
- `paper/tables/table_13_threshold_sweep.tex` - LaTeX appendix table with FNR ranks across thresholds 0.05-0.40
- `paper/main.tex` - Added threshold-invariance paragraph to Section 4.4, JEDM AI declaration, Table 13 in appendix

## Decisions Made
- **Rank order is NOT fully invariant** -- at thresholds 0.25+, other_indigenous overtakes castellano as the probability ceiling compresses discriminative range. This is reported honestly in both the JSON export and the paper text.
- **v1/v2 comparison uses same raw probabilities** with different Platt parameters (v1: A=-6.236, B=4.442; v2: A=-8.156, B=5.069) since the original v1 25-feature model no longer exists after Phase 27 retraining.
- **Table 13 displays 0.05-0.40 range** only, omitting degenerate thresholds above max calibrated probability (0.48) where all groups have FNR=1.0.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Threshold-invariance finding documented; ready for framing strengthening (Phase 31)
- Paper compiles cleanly at 24 pages with all 13 tables

---
*Phase: 30-compliance-foundations*
*Completed: 2026-03-02*
