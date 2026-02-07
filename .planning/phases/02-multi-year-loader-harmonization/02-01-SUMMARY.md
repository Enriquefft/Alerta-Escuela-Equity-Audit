---
phase: 02-multi-year-loader-harmonization
plan: 01
subsystem: data
tags: [polars, enaho, multi-year, harmonization, p300a, mother-tongue, pooling]

# Dependency graph
requires:
  - phase: 01-enaho-single-year-loader
    provides: "load_enaho_year(), ENAHOResult, Module 200/300 loading, dropout construction"
provides:
  - "load_all_years() pools 2018-2023 into PooledENAHOResult (150K rows)"
  - "harmonize_p300a() collapses indigenous codes 10-15 to code 3 for cross-year analysis"
  - "P303-null row dropping handles COVID reduced questionnaire years (2020/2021)"
  - "POOLED_COLUMNS constant defines fixed schema for downstream phases"
  - "Gate test 1.2 validates pooled data integrity"
affects: [03-spatial-merge, 04-feature-engineering, 05-modeling, 08-fairness-audit]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PooledENAHOResult dataclass for multi-year container"
    - "POOLED_COLUMNS constant for schema-consistent vertical concat"
    - "harmonize_p300a dual-column pattern (original + harmonized)"

key-files:
  created:
    - "tests/gates/test_gate_1_2.py"
  modified:
    - "src/data/enaho.py"
    - "tests/unit/test_enaho_loader.py"

key-decisions:
  - "P303-null rows dropped before null-fill logic (affects 2020 ~52.3%, 2021 ~4.6%)"
  - "Harmonization preserves both original and collapsed codes via dual columns"
  - "Fixed _find_module_file to prefer exact module match (300.dta over 300a.dta)"

patterns-established:
  - "Dual-column harmonization: p300a_original preserves raw codes, p300a_harmonized collapses for cross-year"
  - "PooledENAHOResult container with per_year_stats and prefixed warnings"
  - "Exact module file matching before wildcard fallback"

# Metrics
duration: 8min
completed: 2026-02-07
---

# Phase 2 Plan 1: Multi-Year Loader + Harmonization Summary

**6-year ENAHO pooling (150K rows) with P300A mother tongue harmonization (codes 10-15 collapsed to 3) and P303-null COVID year handling**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-07T23:39:31Z
- **Completed:** 2026-02-07T23:47:11Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Pooled 6 years (2018-2023) of ENAHO survey data into 150,135-row DataFrame with consistent schema
- P300A harmonization: disaggregated indigenous language codes 10-15 (introduced 2020+) collapsed to code 3 for cross-year comparability, with original codes preserved
- P303-null handling enables COVID years (2020: 52.3% drop, 2021: 4.6% drop) to load without crashing
- Gate test 1.2 validates all pooled data properties: 6 years, 24,205 dropouts, 1.53x stability ratio

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend enaho.py with P303-null handling, multi-year loader, and P300A harmonization** - `acfd9c8` (feat)
2. **Task 2: Unit tests for harmonization + multi-year loading, and gate test 1.2** - `5fbb432` (test)

## Files Created/Modified

- `src/data/enaho.py` - Added PooledENAHOResult, POOLED_COLUMNS, harmonize_p300a(), load_all_years(), P303-null drop in load_enaho_year(), fixed _find_module_file()
- `tests/unit/test_enaho_loader.py` - Added 9 unit tests: 7 for harmonize_p300a, 2 for PooledENAHOResult
- `tests/gates/test_gate_1_2.py` - Gate test validating 6-year pooled data (row counts, harmonization, stability ratio)

## Decisions Made

- **P303-null rows dropped (not filled):** COVID reduced questionnaire years have large fractions of P303 nulls (2020: 52.3%). These rows lack prior enrollment info and cannot contribute to dropout analysis. Dropping before the existing null-fill threshold check avoids the ValueError that would otherwise block loading.
- **Dual-column harmonization:** `p300a_original` preserves raw INEI codes (including disaggregated 10-15 for 2020+ analysis); `p300a_harmonized` collapses 10-15 to code 3 for cross-year comparability. Both downstream analysis paths are supported.
- **Exact module file matching:** Fixed `_find_module_file()` to prefer exact match (`enaho01a-YEAR-300.dta`) over wildcard match (`enaho01a-YEAR-300*.dta`) which was incorrectly loading `300a.dta` sub-module for 2021 (910 rows, missing P303/P306).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed _find_module_file() matching wrong file for 2021**
- **Found during:** Task 2 (gate test 1.2 execution)
- **Issue:** The glob pattern `{prefix}{sep}{year}{sep}{module}*.{ext}` with module="300" matched both `300.dta` (correct, 109K rows) and `300a.dta` (wrong sub-module, 910 rows). For 2021, `300a.dta` was returned first, causing a ColumnNotFoundError on P303.
- **Fix:** Changed `_find_module_file()` to first try exact match (`{module}.{ext}`) before falling back to wildcard (`{module}*.{ext}`). Also tightened case-insensitive fallback to prefer exact module-number boundary matches.
- **Files modified:** `src/data/enaho.py`
- **Verification:** All 6 years now load correct module files; 39 tests pass including gate 1.2
- **Committed in:** `5fbb432` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix was essential for correct 2021 data loading. No scope creep.

## Issues Encountered

- 2020 COVID year drops 52.3% of school-age rows due to P303 nulls, resulting in only 13,755 rows for that year. This is expected behavior (reduced questionnaire) and is documented in warnings.
- 2020 weighted dropout rate (27.54%) is significantly higher than other years (13-16%), reflecting genuine COVID impact on school enrollment.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Pooled DataFrame ready as input for Phase 3 (spatial merges) and Phase 4 (feature engineering)
- Per-year stats available for longitudinal analysis
- Harmonization columns ready for fairness audit (Phase 8) disaggregated analysis
- All 39 tests passing (37 unit + 2 gate)

### Key Baselines from Gate Test 1.2

| Year | Rows | Dropouts | Weighted Rate |
|------|------|----------|---------------|
| 2018 | 30,559 | 4,821 | 16.21% |
| 2019 | 28,030 | 4,618 | 15.27% |
| 2020 | 13,755 | 3,991 | 27.54% |
| 2021 | 25,679 | 3,465 | 13.09% |
| 2022 | 26,477 | 3,810 | 14.16% |
| 2023 | 25,635 | 3,500 | 13.45% |
| **Total** | **150,135** | **24,205** | -- |

Harmonization stability ratio: 1.53x (well within 2.0x threshold)

## Self-Check: PASSED

---
*Phase: 02-multi-year-loader-harmonization*
*Completed: 2026-02-07*
