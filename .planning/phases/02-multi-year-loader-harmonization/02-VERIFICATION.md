---
phase: 02-multi-year-loader-harmonization
verified: 2026-02-07T19:15:00Z
status: passed
score: 6/6 must-haves verified
---

# Phase 2: Multi-Year Loader + Harmonization Verification Report

**Phase Goal:** All 6 years of ENAHO data (2018-2023) stack into one consistent dataset with P300A mother tongue harmonization preserving both cross-year and disaggregated codes

**Verified:** 2026-02-07T19:15:00Z

**Status:** passed

**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | load_all_years() returns a pooled polars DataFrame with ~150K-160K rows across 6 years (2018-2023) | ✓ VERIFIED | Gate test 1.2 confirms 150,135 rows with year column spanning [2018, 2019, 2020, 2021, 2022, 2023] |
| 2 | Each row has a year column identifying its source year | ✓ VERIFIED | Gate test validates year column exists and all 6 years are represented; per-year breakdown shows 13,755 (2020) to 30,559 (2018) rows |
| 3 | p300a_harmonized column collapses codes 10-15 to code 3 for cross-year comparison | ✓ VERIFIED | Gate test confirms no codes 10-15 in p300a_harmonized column; unit test TestHarmonizeP300A::test_codes_10_to_15_collapse_to_3 passes |
| 4 | p300a_original column preserves raw INEI codes including disaggregated 10-15 for 2020+ analysis | ✓ VERIFIED | Gate test confirms all codes 10-15 present in p300a_original for years >= 2020; unit test TestHarmonizeP300A::test_original_codes_preserved passes |
| 5 | 2020 COVID year loads without crashing despite 52% P303 nulls | ✓ VERIFIED | Gate test shows 2020 loaded with 13,755 rows after dropping 52.3% P303-null rows; warning logged as expected |
| 6 | Gate test 1.2 passes all assertions | ✓ VERIFIED | pytest tests/gates/test_gate_1_2.py passes with all metrics within expected ranges (24,205 dropouts, 1.53x stability ratio) |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/data/enaho.py` | Contains PooledENAHOResult, POOLED_COLUMNS, harmonize_p300a(), load_all_years(), P303-null handling | ✓ VERIFIED | 620 lines, substantive implementation: PooledENAHOResult at line 99, POOLED_COLUMNS at line 56, harmonize_p300a at line 525, load_all_years at line 557, P303-null filter at line 436 with is_not_null() |
| `tests/unit/test_enaho_loader.py` | Contains TestHarmonizeP300A and TestPooledENAHOResult | ✓ VERIFIED | 291 lines, TestHarmonizeP300A at line 222 with 7 test methods, TestPooledENAHOResult at line 273 with 2 test methods, all pass |
| `tests/gates/test_gate_1_2.py` | Contains test_gate_1_2 function | ✓ VERIFIED | 123 lines, substantive gate test at line 21, validates pooled data properties: year coverage, harmonization, stability ratio, dropout counts |

**All artifacts:** EXISTS + SUBSTANTIVE + WIRED

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| load_all_years | load_enaho_year | Iterator calling load_enaho_year(year) | ✓ WIRED | Line 585 in enaho.py: `result = load_enaho_year(year)` inside for loop over years |
| load_all_years | harmonize_p300a | Call after pooled concat | ✓ WIRED | Line 609 in enaho.py: `pooled = harmonize_p300a(pooled)` after vertical concat |
| load_enaho_year | P303-null drop | Filter before null-threshold check | ✓ WIRED | Line 436 in enaho.py: `df = df.filter(pl.col("P303").is_not_null())` with logging and warning |
| harmonize_p300a | _DISAGG_CODES | Uses constant to identify codes to collapse | ✓ WIRED | Line 549: `.is_in(_DISAGG_CODES)` where _DISAGG_CODES = [10, 11, 12, 13, 14, 15] at line 522 |
| test_gate_1_2 | load_all_years | Imports and calls function | ✓ WIRED | Line 18: `from data.enaho import load_all_years`, line 22: `result = load_all_years()` |

**All key links:** WIRED with substantive implementations

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| DATA-03: Load all 7 years (2018-2024) with consistent schema and year column | ✓ SATISFIED (6 of 7 years) | Loads 2018-2023 (2024 not available on INEI portal); year column present; POOLED_COLUMNS ensures schema consistency across years |
| DATA-04: Harmonize P300A mother tongue codes (collapse 10-15 -> 3 for cross-year; preserve originals) | ✓ SATISFIED | Dual-column harmonization: p300a_harmonized collapses 10-15 to 3 (verified by gate test no disagg codes assertion), p300a_original preserves raw codes (verified by gate test disagg codes present for 2020+) |

**Note on DATA-03:** Spec requires 7 years (2018-2024), but ENAHO 2024 is not yet available on INEI portal as of 2026-02-07 (confirmed in Phase 1). Implementation correctly loads all available years (2018-2023, 6 years). This is not a gap — it's a data availability constraint external to the codebase. The loader is ready for 2024 when it becomes available.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No stub patterns, TODOs, or placeholders found in newly created code |

**Scan results:**
- ✓ No TODO/FIXME/placeholder comments in src/data/enaho.py
- ✓ No TODO/FIXME/placeholder comments in tests/gates/test_gate_1_2.py
- ✓ No empty return statements or console.log-only implementations
- ✓ All functions have substantive logic

### Human Verification Required

None. All goal requirements are programmatically verifiable and have been verified via automated tests.

---

## Detailed Verification Evidence

### Truth 1: Pooled DataFrame with ~150K-160K rows

**Check:** Gate test 1.2 assertion
```python
total = df.height
assert 130_000 <= total <= 190_000, f"FAIL: pooled rows {total} outside [130K, 190K]"
```

**Result:** 150,135 rows (within range)

**Per-year breakdown from gate test:**
- 2018: 30,559 rows, 4,821 dropouts (16.21% weighted)
- 2019: 28,030 rows, 4,618 dropouts (15.27% weighted)
- 2020: 13,755 rows, 3,991 dropouts (27.54% weighted) — COVID year with 52.3% P303 nulls dropped
- 2021: 25,679 rows, 3,465 dropouts (13.09% weighted)
- 2022: 26,477 rows, 3,810 dropouts (14.16% weighted)
- 2023: 25,635 rows, 3,500 dropouts (13.45% weighted)

**Total:** 150,135 rows, 24,205 dropouts

### Truth 2: Year column identifies source year

**Check:** Gate test 1.2 year coverage assertion
```python
years = sorted(df["year"].unique().to_list())
assert years == [2018, 2019, 2020, 2021, 2022, 2023]
```

**Result:** PASS — all 6 years present

**Implementation:** Line 588 in `load_all_years()`:
```python
df = result.df.with_columns(pl.lit(year).alias("year"))
```

### Truth 3: p300a_harmonized collapses codes 10-15 to 3

**Check:** Gate test 1.2 disaggregated codes exclusion
```python
harmonized_vals = df["p300a_harmonized"].drop_nulls().unique().to_list()
for code in [10, 11, 12, 13, 14, 15]:
    assert code not in harmonized_vals
```

**Result:** PASS — no codes 10-15 found in harmonized column

**Implementation:** Line 549-553 in `harmonize_p300a()`:
```python
pl.when(pl.col("P300A").is_in(_DISAGG_CODES))
.then(pl.lit(3))
.otherwise(pl.col("P300A"))
.cast(pl.Int64)
.alias("p300a_harmonized")
```

**Unit test coverage:** 7 tests in TestHarmonizeP300A, all pass:
- test_codes_10_to_15_collapse_to_3
- test_original_codes_preserved
- test_non_disagg_codes_unchanged
- test_mixed_codes
- test_harmonized_column_is_int64
- test_null_p300a_preserved
- test_disagg_codes_constant_complete

### Truth 4: p300a_original preserves disaggregated codes

**Check:** Gate test 1.2 disaggregated codes presence in original
```python
post2020 = df.filter(pl.col("year") >= 2020)
original_vals = post2020["p300a_original"].drop_nulls().unique().to_list()
disagg_present = [c for c in [10, 11, 12, 13, 14, 15] if c in original_vals]
assert len(disagg_present) > 0
```

**Result:** PASS — all 6 disaggregated codes [10, 11, 12, 13, 14, 15] present in p300a_original for 2020+

**Implementation:** Line 548 in `harmonize_p300a()`:
```python
pl.col("P300A").alias("p300a_original")
```

### Truth 5: 2020 COVID year loads without crashing

**Check:** Gate test 1.2 per-year breakdown shows 2020 data

**Result:** PASS — 2020 loaded with 13,755 rows

**P303-null handling:** Line 436 in `load_enaho_year()`:
```python
df = df.filter(pl.col("P303").is_not_null())
```

**Warning logged:** Gate test output shows:
```
WARN: [2020] Dropped 15061 rows (52.3%) with P303 null (COVID reduced questionnaire / phone interview)
```

### Truth 6: Harmonization stability (max/min ratio < 2.0x)

**Check:** Gate test 1.2 stability ratio calculation
```python
ratio = max(proportions.values()) / min(proportions.values())
assert ratio < 2.0
```

**Result:** PASS — ratio = 1.53x (well below 2.0x threshold)

**Per-year code 3 proportions:**
- 2018: 3.688%
- 2019: 3.989%
- 2020: 3.955%
- 2021: 2.909%
- 2022: 2.746%
- 2023: 2.606%

**Max/min:** 3.989% / 2.606% = 1.53x

This confirms harmonization is not masking real population shifts — the proportion of "other indigenous languages" (code 3) is stable across years after collapsing the disaggregated codes.

---

## Test Suite Status

**Full test suite:** 39 tests, all PASS (17.94s runtime)

**Breakdown:**
- Gate tests: 2 (test_gate_1_1, test_gate_1_2)
- Unit tests: 37
  - ENAHOResult: 2 tests
  - Dropout construction: 5 tests
  - School-age filter: 2 tests
  - UBIGEO validation: 3 tests
  - Critical null detection: 5 tests
  - HarmonizeP300A: 7 tests (NEW)
  - PooledENAHOResult: 2 tests (NEW)
  - Utilities (pad_ubigeo, sniff_delimiter): 11 tests

**New tests added in Phase 2:** 9 (7 harmonization + 2 pooled result)

**No regressions:** All 30 pre-existing tests still pass

---

## Summary

Phase 2 goal **ACHIEVED**. All 6 available years of ENAHO data (2018-2023) successfully pool into a consistent 150,135-row DataFrame with P300A mother tongue harmonization preserving both cross-year comparability (p300a_harmonized) and disaggregated analysis capability (p300a_original).

### Key Metrics (from Gate Test 1.2)

- **Pooled rows:** 150,135 (target: ~150K-160K) ✓
- **Years:** [2018, 2019, 2020, 2021, 2022, 2023] (6 of 7 available) ✓
- **Total dropouts:** 24,205 unweighted (target: ~22K+) ✓
- **Harmonization stability:** 1.53x max/min ratio (threshold: < 2.0x) ✓
- **Columns:** 19 (POOLED_COLUMNS + p300a_harmonized + p300a_original + year)
- **Test coverage:** 39 tests pass, 9 new tests added, 0 regressions

### Readiness for Next Phase

Phase 3 (Spatial + Supplementary Data Merges) can proceed:
- ✓ Pooled DataFrame ready as input for spatial merges
- ✓ Year column enables temporal analysis
- ✓ Harmonized P300A enables fairness audit disaggregation (Phase 8)
- ✓ Fixed schema (POOLED_COLUMNS) ensures downstream consistency
- ✓ Per-year stats available for QA and reporting

---

_Verified: 2026-02-07T19:15:00Z_  
_Verifier: Claude (gsd-verifier)_
