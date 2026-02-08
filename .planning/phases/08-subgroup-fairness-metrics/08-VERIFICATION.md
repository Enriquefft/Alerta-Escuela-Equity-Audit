---
phase: 08-subgroup-fairness-metrics
verified: 2026-02-08T15:45:00Z
status: passed
score: 5/5 must-haves verified
---

# Phase 8: Subgroup Fairness Metrics Verification Report

**Phase Goal:** Comprehensive fairness metrics are computed across all 6 protected dimensions and 3 intersections, quantifying where the model systematically fails different student populations

**Verified:** 2026-02-08T15:45:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TPR, FPR, FNR, precision, and PR-AUC computed per subgroup for all 6 dimensions with FACTOR07 survey weights | ✓ VERIFIED | fairness_metrics.json has all 5 metrics for 7 dimensions (language, language_disaggregated, sex, geography, region, poverty, nationality) with 34 total groups. Weighted n_weighted values match independent calculation (female: 3,661,180.77). |
| 2 | Calibration-by-group shows actual dropout rate among predicted high-risk (>0.7 uncalibrated) per group | ✓ VERIFIED | All 34 dimension groups have `calibration_high_risk` key. Language dimension shows castellano=0.364, quechua=0.362 actual among high-risk. Groups with <30 high-risk observations correctly report null. |
| 3 | Three intersectional analyses (language x rurality, sex x poverty, language x region) with <50 groups flagged | ✓ VERIFIED | 3 intersections present with 30 total groups. 6 groups flagged small sample: aimara_urban (n=25), aimara_costa (n=8), aimara_selva (n=2), other_indigenous_costa (n=31), other_indigenous_sierra (n=19), quechua_costa (n=34). |
| 4 | fairness_metrics.json matches M4 schema with dimensions, groups, gaps, and intersections | ✓ VERIFIED | JSON has required top-level keys: generated_at, model, threshold, threshold_type, calibration_note, test_set, n_test, n_dropouts, dimensions, intersections. All dimensions have gaps with equalized_odds_tpr, equalized_odds_fpr, predictive_parity, max_fnr_gap, max_fnr_groups. |
| 5 | Gate test 3.1 passes; FNR by language group and calibration table printed for human review | ✓ VERIFIED | 12/12 tests pass. FNR by language group shows equity gap: other_indigenous FNR=0.227/FPR=0.537 vs castellano FNR=0.639/FPR=0.160. Human approved findings with Phase 11 framing notes. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/fairness/metrics.py` | Fairness metrics pipeline (min 200 lines) | ✓ VERIFIED (795 lines) | Implements complete pipeline: data loading, 7 dimensions + 3 intersections, two-MetricFrame pattern, calibration-by-group, gap computation, JSON export. No stub patterns. |
| `tests/gates/test_gate_3_1.py` | Gate test 3.1 validation (min 80 lines) | ✓ VERIFIED (352 lines) | 12 test functions covering JSON structure, dimension completeness, sample sizes, weighted vs unweighted difference, gaps, intersections, FNR consistency. All pass. Human review print block included. |
| `data/exports/fairness_metrics.json` | M4-schema-compliant fairness metrics | ✓ VERIFIED (27.6 KB) | 7 dimensions, 3 intersections, 34 dimension groups, 30 intersection groups. Valid JSON matching M4 schema. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| metrics.py | predictions_lgbm_calibrated.parquet | polars read_parquet + filter test_2023 | ✓ WIRED | Line 389-401: loads predictions, filters to test_2023, asserts 25,635 rows. |
| metrics.py | enaho_with_features.parquet | polars LEFT JOIN on ID keys | ✓ WIRED | Line 408-429: loads features, joins on CONGLOME/VIVIENDA/HOGAR/CODPERSO/year, gets sensitive features. |
| metrics.py | fairlearn MetricFrame | Two-MetricFrame pattern with sample_params | ✓ WIRED | Lines 183-191, 272-280: separate binary (TPR/FPR/FNR/precision) and proba (PR-AUC) MetricFrame instances with nested sample_params dict. |
| metrics.py | fairness_metrics.json | json.dump with M4 schema | ✓ WIRED | Line 702: writes JSON with indent=2, default=str. Schema matches M4 structure (dimensions, intersections, gaps). |
| test_gate_3_1.py | fairness_metrics.json | json.load and structural assertions | ✓ WIRED | Line 71: fixture loads JSON. 12 test functions validate structure, completeness, metrics, gaps. |

### Requirements Coverage

| Requirement | Status | Supporting Truths |
|-------------|--------|-------------------|
| FAIR-01: Compute TPR, FPR, FNR, precision, PR-AUC per subgroup across all 6 dimensions | ✓ SATISFIED | Truth 1 |
| FAIR-02: Compute calibration per group (actual dropout rate among predicted-high-risk >0.7) | ✓ SATISFIED | Truth 2 |
| FAIR-03: Analyze 3 intersections: language x rurality, sex x poverty, language x region | ✓ SATISFIED | Truth 3 |
| FAIR-04: Flag intersectional groups with <50 unweighted observations | ✓ SATISFIED | Truth 3 |
| FAIR-05: Compute equalized odds gap and predictive parity gap per dimension | ✓ SATISFIED | Truth 4 |
| FAIR-06: Export fairness metrics to data/exports/fairness_metrics.json matching M4 schema | ✓ SATISFIED | Truth 4 |

### Anti-Patterns Found

No anti-patterns found. All files scanned for TODO/FIXME/placeholder/stub patterns — zero matches.

### Human Verification Completed

Human approval given in 08-01-SUMMARY.md (Task 3) with findings notes:

**Approved Findings:**
- FNR-FPR trade-off: other_indigenous (FNR=0.227, FPR=0.537) vs castellano (FNR=0.639, FPR=0.160) — surveillance bias vs invisibility bias
- Geography: Urban FNR=0.653 > Rural FNR=0.536 (gap 0.117)
- Calibration gap by region: Selva 28.1% vs Sierra 38.9% actual dropout among high-risk — "high risk" means different things
- Worst intersection: other_indigenous_urban FNR=0.753 (n=89) — indigenous speakers in urban settings most missed
- Sex gap minimal: FNR gap only 0.026

**Human Notes for Phase 11:**
- Use "surveillance bias" (indigenous over-flagged) vs "invisibility bias" (castellano/urban missed) framing
- Selva calibration gap is a real finding worth highlighting
- Report other_indigenous_urban with sample size caveat (n=89)
- Barely mention nationality dimension (n=27 too small)

### Key Equity Findings Verified

**1. FNR-FPR Trade-off by Language (Core Finding):**
- other_indigenous: FNR=0.227, FPR=0.537
- castellano: FNR=0.639, FPR=0.160
- Interpretation: Indigenous students face surveillance bias (over-flagged as high-risk), while castellano/urban students face invisibility bias (model misses actual dropouts).

**2. Geography Gap:**
- Urban: FNR=0.653
- Rural: FNR=0.536
- Gap: 0.117

**3. Calibration by Region:**
- Selva: 28.1% actual dropout among 188 predicted high-risk
- Sierra: 38.9% actual dropout among 348 predicted high-risk
- Costa: 37.8% actual dropout among 166 predicted high-risk
- Interpretation: Model's "high risk" label means different things in different regions.

**4. Intersectional Analysis:**
- other_indigenous_urban: FNR=0.753, n=89 (worst)
- aimara_urban: n=25 (flagged small sample)
- language_x_rural: 8 groups, 1 flagged
- sex_x_poverty: 10 groups, 0 flagged
- language_x_region: 12 groups, 5 flagged

**5. Sex Gap:**
- Female: FNR=0.618
- Male: FNR=0.644
- Gap: 0.026 (minimal)

### Technical Implementation Quality

**Two-MetricFrame Pattern:**
- Correctly separates binary metrics (y_pred) from probability metrics (y_pred_proba)
- sample_params nested dict correctly passes weights to each metric
- Resolves fairlearn 0.13.0 limitation

**Survey Weighting:**
- FACTOR07 weights correctly applied via sample_params
- Weighted n_weighted values independently verified (female: 3,661,180.77)
- Gate test 3.1 verifies weighted differs from unweighted

**Calibration-by-group:**
- Uses uncalibrated probabilities (>0.7) because calibrated max=0.431
- Documented in JSON metadata
- min_high_risk=30 threshold correctly filters unreliable groups

**Small Sample Flagging:**
- Primary dimensions: n_unweighted < 100
- Disaggregated language: n_unweighted < 50
- Intersections: n_unweighted < 50
- 6 intersection groups correctly flagged

**M4 Schema Compliance:**
- All required top-level keys present
- dimensions → groups → metrics + calibration_high_risk + flagged_small_sample
- dimensions → gaps → 5 gap types
- intersections → groups → metrics + flagged_small_sample
- intersections → gaps

### Statistics

- **Total dimension groups:** 34
- **Total intersection groups:** 30
- **Intersection groups flagged:** 6
- **Gate tests:** 12/12 passed
- **Total regression tests:** 166 passed (includes all prior phases)
- **Duration:** ~8 minutes
- **JSON file size:** 27.6 KB

---

_Verified: 2026-02-08T15:45:00Z_
_Verifier: Claude (gsd-verifier)_
