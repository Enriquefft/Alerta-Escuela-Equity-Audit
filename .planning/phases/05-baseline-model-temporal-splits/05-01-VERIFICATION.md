---
phase: 05-baseline-model-temporal-splits
plan: 01
verified: 2026-02-08T11:15:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 5 Plan 1: Baseline Model + Temporal Splits Verification Report

**Phase Goal:** Temporal split discipline is established and a logistic regression baseline validates the modeling pipeline with survey-weighted evaluation, setting patterns for gradient boosting phases

**Verified:** 2026-02-08T11:15:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Temporal splits have zero year overlap: train=2018-2021, validate=2022, test=2023 | ✓ VERIFIED | Gate test passes; train years={2018,2019,2020,2021}, val={2022}, test={2023}; no overlap; 98,023 + 26,477 + 25,635 = 150,135 rows |
| 2 | Logistic regression converges and achieves PR-AUC > 0.14 on validation | ✓ VERIFIED | Converged in 307 iterations; PR-AUC (val, weighted) = 0.2103 > 0.14 |
| 3 | Weighted metrics differ from unweighted metrics (FACTOR07 actually applied) | ✓ VERIFIED | PR-AUC diff = 0.0026 (weighted=0.2103, unweighted=0.2077); gate test asserts difference > 0.001 |
| 4 | model_results.json exists with logistic_regression entry including validate_2022 metrics and threshold analysis at 0.3/0.4/0.5/0.6/0.7 | ✓ VERIFIED | File exists (11 KB); has logistic_regression key with validate_2022 + test_2023 metrics; 6 thresholds (0.3/0.4/0.5/0.6/0.7 + optimal=0.5168) |
| 5 | LR coefficients show sensible signs: poverty increases risk, rural increases risk, indigenous languages increase risk | ✓ VERIFIED | All indigenous language coeffs positive (Quechua OR=1.60, Aimara OR=1.41, Other=2.20); poverty_index_z positive (OR=1.29); age positive (OR=1.13); poverty_quintile negative due to multicollinearity (documented); rural near-zero due to spatial correlation (documented) |
| 6 | Gate test 2.1 passes all assertions | ✓ VERIFIED | 11/11 tests pass in 1.23s |

**Score:** 6/6 truths verified (100%)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/baseline.py` | Temporal splits, LR training, evaluation, threshold analysis, JSON export | ✓ VERIFIED | 893 lines; exports create_temporal_splits, train_logistic_regression, compute_metrics, run_baseline_pipeline; imports MODEL_FEATURES from data.features; uses LogisticRegression from sklearn; uses sm.GLM for coefficient inference; reads enaho_with_features.parquet |
| `data/exports/model_results.json` | Model metrics, coefficients, threshold analysis in M4 schema | ✓ VERIFIED | 11 KB; has logistic_regression entry with metadata, metrics (validate_2022 + test_2023, both weighted/unweighted), threshold_analysis (6 entries), coefficients (26 entries) |
| `data/processed/predictions_lr.parquet` | Per-row validation and test set predictions for Phase 8 fairness and Phase 9 SHAP | ✓ VERIFIED | 598 KB; 52,112 rows (val + test); has ID columns + prob_dropout (range 0.238-0.874) + pred_dropout + model + threshold + split |
| `data/processed/model_lr.joblib` | Persisted sklearn LogisticRegression model | ✓ VERIFIED | 1.1 KB; loads successfully; has predict_proba method; n_features_in_=25 |
| `data/exports/figures/pr_curve_lr.png` | Precision-recall curve visualization with threshold markers | ✓ VERIFIED | 73 KB; valid PNG (1500x1050 RGBA); gate test confirms >10KB |
| `tests/gates/test_gate_2_1.py` | Gate test for baseline model validation | ✓ VERIFIED | 445 lines; 11 test functions covering splits, convergence, PR-AUC, weighted/unweighted diff, threshold analysis, coefficients, predictions, model persistence, PR curve, test metrics |

**All 6 artifacts verified:** Exist, substantive (adequate length/content), and wired (connected to system)

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `src/models/baseline.py` | `data/processed/enaho_with_features.parquet` | polars read_parquet | ✓ WIRED | Line 680: `df = pl.read_parquet(parquet_path)` where parquet_path = "data/processed/enaho_with_features.parquet" |
| `src/models/baseline.py` | `src/data/features.py` | MODEL_FEATURES import | ✓ WIRED | Line 42: `from data.features import MODEL_FEATURES` |
| `src/models/baseline.py` | `sklearn.linear_model` | LogisticRegression training | ✓ WIRED | Line 27 import + Line 217-225 instantiation/fit with sample_weight=w_train |
| `src/models/baseline.py` | `statsmodels` | GLM(Binomial) for coefficient inference | ✓ WIRED | Line 39 import + Line 249: `glm = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w)` |
| `tests/gates/test_gate_2_1.py` | `data/exports/model_results.json` | json.load validation | ✓ WIRED | Lines 42-45: fixture loads JSON; all tests reference model_results fixture |

**All 5 key links verified:** Artifacts are connected and data flows correctly

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **MODL-01**: Create temporal splits with no overlap | ✓ SATISFIED | Gate test confirms train=2018-2021, val=2022, test=2023; zero overlap; correct split sizes |
| **MODL-02**: Train LR with class_weight='balanced' and survey-weighted eval | ✓ SATISFIED | model_results.json metadata shows class_weight='balanced'; gate test confirms weighted != unweighted metrics |
| **MODL-06**: Tune threshold per model; report at 0.3/0.4/0.5/0.6/0.7 | ✓ SATISFIED | threshold_analysis has 6 entries (5 fixed + optimal=0.5168) with weighted/unweighted F1/precision/recall |
| **MODL-08**: Export model results to model_results.json matching M4 schema | ✓ SATISFIED | File exists with logistic_regression entry; includes metadata, metrics, threshold_analysis, coefficients |
| **MODL-09**: All metrics survey-weighted via FACTOR07; also compute unweighted | ✓ SATISFIED | All metrics have weighted + unweighted versions; gate test asserts they differ by > 0.001 |

**All 5 requirements satisfied.** Note: ROADMAP originally specified train max=2022, validate=2023, test=2024, but implementation uses 2018-2021/2022/2023 due to ENAHO 2024 unavailability. This shift is documented in:
- Phase 05-CONTEXT.md line 17
- model_results.json metadata.year_shift_note
- SUMMARY.md (Phase 2 lessons learned, Phase 5 decisions)

The temporal split requirement is satisfied with the shifted years.

### Anti-Patterns Found

**None found.** Scanned src/models/baseline.py and tests/gates/test_gate_2_1.py for:
- TODO/FIXME/XXX/HACK comments: None
- Placeholder content: None
- Empty implementations (return null/{}): None
- Console.log only implementations: None

Both files are fully implemented with substantive logic.

### Human Verification Required

**None required.** The plan included a human-verify checkpoint (Task 3) for reviewing LR coefficients. According to the SUMMARY.md:
- Task 3 was marked as "approved (no commit)"
- Human reviewer accepted the coefficient signs, including:
  - poverty_quintile negative sign (multicollinearity with poverty_index_z, which is correctly positive)
  - rural near-zero (effect absorbed by correlated spatial features)
  - All indigenous language features positive (Quechua OR=1.60, Aimara OR=1.41, Other indigenous OR=2.20)

The human gate passed, and all structural checks are programmable (confirmed above).

---

## Verification Details

### Artifact-Level Checks

**src/models/baseline.py (893 lines)**

Level 1 (Exists): PASS
Level 2 (Substantive): PASS
- Line count: 893 lines (exceeds 15-line minimum for components)
- No stub patterns found (0 TODO/FIXME/placeholder/not implemented)
- Exports 4 functions: create_temporal_splits, train_logistic_regression, compute_metrics, run_baseline_pipeline

Level 3 (Wired): PASS
- Imported by tests/gates/test_gate_2_1.py (indirectly via baseline.py execution)
- Reads from data/processed/enaho_with_features.parquet
- Imports MODEL_FEATURES from data.features
- Uses sklearn.linear_model.LogisticRegression
- Uses statsmodels.api.GLM

**data/exports/model_results.json (11 KB)**

Level 1 (Exists): PASS
Level 2 (Substantive): PASS
- Valid JSON with logistic_regression top-level key
- Has metadata (16 fields), metrics (validate_2022 + test_2023), threshold_analysis (6 entries), coefficients (26 entries)
- All required fields present

Level 3 (Wired): PASS
- Read by tests/gates/test_gate_2_1.py (fixture at lines 42-45)
- Will be read by Phase 6 (LightGBM/XGBoost) to add more model entries
- Will be read by Phase 8 (fairness) for baseline model metrics

**data/processed/predictions_lr.parquet (598 KB)**

Level 1 (Exists): PASS
Level 2 (Substantive): PASS
- 52,112 rows (val=26,477 + test=25,635)
- 13 columns including ID columns + prob_dropout + pred_dropout + model + threshold + split
- prob_dropout range 0.238-0.874 (realistic probabilities)

Level 3 (Wired): PASS
- Read by tests/gates/test_gate_2_1.py
- Will be read by Phase 8 (subgroup fairness analysis)
- Will be read by Phase 9 (SHAP interpretability)

**data/processed/model_lr.joblib (1.1 KB)**

Level 1 (Exists): PASS
Level 2 (Substantive): PASS
- Loadable sklearn LogisticRegression model
- Has predict_proba method
- Has n_features_in_=25

Level 3 (Wired): PASS
- Read by tests/gates/test_gate_2_1.py
- Will be read by Phase 7 (model comparison)

**data/exports/figures/pr_curve_lr.png (73 KB)**

Level 1 (Exists): PASS
Level 2 (Substantive): PASS
- Valid PNG image (1500x1050 pixels, RGBA)
- File size 73 KB (exceeds 10 KB threshold, not empty)

Level 3 (Wired): PASS
- Checked by tests/gates/test_gate_2_1.py
- Will be included in final exports for Phase 11

**tests/gates/test_gate_2_1.py (445 lines)**

Level 1 (Exists): PASS
Level 2 (Substantive): PASS
- 445 lines (exceeds 10-line minimum for tests)
- 11 test functions covering all aspects of baseline pipeline
- No stub patterns found

Level 3 (Wired): PASS
- Imports from data.features (MODEL_FEATURES)
- Reads enaho_with_features.parquet
- Reads model_results.json
- Reads predictions_lr.parquet
- Loads model_lr.joblib
- Checks pr_curve_lr.png

### Key Metrics Summary

**Temporal Splits:**
- Train: 98,023 rows (2018-2021)
- Validate: 26,477 rows (2022)
- Test: 25,635 rows (2023)
- Total: 150,135 rows
- Zero overlap: ✓

**Model Performance (Validation 2022):**
- PR-AUC (weighted): 0.2103 (50.2% above 0.14 baseline)
- PR-AUC (unweighted): 0.2077
- ROC-AUC (weighted): 0.5913
- F1 (weighted, optimal threshold): 0.2877
- Convergence: True (307 iterations)
- Optimal threshold: 0.5168

**Model Performance (Test 2023):**
- PR-AUC (weighted): 0.1927
- PR-AUC (unweighted): 0.1951

**Threshold Analysis:** 6 entries (0.3, 0.4, 0.5, 0.6, 0.7, optimal=0.5168)

**Coefficients:** 26 entries (intercept + 25 features)
- Indigenous languages all positive: Quechua OR=1.60, Aimara OR=1.41, Other=2.20
- Age positive: OR=1.13
- poverty_index_z positive: OR=1.29
- poverty_quintile negative (multicollinearity): OR=0.81 (documented)
- rural near-zero: OR=0.99 (spatial correlation documented)

### Overall Assessment

**Status: PASSED**

All must-haves verified:
1. ✓ Temporal splits correct (zero overlap)
2. ✓ LR converges and achieves PR-AUC > 0.14
3. ✓ Weighted != unweighted metrics
4. ✓ model_results.json exists with correct structure
5. ✓ LR coefficients sensible (human-approved)
6. ✓ Gate test 2.1 passes (11/11 tests)

All artifacts exist, are substantive, and are wired correctly.

All 5 requirements (MODL-01, MODL-02, MODL-06, MODL-08, MODL-09) satisfied.

No anti-patterns found.

Human verification completed (coefficients approved).

**Phase goal achieved.** Temporal split discipline established, logistic regression baseline validates the modeling pipeline with survey-weighted evaluation. Patterns set for gradient boosting phases (6-7).

---

_Verified: 2026-02-08T11:15:00Z_
_Verifier: Claude (gsd-verifier)_
