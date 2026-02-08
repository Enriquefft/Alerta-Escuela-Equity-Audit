---
phase: 07-calibration-onnx-export-final-test
verified: 2026-02-08T21:30:00Z
status: passed
score: 5/5 must-haves verified
---

# Phase 7: Calibration + ONNX Export + Final Test Verification Report

**Phase Goal:** The best model is calibrated, exported to ONNX for browser inference, and evaluated on the 2023 test set exactly once -- the only time test data is touched
**Verified:** 2026-02-08T21:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Calibrated LightGBM has lower Brier score than uncalibrated on validation set | ✓ VERIFIED | Brier: 0.2115 → 0.1156 (45.3% reduction); assertion in calibration.py line 136-139 |
| 2 | ONNX file predictions match Python model within 1e-4 on 100 random samples | ✓ VERIFIED | Max abs diff: 1.03e-07 (< 1e-4); ONNX validation in calibration.py lines 177-197 |
| 3 | Test set (2023) PR-AUC is within 0.07 of validation PR-AUC | ✓ VERIFIED | Gap: 0.0233 (val: 0.2611, test: 0.2378); check in calibration.py lines 246-250 |
| 4 | model_results.json contains test_2023_final and test_2023_calibrated entries | ✓ VERIFIED | Both keys exist: lightgbm.test_2023_final + test_2023_calibrated top-level |
| 5 | Comparison table with Alerta Escuela published metrics is printed for human review | ✓ VERIFIED | Table printed in calibration.py lines 283-310; includes ROC-AUC, FNR, fairness comparison |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/calibration.py` | Calibration + ONNX export + final test pipeline | ✓ VERIFIED | 466 lines, all 13 steps implemented, no stubs |
| `tests/gates/test_gate_2_3.py` | Gate test 2.3 assertions | ✓ VERIFIED | 334 lines, 11 tests, all PASSED |
| `data/exports/onnx/lightgbm_dropout.onnx` | ONNX model for browser inference | ✓ VERIFIED | 0.10 MB (< 50 MB), predictions match within 1e-7 |
| `data/exports/figures/calibration_plot.png` | Calibration reliability diagram | ✓ VERIFIED | 87 KB, exists and > 1 KB |
| `data/processed/model_lgbm_calibrated.joblib` | Persisted calibrated model | ✓ VERIFIED | 177 KB, loads successfully |
| `data/processed/predictions_lgbm_calibrated.parquet` | Calibrated predictions | ✓ VERIFIED | 726 KB, expected columns present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `src/models/calibration.py` | `data/processed/model_lgbm.joblib` | `joblib.load` | ✓ WIRED | Line 103: `lgbm = joblib.load(lgbm_model_path)` |
| `src/models/calibration.py` | `sklearn.frozen.FrozenEstimator` | FrozenEstimator wrapping | ✓ WIRED | Line 129: `frozen = FrozenEstimator(lgbm)` |
| `src/models/calibration.py` | `onnxmltools.convert.convert_lightgbm` | ONNX conversion | ✓ WIRED | Line 163: `onnx_model = convert_lightgbm(lgbm, ...)` |
| `src/models/calibration.py` | `data/exports/model_results.json` | JSON merge | ✓ WIRED | Lines 365-431: loads existing, adds entries, writes back |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| MODL-05: Calibrate best model on validation set | ✓ SATISFIED | Platt sigmoid calibration, 45.3% Brier reduction |
| MODL-07: Evaluate calibrated model on test set exactly once | ✓ SATISFIED | Test evaluation at Step 7 only, metrics in model_results.json |
| EXPO-01: Export LightGBM as ONNX with validation | ✓ SATISFIED | ONNX at data/exports/onnx/lightgbm_dropout.onnx, max diff 1.03e-07 < 1e-4 |

### Anti-Patterns Found

No anti-patterns detected.

**Scan Results:**
- No TODO/FIXME/XXX/HACK comments
- No placeholder content
- No empty implementations (return null/{}/"[]")
- No console.log-only handlers
- All assertions present and enforcing success criteria

### Human Verification Required

None. All success criteria are programmatically verified. Human approval already granted per summary frontmatter (Task 3 approved).

---

## Detailed Verification Results

### Level 1: Existence
All 6 required files exist:
- `src/models/calibration.py` — 466 lines
- `tests/gates/test_gate_2_3.py` — 334 lines
- `data/exports/onnx/lightgbm_dropout.onnx` — 0.10 MB
- `data/exports/figures/calibration_plot.png` — 87 KB
- `data/processed/model_lgbm_calibrated.joblib` — 177 KB
- `data/processed/predictions_lgbm_calibrated.parquet` — 726 KB

### Level 2: Substantive
**calibration.py (466 lines):**
- Main pipeline function `run_calibration_pipeline()` at line 67
- 13 discrete steps implemented (Step 1-13 documented in docstring)
- Key logic verified:
  - Brier improvement assertion (line 136-139)
  - ONNX size assertion (line 168-170)
  - ONNX prediction validation (lines 177-197)
  - Val-test gap check (lines 246-250)
  - Alerta Escuela comparison table (lines 283-310)
- Proper entry point: `if __name__ == "__main__":` at line 462
- No stub patterns detected

**test_gate_2_3.py (334 lines):**
- 11 test functions, all substantive
- Fixtures: model_results, calibrated_model, onnx_path
- Tests cover: Brier improvement, ONNX validation, val-test gap, comparison table, artifacts
- All tests PASSED in execution

**ONNX model:**
- Size: 0.10 MB (well under 50 MB limit)
- Validated: 100 random samples, float32 cast, max abs diff 1.03e-07 < 1e-4

**Calibration plot:**
- Size: 87 KB (> 1 KB threshold)
- Contains reliability diagram for before/after Platt scaling

**Calibrated model:**
- Size: 177 KB
- Contains CalibratedClassifierCV with FrozenEstimator wrapper
- Loads successfully via joblib

**Predictions parquet:**
- Size: 726 KB
- Expected columns: prob_dropout, prob_dropout_uncalibrated, pred_dropout, model, split
- Contains validation (2022) and test (2023) predictions

### Level 3: Wired
**calibration.py connections verified:**
1. Loads uncalibrated LightGBM: `joblib.load(lgbm_model_path)` (line 103)
2. Uses FrozenEstimator wrapper: `FrozenEstimator(lgbm)` (line 129)
3. Converts to ONNX: `convert_lightgbm(lgbm, ...)` (line 163)
4. Updates model_results.json: loads, merges, writes (lines 365-431)

**test_gate_2_3.py connections:**
- Loads model_results.json for verification
- Loads calibrated model for testing
- Validates ONNX file existence and predictions
- All imports resolve correctly

**Pipeline execution verified:**
- Gate test 2.3: 11/11 PASSED
- No errors during pipeline run (per summary)
- All output files generated

### Metrics Validation

**From model_results.json:**
```
Calibration (validation set):
- Brier uncalibrated: 0.2115
- Brier calibrated:   0.1156
- Improvement:        45.3%
- Method:             platt_sigmoid

ONNX Export:
- Path:               data/exports/onnx/lightgbm_dropout.onnx
- Size:               0.10 MB
- Max abs diff:       1.03e-07
- Validation:         PASSED (< 1e-4)

Final Test Evaluation (2023):
- Calibrated PR-AUC (weighted):  0.2378
- Validation PR-AUC (weighted):  0.2611
- Val-test gap:                  0.0233 (< 0.07)
- ROC-AUC:                       0.6314
- FNR:                           62.4%

Alerta Escuela Comparison:
- Their FNR range:               57-64%
- Our FNR:                       62.4% (MATCHES)
- Their ROC-AUC range:           0.84-0.89
- Our ROC-AUC:                   0.6314 (different data source)
```

**Platt Parameters (for browser calibration):**
- A (slope):     -5.278337
- B (intercept):  4.276521
- Formula: `calibrated = 1 / (1 + Math.exp(-5.278337 * raw_prob + 4.276521))`

### Gate Test Execution

```
tests/gates/test_gate_2_3.py::test_calibrated_entry_exists PASSED        [  9%]
tests/gates/test_gate_2_3.py::test_brier_improvement PASSED              [ 18%]
tests/gates/test_gate_2_3.py::test_val_test_prauc_gap PASSED             [ 27%]
tests/gates/test_gate_2_3.py::test_onnx_exists_and_size PASSED           [ 36%]
tests/gates/test_gate_2_3.py::test_onnx_predictions_match PASSED         [ 45%]
tests/gates/test_gate_2_3.py::test_platt_parameters PASSED               [ 54%]
tests/gates/test_gate_2_3.py::test_calibration_plot_exists PASSED        [ 63%]
tests/gates/test_gate_2_3.py::test_calibrated_model_persisted PASSED     [ 72%]
tests/gates/test_gate_2_3.py::test_alerta_escuela_comparison PASSED      [ 81%]
tests/gates/test_gate_2_3.py::test_calibrated_predictions_parquet PASSED [ 90%]
tests/gates/test_gate_2_3.py::test_test_set_metrics_present PASSED       [100%]

11 passed, 1 warning in 2.43s
```

### Implementation Quality

**Strengths:**
- Complete 13-step pipeline following plan exactly
- Proper error handling with assertions at critical points
- Clear documentation and print statements for each step
- Correct sklearn 1.8.0 patterns (FrozenEstimator vs deprecated cv='prefit')
- Test set touched exactly once (Step 7 only)
- Comprehensive gate test coverage (11 assertions)

**Architecture:**
- Pipeline script pattern (like baseline.py, lightgbm_xgboost.py)
- Reuses utilities from baseline.py (create_temporal_splits, compute_metrics, etc.)
- Follows established matplotlib patterns (Agg backend, tight_layout, plt.close)
- JSON merge pattern preserves existing model_results.json entries

**Key Decisions:**
1. FrozenEstimator wrapper for calibration (sklearn 1.8.0 compatibility)
2. Export raw LightGBM to ONNX (not calibrated wrapper, which isn't ONNX-convertible)
3. Store Platt A/B parameters separately for browser-side calibration
4. Test year = 2023 with metadata note explaining ENAHO 2024 unavailability

---

## Summary

**Phase Goal:** ACHIEVED

All 5 observable truths verified:
1. ✓ Calibrated Brier < uncalibrated (45.3% reduction)
2. ✓ ONNX predictions match Python within 1e-4 (actual: 1e-7)
3. ✓ Val-test PR-AUC gap < 0.07 (actual: 0.0233)
4. ✓ model_results.json updated with both test entries
5. ✓ Alerta Escuela comparison table present (FNR: 62.4% matches 57-64% range)

All 6 required artifacts exist, are substantive, and are properly wired.

All 3 requirements (MODL-05, MODL-07, EXPO-01) satisfied.

Gate test 2.3: 11/11 PASSED.

Human approval: Granted (per summary Task 3).

**The best model is calibrated, exported to ONNX for browser inference, and evaluated on the 2023 test set exactly once. Ready to proceed to Phase 8 (subgroup fairness metrics).**

---

_Verified: 2026-02-08T21:30:00Z_
_Verifier: Claude (gsd-verifier)_
