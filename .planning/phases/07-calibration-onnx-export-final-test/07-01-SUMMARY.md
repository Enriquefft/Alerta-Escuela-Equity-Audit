---
phase: 07-calibration-onnx-export-final-test
plan: 01
subsystem: models
tags: [calibration, platt-scaling, onnx, lightgbm, frozen-estimator, onnxruntime]

# Dependency graph
requires:
  - phase: 06-lightgbm-xgboost
    provides: "Trained LightGBM model (model_lgbm.joblib) and model_results.json"
  - phase: 05-baseline-model-temporal-splits
    provides: "Temporal splits, compute_metrics, _threshold_analysis utilities"
provides:
  - "Calibrated LightGBM model (model_lgbm_calibrated.joblib)"
  - "ONNX model for browser inference (lightgbm_dropout.onnx)"
  - "Final 2023 test set metrics (test_2023_final + test_2023_calibrated)"
  - "Platt A/B parameters for JS-side calibration"
  - "Alerta Escuela comparison in model_results.json"
  - "Calibrated predictions parquet"
affects: [08-subgroup-fairness-metrics, 09-shap-interpretability, 11-findings-distillation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FrozenEstimator wrapping for pre-trained model calibration"
    - "ONNX export of raw model + separate Platt parameters for browser calibration"
    - "Test set touched exactly once (final evaluation only)"

key-files:
  created:
    - src/models/calibration.py
    - tests/gates/test_gate_2_3.py
    - data/exports/onnx/lightgbm_dropout.onnx
    - data/exports/figures/calibration_plot.png
    - data/processed/model_lgbm_calibrated.joblib
    - data/processed/predictions_lgbm_calibrated.parquet
  modified:
    - data/exports/model_results.json

key-decisions:
  - "FrozenEstimator over cv='prefit' (removed in sklearn 1.8.0)"
  - "Export raw LightGBM to ONNX, apply Platt scaling in JS with extracted A/B params"
  - "Test year is 2023, not 2024 (ENAHO 2024 unavailable)"
  - "Suppress benign FrozenEstimator sample_weight warning"

patterns-established:
  - "Browser-side calibration: calibrated = 1 / (1 + exp(A * raw_prob + B))"
  - "ONNX validation: 100 random samples, float32 cast, max abs diff < 1e-4"

# Metrics
duration: ~5min
completed: 2026-02-08
---

# Plan 07-01: Calibration + ONNX Export + Final Test Summary

**Platt-calibrated LightGBM with 45.3% Brier reduction, ONNX export (0.10 MB, 1e-7 prediction match), and final 2023 test evaluation showing 62.4% FNR matching Alerta Escuela's 57-64% range**

## Performance

- **Duration:** ~5 min
- **Tasks:** 3 (2 auto + 1 human checkpoint)
- **Files created:** 6
- **Files modified:** 1

## Accomplishments

- Calibrated LightGBM with Platt scaling (Brier: 0.2115 → 0.1156, 45.3% reduction)
- Exported raw LightGBM to ONNX (0.10 MB, max prediction diff 1.03e-07 vs Python)
- Final 2023 test evaluation: PR-AUC 0.2378, ROC-AUC 0.6314, FNR 62.4%
- Val-test PR-AUC gap = 0.0233 (well within 0.07 threshold)
- Alerta Escuela comparison table: FNR matches their 57-64% range, ROC-AUC difference explained by different data sources
- Platt parameters extracted: A=-5.278337, B=4.276521 for browser-side JS calibration
- Gate test 2.3: 11/11 passed, full regression 50/50 passed

## Task Commits

1. **Task 1: Create calibration + ONNX export + final test pipeline** — `43962b3` (feat)
2. **Task 2: Create gate test 2.3** — `de66ebc` (test)
3. **Task 3: Human review** — approved (calibration plot, Brier improvement, Alerta Escuela comparison)

## Files Created/Modified

- `src/models/calibration.py` — Full calibration + ONNX + final test pipeline
- `tests/gates/test_gate_2_3.py` — 11 gate test assertions
- `data/exports/onnx/lightgbm_dropout.onnx` — ONNX model for browser inference (0.10 MB)
- `data/exports/figures/calibration_plot.png` — Calibration reliability diagram (before/after Platt)
- `data/processed/model_lgbm_calibrated.joblib` — Persisted calibrated model
- `data/processed/predictions_lgbm_calibrated.parquet` — Calibrated predictions (val + test)
- `data/exports/model_results.json` — Updated with test_2023_calibrated entry + Alerta Escuela comparison

## Decisions Made

- FrozenEstimator replaces removed cv='prefit' (sklearn 1.8.0 breaking change)
- Raw LightGBM exported to ONNX (not CalibratedClassifierCV wrapper, which isn't ONNX-convertible)
- Platt A/B parameters stored in model_results.json for browser-side calibration formula
- Test year = 2023 with metadata note explaining ENAHO 2024 unavailability

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Calibrated model and predictions ready for Phase 8 (subgroup fairness metrics)
- ONNX model ready for Phase 11 (final exports)
- Platt parameters available for M4 site browser inference
- All model training/evaluation complete — remaining phases are analysis-only

---
*Phase: 07-calibration-onnx-export-final-test*
*Completed: 2026-02-08*
