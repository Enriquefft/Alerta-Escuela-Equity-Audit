# Phase 7: Calibration + ONNX Export + Final Test — Context

## Goal
The best model is calibrated, exported to ONNX for browser inference, and evaluated on the 2024 test set exactly once — the only time test data is touched.

## Requirements
- **MODL-05**: Model calibration (Platt scaling or isotonic regression)
- **MODL-07**: Final test set evaluation (one-time touch)
- **EXPO-01**: ONNX export for browser inference

## Success Criteria
1. Calibrated model has lower Brier score than uncalibrated on validation set
2. Test set (2024) PR-AUC is within 0.07 of validation PR-AUC (no extreme overfitting)
3. ONNX file exists at `data/exports/onnx/lightgbm_dropout.onnx`, is under 50MB, and predictions match Python model within 1e-4 absolute difference on 100 random samples
4. `model_results.json` updated with `test_2024_final` and `test_2024_calibrated` entries plus Alerta Escuela comparison
5. Gate test 2.3 passes; comparison table with Alerta Escuela published metrics printed for human review

## Human Gate
Review calibration plot and metrics vs Alerta Escuela comparison.

## Discussion Decisions
- Skipped discussion (straightforward phase building on Phase 6 outputs)
- LightGBM is the primary model (best_iteration=79, val PR-AUC=0.2611)
- scale_pos_weight distorts probabilities — calibration is critical for meaningful risk scores
- ENAHO 2024 not available — test set is 2023 (already split in Phase 5 as TEST_YEAR=2023)
- ONNX export targets browser inference for M4 scrollytelling site

## Key Upstream Dependencies
- `data/processed/model_lgbm.joblib` — trained LightGBM model
- `data/processed/enaho_with_features.parquet` — 150,135 rows, 25 features
- `src/models/baseline.py` — temporal splits, compute_metrics(), evaluation patterns
- `src/models/lightgbm_xgboost.py` — LightGBM pipeline patterns
- `data/exports/model_results.json` — existing LR + LightGBM + XGBoost entries
- Temporal splits: train=2018-2021 (98,023), val=2022 (26,477), test=2023 (25,635)
- LightGBM val PR-AUC (weighted) = 0.2611, val Brier = 0.2115

## Key Downstream Consumers
- Phase 8: Subgroup fairness metrics (calibrated predictions)
- Phase 9: SHAP interpretability (calibrated model)
- Phase 11: M4 exports (ONNX file for browser)

## Important Notes
- ENAHO 2024 is NOT available (noted in Phase 1 memory). The "test set" is 2023 data (TEST_YEAR=2023 in baseline.py).
- The roadmap says "test set (2024)" but this refers to the test temporal split, which is 2023 data.
- ONNX float32 tolerance may need relaxing to 1e-4 (flagged in STATE.md blockers).
