# Phase 6: LightGBM + XGBoost — Context

## Goal
The primary LightGBM model (matching Alerta Escuela's algorithm) and XGBoost comparison model are trained and Optuna-tuned, establishing algorithm-independence of fairness findings.

## Requirements
- **MODL-03**: LightGBM with Optuna hyperparameter tuning
- **MODL-04**: XGBoost comparison model

## Success Criteria
1. LightGBM achieves higher validation PR-AUC than logistic regression baseline (0.2103)
2. XGBoost validation PR-AUC is within 5% of LightGBM (confirming fairness findings are algorithm-independent)
3. No single feature accounts for more than 50% of LightGBM importance (model uses diverse signal)
4. `model_results.json` updated with lightgbm and xgboost entries including Optuna best hyperparameters, validation metrics, and threshold analysis
5. Gate test 2.2 passes; top-10 feature importances printed for human review

## Human Gate
Review feature importances: age, poverty, rurality should be top-5.

## Discussion Decisions
- Skipped discussion (straightforward phase building on Phase 5 patterns)
- Phase 5 established all reusable patterns: compute_metrics(), temporal splits, model_results.json structure, predictions parquet format, PR curve generation
- LightGBM is the primary model (matches Alerta Escuela's algorithm choice)
- XGBoost serves as algorithm-independence check only

## Key Upstream Dependencies
- `data/processed/enaho_with_features.parquet` — 150,135 rows, 25 features
- `src/models/baseline.py` — temporal splits, compute_metrics(), _df_to_numpy() patterns
- `src/data/features.py` — MODEL_FEATURES (25 features), META_COLUMNS
- `data/exports/model_results.json` — existing LR entry (add lightgbm/xgboost entries)
- LR baseline val PR-AUC (weighted) = 0.2103 (must beat)
- Temporal splits: train=2018-2021 (98,023), val=2022 (26,477), test=2023 (25,635)

## Key Downstream Consumers
- Phase 7: Calibration + ONNX export (best model)
- Phase 8: Subgroup fairness metrics (predictions parquet)
- Phase 9: SHAP interpretability (model + predictions)
