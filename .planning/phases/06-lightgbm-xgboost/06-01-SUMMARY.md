# Plan 06-01 Summary

## Outcome: COMPLETE

All 3 tasks completed. LightGBM and XGBoost pipelines trained, tuned, and evaluated.

## Key Results

- **LightGBM val PR-AUC (weighted):** 0.2611 (beats LR baseline 0.2103)
- **XGBoost val PR-AUC (weighted):** 0.2612 (beats LR baseline 0.2103)
- **Algorithm-independence ratio:** 1.0006 (PASS, >= 0.95)
- **Max feature importance:** age = 0.2585 (PASS, < 0.50)
- **LightGBM best iteration:** 79 (proper ensemble after early-stopping fix)
- **XGBoost best iteration:** 50
- **Gate test 2.2:** 11/11 PASSED

## Commits

1. `88f476a` — feat(06-01): create LightGBM + XGBoost training pipelines with Optuna tuning
2. `e53f59b` — test(06-01): add gate test 2.2 for LightGBM/XGBoost validation
3. `2b0fef7` — fix(06): use first_metric_only=True for LightGBM early stopping

## Artifacts Created

- `src/models/lightgbm_xgboost.py` — Full pipeline with Optuna tuning
- `tests/gates/test_gate_2_2.py` — 11 gate test assertions
- `data/exports/model_results.json` — Updated with lightgbm + xgboost entries
- `data/processed/predictions_lgbm.parquet` — 52,112 rows (val + test)
- `data/processed/predictions_xgb.parquet` — 52,112 rows (val + test)
- `data/processed/model_lgbm.joblib` — Persisted LightGBM model
- `data/processed/model_xgb.joblib` — Persisted XGBoost model
- `data/exports/figures/pr_curve_lgbm.png` — PR curve with threshold markers
- `data/exports/figures/pr_curve_xgb.png` — PR curve with threshold markers

## Deviations

- **Early-stopping fix:** LightGBM's `early_stopping()` callback defaults to monitoring the last eval metric (`binary_logloss`), which degrades rapidly with `scale_pos_weight`. Added `first_metric_only=True` to monitor `average_precision` instead. This improved val PR-AUC from 0.2406 to 0.2611 and best_iteration from 1 to 79.

## Feature Importances (LightGBM, gain-normalized)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | age | 0.2585 |
| 2 | nightlight_intensity_z | 0.1227 |
| 3 | census_electricity_pct_z | 0.0931 |
| 4 | poverty_index_z | 0.0844 |
| 5 | census_literacy_rate_z | 0.0839 |
| 6 | census_indigenous_lang_pct_z | 0.0818 |
| 7 | census_water_access_pct_z | 0.0791 |
| 8 | is_working | 0.0561 |
| 9 | log_income | 0.0555 |
| 10 | district_dropout_rate_admin_z | 0.0278 |

Human gate approved: age (#1), poverty (#4 + #9), rurality proxies (#2, #3) all in top positions.
