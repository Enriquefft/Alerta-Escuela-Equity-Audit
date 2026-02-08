# Phase 5: Baseline Model + Temporal Splits - Context

**Gathered:** 2026-02-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish temporal split discipline and train a logistic regression baseline with survey-weighted evaluation. This phase sets the modeling patterns (split creation, metric computation, threshold tuning, JSON export, prediction storage) that Phases 6-7 will reuse for LightGBM/XGBoost. No gradient boosting models are trained here.

</domain>

<decisions>
## Implementation Decisions

### Split Mechanics
- **Year shift due to 2024 unavailability:** train=2018-2021, validate=2022, test=2023 (shifted back one year from spec's train/validate/test=2024 since ENAHO 2024 is not available on INEI portal)
- **No stratification:** Pure year-based splits. Temporal shift is the point — if 2020 COVID distribution differs, that's a valid finding about temporal robustness
- **Include 2020 in training** despite reduced sample (~13,755 rows vs ~25K normal). Document the COVID caveat but keep the data — models should see COVID-era patterns
- **Gate test baselines use 2023 data:** Test set baselines grounded in known 2023 values (~25,663 rows, ~13.42% weighted rate) rather than spec's original 2024 thresholds

### Evaluation Pipeline
- **Full metric suite:** PR-AUC (primary), ROC-AUC, F1, precision, recall, Brier score, log-loss — all computed both weighted (FACTOR07) and unweighted
- **Survey weights via sklearn sample_weight:** Pass FACTOR07 directly to sklearn metric functions (average_precision_score, roc_auc_score, log_loss, brier_score_loss, f1_score, precision_score, recall_score). No custom weighted implementations needed
- **Store per-row predictions:** Save validation and test set predictions (probabilities + binary at chosen threshold) as parquet for Phase 8 fairness and Phase 9 SHAP
- **Threshold tuning:** Report metrics at all 5 spec thresholds (0.3, 0.4, 0.5, 0.6, 0.7) plus the optimal threshold. Claude's discretion on optimization target (max F1 vs max recall at precision floor)

### Model Output Format
- **model_results.json: model-keyed dict.** Top-level keys are model names (logistic_regression, then lightgbm/xgboost in later phases). Each contains: metrics (weighted + unweighted), threshold_analysis, coefficients/feature_importances, metadata (train years, n_samples, convergence)
- **LR coefficients in JSON + console:** Store coefficients, standard errors, odds ratios, and p-values in model_results.json under logistic_regression.coefficients. Also print to console for human review
- **Predictions saved to data/processed/:** predictions_lr.parquet (later predictions_lgbm.parquet, etc.). Processed data, not exports — downstream phases read internally
- **Model persisted as joblib:** data/processed/model_lr.joblib. Phase 7 needs the best model for calibration and ONNX export without retraining

### Human Review Scope
- **Equity-relevant features highlighted:** Checkpoint focuses on poverty_quintile (should increase risk), rural (should increase risk), lang_other_indigenous (should increase risk), age, es_mujer — the fairness audit's core dimensions
- **Rejection criteria: wrong signs + sub-random PR-AUC.** Reject if poverty DECREASES risk, rural DECREASES risk, or indigenous language has no effect (contradicts descriptive stats). Also reject if PR-AUC < 0.14 (random baseline). Both checks must pass
- **Threshold analysis: table + PR curve PNG.** Table with metrics at 5 thresholds PLUS precision-recall curve visualization
- **Weighted vs unweighted comparison shown:** Side-by-side table of weighted vs unweighted PR-AUC, F1, recall — makes FACTOR07 effect visible and reviewable at checkpoint

### Claude's Discretion
- Threshold optimization target (max weighted F1 vs max recall at precision floor)
- LogisticRegression hyperparameters (solver, max_iter, regularization strength)
- Exact precision-recall curve styling and layout
- Whether to include a confusion matrix visualization

</decisions>

<specifics>
## Specific Ideas

- Year shift (2024 unavailable) must be clearly documented in model_results.json metadata and findings — the audit should be transparent about data availability constraints
- 2020 COVID caveat should appear in training set metadata (reduced sample, potential distribution shift)
- The LR baseline exists to validate the pipeline and provide interpretable coefficients, not to achieve best performance — the fairness audit is the product

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-baseline-model-temporal-splits*
*Context gathered: 2026-02-08*
