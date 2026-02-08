# Phase 6: LightGBM + XGBoost - Research

**Researched:** 2026-02-08
**Domain:** Gradient boosting models with Optuna hyperparameter tuning
**Confidence:** HIGH

## Summary

Phase 6 trains LightGBM (primary, matching Alerta Escuela's algorithm) and XGBoost (comparison) models using Optuna hyperparameter tuning, then evaluates them against the logistic regression baseline (val PR-AUC 0.2103). Both models use the sklearn API and follow Phase 5's established patterns for temporal splits, metric computation, threshold analysis, prediction export, and model_results.json structure.

The installed versions are LightGBM 4.6.0, XGBoost 3.1.3, and Optuna 4.7.0. All three have been verified to work together. The `optuna-integration` package is NOT installed (pruning callbacks unavailable), but this is not needed -- at ~1.5s per LightGBM trial on 98k rows, 100 Optuna trials complete in under 3 minutes. LightGBM's built-in early_stopping callback serves as sufficient pruning within each trial.

**Primary recommendation:** Use the sklearn API for both models (LGBMClassifier, XGBClassifier) with `sample_weight=FACTOR07` for survey weights plus `scale_pos_weight=4.80` for class imbalance. Tune LightGBM with 100 Optuna trials optimizing weighted validation PR-AUC, then apply best params structure to XGBoost tuning (50 trials sufficient for comparison model).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- LightGBM is the primary model (matches Alerta Escuela's algorithm choice)
- XGBoost serves as algorithm-independence check only
- Phase 5 established all reusable patterns: compute_metrics(), temporal splits, model_results.json structure, predictions parquet format, PR curve generation
- model_results.json must be updated with `lightgbm` and `xgboost` keys (add to existing file, do not overwrite LR entry)

### Claude's Discretion
- Optuna trial count and search space ranges
- Whether to use pruning callbacks or rely on early stopping
- Feature importance type (gain vs split)
- Exact hyperparameter search ranges
- Whether XGBoost gets full Optuna tuning or uses LightGBM's best params as starting point

### Deferred Ideas (OUT OF SCOPE)
None -- discussion was skipped (straightforward phase)
</user_constraints>

## Standard Stack

### Core (already installed, versions verified)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | Primary gradient boosting model | Matches Alerta Escuela's algorithm; LGBMClassifier sklearn API |
| xgboost | 3.1.3 | Comparison gradient boosting model | Algorithm-independence verification; XGBClassifier sklearn API |
| optuna | 4.7.0 | Hyperparameter tuning | Bayesian optimization; TPE sampler; built-in study persistence |
| scikit-learn | (installed) | Metrics, evaluation | average_precision_score, roc_auc_score with sample_weight |

### Supporting (already installed)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| joblib | (installed) | Model persistence | Save trained models as .joblib |
| matplotlib | (installed) | PR curve visualization | Reuse baseline.py _plot_pr_curve pattern |
| numpy | (installed) | Array operations | Feature importance arrays, threshold analysis |
| polars | (installed) | Data loading | Read parquet, create splits |

### NOT Needed

| Library | Why Not |
|---------|---------|
| optuna-integration | NOT installed. Pruning callbacks not needed -- each trial is ~1.5s on 98k rows. LightGBM's early_stopping callback provides within-trial pruning. |

**Installation:** No new packages needed. All dependencies already declared in `pyproject.toml`.

## Architecture Patterns

### Recommended File Structure

```
src/models/
    baseline.py          # Existing - imports reused by lightgbm_xgboost.py
    lightgbm_xgboost.py  # NEW - LightGBM + XGBoost training pipeline
```

### Pattern 1: Reuse Phase 5 Utilities Directly

**What:** Import temporal splits, numpy conversion, metric computation, and threshold analysis from baseline.py rather than duplicating.
**When to use:** Always -- single source of truth.

```python
# Import from baseline.py
from models.baseline import (
    create_temporal_splits,
    _df_to_numpy,
    compute_metrics,
    _threshold_analysis,
    TRAIN_YEARS,
    VALIDATE_YEAR,
    TEST_YEAR,
    FIXED_THRESHOLDS,
    ID_COLUMNS,
)
```

### Pattern 2: Optuna Objective Function with Early Stopping

**What:** Define an objective function that creates an LGBMClassifier/XGBClassifier with Optuna-suggested hyperparameters, trains with early stopping, and returns weighted validation PR-AUC.
**When to use:** For both LightGBM and XGBoost hyperparameter tuning.

```python
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score

def lgbm_objective(trial, X_train, y_train, w_train, X_val, y_val, w_val, scale_pos_weight):
    params = {
        "n_estimators": 500,  # High, let early_stopping trim
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "importance_type": "gain",
        "verbose": -1,
        "random_state": 42,
    }

    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="average_precision",
        callbacks=[early_stopping(50), log_evaluation(0)],
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob, sample_weight=w_val)
    return pr_auc


# Run study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(
    lambda trial: lgbm_objective(trial, X_train, y_train, w_train, X_val, y_val, w_val, spw),
    n_trials=100,
)
best_params = study.best_trial.params
```

### Pattern 3: XGBoost Objective Function

**What:** Same pattern as LightGBM but with XGBoost-specific parameter names.
**When to use:** For XGBoost hyperparameter tuning.
**Key differences from LightGBM:**
- `eval_sample_weight` is called `sample_weight_eval_set` in XGBoost
- `callbacks` is a constructor parameter, not a fit parameter
- `early_stopping_rounds` is a constructor parameter
- `eval_metric='aucpr'` (not `'average_precision'`)
- `min_child_weight` replaces `min_child_samples`
- `gamma` parameter exists (minimum loss reduction for split)
- `verbosity=0` (not `verbose=-1`)

```python
from xgboost import XGBClassifier

def xgb_objective(trial, X_train, y_train, w_train, X_val, y_val, w_val, scale_pos_weight):
    params = {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "importance_type": "gain",
        "verbosity": 0,
        "random_state": 42,
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob, sample_weight=w_val)
    return pr_auc
```

### Pattern 4: Retrain Best Model and Extract Feature Importances

**What:** After Optuna tuning, retrain the best model with optimal params, extract feature importances, and compute full evaluation.
**When to use:** After study.optimize() completes.

```python
# Retrain best LightGBM model
best_params = study.best_trial.params
best_lgbm = LGBMClassifier(
    n_estimators=500,
    **best_params,
    scale_pos_weight=spw,
    importance_type="gain",
    verbose=-1,
    random_state=42,
)
best_lgbm.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[w_val],
    eval_metric="average_precision",
    callbacks=[early_stopping(50), log_evaluation(0)],
)

# Feature importances (gain-based, normalized to sum=1)
raw_importances = best_lgbm.feature_importances_
normalized_importances = raw_importances / raw_importances.sum()
importance_dict = dict(zip(MODEL_FEATURES, normalized_importances))
sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Top-10 for human review
print("=== TOP-10 FEATURE IMPORTANCES (GAIN) ===")
for feat, imp in sorted_importances[:10]:
    print(f"  {feat:<40s} {imp:.4f}")
```

### Pattern 5: Merge into Existing model_results.json

**What:** Read existing JSON file, add new model entries, write back.
**When to use:** After training both models.

```python
import json
from pathlib import Path

results_path = root / "data" / "exports" / "model_results.json"

# Read existing
with open(results_path, "r") as f:
    model_results = json.load(f)

# Add new entries
model_results["lightgbm"] = {
    "metadata": {...},
    "metrics": {
        "validate_2022": {"weighted": {...}, "unweighted": {...}},
        "test_2023": {"weighted": {...}, "unweighted": {...}},
    },
    "threshold_analysis": {...},
    "feature_importances": sorted_importances_list,
    "optuna_study": {
        "n_trials": study.n_trials,
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
    },
}

model_results["xgboost"] = {... same structure ...}

with open(results_path, "w") as f:
    json.dump(model_results, f, indent=2)
```

### Pattern 6: Prediction Export (Matching Baseline Pattern)

**What:** Save predictions parquet with same schema as baseline predictions.
**When to use:** For each model's validation and test predictions.

```python
# Adapt _save_predictions from baseline.py for model_name parameter
pred_df = df.select(ID_COLUMNS).with_columns([
    pl.Series("prob_dropout", y_prob),
    pl.Series("pred_dropout", y_pred),
    pl.lit("lightgbm").alias("model"),  # or "xgboost"
    pl.lit(optimal_threshold).alias("threshold"),
    pl.lit(split_name).alias("split"),
])
pred_df.write_parquet(predictions_path)
```

### Anti-Patterns to Avoid

- **DO NOT use `is_unbalance=True` with `sample_weight`:** Use `scale_pos_weight` instead. `is_unbalance` and `scale_pos_weight` are mutually exclusive. When combined with `sample_weight`, class weights are multiplied with sample weights -- this is correct behavior for survey-weighted imbalanced classification.
- **DO NOT use `class_weight='balanced'` for LightGBM:** This is for multi-class. For binary, use `scale_pos_weight`.
- **DO NOT use `optuna.integration.LightGBMPruningCallback`:** Not installed and not needed at this dataset size.
- **DO NOT overwrite model_results.json:** Read existing file, add entries, write back.
- **DO NOT use `eval_sample_weight` with XGBoost:** The parameter is called `sample_weight_eval_set` in XGBClassifier.fit().
- **DO NOT pass `callbacks` to XGBClassifier.fit():** In XGBoost 3.x sklearn API, `callbacks` and `early_stopping_rounds` are constructor parameters.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Temporal splits | Custom split logic | `create_temporal_splits()` from baseline.py | Already verified, handles year coverage assertions |
| Numpy conversion | Custom Polars->numpy | `_df_to_numpy()` from baseline.py | Handles Boolean->Int8 cast, FACTOR07 extraction |
| Metric computation | Manual metric loops | `compute_metrics()` from baseline.py | Handles weighted + unweighted, consistent schema |
| Threshold analysis | Custom threshold search | `_threshold_analysis()` from baseline.py | 5 fixed + optimal threshold, consistent format |
| Hyperparameter search | Grid search / manual | Optuna TPE sampler | Bayesian optimization, much more efficient |
| Early stopping | Manual iteration counting | LightGBM's `early_stopping()` callback / XGBoost's `early_stopping_rounds` | Built-in, handles best_iteration tracking |
| Feature importance normalization | Manual normalization | `model.feature_importances_ / sum` | Simple but must normalize for comparison |

**Key insight:** Phase 5 did the hard work of establishing evaluation patterns. Phase 6 should reuse everything and only add: (1) Optuna objective functions, (2) model-specific training, (3) feature importance extraction, (4) JSON merge logic.

## Common Pitfalls

### Pitfall 1: Survey Weights vs Class Imbalance Weights Confusion

**What goes wrong:** Conflating FACTOR07 survey expansion weights with class rebalancing weights. Using `is_unbalance=True` which sets scale_pos_weight internally AND passing FACTOR07 as sample_weight results in double-weighting of the class imbalance signal.
**Why it happens:** FACTOR07 are survey expansion weights (how many people each respondent represents), NOT class weights. The class imbalance (17.24% dropout in train) needs separate handling.
**How to avoid:** Pass FACTOR07 via `sample_weight` parameter. Handle class imbalance via `scale_pos_weight=4.80` (n_neg/n_pos on training set). These multiply together correctly -- survey weights scale all instances, class weight scales positive instances additionally.
**Warning signs:** If probability calibration looks very wrong, check if double-weighting occurred.

### Pitfall 2: LightGBM vs XGBoost API Differences

**What goes wrong:** Using LightGBM parameter names in XGBoost or vice versa.
**Why it happens:** Similar but not identical APIs.
**How to avoid:** Reference this table:

| Concept | LightGBM (LGBMClassifier) | XGBoost (XGBClassifier) |
|---------|---------------------------|-------------------------|
| Eval weight param | `eval_sample_weight=[w]` | `sample_weight_eval_set=[w]` |
| Callbacks in fit | `callbacks=[...]` in `.fit()` | `callbacks=[...]` in `__init__()` |
| Early stopping | `callbacks=[early_stopping(50)]` in `.fit()` | `early_stopping_rounds=50` in `__init__()` |
| Eval metric (PR-AUC) | `eval_metric='average_precision'` | `eval_metric='aucpr'` |
| Verbosity off | `verbose=-1` | `verbosity=0` + `verbose=False` in fit |
| Min samples in leaf | `min_child_samples` | `min_child_weight` |
| Best iteration | `model.best_iteration_` | `model.best_iteration` (no underscore) |
| L2 regularization default | 0.0 | 1.0 |

### Pitfall 3: Feature Importance Normalization

**What goes wrong:** Raw gain importances are absolute values that vary with n_estimators and tree depth. Comparing across models without normalization is meaningless.
**Why it happens:** LightGBM `feature_importances_` with `importance_type='gain'` returns total gain (not normalized). XGBoost with `importance_type='gain'` also returns raw gain.
**How to avoid:** Normalize to sum=1 for both models: `imp / imp.sum()`. This makes the "no single feature > 50%" check valid across models.
**Warning signs:** Feature importance values that seem very large or not summing to 1.

### Pitfall 4: n_estimators Confusion with Early Stopping

**What goes wrong:** Setting low n_estimators (e.g., 100) thinking early stopping will handle it. But early stopping needs headroom.
**Why it happens:** Optuna might suggest a high learning rate that converges fast, but with n_estimators=100 there's no room for early stopping to improve.
**How to avoid:** Set n_estimators=500 (or even 1000) as a ceiling and let early_stopping(50) trim to the best iteration. The actual number of trees used is `model.best_iteration_`.

### Pitfall 5: Optuna Study Direction

**What goes wrong:** Setting `direction="minimize"` when optimizing PR-AUC.
**Why it happens:** Some metrics (like log_loss, error) should be minimized, while PR-AUC should be maximized.
**How to avoid:** Always use `direction="maximize"` for PR-AUC optimization.

## Code Examples

### Complete LightGBM Training Pipeline (verified pattern)

```python
# Source: Verified locally with LightGBM 4.6.0, Optuna 4.7.0
import json
import optuna
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score

# 1. Calculate scale_pos_weight from training data
n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
spw = n_neg / n_pos  # ~4.80 for this dataset

# 2. Define objective
def lgbm_objective(trial):
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        scale_pos_weight=spw,
        importance_type="gain",
        verbose=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="average_precision",
        callbacks=[early_stopping(50), log_evaluation(0)],
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob, sample_weight=w_val)

# 3. Run study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(lgbm_objective, n_trials=100)

# 4. Retrain with best params
best_params = study.best_trial.params
best_lgbm = LGBMClassifier(
    n_estimators=500,
    **best_params,
    scale_pos_weight=spw,
    importance_type="gain",
    verbose=-1,
    random_state=42,
)
best_lgbm.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[w_val],
    eval_metric="average_precision",
    callbacks=[early_stopping(50), log_evaluation(0)],
)

# 5. Extract feature importances
raw_imp = best_lgbm.feature_importances_
norm_imp = raw_imp / raw_imp.sum()
```

### Complete XGBoost Training Pipeline (verified pattern)

```python
# Source: Verified locally with XGBoost 3.1.3, Optuna 4.7.0
from xgboost import XGBClassifier

def xgb_objective(trial):
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        importance_type="gain",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob, sample_weight=w_val)

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)
```

### Feature Importance Console Output (for human gate)

```python
# Print top-10 importances for human review
print("\n=== TOP-10 LIGHTGBM FEATURE IMPORTANCES (GAIN, NORMALIZED) ===")
for feat, imp in sorted_importances[:10]:
    equity_mark = " ***" if feat in _EQUITY_FEATURES else ""
    print(f"  {feat:<40s} {imp:.4f}{equity_mark}")

# Verify no single feature > 50%
max_imp = sorted_importances[0][1]
assert max_imp < 0.50, (
    f"Feature {sorted_importances[0][0]} has {max_imp:.4f} importance (>50%)"
)
```

### JSON Merge Pattern

```python
# Read existing model_results.json, add entries, write back
results_path = root / "data" / "exports" / "model_results.json"

with open(results_path, "r") as f:
    model_results = json.load(f)

# Verify LR entry still exists
assert "logistic_regression" in model_results, "LR baseline entry missing!"

# Add lightgbm entry
model_results["lightgbm"] = {
    "metadata": {
        "model_type": "LGBMClassifier",
        "train_years": TRAIN_YEARS,
        "validate_year": VALIDATE_YEAR,
        "test_year": TEST_YEAR,
        "n_train": int(X_train.shape[0]),
        "n_validate": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(MODEL_FEATURES),
        "feature_names": list(MODEL_FEATURES),
        "best_iteration": int(best_lgbm.best_iteration_),
        "scale_pos_weight": round(spw, 4),
        "optuna_n_trials": study.n_trials,
        "optuna_best_trial": study.best_trial.number,
        "optuna_best_params": best_params,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    },
    "metrics": {
        "validate_2022": {"weighted": w_val_metrics, "unweighted": uw_val_metrics},
        "test_2023": {"weighted": w_test_metrics, "unweighted": uw_test_metrics},
    },
    "threshold_analysis": threshold_data,
    "feature_importances": [
        {"feature": feat, "importance": round(float(imp), 6)}
        for feat, imp in sorted_importances
    ],
}

# Add xgboost entry (same structure)
model_results["xgboost"] = { ... }

# Write back
with open(results_path, "w") as f:
    json.dump(model_results, f, indent=2)
```

## Data Characteristics (verified)

| Property | Value |
|----------|-------|
| Total rows | 150,135 |
| Training rows | 98,023 (2018-2021) |
| Validation rows | 26,477 (2022) |
| Test rows | 25,635 (2023) |
| Features | 25 (MODEL_FEATURES) |
| Train dropout rate | 17.24% (16,895 positive) |
| scale_pos_weight | 4.8019 (81,128 / 16,895) |
| LR baseline val PR-AUC (weighted) | 0.2103 |
| LR baseline val PR-AUC (unweighted) | 0.2077 |

## Hyperparameter Search Spaces

### LightGBM Search Space (recommended)

| Parameter | Type | Range | Scale | Rationale |
|-----------|------|-------|-------|-----------|
| learning_rate | float | [0.01, 0.3] | log | Standard range; lower allows more trees |
| num_leaves | int | [8, 128] | linear | 31 default; 128 max avoids overfitting on 98k rows |
| max_depth | int | [3, 12] | linear | Constrain with num_leaves for regularization |
| min_child_samples | int | [5, 100] | linear | 20 default; lower risks overfitting |
| subsample | float | [0.5, 1.0] | linear | Row sampling for regularization |
| colsample_bytree | float | [0.5, 1.0] | linear | Feature sampling per tree |
| reg_alpha | float | [1e-8, 10.0] | log | L1 regularization |
| reg_lambda | float | [1e-8, 10.0] | log | L2 regularization |

**Fixed:** n_estimators=500 (early stopping trims), scale_pos_weight=4.80, random_state=42, importance_type='gain'

### XGBoost Search Space (recommended)

| Parameter | Type | Range | Scale | Rationale |
|-----------|------|-------|-------|-----------|
| learning_rate | float | [0.01, 0.3] | log | Same rationale as LightGBM |
| max_depth | int | [3, 10] | linear | XGBoost default=6; narrower range than LGBM |
| min_child_weight | int | [1, 20] | linear | Analogous to min_child_samples |
| gamma | float | [1e-8, 5.0] | log | Minimum loss reduction for split |
| subsample | float | [0.5, 1.0] | linear | Same as LightGBM |
| colsample_bytree | float | [0.5, 1.0] | linear | Same as LightGBM |
| reg_alpha | float | [1e-8, 10.0] | log | Same as LightGBM |
| reg_lambda | float | [1e-8, 10.0] | log | Note: XGBoost default=1.0 (vs LGBM 0.0) |

**Fixed:** n_estimators=500, early_stopping_rounds=50, scale_pos_weight=4.80, random_state=42, importance_type='gain'

## Gate Test Assertions

```python
# Gate test 2.2 assertions

# 1. LightGBM beats LR baseline
lgbm_val_prauc = weighted_val_lgbm["pr_auc"]
assert lgbm_val_prauc > 0.2103, (
    f"LightGBM val PR-AUC {lgbm_val_prauc:.4f} <= LR baseline 0.2103"
)

# 2. XGBoost within 5% of LightGBM (algorithm-independence)
xgb_val_prauc = weighted_val_xgb["pr_auc"]
ratio = xgb_val_prauc / lgbm_val_prauc
assert ratio >= 0.95, (
    f"XGBoost val PR-AUC {xgb_val_prauc:.4f} is {(1-ratio)*100:.1f}% below "
    f"LightGBM {lgbm_val_prauc:.4f} (threshold: 5%)"
)

# 3. No single feature > 50% importance
max_imp_feat, max_imp_val = sorted_importances[0]
assert max_imp_val < 0.50, (
    f"Feature {max_imp_feat} has {max_imp_val:.4f} normalized importance (>50%)"
)

# 4. model_results.json has all three model entries
with open(results_path) as f:
    results = json.load(f)
assert "logistic_regression" in results, "LR entry missing"
assert "lightgbm" in results, "LightGBM entry missing"
assert "xgboost" in results, "XGBoost entry missing"

# 5. Human review: top-10 feature importances printed
# (age, poverty_quintile, rural should be in top-5)
top5_features = [feat for feat, _ in sorted_importances[:5]]
print(f"\nTop-5 features: {top5_features}")
# These are checked by human, not asserted programmatically
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| optuna.integration.X | optuna_integration (separate package) | Optuna 4.x | Import path changed; requires pip install optuna-integration[lightgbm] |
| LightGBM early_stopping_rounds param | early_stopping() callback | LightGBM 4.0+ | Old param deprecated; use callbacks=[early_stopping(N)] |
| XGBoost callbacks in fit() | callbacks in constructor | XGBoost 2.0+ | Must pass callbacks to XGBClassifier(), not .fit() |
| importance_type default 'weight' | importance_type default 'split' (LightGBM) | LightGBM 4.x | Use 'gain' explicitly for meaningful importance comparison |

## Open Questions

1. **Optimal trial count for XGBoost**
   - What we know: 50 trials is sufficient for a comparison model; 100 trials for primary model
   - What's unclear: Whether XGBoost benefits from more trials given similar search space
   - Recommendation: Use 50 trials for XGBoost (comparison only, not primary model)

2. **PR curve for both models or just LightGBM?**
   - What we know: Baseline generates PR curve PNG; Phase 5 established the pattern
   - What's unclear: Whether spec requires PR curves for all models or just the primary
   - Recommendation: Generate PR curves for both (reuse _plot_pr_curve pattern with model name)

3. **Probability calibration warning**
   - What we know: Using scale_pos_weight distorts probability estimates. LightGBM docs note "poor estimates of individual class probabilities"
   - What's unclear: Whether Phase 7 (calibration) handles this or Phase 6 should document it
   - Recommendation: Document the calibration need in model_results.json metadata; Phase 7 handles actual calibration

## Sources

### Primary (HIGH confidence)
- LightGBM 4.6.0 official docs: LGBMClassifier API, Parameters reference
- XGBoost 3.1.1 official docs: Parameters reference, sklearn API
- Optuna 4.7.0 official docs: Study API, trial.suggest_* methods
- Local verification: All code patterns tested against installed versions (LightGBM 4.6.0, XGBoost 3.1.3, Optuna 4.7.0)

### Secondary (MEDIUM confidence)
- [LightGBM LGBMClassifier docs](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Optuna-Integration LightGBMPruningCallback](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.LightGBMPruningCallback.html)
- [GitHub: LightGBM issue #6807 - Class/Sample weights interaction](https://github.com/microsoft/LightGBM/issues/6807)
- [Optuna examples: lightgbm_simple.py](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py)

### Tertiary (LOW confidence)
- None -- all findings verified locally

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All versions verified locally, APIs tested
- Architecture: HIGH - Patterns tested with actual installed libraries, benchmark confirms feasibility
- Pitfalls: HIGH - API differences verified empirically (fit signatures, parameter names, callback locations)
- Hyperparameter ranges: MEDIUM - Based on standard practices and documentation defaults, actual data characteristics verified

**Research date:** 2026-02-08
**Valid until:** 2026-03-08 (stable libraries, no major releases expected)
