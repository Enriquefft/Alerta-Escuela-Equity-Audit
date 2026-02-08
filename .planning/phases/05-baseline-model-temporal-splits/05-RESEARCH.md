# Phase 5: Baseline Model + Temporal Splits - Research

**Researched:** 2026-02-08
**Domain:** Logistic regression with survey-weighted evaluation, temporal data splitting, threshold analysis
**Confidence:** HIGH

## Summary

This phase establishes temporal split discipline and trains a logistic regression baseline to validate the modeling pipeline. The primary technical challenges are: (1) correctly splitting by year with zero overlap, (2) computing the full metric suite with survey weights (FACTOR07), (3) obtaining LR coefficients with standard errors, p-values, and odds ratios for the human review gate, (4) threshold analysis at 5 fixed points plus an optimal threshold, and (5) exporting model_results.json and prediction parquets in formats that Phases 6-9 will consume.

The critical discovery is that sklearn's LogisticRegression does NOT provide standard errors or p-values -- only raw coefficients. The CONTEXT.md requires "coefficients, standard errors, odds ratios, and p-values." This requires a dual-model approach: sklearn for prediction/persistence/evaluation AND statsmodels GLM(Binomial) for statistical inference. Both models produce identical coefficients when configured equivalently.

All sklearn metric functions (average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss, log_loss, precision_recall_curve) accept `sample_weight` and are verified at sklearn 1.8.0. The PrecisionRecallDisplay class also supports sample_weight for visualization.

**Primary recommendation:** Use sklearn LogisticRegression for training/prediction/joblib persistence and statsmodels GLM(Binomial, freq_weights) on the same training data for statistical inference. Both are already installed in the environment.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Split Mechanics:**
- train=2018-2021, validate=2022, test=2023 (shifted back one year from spec due to ENAHO 2024 unavailability)
- No stratification -- pure year-based splits
- Include 2020 in training despite reduced sample (~13,755 rows)
- Gate test baselines use 2023 data (~25,635 rows, ~13.42% weighted rate)

**Evaluation Pipeline:**
- Full metric suite: PR-AUC (primary), ROC-AUC, F1, precision, recall, Brier score, log-loss -- all computed both weighted (FACTOR07) and unweighted
- Survey weights via sklearn sample_weight: Pass FACTOR07 directly to sklearn metric functions
- Store per-row predictions as parquet for Phase 8 fairness and Phase 9 SHAP
- Threshold tuning at 0.3, 0.4, 0.5, 0.6, 0.7 plus optimal threshold

**Model Output Format:**
- model_results.json: model-keyed dict (top-level key "logistic_regression")
- LR coefficients in JSON + console: coefficients, standard errors, odds ratios, p-values
- Predictions saved to data/processed/predictions_lr.parquet
- Model persisted as data/processed/model_lr.joblib

**Human Review Scope:**
- Equity-relevant features highlighted: poverty_quintile, rural, lang_other_indigenous, age, es_mujer
- Rejection criteria: wrong coefficient signs (poverty/rural/indigenous) OR PR-AUC < 0.14
- Threshold analysis table + PR curve PNG
- Weighted vs unweighted comparison shown side-by-side

### Claude's Discretion

- Threshold optimization target (max weighted F1 vs max recall at precision floor)
- LogisticRegression hyperparameters (solver, max_iter, regularization strength)
- Exact precision-recall curve styling and layout
- Whether to include a confusion matrix visualization

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

All libraries are already installed in the project environment (verified).

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | LogisticRegression training, predictions, metric computation | Industry standard for ML; all metric functions support sample_weight |
| statsmodels | 0.14.6 | GLM(Binomial) for LR coefficient inference (SEs, p-values, CIs) | Only way to get standard errors/p-values from logistic regression in Python |
| polars | 1.0+ | Data loading, splitting, feature selection, parquet I/O | Project standard (Polars-first per prior phases) |
| numpy | (bundled) | Array operations at sklearn boundary | Required for Polars -> sklearn conversion |
| joblib | 1.5.3 | Model persistence (dump/load) | sklearn's recommended serialization; already a dependency |
| matplotlib | (installed) | Precision-recall curve visualization | Project standard (used in Phase 4 descriptive.py) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | - | model_results.json export | JSON serialization of metrics/coefficients |
| pathlib (stdlib) | - | File path management | Project convention for all I/O |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| statsmodels GLM | Manual SE computation via Hessian | GLM is simpler, already installed, well-tested |
| joblib.dump | pickle | joblib optimized for numpy arrays in sklearn models |
| sklearn PrecisionRecallDisplay | Manual matplotlib PR curve | PrecisionRecallDisplay supports sample_weight natively |

**Installation:** No new packages needed. All are in pyproject.toml.

## Architecture Patterns

### Recommended Project Structure

```
src/
  models/
    __init__.py          # Re-exports (currently empty)
    baseline.py          # NEW: temporal splits, LR training, evaluation, export
tests/
  gates/
    test_gate_2_1.py     # NEW: gate test for baseline model
data/
  processed/
    enaho_with_features.parquet  # INPUT (from Phase 4)
    predictions_lr.parquet       # OUTPUT: per-row predictions
    model_lr.joblib              # OUTPUT: persisted sklearn model
  exports/
    model_results.json           # OUTPUT: metrics, coefficients, threshold analysis
    figures/
      pr_curve_lr.png            # OUTPUT: precision-recall curve
```

### Pattern 1: Dual-Model Training (sklearn + statsmodels)

**What:** Train sklearn LogisticRegression for prediction and joblib persistence; train statsmodels GLM(Binomial) on identical data for coefficient inference (SEs, p-values, odds ratios).

**When to use:** Any time you need both predictive model persistence AND statistical inference from logistic regression.

**Why:** sklearn's LogisticRegression provides `coef_` and `intercept_` but NO standard errors, p-values, or confidence intervals. statsmodels GLM provides all inference statistics but is not designed for sklearn pipeline integration or joblib persistence.

**Example:**
```python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# Feature matrix X (numpy), target y (numpy), weights w (numpy)
# All converted from Polars at the sklearn boundary

# --- sklearn model (for prediction + persistence) ---
lr = LogisticRegression(
    class_weight="balanced",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
    C=1.0,
)
lr.fit(X_train, y_train, sample_weight=w_train)
y_prob = lr.predict_proba(X_val)[:, 1]  # P(dropout=1)

# --- statsmodels model (for inference) ---
X_train_const = sm.add_constant(X_train)
glm = sm.GLM(
    y_train,
    X_train_const,
    family=sm.families.Binomial(),
    freq_weights=w_train,
)
glm_result = glm.fit()

# Extract inference
coefficients = glm_result.params        # includes intercept
std_errors = glm_result.bse
p_values = glm_result.pvalues
conf_int = glm_result.conf_int()        # 2D array: [n_params, 2]
odds_ratios = np.exp(coefficients)
```

### Pattern 2: Polars-to-Numpy Conversion at sklearn Boundary

**What:** Keep data in Polars for I/O and filtering; convert to numpy arrays only when calling sklearn/statsmodels.

**When to use:** Every interaction with sklearn or statsmodels.

**Example:**
```python
import polars as pl
from data.features import MODEL_FEATURES

df = pl.read_parquet("data/processed/enaho_with_features.parquet")

# Split by year (stays in Polars)
train_df = df.filter(pl.col("year").is_in([2018, 2019, 2020, 2021]))

# Convert at sklearn boundary
X_train = train_df.select(MODEL_FEATURES).to_numpy()  # (n, 25)
y_train = train_df["dropout"].cast(pl.Int8).to_numpy()  # Boolean -> Int8 -> numpy
w_train = train_df["FACTOR07"].to_numpy()               # Float64 -> numpy
```

**Critical:** The `dropout` column is `Boolean` dtype in Polars. Must cast to `Int8` (or `Float64`) before `.to_numpy()` for sklearn compatibility.

### Pattern 3: Metric Computation with Survey Weights

**What:** Compute all metrics both weighted and unweighted, then assert they differ.

**Example:**
```python
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss, log_loss,
)

def compute_metrics(y_true, y_prob, y_pred, weights=None):
    """Compute full metric suite, optionally weighted."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y_true, y_prob, sample_weight=weights)),
        "f1": float(f1_score(y_true, y_pred, sample_weight=weights)),
        "precision": float(precision_score(y_true, y_pred, sample_weight=weights)),
        "recall": float(recall_score(y_true, y_pred, sample_weight=weights)),
        "brier": float(brier_score_loss(y_true, y_prob, sample_weight=weights)),
        "log_loss": float(log_loss(y_true, y_prob, sample_weight=weights)),
    }

weighted = compute_metrics(y_val, y_prob, y_pred, weights=w_val)
unweighted = compute_metrics(y_val, y_prob, y_pred, weights=None)

# Assert they differ (FACTOR07 is actually applied)
assert weighted["pr_auc"] != unweighted["pr_auc"], "Weighted == unweighted!"
```

### Pattern 4: Threshold Analysis at Fixed Points

**What:** Report metrics at 5 fixed thresholds (0.3, 0.4, 0.5, 0.6, 0.7) plus the optimal threshold.

**Example:**
```python
from sklearn.metrics import precision_recall_curve

# Get PR curve with weights
precision, recall, thresholds = precision_recall_curve(
    y_val, y_prob, sample_weight=w_val
)

# Compute F1 at each threshold
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[optimal_idx])

# Report at fixed thresholds
FIXED_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_analysis = []
for t in FIXED_THRESHOLDS:
    y_pred_t = (y_prob >= t).astype(int)
    metrics_t = compute_metrics(y_val, y_prob, y_pred_t, weights=w_val)
    metrics_t["threshold"] = t
    threshold_analysis.append(metrics_t)
```

### Pattern 5: Prediction Storage as Parquet

**What:** Save per-row predictions with identifying columns for downstream Phase 8 (fairness) and Phase 9 (SHAP).

**Example:**
```python
# Build prediction DataFrame in Polars
pred_df = val_df.select(
    ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year", "UBIGEO",
     "FACTOR07", "dropout"]
).with_columns([
    pl.Series("prob_dropout", y_prob),
    pl.Series("pred_dropout", y_pred),
    pl.lit("logistic_regression").alias("model"),
    pl.lit(optimal_threshold).alias("threshold"),
])
pred_df.write_parquet("data/processed/predictions_lr.parquet")
```

### Anti-Patterns to Avoid

- **Converting entire DataFrame to pandas then numpy:** Only convert the specific columns needed at the sklearn boundary. Use `df.select(MODEL_FEATURES).to_numpy()`.
- **Training with FACTOR07 as both class_weight AND sample_weight:** These multiply. If using `class_weight='balanced'`, FACTOR07 as `sample_weight` already adjusts for population representation. Using both is valid per the spec requirement but understand the interaction.
- **Forgetting to cast Boolean dropout to numeric:** Polars Boolean does not convert to numpy int automatically in all paths. Always `.cast(pl.Int8).to_numpy()`.
- **Using freq_weights for inference when data has non-integer weights:** `freq_weights` in statsmodels works with non-integer values (verified) but inflates the effective sample size to the sum of weights (~25M for training set), making ALL p-values effectively 0. This is technically correct for survey-weighted inference but means p-values are not useful for variable selection -- use odds ratios and coefficient signs for interpretation instead.
- **Fitting statsmodels without add_constant:** statsmodels does NOT add an intercept by default, unlike sklearn. Always use `sm.add_constant(X)`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PR-AUC computation | Manual trapezoidal integration | `average_precision_score(sample_weight=w)` | Handles ties, sample weights, edge cases |
| Precision-recall curve | Manual threshold loop | `precision_recall_curve(sample_weight=w)` | Returns precision/recall/thresholds arrays with weight support |
| PR curve visualization | Manual matplotlib curve | `PrecisionRecallDisplay.from_predictions(sample_weight=w)` | Handles styling, chance level, labels |
| Threshold-optimal F1 | Grid search over thresholds | Compute from `precision_recall_curve` output | All thresholds evaluated in one pass |
| Standard errors/p-values | Hessian computation on sklearn model | `statsmodels.GLM(Binomial, freq_weights=w).fit()` | Correct inference with survey weights |
| Odds ratios | Manual exp(coef) | `np.exp(glm_result.params)` | Trivial but use glm_result params to get intercept-inclusive |
| Model persistence | Custom pickle | `joblib.dump(model, path)` / `joblib.load(path)` | Optimized for numpy arrays in sklearn models |
| Weighted Brier score | Manual MSE with weights | `brier_score_loss(sample_weight=w)` | Handles edge cases, pos_label |

**Key insight:** sklearn 1.8.0 has comprehensive `sample_weight` support across ALL metric functions. There is zero need for custom weighted metric implementations.

## Common Pitfalls

### Pitfall 1: Boolean Dropout Column

**What goes wrong:** Passing Polars Boolean column directly to sklearn causes type errors or silent misinterpretation.
**Why it happens:** Polars `Boolean` dtype does not map cleanly to numpy int in all conversion paths.
**How to avoid:** Always cast: `df["dropout"].cast(pl.Int8).to_numpy()` or `.cast(pl.Float64).to_numpy()`.
**Warning signs:** Shape mismatch errors, `ValueError: y should be a 1d array`, or unexpected metric values.

### Pitfall 2: sklearn vs statsmodels Coefficient Order

**What goes wrong:** sklearn stores coefficients as `coef_` (no intercept) + `intercept_` separately. statsmodels stores params as `[intercept, feature1, feature2, ...]` when `add_constant` is used.
**Why it happens:** Different conventions between the two libraries.
**How to avoid:** When building the coefficient table for JSON/console output, use statsmodels params with feature names = ["intercept"] + MODEL_FEATURES. Verify by checking that sklearn `coef_[0]` matches statsmodels `params[1]` (first non-intercept).
**Warning signs:** Coefficients off by one position, intercept mixed into feature coefficients.

### Pitfall 3: class_weight='balanced' with Survey Weights

**What goes wrong:** The interaction between class_weight and sample_weight is multiplicative. With ~14% dropout rate, balanced weights multiply minority class by ~3.5x. Combined with FACTOR07 (1-2039 range), effective weights can be extreme.
**Why it happens:** sklearn multiplies class_weight * sample_weight per-sample.
**How to avoid:** This is the intended behavior per MODL-02 ("class_weight='balanced'"). But document the interaction. The logistic regression should still converge -- the solver handles large weight ranges. If convergence issues arise, increase `max_iter`.
**Warning signs:** ConvergenceWarning from sklearn, unreasonable coefficient magnitudes.

### Pitfall 4: Year Split Leakage

**What goes wrong:** Accidentally including 2022 data in training or 2023 data in validation.
**Why it happens:** Off-by-one errors in year filtering, especially when adapting from spec (which used different years).
**How to avoid:** Assert year ranges after splitting: `assert train_df["year"].max() == 2021`, `assert val_df["year"].unique().to_list() == [2022]`, `assert test_df["year"].unique().to_list() == [2023]`.
**Warning signs:** Overlap in year sets, unusually high validation performance.

### Pitfall 5: Inflated P-values from freq_weights

**What goes wrong:** All p-values are 0.0000 because freq_weights inflates effective sample size to ~25 million (sum of FACTOR07 in training set).
**Why it happens:** Survey expansion weights are large (mean ~269, sum ~25M for training set). With 25M effective observations, even tiny effects are "significant."
**How to avoid:** This is expected behavior. Do NOT interpret p-values as meaningful for variable selection. Instead, focus on: (1) coefficient signs (directional correctness), (2) odds ratios (effect sizes), (3) confidence interval widths (relative precision). Document this in model_results.json metadata.
**Warning signs:** Every feature shows p < 0.001 regardless of effect size.

### Pitfall 6: LogisticRegression Convergence with max_iter=100

**What goes wrong:** ConvergenceWarning with default max_iter=100, especially with 98K training rows and 25 features.
**Why it happens:** LBFGS needs more iterations for large datasets with class_weight='balanced'.
**How to avoid:** Set `max_iter=1000` (or even 5000). The training set is only ~98K rows x 25 features -- convergence should still be fast. Capture the `n_iter_` attribute to report actual iterations in metadata.
**Warning signs:** sklearn ConvergenceWarning, unstable coefficients.

## Code Examples

### Full Temporal Split Creation

```python
# Source: verified against actual data
import polars as pl

df = pl.read_parquet("data/processed/enaho_with_features.parquet")

# Year-based temporal splits per CONTEXT.md
train_df = df.filter(pl.col("year").is_in([2018, 2019, 2020, 2021]))
val_df = df.filter(pl.col("year") == 2022)
test_df = df.filter(pl.col("year") == 2023)

# Verify no overlap
train_years = set(train_df["year"].unique().to_list())
val_years = set(val_df["year"].unique().to_list())
test_years = set(test_df["year"].unique().to_list())
assert train_years == {2018, 2019, 2020, 2021}
assert val_years == {2022}
assert test_years == {2023}
assert not (train_years & val_years)
assert not (train_years & test_years)
assert not (val_years & test_years)

# Verify complete partition
assert train_df.height + val_df.height + test_df.height == df.height

# Expected sizes (from actual data):
# Train: 98,023 rows (2018=30,559 + 2019=28,030 + 2020=13,755 + 2021=25,679)
# Val:   26,477 rows (2022)
# Test:  25,635 rows (2023)
```

### Sklearn Metric Function Signatures (verified 1.8.0)

```python
# ALL of these accept sample_weight as keyword argument:
average_precision_score(y_true, y_score, *, sample_weight=None)
roc_auc_score(y_true, y_score, *, sample_weight=None)
f1_score(y_true, y_pred, *, sample_weight=None)
precision_score(y_true, y_pred, *, sample_weight=None)
recall_score(y_true, y_pred, *, sample_weight=None)
brier_score_loss(y_true, y_proba, *, sample_weight=None)
log_loss(y_true, y_pred, *, sample_weight=None)
precision_recall_curve(y_true, y_score, *, sample_weight=None)

# Visualization also supports sample_weight:
PrecisionRecallDisplay.from_predictions(y_true, y_score, *, sample_weight=None)
```

### Statsmodels GLM for Coefficient Inference

```python
# Source: verified in environment (statsmodels 0.14.6)
import statsmodels.api as sm
import numpy as np

feature_names = ["intercept"] + list(MODEL_FEATURES)  # 26 total

X_const = sm.add_constant(X_train)
glm = sm.GLM(
    y_train.astype(float),
    X_const,
    family=sm.families.Binomial(),
    freq_weights=w_train,
)
glm_result = glm.fit()

# Build coefficient table
coef_table = []
for i, name in enumerate(feature_names):
    coef_table.append({
        "feature": name,
        "coefficient": round(float(glm_result.params[i]), 6),
        "std_error": round(float(glm_result.bse[i]), 6),
        "odds_ratio": round(float(np.exp(glm_result.params[i])), 6),
        "p_value": round(float(glm_result.pvalues[i]), 6),
        "ci_lower": round(float(glm_result.conf_int()[i, 0]), 6),
        "ci_upper": round(float(glm_result.conf_int()[i, 1]), 6),
    })
```

### Model Persistence with joblib

```python
# Source: sklearn 1.8.0 documentation
import joblib

# Save model + metadata
joblib.dump(lr, "data/processed/model_lr.joblib")

# Load model
lr_loaded = joblib.load("data/processed/model_lr.joblib")
assert hasattr(lr_loaded, "predict_proba")
```

### model_results.json Schema

```json
{
  "logistic_regression": {
    "metadata": {
      "model_type": "LogisticRegression",
      "train_years": [2018, 2019, 2020, 2021],
      "validate_year": 2022,
      "test_year": 2023,
      "n_train": 98023,
      "n_validate": 26477,
      "n_test": 25635,
      "n_features": 25,
      "feature_names": ["age", "is_secundaria_age", "..."],
      "class_weight": "balanced",
      "solver": "lbfgs",
      "max_iter": 1000,
      "C": 1.0,
      "n_iter_actual": 42,
      "convergence": true,
      "year_shift_note": "ENAHO 2024 unavailable; train/val/test shifted back by 1 year from spec",
      "covid_note": "2020 has reduced sample (~13,755 rows) due to COVID phone interviews"
    },
    "metrics": {
      "validate_2022": {
        "weighted": {
          "pr_auc": 0.XXX,
          "roc_auc": 0.XXX,
          "f1": 0.XXX,
          "precision": 0.XXX,
          "recall": 0.XXX,
          "brier": 0.XXX,
          "log_loss": 0.XXX
        },
        "unweighted": {
          "pr_auc": 0.XXX,
          "roc_auc": 0.XXX,
          "f1": 0.XXX,
          "precision": 0.XXX,
          "recall": 0.XXX,
          "brier": 0.XXX,
          "log_loss": 0.XXX
        }
      },
      "test_2023": {
        "weighted": { "...": "..." },
        "unweighted": { "...": "..." }
      }
    },
    "threshold_analysis": {
      "optimal_threshold": 0.XXX,
      "optimization_target": "max_weighted_f1",
      "thresholds": [
        {
          "threshold": 0.3,
          "weighted_f1": 0.XXX,
          "weighted_precision": 0.XXX,
          "weighted_recall": 0.XXX,
          "unweighted_f1": 0.XXX,
          "unweighted_precision": 0.XXX,
          "unweighted_recall": 0.XXX
        }
      ]
    },
    "coefficients": [
      {
        "feature": "intercept",
        "coefficient": 0.XXX,
        "std_error": 0.XXX,
        "odds_ratio": 0.XXX,
        "p_value": 0.XXX,
        "ci_lower": 0.XXX,
        "ci_upper": 0.XXX
      }
    ]
  }
}
```

## Discretion Recommendations

### Threshold Optimization Target

**Recommendation: Max weighted F1.**

Rationale: For an equity audit, F1 balances precision and recall, which matters because both false positives (wrongly flagging a student as at-risk) and false negatives (missing a truly at-risk student) have consequences. The alternative (max recall at precision floor) is more appropriate when the cost of false negatives dramatically exceeds false positives, which is not yet established at this baseline stage. F1 is also the spec's stated target (MODL-06: "Tune threshold per model using F1 on validation").

### LogisticRegression Hyperparameters

**Recommendation:**
- `solver="lbfgs"` -- default, efficient for L2 regularization, supports multiclass but we only need binary
- `max_iter=1000` -- 10x default; prevents convergence warnings with 98K rows
- `C=1.0` -- default regularization strength; no need to tune for a baseline
- `class_weight="balanced"` -- required by MODL-02; adjusts for ~14% minority class
- `random_state=42` -- reproducibility

Rationale: This is a baseline model. The goal is pipeline validation and interpretable coefficients, not maximum performance. Default hyperparameters are appropriate. Phase 6 does the serious hyperparameter tuning with Optuna.

### PR Curve Styling

**Recommendation:** Follow Phase 4's matplotlib pattern (Agg backend, consistent figure size). Include:
- PR curve with weighted sample_weight
- Chance level line (horizontal at ~0.14, the base dropout rate)
- Markers at the 5 fixed thresholds
- Optimal threshold highlighted
- Title: "Precision-Recall Curve: Logistic Regression (Validation 2022)"
- Save as `data/exports/figures/pr_curve_lr.png`

### Confusion Matrix Visualization

**Recommendation: Skip.** A confusion matrix at a single threshold adds little value for the human review gate. The threshold analysis table already shows precision/recall at 5 thresholds. The PR curve visualization is more informative. If the reviewer needs it, it can be added in a follow-up.

## Data Characteristics (verified from actual data)

| Split | Rows | Years | FACTOR07 Sum | Unweighted Dropout Rate |
|-------|------|-------|-------------|------------------------|
| Train | 98,023 | 2018-2021 | 25,298,930 | 17.24% |
| Validate | 26,477 | 2022 | 7,512,606 | 14.39% |
| Test | 25,635 | 2023 | 7,517,741 | 13.65% |
| Total | 150,135 | 2018-2023 | 40,329,277 | 16.12% |

**Feature matrix:** 25 features, all numeric (Int32/Int64/Float64), zero nulls.

**Dropout column:** Boolean dtype in Polars. Must cast to Int8/Float64 before numpy conversion.

**FACTOR07:** Float64, range [1.09, 2039.02], mean 268.62, non-integer survey expansion weights.

**Dropout distribution by training year:**
- 2018: ~30,559 rows (normal)
- 2019: ~28,030 rows (normal)
- 2020: ~13,755 rows (reduced -- COVID phone interviews)
- 2021: ~25,679 rows (normal)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual PR curve with matplotlib | `PrecisionRecallDisplay.from_predictions()` | sklearn 1.2+ | Handles sample_weight, chance level, styling |
| `precision_recall_curve` without drop_intermediate | `drop_intermediate=False` (default) | sklearn 1.3 | Parameter added; False is correct for threshold analysis |
| pickle for model persistence | joblib.dump/load | Long-standing | Better numpy array serialization |
| Manual threshold grid search | Compute from `precision_recall_curve` output | N/A | All thresholds in one pass |

**Deprecated/outdated:**
- sklearn's `TunedThresholdClassifierCV` (new in 1.5+) exists but is overly complex for our use case of reporting at 5 fixed thresholds.

## Open Questions

1. **class_weight='balanced' with survey weights interaction**
   - What we know: They multiply. class_weight adjusts for the ~14% minority class (~3.5x), FACTOR07 adjusts for sampling design (1-2039x). The combined effect can produce extreme per-sample weights.
   - What's unclear: Whether using class_weight='balanced' on survey data is methodologically appropriate, since the ~14% dropout rate may already be the true population rate after FACTOR07 weighting.
   - Recommendation: Use class_weight='balanced' as specified by MODL-02. The LR baseline exists to validate the pipeline and provide interpretable coefficients, not to be methodologically perfect. Document the interaction. If coefficients look unreasonable, test without class_weight='balanced' as a diagnostic.

2. **freq_weights p-value interpretation**
   - What we know: freq_weights inflates effective sample size to ~25M, making all p-values effectively 0.
   - What's unclear: Whether to report these p-values as-is or normalize the weights first.
   - Recommendation: Report p-values as-is (they are technically correct for the effective sample size) but add a metadata note explaining that with survey weights, p-values are uninformative for variable selection. Focus human review on odds ratios and coefficient signs.

3. **Validate metrics key naming**
   - What we know: Roadmap says "validate_2023" but with the year shift, validation is now 2022.
   - What's unclear: Whether to keep "validate_2023" for spec compatibility or use "validate_2022" for accuracy.
   - Recommendation: Use `"validate_2022"` and `"test_2023"` in model_results.json. Accuracy is more important than matching the spec's original year labeling. The metadata section documents the year shift.

## Sources

### Primary (HIGH confidence)
- sklearn 1.8.0 official docs -- verified average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss, log_loss, precision_recall_curve, LogisticRegression, PrecisionRecallDisplay all support sample_weight
- statsmodels 0.14.6 -- verified GLM(Binomial, freq_weights) provides params, bse, pvalues, conf_int() (tested in environment)
- Actual data inspection -- verified FACTOR07 range [1.09, 2039.02], split sizes, feature dtypes, zero nulls

### Secondary (MEDIUM confidence)
- sklearn docs: class_weight and sample_weight multiply (stated in LogisticRegression docs)
- sklearn docs: joblib is recommended for model persistence

### Tertiary (LOW confidence)
- class_weight='balanced' with survey weights: methodological appropriateness not fully established in literature

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified in environment with exact version numbers
- Architecture: HIGH -- dual-model pattern tested, Polars-to-numpy conversion verified, metric functions API confirmed
- Pitfalls: HIGH -- Boolean dtype, convergence, p-value inflation all verified through actual testing
- Data characteristics: HIGH -- all numbers from actual data inspection (not estimates)

**Research date:** 2026-02-08
**Valid until:** 2026-03-08 (sklearn/statsmodels stable, low change rate)
