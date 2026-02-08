# Phase 7: Calibration + ONNX Export + Final Test - Research

**Researched:** 2026-02-08
**Domain:** Model calibration, ONNX export, final evaluation
**Confidence:** HIGH

## Summary

Phase 7 calibrates the LightGBM model (distorted by scale_pos_weight=4.80), exports it to ONNX for browser inference, and evaluates on the held-out 2023 test set exactly once. Research verified all three pillars end-to-end with the actual project model and data.

**Key findings:**
- `cv='prefit'` is **removed** in sklearn 1.8.0 -- use `FrozenEstimator` from `sklearn.frozen` instead. The sample_weight warning from FrozenEstimator is benign (weights are correctly passed to the calibration logistic regression, not to the frozen model's non-existent fit).
- Platt scaling (sigmoid) reduces Brier score by ~45% on validation data (0.2115 to 0.1157), confirming calibration is critical.
- ONNX conversion produces a 0.10 MB model with max absolute prediction difference of 1.21e-07 vs Python (well within 1e-4 tolerance).
- For browser inference, export the uncalibrated LightGBM to ONNX and apply Platt scaling post-hoc in JavaScript using the extracted A/B parameters: `calibrated = 1 / (1 + exp(A * raw_prob + B))`.
- Alerta Escuela published metrics: ROC-AUC 0.84-0.89, FNR 57-64%. No published precision, PR-AUC, or fairness analysis.

**Primary recommendation:** Use `CalibratedClassifierCV(FrozenEstimator(lgbm), method='sigmoid')` for calibration, `convert_lightgbm()` from onnxmltools for ONNX export with `zipmap=False`, and `onnxruntime.InferenceSession` for validation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- LightGBM is the primary model (best_iteration=79, val PR-AUC=0.2611)
- scale_pos_weight distorts probabilities -- calibration is critical for meaningful risk scores
- ENAHO 2024 not available -- test set is 2023 (already split in Phase 5 as TEST_YEAR=2023)
- ONNX export targets browser inference for M4 scrollytelling site
- Skipped discussion (straightforward phase building on Phase 6 outputs)

### Claude's Discretion
- No explicit discretion areas defined (straightforward phase)

### Deferred Ideas (OUT OF SCOPE)
- None defined
</user_constraints>

## Standard Stack

The established libraries for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | CalibratedClassifierCV, FrozenEstimator, CalibrationDisplay, brier_score_loss | Already installed; gold standard for calibration |
| onnxmltools | 1.16.0 | convert_lightgbm() for LightGBM -> ONNX | Already installed; official ONNX converter for LightGBM |
| onnxruntime | 1.24.1 | InferenceSession for ONNX validation | Already installed; reference ONNX inference engine |
| onnx | 1.20.1 | ONNX model format, save/load | Already installed; model serialization |
| lightgbm | 4.6.0 | Trained model (loaded from joblib) | Already installed; upstream dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| joblib | (bundled) | Load model_lgbm.joblib | Loading persisted model |
| matplotlib | (installed) | Calibration plot, PR curve | Reliability diagram generation |
| numpy | (installed) | Array operations, float32 casting | ONNX inference requires float32 input |
| polars | (installed) | Data loading, DataFrame operations | Feature matrix, predictions export |

### Not Needed
| Library | Why Not |
|---------|---------|
| onnxconverter_common | Not installed; FloatTensorType available via `onnxmltools.convert.common.data_types` |
| skl2onnx | Not needed for direct LightGBM conversion; onnxmltools handles it |

**Installation:** No new packages needed. All required libraries are already in pyproject.toml.

## Architecture Patterns

### Recommended Project Structure
```
src/models/
    calibration.py          # New file: calibration + ONNX + final test pipeline
    baseline.py             # Existing: temporal splits, compute_metrics, _threshold_analysis
    lightgbm_xgboost.py     # Existing: LightGBM pipeline patterns
data/processed/
    model_lgbm.joblib       # Input: trained LightGBM model
    model_lgbm_calibrated.joblib  # Output: calibrated model
    predictions_lgbm_calibrated.parquet  # Output: calibrated predictions
data/exports/
    onnx/
        lightgbm_dropout.onnx  # Output: ONNX model for browser
    figures/
        calibration_plot.png   # Output: reliability diagram
    model_results.json         # Updated with calibrated + test entries
```

### Pattern 1: Calibration with FrozenEstimator
**What:** Calibrate a pre-trained model without re-fitting it
**When to use:** When the model is already trained and you want to calibrate probabilities on held-out data
**Critical:** `cv='prefit'` is REMOVED in sklearn 1.8.0. Use FrozenEstimator instead.

```python
# Source: sklearn 1.8.0 official docs + verified locally
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

# Load pre-trained LightGBM
lgbm = joblib.load("data/processed/model_lgbm.joblib")

# Wrap in FrozenEstimator (prevents re-fitting during calibration)
frozen = FrozenEstimator(lgbm)

# Calibrate on validation data with survey weights
cal_model = CalibratedClassifierCV(frozen, method="sigmoid")
cal_model.fit(X_val, y_val, sample_weight=w_val)
# NOTE: Warning about FrozenEstimator not accepting sample_weight is BENIGN.
# The sample_weight IS passed to the calibration logistic regression (Platt scaling).
# The FrozenEstimator's model is NOT re-fit (which is correct behavior).

# Get calibrated probabilities
cal_proba = cal_model.predict_proba(X_new)[:, 1]
```

### Pattern 2: ONNX Export with onnxmltools
**What:** Convert LightGBM to ONNX for browser inference
**When to use:** Exporting model for cross-platform inference

```python
# Source: onnxmltools docs + verified locally
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx

# Define input shape: [batch, n_features]
initial_types = [("input", FloatTensorType([None, 25]))]  # 25 features

# Convert with zipmap=False to get tensor output (not dict)
onnx_model = convert_lightgbm(
    lgbm,
    initial_types=initial_types,
    zipmap=False,  # CRITICAL: output probabilities as array, not dict
)

# Save
onnx.save(onnx_model, "data/exports/onnx/lightgbm_dropout.onnx")
```

### Pattern 3: ONNX Inference Validation
**What:** Verify ONNX predictions match Python model
**When to use:** After ONNX export, before deployment

```python
# Source: onnxruntime docs + verified locally
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession(
    "data/exports/onnx/lightgbm_dropout.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = sess.get_inputs()[0].name  # "input"
# Output names: ["label", "probabilities"]

# Input MUST be float32
X_float32 = X_sample.astype(np.float32)
result = sess.run(None, {input_name: X_float32})

labels = result[0]       # int64 array of predicted classes
probas = result[1][:, 1] # float32 array of P(dropout=1)

# Compare with Python
py_proba = lgbm.predict_proba(X_sample)[:, 1]
max_diff = np.max(np.abs(py_proba - probas))
assert max_diff < 1e-4, f"ONNX predictions differ: max_diff={max_diff}"
```

### Pattern 4: Calibration Plot (Reliability Diagram)
**What:** Visualize before/after calibration quality
**When to use:** Human gate review artifact

```python
# Source: sklearn 1.8.0 CalibrationDisplay docs
from sklearn.calibration import CalibrationDisplay

fig, ax = plt.subplots(figsize=(8, 6))

# Plot uncalibrated
CalibrationDisplay.from_predictions(
    y_val, uncal_proba, n_bins=10, strategy="uniform",
    name="LightGBM (uncalibrated)", ax=ax,
)

# Plot calibrated
CalibrationDisplay.from_predictions(
    y_val, cal_proba, n_bins=10, strategy="uniform",
    name="LightGBM (calibrated)", ax=ax,
)

ax.set_title("Calibration Curve: Before vs After Platt Scaling")
plt.tight_layout()
fig.savefig("data/exports/figures/calibration_plot.png", dpi=150)
plt.close(fig)
```

### Pattern 5: Browser-Side Platt Scaling
**What:** Apply calibration in JavaScript after ONNX inference
**When to use:** M4 scrollytelling site needs calibrated probabilities

```python
# Extract Platt parameters from calibrated model
calibrator = cal_model.calibrated_classifiers_[0].calibrators[0]
platt_a = calibrator.a_  # slope
platt_b = calibrator.b_  # intercept

# Export parameters for JavaScript
# JS formula: calibrated = 1 / (1 + Math.exp(A * raw_prob + B))
```

```javascript
// Browser-side calibration (after ONNX inference)
function calibrate(rawProb) {
    const A = -5.278337;  // from Python extraction
    const B = 4.276521;
    return 1.0 / (1.0 + Math.exp(A * rawProb + B));
}
```

### Anti-Patterns to Avoid
- **Using `cv='prefit'`:** Removed in sklearn 1.8.0. Will throw `InvalidParameterError`.
- **Converting CalibratedClassifierCV to ONNX:** The calibrated wrapper is not directly convertible. Export raw LightGBM + Platt parameters separately.
- **Skipping `zipmap=False`:** Default ZipMap outputs dict-of-probabilities which is harder to work with and incompatible with some ONNX runtimes.
- **Using float64 input for ONNX:** ONNX models expect float32. Cast input arrays with `.astype(np.float32)`.
- **Evaluating test set more than once:** The test set (2023) must be touched EXACTLY ONCE. All threshold tuning happens on validation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Probability calibration | Custom sigmoid fitting | `CalibratedClassifierCV` | Handles edge cases, clipping, numerical stability |
| LightGBM to ONNX conversion | Manual tree serialization | `onnxmltools.convert.convert_lightgbm()` | Handles all LightGBM internals (boosting, leaves, thresholds) |
| ONNX inference | Custom tree traversal | `onnxruntime.InferenceSession` | Optimized C++ runtime, matches reference implementation |
| Calibration visualization | Custom matplotlib from scratch | `CalibrationDisplay.from_predictions()` | Handles binning, reference line, formatting |
| Brier score computation | Manual MSE of probabilities | `sklearn.metrics.brier_score_loss()` | Handles sample_weight correctly |

**Key insight:** The entire pipeline chains existing library calls. The only custom code needed is the pipeline orchestration, JSON updates, and Alerta Escuela comparison table printing.

## Common Pitfalls

### Pitfall 1: cv='prefit' Removed in sklearn 1.8
**What goes wrong:** `InvalidParameterError: The 'cv' parameter of CalibratedClassifierCV must be an int...`
**Why it happens:** sklearn 1.8.0 removed `cv='prefit'` (deprecated since 1.6).
**How to avoid:** Use `FrozenEstimator` wrapper: `CalibratedClassifierCV(FrozenEstimator(model))`
**Warning signs:** Import error or InvalidParameterError at runtime.

### Pitfall 2: FrozenEstimator sample_weight Warning (Benign)
**What goes wrong:** Warning: "Since FrozenEstimator does not appear to accept sample_weight, sample weights will only be used for the calibration itself."
**Why it happens:** FrozenEstimator blocks all parameter passing to prevent re-fitting.
**How to avoid:** This warning is EXPECTED and CORRECT. The sample_weight IS used for the Platt scaling logistic regression, which is the calibration step. The LightGBM model was already trained with weights.
**Warning signs:** None -- this is desired behavior. Suppress the specific warning if noisy.

### Pitfall 3: ONNX Float32 Precision
**What goes wrong:** Prediction differences between Python (float64) and ONNX (float32).
**Why it happens:** LightGBM uses float64 internally; ONNX operates in float32.
**How to avoid:** Verified empirically: max absolute diff is ~1.2e-07 for the actual model. The 1e-4 tolerance is easily met.
**Warning signs:** If max_diff exceeds 1e-5, investigate model complexity or numerical edge cases.

### Pitfall 4: LightGBM feature_name_ Attribute
**What goes wrong:** `AttributeError: 'LGBMClassifier' object has no attribute 'feature_names_'`
**Why it happens:** LightGBM uses `feature_name_` (singular), not sklearn's `feature_names_in_`.
**How to avoid:** Use `model.n_features_in_` for count and `model.feature_name_` for names.
**Warning signs:** Attribute error when accessing model metadata.

### Pitfall 5: Test Set Touched Multiple Times
**What goes wrong:** Data leakage from using test metrics for any decision-making.
**Why it happens:** Accidentally evaluating on test before calibration is finalized.
**How to avoid:** Calibrate entirely on validation data. Only after calibration is frozen, run test evaluation once. Structure the pipeline to enforce this order.
**Warning signs:** Any code that computes test metrics before calibration parameters are set.

### Pitfall 6: Calibrating on Training Data
**What goes wrong:** Overfitting the calibration to training data, making Brier score artificially good.
**Why it happens:** Using X_train, y_train for calibration instead of held-out data.
**How to avoid:** Use validation set (2022) for calibration. This is the same split used for threshold tuning.
**Warning signs:** Calibrated Brier on training much better than on validation.

### Pitfall 7: ONNX ZipMap Output Format
**What goes wrong:** ONNX inference returns list of dicts instead of numpy array for probabilities.
**Why it happens:** Default `zipmap=True` in `convert_lightgbm()`.
**How to avoid:** Always use `zipmap=False` when converting. Output will be a float32 tensor of shape (n_samples, n_classes).
**Warning signs:** `result[1]` is a list of dicts like `[{0: 0.9, 1: 0.1}, ...]`.

## Code Examples

### Complete Calibration Pipeline
```python
# Source: Verified locally with actual project model and data
import joblib
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

# 1. Load model and data
lgbm = joblib.load("data/processed/model_lgbm.joblib")
df = pl.read_parquet("data/processed/enaho_with_features.parquet")
_, val_df, test_df = create_temporal_splits(df)
X_val, y_val, w_val = _df_to_numpy(val_df)

# 2. Get uncalibrated validation predictions
uncal_proba_val = lgbm.predict_proba(X_val)[:, 1]
brier_uncal = brier_score_loss(y_val, uncal_proba_val, sample_weight=w_val)

# 3. Calibrate with Platt scaling on validation
frozen = FrozenEstimator(lgbm)
cal_model = CalibratedClassifierCV(frozen, method="sigmoid")
cal_model.fit(X_val, y_val, sample_weight=w_val)

# 4. Get calibrated validation predictions
cal_proba_val = cal_model.predict_proba(X_val)[:, 1]
brier_cal = brier_score_loss(y_val, cal_proba_val, sample_weight=w_val)
assert brier_cal < brier_uncal, "Calibration must improve Brier score"

# 5. ONCE: Evaluate on test set
X_test, y_test, w_test = _df_to_numpy(test_df)
cal_proba_test = cal_model.predict_proba(X_test)[:, 1]
uncal_proba_test = lgbm.predict_proba(X_test)[:, 1]
# ... compute all metrics for both calibrated and uncalibrated ...
```

### Complete ONNX Export and Validation
```python
# Source: Verified locally with actual project model
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import onnxruntime as rt

# 1. Convert
initial_types = [("input", FloatTensorType([None, 25]))]
onnx_model = convert_lightgbm(lgbm, initial_types=initial_types, zipmap=False)
onnx_path = root / "data" / "exports" / "onnx" / "lightgbm_dropout.onnx"
onnx_path.parent.mkdir(parents=True, exist_ok=True)
onnx.save(onnx_model, str(onnx_path))

# 2. Validate size
size_bytes = onnx_path.stat().st_size
assert size_bytes < 50 * 1024 * 1024, f"ONNX too large: {size_bytes}"

# 3. Validate predictions
sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# Sample 100 random rows
rng = np.random.default_rng(42)
indices = rng.choice(X_val.shape[0], 100, replace=False)
X_sample = X_val[indices].astype(np.float32)

py_proba = lgbm.predict_proba(X_sample)[:, 1]
onnx_result = sess.run(None, {input_name: X_sample})
onnx_proba = onnx_result[1][:, 1]

max_diff = np.max(np.abs(py_proba - onnx_proba))
assert max_diff < 1e-4, f"ONNX predictions differ: {max_diff:.2e}"
```

### Alerta Escuela Comparison Table
```python
# Source: PROJECT.md published metrics
alerta_escuela_metrics = {
    "model": "Alerta Escuela (MINEDU)",
    "algorithm": "LightGBM",
    "data_source": "SIAGIE administrative",
    "roc_auc_range": "0.84-0.89",
    "fnr_range": "57-64%",
    "features": 31,
    "fairness_analysis": "None published",
    "notes": "Uses 5 administrative data sources; trained on SIAGIE records",
}

our_metrics = {
    "model": "Equity Audit Replication",
    "algorithm": "LightGBM (Optuna-tuned)",
    "data_source": "ENAHO survey",
    "roc_auc": f"{test_metrics['roc_auc']:.4f}",
    "pr_auc": f"{test_metrics['pr_auc']:.4f}",
    "fnr": f"{1 - test_metrics['recall']:.4f}",
    "features": 25,
    "fairness_analysis": "6 dimensions + 3 intersections (Phase 8)",
}

# Print comparison table for human review
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cv='prefit'` | `FrozenEstimator` wrapper | sklearn 1.6 (deprecated), 1.8 (removed) | Must use `from sklearn.frozen import FrozenEstimator` |
| `onnxconverter_common.FloatTensorType` | `onnxmltools.convert.common.data_types.FloatTensorType` | Current | Either import works, but onnxmltools path avoids extra dependency |
| ZipMap output (dict) | `zipmap=False` (tensor) | Always available | Use False for array output instead of dict-of-probabilities |
| `CalibrationDisplay` | `CalibrationDisplay.from_predictions()` | sklearn 1.0+ | Preferred over manual calibration_curve + matplotlib |

**Deprecated/outdated:**
- `cv='prefit'`: Removed in sklearn 1.8.0. Use `FrozenEstimator` instead.
- `onnxconverter_common` package: Not installed in this project. Use `onnxmltools.convert.common.data_types` instead.

## Verified Baselines (from actual project data)

These numbers were verified by running the actual model against actual data during research:

| Metric | Value | Source |
|--------|-------|--------|
| LightGBM val Brier (uncalibrated) | 0.2115 | model_results.json |
| LightGBM val Brier (calibrated, Platt) | 0.1157 | Verified locally |
| Brier improvement | 45.3% | Computed |
| Platt A (slope) | -5.278337 | Extracted from _SigmoidCalibration |
| Platt B (intercept) | 4.276521 | Extracted from _SigmoidCalibration |
| ONNX model size | 0.10 MB (101,955 bytes) | Verified locally |
| ONNX max abs diff | 1.21e-07 | Verified on 100 samples |
| LightGBM val PR-AUC (weighted) | 0.2611 | model_results.json |
| LightGBM test PR-AUC (weighted) | 0.2378 | model_results.json |
| Val-Test PR-AUC gap | 0.0233 | Computed (well within 0.07 threshold) |

## Alerta Escuela Published Metrics

**Source:** PROJECT.md (extracted from MINEDU repository document)

| Metric | Alerta Escuela | Our Model (LightGBM) | Notes |
|--------|---------------|----------------------|-------|
| Algorithm | LightGBM | LightGBM (Optuna-tuned) | Same algorithm family |
| Data Source | SIAGIE administrative | ENAHO survey | Different data sources |
| ROC-AUC | 0.84-0.89 | 0.6505 (val) / 0.6314 (test) | Not directly comparable (different data) |
| FNR | 57-64% | ~60% (val) / ~62% (test) | Similar FNR range |
| Features | 31 | 25 | Both use socioeconomic + educational features |
| Fairness Analysis | None published | 6 dimensions + 3 intersections | Our key contribution |
| Calibration | Not reported | Platt scaling (45% Brier improvement) | Critical for meaningful risk scores |

**Important context for comparison:**
1. ROC-AUC is not directly comparable because different data sources produce different base rates (~14% survey vs ~2% administrative).
2. The FNR comparison is more meaningful -- both models miss 57-64% of actual dropouts.
3. Our PR-AUC (the more informative metric for imbalanced data) has no Alerta Escuela counterpart.
4. The fairness analysis (Phase 8) is the key differentiation, not predictive accuracy.

## Open Questions

1. **Platt vs Isotonic calibration:**
   - What we know: Platt (sigmoid) produces 45% Brier improvement with proper parametric form. Isotonic is non-parametric and can overfit with small calibration samples.
   - What's unclear: Whether isotonic would perform better given 26,477 validation samples (which is large enough to avoid isotonic overfitting).
   - Recommendation: Use sigmoid (Platt) as primary since it's more interpretable and the Platt parameters (A, B) can be easily applied in JavaScript. Try isotonic as comparison and report both in model_results.json.

2. **model_results.json key naming:**
   - What we know: Context says keys should be `test_2024_final` and `test_2024_calibrated`, but the actual test year is 2023.
   - What's unclear: Whether to use `test_2023_final` / `test_2023_calibrated` (matching actual year) or `test_2024_final` / `test_2024_calibrated` (matching spec language).
   - Recommendation: Use `test_2023_final` and `test_2023_calibrated` to match reality (the temporal split uses TEST_YEAR=2023). Add a note in the metadata explaining the year shift.

3. **ONNX export: raw vs calibrated model:**
   - What we know: CalibratedClassifierCV cannot be directly exported to ONNX. The raw LightGBM can.
   - What's unclear: None -- the architecture is clear.
   - Recommendation: Export raw LightGBM to ONNX, include Platt A/B parameters in model_results.json for browser-side calibration.

## Sources

### Primary (HIGH confidence)
- sklearn 1.8.0 CalibratedClassifierCV documentation: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- sklearn 1.8.0 FrozenEstimator documentation: https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html
- sklearn 1.8.0 CalibrationDisplay documentation: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html
- onnxmltools GitHub + convert.py source: https://github.com/onnx/onnxmltools
- sklearn-onnx LightGBM classifier tutorial: https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm.html
- **Local verification:** All patterns tested end-to-end with actual project model (model_lgbm.joblib) and data (enaho_with_features.parquet)

### Secondary (MEDIUM confidence)
- Alerta Escuela MINEDU repository: https://repositorio.minedu.gob.pe/handle/20.500.12799/10990
- PROJECT.md published metrics: ROC-AUC 0.84-0.89, FNR 57-64%

### Tertiary (LOW confidence)
- Alerta Escuela detailed model metrics (from full PDF): Could not fetch (18.59 MB PDF exceeds size limit). Published precision/recall numbers not extracted.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified installed and working with correct versions
- Architecture: HIGH - End-to-end pipeline tested locally with actual model and data
- Pitfalls: HIGH - cv='prefit' removal, FrozenEstimator warning, float32 precision all verified empirically
- Alerta Escuela comparison: MEDIUM - High-level metrics from PROJECT.md confirmed, but detailed breakdowns not available

**Research date:** 2026-02-08
**Valid until:** 2026-03-08 (stable domain, all libraries verified at current versions)
