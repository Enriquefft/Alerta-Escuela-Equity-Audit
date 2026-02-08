# Phase 8: Subgroup Fairness Metrics - Research

**Researched:** 2026-02-08
**Domain:** ML fairness evaluation, fairlearn MetricFrame, survey-weighted subgroup analysis
**Confidence:** HIGH

## Summary

This phase computes comprehensive fairness metrics across 6 protected dimensions and 3 intersections using fairlearn's MetricFrame with FACTOR07 survey weights. The core technical challenge is that MetricFrame requires separate instances for binary-prediction metrics (TPR, FPR, FNR, precision) and probability-based metrics (PR-AUC), since the `y_pred` argument must be binary for the former and continuous for the latter. The `sample_params` API in fairlearn 0.13.0 uses nested dictionaries where each metric name maps to its own `{"sample_weight": weights}` dict -- there is no global sample_weight parameter.

A critical data constraint was discovered: calibrated probabilities max out at ~0.43 on the test set (Platt scaling compresses the range), so the spec's "high risk > 0.7" threshold for calibration-by-group analysis cannot use calibrated probabilities. The uncalibrated probabilities do reach 0.76 with 702 test observations above 0.7. The implementation should use uncalibrated probabilities for the calibration-by-group analysis (actual dropout rate among predicted high-risk), or alternatively define "high risk" relative to the calibrated distribution (e.g., top decile of calibrated probability).

**Primary recommendation:** Build `src/fairness/metrics.py` with a `compute_fairness_metrics()` function that (1) loads predictions + features, (2) runs two MetricFrames per dimension (binary + proba), (3) computes calibration-by-group, (4) computes gap metrics, and (5) exports `fairness_metrics.json` matching the M4 schema. Use the test set (2023) only. Use the calibrated model's threshold (0.165716) for binary predictions. Use uncalibrated probabilities for the "high risk > 0.7" calibration analysis.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fairlearn | 0.13.0 | MetricFrame for subgroup metrics | Already installed; official fairness toolkit |
| sklearn.metrics | (bundled) | recall_score, precision_score, average_precision_score | Standard metric implementations |
| numpy | (bundled) | Weighted averages, array ops | Standard numerical computing |
| polars | (bundled) | Data loading and joining | Project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | (bundled) | MetricFrame.by_group returns pandas DataFrame | Required for MetricFrame output |
| json | stdlib | Export fairness_metrics.json | Standard JSON serialization |

### Not Needed
| Library | Reason |
|---------|--------|
| aif360 | Heavier, fairlearn already handles everything needed |
| themis-ml | Redundant with fairlearn |

**Installation:** No new dependencies needed -- everything is already installed.

## Architecture Patterns

### Recommended Project Structure
```
src/
  fairness/
    __init__.py          # (already exists, empty)
    metrics.py           # NEW: compute_fairness_metrics() + helpers
tests/
  gates/
    test_gate_3_1.py     # NEW: fairness gate test
data/
  exports/
    fairness_metrics.json  # NEW: M4 schema output
```

### Pattern 1: Two-MetricFrame Approach for Mixed Metric Types
**What:** MetricFrame requires all metrics to use the same y_pred type (binary or continuous). Since TPR/FPR/FNR/precision need binary predictions but PR-AUC needs probabilities, use two separate MetricFrame instances and merge results.
**When to use:** Always -- this is the only correct approach.
**Confidence:** HIGH (verified empirically with fairlearn 0.13.0)
**Example:**
```python
# Source: Verified empirically against fairlearn 0.13.0
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score, average_precision_score

# MetricFrame 1: Binary prediction metrics
mf_binary = MetricFrame(
    metrics={
        'tpr': lambda y, p, sample_weight=None: recall_score(y, p, sample_weight=sample_weight, zero_division=np.nan),
        'precision': lambda y, p, sample_weight=None: precision_score(y, p, sample_weight=sample_weight, zero_division=np.nan),
        'fnr': lambda y, p, sample_weight=None: 1.0 - recall_score(y, p, sample_weight=sample_weight, zero_division=np.nan),
        'fpr': weighted_fpr,  # custom function, see below
    },
    y_true=y_true,
    y_pred=y_pred_binary,  # binary predictions at threshold
    sensitive_features=sensitive_features,
    sample_params={
        'tpr': {'sample_weight': weights},
        'precision': {'sample_weight': weights},
        'fnr': {'sample_weight': weights},
        'fpr': {'sample_weight': weights},
    }
)

# MetricFrame 2: Probability-based metrics
mf_proba = MetricFrame(
    metrics={'pr_auc': average_precision_score},
    y_true=y_true,
    y_pred=y_pred_proba,  # continuous probabilities
    sensitive_features=sensitive_features,
    sample_params={'pr_auc': {'sample_weight': weights}},
)

# Merge results
combined = pd.concat([mf_binary.by_group, mf_proba.by_group], axis=1)
```

### Pattern 2: Custom FPR with Sample Weights
**What:** sklearn does not have a direct `fpr_score` function. Must be implemented manually using weighted averages over true negatives.
**When to use:** Whenever FPR is needed with survey weights.
**Confidence:** HIGH (verified empirically)
**Example:**
```python
def weighted_fpr(y_true, y_pred, sample_weight=None):
    """False Positive Rate with optional sample weights."""
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_weight = np.asarray(sample_weight, dtype=float)
    negatives = (y_true == 0)
    if negatives.sum() == 0:
        return np.nan
    fp = negatives & (y_pred == 1)
    return float(np.average(fp[negatives], weights=sample_weight[negatives]))
```

### Pattern 3: sample_params Nesting (CRITICAL)
**What:** The `sample_params` dict must repeat `sample_weight` for EACH metric by name. There is no "global" sample_weight in fairlearn's API.
**When to use:** Always when passing sample_weight to MetricFrame.
**Confidence:** HIGH (verified from official docs + empirical testing)
**Example:**
```python
# CORRECT -- repeat for each metric
sample_params = {
    'tpr': {'sample_weight': weights},
    'fpr': {'sample_weight': weights},
    'fnr': {'sample_weight': weights},
    'precision': {'sample_weight': weights},
}

# WRONG -- no global sample_weight
# sample_params = {'sample_weight': weights}  # WILL NOT WORK
```

### Pattern 4: Intersectional Groups via DataFrame
**What:** Pass a pandas DataFrame with multiple columns as `sensitive_features` to get intersectional analysis automatically.
**When to use:** For the 3 mandatory intersections (language x rurality, sex x poverty, language x region).
**Confidence:** HIGH (verified empirically + official docs)
**Example:**
```python
import pandas as pd

# Create intersectional sensitive features
sensitive_df = pd.DataFrame({
    'language': language_labels,
    'rural': rural_labels,
})

mf = MetricFrame(
    metrics=metrics_dict,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_df,
    sample_params=sample_params,
)
# mf.by_group has MultiIndex (language, rural)
```

### Pattern 5: Data Loading Strategy
**What:** Load predictions parquet (has prob_dropout, pred_dropout, FACTOR07, dropout) and LEFT JOIN with feature matrix to get sensitive features (p300a_harmonized, region_natural, es_mujer, rural, poverty_quintile, etc.)
**When to use:** At the start of the fairness pipeline.
**Confidence:** HIGH (verified join produces 25,635 matched rows with 0 nulls)
**Example:**
```python
import polars as pl

pred = pl.read_parquet("data/processed/predictions_lgbm_calibrated.parquet")
feat = pl.read_parquet("data/processed/enaho_with_features.parquet")

# Filter to test set
test_pred = pred.filter(pl.col("split") == "test_2023")

join_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"]
meta_cols = ["p300a_harmonized", "p300a_original", "region_natural",
             "es_mujer", "rural", "es_peruano", "poverty_quintile",
             "age", "is_secundaria_age", "log_income", "department"]

merged = test_pred.join(
    feat.select(join_keys + meta_cols),
    on=join_keys,
    how="left",
)
assert merged.height == test_pred.height  # No row gain/loss
```

### Anti-Patterns to Avoid
- **Using y_pred_proba for binary metrics:** MetricFrame will silently compute recall_score with continuous values, producing nonsensical results. Always split into two MetricFrames.
- **Forgetting zero_division parameter:** Subgroups with 0 positive or 0 negative samples will trigger UndefinedMetricWarning and return 0.0. Use `zero_division=np.nan` to get NaN instead (clearer signaling).
- **Applying 0.7 high-risk threshold to calibrated probabilities:** Calibrated probs max at ~0.43 on test set. Zero observations would qualify, making the analysis vacuous.
- **Computing fairness on train or validation sets:** Must use TEST set (2023) only -- the held-out year never seen during training or threshold tuning.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Subgroup metric computation | Manual loops over groups | `fairlearn.MetricFrame` | Handles edge cases, NaN groups, multi-feature intersections |
| Equalized odds gap | Manual max-min over TPR/FPR | `fairlearn.metrics.equalized_odds_difference` | Handles sample_weight natively, tested implementation |
| FNR gap | Manual computation | `fairlearn.metrics.false_negative_rate_difference` | Built-in, handles sample_weight |
| FPR calculation | Confusion matrix math | Custom `weighted_fpr` function (above) | sklearn lacks `fpr_score`; the custom function is needed but simple |
| Group difference/ratio | Manual iteration | `MetricFrame.difference()` and `MetricFrame.ratio()` | Built-in methods with `between_groups` and `to_overall` options |

**Key insight:** fairlearn 0.13.0 provides most gap metrics as standalone functions AND as MetricFrame methods. Use MetricFrame for detailed per-group results, standalone functions for summary gap values.

## Common Pitfalls

### Pitfall 1: Calibrated Probability Range Compression
**What goes wrong:** Platt scaling compresses the probability range. On this dataset, calibrated probs are in [0.027, 0.431]. The spec's "high risk > 0.7" threshold yields 0 students if applied to calibrated probabilities.
**Why it happens:** Platt scaling (sigmoid transform) maps the uncalibrated range to a narrower, better-calibrated range. Since the true base rate is ~14%, even the highest-risk students don't reach 0.7 calibrated probability.
**How to avoid:** Use UNCALIBRATED probabilities for the "high risk > 0.7" calibration-by-group analysis (702 test observations qualify). OR redefine "high risk" as top decile of calibrated probability. Document the choice clearly.
**Data point:** Uncalibrated test probs: max=0.757, 702 above 0.7. Calibrated test probs: max=0.431, 0 above 0.7.

### Pitfall 2: Tiny Subgroups Producing Unreliable Metrics
**What goes wrong:** Several intersections and disaggregated language codes have <50 observations (aimara_urban=25, aimara_costa=8, quechua_costa=47, other_indigenous_sierra=2). Metrics for these groups are statistically unreliable.
**Why it happens:** ENAHO survey samples reflect population proportions; minority language speakers in non-traditional regions are rare.
**How to avoid:** Flag any group with <50 unweighted observations in the JSON output. Still compute metrics but add a `"flagged_small_sample": true` field. Use 100 as the minimum for primary dimensions (per spec).
**Warning signs:** NaN metrics (group has 0 positives or 0 negatives), extreme metric values (0.0 or 1.0).

### Pitfall 3: Boolean dtype for dropout
**What goes wrong:** The predictions parquet stores `dropout` as Boolean. Passing Boolean directly to sklearn metrics may cause unexpected behavior.
**Why it happens:** Polars stores boolean as Boolean dtype; numpy conversion may not produce int arrays.
**How to avoid:** Cast to Int8 before converting to numpy: `y_true = merged["dropout"].cast(pl.Int8).to_numpy()`
**Pattern established in:** Phase 5 baseline (`_df_to_numpy` casts `pl.Int8`).

### Pitfall 4: es_peruano Subgroup Has Only 27 Non-Peruvian Students
**What goes wrong:** With only 27 non-Peruvian students in the 2023 test set, metrics for `es_peruano=0` are unreliable (way below the 100 minimum).
**Why it happens:** ENAHO is a Peruvian household survey; foreign-born children are inherently rare.
**How to avoid:** Compute the metrics but flag as small sample. Note in findings that this dimension cannot support reliable fairness conclusions. Still worth reporting the point estimates with appropriate caveats.

### Pitfall 5: Language Dimension Has Two Levels of Granularity
**What goes wrong:** The spec requires both harmonized (4 groups: castellano, quechua, aimara, other_indigenous) and disaggregated (original codes: Ashaninka, Awajun, Shipibo, etc.) language analysis.
**Why it happens:** Post-2020 ENAHO surveys added disaggregated indigenous language codes.
**How to avoid:** Run the "language" dimension twice: once with `p300a_harmonized` (4 groups, all above 50 threshold) and once with `p300a_original` (14 codes, many flagged <50). Structure the JSON output with both levels.

### Pitfall 6: Poverty Dimension Should Use poverty_quintile (Not log_income Binned)
**What goes wrong:** The spec says "log_income binned into quintiles" for poverty dimension, but `poverty_quintile` is already computed (weighted quintiles from census_poverty_rate) in the feature matrix.
**Why it happens:** Spec wording may refer to the existing quintile feature.
**How to avoid:** Use the existing `poverty_quintile` column (values 1-5). It was computed with FACTOR07 survey weights to ensure each quintile has exactly 20% of weighted population. This is better than ad-hoc binning of log_income.

## Code Examples

### Complete Dimension Analysis Pattern
```python
# Source: Verified against fairlearn 0.13.0 empirically
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score, average_precision_score

def analyze_dimension(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    dimension_name: str,
    min_sample: int = 100,
) -> dict:
    """Compute fairness metrics for a single dimension."""

    def safe_recall(y, p, sample_weight=None):
        return recall_score(y, p, sample_weight=sample_weight, zero_division=np.nan)

    def safe_precision(y, p, sample_weight=None):
        return precision_score(y, p, sample_weight=sample_weight, zero_division=np.nan)

    def safe_fnr(y, p, sample_weight=None):
        r = recall_score(y, p, sample_weight=sample_weight, zero_division=np.nan)
        return 1.0 - r if not np.isnan(r) else np.nan

    def weighted_fpr(y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y_true, dtype=float)
        negatives = (np.asarray(y_true) == 0)
        if negatives.sum() == 0:
            return np.nan
        fp = negatives & (np.asarray(y_pred) == 1)
        return float(np.average(fp[negatives], weights=np.asarray(sample_weight)[negatives]))

    binary_metrics = {
        'tpr': safe_recall,
        'fpr': weighted_fpr,
        'fnr': safe_fnr,
        'precision': safe_precision,
    }
    binary_params = {k: {'sample_weight': weights} for k in binary_metrics}

    mf_bin = MetricFrame(
        metrics=binary_metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        sample_params=binary_params,
    )

    mf_prob = MetricFrame(
        metrics={'pr_auc': average_precision_score},
        y_true=y_true,
        y_pred=y_prob,
        sensitive_features=groups,
        sample_params={'pr_auc': {'sample_weight': weights}},
    )

    combined = pd.concat([mf_bin.by_group, mf_prob.by_group], axis=1)

    # Build per-group output
    result = {"groups": {}}
    for group_name in combined.index:
        group_mask = (groups == group_name)
        n_unweighted = int(group_mask.sum())
        n_weighted = float(weights[group_mask].sum())

        group_data = {
            "n_unweighted": n_unweighted,
            "n_weighted": round(n_weighted, 2),
            "tpr": round(float(combined.loc[group_name, "tpr"]), 6),
            "fpr": round(float(combined.loc[group_name, "fpr"]), 6),
            "fnr": round(float(combined.loc[group_name, "fnr"]), 6),
            "precision": round(float(combined.loc[group_name, "precision"]), 6),
            "pr_auc": round(float(combined.loc[group_name, "pr_auc"]), 6),
        }

        if n_unweighted < min_sample:
            group_data["flagged_small_sample"] = True

        result["groups"][str(group_name)] = group_data

    return result
```

### Calibration by Group Pattern
```python
# Use UNCALIBRATED probabilities for >0.7 threshold
def compute_calibration_by_group(
    y_true: np.ndarray,
    y_prob_uncal: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.7,
    min_high_risk: int = 30,
) -> dict:
    """For each group: among predicted high-risk, what is actual dropout rate?"""
    result = {}
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = (groups == g)
        high_risk = mask & (y_prob_uncal > threshold)
        n_high = int(high_risk.sum())
        if n_high >= min_high_risk:
            actual_rate = float(np.average(y_true[high_risk], weights=weights[high_risk]))
        else:
            actual_rate = None  # insufficient sample
        result[str(g)] = {
            "n_predicted_high": n_high,
            "actual_dropout_rate": round(actual_rate, 6) if actual_rate is not None else None,
        }
    return result
```

### Gap Computation Pattern
```python
from fairlearn.metrics import equalized_odds_difference

def compute_gaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    mf_binary: MetricFrame,
) -> dict:
    """Compute equalized odds, predictive parity, and max FNR gap."""
    by_group = mf_binary.by_group

    tpr_vals = by_group["tpr"].dropna()
    fpr_vals = by_group["fpr"].dropna()
    fnr_vals = by_group["fnr"].dropna()
    prec_vals = by_group["precision"].dropna()

    # Equalized odds: max TPR gap and max FPR gap
    eo_tpr = float(tpr_vals.max() - tpr_vals.min()) if len(tpr_vals) > 1 else 0.0
    eo_fpr = float(fpr_vals.max() - fpr_vals.min()) if len(fpr_vals) > 1 else 0.0

    # Predictive parity: max precision gap
    pp = float(prec_vals.max() - prec_vals.min()) if len(prec_vals) > 1 else 0.0

    # Max FNR gap with group names
    if len(fnr_vals) > 1:
        max_fnr_gap = float(fnr_vals.max() - fnr_vals.min())
        max_fnr_groups = [str(fnr_vals.idxmax()), str(fnr_vals.idxmin())]
    else:
        max_fnr_gap = 0.0
        max_fnr_groups = []

    return {
        "equalized_odds_tpr": round(eo_tpr, 6),
        "equalized_odds_fpr": round(eo_fpr, 6),
        "predictive_parity": round(pp, 6),
        "max_fnr_gap": round(max_fnr_gap, 6),
        "max_fnr_groups": max_fnr_groups,
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual loop over groups | fairlearn MetricFrame | fairlearn 0.4+ (2020) | Standardized, tested, handles edge cases |
| Single metric (accuracy parity) | Multi-metric evaluation (TPR, FPR, FNR, precision, PR-AUC) | 2019-2020 | More nuanced fairness assessment |
| Ignore intersectionality | Multi-feature MetricFrame | fairlearn 0.5+ | Reveals compounded disadvantage |
| Unweighted metrics only | sample_params for survey weights | fairlearn 0.8+ | Correct for complex survey designs |

**Deprecated/outdated:**
- `fairlearn.widget` / dashboard: Replaced by `fairlearn.metrics.MetricFrame`
- `_create_group_metric_set`: Internal, replaced by public MetricFrame API

## Critical Data Characteristics (Test Set 2023)

### Test Set Summary
- Total rows: 25,635
- Dropouts: 3,500 (13.64% unweighted)
- Calibrated threshold: 0.165716
- Calibrated prob range: [0.027, 0.431]
- Uncalibrated prob range: [varied, max 0.757]

### Dimension Group Sizes (Test 2023, Unweighted)
| Dimension | Groups | Min Group Size | All > 100? |
|-----------|--------|----------------|------------|
| Sex (es_mujer) | Male=13,211, Female=12,424 | 12,424 | Yes |
| Language (harmonized) | Castellano=23,170, Quechua=1,624, Other_indig=668, Aimara=76 | 76 (aimara) | No (aimara=76) |
| Geography (rural) | Urban=16,013, Rural=9,622 | 9,622 | Yes |
| Region | Costa=10,125, Sierra=8,458, Selva=7,052 | 7,052 | Yes |
| Poverty (quintile) | Q1=2,578 to Q5=6,414 | 2,578 | Yes |
| Nationality (es_peruano) | Peruvian=25,608, Non-Peruvian=27 | 27 | No (non-Peruvian=27) |

### Intersectional Small Groups (<50 unweighted)
| Intersection | Group | N | Action |
|-------------|-------|---|--------|
| Language x Rural | aimara_urban | 25 | Flag |
| Language x Rural | foreign_other_rural | 30 | Flag |
| Language x Region | aimara_costa | 8 | Flag |
| Language x Region | aimara_selva | 3 | Flag |
| Language x Region | other_indigenous_sierra | 2 | Flag |
| Language x Region | quechua_costa | 47 | Flag |

### Decision: Use Calibrated Probabilities for Fairness, Uncalibrated for High-Risk Analysis
- **Binary predictions (TPR, FPR, FNR, precision):** Use calibrated model's threshold (0.165716) applied to calibrated probabilities. These are the actual deployed predictions.
- **PR-AUC:** Use calibrated probabilities (ranking-based metric, threshold-invariant).
- **Calibration-by-group (>0.7 high-risk):** Use UNCALIBRATED probabilities. Rationale: 702 test observations exceed 0.7 uncalibrated vs. 0 calibrated. Document this choice in the JSON.
- **Alternative approach (if spec requires calibrated):** Redefine "high risk" as top decile of calibrated distribution (top 2,564 students; max calibrated prob ~0.43). Report actual dropout rate for this top decile per group.

## Open Questions

Things that couldn't be fully resolved:

1. **High-risk threshold for calibration-by-group**
   - What we know: Spec says ">0.7 probability" but calibrated probs max at 0.43. Uncalibrated probs have 702 observations above 0.7.
   - What's unclear: Does the spec intend calibrated or uncalibrated probabilities? The M4 site may expect a specific threshold.
   - Recommendation: Use uncalibrated >0.7 as primary. Add calibrated top-decile as secondary. Document both in JSON. Report the issue.

2. **Disaggregated language (p300a_original) dimension placement in JSON**
   - What we know: Spec mentions "Language (disagg.)" with p300a_original, 50 observation minimum (not 100).
   - What's unclear: Should disaggregated language be a separate dimension in the JSON or nested under "language"?
   - Recommendation: Make it a separate dimension key `"language_disaggregated"` with the lower 50-observation threshold.

3. **Which model to use: "lightgbm" in JSON schema**
   - What we know: M4 schema says `"model": "lightgbm"`. The predictions come from the calibrated LightGBM model.
   - What's unclear: Should it say "lightgbm" or "lightgbm_calibrated"?
   - Recommendation: Use `"model": "lightgbm"` per schema, note calibration in metadata.

## Sources

### Primary (HIGH confidence)
- fairlearn 0.13.0 API: Empirically tested MetricFrame with `sample_params`, verified nested dict structure works correctly
- fairlearn 0.13.0 docs: [MetricFrame API Reference](https://fairlearn.org/v0.13/api_reference/generated/fairlearn.metrics.MetricFrame.html)
- fairlearn 0.13.0 examples: [Metrics with Multiple Features](https://fairlearn.org/v0.13/auto_examples/plot_new_metrics.html)
- Codebase: `src/data/features.py` (MODEL_FEATURES, META_COLUMNS), `src/models/baseline.py` (_df_to_numpy, compute_metrics)
- Data inspection: `predictions_lgbm_calibrated.parquet` and `enaho_with_features.parquet` schemas and contents verified

### Secondary (MEDIUM confidence)
- Best practice: Use calibrated probabilities for threshold-based fairness metrics when probabilities allow meaningful thresholds
- [ACM: Is calibration a fairness requirement?](https://dl.acm.org/doi/fullHtml/10.1145/3531146.3533245) -- Calibration and equalized odds have inherent tradeoffs

### Tertiary (LOW confidence)
- None -- all critical claims verified empirically or from official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - fairlearn 0.13.0 verified empirically, all API calls tested
- Architecture: HIGH - Two-MetricFrame pattern verified, data joins verified, group sizes checked
- Pitfalls: HIGH - Calibration range compression verified empirically (critical finding), group sizes verified from data

**Research date:** 2026-02-08
**Valid until:** 2026-03-08 (stable -- fairlearn API unlikely to change within a minor version)
