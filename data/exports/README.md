# Alerta Escuela Equity Audit -- Export Files

Developer-facing documentation for integrating export files into the M4 scrollytelling site.

## Overview

| File | Size | Site Section | Producing Script | Phase |
|------|------|-------------|-----------------|-------|
| `findings.json` | ~10 KB | Narrative findings | `src/fairness/findings.py` | 11 |
| `fairness_metrics.json` | 28 KB | Subgroup fairness dashboard | `src/fairness/metrics.py` | 8 |
| `shap_values.json` | 29 KB | SHAP interpretability | `src/fairness/shap_analysis.py` | 9 |
| `choropleth.json` | 402 KB | District-level map | `src/fairness/cross_validation.py` | 10 |
| `model_results.json` | 30 KB | Model performance summary | `src/models/*.py` | 5-7 |
| `descriptive_tables.json` | 195 KB | Descriptive statistics | `src/data/descriptive.py` | 4 |
| `onnx/lightgbm_dropout.onnx` | 0.10 MB | Client-side inference | `src/models/calibration.py` | 7 |

---

## findings.json

**Purpose:** 8 bilingual (ES/EN) media-ready findings ordered by narrative arc, each linking to its source data.

**Schema:**

```json
{
  "generated_at": "ISO 8601 timestamp",
  "n_findings": 8,
  "findings": [
    {
      "id": "string (unique identifier, e.g. 'fnr_overall')",
      "stat": "string (human-readable statistic)",
      "headline_es": "string (Spanish headline, stat-forward)",
      "headline_en": "string (English headline, stat-forward)",
      "explanation_es": "string (2-3 sentence Spanish explanation)",
      "explanation_en": "string (2-3 sentence English explanation)",
      "metric_source": {
        "path": "string (format: filename.json#dot.path)",
        "label": "string (human-readable metric description)"
      },
      "visualization_type": "bar_chart | grouped_bar | heatmap | choropleth | comparison_bar",
      "data_key": "string (key in source export for rendering)",
      "severity": "critical | high | medium | low"
    }
  ]
}
```

**M4 component:** Main scrollytelling narrative. Each finding drives one scroll section. Use `data_key` to load the corresponding visualization data from the referenced export file.

**Data provenance:** Assembled from all other export files at runtime. Every `metric_source.path` is validated to resolve to a non-null value in the referenced export.

---

## fairness_metrics.json

**Purpose:** Subgroup fairness metrics (TPR, FPR, FNR, PR-AUC, precision) across 7 demographic dimensions and 3 intersections.

**Schema (top-level keys):**

| Key | Type | Description |
|-----|------|-------------|
| `generated_at` | string | ISO 8601 timestamp |
| `model` | string | Model name (`"lightgbm"`) |
| `threshold` | float | Classification threshold (calibrated) |
| `threshold_type` | string | Threshold type (`"calibrated"`) |
| `calibration_note` | string | Note on calibrated prob range and high-risk cutoff |
| `test_set` | string | Test year (`"2023"`) |
| `n_test` | int | Test set size |
| `n_dropouts` | int | Positive class count in test |
| `dimensions` | object | 7 fairness dimensions (see below) |
| `intersections` | object | 3 intersection analyses |

**Dimensions structure:** Each dimension (e.g., `dimensions.language`) contains:

```json
{
  "groups": {
    "castellano": {
      "n_unweighted": 23170,
      "n_weighted": 6993371.01,
      "tpr": 0.367,
      "fpr": 0.175,
      "fnr": 0.633,
      "precision": 0.243,
      "pr_auc": 0.235,
      "calibration_high_risk": { "n_predicted_high": 0, "actual_dropout_rate": null },
      "flagged_small_sample": false
    }
  },
  "gaps": {
    "equalized_odds_tpr": 0.707,
    "equalized_odds_fpr": 0.345,
    "predictive_parity": 0.181,
    "max_fnr_gap": 0.707,
    "max_fnr_groups": ["unknown", "other_indigenous"]
  }
}
```

**Dimensions available:** `language`, `language_disaggregated`, `sex`, `geography`, `region`, `poverty`, `nationality`.

**Intersections available:** `language_x_rural`, `sex_x_poverty`, `language_x_region`.

**M4 component:** Fairness dashboard with grouped bar charts (FNR/FPR by group) and heatmaps (intersection matrices).

**Data provenance:**
- Source: ENAHO 2018-2023 microdata (INEI), LightGBM test 2023 predictions
- Pipeline: fairlearn MetricFrame on calibrated predictions
- Script: `src/fairness/metrics.py` (Phase 8)

---

## shap_values.json

**Purpose:** SHAP feature importance analysis including global rankings, top-5 comparisons, regional breakdowns, and interaction effects.

**Schema (top-level keys):**

| Key | Type | Description |
|-----|------|-------------|
| `generated_at` | string | ISO 8601 timestamp |
| `model` | string | Model name (`"lightgbm"`) |
| `computed_on` | string | Dataset used (`"test_2023"`) |
| `n_test` | int | Number of test observations |
| `shap_space` | string | SHAP value space (`"log_odds"`) |
| `base_value` | float | Expected value (log-odds) |
| `feature_names` | array[25] | All feature names (English) |
| `feature_labels_es` | object | Spanish labels keyed by feature name |
| `global_importance` | object | Mean absolute SHAP per feature |
| `top_5_shap` | array[5] | Top 5 SHAP features |
| `top_5_lr` | array[5] | Top 5 logistic regression features |
| `overlap_count` | int | Overlap between SHAP and LR top-5 |
| `overlap_features` | array | Features appearing in both top-5 lists |
| `overlap_note` | string | Explanation of overlap result |
| `regional` | object | Per-region (costa/sierra/selva) SHAP |
| `es_peruano` | object | Nationality feature analysis |
| `es_mujer` | object | Sex feature analysis |
| `interactions` | object | Feature interaction effects |
| `profiles` | array[10] | Individual student SHAP profiles |

**M4 component:** SHAP bar chart (global importance), beeswarm plot, force plots for individual profiles.

**Data provenance:**
- Source: LightGBM model + ENAHO test 2023 features
- Pipeline: SHAP TreeExplainer on raw (uncalibrated) LightGBM model
- Script: `src/fairness/shap_analysis.py` (Phase 9)

---

## choropleth.json

**Purpose:** District-level cross-validation comparing model dropout predictions with administrative dropout rates, for choropleth map visualization.

**Schema (top-level keys):**

| Key | Type | Description |
|-----|------|-------------|
| `generated_at` | string | ISO 8601 timestamp |
| `n_districts` | int | Total districts (1,891) |
| `n_with_predictions` | int | Districts with model predictions (1,540) |
| `correlation` | object | Pearson r, p-value, n, caveat |
| `error_by_indigenous_group` | object | MAE by indigenous language prevalence |
| `districts` | array[1891] | Per-district records |

**District record schema:**

```json
{
  "ubigeo": "010101",
  "predicted_dropout_rate": 0.142,
  "admin_dropout_rate": 0.023,
  "model_error": 0.119,
  "indigenous_language_pct": 5.2,
  "poverty_index": -0.31
}
```

**M4 component:** Choropleth map colored by `model_error` or `predicted_dropout_rate`. Tooltip shows district stats. Overlay toggle for indigenous language prevalence.

**Data provenance:**
- Source: ENAHO 2023 test predictions aggregated to district level + MINEDU admin data (datosabiertos.gob.pe)
- Pipeline: District-level aggregation with Pearson correlation and stratified error analysis
- Script: `src/fairness/cross_validation.py` (Phase 10)

---

## model_results.json

**Purpose:** Model performance metrics, coefficients, and threshold analysis for logistic regression, LightGBM, and XGBoost.

**Schema (top-level keys):**

| Key | Type | Description |
|-----|------|-------------|
| `logistic_regression` | object | LR metrics, coefficients, odds ratios |
| `lightgbm` | object | LightGBM metrics, feature importances |
| `xgboost` | object | XGBoost metrics, feature importances |
| `test_2023_calibrated` | object | Final calibrated test results |

**Metrics structure (per model):**

```json
{
  "metadata": { "n_train": 98023, "n_val": 26477, "n_test": 25635 },
  "metrics": {
    "validate_2022": {
      "weighted": { "pr_auc": 0.2611, "roc_auc": 0.748 },
      "unweighted": { "pr_auc": 0.233 }
    }
  },
  "threshold_analysis": { "optimal_threshold": 0.167, "method": "max_weighted_f1" },
  "feature_importances": { "age": 0.259, "nightlight_intensity_z": 0.103 }
}
```

**M4 component:** Model comparison table, PR curves, calibration plot.

**Data provenance:**
- Source: ENAHO 2018-2023 train/val/test splits with survey weights
- Pipeline: sklearn LR + LightGBM + XGBoost + Platt calibration
- Scripts: `src/models/baseline.py` (Phase 5), `src/models/gbm.py` (Phase 6), `src/models/calibration.py` (Phase 7)

---

## descriptive_tables.json

**Purpose:** Weighted descriptive dropout statistics across demographic dimensions, with confidence intervals and intersection heatmaps.

**Schema (top-level keys):**

| Key | Type | Description |
|-----|------|-------------|
| `_metadata` | object | Generation info, source rows, years |
| `language` | array[7] | Dropout rates by 7 language groups |
| `language_binary` | array[2] | Indigenous vs castellano binary |
| `sex` | array[2] | Male vs female |
| `sex_x_level` | array[4] | Sex x education level |
| `rural` | array[2] | Urban vs rural |
| `region` | array[3] | Costa/Sierra/Selva |
| `poverty` | array[5] | Poverty quintiles |
| `heatmap_language_x_rural` | object | Intersection matrix |
| `heatmap_language_x_poverty` | object | Intersection matrix |
| `heatmap_language_x_region` | object | Intersection matrix |
| `choropleth_prep` | array[1540] | District-level descriptive data |
| `temporal` | object | Year-over-year trends |

**Table record schema (e.g., `language[0]`):**

```json
{
  "group": "other_indigenous",
  "weighted_rate": 0.2187,
  "lower_ci": 0.2176,
  "upper_ci": 0.2199,
  "n_unweighted": 3947,
  "n_weighted": 496036.0
}
```

**M4 component:** Descriptive statistics tables, intersection heatmaps (3 matrices), temporal line chart.

**Data provenance:**
- Source: ENAHO 2018-2023 pooled microdata with FACTOR07 survey weights
- Pipeline: statsmodels DescrStatsW with linearization confidence intervals
- Script: `src/data/descriptive.py` (Phase 4)

---

## onnx/lightgbm_dropout.onnx

**Purpose:** LightGBM dropout prediction model exported in ONNX format for client-side inference in the browser.

**Details:**

| Property | Value |
|----------|-------|
| File size | 0.10 MB |
| Input | 25 float32 features |
| Output | Raw probability (pre-calibration) |
| Max diff vs Python | 1.03e-07 |

**Calibration:** Apply Platt scaling in JavaScript after ONNX inference:

```javascript
const A = -6.236085;
const B = 4.442308;
const calibrated = 1 / (1 + Math.exp(A * raw_prob + B));
```

**M4 component:** Client-side "What if?" tool. User adjusts feature sliders, ONNX.js runs inference, Platt scaling produces calibrated probability.

**Data provenance:**
- Source: LightGBM model trained on ENAHO 2018-2021 (train) with 2022 validation
- Pipeline: skl2onnx conversion of raw (uncalibrated) LightGBM model
- Script: `src/models/calibration.py` (Phase 7)
