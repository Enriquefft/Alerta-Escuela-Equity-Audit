# Phase 10: Cross-Validation with Admin Data - Research

**Researched:** 2026-02-08
**Domain:** District-level aggregation, spatial correlation analysis, GeoJSON export
**Confidence:** HIGH

## Summary

Phase 10 requires aggregating individual-level model predictions to the district level, correlating them with administrative (MINEDU) dropout rates, quantifying prediction error by indigenous language prevalence, and exporting a choropleth-ready JSON file with >1,500 districts.

The technical challenge is straightforward: score the full 150,135-row feature matrix through the calibrated LightGBM model, compute weighted-mean predicted dropout rates per district (using FACTOR07 survey weights), join with admin and census data on UBIGEO, compute Pearson correlation with scipy.stats.pearsonr, and export. The data infrastructure from Phases 3-9 (loaders, parquets, model artifacts) is mature and stable.

The key insight is that 1,540 ENAHO districts have model predictions while admin/census data covers 1,890 districts. Including all 1,890 districts (350 with null predictions) satisfies the >1,500 requirement and provides a complete choropleth. Correlation and error analysis should use only the 1,540 districts with both predicted and admin rates.

**Primary recommendation:** Score ALL rows in enaho_with_features.parquet through the calibrated model, aggregate to district level with FACTOR07 weighting, join admin+census on UBIGEO, export 1,890-district choropleth.json, validate with Pearson correlation and stratified error analysis.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| polars | (project version) | DataFrame operations, aggregation, joins | Already used throughout pipeline |
| scipy.stats | (project version) | pearsonr for Pearson correlation + p-value | Standard for statistical correlation tests |
| joblib | (project version) | Load calibrated model for scoring | Already used for model persistence |
| numpy | (project version) | Array operations for model scoring | Already used throughout |
| json (stdlib) | - | Export choropleth.json | Consistent with existing exports |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn | (project version) | predict_proba on CalibratedClassifierCV | Model scoring |
| matplotlib | (project version) | Optional scatter plot of correlation | If visual needed for gate review |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.pearsonr | numpy.corrcoef | pearsonr gives p-value directly; numpy.corrcoef does not |
| polars group_by | pandas groupby | polars is already the project standard; no reason to switch |

**Installation:** No new packages needed. All dependencies are already installed.

## Architecture Patterns

### Recommended Project Structure
```
src/
  fairness/
    cross_validation.py    # NEW: district-level cross-validation pipeline
tests/
  gates/
    test_gate_3_3.py       # NEW: gate test 3.3
data/
  exports/
    choropleth.json        # NEW: output file
```

### Pattern 1: Score-Aggregate-Join-Export Pipeline
**What:** A single-file pipeline (`cross_validation.py`) that loads the calibrated model, scores the full feature matrix, aggregates predictions to district level, joins admin/census data, computes correlation statistics, and exports choropleth.json.
**When to use:** This is the only pattern needed for Phase 10.
**Why:** Follows the same pattern as `src/fairness/metrics.py` and `src/fairness/shap_analysis.py` -- standalone script with `run_*_pipeline()` function.

```python
# Pattern from existing codebase (metrics.py, shap_analysis.py)
def run_cross_validation_pipeline() -> dict:
    """Run district-level cross-validation and export choropleth.json."""
    root = find_project_root()

    # Step 1: Load model + full feature matrix
    # Step 2: Score all 150,135 rows through calibrated model
    # Step 3: Aggregate to district level (weighted mean predicted dropout)
    # Step 4: Join admin + census data on UBIGEO
    # Step 5: Compute Pearson correlation
    # Step 6: Stratify error by indigenous language prevalence
    # Step 7: Export choropleth.json
    # Step 8: Console summary for human review

    return choropleth_data
```

### Pattern 2: District-Level Aggregation with Survey Weights
**What:** Compute weighted mean predicted dropout rate per district using FACTOR07.
**When to use:** This is the correct way to aggregate individual predictions to district level since ENAHO is a survey with expansion factors.

```python
import polars as pl

# Weighted aggregation: sum(prob * FACTOR07) / sum(FACTOR07)
district_predictions = (
    scored_df
    .group_by("UBIGEO")
    .agg([
        pl.len().alias("n_students"),
        pl.col("FACTOR07").sum().alias("total_weight"),
        (pl.col("prob_dropout") * pl.col("FACTOR07")).sum().alias("_weighted_sum"),
    ])
    .with_columns(
        (pl.col("_weighted_sum") / pl.col("total_weight")).alias("predicted_dropout_rate")
    )
    .drop("_weighted_sum")
)
```

### Pattern 3: Admin Rate Unit Harmonization
**What:** Admin dropout rates are in percentage (0-100, e.g., 2.05%), while model predictions are probability (0-1, e.g., 0.1342). For correlation and error calculation, convert to the SAME units.
**When to use:** Before computing Pearson r and mean absolute error.

```python
# Convert admin rates from percentage to proportion (0-1) for comparability
# OR convert model predictions from proportion to percentage
# Recommendation: work in percentage (multiply predictions by 100)
# This makes the choropleth data more human-readable
district_df = district_df.with_columns(
    (pl.col("predicted_dropout_rate") * 100).alias("predicted_dropout_rate")
)
# Now both are in percentage units (0-100)
```

**IMPORTANT:** Pearson correlation is scale-invariant, so it does not matter which unit you use for r and p. But for mean absolute error (MAE), consistent units are essential.

### Pattern 4: Stratified Error Analysis
**What:** Split districts into high-indigenous (>50% indigenous language) and low-indigenous (<10%) groups, compute MAE separately.
**When to use:** Required by XVAL-02.

```python
from scipy.stats import pearsonr

# Census provides indigenous_lang_pct (0-100)
high_indig = merged.filter(pl.col("census_indigenous_lang_pct") > 50)
low_indig = merged.filter(pl.col("census_indigenous_lang_pct") < 10)

# MAE for each group
high_mae = (high_indig["model_error"].abs()).mean()
low_mae = (low_indig["model_error"].abs()).mean()
```

### Anti-Patterns to Avoid
- **Using only test/validation predictions:** Only 1,296 districts covered. Must score ALL rows through the model.
- **Unweighted aggregation:** ENAHO is a survey; FACTOR07 weights are essential for population-representative district estimates.
- **Comparing probabilities to percentages without unit conversion:** Will produce inflated error values if units differ.
- **Dropping districts with small samples:** Include all districts but flag those with n_students < 5 or n_students < 10.
- **Using admin_secundaria_rate only:** 44 districts have null secundaria. Use a combined admin rate (secundaria for ages 12+, primaria for ages 6-11, or overall average).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pearson correlation + p-value | Manual computation | `scipy.stats.pearsonr` | Handles edge cases, numerically stable |
| Model scoring | Re-implement prediction logic | `cal_model.predict_proba(X)` | Calibration is already built into the saved model |
| JSON export | Custom serialization | `json.dump()` with `default=str` | Consistent with existing exports pattern |
| Admin/Census loading | Read CSVs directly | `load_admin_dropout_rates()`, `load_census_2017()` | Already validated, UBIGEO-padded, tested |
| Feature matrix extraction | Manual column selection | `MODEL_FEATURES` from `data.features` | Single source of truth for feature list |
| Survey weighting | Simple mean | Weighted mean with FACTOR07 | ENAHO is a survey; unweighted means are biased |

**Key insight:** Phase 10 combines existing infrastructure (model, loaders, features) with straightforward aggregation. Almost no new algorithms needed -- just plumbing.

## Common Pitfalls

### Pitfall 1: Scoring Training Data Through the Model
**What goes wrong:** Concern that using training data predictions inflates correlation (data leakage).
**Why it happens:** The model was trained on 2018-2021 data, so scoring those rows produces in-sample predictions.
**How to avoid:** This is actually fine for the choropleth purpose. The goal is to produce district-level predicted rates for spatial visualization, not to evaluate model performance. The correlation test with admin rates validates external agreement. Document the caveat: "Predictions for 2018-2021 rows are in-sample; 2022-2023 rows are out-of-sample."
**Alternative approach:** If purity is needed, use only 2022-2023 predictions (1,296 districts) for the correlation test, but use ALL years for the choropleth aggregation to maximize coverage.
**Recommended:** Use ALL years for choropleth export (>1,500 districts). Use ONLY test_2023 predictions (1,242 districts) for the Pearson correlation to avoid data leakage.

### Pitfall 2: Admin Rate Interpretation
**What goes wrong:** Treating admin dropout rates (MINEDU interannual rates) and model predictions (ENAHO survey-based) as measuring the same thing.
**Why it happens:** Both are "dropout rates" but computed differently. Admin rates are from SIAGIE administrative records; model predictions are from survey responses.
**How to avoid:** Frame the correlation as "external validation" not "accuracy check." A moderate positive correlation (r=0.1-0.5) is a strong finding given synthetic admin data and methodological differences.
**Warning signs:** Very high correlation (r>0.8) might indicate data leakage through the district_dropout_rate_admin feature already in the model.

### Pitfall 3: district_dropout_rate_admin Leakage
**What goes wrong:** The model already uses `district_dropout_rate_admin_z` as a feature (admin dropout rate z-scored). If we correlate model predictions with admin dropout rates, the correlation is partially mechanical.
**Why it happens:** The feature engineering pipeline (Phase 4) included admin dropout rates as a spatial predictor.
**How to avoid:**
1. Report the correlation honestly with this caveat.
2. Optionally, compute a "partial correlation" controlling for district_dropout_rate_admin.
3. The spec-required analysis is straightforward correlation; the caveat enriches the finding.
**Warning signs:** This is expected and documented. The SHAP analysis (Phase 9) already showed district_dropout_rate_admin_z has importance ~0.03.

### Pitfall 4: Choropleth JSON Size
**What goes wrong:** 1,890 district records with 6+ fields could produce a large JSON file.
**Why it happens:** JSON is verbose compared to parquet/CSV.
**How to avoid:** Round floating-point values to 4 decimal places. Expected size: ~200-400 KB for 1,890 records with 6-8 fields each. This is fine for browser consumption.

### Pitfall 5: Admin Combined Rate Computation
**What goes wrong:** Using only secundaria admin rate when 44 districts are null, or only primaria rate when the analysis covers ages 6-17.
**Why it happens:** The model covers school-age children 6-17 (both primaria and secundaria ages).
**How to avoid:** Compute a combined admin rate: average of primaria and secundaria where both exist, or whichever is available. This matches the model's population better than either rate alone.

### Pitfall 6: Float NaN in JSON Export
**What goes wrong:** Python's `float('nan')` serializes to `NaN` in JSON which is not valid JSON.
**Why it happens:** Null predictions for 350 admin-only districts, or NaN from calculations.
**How to avoid:** Use `None` (serializes to `null` in JSON) instead of `float('nan')`. Filter or replace NaN values before serialization.

## Code Examples

### Scoring Full Dataset
```python
# Source: existing codebase patterns (models/calibration.py, models/baseline.py)
import joblib
import polars as pl
from data.features import MODEL_FEATURES

# Load calibrated model
cal_model = joblib.load(root / "data" / "processed" / "model_lgbm_calibrated.joblib")

# Load full feature matrix
feat = pl.read_parquet(root / "data" / "processed" / "enaho_with_features.parquet")

# Extract feature matrix (25 features)
X = feat.select(MODEL_FEATURES).to_pandas().to_numpy()

# Score through calibrated model
cal_probs = cal_model.predict_proba(X)[:, 1]

# Add predictions back to DataFrame
scored = feat.with_columns(
    pl.Series("prob_dropout", cal_probs)
)
```

### District-Level Aggregation
```python
# Weighted mean predicted dropout rate per district
district_pred = (
    scored
    .group_by("UBIGEO")
    .agg([
        pl.len().alias("n_students"),
        pl.col("FACTOR07").sum().alias("total_weight"),
        (pl.col("prob_dropout") * pl.col("FACTOR07")).sum().alias("_ws"),
        (pl.col("dropout").cast(pl.Float64) * pl.col("FACTOR07")).sum().alias("_wd"),
    ])
    .with_columns([
        (pl.col("_ws") / pl.col("total_weight") * 100).alias("predicted_dropout_rate"),
        (pl.col("_wd") / pl.col("total_weight") * 100).alias("actual_dropout_rate_enaho"),
    ])
    .drop("_ws", "_wd")
)
# predicted_dropout_rate is now in percentage (0-100) for consistency with admin rates
```

### Pearson Correlation
```python
from scipy.stats import pearsonr
import numpy as np

# Filter to districts with BOTH predicted and admin rates (non-null)
valid = merged.filter(
    pl.col("predicted_dropout_rate").is_not_null()
    & pl.col("admin_dropout_rate").is_not_null()
)

pred_rates = valid["predicted_dropout_rate"].to_numpy()
admin_rates = valid["admin_dropout_rate"].to_numpy()

r, p_value = pearsonr(pred_rates, admin_rates)
print(f"Pearson r = {r:.4f}, p = {p_value:.6f}")
# Expected: positive r with p < 0.05
```

### Stratified Error Analysis
```python
# Model error = predicted - admin (positive = overprediction)
merged = merged.with_columns(
    (pl.col("predicted_dropout_rate") - pl.col("admin_dropout_rate")).alias("model_error")
)

# High-indigenous districts (>50% indigenous lang)
high_indig = merged.filter(pl.col("indigenous_language_pct") > 50)
# Low-indigenous districts (<10% indigenous lang)
low_indig = merged.filter(pl.col("indigenous_language_pct") < 10)

high_mae = high_indig["model_error"].abs().mean()
low_mae = low_indig["model_error"].abs().mean()

print(f"High-indigenous MAE: {high_mae:.4f} pp")
print(f"Low-indigenous MAE: {low_mae:.4f} pp")
print(f"N high-indig districts: {high_indig.height}")
print(f"N low-indig districts: {low_indig.height}")
```

### Choropleth JSON Export
```python
# Based on requirements: ubigeo, predicted_dropout_rate, admin_dropout_rate,
# model_error, indigenous_language_pct, poverty_index
choropleth_records = []
for row in final_df.iter_rows(named=True):
    record = {
        "ubigeo": row["UBIGEO"],
        "predicted_dropout_rate": round(row["predicted_dropout_rate"], 4) if row["predicted_dropout_rate"] is not None else None,
        "admin_dropout_rate": round(row["admin_dropout_rate"], 4) if row["admin_dropout_rate"] is not None else None,
        "model_error": round(row["model_error"], 4) if row["model_error"] is not None else None,
        "indigenous_language_pct": round(row["indigenous_language_pct"], 4) if row["indigenous_language_pct"] is not None else None,
        "poverty_index": round(row["poverty_index"], 4) if row["poverty_index"] is not None else None,
    }
    choropleth_records.append(record)

choropleth_json = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "n_districts": len(choropleth_records),
    "n_with_predictions": sum(1 for r in choropleth_records if r["predicted_dropout_rate"] is not None),
    "correlation": {
        "pearson_r": round(r, 6),
        "p_value": round(p_value, 6),
        "n_districts_correlated": n_valid,
    },
    "error_by_indigenous_group": {
        "high_indigenous_gt50": {
            "n_districts": high_indig.height,
            "mean_absolute_error": round(float(high_mae), 4),
        },
        "low_indigenous_lt10": {
            "n_districts": low_indig.height,
            "mean_absolute_error": round(float(low_mae), 4),
        },
    },
    "districts": choropleth_records,
}
```

### Gate Test Pattern
```python
# Source: existing gate test pattern (test_gate_3_1.py, test_gate_3_2.py)
import json
import pytest

ROOT = find_project_root()
CHOROPLETH_PATH = ROOT / "data" / "exports" / "choropleth.json"

@pytest.fixture(scope="module")
def choropleth_data():
    assert CHOROPLETH_PATH.exists(), "choropleth.json not found"
    with open(CHOROPLETH_PATH) as f:
        return json.load(f)

def test_positive_correlation(choropleth_data):
    """Pearson r is positive and p < 0.05."""
    corr = choropleth_data["correlation"]
    assert corr["pearson_r"] > 0, f"r={corr['pearson_r']} is not positive"
    assert corr["p_value"] < 0.05, f"p={corr['p_value']} is not < 0.05"

def test_district_count(choropleth_data):
    """choropleth.json has >1500 districts."""
    assert choropleth_data["n_districts"] > 1500

def test_district_fields(choropleth_data):
    """Each district has required fields."""
    required = {"ubigeo", "predicted_dropout_rate", "admin_dropout_rate",
                "model_error", "indigenous_language_pct", "poverty_index"}
    for d in choropleth_data["districts"][:10]:  # Sample check
        assert required.issubset(d.keys())
```

## Data Architecture: Key Numbers

These verified numbers inform the plan directly:

| Metric | Value | Source |
|--------|-------|--------|
| Full dataset rows | 150,135 | enaho_with_features.parquet |
| Full dataset districts (UBIGEO) | 1,540 | enaho_with_features.parquet |
| Admin districts | 1,890 | data/raw/admin/ |
| Census districts | 1,890 | data/raw/census/ |
| Admin-only districts (no ENAHO) | 350 | admin - enaho |
| Test 2023 districts | 1,242 | predictions_lgbm_calibrated.parquet |
| Val+Test districts | 1,296 | predictions_lgbm_calibrated.parquet |
| High-indigenous (>50%) districts | 392 | census indigenous_lang_pct |
| Low-indigenous (<10%) districts | 350 | census indigenous_lang_pct |
| Admin primaria range | 0.00-3.57% | admin data |
| Admin secundaria range | 0.00-6.80% | admin data |
| Calibrated prob range | 0.027-0.431 | predictions parquet |
| Model features | 25 | MODEL_FEATURES |
| Admin secundaria null | 44 districts | admin data |
| Districts with <5 ENAHO students | 51 | enaho_with_features.parquet |
| Districts with <10 ENAHO students | 148 | enaho_with_features.parquet |

## Admin Rate Strategy

**IMPORTANT DECISION:** Which admin dropout rate to use for the district-level comparison?

Options:
1. **admin_secundaria_rate only** -- Best match for school dropout concept but 44 nulls.
2. **admin_primaria_rate only** -- No nulls but captures younger age group only.
3. **Combined rate** -- Average of primaria + secundaria where both exist, single rate where one is null. Best match for the model's 6-17 age range.
4. **Weighted by age composition** -- Use primaria rate for age-6-11 predictions, secundaria for age-12-17. Most precise.

**Recommendation:** Use option 3 (combined average) as the primary `admin_dropout_rate` in the choropleth. This is the simplest approach that covers the full age range without nulls. Document the methodology.

## Correlation Approach Strategy

**IMPORTANT:** The model already uses `district_dropout_rate_admin_z` as a feature (z-scored admin dropout rate). This creates a partially mechanical correlation between model predictions and admin rates.

**Recommendation:** Report the correlation honestly with this caveat. The correlation is still meaningful because:
1. The model has 24 OTHER features beyond the admin rate feature.
2. The admin rate feature's SHAP importance is relatively low (~0.03 mean |SHAP|).
3. The correlation validates that the model's spatial pattern matches external data, even if partially by construction.

For additional rigor, the pipeline could also report:
- Correlation using ONLY test_2023 districts (out-of-sample predictions) -- 1,242 districts
- The admin rate feature's SHAP importance as context

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CSV-based admin data loading | Polars-based loader with UBIGEO validation | Phase 3 | Use existing `load_admin_dropout_rates()` |
| Manual feature extraction | MODEL_FEATURES constant | Phase 4 | Use the constant for scoring |
| sklearn cv='prefit' | FrozenEstimator | sklearn 1.8 | Calibrated model already saved with new API |

## Open Questions

1. **Admin rate as combined vs. secundaria-only?**
   - What we know: Admin data has primaria and secundaria rates; 44 districts lack secundaria.
   - What's unclear: The spec says "admin dropout rates" without specifying which level.
   - Recommendation: Use combined average for choropleth, report both in the analysis.

2. **In-sample vs out-of-sample for correlation?**
   - What we know: Using all data gives 1,540 districts; using only test gives 1,242.
   - What's unclear: Whether the spec intends in-sample or out-of-sample predictions.
   - Recommendation: Compute correlation on test_2023 only (purest), but generate predictions for ALL data for the choropleth.

3. **Should the choropleth include the 350 admin-only districts (no ENAHO data)?**
   - What we know: Spec requires >1,500 districts. With ENAHO-only we have 1,540.
   - What's unclear: Whether districts without predictions are useful for the M4 site.
   - Recommendation: Include all 1,890 districts with null predictions for the 350 admin-only ones. This provides a complete Peru map.

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/data/admin.py` -- AdminResult with UBIGEO, admin_primaria_rate, admin_secundaria_rate
- Codebase inspection: `src/data/census.py` -- CensusResult with UBIGEO, census_indigenous_lang_pct, census_poverty_rate
- Codebase inspection: `src/data/features.py` -- MODEL_FEATURES (25 features), META_COLUMNS
- Codebase inspection: `src/fairness/metrics.py` -- Pipeline pattern (run_fairness_pipeline)
- Codebase inspection: `src/models/calibration.py` -- Calibrated model loading pattern
- Codebase inspection: `src/models/baseline.py` -- ID_COLUMNS, create_temporal_splits
- Data verification: `data/processed/predictions_lgbm_calibrated.parquet` -- 52,112 rows, 14 columns, UBIGEO included
- Data verification: `data/processed/enaho_with_features.parquet` -- 150,135 rows, 65 columns, 1,540 unique UBIGEO
- Data verification: Admin data -- 1,890 districts, 0-3.57% primaria, 0-6.80% secundaria
- Data verification: Census data -- 1,890 districts, 392 with >50% indigenous, 350 with <10%
- Library verification: scipy.stats.pearsonr available in project environment
- Gate test patterns: test_gate_3_1.py, test_gate_3_2.py (JSON validation + human review print)

### Secondary (MEDIUM confidence)
- District coverage analysis: 244 districts are train-only (not in val/test predictions). Scoring full dataset adds these districts.

### Tertiary (LOW confidence)
- None. All findings verified through codebase inspection and data verification.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and used in project
- Architecture: HIGH -- follows established patterns from Phases 8-9
- Data coverage: HIGH -- verified exact district counts and overlaps
- Pitfalls: HIGH -- identified through codebase analysis (admin feature leakage, unit mismatch, NaN handling)
- Correlation approach: MEDIUM -- synthetic admin data makes expected r magnitude uncertain

**Research date:** 2026-02-08
**Valid until:** 2026-03-08 (30 days -- stable domain, no fast-moving dependencies)
