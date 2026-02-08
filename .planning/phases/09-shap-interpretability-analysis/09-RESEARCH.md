# Phase 9: SHAP Interpretability Analysis - Research

**Researched:** 2026-02-08
**Domain:** SHAP interpretability for LightGBM binary classification
**Confidence:** HIGH

## Summary

This research covers SHAP (SHapley Additive exPlanations) TreeExplainer with LightGBM for the Alerta Escuela equity audit's interpretability analysis. The goal is to compute global, regional, and interaction SHAP values on the 2023 test set, quantify ES_PERUANO and ES_MUJER contributions, select 10 representative student profiles, and export to M4-schema-compliant JSON.

**Key verified finding:** SHAP 0.50.0 with LightGBM 4.6.0 returns a **single 2D ndarray** (n_samples, n_features) from `TreeExplainer.shap_values()` for binary classification -- NOT a list. The old `shap_values[1]` pattern from the spec is outdated. The `expected_value` is a scalar (log-odds). This was verified by running the actual installed library.

**Primary recommendation:** Use the raw (uncalibrated) `model_lgbm.joblib` for SHAP, not the calibrated wrapper. TreeExplainer requires direct access to the tree structure. Compute SHAP in log-odds space (default `model_output='raw'`). Use the new Explanation API (`explainer(X)`) for modern plot functions, and the legacy `explainer.shap_values(X)` for interaction values and backward-compatible summary plots.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| shap | 0.50.0 | SHAP value computation + plots | Already installed, TreeExplainer optimized for LightGBM |
| lightgbm | 4.6.0 | Trained model to explain | Raw model stored at `model_lgbm.joblib` |
| matplotlib | (installed) | Headless plot generation (Agg backend) | Project standard for figure export |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| joblib | (installed) | Model loading | Load `model_lgbm.joblib` |
| numpy | (installed) | Array operations | Feature extraction, mean SHAP computation |
| polars | (installed) | Data loading + filtering | Load parquet, filter test set, select profiles |
| json | stdlib | JSON export | Write `shap_values.json` |

### No New Dependencies
All required libraries are already installed. No additions to `pyproject.toml` needed.

## Architecture Patterns

### Recommended Project Structure
```
src/
  fairness/
    shap_analysis.py          # NEW: SHAP pipeline (global + regional + profiles)
tests/
  gates/
    test_gate_3_2.py          # NEW: Gate test for SHAP
data/
  exports/
    shap_values.json          # M4 export
    figures/
      shap_beeswarm_global.png
      shap_bar_top10.png
      shap_regional_comparison.png
      shap_force_es_peruano.png
      shap_force_es_mujer.png
```

### Pattern 1: Raw Model for SHAP, Not Calibrated
**What:** TreeExplainer requires direct tree access. CalibratedClassifierCV wraps the model in a way that TreeExplainer cannot decompose.
**When to use:** Always for SHAP with tree models.
**Verified:** GitHub issues #899 and #1196 confirm CalibratedClassifierCV is not compatible with TreeExplainer.
```python
# CORRECT: Use raw model
lgbm = joblib.load("data/processed/model_lgbm.joblib")
explainer = shap.TreeExplainer(lgbm)

# WRONG: Calibrated model will fail or give wrong results
# cal_model = joblib.load("data/processed/model_lgbm_calibrated.joblib")
# explainer = shap.TreeExplainer(cal_model)  # ERROR
```

### Pattern 2: SHAP Return Type (SHAP 0.50.0 + LightGBM Binary)
**What:** `shap_values()` returns a single 2D ndarray, not a list. `expected_value` is a scalar.
**Critical:** The spec suggests `shap_values[1]` -- this will NOT work with shap 0.50.0.
**Verified by running on installed versions (shap 0.50.0, lightgbm 4.6.0):**
```python
explainer = shap.TreeExplainer(lgbm)
sv = explainer.shap_values(X_test)
# sv.shape = (25635, 25)  -- single 2D array in LOG-ODDS space
# explainer.expected_value = scalar (base log-odds)
# sv[i].sum() + expected_value ≈ raw model prediction for row i

# WARNING: shap prints a deprecation-style warning:
# "LightGBM binary classifier with TreeExplainer shap values output
#  has changed to a list of ndarray"
# Despite the warning text, the actual return is ndarray, not list.
# This was confirmed by direct testing.
```

### Pattern 3: New vs Legacy API
**What:** SHAP 0.50.0 supports both old and new APIs.
**Use both strategically:**
```python
# NEW API: For beeswarm, bar, waterfall plots
explanation = explainer(X_test)  # Returns Explanation object
shap.plots.beeswarm(explanation, show=False)  # Modern plots accept Explanation

# LEGACY API: For force_plot, summary_plot, interaction_values
sv = explainer.shap_values(X_test)  # Returns ndarray
shap.force_plot(explainer.expected_value, sv[0:1], X_test[0:1],
                feature_names=feature_names, matplotlib=True, show=False)
```

### Pattern 4: Regional Cohort Comparison
**What:** SHAP's bar plot supports multi-cohort comparison via dictionary of Explanations.
**Use for regional comparison (Costa/Sierra/Selva):**
```python
explanation = explainer(X_test)
regions = [...]  # array of "costa", "sierra", "selva" per test row
shap.plots.bar(explanation.cohorts(regions).abs.mean(0), show=False)
```

### Pattern 5: Pipeline Structure (Consistent with Phases 5-8)
```python
# src/fairness/shap_analysis.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")

def run_shap_pipeline() -> dict:
    """Run SHAP analysis pipeline. Returns shap_values.json content."""
    # 1. Load model + data
    # 2. Compute global SHAP
    # 3. Compute regional SHAP
    # 4. Compute interaction values
    # 5. Select 10 profiles
    # 6. Generate 5 figures
    # 7. Export JSON
    # 8. Print for human review
    pass

if __name__ == "__main__":
    run_shap_pipeline()
```

### Anti-Patterns to Avoid
- **Using calibrated model with TreeExplainer:** Will fail or produce incorrect SHAP values
- **Indexing `shap_values[1]`:** In shap 0.50.0 with LightGBM binary, return is 2D ndarray, not list
- **Computing interaction values on full test set:** Use subsample (1000 rows is fine, takes ~8s)
- **Using `show=True` in headless environment:** Always `show=False` then `plt.savefig()`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature importance ranking | Manual permutation importance | `shap.TreeExplainer` + `mean(abs(shap_values), axis=0)` | SHAP is theoretically grounded, additive |
| Force plots | Custom waterfall visualization | `shap.force_plot(matplotlib=True)` | Built-in, publication quality |
| Beeswarm/summary plots | Manual scatter + jitter | `shap.plots.beeswarm()` or `shap.summary_plot()` | Complex density + color encoding handled |
| Regional comparison bars | Manual grouped bar chart | `shap.plots.bar(explanation.cohorts(regions).abs.mean(0))` | Built-in multi-cohort support |
| Interaction values | Manual computation | `explainer.shap_interaction_values()` | O(TLD) algorithm, exact, fast |

**Key insight:** SHAP has dedicated optimized C++ implementations for LightGBM. The TreeSHAP algorithm is O(TLD) where T=trees, L=leaves, D=depth. Regular SHAP values for 25,635 rows x 25 features with 79 trees will take ~1-5 seconds. Interaction values for 1000 rows will take ~8 seconds.

## Common Pitfalls

### Pitfall 1: Stale SHAP API for LightGBM Binary Classification
**What goes wrong:** Code uses `shap_values[1]` assuming list return, gets IndexError or wrong values.
**Why it happens:** SHAP 0.45.0 changed return types. Many tutorials and even the spec reference the old behavior.
**How to avoid:** Verify return type at runtime. With shap 0.50.0 + LightGBM 4.6.0, `shap_values()` returns `ndarray` shape `(n, p)` directly.
**Warning signs:** Warning message about "changed to a list of ndarray" (confusingly worded but return IS ndarray).

### Pitfall 2: SHAP Values Are in Log-Odds Space
**What goes wrong:** Interpreting SHAP values as probability contributions.
**Why it happens:** Default `model_output='raw'` returns log-odds space SHAP values.
**How to avoid:** Document clearly that SHAP values are in log-odds space. sum(SHAP) + base_value = raw log-odds prediction. To get probability: `1 / (1 + exp(-raw_prediction))`.
**Warning signs:** SHAP values that seem "too large" (they're log-odds, not probabilities).

### Pitfall 3: LR Coefficient vs SHAP Top-5 Overlap
**What goes wrong:** Gate test requires 3/5 overlap but LR and LightGBM use fundamentally different importance measures.
**Why it happens:** LR coefficients reflect linear associations (language dummies dominate). LightGBM gain-based importance reflects split utility (continuous features like age dominate). SHAP should bridge this gap somewhat.
**How to avoid:** SHAP global importance (mean |SHAP|) should show more overlap with LR than gain-based importance does. Current LR top-5 by |coef|: `lang_other_indigenous, lang_foreign, lang_quechua, is_secundaria_age, lang_aimara`. Current LightGBM gain top-5: `age, nightlight_intensity_z, census_electricity_pct_z, poverty_index_z, census_literacy_rate_z`. There is ZERO overlap between these. SHAP top-5 may partially bridge this. If 3/5 overlap is not achieved, document and explain.
**Warning signs:** If SHAP top-5 looks identical to gain-based top-5, the overlap check may fail.

### Pitfall 4: Insufficient Profiles for Some Categories
**What goes wrong:** Too few students match profile criteria (e.g., Lima urban foreign: only 2 in test set).
**Why it happens:** ES_PERUANO=0 is very rare (~27/25,635 = 0.1%).
**How to avoid:** For "Lima urban foreign" profile, select from the available 2 students. Use the one with prediction closest to overall mean (most "representative"). Flag the small sample.
**Warning signs:** Empty filter results when selecting profiles.

### Pitfall 5: Force Plot File Size
**What goes wrong:** Force plots with 25 features produce very wide images that are hard to read.
**Why it happens:** Default force plot shows all features.
**How to avoid:** Use `contribution_threshold` parameter or `max_display` to limit shown features. Or use waterfall plots instead (more readable for single observations).
**Warning signs:** Truncated or unreadable saved PNGs.

### Pitfall 6: Interaction Values Memory
**What goes wrong:** Computing interaction values on full test set uses excessive memory.
**Why it happens:** Return shape is (n, 25, 25) = n * 625 floats. For 25,635 rows = 128 MB.
**How to avoid:** Subsample to 1000 rows (spec says if > 5000). 1000 rows at 25 features = 4.8 MB, 8 seconds compute.
**Warning signs:** Memory spikes during interaction computation.

## Code Examples

### Example 1: Global SHAP Computation
```python
# Verified pattern for shap 0.50.0 + lightgbm 4.6.0
import shap
import joblib
import numpy as np

lgbm = joblib.load("data/processed/model_lgbm.joblib")
explainer = shap.TreeExplainer(lgbm)

# Get SHAP values (2D ndarray, log-odds space)
sv = explainer.shap_values(X_test)
# sv.shape = (n_test, 25)

# Global importance: mean absolute SHAP value per feature
global_importance = np.abs(sv).mean(axis=0)
# Shape: (25,)

# Map to feature names
importance_dict = dict(zip(MODEL_FEATURES, global_importance.tolist()))
# Sort by importance
sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
```

### Example 2: Beeswarm Plot (Global)
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

explanation = explainer(X_test)
shap.plots.beeswarm(explanation, max_display=25, show=False)
plt.tight_layout()
plt.savefig("data/exports/figures/shap_beeswarm_global.png",
            bbox_inches="tight", dpi=150)
plt.close("all")
```

### Example 3: Regional SHAP with Cohort Comparison
```python
# Build region labels for test set
regions = test_df["region_natural"].to_numpy()

# Compute SHAP for full test set
explanation = explainer(X_test)

# Cohort comparison bar plot
shap.plots.bar(
    explanation.cohorts(regions).abs.mean(0),
    max_display=10,
    show=False,
)
plt.tight_layout()
plt.savefig("data/exports/figures/shap_regional_comparison.png",
            bbox_inches="tight", dpi=150)
plt.close("all")

# Also compute per-region mean |SHAP| for JSON export
for region_name in ["costa", "sierra", "selva"]:
    mask = regions == region_name
    regional_mean_shap = np.abs(sv[mask]).mean(axis=0)
    # Store in JSON structure
```

### Example 4: Force Plot Saved as PNG
```python
# Force plot for a single observation
idx = 42  # index of representative student
shap.force_plot(
    explainer.expected_value,
    sv[idx:idx+1],
    X_test[idx:idx+1],
    feature_names=list(MODEL_FEATURES),
    matplotlib=True,
    show=False,
)
plt.savefig("data/exports/figures/shap_force_es_peruano.png",
            bbox_inches="tight", dpi=150)
plt.close("all")
```

### Example 5: Interaction Values
```python
# Subsample for interaction values (spec: <= 1000 if test set > 5000)
rng = np.random.default_rng(42)
sub_idx = rng.choice(X_test.shape[0], 1000, replace=False)
X_sub = X_test[sub_idx]

interaction_values = explainer.shap_interaction_values(X_sub)
# Shape: (1000, 25, 25)

# Focus on poverty x language interaction
poverty_idx = MODEL_FEATURES.index("poverty_index_z")
lang_idx = MODEL_FEATURES.index("lang_other_indigenous")
interaction_strength = np.abs(interaction_values[:, poverty_idx, lang_idx]).mean()
```

### Example 6: Profile Selection
```python
# Select representative profiles systematically
# Strategy: for each profile category, find the row closest to the
# category's median predicted probability (most "representative")
def select_representative(df, mask, probas):
    """Select the row with prediction closest to group median."""
    group_proba = probas[mask]
    median_prob = np.median(group_proba)
    # Find index of closest to median
    group_indices = np.where(mask)[0]
    closest_idx = group_indices[np.argmin(np.abs(group_proba - median_prob))]
    return closest_idx
```

### Example 7: M4 JSON Schema Construction
```python
shap_json = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "model": "lightgbm",
    "computed_on": "test_2023",
    "feature_names": list(MODEL_FEATURES),
    "feature_labels_es": FEATURE_LABELS_ES,
    "global_importance": {
        feat: round(float(val), 6)
        for feat, val in zip(MODEL_FEATURES, global_importance)
    },
    "regional": {
        "costa": {feat: round(float(v), 6) for feat, v in zip(MODEL_FEATURES, costa_shap)},
        "sierra": {feat: round(float(v), 6) for feat, v in zip(MODEL_FEATURES, sierra_shap)},
        "selva": {feat: round(float(v), 6) for feat, v in zip(MODEL_FEATURES, selva_shap)},
    },
    "profiles": profiles_list,  # 10 profile dicts
}
```

## Feature Labels (Spanish)

The M4 schema requires `feature_labels_es`. These labels are for the scrollytelling site audience (Spanish-speaking, Peru context):

```python
FEATURE_LABELS_ES: list[str] = [
    "Edad",                                    # age
    "Edad de secundaria (12+)",                # is_secundaria_age
    "Sexo femenino",                           # es_mujer
    "Lengua castellana",                       # lang_castellano
    "Lengua quechua",                          # lang_quechua
    "Lengua aimara",                           # lang_aimara
    "Otra lengua indigena",                    # lang_other_indigenous
    "Lengua extranjera",                       # lang_foreign
    "Zona rural",                              # rural
    "Region Sierra",                           # is_sierra
    "Region Selva",                            # is_selva
    "Tasa de desercion distrital (admin, z)",  # district_dropout_rate_admin_z
    "Intensidad de luces nocturnas (z)",       # nightlight_intensity_z
    "Indice de pobreza (z)",                   # poverty_index_z
    "Quintil de pobreza",                      # poverty_quintile
    "Nacionalidad peruana",                    # es_peruano
    "Tiene discapacidad",                      # has_disability
    "Trabaja",                                 # is_working
    "Beneficiario JUNTOS",                     # juntos_participant
    "Ingreso del hogar (log)",                 # log_income
    "Educacion de los padres (anos)",          # parent_education_years
    "Poblacion indigena del distrito (z)",     # census_indigenous_lang_pct_z
    "Tasa de alfabetismo del distrito (z)",    # census_literacy_rate_z
    "Acceso a electricidad del distrito (z)",  # census_electricity_pct_z
    "Acceso a agua del distrito (z)",          # census_water_access_pct_z
]
```

**Confidence:** MEDIUM -- Labels are based on standard INEI ENAHO terminology. "z" suffix indicates z-score standardized. May need refinement from Spanish-speaking reviewer.

## Profile Selection Strategy

The spec requires 10 specific profile types. Based on test set composition:

| Profile ID | Filter Criteria | Available Rows | Notes |
|---|---|---|---|
| `lima_urban_castellano_male` | dept=15, rural=0, lang_castellano=1, es_mujer=0 | ~1,200 | Abundant |
| `lima_urban_foreign` | dept=15, rural=0, es_peruano=0 | 2 | Very sparse |
| `sierra_rural_quechua` | region=sierra, rural=1, lang_quechua=1 | 1,327 | Abundant |
| `sierra_rural_castellano` | region=sierra, rural=1, lang_castellano=1 | 3,608 | Abundant |
| `selva_rural_indigenous` | region=selva, rural=1, lang_other_indigenous=1 | 577 | Moderate |
| `selva_rural_castellano` | region=selva, rural=1, lang_castellano=1 | 2,769 | Abundant |
| `female_secundaria_urban` | es_mujer=1, is_secundaria_age=1, rural=0 | 4,120 | Abundant |
| `female_secundaria_rural` | es_mujer=1, is_secundaria_age=1, rural=1 | 2,463 | Abundant |
| `male_secundaria_urban` | es_mujer=0, is_secundaria_age=1, rural=0 | 4,280 | Abundant |
| `male_secundaria_rural` | es_mujer=0, is_secundaria_age=1, rural=1 | 2,645 | Abundant |

**Selection algorithm:** For each profile type, find the row whose predicted probability is closest to the median of that subgroup. This selects the most "typical" student rather than an outlier.

**Lima urban foreign caveat:** Only 2 students match (es_peruano=0 in Lima urban). Use whichever is closer to the foreign-student median. Document the small sample.

## Interaction Values Analysis

### Performance Benchmarks (Verified)
| Subsample Size | Compute Time | Memory | Recommended |
|---|---|---|---|
| 100 rows | 0.78s | 0.5 MB | Too small for stability |
| 500 rows | 3.84s | 2.4 MB | Good for quick analysis |
| 1000 rows | 7.83s | 4.8 MB | Recommended (spec says <= 1000 if > 5000) |

**Note:** The spec says interaction values are O(n^2). This is WRONG. TreeSHAP interaction values are O(n * T * L * D) -- linear in n. The benchmarks confirm linear scaling. 1000 rows is the right subsample size per the spec's guidance.

### Key Interactions to Focus On
Per spec:
- **poverty x language** (`poverty_index_z` x `lang_other_indigenous`)
- **rurality x sex** (`rural` x `es_mujer`)

Compute mean absolute interaction strength for these pairs.

## LR Coefficient vs SHAP Overlap Analysis

**Current state (before SHAP computation):**

LR top-5 by |coefficient|:
1. `lang_other_indigenous` (0.788)
2. `lang_foreign` (0.576)
3. `lang_quechua` (0.471)
4. `is_secundaria_age` (0.438)
5. `lang_aimara` (0.347)

LightGBM gain-based top-5:
1. `age` (0.259)
2. `nightlight_intensity_z` (0.123)
3. `census_electricity_pct_z` (0.093)
4. `poverty_index_z` (0.084)
5. `census_literacy_rate_z` (0.084)

**Overlap: 0/5.** SHAP values often differ from both LR coefficients AND gain-based importance. SHAP marginal contributions may elevate features like `age`, `poverty_index_z`, or `is_working` that appear in both paradigms. The 3/5 overlap gate test criterion should be realistic but may need flexibility depending on actual SHAP results.

**Mitigation:** If strict 3/5 overlap fails, document the explanation: LR is linear, LightGBM is nonlinear. SHAP captures nonlinear contributions. The important thing is that equity-relevant features appear in both rankings.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| `shap_values()` returns list for binary | Returns single ndarray | shap 0.45.0 (Mar 2024) | Must NOT index `[1]` |
| `shap.summary_plot()` with numpy arrays | `shap.plots.beeswarm()` with Explanation | shap 0.36.0+ | Modern API, richer metadata |
| Manual cohort comparison | `explanation.cohorts(groups).abs.mean(0)` | shap ~0.40+ | Built-in multi-bar plots |
| `shap.force_plot()` returns JS/HTML | `matplotlib=True, show=False` for static PNG | Existing | Required for headless Agg backend |

**Deprecated/outdated:**
- `shap.summary_plot()`: Still works but `shap.plots.beeswarm()` is preferred
- `shap_values[1]` for binary: Returns single array now, not list
- `model_output='probability'` with TreeExplainer: Only works with `feature_perturbation='interventional'`, adds complexity

## Open Questions

1. **SHAP-LR overlap threshold**
   - What we know: LR and LightGBM gain have 0/5 overlap. SHAP should partially bridge this.
   - What's unclear: Whether 3/5 overlap will be achieved depends on actual SHAP values.
   - Recommendation: Compute SHAP, check overlap, document findings. If <3, explain why (LR is linear, SHAP captures nonlinear effects).

2. **Spanish labels accuracy**
   - What we know: Labels follow standard INEI ENAHO terminology.
   - What's unclear: Some labels may need refinement for M4 scrollytelling context.
   - Recommendation: Use provided labels; they can be adjusted in Phase 11 (Findings Distillation) if needed.

3. **Interaction plot visualization**
   - What we know: `shap_interaction_values()` returns 3D tensor; SHAP doesn't have a built-in "interaction heatmap" plot.
   - What's unclear: Best way to visualize key interaction pairs.
   - Recommendation: Use custom matplotlib heatmap for top interaction pairs, or `shap.dependence_plot()` with interaction_index for pairwise visualization.

## Sources

### Primary (HIGH confidence)
- SHAP 0.50.0 installed and tested directly on project environment
- LightGBM 4.6.0 installed and tested directly
- [SHAP TreeExplainer API docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) - API signatures verified
- [SHAP Release Notes](https://shap.readthedocs.io/en/latest/release_notes.html) - v0.45-0.50 changes documented
- [SHAP Migration Guide](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/migrating-to-new-api.html) - Old vs new API
- [SHAP force plot API](https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html) - matplotlib=True pattern
- [Census income LightGBM example](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20LightGBM.html)
- [SHAP bar plot cohorts](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html) - Multi-cohort pattern

### Secondary (MEDIUM confidence)
- [SHAP issue #899](https://github.com/shap/shap/issues/899) - CalibratedClassifierCV incompatibility
- [SHAP issue #526](https://github.com/shap/shap/issues/526) - LightGBM 2D array return
- [SHAP issue #2334](https://github.com/slundberg/shap/issues/2334) - LightGBM vs XGBoost binary output
- [SHAP issue #153](https://github.com/slundberg/shap/issues/153) - Saving plots as PNG

### Tertiary (LOW confidence)
- Spanish feature labels: based on INEI ENAHO codebook terminology (not directly verified against M4 site)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries installed and tested directly
- Architecture: HIGH - Patterns verified by running code on installed shap 0.50.0
- Return types: HIGH - Directly tested shap_values() and explainer() return types
- Performance: HIGH - Benchmarked interaction values timing on similar model size
- Spanish labels: MEDIUM - Based on standard ENAHO terminology, may need refinement
- LR overlap: MEDIUM - Depends on actual SHAP values (not yet computed)

**Research date:** 2026-02-08
**Valid until:** 2026-03-08 (stable -- shap 0.50.0 released Nov 2025, unlikely to change soon)
