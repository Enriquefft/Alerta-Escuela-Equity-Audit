---
phase: 09-shap-interpretability-analysis
plan: 01
subsystem: fairness
tags: [shap, treeshap, lightgbm, interpretability, profiles, regional, interaction, log-odds]

# Dependency graph
requires:
  - phase: 07-calibration-onnx-export-final-test
    provides: "Raw LightGBM model (model_lgbm.joblib), calibrated predictions parquet"
  - phase: 04-feature-engineering-descriptive-statistics
    provides: "enaho_with_features.parquet with MODEL_FEATURES and region_natural"
  - phase: 05-baseline-model-temporal-splits
    provides: "model_results.json with logistic_regression coefficients for overlap check"
provides:
  - "Global SHAP values (25 features, mean |SHAP| importance ranking)"
  - "Regional SHAP (Costa/Sierra/Selva per-region mean |SHAP|)"
  - "ES_PERUANO and ES_MUJER SHAP magnitudes with rank and direction"
  - "10 representative student profiles with feature values + SHAP values + predicted probability"
  - "5 publication-quality PNG figures (beeswarm, bar, regional, 2 force plots)"
  - "M4-schema-compliant shap_values.json (28 KB)"
affects: [11-findings-distillation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Raw model for TreeExplainer (NOT calibrated wrapper)"
    - "SHAP 0.50.0 returns single 2D ndarray for LightGBM binary (NOT list)"
    - "New API explainer(X) for beeswarm/bar plots; legacy explainer.shap_values(X) for force/interaction"
    - "explanation.cohorts(regions).abs.mean(0) for regional comparison"
    - "Interaction values on 1000-row subsample (linear in n, ~8s)"

key-files:
  created:
    - src/fairness/shap_analysis.py
    - tests/gates/test_gate_3_2.py
    - data/exports/shap_values.json
    - data/exports/figures/shap_beeswarm_global.png
    - data/exports/figures/shap_bar_top10.png
    - data/exports/figures/shap_regional_comparison.png
    - data/exports/figures/shap_force_es_peruano.png
    - data/exports/figures/shap_force_es_mujer.png

key-decisions:
  - "0/5 LR overlap is expected and documented — LR favors categorical dummies, SHAP captures nonlinear continuous effects"
  - "ES_PERUANO rank 25/25 (zero contribution) — consistent with n=27, no model signal"
  - "ES_MUJER rank 16/25 (tiny direct effect) — gender mediated through correlated features"
  - "Model predicts dropout through spatial/structural features (nightlights, literacy, poverty) not identity features (language, gender)"
  - "lima_urban_foreign profile: n=2, flagged as small sample"

patterns-established:
  - "TreeExplainer with raw LightGBM model for SHAP analysis"
  - "Profile selection via median-closest representative"

# Metrics
duration: ~10min
completed: 2026-02-08
---

# Plan 09-01: SHAP Interpretability Analysis Summary

**Global SHAP reveals dropout prediction driven by spatial/structural features (age, literacy, nightlights, poverty) not identity features (language, gender, nationality) — reinforcing Phase 8's finding that the model's equity issues are structural, not individual-level**

## Performance

- **Duration:** ~10 min
- **Tasks:** 3 (2 auto + 1 human checkpoint)
- **Files created:** 8 (2 .py, 1 .json, 5 .png)
- **Gate tests:** 11/11 pass

## Accomplishments

- Global SHAP values computed on 25,635 test rows x 25 features (log-odds space)
- Top-5 SHAP: age (0.170), census_literacy_rate_z, nightlight_intensity_z, is_working, poverty_index_z
- Regional SHAP for Costa, Sierra, Selva — age dominates all regions, Sierra emphasizes literacy, Selva emphasizes nightlights
- ES_PERUANO: rank 25/25 (zero contribution, consistent with n=27)
- ES_MUJER: rank 16/25 (0.003 mean |SHAP|, slightly protective for females)
- 10 representative student profiles with feature values + SHAP values + calibrated probabilities
- Interaction values on 1000-row subsample — poverty x language and rural x gender pairs quantified
- 5 publication-quality PNG figures
- shap_values.json (28 KB) matches M4 schema

## Key SHAP Findings

- **Top-5 SHAP features:** age, census_literacy_rate_z, nightlight_intensity_z, is_working, poverty_index_z — all continuous/spatial features
- **LR vs SHAP overlap: 0/5** — LR top-5 are language dummies; SHAP top-5 are continuous. Different model paradigms rank differently; both capture equity-relevant signal
- **ES_PERUANO:** Mean |SHAP| ≈ 0.000, rank 25/25. Zero contribution (n=27 non-Peruvian)
- **ES_MUJER:** Mean |SHAP| = 0.003, rank 16/25. Signed: -0.003 females (slightly protective), +0.003 males. Gender mediated through correlated features
- **Regional:** Age dominates all regions (0.156-0.178). Sierra: literacy + working. Selva: nightlights + working. Language features absorbed by spatial z-scores
- **Profiles:** Probabilities range 0.11 (Lima urban) to 0.17 (Selva rural indigenous) — directionally correct

## Task Commits

1. **Task 1: Implement SHAP analysis pipeline** — `a3a5683` (feat)
2. **Task 2: Write gate test 3.2** — `b3b4bcd` (test)
3. **Task 3: Human review** — approved with notes:
   - 0/5 LR overlap is expected (model paradigm difference, not contradictory findings)
   - ES_PERUANO zero is consistent with Phase 8 (barely mention)
   - ES_MUJER minimal direct effect, mediated through correlated features
   - Model predicts through spatial/structural features, not identity — key Phase 11 narrative
   - All 5 figures verified

## Files Created/Modified

- `src/fairness/shap_analysis.py` — Full SHAP pipeline (global + regional + interaction + profiles + figures + JSON)
- `tests/gates/test_gate_3_2.py` — 11 gate test assertions + human review print block
- `data/exports/shap_values.json` — M4-schema-compliant SHAP values (28 KB)
- `data/exports/figures/shap_beeswarm_global.png` — Global beeswarm (178 KB)
- `data/exports/figures/shap_bar_top10.png` — Top-10 bar chart (67 KB)
- `data/exports/figures/shap_regional_comparison.png` — Regional comparison (121 KB)
- `data/exports/figures/shap_force_es_peruano.png` — Force plot for foreign student (94 KB)
- `data/exports/figures/shap_force_es_mujer.png` — Force plot for rural female student (110 KB)

## Decisions Made

- 0/5 LR overlap documented with explanation (linear vs nonlinear paradigm)
- ES_PERUANO: zero SHAP confirms no model signal (consistent with Phase 8 n=27)
- ES_MUJER: minimal direct SHAP, mediated through other features
- lima_urban_foreign: n=2, flagged as small sample

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Phase 11 Framing Notes (from human review)

1. **Core finding:** Model predicts dropout through *where you live* (spatial/structural features) more than *who you are* (identity features). Nightlights, literacy rates, and poverty are the top drivers — not language, gender, or nationality directly.
2. **LR vs SHAP narrative:** Linear models highlight categorical identity variables; tree models highlight continuous spatial variables. Both capture equity-relevant signal through different lenses. Neither is "wrong."
3. **ES_PERUANO:** Zero SHAP confirms the dimension is unusable (n=27). Exclude from findings.
4. **ES_MUJER:** Minimal direct effect. Gender equity gap (Phase 8: FNR gap 0.026) is driven by correlated features, not gender itself. Report as "gender is not a direct predictor, but intersects with other risk factors."
5. **Regional:** Selva's reliance on nightlights for prediction → infrastructure proxy. Sierra's reliance on literacy → education system quality proxy. These are *structural* explanations of dropout, suitable for policy framing.

---
*Phase: 09-shap-interpretability-analysis*
*Completed: 2026-02-08*
