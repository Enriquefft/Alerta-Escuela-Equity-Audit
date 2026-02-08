---
phase: 09-shap-interpretability-analysis
verified: 2026-02-08T20:30:00Z
status: passed
score: 7/7 must-haves verified
---

# Phase 9: SHAP Interpretability Analysis Verification Report

**Phase Goal:** Global, regional, and interaction SHAP values quantify each feature's contribution to dropout predictions, with specific attention to ES_PERUANO (nationality) and ES_MUJER (gender) effects.

**Verified:** 2026-02-08T20:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Global SHAP values computed on 2023 test set (25,635 rows x 25 features) | ✓ VERIFIED | shap_values.json confirms n_test=25635, 25 features in global_importance, shap_analysis.py lines 265-276 compute TreeExplainer and SHAP values |
| 2 | Top-5 SHAP features documented alongside top-5 LR coefficient features with overlap count | ✓ VERIFIED | JSON has top_5_shap, top_5_lr, overlap_count=0, overlap_note explaining paradigm difference. Gate test passes. Human approved 0/5 overlap as expected. |
| 3 | Regional SHAP computed separately for Costa, Sierra, Selva with per-region mean &#124;SHAP&#124; per feature | ✓ VERIFIED | JSON regional section has costa/sierra/selva keys, each with 25 features. Lines 345-381 compute per-region masks and mean absolute SHAP. Gate test verifies all 3 regions present. |
| 4 | ES_PERUANO and ES_MUJER average SHAP magnitudes quantified and printed for review | ✓ VERIFIED | JSON es_peruano: rank 25/25, mean_abs_shap=0.0. JSON es_mujer: rank 16/25, mean_abs_shap=0.003072. Lines 422-468 compute magnitudes and conditional means. Gate test output shows both magnitudes. |
| 5 | 10 representative student profiles exported with feature values, SHAP values, and calibrated probability | ✓ VERIFIED | JSON profiles array has 10 entries. Each has profile_id, feature_values (25 keys), shap_values (25 keys), predicted_probability, base_value, n_in_group. Lines 470-554 implement profile selection. Gate test confirms 10 profiles. |
| 6 | shap_values.json matches M4 schema with global_importance, regional, profiles, and feature_labels_es | ✓ VERIFIED | JSON has all required top-level keys: global_importance, regional, profiles, feature_labels_es (25 Spanish labels), es_peruano, es_mujer, interactions, top_5_shap, top_5_lr, overlap_count. 29 KB file size. Gate test validates structure. |
| 7 | Gate test 3.2 passes all assertions | ✓ VERIFIED | `uv run pytest tests/gates/test_gate_3_2.py -v -s` shows 11/11 tests PASSED. No failures. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/fairness/shap_analysis.py` | SHAP pipeline: global + regional + interaction + profiles + figures + JSON (min 200 lines) | ✓ VERIFIED | EXISTS: 715 lines. SUBSTANTIVE: imports shap, joblib, loads model_lgbm.joblib (line 200), creates TreeExplainer (265), computes SHAP values (266-276), generates 5 figures, exports JSON (623-626). NO STUBS: no TODO/FIXME/placeholder. WIRED: imports MODEL_FEATURES from data.features (line 30), called by __main__ (711-715). |
| `tests/gates/test_gate_3_2.py` | Gate test validating SHAP outputs (min 80 lines) | ✓ VERIFIED | EXISTS: 326 lines. SUBSTANTIVE: 11 test functions + fixture loading shap_values.json (34-40). NO STUBS. WIRED: imports from utils, loads JSON, asserts structure and values. All tests pass. |
| `data/exports/shap_values.json` | M4-schema-compliant SHAP export (contains "global_importance") | ✓ VERIFIED | EXISTS: 29 KB. SUBSTANTIVE: contains global_importance (25 features), regional (3 regions x 25 features), profiles (10 entries), es_peruano, es_mujer, interactions, feature_labels_es. Valid JSON structure. WIRED: written by shap_analysis.py line 623-626, read by test_gate_3_2.py fixture. |
| `data/exports/figures/shap_beeswarm_global.png` | Global beeswarm visualization | ✓ VERIFIED | EXISTS: 174 KB. Created by shap_analysis.py lines 316-323. Gate test confirms file exists and size > 0. |
| `data/exports/figures/shap_bar_top10.png` | Top 10 feature importance bar chart | ✓ VERIFIED | EXISTS: 66 KB. Created by shap_analysis.py lines 328-333. Gate test confirms existence. |
| `data/exports/figures/shap_regional_comparison.png` | Regional cohort comparison bar chart | ✓ VERIFIED | EXISTS: 118 KB. Created by shap_analysis.py lines 396-401. Gate test confirms existence. |
| `data/exports/figures/shap_force_es_peruano.png` | Force plot for ES_PERUANO representative profile | ✓ VERIFIED | EXISTS: 93 KB. Created by shap_analysis.py lines 527-536. Gate test confirms existence. |
| `data/exports/figures/shap_force_es_mujer.png` | Force plot for ES_MUJER representative profile | ✓ VERIFIED | EXISTS: 108 KB. Created by shap_analysis.py lines 538-547. Gate test confirms existence. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| src/fairness/shap_analysis.py | data/processed/model_lgbm.joblib | joblib.load raw LightGBM model (NOT calibrated wrapper) | ✓ WIRED | Line 200: `lgbm = joblib.load(root / "data" / "processed" / "model_lgbm.joblib")`. Pattern match found. Model loaded successfully (gate test passes). |
| src/fairness/shap_analysis.py | shap.TreeExplainer | TreeExplainer with raw model, shap_values returns 2D ndarray | ✓ WIRED | Line 265: `explainer = shap.TreeExplainer(lgbm)`. Line 266: `sv = explainer.shap_values(X_test)`. Assertions verify shape (25635, 25). SHAP computation complete and validated. |
| src/fairness/shap_analysis.py | data/exports/shap_values.json | json.dump with M4 schema structure | ✓ WIRED | Line 626: `json.dump(shap_json, f, indent=2, default=str)`. JSON file created (29 KB), schema-compliant structure verified by gate test. |
| tests/gates/test_gate_3_2.py | data/exports/shap_values.json | json.load + assertions on structure and values | ✓ WIRED | Lines 34-40: fixture loads JSON with `json.load(f)`. All 11 test functions assert on loaded data. Tests pass, confirming read + validation works. |

### Requirements Coverage

Phase 9 requirements from ROADMAP (SHAP-01 through SHAP-06):

| Requirement | Status | Supporting Truths |
|-------------|--------|-------------------|
| SHAP-01: Global SHAP values on test set | ✓ SATISFIED | Truth 1: Global SHAP verified |
| SHAP-02: Regional SHAP (Costa/Sierra/Selva) | ✓ SATISFIED | Truth 3: Regional SHAP verified |
| SHAP-03: ES_PERUANO quantification | ✓ SATISFIED | Truth 4: ES_PERUANO magnitude verified (rank 25/25, zero contribution) |
| SHAP-04: ES_MUJER quantification | ✓ SATISFIED | Truth 4: ES_MUJER magnitude verified (rank 16/25, minimal direct effect) |
| SHAP-05: 10 representative profiles | ✓ SATISFIED | Truth 5: 10 profiles verified with feature values + SHAP values + probabilities |
| SHAP-06: M4-schema shap_values.json export | ✓ SATISFIED | Truth 6: JSON structure verified, schema-compliant |

### Anti-Patterns Found

**No blocker or warning anti-patterns detected.**

Scan of modified files (src/fairness/shap_analysis.py, tests/gates/test_gate_3_2.py):
- 🟢 No TODO/FIXME/placeholder comments
- 🟢 No empty return statements
- 🟢 No console.log-only implementations
- 🟢 No hardcoded placeholders
- 🟢 All functions have substantive implementations

### Human Verification Notes

**Human gate was completed and APPROVED** (per SUMMARY.md and user context).

Key human-verified findings:
1. **0/5 LR overlap is expected and documented** — Linear models (LR) favor categorical language dummies; tree models (LightGBM) favor continuous spatial features (age, poverty, nightlights). Both identify equity-relevant features through different paradigms. Not contradictory.
2. **ES_PERUANO rank 25/25 (zero contribution)** — Consistent with n=27 non-Peruvian students in test set. No model signal. Phase 8 already flagged this dimension as unusable.
3. **ES_MUJER rank 16/25 (minimal direct effect)** — Mean |SHAP| = 0.003072, slightly protective for females (-0.003166 vs males +0.002976). Gender equity gap (Phase 8 FNR gap 0.026) is mediated through correlated features, not gender itself.
4. **Model predicts through spatial/structural features** — Top-5 SHAP: age, census_literacy_rate_z, nightlight_intensity_z, is_working, poverty_index_z. All continuous/spatial, not identity features (language, gender, nationality). Key narrative for Phase 11 findings.
5. **Regional differences** — Age dominates all regions (0.156-0.178). Sierra emphasizes literacy + working. Selva emphasizes nightlights + working. Infrastructure/education system quality proxies, suitable for policy framing.
6. **All 5 figures verified as publication-quality** — Beeswarm, bar, regional comparison, and 2 force plots all exist and render correctly.

---

## Overall Status: PASSED ✓

**All must-haves verified. No gaps found. Human gate approved.**

### Success Criteria Met

Per ROADMAP Phase 9 success criteria:
1. ✓ Global SHAP values computed on 2023 test set; top-5 SHAP features overlap with top-5 LR coefficient magnitudes (at least 3 in common) **→ 0/5 overlap documented and approved by human as expected paradigm difference**
2. ✓ Regional SHAP computed separately for Costa, Sierra, Selva — revealing where features like mother tongue matter more
3. ✓ ES_PERUANO and ES_MUJER average SHAP magnitudes quantified specifically; 10 representative student profiles exported with feature values + SHAP values + predicted probability
4. ✓ `data/exports/shap_values.json` exists, matches M4 schema with global_importance, regional, and profiles sections
5. ✓ Gate test 3.2 passes; top-5 global SHAP features and ES_PERUANO/ES_MUJER magnitudes printed for human review

**Note on SC-1 (LR overlap criterion):** The roadmap states "at least 3 in common." Actual overlap is 0/5. This was **explicitly approved by the human reviewer** with documented explanation that LR (linear model with categorical dummies) and SHAP (tree model with continuous features) rank features differently due to model paradigm, not contradictory findings. Both identify equity-relevant features. Per user context: "This was approved by the human reviewer with documented explanation. Do NOT fail verification for this — it is expected and documented."

### Key Metrics

- **Gate tests:** 11/11 passed
- **Artifacts:** 8/8 verified (2 .py files, 1 .json, 5 .png figures)
- **Key links:** 4/4 wired
- **Must-have score:** 7/7 truths verified
- **Anti-patterns:** 0 blockers, 0 warnings
- **Human gate:** APPROVED

### Phase Outputs Ready for Downstream Use

Phase 11 (Findings Distillation) can consume:
- `data/exports/shap_values.json` — Global/regional SHAP importance rankings, ES_PERUANO/ES_MUJER magnitudes, 10 representative profiles with SHAP breakdowns, interaction strengths
- All 5 SHAP figures — Ready for bilingual narrative integration
- Key narrative: Model predicts dropout through *where you live* (spatial/structural features) more than *who you are* (identity features)

---

_Verified: 2026-02-08T20:30:00Z_
_Verifier: Claude (gsd-verifier)_
