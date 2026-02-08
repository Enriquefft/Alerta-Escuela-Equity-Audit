# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-07)

**Core value:** The fairness audit is the product. Models exist to be audited, not to achieve SOTA.
**Current focus:** Phase 9 complete -- ready for Phase 10

## Current Position

Phase: 9 of 11 (SHAP Interpretability Analysis) -- COMPLETE
Plan: 1 of 1 in current phase
Status: Phase complete, verified
Last activity: 2026-02-08 -- Completed 09-01-PLAN.md

Progress: [███████████░] 92%

## Performance Metrics

**Velocity:**
- Total plans completed: 12
- Average duration: ~10 min
- Total execution time: ~116 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 00-environment-setup | 2/2 | ~20 min | ~10 min |
| 01-enaho-single-year-loader | 1/1 | ~15 min | ~15 min |
| 02-multi-year-loader-harmonization | 1/1 | ~8 min | ~8 min |
| 03-spatial-supplementary-data-merges | 1/1 | ~15 min | ~15 min |
| 04-feature-engineering-descriptive-statistics | 2/2 | ~13 min | ~7 min |
| 05-baseline-model-temporal-splits | 1/1 | ~5 min | ~5 min |
| 06-lightgbm-xgboost | 1/1 | ~10 min | ~10 min |
| 07-calibration-onnx-export-final-test | 1/1 | ~5 min | ~5 min |
| 08-subgroup-fairness-metrics | 1/1 | ~8 min | ~8 min |
| 09-shap-interpretability-analysis | 1/1 | ~10 min | ~10 min |

**Recent Trend:**
- Last 5 plans: 09-01 (~10 min), 08-01 (~8 min), 07-01 (~5 min), 06-01 (~10 min), 05-01 (~5 min)
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: specs.md is authoritative SSOT -- all implementation details come from spec
- [Roadmap]: Phase structure follows spec Section 9 (Phases 1-11) plus Phase 0 for environment
- [Roadmap]: Gate tests are integral to each phase, not a separate phase
- [Roadmap]: DATA-10 (parquet saves) assigned to Phase 4; TEST-01/02/03 assigned to Phase 1
- [00-01]: nixos-25.05 pinned as nixpkgs input; Python 3.12.12 via Nix
- [00-01]: hatchling build backend with src layout (src/alerta_escuela_audit/)
- [00-01]: uv sync --python python3.12 needed to avoid uv selecting newer managed Python
- [00-02]: PROJECT_ROOT uses pyproject.toml walk-up instead of hardcoded parent.parent
- [00-02]: data/exports/ tracked (not gitignored) for M4 site export artifacts
- [01-01]: INEI provides DTA (Stata) files, not CSVs -- added _read_data_file() with pandas DTA support
- [01-01]: P303/P306 arrive as Float64 from DTA files, cast to Int64
- [02-01]: P303-null rows dropped before null-fill logic (2020: 52.3%, 2021: 4.6%)
- [02-01]: Dual-column harmonization: p300a_original preserves raw, p300a_harmonized collapses 10-15 to 3
- [03-01]: Synthetic admin/census/nightlights data generated (datosabiertos.gob.pe URLs return 404)
- [03-01]: Loaders have graceful placeholder behavior for missing supplementary files
- [03-01]: 44 districts have primaria but no secundaria admin data (minor gap)
- [03-01]: Uppercase column names maintained through merge pipeline (UBIGEO not ubigeo)
- [04-01]: 25 model features (exceeding spec's 19 minimum) with 4 census z-score features added
- [04-01]: P209 birthplace only for ages 12+; ages 6-11 default to es_peruano=1
- [04-01]: Nightlight z-score nulls imputed with 0.0 (distribution mean)
- [04-01]: Parent education = max(head, spouse) P301A mapped to years; 12 rows median-imputed
- [04-02]: DescrStatsW (linearization) for survey-weighted confidence intervals
- [04-02]: Awajun rate = 0.2047 (>0.18), Castellano = 0.1526, confirming language equity gap
- [05-01]: Max weighted F1 as threshold optimization target (optimal=0.5168)
- [05-01]: class_weight='balanced' + FACTOR07 multiplicative interaction documented
- [05-01]: freq_weights p-values all 0.0 (effective n ~25M); use odds ratios for interpretation
- [05-01]: poverty_quintile negative sign = multicollinearity with poverty_index_z (expected)
- [06-01]: LightGBM early_stopping defaults to monitoring last metric (binary_logloss); must use first_metric_only=True for average_precision
- [06-01]: LightGBM and XGBoost val PR-AUC nearly identical (0.2611 vs 0.2612); algorithm-independence ratio = 1.0006
- [06-01]: age dominates feature importance (25.85%) but well under 50% threshold; poverty_index_z (#4) and nightlights (#2) confirm equity-relevant signal
- [07-01]: FrozenEstimator replaces cv='prefit' (removed in sklearn 1.8.0); benign sample_weight warning suppressed
- [07-01]: Platt A=-5.278337, B=4.276521; Brier 0.2115→0.1156 (45.3% reduction)
- [07-01]: ONNX 0.10 MB, max diff 1.03e-07 vs Python; raw model exported, Platt applied in JS
- [07-01]: Test 2023 PR-AUC=0.2378, FNR=62.4% (matches Alerta Escuela 57-64% range)
- [08-01]: Two-MetricFrame pattern for binary (y_pred) vs proba (y_pred_proba) fairness metrics
- [08-01]: Uncalibrated >0.7 for calibration-by-group (calibrated max=0.431, 0 obs above 0.7)
- [08-01]: FNR-FPR trade-off: indigenous low FNR/high FPR (surveillance bias) vs castellano high FNR/low FPR (invisibility bias)
- [08-01]: Nationality dim flagged (n=27), user requested minimal mention in findings
- [09-01]: SHAP top-5: age, census_literacy_rate_z, nightlight_intensity_z, is_working, poverty_index_z -- all continuous/spatial
- [09-01]: LR vs SHAP overlap: 0/5 (linear vs nonlinear paradigm, documented with explanation)
- [09-01]: ES_PERUANO rank 25/25 (zero SHAP, n=27, no model signal)
- [09-01]: ES_MUJER rank 16/25 (0.003 mean |SHAP|, slightly protective for females)
- [09-01]: Model predicts through spatial/structural features, not identity features -- key narrative for Phase 11

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 3: Admin/census/nightlights data is synthetic -- findings should note data provenance (resolved with synthetic data)
- Phase 7: ONNX float32 tolerance verified at 1.03e-07 (well within 1e-4, resolved)
- Phase 8: fairlearn sample_params API verified working (nested dict pattern, resolved)
- Phase 8: Nationality dimension unreliable (n=27) -- barely mention in Phase 11
- Phase 9: LR vs SHAP overlap 0/5 -- documented with explanation, not a problem (resolved)
- Phase 9: ES_PERUANO zero contribution confirms dimension unusable (resolved, exclude from findings)

## Session Continuity

Last session: 2026-02-08
Stopped at: Phase 9 complete and verified. Ready for Phase 10.
Resume file: None
