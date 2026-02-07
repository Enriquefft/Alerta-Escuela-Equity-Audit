# Alerta Escuela Equity Audit

## What This Is

The first independent equity audit of Peru's "Alerta Escuela" — a LightGBM-based ML system deployed by MINEDU in October 2020 to predict student dropout risk. The system uses gender, mother tongue, and nationality as predictive features with zero published fairness analysis. We build a reproducible ML pipeline using publicly available ENAHO survey data (2018–2024) to replicate the dropout prediction task, quantify fairness gaps across protected groups, and produce media-ready findings + exportable data for a scrollytelling website targeting Peruvian journalists.

## Core Value

**The fairness audit is the product.** Models exist to be audited, not to achieve SOTA. If the LightGBM gets 0.85 AUC, great. If it gets 0.75, also fine — the fairness gaps are the finding, not the accuracy.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Load ENAHO microdata (2018–2024) with correct delimiter detection and P300A harmonization
- [ ] Construct dropout target as (P303==1 & P306==2) with survey-weighted validation
- [ ] Merge district-level admin dropout rates, Census 2017, VIIRS nightlights, Censo Escolar via UBIGEO
- [ ] Engineer all features per proxy mapping (19+ model features, 5 protected attributes)
- [ ] Compute survey-weighted descriptive statistics across 6 fairness dimensions
- [ ] Train logistic regression baseline, LightGBM (Optuna-tuned), and XGBoost with temporal splits
- [ ] Calibrate best model, export LightGBM to ONNX, evaluate on test set exactly once
- [ ] Compute subgroup fairness metrics (TPR, FPR, FNR, precision, PR-AUC) across all dimensions + 3 intersections
- [ ] Global + regional SHAP analysis with ES_PERUANO and ES_MUJER effect quantification
- [ ] Cross-validate fairness gaps against district administrative data
- [ ] Distill 5–7 media-ready findings with Spanish translations and export all JSON files for M4 site

### Out of Scope

- Replicating Alerta Escuela's exact model — we don't have SIAGIE access
- Web application — M4 scrollytelling site is a separate repo
- Real-time prediction or deployment
- Deep learning (no TabNet, neural networks)
- Experiment tracking tools (MLflow, DVC, W&B) — export metrics as JSON
- Geospatial processing libraries (geopandas, GDAL) — all geo data is pre-aggregated
- OAuth login, mobile apps, real-time chat

## Context

- **End consumer:** Peruvian journalists (Ojo Público, El Comercio, La República). Everything must be explainable to a non-technical audience in Spanish.
- **Data:** ENAHO is cross-sectional household survey (~25K school-age per year). Not the same as SIAGIE administrative records. Our dropout measure (~14% survey-based) differs from administrative (~2%) — this is well-documented measurement difference, not error.
- **Alerta Escuela performance:** Published ROC-AUC 0.84–0.89 across levels, but FNR of 57–64% (misses most actual dropouts). Uses 31 features from 5 administrative sources. Zero fairness analysis published.
- **Critical data quirk:** ENAHO delimiter changes (pipe ≤2019, comma ≥2020). P300A mother tongue codes disaggregated in 2020 (code 3 split into codes 10–15). UBIGEO needs zero-padding.
- **Exported data feeds a Next.js scrollytelling site** with specific JSON schema contracts (findings.json, fairness_metrics.json, shap_values.json, choropleth.json, model_results.json, descriptive_tables.json, ONNX model).

## Constraints

- **Tech stack**: Python 3.12, polars (primary), sklearn/lightgbm/xgboost/fairlearn/shap, Nix flakes + uv — locked, do not change
- **No pandas**: Use polars for data processing. `.to_pandas()` only at sklearn/fairlearn/shap boundary
- **Models**: Exactly 3 — LogReg, LightGBM, XGBoost. No more.
- **Survey weights**: ALL metrics must use FACTOR07. Every metric function must accept sample_weight.
- **Temporal split**: Train 2018–2022, validate 2023, test 2024. Test set touched ONCE in Phase 7.
- **Primary metric**: PR-AUC (not ROC-AUC) due to class imbalance
- **Export schemas**: Must match M4 site JSON contracts exactly (Section 11 of specs.md)
- **Environment**: Nix flakes for system dependencies, uv for Python packages

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use ENAHO survey data instead of SIAGIE | SIAGIE is not publicly accessible; ENAHO enables reproducible independent audit | — Pending |
| Polars over pandas | Performance on ~180K rows, modern API; pandas only at sklearn boundary | — Pending |
| PR-AUC as primary metric | Class imbalance (~14% dropout rate) makes ROC-AUC misleading | — Pending |
| Fairlearn over AIF360 | Better MetricFrame API with sample_weight support | — Pending |
| 7-year pooled dataset | More statistical power for subgroup analysis, especially small indigenous groups | — Pending |
| Temporal split over random split | Prevents future data leakage, mirrors real deployment scenario | — Pending |
| specs.md as authoritative source | Single source of truth for all implementation details | — Pending |

---
*Last updated: 2026-02-07 after initialization*
