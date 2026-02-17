# Alerta Escuela Equity Audit

## What This Is

The first independent equity audit of Peru's "Alerta Escuela" — a LightGBM-based ML system deployed by MINEDU in October 2020 to predict student dropout risk. The system uses gender, mother tongue, and nationality as predictive features with zero published fairness analysis. We build a reproducible ML pipeline using publicly available ENAHO survey data (2018–2023) to replicate the dropout prediction task, quantify fairness gaps across protected groups, and produce media-ready findings + exportable data for a scrollytelling website targeting Peruvian journalists.

## Core Value

**The fairness audit is the product.** Models exist to be audited, not to achieve SOTA. If the LightGBM gets 0.85 AUC, great. If it gets 0.75, also fine — the fairness gaps are the finding, not the accuracy.

## Current State

**Shipped: v1.0-analysis-complete** (tagged 2026-02-17)

All v1 requirements delivered:
- 150,135 ENAHO rows across 6 years (2018-2023) with 25 model features
- 3 trained models (LogReg, LightGBM, XGBoost) with Platt-calibrated LightGBM as primary
- Fairness analysis across 7 dimensions + 3 intersections using fairlearn MetricFrame
- SHAP interpretability (global, regional, interaction, 10 student profiles)
- 7 bilingual findings distilled for M4 scrollytelling site
- 7 JSON exports + ONNX model validated against M4 schema contracts
- 95 gate tests passing across 12 test files

### Key Findings

1. System misses 63.3% of actual dropouts (FNR=0.633 for castellano speakers)
2. Indigenous students over-flagged (FPR=0.537) — "surveillance bias" vs "invisibility bias"
3. Dropout predicted by spatial/structural features (age, literacy, poverty), not identity features
4. Algorithm-independent: LightGBM/XGBoost ratio = 1.0006

### Exports

All in `data/exports/`: findings.json, fairness_metrics.json, shap_values.json, choropleth.json, model_results.json, descriptive_tables.json, onnx/lightgbm_dropout.onnx

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use ENAHO survey data instead of SIAGIE | SIAGIE is not publicly accessible; ENAHO enables reproducible independent audit | Validated — 150K rows, sufficient for subgroup analysis |
| Polars over pandas | Performance on ~150K rows, modern API; pandas only at sklearn boundary | Validated — clean pipeline |
| PR-AUC as primary metric | Class imbalance (~14% dropout rate) makes ROC-AUC misleading | Validated — PR-AUC=0.2611 |
| Fairlearn over AIF360 | Better MetricFrame API with sample_weight support | Validated — 7 dims + 3 intersections |
| 6-year pooled dataset | More statistical power for subgroup analysis, especially small indigenous groups | Validated — other_indigenous n=668 |
| Temporal split over random split | Prevents future data leakage, mirrors real deployment scenario | Validated — val-test gap 0.0233 |
| specs.md as authoritative source | Single source of truth for all implementation details | Validated |

## Next Milestone: v2.0-publication

Transform v1.0 analysis into citable academic paper:
- Phase 12: Publication figures (FNR bar chart, FNR×rurality heatmap, combined PR curves, polish)
- Phase 13: LaTeX template + auto-generated tables from JSON exports
- Phase 14: Repo cleanup for public release (README, LICENSE, CITATION.cff, reproducibility)

### Out of Scope for v2.0 (Manual Work TODO)

- Write paper sections (Introduction through Conclusion)
- Write press summaries (Spanish + English)
- Write blog posts
- Pitch media outlets (Ojo Publico, El Comercio)
- Submit to arXiv, FAccT 2027, AIES 2026

### Deferred to v3.0+

- Automated pipeline re-run when new ENAHO year released
- Interactive fairness metrics dashboard
- Causal analysis of language → dropout pathway
- Bias mitigation experiments
- Censo Escolar school-level features

## Constraints

- **Tech stack**: Python 3.12, polars (primary), sklearn/lightgbm/xgboost/fairlearn/shap, Nix flakes + uv — locked
- **No pandas**: Use polars for data processing. `.to_pandas()` only at sklearn/fairlearn/shap boundary
- **Models**: Exactly 3 — LogReg, LightGBM, XGBoost. No more.
- **Survey weights**: ALL metrics must use FACTOR07
- **Primary metric**: PR-AUC (not ROC-AUC)
- **Export schemas**: Must match M4 site JSON contracts (Section 11 of specs.md)

## Context

- **End consumer:** Peruvian journalists (Ojo Público, El Comercio, La República)
- **Data:** ENAHO cross-sectional household survey (~25K school-age per year)
- **Alerta Escuela:** Published ROC-AUC 0.84–0.89, FNR 57–64%. Zero fairness analysis published.
- **Exported data feeds** a Next.js scrollytelling site with specific JSON schema contracts

## Out of Scope

- Replicating Alerta Escuela's exact model — no SIAGIE access
- Web application / M4 site — separate repo
- Real-time prediction or deployment
- Deep learning, experiment tracking, geospatial processing
- Bias mitigation / debiasing — audit scope only

---
*Last updated: 2026-02-17 after v1.0 milestone completion*
