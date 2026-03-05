# Alerta Escuela Equity Audit

## What This Is

An independent equity audit of dropout prediction in Peru using ENAHO survey data — a proxy for the "Alerta Escuela" ML system deployed by MINEDU. We build a reproducible ML pipeline using publicly available ENAHO survey data (2018–2023) to replicate the dropout prediction task, quantify fairness gaps across protected groups with statistical rigor, and produce media-ready findings for Peruvian journalists.

## Core Value

**The fairness audit is the product.** Models exist to be audited, not to achieve SOTA. If the LightGBM gets 0.85 AUC, great. If it gets 0.75, also fine — the fairness gaps are the finding, not the accuracy.

## Current State

**Shipped: v4.1-readability-polish** (2026-03-05)

Paper polished for JEDM submission — framing, compliance, and structural improvements:
- FNR rank order verified threshold-invariant (0.05–0.20) and stated in Methods Section 4.4
- AI declaration rewritten with section-specific Claude Code usage per JEDM format
- Abstract leads with generalizable household-survey proxy audit claim
- Contribution bullets consolidated from 4→3; surveillance-invisibility axis elevated
- Three proxy audit citations added (Sandvig 2014, Adler 2018, Obermeyer 2019)
- EWS generalization paragraph in Discussion establishes findings as cross-domain
- Section 6.2 folded into 6.1; appendix reduced to 5 tables

**Paper state:** 23 pages, compiles cleanly, ready for JEDM submission

<details>
<summary>v4.0-model-experiments (2026-02-07 → 2026-03-01)</summary>

- 6 new features engineered (overage-for-grade + 4 interaction features)
- All 5 models retrained with 31-feature matrix (+9-14% PR-AUC improvement)
- Castellano FNR disparity confirmed PERSISTENT and algorithm-independent across all 5 models
- Paper updated to 23 pages with v2 robustness analysis
- Panel linkage assessed as infeasible (18.9% effective rate — negative result documented)
</details>

<details>
<summary>v3.2-jedm-revision (2026-02-18)</summary>

- Pooled CI analysis, feature ablation, power analysis
- Paper reduced to 18 pages for JEDM submission
</details>

<details>
<summary>v3.1-paper-strengthening (2026-02-17)</summary>

Paper strengthened with statistical rigor: bootstrap CIs, hypothesis tests, RF + MLP models, lift analysis, proxy framing rewrite, review fixes.
</details>

<details>
<summary>v3.0-submission (2026-02-17)</summary>

Paper expanded to full academic depth (21 pages), 43 bibliography entries, Limitations + Ethical Considerations added.
</details>

<details>
<summary>v2.0-publication (2026-02-17)</summary>

Publication-quality figures, ACM LaTeX paper, 8 auto-generated tables, public repo cleanup.
</details>

<details>
<summary>v1.0-analysis-complete (2026-02-07 → 2026-02-13)</summary>

150,135 ENAHO rows, 5 models, fairness across 7 dimensions + 3 intersections, SHAP, 7 bilingual findings, 7 JSON exports + ONNX.
</details>

### Key Findings

1. System misses 63.3% of actual dropouts (FNR=0.633 for castellano speakers)
2. Indigenous students over-flagged (FPR=0.537) — "surveillance bias" vs "invisibility bias"
3. Dropout predicted by spatial/structural features (age, literacy, poverty), not identity features
4. Cross-architecture consistent: all 5 model families show same FNR rank order

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
| JEDM over FAccT | Q3 journal, free APC, ~3-month review, no page limit pressure | v3.2 |
| Proxy audit framing | Position paper as replicable methodology, not single-country result | ✓ v4.1 — abstract/intro updated |
| Threshold-invariance verification | FNR rank order must hold across operational thresholds, not just optimal | ✓ v4.1 — confirmed 0.05–0.20 |

## Next Milestone: v4.2 (Planning)

**Current focus:** JEDM submission and post-submission work.

**Deferred (next milestone candidates):**
- Submit to JEDM via portal (SUBM-02)
- Zenodo archival for DOI (SUBM-01)
- Press & media materials in ES/EN (SUBM-03)
- Press & media materials (summaries, blog posts, pitch emails)

## Potential Future Milestones

- Automated pipeline re-run when new ENAHO year released
- Interactive fairness metrics dashboard
- Causal analysis of language → dropout pathway
- Bias mitigation experiments
- Censo Escolar school-level features

## Constraints

- **Tech stack**: Python 3.12, polars (primary), sklearn/lightgbm/xgboost/fairlearn/shap, Nix flakes + uv — locked
- **No pandas**: Use polars for data processing. `.to_pandas()` only at sklearn/fairlearn/shap boundary
- **Models**: 5 — LogReg, LightGBM, XGBoost, Random Forest, MLP
- **Survey weights**: ALL metrics must use FACTOR07
- **Primary metric**: PR-AUC (not ROC-AUC)
- **Export schemas**: Must match M4 site JSON contracts (Section 11 of specs.md)

## Context

- **End consumer:** Peruvian journalists (Ojo Público, El Comercio, La República)
- **Target venue:** JEDM (Journal of Educational Data Mining)
- **Data:** ENAHO cross-sectional household survey (~25K school-age per year)
- **Alerta Escuela:** Published ROC-AUC 0.84–0.89, FNR 57–64%. Zero fairness analysis published.
- **Exported data feeds** a Next.js scrollytelling site with specific JSON schema contracts

## Out of Scope

- Replicating Alerta Escuela's exact model — no SIAGIE access
- Web application / M4 site — separate repo
- Real-time prediction or deployment
- Deep learning (beyond sklearn MLP), experiment tracking, geospatial processing
- Bias mitigation / debiasing — audit scope only

---
*Last updated: 2026-03-05 after v4.1 milestone*
