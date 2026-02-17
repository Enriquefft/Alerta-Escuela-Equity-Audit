# Roadmap: Alerta Escuela Equity Audit

## Completed Milestones

- [x] **v1.0-analysis-complete** (2026-02-07 → 2026-02-13) — Full equity audit pipeline: ENAHO data loading, 3 models, fairness analysis across 7 dimensions + 3 intersections, SHAP interpretability, 7 bilingual findings exported for M4 site. [Details](milestones/v1.0-ROADMAP.md)
- [x] **v2.0-publication** (2026-02-17) — Publication-quality figures, ACM LaTeX paper with 8 auto-generated tables, public repo cleanup (README, CITATION.cff, LICENSE, pipeline script). [Details](milestones/v2.0-ROADMAP.md)
- [x] **v3.0-submission** (2026-02-17) — Paper expanded to full academic depth (21 pages), 43 bibliography entries, Limitations + Ethical Considerations sections. [Details](milestones/v3.0-ROADMAP.md)

## Current Milestone: v3.1-paper-strengthening

Address 5 critical weaknesses identified in peer review to make the paper defensible at FAccT 2027:
1. Proxy problem (ENAHO ≠ SIAGIE) — reframe title and claims
2. Low PR-AUC barely above baseline — add lift/calibration evidence
3. Small sample sizes at intersections — add bootstrap CIs
4. No statistical testing for disparities — add hypothesis tests
5. Algorithm independence overstated — add RF + MLP models

### Phases

**Phase Numbering:**
- Continues from v3.0 (Phase 16+)

- [ ] **Phase 16: Statistical Rigor** — Bootstrap CIs for all subgroup fairness metrics, survey-weighted hypothesis tests for disparity significance
- [ ] **Phase 17: Model Expansion + Predictive Validity** — Train RF + MLP, compute fairness metrics for 5 models, lift analysis, calibration-by-decile
- [ ] **Phase 18: Paper Rewrite** — New title, ENAHO-vs-SIAGIE comparison, integrate CIs/tests/lift/new models, reframe proxy claims

### Phase Details

#### Phase 16: Statistical Rigor
**Goal**: Add confidence intervals and hypothesis tests to all fairness metrics so every disparity claim in the paper is backed by statistical evidence.
**Depends on**: Phase 15 (complete — paper exists to strengthen)
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04
**Success Criteria** (what must be TRUE):
  1. Bootstrap CIs (1000+ replicates) computed for FNR, FPR, precision, PR-AUC per subgroup across all 7 dimensions + 3 intersections
  2. Survey-weighted Wald tests produce p-values for key FNR/FPR comparisons (castellano vs each indigenous group)
  3. Intersectional CIs flag groups where width > 0.3 (e.g., urban other-indigenous)
  4. `fairness_metrics.json` updated with `ci_lower`, `ci_upper`, and `p_value` fields
  5. All existing gate tests still pass
**Human Gate**: No — code-only, verified by tests
**Plans**: 1 plan

#### Phase 17: Model Expansion + Predictive Validity
**Goal**: Train RF and MLP to test algorithm independence across model families (not just GBM variants), and add lift/calibration analysis to demonstrate the model provides meaningful signal above base rate.
**Depends on**: Phase 16 (CIs infrastructure reused for new models)
**Requirements**: VALID-01, VALID-02, VALID-03, MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05
**Success Criteria** (what must be TRUE):
  1. Random Forest trained with FACTOR07 weights, hyperparameters tuned
  2. MLP trained with standardized features and FACTOR07 weights
  3. Fairness metrics (FNR/FPR/precision/PR-AUC per subgroup) computed for RF and MLP
  4. Cross-model comparison table: FNR by language for all 5 models
  5. `model_results.json` updated with RF and MLP sections
  6. Lift > 1.0 at top decile demonstrated
  7. Calibration-by-decile analysis shows calibrated probabilities track observed rates
  8. Brier skill score > 0 (model beats prevalence baseline)
  9. All existing gate tests still pass
**Human Gate**: No — code-only, verified by tests
**Plans**: 2 plans (17-01: code, 17-02: execute)

#### Phase 18: Paper Rewrite
**Goal**: Rewrite paper sections to incorporate all new evidence (CIs, tests, RF+MLP, lift analysis) and reframe the proxy audit claim honestly.
**Depends on**: Phase 16 + Phase 17 (all new analysis complete)
**Requirements**: PAPER-13, PAPER-14, PAPER-15, PAPER-16, PAPER-17, PAPER-18, PAPER-19, VALID-04
**Success Criteria** (what must be TRUE):
  1. Title changed to reflect proxy audit framing (no "Alerta Escuela" as if auditing the actual system)
  2. ENAHO-vs-SIAGIE feature comparison table present in Data section
  3. Introduction frames paper as "what CAN emerge" not "what Alerta Escuela DOES"
  4. All fairness tables include CI columns
  5. Predictive Validity subsection discusses PR-AUC vs baseline with lift evidence
  6. Algorithm-independence section covers 5 model families (LR, LightGBM, XGBoost, RF, MLP)
  7. Statistical significance reported for key disparity comparisons
  8. n=89 urban-indigenous CI reported prominently with width caveat
  9. Paper compiles without errors
**Human Gate**: Yes — review strengthened paper before proceeding to media/arXiv
**Plans**: 3 plans

Plans:
- [ ] 18-01-PLAN.md — Framing sweep: title, abstract, intro proxy framing, "What We Are NOT Claiming", Discussion/Conclusion/Limitations updates
- [ ] 18-02-PLAN.md — New evidence: ENAHO-vs-SIAGIE table, Predictive Validity subsection, Algorithm Independence subsection, CI integration into fairness tables, fig08, Related Work cuts
- [ ] 18-03-PLAN.md — Final compile, page count verification, human review gate

### Deferred (from v3.0)

These phases resume after v3.1 is approved:
- Phase 19: Press & Media Materials (was v3.0 Phase 16)
- Phase 20: arXiv Submission Prep (was v3.0 Phase 17)

### Progress

**Execution Order:**
Phase 16 → 17 (partially parallelizable with 16) → 18

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 16. Statistical Rigor | 0/1 | Pending | — |
| 17. Model Expansion + Predictive Validity | 2/2 | Planned | — |
| 18. Paper Rewrite | 0/3 | Planned | — |
