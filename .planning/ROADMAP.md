# Roadmap: Alerta Escuela Equity Audit

## Completed Milestones

- [x] **v1.0-analysis-complete** (2026-02-07 -> 2026-02-13) — Full equity audit pipeline: ENAHO data loading, 3 models, fairness analysis across 7 dimensions + 3 intersections, SHAP interpretability, 7 bilingual findings exported for M4 site. [Details](milestones/v1.0-ROADMAP.md)
- [x] **v2.0-publication** (2026-02-17) — Publication-quality figures, ACM LaTeX paper with 8 auto-generated tables, public repo cleanup (README, CITATION.cff, LICENSE, pipeline script). [Details](milestones/v2.0-ROADMAP.md)
- [x] **v3.0-submission** (2026-02-17) — Paper expanded to full academic depth (21 pages), 43 bibliography entries, Limitations + Ethical Considerations sections. [Details](milestones/v3.0-ROADMAP.md)
- [x] **v3.1-paper-strengthening** (2026-02-17) — Bootstrap CIs, hypothesis tests, RF + MLP models, lift/calibration analysis, paper rewrite with proxy framing, review fixes. [Details](milestones/v3.1-ROADMAP.md)
- [x] **v3.2-jedm-revision** (2026-02-18) — Paper revised to 18 pages for JEDM: pooled CI analysis, feature ablation, power analysis, page reduction. [Details](milestones/v3.2-ROADMAP.md)

<details>
<summary>v4.0-model-experiments (Phases 26-29) - SHIPPED 2026-03-01</summary>

Experiment with model improvements to test whether better features strengthen or change fairness findings. 6 new features engineered, all 5 models retrained (+9-14% PR-AUC), castellano FNR disparity confirmed as persistent and algorithm-independent. Paper updated to 23 pages.

- [x] **Phase 26: Feature Engineering** - 3/3 plans complete
- [x] **Phase 27: Model Retraining** - 2/2 plans complete
- [x] **Phase 28: Fairness Re-Analysis** - 2/2 plans complete
- [x] **Phase 29: Interpretation & Paper Update** - 2/2 plans complete

</details>

## Current Milestone: v4.1-readability-polish

Pre-submission content polish addressing anticipated JEDM reviewer concerns. Strengthen framing, tighten structure, fix compliance gaps. All work is paper-focused (paper/main.tex) except COMP-01 which requires running analysis code.

### Phases

**Phase Numbering:**
- Continues from v4.0 (Phase 30+). Phases 26-29 completed in v4.0.

- [ ] **Phase 30: Compliance Foundations** - Verify threshold-invariance of FNR rank order and rewrite AI declaration per JEDM format
- [ ] **Phase 31: Framing & Literature** - Strengthen generalizability framing, consolidate contributions, connect to broader EWS literature, cite proxy audit work
- [ ] **Phase 32: Structural Tightening** - Tighten Section 6.2, reduce cross-architecture redundancy, simplify abstract, review table necessity

### Phase Details

#### Phase 30: Compliance Foundations
**Goal**: Paper's methodological claims are empirically verified and its AI declaration meets JEDM's section-specific format requirements.
**Depends on**: v4.0 complete (23-page paper with v2 model results)
**Requirements**: COMP-01, COMP-02
**Success Criteria** (what must be TRUE):
  1. FNR rank order (castellano > quechua > other_indigenous) is verified as invariant across a range of classification thresholds (not just the optimal threshold), with results stated in Methods Section 4.4
  2. AI declaration names specific paper sections where AI tools were used, following JEDM's required format (not just task categories like "writing assistance")
  3. Paper compiles without errors after both changes
**Plans:** 1 plan
Plans:
- [ ] 30-01-PLAN.md — Threshold sweep analysis + paper compliance updates (Methods 4.4 + AI declaration)

#### Phase 31: Framing & Literature
**Goal**: Paper positions the proxy audit framework as a generalizable contribution beyond the Peru case study, grounded in cross-domain proxy audit literature.
**Depends on**: Phase 30 (threshold-invariance result informs framing language)
**Requirements**: FRAME-01, FRAME-02, FRAME-03, FRAME-04
**Success Criteria** (what must be TRUE):
  1. Abstract opens with the generalizable proxy audit framework claim before introducing Peru-specific details, establishing replicability for any country with nationally representative household surveys
  2. Introduction contribution bullets are consolidated (original 1 & 4 merged) and the surveillance-invisibility axis is elevated to the contribution list
  3. Discussion contains a paragraph connecting spatial-feature SHAP findings to implications for any EWS using geographic/structural predictors, not limited to Peru
  4. Proxy audit literature from outside education (criminal justice, lending, hiring) is cited or explicitly differentiated, strengthening the Related Work section
  5. Paper compiles without errors after all framing changes
**Plans**: TBD

#### Phase 32: Structural Tightening
**Goal**: Paper is tighter and more focused, with redundancy removed and every table earning its place.
**Depends on**: Phase 31 (framing changes may affect section flow)
**Requirements**: STRC-01, STRC-02, STRC-03, STRC-04
**Success Criteria** (what must be TRUE):
  1. Section 6.2 (Other Demographic Dimensions) is either folded into 6.1 or condensed to one paragraph, eliminating standalone section bloat
  2. Cross-architecture model comparison content appears once (in Section 5 or Section 6, not both), reducing redundancy between Sections 5.1/5.2 and 6.1
  3. Abstract mentions v2/31-feature model in at most one clause (or removes it entirely), focusing reader attention on primary findings
  4. All 12 tables reviewed: unnecessary tables trimmed or moved to appendix, with a clear rationale for each kept in-body
  5. Paper compiles without errors and reads as a cohesive narrative without structural gaps from removed content
**Plans**: TBD

### Deferred

- Phase 20: Press & Media Materials (from v3.0, after JEDM submission)

### Progress

**Execution Order:**
Phase 30 -> 31 -> 32

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 30. Compliance Foundations | 0/1 | Not started | - |
| 31. Framing & Literature | 0/TBD | Not started | - |
| 32. Structural Tightening | 0/TBD | Not started | - |
