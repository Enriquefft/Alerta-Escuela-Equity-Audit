# Roadmap: Alerta Escuela Equity Audit

## Completed Milestones

- [x] **v1.0-analysis-complete** (2026-02-07 → 2026-02-13) — Full equity audit pipeline: ENAHO data loading, 3 models, fairness analysis across 7 dimensions + 3 intersections, SHAP interpretability, 7 bilingual findings exported for M4 site. [Details](milestones/v1.0-ROADMAP.md)
- [x] **v2.0-publication** (2026-02-17) — Publication-quality figures, ACM LaTeX paper with 8 auto-generated tables, public repo cleanup (README, CITATION.cff, LICENSE, pipeline script). [Details](milestones/v2.0-ROADMAP.md)
- [x] **v3.0-submission** (2026-02-17) — Paper expanded to full academic depth (21 pages), 43 bibliography entries, Limitations + Ethical Considerations sections. [Details](milestones/v3.0-ROADMAP.md)
- [x] **v3.1-paper-strengthening** (2026-02-17) — Bootstrap CIs, hypothesis tests, RF + MLP models, lift/calibration analysis, paper rewrite with proxy framing, review fixes. [Details](milestones/v3.1-ROADMAP.md)

## Current Milestone: v3.2-jedm-revision

Revise paper for JEDM (Journal of Educational Data Mining) submission. Fix critical issues (abstract framing, urban indigenous CI, length), add missing analysis (feature ablation, power analysis), translate figures to English, and trim from 22 to 15–18 pages.

### Phases

**Phase Numbering:**
- Continues from v3.1 (Phase 22+). Phases 20-21 removed/deferred.

- [ ] **Phase 22: Analysis Code** — Pool val+test for urban indigenous CI, feature ablation, power analysis, English figure labels
- [ ] **Phase 23: Paper Restructure** — Abstract rewrite, structural cuts (delete 7.1, compress related work, move figures to appendix), content additions (normative discussion, expanded conclusion, ablation/power results), moderate fixes
- [ ] **Phase 24: Style & Polish** — Rename "algorithm independence", trim prose restating tables, writing style cleanup, final page count verification

### Phase Details

#### Phase 22: Analysis Code
**Goal**: Produce the new analysis artifacts that the paper rewrite depends on — pooled urban indigenous CI, feature ablation FNR table, power analysis calculation, and English-labeled figures.
**Depends on**: v3.1 complete (all models trained, fairness metrics computed)
**Requirements**: CODE-01, CODE-02, CODE-03, CODE-04
**Success Criteria** (what must be TRUE):
  1. Pooled val+test urban indigenous FNR computed with bootstrap CI (n≈170–180)
  2. Feature ablation: two LightGBM variants trained (individual-only, spatial-only), FNR by language reported
  3. Power analysis: required n for FNR=0.75 vs 0.63 at 80% power calculated
  4. All figures regenerated with English axis labels
  5. All existing gate tests still pass
**Human Gate**: No — code-only, verified by tests
**Plans**: 1 plan

#### Phase 23: Paper Restructure
**Goal**: Major paper revision — fix abstract, cut ~4.5 pages (delete 7.1, compress related work, move figures to appendix), add new content (normative discussion, expanded conclusion, ablation/power results, PR-AUC interpretation, restored references), condense proxy disclaimer.
**Depends on**: Phase 22 (ablation table, pooled CI, power analysis, English figures)
**Requirements**: PAPER-20 through PAPER-32
**Success Criteria** (what must be TRUE):
  1. Abstract leads with contribution, not negation; urban indigenous FNR framed as hypothesis-generating
  2. Section 7.1 deleted, discussion starts at spatial proxy mechanism
  3. Related work ≤1 page, three thematic paragraphs
  4. Figures 2, 3, 5 in appendix; body retains Figures 1, 4, 6, 7, 8
  5. Normative fairness discussion present (1–2 paragraphs)
  6. Conclusion expanded to 5–7 sentences
  7. Feature ablation table and power analysis paragraph integrated
  8. Kearns et al. (2018) restored in Related Work
  9. Proxy disclaimer condensed to 2 sentences in contributions paragraph
  10. Paper compiles without errors
**Human Gate**: No — style pass follows
**Plans**: 1 plan

#### Phase 24: Style & Polish
**Goal**: Final pass — rename terminology, trim verbosity, verify page count is 15–18 pages.
**Depends on**: Phase 23 (structure finalized)
**Requirements**: PAPER-25, PAPER-33, PAPER-34, PAPER-35
**Success Criteria** (what must be TRUE):
  1. "Algorithm independence" renamed to "cross-architecture consistency" throughout
  2. Prose accompanying tables/figures contains only interpretation, not description
  3. Hedging stacks, meta-commentary, and redundant connectives removed
  4. Active voice used by default
  5. Paper compiles, 15–18 pages
**Human Gate**: Yes — final review before JEDM submission
**Plans**: 1 plan

### Deferred

- Phase 20: Press & Media Materials (from v3.0, after JEDM submission)

### Removed

- Phase 21: arXiv Submission Prep (removed — JEDM has its own portal)

### Progress

**Execution Order:**
Phase 22 → 23 → 24

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 22. Analysis Code | 0/1 | Pending | — |
| 23. Paper Restructure | 0/1 | Pending | — |
| 24. Style & Polish | 0/1 | Pending | — |
