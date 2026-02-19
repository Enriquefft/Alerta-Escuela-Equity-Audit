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

- [x] **Phase 22: Analysis Code** — Pool val+test for urban indigenous CI, feature ablation, power analysis (reframed as data ceiling argument), English figure labels, fix JSON flagging
- [x] **Phase 23: Paper Restructure** — Abstract rewrite, structural cuts (delete 7.1, compress related work, move figures to appendix), content additions (normative discussion, expanded conclusion, ablation/power results), moderate fixes
- [x] **Phase 24: Style & Polish** — Rename "algorithm independence", trim prose restating tables, writing style cleanup, final page count verification
- [x] **Phase 25: Page Count Reduction** — Moved 4 tables + 2 figures to appendix, condensed discussion/data/intro/limitations, removed PR curves figure (22→18 pages)

### Phase Details

#### Phase 22: Analysis Code
**Goal**: Produce the new analysis artifacts that the paper rewrite depends on — pooled urban indigenous CI (with contingency if it doesn't narrow), feature ablation FNR table, power analysis as methodological limitation argument, English-labeled figures, and JSON flag fix.
**Depends on**: v3.1 complete (all models trained, fairness metrics computed)
**Requirements**: CODE-01, CODE-02, CODE-03, CODE-04, CODE-05
**Success Criteria** (what must be TRUE):
  1. Pooled val+test urban indigenous FNR computed with simple weighted bootstrap CI (n≈170–180). Credibility threshold: CI lower bound > 0.50
  2. Contingency documented: if CI lower bound ≤ 0.50, castellano invisibility (FNR=0.633) becomes headline finding
  3. Feature ablation: two LightGBM variants trained (individual-only, spatial-only), FNR by language reported
  4. Power analysis: minimum n for credible intersectional auditing, translated to ENAHO years, framed as data ceiling argument for SIAGIE access
  5. All figures regenerated with English axis labels
  6. `flagged_small_sample` set for other_indigenous_urban (n=89) in fairness_metrics.json
  7. All existing gate tests still pass
**Human Gate**: No — code-only, verified by tests
**Plans**: 1 plan

#### Phase 23: Paper Restructure
**Goal**: Major paper revision — fix abstract, cut ~4.5 pages (delete 7.1, compress related work, move figures to appendix), add new content (normative discussion, expanded conclusion, ablation/power results, PR-AUC interpretation, restored references), condense proxy disclaimer.
**Depends on**: Phase 22 (ablation table, pooled CI, power analysis, English figures)
**Requirements**: PAPER-20 through PAPER-32
**Success Criteria** (what must be TRUE):
  1. Abstract leads with castellano invisibility as primary finding; urban indigenous framed as hypothesis-generating or secondary (depending on CODE-01 result)
  2. Section 7.1 deleted, discussion starts at spatial proxy mechanism
  3. Related work ≤1 page, three thematic paragraphs
  4. Figures 2, 3, 5 in appendix; body retains Figures 1, 4, 6, 7, 8
  5. Normative fairness discussion present (1–2 paragraphs)
  6. Conclusion expanded to 5–7 sentences
  7. Feature ablation table and power analysis integrated
  7a. Power analysis framed as methodological contribution: survey data hits a ceiling for intersectional auditing, arguing for SIAGIE access
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

#### Phase 25: Page Count Reduction
**Goal**: Evaluate what can be trimmed vs what's essential, then cut paper from 22 to 15–18 pages. Research-first: audit each section's contribution before cutting.
**Depends on**: Phase 24 (style finalized)
**Success Criteria** (what must be TRUE):
  1. Section-by-section audit completed identifying trimmable vs essential content
  2. Cuts preserve all key findings, methodology, and analytical rigor
  3. Paper compiles, 15–18 pages
  4. No orphaned references, figures, or table citations
**Human Gate**: Yes — final review before JEDM submission
**Plans**: 1 plan

### Deferred

- Phase 20: Press & Media Materials (from v3.0, after JEDM submission)

### Removed

- Phase 21: arXiv Submission Prep (removed — JEDM has its own portal)

### Progress

**Execution Order:**
Phase 22 → 23 → 24 → 25

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 22. Analysis Code | 1/1 | Complete | 2026-02-18 |
| 23. Paper Restructure | 1/1 | Complete | 2026-02-18 |
| 24. Style & Polish | 1/1 | Complete | 2026-02-18 |
| 25. Page Count Reduction | 1/1 | Complete | 2026-02-18 |
