# Roadmap: Alerta Escuela Equity Audit

## Completed Milestones

- [x] **v1.0-analysis-complete** (2026-02-07 → 2026-02-13) — Full equity audit pipeline: ENAHO data loading, 3 models, fairness analysis across 7 dimensions + 3 intersections, SHAP interpretability, 7 bilingual findings exported for M4 site. [Details](milestones/v1.0-ROADMAP.md)

## Current Milestone: v2.0-publication

Transform v1.0 analysis into citable academic paper with publication-quality figures, auto-generated LaTeX tables, and a clean public repo.

### Phases

**Phase Numbering:**
- Continues from v1.0 (Phase 12+)
- Decimal phases (e.g., 12.1): Urgent insertions

- [ ] **Phase 12: Publication Figures** - New headline figures (FNR bar, FNR×rurality heatmap, combined PR curves) + polish existing for print
- [ ] **Phase 13: LaTeX Template + Table Generation** - ACM template with structure, auto-generate 8 tables from JSON exports
- [ ] **Phase 14: Repo Cleanup for Public Release** - README, LICENSE, CITATION.cff, reproducibility verification

### Phase Details

#### Phase 12: Publication Figures
**Goal**: Create 2 new headline figures and polish all existing figures to publication quality with consistent styling, proper feature names, and dual-format export (PNG 300dpi + PDF vector)
**Depends on**: v1.0 exports (complete)
**Requirements**: FIG-01, FIG-02, FIG-03, FIG-04, FIG-05
**Success Criteria** (what must be TRUE):
  1. FNR/FPR grouped bar chart by language group exists — the "money figure" for paper and media
  2. FNR heatmap for language × rurality intersection exists — shows urban indigenous blind spot (FNR=75.3% vs 17.1%)
  3. Combined PR curves panel (3 models) exists as single figure
  4. All figures use consistent styling: same font family/sizes, color palette, axis labels
  5. All figures exported as both PNG (300dpi) and PDF vector in `paper/figures/`
**Human Gate**: Yes — review figures for publication readiness
**Plans**: 1 plan

#### Phase 13: LaTeX Template + Table Generation
**Goal**: ACM-format LaTeX template with full paper structure and 8 auto-generated tables pulling data directly from v1.0 JSON exports
**Depends on**: Phase 12 (figures needed for template)
**Requirements**: TEX-01, TEX-02, TEX-03, TEX-04, TEX-05, TEX-06, TEX-07, TEX-08
**Success Criteria** (what must be TRUE):
  1. `paper/main.tex` compiles with pdflatex/latexmk and produces a well-formatted PDF
  2. All 8 tables auto-generated from JSON exports via Python script — no manual data entry
  3. Table numbers match JSON source values exactly (spot-check 5 cells per table)
  4. Paper structure has all 8 sections + appendix with placeholder text
  5. Figures from Phase 12 included and properly referenced
**Human Gate**: Yes — review table accuracy and template layout
**Plans**: 1 plan

#### Phase 14: Repo Cleanup for Public Release
**Goal**: Repository is ready for public GitHub release with proper documentation, licensing, citation info, and verified reproducibility
**Depends on**: Phase 13
**Requirements**: REPO-01, REPO-02, REPO-03, REPO-04, REPO-05
**Success Criteria** (what must be TRUE):
  1. Root README.md explains project, setup instructions, and how to reproduce
  2. LICENSE file present (MIT or Apache 2.0)
  3. CITATION.cff file present with proper academic citation metadata
  4. `scripts/rerun_pipeline.sh` runs end-to-end without errors (or documents exactly what manual steps are needed)
  5. No internal files (.planning/, .claude/) exposed in public-facing content
**Human Gate**: Yes — final review before making repo public
**Plans**: 1 plan

### Progress

**Execution Order:**
Phase 12 → 13 → 14

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 12. Publication Figures | 0/1 | Pending | — |
| 13. LaTeX Template + Table Generation | 0/1 | Pending | — |
| 14. Repo Cleanup for Public Release | 0/1 | Pending | — |
