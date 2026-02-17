# Roadmap: Alerta Escuela Equity Audit

## Completed Milestones

- [x] **v1.0-analysis-complete** (2026-02-07 → 2026-02-13) — Full equity audit pipeline: ENAHO data loading, 3 models, fairness analysis across 7 dimensions + 3 intersections, SHAP interpretability, 7 bilingual findings exported for M4 site. [Details](milestones/v1.0-ROADMAP.md)
- [x] **v2.0-publication** (2026-02-17) — Publication-quality figures, ACM LaTeX paper with 8 auto-generated tables, public repo cleanup (README, CITATION.cff, LICENSE, pipeline script). [Details](milestones/v2.0-ROADMAP.md)

## Current Milestone: v3.0-submission

Get the paper published (arXiv preprint first, FAccT 2027 later) and generate media coverage for O-1 visa evidence.

### Phases

**Phase Numbering:**
- Continues from v2.0 (Phase 15+)

- [ ] **Phase 15: Paper Expansion** - Expand all sections to full academic length, add Limitations + Ethical Considerations, expand bibliography
- [ ] **Phase 16: Press & Media Materials** - Press summaries (ES/EN), blog posts (ES/EN), media pitch email templates
- [ ] **Phase 17: arXiv Submission Prep** - .bbl generation, figure packaging, tar.gz submission package, test compilation

### Phase Details

#### Phase 15: Paper Expansion
**Goal**: Expand draft paper from ~4 pages of concise text to full academic paper (15-20 pages for arXiv, no page limit). All sections fleshed out with proper academic depth, new sections added (Limitations, Ethical Considerations).
**Depends on**: v2.0 (complete)
**Requirements**: PAPER-01 through PAPER-12
**Success Criteria** (what must be TRUE):
  1. Paper compiles and produces 15+ page PDF (excluding references)
  2. Introduction frames 2-3 explicit research questions answered by the paper
  3. Related Work positions this paper relative to at least 10 prior works
  4. Limitations subsection acknowledges ENAHO proxy, synthetic data, COVID 2020, small samples
  5. Ethical Considerations includes positionality statement and generative AI disclosure
  6. Bibliography has 30+ entries
**Human Gate**: Yes — review paper for academic quality before arXiv
**Plans**: 2 plans

Plans:
- [ ] 15-01-PLAN.md — Bibliography expansion (43 entries) + Introduction with 3 RQs + Related Work with 4 subsections
- [ ] 15-02-PLAN.md — Expand Data/Methods/Results/Fairness + add Limitations, Ethical Considerations, Conclusion

#### Phase 16: Press & Media Materials
**Goal**: Draft all media materials needed to generate coverage for O-1 visa evidence: press summaries, blog posts, and pitch emails in both Spanish and English.
**Depends on**: Phase 15 (paper content needed for summaries)
**Requirements**: MEDIA-01 through MEDIA-05
**Success Criteria** (what must be TRUE):
  1. Spanish press summary exists (1 page, quotes-ready language)
  2. English press summary exists (1 page, international framing)
  3. Spanish blog post exists (800-1200 words, accessible narrative)
  4. English blog post exists (800-1200 words)
  5. Media pitch email templates exist for Ojo Publico, El Comercio, and generic
**Human Gate**: Yes — review tone and accuracy before sending
**Plans**: 1 plan

#### Phase 17: arXiv Submission Prep
**Goal**: Package paper for arXiv submission — .bbl file, relative paths, clean tar.gz, verified compilation.
**Depends on**: Phase 15 (final paper content)
**Requirements**: ARXIV-01 through ARXIV-05
**Success Criteria** (what must be TRUE):
  1. .bbl file generated and paper compiles without .bib
  2. All figure paths are relative (no absolute/local paths)
  3. Submission tar.gz contains only needed files (no .planning/, .claude/, unused figures)
  4. Paper compiles from extracted tar.gz in a clean directory
  5. arXiv metadata prepared (title, abstract, authors, cs.CY + cs.LG categories)
**Human Gate**: Yes — final review before arXiv upload
**Plans**: 1 plan

### Progress

**Execution Order:**
Phase 15 → 16 (can start during 15 review) → 17

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 15. Paper Expansion | 0/2 | Pending | — |
| 16. Press & Media Materials | 0/1 | Pending | — |
| 17. arXiv Submission Prep | 0/1 | Pending | — |
