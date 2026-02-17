# Requirements: v2.0-publication

**Defined:** 2026-02-17
**Core Value:** Transform v1.0 analysis into citable academic paper with publication-quality figures and reproducible artifacts.

## v2 Requirements

### Publication Figures

- [ ] **FIG-01**: FNR/FPR grouped bar chart by language group — headline figure for paper + media
- [ ] **FIG-02**: FNR heatmap for language x rurality intersection — shows urban indigenous blind spot
- [ ] **FIG-03**: Combined PR curves (3 models) in single panel figure
- [ ] **FIG-04**: Polish all existing figures for print (consistent font sizes, axis labels, color scheme, feature names)
- [ ] **FIG-05**: All figures export as both PNG (300dpi) and PDF vector for LaTeX

### LaTeX + Tables

- [ ] **TEX-01**: ACM or arXiv-friendly LaTeX template with full paper structure pre-filled (8 sections + appendix)
- [ ] **TEX-02**: Auto-generate Table 1: Model performance comparison (3 models x val/test) from model_results.json
- [ ] **TEX-03**: Auto-generate Table 2: Weighted dropout rates by language group (with CIs) from descriptive_tables.json
- [ ] **TEX-04**: Auto-generate Table 3: Fairness metrics by language group (FNR, FPR, TPR, precision, n) from fairness_metrics.json
- [ ] **TEX-05**: Auto-generate Table 4: Language x rurality intersection (FNR, n) from fairness_metrics.json
- [ ] **TEX-06**: Auto-generate Table 5: SHAP top 10 vs LR top 5 from shap_values.json
- [ ] **TEX-07**: Auto-generate Table A1: Full feature list with Alerta Escuela proxy mapping
- [ ] **TEX-08**: Auto-generate Table A2: P300A harmonization table

### Repo Cleanup

- [ ] **REPO-01**: Clean README.md for public release (project overview, setup, reproduce instructions)
- [ ] **REPO-02**: Add LICENSE file (MIT or Apache 2.0)
- [ ] **REPO-03**: Verify reproducibility: `scripts/rerun_pipeline.sh` runs end-to-end from raw data
- [ ] **REPO-04**: Remove or gitignore sensitive/internal files (.planning/, .claude/, internal notes)
- [ ] **REPO-05**: Add CITATION.cff for proper academic citation

## Out of Scope (Manual Work — TODO List)

These are tracked here for reference but are NOT code deliverables:

- [ ] Write paper Sections 1-3 (Introduction, Related Work, System Description)
- [ ] Write paper Section 4 (Data & Methods)
- [ ] Write paper Section 5 (Results) — pull from JSON exports
- [ ] Write paper Section 6 (Discussion) — urban indigenous blind spot narrative
- [ ] Write paper Section 7 (Recommendations) + Conclusion
- [ ] Write Abstract (after results finalized)
- [ ] Write 1-page press summary (Spanish)
- [ ] Write 1-page press summary (English)
- [ ] Write blog post (Spanish, 800-1200 words)
- [ ] Write blog post (English)
- [ ] Pitch Ojo Publico after arXiv live
- [ ] Pitch El Comercio data desk
- [ ] Submit to arXiv (cs.CY primary, cs.LG secondary)
- [ ] Submit to FAccT 2027 or AIES 2026

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIG-01 | Phase 12 | Pending |
| FIG-02 | Phase 12 | Pending |
| FIG-03 | Phase 12 | Pending |
| FIG-04 | Phase 12 | Pending |
| FIG-05 | Phase 12 | Pending |
| TEX-01 | Phase 13 | Pending |
| TEX-02 | Phase 13 | Pending |
| TEX-03 | Phase 13 | Pending |
| TEX-04 | Phase 13 | Pending |
| TEX-05 | Phase 13 | Pending |
| TEX-06 | Phase 13 | Pending |
| TEX-07 | Phase 13 | Pending |
| TEX-08 | Phase 13 | Pending |
| REPO-01 | Phase 14 | Pending |
| REPO-02 | Phase 14 | Pending |
| REPO-03 | Phase 14 | Pending |
| REPO-04 | Phase 14 | Pending |
| REPO-05 | Phase 14 | Pending |

**Coverage:**
- v2 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0

---
*Requirements defined: 2026-02-17*
