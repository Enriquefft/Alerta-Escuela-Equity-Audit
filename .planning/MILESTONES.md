# Milestones

## v4.1 readability-polish (Shipped: 2026-03-05) — Public release: v1.0

**Phases completed:** 3 phases (30–32), 6 plans
**Git range:** `feat(30-01)` → `feat(32-02)`
**Timeline:** 4 days (2026-03-01 → 2026-03-05)
**Files modified:** 14 | Lines: +2,089 / −142

**Key accomplishments:**
- FNR rank order (castellano > quechua > other_indigenous) verified threshold-invariant across 0.05–0.20; stated in Methods Section 4.4 (COMP-01)
- AI declaration rewritten to name specific paper sections per JEDM format (COMP-02)
- Three proxy audit bib entries (Sandvig 2014, Adler 2018, Obermeyer 2019) added; Related Work footnote grounds methodology in cross-domain auditing tradition (FRAME-04)
- Abstract restructured to lead with generalizable household-survey proxy audit claim; contribution bullets consolidated from 4 to 3 with surveillance-invisibility axis elevated (FRAME-01, FRAME-02)
- EWS generalization paragraph added in Discussion — surveillance-invisibility axis framed as emergent property of any EWS using geographic/structural aggregates (FRAME-03)
- Section 6.2 folded into 6.1; ENAHO vs. SIAGIE appendix table removed — appendix reduced to 5 tables (STRC-01-04)

---

## v4.0 model-experiments (Shipped: 2026-03-01)

**Phases completed:** 4 phases, 9 plans, 0 tasks

**Key accomplishments:**
- 6 new features engineered (overage-for-grade + 4 interaction features)
- Panel linkage assessed as infeasible (18.9% effective rate — negative result documented)
- All 5 models retrained with 31-feature matrix (+9-14% PR-AUC improvement)
- Castellano FNR disparity confirmed as PERSISTENT and algorithm-independent across all 5 models
- Paper updated to 23 pages with v2 robustness analysis, compiles cleanly for JEDM submission

**Stats:** 143 commits, 197 files changed, 23 days (Feb 7 → Mar 1, 2026)
**Tech debt:** shap_values.json stale (25 features vs 31) — SHAP re-run deferred

---

