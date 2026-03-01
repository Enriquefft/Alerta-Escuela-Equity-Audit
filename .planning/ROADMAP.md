# Roadmap: Alerta Escuela Equity Audit

## Completed Milestones

- [x] **v1.0-analysis-complete** (2026-02-07 -> 2026-02-13) — Full equity audit pipeline: ENAHO data loading, 3 models, fairness analysis across 7 dimensions + 3 intersections, SHAP interpretability, 7 bilingual findings exported for M4 site. [Details](milestones/v1.0-ROADMAP.md)
- [x] **v2.0-publication** (2026-02-17) — Publication-quality figures, ACM LaTeX paper with 8 auto-generated tables, public repo cleanup (README, CITATION.cff, LICENSE, pipeline script). [Details](milestones/v2.0-ROADMAP.md)
- [x] **v3.0-submission** (2026-02-17) — Paper expanded to full academic depth (21 pages), 43 bibliography entries, Limitations + Ethical Considerations sections. [Details](milestones/v3.0-ROADMAP.md)
- [x] **v3.1-paper-strengthening** (2026-02-17) — Bootstrap CIs, hypothesis tests, RF + MLP models, lift/calibration analysis, paper rewrite with proxy framing, review fixes. [Details](milestones/v3.1-ROADMAP.md)
- [x] **v3.2-jedm-revision** (2026-02-18) — Paper revised to 18 pages for JEDM: pooled CI analysis, feature ablation, power analysis, page reduction. [Details](milestones/v3.2-ROADMAP.md)

## Current Milestone: v4.0-model-experiments

Experiment with model improvements to test whether better features strengthen or change fairness findings. Research-oriented: all outcomes (persist/narrow/disappear/no improvement) are valid and publishable.

### Phases

**Phase Numbering:**
- Continues from v3.2 (Phase 26+). Phases 22-25 completed in v3.2.

- [ ] **Phase 26: Feature Engineering** - Build overage-for-grade, assess panel linkage feasibility, engineer interaction features
- [ ] **Phase 27: Model Retraining** - Retrain all 5 model families with new features, re-calibrate, compare PR-AUC before/after
- [ ] **Phase 28: Fairness Re-Analysis** - Full fairness pipeline on updated models, assess FNR disparity change, cross-architecture consistency
- [ ] **Phase 29: Interpretation & Paper Update** - Interpret results per outcome scenario, update paper, document panel linkage outcome

### Phase Details

#### Phase 26: Feature Engineering
**Goal**: New features are computed and validated, ready for model consumption. Panel linkage feasibility is determined with a clear go/no-go decision.
**Depends on**: v3.2 complete (existing 25-feature dataset in data/processed/)
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04
**Success Criteria** (what must be TRUE):
  1. Overage-for-grade feature computed for all 150,135 rows with zero nulls, correctly reflecting age minus expected grade age
  2. Panel linkage rate measured and documented; if <20% linkable, trajectory features skipped and negative result recorded
  3. If panel linkage >=20%, trajectory features (income change, sibling dropout, work transitions) computed for linkable subset
  4. Interaction features (age x working, age x poverty, rural x parental_education, secondary_age x income) computed with zero nulls
  5. Updated feature matrix saved as parquet with all new columns, existing gate tests still pass
**Plans:** 2/3 plans executed
Plans:
- [ ] 26-01-PLAN.md — Overage-for-grade and interaction features
- [ ] 26-02-PLAN.md — Panel linkage assessment and trajectory features
- [ ] 26-03-PLAN.md — Feature integration, final parquet, and validation

#### Phase 27: Model Retraining
**Goal**: All 5 model families retrained with expanded feature set, primary LightGBM re-calibrated and re-exported as ONNX, with clear before/after PR-AUC comparison.
**Depends on**: Phase 26 (new feature matrix ready)
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04
**Success Criteria** (what must be TRUE):
  1. LightGBM retrained with Optuna re-optimization using new features, best hyperparameters logged
  2. All 5 model families (LR, LightGBM, XGBoost, RF, MLP) retrained with new feature set
  3. Primary LightGBM re-calibrated (Platt scaling) with updated coefficients, ONNX re-exported and validated (max diff < 1e-4)
  4. Before/after PR-AUC table produced for all 5 models showing whether new features improve prediction
  5. model_results.json updated with new metrics (weighted PR-AUC using FACTOR07)
**Plans**: TBD

#### Phase 28: Fairness Re-Analysis
**Goal**: Full fairness analysis re-run with updated models to determine whether FNR disparities persist, narrow, or disappear with better features.
**Depends on**: Phase 27 (all 5 models retrained)
**Requirements**: FAIR-01, FAIR-02, FAIR-03
**Success Criteria** (what must be TRUE):
  1. Full fairness pipeline re-run on updated primary LightGBM (7 dimensions + 3 intersections, all survey-weighted)
  2. FNR disparity change quantified: before/after comparison showing persist/narrow/disappear for castellano vs indigenous groups
  3. Cross-architecture consistency checked across all 5 updated models (FNR rank order comparison)
  4. Updated fairness_metrics.json exported with new results
**Plans**: TBD

#### Phase 29: Interpretation & Paper Update
**Goal**: Results interpreted per outcome scenario and paper updated with new findings, tables, and figures. Panel linkage documented regardless of outcome.
**Depends on**: Phase 28 (fairness re-analysis complete)
**Requirements**: PAPER-01, PAPER-02, PAPER-03
**Success Criteria** (what must be TRUE):
  1. Results interpreted through correct scenario lens (persist: features confirm structural finding; narrow: features partially explain gap; disappear: previous finding was feature artifact; ceiling: no PR-AUC improvement)
  2. Paper tables and figures updated with new model results and fairness metrics
  3. Panel linkage outcome documented in Limitations section (whether positive or negative result)
  4. Paper compiles without errors, narrative consistent with empirical results
**Plans**: TBD

### Deferred

- Phase 20: Press & Media Materials (from v3.0, after JEDM submission)

### Progress

**Execution Order:**
Phase 26 -> 27 -> 28 -> 29

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 26. Feature Engineering | 2/3 | In Progress|  |
| 27. Model Retraining | 0/? | Not started | - |
| 28. Fairness Re-Analysis | 0/? | Not started | - |
| 29. Interpretation & Paper Update | 0/? | Not started | - |
