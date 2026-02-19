# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** The fairness audit is the product. Models exist to be audited, not to achieve SOTA.
**Current focus:** v3.2-jedm-revision — COMPLETE

## Current Position

Phase: 25 of 25 (Page Count Reduction) -- COMPLETE
Plan: 1 of 1 in current phase
Status: All phases complete. Paper at 18 pages (down from 22).
Last activity: 2026-02-18 -- Phase 25 executed

Progress: [################] 100% (4 of 4 phases complete)

## Performance Metrics

**Velocity (cumulative):**
- v1.0: 15 plans, ~132 min
- v2.0: 3 plans, ~18 min
- v3.0: 2 plans (Phase 15)
- v3.1: 7 plans (Phase 16: 1, Phase 17: 2, Phase 18: 3, Phase 19: 1)
- v3.2: 4 plans (Phase 22: 1, Phase 23: 1, Phase 24: 1, Phase 25: 1)
- Total: 31 plans completed

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 22    | 01   | ~36 min  | 5     | 26    |
| 23    | 01   | ~8 min   | 13    | 3     |
| 24    | 01   | ~8 min   | 5     | 2     |
| 25    | 01   | ~10 min  | 9     | 1     |

## Accumulated Context

### Decisions

- [v3.2]: Target JEDM instead of FAccT 2027 (free APC, ~3-month review)
- [v3.2]: Pool val+test data for urban indigenous CI, flag non-test data prominently
- [v3.2]: Remove arXiv prep phase (JEDM has own portal)
- [v3.2]: Keep media materials phase deferred (Phase 20)
- [v3.2]: Feature ablation to experimentally test spatial proxy mechanism
- [v3.2]: Rename "algorithm independence" → "cross-architecture consistency"
- [v3.2]: English figure labels (JEDM is English-language)
- [v3.2]: Phase 25 added — paper is 22pp, needs cuts to reach 15-18pp target

### Pending Todos

None.

### Decisions (from discuss-phase 22)

- [v3.2]: Pool val+test replaces test-only FNR (not alongside)
- [v3.2]: Credibility threshold: CI lower bound > 0.50
- [v3.2]: Strictly other-indigenous urban cell (don't mix with Quechua/Aymara)
- [v3.2]: Simple weighted bootstrap (pooling breaks PSU design)
- [v3.2]: If pooled CI lower bound ≤ 0.50, castellano invisibility (FNR=0.633, n=22K+) becomes headline
- [v3.2]: Power analysis reframed as methodological limitation — survey data ceiling for intersectional auditing
- [v3.2]: Fix flagged_small_sample for other_indigenous_urban in JSON
- [v3.2]: Paper should argue this kind of study can't produce significant intersectional results due to data limitations → SIAGIE access argument

### Decisions (from Phase 22 execution)

- [v3.2]: Pooled CI credible=false (CI lower 0.392 < 0.50) -- castellano invisibility (FNR=0.633) is headline
- [v3.2]: Ablation interpretation "mixed" -- castellano consistently highest FNR across all variants
- [v3.2]: Per-variant thresholds needed for ablation (scale_pos_weight shifts prob distributions)
- [v3.2]: Need 46 dropouts (~8 ENAHO years) to confirm FNR > 0.50 for urban other-indigenous

### Decisions (from Phase 23+24 execution)

- [v3.2]: Phase 23 restructure + Phase 24 style polish applied together (committed b692b5b)
- [v3.2]: Paper compiles clean at 22 pages — content additions offset structural cuts
- [v3.2]: JEDM accepts 15-25pp; 22pp is within range but above self-imposed 18pp target

### Decisions (from Phase 25 execution)

- [v3.2]: Paper reduced from 22 to 18 pages
- [v3.2]: Tables 5, 9, 10, 11 moved to Supplementary Tables appendix
- [v3.2]: Figures 4 (dropout heatmap), 7 (SHAP beeswarm) moved to appendix
- [v3.2]: Figure 1 (PR curves) removed entirely — Table 4 has all numbers
- [v3.2]: Discussion subsections merged: EWS Operators condensed, Generalizability deleted, Normative condensed
- [v3.2]: Additional prose tightening in Data, Methods, Results, Fairness, Limitations, Introduction

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18
Stopped at: v3.2 milestone complete. Ready for JEDM submission.
Resume file: None
