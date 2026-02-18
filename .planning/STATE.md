# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** The fairness audit is the product. Models exist to be audited, not to achieve SOTA.
**Current focus:** v3.2-jedm-revision — revise paper for JEDM submission

## Current Position

Phase: 23 of 24 (Paper Restructure) -- PENDING
Plan: 0 of ? in current phase
Status: Phase 22 complete. Ready for Phase 23 planning.
Last activity: 2026-02-18 -- Phase 22 executed (22-01-PLAN.md)

Progress: [#####░░░░░░░░░░░] 33% (1 of 3 phases complete)

## Performance Metrics

**Velocity (cumulative):**
- v1.0: 15 plans, ~132 min
- v2.0: 3 plans, ~18 min
- v3.0: 2 plans (Phase 15)
- v3.1: 7 plans (Phase 16: 1, Phase 17: 2, Phase 18: 3, Phase 19: 1)
- v3.2: 1 plan (Phase 22: 1)
- Total: 28 plans completed

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 22    | 01   | ~36 min  | 5     | 26    |

## Accumulated Context

### Decisions

- [v3.2]: Target JEDM instead of FAccT 2027 (free APC, ~3-month review)
- [v3.2]: Pool val+test data for urban indigenous CI, flag non-test data prominently
- [v3.2]: Remove arXiv prep phase (JEDM has own portal)
- [v3.2]: Keep media materials phase deferred (Phase 20)
- [v3.2]: Feature ablation to experimentally test spatial proxy mechanism
- [v3.2]: Rename "algorithm independence" → "cross-architecture consistency"
- [v3.2]: English figure labels (JEDM is English-language)

### Pending Todos

None yet.

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

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 22-01-PLAN.md. Phase 22 done. Ready for Phase 23 planning.
Resume file: None
