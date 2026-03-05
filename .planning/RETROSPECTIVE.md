# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

---

## Milestone: v4.1 — readability-polish

**Shipped:** 2026-03-05
**Phases:** 3 (30–32) | **Plans:** 6 | **Requirements:** 10/10

### What Was Built

- Threshold-invariance verification for FNR rank order across 0.05–0.20 range (new `src/fairness/threshold_sweep.py`)
- JEDM-compliant AI declaration with section-specific Claude Code usage
- Abstract restructured to lead with generalizable proxy audit claim; contribution bullets consolidated 4→3
- Three cross-domain proxy audit citations (Sandvig 2014, Adler 2018, Obermeyer 2019)
- EWS generalization paragraph in Discussion — surveillance-invisibility axis as cross-domain finding
- Section 6.2 folded into 6.1; ENAHO vs. SIAGIE appendix table removed (appendix: 13→12 tables)

### What Worked

- **All work was paper-only** (except threshold sweep script) — very low blast radius, fast execution
- **Phase sequencing was clean**: COMP-01 result informed FRAME language; FRAME changes informed STRC ordering
- **STRC-04 undercount**: REQUIREMENTS.md checkbox was missed during plan execution but the work was done — caught at milestone completion. Consider a checklist step in plan execution to mark requirements as each plan closes.

### What Was Inefficient

- REQUIREMENTS.md STRC-04 checkbox not updated during Phase 32 execution — required correction at milestone time. Small issue but avoidable.
- `gsd-tools milestone complete` didn't extract accomplishments (SUMMARY.md files use `one_liner` in frontmatter bolt-on pattern, not yet in this project's files) — had to manually write MILESTONES.md entry.

### Patterns Established

- **Threshold-invariance as a methodological claim**: FNR rank order must hold across a range of operational thresholds, not just the optimal threshold. Establish this as a standard verification step in any future fairness audit milestone.
- **Proxy audit framing pattern**: Abstract → Introduction → Discussion flow for generalizable methodology claims: (1) lead abstract with replicable framework, (2) consolidate contributions, (3) connect findings to cross-domain implications.

### Key Lessons

1. **Mark requirements complete within the plan execution, not retroactively** — STRC-04 slip shows REQUIREMENTS.md checkboxes need to be updated as part of each plan's commit, not deferred.
2. **Paper-only milestones benefit from tight phase ordering** — COMP → FRAME → STRC worked because empirical results (threshold sweep) informed framing language, which informed structural decisions.
3. **v2 mention removal**: Removing model improvement details from the abstract (STRC-03) was straightforward and the paper reads better focused on primary findings.

### Cost Observations

- Model: claude-sonnet-4-6
- All work: paper editing (LaTeX) + one analysis script
- Sessions: ~4 sessions across 4 days
- Notable: 6 plans in 4 days at very low computational cost — paper editing is efficient with GSD

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 | 12 | 15 | Baseline: full pipeline from scratch |
| v2.0 | 3 | 3 | Pivot to publication-quality output |
| v3.0-v3.2 | 9 | 13 | Paper depth + statistical rigor + JEDM target |
| v4.0 | 4 | 9 | Model experiments confirming robustness |
| v4.1 | 3 | 6 | Pre-submission polish: framing + compliance |

### Cumulative Quality

| Milestone | Paper Pages | Test Suite | Key Addition |
|-----------|-------------|------------|--------------|
| v1.0 | — | 95 gate tests | Full analysis pipeline |
| v3.1 | 22 | 95+ | Bootstrap CIs, 5 models |
| v4.0 | 23 | 71 (47 gate + 24 fairness) | 31-feature models |
| v4.1 | 23 | + threshold sweep | Pre-submission polish |

### Top Lessons (Verified Across Milestones)

1. **DTA format for ENAHO**: INEI provides Stata DTA files, not CSVs — always support both formats with uppercase column names.
2. **LightGBM early stopping monitor**: Must use `first_metric_only=True` when using `average_precision` with `scale_pos_weight`.
3. **REQUIREMENTS.md checkboxes**: Mark complete during plan execution, not deferred to milestone close.
4. **Fairness audit core finding is structural**: Castellano FNR disparity persists across all 5 models, all feature sets, all operational thresholds — it is robust.
