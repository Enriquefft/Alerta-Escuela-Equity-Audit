# Phase 30: Compliance Foundations - Context

**Gathered:** 2026-03-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify that the FNR rank order (castellano > quechua > other_indigenous) is threshold-invariant, state this in Methods Section 4.4 with an appendix table as evidence. Rewrite the AI declaration to name specific paper sections per JEDM format.

</domain>

<decisions>
## Implementation Decisions

### Threshold Sweep
- Sweep from 0.05 to 0.95 in steps chosen by Claude (0.05 or 0.10 — whatever produces a clean appendix table)
- Use v1 (25-feature) model as primary; v2 (31-feature) as robustness check
- Show all language groups in appendix table but narrative focuses on top 3 (castellano, quechua, otros indigenas)
- Footnote about small-sample instability for Aimara (n=76) and others
- Present as: short paragraph in Methods Section 4.4 + appendix table (Table 13 or next available number)

### AI Declaration
- Rewrite to name specific paper sections, not just task categories
- Section mapping:
  - Data pipeline code: Section 3 (Data), Section 4 (Methods)
  - Analysis code (models, fairness): Section 5 (Results), Section 6 (Fairness Analysis)
  - Figure generation: Section 5, Section 6, Appendix A
  - Editorial refinement: All sections
- Use "Claude Code (Anthropic)" as the tool name (distinguishes CLI agent from chat)

### Claude's Discretion
- Exact threshold step size (0.05 vs 0.10)
- Table formatting and column layout
- Exact wording of the Methods paragraph (must state "threshold-invariant" clearly)

</decisions>

<specifics>
## Specific Ideas

- The threshold-invariance claim pre-empts a reviewer arguing FNR findings are artifacts of threshold choice
- If rank order does NOT hold at some thresholds, report honestly — note the range where it holds and where it breaks

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/fairness/metrics.py`: `_THRESHOLD_MAP` dict already loads optimal thresholds per model; `_load_and_prepare_data(model_name)` loads test data with y_true/y_prob/weights
- `src/fairness/metrics.py`: `run_fairness_pipeline(model_name)` computes all fairness metrics — can be adapted to loop over thresholds
- `data/exports/model_results.json`: contains optimal thresholds and all model metadata

### Established Patterns
- All fairness analysis uses survey weights (FACTOR07)
- v1 model files in `data/processed/`: `model_lgbm.joblib`, `predictions_lgbm.parquet`
- v2 model files: `model_lgbm_v2.joblib`, `predictions_lgbm_v2.parquet`

### Integration Points
- New script likely in `src/fairness/` (e.g., `threshold_sweep.py`)
- New appendix table in `paper/tables/` (auto-generated from JSON/CSV)
- Methods paragraph in `paper/main.tex` around line 147-150 (Section 4.4 Fairness Evaluation)
- AI declaration at `paper/main.tex` line 303-305

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 30-compliance-foundations*
*Context gathered: 2026-03-01*
