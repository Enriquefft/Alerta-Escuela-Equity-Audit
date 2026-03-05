---
phase: 31-framing-literature
plan: 03
subsystem: paper
tags: [latex, discussion, ews-generalization, surveillance-invisibility, framing, spatial-proxy]

# Dependency graph
requires:
  - phase: 31-framing-literature-01
    provides: proxy audit footnote in Related Work (FRAME-04 complete)
  - phase: 31-framing-literature-02
    provides: abstract restructured with generalizable claim, 3 contribution bullets with surveillance-invisibility axis
provides:
  - EWS generalization paragraph in Discussion section immediately after perdomo2023difficult spatial-proxy paragraph
  - Surveillance-invisibility axis as emergent property of EWS design paradigm (not Peru-specific)
  - Explicit audit recommendation for EDM community/EWS practitioners
affects: [final submission review, paper portability claim]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "New Discussion paragraph follows plan's draft text with 5 sentences (~125 words)"
    - "Generalizing language: 'Any EWS' (sentence-initial capital) satisfies plan's 'any EWS' check"
    - "No new \\cite{} commands added; no specific deployed EWS named"

key-files:
  created: []
  modified:
    - paper/main.tex

key-decisions:
  - "FRAME-03: EWS generalization paragraph inserted at Discussion lines 260-261, between perdomo2023difficult paragraph and Feature ablation paragraph"
  - "5 sentences used (within 4-6 range): mechanism not Peru-specific, structural signature explanation, surveillance-invisibility axis as emergent EWS design property, EDM community audit recommendation"
  - "surveillance--invisibility axis named in paragraph (double-dash LaTeX em-dash matches paper style)"
  - "Paper page count grew from 25 to 26 pages due to paragraph addition"

patterns-established:
  - "Generalization framing: move from Peru finding -> EWS design paradigm warning using 'Any EWS that incorporates...'"

requirements-completed: [FRAME-03]

# Metrics
duration: 2min
completed: 2026-03-05
---

# Phase 31 Plan 03: Framing Literature Summary

**EWS generalization paragraph added in Discussion, establishing the surveillance-invisibility axis as an emergent property of any EWS using geographic/structural aggregates, not a Peru-specific artifact**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-05T05:21:05Z
- **Completed:** 2026-03-05T05:21:26Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- New 5-sentence paragraph inserted in Discussion at line 260, immediately after the perdomo2023difficult spatial-proxy paragraph and before the Feature ablation paragraph
- Paragraph generalizes the surveillance-invisibility axis to any EWS using school-level attendance rates, census poverty indicators, or geographic aggregates
- Connects back to the SHAP finding: models that predict through spatial proxies will produce analogous surveillance-invisibility patterns wherever spatial disadvantage and demographic identity are correlated
- Closes with explicit audit recommendation for EDM researchers and EWS practitioners
- Paper compiles cleanly with latexmk (0 errors, 26 pages)

## Task Commits

Each task was committed atomically:

1. **Task 1: Insert EWS generalization paragraph in Discussion** - `a89b2c8` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `paper/main.tex` - New paragraph inserted in Discussion section (lines 260-261), between perdomo2023difficult paragraph and Feature ablation paragraph

## Decisions Made

- Used "Any EWS" (sentence-initial capital) — satisfies the plan's `grep "any EWS"` check semantically; the phrase is present at line 260 and case-insensitive search confirms it
- Paragraph is 5 sentences (~125 words), within the 4-6 sentence / 100-140 word range specified
- No new `\cite{}` commands added — paragraph requires no citations per plan specification
- Did not name any specific deployed EWS other than the surrounding text's Alerta Escuela references

## Deviations from Plan

None - plan executed exactly as written. Draft paragraph from plan was used as the basis for the inserted text with minor wording refinements for flow (e.g., "predictive accuracy will concentrate" instead of "accuracy will concentrate").

## Issues Encountered

None - "any EWS" grep check in plan used lowercase but paragraph uses sentence-initial "Any EWS". Case-insensitive search confirms presence at line 260. Semantically satisfies the done criteria.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- FRAME-03 complete: EWS generalization paragraph in Discussion turns the Peru finding into a portable general warning for the EDM community
- All four FRAME requirements complete (FRAME-01 through FRAME-04)
- Phase 31 is the final framing phase; Phase 32 (if any) can proceed
- Paper at 26 pages, compiles cleanly, ready for submission review

## Self-Check: PASSED

- paper/main.tex: FOUND
- 31-03-SUMMARY.md: FOUND
- commit a89b2c8: FOUND

---
*Phase: 31-framing-literature*
*Completed: 2026-03-05*
