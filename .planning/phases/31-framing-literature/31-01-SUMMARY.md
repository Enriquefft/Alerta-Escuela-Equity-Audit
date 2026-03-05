---
phase: 31-framing-literature
plan: 01
subsystem: paper
tags: [latex, bibtex, proxy-audit, related-work, citations, footnote]

# Dependency graph
requires:
  - phase: 30-compliance-foundations
    provides: Threshold sweep and paper compliance baseline; paper/main.tex at current state
provides:
  - Three new bib entries (sandvig2014auditing, adler2018auditing, obermeyer2019dissecting) in references.bib
  - Proxy audit footnote in Related Work anchoring methodology in 10-year cross-domain tradition
affects: [31-framing-literature, paper submission]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Footnote citation in Related Work uses \\citeN{} (same as inline style) — not \\citep{} or \\citet{}"

key-files:
  created: []
  modified:
    - paper/references.bib
    - paper/main.tex

key-decisions:
  - "Used \\citeN{} in footnote (not \\citep{}) — paper uses acmtrans bibliographystyle which defines \\citeN but not \\citep/\\citet"
  - "Footnote attaches to end of the sentence introducing our proxy audit contribution in Related Work paragraph 1"
  - "angwin2016machine (ProPublica journalism) not cited — chouldechova2017fair used for criminal justice exemplar (already in bib, peer-reviewed)"

patterns-established:
  - "Proxy audit framing: cite cross-domain exemplars in footnote, assert education-EWS gap as paper contribution"

requirements-completed: [FRAME-04]

# Metrics
duration: 2min
completed: 2026-03-05
---

# Phase 31 Plan 01: Framing Literature Summary

**Three proxy audit bib entries (Sandvig 2014, Adler 2018, Obermeyer 2019) added and a four-citation footnote in Related Work grounds our methodology in the established cross-domain auditing-without-access tradition**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-05T05:13:22Z
- **Completed:** 2026-03-05T05:14:55Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added three new bib entries under "Proxy Audit Literature (Cross-Domain)" section in references.bib
- Added footnote attached to the proxy audit contribution sentence in Related Work paragraph 1, citing all four cross-domain exemplars
- Paper compiles cleanly with latexmk (0 errors, all targets up-to-date)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add three bib entries to references.bib** - `af6aaf6` (feat)
2. **Task 2: Add proxy audit footnote in Related Work** - `c9446b0` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `paper/references.bib` - Added sandvig2014auditing, adler2018auditing, obermeyer2019dissecting bib entries
- `paper/main.tex` - Added \footnote{} on proxy audit sentence in Related Work (Section 2, paragraph 1)

## Decisions Made

- Used `\citeN{}` in the footnote body instead of `\citep{}` — the paper's acmtrans bibliographystyle defines `\citeN` but not `\citep` or `\citet`. The plan context mentioned `\citep{}` for footnotes but that command is undefined in this paper's preamble.
- Footnote ends the paragraph-1 sentence that introduces our proxy audit contribution (line 84), replacing the terminal period with `\footnote{...}.` as specified.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Replaced \\citet{} with \\citeN{} in footnote**
- **Found during:** Task 2 (Add proxy audit footnote in Related Work)
- **Issue:** Plan specified `\citet{}` commands in footnote text. The paper's acmtrans bibliographystyle defines `\citeN{}` for inline citations but not `\citet{}`, causing three "Undefined control sequence" errors and compile failure.
- **Fix:** Changed all four `\citet{}` calls in the footnote to `\citeN{}`, matching the existing citation style throughout the document.
- **Files modified:** paper/main.tex
- **Verification:** latexmk reports "All targets (main.pdf) are up-to-date" with 0 errors.
- **Committed in:** c9446b0 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - citation command undefined in this paper's style)
**Impact on plan:** Required fix for compilation. Semantically identical output — `\citeN{}` and `\citet{}` both produce author-name inline citations; only command name differs per bibliographystyle.

## Issues Encountered

- latexmk initially failed with "Undefined control sequence" for `\citet{}` — resolved by switching to `\citeN{}` matching the paper's established style.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- FRAME-04 complete: proxy audit literature cited and cross-domain tradition established
- Paper compiles cleanly
- Next plans in Phase 31 can proceed

---
*Phase: 31-framing-literature*
*Completed: 2026-03-05*
