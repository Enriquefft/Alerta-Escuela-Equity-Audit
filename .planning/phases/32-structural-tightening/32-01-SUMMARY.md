---
phase: 32-structural-tightening
plan: "01"
subsystem: paper
tags: [latex, section-restructure, fairness-analysis]
dependency_graph:
  requires: [31-03-SUMMARY.md]
  provides: [revised-section-6-structure]
  affects: [paper/main.tex, paper/main.pdf]
tech_stack:
  added: []
  patterns: [surgical-latex-edit, subsection-absorption]
key_files:
  created: []
  modified: [paper/main.tex]
decisions:
  - "Absorbed former 6.2 paragraph into end of 6.1 with bridge phrase 'Language dominates the fairness picture'"
  - "Cross-arch sentence ('This FNR rank order holds across all five model families.') was absent from Section 6.1 lines 201-220 — no removal needed there (present only in Abstract)"
  - "Age sentence dropped — redundant with model feature analysis in SHAP section"
metrics:
  duration: "4 minutes"
  completed: "2026-03-05"
  tasks_completed: 1
  files_modified: 1
---

# Phase 32 Plan 01: Fold Section 6.2 into 6.1 Summary

Removed standalone Section 6.2 (Other Demographic Dimensions) by absorbing its content (minus age sentence) as a closing paragraph in Section 6.1, leaving Section 6 with 3 clean subsections.

## What Was Done

**Task 1: Fold Section 6.2 into 6.1 and remove cross-arch duplicate**

Three edits to `paper/main.tex`:

- **Edit A:** Removed `\subsection{Other Demographic Dimensions}` heading (line 217)
- **Edit B:** Replaced the standalone 6.2 paragraph with an absorbed version at the end of 6.1, using bridge phrase "Language dominates the fairness picture:" and dropping the age sentence ("Older students (ages 15--17)...")
- **Edit C:** Verified cross-arch sentence absent from Section 6.1 lines 201-220 — no action needed

Resulting Section 6 structure:
- 6.1 Language Dimension: The Surveillance--Invisibility Axis (with absorbed closing paragraph)
- 6.2 Intersectional Analysis (was 6.3)
- 6.3 SHAP Interpretability (was 6.4)

LaTeX auto-renumbers subsections; no manual label changes required. No `\label{}` existed on the removed subsection. Paper compiles cleanly with latexmk.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `grep -n "Other Demographic" paper/main.tex` — no matches (confirmed)
- `grep -n "ages 15--17" paper/main.tex` — no matches (confirmed)
- `grep -n "subsection{Intersectional" paper/main.tex` — line 219 (confirmed present)
- `grep -n "subsection{SHAP" paper/main.tex` — line 231 (confirmed present)
- `latexmk` — exits 0, no undefined references

## Self-Check: PASSED

- `paper/main.tex` modified and committed at `4ba831c`
- Task commit exists: `git log --oneline` confirms `4ba831c feat(32-01): fold Section 6.2 into 6.1, remove cross-arch duplicate`
