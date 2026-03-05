---
phase: 32-structural-tightening
plan: "02"
subsystem: paper
tags: [latex, appendix-cleanup, table-removal]
dependency_graph:
  requires: [32-01-SUMMARY.md]
  provides: [appendix-without-enaho-siagie-table]
  affects: [paper/main.tex, paper/main.pdf]
tech_stack:
  added: []
  patterns: [surgical-latex-edit]
key_files:
  created: []
  modified: [paper/main.tex]
decisions:
  - "Deleted appendix table block for tab:enaho_siagie; physical file table_09_enaho_siagie.tex preserved on disk"
  - "Removed two parenthetical citations from Section 3 and Section 8; factual sentences retained"
metrics:
  duration: "3 minutes"
  completed: "2026-03-05"
  tasks_completed: 1
  files_modified: 1
---

# Phase 32 Plan 02: Remove ENAHO vs. SIAGIE Appendix Table Summary

Removed the ENAHO vs. SIAGIE feature availability table (tab:enaho_siagie) from the appendix and all in-text references, reducing appendix to 5 tables and eliminating the inferential disclaimer that invited reviewer scrutiny.

## What Was Done

**Task 1: Remove tab:enaho_siagie table block and in-text references**

Three edits to `paper/main.tex`:

- **Edit A (Section 3, line 96):** Removed `(Table~\ref{tab:enaho_siagie}, Appendix~\ref{sec:appendix_tables})` parenthetical from the ENAHO-vs-SIAGIE sentence; factual claim about missing attendance/grade/trajectory data retained
- **Edit B (Section 8, line 272):** Removed `(Table~\ref{tab:enaho_siagie}, Appendix~\ref{sec:appendix_tables})` parenthetical from the proxy-model limitations opening sentence
- **Edit C (Appendix, lines 364-368):** Deleted the entire `\begin{table}[H]...\end{table}` block containing the ENAHO vs. SIAGIE comparison; `paper/tables/table_09_enaho_siagie.tex` left on disk unchanged

Resulting appendix: 12 `\begin{table}` blocks (was 13, reduced by 1). Paper compiles with `latexmk` exit 0, no undefined reference warnings.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `grep -c "tab:enaho_siagie" paper/main.tex` — returns 0 (confirmed)
- `ls paper/tables/table_09_enaho_siagie.tex` — file exists (confirmed)
- `grep -c '\begin{table}' paper/main.tex` — returns 12 (was 13, confirmed)
- `latexmk` — exits 0, no undefined references for tab:enaho_siagie

## Self-Check: PASSED

- `paper/main.tex` and `paper/main.pdf` committed at `9f1991c`
- Task commit exists: `git log --oneline` confirms `9f1991c feat(32-02): remove tab:enaho_siagie table and references`
