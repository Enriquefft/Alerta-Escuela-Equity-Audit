---
phase: 32-structural-tightening
verified: 2026-03-05T00:00:00Z
status: gaps_found
score: 4/5 must-haves verified
gaps:
  - truth: "Cross-architecture model comparison content appears once (not duplicated between Sections 5 and 6)"
    status: failed
    reason: "The sentence 'This FNR rank order holds across all five model families.' appears in the Abstract (line 42), not removed from Section 6.1. The plan verified it was absent from lines 201-220 of Section 6 — it is — but the sentence was never removed from the Abstract. STRC-02 concerns duplication between Sections 5 and 6; that is resolved. However the Abstract still carries this sentence as a standalone claim separate from its single appearance in Section 5.2, which may constitute residual duplication. Low severity but flagged for review."
    artifacts:
      - path: "paper/main.tex"
        issue: "'This FNR rank order holds across all five model families.' remains in Abstract (line 42). Plan 01 only verified it was absent from Section 6 lines 201-220, not from the Abstract."
    missing:
      - "Confirm whether STRC-02 scope covers the Abstract or only Sections 5/6 body text. If Abstract is in scope, remove or fold the sentence into the existing abstract claim."
---

# Phase 32: Structural Tightening Verification Report

**Phase Goal:** Paper is tighter and more focused, with redundancy removed and every table earning its place.
**Verified:** 2026-03-05
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Section 6.2 (Other Demographic Dimensions) folded into 6.1 — standalone subsection heading removed | VERIFIED | `grep -n "Other Demographic" paper/main.tex` returns zero matches. Section 6 subsections at lines 201, 219, 231 are: Language Dimension, Intersectional Analysis, SHAP Interpretability — exactly 3. |
| 2 | Cross-architecture model comparison content appears once (not duplicated between Sections 5 and 6) | PARTIAL | Sentence is absent from Section 6 body (lines 201-231). However the sentence "This FNR rank order holds across all five model families." appears in the Abstract at line 42. Scope of STRC-02 is ambiguous — plan only checked lines 201-220. Section 5/6 duplication is resolved. |
| 3 | Abstract mentions v2/31-feature model in at most one clause (completed in Phase 31) | VERIFIED | Confirmed by plan note "STRC-03: Abstract v2 mention was already removed in Phase 31. No tasks needed." Not re-verified here per plan. |
| 4 | tab:enaho_siagie removed — no \begin{table}...\end{table} block, no in-text references remain | VERIFIED | `grep "tab:enaho_siagie" paper/main.tex` returns zero matches. Table count in appendix is 5 (lines 358, 364, 370, 376, 382). Physical file `paper/tables/table_09_enaho_siagie.tex` still exists on disk. |
| 5 | Paper compiles without errors | VERIFIED | `latexmk` reports "All targets (main.pdf) are up-to-date" with exit 0, no Error or Undefined reference warnings. |

**Score:** 4/5 truths verified (1 partial)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `paper/main.tex` | Revised Section 6 with 3 subsections | VERIFIED | Subsections at lines 201, 219, 231 confirm 3-subsection structure |
| `paper/main.tex` | Appendix tables section without tab:enaho_siagie block | VERIFIED | Zero grep matches for `tab:enaho_siagie`; 5 appendix table blocks at lines 358-382 |
| `paper/tables/table_09_enaho_siagie.tex` | File preserved on disk (not deleted) | VERIFIED | `ls` confirms file exists |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Section 6.1 closing paragraph | Absorbed former 6.2 text | "Language dominates the fairness picture..." | VERIFIED | Line 217: "Language dominates the fairness picture: region shows FNR variation driven by spatial profiles, poverty quintiles show a monotonic flagging gradient that partially tracks base rates, and the sex gap is minimal (FNR difference of 0.026). Nationality ($n=27$ non-Peruvian) is unusable for inference." — Age sentence ("Older students ages 15--17") is absent. |
| Section labels in text | Correct subsection labels after renumbering | `\label{}` and `\ref{}` matching | VERIFIED | LaTeX compiles cleanly with no undefined reference warnings. Auto-numbering handled renaming from 6.3/6.4 to 6.2/6.3. |
| Section 8 (Limitations) ~line 274 | tab:enaho_siagie reference removed | Parenthetical deleted | VERIFIED | Zero matches for `tab:enaho_siagie` in main.tex. |
| Section 3 (Data) ~line 96 | tab:enaho_siagie reference removed | Parenthetical deleted | VERIFIED | Zero matches for `tab:enaho_siagie` in main.tex. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| STRC-01 | 32-01 | Fold Section 6.2 into 6.1, remove standalone heading | SATISFIED | "Other Demographic" absent; 3-subsection structure confirmed |
| STRC-02 | 32-01 | Remove duplicate cross-architecture sentence from Section 6 | SATISFIED (body) | Absent from Section 6 lines 201-231; remains in Abstract — scope ambiguous |
| STRC-03 | 32-01 | Abstract v2/31-feature mention reduced to one clause | SATISFIED | Completed in Phase 31; noted in plan as no-op |
| STRC-04 | 32-02 | Remove tab:enaho_siagie table and all references | SATISFIED | Zero grep matches; 5 appendix tables remain; file preserved; clean compile |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `paper/main.tex` line 42 | "This FNR rank order holds across all five model families." in Abstract | Info | Sentence also appears in Section 5 body (Section 5.2 Predictive Validity area). If STRC-02 scope includes Abstract, this is a residual duplication. If Abstract is intentionally a summary, no issue. |

### Human Verification Required

None — all checks are programmatic grep and compile verification.

### Gaps Summary

One partial gap: The cross-architecture sentence "This FNR rank order holds across all five model families." was correctly removed from Section 6 (the explicit STRC-02 target). However it remains in the Abstract at line 42, where it appears as a standalone sentence. The plan's verification tasks only checked lines 201-220 of Section 6, not the Abstract.

Whether this is a true gap depends on STRC-02's intent. If STRC-02 only required removing the duplicate from Section 6 (which duplicated Section 5.2), then STRC-02 is fully satisfied and the Abstract occurrence is intentional. If STRC-02 required eliminating all redundant occurrences across the paper, the Abstract still needs attention.

Given the plan's explicit scope ("remove from Section 6.1"), the conservative interpretation is that STRC-02 is satisfied and the Abstract occurrence is intentional. The gap is flagged for human confirmation only.

All other success criteria are fully verified: Section 6 has 3 subsections, the absorbed paragraph is present, the age sentence is gone, tab:enaho_siagie is fully removed with no stray references, the source file is preserved, and the paper compiles cleanly.

---

_Verified: 2026-03-05_
_Verifier: Claude (gsd-verifier)_
