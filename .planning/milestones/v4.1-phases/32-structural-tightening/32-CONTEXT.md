# Phase 32: Structural Tightening — Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove structural redundancy and dead weight from paper/main.tex. Three changes: fold Section 6.2 into 6.1, eliminate cross-architecture duplication between Sections 5.2 and 6.1, and trim unnecessary tables.

</domain>

<decisions>
## Implementation Decisions

### Section 6.2 (STRC-01)
- **Action:** Fold Section 6.2 ("Other Demographic Dimensions") into Section 6.1 — remove the `\subsection` heading and absorb the paragraph as a transition/closing at the end of 6.1.
- **Age finding:** Drop the age sentence ("Older students ages 15-17 are flagged more accurately...") — SHAP section (6.4) already covers age importance; redundant here.
- **Result:** Section 6 will have 3 subsections (6.1, 6.3→6.2, 6.4→6.3) instead of 4.
- **Renumber:** Remaining subsections (Intersectional Analysis, SHAP Interpretability) must be renumbered and any cross-references updated.

### Cross-Architecture Redundancy (STRC-02)
- **Action:** Keep the full cross-architecture paragraph in Section 5.2 (Predictive Validity). Remove the duplicate sentence from Section 6.1 ("This FNR rank order holds across all five model families.").
- **Rationale:** Robustness claim belongs in Results (Section 5), not inside the Fairness Analysis narrative.

### Tables (STRC-04)
- **ENAHO vs SIAGIE table (Appendix Table 9 / tab:enaho_siagie):** Remove entirely — it's inferential/hypothetical, awkward caption disclaimer, invites more scrutiny than insight.
- **All 7 in-body tables:** Keep as-is. Each is directly referenced in the narrative and earns its place.
- **Remaining 5 appendix tables:** Keep. The threshold sweep (Table 13) is new from Phase 30 and needed.
- **Result:** 12 total tables (down from 13), 7 in-body + 5 appendix.

### STRC-03 (Abstract v2 mention)
- **Already complete in Phase 31** — v2 sentence removed from abstract entirely. No action needed.

### Claude's Discretion
- Exact wording of the folded 6.2 content at the end of 6.1
- Any bridge sentence needed after removing the cross-arch sentence from 6.1
- Handling of any cross-references to removed/renumbered subsections

</decisions>

<specifics>
## Specific Ideas

- When folding 6.2 into 6.1, the content should read as a natural closing paragraph: "Language dominates the fairness picture; other dimensions show [X, Y, Z]." Not a new header — just a paragraph transition.
- Removing ENAHO vs SIAGIE table may require updating any in-text reference to it — check the Methods section or Related Work for citations to `tab:enaho_siagie`.

</specifics>

<code_context>
## Existing Paper Insights

### Section 6.2 current text (verbatim, ~80 words)
"Language dominates: region shows FNR variation driven by spatial profiles, poverty quintiles show a monotonic flagging gradient that partially tracks base rates, and the sex gap is minimal (FNR difference of 0.026). Nationality ($n=27$ non-Peruvian) is unusable for inference. Older students (ages 15--17) are flagged more accurately than younger students, reflecting both higher base rates and the model's reliance on age as a predictive feature."

**Plan:** Keep everything except the age sentence. Absorb remainder at end of 6.1.

### Cross-arch sentence in 6.1 (to remove)
"This FNR rank order holds across all five model families."

**Plan:** Remove this sentence. Full cross-arch paragraph stays in 5.2.

### Table labels
- Remove: `tab:enaho_siagie` (Appendix Table 9)
- Keep all others

### Key files
- `paper/main.tex` — all edits here
- `paper/tables/table_09_enaho_siagie.tex` — referenced file (no need to delete, just remove `\input` and `\begin{table}`)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---
*Phase: 32-structural-tightening*
*Context gathered: 2026-03-05*
