# Phase 31: Framing & Literature — Context

**Gathered:** 2026-03-04
**Status:** Ready for planning (with research step)

<domain>
## Phase Boundary

Strengthen the paper's framing so a skimming reviewer sees it as a generalizable methodological contribution, not a Peru case study. Four changes: abstract restructure, contribution bullets consolidation, EWS generalization paragraph in Discussion, proxy audit literature in Related Work (footnote).

</domain>

<decisions>
## Implementation Decisions

### FRAME-01: Abstract Restructuring
- **Opening:** Rewrite sentence 1 to lead with the generalizable claim using "strong" framing: household surveys as the enabling mechanism. Example direction: "Nationally representative household surveys—available in most countries—enable proxy audits of deployed EWS without requiring system access."
- **Sentence order:** Generalizable claim first → Peru as the case study second → ENAHO setup → findings → contributions.
- **v2 mention:** Remove entirely from the abstract. The body covers it. Abstract focuses on primary v1 findings only.

### FRAME-02: Contribution Bullets
- **Merge bullets 1 & 4:** Combine into a single bullet leading with the proxy audit method as replicable in any country with household surveys, mentioning open-source as secondary.
- **Result:** 3 bullets total (down from 4).
- **Surveillance-invisibility axis:** Fold concept into existing bullet 2 or 3 (not a new bullet). Enhance the wording of one of those bullets to name the axis.

### FRAME-03: EWS Generalization Paragraph
- **Placement:** Immediately after the existing SHAP/spatial-feature paragraph in Discussion (the one starting "SHAP analysis reveals that nightlight intensity...").
- **Tone:** Stay abstract — describe the pattern ("any EWS using school-level attendance rates, census poverty data, or geographic aggregates") without naming specific deployed systems we haven't audited.
- **Purpose:** Turn a Peru finding into a general warning for EDM community.

### FRAME-04: Proxy Audit Literature
- **Discovery method:** Spawn research agent during planning to find proxy audit papers in criminal justice/lending/hiring BEFORE writing. Use findings to decide cite vs. differentiate.
- **Citation location:** Footnote (not in-body Related Work, not intro). Acknowledge cross-domain precedent in a footnote, keep body text focused on EDM context.
- **If nothing found:** Claude drafts a differentiation sentence noting that proxy auditing has been applied in [domain] (e.g., ProPublica COMPAS analysis used public criminal records as a proxy) and distinguishing our contribution as the first in educational EWS.

### Claude's Discretion
- Exact abstract sentence wording (must be LaTeX-compilable, match existing citation style)
- Which existing bullet (2 or 3) absorbs the surveillance-invisibility axis framing
- Length of EWS paragraph (1 paragraph, ~4-6 sentences)
- Footnote wording for proxy audit literature

</decisions>

<specifics>
## Specific Ideas

- The abstract restructure is the highest-leverage change for "skimming reviewer" risk. Do this carefully — don't lose the Peru specificity that grounds the empirical contribution.
- The v2 removal from abstract makes the abstract tighter and cleaner. The 9-14% improvement numbers add complexity a first-read reviewer doesn't need.
- The EWS paragraph should connect back to the SHAP finding: models that predict through nightlights and district rates will produce analogous surveillance-invisibility patterns wherever spatial disadvantage and demographic identity are correlated.

</specifics>

<code_context>
## Existing Paper Insights

### Current Abstract
Opens with: "Dropout prediction systems are proliferating across Latin America, yet their fairness properties remain unaudited."
V2 mention at: "Expanding to 31 features with overage and interaction terms improves prediction by 9--14\% but the castellano invisibility pattern persists..."
Contributions at end of abstract (short form) and in intro bullets.

### Current Contribution Bullets (verbatim)
1. "A proxy equity audit framework demonstrating independent algorithmic accountability using only public survey data."
2. "A survey-weighted fairness audit spanning seven dimensions and three intersections, revealing disparities invisible to single-axis analysis."
3. "Evidence that dropout models encode structural inequities through spatial proxy features, not protected attributes."
4. "An open-source, replicable audit framework for educational EWS auditing."

### Current Discussion (spatial paragraph)
Starts: "SHAP analysis reveals that nightlight intensity, district-level dropout rates, and census literacy rates collectively encode the spatial concentration of disadvantage..."
Ends with perdomo2023difficult citation.
EWS generalization paragraph goes AFTER this block.

### Key files
- `paper/main.tex` — single source file
- `paper/references.bib` — bibliography (add new citations here)
- Phase 30 finding: threshold-invariance paragraph already added to Section 4.4 (Methods)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---
*Phase: 31-framing-literature*
*Context gathered: 2026-03-04*
