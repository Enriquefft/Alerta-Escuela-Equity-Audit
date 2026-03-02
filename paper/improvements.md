## "Mere Application" Risk

JEDM says submissions must go beyond "mere application" and include "generalizable methodologies or comparative analyses." Your paper is **not** mere application — but you need to make sure the framing doesn't read that way to a skimming reviewer.

**What you have going for you:**
- The proxy audit framework is explicitly positioned as transferable ("applicable wherever direct system access is unavailable")
- You compare five model families, which counts as comparative analysis
- The impossibility theorem connection (Chouldechova) gives it theoretical grounding
- The intersectional methodology is a methodological contribution, not just an applied finding

**Where the risk is:**
- The title says "in Peru" — a reviewer skimming might categorize it as a country case study
- The abstract leads with Peru-specific details before establishing the methodological contribution
- The introduction's contribution bullets are good but the first one ("proxy equity audit framework") could be more explicit about *why* this generalizes. Right now it says "demonstrating independent algorithmic accountability using only public survey data" — consider adding something about the approach being replicable in any country with nationally representative household surveys (which is most countries)

**Recommendation:** You don't need to restructure, but strengthen two sentences. In the abstract, make the generalizability claim earlier and more explicit. In the conclusion, the sentence about "an approach applicable wherever direct system access is unavailable" is doing the heavy lifting — consider echoing this framing in the introduction's contribution list too.

---

## 12 Reviewer Questions

### 1. How relevant is this to JEDM's scope?

**Assessment: Strong.** JEDM's scope includes "innovative processes or methodologies for analyzing educational data" and "advancing understanding of learner cognition" (less relevant) and "broader applicability of educational software." Your paper fits squarely in methodology for analyzing educational data. The fairness angle is increasingly prominent in EDM — Gardner et al. (2024), Baker and Hawn (2022), and Pan and Zhang (2024) are all published in or adjacent to EDM venues, so there's clear precedent.

**No action needed.**

### 2. How novel is the research? Are the authors aware of related work?

**Assessment: Strong with one gap.** Your Related Work section is thorough — you cover EWS history, fairness in education, intersectionality theory, and Peru-specific education literature. Novelty claims are well-scoped: proxy audit with public survey data, survey-weighted fairness metrics, developing-country multilingual context.

**Potential concern:** A reviewer might ask whether proxy auditing itself is novel or whether prior work has done something similar outside education (e.g., ProPublica's COMPAS audit, or algorithmic auditing in hiring). You cite Buolamwini and Gebru but that's a direct audit, not a proxy one. Consider whether there's proxy audit literature in criminal justice or lending that you should cite and differentiate from. If there is and you're aware of it but chose not to include it, that's a judgment call. If you haven't checked, check.

### 3. What is the scientific contribution? Is it clearly explained?

**Assessment: Good, could be sharper.** You have four contribution bullets in the introduction. The issue is that bullets 1 and 4 overlap (both about the audit framework being open/replicable), and bullet 3 (spatial proxy features) is the most analytically interesting but reads as secondary.

**Recommendation:** Consider consolidating bullets 1 and 4 into one, and giving the spatial proxy mechanism finding its own more prominent framing. The "surveillance–invisibility axis" concept is your most memorable analytical contribution — make sure it's in the contribution list, not just in the body.

### 4. Is the work technically sound? Enough methodological details? Claims substantiated?

**Assessment: Strong overall, with two soft spots.**

**Soft spot 1: Threshold selection.** You use max weighted F1 on validation to set the classification threshold, but this choice isn't well motivated. Why F1 and not a threshold that optimizes for the specific fairness-performance tradeoff you care about? A reviewer could argue that your FNR findings are partially an artifact of the threshold choice. You mention this implicitly in the Discussion (group-specific threshold adjustment could reduce castellano invisibility) but don't address it methodologically. Adding a sentence in Section 4.4 acknowledging that threshold choice affects absolute FNR values while the rank order is threshold-invariant (if that's true — verify) would pre-empt this.

**Soft spot 2: The v2 model (31 features) is only reported on validation for PR-AUC but on test for FNR (Table 10).** This is slightly inconsistent. A reviewer might ask: why not report test PR-AUC for v2 as well? You do mention v2 test PR-AUC = 0.257 in Section 5.1, but it's not in Table 4. Consider adding it to Table 4 or making the reporting more symmetric.

### 5. Data/code available for inspection and re-use?

**Assessment: Good.** You have the anonymous repo link. Before submission, archive on Zenodo for a DOI. JEDM specifically asks reviewers to check this.

**Action needed:** Zenodo archival.

### 6. Generative AI use properly documented?

**Assessment: Nearly compliant.** Your declaration section exists and follows the right structure. JEDM's required format asks for "precisely all sections or subsections." Your current version says "data pipeline implementation, figure generation, and editorial refinement" — these are task categories, not section names. A reviewer or editor could flag this.

**Recommendation:** Rewrite to something like: "...in Sections 3 (Data) and 4 (Methods) for data pipeline implementation, Section 5 (Results) and Appendix A for figure generation, and across all sections for editorial refinement..." This maps tasks to actual paper sections.

### 7. Limitations described satisfactorily?

**Assessment: Very strong.** This is one of the paper's strengths. Six clearly articulated limitations including the proxy audit caveat, self-report language variable, COVID wave, small intersectional samples with a power analysis, panel linkage ceiling, and survey-weighted gradient boosting statistical guarantees. The power analysis in particular (8 ENAHO years to confirm FNR > 0.50) is the kind of quantified limitation reviewers respect.

**No action needed.**

### 8. How significant is the research? Likely impact on the community?

**Assessment: Moderate-to-strong.** The significance depends on how much the reviewer cares about fairness in education and developing-country contexts. For reviewers aligned with the Baker/Kizilcec/Gardner fairness thread in EDM, this is clearly significant. For a reviewer more focused on learning analytics or adaptive systems, it may feel peripheral.

**Risk mitigation:** The paper could benefit from one paragraph (perhaps in the Discussion) connecting your findings to implications for the broader EDM community — not just Peru's EWS but any system that relies on spatial/structural features. Something like: "Any dropout prediction system that achieves discrimination primarily through geographic features will produce analogous fairness failures. This includes systems in [examples] that use school-level or district-level aggregates." This transforms a Peru finding into a general warning.

### 9. Title clearly reflects contents?

**Assessment: Good.** "Who Gets Missed? A Proxy Equity Audit of Survey-Derived Dropout Risk in Peru" is clear, informative, and has a hook. The question format works for the surveillance–invisibility framing.

**Minor thought:** "Survey-Derived" is accurate but slightly technical for a title. Not a problem — just noting that a reviewer might find it dense on first read.

### 10. Presentation, organization, and length satisfactory?

**Assessment: Moderate concern.** The paper is 23 pages with appendices. Check recent JEDM articles for typical length — my sense is most JEDM papers are 20–30 pages, so you're within range. However, the body text could be tighter in places.

**Specific areas to tighten:**
- Section 6.2 (Other Demographic Dimensions) is quite short and could be folded into 6.1 or summarized in one paragraph
- The transition from Section 5 to Section 6 has some redundancy — the cross-architecture consistency is mentioned in both 5.1/5.2 and again in 6.1
- Table 4 has a complex v1/v2 structure that could be simplified (report v2 in a separate table or appendix since it's a robustness check)

### 11. Illustrations and tables necessary and acceptable?

**Assessment: Good, with one note.** Your figures are clear and the heatmaps (Figures 4, 6) are effective. Figure 2 (FNR/FPR bar chart) communicates the surveillance–invisibility axis well.

**One concern:** You have 12 tables (8 in appendix). That's a lot. A reviewer might ask whether all are necessary. Tables 8 (all 25 LR coefficients) and 9 (ENAHO vs SIAGIE comparison) are useful but could potentially be trimmed. Table 9 in particular is mostly inferential about SIAGIE — a reviewer might question its necessity since you explicitly disclaim knowledge of SIAGIE's actual features.

### 12. Keywords and abstract informative?

**Assessment: Good.** Keywords cover the right terms. The abstract is dense but informative — it includes the key numbers (FNR 63.3% vs 21.6%, PR-AUC 0.236, N = 150,135) and frames the contribution.

**One suggestion:** The abstract currently includes the v2 (31-feature) robustness finding, which adds complexity. Consider whether the abstract needs it or whether the v1 findings alone tell the story cleanly enough. The v2 result can live in the body without cluttering the abstract.

---

## Summary of Priority Actions

**Must-do (would likely affect review outcome):**
1. Strengthen generalizability framing in abstract and introduction — don't let a skimmer think this is just a case study
2. Fix the AI declaration to name specific paper sections, not just task categories
3. Verify the FNR rank order is threshold-invariant (if it is, say so explicitly in Methods)
4. Archive code on Zenodo

**Should-do (would improve the paper):**
5. Add a paragraph connecting findings to any EWS using spatial features, not just Peru's
6. Check if proxy audit literature exists outside education (criminal justice, lending) and cite or differentiate
7. Tighten Section 6.2 and reduce cross-architecture redundancy between Sections 5 and 6
8. Simplify the abstract by removing or shortening the v2 mention

**Nice-to-do (polish):**
9. Consolidate contribution bullets (merge 1 and 4)
10. Consider whether all 12 tables are necessary
