# Revision Guide — *Who Gets Missed?* (v2)

**Target venue:** JEDM (Journal of Educational Data Mining) — Q3, free APC, ~3-month review, no strict page limit
**Target length:** 15–18 pages (trim redundancy, don't gut the paper)

---

## Summary

| # | Issue | Priority | Effort |
|---|-------|----------|--------|
| 1 | Abstract opens with negation | CRITICAL | Low |
| 2 | Urban indigenous CI [0.211, 1.000] undermines headline | CRITICAL | Medium |
| 3 | Abstract overstates 75% FNR confidence | CRITICAL | Low |
| 4 | Paper too long (22pp → 15–18pp for JEDM) | CRITICAL | Medium |
| 5 | No feature ablation robustness check | IMPORTANT | Medium |
| 6 | No normative fairness criterion discussion | IMPORTANT | Medium |
| 7 | Discussion 7.1 repeats results — delete | IMPORTANT | Low |
| 8 | Related work too long and sequential | IMPORTANT | Medium |
| 9 | Figures 2, 3, 5 duplicate tables — move to appendix | IMPORTANT | Low |
| 10 | Conclusion too compressed | IMPORTANT | Low |
| 11 | Kearns et al. and other key refs dropped | MODERATE | Low |
| 12 | PR-AUC 0.236 → connect to spatial proxy mechanism | MODERATE | Low |
| 13 | Power analysis for intersectional cells | MODERATE | Medium |
| 14 | Disclaimer paragraph too heavy | MODERATE | Low |
| 15 | Prose restates what tables/figures already show | MODERATE | Low |
| 16 | Spanish axis labels in English paper | MINOR | Medium |
| 17 | "Algorithm independence" → "cross-architecture consistency" | MINOR | Low |
| 18 | Writing style: trim verbosity throughout | MINOR | Medium |

---

## CRITICAL

### #1 — Abstract opens with negation

**Current first sentence:** *"This paper does not audit Peru's Alerta Escuela early warning system directly."*

Opening with what you didn't do is an apology, not a hook. The disclaimer is essential — but it belongs after the contribution.

**Fix:** Restructure to: problem → contribution → scope caveat → findings.

> "Dropout prediction systems are proliferating across Latin America, yet their fairness properties remain unaudited. We construct a proxy dropout prediction model from publicly available ENAHO survey data (2018–2023, N=150,135) targeting the same school-age population as Peru's Alerta Escuela early warning system. We have not accessed Alerta Escuela's predictions, training data, or operational feature set; our findings characterize disparities in survey-derived dropout risk modeling, not the deployed system itself."

Same information, contribution-first ordering.

---

### #2 — Urban indigenous CI [0.211, 1.000] undermines headline

The 95% CI spans nearly the full [0, 1] range. The data are compatible with:

- FNR ≈ 0.21 → no problem exists, finding is an artifact
- FNR ≈ 0.50 → moderate, comparable to Quechua
- FNR ≈ 0.75 → the point estimate, catastrophic
- FNR ≈ 1.00 → worse than claimed

With n=89 and ~20% dropout rate, you have ~18 actual dropouts. The model missed ~13–14. This is not "robust point estimate with high uncertainty" — it's maximum uncertainty.

**Fix (do both A and B):**

**A. Downgrade the rhetoric.** Reframe as hypothesis-generating. Lead with Castellano vs. indigenous FNR gap (well-powered, p < 0.001) as the primary finding. Urban indigenous becomes "a suggestive pattern requiring confirmation."

**B. Pool validation + test data.** The 2022 set likely has another ~80–100 urban other-indigenous students. Pooling to n≈170–180 roughly halves CI width. If the estimate holds at ~0.75 with CI ≈ [0.55, 0.90], the finding is credible. If it collapses, you've avoided a false claim. Flag clearly that this uses non-test data.

---

### #3 — Abstract overstates 75% FNR confidence

Most readers only read the abstract. Your paper will be cited as "Flores Teniente showed dropout prediction misses 75% of urban indigenous students" — a claim the data don't support.

**Fix:** Replace with:

> "Intersectional analysis identifies urban indigenous students as a potentially high-FNR subgroup (point estimate 0.75, n=89), though small sample size limits precision."

---

### #4 — Paper length (22pp → 15–18pp)

JEDM typical range is 15–25 pages. Aim for ~16–17 after these cuts:

| Cut | Pages saved |
|-----|-------------|
| Delete Section 7.1 (#7) | ~1 |
| Compress related work (#8) | ~1.5 |
| Move Figures 2, 3, 5 to appendix (#9) | ~1.5 |
| Trim prose restating tables (#15) | ~0.5 |
| **Total** | **~4.5** |

---

## IMPORTANT

### #5 — No feature ablation

The spatial proxy claim is supported by SHAP correlation but never tested experimentally.

**Fix:** Run LightGBM on two additional feature subsets:

1. Individual + household only (drop all 9 district-level spatial features)
2. Spatial features only (drop individual/household)

Report FNR by language group for each. One table, one paragraph. Disparities persist without spatial features → mechanism is more complex. Disparities disappear → mechanism confirmed experimentally.

---

### #6 — No normative fairness discussion

The paper documents the FNR–FPR trade-off and cites Chouldechova but never argues which side matters more for this context.

**Fix:** Add 1–2 paragraphs to Section 7.3:

1. State which fairness criterion the paper privileges and why
2. Acknowledge what equalizing FNR would cost (higher FPR for Spanish speakers)
3. Note that the right trade-off depends on what happens after flagging — low-cost intervention (phone call) makes FPR tolerable; high-cost (home visit) doesn't

Don't resolve the question — show you've thought about it.

---

### #7 — Discussion 7.1 repeats results

Section 7.1 restates the three RQ answers just presented in Section 6. Full page, zero new content.

**Fix:** Delete entirely. Start discussion at 7.2 (spatial proxy mechanism).

---

### #8 — Related work too long

~2.5 pages of sequential summaries. Re-explains concepts the JEDM audience already knows.

**Fix:** Three thematic paragraphs, ~1 page total:

1. **Dropout EWS:** Bowers → Lakkaraju → Knowles → Adelman. Point: fairness audits didn't follow EWS expansion to developing countries. (4–5 sentences)
2. **Fairness in education:** Kizilcec/Lee, Baker/Hawn, Chouldechova, Pan/Zhang. Point: most work is US/European, race/gender, no survey weights. (4–5 sentences)
3. **Intersectionality + LatAm:** Crenshaw, Buolamwini/Gebru, Cueto et al. Point: Peru's axes of disadvantage differ from Global North. (4–5 sentences)

---

### #9 — Figures duplicate tables

- **Figure 2** (PR curves) → Table 5 has the numbers
- **Figure 3** (calibration plot) → Brier scores in text
- **Figure 5** (FNR/FPR bars) → Table 8 has same data plus CIs and p-values

**Fix:** Move all three to appendix. Keep Figures 1, 4, 6, 7, 8.

---

### #10 — Conclusion too compressed

One sentence. Too abrupt.

**Fix:** Expand to 5–7 sentences:

1. Key findings (one sentence)
2. Proxy audit as methodology — independent accountability using public data
3. Limitation: audits proxy, not deployed system; MINEDU transparency would enable direct evaluation
4. Future work: larger intersectional samples, community engagement
5. Closing hook: who decides the right fairness trade-off for Peru's students?

---

## MODERATE

### #11 — Key references dropped

Restore **Kearns et al. (2018)** — proves single-axis auditing is provably insufficient. Theoretical backbone of your intersectional argument. One sentence in Section 2.

Restore Hébert-Johnson and Mehrabi if space permits.

---

### #12 — PR-AUC 0.236 → connect to spatial proxy

Low PR-AUC isn't just a limitation — it's consistent with the model being a geographic stratifier.

**Fix:** Add to Section 5.1 or 7.2:

> "The modest PR-AUC is itself informative: a model achieving lift primarily through geographic stratification will produce predictable fairness failures where spatial and demographic profiles diverge — precisely the pattern documented here."

---

### #13 — Power analysis for intersectional cells

Transform vague "n is small" caveat into a precise research agenda.

**Fix:** One paragraph calculating required n to detect FNR = 0.75 vs. 0.63 (Castellano) at 80% power, and how many ENAHO years that requires.

---

### #14 — Disclaimer paragraph too heavy

End of intro has a standalone five-negative-claims block. Reads defensively.

**Fix:** Condense to 2 sentences woven into contributions paragraph:

> "We emphasize that this is a proxy audit: we have not accessed Alerta Escuela's predictions, training data, or operational feature set, and make no claims about the deployed system's specific fairness properties. Our findings characterize disparities that can emerge from survey-derived dropout prediction in Peru's demographic context."

---

### #15 — Prose restates tables/figures

**Rule:** Prose accompanying a table/figure should only contain information the reader *cannot* see by looking at it. Cut description, keep interpretation.

**Example:** Instead of *"LightGBM, XGBoost, and RF achieve near-identical validation PR-AUC (0.262, 0.263, and 0.261 respectively)"* → *"Near-identical PR-AUC across three tree-based ensembles (Table 5) confirms fairness findings reflect data structure, not model artifacts."*

---

## MINOR

### #16 — Spanish axis labels

All figures have Spanish labels. JEDM is English-language. Regenerate with English labels.

---

### #17 — "Algorithm independence" → "cross-architecture consistency"

More precise. You tested five implementations, not proved universality. Rename section header and all references.

---

### #18 — Writing style: trim verbosity

Apply your own stated preference (concise, zero redundancy) to the paper itself:

- Cut hedging stacks: "it is worth noting that this finding suggests" → state the finding
- Cut meta-commentary: "We now turn to" → just start
- Cut redundant connectives: "Moreover," "Furthermore" when the connection is obvious
- Active voice by default

---

## Revision Order

1. **Pool val+test data** (#2B) — determines if headline finding survives
2. **Run feature ablation** (#5) — determines if spatial proxy claim holds
3. **Rewrite abstract** (#1, #3)
4. **Delete 7.1, compress related work, move figures** (#7, #8, #9)
5. **Add normative discussion + expand conclusion** (#6, #10)
6. **Fix remaining issues** (#11–18)
7. **Final pass:** writing style cleanup (#15, #18) on entire manuscript
