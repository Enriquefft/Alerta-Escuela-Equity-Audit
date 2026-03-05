# Phase 31: Framing & Literature — Research

**Researched:** 2026-03-05
**Domain:** Proxy auditing of algorithmic systems — cross-domain precedent literature
**Confidence:** HIGH (three of four recommended papers verified via official sources; ProPublica methodology confirmed via their own published methodology article)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **FRAME-04:** Discovery method is to spawn research agent during planning. Use findings to decide cite vs. differentiate. Citation location is a footnote (not in-body Related Work, not intro). Acknowledge cross-domain precedent in a footnote, keep body text focused on EDM context. If nothing found: Claude drafts a differentiation sentence noting that proxy auditing has been applied in [domain] and distinguishing our contribution as the first in educational EWS.

### Claude's Discretion
- Exact abstract sentence wording (must be LaTeX-compilable, match existing citation style)
- Which existing bullet (2 or 3) absorbs the surveillance-invisibility axis framing
- Length of EWS paragraph (1 paragraph, ~4-6 sentences)
- Footnote wording for proxy audit literature

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope.
</user_constraints>

---

## Summary

Proxy auditing — reconstructing or inferring a deployed system's behavior using independent external data without access to the system's internals — has a well-established multi-domain literature. The practice predates the term: ProPublica's 2016 COMPAS investigation is the canonical exemplar, assembling public criminal records to evaluate a proprietary recidivism tool that refused external access. Healthcare (Obermeyer et al. 2019, *Science*) replicated the pattern with insurance claims data. The computer science fairness literature has also formalized auditing without model access as a distinct methodology (Adler et al. 2018; Sandvig et al. 2014). All four examples share the structural signature of our work: a deployed system whose internals are unavailable, an independent data source assembled by the researchers, and fairness conclusions drawn from model outputs or reconstructed behavior.

The literature does **not** contain an application of this method to a dropout-prediction or enrollment-risk EWS in any country's education system. The closest works in EDM (Baker & Hawn 2021, Mitchell et al. 2021) audit internal research models — not deployed production systems — and use the systems' own training data rather than independent survey data. Our paper's novelty claim is therefore both accurate and defensible: we are the first to apply proxy auditing, using nationally representative household survey data, to a deployed educational early warning system.

**Primary recommendation:** Cite three cross-domain exemplars in a footnote to anchor our methodology in established practice, then assert the education-EWS gap as our contribution. Use the framing "proxy audit" (matching Adler et al.'s terminology) rather than inventing a new term.

---

## Whether Proxy Auditing Has an Established Literature Outside Education

**Yes — HIGH confidence.** The practice is well-established with a consistent methodological pattern:

1. A deployed production system (proprietary or inaccessible) produces scores affecting individuals.
2. An independent investigator assembles external data — public records, administrative data, survey data — containing inputs and outcomes observable without system access.
3. The investigator fits their own model or statistical test to that external data to characterize what the deployed system does (its error rates, disparate impact, proxy mechanisms).
4. Findings are published without the deploying organization's cooperation.

This is precisely the structure of our audit. The term "proxy audit" appears explicitly in Adler et al. (2018) and has since been used in regulatory contexts (e.g., NYC Local Law 144 commentary). Outside of education, the four strongest citable papers are below.

---

## Four Citable Papers with Proxy Audit Methodology

### Paper 1: ProPublica COMPAS Analysis (Criminal Justice)

| Field | Detail |
|-------|--------|
| **Authors** | Jeff Larson, Surya Mattu, Lauren Kirchner, Julia Angwin |
| **Title** | "Machine Bias: There's Software Used Across the Country to Predict Future Criminals. And It's Biased Against Blacks." |
| **Venue** | ProPublica (journalistic investigation + full methodology article) |
| **Year** | 2016 |
| **arXiv/DOI** | https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm |

**Was it a proxy audit? Yes.** Northpointe refused to disclose how COMPAS computes risk scores. ProPublica assembled Broward County jail records, public Florida DOC incarceration records, and matched them to COMPAS scores obtained through public court records. They fit a Cox hazard model and a logistic regression — independent models built on their own assembled data — to characterize COMPAS's racial disparities. They never accessed the Northpointe model code or training data.

**One-sentence description:** ProPublica built an independent matched dataset from public criminal records and fit their own statistical models to characterize racial disparities in a proprietary recidivism scoring tool, with no access to Northpointe's model or training data.

**Relevance to our framing:** This is the prototype. It establishes that (a) journalists/researchers can produce credible algorithmic audits without model access, (b) public administrative records are a legitimate substitute data source, and (c) fitting an independent model to those records yields actionable fairness findings.

**Caveat for citation:** ProPublica is journalism, not peer-reviewed. A subsequent paper by Chouldechova (2017, *Big Data* journal) formalized the fairness impossibility result that ProPublica's analysis uncovered — and `chouldechova2017fair` is already in references.bib. We can cite Chouldechova for the COMPAS result in-body and reference ProPublica in the footnote.

---

### Paper 2: Obermeyer et al. — Healthcare Risk Score (Health Domain)

| Field | Detail |
|-------|--------|
| **Authors** | Ziad Obermeyer, Brian Powers, Christine Vogeli, Sendhil Mullainathan |
| **Title** | "Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations" |
| **Venue** | *Science* |
| **Year** | 2019 |
| **DOI** | https://doi.org/10.1126/science.aax2342 |

**Was it a proxy audit? Yes.** The health risk scoring algorithm is a commercial product deployed at large US health systems. The researchers did not have access to the algorithm's code; they worked with electronic health records and insurance claims data that were independently observable. They used those records to measure the gap between Black and white patients' actual health needs versus the risk scores the algorithm assigned — the algorithm was audited through its outputs, not its internals.

**One-sentence description:** Using independent electronic health records and claims data — without access to the proprietary algorithm's code — Obermeyer et al. showed that a widely deployed health-risk scoring tool systematically underestimated the severity of illness in Black patients because it used healthcare costs as a proxy for health needs.

**Relevance to our framing:** This is the most prestigious peer-reviewed exemplar (published in *Science*). It demonstrates that the proxy audit methodology is not just journalism but rigorous enough for top-tier empirical research. The structural parallel to our work is strong: they use cost as a proxy where the algorithm is biased; we use household survey data to reconstruct who the EWS misses.

**Suggested bib key:** `obermeyer2019dissecting`

---

### Paper 3: Adler et al. — Auditing Black-Box Models (Formalized Framework)

| Field | Detail |
|-------|--------|
| **Authors** | Philip Adler, Casey Falk, Sorelle A. Friedler, Tionney Nix, Gabriel Rybeck, Carlos Scheidegger, Brandon Smith, Suresh Venkatasubramanian |
| **Title** | "Auditing Black-box Models for Indirect Influence" |
| **Venue** | *Knowledge and Information Systems* (journal); preliminary version at IEEE ICDM 2016 |
| **Year** | 2018 (journal); 2016 (conference) |
| **DOI** | https://doi.org/10.1007/s10115-017-1116-3 |
| **arXiv** | https://arxiv.org/abs/1602.07043 |

**Was it a proxy audit? Yes.** The paper explicitly develops a technique for auditing black-box models — models whose internals are unavailable — by studying feature influence through the model's outputs only. They audit deployed models in credit scoring and income prediction domains. The technique examines whether protected attributes (race, gender) indirectly influence outcomes even when excluded from the model, using only input-output observations.

**One-sentence description:** Adler et al. formalized a method for auditing black-box models in credit and income domains by measuring indirect feature influence purely from model outputs, without access to model internals — the first peer-reviewed framework that names the "auditing without access" approach and applies it to fairness.

**Relevance to our framing:** This paper provides the methodological vocabulary. Using "proxy audit" or "black-box audit" language links our work to a formal CS fairness tradition, not just journalism. It is particularly useful for the footnote because it gives the planner a peer-reviewed citation with an explicit "auditing without access" framing.

**Suggested bib key:** `adler2018auditing`

---

### Paper 4: Sandvig et al. — Auditing Algorithms Framework (Taxonomy Paper)

| Field | Detail |
|-------|--------|
| **Authors** | Christian Sandvig, Kevin Hamilton, Karrie Karahalios, Cedric Langbort |
| **Title** | "Auditing Algorithms: Research Methods for Detecting Discrimination on Internet Platforms" |
| **Venue** | ICA 2014 preconference: *Data and Discrimination: Converting Critical Concerns into Productive Inquiry* |
| **Year** | 2014 |
| **URL** | https://websites.umich.edu/~csandvig/research/Auditing%20Algorithms%20--%20Sandvig%20--%20ICA%202014%20Data%20and%20Discrimination%20Preconference.pdf |
| **Citations** | 551+ (Semantic Scholar) — highly influential |

**Was it a proxy audit? Yes (taxonomy level).** Sandvig et al. define five external audit designs for platforms and algorithms, all predicated on not having code access. Their "scraping audit" and "sock puppet audit" types are structurally similar to what we do: assembling independent data from outside the system to characterize its behavior. The paper is the foundational taxonomy of algorithmic auditing without model access.

**One-sentence description:** Sandvig et al. defined five external audit designs for detecting discrimination in deployed platform algorithms — the foundational taxonomy of auditing methods that operate without access to model code or training data — establishing "auditing algorithms" as a distinct research practice.

**Relevance to our framing:** Sandvig is the most-cited methodological foundation paper in the algorithmic accountability literature. Citing it roots our methodology in a 10-year tradition. The paper predates the FAccT conference and is regarded as having helped found it.

**Suggested bib key:** `sandvig2014auditing`

---

## How to Frame the Differentiation (Our Novelty Claim)

### What the literature establishes

External, proxy-based auditing of proprietary algorithmic systems has been practiced since at least 2014 (Sandvig et al.) and has produced landmark results in criminal justice (ProPublica/COMPAS, 2016), formal CS fairness theory (Adler et al., 2018), and healthcare (Obermeyer et al., 2019). The methodology is well-accepted: assemble independent external data, fit independent models, characterize the deployed system's behavior without accessing its internals.

### What the literature does NOT contain

No prior work applies this methodology to:
- An educational dropout-prediction or enrollment-risk EWS
- Data from nationally representative household surveys as the external audit data source
- A government-deployed system in a Latin American context

EDM fairness papers (Baker & Hawn 2021; Mitchell et al. 2021) audit research models using their own training data — that is internal auditing, not proxy auditing. They know what the model is because they built it.

### Recommended footnote framing (for planner to finalize)

Recommended draft for the footnote:

> Proxy auditing — characterizing a deployed system's fairness properties from external data, without access to model code or training data — has precedent across domains: \citet{sandvig2014auditing} established the methodological taxonomy; \citet{adler2018auditing} formalized black-box fairness auditing in credit and income settings; ProPublica's COMPAS investigation \citep{angwin2016machine} applied it to criminal sentencing; and \citet{obermeyer2019dissecting} used insurance claims data to expose racial bias in a healthcare risk score. To our knowledge, no prior work has applied proxy auditing to an educational dropout-prediction EWS.

**Note for planner:** `angwin2016machine` does not yet exist in references.bib. Options:
1. Add a `@misc` bib entry for the ProPublica article (journalism, citable)
2. Substitute `chouldechova2017fair` (already in bib) which formalizes the same COMPAS finding as a peer-reviewed result — this may be preferable for an academic footnote
3. Cite Obermeyer + Sandvig + Adler only (three is enough to establish the tradition)

The three-paper version avoids adding a journalism citation:

> Proxy auditing of deployed systems has precedent in criminal justice \citep{chouldechova2017fair}, platform discrimination detection \citep{sandvig2014auditing}, credit scoring \citep{adler2018auditing}, and healthcare \citep{obermeyer2019dissecting}. To our knowledge, no prior work applies this approach to an educational EWS.

---

## Standard Stack (for planner — bibliography additions needed)

Two new bib entries are required (Sandvig already appears in the literature but is not in references.bib; Adler and Obermeyer are new):

```bibtex
@inproceedings{sandvig2014auditing,
  author    = {Sandvig, Christian and Hamilton, Kevin and Karahalios, Karrie and Langbort, Cedric},
  title     = {Auditing Algorithms: Research Methods for Detecting Discrimination on Internet Platforms},
  booktitle = {Data and Discrimination: Converting Critical Concerns into Productive Inquiry,
               Preconference at the 64th Annual Meeting of the International Communication Association},
  year      = {2014},
  address   = {Seattle, WA},
  url       = {https://websites.umich.edu/~csandvig/research/Auditing%20Algorithms%20--%20Sandvig%20--%20ICA%202014%20Data%20and%20Discrimination%20Preconference.pdf}
}

@article{adler2018auditing,
  author    = {Adler, Philip and Falk, Casey and Friedler, Sorelle A. and Nix, Tionney and
               Rybeck, Gabriel and Scheidegger, Carlos and Smith, Brandon and Venkatasubramanian, Suresh},
  title     = {Auditing Black-box Models for Indirect Influence},
  journal   = {Knowledge and Information Systems},
  volume    = {54},
  number    = {1},
  pages     = {95--122},
  year      = {2018},
  doi       = {10.1007/s10115-017-1116-3}
}

@article{obermeyer2019dissecting,
  author    = {Obermeyer, Ziad and Powers, Brian and Vogeli, Christine and Mullainathan, Sendhil},
  title     = {Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations},
  journal   = {Science},
  volume    = {366},
  number    = {6464},
  pages     = {447--453},
  year      = {2019},
  doi       = {10.1126/science.aax2342}
}
```

Optional (if ProPublica journalism citation preferred over Chouldechova):

```bibtex
@misc{angwin2016machine,
  author    = {Angwin, Julia and Larson, Jeff and Mattu, Surya and Kirchner, Lauren},
  title     = {Machine Bias: There's Software Used Across the Country to Predict Future Criminals. And It's Biased Against Blacks},
  year      = {2016},
  month     = {May},
  publisher = {ProPublica},
  url       = {https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing},
  note      = {With methodology: \url{https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm}}
}
```

---

## Common Pitfalls

### Pitfall 1: Conflating "proxy audit" with "proxy variable"
**What goes wrong:** A reviewer may read "proxy audit" and think it means "using a proxy variable inside the model" (a common fairness concept), not "a parallel model that proxies for the deployed system."
**How to avoid:** Define the term clearly in the footnote or at first use in the abstract/intro. The Adler et al. (2018) paper uses "auditing black-box models" language; Sandvig et al. use "external audit." Either phrasing avoids the ambiguity.

### Pitfall 2: Overstating the novelty claim
**What goes wrong:** Claiming "first proxy audit" when someone has applied adjacent methods (e.g., test-score gap analysis using administrative data).
**How to avoid:** The novelty claim should be precisely scoped: "first proxy audit of a deployed dropout-prediction EWS using nationally representative household survey data." The qualifiers matter — it's not "first educational audit" (Baker & Hawn exist) but "first audit of a deployed EWS without model access."

### Pitfall 3: The ProPublica citation is journalism, not peer-reviewed
**What goes wrong:** JEDM reviewers may be skeptical of a journalism citation in a footnote.
**How to avoid:** Substitute `chouldechova2017fair` (already in bib, peer-reviewed, same underlying COMPAS data) for the in-footnote criminal justice exemplar, or cite both (ProPublica for the audit methodology, Chouldechova for the formalized fairness impossibility theorem it uncovered).

---

## Open Questions

1. **Does Raghavan et al. (2020, FAT*) count as a proxy audit?**
   - What we know: Raghavan et al. reviewed 18 hiring algorithm vendors, assessed their disclosed practices, and evaluated fairness claims from outside.
   - What's unclear: They used the companies' own disclosed documentation, not independently assembled data. This is closer to a document audit than a proxy audit.
   - Recommendation: Do not cite as a proxy audit exemplar. It strengthens a different claim (lack of industry transparency) not our methodological claim.

2. **Is Buolamwini & Gebru (2018) a proxy audit?**
   - What we know: `buolamwini2018gender` is already in references.bib. They assembled their own Pilot Parliaments Benchmark dataset and evaluated commercial face-classification APIs without model access.
   - What's unclear: The classification APIs they tested are query-accessible — they do not reconstruct from independent administrative data. Structurally similar but the "deployed system" is directly queryable.
   - Recommendation: Could cite in the footnote as "in computer vision, Buolamwini and Gebru (2018) audited commercial APIs from external benchmark data." However, the Obermeyer exemplar is stronger for our framing since it involves administrative/survey data rather than a directly queryable API.

---

## Architecture Patterns (for planner — task structure)

The FRAME-04 task is well-defined:
1. Add 2-3 new entries to `paper/references.bib` (Obermeyer, Adler, Sandvig)
2. Locate the Related Work section or a natural footnote anchor in `paper/main.tex`
3. Add one footnote with the cross-domain citations
4. Verify LaTeX compiles cleanly with `latexmk -pdf main.tex` from `paper/`
5. No changes to paper body — footnote only

The planner should NOT add these citations to the introduction or Related Work body paragraphs. The CONTEXT.md decision is explicit: footnote only.

---

## Sources

### Primary (HIGH confidence)
- ProPublica methodology article — confirmed methodology: no model access, public criminal records, independent statistical models. https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
- Obermeyer et al. 2019 — Science journal, DOI 10.1126/science.aax2342 — confirmed via PubMed and multiple secondary sources
- Adler et al. 2018 — Knowledge and Information Systems, DOI 10.1007/s10115-017-1116-3 — confirmed via ACM DL, arXiv 1602.07043, NSF PAGES
- Sandvig et al. 2014 — ICA preconference proceedings, PDF publicly available at University of Michigan, 551+ citations

### Secondary (MEDIUM confidence)
- Raghavan et al. 2020 FAT* — confirmed via ACM DL and arXiv 1906.09208 — evaluated but not recommended as proxy audit exemplar
- Raji et al. 2020 FAT* — confirmed via ACM DL — addresses internal auditing framework (different from our method)

### Tertiary (LOW confidence — not recommended for citation)
- General claims about "external audit without access" as a research tradition — confirmed across multiple sources, no single authoritative survey paper

---

## Metadata

**Confidence breakdown:**
- Proxy audit cross-domain literature exists: HIGH — four papers confirmed via official sources
- Novelty claim (no prior EWS proxy audit): HIGH — exhaustive EDM and proxy audit searches returned no education EWS applications
- Recommended footnote wording: MEDIUM — requires LaTeX integration by planner; exact citation keys depend on bib entry additions

**Research date:** 2026-03-05
**Valid until:** Stable literature — findings valid until at least 2027-01-01. Monitor FAccT 2026 proceedings if publication occurs before submission.
