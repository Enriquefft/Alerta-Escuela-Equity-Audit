# Phase 11: Findings Distillation + Final Exports — Context

## Finding Selection & Impact Ordering

### Ordering Strategy: Narrative Arc
Findings are ordered to tell a story, not by statistical magnitude alone:
1. **The system exists** — what Alerta Escuela does and how our audit replicates it
2. **Who it misses** — FNR disparities across protected dimensions
3. **Why it misses them** — SHAP reveals structural features drive predictions, not identity
4. **Proof** — admin data cross-validation confirms spatial equity patterns

### Scope: Equity-Focused
- All 5-7 findings center on fairness and who the system fails
- Model performance (PR-AUC, calibration) is supporting context, NOT a standalone finding
- Method credibility comes through the narrative, not as separate findings

### Media-Ready Bar: Dual Test
Every finding must pass BOTH:
1. **One-sentence headline test** — a journalist could print the headline as-is
2. **Stat + so-what test** — concrete number AND clear "this means X for Y students" implication

If a finding fails either test, it's background context, not a finding.

### metric_source Format: Path + Label
Each finding's `metric_source` contains:
- **path**: Machine-resolvable JSON path (e.g., `fairness_metrics.json#dimensions.language.other_indigenous.fnr`)
- **label**: Human-readable description (e.g., "FNR for Other Indigenous language group")

Gate test 3.4 will verify every path resolves to an actual value.

## Bilingual Tone & Style

### Spanish Register: Journalistic-Accessible with Academic Grounding
- Clear, direct language like El Comercio or RPP reporting
- Doesn't shy from precise terms (tasa de falsos negativos) but avoids unnecessary jargon
- NOT activist/advocacy tone, NOT dry academic passive voice
- Peruvian context used naturally (Selva, Sierra, MINEDU, etc.)

### Headline Format: Stat-Forward
- Number hits first in the headline
- Example: "6 de cada 10 estudiantes indígenas en riesgo no son detectados por el sistema"
- NOT: "Los estudiantes que el sistema no ve" (that's narrative-forward)

### English-Spanish Relationship: Parallel Adaptation
- NOT literal translation — each version adapted for its audience
- Spanish uses Peruvian context naturally: "la región Selva", "MINEDU"
- English adds geographic/institutional context: "the Selva region (Amazon basin)", "Peru's Ministry of Education"
- Same finding, same stat, different framing

### Explanation Depth: 2-3 Sentences Max
- Sentence 1: State the finding with the number
- Sentence 2: What it means for students/policy
- Optional sentence 3: Methodological anchor if needed
- Scrollytelling site handles visualization — explanations don't describe charts

## Export README

### Audience: M4 Site Developers
- Primary purpose: help developers integrate exports into the scrollytelling site
- Assumes developer knows JavaScript/React but not the analysis methodology
- Methodology details are secondary (for reproducibility, not integration)

### Structure: Overview Table + Per-File Sections
1. **Overview table** — quick reference mapping each file to its site section
2. **Per-file sections** — each of the 7 exports gets:
   - Purpose (1 sentence)
   - Schema (field names, types, example values)
   - Which M4 site component consumes it
   - File size / record count

### Data Provenance: Source + Pipeline + Phase
Each file section includes provenance:
- **Data source**: e.g., "ENAHO 2018-2023 microdata (INEI)"
- **Pipeline step**: e.g., "SHAP TreeExplainer on LightGBM test predictions"
- **Producing script**: e.g., `src/fairness/shap_analysis.py` (Phase 9)

### Language: English Only
Technical documentation in English — standard for code repos.

## Deferred Ideas
*(None raised during discussion)*
