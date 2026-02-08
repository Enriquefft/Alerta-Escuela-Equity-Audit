# Phase 4: Feature Engineering + Descriptive Statistics - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Engineer all 19+ model features from the merged dataset (full_dataset.parquet) and compute survey-weighted descriptive statistics quantifying dropout gaps across 6 fairness dimensions. Produce the first JSON export (descriptive_tables.json) for the M4 scrollytelling site and 7 visualizations for human review.

Requirements: DATA-09, DATA-10, DESC-01 through DESC-06.

</domain>

<decisions>
## Implementation Decisions

### Feature construction
- **Poverty quintiles:** Use FACTOR07 survey weights to split into 5 groups so each quintile represents ~20% of the weighted population
- **Mother tongue feature:** Claude's discretion on encoding — follow spec Section 5 for exact column names. Preserve disaggregated codes for descriptive analysis regardless of model encoding choice
- **Age feature:** Include BOTH raw continuous age (6-17) AND primaria-age (6-11) vs secundaria-age (12-17) binary. Captures the known dropout cliff at the primaria→secundaria transition
- **Spatial features:** Standardize (z-score) all district-level features from the spatial merge (admin dropout rates, census indicators, nightlights radiance) before they enter the model. Helps logistic regression convergence

### Descriptive breakdowns
- **Language disaggregation:** Top 5 + Other — Castellano, Quechua, Aymara, Awajun, Ashaninka, Other indigenous. Binary indigenous/Castellano split as headline stat
- **Temporal trends:** Show all 6 individual years (2018-2023). Captures COVID impact in 2020, recovery trajectory
- **Heatmaps:** Three heatmaps, not just one — language x rurality, language x poverty quintile, language x region
- **Sex breakdown:** Sex x education level (male/female rates for primaria-age and secundaria-age separately). Captures the known gender flip where boys drop out more in secundaria

### Visualization approach
- **Format:** Both console-printed tables for quick gate review AND matplotlib PNGs saved as image files
- **PNG location:** data/exports/figures/ (tracked in git alongside JSON exports)
- **7 visualizations:** All 7 as specified — (1) language bars, (2) sex x education bars, (3) rural/urban bars, (4) region bars, (5) poverty quintile bars, (6) language x rurality heatmap, (7) temporal trend lines
- **Color palette:** Claude's discretion — pick a clean, accessible palette appropriate for an equity audit

### JSON export schema
- **Structure:** One object per breakdown — top-level keys: language, sex, rural, region, poverty, heatmap, temporal. Each contains its own array of records
- **Metadata:** Include _metadata key with timestamp, source_rows, years_covered, and sample sizes per subgroup
- **Number precision:** Rounded to 4 decimal places (0.1342 not 0.13421587)
- **Uncertainty:** Include 95% confidence intervals (lower_ci, upper_ci) for each weighted rate. Shows precision varies by group size

### Claude's Discretion
- Mother tongue model encoding (binary vs multi-category dummies)
- Exact matplotlib color palette and styling
- Confidence interval computation method (linearization vs bootstrap)
- Any additional features beyond the 19 specified if they clearly serve the fairness audit

</decisions>

<specifics>
## Specific Ideas

- Awajun dropout rate >18% for 2020+ years is a key success criterion — make sure this is prominently visible in both console output and visualizations
- The 3 heatmaps (language x rurality, language x poverty, language x region) go beyond the spec's single heatmap — extra context for understanding intersectional patterns before Phase 8 formal analysis
- Both primaria-age and secundaria-age buckets are important because the dropout cliff happens at transition — the sex x education breakdown should highlight this

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-feature-engineering-descriptive-statistics*
*Context gathered: 2026-02-07*
