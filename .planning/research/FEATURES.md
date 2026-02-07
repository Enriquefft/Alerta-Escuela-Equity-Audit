# Feature Research

**Domain:** ML Fairness Audit of Education Dropout Prediction (Peru / Alerta Escuela)
**Researched:** 2026-02-07
**Confidence:** MEDIUM (based on training data through May 2025; WebSearch/WebFetch unavailable for verification of latest library versions)

---

## Feature Landscape

### Table Stakes (Audit Loses Credibility Without These)

Features that any credible ML fairness audit must include. Omitting any of these would be flagged by peer reviewers, journalists, or policymakers as a methodological gap.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Survey-weighted metrics throughout** | ENAHO is a complex survey with expansion factors (FACTOR07). Unweighted metrics describe the sample, not the population. Any published finding using unweighted ENAHO metrics is methodologically wrong. Peer reviewers will reject immediately. | MEDIUM | Every sklearn metric function accepts `sample_weight`. Fairlearn's `MetricFrame` supports `sample_params` dict to pass weights per metric. The complexity is not in the API but in the discipline: every single metric call, including descriptive stats, must pass weights. Produce both weighted and unweighted and assert they differ as a sanity check. |
| **Subgroup fairness metrics (6 dimensions)** | The audit's core purpose. Disparate impact across sex, language, geography, region, poverty, and nationality is the central claim. Without per-group TPR, FPR, FNR, precision, and PR-AUC, there is no audit. | MEDIUM | Use fairlearn `MetricFrame` with `sensitive_features` parameter. Each of the 6 dimensions gets its own MetricFrame. Compute `mf.by_group`, `mf.difference()`, `mf.ratio()`. For multiple metrics per dimension, pass a dict of metric functions. |
| **Intersectional analysis (3 intersections)** | Single-axis analysis masks compounding disadvantage. A Quechua-speaking rural female student faces different risk than any single axis reveals. Intersectional analysis is standard in fairness literature since Buolamwini & Gebru (2018). | HIGH | Create composite group columns (e.g., `language_rurality = p300a_harmonized + "_" + rural`). Pass as `sensitive_features` to MetricFrame. Challenge: cell sizes shrink rapidly. Must flag groups with <50 unweighted observations. Three intersections specified: language x rurality, sex x poverty, language x region. |
| **False Negative Rate (FNR) by group** | FNR = missed dropouts. If the model systematically misses indigenous-language dropouts more than Castellano-speakers, those students lose access to intervention. FNR disparity is the most consequential fairness metric for a dropout prediction system because it determines who gets help. | LOW | FNR = 1 - recall. Already computed as part of subgroup metrics. The key is to foreground it in findings because it is the most media-understandable metric: "X% of indigenous dropouts are missed vs Y% of Spanish-speaking dropouts." |
| **Calibration analysis by group** | If the model says "70% dropout risk" for an indigenous student but the actual rate is 30%, the model is miscalibrated for that group. Calibration fairness (sufficiency) is distinct from equalized odds. Both must be checked. | MEDIUM | For each subgroup, among predicted-high-risk (>threshold, e.g., 0.7), compute actual dropout rate. Use sklearn `calibration_curve` per group. Also compare Brier scores per group. The key output is: "Does 'high risk' mean the same thing for every group?" |
| **SHAP-based feature attribution** | Claiming "nationality is a biased predictor" without quantifying its marginal contribution is an unsupported assertion. SHAP provides theoretically grounded, additive feature attributions for LightGBM. Required to answer: "How much does being non-Peruvian independently increase predicted risk?" | MEDIUM | `shap.TreeExplainer` is fast for LightGBM. Compute on test set (2024), not train. For binary classification, LightGBM returns list of arrays; take `shap_values[1]` for positive class. Mean absolute SHAP per feature gives global importance. Per-instance SHAP enables waterfall plots for representative profiles. |
| **Temporal train/validate/test split** | Using random splits on panel data leaks temporal information. Dropout patterns shift year-to-year (e.g., COVID in 2020). A credible audit uses temporal splits: train on past, evaluate on future. | LOW | Straightforward: filter by year column. Train = 2018-2022, Validate = 2023, Test = 2024. The discipline is never touching the test set until final evaluation. |
| **PR-AUC as primary metric (not ROC-AUC)** | With ~14% dropout rate, ROC-AUC is inflated by the large number of true negatives. PR-AUC is sensitive to the minority class performance, which is what matters for dropout prediction. Alerta Escuela reports ROC-AUC, which hides poor precision (~20%). | LOW | `sklearn.metrics.average_precision_score(y_true, y_pred_proba, sample_weight=w)`. Report ROC-AUC secondarily for comparability with Alerta Escuela's published numbers. |
| **Reproducibility infrastructure** | An audit that cannot be reproduced is not an audit. Fixed random seeds, version-pinned dependencies, deterministic data pipeline, saved intermediate artifacts. | LOW | `random_state=42` everywhere. Pin versions in pyproject.toml. Save processed data as parquet. Export all metrics as JSON. The key insight: reproducibility is not optional for an audit claiming policy implications. |
| **Bilingual findings (Spanish + English)** | End consumers are Peruvian journalists. Findings must be in natural Spanish, not machine-translated. English for international audience and code documentation. | LOW | Every finding in `findings.json` has `headline_es`, `headline_en`, `explanation_es`, `explanation_en`. Feature labels in `shap_values.json` include `feature_labels_es`. This is a content task, not a technical one, but omitting it makes the audit useless for its intended audience. |
| **JSON export contracts for downstream visualization** | The audit feeds a Next.js scrollytelling site. Without clean, schema-documented JSON exports, the data-to-visualization pipeline breaks. The export schema IS the API between the audit and the media product. | MEDIUM | Seven export files with documented schemas (Section 11 of spec). Each must validate against its schema. The README.md in exports/ documents provenance. The key discipline: exports are committed to git (unlike raw data), making them the single source of truth for the site. |

### Differentiators (Competitive Advantage / Novel Contribution)

Features that distinguish this audit from a generic fairness assessment. These make the audit publishable, newsworthy, and methodologically novel.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Cross-validation against administrative district data** | Most fairness audits use only the model's own predictions. This audit cross-references ENAHO-based predictions against MINEDU administrative dropout rates at the district level, revealing whether model errors are geographically patterned. If indigenous-majority districts have higher prediction error, that is a structural finding beyond any single model metric. | HIGH | Aggregate model predictions to district level (mean predicted proba per UBIGEO). Merge with admin rates on UBIGEO. Compute Pearson correlation (expect positive). Split by % indigenous speakers (from census): compare mean absolute error for high-indigenous (>50%) vs low-indigenous (<10%) districts. This is a novel validation approach that strengthens causal claims. |
| **Regional SHAP decomposition (Costa/Sierra/Selva)** | Peru's three natural regions have fundamentally different dropout dynamics. A national-level SHAP summary hides that language matters more in Selva than in Lima. Regional SHAP comparison reveals geographic heterogeneity in how the model uses features, which has direct policy implications for regional interventions. | MEDIUM | Split test set by `region_natural`. Run `shap.TreeExplainer` separately on each subset. Compare top-5 feature importance rankings across regions. Visualize as side-by-side bar charts. The insight is not just "what matters" but "where it matters differently." |
| **P300A mother tongue harmonization across the 2020 INEI structural break** | In 2020, INEI disaggregated code 3 ("otra lengua nativa") into 6 specific indigenous languages (Ashaninka, Awajun, Shipibo-Konibo, Shawi, Matsigenka, Achuar). This is a coding change, not a population shift. Without harmonization, cross-year analysis produces artifactual trends. With harmonization, this audit can show both the aggregated cross-year picture AND the 2020+ disaggregated picture. No published analysis of Alerta Escuela handles this correctly. | MEDIUM | Implement `harmonize_p300a()` exactly as specified. Preserve BOTH `p300a_harmonized` (cross-year) and `p300a_original` (disaggregated). For findings: use harmonized for trend analysis, original for 2020+ deep-dives. The Awajun-specific finding (~22% dropout) is only possible with the disaggregated codes. |
| **Survey-weighted descriptive gaps as standalone findings** | Before any model is trained, the raw survey data reveals dramatic equity gaps (e.g., Awajun ~22% vs Castellano ~14%). These descriptive findings are model-independent, making them harder to dismiss as "algorithmic artifact." The descriptive layer provides the "so what" context that makes the model-based findings meaningful. | LOW | Weighted crosstabs using polars with FACTOR07. Group by each dimension, compute weighted dropout rate. The simplicity is the strength: "These gaps exist in the population data before any algorithm touches them." |
| **SHAP interaction effects for compound disadvantage** | Standard SHAP gives marginal feature effects. Interaction SHAP reveals whether poverty x language produces a super-additive risk increase -- i.e., being poor AND indigenous is worse than the sum of being poor and being indigenous separately. This is the quantitative evidence for "intersectional harm" beyond just computing metrics for intersection groups. | HIGH | `explainer.shap_interaction_values(X_subsample)` is O(n^2 * n_features^2) for TreeExplainer. Subsample to 1000 rows. Focus on poverty x language and rurality x sex interactions. The output is a 3D array (n_samples x n_features x n_features). Extract the off-diagonal elements for the pairs of interest. |
| **10 representative student profiles with SHAP waterfall** | Aggregate statistics are abstract. A SHAP waterfall for "14-year-old Awajun girl in rural Loreto" makes the audit concrete and media-friendly. Each profile shows: features, predicted probability, base value, and per-feature SHAP contributions as a waterfall. This is the bridge between statistical finding and human story. | MEDIUM | Select 10 profiles spanning: Lima urban Castellano, Sierra rural Quechua, Selva rural indigenous, female secundaria, male secundaria. For each: extract feature values, SHAP values, predicted proba. Export as structured JSON. The scrollytelling site uses these for its interactive "meet the student" section. |
| **Threshold sensitivity analysis** | Alerta Escuela's threshold determines who receives intervention. Reporting metrics at a single threshold hides how fairness gaps change with threshold choice. Showing metrics at 0.3, 0.4, 0.5, 0.6, 0.7 reveals whether there exists ANY threshold that achieves acceptable fairness. If no threshold works, the problem is the model, not the threshold. | LOW | `evaluate_at_thresholds()` with 5 threshold values. For each threshold, compute all fairness metrics across all groups. Export as nested JSON. The key finding: "The FNR gap between indigenous and Castellano speakers persists across all thresholds." |
| **Choropleth-ready spatial data export** | A map of model prediction error by district, colored by indigenous language prevalence, is the most powerful single visualization for the media product. It shows WHERE the model fails and for WHOM. No other visualization combines geographic, demographic, and algorithmic dimensions as compactly. | MEDIUM | Per-district: predicted rate, admin rate, error (predicted - admin), indigenous %, poverty index, lat/lon. Export as `choropleth.json`. The Next.js site renders this with Mapbox or D3 geo. The audit produces the data; the site produces the map. |
| **ONNX model export for browser-based inference** | Exporting the LightGBM model as ONNX enables the scrollytelling site to run live predictions in the browser. Visitors can adjust student features and see how predicted risk changes. This makes the audit interactive, not just a static report. | MEDIUM | `onnxmltools.convert_lightgbm()`. Validate predictions match Python model within 1e-5. The ONNX file is committed to `data/exports/onnx/`. The site loads it with ONNX Runtime Web (onnxruntime-web). Feature names must match exactly between Python training and ONNX input. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but should be deliberately NOT built. Including these would either undermine the audit's credibility, waste effort, or introduce methodological problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Bias mitigation / debiased model** | "If you found bias, why not fix it?" Natural follow-up question. | This is an AUDIT, not a model improvement project. Adding mitigation conflates the auditor role with the developer role. It also implies the current data/model CAN be debiased, which may not be true -- the bias may be structural (e.g., SIAGIE data quality gaps). Debiasing a proxy model built on ENAHO has no effect on the deployed Alerta Escuela system. The audit's power is in documenting problems, not pretending to solve them. | Recommend that MINEDU conduct their own fairness analysis using SIAGIE data. Document specific mitigation strategies they could evaluate (reweighting, threshold adjustment per group, feature exclusion studies) without implementing them. |
| **AIF360 / multiple fairness libraries** | "Why not also use IBM AIF360 for validation?" | AIF360 and fairlearn compute the same metrics with different APIs. Using both creates maintenance burden, API confusion, and "which result is correct?" disputes without adding analytical value. The spec explicitly excludes AIF360. | Use fairlearn exclusively. It has the cleanest API for MetricFrame with sample_weight, active maintenance, and Microsoft Research backing. |
| **Causal fairness analysis** | "Correlation is not causation. Do causal analysis to prove the model is biased." | Causal inference requires a causal DAG, which requires domain assumptions about the data-generating process. ENAHO is a cross-sectional survey, not a randomized experiment. Causal claims about Alerta Escuela's model require SIAGIE data we do not have. Attempting causal analysis with observational survey data would be methodologically weaker than the descriptive + predictive approach. | Frame findings as "disparate impact" (observed outcome differences), not "causal discrimination." This is the legally and methodologically appropriate framing. The audit documents WHAT happens, not WHY it happens. |
| **Deep learning models (TabNet, neural nets)** | "Modern architectures might reveal different fairness patterns." | The spec locks the stack to LogReg, LightGBM, XGBoost. Adding neural nets: (1) requires PyTorch/TensorFlow, (2) makes SHAP analysis intractable (TreeExplainer does not work, KernelExplainer is slow and approximate), (3) the audit is about Alerta Escuela's LightGBM, not about finding the best model, (4) if fairness gaps appear in both a simple model (LogReg) and a complex model (LightGBM), that is stronger evidence of structural bias than adding a third complex model. | Keep exactly 3 models. The LogReg baseline is the most valuable comparison: if linear and nonlinear models both show the same fairness gaps, the problem is in the data, not the algorithm. |
| **Experiment tracking (MLflow, W&B, DVC)** | "Track all model runs for reproducibility." | Overkill for 3 models with one Optuna sweep each. Adds infrastructure dependencies (MLflow server, database, artifact store) that complicate the Nix-based reproducibility story. The audit will be run once, not iterated on weekly. | Export all metrics as JSON to `data/exports/`. Pin random seeds. Save processed data as parquet. This is sufficient reproducibility for a one-shot audit. |
| **Real-time dashboard / interactive Streamlit app** | "Make the results explorable in a dashboard." | The downstream consumer is a Next.js scrollytelling site, not a dashboard. Building a Streamlit app duplicates visualization effort and creates a second source of truth. The scrollytelling format is journalistic, not analytical -- it guides the reader through a narrative, not an open-ended exploration. | Export clean JSON with documented schemas. The Next.js site IS the interactive product. If an analytical dashboard is needed later, it is built FROM the same JSON exports, not as a separate pipeline. |
| **Individual-level privacy-preserving output** | "Export individual student records for the site." | ENAHO microdata has individual survey responses. Even though ENAHO is publicly available, exporting individual-level records with SHAP values to a public website creates reidentification risk, especially for small indigenous groups in specific districts. | Export only aggregate statistics and the 10 curated representative profiles. Profiles should be composite/representative, not actual individuals. All exports are at the group or district level. |
| **Geospatial processing (geopandas, GDAL)** | "Generate choropleth maps directly from the audit pipeline." | Adds heavy geospatial dependencies (GDAL is notoriously difficult to install). The audit produces DATA for choropleths, not the maps themselves. Map rendering is the Next.js site's job using Mapbox/D3. | Export `choropleth.json` with ubigeo, lat/lon, and metric values. The site renders the map. Keep geospatial processing out of the Python pipeline entirely. |
| **Automated Spanish translation** | "Use an LLM to translate all findings to Spanish." | Machine translation of statistical findings produces unnatural, sometimes misleading text. "Tasa de falsos negativos" is correct but stilted; a Peruvian journalist would say "porcentaje de desertores que el modelo no detecta." The Spanish must be natural for the Peruvian audience. | Write Spanish findings manually (or with LLM assistance reviewed by a Spanish speaker). The 5-7 findings and feature labels are a small, high-stakes corpus where quality matters more than automation. |

---

## Feature Dependencies

```
[Survey-Weighted Metrics]
    |
    +--requires--> [ENAHO Data Pipeline with FACTOR07]
    |
    +--feeds into--> [Subgroup Fairness Metrics]
    |                     |
    |                     +--requires--> [Trained Models (LogReg, LightGBM, XGB)]
    |                     |
    |                     +--extends to--> [Intersectional Analysis]
    |                     |                     |
    |                     |                     +--requires--> [Composite Group Columns]
    |                     |
    |                     +--extends to--> [Calibration by Group]
    |                     |                     |
    |                     |                     +--requires--> [Calibrated Model]
    |                     |
    |                     +--feeds into--> [Threshold Sensitivity Analysis]
    |
    +--feeds into--> [SHAP Analysis]
                          |
                          +--requires--> [Trained LightGBM Model]
                          |
                          +--extends to--> [Regional SHAP Decomposition]
                          |                     |
                          |                     +--requires--> [region_natural Feature]
                          |
                          +--extends to--> [SHAP Interaction Effects]
                          |
                          +--extends to--> [10 Representative Profiles]

[P300A Harmonization]
    |
    +--feeds into--> [Subgroup Fairness Metrics (language dimension)]
    +--feeds into--> [Survey-Weighted Descriptive Gaps]
    +--feeds into--> [Intersectional Analysis (language x *)]

[Admin District Data]
    |
    +--feeds into--> [Cross-Validation Against Admin Data]
    |                     |
    |                     +--requires--> [Trained Model Predictions]
    |                     +--requires--> [UBIGEO Merge Infrastructure]
    |
    +--feeds into--> [Choropleth Export]

[All Fairness Metrics + SHAP + Cross-Validation]
    |
    +--feeds into--> [Findings Distillation]
                          |
                          +--produces--> [findings.json (bilingual)]
                          +--produces--> [All Export JSONs]
                          +--produces--> [ONNX Export]
```

### Dependency Notes

- **Subgroup Fairness requires Trained Models:** Cannot compute FNR/FPR/calibration without predictions. Models must be trained and threshold-tuned before fairness analysis begins.
- **Intersectional Analysis requires Subgroup Analysis:** Intersections are extensions of single-axis analysis. The code infrastructure (MetricFrame with sample_params) is the same; the difference is the composite group column.
- **SHAP requires LightGBM specifically:** TreeExplainer only works with tree-based models. LogReg coefficients provide a simpler interpretability baseline, but SHAP interaction values are only available through TreeExplainer.
- **Cross-Validation requires both Model Predictions and Admin Data:** This is a multi-source validation step that cannot begin until both the model pipeline and the admin data pipeline are complete.
- **Findings Distillation requires ALL upstream analyses:** This is the final synthesis step. It reads all previously generated JSON exports and produces the media-ready `findings.json`.
- **P300A Harmonization is upstream of everything language-related:** Must be implemented in the data pipeline before any language-disaggregated analysis. Both harmonized and original columns must coexist.
- **Calibration by Group requires Calibrated Model:** The CalibratedClassifierCV step must complete before group-level calibration analysis. Calibrate on validation set, then analyze calibration per group on test set.

---

## MVP Definition

### Launch With (v1 -- The Audit)

The audit is not a product with iterative versions. It is a one-shot research artifact. "MVP" here means: the minimum set of analyses that constitute a credible, publishable equity audit.

- [ ] **Survey-weighted descriptive gaps (6 dimensions)** -- This alone is newsworthy: "Awajun dropout rate 63% higher than Castellano speakers." No model needed.
- [ ] **Temporal-split LightGBM + LogReg baseline** -- The model exists to be audited. LogReg provides interpretable baseline; LightGBM matches Alerta Escuela's algorithm.
- [ ] **Subgroup fairness metrics (FNR, FPR, TPR, precision, PR-AUC) for all 6 dimensions** -- The core audit output. FNR disparity is the headline finding.
- [ ] **SHAP global importance + ES_PERUANO / ES_MUJER specific analysis** -- Answers: "How much does nationality independently contribute to predicted risk?"
- [ ] **Calibration by group** -- Answers: "Does 'high risk' mean the same thing for every group?"
- [ ] **3 intersectional analyses** -- Compounding disadvantage evidence.
- [ ] **findings.json with 5-7 bilingual findings** -- Media-ready output.
- [ ] **All 7 export JSONs matching schema contracts** -- Downstream site cannot function without these.

### Add After Validation (v1.x -- Strengthening Evidence)

Features that strengthen the audit but can be added after the core findings are reviewed.

- [ ] **Cross-validation against admin district data** -- Strengthens causal interpretation but requires admin data pipeline to be complete. If admin data is unavailable or unreliable, the audit still stands on survey data alone.
- [ ] **Regional SHAP decomposition (Costa/Sierra/Selva)** -- Adds geographic nuance but is not required for the core fairness claims.
- [ ] **SHAP interaction effects** -- Computationally expensive, analytically sophisticated. Adds "compound disadvantage" evidence but the intersectional MetricFrame analysis already captures group-level compounding.
- [ ] **10 representative student profiles** -- High media impact but depends on careful profile selection. Can be curated after initial findings are validated.
- [ ] **Choropleth-ready spatial export** -- High visualization impact but requires Census 2017 merge for indigenous language prevalence data.
- [ ] **ONNX export for browser inference** -- Only needed when the Next.js site is ready for the interactive demo component.

### Future Consideration (v2+ -- If Audit Leads to Ongoing Monitoring)

Features to defer unless the audit evolves into a recurring monitoring program.

- [ ] **Automated pipeline for new ENAHO years** -- If INEI releases 2025 data, the pipeline should re-run. But automating this is premature for a one-shot audit.
- [ ] **Longitudinal fairness trend analysis** -- Track whether fairness gaps widen or narrow over time. Requires 2025+ data.
- [ ] **Comparison with other countries' dropout prediction audits** -- Context for the findings, but not required for the Peru-specific audit.
- [ ] **Fairness-constrained model training** -- If MINEDU engages, provide debiased model alternatives. But this is a different project with different goals.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Survey-weighted descriptive gaps | HIGH | LOW | P1 |
| Subgroup fairness metrics (6 dimensions) | HIGH | MEDIUM | P1 |
| FNR disparity by group (foregrounded) | HIGH | LOW | P1 |
| SHAP global + ES_PERUANO/ES_MUJER | HIGH | MEDIUM | P1 |
| Calibration by group | HIGH | MEDIUM | P1 |
| Intersectional analysis (3 intersections) | HIGH | HIGH | P1 |
| Bilingual findings export | HIGH | LOW | P1 |
| JSON export contracts (7 files) | HIGH | MEDIUM | P1 |
| Threshold sensitivity analysis | MEDIUM | LOW | P1 |
| Reproducibility (seeds, versions, parquet) | MEDIUM | LOW | P1 |
| PR-AUC as primary metric | MEDIUM | LOW | P1 |
| P300A harmonization (cross-year) | HIGH | MEDIUM | P1 |
| Cross-validation with admin data | HIGH | HIGH | P2 |
| Regional SHAP decomposition | MEDIUM | MEDIUM | P2 |
| SHAP interaction effects | MEDIUM | HIGH | P2 |
| 10 representative student profiles | HIGH | MEDIUM | P2 |
| Choropleth spatial export | HIGH | MEDIUM | P2 |
| ONNX model export | MEDIUM | MEDIUM | P2 |
| Automated pipeline for new years | LOW | MEDIUM | P3 |
| Longitudinal trend analysis | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for a credible audit. Omission = methodological gap.
- P2: Should have. Strengthens evidence and media impact. Add in parallel or immediately after P1.
- P3: Nice to have. Defer unless the project scope expands.

---

## Competitor / Reference Feature Analysis

Other ML fairness audits and what they include, informing what is expected in this domain.

| Feature | NIST AI RMF (2023) | Gender Shades (Buolamwini 2018) | ProPublica COMPAS (2016) | UK ICO Audit Framework | Our Approach |
|---------|-----|-----|-----|-----|-----|
| Subgroup metrics | Required | Accuracy by skin type x gender | FPR/FNR by race | Required by dimension | All 6 dimensions + 3 intersections |
| Intersectional analysis | Mentioned | Core methodology (skin type x gender) | Not done (single-axis race) | Recommended | 3 mandatory intersections |
| Calibration by group | Mentioned | Not done | Calibration was central finding | Required | Per-group calibration at high-risk threshold |
| SHAP / explanations | Recommended | Not applicable (black-box CV) | Not done (logistic regression, used coefficients) | Required | TreeExplainer global + regional + interaction |
| Survey weights | N/A | N/A | N/A | N/A | Required (ENAHO FACTOR07) -- unique to this audit |
| Cross-validation with external data | Recommended | Used ground truth labels | Compared with recidivism data | Recommended | Admin district dropout rates |
| Media-ready outputs | N/A | Yes (visualizations drove media coverage) | Yes (interactive article) | N/A | Bilingual JSON for scrollytelling site |
| Temporal validation | Mentioned | Cross-dataset | Not done (single dataset) | Recommended | Strict temporal split 2018-2022/2023/2024 |
| Reproducibility | Required | Code published | Methodology published | Required | Seeds, versions, parquet, JSON, git |

**Key takeaway:** Our audit is more comprehensive than any single reference in the table. The combination of survey-weighted metrics, intersectional analysis, SHAP explanations, cross-validation with admin data, AND media-ready bilingual exports is novel. The closest comparable is ProPublica's COMPAS analysis (for methodology) combined with Gender Shades (for media impact), but neither handled survey weights or produced structured data exports for a downstream visualization product.

---

## Domain-Specific Feature Notes

### Survey Weight Handling (FACTOR07)

**Confidence: HIGH** (well-established survey methodology, sklearn/fairlearn API is stable)

This is the single most important methodological discipline in the entire audit. ENAHO's expansion factors (FACTOR07) convert sample observations into population estimates. Without them, every finding describes ~25,000 surveyed households, not Peru's ~8 million school-age children.

Implementation pattern:
- Every call to `sklearn.metrics.*` must include `sample_weight=factor07`
- Every call to `fairlearn.metrics.MetricFrame` must include `sample_params={'metric_name': {'sample_weight': weights}}` for each metric in the metrics dict
- Descriptive statistics in polars: `(pl.col('dropout') * pl.col('factor07')).sum() / pl.col('factor07').sum()` for weighted means
- For weighted quantiles (poverty quintile construction), use `statsmodels` weighted quantile functions or manual implementation
- Always compute BOTH weighted and unweighted versions, assert they differ, report both

### Fairlearn MetricFrame Usage Pattern

**Confidence: MEDIUM** (API was stable through May 2025 training data; exact parameter names should be verified against current fairlearn docs)

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score

metrics = {
    'recall': recall_score,
    'precision': precision_score,
}

# sample_params maps metric name -> dict of extra kwargs
sample_params = {
    'recall': {'sample_weight': weights},
    'precision': {'sample_weight': weights},
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_df,  # Can be DataFrame with multiple columns for intersections
    sample_params=sample_params,
)

# Key outputs:
mf.by_group           # DataFrame: metric values per group
mf.difference()       # Series: max gap per metric
mf.ratio()            # Series: min ratio per metric
mf.group_min()        # Series: worst-performing group per metric
mf.group_max()        # Series: best-performing group per metric
```

**NOTE:** Verify that `sample_params` syntax is current. Older fairlearn versions used a different parameter passing mechanism. The spec's example code uses this pattern and should be treated as authoritative.

### SHAP Best Practices for Fairness Audits

**Confidence: MEDIUM** (SHAP API stable but version-specific behaviors exist)

1. **Compute on test set, not train.** SHAP on training data shows what the model learned from; SHAP on test data shows what drives predictions on unseen data. For an audit, test-set SHAP is the correct choice.
2. **Binary classification gotcha:** LightGBM's `TreeExplainer` may return a list of two arrays (one per class) or a single array depending on the model configuration and SHAP version. Always check: `if isinstance(shap_values, list): shap_values = shap_values[1]` to get positive-class contributions.
3. **Interaction values are expensive:** `explainer.shap_interaction_values(X)` returns an (n, p, p) array. For n=5000 and p=19, this is manageable. But subsample to ~1000 rows if the test set is larger.
4. **Regional decomposition:** Run TreeExplainer on the SAME model but DIFFERENT subsets of the test data (filtered by region). This shows how the model's behavior varies across regions, not how different models behave.
5. **Feature name consistency:** SHAP uses the feature names from the input DataFrame. If training used different column names than the test set, SHAP plots will be mislabeled. Use a single `MODEL_FEATURES` list throughout.

### Calibration Analysis Pattern

**Confidence: HIGH** (standard sklearn API)

```python
from sklearn.calibration import calibration_curve

# Per-group calibration
for group_name, group_mask in groups.items():
    prob_true, prob_pred = calibration_curve(
        y_true[group_mask],
        y_pred_proba[group_mask],
        n_bins=10,
        strategy='quantile',  # Better for imbalanced data
        sample_weight=weights[group_mask],  # NOTE: verify sample_weight support in calibration_curve
    )
    # Plot reliability diagram per group
```

**WARNING:** `sklearn.calibration.calibration_curve` may NOT support `sample_weight` in all versions. If not, implement weighted calibration manually: bin predictions, compute weighted mean actual rate per bin. Verify against current sklearn docs.

### Media-Ready Export Patterns

**Confidence: HIGH** (JSON schema design is framework-agnostic)

Key principles for exports consumed by a Next.js scrollytelling site:
1. **Flat where possible:** Nested JSON is harder to consume in JavaScript. Flatten group metrics into key-value pairs.
2. **Include metadata:** `generated_at` timestamp, `model` name, `threshold` used. The site needs to display provenance.
3. **Spanish labels alongside English keys:** `feature_names` for code, `feature_labels_es` for display. Never make the frontend do translation.
4. **Self-documenting schemas:** `data/exports/README.md` documents every file, every field, and the code that generated it.
5. **Committed to git:** Export files are committed (unlike raw data). They are the contract between the audit and the site. Schema changes require coordination.

---

## Sources

- Project specification (`specs.md`) -- PRIMARY source for feature requirements, metric definitions, and export contracts [HIGH confidence]
- Fairlearn documentation (training data knowledge through May 2025) -- MetricFrame API, sample_params, sensitive_features [MEDIUM confidence; recommend verifying current API against fairlearn.org docs]
- SHAP documentation (training data knowledge through May 2025) -- TreeExplainer, interaction values, binary classification handling [MEDIUM confidence]
- sklearn documentation (training data knowledge through May 2025) -- calibration_curve, average_precision_score, sample_weight support [MEDIUM confidence]
- Buolamwini & Gebru (2018), "Gender Shades" -- Intersectional fairness methodology reference [HIGH confidence; seminal paper]
- ProPublica COMPAS analysis (2016) -- FPR/FNR disparity methodology reference [HIGH confidence; seminal work]
- NIST AI Risk Management Framework (2023) -- Federal audit requirements reference [MEDIUM confidence]
- WebSearch/WebFetch unavailable for verification of current library versions and community best practices [flagged as gap]

**Known gaps:**
- Could not verify current fairlearn `sample_params` API against live docs
- Could not verify sklearn `calibration_curve` sample_weight support in latest version
- Could not search for recent (2025-2026) ML fairness audit methodologies or new tools
- Could not verify SHAP version-specific behavior for LightGBM binary classification output format

---
*Feature research for: ML Fairness Audit of Education Dropout Prediction (Peru / Alerta Escuela)*
*Researched: 2026-02-07*
