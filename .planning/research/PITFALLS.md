# Pitfalls Research

**Domain:** ML fairness audit of ENAHO survey data (Peru education equity)
**Researched:** 2026-02-07
**Confidence:** HIGH (most pitfalls verified via official docs, GitHub issues, or academic literature)

---

## Critical Pitfalls

Mistakes that cause incorrect results, invalidate the audit, or require full pipeline rewrites.

---

### Pitfall 1: ENAHO Delimiter Mismatch Silently Loads Garbage Data

**What goes wrong:**
INEI changed the ENAHO CSV delimiter from pipe `|` (2018--2019) to comma `,` (2020--2024). If you use the wrong delimiter, polars/pandas will read the entire row as a single column. The loader will not crash -- it will return a DataFrame with 1 column and thousands of rows, or parse quoted fields incorrectly, producing NaN-filled data that passes basic shape checks but contains nonsense values.

**Why it happens:**
Polars `read_csv` defaults to comma delimiter. If you forget to set `separator='|'` for 2018--2019 files, it reads the pipe-delimited data as a single wide column. The error is silent because CSV parsers are lenient.

**How to avoid:**
- Hardcode delimiter selection: `separator = '|' if year <= 2019 else ','`
- Do NOT rely on CSV auto-detection (polars does not auto-detect delimiters reliably for pipe-delimited files)
- After every load, assert the DataFrame has at least 20 columns. A single-column DataFrame is the telltale sign of wrong delimiter
- Assert that key columns (`P300A`, `P303`, `P306`, `FACTOR07`) exist and are non-null

**Warning signs:**
- DataFrame has 1 column or far fewer columns than expected
- Column names contain pipe characters
- All values in P300A are null after load
- Row count is correct but column count is wrong

**Phase to address:** Phase 1 (ENAHO Single-Year Loader)

**Confidence:** HIGH -- this is documented in the project spec (Section 4.1) and confirmed by the [PeruData/ENAHO](https://github.com/PeruData/ENAHO) processing scripts.

---

### Pitfall 2: UBIGEO Leading Zero Loss Creates Silent Merge Failures

**What goes wrong:**
Peru's UBIGEO codes are 6-digit strings where the first two digits identify the department. Departments 01--09 (Amazonas through Huanuco) have codes starting with "0". When UBIGEO is read as an integer (common default), leading zeros are stripped: `"010101"` becomes `10101`. This causes all merge operations with admin/census/nightlights data to silently fail for 9 out of 25 departments -- roughly 36% of districts. The merge returns null for those rows, and because it is a left join, no error is raised.

**Why it happens:**
- CSV parsers default to numeric inference for digit-only columns
- Polars infers UBIGEO as Int64, pandas as int64
- Even `str(ubigeo)` does not restore the leading zero once it is parsed as an integer
- The problem is invisible unless you specifically check UBIGEO string lengths

**How to avoid:**
- Force UBIGEO to string type immediately at load time: `pl.col('UBIGEO').cast(pl.Utf8).str.zfill(6)`
- Create a `pad_ubigeo()` utility in `src/utils.py` and call it in every loader (ENAHO, admin, census, nightlights)
- After every merge, assert: `assert df['ubigeo'].str.len_chars().min() == 6`
- After every spatial merge, assert merge rate > 0.85 and check that departments 01--09 have non-null enrichment data

**Warning signs:**
- Merge coverage drops below 85%
- Departments like Amazonas, Ancash, Apurimac have null admin/census data
- UBIGEO values have 5 characters instead of 6
- District count after merge is suspiciously lower than expected (~1890)

**Phase to address:** Phase 1 (utils.py), validated in every subsequent merge phase (Phase 3, Phase 4)

**Confidence:** HIGH -- documented in spec Section 4.2, confirmed by [UBIGEO Peru](https://github.com/ernestorivero/Ubigeo-Peru) repository and INEI coding standards.

---

### Pitfall 3: P300A Harmonization Failure Creates Fake Demographic Shifts

**What goes wrong:**
In 2020, INEI disaggregated mother tongue code 3 ("Otra lengua nativa") into six specific indigenous languages (codes 10--15: Ashaninka, Awajun, Shipibo-Konibo, Shawi, Matsigenka, Achuar). If you analyze P300A without harmonization, code 3 appears to drop precipitously in 2020, and six "new" language groups appear. This looks like a real demographic shift but is purely a coding change. Any cross-year trend analysis of indigenous language groups will be wrong.

**Why it happens:**
- The coding change is not documented in the CSV headers or column metadata
- Pre-2020 files do not contain codes 10--15, so a naive analysis sees them as "new populations"
- If you build language dummies from raw P300A, you get different feature sets pre/post 2020, which breaks model training

**How to avoid:**
- Implement `harmonize_p300a()` exactly as specified in the project spec Section 4.1
- Always create BOTH `p300a_harmonized` (codes 10--15 collapsed to 3) and `p300a_original`
- Use `p300a_harmonized` for all cross-year analyses and model features
- Use `p300a_original` only for 2020+ disaggregated deep-dive analyses
- Validate: the sum of codes 3+10+11+12+13+14+15 should be stable across years (within 30% of each other)

**Warning signs:**
- Code 3 count drops >80% from 2019 to 2020
- Codes 10--15 appear only in 2020+ data
- Feature matrix has different column counts for different years
- Language dummy variables have different names across years

**Phase to address:** Phase 2 (Multi-Year Loader + Harmonization)

**Confidence:** HIGH -- documented in spec, confirmed by INEI's own methodological notes on the 2020 ENAHO revision.

---

### Pitfall 4: Survey Weights Omitted from Metrics -- All Reported Numbers Are Wrong

**What goes wrong:**
ENAHO uses complex stratified sampling with expansion factors (FACTOR07). Unweighted statistics describe the sample, not Peru's population. For example, ENAHO oversamples rural areas, so unweighted dropout rates will be biased upward vs. the true national rate. If survey weights are omitted from any metric -- descriptive stats, model evaluation, fairness metrics -- the results do not generalize to Peru's student population and the entire audit is methodologically invalid.

**Why it happens:**
- sklearn metrics default to `sample_weight=None` -- easy to forget
- fairlearn MetricFrame requires explicit `sample_params` configuration per metric -- it has no global default (see Pitfall 5 for the specific API gotcha)
- Polars has no built-in weighted aggregation function, requiring manual `(col * weight).sum() / weight.sum()` patterns
- Weighted and unweighted results are often "close enough" that the omission is not obvious without explicit comparison

**How to avoid:**
- Every metric function signature MUST include a `sample_weight` parameter
- After computing any metric, also compute the unweighted version and assert they differ: `assert abs(weighted_f1 - unweighted_f1) > 0.001`
- In polars, use the pattern: `(pl.col('dropout') * pl.col('factor07')).sum() / pl.col('factor07').sum()`
- In sklearn: always pass `sample_weight=weights` to every call of `f1_score`, `precision_score`, `recall_score`, `average_precision_score`, `brier_score_loss`
- Create a wrapper function `evaluate_model()` that enforces weight passing

**Warning signs:**
- Weighted and unweighted metrics are identical (weights are not being applied)
- National dropout rate does not match expected ~13--15% weighted range
- Rural dropout rate is not higher than urban (weights not correcting sampling bias)
- Any metric function call without `sample_weight=` in the source code

**Phase to address:** Phase 4 (Feature Engineering), enforced in every subsequent phase

**Confidence:** HIGH -- verified via [PLOS ONE study on survey weights in ML](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0280387) which demonstrates 5 percentage point F1 overestimation when weights are ignored.

---

### Pitfall 5: Fairlearn MetricFrame sample_params API Silently Drops Weights

**What goes wrong:**
When passing multiple metrics to MetricFrame as a dictionary, `sample_params` must be a nested dictionary where the outer keys match metric names exactly. There is no "global" sample_params concept. If you pass a flat dictionary (as you would for a single metric), fairlearn will silently ignore the weights for some or all metrics, producing unweighted results without any error or warning.

**Why it happens:**
- The API behaves differently for single-metric vs. multi-metric usage
- For a single metric (callable), `sample_params` is a flat dict: `{'sample_weight': w}`
- For multiple metrics (dict of callables), `sample_params` must be nested: `{'metric_name': {'sample_weight': w}}`
- If you use the flat dict format with multiple metrics, fairlearn does not raise an error -- it just ignores the weights
- Metrics not listed in the nested `sample_params` receive no per-sample arguments at all

**How to avoid:**
- Always use the nested dictionary format, even for a single metric:
  ```python
  sample_params = {
      'recall': {'sample_weight': weights},
      'precision': {'sample_weight': weights},
      'pr_auc': {'sample_weight': weights}
  }
  ```
- After computing MetricFrame, verify that `mf.overall` values differ from a manually computed unweighted version
- Write a helper function that auto-generates the nested `sample_params` dict from a metric dict and a single weight array
- Unit test: compute one metric both via MetricFrame and manually with `sample_weight`; assert they match

**Warning signs:**
- MetricFrame overall metrics are identical to unweighted sklearn calls
- `mf.by_group` values look suspiciously like unweighted per-group metrics
- No error/warning but weights are not being applied

**Phase to address:** Phase 8 (Subgroup Fairness Metrics)

**Confidence:** HIGH -- verified via [Fairlearn 0.14 Advanced MetricFrame documentation](https://fairlearn.org/main/user_guide/assessment/advanced_metricframe.html) which explicitly states "there is no concept of a 'global' sample parameter."

---

### Pitfall 6: SHAP TreeExplainer Output Shape Varies by Model Type and SHAP Version

**What goes wrong:**
SHAP TreeExplainer returns different output shapes for LightGBM binary classification depending on the SHAP library version and how you call it:

- **SHAP < 0.45**: `shap_values()` returns a **list of two arrays** `[neg_class_shap, pos_class_shap]`, each of shape `(n_samples, n_features)`. You need `shap_values[1]` for the positive class.
- **SHAP >= 0.45**: `shap_values()` returns an **ndarray** of shape `(n_samples, n_features)` for LightGBM (because LightGBM outputs a single raw margin/log-odds). There is no `[1]` index needed.
- **Scikit-learn classifiers** in all versions: return shape `(n_samples, n_features, 2)` or a list of length 2.

If you write code assuming the list format but use SHAP >= 0.45, you will index into the features dimension instead of the class dimension, producing complete garbage. If you write code assuming the ndarray format but use SHAP < 0.45, you will get a list and numpy operations will fail.

**Why it happens:**
- SHAP v0.45.0 (March 2024) explicitly changed the return type: "Changed type and shape of returned SHAP values in some cases, to be consistent with model outputs. SHAP values for models with multiple outputs are now np.ndarray rather than list."
- LightGBM's native API outputs a single log-odds value for binary classification, so SHAP >= 0.45 returns a 2D array. But LGBMClassifier (sklearn API) may behave differently.
- The spec says `shap >= 0.45` is required, but the spec code example still uses the old `shap_values[1]` pattern

**How to avoid:**
- Pin SHAP version in pyproject.toml and document expected output shape
- Write defensive code that handles both formats:
  ```python
  sv = explainer.shap_values(X_test)
  if isinstance(sv, list):
      shap_vals = sv[1]  # Old format: take positive class
  else:
      shap_vals = sv      # New format: already single output
  ```
- After computing SHAP values, assert shape: `assert shap_vals.shape == (n_samples, n_features)`
- Prefer using `explainer(X_test)` (the `__call__` method) which returns an `Explanation` object with more consistent behavior
- Verify `expected_value` is a scalar (LightGBM) not an array (sklearn classifiers)

**Warning signs:**
- SHAP summary plot shows only 2 features when you expect 19+
- SHAP values have wrong shape `(n_samples, 2)` instead of `(n_samples, n_features)`
- TypeError when indexing SHAP output
- Mean absolute SHAP values do not sum approximately to mean prediction deviation

**Phase to address:** Phase 9 (SHAP Analysis)

**Confidence:** HIGH -- verified via [SHAP release notes](https://shap.readthedocs.io/en/latest/release_notes.html) and [SHAP TreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) confirming the v0.45 breaking change.

---

### Pitfall 7: Temporal Data Leakage -- Using 2024 Test Data Before Phase 7

**What goes wrong:**
The spec mandates that 2024 data (test set) is touched exactly once, during Phase 7 (final evaluation). If any code before Phase 7 loads 2024 data for feature engineering, threshold tuning, hyperparameter optimization, descriptive statistics, or even casual EDA, the test set is contaminated. Results on the test set will be optimistically biased, and the audit's credibility is destroyed.

The subtler form: computing weighted income quintile breaks or scaling parameters on the full dataset (including 2024) and then applying them to the "train" split. This leaks distributional information from the future.

**Why it happens:**
- `load_all_years()` loads 2018--2024 by default, making it easy to accidentally include 2024 in preprocessing
- Feature engineering (income quintile breaks, StandardScaler fit) on the full dataset before splitting is a common ML pattern that constitutes leakage in temporal settings
- Threshold tuning on test data feels "harmless" but directly optimizes for the held-out set
- Even descriptive statistics on 2024 can unconsciously influence modeling decisions

**How to avoid:**
- `create_temporal_splits()` MUST be called before any model-related computation
- After splitting, assert: `assert X_train['year'].max() == 2022` and `assert X_val['year'].unique() == [2023]`
- Fit all preprocessing (StandardScaler, income quintile breaks) on train set ONLY, then transform val/test
- Add a CI/test assertion: grep all code files for references to 2024 before Phase 7; flag any occurrence
- In `load_all_years()`, add an optional parameter `exclude_test=True` that excludes 2024 by default

**Warning signs:**
- Test set PR-AUC is higher than validation set PR-AUC (suspicious -- suggests overfitting to test)
- Any code before Phase 7 that references year 2024
- StandardScaler or income quintile computation using the full dataset
- Unusually high model performance ("too good to be true")

**Phase to address:** Phase 5 (Temporal Splits), with assertions in every subsequent phase

**Confidence:** HIGH -- standard ML best practice, reinforced by [IBM's data leakage documentation](https://www.ibm.com/think/topics/data-leakage-machine-learning) and confirmed as explicit spec requirement (Section 14, Rule 10).

---

### Pitfall 8: ONNX Float32 Accumulation Error Grows with Tree Count

**What goes wrong:**
LightGBM uses double precision (float64) internally, but the ONNX TreeEnsembleClassifier operator uses float32 by default. Each tree's prediction is accumulated in float32, and the rounding error compounds across trees. With 500 trees (the spec's `n_estimators`), the max absolute prediction difference between LightGBM and ONNX can exceed the spec's `1e-5` tolerance, causing the validation assertion to fail.

**Why it happens:**
- ONNX's TreeEnsembleRegressor/Classifier nodes default to float32 computation
- LightGBM stores tree thresholds and leaf values in float64
- Prediction = sum of 500 tree outputs; each summation step introduces float32 rounding
- The error is proportional to `O(n_trees * epsilon_float32)` where `epsilon_float32 ~ 1.19e-7`
- At 500 trees: worst case ~60 * 1e-7 = 6e-6, but empirically can be larger depending on tree depth and value ranges

**How to avoid:**
- Use `onnxmltools.convert_lightgbm()` with the `zipmap=False` option to avoid unnecessary post-processing
- Convert input features to float32 BEFORE calling LightGBM's predict (to match what ONNX will receive): `X_test.astype(np.float32)`
- Relax the tolerance if needed: `assert np.max(np.abs(onnx_preds - python_preds)) < 1e-4` (the spec says 1e-5, which may be too tight for 500 trees)
- Consider using `sklearn-onnx` with `options={LGBMClassifier: {'zipmap': False}}` for better conversion control
- Validate on 100+ samples across different prediction ranges (low probability, mid, high)
- If tolerance cannot be met, reduce `n_estimators` or use the `split` option in sklearn-onnx to partition tree summation into smaller groups

**Warning signs:**
- ONNX validation assertion fails intermittently (error near tolerance boundary)
- Max prediction difference is 1e-4 to 1e-3 range
- Differences are larger for extreme predictions (near 0 or 1 in probability space)
- ONNX output `sess.run(None, ...)[1][:, 1]` -- if the index is wrong, you get class labels instead of probabilities

**Phase to address:** Phase 7 (ONNX Export)

**Confidence:** HIGH -- verified via [sklearn-onnx LightGBM documentation](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm_reg.html) and [onnxmltools GitHub issues](https://github.com/onnx/onnxmltools/issues/150) documenting the float32 accumulation problem.

---

### Pitfall 9: Polars-to-Pandas Conversion Corrupts Data Types at sklearn/fairlearn Boundary

**What goes wrong:**
When calling `.to_pandas()` to pass data to sklearn/fairlearn/shap, several data type corruptions can occur silently:

1. **Nullable integers become float64**: Polars Int64 columns with any null values convert to pandas float64 (not Int64). If P300A has a single null, the entire column becomes float64, and categorical operations break.
2. **Categorical columns lose encoding**: Polars Categorical/Enum types may not convert cleanly to pandas Categorical, especially with string cache issues in polars >= 1.32.
3. **Column order is not guaranteed**: Polars DataFrames do not guarantee column order the same way pandas does. If you convert to pandas and then to numpy, the feature column order may not match what the model was trained on.
4. **String columns become object dtype**: UBIGEO and other string columns become pandas `object` dtype, which sklearn may reject or handle differently.

**Why it happens:**
- NumPy does not support nullable integers, so pandas falls back to float64 for nullable int columns
- Polars and pandas have fundamentally different type systems
- The conversion is a lossy operation that cannot preserve all polars semantics

**How to avoid:**
- Before `.to_pandas()`, fill nulls in integer columns: `df.with_columns(pl.col('P300A').fill_null(99))`
- Use `use_pyarrow_extension_array=True` in `.to_pandas()` to preserve nullable types (but verify sklearn compatibility)
- Explicitly select and order columns before conversion:
  ```python
  X_pd = df.select(MODEL_FEATURES).to_pandas()
  assert list(X_pd.columns) == MODEL_FEATURES  # Order check
  ```
- Convert to numpy explicitly where possible: `X_np = df.select(MODEL_FEATURES).to_numpy()` (avoids pandas entirely)
- Never pass polars DataFrames directly to sklearn (it may partially work in sklearn 1.5+ but is unreliable)

**Warning signs:**
- `dtype: float64` for columns that should be integer (P300A, P207, P303)
- Feature importance order does not match expected feature names (column reordering)
- sklearn warns about feature name mismatch between fit and predict
- `SettingWithCopyWarning` from pandas after conversion

**Phase to address:** Phase 1 (establish conversion pattern), enforced everywhere

**Confidence:** HIGH -- verified via [polars issue #8204](https://github.com/pola-rs/polars/issues/8204) (Int64-to-float64) and [scikit-learn issue #32167](https://github.com/scikit-learn/scikit-learn/issues/32167) (polars integration issues).

---

## Moderate Pitfalls

Mistakes that cause delays, incorrect subgroup analysis, or technical debt.

---

### Pitfall 10: Intersectional Analysis with Small Sample Sizes Produces Noise, Not Findings

**What goes wrong:**
When computing fairness metrics for intersectional groups (e.g., Quechua + Rural + Female), sample sizes can drop below statistical significance thresholds. A group with n=30 unweighted observations can produce wildly unstable metric estimates -- a single misclassification swings FNR by 3 percentage points. Reporting these as "findings" misleads journalists who will treat them as population-level facts.

**Why it happens:**
- Intersections grow combinatorially: 4 languages x 2 rurality x 2 sex x 5 poverty quintiles = 80 groups
- ENAHO samples ~25K school-age children per year; many intersections have <50 observations
- Disaggregated indigenous language codes (Awajun, Shawi, etc.) have very small samples even without intersections (~150 unweighted per code across all 2020+ years)
- Survey weights amplify the problem: a group with n=20 unweighted can represent n_weighted=50,000 with huge confidence intervals

**How to avoid:**
- Enforce minimum sample sizes: n >= 100 unweighted for primary dimensions, n >= 50 for intersections
- Flag (do not suppress) groups below threshold: report metrics but add explicit caveats
- Compute confidence intervals using bootstrap or survey-weighted standard errors
- For intersections, focus on the 3 specified pairs (language x rurality, sex x poverty, language x region) rather than all possible combinations
- For disaggregated indigenous codes, report but caveat that findings are exploratory due to sample size
- Never report intersectional findings without the unweighted sample size alongside

**Warning signs:**
- Any intersectional group with n < 50 unweighted observations
- Metric values of exactly 0.0 or 1.0 for a subgroup (usually means n is tiny)
- Wide variance in metrics across bootstrap replicates
- FNR or TPR differences that reverse direction when a few observations are added/removed

**Phase to address:** Phase 8 (Subgroup Fairness Metrics), Phase 9 (SHAP by region)

**Confidence:** HIGH -- confirmed by [academic literature on intersectional fairness sample sizes](https://arxiv.org/html/2312.04745) and the spec's own minimum sample thresholds (Section 7).

---

### Pitfall 11: Class Imbalance Handling Conflicts with Survey Weights

**What goes wrong:**
The project uses both `is_unbalance=True` in LightGBM AND survey weights (`FACTOR07`) as `sample_weight`. These two mechanisms can conflict: `is_unbalance` adjusts the initial score based on unweighted class proportions, while survey weights adjust the gradient updates. If the weighted class ratio differs from the unweighted ratio (which it will, because ENAHO oversamples rural areas), the model receives contradictory signals about the base rate.

Additionally, using `class_weight='balanced'` in LogisticRegression alongside survey weights double-counts the imbalance correction for weighted observations.

**Why it happens:**
- `is_unbalance=True` computes `scale_pos_weight` from the unweighted sample, ignoring survey weights
- Survey weights change the effective class ratio, but `is_unbalance` does not know about them
- The interaction between these two weighting mechanisms is poorly documented in LightGBM
- LightGBM issue #6807 shows different initialization paths for class weights vs. `is_unbalance`

**How to avoid:**
- Choose ONE imbalance strategy, not both:
  - **Option A (recommended):** Use survey weights as `sample_weight` and set `is_unbalance=False`. Let the survey weights handle both sampling design correction AND class imbalance.
  - **Option B:** Compute `scale_pos_weight` from the WEIGHTED class ratio (not unweighted), and do NOT pass survey weights during training. Pass weights only during evaluation.
- For LogisticRegression: use `class_weight=None` if passing survey weights as `sample_weight`, OR use `class_weight='balanced'` without survey weights in training
- Always evaluate with survey weights regardless of training strategy
- Document which strategy was chosen and why

**Warning signs:**
- Predicted probabilities are miscalibrated (mean predicted probability differs substantially from weighted base rate)
- Model is overconfident on minority class (predicting >0.8 for many dropout students)
- Brier score does not improve with calibration (underlying probability estimates are systematically biased)

**Phase to address:** Phase 5 (Baseline Model), Phase 6 (LightGBM/XGBoost)

**Confidence:** MEDIUM -- the interaction between survey weights and `is_unbalance` is not well-documented. [LightGBM issue #6807](https://github.com/microsoft/LightGBM/issues/6807) provides some insight but no definitive guidance. The recommendation is based on sound statistical reasoning.

---

### Pitfall 12: SHAP Computed on Train Set Instead of Test Set

**What goes wrong:**
If SHAP values are computed on the training set, the explanations reflect how the model memorizes training patterns, not how it generalizes. Feature importance will be inflated for overfit features, and the fairness implications will be overstated for features the model has "seen."

**Why it happens:**
- The training DataFrame is readily available in the same notebook/script
- Training set is larger (5 years) and produces smoother SHAP beeswarm plots
- No runtime error occurs -- SHAP will happily explain predictions on any data
- Interaction values are expensive (`O(n^2)`), tempting developers to use the already-loaded train set

**How to avoid:**
- Assert year == 2024 on the DataFrame passed to SHAP: `assert X_shap['year'].unique() == [2024]`
- Compute SHAP only after Phase 7 (final test evaluation), using the same test set
- For interaction values, subsample the test set to 1000 rows, not the train set
- In the SHAP export JSON, include `"computed_on": "test_2024"` as metadata

**Warning signs:**
- SHAP input has >10,000 rows (test set should be ~25K for one year, train is ~125K)
- SHAP computation takes >10 minutes (test set is 5x smaller and should be faster)
- Year column in SHAP DataFrame contains values other than 2024

**Phase to address:** Phase 9 (SHAP Analysis)

**Confidence:** HIGH -- documented in spec Section 8 and Section 14 Rule 10, and standard ML best practice.

---

### Pitfall 13: ONNX Binary Classification Output Index Confusion

**What goes wrong:**
When running inference with ONNX Runtime on a converted LightGBM classifier, `sess.run(None, input_feed)` returns a list of two outputs: `[labels, probabilities]`. The probabilities output (`[1]`) is a 2D array of shape `(n_samples, 2)` where column 0 is P(class=0) and column 1 is P(class=1). If you take `[1]` (thinking it is the positive class probability) you get the entire probability array, not P(dropout=1). You need `[1][:, 1]`.

Additionally, the `[0]` output contains predicted labels, not probabilities. Confusing `[0]` and `[1]` gives you integer labels (0/1) instead of continuous probabilities, which will produce nonsensical comparisons with `predict_proba`.

**Why it happens:**
- ONNX classification models return two outputs by default: labels and probabilities
- The indexing is `output[output_index][sample_index, class_index]` -- three levels of indexing
- Different ONNX converter versions may change the output structure (some include ZipMap post-processing, some do not)
- The spec code (`onnx_preds = sess.run(None, ...)[1][:, 1]`) is correct but easy to mistype

**How to avoid:**
- Always use the exact indexing: `sess.run(None, feed)[1][:, 1]` for positive class probabilities
- Assert output shape: `assert onnx_output[1].shape == (n_samples, 2)`
- Assert probability range: `assert 0 <= onnx_output[1][:, 1].min()` and `assert onnx_output[1][:, 1].max() <= 1`
- Convert with `zipmap=False` in sklearn-onnx to get raw probabilities without ZipMap wrapping (ZipMap returns list of dicts instead of arrays)
- Name your ONNX session output variables and check them: `sess.get_outputs()[0].name`

**Warning signs:**
- ONNX predictions are all 0 or 1 (you are reading labels, not probabilities)
- ONNX predictions are all in a narrow range (you are reading P(class=0) instead of P(class=1))
- Validation assertion fails because you are comparing labels vs. probabilities
- TypeError: cannot compare integer labels with float probabilities

**Phase to address:** Phase 7 (ONNX Export)

**Confidence:** HIGH -- verified via [onnxruntime issue #12629](https://github.com/microsoft/onnxruntime/issues/12629) documenting the output shape confusion across environments.

---

### Pitfall 14: Cartesian Product from UBIGEO Joins Multiplies Rows Silently

**What goes wrong:**
If the admin/census/nightlights data has duplicate UBIGEO entries (e.g., separate rows for primaria and secundaria in admin data, or multiple years of nightlights), a left join on UBIGEO produces a Cartesian product. A dataset with 180K rows can silently balloon to 360K rows (one admin dataset) or 1.26M rows (both admin levels x 7 nightlights years). The downstream model trains on duplicated observations, producing overconfident results.

**Why it happens:**
- Admin data has separate rows for primaria and secundaria per district -- joining without filtering creates 2x multiplication
- Nightlights data may have multiple years per district
- Census data should be unique per UBIGEO but data quality issues may create duplicates
- Polars/pandas left join does not warn about many-to-many joins by default (pandas 2.0+ warns, but polars does not)

**How to avoid:**
- Before any join, assert uniqueness of the join key in the right table: `assert right_df['ubigeo'].is_unique()`
- If admin data has both primaria and secundaria, create separate columns (`admin_dropout_primaria`, `admin_dropout_secundaria`) via pivot before joining
- After every join, assert row count unchanged: `assert len(merged) == len(base_df)`
- Deduplicate right tables before joining if appropriate

**Warning signs:**
- Row count increases after a "left join" (should never happen with unique right keys)
- Same student appears multiple times with different admin/census values
- Model training is slower than expected (more rows)
- Pooled dataset exceeds 200K rows (expected: 140K--180K)

**Phase to address:** Phase 3 (Spatial Merges)

**Confidence:** HIGH -- documented in spec Section 15 (Known Pitfalls table).

---

## Minor Pitfalls

Mistakes that cause annoyance, confusion, or minor inaccuracies but are fixable without major rework.

---

### Pitfall 15: Column Name Variations Across ENAHO Years

**What goes wrong:**
ENAHO column names may have slight variations across years: trailing spaces, case differences (`FACTOR07` vs `factor07`), or slight naming changes (e.g., `HOESSION` vs `HOGAR`). If you merge Module 200 and Module 300 using hardcoded column names, some years will fail silently (returning all-null merged columns).

**How to avoid:**
- Normalize all column names immediately after loading: `.columns = [c.strip().upper() for c in df.columns]` (pandas) or `df = df.rename({c: c.strip().upper() for c in df.columns})` (polars)
- After normalization, assert that merge keys exist: `assert all(k in df.columns for k in ['CONGLOME', 'VIVIENDA', 'CODPERSO'])`
- Log which columns were found per year for debugging

**Phase to address:** Phase 1 (Single-Year Loader)

**Confidence:** HIGH -- documented in spec Section 4.1.

---

### Pitfall 16: Threshold Tuning at 0.5 Default Misses Optimal Operating Point

**What goes wrong:**
Using the default 0.5 threshold for binary classification with imbalanced data (~14% positive rate) produces very low recall. The model predicts "no dropout" for most students because the base rate is well below 0.5. The optimal threshold for F1 maximization is typically 0.3--0.4 for this class ratio.

**How to avoid:**
- Tune threshold on validation set by sweeping [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] and selecting max weighted F1
- Report metrics at all thresholds for transparency (spec Section 6 requires this)
- Use precision-recall curves to visualize the threshold tradeoff
- Never use the test set for threshold selection

**Phase to address:** Phase 5 (Baseline Model), Phase 6 (LightGBM/XGBoost)

**Confidence:** HIGH -- standard practice for imbalanced classification.

---

### Pitfall 17: model_output="raw" vs "probability" in SHAP Changes Interpretation

**What goes wrong:**
SHAP TreeExplainer defaults to `model_output="raw"`, which for LightGBM means SHAP values explain the log-odds output. The SHAP values sum to the log-odds prediction, NOT to the probability prediction. If you treat these as probability-space contributions in your media export, the numbers will not be interpretable by journalists.

**How to avoid:**
- If you need probability-space SHAP values, use `model_output="probability"` (requires `feature_perturbation="interventional"`)
- If using `model_output="raw"` (default), clearly label all SHAP exports as "log-odds contributions" not "probability contributions"
- For media-facing exports, consider converting SHAP values to probability space or using the `Explanation` object's `.abs.mean()` for feature importance (which is interpretable regardless of scale)

**Phase to address:** Phase 9 (SHAP Analysis), Phase 11 (Findings Distillation)

**Confidence:** HIGH -- verified via [SHAP TreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

---

### Pitfall 18: Calibration Fitted on Wrong Set

**What goes wrong:**
`CalibratedClassifierCV` with `cv='prefit'` must be fitted on the validation set, not the training set. If fitted on the training set, the calibration will overfit to the training distribution and not improve (or may worsen) calibration on unseen data.

**How to avoid:**
- Use `cv='prefit'` and call `calibrated_model.fit(X_val, y_val, sample_weight=w_val)` with validation data only
- Verify Brier score improves: `assert brier_post_cal < brier_pre_cal`
- Calibration must happen BEFORE test set evaluation

**Phase to address:** Phase 7 (Calibration)

**Confidence:** HIGH -- standard ML practice, documented in spec Section 6.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoding column names per year | Handles year-specific quirks | Breaks when adding new years | Never -- normalize column names instead |
| Using pandas throughout instead of polars | Simpler sklearn integration | Inconsistent with spec, slower for data processing | Never -- spec mandates polars |
| Skipping survey weight validation | Faster development | All metrics are wrong, audit is invalid | Never |
| Computing SHAP on train for speed | 5x faster computation | Explains memorization, not generalization | Never |
| Using `is_unbalance=True` with survey weights | Quick imbalance fix | Conflicting weight signals, miscalibration | Only if you understand the interaction and explicitly choose this |
| Relaxing ONNX tolerance to 1e-2 | ONNX validation passes | Predictions in the M4 website diverge noticeably from Python model | Only as documented last resort after trying all other mitigations |

## Integration Gotchas

Common mistakes when connecting libraries in this pipeline.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| polars -> sklearn | Passing polars DataFrame directly | Always `.to_pandas()` or `.to_numpy()` first; select columns in consistent order |
| polars -> fairlearn MetricFrame | Passing polars Series as `sensitive_features` | Convert to pandas Series or numpy array first |
| LightGBM -> SHAP | Passing LGBMClassifier vs Booster object | TreeExplainer accepts both, but output shape may differ; prefer Booster for consistency |
| LightGBM -> onnxmltools | Using sklearn API model vs Booster | `convert_lightgbm` expects the Booster object, not the sklearn wrapper |
| sklearn CalibratedClassifierCV -> ONNX | Calibrating then exporting | ONNX export should be the uncalibrated model; calibration is Python-side only |
| fairlearn -> shap | Using different datasets | Both must use same dataset (test set); MetricFrame predictions must match SHAP input data |

## Performance Traps

Patterns that work at small scale but fail with full dataset.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| SHAP interaction values on full test set | OOM or multi-hour computation | Subsample to 1000 rows for interactions | n > 5000 rows |
| Loading all 7 years into memory as separate DataFrames | Works but wastes memory | Use `pl.concat()` and process as single frame | Not a problem at 180K rows, but bad practice |
| Bootstrap CI for all intersectional groups | Hours of computation | Limit to flagged groups with n > 50 | >20 intersectional groups x 1000 bootstrap reps |
| Optuna 50 trials with full train set | Each trial takes minutes | Use early stopping aggressively; consider subsample for initial search | 50 trials x 5 minutes = 4 hours |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Survey-weighted metrics:** Often reported without weights -- verify by comparing weighted vs unweighted values and asserting they differ
- [ ] **P300A harmonization:** Code exists but only `p300a_harmonized` is created -- verify `p300a_original` also exists for disaggregated analysis
- [ ] **Temporal split:** Data is split by year but StandardScaler is fit on full data -- verify scaler is fit on train only
- [ ] **ONNX export:** File is created but predictions are not validated -- verify max abs diff assertion passes on 100 samples
- [ ] **Fairness metrics:** MetricFrame runs without error but `sample_params` was flat dict -- verify weighted metrics differ from unweighted
- [ ] **SHAP values:** Computed and plotted but on training set -- verify input data year == 2024
- [ ] **Intersectional analysis:** Metrics computed for all groups but small-sample groups not flagged -- verify n >= 50 assertion exists
- [ ] **UBIGEO padding:** Applied to ENAHO data but not to admin/census data -- verify all data sources have 6-char UBIGEO
- [ ] **Threshold tuning:** Best threshold selected but only at 0.5 -- verify multiple thresholds evaluated on validation set
- [ ] **Model comparison:** LightGBM vs XGBoost compared on accuracy only -- verify fairness gaps compared across models too

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong delimiter loaded | LOW | Re-run loader with correct delimiter; no downstream impact if caught early |
| UBIGEO zero loss | MEDIUM | Re-pad all data sources and re-run all merges; check downstream models |
| P300A not harmonized | MEDIUM | Add harmonization and re-run feature engineering + all downstream |
| Survey weights omitted | HIGH | Re-run ALL metrics (descriptive, model eval, fairness); all exports invalid |
| Test data leaked | HIGH | Cannot recover -- must retrain models excluding leaked data and re-evaluate |
| SHAP on wrong set | LOW | Re-run SHAP on test set; no model changes needed |
| ONNX tolerance too tight | LOW | Relax tolerance or use double-precision tree summation; no model changes |
| Cartesian join | MEDIUM | Fix join, re-run feature engineering + downstream; verify no model contamination |
| MetricFrame weights dropped | MEDIUM | Fix sample_params nesting, re-run all fairness metrics and exports |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Delimiter mismatch (P1) | Phase 1 | Gate 1.1: assert column count >= 20 per loaded year |
| UBIGEO zero loss (P2) | Phase 1 (utils.py) | Gate 1.3: assert all UBIGEO len == 6, departments 01-09 have data |
| P300A harmonization (P3) | Phase 2 | Gate 1.2: code 3 count stable across years, both columns exist |
| Survey weights omitted (P4) | Phase 4+ | Every gate: assert weighted != unweighted metrics |
| MetricFrame sample_params (P5) | Phase 8 | Gate 3.1: compare MF overall to manual weighted computation |
| SHAP output shape (P6) | Phase 9 | Gate 3.2: assert shap_values.shape == (n_test, n_features) |
| Temporal leakage (P7) | Phase 5 | Gate 2.1: assert max train year == 2022, test year == 2024 |
| ONNX float32 error (P8) | Phase 7 | Gate 2.3: assert max abs diff < tolerance on 100 samples |
| Polars-pandas conversion (P9) | Phase 1 | Every phase: assert column order matches MODEL_FEATURES |
| Small sample intersections (P10) | Phase 8 | Gate 3.1: flag groups with n < 50, do not report without caveat |
| Class imbalance + weights (P11) | Phase 5-6 | Gate 2.1: verify calibration after training with chosen strategy |
| SHAP on train set (P12) | Phase 9 | Gate 3.2: assert SHAP input year == 2024 |
| ONNX output index (P13) | Phase 7 | Gate 2.3: assert output shape and probability range [0,1] |
| Cartesian join (P14) | Phase 3 | Gate 1.3: assert row count unchanged after merge |
| Column name variations (P15) | Phase 1 | Gate 1.1: assert key columns exist after normalization |
| Default 0.5 threshold (P16) | Phase 5-6 | Gate 2.1-2.2: metrics reported at multiple thresholds |
| SHAP model_output (P17) | Phase 9 | Gate 3.2: document whether values are log-odds or probability |
| Calibration on wrong set (P18) | Phase 7 | Gate 2.3: Brier score improves post-calibration |

## Sources

### Official Documentation (HIGH confidence)
- [Fairlearn 0.14 MetricFrame API](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html) -- sample_params API behavior
- [Fairlearn Advanced MetricFrame Usage](https://fairlearn.org/main/user_guide/assessment/advanced_metricframe.html) -- no global sample_params concept
- [SHAP TreeExplainer Documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) -- output shape for LightGBM binary classification
- [SHAP Release Notes v0.45](https://shap.readthedocs.io/en/latest/release_notes.html) -- breaking change from list to ndarray
- [sklearn-onnx LightGBM Conversion](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm_reg.html) -- float32 accumulation error
- [Polars to_pandas Documentation](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.to_pandas.html) -- conversion behavior
- [LightGBM Parameters Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) -- is_unbalance and scale_pos_weight

### GitHub Issues (HIGH confidence)
- [SHAP #526: TreeExplainer binary classification output shape](https://github.com/shap/shap/issues/526)
- [Polars #8204: Int64 to float64 conversion with nulls](https://github.com/pola-rs/polars/issues/8204)
- [LightGBM #6807: Class/sample weights vs is_unbalance interaction](https://github.com/microsoft/LightGBM/issues/6807)
- [onnxmltools #150: ONNX predictions differ from LightGBM](https://github.com/onnx/onnxmltools/issues/150)
- [onnxruntime #12629: Binary classifier output shape across environments](https://github.com/microsoft/onnxruntime/issues/12629)
- [scikit-learn #32167: Polars DataFrame compatibility issues](https://github.com/scikit-learn/scikit-learn/issues/32167)

### Academic Literature (HIGH confidence)
- [Survey Weights in ML (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0280387) -- 5pp F1 overestimation when ignoring survey weights
- [Sample Size for Fairness Audits (arXiv)](https://arxiv.org/html/2312.04745) -- minimum samples for intersectional analysis
- [Intersectional Fairness Survey (IJCAI 2023)](https://www.ijcai.org/proceedings/2023/0742.pdf) -- small sample challenges

### Project-Specific (HIGH confidence)
- [PeruData/ENAHO](https://github.com/PeruData/ENAHO) -- ENAHO data processing patterns
- [UBIGEO Peru Repository](https://github.com/ernestorivero/Ubigeo-Peru) -- UBIGEO coding systems
- Project specs.md Section 4.1, 15 -- known pitfalls enumerated by project author

---
*Pitfalls research for: ML fairness audit of ENAHO/education data (Alerta Escuela Equity Audit)*
*Researched: 2026-02-07*
