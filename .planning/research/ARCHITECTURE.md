# Architecture Research

**Domain:** ML fairness audit pipeline (survey microdata to media-ready findings)
**Researched:** 2026-02-07
**Confidence:** HIGH (project spec is authoritative; architecture patterns are well-established for batch ML pipelines)

## Standard Architecture

### System Overview

```
+---------------------------------------------------------------------------+
|                        DATA INGESTION LAYER                               |
|                                                                           |
|  +-------------+  +------------+  +-------------+  +---------------+      |
|  | enaho.py    |  | admin.py   |  | census.py   |  | nightlights.py|      |
|  | (7yr ENAHO) |  | (district  |  | (Census '17)|  | (VIIRS)       |      |
|  |             |  |  dropout)  |  |             |  |               |      |
|  +------+------+  +-----+------+  +------+------+  +-------+-------+      |
|         |               |                |                  |              |
+---------+---------------+----------------+------------------+--------------+
          |               |                |                  |
          v               v                v                  v
+---------------------------------------------------------------------------+
|                     HARMONIZATION + MERGE LAYER                           |
|                                                                           |
|  +--------------------------------------------------------------------+  |
|  |                    features.py                                      |  |
|  |  merge_all_sources() -> LEFT JOINs on UBIGEO -> build_features()   |  |
|  |  P300A harmonization, language dummies, region mapping, quintiles   |  |
|  +--------------------------------------------------------------------+  |
|                                                                           |
|  Output: data/processed/enaho_with_features.parquet  (polars DataFrame)   |
+---------------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------------+
|                        MODEL TRAINING LAYER                               |
|                                                                           |
|  +--------------------+    +--------------------+                         |
|  | train.py           |    | evaluate.py        |                         |
|  | create_temporal_   |    | evaluate_model()   |                         |
|  |   splits()         |    | evaluate_at_       |                         |
|  | train_logistic_    |    |   thresholds()     |                         |
|  |   regression()     +--->| survey-weighted    |                         |
|  | train_lightgbm()   |    |   PR-AUC, F1, etc  |                         |
|  | train_xgboost()    |    +--------------------+                         |
|  +--------------------+                                                   |
|                                                                           |
|  +--------------------+    +--------------------+                         |
|  | calibrate.py       |    | export_onnx.py     |                         |
|  | Platt/isotonic     |    | LightGBM -> ONNX   |                         |
|  | on validation set  |    | + prediction check  |                         |
|  +--------------------+    +--------------------+                         |
|                                                                           |
|  BOUNDARY: polars -> .to_pandas() happens HERE for sklearn/lightgbm      |
+---------------------------------------------------------------------------+
          |
          |  (predictions: y_pred_proba, y_pred, + meta columns)
          v
+---------------------------------------------------------------------------+
|                      FAIRNESS ANALYSIS LAYER                              |
|                                                                           |
|  +--------------------+    +--------------------+                         |
|  | metrics.py         |    | shap_analysis.py   |                         |
|  | MetricFrame per    |    | TreeExplainer on   |                         |
|  |   6 dimensions     |    |   test set (2024)  |                         |
|  | 3 intersections    |    | global, regional,  |                         |
|  | calibration per    |    |   interaction SHAP  |                         |
|  |   group            |    | 10 representative  |                         |
|  | cross_validate_    |    |   student profiles  |                         |
|  |   admin()          |    |                    |                         |
|  +--------------------+    +--------------------+                         |
|                                                                           |
|  BOUNDARY: polars -> .to_pandas() for fairlearn MetricFrame + shap       |
+---------------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------------+
|                      DISTILLATION + EXPORT LAYER                          |
|                                                                           |
|  +--------------------------------------------------------------------+  |
|  |                       distill.py                                    |  |
|  |  Reads all intermediate JSONs -> produces findings.json             |  |
|  |  5-7 media-ready findings with Spanish/English translations         |  |
|  +--------------------------------------------------------------------+  |
|                                                                           |
|  Output: data/exports/                                                    |
|    findings.json, fairness_metrics.json, shap_values.json,               |
|    choropleth.json, model_results.json, descriptive_tables.json,         |
|    onnx/lightgbm_dropout.onnx, README.md                                |
+---------------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------------+
|                      VALIDATION LAYER (cross-cutting)                     |
|                                                                           |
|  tests/gates/                    tests/unit/                              |
|  test_gate_1_1.py ... 3_4.py    test_enaho_loader.py                     |
|                                  test_harmonization.py                    |
|  Gate tests validate each        test_ubigeo.py                           |
|  phase output before the                                                  |
|  pipeline can proceed.           Unit tests validate                      |
|                                  individual functions.                    |
+---------------------------------------------------------------------------+
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation | Polars/Pandas Boundary |
|-----------|----------------|------------------------|------------------------|
| `src/data/enaho.py` | Load ENAHO CSVs (7 years), detect delimiters, normalize columns, construct dropout target, harmonize P300A | `load_enaho_year()`, `load_all_years()`, `harmonize_p300a()` | Pure polars |
| `src/data/admin.py` | Load district-level administrative dropout rates, zero-pad UBIGEO | `load_admin_dropout_rates()` | Pure polars |
| `src/data/census.py` | Load Census 2017 district aggregates for enrichment | `load_census_data()` | Pure polars |
| `src/data/nightlights.py` | Load VIIRS nighttime lights district data | `load_nightlights()` | Pure polars |
| `src/data/features.py` | Merge all sources via LEFT JOIN on UBIGEO, build all 19+ model features, create language dummies, region mapping, poverty quintiles | `merge_all_sources()`, `build_features()` | Pure polars; output saved as parquet |
| `src/models/train.py` | Create temporal splits, train 3 models (LogReg, LightGBM, XGBoost), Optuna hyperparameter tuning | `create_temporal_splits()`, `train_logistic_regression()`, `train_lightgbm()`, `train_xgboost()` | **Conversion boundary:** `.to_pandas()` for X/y into sklearn/lgb/xgb |
| `src/models/evaluate.py` | Survey-weighted evaluation (PR-AUC, F1, precision, recall, Brier), multi-threshold reporting | `evaluate_model()`, `evaluate_at_thresholds()` | Receives pandas arrays from train.py |
| `src/models/calibrate.py` | Platt/isotonic calibration on validation set, select by Brier score | `calibrate_model()` | Receives pandas, uses sklearn CalibratedClassifierCV |
| `src/models/export_onnx.py` | Convert LightGBM to ONNX, validate prediction parity (<1e-5 diff) | `export_to_onnx()`, `validate_onnx()` | Receives pandas/numpy arrays |
| `src/fairness/metrics.py` | Compute fairness metrics via fairlearn MetricFrame across 6 dimensions + 3 intersections, calibration per group, cross-validate against admin data | `compute_fairness_metrics()`, `cross_validate_admin()` | **Conversion boundary:** `.to_pandas()` for fairlearn MetricFrame |
| `src/fairness/shap_analysis.py` | Global, regional, and interaction SHAP values via TreeExplainer, generate 10 representative student profiles | `compute_global_shap()`, `compute_regional_shap()`, `compute_interaction_shap()` | **Conversion boundary:** `.to_pandas()` for shap TreeExplainer |
| `src/fairness/distill.py` | Read all intermediate JSON exports, synthesize 5-7 media-ready findings with Spanish translations | `distill_findings()` | Pure Python (reads JSON, writes JSON) |
| `src/utils.py` | UBIGEO padding, constants (MODEL_FEATURES, PROTECTED_ATTRIBUTES, META_COLUMNS), paths, survey weight helpers | `pad_ubigeo()`, constants | Pure Python/polars |

## Recommended Project Structure

```
alerta-escuela-audit/
├── pyproject.toml              # uv project config, dependencies
├── flake.nix                   # Nix flake for system deps
├── specs.md                    # Authoritative project specification
├── src/
│   ├── __init__.py
│   ├── data/                   # LAYER 1: Data ingestion + harmonization
│   │   ├── __init__.py
│   │   ├── enaho.py            # ENAHO loading, merging, harmonization
│   │   ├── admin.py            # District-level dropout rates
│   │   ├── census.py           # Census 2017 district enrichment
│   │   ├── nightlights.py     # VIIRS district-level economic proxy
│   │   └── features.py        # Feature engineering + merge orchestration
│   ├── models/                 # LAYER 2: Model training + evaluation
│   │   ├── __init__.py
│   │   ├── train.py            # Train all 3 models, temporal splits
│   │   ├── evaluate.py         # Survey-weighted metrics + JSON export
│   │   ├── calibrate.py        # Platt/isotonic calibration
│   │   └── export_onnx.py     # LightGBM -> ONNX
│   ├── fairness/               # LAYER 3: Fairness analysis
│   │   ├── __init__.py
│   │   ├── metrics.py          # Subgroup fairness + cross-validation
│   │   ├── shap_analysis.py   # SHAP global/regional/interaction
│   │   └── distill.py         # Findings distillation -> JSON exports
│   └── utils.py                # Shared: UBIGEO, constants, paths, helpers
├── notebooks/                  # EDA and narrative
│   ├── 01_enaho_exploration.ipynb
│   ├── 02_descriptive_gaps.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_fairness_audit.ipynb
├── tests/
│   ├── conftest.py             # Shared fixtures (sample data, paths)
│   ├── gates/                  # Phase gate validation tests
│   │   └── test_gate_{N}_{M}.py
│   └── unit/                   # Function-level unit tests
│       ├── test_enaho_loader.py
│       ├── test_harmonization.py
│       └── test_ubigeo.py
├── data/
│   ├── raw/                    # .gitignored -- user-provided source data
│   ├── processed/              # .gitignored -- intermediate parquets
│   └── exports/                # COMMITTED -- JSON exports for M4 site
└── outputs/
    └── figures/                # Generated plots
```

### Structure Rationale

- **`src/data/`:** One file per data source creates clean responsibility boundaries. When a loader breaks (and they will, because government CSVs are messy), you know exactly which file to fix. `features.py` acts as the orchestrator that pulls all sources together -- it depends on all other files in `src/data/` but nothing in `src/data/` depends on it.

- **`src/models/`:** Separating train, evaluate, calibrate, and export into distinct files enforces the discipline of not leaking evaluation logic into training and not touching the test set prematurely. `train.py` produces models; `evaluate.py` measures them; `calibrate.py` post-processes them; `export_onnx.py` serializes them. This separation makes the "test set touched once" rule auditable.

- **`src/fairness/`:** Mirrors the modeling layer but for the actual product (the audit). `metrics.py` uses fairlearn to compute subgroup metrics. `shap_analysis.py` uses shap for interpretability. `distill.py` synthesizes everything into journalist-facing output. This separation matters because fairness analysis requires the complete modeling pipeline to be done first.

- **`tests/gates/`:** Gate tests are the "stop and verify" mechanism. They are numbered by phase (1.1, 1.2, ..., 3.4) and enforce invariants before proceeding. This is different from unit tests -- gate tests validate pipeline stage outputs, not individual functions.

## Architectural Patterns

### Pattern 1: Polars-First with Boundary Conversion

**What:** Use polars for all data loading, transformation, and feature engineering. Convert to pandas (via `.to_pandas()`) only at the exact point where sklearn, fairlearn, or shap require pandas/numpy input. Convert back to polars immediately after if further polars processing is needed.

**When to use:** Every stage of this pipeline. The boundary is explicit: `src/models/train.py` is where conversion first happens, and it stays in pandas/numpy through `src/models/` and `src/fairness/` for sklearn/fairlearn/shap calls.

**Trade-offs:**
- Pro: Polars is faster for data wrangling on ~180K rows, has a cleaner API for group-by/window operations, and avoids the "SettingWithCopyWarning" class of pandas bugs entirely
- Pro: Explicit boundary prevents accidental mixing -- you always know what type you're dealing with
- Con: Some cognitive overhead maintaining two mental models
- Con: If fairlearn or shap APIs change how they accept data, the boundary code needs updating

**Example:**
```python
# In src/data/features.py -- pure polars
def build_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("P300A").map_elements(harmonize_p300a, return_dtype=pl.Int64))
            .alias("p300a_harmonized"),
        pl.when(pl.col("P207") == 2).then(1).otherwise(0).alias("es_mujer"),
        # ... more feature columns
    ])

# In src/models/train.py -- conversion boundary
def create_temporal_splits(df: pl.DataFrame) -> tuple:
    train_pl = df.filter(pl.col("year").is_in([2018, 2019, 2020, 2021, 2022]))
    val_pl = df.filter(pl.col("year") == 2023)
    test_pl = df.filter(pl.col("year") == 2024)

    # Convert at boundary -- these DataFrames go into sklearn
    X_train = train_pl.select(MODEL_FEATURES).to_pandas()
    y_train = train_pl.select("dropout").to_pandas().values.ravel()
    w_train = train_pl.select("factor07").to_pandas().values.ravel()
    # ... same for val and test

    return (X_train, y_train, w_train), (X_val, y_val, w_val), (X_test, y_test, w_test)
```

### Pattern 2: Gate-Guarded Pipeline Stages

**What:** Each pipeline stage writes its output to a known location (parquet file or JSON export), and a corresponding gate test validates that output before the next stage is permitted to run. Stages are designed to be independently re-runnable.

**When to use:** This is the primary execution pattern for the entire pipeline. Phases 1-11 in the spec each have gate tests.

**Trade-offs:**
- Pro: Catches data quality issues early (bad UBIGEO, wrong delimiter, harmonization bugs)
- Pro: Makes human review explicit -- some gates require human confirmation
- Pro: Supports incremental development -- you can stop, fix, and restart from any checkpoint
- Con: Adds test-writing overhead at each stage
- Con: Parquet checkpoints consume disk space (but ~180K rows is tiny)

**Example:**
```python
# Gate test pattern -- tests/gates/test_gate_1_1.py
import polars as pl
import pytest

@pytest.fixture
def enaho_2023():
    from src.data.enaho import load_enaho_year
    return load_enaho_year(2023)

def test_row_count(enaho_2023):
    n = len(enaho_2023)
    assert 23_000 <= n <= 28_000, f"Expected 23K-28K rows, got {n}"

def test_dropout_rate(enaho_2023):
    weighted_rate = (
        (enaho_2023["dropout"] * enaho_2023["FACTOR07"]).sum()
        / enaho_2023["FACTOR07"].sum()
    )
    assert 0.12 <= weighted_rate <= 0.16, f"Expected 12-16% weighted dropout, got {weighted_rate:.3f}"

def test_ubigeo_format(enaho_2023):
    assert enaho_2023["UBIGEO"].str.len_chars().min() == 6
    assert enaho_2023["UBIGEO"].str.len_chars().max() == 6
```

### Pattern 3: Survey-Weight-First Metric Functions

**What:** Every metric function accepts `sample_weight` as a required parameter, not optional. Compute both weighted and unweighted metrics, and assert they differ (proving weights are actually applied).

**When to use:** All evaluation and fairness metric computation. This is non-negotiable for survey microdata -- unweighted ENAHO metrics are statistically invalid representations of the Peruvian population.

**Trade-offs:**
- Pro: Prevents the single most common error in survey-based ML (ignoring weights)
- Pro: The "assert weighted != unweighted" check catches silent weight-ignoring bugs
- Con: Slightly more verbose function signatures
- Con: Some sklearn functions handle `sample_weight` inconsistently (e.g., `precision_recall_curve` does not support `sample_weight` in all sklearn versions -- verify at implementation time)

**Example:**
```python
# In src/models/evaluate.py
def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,  # Required, not optional
) -> dict:
    weighted = {
        "pr_auc": average_precision_score(y_true, y_pred_proba, sample_weight=sample_weight),
        "f1": f1_score(y_true, y_pred, sample_weight=sample_weight),
        "recall": recall_score(y_true, y_pred, sample_weight=sample_weight),
        "precision": precision_score(y_true, y_pred, sample_weight=sample_weight),
        "brier": brier_score_loss(y_true, y_pred_proba, sample_weight=sample_weight),
    }
    weighted["fnr"] = 1.0 - weighted["recall"]

    # Sanity check: weights must actually affect results
    unweighted_f1 = f1_score(y_true, y_pred)
    assert weighted["f1"] != unweighted_f1, "Weights had no effect on F1 -- check weight array"

    return weighted
```

### Pattern 4: Immutable Intermediate Artifacts

**What:** Each pipeline stage reads from a known input artifact (parquet or JSON) and writes to a known output artifact. Stages never modify their inputs. This creates a reproducible chain: raw CSVs -> enaho_merged.parquet -> enaho_with_features.parquet -> full_dataset.parquet -> model predictions -> fairness metrics -> findings.

**When to use:** All data processing stages. The spec mandates `data/processed/` for intermediates and `data/exports/` for finals.

**Trade-offs:**
- Pro: Full reproducibility -- delete `data/processed/` and re-run from raw
- Pro: Debugging is easy -- inspect any intermediate parquet to find where corruption entered
- Pro: Supports the "re-run from checkpoint" pattern when iterating on later stages
- Con: Some redundant disk I/O (writing then re-reading parquet between stages)

## Data Flow

### Complete Pipeline Flow

```
data/raw/enaho/{YEAR}/*.csv
    |
    | enaho.py: load_enaho_year() x7, merge mod200+mod300, harmonize P300A
    v
data/processed/enaho_merged.parquet  (~180K rows, ~30 columns)
    |
    | admin.py, census.py, nightlights.py: load supplementary sources
    | features.py: merge_all_sources() via LEFT JOIN on UBIGEO
    v
data/processed/full_dataset.parquet  (~180K rows, ~45 columns)
    |
    | features.py: build_features() -- language dummies, region map, quintiles
    v
data/processed/enaho_with_features.parquet  (~180K rows, ~50+ columns)
    |
    | train.py: create_temporal_splits()
    |   Train: 2018-2022 (~130K rows)
    |   Validate: 2023 (~25K rows)
    |   Test: 2024 (~25K rows)
    |
    | .to_pandas() BOUNDARY -- polars -> pandas/numpy for sklearn
    v
[sklearn Pipeline: StandardScaler + LogisticRegression]  --> validate metrics
[LightGBM: Optuna 50 trials, early stopping]             --> validate metrics
[XGBoost: tuned hyperparameters]                          --> validate metrics
    |
    | calibrate.py: CalibratedClassifierCV on validation set
    | export_onnx.py: LightGBM -> ONNX + validation
    | evaluate.py: FINAL test evaluation (2024 -- touched ONCE)
    v
data/exports/model_results.json
data/exports/onnx/lightgbm_dropout.onnx
    |
    | .to_pandas() BOUNDARY -- for fairlearn MetricFrame + shap
    v
[fairlearn MetricFrame: 6 dimensions x 5 metrics + 3 intersections]
[shap TreeExplainer: global + 3 regional + interaction values]
[cross_validate_admin: district-level correlation]
    |
    v
data/exports/fairness_metrics.json
data/exports/shap_values.json
data/exports/choropleth.json
data/exports/descriptive_tables.json
    |
    | distill.py: synthesize all JSONs -> media-ready findings
    v
data/exports/findings.json  (5-7 findings with es/en translations)
data/exports/README.md      (schema documentation)
```

### Key Data Schemas at Each Stage

| Stage | Key Columns | Row Count | Notes |
|-------|-------------|-----------|-------|
| Raw ENAHO CSVs | Module-specific (P207, P208A, P300A, P303, P306, UBIGEO, FACTOR07, etc.) | ~25K per year | Delimiter varies by year |
| `enaho_merged.parquet` | Normalized superset of mod200+mod300 columns + `year`, `dropout`, `p300a_harmonized`, `p300a_original` | ~180K (7 years) | All columns uppercased, UBIGEO zero-padded |
| `full_dataset.parquet` | Above + `district_dropout_rate_admin`, `nightlight_intensity`, `poverty_index`, `school_student_teacher_ratio` | ~180K (same -- LEFT JOIN) | Row count MUST equal enaho_merged |
| `enaho_with_features.parquet` | Above + all `MODEL_FEATURES` columns: `es_mujer`, `lang_*`, `rural`, `region_natural`, `log_income`, `parent_education_years`, etc. | ~180K (same) | Ready for model training |
| Train split (pandas) | `MODEL_FEATURES` (19 columns) + `dropout` + `factor07` | ~130K | Years 2018-2022 |
| Validate split (pandas) | Same schema | ~25K | Year 2023 |
| Test split (pandas) | Same schema + `META_COLUMNS` for fairness | ~25K | Year 2024 -- touched ONCE |

### Polars-to-Pandas Conversion Points

There are exactly **three** conversion boundaries in this pipeline. All other code is pure polars.

| Boundary | Location | What Converts | Why |
|----------|----------|---------------|-----|
| **Model Training** | `src/models/train.py: create_temporal_splits()` | Feature matrix (X), target (y), weights (w) for each split | sklearn, lightgbm, xgboost all require pandas/numpy input |
| **Fairness Metrics** | `src/fairness/metrics.py: compute_fairness_metrics()` | Predictions + sensitive features + weights | fairlearn MetricFrame requires pandas Series/DataFrame |
| **SHAP Analysis** | `src/fairness/shap_analysis.py: compute_global_shap()` | Test set feature matrix | shap TreeExplainer requires pandas/numpy input |

**Architectural rule:** Conversion happens at function call boundaries, not mid-function. The calling function receives polars, converts to pandas, calls the sklearn/fairlearn/shap API, and returns Python dicts or numpy arrays. Polars DataFrames never flow into sklearn-adjacent code.

## Build Order (Dependency Graph)

```
Phase 1: enaho.py + utils.py (single-year loader)
    |
    v
Phase 2: enaho.py extension (multi-year + P300A harmonization)
    |
    v
Phase 3: admin.py + census.py + nightlights.py + features.py (merge)
    |     \
    |      Depends on: enaho_merged.parquet existing
    v
Phase 4: features.py (build_features) + notebooks/02_descriptive_gaps.ipynb
    |     \
    |      Depends on: full_dataset.parquet existing
    v
Phase 5: train.py + evaluate.py (LogReg baseline)
    |     \
    |      Depends on: enaho_with_features.parquet existing
    v
Phase 6: train.py extension (LightGBM + XGBoost)
    |     \
    |      Depends on: LogReg baseline working (validates pipeline)
    v
Phase 7: calibrate.py + export_onnx.py (final test eval)
    |     \
    |      Depends on: trained LightGBM model
    |      CRITICAL: First and ONLY time test set (2024) is used
    v
Phase 8: metrics.py (fairness metrics)
    |     \
    |      Depends on: calibrated model + test set predictions
    v
Phase 9: shap_analysis.py (SHAP)
    |     \
    |      Depends on: trained LightGBM model + test set
    v
Phase 10: metrics.py extension (cross-validate admin)
    |      \
    |       Depends on: district-level predictions + admin dropout data
    v
Phase 11: distill.py (findings + exports)
           \
            Depends on: ALL prior JSON exports existing
```

**Critical ordering constraints:**

1. **Data before models:** Phases 1-4 must complete before Phase 5. There is no way to train models without the feature-engineered dataset.

2. **Baseline before boost:** Phase 5 (LogReg) must complete before Phase 6 (LightGBM/XGBoost). The baseline validates the train/evaluate pipeline works, and provides the benchmark that boosted models must beat.

3. **Training before fairness:** Phases 5-7 must complete before Phases 8-10. Fairness analysis operates on model predictions, which do not exist until models are trained.

4. **Test set isolation:** The test set (2024) must NOT be touched until Phase 7. Phases 5-6 use only the validation set (2023). This is an architectural invariant that gate tests enforce.

5. **All exports before distillation:** Phase 11 reads all prior JSON exports. It cannot run until Phases 8-10 have each written their export files.

**Parallelization opportunities:**

- Phases 8 (fairness metrics) and 9 (SHAP) are independent of each other -- they both read from the trained model and test set predictions. They COULD run in parallel if needed, though the spec sequences them.
- Phase 10 (admin cross-validation) depends on Phase 8 outputs (district-aggregated predictions), so it must follow Phase 8.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Pandas Throughout

**What people do:** Use pandas for everything because it is more familiar, ignoring the polars specification.

**Why it is wrong:** The spec explicitly requires polars as primary. More importantly, pandas has pitfalls that polars avoids: SettingWithCopyWarning, implicit type coercion, index alignment bugs. With survey data that has specific merge-key requirements (UBIGEO), polars' explicit join semantics are safer.

**Do this instead:** Use polars for all data loading and transformation. Convert to pandas only at the three documented boundary points.

### Anti-Pattern 2: Leaking Test Data

**What people do:** Use the test set (2024) for hyperparameter tuning, threshold selection, or model comparison, then "evaluate" on the same test set and report inflated metrics.

**Why it is wrong:** This is the most common ML methodology error. In this project, it would invalidate the entire fairness audit. If thresholds are tuned on 2024 data, the fairness metrics computed on 2024 data are not trustworthy.

**Do this instead:** Enforce the temporal split strictly. Gate tests in Phases 5-6 should assert that no 2024 data appears in training or validation. Phase 7 is the single point where 2024 data enters the evaluation pipeline.

### Anti-Pattern 3: Unweighted Metrics

**What people do:** Call `f1_score(y_true, y_pred)` without `sample_weight`, getting unweighted metrics that do not represent the Peruvian population.

**Why it is wrong:** ENAHO uses complex stratified sampling with expansion factors (FACTOR07). Unweighted metrics over-represent densely sampled strata (urban Lima) and under-represent sparsely sampled strata (rural Amazonia). For a fairness audit focused on marginalized populations, this error is catastrophic -- it would systematically undercount the very groups we are auditing.

**Do this instead:** Every metric function takes `sample_weight` as a required parameter. Compute both weighted and unweighted, assert they differ.

### Anti-Pattern 4: Cartesian Join Explosion

**What people do:** Join ENAHO with district-level data on UBIGEO without verifying join type and cardinality. If the admin data has duplicate UBIGEOs (e.g., separate rows for primaria and secundaria), a naive join duplicates ENAHO rows.

**Why it is wrong:** Silently doubles or triples the dataset, corrupting all downstream statistics. The row count invariant (output rows == input rows for LEFT JOIN) is the only reliable detection mechanism.

**Do this instead:** Assert row count before and after every merge in `features.py`. If admin data has multiple rows per UBIGEO (by education level), aggregate or filter before joining.

### Anti-Pattern 5: Monolithic Pipeline Script

**What people do:** Write one massive `pipeline.py` that loads data, engineers features, trains models, computes fairness, and exports -- all in a single function.

**Why it is wrong:** Makes debugging impossible when something breaks at step 7 of 11. Makes gate testing impossible. Makes re-running from a checkpoint impossible. Violates the "immutable intermediate artifacts" pattern.

**Do this instead:** Each module produces a parquet or JSON artifact. Each module can be run independently given its input artifact exists. Gate tests validate each artifact before the next module runs.

### Anti-Pattern 6: SHAP on Training Data

**What people do:** Compute SHAP values on the training set because it is larger and produces smoother plots.

**Why it is wrong:** SHAP on training data shows what the model memorized, not what it generalizes. For a fairness audit, this is misleading -- training-set SHAP may hide overfitting to specific subgroups. The spec requires SHAP on the test set (2024).

**Do this instead:** Gate test 3.2 asserts that the SHAP input DataFrame's year column contains only 2024.

## Testing Strategy

### Three-Tier Testing

| Tier | Location | Purpose | When Run |
|------|----------|---------|----------|
| **Unit tests** | `tests/unit/` | Test individual functions in isolation (harmonization logic, UBIGEO padding, delimiter detection) | During development, on every commit |
| **Gate tests** | `tests/gates/` | Validate pipeline stage outputs against known invariants (row counts, rate bounds, column presence) | After each phase completes, before proceeding to next phase |
| **Integration assertions** | Inline in pipeline code | Runtime checks embedded in production code (row count after merge, weight != 0, year boundaries) | Every pipeline execution |

### Gate Test Design Principles

1. **Bound, don't hardcode:** Use ranges (`23_000 <= n <= 28_000`) not exact values (`n == 25_412`). Survey data varies by year and exact counts may shift with minor preprocessing changes.

2. **Test invariants, not implementation:** "Weighted dropout rate is between 12-16%" is an invariant. "The third column is named P300A" is implementation detail that might change with column reordering.

3. **Print for human review:** Gate tests that precede human review gates should print diagnostic information (sample rows, coefficient tables, metric comparisons) so the human can make an informed decision.

4. **Mark stop points:** Gate tests that require human review should include a visible marker: `# GATE X.Y -- STOP AND REVIEW BEFORE PROCEEDING`.

### What Gate Tests Validate at Each Stage

| Gate | Validates | Critical Check |
|------|-----------|----------------|
| 1.1 | Single-year ENAHO load | Row count, dropout rate, UBIGEO format |
| 1.2 | Multi-year load + harmonization | P300A stability across years, delimiter detection |
| 1.3 | Spatial merge (admin data) | UBIGEO format, merge rate, row count unchanged |
| 1.4 | Enrichment merge (census, nightlights) | Coverage rates, no duplicate rows |
| 1.5 | Feature engineering | Feature count, binary values, correlation check |
| 2.1 | Baseline model | No temporal leakage, PR-AUC > random, weights applied |
| 2.2 | LightGBM + XGBoost | LGB > LR, feature importance distribution |
| 2.3 | Calibration + ONNX | Brier improvement, ONNX parity, test-val metric proximity |
| 3.1 | Fairness metrics | All 6 dimensions covered, weighted != unweighted |
| 3.2 | SHAP | Computed on 2024, shape matches, top features overlap with LR |
| 3.3 | Admin cross-validation | Positive correlation, statistical significance |
| 3.4 | Exports | All 7 files present, schemas valid, findings complete |

### Inline Assertion Pattern

Beyond gate tests, the pipeline code itself should contain defensive assertions at critical junctures.

```python
# In features.py: merge_all_sources()
def merge_all_sources(enaho_df, admin_df, census_df, nightlights_df):
    n_before = len(enaho_df)

    merged = enaho_df.join(admin_df, on="ubigeo", how="left")
    assert len(merged) == n_before, (
        f"Row count changed after admin merge: {n_before} -> {len(merged)}. "
        "Check for duplicate UBIGEO in admin data."
    )

    merged = merged.join(census_df, on="ubigeo", how="left")
    assert len(merged) == n_before, (
        f"Row count changed after census merge: {n_before} -> {len(merged)}."
    )

    # ... nightlights merge with same assertion

    return merged
```

## Scaling Considerations

This pipeline processes ~180K rows, which is tiny by ML standards. Scaling is not a concern for the data processing itself. The relevant "scaling" considerations are computational:

| Concern | At ~180K rows (current) | If dataset grew 10x | Notes |
|---------|------------------------|---------------------|-------|
| Data loading | <5 seconds with polars | <30 seconds | polars handles millions efficiently |
| Feature engineering | <10 seconds | <60 seconds | All operations are vectorized |
| LightGBM training (50 Optuna trials) | ~10-30 minutes | ~2-5 hours | Main bottleneck; can reduce trials |
| SHAP TreeExplainer (global) | ~1-5 minutes on test set | ~10-30 minutes | Linear in n_samples |
| SHAP interaction values | ~5-30 minutes (subsample to 1000) | Same (subsampled) | O(n * n_features^2), always subsample |
| fairlearn MetricFrame | <30 seconds | <5 minutes | Lightweight computation |

**First bottleneck:** SHAP interaction values. Always subsample to 1000 rows regardless of test set size.

**Second bottleneck:** Optuna trials for LightGBM. With 50 trials and early stopping, this is manageable but is the longest single computation. Reduce to 20 trials for faster iteration during development.

## Integration Points

### External Data Sources

| Source | Integration Pattern | Gotchas |
|--------|---------------------|---------|
| ENAHO CSVs (INEI) | User manually downloads and places in `data/raw/enaho/{YEAR}/` | Delimiter changes at 2020 boundary; column name whitespace; UBIGEO zero-padding |
| Admin dropout rates (datosabiertos) | User downloads CSV to `data/raw/admin/` | UBIGEO leading zeros stripped; may have separate rows for primaria/secundaria |
| Census 2017 (INEI) | User downloads to `data/raw/census/` | Potentially different UBIGEO format; may need column name mapping |
| VIIRS Nightlights | Pre-aggregated CSV in `data/raw/nightlights/` | Coverage may not reach all districts; no negative values expected |
| Censo Escolar (ESCALE) | Aggregated CSV in `data/raw/escolar/` | Student-teacher ratio computation may need validation |

### Internal Module Boundaries

| Boundary | Communication | Data Format | Key Constraint |
|----------|---------------|-------------|----------------|
| `src/data/*` -> `src/models/train.py` | File: `enaho_with_features.parquet` | polars -> pandas at boundary | MODEL_FEATURES list must match column names exactly |
| `src/models/train.py` -> `src/models/evaluate.py` | Function call: model + predictions + weights | numpy arrays | sample_weight must be passed to every metric |
| `src/models/train.py` -> `src/fairness/metrics.py` | File: exported predictions + meta columns | pandas DataFrame or saved parquet | Must include sensitive features + weights + predictions |
| `src/models/train.py` -> `src/fairness/shap_analysis.py` | Model object + test set feature matrix | trained LightGBM model + pandas/numpy | Feature names must match training feature names exactly |
| `src/fairness/metrics.py` -> `src/fairness/distill.py` | File: `fairness_metrics.json` | JSON | Schema must match M4 contract (Section 11.2 of spec) |
| All `src/fairness/*` -> `src/fairness/distill.py` | Files: all JSON exports | JSON | distill.py reads all prior exports; fails if any missing |
| `src/utils.py` -> everywhere | Import: constants, helper functions | Python module | MODEL_FEATURES, PROTECTED_ATTRIBUTES, META_COLUMNS are single source of truth |

### M4 Site Contract (Downstream)

The JSON exports in `data/exports/` are the contract with the downstream Next.js scrollytelling site. The schemas are defined in Section 11 of specs.md and are non-negotiable. Key constraints:

- `findings.json`: Must have `headline_es` and `headline_en` for bilingual support
- `fairness_metrics.json`: Nested structure by dimension -> group -> metric
- `shap_values.json`: Must include `feature_labels_es` for Spanish feature names
- `choropleth.json`: Must include `latitude`/`longitude` for map rendering (may need geocoding of UBIGEO)
- All JSON files: Must include `generated_at` ISO timestamp for provenance

## Sources

- **specs.md** (authoritative project specification) -- Section 3 (directory structure), Section 5 (feature engineering), Section 6 (modeling), Section 7 (fairness audit), Section 8 (SHAP), Section 9 (roadmap/phases), Section 11 (export contracts). **Confidence: HIGH** -- this is the project's own specification document.
- **polars documentation** -- `.to_pandas()` conversion API, join semantics, group-by API. **Confidence: MEDIUM** -- based on training knowledge of polars >=1.0 API. Verify exact method signatures at implementation time.
- **sklearn documentation** -- `CalibratedClassifierCV`, metric functions with `sample_weight`, `Pipeline`. **Confidence: HIGH** -- stable API across sklearn 1.x versions.
- **fairlearn documentation** -- `MetricFrame` with `sample_params` for weighted metrics. **Confidence: MEDIUM** -- fairlearn 0.11 API based on training knowledge. The `sample_params` dict-of-dicts pattern should be verified against current fairlearn docs at implementation time.
- **shap documentation** -- `TreeExplainer`, `shap_values()`, `shap_interaction_values()`. **Confidence: MEDIUM** -- shap API has changed across versions (list vs. array return for binary classification). Verify whether `shap_values[1]` or direct array is returned for the installed shap version.
- **LightGBM/XGBoost documentation** -- Training API, Optuna integration. **Confidence: HIGH** -- stable APIs.
- **General ML pipeline architecture patterns** -- Gate testing, temporal splits, immutable artifacts. **Confidence: HIGH** -- well-established patterns in applied ML.

**Note:** WebSearch and WebFetch were unavailable during this research session. All findings are based on training knowledge and the project's own specs.md. Library API details (especially for polars interop, fairlearn MetricFrame sample_params, and shap return types) should be verified with current documentation during implementation.

---
*Architecture research for: ML fairness audit pipeline (ENAHO survey data to media-ready equity findings)*
*Researched: 2026-02-07*
