# Alerta Escuela Equity Audit — Project Specification

> **Purpose:** Single source of truth for agentic development of an independent equity audit of Peru's national dropout prediction system. This document contains everything needed to implement phases M1–M3 (data pipeline, modeling, fairness audit) plus M4 export contracts using GSD (get-shit-done) with Claude Code.

> **CRITICAL RULE FOR THE AGENT:** Before writing ANY code in a new phase, re-read the relevant phase section of this document. Context rot will cause you to drift from the spec. This document is authoritative — if your memory contradicts this doc, the doc wins.

---

## 1. PROJECT VISION

### What We're Building

The first independent equity audit of Peru's "Alerta Escuela" — a LightGBM-based ML system deployed by MINEDU (Ministry of Education) in October 2020 to predict student dropout risk. The system uses gender, mother tongue, and nationality as predictive features. Non-Peruvian nationality is a top predictor. Female gender increases predicted dropout risk in secundaria. **Zero fairness analysis has been published.**

We build a reproducible ML pipeline using publicly available survey data (ENAHO) to:
1. Replicate the dropout prediction task with comparable features
2. Quantify fairness gaps across protected groups (language, sex, geography, poverty, school type)
3. Produce media-ready findings + exportable data for a scrollytelling website

### What We're NOT Building

- We are NOT replicating Alerta Escuela's exact model (we don't have SIAGIE access)
- We are NOT building a web application in this phase (M4 site is a separate repo)
- We are NOT doing real-time prediction or deployment
- We are NOT using deep learning (no TabNet, no neural networks)

### Core Priority

**The fairness audit is the product.** Models exist to be audited, not to achieve SOTA. If the LightGBM gets 0.85 AUC, great. If it gets 0.75, also fine — the fairness gaps are the finding, not the accuracy.

### End Consumer

Peruvian journalists (Ojo Público, El Comercio, La República). The exported data feeds a Next.js scrollytelling site (separate repo). Everything we produce must be explainable to a non-technical audience in Spanish.

---

## 2. TECH STACK (LOCKED — DO NOT CHANGE)

| Layer | Tool | Version | Notes |
|---|---|---|---|
| Language | Python | 3.12 | Managed via Nix flakes |
| Package manager | uv | latest | Already configured in repo |
| Data processing | polars | >=1.0 | Primary. Convert to pandas ONLY at sklearn/fairlearn boundary |
| ML framework | scikit-learn | >=1.5 | Logistic regression, calibration, metrics |
| Gradient boosting | lightgbm | >=4.0 | Primary model (matches Alerta Escuela) |
| Gradient boosting | xgboost | >=2.0 | Comparison model only |
| Fairness | fairlearn | >=0.11 | MetricFrame with sample_weight |
| Interpretability | shap | >=0.45 | TreeExplainer for LightGBM |
| Statistics | statsmodels | latest | Survey-weighted descriptive stats |
| Visualization | matplotlib + seaborn | latest | Paper-quality static figures |
| Visualization | plotly | latest | Interactive plots in notebooks |
| ONNX export | onnx + onnxmltools | latest | LightGBM → ONNX for M4 site demo |
| Testing | pytest | latest | Gate validation tests |
| Linting | ruff | latest | Already in Nix config |
| Notebooks | jupyterlab | latest | EDA and narrative |

### Libraries NOT to use

- **pandas** — Use polars. Convert with `.to_pandas()` only when passing to sklearn/fairlearn/shap
- **TabNet / PyTorch / TensorFlow** — No neural networks. Three models only: LogReg, LightGBM, XGBoost
- **MLflow / DVC / Weights & Biases** — No experiment tracking tools. Export metrics as JSON
- **AIF360** — Use fairlearn exclusively
- **geopandas / GDAL** — No geospatial processing libraries. All geo data is pre-aggregated CSV/JSON
- **dask / spark** — Dataset fits in memory (~180K rows)

---

## 3. DIRECTORY STRUCTURE (ENFORCE EXACTLY)

```
alerta-escuela-audit/
├── pyproject.toml
├── flake.nix
├── README.md
├── .planning/                    # GSD planning directory
│   └── ...
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── enaho.py              # ENAHO loading, merging, harmonization
│   │   ├── admin.py              # District-level dropout rates
│   │   ├── census.py             # Census 2017 district enrichment
│   │   ├── nightlights.py        # VIIRS district-level economic proxy
│   │   └── features.py           # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # Train all 3 models
│   │   ├── evaluate.py           # Survey-weighted metrics + JSON export
│   │   ├── calibrate.py          # Platt/isotonic calibration
│   │   └── export_onnx.py        # LightGBM → ONNX
│   ├── fairness/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Subgroup fairness metrics via fairlearn
│   │   ├── shap_analysis.py      # Global + regional + interaction SHAP
│   │   └── distill.py            # Distill findings → M4 JSON exports
│   └── utils.py                  # UBIGEO padding, constants, paths, survey weight helpers
├── notebooks/
│   ├── 01_enaho_exploration.ipynb
│   ├── 02_descriptive_gaps.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_fairness_audit.ipynb
├── tests/
│   ├── conftest.py               # Shared fixtures (sample data, paths)
│   ├── gates/
│   │   ├── test_gate_1_1.py      # Target variable validation
│   │   ├── test_gate_1_2.py      # Cross-year consistency
│   │   ├── test_gate_1_3.py      # Spatial merge
│   │   ├── test_gate_1_4.py      # Enrichment merge
│   │   ├── test_gate_1_5.py      # Feature + EDA
│   │   ├── test_gate_2_1.py      # Baseline model
│   │   ├── test_gate_2_2.py      # LightGBM + XGBoost
│   │   ├── test_gate_2_3.py      # Calibration + ONNX
│   │   ├── test_gate_3_1.py      # Subgroup fairness
│   │   ├── test_gate_3_2.py      # SHAP
│   │   ├── test_gate_3_3.py      # Cross-validation admin data
│   │   └── test_gate_3_4.py      # Distillation + exports
│   └── unit/
│       ├── test_enaho_loader.py
│       ├── test_harmonization.py
│       └── test_ubigeo.py
├── data/
│   ├── raw/                      # .gitignored — user places ENAHO ZIPs here manually
│   │   ├── enaho/
│   │   │   ├── 2018/             # Contains Enaho01a-2018-200.csv, Enaho01a-2018-300.csv
│   │   │   ├── 2019/
│   │   │   ├── 2020/
│   │   │   ├── 2021/
│   │   │   ├── 2022/
│   │   │   ├── 2023/
│   │   │   └── 2024/
│   │   ├── admin/                # District dropout CSVs from datosabiertos
│   │   ├── census/               # Census 2017 district aggregates
│   │   ├── nightlights/          # Pre-aggregated VIIRS district data
│   │   └── escolar/              # Censo Escolar aggregates
│   ├── processed/                # .gitignored — intermediate cleaned datasets
│   │   ├── enaho_merged.parquet
│   │   ├── enaho_with_features.parquet
│   │   └── full_dataset.parquet
│   └── exports/                  # COMMITTED — JSON exports for M4 site
│       ├── findings.json
│       ├── fairness_metrics.json
│       ├── shap_values.json
│       ├── choropleth.json
│       ├── model_results.json
│       ├── descriptive_tables.json
│       ├── onnx/
│       │   └── lightgbm_dropout.onnx
│       └── README.md             # Documents each export's schema and provenance
├── outputs/
│   └── figures/                  # Generated plots for notebooks/paper
└── .gitignore
```

### .gitignore (create this)

```
data/raw/
data/processed/
*.pyc
__pycache__/
.ipynb_checkpoints/
.DS_Store
*.egg-info/
dist/
build/
.env
```

---

## 4. DATA SOURCES — COMPLETE SPECIFICATIONS

### 4.1 ENAHO Microdata (PRIMARY — Individual Level)

**Source:** INEI (user downloads manually and places in `data/raw/enaho/{YEAR}/`)

**Years:** 2018, 2019, 2020, 2021, 2022, 2023, 2024 (7 years)

**Files per year:**
- `Enaho01a-{YEAR}-300.csv` — Education module
- `Enaho01a-{YEAR}-200.csv` — Household members module

**CRITICAL FORMAT QUIRK:**
- 2018–2019: pipe `|` delimiter
- 2020–2024: comma `,` delimiter
- Auto-detect using: `sep='|' if year <= 2019 else ','`

**Merge keys between Module 200 and Module 300:**
`CONGLOME`, `VIVIENDA`, `HOESSION` (may appear as `HOESSION` or similar — verify in first year loaded), `CODPERSO`

**IMPORTANT:** Column names may have slight variations across years (case, extra spaces). Normalize all column names with `.strip().upper()` after loading.

**School-age filter:** Ages 6–17 (construct from birth date or age column in Module 200)

#### Key columns from Module 300 (Education):

| Column | Description | Values | Notes |
|---|---|---|---|
| P300A | Mother tongue | 1=Quechua, 2=Aimara, 3=Otra lengua nativa, 4=Castellano, 6=Inglés, 7=Portugués, 8=Otra lengua extranjera, 9=Sordomudo, 10=Asháninka, 11=Awajún, 12=Shipibo-Konibo, 13=Shawi, 14=Matsigenka, 15=Achuar, 99=No sabe/NR | Codes 10–15 only exist from 2020+ |
| P303 | Enrolled previous year | 1=Yes, 2=No | |
| P306 | Currently enrolled | 1=Yes, 2=No | |
| P308A | Grade currently attending | Numeric | Often blank if not enrolled |
| P301A | Highest education level attained | Varies | |
| FACTOR07 | Survey expansion factor (weight) | Float >0 | NEVER null. Use this for all weighted computations |

#### Key columns from Module 200 (Members):

| Column | Description | Values |
|---|---|---|
| P207 | Sex | 1=Male, 2=Female |
| P208A | Age | Integer |
| UBIGEO | 6-digit district code | String (may need zero-padding) |
| ESTRATO | Stratum (urban/rural proxy) | Varies |

#### Dropout target construction:

```python
# EXACT LOGIC — DO NOT MODIFY
DROPOUT = (P303 == 1) & (P306 == 2)
# Translation: "Was enrolled last year AND is NOT enrolled this year"
```

#### Expected validation values (2023):

| Metric | Expected Value | Tolerance |
|---|---|---|
| School-age sample (6–17) | ~25,000–26,000 | ±2,000 |
| Unweighted dropout count | ~3,500 | ±500 |
| Weighted dropout rate | ~13–15% | ±2% |
| P300A null rate (ages 6–17) | <1% | |

#### P300A Harmonization Logic (EXACT — implement verbatim):

```python
def harmonize_p300a(code: int) -> int:
    """Collapse post-2020 disaggregated indigenous codes back to code 3.

    In 2020, INEI disaggregated code 3 ("Otra lengua nativa") into:
    10=Asháninka, 11=Awajún, 12=Shipibo-Konibo, 13=Shawi, 14=Matsigenka, 15=Achuar.
    This is a coding change, NOT a population shift. Sum of 3+10–15 is stable across years.

    For cross-year analysis, collapse 10–15 back to 3.
    For 2020+ deep-dive analysis, keep disaggregated codes.
    """
    if code in (10, 11, 12, 13, 14, 15):
        return 3
    if code == 99 or code is None:
        return 99
    return code
```

**Preserve BOTH versions:** `P300A_harmonized` (for cross-year) and `P300A_original` (for disaggregated 2020+ analysis).

#### Analytical language groupings:

| Group Label | Codes | Use |
|---|---|---|
| `castellano` | 4 | All years |
| `quechua` | 1 | All years |
| `aimara` | 2 | All years |
| `other_indigenous` | 3 (harmonized: 3+10–15) | All years, cross-year analysis |
| `ashaninka` | 10 | 2020–2024 disaggregated only |
| `awajun` | 11 | 2020–2024 disaggregated only |
| `shipibo_konibo` | 12 | 2020–2024 disaggregated only |
| `shawi` | 13 | 2020–2024 disaggregated only |
| `matsigenka` | 14 | 2020–2024 disaggregated only |
| `achuar` | 15 | 2020–2024 disaggregated only |
| `foreign` | 6, 7 | All years |
| `other_unknown` | 8, 9, 99, null | All years |

### 4.2 District Dropout Rates (Administrative Ground Truth)

**Source:** `datosabiertos.gob.pe` → "Tasa y número de desertores en Educación Primaria y Secundaria 2023/2024"

**Location:** `data/raw/admin/`

**Schema:** `ubigeo, Departamento, Provincia, Distrito, desertor, denominador, Tasa`

**CRITICAL FIX:** Zero-pad UBIGEO: `str(ubigeo).zfill(6)` — leading zeros stripped for departments 01–09 (Amazonas, Áncash, Apurímac, Arequipa, Ayacucho, Cajamarca, Cusco, Huancavelica, Huánuco).

**Expected values:**

| Metric | Primaria | Secundaria |
|---|---|---|
| Districts | ~1,890 | ~1,846 |
| Departments | 25 | 25 |
| Mean dropout rate | ~0.93% | ~2.05% |
| Zero-dropout districts | ~538 | ~414 |
| Nulls | 0 | 0 |

### 4.3 Census 2017 (District-Level Enrichment)

**Source:** INEI Census 2017 district-level aggregates
**Location:** `data/raw/census/`
**Merge key:** UBIGEO (zero-padded to 6 digits)
**Use:** Poverty indices, indigenous language prevalence, access to services

### 4.4 VIIRS Nighttime Lights (Economic Proxy)

**Source:** Pre-aggregated district-level data (Jiaxiong Yao research site or similar)
**Location:** `data/raw/nightlights/`
**Merge key:** UBIGEO
**Expected:** No negative values. Coverage >90% of districts.
**Use:** Economic activity proxy for rural areas where income data is sparse

### 4.5 Censo Escolar (School/District Aggregates)

**Source:** `datosabiertos.gob.pe/dataset/censo-escolar` and ESCALE
**Location:** `data/raw/escolar/`
**Merge key:** UBIGEO
**Use:** Teacher counts, school infrastructure, student-teacher ratios

---

## 5. FEATURE ENGINEERING SPECIFICATION

### Target Variable

```python
# Binary classification target
DROPOUT: int  # 1 if (P303 == 1 AND P306 == 2), else 0
```

### Feature Proxy Mapping (Alerta Escuela → ENAHO)

Each feature below MUST be created in `src/data/features.py`. Use these exact column names in the output DataFrame:

| Feature Name | Source | Construction | Type |
|---|---|---|---|
| `age` | Module 200 P208A | Direct | int |
| `es_mujer` | Module 200 P207 | `1 if P207 == 2 else 0` | binary |
| `p300a_harmonized` | Module 300 P300A | Harmonization function | int |
| `p300a_original` | Module 300 P300A | Raw value | int |
| `lang_castellano` | P300A | `1 if P300A == 4 else 0` | binary |
| `lang_quechua` | P300A | `1 if P300A == 1 else 0` | binary |
| `lang_aimara` | P300A | `1 if P300A == 2 else 0` | binary |
| `lang_other_indigenous` | P300A harmonized | `1 if P300A_harmonized == 3 else 0` | binary |
| `lang_foreign` | P300A | `1 if P300A in (6, 7) else 0` | binary |
| `es_peruano` | Module 200 birthplace/nationality | `1 if Peruvian, 0 if foreign` | binary |
| `rural` | ESTRATO or district classification | Binary urban/rural | binary |
| `parent_education_years` | Module 200 parent/guardian fields | Years of education | float |
| `is_working` | Employment module | `1 if currently working` | binary |
| `juntos_participant` | Social program module | `1 if household receives JUNTOS` | binary |
| `log_income` | Income/expenditure module | `log(household_income + 1)` | float |
| `has_disability` | Disability module | `1 if any disability reported` | binary |
| `ubigeo` | Module 200 | Zero-padded 6-digit string | string (not a model feature — merge key) |
| `year` | Derived | Survey year | int (not a model feature — split key) |
| `factor07` | Module 300 FACTOR07 | Survey weight | float (not a model feature — weight) |
| `region_natural` | Derived from UBIGEO first 2 digits | Costa/Sierra/Selva classification | categorical |
| `department` | UBIGEO[:2] | Department code | categorical |
| `district_dropout_rate_admin` | Admin data merge | Administrative dropout rate | float |
| `nightlight_intensity` | VIIRS merge | Mean radiance | float |
| `poverty_index` | Census merge | District poverty index | float |
| `school_student_teacher_ratio` | Escolar merge | District avg student-teacher ratio | float |

### Features NOT available (document in code comments):

- Z-score math/language grades (requires SIAGIE)
- Prior dropout history (requires longitudinal tracking, ENAHO is cross-sectional)
- Prior enrollment cost (SIAGIE-only)
- Birth proximity to school (requires SIAGIE geolocation)

### Feature matrix for model training:

```python
# Features that enter the model (exclude keys, weights, and string identifiers)
MODEL_FEATURES = [
    'age', 'es_mujer', 'lang_castellano', 'lang_quechua', 'lang_aimara',
    'lang_other_indigenous', 'lang_foreign', 'es_peruano', 'rural',
    'parent_education_years', 'is_working', 'juntos_participant',
    'log_income', 'has_disability', 'region_natural_encoded',
    'district_dropout_rate_admin', 'nightlight_intensity',
    'poverty_index', 'school_student_teacher_ratio'
]

# Protected attributes for fairness analysis (NOT excluded from model — we audit their effect)
PROTECTED_ATTRIBUTES = ['es_mujer', 'p300a_harmonized', 'rural', 'region_natural', 'log_income']

# Columns that are NOT model features but needed for analysis
META_COLUMNS = ['ubigeo', 'year', 'factor07', 'p300a_original', 'department', 'dropout']
```

---

## 6. MODELING SPECIFICATION

### Temporal Split (ENFORCE STRICTLY)

```python
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]
VALIDATE_YEAR = 2023
TEST_YEAR = 2024
```

**NEVER use test data for any decision.** Hyperparameter tuning, threshold selection, and model comparison all use the validation set. Test set is touched ONCE for final evaluation.

### Models (exactly 3, no more)

#### Model 1: Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    ))
])
```

**Purpose:** Interpretable baseline. Odds ratios are media-friendly. If fairness gaps appear here, they're structural, not algorithmic.

#### Model 2: LightGBM (Primary — matches Alerta Escuela)

```python
import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'metric': 'average_precision',  # PR-AUC, not ROC-AUC
    'is_unbalance': True,           # Handle class imbalance
    'verbosity': -1,
    'random_state': 42,
    'n_estimators': 500,            # Tune via early stopping on validation
    'learning_rate': 0.05,          # Tune: [0.01, 0.05, 0.1]
    'num_leaves': 31,               # Tune: [15, 31, 63]
    'min_child_samples': 50,        # Conservative given ~14% positive rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

**Tuning:** Use Optuna with 50 trials on validation set. Search space: `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.

#### Model 3: XGBoost (Comparison)

```python
import xgboost as xgb

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'scale_pos_weight': (n_negative / n_positive),
    'random_state': 42,
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

**Purpose:** If fairness gaps appear in both LightGBM and XGBoost, findings are algorithm-independent.

### Evaluation Metrics (ALL must be survey-weighted)

```python
# Primary metric: PR-AUC (not ROC-AUC) — because of class imbalance
# Secondary: F1, Precision, Recall at tuned threshold
# Calibration: Brier score

# CRITICAL: All metrics must pass sample_weight=FACTOR07
from sklearn.metrics import (
    average_precision_score,  # PR-AUC
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    precision_recall_curve
)

def evaluate_model(y_true, y_pred_proba, y_pred_binary, sample_weight):
    """All metrics are survey-weighted."""
    return {
        'pr_auc': average_precision_score(y_true, y_pred_proba, sample_weight=sample_weight),
        'f1': f1_score(y_true, y_pred_binary, sample_weight=sample_weight),
        'precision': precision_score(y_true, y_pred_binary, sample_weight=sample_weight),
        'recall': recall_score(y_true, y_pred_binary, sample_weight=sample_weight),
        'brier': brier_score_loss(y_true, y_pred_proba, sample_weight=sample_weight),
        'fnr': 1 - recall_score(y_true, y_pred_binary, sample_weight=sample_weight),
    }
```

### Threshold Tuning

Do NOT use 0.5 as default. Tune threshold on validation set to maximize F1. Report metrics at thresholds 0.3, 0.4, 0.5, 0.6, 0.7 for transparency.

### Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate on validation set
calibrated_lgb = CalibratedClassifierCV(
    best_lgb_model,
    method='isotonic',  # Try both 'sigmoid' and 'isotonic', keep better Brier score
    cv='prefit'         # Model already trained
)
calibrated_lgb.fit(X_validate, y_validate, sample_weight=w_validate)
```

### ONNX Export

```python
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

initial_type = [('features', FloatTensorType([None, len(MODEL_FEATURES)]))]
onnx_model = convert_lightgbm(best_lgb_model, initial_types=initial_type)

# Save
with open('data/exports/onnx/lightgbm_dropout.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# VALIDATE: predictions must match Python model
import onnxruntime as rt
sess = rt.InferenceSession('data/exports/onnx/lightgbm_dropout.onnx')
onnx_preds = sess.run(None, {'features': X_test_sample.astype(np.float32)})[1][:, 1]
python_preds = best_lgb_model.predict_proba(X_test_sample)[:, 1]
assert np.max(np.abs(onnx_preds - python_preds)) < 1e-5
```

---

## 7. FAIRNESS AUDIT SPECIFICATION

### Protected Attribute Dimensions (6 total)

| Dimension | Column | Groups | Minimum sample per group |
|---|---|---|---|
| Sex | `es_mujer` | Male (0), Female (1) | 100 unweighted |
| Language | `p300a_harmonized` | castellano, quechua, aimara, other_indigenous | 100 unweighted |
| Language (disagg.) | `p300a_original` | All 2020+ codes | 50 unweighted (flag if smaller) |
| Geography | `rural` | Urban (0), Rural (1) | 100 unweighted |
| Region | `region_natural` | Costa, Sierra, Selva | 100 unweighted |
| Poverty | `log_income` binned into quintiles | Q1 (poorest) through Q5 (wealthiest) | 100 unweighted |

### Intersectional Analysis (mandatory)

Compute metrics for these intersections:
- `language × rurality` (e.g., Quechua + Rural vs Castellano + Urban)
- `sex × poverty quintile`
- `language × region` (e.g., Indigenous + Selva vs Castellano + Costa)

Flag any intersection with <50 unweighted observations. Report but caveat.

### Fairness Metrics (per subgroup, survey-weighted)

```python
from fairlearn.metrics import MetricFrame

# Core metrics to compute per subgroup
FAIRNESS_METRICS = {
    'tpr': lambda y, p, sw: recall_score(y, p, sample_weight=sw),           # True Positive Rate (recall)
    'fpr': lambda y, p, sw: fpr_score(y, p, sw),                            # False Positive Rate
    'fnr': lambda y, p, sw: 1 - recall_score(y, p, sample_weight=sw),       # False Negative Rate (missed dropouts)
    'precision': lambda y, p, sw: precision_score(y, p, sample_weight=sw),   # Predictive parity
    'pr_auc': lambda y, pp, sw: average_precision_score(y, pp, sample_weight=sw),  # Discrimination ability
}

# Use MetricFrame
mf = MetricFrame(
    metrics={'recall': recall_score, 'precision': precision_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_df,
    sample_params={'recall': {'sample_weight': weights}, 'precision': {'sample_weight': weights}}
)

# Key outputs:
# mf.by_group         → per-group metrics table
# mf.difference()     → max gap between groups
# mf.ratio()          → min ratio between groups
```

### Calibration by Group

For each subgroup: among students predicted as "high risk" (>0.7 probability), what is the actual dropout rate? If the model says "70% risk" for a rural indigenous student but the true rate is 30%, that's a calibration fairness failure.

```python
# Per-group calibration
for group in groups:
    mask = (sensitive_features == group)
    high_risk_mask = mask & (y_pred_proba > 0.7)
    if high_risk_mask.sum() > 30:
        actual_rate = np.average(y_true[high_risk_mask], weights=weights[high_risk_mask])
        # Record: group, predicted_risk=0.7+, actual_rate
```

### Specific Audit Questions (map to findings)

1. **ES_PERUANO effect:** What is the SHAP value magnitude for nationality? Is non-Peruvian status independently increasing predicted risk beyond what poverty/geography explain?
2. **ES_MUJER in secundaria:** Isolate to secondary school ages (12–17). Is the gender effect present? What's the FNR gap between male and female?
3. **FNR by language:** What percentage of actual indigenous-language dropouts does the model miss vs. Castellano dropouts?
4. **Calibration by region:** Does "high risk" mean the same thing in Lima vs. Amazonas vs. Loreto?
5. **Poverty quintile interaction:** Does the model work equally well for Q1 (poorest) and Q5 (wealthiest)?

---

## 8. SHAP ANALYSIS SPECIFICATION

### Global SHAP

```python
import shap

explainer = shap.TreeExplainer(best_lgb_model)
# Compute on TEST SET (2024), not train
shap_values = explainer.shap_values(X_test)

# If LightGBM returns list (binary classification), take shap_values[1] for positive class
```

### Regional SHAP Comparison

Split test set by `region_natural`. Compute SHAP for each region separately. Compare feature importance rankings.

**Expected finding:** Mother tongue SHAP is higher in Sierra/Selva than in Lima/Costa.

### Interaction Effects

```python
# Compute SHAP interaction values (expensive — subsample if needed)
shap_interaction = explainer.shap_interaction_values(X_test_subsample)
# Focus on: poverty × language, rurality × sex
```

### Export for M4

For the interactive SHAP waterfall demo on the site, export:
- Mean absolute SHAP values per feature (global importance)
- SHAP values for 10 representative student profiles (diverse across regions/languages)
- Feature names and display labels in Spanish

---

## 9. ROADMAP — GSD PHASES

Each phase maps to a GSD `/gsd:execute-phase N` command. Tasks use GSD's XML format.

### Phase 1: ENAHO Single-Year Loader (M1.1)

**Requirements:** REQ-DATA-01, REQ-DATA-02

<tasks>
<task type="auto">
<name>Create ENAHO single-year loader</name>
<files>src/data/enaho.py, src/utils.py</files>
<action>
Implement `load_enaho_year(year: int, data_dir: str = 'data/raw/enaho') -> pl.DataFrame`.
- Reads Module 300 and Module 200 CSVs from `data_dir/{year}/`
- Auto-detects delimiter: pipe for <=2019, comma for >=2020
- Normalizes column names: `.strip().upper()`
- Merges Module 200 and Module 300 on household+person key columns
- Filters to school-age (6-17) based on age from Module 200
- Constructs DROPOUT target: (P303 == 1) & (P306 == 2) → 1, else 0
- Zero-pads UBIGEO to 6 digits
- Returns single merged DataFrame with columns from both modules

In utils.py: implement `pad_ubigeo(code) -> str` that zero-pads to 6 digits.
</action>
<verify>pytest tests/gates/test_gate_1_1.py passes all assertions</verify>
<done>load_enaho_year(2023) returns DataFrame with ~25K rows, ~3500 dropouts, ~14% weighted rate</done>
</task>

<task type="auto">
<name>Write Gate 1.1 validation tests</name>
<files>tests/gates/test_gate_1_1.py</files>
<action>
Write pytest tests that validate:
- Unweighted dropout count for 2023 is between 3000 and 4000
- Weighted dropout rate (using FACTOR07) is between 0.12 and 0.16
- Total school-age rows between 23000 and 28000
- Zero nulls in P303, P306, FACTOR07 for the filtered population
- P207 (sex) has exactly 2 unique non-null values
- P300A has >99% non-null rate for ages 6-17
- All UBIGEO values are exactly 6 characters long

Mark the test file with: `# GATE 1.1 — STOP AND REVIEW BEFORE PROCEEDING`
Print 10 random dropout rows at end for human inspection.
</action>
<verify>All assertions pass when run against 2023 ENAHO data</verify>
<done>Gate 1.1 test file exists and passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 1.1:** After this phase, the human reviews the 10 printed dropout rows. Do they look like real students? Proceed only after human confirmation.

---

### Phase 2: Multi-Year Loader + Harmonization (M1.2)

**Requirements:** REQ-DATA-03, REQ-DATA-04

<tasks>
<task type="auto">
<name>Extend ENAHO loader to all 7 years with P300A harmonization</name>
<files>src/data/enaho.py</files>
<action>
Add to enaho.py:
- `harmonize_p300a(code: int) -> int` — EXACT logic from Section 4.1 of this spec
- `load_all_years(data_dir: str = 'data/raw/enaho') -> pl.DataFrame` — loads 2018-2024,
  stacks with a `year` column, adds `p300a_harmonized` and `p300a_original`
- Save result as parquet: `data/processed/enaho_merged.parquet`

The function should print per-year: rows loaded, dropout count, delimiter used.
</action>
<verify>pytest tests/gates/test_gate_1_2.py passes</verify>
<done>7 years loaded, P300A harmonized, pooled dataset has ~150K-180K rows, ~20K+ dropouts</done>
</task>

<task type="auto">
<name>Write Gate 1.2 validation tests</name>
<files>tests/gates/test_gate_1_2.py</files>
<action>
Validate:
- All 7 years load without exceptions
- Column names are identical across all years after normalization
- P300A code 3 count: >2000 for pre-2020 years, <1000 for post-2020 years
- Sum of codes 3+10+11+12+13+14+15: stable across years (within 30% of each other)
- No single year has >2x the row count of another (no data corruption)
- Pooled row count between 140000 and 200000
- Pooled unweighted dropout count > 18000
- Confirm 2019 uses pipe delimiter, 2020 uses comma
</action>
<verify>All tests pass</verify>
<done>Gate 1.2 passes — cross-year consistency confirmed</done>
</task>
</tasks>

**Gate 1.2: Fully automatable.** No human review needed.

---

### Phase 3: Spatial + Supplementary Data Merges (M1.3 + M1.4)

**Requirements:** REQ-DATA-05, REQ-DATA-06, REQ-DATA-07, REQ-DATA-08

<tasks>
<task type="auto">
<name>Load and merge district admin dropout rates</name>
<files>src/data/admin.py</files>
<action>
Implement `load_admin_dropout_rates(data_dir: str = 'data/raw/admin') -> pl.DataFrame`.
- Load primaria and secundaria CSVs
- Zero-pad UBIGEO
- Return combined DataFrame with columns: ubigeo, level (primaria/secundaria),
  desertor, denominador, tasa
</action>
<verify>Mean rates ~0.93% primaria, ~2.05% secundaria, 25 departments</verify>
<done>Admin data loaded with expected statistics</done>
</task>

<task type="auto">
<name>Load supplementary data sources and merge all</name>
<files>src/data/census.py, src/data/nightlights.py, src/data/features.py</files>
<action>
Implement loaders for Census 2017 and VIIRS nightlights (both district-level CSVs).
Each returns a polars DataFrame with UBIGEO as merge key.

In features.py, implement `merge_all_sources(enaho_df, admin_df, census_df, nightlights_df) -> pl.DataFrame`:
- All merges are LEFT JOINS on UBIGEO (ENAHO is the base — no rows should be gained or lost)
- After merge, row count must equal input enaho_df row count
- Save to `data/processed/full_dataset.parquet`
</action>
<verify>pytest tests/gates/test_gate_1_3.py and test_gate_1_4.py pass</verify>
<done>Full merged dataset with supplementary columns, no row duplication</done>
</task>

<task type="auto">
<name>Write Gate 1.3 + 1.4 validation tests</name>
<files>tests/gates/test_gate_1_3.py, tests/gates/test_gate_1_4.py</files>
<action>
Gate 1.3:
- All UBIGEO exactly 6 chars
- Admin mean rates within 20% of expected (0.93% primaria, 2.05% secundaria)
- 25 unique departments
- ENAHO-to-admin merge rate > 0.85
- Row count unchanged after merge

Gate 1.4:
- Census merge rate > 0.90
- Nightlights: no negatives, coverage > 0.85
- No duplicate rows (check with .is_duplicated())
- Final column count is sum of unique columns from all sources (minus duplicate keys)
- Print list of columns with >10% nulls for human review
</action>
<verify>All gate tests pass</verify>
<done>Spatial and enrichment merges validated</done>
</task>
</tasks>

**🛑 HUMAN GATE 1.3/1.4:** Human reviews: (1) Are the 5 spot-checked districts directionally correct (Lima low dropout, Amazonas high)? (2) Are >10% null columns expected or bugs?

---

### Phase 4: Feature Engineering + Descriptive Statistics (M1.5)

**Requirements:** REQ-DATA-09, REQ-ANALYSIS-01

<tasks>
<task type="auto">
<name>Build feature engineering pipeline</name>
<files>src/data/features.py</files>
<action>
Implement `build_features(df: pl.DataFrame) -> pl.DataFrame`:
- Creates ALL features listed in Section 5 of this spec
- Uses exact column names from the spec
- Language dummies from P300A
- Rural from ESTRATO or district classification
- Income quintiles from log_income (weighted quintile breaks using FACTOR07)
- Region natural derived from first 2 digits of UBIGEO (map department codes to Costa/Sierra/Selva)

Implement region mapping:
- Costa: departments 07, 08, 09, 13, 14, 20, 23, 25 (Callao, Cusco varies — use standard INEI mapping)
- Sierra: departments 03, 04, 05, 06, 09, 10, 11, 12, 15, 18
- Selva: departments 01, 16, 17, 22, 24, 25
NOTE: Verify exact mapping from INEI — some departments span multiple regions.
Use the standard "región natural" classification from INEI.

Save: `data/processed/enaho_with_features.parquet`
</action>
<verify>Feature matrix has 19+ model features, all binaries are 0/1, quintiles have 5 groups</verify>
<done>Feature pipeline complete</done>
</task>

<task type="auto">
<name>Compute descriptive statistics and initial fairness gaps</name>
<files>notebooks/02_descriptive_gaps.ipynb, src/data/features.py</files>
<action>
In the notebook, compute and visualize (save all figures to outputs/figures/):
1. Weighted dropout rate by mother tongue (harmonized). Include Awajún 2020+ disaggregated.
   EXPECTED: Awajún ~22% vs Castellano ~14% (63% gap)
2. Weighted dropout rate by sex × education level (primaria vs secundaria)
3. Weighted dropout rate by rural vs urban
4. Weighted dropout rate by region natural (Costa/Sierra/Selva)
5. Weighted dropout rate by poverty quintile
6. Heatmap: language × rurality dropout rates
7. Choropleth prep: district-level weighted dropout rates (export for later mapping)

All computations MUST use FACTOR07 as weights via:
```python
# Weighted mean using polars
df.group_by('group_col').agg(
    (pl.col('dropout') * pl.col('factor07')).sum() / pl.col('factor07').sum()
)
```

Export descriptive tables to `data/exports/descriptive_tables.json`.
</action>
<verify>pytest tests/gates/test_gate_1_5.py passes</verify>
<done>Descriptive gaps quantified, Awajún gap confirmed, tables exported</done>
</task>

<task type="auto">
<name>Write Gate 1.5 validation tests</name>
<files>tests/gates/test_gate_1_5.py</files>
<action>
Validate:
- Feature matrix has >= 19 model features
- All binary features contain only values {0, 1} (no nulls unless documented)
- Poverty quintile has exactly 5 groups with roughly equal weighted population (±30%)
- Awajún (code 11) weighted non-enrollment rate > 0.18 for 2020+ years
- Castellano (code 4) weighted non-enrollment rate between 0.10 and 0.18
- No feature has > 0.30 null rate
- No pair of features has Pearson correlation > 0.95
- descriptive_tables.json exists and is valid JSON
</action>
<verify>All tests pass</verify>
<done>Gate 1.5 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 1.5:** Human reviews: (1) Weighted dropout rates by subgroup — do directions match Peruvian education reality? (2) Do the 7 visualizations tell a coherent story? (3) Is the Awajún gap real or a sample size artifact?

---

### Phase 5: Baseline Model + Temporal Splits (M2.1)

**Requirements:** REQ-MODEL-01, REQ-MODEL-02

<tasks>
<task type="auto">
<name>Create temporal splits and train logistic regression baseline</name>
<files>src/models/train.py, src/models/evaluate.py</files>
<action>
In train.py:
- `create_temporal_splits(df) -> (train, validate, test)`: Split by year column.
  Train=2018-2022, Validate=2023, Test=2024.
  ASSERT: train years max == 2022, validate year == 2023, test year == 2024.
- `train_logistic_regression(X_train, y_train, sample_weight) -> Pipeline`:
  As specified in Section 6.

In evaluate.py:
- `evaluate_model(y_true, y_pred_proba, y_pred, sample_weight) -> dict`:
  Returns dict with pr_auc, f1, precision, recall, brier, fnr.
  ALL metrics use sample_weight.
- `evaluate_at_thresholds(y_true, y_pred_proba, sample_weight, thresholds=[0.3,0.4,0.5,0.6,0.7]) -> dict`:
  Returns metrics at each threshold.
- Export results to JSON in `data/exports/model_results.json`.

Train logistic regression on train set, evaluate on validate set.
Also compute and store unweighted metrics separately — assert they differ from weighted ones.
</action>
<verify>pytest tests/gates/test_gate_2_1.py passes</verify>
<done>LR baseline trained, PR-AUC > 0.14 (base rate), weighted ≠ unweighted metrics</done>
</task>

<task type="auto">
<name>Write Gate 2.1 validation tests</name>
<files>tests/gates/test_gate_2_1.py</files>
<action>
Validate:
- Train set years: max == 2022, min == 2018
- Validate set years: all == 2023
- Test set years: all == 2024
- No overlap between splits (set intersection is empty)
- Class rate in each split is between 0.10 and 0.20
- LR converged (check .n_iter_ < max_iter)
- PR-AUC on validation > 0.14 (must beat random)
- Weighted F1 != unweighted F1 (weights are applied)
- model_results.json exists, is valid JSON, contains 'logistic_regression' key
- Print LR coefficients with feature names for human review
</action>
<verify>All tests pass</verify>
<done>Gate 2.1 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 2.1:** Human reviews LR coefficients. Do poverty ↑ risk, urban ↓ risk, age effects make sense? Are odds ratios for sex, language, rurality directionally correct?

---

### Phase 6: LightGBM + XGBoost (M2.2)

**Requirements:** REQ-MODEL-03, REQ-MODEL-04

<tasks>
<task type="auto">
<name>Train and tune LightGBM + XGBoost</name>
<files>src/models/train.py</files>
<action>
Add to train.py:
- `train_lightgbm(X_train, y_train, X_val, y_val, sample_weight_train, sample_weight_val) -> lgb.Booster`:
  Use Optuna for hyperparameter tuning (50 trials). Optimize PR-AUC on validation set.
  Early stopping using validation set.
  Search space from Section 6.

- `train_xgboost(X_train, y_train, X_val, y_val, sample_weight_train, sample_weight_val) -> xgb.Booster`:
  Similar setup, optimize PR-AUC.

Save trained models as pickle (for Python use) alongside ONNX export.
Evaluate both on validation set. Add to model_results.json.
Tune optimal threshold per model using F1 on validation set.
Export validation set predictions (actual, predicted_proba, predicted_binary) for all rows —
needed for M3 fairness analysis.
</action>
<verify>pytest tests/gates/test_gate_2_2.py passes</verify>
<done>LightGBM PR-AUC > LR, XGBoost within 3% of LightGBM, no feature >50% importance</done>
</task>

<task type="auto">
<name>Write Gate 2.2 validation tests</name>
<files>tests/gates/test_gate_2_2.py</files>
<action>
Validate:
- LightGBM PR-AUC > logistic regression PR-AUC
- abs(LightGBM PR-AUC - XGBoost PR-AUC) < 0.05
- LightGBM max single feature importance < 0.50
- Models evaluated on validation set only (assert test set not used)
- Metrics at 3+ thresholds reported per model
- Validation predictions exported (row count matches validation set)
- model_results.json updated with lightgbm and xgboost entries
- Print top-10 feature importances for human review
</action>
<verify>All tests pass</verify>
<done>Gate 2.2 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 2.2:** Human reviews feature importances. Are age, poverty, rurality in top 5? Is anything suspicious?

---

### Phase 7: Calibration + ONNX Export + Final Test (M2.3)

**Requirements:** REQ-MODEL-05, REQ-MODEL-06, REQ-EXPORT-01

<tasks>
<task type="auto">
<name>Calibrate best model, run final test evaluation, export ONNX</name>
<files>src/models/calibrate.py, src/models/export_onnx.py</files>
<action>
In calibrate.py:
- Calibrate best LightGBM on validation set using both 'sigmoid' and 'isotonic'
- Keep whichever has lower Brier score on validation
- Evaluate calibrated model on TEST SET (2024) — this is the ONLY time test is used
- Save calibration plot to outputs/figures/

In export_onnx.py:
- Convert best LightGBM to ONNX as specified in Section 6
- Validate: max absolute prediction difference < 1e-5 on 100 random test samples
- Save to data/exports/onnx/lightgbm_dropout.onnx

Update model_results.json with final test metrics + calibration metrics.
Export test set predictions (with all meta columns) for M3 fairness analysis.
</action>
<verify>pytest tests/gates/test_gate_2_3.py passes</verify>
<done>Calibrated model, ONNX exported, final test metrics recorded</done>
</task>

<task type="auto">
<name>Write Gate 2.3 validation tests</name>
<files>tests/gates/test_gate_2_3.py</files>
<action>
Validate:
- Brier score improves after calibration (post < pre)
- Test PR-AUC within 0.07 of validation PR-AUC (no extreme overfitting)
- ONNX file exists and is < 50MB
- ONNX predictions vs Python: max abs diff < 1e-5 on 100 samples
- model_results.json has 'test_final' entry
- Calibration plot saved to outputs/figures/
- Print comparison table: your metrics vs Alerta Escuela published metrics (from Section 1)
</action>
<verify>All tests pass</verify>
<done>Gate 2.3 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 2.3:** Human reviews: (1) Does calibration plot look diagonal? (2) Do your metrics vs Alerta Escuela's comparison table tell a coherent story?

---

### Phase 8: Subgroup Fairness Metrics (M3.1)

**Requirements:** REQ-FAIRNESS-01, REQ-FAIRNESS-02, REQ-FAIRNESS-03

<tasks>
<task type="auto">
<name>Compute comprehensive fairness metrics across all dimensions</name>
<files>src/fairness/metrics.py</files>
<action>
Implement `compute_fairness_metrics(y_true, y_pred, y_pred_proba, sensitive_df, sample_weight) -> dict`:

Using fairlearn MetricFrame, compute for EACH of the 6 dimensions (Section 7):
- TPR (recall) per group
- FPR per group
- FNR per group (= 1 - TPR)
- Precision per group
- PR-AUC per group
- Equalized odds gap (max TPR difference between any two groups)
- Predictive parity gap (max precision difference)

Also compute:
- Calibration per group: actual dropout rate among predicted-high-risk (>0.7 proba)
- Coverage gaps: FNR per group (which students does the model miss?)

For intersections (language×rurality, sex×poverty, language×region):
- Create combined group column
- Compute same metrics
- Flag groups with < 50 observations

ALL metrics must pass sample_weight=FACTOR07.

Output: nested dict structure per dimension, per group, per metric.
Save to `data/exports/fairness_metrics.json`.
</action>
<verify>pytest tests/gates/test_gate_3_1.py passes</verify>
<done>Fairness metrics for all 6 dimensions + 3 intersections, exported as JSON</done>
</task>

<task type="auto">
<name>Write Gate 3.1 validation tests</name>
<files>tests/gates/test_gate_3_1.py</files>
<action>
Validate:
- Metrics exist for all 6 dimensions
- Each group has sample size >= threshold (100 for main, 50 for disaggregated)
- Weighted metrics differ from unweighted
- fairness_metrics.json exists, is valid JSON
- At least 3 intersection analyses present
- Print FNR by language group and FNR by rural/urban for human review
- Print calibration table (predicted-high-risk actual rates per group) for human review
</action>
<verify>All tests pass</verify>
<done>Gate 3.1 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 3.1:** Human reviews: (1) FNR gap direction — do indigenous languages have higher FNR? (2) Calibration by group — does "high risk" mean different things for different groups? (3) Intersectional findings — are they meaningful or noise?

---

### Phase 9: SHAP Analysis (M3.2)

**Requirements:** REQ-FAIRNESS-04, REQ-FAIRNESS-05

<tasks>
<task type="auto">
<name>Compute global, regional, and interaction SHAP values</name>
<files>src/fairness/shap_analysis.py</files>
<action>
Implement using Section 8 spec:

1. `compute_global_shap(model, X_test) -> np.ndarray`: TreeExplainer on test set (2024)
2. `compute_regional_shap(model, X_test, regions) -> dict`: SHAP per region_natural group
3. `compute_interaction_shap(model, X_test_subsample) -> np.ndarray`:
   Subsample to 1000 rows if test set > 5000 (interaction values are O(n²))

Generate and save to outputs/figures/:
- Global SHAP summary plot (beeswarm)
- Top-10 feature importance bar chart
- Regional comparison: side-by-side SHAP bars for Costa/Sierra/Selva
- ES_PERUANO force plot for a representative non-Peruvian student
- ES_MUJER force plot for a representative female secundaria student

For M4 export, create 10 representative student profiles:
- 2 from Lima urban (1 Castellano, 1 foreign)
- 2 from Sierra rural (1 Quechua, 1 Castellano)
- 2 from Selva rural (1 Awajún/indigenous, 1 Castellano)
- 2 female secundaria (1 urban, 1 rural)
- 2 male secundaria (1 urban, 1 rural)
Each profile: feature values + SHAP values + predicted probability.

Save to `data/exports/shap_values.json`.
</action>
<verify>pytest tests/gates/test_gate_3_2.py passes</verify>
<done>SHAP computed, regional comparison done, 10 profiles exported, figures saved</done>
</task>

<task type="auto">
<name>Write Gate 3.2 validation tests</name>
<files>tests/gates/test_gate_3_2.py</files>
<action>
Validate:
- SHAP computed on 2024 test set (check year column)
- Top-5 SHAP features overlap with top-5 LR coefficient magnitudes (>= 3 overlap)
- SHAP values array shape matches (n_test_samples, n_features)
- shap_values.json exists, contains 'global_importance', 'regional', 'profiles' keys
- 10 representative profiles in 'profiles'
- Each profile has feature values + shap values + predicted_proba
- Print top-5 global SHAP features for human review
- Print ES_PERUANO and ES_MUJER average SHAP values for human review
</action>
<verify>All tests pass</verify>
<done>Gate 3.2 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 3.2:** Human reviews: (1) Are top-5 features intuitive? (2) ES_PERUANO magnitude — is it alarming? (3) ES_MUJER in secundaria — structural or bias? (4) Regional differences — does language matter more in Selva than Lima? (5) Are interaction effects meaningful?

---

### Phase 10: Cross-Validation with Admin Data (M3.3)

**Requirements:** REQ-FAIRNESS-06

<tasks>
<task type="auto">
<name>Validate fairness gaps against administrative district data</name>
<files>src/fairness/metrics.py (extend)</files>
<action>
Add to metrics.py:

`cross_validate_admin(predictions_df, admin_df) -> dict`:
1. Aggregate model predictions to district level: mean predicted_proba per UBIGEO
2. Merge with admin dropout rates on UBIGEO
3. Compute Pearson correlation between model predictions and admin rates
4. Split districts by % indigenous language speakers (from census):
   - High indigenous (>50%)
   - Low indigenous (<10%)
   Compare mean absolute prediction error between groups
5. Generate choropleth data: per-district predicted rate, admin rate, error,
   indigenous %, poverty index

Save choropleth data to `data/exports/choropleth.json`.
</action>
<verify>pytest tests/gates/test_gate_3_3.py passes</verify>
<done>Positive correlation with admin data, indigenous-majority district error quantified</done>
</task>

<task type="auto">
<name>Write Gate 3.3 validation tests</name>
<files>tests/gates/test_gate_3_3.py</files>
<action>
Validate:
- Correlation between model predictions and admin rates is positive (r > 0)
- Correlation is statistically significant (p < 0.05)
- choropleth.json exists, has ubigeo + predicted_rate + admin_rate + error fields
- Number of districts in choropleth > 1500
- Print correlation, p-value, and mean error by indigenous % group for human review
</action>
<verify>All tests pass</verify>
<done>Gate 3.3 passes</done>
</task>
</tasks>

**🛑 HUMAN GATE 3.3:** Human reviews: (1) Do indigenous-majority districts have higher model error? (2) Does the spatial pattern of errors make geographic sense? (3) If ENAHO and admin data diverge, is the explanation valid?

---

### Phase 11: Findings Distillation + M4 Exports (M3.4)

**Requirements:** REQ-EXPORT-02, REQ-EXPORT-03, REQ-EXPORT-04

<tasks>
<task type="auto">
<name>Distill findings and generate all M4 export files</name>
<files>src/fairness/distill.py, data/exports/README.md</files>
<action>
Implement `distill_findings() -> dict`:

Read all previously generated exports (fairness_metrics.json, shap_values.json,
model_results.json, choropleth.json, descriptive_tables.json).

Produce `data/exports/findings.json` with 5-7 findings, each structured as:

```json
{
  "findings": [
    {
      "id": 1,
      "stat": "63%",
      "stat_context": "higher non-enrollment rate",
      "headline_es": "Los estudiantes Awajún tienen una tasa de deserción 63% mayor que los hispanohablantes",
      "headline_en": "Awajún students have a 63% higher dropout rate than Spanish speakers",
      "explanation_es": "...",
      "explanation_en": "...",
      "metric_source": "descriptive_tables.json → language_dropout_rates → awajun",
      "visualization_type": "bar_chart",
      "data_key": "language_dropout_rates"
    }
  ]
}
```

Order findings by impact (most striking first).

Write data/exports/README.md documenting:
- Each JSON file's purpose
- Schema for each file
- How they connect to the M4 site components
- Data provenance chain

Verify ALL export files exist with correct schemas:
□ findings.json
□ fairness_metrics.json
□ shap_values.json
□ choropleth.json
□ model_results.json
□ descriptive_tables.json
□ onnx/lightgbm_dropout.onnx
□ README.md
</action>
<verify>pytest tests/gates/test_gate_3_4.py passes</verify>
<done>All 7 export files present, findings distilled, README documents schemas</done>
</task>

<task type="auto">
<name>Write Gate 3.4 validation tests</name>
<files>tests/gates/test_gate_3_4.py</files>
<action>
Validate:
- findings.json has 5-7 entries
- Each finding has: id, stat, headline_es, headline_en, metric_source, visualization_type
- Every metric_source path resolves to an actual value in the referenced export file
- All 7 export files exist in data/exports/
- ONNX file > 1KB (not empty)
- README.md exists and references all 7 files
- Print all findings (id + headline_en) for human review
</action>
<verify>All tests pass</verify>
<done>Gate 3.4 passes — analysis complete, ready for M4 site</done>
</task>
</tasks>

**🛑 HUMAN GATE 3.4:** Human reviews: (1) Is finding #1 the most striking? (2) Can a non-technical person understand each finding? (3) Are Spanish translations natural for Peruvian audience? (4) Tag this commit: `git tag v1.0-analysis-complete`

---

## 10. REQUIREMENTS

### Data Requirements

- **REQ-DATA-01:** Load single year ENAHO with correct delimiter detection
- **REQ-DATA-02:** Construct dropout target as (P303==1 & P306==2)
- **REQ-DATA-03:** Load all 7 years (2018-2024) with consistent schema
- **REQ-DATA-04:** Harmonize P300A mother tongue codes across 2020 structural break
- **REQ-DATA-05:** Load and zero-pad district admin dropout rates
- **REQ-DATA-06:** Load Census 2017 district-level data
- **REQ-DATA-07:** Load VIIRS nightlights district-level data
- **REQ-DATA-08:** Merge all sources via left join on UBIGEO, preserving row count
- **REQ-DATA-09:** Engineer all features per Section 5 specification

### Analysis Requirements

- **REQ-ANALYSIS-01:** Compute weighted descriptive statistics across all 6 dimensions

### Model Requirements

- **REQ-MODEL-01:** Temporal split: train 2018-2022, validate 2023, test 2024
- **REQ-MODEL-02:** Train logistic regression baseline with survey-weighted evaluation
- **REQ-MODEL-03:** Train and tune LightGBM with Optuna (50 trials)
- **REQ-MODEL-04:** Train XGBoost for comparison
- **REQ-MODEL-05:** Calibrate best model, evaluate on test set exactly once
- **REQ-MODEL-06:** Export LightGBM to ONNX with validation

### Fairness Requirements

- **REQ-FAIRNESS-01:** Compute TPR, FPR, FNR, precision, PR-AUC per subgroup across all 6 dimensions
- **REQ-FAIRNESS-02:** Compute calibration per group (actual rate among predicted-high-risk)
- **REQ-FAIRNESS-03:** Analyze 3 intersections: language×rurality, sex×poverty, language×region
- **REQ-FAIRNESS-04:** Global + regional SHAP analysis
- **REQ-FAIRNESS-05:** Quantify ES_PERUANO and ES_MUJER SHAP effects specifically
- **REQ-FAIRNESS-06:** Cross-validate fairness gaps against district admin data

### Export Requirements

- **REQ-EXPORT-01:** Export LightGBM as ONNX to data/exports/onnx/
- **REQ-EXPORT-02:** Distill 5-7 media-ready findings with Spanish translations
- **REQ-EXPORT-03:** All 7 export JSON files present with documented schemas
- **REQ-EXPORT-04:** README.md documenting all exports and their provenance

---

## 11. M4 EXPORT CONTRACTS (JSON SCHEMAS)

These are the exact schemas the Next.js scrollytelling site expects. The agent MUST produce files matching these schemas.

### 11.1 findings.json

```json
{
  "generated_at": "2025-01-15T10:30:00Z",
  "model_version": "lightgbm_v1",
  "findings": [
    {
      "id": 1,
      "stat": "63%",
      "stat_context": "higher non-enrollment rate for Awajún vs Castellano speakers",
      "headline_es": "string — one sentence, newspaper-ready Spanish",
      "headline_en": "string — one sentence English",
      "explanation_es": "string — 2-3 sentences expanding the finding in Spanish",
      "explanation_en": "string — 2-3 sentences English",
      "metric_source": "descriptive_tables.json → language_dropout_rates → awajun",
      "visualization_type": "bar_chart | choropleth | waterfall | line_chart | heatmap",
      "data_key": "string — key in the relevant export JSON that has the viz data",
      "severity": "high | medium | low"
    }
  ]
}
```

### 11.2 fairness_metrics.json

```json
{
  "generated_at": "ISO timestamp",
  "model": "lightgbm",
  "threshold": 0.45,
  "dimensions": {
    "language": {
      "groups": {
        "castellano": {
          "n_unweighted": 15000,
          "n_weighted": 2500000,
          "tpr": 0.42,
          "fpr": 0.08,
          "fnr": 0.58,
          "precision": 0.21,
          "pr_auc": 0.35,
          "calibration_high_risk": {
            "n_predicted_high": 500,
            "actual_dropout_rate": 0.68
          }
        },
        "quechua": { "..." : "same structure" },
        "aimara": { "..." : "same structure" },
        "other_indigenous": { "..." : "same structure" }
      },
      "gaps": {
        "equalized_odds_tpr": 0.15,
        "equalized_odds_fpr": 0.03,
        "predictive_parity": 0.08,
        "max_fnr_gap": 0.22,
        "max_fnr_groups": ["other_indigenous", "castellano"]
      }
    },
    "sex": { "..." : "same structure with groups male/female" },
    "rural": { "..." : "same structure with groups urban/rural" },
    "region": { "..." : "same structure with groups costa/sierra/selva" },
    "poverty": { "..." : "same structure with groups q1-q5" },
    "intersections": {
      "language_x_rural": {
        "groups": {
          "quechua_rural": { "..." : "same metric structure" },
          "quechua_urban": { "..." : "same metric structure" },
          "castellano_rural": { "..." : "same metric structure" },
          "castellano_urban": { "..." : "same metric structure" }
        }
      }
    }
  }
}
```

### 11.3 shap_values.json

```json
{
  "generated_at": "ISO timestamp",
  "model": "lightgbm",
  "computed_on": "test_2024",
  "feature_names": ["age", "es_mujer", "lang_castellano", "..."],
  "feature_labels_es": ["Edad", "Sexo femenino", "Lengua castellana", "..."],
  "global_importance": {
    "feature_name": [0.15, 0.08, 0.06, "...mean_abs_shap_per_feature"]
  },
  "regional": {
    "costa": { "feature_name": [0.12, 0.05, "..."] },
    "sierra": { "feature_name": [0.18, 0.10, "..."] },
    "selva": { "feature_name": [0.20, 0.12, "..."] }
  },
  "profiles": [
    {
      "id": "lima_urban_castellano_male",
      "label_es": "Estudiante masculino, Lima urbana, hispanohablante",
      "features": { "age": 14, "es_mujer": 0, "lang_castellano": 1, "...": "..." },
      "shap_values": { "age": 0.02, "es_mujer": -0.05, "...": "..." },
      "predicted_proba": 0.08,
      "base_value": 0.14
    }
  ]
}
```

### 11.4 choropleth.json

```json
{
  "generated_at": "ISO timestamp",
  "districts": [
    {
      "ubigeo": "010101",
      "department": "Amazonas",
      "provincia": "Chachapoyas",
      "distrito": "Chachapoyas",
      "predicted_dropout_rate": 0.18,
      "admin_dropout_rate": 0.025,
      "model_error": 0.155,
      "indigenous_language_pct": 0.05,
      "poverty_index": 0.45,
      "n_students_survey": 45,
      "latitude": -6.2316,
      "longitude": -77.8691
    }
  ]
}
```

### 11.5 model_results.json

```json
{
  "generated_at": "ISO timestamp",
  "models": {
    "logistic_regression": {
      "validate_2023": {
        "pr_auc": 0.28,
        "f1": 0.25,
        "precision": 0.22,
        "recall": 0.35,
        "brier": 0.12,
        "threshold": 0.5
      }
    },
    "lightgbm": {
      "validate_2023": { "...": "same metrics" },
      "test_2024_final": { "...": "same metrics" },
      "test_2024_calibrated": { "...": "same metrics + brier_pre_calibration" },
      "feature_importance": { "feature_name": "importance_value" },
      "hyperparameters": { "...": "best Optuna params" }
    },
    "xgboost": {
      "validate_2023": { "...": "same metrics" }
    }
  },
  "alerta_escuela_comparison": {
    "source": "MINEDU methodology document, October 2024",
    "their_metrics": {
      "inicial": { "roc_auc": 0.84, "precision": 0.23, "recall": 0.37 },
      "primaria": { "roc_auc": 0.89, "precision": 0.22, "recall": 0.43 },
      "secundaria": { "roc_auc": 0.87, "precision": 0.19, "recall": 0.36 }
    },
    "our_metrics_summary": "string — brief comparison narrative"
  },
  "thresholds": {
    "lightgbm": {
      "0.3": { "...": "metrics at threshold 0.3" },
      "0.4": { "...": "metrics at threshold 0.4" },
      "0.5": { "...": "metrics at threshold 0.5" }
    }
  }
}
```

### 11.6 descriptive_tables.json

```json
{
  "generated_at": "ISO timestamp",
  "dropout_by_language": {
    "castellano": { "weighted_rate": 0.137, "n_unweighted": 18000, "n_weighted": 3200000 },
    "quechua": { "weighted_rate": 0.165, "n_unweighted": 2500, "n_weighted": 380000 },
    "aimara": { "...": "..." },
    "other_indigenous": { "...": "..." },
    "awajun_2020plus": { "weighted_rate": 0.224, "n_unweighted": 150, "n_weighted": 25000 }
  },
  "dropout_by_sex": { "...": "same structure" },
  "dropout_by_rural": { "...": "same structure" },
  "dropout_by_region": { "...": "same structure" },
  "dropout_by_poverty_quintile": { "...": "same structure" },
  "dropout_by_sex_x_level": {
    "male_primaria": { "...": "..." },
    "female_primaria": { "...": "..." },
    "male_secundaria": { "...": "..." },
    "female_secundaria": { "...": "..." }
  },
  "heatmap_language_x_rural": {
    "rows": ["castellano", "quechua", "aimara", "other_indigenous"],
    "columns": ["urban", "rural"],
    "values": [[0.12, 0.18], [0.14, 0.22], [0.13, 0.20], [0.15, 0.25]]
  },
  "temporal_trend": {
    "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "overall_rate": [0.13, 0.14, 0.16, 0.15, 0.14, 0.14, 0.13],
    "by_language": { "...": "rate per year per language group" }
  }
}
```

---

## 12. ALERTA ESCUELA REFERENCE (for comparison)

The system we're auditing. This information comes from their published methodology document.

**Published performance (what they report):**

| Metric | Inicial | Primaria | Secundaria |
|---|---|---|---|
| ROC AUC | 0.84 | 0.89 | 0.87 |
| Precision | 0.23 | 0.22 | 0.19 |
| Recall | 0.37 | 0.43 | 0.36 |
| Missed dropouts (FNR) | 63% | 57% | 64% |
| False alarm rate (1-Precision) | 77% | 78% | 81% |

**Their algorithm:** LightGBM. 60 models (12 grades × 5 macro regions).

**Their features (31 total from 5 sources):** SIAGIE (student), ESCALE (school), NEXUS (teachers), ECE (test scores), JUNTOS (social program), UE (household economics).

**What they excluded:** Attendance data — because SIAGIE records lack uniform criteria and may be biased for JUNTOS-affiliated students. They noted this data quality issue but did NOT examine whether it creates fairness gaps.

**What they never examined:** Any fairness metric. The words "sesgo," "equidad," "fairness," "bias," or "discriminación" appear nowhere in their documentation.

---

## 13. METHODOLOGICAL FRAMING

Include this text (or close paraphrase) in any documentation, notebooks, or README:

> We use ENAHO's survey-based non-enrollment indicator (P303 × P306), which captures a broader phenomenon than MINEDU's administrative dropout metric from SIAGIE. Our individual-level fairness analysis uses ENAHO pooled 2018–2024; we validate spatial patterns against district-level administrative rates from datosabiertos.gob.pe. The discrepancy between measures (~14% survey-based vs ~2% administrative) is well-documented and reflects measurement differences, not data error. Both are valid constructs; our audit examines equity across both.

---

## 14. AGENT EXECUTION RULES

1. **Read this document before each phase.** Context rot is real. Re-read the relevant phase section.
2. **Do not modify the tech stack.** No new libraries unless explicitly blocked by a bug.
3. **All metrics must be survey-weighted using FACTOR07.** If you produce an unweighted metric, also produce the weighted one and assert they differ.
4. **Use polars for data processing, pandas only at the sklearn boundary.** `.to_pandas()` when calling sklearn/fairlearn/shap, polars everywhere else.
5. **Preserve intermediate datasets as parquet** in `data/processed/`.
6. **Export final results as JSON** in `data/exports/` matching the schemas in Section 11.
7. **Write gate tests BEFORE or ALONGSIDE the implementation code.** Not after.
8. **When a gate says "print X for human review" — actually print it and STOP.** Do not proceed to the next phase without human confirmation.
9. **Commit atomically per task.** One commit per task, descriptive message.
10. **Never touch the test set (2024) before Phase 7 (M2.3).** If any code before Phase 7 loads 2024 data into a model or metric, that's a critical error.

---

## 15. KNOWN PITFALLS

| Pitfall | When | Prevention |
|---|---|---|
| ENAHO delimiter mismatch | Phase 1-2 | Hardcode: pipe for <=2019, comma for >=2020 |
| UBIGEO leading zero loss | Phase 3 | Always `.zfill(6)` immediately after loading |
| Cartesian join on UBIGEO | Phase 3 | Assert row count unchanged after every merge |
| P300A harmonization forgotten | Phase 2 | Always create BOTH harmonized and original columns |
| Survey weights ignored | Phase 4+ | Every metric function MUST accept sample_weight parameter |
| Test data leakage | Phase 5-7 | Assert test year ==2024 never appears in training data |
| SHAP on train instead of test | Phase 9 | Assert SHAP input DataFrame year column == 2024 |
| ONNX prediction mismatch | Phase 7 | Validate 100 samples with max diff < 1e-5 |
| Polars/pandas confusion | All | Use polars for processing, .to_pandas() only at sklearn boundary |
| Feature name mismatch between training and ONNX | Phase 7 | Store feature names list, assert ONNX input matches |
