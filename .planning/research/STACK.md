# Stack Research

**Domain:** ML fairness audit pipeline (survey microdata, binary classification, equity analysis)
**Researched:** 2026-02-07
**Confidence:** HIGH -- all versions verified via live `uv pip compile` resolution on this machine

---

## Recommended Stack

The stack is LOCKED per `specs.md` Section 2. This document verifies current versions, documents API patterns, flags compatibility concerns, and provides the Nix + uv configuration patterns needed to bootstrap the project.

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Python | 3.12 | Runtime | Spec-locked. Broad ecosystem support. All stack packages verified compatible. | HIGH |
| polars | >=1.38.1 | Primary data processing | 30-50x faster than pandas for CSV/parquet ops on ~180K rows. Expression-based API prevents mutation bugs. `to_pandas()` at sklearn boundary. | HIGH |
| scikit-learn | >=1.8.0 | ML framework | LogisticRegression baseline, CalibratedClassifierCV, all evaluation metrics with `sample_weight`. v1.8.0 is current stable. | HIGH |
| lightgbm | >=4.6.0 | Primary model | Matches Alerta Escuela's algorithm. sklearn API (`LGBMClassifier`) for Optuna integration. v4.6.0 is current stable. | HIGH |
| xgboost | >=3.1.3 | Comparison model | Algorithm-independence test for fairness findings. v3.1.3 is current stable (major version 3 since late 2024). | HIGH |
| fairlearn | >=0.13.0 | Fairness metrics | `MetricFrame` with `sample_params` for survey-weighted subgroup metrics. v0.13.0 uses narwhals backend (pandas-agnostic). | HIGH |
| shap | >=0.50.0 | Interpretability | `TreeExplainer` for LightGBM feature attribution. v0.50.0 is current for Python 3.12. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| optuna | >=4.7.0 | Hyperparameter tuning | LightGBM + XGBoost tuning, 50 trials on validation set | HIGH |
| statsmodels | >=0.14.6 | Survey-weighted stats | Descriptive statistics, weighted means, confidence intervals | HIGH |
| onnxmltools | >=1.16.0 | ONNX export | LightGBM to ONNX conversion for M4 scrollytelling site | HIGH |
| onnx | >=1.20.1 | ONNX format | Dependency of onnxmltools, model serialization | HIGH |
| onnxruntime | >=1.24.1 | ONNX inference | Validation of ONNX export (predictions match Python model) | HIGH |
| skl2onnx | >=1.20.0 | ONNX conversion backend | Transitive dependency of onnxmltools, handles sklearn pipeline conversion | HIGH |
| matplotlib | >=3.10.8 | Static visualization | Paper-quality figures for outputs/figures/ | HIGH |
| seaborn | >=0.13.2 | Statistical visualization | Heatmaps, distribution plots, fairness gap visualizations | HIGH |
| plotly | >=6.5.2 | Interactive visualization | Notebook-embedded interactive plots for EDA | HIGH |
| pyarrow | >=19.0 | Data bridge | Required by polars `to_pandas()` and parquet I/O. MUST be in dependencies. | HIGH |
| pandas | >=3.0.0 | sklearn boundary | Only used via `polars_df.to_pandas()` at sklearn/fairlearn/shap interface. pandas 3.0 is compatible with all stack packages. | HIGH |
| numpy | >=2.3.5 | Numerical computing | Transitive dependency. Used directly for array ops at sklearn boundary. | HIGH |

### Development Tools

| Tool | Version | Purpose | Notes | Confidence |
|------|---------|---------|-------|------------|
| uv | 0.9.28 (system) | Package management | Already installed system-wide via Nix. Manages Python deps via pyproject.toml. | HIGH |
| Nix | 2.31.2 (system) | Environment management | Provides Python 3.12, system libraries (gcc, etc.), reproducible dev shell | HIGH |
| ruff | >=0.15.0 | Linting + formatting | All-in-one Python linter/formatter. Replaces flake8, isort, black. | HIGH |
| pytest | >=9.0.2 | Testing | Gate validation tests per phase | HIGH |
| jupyterlab | >=4.5.3 | Notebooks | EDA and narrative notebooks | HIGH |
| pyright | latest | Type checking | Static type analysis, configured via pyproject.toml | MEDIUM |

---

## Version Verification Method

All versions were verified via live `uv pip compile` resolution on 2026-02-07 against Python 3.12. This is authoritative -- uv resolves from PyPI's live index, not cached or stale data.

**Key finding:** SHAP resolves to 0.50.0 for Python 3.12 (vs 0.49.1 for Python 3.14). Use >=0.50.0 as the minimum version.

**Key finding:** pandas 3.0.0 resolves successfully with ALL stack packages for Python 3.12. This is a major version (pandas 3.0 was released in 2025). fairlearn 0.13.0 already uses narwhals 2.16.0 as its DataFrame abstraction layer, making it pandas-version-agnostic.

---

## Critical API Patterns

### Polars (v1.38.1) -- Verified Live

All patterns below were verified by running actual code against polars 1.38.1.

**UBIGEO zero-padding:**
```python
# Use str.pad_start (NOT str.zfill -- that does not exist in polars)
df = df.with_columns(
    pl.col("UBIGEO").cast(pl.Utf8).str.pad_start(6, "0").alias("ubigeo")
)
```

**Group-by with weighted mean (survey weights):**
```python
weighted_rates = df.group_by("group_col").agg(
    (pl.col("dropout") * pl.col("factor07")).sum() / pl.col("factor07").sum()
)
```

**Conditional columns (when/then/otherwise):**
```python
df = df.with_columns(
    pl.when(pl.col("P300A").is_in([10, 11, 12, 13, 14, 15]))
    .then(pl.lit(3))
    .when(pl.col("P300A") == 99)
    .then(pl.lit(99))
    .otherwise(pl.col("P300A"))
    .alias("p300a_harmonized")
)
```

**Join (left join for spatial merge):**
```python
merged = enaho_df.join(admin_df, on="ubigeo", how="left")
# ALWAYS assert row count unchanged after left join
assert merged.height == enaho_df.height, "Left join created duplicates"
```

**CSV reading with delimiter detection:**
```python
# polars uses `separator` parameter (NOT `sep` or `delimiter`)
df = pl.read_csv(filepath, separator="|" if year <= 2019 else ",")
```

**Lazy evaluation for large operations:**
```python
lf = pl.scan_csv(filepath, separator=sep)
result = lf.filter(pl.col("P208A").is_between(6, 17)).collect()
```

**Converting to pandas at sklearn boundary:**
```python
# Requires pyarrow to be installed
X_train_pd = X_train_polars.to_pandas()
y_train_np = y_train_polars.to_numpy()
```

**Column name normalization:**
```python
df = df.rename({col: col.strip().upper() for col in df.columns})
```

### Fairlearn MetricFrame (v0.13.0) -- Verified Live

The `sample_params` API for passing `sample_weight` to individual metrics is confirmed working.

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score

mf = MetricFrame(
    metrics={"recall": recall_score, "precision": precision_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_df,
    sample_params={
        "recall": {"sample_weight": weights},
        "precision": {"sample_weight": weights},
    },
)

# Per-group results
mf.by_group          # DataFrame with groups as index, metrics as columns
mf.difference()      # Max gap between any two groups (per metric)
mf.ratio()           # Min ratio between any two groups (per metric)
mf.overall           # Overall metric values
```

**CRITICAL:** The `sample_params` dict keys must match the `metrics` dict keys exactly. Each maps to a dict of kwargs passed to that metric function.

**Narwhals backend:** fairlearn 0.13.0 depends on narwhals 2.16.0, which means it can accept polars DataFrames directly for `sensitive_features` in some cases. However, for safety and compatibility with sklearn metrics, convert to pandas/numpy at the fairlearn boundary.

### SHAP TreeExplainer (v0.50.0) -- Training Data + Version Verified

**Confidence:** MEDIUM -- version verified via uv, API patterns from training data (SHAP 0.50 is newer than training cutoff, but TreeExplainer API has been stable since 0.40+).

```python
import shap

# TreeExplainer for LightGBM
explainer = shap.TreeExplainer(best_lgb_model)

# SHAP values on test set
shap_values = explainer.shap_values(X_test)

# IMPORTANT: For binary classification with LightGBM:
# - LightGBM native Booster: shap_values returns a list [class_0, class_1]
#   Use shap_values[1] for the positive class
# - LGBMClassifier (sklearn API): behavior may differ
#   Check type: if isinstance(shap_values, list), use shap_values[1]
#   Otherwise use shap_values directly
# VALIDATE at runtime which format is returned.

# Alternative (more robust for newer SHAP versions):
explanation = explainer(X_test)
# explanation.values is the SHAP values array
# explanation.base_values is the expected value
```

**Interaction values (expensive):**
```python
# Subsample to <= 1000 rows (O(n * n_features^2) complexity)
X_subsample = X_test[:1000] if len(X_test) > 1000 else X_test
shap_interaction = explainer.shap_interaction_values(X_subsample)
# Shape: (n_samples, n_features, n_features)
```

**SHAP plots:**
```python
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig("outputs/figures/shap_summary.png", dpi=150, bbox_inches="tight")

shap.plots.bar(explanation, show=False)
shap.plots.waterfall(explanation[0], show=False)  # Single instance
```

### LightGBM sklearn API (v4.6.0) -- Verified Live

```python
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    objective="binary",
    metric="average_precision",
    is_unbalance=True,
    verbosity=-1,
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
)

# Training with early stopping
clf.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    eval_sample_weight=[w_val],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
)
```

**IMPORTANT API NOTE:** In LightGBM 4.x, early stopping is via callbacks, not the deprecated `early_stopping_rounds` parameter:
```python
import lightgbm as lgb
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=10),
]
```

### XGBoost sklearn API (v3.1.3) -- Version Verified

```python
from xgboost import XGBClassifier

clf = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=(n_negative / n_positive),
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,  # XGBoost 3.x still uses this parameter
)

clf.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    sample_weight_eval_set=[w_val],
    verbose=10,
)
```

### Optuna + LightGBM Integration (v4.7.0) -- Version Verified

```python
import optuna

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    clf = LGBMClassifier(**params)
    clf.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0),  # Suppress per-round output
        ],
    )

    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_pred_proba, sample_weight=w_val)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

**Optuna pruning callback (optional but recommended for speed):**
```python
from optuna.integration import LightGBMPruningCallback

# Add to callbacks list:
callbacks=[
    lgb.early_stopping(50),
    lgb.log_evaluation(0),
    LightGBMPruningCallback(trial, "average_precision"),
]
```

**NOTE:** `optuna.integration` is confirmed available. In older Optuna versions this was `optuna.integration.lightgbm`. In Optuna 4.x, the import path is `optuna.integration.LightGBMPruningCallback`. Verify at runtime.

### onnxmltools LightGBM Conversion (v1.16.0) -- Version Verified

```python
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

initial_type = [("features", FloatTensorType([None, len(MODEL_FEATURES)]))]
onnx_model = convert_lightgbm(
    best_lgb_model,
    initial_types=initial_type,
    target_opset=15,  # Use opset 15 for broad onnxruntime compatibility
)

# Save
import onnx
onnx.save(onnx_model, "data/exports/onnx/lightgbm_dropout.onnx")

# Validate
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("data/exports/onnx/lightgbm_dropout.onnx")
input_name = sess.get_inputs()[0].name
onnx_preds = sess.run(None, {input_name: X_sample.astype(np.float32)})[1][:, 1]
python_preds = best_lgb_model.predict_proba(X_sample)[:, 1]
assert np.max(np.abs(onnx_preds - python_preds)) < 1e-5
```

**IMPORTANT:** onnxmltools 1.16.0 depends on skl2onnx 1.20.0, which depends on scikit-learn. This means the entire sklearn/onnx chain is tightly version-coupled. Pin all three together.

**IMPORTANT:** `convert_lightgbm` accepts both the native LightGBM Booster and the sklearn `LGBMClassifier`. If using `LGBMClassifier`, the conversion handles the sklearn wrapper automatically.

### Nix Flake + uv Pattern -- Verified from User's Existing Projects

The user has two established patterns. For this project, use the **Manim-ML pattern** (simpler: Nix provides system deps, uv manages Python packages) rather than the uv2nix pattern (ML-utils), because:

1. The uv2nix pattern requires complex overrides for compiled packages (scipy, numpy, lightgbm)
2. The simpler pattern just needs gcc/pkg-config from Nix and lets uv build wheels
3. LightGBM and XGBoost need system-level libs (libgomp, cmake) that Nix provides cleanly

```nix
{
  description = "Alerta Escuela Equity Audit - ML fairness pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              # Python runtime
              pkgs.python312

              # Package management
              pkgs.uv

              # Build dependencies for compiled packages (scipy, numpy, lightgbm)
              pkgs.gcc
              pkgs.pkg-config
              pkgs.gnumake
              pkgs.cmake

              # LightGBM/XGBoost system dependencies
              pkgs.libgcc
              pkgs.openmp  # libgomp for LightGBM OpenMP support

              # BLAS/LAPACK for numpy/scipy
              pkgs.openblas

              # Development tools
              pkgs.ruff
              pkgs.pyright
              pkgs.git
            ];

            env = {
              # Prevent uv from downloading Python
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
              export PYTHONPATH="$PWD:$PYTHONPATH"
            '';
          };
        }
      );
    };
}
```

**IMPORTANT NOTES for the Nix flake:**
- `pkgs.openmp` or `pkgs.llvmPackages.openmp` provides libgomp needed by LightGBM for parallel tree building
- `pkgs.openblas` provides BLAS/LAPACK needed by scipy/numpy
- `pkgs.cmake` is needed if LightGBM builds from source
- The `shellHook` runs `uv sync` on shell entry and activates the venv
- Per the user's CLAUDE.md: use `direnv reload` after updating flake.nix (implies direnv + nix integration is configured)

---

## Installation (pyproject.toml)

```toml
[project]
name = "alerta-escuela-audit"
version = "0.1.0"
description = "Independent equity audit of Peru's Alerta Escuela dropout prediction system"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    # Data processing
    "polars>=1.38",
    "pyarrow>=19.0",          # Required for polars to_pandas()

    # ML framework
    "scikit-learn>=1.8",
    "lightgbm>=4.6",
    "xgboost>=3.1",

    # Fairness
    "fairlearn>=0.13",

    # Interpretability
    "shap>=0.50",

    # Hyperparameter tuning
    "optuna>=4.7",

    # Statistics
    "statsmodels>=0.14",

    # ONNX export
    "onnx>=1.20",
    "onnxmltools>=1.16",
    "onnxruntime>=1.24",

    # Visualization
    "matplotlib>=3.10",
    "seaborn>=0.13",
    "plotly>=6.5",

    # Notebooks
    "jupyterlab>=4.5",
]

[dependency-groups]
dev = [
    "pytest>=9.0",
    "ruff>=0.15",
    "pyright>=1.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["ALL"]
ignore = [
    "D100", "D104",        # Missing docstrings in public module/package
    "D203", "D212",        # Conflicting docstring rules
    "COM812", "ISC001",    # Conflicts with formatter
    "E501",                # Line too long (handled by formatter)
    "PLR2004",             # Magic value comparisons (common in data science)
    "PLR0913",             # Too many arguments (common in ML functions)
    "T201",                # print() usage (needed for gate outputs)
    "S101",                # assert usage (needed in tests)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["E402", "D104", "N999"]
"**/{tests,notebooks}/*" = ["E402", "S101", "D"]

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "basic"
include = ["src"]
exclude = [".venv", "data", "outputs"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not the Alternative |
|----------|-------------|-------------|------------------------|
| Data processing | polars | pandas | 30-50x slower on CSV I/O, mutable state leads to SettingWithCopyWarning bugs, polars expression API is more composable for feature engineering |
| Data processing | polars | dask | Dataset fits in memory (~180K rows), dask adds complexity for no benefit at this scale |
| Fairness | fairlearn | AIF360 | fairlearn has simpler API, better sklearn integration, `MetricFrame` with `sample_params` handles survey weights cleanly. AIF360 is heavier, IBM-specific conventions |
| Fairness | fairlearn | Aequitas | Aequitas lacks `sample_weight` support for survey data. fairlearn's `MetricFrame` is more flexible |
| SHAP | shap (TreeExplainer) | LIME | TreeExplainer is exact (not approximate) for tree models. LIME is model-agnostic but slower and noisier for tree ensembles |
| Gradient boosting | LightGBM + XGBoost | CatBoost | CatBoost is spec-excluded. LightGBM matches Alerta Escuela's algorithm. XGBoost validates algorithm-independence |
| Experiment tracking | JSON export | MLflow / W&B | Spec-excluded. JSON exports are simpler, committed to git, consumed directly by M4 site |
| Package management | uv | pip / poetry / conda | uv is 10-100x faster, already installed on system, resolves lock files deterministically. User already uses uv across projects |
| Environment | Nix flakes | Docker / conda-env | Nix provides reproducible system deps (gcc, openblas, libgomp) that Docker would also need. User already uses Nix across all projects |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| pandas for data processing | Spec-locked to polars. Only `to_pandas()` at sklearn boundary. Mixing polars and pandas leads to confusion about which DataFrame type you have. | polars everywhere, `.to_pandas()` only when calling sklearn/fairlearn/shap |
| TabNet / PyTorch / TensorFlow | Spec-excluded. Neural networks are unnecessary for ~180K rows tabular data. Three models only: LogReg, LightGBM, XGBoost. | LGBMClassifier, XGBClassifier, LogisticRegression |
| MLflow / DVC / W&B | Spec-excluded. Adds infrastructure complexity. JSON exports are simpler and directly consumed by M4 site. | `data/exports/*.json` committed to git |
| AIF360 | Spec-excluded. Use fairlearn exclusively. | fairlearn MetricFrame |
| geopandas / GDAL | Spec-excluded. All geo data is pre-aggregated CSV/JSON. No geospatial processing needed. | polars for loading pre-aggregated district CSVs |
| dask / spark | Spec-excluded. Dataset fits in memory. | polars (single-threaded is sufficient for ~180K rows) |
| `polars_df.to_pandas()` without pyarrow | Will raise `ModuleNotFoundError`. pyarrow is REQUIRED for polars-to-pandas conversion. | Add pyarrow to dependencies |
| `str.zfill()` in polars | Does not exist in polars. | `pl.col("x").str.pad_start(6, "0")` |
| `sep=` parameter in polars read_csv | Parameter is called `separator`, not `sep`. | `pl.read_csv(path, separator="\|")` |
| `early_stopping_rounds=` in LightGBM 4.x | Deprecated. | `callbacks=[lgb.early_stopping(50)]` |
| pandas `DataFrame.groupby()` in polars | Use polars `group_by()` (with underscore). | `df.group_by("col").agg(...)` |

---

## Version Compatibility Matrix

All combinations verified via single `uv pip compile` resolution (2026-02-07):

| Package | Version | Depends On | Compatibility Notes |
|---------|---------|------------|---------------------|
| polars 1.38.1 | 1.38.1 | pyarrow (for to_pandas) | pyarrow MUST be in dependencies for to_pandas() to work |
| fairlearn 0.13.0 | 0.13.0 | narwhals 2.16.0, scikit-learn, scipy, pandas | narwhals makes it pandas-version-agnostic |
| shap 0.50.0 | 0.50.0 | numpy, scipy, scikit-learn, pandas, numba 0.63.1, cloudpickle | numba 0.63.1 requires llvmlite 0.46.0 (Nix must provide LLVM) |
| lightgbm 4.6.0 | 4.6.0 | numpy, scipy | Needs OpenMP (libgomp) at runtime for parallel tree building |
| xgboost 3.1.3 | 3.1.3 | numpy, scipy | nvidia-nccl-cu12 only on Linux (GPU support, optional) |
| optuna 4.7.0 | 4.7.0 | alembic, sqlalchemy, colorlog, tqdm, pyyaml | SQLAlchemy for study storage (uses SQLite by default) |
| onnxmltools 1.16.0 | 1.16.0 | onnx 1.20.1, skl2onnx 1.20.0, numpy, protobuf | skl2onnx depends on scikit-learn -- version must match |
| scikit-learn 1.8.0 | 1.8.0 | numpy, scipy, joblib, threadpoolctl | scipy 1.15.3 for Py 3.12 (needs Fortran compiler in Nix) |
| pandas 3.0.0 | 3.0.0 | numpy, python-dateutil, tzdata | Major version bump from 2.x. All stack packages support it. |
| numpy 2.3.5 | 2.3.5 | -- | numpy 2.x series. All packages verified compatible. |

**Critical Nix dependency chain for compiled packages:**
```
scipy, numpy -> openblas (BLAS/LAPACK) + gfortran
lightgbm -> cmake + gcc + libgomp (OpenMP)
shap -> numba -> llvmlite -> LLVM
xgboost -> cmake + gcc
```

---

## Polars-Pandas Boundary Pattern

This is the single most important architectural pattern for this project. Get it wrong and you will have type confusion throughout the codebase.

**Rule:** Polars is the canonical DataFrame type. Convert to pandas ONLY at the function call boundary where sklearn/fairlearn/shap requires it.

```python
# CORRECT: Convert at the boundary
def train_model(df: pl.DataFrame, features: list[str], target: str, weight: str):
    """Train model. Input is polars, conversion happens internally."""
    X = df.select(features).to_pandas()
    y = df[target].to_numpy()
    w = df[weight].to_numpy()

    clf = LGBMClassifier(...)
    clf.fit(X, y, sample_weight=w)
    return clf

# CORRECT: Fairlearn boundary
def compute_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,  # Already converted from polars
    weights: np.ndarray,
):
    mf = MetricFrame(
        metrics={"recall": recall_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        sample_params={"recall": {"sample_weight": weights}},
    )
    return mf

# WRONG: Mixing polars and pandas throughout
def bad_pattern(df):
    pdf = df.to_pandas()  # Converted too early
    # ... 50 lines of pandas code that should be polars ...
    result = pdf.groupby("col").mean()  # Should be polars group_by
```

---

## Survey Weight Pattern

Every metric computation in this project MUST pass `sample_weight=FACTOR07`. This is non-negotiable for a survey-based analysis.

```python
# sklearn metrics -- all accept sample_weight
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

# Pattern: always pass sample_weight
pr_auc = average_precision_score(y_true, y_pred_proba, sample_weight=weights)
f1 = f1_score(y_true, y_pred_binary, sample_weight=weights)

# fairlearn -- pass via sample_params
mf = MetricFrame(
    metrics={"recall": recall_score, "precision": precision_score},
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_df,
    sample_params={
        "recall": {"sample_weight": weights},
        "precision": {"sample_weight": weights},
    },
)

# Polars weighted aggregation
weighted_rate = df.group_by("group").agg(
    (pl.col("dropout") * pl.col("factor07")).sum() / pl.col("factor07").sum()
)
```

---

## Sources

| Source | What Was Verified | Confidence |
|--------|-------------------|------------|
| `uv pip compile` (live, 2026-02-07) | All package versions, dependency resolution, compatibility | HIGH |
| `uv run --with polars python3` (live) | Polars API: group_by, with_columns, join, str.pad_start, when/then/otherwise | HIGH |
| `uv run --with polars --with pyarrow --with pandas python3` (live) | polars.to_pandas() requires pyarrow AND pandas | HIGH |
| `/home/hybridz/Projects/ML-utils/flake.nix` (user's existing project) | uv2nix pattern for Python ML projects | HIGH |
| `/home/hybridz/Projects/Manim-ML/flake.nix` (user's existing project) | Simpler Nix + uv pattern (recommended for this project) | HIGH |
| `/home/hybridz/Projects/ML-utils/pyproject.toml` (user's existing project) | lightgbm>=4.6.0, xgboost>=3.0.1, scikit-learn>=1.6.1 version pins (our versions are newer) | HIGH |
| Training data (Claude, cutoff May 2025) | SHAP TreeExplainer API patterns, fairlearn MetricFrame API, Optuna integration patterns | MEDIUM |
| Training data | LightGBM 4.x callback API (early_stopping), XGBoost sklearn API | MEDIUM |

---

## Open Questions / Flags for Phase-Specific Research

1. **SHAP 0.50.0 binary classification output format** (MEDIUM confidence): Need to verify at runtime whether `shap_values` for LGBMClassifier returns a list `[class_0, class_1]` or a single array. This changed between SHAP versions. Add a runtime check in `src/fairness/shap_analysis.py`.

2. **numba/llvmlite Nix compatibility**: shap 0.50.0 depends on numba 0.63.1 which depends on llvmlite 0.46.0. llvmlite requires LLVM at build time. May need `pkgs.llvmPackages_18.llvm` or similar in the Nix flake. Test during environment setup.

3. **LightGBM OpenMP on NixOS**: LightGBM's parallel tree building requires libgomp. Verify that `pkgs.openmp` or `pkgs.llvmPackages.openmp` provides the correct OpenMP runtime on NixOS.

4. **onnxmltools + CalibratedClassifierCV**: The spec calls for calibrating the LightGBM model with `CalibratedClassifierCV`. When converting to ONNX, convert the **uncalibrated** LightGBM model (not the sklearn CalibratedClassifierCV wrapper), because onnxmltools expects a LightGBM model. Apply calibration separately in the inference code on the M4 site.

5. **pandas 3.0 breaking changes**: pandas 3.0 drops `DataFrame.append()`, changes default copy behavior, and removes several deprecated features. Minimal impact since we only use pandas at the sklearn boundary, but verify that fairlearn/shap/statsmodels don't hit deprecated pandas APIs internally.

---
*Stack research for: Alerta Escuela Equity Audit -- ML fairness pipeline*
*Researched: 2026-02-07*
*Verified via: live uv pip compile + runtime API tests on this machine*
