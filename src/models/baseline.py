"""Baseline logistic regression model with temporal splits and survey-weighted evaluation.

Establishes temporal split discipline (train=2018-2021, validate=2022, test=2023)
and trains a logistic regression baseline with survey-weighted evaluation, threshold
analysis, and coefficient inference. This module sets patterns reused by Phases 6-7.

Usage::

    uv run python src/models/baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import MODEL_FEATURES
from utils import find_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALIDATE_YEAR = 2022
TEST_YEAR = 2023
FIXED_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Identifying columns to carry into predictions parquet
ID_COLUMNS = [
    "CONGLOME",
    "VIVIENDA",
    "HOGAR",
    "CODPERSO",
    "year",
    "UBIGEO",
    "FACTOR07",
    "dropout",
]


# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------


def create_temporal_splits(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split the dataset by year into train/validate/test with zero overlap.

    Parameters
    ----------
    df : pl.DataFrame
        Full feature matrix with ``year`` column.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (train_df, val_df, test_df)
    """
    train_df = df.filter(pl.col("year").is_in(TRAIN_YEARS))
    val_df = df.filter(pl.col("year") == VALIDATE_YEAR)
    test_df = df.filter(pl.col("year") == TEST_YEAR)

    # Verify year coverage
    train_years = set(train_df["year"].unique().to_list())
    val_years = set(val_df["year"].unique().to_list())
    test_years = set(test_df["year"].unique().to_list())

    assert train_years == set(TRAIN_YEARS), f"Train years mismatch: {train_years}"
    assert val_years == {VALIDATE_YEAR}, f"Val years mismatch: {val_years}"
    assert test_years == {TEST_YEAR}, f"Test years mismatch: {test_years}"

    # Verify no overlap
    assert not (train_years & val_years), "Train/val year overlap"
    assert not (train_years & test_years), "Train/test year overlap"
    assert not (val_years & test_years), "Val/test year overlap"

    # Verify complete partition
    assert train_df.height + val_df.height + test_df.height == df.height, (
        f"Split rows ({train_df.height} + {val_df.height} + {test_df.height}) "
        f"!= total ({df.height})"
    )

    print(
        f"Train: {train_df.height:,} rows ({sorted(train_years)}), "
        f"Validate: {val_df.height:,} rows ({VALIDATE_YEAR}), "
        f"Test: {test_df.height:,} rows ({TEST_YEAR})"
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Numpy conversion
# ---------------------------------------------------------------------------


def _df_to_numpy(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Polars DataFrame to numpy arrays at the sklearn boundary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, y, w) -- features, target, weights.
    """
    X = df.select(MODEL_FEATURES).to_numpy()
    y = df["dropout"].cast(pl.Int8).to_numpy()
    w = df["FACTOR07"].to_numpy()
    return X, y, w


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Compute the full metric suite with optional survey weights.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    y_pred : np.ndarray
        Predicted binary labels (at some threshold).
    weights : np.ndarray or None
        Survey weights (FACTOR07). None for unweighted.

    Returns
    -------
    dict
        Metric name -> float value.
    """
    return {
        "pr_auc": float(
            average_precision_score(y_true, y_prob, sample_weight=weights)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob, sample_weight=weights)
        ),
        "f1": float(f1_score(y_true, y_pred, sample_weight=weights)),
        "precision": float(
            precision_score(y_true, y_pred, sample_weight=weights)
        ),
        "recall": float(recall_score(y_true, y_pred, sample_weight=weights)),
        "brier": float(
            brier_score_loss(y_true, y_prob, sample_weight=weights)
        ),
        "log_loss": float(log_loss(y_true, y_prob, sample_weight=weights)),
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
) -> LogisticRegression:
    """Train sklearn LogisticRegression with balanced class weights and survey weights.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Binary target.
    w_train : np.ndarray
        Survey expansion weights (FACTOR07).

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_train, y_train, sample_weight=w_train)
    print(f"LR converged in {lr.n_iter_[0]} iterations")
    return lr


# ---------------------------------------------------------------------------
# Coefficient inference (statsmodels)
# ---------------------------------------------------------------------------


def _get_coefficient_inference(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
) -> list[dict]:
    """Train statsmodels GLM(Binomial) for coefficient inference.

    Uses freq_weights to incorporate survey weights. Returns a list of dicts
    with feature, coefficient, std_error, odds_ratio, p_value, ci_lower,
    ci_upper for each coefficient (including intercept).

    Note: freq_weights inflates effective n to ~25M, making all p-values
    effectively 0.  Use odds ratios and coefficient signs for interpretation.
    """
    X_const = sm.add_constant(X_train)
    glm = sm.GLM(
        y_train.astype(float),
        X_const,
        family=sm.families.Binomial(),
        freq_weights=w_train,
    )
    glm_result = glm.fit()

    feature_names = ["intercept"] + list(MODEL_FEATURES)
    conf_int = glm_result.conf_int()

    coef_table: list[dict] = []
    for i, name in enumerate(feature_names):
        coef_table.append(
            {
                "feature": name,
                "coefficient": round(float(glm_result.params[i]), 6),
                "std_error": round(float(glm_result.bse[i]), 6),
                "odds_ratio": round(float(np.exp(glm_result.params[i])), 6),
                "p_value": round(float(glm_result.pvalues[i]), 6),
                "ci_lower": round(float(conf_int[i, 0]), 6),
                "ci_upper": round(float(conf_int[i, 1]), 6),
            }
        )

    return coef_table


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def _threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
) -> dict:
    """Threshold tuning at 5 fixed thresholds plus the optimal (max weighted F1).

    Returns
    -------
    dict
        optimal_threshold, optimization_target, thresholds list.
    """
    # Weighted PR curve for optimal threshold search
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
        y_true, y_prob, sample_weight=w
    )

    # Compute F1 at each curve threshold
    f1_arr = (
        2
        * (precision_arr[:-1] * recall_arr[:-1])
        / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
    )
    optimal_idx = int(np.argmax(f1_arr))
    optimal_threshold = float(thresholds_arr[optimal_idx])

    # Evaluate at fixed thresholds
    threshold_entries: list[dict] = []
    for t in FIXED_THRESHOLDS:
        y_pred_t = (y_prob >= t).astype(int)
        w_metrics = {
            "weighted_f1": float(f1_score(y_true, y_pred_t, sample_weight=w)),
            "weighted_precision": float(
                precision_score(y_true, y_pred_t, sample_weight=w)
            ),
            "weighted_recall": float(
                recall_score(y_true, y_pred_t, sample_weight=w)
            ),
        }
        uw_metrics = {
            "unweighted_f1": float(f1_score(y_true, y_pred_t)),
            "unweighted_precision": float(precision_score(y_true, y_pred_t)),
            "unweighted_recall": float(recall_score(y_true, y_pred_t)),
        }
        threshold_entries.append({"threshold": t, **w_metrics, **uw_metrics})

    # Add optimal threshold entry
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    opt_w_metrics = {
        "weighted_f1": float(
            f1_score(y_true, y_pred_opt, sample_weight=w)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred_opt, sample_weight=w)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred_opt, sample_weight=w)
        ),
    }
    opt_uw_metrics = {
        "unweighted_f1": float(f1_score(y_true, y_pred_opt)),
        "unweighted_precision": float(precision_score(y_true, y_pred_opt)),
        "unweighted_recall": float(recall_score(y_true, y_pred_opt)),
    }
    threshold_entries.append(
        {
            "threshold": round(optimal_threshold, 6),
            "is_optimal": True,
            **opt_w_metrics,
            **opt_uw_metrics,
        }
    )

    return {
        "optimal_threshold": round(optimal_threshold, 6),
        "optimization_target": "max_weighted_f1",
        "thresholds": threshold_entries,
    }


# ---------------------------------------------------------------------------
# Predictions parquet
# ---------------------------------------------------------------------------


def _save_predictions(
    df: pl.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    optimal_threshold: float,
    split_name: str,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Build a predictions DataFrame with ID columns and model outputs.

    Parameters
    ----------
    df : pl.DataFrame
        Source DataFrame with ID_COLUMNS.
    y_prob : np.ndarray
        Predicted probabilities.
    y_pred : np.ndarray
        Predicted binary labels.
    optimal_threshold : float
        Threshold used for binary predictions.
    split_name : str
        Split identifier (e.g. "validate_2022").
    output_path : Path or None
        If provided, write parquet to this path.

    Returns
    -------
    pl.DataFrame
        Predictions DataFrame.
    """
    pred_df = df.select(ID_COLUMNS).with_columns(
        [
            pl.Series("prob_dropout", y_prob),
            pl.Series("pred_dropout", y_pred),
            pl.lit("logistic_regression").alias("model"),
            pl.lit(optimal_threshold).alias("threshold"),
            pl.lit(split_name).alias("split"),
        ]
    )

    if output_path is not None:
        pred_df.write_parquet(output_path)
        print(f"Predictions saved: {pred_df.height:,} rows to {output_path}")

    return pred_df


# ---------------------------------------------------------------------------
# PR curve visualization
# ---------------------------------------------------------------------------


def _plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    thresholds_data: dict,
    optimal_threshold: float,
    output_path: Path,
) -> None:
    """Generate precision-recall curve PNG with threshold markers.

    Follows Phase 4 matplotlib patterns (Agg backend, tight_layout, plt.close).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # PR curve via PrecisionRecallDisplay
    disp = PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        sample_weight=w,
        name="LR (weighted)",
        ax=ax,
    )

    # Chance level: weighted base rate
    base_rate = float(np.average(y_true, weights=w))
    ax.axhline(
        y=base_rate,
        color="gray",
        linestyle="--",
        label=f"Chance level ({base_rate:.3f})",
    )

    # Get the full PR curve for marker placement
    prec_arr, rec_arr, thr_arr = precision_recall_curve(
        y_true, y_prob, sample_weight=w
    )

    # Plot markers at fixed thresholds
    for entry in thresholds_data["thresholds"]:
        t = entry["threshold"]
        is_opt = entry.get("is_optimal", False)

        # Find nearest threshold in the curve
        idx = int(np.argmin(np.abs(thr_arr - t)))
        p_at_t = prec_arr[idx]
        r_at_t = rec_arr[idx]

        if is_opt:
            ax.plot(
                r_at_t,
                p_at_t,
                marker="*",
                markersize=15,
                color="red",
                zorder=5,
                label=f"Optimal t={t:.3f}",
            )
            ax.annotate(
                f"  t={t:.3f}",
                (r_at_t, p_at_t),
                fontsize=9,
                color="red",
                fontweight="bold",
            )
        else:
            ax.plot(
                r_at_t,
                p_at_t,
                marker="o",
                markersize=8,
                color="darkorange",
                zorder=4,
            )
            ax.annotate(
                f" {t:.1f}",
                (r_at_t, p_at_t),
                fontsize=8,
                color="darkorange",
            )

    ax.set_title(
        "Precision-Recall Curve: Logistic Regression (Validation 2022)"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"PR curve saved: {output_path}")


# ---------------------------------------------------------------------------
# Console printing helpers
# ---------------------------------------------------------------------------

# Equity-relevant features for highlighting
_EQUITY_FEATURES = {
    "poverty_quintile",
    "rural",
    "lang_other_indigenous",
    "lang_quechua",
    "lang_aimara",
    "age",
    "es_mujer",
    "es_peruano",
}


def _print_coefficient_table(coef_table: list[dict]) -> None:
    """Print coefficient table to console for human review."""
    print("\n=== LOGISTIC REGRESSION COEFFICIENTS ===")
    header = f"{'Feature':<35s} {'Coef':>10s} {'SE':>10s} {'OR':>10s} {'p-value':>10s}"
    print(header)
    print("-" * len(header))

    for row in coef_table:
        mark = " ***" if row["feature"] in _EQUITY_FEATURES else ""
        print(
            f"{row['feature']:<35s} "
            f"{row['coefficient']:>10.4f} "
            f"{row['std_error']:>10.4f} "
            f"{row['odds_ratio']:>10.4f} "
            f"{row['p_value']:>10.4f}"
            f"{mark}"
        )
    print()


def _print_metrics_comparison(
    weighted: dict, unweighted: dict, split_name: str
) -> None:
    """Print weighted vs unweighted metrics side-by-side."""
    print(f"\n=== METRICS: {split_name} ===")
    header = f"{'Metric':<15s} {'Weighted':>12s} {'Unweighted':>12s} {'Diff':>10s}"
    print(header)
    print("-" * len(header))

    for key in weighted:
        w_val = weighted[key]
        uw_val = unweighted[key]
        diff = w_val - uw_val
        print(f"{key:<15s} {w_val:>12.4f} {uw_val:>12.4f} {diff:>+10.4f}")
    print()


def _print_threshold_table(threshold_data: dict) -> None:
    """Print threshold analysis table to console."""
    print("\n=== THRESHOLD ANALYSIS ===")
    header = (
        f"{'Threshold':>10s}  "
        f"{'W-F1':>8s} {'W-Prec':>8s} {'W-Recall':>8s}  "
        f"{'UW-F1':>8s} {'UW-Prec':>8s} {'UW-Recall':>8s}"
    )
    print(header)
    print("-" * len(header))

    for entry in threshold_data["thresholds"]:
        is_opt = entry.get("is_optimal", False)
        t_str = f"{entry['threshold']:.4f}" + ("*" if is_opt else " ")
        print(
            f"{t_str:>10s}  "
            f"{entry['weighted_f1']:>8.4f} "
            f"{entry['weighted_precision']:>8.4f} "
            f"{entry['weighted_recall']:>8.4f}  "
            f"{entry['unweighted_f1']:>8.4f} "
            f"{entry['unweighted_precision']:>8.4f} "
            f"{entry['unweighted_recall']:>8.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# JSON export builder
# ---------------------------------------------------------------------------


def _build_model_results_json(
    metadata: dict,
    metrics_val: dict,
    metrics_test: dict,
    threshold_data: dict,
    coef_table: list[dict],
) -> dict:
    """Build model_results.json structure.

    Parameters
    ----------
    metadata : dict
        Model metadata (hyperparams, convergence, etc.).
    metrics_val : dict
        {"weighted": {...}, "unweighted": {...}} for validation.
    metrics_test : dict
        {"weighted": {...}, "unweighted": {...}} for test.
    threshold_data : dict
        Threshold analysis results.
    coef_table : list[dict]
        Coefficient table from statsmodels.

    Returns
    -------
    dict
        Full model_results.json content.
    """

    def _round_dict(d: dict, decimals: int = 6) -> dict:
        return {
            k: round(v, decimals) if isinstance(v, float) else v
            for k, v in d.items()
        }

    return {
        "logistic_regression": {
            "metadata": metadata,
            "metrics": {
                "validate_2022": {
                    "weighted": _round_dict(metrics_val["weighted"]),
                    "unweighted": _round_dict(metrics_val["unweighted"]),
                },
                "test_2023": {
                    "weighted": _round_dict(metrics_test["weighted"]),
                    "unweighted": _round_dict(metrics_test["unweighted"]),
                },
            },
            "threshold_analysis": threshold_data,
            "coefficients": coef_table,
        }
    }


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_baseline_pipeline() -> dict:
    """Run the full baseline modeling pipeline.

    1. Load enaho_with_features.parquet
    2. Create temporal splits
    3. Train sklearn LR
    4. Get statsmodels coefficient inference
    5. Evaluate on validation and test sets
    6. Threshold analysis
    7. Save predictions, model, PR curve, model_results.json

    Returns
    -------
    dict
        model_results.json content.
    """
    root = find_project_root()
    parquet_path = root / "data" / "processed" / "enaho_with_features.parquet"
    results_path = root / "data" / "exports" / "model_results.json"
    predictions_path = root / "data" / "processed" / "predictions_lr.parquet"
    model_path = root / "data" / "processed" / "model_lr.joblib"
    pr_curve_path = root / "data" / "exports" / "figures" / "pr_curve_lr.png"

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("Loading feature matrix...")
    df = pl.read_parquet(parquet_path)
    print(f"Loaded: {df.height:,} rows, {df.width} columns")

    # -----------------------------------------------------------------------
    # 2. Create temporal splits
    # -----------------------------------------------------------------------
    print("\nCreating temporal splits...")
    train_df, val_df, test_df = create_temporal_splits(df)

    # -----------------------------------------------------------------------
    # 3. Convert to numpy
    # -----------------------------------------------------------------------
    X_train, y_train, w_train = _df_to_numpy(train_df)
    X_val, y_val, w_val = _df_to_numpy(val_df)
    X_test, y_test, w_test = _df_to_numpy(test_df)

    print(f"\nFeature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # -----------------------------------------------------------------------
    # 4. Train sklearn LR
    # -----------------------------------------------------------------------
    print("\nTraining logistic regression...")
    lr = train_logistic_regression(X_train, y_train, w_train)

    # -----------------------------------------------------------------------
    # 5. Get predictions
    # -----------------------------------------------------------------------
    y_prob_val = lr.predict_proba(X_val)[:, 1]
    y_prob_test = lr.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------------------
    # 6. Coefficient inference from statsmodels
    # -----------------------------------------------------------------------
    print("\nGetting coefficient inference from statsmodels...")
    coef_table = _get_coefficient_inference(X_train, y_train, w_train)
    _print_coefficient_table(coef_table)

    # Verify sklearn/statsmodels consistency (first non-intercept feature)
    sklearn_coef0 = float(lr.coef_[0][0])
    sm_coef1 = coef_table[1]["coefficient"]
    coef_diff = abs(sklearn_coef0 - sm_coef1)
    if coef_diff > 0.1:
        logger.warning(
            "sklearn/statsmodels coefficient mismatch for %s: "
            "sklearn=%.6f, statsmodels=%.6f (diff=%.6f). "
            "Expected due to class_weight='balanced' interaction.",
            MODEL_FEATURES[0],
            sklearn_coef0,
            sm_coef1,
            coef_diff,
        )
    else:
        print(
            f"sklearn/statsmodels consistency: diff={coef_diff:.6f} "
            f"(threshold=0.1) -- OK"
        )

    # -----------------------------------------------------------------------
    # 7. Threshold analysis on validation
    # -----------------------------------------------------------------------
    print("\nRunning threshold analysis...")
    threshold_data = _threshold_analysis(y_val, y_prob_val, w_val)
    optimal_threshold = threshold_data["optimal_threshold"]
    _print_threshold_table(threshold_data)

    # -----------------------------------------------------------------------
    # 8. Apply optimal threshold for evaluation
    # -----------------------------------------------------------------------
    y_pred_val = (y_prob_val >= optimal_threshold).astype(int)
    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    # -----------------------------------------------------------------------
    # 9. Compute weighted and unweighted metrics
    # -----------------------------------------------------------------------
    print("Computing metrics...")
    weighted_val = compute_metrics(y_val, y_prob_val, y_pred_val, weights=w_val)
    unweighted_val = compute_metrics(y_val, y_prob_val, y_pred_val, weights=None)

    weighted_test = compute_metrics(
        y_test, y_prob_test, y_pred_test, weights=w_test
    )
    unweighted_test = compute_metrics(
        y_test, y_prob_test, y_pred_test, weights=None
    )

    # Assert weighted != unweighted
    pr_auc_diff = abs(weighted_val["pr_auc"] - unweighted_val["pr_auc"])
    assert pr_auc_diff > 0.001, (
        f"Weighted and unweighted PR-AUC are too similar: "
        f"diff={pr_auc_diff:.6f} (threshold=0.001)"
    )
    print(f"\nWeighted != Unweighted check: PR-AUC diff = {pr_auc_diff:.6f} > 0.001 -- PASS")

    _print_metrics_comparison(weighted_val, unweighted_val, "Validation 2022")
    _print_metrics_comparison(weighted_test, unweighted_test, "Test 2023")

    # -----------------------------------------------------------------------
    # 10. Save predictions (val + test combined)
    # -----------------------------------------------------------------------
    print("\nSaving predictions...")
    val_preds = _save_predictions(
        val_df, y_prob_val, y_pred_val, optimal_threshold, "validate_2022"
    )
    test_preds = _save_predictions(
        test_df, y_prob_test, y_pred_test, optimal_threshold, "test_2023"
    )
    combined_preds = pl.concat([val_preds, test_preds])
    combined_preds.write_parquet(predictions_path)
    print(f"Combined predictions saved: {combined_preds.height:,} rows to {predictions_path}")

    # -----------------------------------------------------------------------
    # 11. Persist model
    # -----------------------------------------------------------------------
    joblib.dump(lr, model_path)
    print(f"Model saved: {model_path}")

    # -----------------------------------------------------------------------
    # 12. Generate PR curve
    # -----------------------------------------------------------------------
    _plot_pr_curve(
        y_val,
        y_prob_val,
        w_val,
        threshold_data,
        optimal_threshold,
        pr_curve_path,
    )

    # -----------------------------------------------------------------------
    # 13. Build and save model_results.json
    # -----------------------------------------------------------------------
    metadata = {
        "model_type": "LogisticRegression",
        "train_years": TRAIN_YEARS,
        "validate_year": VALIDATE_YEAR,
        "test_year": TEST_YEAR,
        "n_train": int(train_df.height),
        "n_validate": int(val_df.height),
        "n_test": int(test_df.height),
        "n_features": len(MODEL_FEATURES),
        "feature_names": list(MODEL_FEATURES),
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "n_iter_actual": int(lr.n_iter_[0]),
        "convergence": True,
        "year_shift_note": (
            "ENAHO 2024 unavailable; train/val/test shifted back by 1 year from spec"
        ),
        "covid_note": (
            "2020 has reduced sample (~13,755 rows) due to COVID phone interviews"
        ),
        "weight_note": (
            "freq_weights inflates effective n to ~25M; all p-values are "
            "effectively 0. Use odds ratios and coefficient signs for interpretation."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_val_combined = {
        "weighted": weighted_val,
        "unweighted": unweighted_val,
    }
    metrics_test_combined = {
        "weighted": weighted_test,
        "unweighted": unweighted_test,
    }

    model_results = _build_model_results_json(
        metadata,
        metrics_val_combined,
        metrics_test_combined,
        threshold_data,
        coef_table,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"\nmodel_results.json saved: {results_path}")

    # -----------------------------------------------------------------------
    # 14. Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:            LogisticRegression (class_weight='balanced')")
    print(f"  Train:            {train_df.height:,} rows ({TRAIN_YEARS})")
    print(f"  Validate:         {val_df.height:,} rows ({VALIDATE_YEAR})")
    print(f"  Test:             {test_df.height:,} rows ({TEST_YEAR})")
    print(f"  Features:         {len(MODEL_FEATURES)}")
    print(f"  Convergence:      {lr.n_iter_[0]} iterations")
    print(f"  Val PR-AUC (W):   {weighted_val['pr_auc']:.4f}")
    print(f"  Val PR-AUC (UW):  {unweighted_val['pr_auc']:.4f}")
    print(f"  Test PR-AUC (W):  {weighted_test['pr_auc']:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Coefficients:     {len(coef_table)}")
    print(f"  Predictions:      {combined_preds.height:,} rows")
    print("=" * 60)

    return model_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_baseline_pipeline()
