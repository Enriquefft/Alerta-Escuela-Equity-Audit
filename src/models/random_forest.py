"""Random Forest model with Optuna hyperparameter tuning.

Trains an Optuna-tuned RandomForestClassifier using the same temporal splits
and evaluation patterns established in baseline.py.

Usage::

    uv run python src/models/random_forest.py
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
import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import MODEL_FEATURES
from models.baseline import (
    create_temporal_splits,
    _df_to_numpy,
    compute_metrics,
    _threshold_analysis,
    TRAIN_YEARS,
    VALIDATE_YEAR,
    TEST_YEAR,
    ID_COLUMNS,
)
from utils import find_project_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_dict(d: dict, decimals: int = 6) -> dict:
    return {
        k: round(v, decimals) if isinstance(v, float) else v for k, v in d.items()
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _rf_objective(trial, X_train, y_train, w_train, X_val, y_val, w_val):
    """Optuna objective for RF. Returns weighted validation PR-AUC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    rf = RandomForestClassifier(
        **params, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    y_prob = rf.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob, sample_weight=w_val)


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def _extract_feature_importances(model, feature_names: list[str]) -> list[tuple[str, float]]:
    raw_imp = model.feature_importances_
    norm_imp = raw_imp / raw_imp.sum()
    return sorted(zip(feature_names, norm_imp), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _save_predictions_rf(
    df: pl.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    optimal_threshold: float,
    split_name: str,
) -> pl.DataFrame:
    return df.select(ID_COLUMNS).with_columns(
        [
            pl.Series("prob_dropout", y_prob),
            pl.Series("pred_dropout", y_pred),
            pl.lit("random_forest").alias("model"),
            pl.lit(optimal_threshold).alias("threshold"),
            pl.lit(split_name).alias("split"),
        ]
    )


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------


def _plot_pr_curve_rf(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    thresholds_data: dict,
    optimal_threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, sample_weight=w, name="RF (weighted)", ax=ax,
    )
    base_rate = float(np.average(y_true, weights=w))
    ax.axhline(y=base_rate, color="gray", linestyle="--", label=f"Chance ({base_rate:.3f})")

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob, sample_weight=w)
    for entry in thresholds_data["thresholds"]:
        t = entry["threshold"]
        is_opt = entry.get("is_optimal", False)
        idx = int(np.argmin(np.abs(thr_arr - t)))
        if is_opt:
            ax.plot(rec_arr[idx], prec_arr[idx], marker="*", markersize=15, color="red", zorder=5, label=f"Optimal t={t:.3f}")
        else:
            ax.plot(rec_arr[idx], prec_arr[idx], marker="o", markersize=8, color="darkorange", zorder=4)
            ax.annotate(f" {t:.1f}", (rec_arr[idx], prec_arr[idx]), fontsize=8, color="darkorange")

    ax.set_title(f"Precision-Recall Curve: Random Forest (Validation {VALIDATE_YEAR})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"PR curve saved: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_rf_pipeline() -> dict:
    """Train Optuna-tuned Random Forest, evaluate, export artifacts."""
    root = find_project_root()
    parquet_path = root / "data" / "processed" / "enaho_with_features.parquet"
    results_path = root / "data" / "exports" / "model_results.json"
    predictions_path = root / "data" / "processed" / "predictions_rf.parquet"
    model_path = root / "data" / "processed" / "model_rf.joblib"
    pr_curve_path = root / "data" / "exports" / "figures" / "pr_curve_rf.png"

    # 1. Load data
    print("Loading feature matrix...")
    df = pl.read_parquet(parquet_path)
    print(f"Loaded: {df.height:,} rows, {df.width} columns")

    # 2. Temporal splits
    print("\nCreating temporal splits...")
    train_df, val_df, test_df = create_temporal_splits(df)

    # 3. Convert to numpy
    X_train, y_train, w_train = _df_to_numpy(train_df)
    X_val, y_val, w_val = _df_to_numpy(val_df)
    X_test, y_test, w_test = _df_to_numpy(test_df)
    print(f"\nFeature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # 4. Optuna tuning (50 trials)
    print("\n" + "=" * 60)
    print("RANDOM FOREST OPTUNA TUNING (50 trials)")
    print("=" * 60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="rf_prauc")
    study.optimize(
        lambda trial: _rf_objective(trial, X_train, y_train, w_train, X_val, y_val, w_val),
        n_trials=50,
    )
    print(f"\nBest RF trial: #{study.best_trial.number}")
    print(f"Best RF val PR-AUC: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    # 5. Retrain best RF
    print("\nRetraining best Random Forest model...")
    best_params = study.best_trial.params
    best_rf = RandomForestClassifier(
        **best_params, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    best_rf.fit(X_train, y_train, sample_weight=w_train)

    # 6. Get predictions
    y_prob_val = best_rf.predict_proba(X_val)[:, 1]
    y_prob_test = best_rf.predict_proba(X_test)[:, 1]

    # 7. Threshold analysis
    threshold_data = _threshold_analysis(y_val, y_prob_val, w_val)
    optimal_threshold = threshold_data["optimal_threshold"]

    # 8. Apply threshold
    y_pred_val = (y_prob_val >= optimal_threshold).astype(int)
    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    # 9. Compute metrics
    w_val_metrics = compute_metrics(y_val, y_prob_val, y_pred_val, weights=w_val)
    uw_val_metrics = compute_metrics(y_val, y_prob_val, y_pred_val, weights=None)
    w_test_metrics = compute_metrics(y_test, y_prob_test, y_pred_test, weights=w_test)
    uw_test_metrics = compute_metrics(y_test, y_prob_test, y_pred_test, weights=None)

    # Print metrics
    print(f"\n=== RF METRICS: Validation {VALIDATE_YEAR} ===")
    for key in w_val_metrics:
        print(f"  {key:<15s} W={w_val_metrics[key]:.4f}  UW={uw_val_metrics[key]:.4f}")
    print(f"\n=== RF METRICS: Test {TEST_YEAR} ===")
    for key in w_test_metrics:
        print(f"  {key:<15s} W={w_test_metrics[key]:.4f}  UW={uw_test_metrics[key]:.4f}")

    # 10. Feature importances
    sorted_imp = _extract_feature_importances(best_rf, list(MODEL_FEATURES))
    print(f"\n=== TOP-10 RF FEATURE IMPORTANCES ===")
    for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
        print(f"  {rank:2d}. {feat:<40s} {imp:.4f}")
    max_imp = sorted_imp[0][1]
    assert max_imp < 0.50, f"Feature {sorted_imp[0][0]} has {max_imp:.4f} importance (>50%)"

    # 11. Save predictions (val + test)
    val_preds = _save_predictions_rf(val_df, y_prob_val, y_pred_val, optimal_threshold, f"validate_{VALIDATE_YEAR}")
    test_preds = _save_predictions_rf(test_df, y_prob_test, y_pred_test, optimal_threshold, f"test_{TEST_YEAR}")
    combined = pl.concat([val_preds, test_preds])
    combined.write_parquet(predictions_path)
    print(f"Predictions saved: {combined.height:,} rows to {predictions_path}")

    # 12. Persist model
    joblib.dump(best_rf, model_path)
    print(f"Model saved: {model_path}")

    # 13. PR curve
    _plot_pr_curve_rf(y_val, y_prob_val, w_val, threshold_data, optimal_threshold, pr_curve_path)

    # 14. Build entry and merge into model_results.json
    entry = {
        "metadata": {
            "model_type": "RandomForestClassifier",
            "train_years": TRAIN_YEARS,
            "validate_year": VALIDATE_YEAR,
            "test_year": TEST_YEAR,
            "n_train": int(X_train.shape[0]),
            "n_validate": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": len(MODEL_FEATURES),
            "feature_names": list(MODEL_FEATURES),
            "optuna_n_trials": study.trials[-1].number + 1,
            "optuna_best_trial": study.best_trial.number,
            "optuna_best_params": study.best_trial.params,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": {
            f"validate_{VALIDATE_YEAR}": {
                "weighted": _round_dict(w_val_metrics),
                "unweighted": _round_dict(uw_val_metrics),
            },
            f"test_{TEST_YEAR}": {
                "weighted": _round_dict(w_test_metrics),
                "unweighted": _round_dict(uw_test_metrics),
            },
        },
        "threshold_analysis": threshold_data,
        "feature_importances": [
            {"feature": feat, "importance": round(float(imp), 6)}
            for feat, imp in sorted_imp
        ],
    }

    # Merge into existing model_results.json
    with open(results_path, "r") as f:
        model_results = json.load(f)
    model_results["random_forest"] = entry
    with open(results_path, "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"model_results.json updated with 'random_forest' key")

    # 15. Summary
    print("\n" + "=" * 60)
    print("RANDOM FOREST PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Val PR-AUC (W):   {w_val_metrics['pr_auc']:.4f}")
    print(f"  Test PR-AUC (W):  {w_test_metrics['pr_auc']:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Predictions:      {combined.height:,} rows")
    print(f"  Optuna trials:    {len(study.trials)}")
    print("=" * 60)

    return model_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_rf_pipeline()
