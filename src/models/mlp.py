"""MLP model with StandardScaler and Optuna hyperparameter tuning.

Trains an Optuna-tuned MLPClassifier with StandardScaler preprocessing,
using the same temporal splits and evaluation patterns from baseline.py.

Usage::

    uv run python src/models/mlp.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import optuna
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
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


def _mlp_objective(trial, X_train_s, y_train, w_train, X_val_s, y_val, w_val):
    """Optuna objective for MLP. Returns weighted validation PR-AUC."""
    n_layers = trial.suggest_int("n_layers", 2, 3)
    layers = tuple(trial.suggest_int(f"h{i}", 32, 256) for i in range(n_layers))
    params = {
        "hidden_layer_sizes": layers,
        "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        mlp = MLPClassifier(
            **params, activation="relu", solver="adam", max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
        mlp.fit(X_train_s, y_train, sample_weight=w_train)
    y_prob = mlp.predict_proba(X_val_s)[:, 1]
    return average_precision_score(y_val, y_prob, sample_weight=w_val)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _save_predictions_mlp(
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
            pl.lit("mlp").alias("model"),
            pl.lit(optimal_threshold).alias("threshold"),
            pl.lit(split_name).alias("split"),
        ]
    )


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------


def _plot_pr_curve_mlp(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    thresholds_data: dict,
    optimal_threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, sample_weight=w, name="MLP (weighted)", ax=ax,
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

    ax.set_title(f"Precision-Recall Curve: MLP (Validation {VALIDATE_YEAR})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"PR curve saved: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_mlp_pipeline() -> dict:
    """Train Optuna-tuned MLP, evaluate, export artifacts."""
    root = find_project_root()
    parquet_path = root / "data" / "processed" / "enaho_with_features.parquet"
    results_path = root / "data" / "exports" / "model_results.json"
    predictions_path = root / "data" / "processed" / "predictions_mlp.parquet"
    model_path = root / "data" / "processed" / "model_mlp.joblib"
    scaler_path = root / "data" / "processed" / "scaler_mlp.joblib"
    pr_curve_path = root / "data" / "exports" / "figures" / "pr_curve_mlp.png"

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

    # 4. StandardScaler (fit on train only)
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # 5. Optuna tuning (50 trials)
    print("\n" + "=" * 60)
    print("MLP OPTUNA TUNING (50 trials)")
    print("=" * 60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="mlp_prauc")
    study.optimize(
        lambda trial: _mlp_objective(trial, X_train_s, y_train, w_train, X_val_s, y_val, w_val),
        n_trials=50,
    )
    print(f"\nBest MLP trial: #{study.best_trial.number}")
    print(f"Best MLP val PR-AUC: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    # Check for fallback condition
    if study.best_trial.value < 0.12:
        print("\nWARNING: Best PR-AUC < 0.12. Re-running without sample_weight...")
        study2 = optuna.create_study(direction="maximize", study_name="mlp_prauc_nosw")
        def _mlp_objective_nosw(trial):
            n_layers = trial.suggest_int("n_layers", 2, 3)
            layers = tuple(trial.suggest_int(f"h{i}", 32, 256) for i in range(n_layers))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                mlp = MLPClassifier(
                    hidden_layer_sizes=layers,
                    alpha=trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                    learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                    activation="relu", solver="adam", max_iter=500,
                    early_stopping=True, validation_fraction=0.1, random_state=42,
                )
                mlp.fit(X_train_s, y_train)
            y_prob = mlp.predict_proba(X_val_s)[:, 1]
            return average_precision_score(y_val, y_prob, sample_weight=w_val)

        study2.optimize(_mlp_objective_nosw, n_trials=50)
        if study2.best_trial.value > study.best_trial.value:
            print(f"Fallback improved PR-AUC: {study2.best_trial.value:.4f}")
            study = study2
            use_sample_weight = False
        else:
            use_sample_weight = True
    else:
        use_sample_weight = True

    # 6. Retrain best MLP
    print("\nRetraining best MLP model...")
    best_params = study.best_trial.params
    n_layers = best_params.pop("n_layers")
    layers = tuple(best_params.pop(f"h{i}") for i in range(n_layers))
    lr_init = best_params.pop("lr")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        best_mlp = MLPClassifier(
            hidden_layer_sizes=layers,
            alpha=best_params["alpha"],
            learning_rate_init=lr_init,
            batch_size=best_params["batch_size"],
            activation="relu", solver="adam", max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
        if use_sample_weight:
            best_mlp.fit(X_train_s, y_train, sample_weight=w_train)
        else:
            best_mlp.fit(X_train_s, y_train)

    print(f"MLP converged in {best_mlp.n_iter_} iterations")

    # 7. Get predictions
    y_prob_val = best_mlp.predict_proba(X_val_s)[:, 1]
    y_prob_test = best_mlp.predict_proba(X_test_s)[:, 1]

    # 8. Threshold analysis
    threshold_data = _threshold_analysis(y_val, y_prob_val, w_val)
    optimal_threshold = threshold_data["optimal_threshold"]

    # 9. Apply threshold
    y_pred_val = (y_prob_val >= optimal_threshold).astype(int)
    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    # 10. Compute metrics
    w_val_metrics = compute_metrics(y_val, y_prob_val, y_pred_val, weights=w_val)
    uw_val_metrics = compute_metrics(y_val, y_prob_val, y_pred_val, weights=None)
    w_test_metrics = compute_metrics(y_test, y_prob_test, y_pred_test, weights=w_test)
    uw_test_metrics = compute_metrics(y_test, y_prob_test, y_pred_test, weights=None)

    # Print metrics
    print(f"\n=== MLP METRICS: Validation {VALIDATE_YEAR} ===")
    for key in w_val_metrics:
        print(f"  {key:<15s} W={w_val_metrics[key]:.4f}  UW={uw_val_metrics[key]:.4f}")
    print(f"\n=== MLP METRICS: Test {TEST_YEAR} ===")
    for key in w_test_metrics:
        print(f"  {key:<15s} W={w_test_metrics[key]:.4f}  UW={uw_test_metrics[key]:.4f}")

    # 11. Permutation importance
    print("\nComputing permutation importance (10 repeats)...")
    perm_imp = permutation_importance(
        best_mlp, X_val_s, y_val, scoring="average_precision",
        n_repeats=10, random_state=42, sample_weight=w_val,
    )
    imp_means = perm_imp.importances_mean
    imp_norm = imp_means / (imp_means.sum() + 1e-10)
    sorted_imp = sorted(
        zip(list(MODEL_FEATURES), imp_norm),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n=== TOP-10 MLP PERMUTATION IMPORTANCES ===")
    for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
        print(f"  {rank:2d}. {feat:<40s} {imp:.4f}")

    # 12. Save predictions (val + test)
    val_preds = _save_predictions_mlp(val_df, y_prob_val, y_pred_val, optimal_threshold, f"validate_{VALIDATE_YEAR}")
    test_preds = _save_predictions_mlp(test_df, y_prob_test, y_pred_test, optimal_threshold, f"test_{TEST_YEAR}")
    combined = pl.concat([val_preds, test_preds])
    combined.write_parquet(predictions_path)
    print(f"Predictions saved: {combined.height:,} rows to {predictions_path}")

    # 13. Persist model
    joblib.dump(best_mlp, model_path)
    print(f"Model saved: {model_path}")

    # 14. PR curve
    _plot_pr_curve_mlp(y_val, y_prob_val, w_val, threshold_data, optimal_threshold, pr_curve_path)

    # 15. Build entry and merge into model_results.json
    # Reconstruct tuned params for JSON serialization
    tuned_params = study.best_trial.params.copy()

    entry = {
        "metadata": {
            "model_type": "MLPClassifier",
            "train_years": TRAIN_YEARS,
            "validate_year": VALIDATE_YEAR,
            "test_year": TEST_YEAR,
            "n_train": int(X_train.shape[0]),
            "n_validate": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": len(MODEL_FEATURES),
            "feature_names": list(MODEL_FEATURES),
            "hidden_layer_sizes": list(layers),
            "scaler_fitted": True,
            "sample_weight_used": use_sample_weight,
            "n_iter": int(best_mlp.n_iter_),
            "optuna_n_trials": study.trials[-1].number + 1,
            "optuna_best_trial": study.best_trial.number,
            "optuna_best_params": tuned_params,
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
        "permutation_feature_importance": [
            {"feature": feat, "importance": round(float(imp), 6)}
            for feat, imp in sorted_imp
        ],
    }

    with open(results_path, "r") as f:
        model_results = json.load(f)
    model_results["mlp"] = entry
    with open(results_path, "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"model_results.json updated with 'mlp' key")

    # 16. Summary
    print("\n" + "=" * 60)
    print("MLP PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Val PR-AUC (W):   {w_val_metrics['pr_auc']:.4f}")
    print(f"  Test PR-AUC (W):  {w_test_metrics['pr_auc']:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Hidden layers:    {layers}")
    print(f"  Sample weight:    {use_sample_weight}")
    print(f"  Predictions:      {combined.height:,} rows")
    print(f"  Optuna trials:    {len(study.trials)}")
    print("=" * 60)

    return model_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_mlp_pipeline()
