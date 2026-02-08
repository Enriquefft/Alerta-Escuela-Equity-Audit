"""LightGBM and XGBoost models with Optuna hyperparameter tuning.

Trains Optuna-tuned LightGBM (primary, matching Alerta Escuela) and XGBoost
(comparison) models using the same temporal splits and evaluation patterns
established in baseline.py.

Usage::

    uv run python src/models/lightgbm_xgboost.py
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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
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
    FIXED_THRESHOLDS,
    ID_COLUMNS,
)
from utils import find_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_dict(d: dict, decimals: int = 6) -> dict:
    """Round all float values in a dict to *decimals* places."""
    return {
        k: round(v, decimals) if isinstance(v, float) else v for k, v in d.items()
    }


def _compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight = n_neg / n_pos from training labels."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / n_pos
    print(
        f"Class imbalance: {n_neg:,} neg / {n_pos:,} pos = scale_pos_weight={spw:.4f}"
    )
    return spw


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------


def _lgbm_objective(
    trial,
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
    w_val,
    spw,
) -> float:
    """Optuna objective for LightGBM. Returns weighted validation PR-AUC."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(
        n_estimators=500,
        **params,
        scale_pos_weight=spw,
        importance_type="gain",
        verbose=-1,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="average_precision",
        callbacks=[early_stopping(50, first_metric_only=True), log_evaluation(0)],
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob, sample_weight=w_val)


def _xgb_objective(
    trial,
    X_train,
    y_train,
    w_train,
    X_val,
    y_val,
    w_val,
    spw,
) -> float:
    """Optuna objective for XGBoost. Returns weighted validation PR-AUC."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = XGBClassifier(
        n_estimators=500,
        **params,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        importance_type="gain",
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob, sample_weight=w_val)


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def _extract_feature_importances(
    model, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Extract and normalize feature importances (gain, sum=1)."""
    raw_imp = model.feature_importances_
    norm_imp = raw_imp / raw_imp.sum()
    importance_pairs = list(zip(feature_names, norm_imp))
    sorted_imp = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
    return sorted_imp


def _print_feature_importances(
    sorted_importances: list[tuple[str, float]], model_name: str
) -> None:
    """Print top-10 feature importances for human review."""
    print(
        f"\n=== TOP-10 {model_name.upper()} FEATURE IMPORTANCES (GAIN, NORMALIZED) ==="
    )
    for rank, (feat, imp) in enumerate(sorted_importances[:10], 1):
        equity_mark = " ***" if feat in _EQUITY_FEATURES else ""
        print(f"  {rank:2d}. {feat:<40s} {imp:.4f}{equity_mark}")

    max_feat, max_imp = sorted_importances[0]
    print(f"\n  Max importance: {max_feat} = {max_imp:.4f} (must be < 0.50)")
    assert max_imp < 0.50, (
        f"Feature {max_feat} has {max_imp:.4f} normalized importance (>50%)"
    )
    print("  Importance concentration check: PASS")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _save_predictions_gbm(
    df: pl.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    optimal_threshold: float,
    split_name: str,
    model_name: str,
) -> pl.DataFrame:
    """Build predictions DataFrame for a GBM model."""
    pred_df = df.select(ID_COLUMNS).with_columns(
        [
            pl.Series("prob_dropout", y_prob),
            pl.Series("pred_dropout", y_pred),
            pl.lit(model_name).alias("model"),
            pl.lit(optimal_threshold).alias("threshold"),
            pl.lit(split_name).alias("split"),
        ]
    )
    return pred_df


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------


def _plot_pr_curve_gbm(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    thresholds_data: dict,
    optimal_threshold: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Generate PR curve PNG, parameterized by model_name."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # PR curve via PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        sample_weight=w,
        name=f"{model_name} (weighted)",
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

    # Get full PR curve for marker placement
    prec_arr, rec_arr, thr_arr = precision_recall_curve(
        y_true, y_prob, sample_weight=w
    )

    # Plot markers at fixed thresholds
    for entry in thresholds_data["thresholds"]:
        t = entry["threshold"]
        is_opt = entry.get("is_optimal", False)

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
        f"Precision-Recall Curve: {model_name} (Validation {VALIDATE_YEAR})"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"PR curve saved: {output_path}")


# ---------------------------------------------------------------------------
# Common evaluation
# ---------------------------------------------------------------------------


def _train_and_evaluate_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_test: np.ndarray,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_names: list[str],
    study,
    spw: float,
    root: Path,
) -> dict:
    """Evaluate a trained model, save artifacts, return JSON entry."""
    # 1. Get predictions
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # 2. Threshold analysis on validation
    threshold_data = _threshold_analysis(y_val, y_prob_val, w_val)
    optimal_threshold = threshold_data["optimal_threshold"]

    # 3. Apply threshold
    y_pred_val = (y_prob_val >= optimal_threshold).astype(int)
    y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

    # 4. Compute metrics
    w_val_metrics = compute_metrics(y_val, y_prob_val, y_pred_val, weights=w_val)
    uw_val_metrics = compute_metrics(y_val, y_prob_val, y_pred_val, weights=None)
    w_test_metrics = compute_metrics(y_test, y_prob_test, y_pred_test, weights=w_test)
    uw_test_metrics = compute_metrics(y_test, y_prob_test, y_pred_test, weights=None)

    # 5. Print metrics comparison
    print(f"\n=== {model_name.upper()} METRICS: Validation {VALIDATE_YEAR} ===")
    print(f"  {'Metric':<15s} {'Weighted':>12s} {'Unweighted':>12s} {'Diff':>10s}")
    for key in w_val_metrics:
        w_v = w_val_metrics[key]
        uw_v = uw_val_metrics[key]
        print(f"  {key:<15s} {w_v:>12.4f} {uw_v:>12.4f} {w_v - uw_v:>+10.4f}")

    print(f"\n=== {model_name.upper()} METRICS: Test {TEST_YEAR} ===")
    print(f"  {'Metric':<15s} {'Weighted':>12s} {'Unweighted':>12s} {'Diff':>10s}")
    for key in w_test_metrics:
        w_v = w_test_metrics[key]
        uw_v = uw_test_metrics[key]
        print(f"  {key:<15s} {w_v:>12.4f} {uw_v:>12.4f} {w_v - uw_v:>+10.4f}")

    # 6. Feature importances
    sorted_imp = _extract_feature_importances(model, feature_names)
    _print_feature_importances(sorted_imp, model_name)

    # 7. Save predictions (val + test combined)
    val_preds = _save_predictions_gbm(
        val_df,
        y_prob_val,
        y_pred_val,
        optimal_threshold,
        f"validate_{VALIDATE_YEAR}",
        model_name,
    )
    test_preds = _save_predictions_gbm(
        test_df,
        y_prob_test,
        y_pred_test,
        optimal_threshold,
        f"test_{TEST_YEAR}",
        model_name,
    )
    combined = pl.concat([val_preds, test_preds])
    abbrev = {"lightgbm": "lgbm", "xgboost": "xgb"}[model_name]
    pred_path = root / "data" / "processed" / f"predictions_{abbrev}.parquet"
    combined.write_parquet(pred_path)
    print(f"Predictions saved: {combined.height:,} rows to {pred_path}")

    # 8. Persist model
    model_path = root / "data" / "processed" / f"model_{abbrev}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    # 9. Generate PR curve
    pr_path = root / "data" / "exports" / "figures" / f"pr_curve_{abbrev}.png"
    _plot_pr_curve_gbm(
        y_val, y_prob_val, w_val, threshold_data, optimal_threshold, model_name, pr_path
    )

    # 10. Determine best_iteration
    if model_name == "lightgbm":
        best_iter = model.best_iteration_  # trailing underscore for LightGBM
    else:
        best_iter = model.best_iteration  # no trailing underscore for XGBoost

    # 11. Build entry dict
    entry = {
        "metadata": {
            "model_type": (
                "LGBMClassifier" if model_name == "lightgbm" else "XGBClassifier"
            ),
            "train_years": TRAIN_YEARS,
            "validate_year": VALIDATE_YEAR,
            "test_year": TEST_YEAR,
            "n_train": int(X_train.shape[0]),
            "n_validate": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": len(feature_names),
            "feature_names": list(feature_names),
            "best_iteration": int(best_iter),
            "scale_pos_weight": round(float(spw), 4),
            "optuna_n_trials": study.trials[-1].number + 1,
            "optuna_best_trial": study.best_trial.number,
            "optuna_best_params": study.best_trial.params,
            "calibration_note": (
                "scale_pos_weight distorts probability estimates; "
                "Phase 7 handles calibration"
            ),
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
    return entry


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_lgbm_xgb_pipeline() -> dict:
    """Train Optuna-tuned LightGBM and XGBoost, evaluate, export artifacts."""
    root = find_project_root()
    parquet_path = root / "data" / "processed" / "enaho_with_features.parquet"
    results_path = root / "data" / "exports" / "model_results.json"

    # 1. Load data
    print("Loading feature matrix...")
    df = pl.read_parquet(parquet_path)
    print(f"Loaded: {df.height:,} rows, {df.width} columns")

    # 2. Create temporal splits
    print("\nCreating temporal splits...")
    train_df, val_df, test_df = create_temporal_splits(df)

    # 3. Convert to numpy
    X_train, y_train, w_train = _df_to_numpy(train_df)
    X_val, y_val, w_val = _df_to_numpy(val_df)
    X_test, y_test, w_test = _df_to_numpy(test_df)
    print(
        f"\nFeature matrix shapes: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )

    # 4. Compute scale_pos_weight
    spw = _compute_scale_pos_weight(y_train)

    # 5. LightGBM Optuna tuning (100 trials)
    print("\n" + "=" * 60)
    print("LIGHTGBM OPTUNA TUNING (100 trials)")
    print("=" * 60)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lgbm_study = optuna.create_study(
        direction="maximize", study_name="lgbm_prauc"
    )
    lgbm_study.optimize(
        lambda trial: _lgbm_objective(
            trial, X_train, y_train, w_train, X_val, y_val, w_val, spw
        ),
        n_trials=100,
    )
    print(f"\nBest LightGBM trial: #{lgbm_study.best_trial.number}")
    print(f"Best LightGBM val PR-AUC: {lgbm_study.best_trial.value:.4f}")
    print(f"Best params: {lgbm_study.best_trial.params}")

    # 6. Retrain best LightGBM
    print("\nRetraining best LightGBM model...")
    best_lgbm_params = lgbm_study.best_trial.params
    best_lgbm = LGBMClassifier(
        n_estimators=500,
        **best_lgbm_params,
        scale_pos_weight=spw,
        importance_type="gain",
        verbose=-1,
        random_state=42,
    )
    best_lgbm.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="average_precision",
        callbacks=[early_stopping(50, first_metric_only=True), log_evaluation(0)],
    )
    print(f"LightGBM best iteration: {best_lgbm.best_iteration_}")

    # 7. Evaluate LightGBM
    lgbm_entry = _train_and_evaluate_model(
        "lightgbm",
        best_lgbm,
        X_train,
        y_train,
        w_train,
        X_val,
        y_val,
        w_val,
        X_test,
        y_test,
        w_test,
        val_df,
        test_df,
        list(MODEL_FEATURES),
        lgbm_study,
        spw,
        root,
    )

    # 8. XGBoost Optuna tuning (50 trials)
    print("\n" + "=" * 60)
    print("XGBOOST OPTUNA TUNING (50 trials)")
    print("=" * 60)
    xgb_study = optuna.create_study(
        direction="maximize", study_name="xgb_prauc"
    )
    xgb_study.optimize(
        lambda trial: _xgb_objective(
            trial, X_train, y_train, w_train, X_val, y_val, w_val, spw
        ),
        n_trials=50,
    )
    print(f"\nBest XGBoost trial: #{xgb_study.best_trial.number}")
    print(f"Best XGBoost val PR-AUC: {xgb_study.best_trial.value:.4f}")
    print(f"Best params: {xgb_study.best_trial.params}")

    # 9. Retrain best XGBoost
    print("\nRetraining best XGBoost model...")
    best_xgb_params = xgb_study.best_trial.params
    best_xgb = XGBClassifier(
        n_estimators=500,
        **best_xgb_params,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        importance_type="gain",
        verbosity=0,
        random_state=42,
    )
    best_xgb.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    print(f"XGBoost best iteration: {best_xgb.best_iteration}")

    # 10. Evaluate XGBoost
    xgb_entry = _train_and_evaluate_model(
        "xgboost",
        best_xgb,
        X_train,
        y_train,
        w_train,
        X_val,
        y_val,
        w_val,
        X_test,
        y_test,
        w_test,
        val_df,
        test_df,
        list(MODEL_FEATURES),
        xgb_study,
        spw,
        root,
    )

    # 11. Algorithm-independence check
    lgbm_prauc = lgbm_entry["metrics"][f"validate_{VALIDATE_YEAR}"]["weighted"][
        "pr_auc"
    ]
    xgb_prauc = xgb_entry["metrics"][f"validate_{VALIDATE_YEAR}"]["weighted"][
        "pr_auc"
    ]
    ratio = xgb_prauc / lgbm_prauc

    print(f"\n=== ALGORITHM-INDEPENDENCE CHECK ===")
    print(f"  LightGBM val PR-AUC (weighted): {lgbm_prauc:.4f}")
    print(f"  XGBoost  val PR-AUC (weighted): {xgb_prauc:.4f}")
    print(f"  Ratio (XGB/LGBM):               {ratio:.4f}")
    print(f"  Threshold:                       >= 0.95 (within 5%)")

    if ratio >= 0.95:
        print("  Algorithm-independence:          PASS")
    else:
        print(
            f"  Algorithm-independence:          WARNING -- XGBoost is "
            f"{(1 - ratio) * 100:.1f}% below LightGBM"
        )

    # 12. LR baseline comparison
    print(f"\n=== VS LOGISTIC REGRESSION BASELINE ===")
    print(f"  LR      val PR-AUC (weighted): 0.2103")
    print(
        f"  LightGBM val PR-AUC (weighted): {lgbm_prauc:.4f} "
        f"({'BEATS LR' if lgbm_prauc > 0.2103 else 'DOES NOT BEAT LR'})"
    )
    print(
        f"  XGBoost  val PR-AUC (weighted): {xgb_prauc:.4f} "
        f"({'BEATS LR' if xgb_prauc > 0.2103 else 'DOES NOT BEAT LR'})"
    )

    # 13. Merge into model_results.json
    print(f"\nUpdating model_results.json...")
    with open(results_path, "r") as f:
        model_results = json.load(f)

    assert "logistic_regression" in model_results, (
        "LR baseline entry missing from model_results.json!"
    )

    model_results["lightgbm"] = lgbm_entry
    model_results["xgboost"] = xgb_entry

    with open(results_path, "w") as f:
        json.dump(model_results, f, indent=2)

    print(f"model_results.json updated with lightgbm and xgboost entries")
    print(f"  Keys: {list(model_results.keys())}")

    # 14. Final summary
    print("\n" + "=" * 60)
    print("LIGHTGBM + XGBOOST PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  LightGBM val PR-AUC (W):  {lgbm_prauc:.4f} (LR baseline: 0.2103)")
    print(f"  XGBoost  val PR-AUC (W):  {xgb_prauc:.4f}")
    print(
        f"  Algorithm-independence:    ratio={ratio:.4f} "
        f"{'PASS' if ratio >= 0.95 else 'WARNING'}"
    )
    print(
        f"  Optuna trials:            LGBM={len(lgbm_study.trials)}, "
        f"XGB={len(xgb_study.trials)}"
    )
    print(f"  LightGBM best iteration:  {best_lgbm.best_iteration_}")
    print(f"  XGBoost best iteration:   {best_xgb.best_iteration}")
    print(
        f"  Max feature importance:   "
        f"{lgbm_entry['feature_importances'][0]['feature']} = "
        f"{lgbm_entry['feature_importances'][0]['importance']:.4f}"
    )
    print("=" * 60)

    return model_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_lgbm_xgb_pipeline()
