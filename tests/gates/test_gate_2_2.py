"""Gate Test 2.2: LightGBM + XGBoost Validation.

Validates:
- LightGBM beats LR baseline (val PR-AUC > 0.2103)
- XGBoost within 5% of LightGBM (algorithm-independence)
- No single feature > 50% importance
- model_results.json has all three model entries with correct schema
- Feature importances normalized and sorted
- Predictions parquets exist with correct schema
- Model joblib files persist and have predict_proba
- PR curve PNGs exist
- Threshold analysis complete for both models
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.features import MODEL_FEATURES
from utils import find_project_root

ROOT = find_project_root()
RESULTS_PATH = ROOT / "data" / "exports" / "model_results.json"
LGBM_PRED_PATH = ROOT / "data" / "processed" / "predictions_lgbm.parquet"
XGB_PRED_PATH = ROOT / "data" / "processed" / "predictions_xgb.parquet"
LGBM_MODEL_PATH = ROOT / "data" / "processed" / "model_lgbm.joblib"
XGB_MODEL_PATH = ROOT / "data" / "processed" / "model_xgb.joblib"
LGBM_PR_PATH = ROOT / "data" / "exports" / "figures" / "pr_curve_lgbm.png"
XGB_PR_PATH = ROOT / "data" / "exports" / "figures" / "pr_curve_xgb.png"


@pytest.fixture(scope="module")
def model_results() -> dict:
    """Load model_results.json once per test module."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# model_results.json structure
# ---------------------------------------------------------------------------


def test_model_results_has_all_models(model_results: dict) -> None:
    """model_results.json has logistic_regression, lightgbm, and xgboost."""
    assert "logistic_regression" in model_results, "LR entry missing (was it overwritten?)"
    assert "lightgbm" in model_results, "LightGBM entry missing"
    assert "xgboost" in model_results, "XGBoost entry missing"
    print(f"\n  model_results.json keys: {list(model_results.keys())}")


# ---------------------------------------------------------------------------
# LightGBM vs LR baseline
# ---------------------------------------------------------------------------


def test_lgbm_beats_lr_baseline(model_results: dict) -> None:
    """LightGBM validation PR-AUC > 0.2103 (LR baseline)."""
    lgbm_prauc = model_results["lightgbm"]["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    assert lgbm_prauc > 0.2103, (
        f"LightGBM val PR-AUC {lgbm_prauc:.4f} does not beat LR baseline 0.2103"
    )
    print(
        f"\n  LightGBM val PR-AUC (weighted): {lgbm_prauc:.4f} > 0.2103 "
        f"(LR baseline) PASS"
    )


# ---------------------------------------------------------------------------
# Algorithm-independence
# ---------------------------------------------------------------------------


def test_algorithm_independence(model_results: dict) -> None:
    """XGBoost val PR-AUC within 5% of LightGBM."""
    lgbm_prauc = model_results["lightgbm"]["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    xgb_prauc = model_results["xgboost"]["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    ratio = xgb_prauc / lgbm_prauc

    assert ratio >= 0.95, (
        f"XGBoost val PR-AUC {xgb_prauc:.4f} is {(1 - ratio) * 100:.1f}% below "
        f"LightGBM {lgbm_prauc:.4f} (threshold: 5%)"
    )
    print(f"\n  LightGBM val PR-AUC: {lgbm_prauc:.4f}")
    print(f"  XGBoost  val PR-AUC: {xgb_prauc:.4f}")
    print(f"  Ratio (XGB/LGBM):    {ratio:.4f} >= 0.95 PASS")


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def test_lgbm_no_feature_dominance(model_results: dict) -> None:
    """No single feature accounts for more than 50% of LightGBM importance."""
    importances = model_results["lightgbm"]["feature_importances"]
    max_feat = importances[0]["feature"]
    max_imp = importances[0]["importance"]

    assert max_imp < 0.50, (
        f"Feature {max_feat} has {max_imp:.4f} normalized importance (>50%)"
    )

    print(f"\n  Top-5 LightGBM feature importances:")
    for i, f in enumerate(importances[:5], 1):
        print(f"    {i}. {f['feature']:<40s} {f['importance']:.4f}")
    print(f"\n  Max importance: {max_feat} = {max_imp:.4f} < 0.50 PASS")


def test_lgbm_importances_normalized(model_results: dict) -> None:
    """Feature importances sum to ~1.0 and are sorted descending."""
    importances = model_results["lightgbm"]["feature_importances"]

    # Sum check
    total = sum(f["importance"] for f in importances)
    assert abs(total - 1.0) < 0.01, (
        f"Importances sum to {total:.4f}, not ~1.0"
    )

    # Count check
    assert len(importances) == 25, (
        f"Expected 25 features, got {len(importances)}"
    )

    # Sorted descending check
    values = [f["importance"] for f in importances]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], (
            f"Importances not sorted: [{i}]={values[i]:.6f} < [{i + 1}]={values[i + 1]:.6f}"
        )

    print(f"\n  Importances sum: {total:.4f} (~1.0 PASS)")
    print(f"  Feature count: {len(importances)} (25 PASS)")
    print(f"  Sorted descending: PASS")


# ---------------------------------------------------------------------------
# LightGBM metadata
# ---------------------------------------------------------------------------


def test_lgbm_metadata(model_results: dict) -> None:
    """LightGBM metadata has required fields with correct values."""
    meta = model_results["lightgbm"]["metadata"]

    assert meta["model_type"] == "LGBMClassifier", (
        f"model_type={meta['model_type']}"
    )
    assert meta["n_features"] == 25, f"n_features={meta['n_features']}"
    assert meta["train_years"] == [2018, 2019, 2020, 2021]
    assert meta["validate_year"] == 2022
    assert meta["test_year"] == 2023
    assert meta["best_iteration"] > 0, (
        f"best_iteration={meta['best_iteration']}"
    )
    assert meta["scale_pos_weight"] > 4.0, (
        f"scale_pos_weight={meta['scale_pos_weight']}"
    )
    assert meta["optuna_n_trials"] >= 100, (
        f"optuna_n_trials={meta['optuna_n_trials']}"
    )
    assert meta["optuna_best_trial"] >= 0
    assert isinstance(meta["optuna_best_params"], dict)
    assert "learning_rate" in meta["optuna_best_params"]

    print(f"\n  LightGBM metadata:")
    print(f"    model_type: {meta['model_type']}")
    print(f"    n_features: {meta['n_features']}")
    print(f"    best_iteration: {meta['best_iteration']}")
    print(f"    scale_pos_weight: {meta['scale_pos_weight']:.4f}")
    print(f"    optuna_n_trials: {meta['optuna_n_trials']}")
    print(f"    optuna_best_trial: {meta['optuna_best_trial']}")


# ---------------------------------------------------------------------------
# XGBoost metadata
# ---------------------------------------------------------------------------


def test_xgb_metadata(model_results: dict) -> None:
    """XGBoost metadata has required fields."""
    meta = model_results["xgboost"]["metadata"]

    assert meta["model_type"] == "XGBClassifier", (
        f"model_type={meta['model_type']}"
    )
    assert meta["n_features"] == 25
    assert meta["optuna_n_trials"] >= 50, (
        f"optuna_n_trials={meta['optuna_n_trials']}"
    )
    assert isinstance(meta["optuna_best_params"], dict)
    assert "learning_rate" in meta["optuna_best_params"]

    print(f"\n  XGBoost metadata:")
    print(f"    model_type: {meta['model_type']}")
    print(f"    optuna_n_trials: {meta['optuna_n_trials']}")
    print(f"    optuna_best_trial: {meta['optuna_best_trial']}")


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def test_threshold_analysis_both_models(model_results: dict) -> None:
    """Both models have complete threshold analysis."""
    for model_name in ["lightgbm", "xgboost"]:
        ta = model_results[model_name]["threshold_analysis"]

        # Optimal threshold range
        opt = ta["optimal_threshold"]
        assert 0.05 < opt < 0.95, (
            f"{model_name} optimal_threshold {opt} outside (0.05, 0.95)"
        )
        assert ta["optimization_target"] == "max_weighted_f1"

        # Check fixed thresholds present
        fixed_present = set()
        for entry in ta["thresholds"]:
            t = entry["threshold"]
            if t in [0.3, 0.4, 0.5, 0.6, 0.7]:
                fixed_present.add(t)

            # Required keys
            for key in ["weighted_f1", "weighted_precision", "weighted_recall"]:
                assert key in entry, (
                    f"{model_name} threshold {t} missing key: {key}"
                )

        assert fixed_present == {0.3, 0.4, 0.5, 0.6, 0.7}, (
            f"{model_name} missing fixed thresholds: "
            f"{{0.3, 0.4, 0.5, 0.6, 0.7}} - {fixed_present}"
        )

    lgbm_opt = model_results["lightgbm"]["threshold_analysis"]["optimal_threshold"]
    xgb_opt = model_results["xgboost"]["threshold_analysis"]["optimal_threshold"]
    print(f"\n  LightGBM optimal threshold: {lgbm_opt:.4f}")
    print(f"  XGBoost  optimal threshold: {xgb_opt:.4f}")
    print(f"  Both have 5 fixed + 1 optimal thresholds: PASS")


# ---------------------------------------------------------------------------
# Predictions parquets
# ---------------------------------------------------------------------------


def test_predictions_parquets_exist() -> None:
    """Both predictions parquets exist with correct schema and values."""
    required_cols = [
        "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year",
        "UBIGEO", "FACTOR07", "dropout", "prob_dropout", "pred_dropout",
        "model", "threshold", "split",
    ]

    for pred_path, expected_model in [
        (LGBM_PRED_PATH, "lightgbm"),
        (XGB_PRED_PATH, "xgboost"),
    ]:
        assert pred_path.exists(), f"Predictions file not found: {pred_path}"
        pred = pl.read_parquet(pred_path)

        # Column check
        for col in required_cols:
            assert col in pred.columns, (
                f"{expected_model}: missing column {col}"
            )

        # prob_dropout range
        min_p = pred["prob_dropout"].min()
        max_p = pred["prob_dropout"].max()
        assert min_p >= 0.0, f"{expected_model} prob_dropout min {min_p} < 0"
        assert max_p <= 1.0, f"{expected_model} prob_dropout max {max_p} > 1"

        # pred_dropout binary
        pred_vals = set(pred["pred_dropout"].unique().to_list())
        assert pred_vals.issubset({0, 1, 0.0, 1.0}), (
            f"{expected_model} pred_dropout unexpected values: {pred_vals}"
        )

        # Model column
        models = pred["model"].unique().to_list()
        assert models == [expected_model], (
            f"{expected_model} model column: {models}"
        )

        # Row count
        assert abs(pred.height - 52_112) <= 200, (
            f"{expected_model} predictions {pred.height} not near 52,112"
        )

    lgbm_pred = pl.read_parquet(LGBM_PRED_PATH)
    xgb_pred = pl.read_parquet(XGB_PRED_PATH)
    print(f"\n  LGBM predictions: {lgbm_pred.height:,} rows")
    print(f"  XGB predictions:  {xgb_pred.height:,} rows")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def test_model_joblibs_exist() -> None:
    """Both model joblib files load and have predict_proba."""
    import joblib

    for model_path in [LGBM_MODEL_PATH, XGB_MODEL_PATH]:
        assert model_path.exists(), f"Model file not found: {model_path}"
        model = joblib.load(model_path)
        assert hasattr(model, "predict_proba"), (
            f"{model_path.name} missing predict_proba"
        )

    print(f"\n  Both models loaded, have predict_proba: True")


# ---------------------------------------------------------------------------
# PR curve figures
# ---------------------------------------------------------------------------


def test_pr_curve_figures_exist() -> None:
    """Both PR curve PNGs exist and are > 10KB."""
    for pr_path in [LGBM_PR_PATH, XGB_PR_PATH]:
        assert pr_path.exists(), f"PR curve not found: {pr_path}"
        size_kb = pr_path.stat().st_size / 1024
        assert size_kb > 10, f"{pr_path.name} too small: {size_kb:.1f} KB"

    lgbm_kb = LGBM_PR_PATH.stat().st_size / 1024
    xgb_kb = XGB_PR_PATH.stat().st_size / 1024
    print(f"\n  PR curve LGBM: {lgbm_kb:.1f} KB")
    print(f"  PR curve XGB:  {xgb_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Standalone summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    lgbm = data["lightgbm"]
    xgb = data["xgboost"]
    lgbm_prauc = lgbm["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    xgb_prauc = xgb["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    ratio = xgb_prauc / lgbm_prauc

    print("=== GATE TEST 2.2 SUMMARY ===")
    print(
        f"LightGBM val PR-AUC (weighted): {lgbm_prauc:.4f} > 0.2103 "
        f"{'PASS' if lgbm_prauc > 0.2103 else 'FAIL'}"
    )
    print(f"XGBoost  val PR-AUC (weighted): {xgb_prauc:.4f}")
    print(
        f"Algorithm-independence: ratio={ratio:.4f} "
        f"{'PASS' if ratio >= 0.95 else 'FAIL'}"
    )

    top5 = lgbm["feature_importances"][:5]
    print(f"\nTop-5 LightGBM features:")
    for i, f in enumerate(top5, 1):
        print(f"  {i}. {f['feature']:<40s} {f['importance']:.4f}")

    max_imp = lgbm["feature_importances"][0]["importance"]
    print(f"\nMax importance: {max_imp:.4f} < 0.50 {'PASS' if max_imp < 0.50 else 'FAIL'}")
    print(f"Optuna trials: LGBM={lgbm['metadata']['optuna_n_trials']}, XGB={xgb['metadata']['optuna_n_trials']}")
    print(f"Models persisted: model_lgbm.joblib, model_xgb.joblib")
    print(f"Predictions: predictions_lgbm.parquet, predictions_xgb.parquet")
    print(f"PR curves: pr_curve_lgbm.png, pr_curve_xgb.png")
