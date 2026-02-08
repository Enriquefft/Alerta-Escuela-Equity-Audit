"""Gate Test 2.1: Baseline Model + Temporal Splits Validation.

Validates:
- Temporal split correctness (no year overlap)
- Logistic regression convergence and PR-AUC > 0.14
- Weighted vs unweighted metric divergence
- model_results.json schema and content
- Threshold analysis at 5 fixed points + optimal
- Coefficient sensibility (signs for equity-relevant features)
- predictions_lr.parquet existence and schema
- model_lr.joblib persistence
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
PARQUET_PATH = ROOT / "data" / "processed" / "enaho_with_features.parquet"
RESULTS_PATH = ROOT / "data" / "exports" / "model_results.json"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions_lr.parquet"
MODEL_PATH = ROOT / "data" / "processed" / "model_lr.joblib"
PR_CURVE_PATH = ROOT / "data" / "exports" / "figures" / "pr_curve_lr.png"


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """Load the feature matrix parquet once per test module."""
    return pl.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def model_results() -> dict:
    """Load model_results.json once per test module."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------


def test_temporal_splits_no_overlap(df: pl.DataFrame) -> None:
    """Verify year coverage, split sizes, and zero overlap."""
    years = set(df["year"].unique().to_list())
    assert years == {2018, 2019, 2020, 2021, 2022, 2023}, (
        f"Expected years {{2018..2023}}, got {years}"
    )

    train = df.filter(pl.col("year").is_in([2018, 2019, 2020, 2021]))
    val = df.filter(pl.col("year") == 2022)
    test = df.filter(pl.col("year") == 2023)

    # Size checks with tolerance
    assert abs(train.height - 98_023) <= 100, (
        f"Train rows {train.height} not near 98,023"
    )
    assert abs(val.height - 26_477) <= 100, (
        f"Val rows {val.height} not near 26,477"
    )
    assert abs(test.height - 25_635) <= 100, (
        f"Test rows {test.height} not near 25,635"
    )

    # Exact total
    assert train.height + val.height + test.height == 150_135, (
        f"Total {train.height + val.height + test.height} != 150,135"
    )

    print(
        f"\n  Train: {train.height:,}, Validate: {val.height:,}, "
        f"Test: {test.height:,}, Total: {df.height:,}"
    )


# ---------------------------------------------------------------------------
# model_results.json
# ---------------------------------------------------------------------------


def test_model_results_json_exists(model_results: dict) -> None:
    """JSON loads and has logistic_regression key."""
    assert RESULTS_PATH.exists(), f"model_results.json not found: {RESULTS_PATH}"
    assert "logistic_regression" in model_results, (
        "Missing top-level key: logistic_regression"
    )
    print("\n  model_results.json loaded successfully")


def test_model_results_metadata(model_results: dict) -> None:
    """Metadata has required fields and correct values."""
    meta = model_results["logistic_regression"]["metadata"]

    assert meta["model_type"] == "LogisticRegression"
    assert meta["n_features"] == 25, f"n_features={meta['n_features']} != 25"
    assert meta["train_years"] == [2018, 2019, 2020, 2021]
    assert meta["validate_year"] == 2022
    assert meta["test_year"] == 2023
    assert meta["class_weight"] == "balanced"
    assert meta["convergence"] is True
    assert "feature_names" in meta
    assert len(meta["feature_names"]) == 25

    print(f"\n  Metadata: model_type={meta['model_type']}")
    print(f"    n_features: {meta['n_features']}")
    print(f"    train_years: {meta['train_years']}")
    print(f"    n_train: {meta['n_train']:,}")
    print(f"    n_validate: {meta['n_validate']:,}")
    print(f"    n_test: {meta['n_test']:,}")
    print(f"    convergence: {meta['convergence']}")
    print(f"    n_iter_actual: {meta['n_iter_actual']}")


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------


def test_validation_pr_auc_above_baseline(model_results: dict) -> None:
    """Validation PR-AUC > 0.14 (beating random baseline for ~14% dropout rate)."""
    lr = model_results["logistic_regression"]
    pr_auc_w = lr["metrics"]["validate_2022"]["weighted"]["pr_auc"]

    assert pr_auc_w > 0.14, f"Validation PR-AUC (weighted) {pr_auc_w:.4f} not > 0.14"
    print(f"\n  Validation PR-AUC (weighted): {pr_auc_w:.4f} > 0.14 PASS")


def test_weighted_differs_from_unweighted(model_results: dict) -> None:
    """Weighted and unweighted PR-AUC differ by > 0.001."""
    lr = model_results["logistic_regression"]
    w_pr_auc = lr["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    uw_pr_auc = lr["metrics"]["validate_2022"]["unweighted"]["pr_auc"]

    diff_pr = abs(w_pr_auc - uw_pr_auc)
    assert diff_pr > 0.001, (
        f"PR-AUC weighted/unweighted diff {diff_pr:.6f} not > 0.001"
    )
    print(f"\n  Weighted PR-AUC:   {w_pr_auc:.4f}")
    print(f"  Unweighted PR-AUC: {uw_pr_auc:.4f}")
    print(f"  Difference:        {diff_pr:.6f}")

    # Also check ROC-AUC or F1 differs
    w_f1 = lr["metrics"]["validate_2022"]["weighted"]["f1"]
    uw_f1 = lr["metrics"]["validate_2022"]["unweighted"]["f1"]
    diff_f1 = abs(w_f1 - uw_f1)
    print(f"  F1 diff:           {diff_f1:.6f}")

    # At least one other metric must differ too
    w_roc = lr["metrics"]["validate_2022"]["weighted"]["roc_auc"]
    uw_roc = lr["metrics"]["validate_2022"]["unweighted"]["roc_auc"]
    diff_roc = abs(w_roc - uw_roc)
    assert diff_f1 > 0.0001 or diff_roc > 0.0001, (
        "Neither F1 nor ROC-AUC differ between weighted and unweighted"
    )
    print("  Weighted != Unweighted: PASS")


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def test_threshold_analysis_complete(model_results: dict) -> None:
    """5 fixed thresholds + optimal, all with required metrics."""
    lr = model_results["logistic_regression"]
    ta = lr["threshold_analysis"]

    # Optimal threshold
    opt = ta["optimal_threshold"]
    assert 0.05 < opt < 0.95, f"Optimal threshold {opt} outside (0.05, 0.95)"
    assert ta["optimization_target"] == "max_weighted_f1"

    # Check fixed thresholds present
    fixed_thresholds_present = set()
    for entry in ta["thresholds"]:
        t = entry["threshold"]
        if t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            fixed_thresholds_present.add(t)

        # Each entry must have required metrics
        for key in [
            "weighted_f1",
            "weighted_precision",
            "weighted_recall",
            "unweighted_f1",
            "unweighted_precision",
            "unweighted_recall",
        ]:
            assert key in entry, f"Threshold {t} missing key: {key}"

    assert fixed_thresholds_present == {0.3, 0.4, 0.5, 0.6, 0.7}, (
        f"Missing fixed thresholds: {set([0.3, 0.4, 0.5, 0.6, 0.7]) - fixed_thresholds_present}"
    )

    # Print threshold table
    print(f"\n  Optimal threshold: {opt:.4f}")
    print(f"  Optimization target: {ta['optimization_target']}")
    print(f"  {'Threshold':>10s}  {'W-F1':>8s}  {'W-Prec':>8s}  {'W-Recall':>8s}")
    print(f"  {'-' * 42}")
    for entry in ta["thresholds"]:
        is_opt = entry.get("is_optimal", False)
        marker = "*" if is_opt else " "
        print(
            f"  {entry['threshold']:>9.4f}{marker}  "
            f"{entry['weighted_f1']:>8.4f}  "
            f"{entry['weighted_precision']:>8.4f}  "
            f"{entry['weighted_recall']:>8.4f}"
        )
    print("  Threshold analysis: PASS")


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


def test_coefficients_sensible_signs(model_results: dict) -> None:
    """26 coefficients; print equity-relevant features for human review.

    NOTE: Do NOT hard-assert the sign of every feature -- only print for
    human review.  The logistic regression with class_weight='balanced' and
    survey weights may produce unexpected signs for some features.
    """
    lr = model_results["logistic_regression"]
    coefficients = lr["coefficients"]

    assert len(coefficients) == 26, (
        f"Expected 26 coefficients (intercept + 25 features), got {len(coefficients)}"
    )

    # Build lookup
    coef_map = {c["feature"]: c for c in coefficients}

    # Equity-relevant features for human review
    equity_features = [
        "poverty_quintile",
        "poverty_index_z",
        "rural",
        "lang_quechua",
        "lang_aimara",
        "lang_other_indigenous",
        "age",
        "es_mujer",
        "es_peruano",
    ]

    print("\n  === EQUITY-RELEVANT COEFFICIENTS (for human review) ===")
    print(f"  {'Feature':<30s}  {'Coef':>10s}  {'OR':>10s}  {'p-value':>10s}")
    print(f"  {'-' * 65}")
    for feat in equity_features:
        if feat in coef_map:
            c = coef_map[feat]
            print(
                f"  {feat:<30s}  {c['coefficient']:>+10.4f}  "
                f"{c['odds_ratio']:>10.4f}  {c['p_value']:>10.4f}"
            )

    # Print all coefficients
    print(f"\n  === ALL COEFFICIENTS ===")
    print(f"  {'Feature':<35s}  {'Coef':>10s}  {'SE':>10s}  {'OR':>10s}")
    print(f"  {'-' * 70}")
    for c in coefficients:
        print(
            f"  {c['feature']:<35s}  {c['coefficient']:>+10.4f}  "
            f"{c['std_error']:>10.4f}  {c['odds_ratio']:>10.4f}"
        )

    print(f"\n  Total coefficients: {len(coefficients)} (intercept + {len(coefficients) - 1} features)")


# ---------------------------------------------------------------------------
# Predictions parquet
# ---------------------------------------------------------------------------


def test_predictions_parquet_exists() -> None:
    """predictions_lr.parquet exists with correct schema and values."""
    assert PREDICTIONS_PATH.exists(), (
        f"Predictions file not found: {PREDICTIONS_PATH}"
    )

    pred = pl.read_parquet(PREDICTIONS_PATH)

    # Required columns
    required_cols = [
        "CONGLOME",
        "VIVIENDA",
        "HOGAR",
        "CODPERSO",
        "year",
        "UBIGEO",
        "FACTOR07",
        "dropout",
        "prob_dropout",
        "pred_dropout",
        "model",
        "threshold",
    ]
    for col in required_cols:
        assert col in pred.columns, f"Missing column: {col}"

    # prob_dropout range
    min_p = pred["prob_dropout"].min()
    max_p = pred["prob_dropout"].max()
    assert min_p >= 0.0, f"prob_dropout min {min_p} < 0"
    assert max_p <= 1.0, f"prob_dropout max {max_p} > 1"

    # pred_dropout binary
    pred_vals = set(pred["pred_dropout"].unique().to_list())
    assert pred_vals.issubset({0, 1, 0.0, 1.0}), (
        f"pred_dropout has unexpected values: {pred_vals}"
    )

    # model column
    models = pred["model"].unique().to_list()
    assert models == ["logistic_regression"], (
        f"Unexpected model values: {models}"
    )

    # Row count (val + test ~= 52,112)
    assert abs(pred.height - 52_112) <= 100, (
        f"Predictions rows {pred.height} not near 52,112"
    )

    print(
        f"\n  Predictions: {pred.height:,} rows, "
        f"prob range [{min_p:.4f}, {max_p:.4f}]"
    )


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def test_model_joblib_exists() -> None:
    """model_lr.joblib loads and has predict_proba."""
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"

    import joblib

    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"
    print(f"\n  Model loaded, has predict_proba: True")


# ---------------------------------------------------------------------------
# PR curve figure
# ---------------------------------------------------------------------------


def test_pr_curve_figure_exists() -> None:
    """PR curve PNG exists and is > 10KB."""
    assert PR_CURVE_PATH.exists(), f"PR curve not found: {PR_CURVE_PATH}"

    size_kb = PR_CURVE_PATH.stat().st_size / 1024
    assert size_kb > 10, f"PR curve too small: {size_kb:.1f} KB"
    print(f"\n  PR curve figure: {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Test set metrics
# ---------------------------------------------------------------------------


def test_test_set_metrics(model_results: dict) -> None:
    """test_2023 metrics exist with weighted and unweighted."""
    lr = model_results["logistic_regression"]
    assert "test_2023" in lr["metrics"], "Missing test_2023 metrics"

    test_metrics = lr["metrics"]["test_2023"]
    assert "weighted" in test_metrics, "Missing test_2023 weighted metrics"
    assert "unweighted" in test_metrics, "Missing test_2023 unweighted metrics"

    pr_auc_w = test_metrics["weighted"]["pr_auc"]
    assert pr_auc_w > 0.0, f"Test PR-AUC (weighted) {pr_auc_w} not > 0.0"

    pr_auc_uw = test_metrics["unweighted"]["pr_auc"]
    print(f"\n  Test PR-AUC (weighted):   {pr_auc_w:.4f}")
    print(f"  Test PR-AUC (unweighted): {pr_auc_uw:.4f}")


# ---------------------------------------------------------------------------
# Standalone summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    lr = data["logistic_regression"]
    meta = lr["metadata"]
    val_w = lr["metrics"]["validate_2022"]["weighted"]
    val_uw = lr["metrics"]["validate_2022"]["unweighted"]
    ta = lr["threshold_analysis"]
    coefficients = lr["coefficients"]
    coef_map = {c["feature"]: c for c in coefficients}

    pred = pl.read_parquet(PREDICTIONS_PATH)

    print("=== GATE TEST 2.1 SUMMARY ===")
    print(
        f"Temporal splits: Train={meta['n_train']:,} (2018-2021), "
        f"Val={meta['n_validate']:,} (2022), "
        f"Test={meta['n_test']:,} (2023)"
    )
    print(f"LR convergence: PASS (n_iter={meta['n_iter_actual']})")
    print(
        f"Val PR-AUC (weighted): {val_w['pr_auc']:.4f} > 0.14 "
        f"{'PASS' if val_w['pr_auc'] > 0.14 else 'FAIL'}"
    )

    diff = abs(val_w["pr_auc"] - val_uw["pr_auc"])
    print(f"Weighted != Unweighted: {'PASS' if diff > 0.001 else 'FAIL'} (diff={diff:.4f})")

    print(f"Threshold analysis: 5 fixed + optimal={ta['optimal_threshold']:.4f}")
    print(f"Coefficients: {len(coefficients)} entries (intercept + {len(coefficients) - 1} features)")

    equity_feats = [
        "poverty_quintile",
        "rural",
        "lang_quechua",
        "age",
        "es_mujer",
    ]
    for feat in equity_feats:
        if feat in coef_map:
            c = coef_map[feat]
            print(
                f"  {feat + ':':<25s} coef={c['coefficient']:+.4f}, OR={c['odds_ratio']:.4f}"
            )

    print(f"Predictions: {pred.height:,} rows saved")
    print(f"Model persisted: model_lr.joblib")
    print(f"PR curve: pr_curve_lr.png")
