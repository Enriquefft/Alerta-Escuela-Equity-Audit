"""Gate Test 2.3: Calibration + ONNX Export + Final Test Validation.

Validates:
- Calibrated Brier score < uncalibrated Brier score (Platt scaling works)
- ONNX file exists, is < 50 MB, predictions match Python within 1e-4
- Val-test PR-AUC gap < 0.07 (no extreme overfitting)
- model_results.json has test_2023_calibrated entry with Platt params
- Alerta Escuela comparison data is present
- Calibrated model persists and loads
- Calibration plot exists
- Calibrated predictions parquet has correct schema
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
ONNX_PATH = ROOT / "data" / "exports" / "onnx" / "lightgbm_dropout.onnx"
CAL_MODEL_PATH = ROOT / "data" / "processed" / "model_lgbm_calibrated.joblib"
CAL_PRED_PATH = ROOT / "data" / "processed" / "predictions_lgbm_calibrated.parquet"
CAL_PLOT_PATH = ROOT / "data" / "exports" / "figures" / "calibration_plot.png"
LGBM_MODEL_PATH = ROOT / "data" / "processed" / "model_lgbm.joblib"
PARQUET_PATH = ROOT / "data" / "processed" / "enaho_with_features.parquet"


@pytest.fixture(scope="module")
def model_results() -> dict:
    """Load model_results.json once per test module."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. test_2023_calibrated entry exists
# ---------------------------------------------------------------------------


def test_calibrated_entry_exists(model_results: dict) -> None:
    """model_results.json has test_2023_calibrated top-level key."""
    assert "test_2023_calibrated" in model_results, (
        f"test_2023_calibrated missing. Keys: {list(model_results.keys())}"
    )
    print(f"\n  model_results.json keys: {list(model_results.keys())}")


# ---------------------------------------------------------------------------
# 2. Brier improvement
# ---------------------------------------------------------------------------


def test_brier_improvement(model_results: dict) -> None:
    """Calibrated Brier < uncalibrated Brier on validation."""
    meta = model_results["test_2023_calibrated"]["metadata"]
    brier_uncal = meta["brier_uncalibrated"]
    brier_cal = meta["brier_calibrated"]

    assert brier_cal < brier_uncal, (
        f"Calibrated Brier {brier_cal:.4f} not less than uncalibrated {brier_uncal:.4f}"
    )

    improvement = (1 - brier_cal / brier_uncal) * 100
    print(f"\n  Brier uncalibrated: {brier_uncal:.4f}")
    print(f"  Brier calibrated:   {brier_cal:.4f}")
    print(f"  Improvement:        {improvement:.1f}%")


# ---------------------------------------------------------------------------
# 3. Val-test PR-AUC gap
# ---------------------------------------------------------------------------


def test_val_test_prauc_gap(model_results: dict) -> None:
    """Calibrated val PR-AUC and calibrated test PR-AUC differ by < 0.07."""
    cal = model_results["test_2023_calibrated"]["metrics"]
    val_prauc = cal["validate_2022"]["weighted"]["pr_auc"]
    test_prauc = cal["test_2023"]["weighted"]["pr_auc"]
    gap = abs(val_prauc - test_prauc)

    assert gap < 0.07, (
        f"Val-test PR-AUC gap {gap:.4f} exceeds 0.07 threshold"
    )
    print(f"\n  Val PR-AUC (cal, W):  {val_prauc:.4f}")
    print(f"  Test PR-AUC (cal, W): {test_prauc:.4f}")
    print(f"  Gap:                  {gap:.4f} < 0.07 PASS")


# ---------------------------------------------------------------------------
# 4. ONNX exists and size
# ---------------------------------------------------------------------------


def test_onnx_exists_and_size() -> None:
    """ONNX file exists at expected path and is < 50 MB."""
    assert ONNX_PATH.exists(), f"ONNX file not found: {ONNX_PATH}"

    size_bytes = ONNX_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    assert size_bytes < 50 * 1024 * 1024, (
        f"ONNX file too large: {size_mb:.2f} MB"
    )
    print(f"\n  ONNX size: {size_mb:.2f} MB (< 50 MB PASS)")


# ---------------------------------------------------------------------------
# 5. ONNX predictions match Python
# ---------------------------------------------------------------------------


def test_onnx_predictions_match() -> None:
    """ONNX predictions match Python LightGBM within 1e-4 on 100 samples."""
    import joblib
    import onnxruntime as ort

    lgbm = joblib.load(LGBM_MODEL_PATH)
    df = pl.read_parquet(PARQUET_PATH)

    from models.baseline import create_temporal_splits, _df_to_numpy
    _, val_df, _ = create_temporal_splits(df)
    X_val, _, _ = _df_to_numpy(val_df)

    # Sample 100 random rows
    rng = np.random.default_rng(42)
    indices = rng.choice(X_val.shape[0], 100, replace=False)
    X_sample = X_val[indices].astype(np.float32)

    # Python predictions
    py_proba = lgbm.predict_proba(X_sample)[:, 1]

    # ONNX predictions
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_result = sess.run(None, {input_name: X_sample})
    onnx_proba = onnx_result[1][:, 1]

    max_diff = float(np.max(np.abs(py_proba - onnx_proba)))
    assert max_diff < 1e-4, (
        f"ONNX predictions differ: max_diff={max_diff:.2e}"
    )
    print(f"\n  ONNX max absolute diff: {max_diff:.2e} (< 1e-4 PASS)")


# ---------------------------------------------------------------------------
# 6. Platt parameters
# ---------------------------------------------------------------------------


def test_platt_parameters(model_results: dict) -> None:
    """Platt A and B are present in metadata and are finite floats. A < 0."""
    meta = model_results["test_2023_calibrated"]["metadata"]

    assert "platt_a" in meta, "platt_a missing from metadata"
    assert "platt_b" in meta, "platt_b missing from metadata"

    platt_a = meta["platt_a"]
    platt_b = meta["platt_b"]

    assert isinstance(platt_a, (int, float)), f"platt_a not numeric: {type(platt_a)}"
    assert isinstance(platt_b, (int, float)), f"platt_b not numeric: {type(platt_b)}"
    assert np.isfinite(platt_a), f"platt_a not finite: {platt_a}"
    assert np.isfinite(platt_b), f"platt_b not finite: {platt_b}"
    assert platt_a < 0, f"platt_a should be negative (sigmoid slope): {platt_a}"

    print(f"\n  Platt A: {platt_a:.6f} (negative PASS)")
    print(f"  Platt B: {platt_b:.6f}")


# ---------------------------------------------------------------------------
# 7. Calibration plot exists
# ---------------------------------------------------------------------------


def test_calibration_plot_exists() -> None:
    """data/exports/figures/calibration_plot.png exists and is > 1 KB."""
    assert CAL_PLOT_PATH.exists(), f"Calibration plot not found: {CAL_PLOT_PATH}"

    size_kb = CAL_PLOT_PATH.stat().st_size / 1024
    assert size_kb > 1, f"Calibration plot too small: {size_kb:.1f} KB"
    print(f"\n  Calibration plot: {size_kb:.1f} KB (> 1 KB PASS)")


# ---------------------------------------------------------------------------
# 8. Calibrated model persisted
# ---------------------------------------------------------------------------


def test_calibrated_model_persisted() -> None:
    """model_lgbm_calibrated.joblib exists and loads successfully."""
    import joblib

    assert CAL_MODEL_PATH.exists(), f"Calibrated model not found: {CAL_MODEL_PATH}"

    cal_model = joblib.load(CAL_MODEL_PATH)
    assert hasattr(cal_model, "predict_proba"), (
        "Calibrated model missing predict_proba"
    )
    print(f"\n  Calibrated model loaded: {CAL_MODEL_PATH.name}")
    print(f"  Has predict_proba: True")


# ---------------------------------------------------------------------------
# 9. Alerta Escuela comparison
# ---------------------------------------------------------------------------


def test_alerta_escuela_comparison(model_results: dict) -> None:
    """alerta_escuela_comparison key exists with required fields."""
    cal = model_results["test_2023_calibrated"]
    assert "alerta_escuela_comparison" in cal, (
        "alerta_escuela_comparison missing from test_2023_calibrated"
    )

    comp = cal["alerta_escuela_comparison"]
    for key in ["roc_auc_range", "fnr_range", "fairness_analysis"]:
        assert key in comp, f"alerta_escuela_comparison missing key: {key}"

    print(f"\n  Alerta Escuela comparison keys: {list(comp.keys())}")
    print(f"  ROC-AUC range: {comp['roc_auc_range']}")
    print(f"  FNR range: {comp['fnr_range']}")
    print(f"  Fairness: {comp['fairness_analysis']}")


# ---------------------------------------------------------------------------
# 10. Calibrated predictions parquet
# ---------------------------------------------------------------------------


def test_calibrated_predictions_parquet() -> None:
    """predictions_lgbm_calibrated.parquet exists with correct schema."""
    assert CAL_PRED_PATH.exists(), (
        f"Calibrated predictions not found: {CAL_PRED_PATH}"
    )

    pred = pl.read_parquet(CAL_PRED_PATH)
    assert pred.height > 0, "Calibrated predictions is empty"

    required_cols = [
        "prob_dropout",
        "prob_dropout_uncalibrated",
        "pred_dropout",
        "model",
        "split",
    ]
    for col in required_cols:
        assert col in pred.columns, f"Missing column: {col}"

    # Model column should be lightgbm_calibrated
    models = pred["model"].unique().to_list()
    assert models == ["lightgbm_calibrated"], (
        f"Unexpected model values: {models}"
    )

    # prob_dropout range
    min_p = pred["prob_dropout"].min()
    max_p = pred["prob_dropout"].max()
    assert min_p >= 0.0, f"prob_dropout min {min_p} < 0"
    assert max_p <= 1.0, f"prob_dropout max {max_p} > 1"

    print(f"\n  Calibrated predictions: {pred.height:,} rows")
    print(f"  Columns: {pred.columns}")
    print(f"  prob_dropout range: [{min_p:.4f}, {max_p:.4f}]")


# ---------------------------------------------------------------------------
# 11. Test set metrics present
# ---------------------------------------------------------------------------


def test_test_set_metrics_present(model_results: dict) -> None:
    """test_2023 metrics exist under test_2023_calibrated with full schema."""
    cal_metrics = model_results["test_2023_calibrated"]["metrics"]

    assert "test_2023" in cal_metrics, (
        f"test_2023 missing from calibrated metrics. Keys: {list(cal_metrics.keys())}"
    )

    required_metric_keys = [
        "pr_auc", "roc_auc", "brier", "f1", "precision", "recall", "log_loss",
    ]

    for weight_type in ["weighted", "unweighted"]:
        assert weight_type in cal_metrics["test_2023"], (
            f"test_2023 missing {weight_type} sub-dict"
        )
        metrics = cal_metrics["test_2023"][weight_type]
        for key in required_metric_keys:
            assert key in metrics, (
                f"test_2023.{weight_type} missing metric: {key}"
            )

    w = cal_metrics["test_2023"]["weighted"]
    uw = cal_metrics["test_2023"]["unweighted"]
    print(f"\n  Test 2023 weighted PR-AUC:   {w['pr_auc']:.4f}")
    print(f"  Test 2023 unweighted PR-AUC: {uw['pr_auc']:.4f}")
    print(f"  Test 2023 weighted Brier:    {w['brier']:.4f}")
    print(f"  All 7 metric keys present in both weighted and unweighted: PASS")


# ---------------------------------------------------------------------------
# Standalone summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    cal = data["test_2023_calibrated"]
    meta = cal["metadata"]

    print("=== GATE TEST 2.3 SUMMARY ===")
    print(f"Brier: {meta['brier_uncalibrated']:.4f} -> {meta['brier_calibrated']:.4f} ({meta['brier_improvement_pct']:.1f}% reduction)")
    print(f"Platt A: {meta['platt_a']:.6f}, B: {meta['platt_b']:.6f}")
    print(f"ONNX: {meta['onnx_path']} ({meta['onnx_size_bytes']} bytes)")
    print(f"ONNX max diff: {meta['onnx_max_abs_diff']:.2e}")

    val_prauc = cal["metrics"]["validate_2022"]["weighted"]["pr_auc"]
    test_prauc = cal["metrics"]["test_2023"]["weighted"]["pr_auc"]
    print(f"Val PR-AUC (cal, W): {val_prauc:.4f}")
    print(f"Test PR-AUC (cal, W): {test_prauc:.4f}")
    print(f"Gap: {abs(val_prauc - test_prauc):.4f}")

    comp = cal["alerta_escuela_comparison"]
    print(f"\nAlerta Escuela: ROC-AUC {comp['roc_auc_range']}, FNR {comp['fnr_range']}")
    print(f"Equity Audit: ROC-AUC {cal['metrics']['test_2023']['weighted']['roc_auc']:.4f}")
