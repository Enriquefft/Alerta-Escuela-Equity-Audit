"""Calibration (Platt scaling), ONNX export, and final test evaluation.

Calibrates the LightGBM model to correct scale_pos_weight probability distortion,
exports the raw model to ONNX for browser inference, and evaluates on the 2023
test set exactly once. Produces calibrated predictions, calibration plot,
updated model_results.json, and Alerta Escuela comparison table.

Usage::

    uv run python src/models/calibration.py
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import onnxruntime as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import MODEL_FEATURES
from models.baseline import (
    create_temporal_splits,
    _df_to_numpy,
    compute_metrics,
    _threshold_analysis,
    VALIDATE_YEAR,
    TEST_YEAR,
    ID_COLUMNS,
)
from utils import find_project_root

logger = logging.getLogger(__name__)

# Suppress the benign FrozenEstimator sample_weight warning.
# The warning is expected: sample_weight flows correctly to the calibration LR,
# not to the frozen model's non-existent fit method.
warnings.filterwarnings(
    "ignore",
    message=".*FrozenEstimator.*sample_weight.*",
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_calibration_pipeline() -> dict:
    """Run the full calibration, ONNX export, and final test pipeline.

    Steps:
    1. Load model and data
    2. Compute uncalibrated validation baseline
    3. Calibrate with Platt scaling on validation set
    4. Extract Platt parameters for browser-side calibration
    5. Export raw LightGBM to ONNX
    6. Validate ONNX predictions
    7. Final test evaluation (EXACTLY ONCE)
    8. Calibration reliability diagram
    9. Alerta Escuela comparison table
    10. Save calibrated predictions parquet
    11. Persist calibrated model
    12. Update model_results.json
    13. Final summary

    Returns
    -------
    dict
        Updated model_results.json content.
    """
    root = find_project_root()
    parquet_path = root / "data" / "processed" / "enaho_with_features.parquet"
    results_path = root / "data" / "exports" / "model_results.json"
    onnx_path = root / "data" / "exports" / "onnx" / "lightgbm_dropout.onnx"
    cal_plot_path = root / "data" / "exports" / "figures" / "calibration_plot.png"
    cal_model_path = root / "data" / "processed" / "model_lgbm_calibrated.joblib"
    cal_pred_path = root / "data" / "processed" / "predictions_lgbm_calibrated.parquet"
    lgbm_model_path = root / "data" / "processed" / "model_lgbm.joblib"

    # -----------------------------------------------------------------------
    # Step 1 -- Load model and data
    # -----------------------------------------------------------------------
    print("Step 1: Loading model and data...")
    lgbm = joblib.load(lgbm_model_path)
    print(f"  LightGBM model loaded: {lgbm_model_path.name}")
    print(f"  best_iteration: {lgbm.best_iteration_}")

    df = pl.read_parquet(parquet_path)
    print(f"  Data loaded: {df.height:,} rows, {df.width} columns")

    print("\n  Creating temporal splits...")
    _, val_df, test_df = create_temporal_splits(df)

    X_val, y_val, w_val = _df_to_numpy(val_df)
    X_test, y_test, w_test = _df_to_numpy(test_df)
    print(f"  Val: {X_val.shape}, Test: {X_test.shape}")

    # -----------------------------------------------------------------------
    # Step 2 -- Uncalibrated validation baseline
    # -----------------------------------------------------------------------
    print("\nStep 2: Uncalibrated validation baseline...")
    uncal_proba_val = lgbm.predict_proba(X_val)[:, 1]
    brier_uncal = brier_score_loss(y_val, uncal_proba_val, sample_weight=w_val)
    print(f"  Uncalibrated Brier score (val): {brier_uncal:.4f}")

    # -----------------------------------------------------------------------
    # Step 3 -- Calibrate with Platt scaling on validation set
    # -----------------------------------------------------------------------
    print("\nStep 3: Platt scaling calibration on validation set...")
    frozen = FrozenEstimator(lgbm)
    cal_model = CalibratedClassifierCV(frozen, method="sigmoid")
    cal_model.fit(X_val, y_val, sample_weight=w_val)

    cal_proba_val = cal_model.predict_proba(X_val)[:, 1]
    brier_cal = brier_score_loss(y_val, cal_proba_val, sample_weight=w_val)

    assert brier_cal < brier_uncal, (
        f"Calibration did NOT improve Brier score: "
        f"{brier_uncal:.4f} -> {brier_cal:.4f}"
    )

    brier_improvement_pct = (1 - brier_cal / brier_uncal) * 100
    print(
        f"  Brier improvement: {brier_uncal:.4f} -> {brier_cal:.4f} "
        f"({brier_improvement_pct:.1f}% reduction)"
    )

    # -----------------------------------------------------------------------
    # Step 4 -- Extract Platt parameters for browser-side calibration
    # -----------------------------------------------------------------------
    print("\nStep 4: Extracting Platt parameters...")
    calibrator = cal_model.calibrated_classifiers_[0].calibrators[0]
    platt_a = float(calibrator.a_)
    platt_b = float(calibrator.b_)
    print(f"  Platt A (slope):     {platt_a:.6f}")
    print(f"  Platt B (intercept): {platt_b:.6f}")
    print(f"  JS formula: calibrated = 1 / (1 + Math.exp({platt_a:.6f} * raw_prob + {platt_b:.6f}))")

    # -----------------------------------------------------------------------
    # Step 5 -- ONNX export of RAW LightGBM (NOT calibrated wrapper)
    # -----------------------------------------------------------------------
    print("\nStep 5: ONNX export...")
    initial_types = [("input", FloatTensorType([None, len(MODEL_FEATURES)]))]
    onnx_model = convert_lightgbm(lgbm, initial_types=initial_types, zipmap=False)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(onnx_path))

    onnx_size_bytes = onnx_path.stat().st_size
    assert onnx_size_bytes < 50 * 1024 * 1024, (
        f"ONNX model too large: {onnx_size_bytes / (1024 * 1024):.2f} MB"
    )
    print(f"  ONNX saved: {onnx_path}")
    print(f"  ONNX size: {onnx_size_bytes / (1024 * 1024):.2f} MB")

    # -----------------------------------------------------------------------
    # Step 6 -- ONNX prediction validation
    # -----------------------------------------------------------------------
    print("\nStep 6: ONNX prediction validation...")
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(42)
    indices = rng.choice(X_val.shape[0], 100, replace=False)
    X_sample = X_val[indices]
    X_float32 = X_sample.astype(np.float32)

    py_proba = lgbm.predict_proba(X_sample)[:, 1]
    onnx_result = sess.run(None, {input_name: X_float32})
    onnx_proba = onnx_result[1][:, 1]

    max_diff = float(np.max(np.abs(py_proba - onnx_proba)))
    assert max_diff < 1e-4, (
        f"ONNX predictions differ from Python: max_diff={max_diff:.2e}"
    )
    print(f"  Max absolute difference: {max_diff:.2e} (threshold: 1e-4)")
    print(f"  ONNX validation: PASS")

    # -----------------------------------------------------------------------
    # Step 7 -- FINAL TEST EVALUATION (EXACTLY ONCE)
    # -----------------------------------------------------------------------
    print("\nStep 7: Final test evaluation (EXACTLY ONCE)...")
    uncal_proba_test = lgbm.predict_proba(X_test)[:, 1]
    cal_proba_test = cal_model.predict_proba(X_test)[:, 1]

    # Threshold analysis on VALIDATION (not test)
    cal_threshold_data = _threshold_analysis(y_val, cal_proba_val, w_val)
    cal_optimal_threshold = cal_threshold_data["optimal_threshold"]
    print(f"  Calibrated optimal threshold (from val): {cal_optimal_threshold:.4f}")

    # Also get uncalibrated threshold for fair comparison
    uncal_threshold_data = _threshold_analysis(y_val, uncal_proba_val, w_val)
    uncal_optimal_threshold = uncal_threshold_data["optimal_threshold"]
    print(f"  Uncalibrated optimal threshold (from val): {uncal_optimal_threshold:.4f}")

    # Apply thresholds to test predictions
    uncal_pred_test = (uncal_proba_test >= uncal_optimal_threshold).astype(int)
    cal_pred_test = (cal_proba_test >= cal_optimal_threshold).astype(int)

    # Also for validation (for JSON)
    uncal_pred_val = (uncal_proba_val >= uncal_optimal_threshold).astype(int)
    cal_pred_val = (cal_proba_val >= cal_optimal_threshold).astype(int)

    # Compute full metric suite for test_2023_final (uncalibrated LightGBM on test)
    test_final_weighted = compute_metrics(
        y_test, uncal_proba_test, uncal_pred_test, weights=w_test
    )
    test_final_unweighted = compute_metrics(
        y_test, uncal_proba_test, uncal_pred_test, weights=None
    )

    # Compute full metric suite for test_2023_calibrated (calibrated LightGBM on test)
    test_cal_weighted = compute_metrics(
        y_test, cal_proba_test, cal_pred_test, weights=w_test
    )
    test_cal_unweighted = compute_metrics(
        y_test, cal_proba_test, cal_pred_test, weights=None
    )

    # Calibrated validation metrics (for JSON)
    val_cal_weighted = compute_metrics(
        y_val, cal_proba_val, cal_pred_val, weights=w_val
    )
    val_cal_unweighted = compute_metrics(
        y_val, cal_proba_val, cal_pred_val, weights=None
    )

    # Check val-test gap (calibrated)
    val_prauc = val_cal_weighted["pr_auc"]
    test_prauc = test_cal_weighted["pr_auc"]
    val_test_gap = abs(val_prauc - test_prauc)
    assert val_test_gap < 0.07, (
        f"Val-test PR-AUC gap too large: {val_test_gap:.4f} (threshold: 0.07)"
    )

    # Print metrics side by side
    print(f"\n  === TEST {TEST_YEAR}: Uncalibrated vs Calibrated ===")
    print(f"  {'Metric':<15s} {'Uncalibrated':>14s} {'Calibrated':>14s} {'Diff':>10s}")
    print(f"  {'-' * 55}")
    for key in test_final_weighted:
        uncal_v = test_final_weighted[key]
        cal_v = test_cal_weighted[key]
        diff = cal_v - uncal_v
        print(f"  {key:<15s} {uncal_v:>14.4f} {cal_v:>14.4f} {diff:>+10.4f}")

    print(f"\n  Val PR-AUC (calibrated, weighted): {val_prauc:.4f}")
    print(f"  Test PR-AUC (calibrated, weighted): {test_prauc:.4f}")
    print(f"  Val-test PR-AUC gap: {val_test_gap:.4f} (threshold: 0.07)")

    # -----------------------------------------------------------------------
    # Step 8 -- Calibration reliability diagram
    # -----------------------------------------------------------------------
    print("\nStep 8: Calibration reliability diagram...")
    fig, ax = plt.subplots(figsize=(8, 6))

    CalibrationDisplay.from_predictions(
        y_val, uncal_proba_val, n_bins=10, strategy="uniform",
        name="LightGBM (uncalibrated)", ax=ax,
    )
    CalibrationDisplay.from_predictions(
        y_val, cal_proba_val, n_bins=10, strategy="uniform",
        name="LightGBM (calibrated)", ax=ax,
    )

    ax.set_title(
        "Calibration Curve: LightGBM Before vs After Platt Scaling "
        f"(Validation {VALIDATE_YEAR})"
    )
    plt.tight_layout()
    cal_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cal_plot_path, dpi=150)
    plt.close(fig)
    print(f"  Calibration plot saved: {cal_plot_path}")

    # -----------------------------------------------------------------------
    # Step 9 -- Alerta Escuela comparison table
    # -----------------------------------------------------------------------
    print("\nStep 9: Alerta Escuela comparison table...")

    test_roc_auc = test_cal_weighted["roc_auc"]
    test_recall = test_cal_weighted["recall"]
    test_fnr = (1 - test_recall) * 100
    test_prauc_w = test_cal_weighted["pr_auc"]

    print("\n" + "=" * 75)
    print("=== ALERTA ESCUELA vs EQUITY AUDIT COMPARISON ===")
    print("=" * 75)
    print(f"{'Metric':<25s} {'Alerta Escuela':<25s} {'Equity Audit (calibrated)':<25s}")
    print("-" * 75)
    print(f"{'Algorithm':<25s} {'LightGBM':<25s} {'LightGBM (Optuna)':<25s}")
    print(f"{'Data Source':<25s} {'SIAGIE admin':<25s} {'ENAHO survey':<25s}")
    print(f"{'ROC-AUC':<25s} {'0.84-0.89':<25s} {test_roc_auc:<25.4f}")
    print(f"{'FNR':<25s} {'57-64%':<25s} {test_fnr:.1f}%")
    print(f"{'PR-AUC':<25s} {'Not published':<25s} {test_prauc_w:<25.4f}")
    print(f"{'Features':<25s} {'31':<25s} {'25':<25s}")
    print(
        f"{'Calibration':<25s} {'Not reported':<25s} "
        f"{'Platt scaling (' + f'{brier_improvement_pct:.1f}' + '% Brier red.)':<25s}"
    )
    print(
        f"{'Fairness Analysis':<25s} {'None published':<25s} "
        f"{'6 dims + 3 intersections':<25s}"
    )
    print("=" * 75)
    print(
        "\nNote: ROC-AUC is not directly comparable due to different data sources "
        "and base rates (~14% survey vs ~2% admin). FNR comparison is more meaningful."
    )

    # -----------------------------------------------------------------------
    # Step 10 -- Save calibrated predictions parquet
    # -----------------------------------------------------------------------
    print("\nStep 10: Saving calibrated predictions parquet...")

    # Build validation predictions
    val_pred_df = val_df.select(ID_COLUMNS).with_columns([
        pl.Series("prob_dropout", cal_proba_val),
        pl.Series("prob_dropout_uncalibrated", uncal_proba_val),
        pl.Series("pred_dropout", cal_pred_val),
        pl.lit("lightgbm_calibrated").alias("model"),
        pl.lit(cal_optimal_threshold).alias("threshold"),
        pl.lit(f"validate_{VALIDATE_YEAR}").alias("split"),
    ])

    # Build test predictions
    test_pred_df = test_df.select(ID_COLUMNS).with_columns([
        pl.Series("prob_dropout", cal_proba_test),
        pl.Series("prob_dropout_uncalibrated", uncal_proba_test),
        pl.Series("pred_dropout", cal_pred_test),
        pl.lit("lightgbm_calibrated").alias("model"),
        pl.lit(cal_optimal_threshold).alias("threshold"),
        pl.lit(f"test_{TEST_YEAR}").alias("split"),
    ])

    combined_preds = pl.concat([val_pred_df, test_pred_df])
    combined_preds.write_parquet(cal_pred_path)
    print(f"  Calibrated predictions: {combined_preds.height:,} rows -> {cal_pred_path}")

    # -----------------------------------------------------------------------
    # Step 11 -- Persist calibrated model
    # -----------------------------------------------------------------------
    print("\nStep 11: Persisting calibrated model...")
    joblib.dump(cal_model, cal_model_path)
    print(f"  Calibrated model saved: {cal_model_path}")

    # -----------------------------------------------------------------------
    # Step 12 -- Update model_results.json
    # -----------------------------------------------------------------------
    print("\nStep 12: Updating model_results.json...")
    with open(results_path, "r") as f:
        model_results = json.load(f)

    def _round_dict(d: dict, decimals: int = 6) -> dict:
        return {
            k: round(v, decimals) if isinstance(v, float) else v
            for k, v in d.items()
        }

    # Add test_2023_final under lightgbm key (uncalibrated test metrics)
    model_results["lightgbm"]["test_2023_final"] = {
        "weighted": _round_dict(test_final_weighted),
        "unweighted": _round_dict(test_final_unweighted),
    }

    # Add test_2023_calibrated top-level key
    model_results["test_2023_calibrated"] = {
        "metadata": {
            "calibration_method": "platt_sigmoid",
            "platt_a": round(platt_a, 6),
            "platt_b": round(platt_b, 6),
            "brier_uncalibrated": round(float(brier_uncal), 6),
            "brier_calibrated": round(float(brier_cal), 6),
            "brier_improvement_pct": round(float(brier_improvement_pct), 2),
            "onnx_path": str(onnx_path.relative_to(root)),
            "onnx_size_bytes": onnx_size_bytes,
            "onnx_max_abs_diff": round(max_diff, 10),
            "optimal_threshold": round(float(cal_optimal_threshold), 6),
            "base_model": "lightgbm",
            "frozen_estimator": True,
            "year_note": "Test year is 2023 (ENAHO 2024 unavailable)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": {
            f"validate_{VALIDATE_YEAR}": {
                "weighted": _round_dict(val_cal_weighted),
                "unweighted": _round_dict(val_cal_unweighted),
            },
            f"test_{TEST_YEAR}": {
                "weighted": _round_dict(test_cal_weighted),
                "unweighted": _round_dict(test_cal_unweighted),
            },
        },
        "threshold_analysis": cal_threshold_data,
        "alerta_escuela_comparison": {
            "model": "Alerta Escuela (MINEDU)",
            "algorithm": "LightGBM",
            "data_source": "SIAGIE administrative",
            "roc_auc_range": "0.84-0.89",
            "fnr_range": "57-64%",
            "features": 31,
            "fairness_analysis": "None published",
            "notes": "Uses 5 administrative data sources; trained on SIAGIE records",
            "comparison_caveats": (
                "ROC-AUC is not directly comparable due to different data sources "
                "and base rates (~14% ENAHO survey vs ~2% SIAGIE admin). "
                "FNR comparison is more meaningful."
            ),
        },
    }

    with open(results_path, "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"  model_results.json updated: {results_path}")
    print(f"  Keys: {list(model_results.keys())}")

    # -----------------------------------------------------------------------
    # Step 13 -- Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CALIBRATION + ONNX EXPORT + FINAL TEST COMPLETE")
    print("=" * 60)
    print(f"  Brier (uncal -> cal):     {brier_uncal:.4f} -> {brier_cal:.4f} ({brier_improvement_pct:.1f}% reduction)")
    print(f"  Platt A:                  {platt_a:.6f}")
    print(f"  Platt B:                  {platt_b:.6f}")
    print(f"  ONNX size:                {onnx_size_bytes / (1024 * 1024):.2f} MB")
    print(f"  ONNX max diff:            {max_diff:.2e}")
    print(f"  Val PR-AUC (cal, W):      {val_prauc:.4f}")
    print(f"  Test PR-AUC (cal, W):     {test_prauc:.4f}")
    print(f"  Val-test PR-AUC gap:      {val_test_gap:.4f}")
    print(f"  Optimal threshold (cal):  {cal_optimal_threshold:.4f}")
    print(f"  Test FNR:                 {test_fnr:.1f}%")
    print(f"  Calibrated predictions:   {combined_preds.height:,} rows")
    print(f"  ONNX:                     {onnx_path}")
    print(f"  Calibration plot:         {cal_plot_path}")
    print("=" * 60)

    return model_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_calibration_pipeline()
