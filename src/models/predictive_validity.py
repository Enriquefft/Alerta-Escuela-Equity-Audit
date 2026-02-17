"""Predictive validity analysis: lift, calibration-by-decile, BSS, cross-model comparison.

Computes lift analysis, calibration-by-decile, Brier skill scores, and
cross-model FNR comparison across all 5 model families.

Usage::

    uv run python src/models/predictive_validity.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import MODEL_FEATURES
from models.baseline import create_temporal_splits, TEST_YEAR
from utils import find_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prediction file mapping
# ---------------------------------------------------------------------------

PREDICTION_FILES = {
    "logistic_regression": "predictions_lr.parquet",
    "lightgbm_calibrated": "predictions_lgbm_calibrated.parquet",
    "xgboost": "predictions_xgb.parquet",
    "random_forest": "predictions_rf.parquet",
    "mlp": "predictions_mlp.parquet",
}

THRESHOLD_PATHS = {
    "logistic_regression": ["logistic_regression", "threshold_analysis", "optimal_threshold"],
    "lightgbm_calibrated": ["test_2023_calibrated", "metadata", "optimal_threshold"],
    "xgboost": ["xgboost", "threshold_analysis", "optimal_threshold"],
    "random_forest": ["random_forest", "threshold_analysis", "optimal_threshold"],
    "mlp": ["mlp", "threshold_analysis", "optimal_threshold"],
}


def _get_nested(d: dict, keys: list[str]):
    """Navigate nested dict by key path."""
    for k in keys:
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# Lift analysis
# ---------------------------------------------------------------------------


def compute_lift_by_decile(
    y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Compute lift by decile of predicted probability.

    Returns list of dicts with decile, n, lift, rate, cumulative_lift.
    Decile 10 = highest predicted risk.
    """
    # Sort by predicted probability descending
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    w_sorted = weights[order]

    # Split into deciles
    bin_size = len(y_sorted) // n_bins
    overall_rate = float(np.average(y_true, weights=weights))

    results = []
    cumulative_positives_w = 0.0
    cumulative_total_w = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y_sorted)
        y_bin = y_sorted[start:end]
        w_bin = w_sorted[start:end]

        bin_rate = float(np.average(y_bin, weights=w_bin))
        lift = bin_rate / overall_rate if overall_rate > 0 else 0.0

        cumulative_positives_w += float(np.sum(y_bin * w_bin))
        cumulative_total_w += float(np.sum(w_bin))
        cum_rate = cumulative_positives_w / cumulative_total_w if cumulative_total_w > 0 else 0.0
        cum_lift = cum_rate / overall_rate if overall_rate > 0 else 0.0

        results.append({
            "decile": n_bins - i,  # 10 = highest risk
            "n": int(end - start),
            "rate": round(bin_rate, 6),
            "lift": round(lift, 6),
            "cumulative_lift": round(cum_lift, 6),
        })

    # Reverse so decile 1 = lowest risk first
    results.reverse()
    return results


# ---------------------------------------------------------------------------
# Calibration-by-decile
# ---------------------------------------------------------------------------


def compute_calibration_by_decile(
    y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute calibration by decile of predicted probability.

    Returns dict with deciles list + mean_absolute_calibration_error.
    """
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    w_sorted = weights[order]

    bin_size = len(y_sorted) // n_bins
    deciles = []
    total_abs_error = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y_sorted)
        y_bin = y_sorted[start:end]
        p_bin = p_sorted[start:end]
        w_bin = w_sorted[start:end]

        mean_pred = float(np.average(p_bin, weights=w_bin))
        mean_obs = float(np.average(y_bin, weights=w_bin))
        abs_error = abs(mean_pred - mean_obs)
        total_abs_error += abs_error

        deciles.append({
            "decile": i + 1,
            "n": int(end - start),
            "mean_predicted": round(mean_pred, 6),
            "mean_observed": round(mean_obs, 6),
            "absolute_error": round(abs_error, 6),
        })

    mace = total_abs_error / n_bins
    return {
        "deciles": deciles,
        "mean_absolute_calibration_error": round(mace, 6),
    }


# ---------------------------------------------------------------------------
# Brier skill score
# ---------------------------------------------------------------------------


def compute_brier_skill_score(
    y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray
) -> dict:
    """Compute Brier skill score (BSS = 1 - Brier_model / Brier_baseline)."""
    prevalence = float(np.average(y_true, weights=weights))
    brier_baseline = prevalence * (1 - prevalence)
    brier_model = float(np.average((y_true - y_prob) ** 2, weights=weights))
    bss = 1 - (brier_model / brier_baseline) if brier_baseline > 0 else 0.0

    return {
        "brier_model": round(brier_model, 6),
        "brier_baseline": round(brier_baseline, 6),
        "brier_skill_score": round(bss, 6),
        "prevalence": round(prevalence, 6),
    }


# ---------------------------------------------------------------------------
# Cross-model FNR table
# ---------------------------------------------------------------------------


def compute_cross_model_fnr_table(
    model_preds: dict[str, dict],
    language_groups: np.ndarray,
) -> dict:
    """Compute FNR by language group across all models.

    Parameters
    ----------
    model_preds : dict
        model_name -> {"y_true", "y_pred", "weights"} arrays
    language_groups : np.ndarray
        Language group label per observation (test set)

    Returns
    -------
    dict
        Nested: language_group -> model_name -> FNR
    """
    unique_langs = sorted(set(language_groups))
    table = {}

    for lang in unique_langs:
        mask = language_groups == lang
        table[lang] = {}

        for model_name, data in model_preds.items():
            y_true_g = data["y_true"][mask]
            y_pred_g = data["y_pred"][mask]
            w_g = data["weights"][mask]

            # FNR = weighted FN / weighted positives
            positives_mask = y_true_g == 1
            if positives_mask.sum() == 0:
                table[lang][model_name] = None
                continue

            fn_mask = positives_mask & (y_pred_g == 0)
            w_fn = float(np.sum(w_g[fn_mask]))
            w_pos = float(np.sum(w_g[positives_mask]))
            fnr = w_fn / w_pos if w_pos > 0 else 0.0
            table[lang][model_name] = round(fnr, 6)

    return table


def assess_algorithm_independence(fnr_table: dict) -> dict:
    """Assess whether FNR rank order is consistent across models."""
    # Get models from first language group
    first_lang = next(iter(fnr_table.values()))
    model_names = list(first_lang.keys())

    # For each model, rank languages by FNR (highest = worst)
    rankings = {}
    for model_name in model_names:
        lang_fnrs = []
        for lang, model_fnrs in fnr_table.items():
            fnr = model_fnrs.get(model_name)
            if fnr is not None:
                lang_fnrs.append((lang, fnr))
        sorted_langs = sorted(lang_fnrs, key=lambda x: x[1], reverse=True)
        rankings[model_name] = [x[0] for x in sorted_langs]

    # Check if worst group is consistent
    worst_groups = {model: ranks[0] for model, ranks in rankings.items() if ranks}
    worst_set = set(worst_groups.values())
    consistent = len(worst_set) == 1

    # Compute max FNR range across models per language
    max_range = 0.0
    max_range_lang = None
    for lang, model_fnrs in fnr_table.items():
        fnrs = [v for v in model_fnrs.values() if v is not None]
        if len(fnrs) >= 2:
            rng = max(fnrs) - min(fnrs)
            if rng > max_range:
                max_range = rng
                max_range_lang = lang

    return {
        "fnr_rank_order_consistent": consistent,
        "worst_group_per_model": worst_groups,
        "worst_group_all_models": worst_set.pop() if consistent else list(worst_set),
        "max_fnr_range": round(max_range, 6),
        "max_fnr_range_language": max_range_lang,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_predictive_validity_pipeline() -> dict:
    """Run predictive validity analysis across all 5 models."""
    root = find_project_root()
    results_path = root / "data" / "exports" / "model_results.json"
    feat_path = root / "data" / "processed" / "enaho_with_features.parquet"
    output_path = root / "data" / "exports" / "predictive_validity.json"

    with open(results_path) as f:
        model_results = json.load(f)

    # Load features for language groups
    print("Loading features for language group mapping...")
    feat = pl.read_parquet(feat_path)

    # Load all predictions (test set only)
    print("\nLoading predictions for all 5 models...")
    model_data = {}
    join_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"]

    for model_name, pred_file in PREDICTION_FILES.items():
        pred_path = root / "data" / "processed" / pred_file
        if not pred_path.exists():
            print(f"  SKIP: {pred_file} not found")
            continue

        pred = pl.read_parquet(pred_path)
        test_pred = pred.filter(pl.col("split") == f"test_{TEST_YEAR}")

        # Get threshold
        threshold_keys = THRESHOLD_PATHS[model_name]
        threshold = _get_nested(model_results, threshold_keys)

        y_true = test_pred["dropout"].cast(pl.Int8).to_numpy()
        y_prob = test_pred["prob_dropout"].to_numpy()
        y_pred = (y_prob >= threshold).astype(int)
        weights = test_pred["FACTOR07"].to_numpy()

        model_data[model_name] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "weights": weights,
            "test_pred": test_pred,
        }
        print(f"  {model_name}: {test_pred.height} rows, threshold={threshold:.4f}")

    assert len(model_data) >= 5, f"Expected 5 models, found {len(model_data)}"

    # Build language groups from features (match to test set)
    # Use first model's test predictions to get the correct order
    first_model = next(iter(model_data.values()))
    test_pred_for_join = first_model["test_pred"]

    lang_cols = ["lang_castellano", "lang_quechua", "lang_aimara", "lang_other_indigenous", "lang_foreign"]
    merged = test_pred_for_join.join(
        feat.select(join_keys + lang_cols),
        on=join_keys,
        how="left",
    )

    # Build language group array
    language_groups = []
    for i in range(merged.height):
        found = False
        for col, label in [
            ("lang_castellano", "castellano"),
            ("lang_quechua", "quechua"),
            ("lang_aimara", "aimara"),
            ("lang_other_indigenous", "other_indigenous"),
            ("lang_foreign", "foreign"),
        ]:
            if merged[col][i] == 1:
                language_groups.append(label)
                found = True
                break
        if not found:
            language_groups.append("unknown")
    language_groups = np.array(language_groups)

    # -----------------------------------------------------------------------
    # Compute analyses
    # -----------------------------------------------------------------------

    # 1. Lift analysis (all models, test set)
    print("\n=== LIFT ANALYSIS ===")
    lift_results = {}
    for model_name, data in model_data.items():
        lift = compute_lift_by_decile(data["y_true"], data["y_prob"], data["weights"])
        lift_results[model_name] = lift
        top_decile = [d for d in lift if d["decile"] == 10][0]
        print(f"  {model_name}: top decile lift = {top_decile['lift']:.2f}")

    # 2. Calibration-by-decile (all models, test set)
    print("\n=== CALIBRATION BY DECILE ===")
    calibration_results = {}
    for model_name, data in model_data.items():
        cal = compute_calibration_by_decile(data["y_true"], data["y_prob"], data["weights"])
        calibration_results[model_name] = cal
        print(f"  {model_name}: MACE = {cal['mean_absolute_calibration_error']:.4f}")

    # 3. Brier skill score (all models, test set)
    print("\n=== BRIER SKILL SCORES ===")
    bss_results = {}
    for model_name, data in model_data.items():
        bss = compute_brier_skill_score(data["y_true"], data["y_prob"], data["weights"])
        bss_results[model_name] = bss
        print(f"  {model_name}: BSS = {bss['brier_skill_score']:.4f}")

    # 4. Cross-model FNR table
    print("\n=== CROSS-MODEL FNR BY LANGUAGE ===")
    fnr_table = compute_cross_model_fnr_table(model_data, language_groups)
    for lang, model_fnrs in sorted(fnr_table.items()):
        fnr_vals = " | ".join(f"{m}: {v:.4f}" if v else f"{m}: N/A" for m, v in model_fnrs.items())
        print(f"  {lang}: {fnr_vals}")

    # 5. Algorithm independence
    independence = assess_algorithm_independence(fnr_table)
    print(f"\n  Rank order consistent: {independence['fnr_rank_order_consistent']}")
    print(f"  Worst group per model: {independence['worst_group_per_model']}")
    print(f"  Max FNR range: {independence['max_fnr_range']:.4f} ({independence['max_fnr_range_language']})")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    validity_json = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "test_year": TEST_YEAR,
            "n_models": len(model_data),
            "models": list(model_data.keys()),
        },
        "lift_analysis": lift_results,
        "calibration_by_decile": calibration_results,
        "brier_skill_scores": bss_results,
        "cross_model_fnr": fnr_table,
        "algorithm_independence": independence,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(validity_json, f, indent=2)
    print(f"\nSaved: {output_path}")

    return validity_json


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_predictive_validity_pipeline()
