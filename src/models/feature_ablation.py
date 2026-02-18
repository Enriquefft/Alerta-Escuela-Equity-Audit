"""Feature ablation: spatial-only vs individual-only LightGBM variants.

Trains two LightGBM variants with feature subsets to test whether FNR
disparities persist when spatial or individual features are removed.
Reuses the same Optuna-tuned hyperparameters (no re-tuning).

Usage::

    PYTHONPATH=src uv run python src/models/feature_ablation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import MODEL_FEATURES
from models.baseline import (
    create_temporal_splits,
    _df_to_numpy,
    TRAIN_YEARS,
    VALIDATE_YEAR,
    TEST_YEAR,
    ID_COLUMNS,
)
from fairness.bootstrap import _fast_weighted_fnr
from utils import find_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature subsets
# ---------------------------------------------------------------------------

SPATIAL_FEATURES = [
    "district_dropout_rate_admin_z",
    "nightlight_intensity_z",
    "poverty_index_z",
    "census_indigenous_lang_pct_z",
    "census_literacy_rate_z",
    "census_electricity_pct_z",
    "census_water_access_pct_z",
]

INDIVIDUAL_HOUSEHOLD_FEATURES = [
    "age", "es_mujer", "es_peruano", "is_secundaria_age",
    "lang_castellano", "lang_quechua", "lang_aimara",
    "lang_other_indigenous", "lang_foreign",
    "rural", "is_sierra", "is_selva",
    "poverty_quintile", "log_income",
    "parent_education_years", "has_disability", "is_working",
    "juntos_participant",
]

# Language dummy columns for building language group labels
LANG_COLS = [
    ("lang_castellano", "castellano"),
    ("lang_quechua", "quechua"),
    ("lang_aimara", "aimara"),
    ("lang_other_indigenous", "other_indigenous"),
    ("lang_foreign", "foreign"),
]


def _build_language_labels(df: pl.DataFrame) -> np.ndarray:
    """Assign harmonized language labels from dummies."""
    labels = []
    for i in range(df.height):
        found = False
        for col, label in LANG_COLS:
            if df[col][i] == 1:
                labels.append(label)
                found = True
                break
        if not found:
            labels.append("unknown")
    return np.array(labels)


def _compute_fnr_by_language(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    lang_labels: np.ndarray,
) -> dict[str, float]:
    """Compute weighted FNR for each language group."""
    result = {}
    for lang in ["castellano", "quechua", "aimara", "other_indigenous", "foreign"]:
        mask = lang_labels == lang
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            result[lang] = None
            continue
        fnr = float(_fast_weighted_fnr(y_true[mask], y_pred[mask], weights[mask]))
        result[lang] = round(fnr, 6)
    return result


def run_ablation() -> dict:
    """Train spatial-only and individual-only LightGBM, compare FNR by language."""
    root = find_project_root()

    # Load data
    print("Loading data...")
    df = pl.read_parquet(root / "data" / "processed" / "enaho_with_features.parquet")
    train_df, val_df, test_df = create_temporal_splits(df)

    # Load existing hyperparameters and metadata
    with open(root / "data" / "exports" / "model_results.json") as f:
        model_results = json.load(f)
    best_params = model_results["lightgbm"]["metadata"]["optuna_best_params"]
    spw = model_results["lightgbm"]["metadata"]["scale_pos_weight"]
    threshold = model_results["test_2023_calibrated"]["metadata"]["optimal_threshold"]

    print(f"Reusing hyperparameters: {best_params}")
    print(f"scale_pos_weight={spw}, full_model_threshold={threshold}")

    # Prepare numpy arrays for each feature set

    # Individual-only
    X_train_ind = train_df.select(INDIVIDUAL_HOUSEHOLD_FEATURES).to_numpy()
    X_val_ind = val_df.select(INDIVIDUAL_HOUSEHOLD_FEATURES).to_numpy()
    X_test_ind = test_df.select(INDIVIDUAL_HOUSEHOLD_FEATURES).to_numpy()

    # Spatial-only
    X_train_sp = train_df.select(SPATIAL_FEATURES).to_numpy()
    X_val_sp = val_df.select(SPATIAL_FEATURES).to_numpy()
    X_test_sp = test_df.select(SPATIAL_FEATURES).to_numpy()

    # Common arrays
    y_train = train_df["dropout"].cast(pl.Int8).to_numpy()
    y_val = val_df["dropout"].cast(pl.Int8).to_numpy()
    y_test = test_df["dropout"].cast(pl.Int8).to_numpy()
    w_train = train_df["FACTOR07"].to_numpy()
    w_val = val_df["FACTOR07"].to_numpy()
    w_test = test_df["FACTOR07"].to_numpy()

    # Build language labels for test set
    lang_labels = _build_language_labels(test_df)

    # Train variants
    variants = {
        "individual_only": (X_train_ind, X_val_ind, X_test_ind, INDIVIDUAL_HOUSEHOLD_FEATURES),
        "spatial_only": (X_train_sp, X_val_sp, X_test_sp, SPATIAL_FEATURES),
    }

    # Adapt num_leaves for smaller feature sets
    results = {"metadata": {}, "fnr_by_language": {}, "val_pr_auc": {}, "interpretation": ""}

    # Full model reference: load existing predictions for FNR
    preds_cal = pl.read_parquet(root / "data" / "processed" / "predictions_lgbm_calibrated.parquet")
    test_preds = preds_cal.filter(pl.col("split") == "test_2023")
    y_pred_full = test_preds["pred_dropout"].to_numpy()
    y_prob_full = test_preds["prob_dropout"].to_numpy()
    full_val_pr_auc = model_results["lightgbm"]["metrics"]["validate_2022"]["weighted"]["pr_auc"]

    fnr_full = _compute_fnr_by_language(y_test, y_pred_full, w_test, lang_labels)
    results["fnr_by_language"] = {
        lang: {"full": fnr_full.get(lang)}
        for lang in ["castellano", "quechua", "aimara", "other_indigenous", "foreign"]
    }
    results["val_pr_auc"]["full"] = full_val_pr_auc

    for variant_name, (X_tr, X_va, X_te, feat_list) in variants.items():
        print(f"\nTraining {variant_name} ({len(feat_list)} features)...")

        # Adjust num_leaves for smaller feature space
        adjusted_params = dict(best_params)
        if len(feat_list) < 10:
            adjusted_params["num_leaves"] = min(best_params["num_leaves"], 20)
            adjusted_params["max_depth"] = min(best_params["max_depth"], 6)

        model = LGBMClassifier(
            n_estimators=500,
            **adjusted_params,
            scale_pos_weight=spw,
            importance_type="gain",
            verbose=-1,
            random_state=42,
        )
        model.fit(
            X_tr, y_train,
            sample_weight=w_train,
            eval_set=[(X_va, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="average_precision",
            callbacks=[early_stopping(50, first_metric_only=True), log_evaluation(0)],
        )

        # Evaluate
        y_prob_val = model.predict_proba(X_va)[:, 1]
        y_prob_test = model.predict_proba(X_te)[:, 1]
        val_pr_auc = average_precision_score(y_val, y_prob_val, sample_weight=w_val)

        # Find optimal threshold for THIS variant on validation set (max weighted F1)
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.005):
            yp = (y_prob_val >= t).astype(int)
            f = f1_score(y_val, yp, sample_weight=w_val)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        print(f"  Variant threshold (max weighted F1): {best_t:.3f} (F1={best_f1:.4f})")
        y_pred_test = (y_prob_test >= best_t).astype(int)

        # FNR by language
        fnr_variant = _compute_fnr_by_language(y_test, y_pred_test, w_test, lang_labels)

        key = variant_name.replace("_only", "")  # "individual" or "spatial"
        for lang in ["castellano", "quechua", "aimara", "other_indigenous", "foreign"]:
            results["fnr_by_language"][lang][f"{key}_only"] = fnr_variant.get(lang)

        results["val_pr_auc"][variant_name] = round(val_pr_auc, 6)
        results.setdefault("thresholds", {})[variant_name] = round(best_t, 6)

        # Save model
        model_path = root / "data" / "processed" / f"model_lgbm_{variant_name.replace('_only', '')}.joblib"
        joblib.dump(model, model_path)
        print(f"  Val PR-AUC: {val_pr_auc:.4f}, saved to {model_path.name}")
        print(f"  Best iteration: {model.best_iteration_}")

    # Determine interpretation: do FNR rank orders persist?
    rank_orders = []
    for variant_key in ["full", "individual_only", "spatial_only"]:
        fnrs = {}
        for lang in ["castellano", "quechua", "aimara", "other_indigenous"]:
            val = results["fnr_by_language"][lang].get(variant_key)
            if val is not None:
                fnrs[lang] = val
        if fnrs:
            ranked = sorted(fnrs.keys(), key=lambda l: fnrs[l], reverse=True)
            rank_orders.append(ranked)

    # Check highest-FNR group consistency and rank correlation
    if len(rank_orders) >= 2:
        highest_groups = [r[0] for r in rank_orders]
        highest_consistent = all(g == highest_groups[0] for g in highest_groups)

        # Spearman-like: check if top-2 sets match
        top2_sets = [set(r[:2]) for r in rank_orders]
        top2_consistent = all(t == top2_sets[0] for t in top2_sets)

        if top2_consistent:
            results["interpretation"] = "disparities_persist"
        elif highest_consistent:
            results["interpretation"] = "mixed"
        else:
            results["interpretation"] = "disparities_disappear"

        results["rank_analysis"] = {
            "highest_fnr_group_by_variant": {
                k: r[0] for k, r in zip(["full", "individual_only", "spatial_only"], rank_orders)
            },
            "highest_fnr_consistent": highest_consistent,
            "top2_consistent": top2_consistent,
        }
    else:
        results["interpretation"] = "insufficient_data"

    results["metadata"] = {
        "hyperparameters": "reused from lightgbm best_params",
        "full_model_n_features": len(MODEL_FEATURES),
        "individual_n_features": len(INDIVIDUAL_HOUSEHOLD_FEATURES),
        "spatial_n_features": len(SPATIAL_FEATURES),
        "full_model_threshold": threshold,
        "note": "Each variant uses its own optimal threshold (max weighted F1 on validation set) to ensure meaningful FNR comparisons",
    }

    # Save
    output_path = root / "data" / "exports" / "feature_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Summary
    print("\n=== FNR BY LANGUAGE GROUP ===")
    for lang in ["castellano", "quechua", "aimara", "other_indigenous", "foreign"]:
        vals = results["fnr_by_language"][lang]
        parts = [f"{k}={v:.3f}" if v is not None else f"{k}=N/A" for k, v in vals.items()]
        print(f"  {lang}: {', '.join(parts)}")
    print(f"\nInterpretation: {results['interpretation']}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_ablation()
