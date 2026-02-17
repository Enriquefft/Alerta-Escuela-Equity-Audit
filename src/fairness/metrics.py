"""Subgroup fairness metrics for the Alerta Escuela Equity Audit.

Computes comprehensive fairness metrics across 6 protected dimensions and
3 intersections using fairlearn MetricFrame with FACTOR07 survey weights.
Exports results to M4-schema-compliant fairness_metrics.json.

Dimensions:
  1. language (harmonized 5 groups + unknown)
  2. language_disaggregated (p300a_original codes)
  3. sex (male/female)
  4. geography (urban/rural)
  5. region (costa/sierra/selva)
  6. poverty (Q1-Q5)
  7. nationality (peruvian/non_peruvian)

Intersections:
  1. language x rural
  2. sex x poverty
  3. language x region

Usage::

    uv run python src/fairness/metrics.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from fairlearn.metrics import MetricFrame
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import find_project_root
from fairness.bootstrap import bootstrap_subgroup_cis
from fairness.hypothesis_tests import test_all_disparities

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language code mappings
# ---------------------------------------------------------------------------

# p300a_original -> readable label
P300A_LABEL_MAP: dict[int, str] = {
    1: "quechua",
    2: "aimara",
    3: "other_indigenous",
    4: "castellano",
    5: "portuguese",
    6: "english",
    7: "other_foreign",
    10: "ashaninka",
    11: "awajun",
    12: "shipibo",
    13: "shawi",
    14: "matsigenka",
    15: "achuar",
}

# Harmonized language from dummies -> label
HARMONIZED_LANGUAGE_MAP: dict[str, str] = {
    "lang_castellano": "castellano",
    "lang_quechua": "quechua",
    "lang_aimara": "aimara",
    "lang_other_indigenous": "other_indigenous",
    "lang_foreign": "foreign",
}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def _safe_recall(y_true, y_pred, sample_weight=None):
    """Recall (TPR) with NaN for undefined groups."""
    return recall_score(
        y_true, y_pred, sample_weight=sample_weight, zero_division=np.nan
    )


def _safe_precision(y_true, y_pred, sample_weight=None):
    """Precision with NaN for undefined groups."""
    return precision_score(
        y_true, y_pred, sample_weight=sample_weight, zero_division=np.nan
    )


def _safe_fnr(y_true, y_pred, sample_weight=None):
    """False Negative Rate = 1 - TPR."""
    r = recall_score(
        y_true, y_pred, sample_weight=sample_weight, zero_division=np.nan
    )
    return 1.0 - r if not np.isnan(r) else np.nan


def weighted_fpr(y_true, y_pred, sample_weight=None):
    """False Positive Rate with optional sample weights.

    sklearn does not provide a direct FPR score function.
    FPR = FP / (FP + TN) = weighted average of (y_pred==1) among true negatives.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_weight = np.asarray(sample_weight, dtype=float)
    negatives = y_true == 0
    if negatives.sum() == 0:
        return np.nan
    fp = negatives & (y_pred == 1)
    return float(np.average(fp[negatives], weights=sample_weight[negatives]))


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

BINARY_METRICS = {
    "tpr": _safe_recall,
    "fpr": weighted_fpr,
    "fnr": _safe_fnr,
    "precision": _safe_precision,
}

PROBA_METRICS = {
    "pr_auc": average_precision_score,
}


# ---------------------------------------------------------------------------
# Bootstrap / permutation cache
# ---------------------------------------------------------------------------

def _data_hash(*arrays: np.ndarray) -> str:
    """Compute a fast hash over numpy arrays to detect data changes."""
    h = hashlib.sha256()
    for a in arrays:
        # Object arrays (string labels) need special handling
        if a.dtype == object:
            h.update(str(a[:100].tolist()).encode())
        else:
            h.update(a.tobytes()[:4096])
        h.update(str(len(a)).encode())
    return h.hexdigest()[:16]


def _load_cache(cache_path: Path, expected_hash: str) -> dict | None:
    """Load cached bootstrap/permutation results if hash matches."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("data_hash") == expected_hash:
            logger.info("Cache hit: %s", cache_path.name)
            return cached
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(cache_path: Path, data_hash: str, boot_cis: dict, perm_results: dict) -> None:
    """Save bootstrap/permutation results to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_hash": data_hash,
        "bootstrap_cis": boot_cis,
        "permutation_results": perm_results,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    logger.info("Cache saved: %s", cache_path.name)


def _analyze_dimension(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    y_prob_uncal: np.ndarray,
    dimension_name: str,
    min_sample: int = 100,
    min_high_risk: int = 30,
    n_bootstrap: int = 1000,
    n_permutations: int = 10000,
    reference_group: str | None = None,
) -> dict:
    """Compute fairness metrics for a single dimension.

    Parameters
    ----------
    y_true : array
        Binary ground truth (0/1).
    y_pred : array
        Binary predictions at threshold.
    y_prob : array
        Calibrated probabilities (for PR-AUC).
    weights : array
        FACTOR07 survey weights.
    groups : array
        Group labels for the dimension.
    y_prob_uncal : array
        Uncalibrated probabilities (for high-risk calibration analysis).
    dimension_name : str
        Name of the dimension (for logging).
    min_sample : int
        Minimum unweighted sample size before flagging.
    min_high_risk : int
        Minimum high-risk observations for calibration analysis.

    Returns
    -------
    dict
        Dimension results with groups, gaps, and metadata.
    """
    binary_params = {k: {"sample_weight": weights} for k in BINARY_METRICS}
    proba_params = {k: {"sample_weight": weights} for k in PROBA_METRICS}

    mf_binary = MetricFrame(
        metrics=BINARY_METRICS,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
        sample_params=binary_params,
    )

    mf_proba = MetricFrame(
        metrics=PROBA_METRICS,
        y_true=y_true,
        y_pred=y_prob,
        sensitive_features=groups,
        sample_params=proba_params,
    )

    combined = pd.concat([mf_binary.by_group, mf_proba.by_group], axis=1)

    # ---- Bootstrap CIs + Permutation tests (cached) ----
    ci_metric_fns = {
        "tpr": (_safe_recall, "binary"),
        "fpr": (weighted_fpr, "binary"),
        "fnr": (_safe_fnr, "binary"),
        "precision": (_safe_precision, "binary"),
        "pr_auc": (average_precision_score, "prob"),
    }

    root = find_project_root()
    cache_dir = root / "data" / "processed" / "cache"
    dim_safe = dimension_name.replace(" ", "_").replace("/", "_")
    cache_path = cache_dir / f"stat_cache_{dim_safe}.json"
    dhash = _data_hash(y_true, y_pred, y_prob, weights, groups)

    cached = _load_cache(cache_path, dhash)
    if cached is not None:
        boot_cis = cached["bootstrap_cis"]
        perm_results = cached["permutation_results"]
        print(f"    [cache hit] {dimension_name}")
    else:
        print(f"    [computing] bootstrap + permutation for {dimension_name}...")
        boot_cis = bootstrap_subgroup_cis(
            y_true, y_pred, y_prob, weights, groups,
            metric_fns=ci_metric_fns,
            n_replicates=n_bootstrap,
            seed=42,
        )
        perm_results = {}
        if reference_group is not None:
            perm_results = test_all_disparities(
                y_true, y_pred, y_prob, weights, groups,
                reference_group=reference_group,
                metric_fns=ci_metric_fns,
                n_permutations=n_permutations,
                seed=42,
            )
        _save_cache(cache_path, dhash, boot_cis, perm_results)

    # Build per-group output
    result_groups = {}
    for group_name in combined.index:
        group_mask = groups == group_name
        n_unweighted = int(group_mask.sum())
        n_weighted = float(weights[group_mask].sum())

        group_data = {
            "n_unweighted": n_unweighted,
            "n_weighted": round(n_weighted, 2),
            "tpr": round(float(combined.loc[group_name, "tpr"]), 6),
            "fpr": round(float(combined.loc[group_name, "fpr"]), 6),
            "fnr": round(float(combined.loc[group_name, "fnr"]), 6),
            "precision": round(float(combined.loc[group_name, "precision"]), 6),
            "pr_auc": round(float(combined.loc[group_name, "pr_auc"]), 6),
        }

        # Add CI fields
        g_str = str(group_name)
        g_boot = boot_cis.get(g_str, {})
        for metric in ["tpr", "fpr", "fnr", "precision", "pr_auc"]:
            m_ci = g_boot.get(metric, {})
            ci_l = m_ci.get("ci_lower")
            ci_u = m_ci.get("ci_upper")
            group_data[f"{metric}_ci_lower"] = round(ci_l, 6) if ci_l is not None else None
            group_data[f"{metric}_ci_upper"] = round(ci_u, 6) if ci_u is not None else None

        # Add p-value fields
        g_perm = perm_results.get(g_str, {})
        for metric in ["tpr", "fpr", "fnr", "precision", "pr_auc"]:
            m_perm = g_perm.get(metric, {})
            group_data[f"{metric}_p_value"] = m_perm.get("p_value")

        # CI width warning
        fnr_ci_l = group_data.get("fnr_ci_lower")
        fnr_ci_u = group_data.get("fnr_ci_upper")
        if fnr_ci_l is not None and fnr_ci_u is not None:
            fnr_width = fnr_ci_u - fnr_ci_l
            if fnr_width > 0.3:
                group_data["ci_warning"] = "wide_interval"
            elif n_unweighted < 30:
                group_data["ci_warning"] = "insufficient_sample"
        elif n_unweighted < 30:
            group_data["ci_warning"] = "insufficient_sample"

        # Calibration by group: among predicted high-risk (uncalibrated > 0.7)
        high_risk_mask = group_mask & (y_prob_uncal > 0.7)
        n_high = int(high_risk_mask.sum())
        if n_high >= min_high_risk:
            actual_rate = float(
                np.average(y_true[high_risk_mask], weights=weights[high_risk_mask])
            )
            group_data["calibration_high_risk"] = {
                "n_predicted_high": n_high,
                "actual_dropout_rate": round(actual_rate, 6),
            }
        else:
            group_data["calibration_high_risk"] = {
                "n_predicted_high": n_high,
                "actual_dropout_rate": None,
            }

        if n_unweighted < min_sample:
            group_data["flagged_small_sample"] = True

        result_groups[str(group_name)] = group_data

    # Compute gaps
    gaps = _compute_gaps(mf_binary)

    return {
        "sensitive_feature": dimension_name,
        "min_sample": min_sample,
        "groups": result_groups,
        "gaps": gaps,
    }


def _analyze_intersection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    sensitive_df: pd.DataFrame,
    y_prob_uncal: np.ndarray,
    intersection_name: str,
    min_sample: int = 50,
    min_high_risk: int = 30,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute fairness metrics for an intersectional analysis.

    Parameters
    ----------
    sensitive_df : pd.DataFrame
        DataFrame with 2 columns representing the intersection dimensions.
    """
    binary_params = {k: {"sample_weight": weights} for k in BINARY_METRICS}
    proba_params = {k: {"sample_weight": weights} for k in PROBA_METRICS}

    mf_binary = MetricFrame(
        metrics=BINARY_METRICS,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_df,
        sample_params=binary_params,
    )

    mf_proba = MetricFrame(
        metrics=PROBA_METRICS,
        y_true=y_true,
        y_pred=y_prob,
        sensitive_features=sensitive_df,
        sample_params=proba_params,
    )

    combined = pd.concat([mf_binary.by_group, mf_proba.by_group], axis=1)

    # Build combined group labels for bootstrap
    intersection_labels = np.array([
        "_".join(str(sensitive_df[col].values[i]) for col in sensitive_df.columns)
        for i in range(len(y_true))
    ])

    # ---- Bootstrap CIs for intersection (cached) ----
    ci_metric_fns = {
        "tpr": (_safe_recall, "binary"),
        "fpr": (weighted_fpr, "binary"),
        "fnr": (_safe_fnr, "binary"),
        "precision": (_safe_precision, "binary"),
        "pr_auc": (average_precision_score, "prob"),
    }

    root = find_project_root()
    cache_dir = root / "data" / "processed" / "cache"
    cache_path = cache_dir / f"stat_cache_int_{intersection_name}.json"
    dhash = _data_hash(y_true, y_pred, y_prob, weights, intersection_labels)

    cached = _load_cache(cache_path, dhash)
    if cached is not None:
        boot_cis = cached["bootstrap_cis"]
        print(f"    [cache hit] {intersection_name}")
    else:
        print(f"    [computing] bootstrap for {intersection_name}...")
        boot_cis = bootstrap_subgroup_cis(
            y_true, y_pred, y_prob, weights, intersection_labels,
            metric_fns=ci_metric_fns,
            n_replicates=n_bootstrap,
            seed=42,
        )
        _save_cache(cache_path, dhash, boot_cis, {})

    # Build per-group output (MultiIndex -> string key)
    result_groups = {}
    for idx in combined.index:
        # MultiIndex tuple -> "val1_val2" string
        group_label = "_".join(str(v) for v in idx)

        # Build mask for this intersection group
        mask = np.ones(len(y_true), dtype=bool)
        for col_idx, col_name in enumerate(sensitive_df.columns):
            mask = mask & (sensitive_df[col_name].values == idx[col_idx])

        n_unweighted = int(mask.sum())
        n_weighted = float(weights[mask].sum())

        group_data = {
            "n_unweighted": n_unweighted,
            "n_weighted": round(n_weighted, 2),
            "tpr": round(float(combined.loc[idx, "tpr"]), 6),
            "fpr": round(float(combined.loc[idx, "fpr"]), 6),
            "fnr": round(float(combined.loc[idx, "fnr"]), 6),
            "precision": round(float(combined.loc[idx, "precision"]), 6),
            "pr_auc": round(float(combined.loc[idx, "pr_auc"]), 6),
        }

        # Add CI fields from bootstrap
        g_boot = boot_cis.get(group_label, {})
        for metric in ["tpr", "fpr", "fnr", "precision", "pr_auc"]:
            m_ci = g_boot.get(metric, {})
            ci_l = m_ci.get("ci_lower")
            ci_u = m_ci.get("ci_upper")
            group_data[f"{metric}_ci_lower"] = round(ci_l, 6) if ci_l is not None else None
            group_data[f"{metric}_ci_upper"] = round(ci_u, 6) if ci_u is not None else None

        # CI width warning
        fnr_ci_l = group_data.get("fnr_ci_lower")
        fnr_ci_u = group_data.get("fnr_ci_upper")
        if fnr_ci_l is not None and fnr_ci_u is not None:
            fnr_width = fnr_ci_u - fnr_ci_l
            if fnr_width > 0.3:
                group_data["ci_warning"] = "wide_interval"
            elif n_unweighted < 30:
                group_data["ci_warning"] = "insufficient_sample"
        elif n_unweighted < 30:
            group_data["ci_warning"] = "insufficient_sample"

        # Calibration by group
        high_risk_mask = mask & (y_prob_uncal > 0.7)
        n_high = int(high_risk_mask.sum())
        if n_high >= min_high_risk:
            actual_rate = float(
                np.average(y_true[high_risk_mask], weights=weights[high_risk_mask])
            )
            group_data["calibration_high_risk"] = {
                "n_predicted_high": n_high,
                "actual_dropout_rate": round(actual_rate, 6),
            }
        else:
            group_data["calibration_high_risk"] = {
                "n_predicted_high": n_high,
                "actual_dropout_rate": None,
            }

        if n_unweighted < min_sample:
            group_data["flagged_small_sample"] = True

        result_groups[group_label] = group_data

    # Compute gaps
    gaps = _compute_gaps(mf_binary)

    return {
        "groups": result_groups,
        "gaps": gaps,
    }


def _compute_gaps(mf_binary: MetricFrame) -> dict:
    """Compute equalized odds, predictive parity, and max FNR gap."""
    by_group = mf_binary.by_group

    tpr_vals = by_group["tpr"].dropna()
    fpr_vals = by_group["fpr"].dropna()
    fnr_vals = by_group["fnr"].dropna()
    prec_vals = by_group["precision"].dropna()

    eo_tpr = float(tpr_vals.max() - tpr_vals.min()) if len(tpr_vals) > 1 else 0.0
    eo_fpr = float(fpr_vals.max() - fpr_vals.min()) if len(fpr_vals) > 1 else 0.0
    pp = float(prec_vals.max() - prec_vals.min()) if len(prec_vals) > 1 else 0.0

    if len(fnr_vals) > 1:
        max_fnr_gap = float(fnr_vals.max() - fnr_vals.min())
        max_fnr_groups = [str(fnr_vals.idxmax()), str(fnr_vals.idxmin())]
    else:
        max_fnr_gap = 0.0
        max_fnr_groups = []

    return {
        "equalized_odds_tpr": round(eo_tpr, 6),
        "equalized_odds_fpr": round(eo_fpr, 6),
        "predictive_parity": round(pp, 6),
        "max_fnr_gap": round(max_fnr_gap, 6),
        "max_fnr_groups": max_fnr_groups,
    }


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


_PRED_FILE_MAP = {
    "lgbm_calibrated": "predictions_lgbm_calibrated.parquet",
    "rf": "predictions_rf.parquet",
    "mlp": "predictions_mlp.parquet",
    "lr": "predictions_lr.parquet",
    "xgb": "predictions_xgb.parquet",
}

_THRESHOLD_MAP = {
    "lgbm_calibrated": lambda r: r["test_2023_calibrated"]["metadata"]["optimal_threshold"],
    "rf": lambda r: r["random_forest"]["threshold_analysis"]["optimal_threshold"],
    "mlp": lambda r: r["mlp"]["threshold_analysis"]["optimal_threshold"],
    "lr": lambda r: r["logistic_regression"]["threshold_analysis"]["optimal_threshold"],
    "xgb": lambda r: r["xgboost"]["threshold_analysis"]["optimal_threshold"],
}


def _load_and_prepare_data(model_name: str = "lgbm_calibrated") -> dict:
    """Load predictions + features and prepare arrays for fairness analysis.

    Parameters
    ----------
    model_name : str
        Model key: lgbm_calibrated, rf, mlp, lr, xgb.

    Returns
    -------
    dict
        Keys: y_true, y_pred, y_prob, y_prob_uncal, weights, merged_df,
              threshold, n_test, n_dropouts
    """
    root = find_project_root()
    pred_path = root / "data" / "processed" / _PRED_FILE_MAP[model_name]
    feat_path = root / "data" / "processed" / "enaho_with_features.parquet"
    results_path = root / "data" / "exports" / "model_results.json"

    # Load threshold from model_results.json
    with open(results_path) as f:
        model_results = json.load(f)
    threshold = _THRESHOLD_MAP[model_name](model_results)
    print(f"  Threshold ({model_name}): {threshold}")

    # Load predictions and filter to test_2023
    pred = pl.read_parquet(pred_path)
    test_pred = pred.filter(pl.col("split") == "test_2023")
    assert test_pred.height == 25635, (
        f"Expected 25,635 test rows, got {test_pred.height}"
    )
    print(f"  Test set: {test_pred.height:,} rows")

    # Load features and JOIN to get sensitive features
    feat = pl.read_parquet(feat_path)
    join_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"]
    meta_cols = [
        "p300a_harmonized",
        "p300a_original",
        "region_natural",
        "es_mujer",
        "rural",
        "es_peruano",
        "poverty_quintile",
        "lang_castellano",
        "lang_quechua",
        "lang_aimara",
        "lang_other_indigenous",
        "lang_foreign",
        "department",
    ]

    merged = test_pred.join(
        feat.select(join_keys + meta_cols),
        on=join_keys,
        how="left",
    )
    assert merged.height == test_pred.height, (
        f"Join changed row count: {test_pred.height} -> {merged.height}"
    )
    print(f"  Merged: {merged.height:,} rows, {merged.width} columns")

    # Extract numpy arrays
    y_true = merged["dropout"].cast(pl.Int8).to_numpy()
    y_prob = merged["prob_dropout"].to_numpy()
    if model_name == "lgbm_calibrated" and "prob_dropout_uncalibrated" in merged.columns:
        y_prob_uncal = merged["prob_dropout_uncalibrated"].to_numpy()
    else:
        y_prob_uncal = y_prob  # Same as predicted for non-calibrated models
    weights = merged["FACTOR07"].to_numpy()

    # Binary predictions using calibrated threshold
    y_pred = (y_prob >= threshold).astype(int)

    n_dropouts = int(y_true.sum())
    print(f"  Dropouts: {n_dropouts}")
    print(f"  Predicted positive: {y_pred.sum()}")
    print(f"  Uncalibrated > 0.7: {(y_prob_uncal > 0.7).sum()}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_prob_uncal": y_prob_uncal,
        "weights": weights,
        "merged_df": merged,
        "threshold": threshold,
        "n_test": test_pred.height,
        "n_dropouts": n_dropouts,
    }


def _build_language_groups(merged: pl.DataFrame) -> np.ndarray:
    """Build harmonized language group labels from language dummies."""
    lang_cols = [
        ("lang_castellano", "castellano"),
        ("lang_quechua", "quechua"),
        ("lang_aimara", "aimara"),
        ("lang_other_indigenous", "other_indigenous"),
        ("lang_foreign", "foreign"),
    ]
    labels = []
    for i in range(merged.height):
        found = False
        for col, label in lang_cols:
            if merged[col][i] == 1:
                labels.append(label)
                found = True
                break
        if not found:
            labels.append("unknown")
    return np.array(labels)


def _build_disaggregated_language(merged: pl.DataFrame) -> np.ndarray:
    """Build disaggregated language labels from p300a_original."""
    codes = merged["p300a_original"].to_numpy()
    labels = []
    for code in codes:
        int_code = int(code) if not np.isnan(code) else -1
        labels.append(P300A_LABEL_MAP.get(int_code, f"code_{int_code}"))
    return np.array(labels)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


REFERENCE_GROUPS = {
    "language": "castellano",
    "language_disaggregated": "castellano",
    "sex": "male",
    "geography": "urban",
    "region": "costa",
    "poverty": "Q5",
    "nationality": None,  # skip — n=27 non-Peruvian
}


def run_fairness_pipeline(
    model_name: str = "lgbm_calibrated",
    n_bootstrap: int = 1000,
    n_permutations: int = 5000,
) -> dict:
    """Run the full fairness metrics pipeline.

    Parameters
    ----------
    model_name : str
        Model key: lgbm_calibrated, rf, mlp, lr, xgb.
    n_bootstrap : int
        Number of bootstrap replicates for CIs.
    n_permutations : int
        Number of permutations for hypothesis tests.

    Returns
    -------
    dict
        The fairness_metrics.json content.
    """
    root = find_project_root()
    if model_name == "lgbm_calibrated":
        output_path = root / "data" / "exports" / "fairness_metrics.json"
    else:
        output_path = root / "data" / "exports" / f"fairness_metrics_{model_name}.json"

    # -----------------------------------------------------------------------
    # Step 1: Load and prepare data
    # -----------------------------------------------------------------------
    print(f"Step 1: Loading and preparing data for model '{model_name}'...")
    data = _load_and_prepare_data(model_name=model_name)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data["y_prob"]
    y_prob_uncal = data["y_prob_uncal"]
    weights = data["weights"]
    merged = data["merged_df"]
    threshold = data["threshold"]

    # -----------------------------------------------------------------------
    # Step 2: Build group arrays for all dimensions
    # -----------------------------------------------------------------------
    print("\nStep 2: Building group arrays...")

    # 1. Language (harmonized)
    language_groups = _build_language_groups(merged)
    print(f"  language: {np.unique(language_groups, return_counts=True)}")

    # 2. Language disaggregated
    language_disagg = _build_disaggregated_language(merged)
    print(f"  language_disagg: {len(np.unique(language_disagg))} unique codes")

    # 3. Sex
    sex_groups = np.where(
        merged["es_mujer"].to_numpy() == 1, "female", "male"
    )

    # 4. Geography
    geo_groups = np.where(
        merged["rural"].to_numpy() == 1, "rural", "urban"
    )

    # 5. Region
    region_groups = merged["region_natural"].to_numpy()

    # 6. Poverty
    poverty_groups = np.array(
        [f"Q{q}" for q in merged["poverty_quintile"].to_numpy()]
    )

    # 7. Nationality
    nationality_groups = np.where(
        merged["es_peruano"].to_numpy() == 1, "peruvian", "non_peruvian"
    )

    # -----------------------------------------------------------------------
    # Step 3: Analyze each dimension
    # -----------------------------------------------------------------------
    print("\nStep 3: Analyzing dimensions...")

    dimensions = {}

    dim_configs = [
        ("language", language_groups, "p300a_harmonized (dummies)", 100),
        ("language_disaggregated", language_disagg, "p300a_original", 50),
        ("sex", sex_groups, "es_mujer", 100),
        ("geography", geo_groups, "rural", 100),
        ("region", region_groups, "region_natural", 100),
        ("poverty", poverty_groups, "poverty_quintile", 100),
        ("nationality", nationality_groups, "es_peruano", 100),
    ]

    for dim_name, groups, feature_name, min_sample in dim_configs:
        print(f"\n  Analyzing: {dim_name} (min_sample={min_sample})...")
        ref = REFERENCE_GROUPS.get(dim_name)
        result = _analyze_dimension(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            weights=weights,
            groups=groups,
            y_prob_uncal=y_prob_uncal,
            dimension_name=feature_name,
            min_sample=min_sample,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            reference_group=ref,
        )
        dimensions[dim_name] = result

        # Print summary
        for gname, gdata in result["groups"].items():
            flag = " [SMALL SAMPLE]" if gdata.get("flagged_small_sample") else ""
            print(
                f"    {gname}: n={gdata['n_unweighted']}, "
                f"FNR={gdata['fnr']:.4f}, TPR={gdata['tpr']:.4f}, "
                f"PR-AUC={gdata['pr_auc']:.4f}{flag}"
            )
        print(
            f"    Gaps: FNR gap={result['gaps']['max_fnr_gap']:.4f} "
            f"({result['gaps']['max_fnr_groups']})"
        )

    # -----------------------------------------------------------------------
    # Step 4: Intersectional analyses
    # -----------------------------------------------------------------------
    print("\nStep 4: Intersectional analyses...")

    intersections = {}

    # 4a. Language x Rural (exclude foreign for meaningful intersection)
    lang_for_intersect = language_groups.copy()
    # Keep: castellano, quechua, aimara, other_indigenous. Exclude: foreign, unknown
    lang_mask = np.isin(
        lang_for_intersect,
        ["castellano", "quechua", "aimara", "other_indigenous"],
    )

    if lang_mask.sum() > 0:
        print(f"\n  language_x_rural: {lang_mask.sum()} rows (excluding foreign/unknown)...")
        inter_df = pd.DataFrame(
            {
                "language": lang_for_intersect[lang_mask],
                "rural": geo_groups[lang_mask],
            }
        )
        result = _analyze_intersection(
            y_true=y_true[lang_mask],
            y_pred=y_pred[lang_mask],
            y_prob=y_prob[lang_mask],
            weights=weights[lang_mask],
            sensitive_df=inter_df,
            y_prob_uncal=y_prob_uncal[lang_mask],
            intersection_name="language_x_rural",
            min_sample=50,
            n_bootstrap=n_bootstrap,
        )
        intersections["language_x_rural"] = result

    # 4b. Sex x Poverty
    print(f"\n  sex_x_poverty: {len(y_true)} rows...")
    inter_df = pd.DataFrame(
        {
            "sex": sex_groups,
            "poverty": poverty_groups,
        }
    )
    result = _analyze_intersection(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        weights=weights,
        sensitive_df=inter_df,
        y_prob_uncal=y_prob_uncal,
        intersection_name="sex_x_poverty",
        min_sample=50,
        n_bootstrap=n_bootstrap,
    )
    intersections["sex_x_poverty"] = result

    # 4c. Language x Region (exclude foreign/unknown)
    if lang_mask.sum() > 0:
        print(f"\n  language_x_region: {lang_mask.sum()} rows...")
        inter_df = pd.DataFrame(
            {
                "language": lang_for_intersect[lang_mask],
                "region": region_groups[lang_mask],
            }
        )
        result = _analyze_intersection(
            y_true=y_true[lang_mask],
            y_pred=y_pred[lang_mask],
            y_prob=y_prob[lang_mask],
            weights=weights[lang_mask],
            sensitive_df=inter_df,
            y_prob_uncal=y_prob_uncal[lang_mask],
            intersection_name="language_x_region",
            min_sample=50,
            n_bootstrap=n_bootstrap,
        )
        intersections["language_x_region"] = result

    # -----------------------------------------------------------------------
    # Step 5: Build and export JSON
    # -----------------------------------------------------------------------
    print("\nStep 5: Exporting fairness_metrics.json...")

    fairness_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "threshold": threshold,
        "threshold_type": "calibrated",
        "calibration_note": (
            "Calibrated probs max at 0.431; high-risk (>0.7) uses uncalibrated probs"
        ),
        "test_set": "2023",
        "n_test": data["n_test"],
        "n_dropouts": data["n_dropouts"],
        "bootstrap_replicates": n_bootstrap,
        "permutation_replicates": n_permutations,
        "ci_level": 0.95,
        "ci_method": "percentile_bootstrap",
        "test_method": "permutation_test",
        "dimensions": dimensions,
        "intersections": intersections,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fairness_json, f, indent=2, default=str)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({file_size_kb:.1f} KB)")

    # -----------------------------------------------------------------------
    # Step 6: Console summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FAIRNESS METRICS SUMMARY")
    print("=" * 70)

    # FNR by language group (harmonized)
    print("\n=== FNR BY LANGUAGE GROUP (HARMONIZED) ===")
    lang_dim = dimensions["language"]
    for gname, gdata in sorted(
        lang_dim["groups"].items(), key=lambda x: x[1]["fnr"], reverse=True
    ):
        flag = " *" if gdata.get("flagged_small_sample") else ""
        print(f"  {gname:<20s}  FNR={gdata['fnr']:.4f}  n={gdata['n_unweighted']}{flag}")

    # FNR by geography
    print("\n=== FNR BY GEOGRAPHY ===")
    geo_dim = dimensions["geography"]
    for gname, gdata in sorted(
        geo_dim["groups"].items(), key=lambda x: x[1]["fnr"], reverse=True
    ):
        print(f"  {gname:<20s}  FNR={gdata['fnr']:.4f}  n={gdata['n_unweighted']}")

    # Calibration table
    print("\n=== CALIBRATION BY GROUP (HIGH RISK >0.7 UNCALIBRATED) ===")
    for dim_name in ["language", "sex", "geography", "region", "poverty"]:
        dim = dimensions[dim_name]
        print(f"\n  {dim_name}:")
        for gname, gdata in dim["groups"].items():
            cal = gdata["calibration_high_risk"]
            rate_str = f"{cal['actual_dropout_rate']:.4f}" if cal["actual_dropout_rate"] is not None else "N/A (n<30)"
            print(f"    {gname:<25s}  n_high={cal['n_predicted_high']:>4d}  actual_rate={rate_str}")

    # Max FNR gaps
    print("\n=== MAX FNR GAPS BY DIMENSION ===")
    for dim_name, dim_data in dimensions.items():
        gaps = dim_data["gaps"]
        groups_str = " vs ".join(gaps["max_fnr_groups"]) if gaps["max_fnr_groups"] else "N/A"
        print(f"  {dim_name:<30s}  gap={gaps['max_fnr_gap']:.4f}  ({groups_str})")

    # Flagged small sample groups
    print("\n=== FLAGGED SMALL SAMPLE GROUPS ===")
    for dim_name, dim_data in dimensions.items():
        for gname, gdata in dim_data["groups"].items():
            if gdata.get("flagged_small_sample"):
                print(f"  {dim_name}/{gname}: n={gdata['n_unweighted']}")
    for int_name, int_data in intersections.items():
        for gname, gdata in int_data["groups"].items():
            if gdata.get("flagged_small_sample"):
                print(f"  {int_name}/{gname}: n={gdata['n_unweighted']}")

    # Intersectional highlights: language_x_rural sorted by FNR
    if "language_x_rural" in intersections:
        print("\n=== INTERSECTIONAL HIGHLIGHTS: LANGUAGE x RURAL ===")
        int_data = intersections["language_x_rural"]
        sorted_groups = sorted(
            int_data["groups"].items(),
            key=lambda x: x[1]["fnr"],
            reverse=True,
        )
        for gname, gdata in sorted_groups:
            flag = " *" if gdata.get("flagged_small_sample") else ""
            print(
                f"  {gname:<30s}  FNR={gdata['fnr']:.4f}  "
                f"n={gdata['n_unweighted']}{flag}"
            )

    # Overall stats
    n_dims = len(dimensions)
    n_ints = len(intersections)
    n_int_groups = sum(len(v["groups"]) for v in intersections.values())
    print(f"\n  Dimensions: {n_dims}")
    print(f"  Intersections: {n_ints} ({n_int_groups} groups)")
    print(f"  JSON file: {file_size_kb:.1f} KB")
    print("=" * 70)

    return fairness_json


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_fairness_pipeline()
