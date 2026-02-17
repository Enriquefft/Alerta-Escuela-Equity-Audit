"""Bootstrap confidence intervals for survey-weighted fairness metrics.

Computes percentile bootstrap CIs by resampling individuals with replacement,
preserving FACTOR07 survey weights. Treats data as weighted simple random sample
(not cluster bootstrap on PSUs).

Optimized: uses fast numpy-only metric implementations to avoid sklearn
function call overhead in tight loops.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Fast numpy-only metric implementations (avoid sklearn overhead in loops)
# ---------------------------------------------------------------------------

def _fast_weighted_tpr(y_true, y_pred, w):
    """TPR = weighted recall among positives."""
    pos = y_true == 1
    if pos.sum() == 0:
        return np.nan
    return np.average(y_pred[pos], weights=w[pos])


def _fast_weighted_fnr(y_true, y_pred, w):
    """FNR = 1 - TPR."""
    tpr = _fast_weighted_tpr(y_true, y_pred, w)
    return 1.0 - tpr if np.isfinite(tpr) else np.nan


def _fast_weighted_fpr(y_true, y_pred, w):
    """FPR = weighted FP rate among negatives."""
    neg = y_true == 0
    if neg.sum() == 0:
        return np.nan
    return np.average(y_pred[neg], weights=w[neg])


def _fast_weighted_precision(y_true, y_pred, w):
    """Precision = weighted TP / weighted predicted positive."""
    pred_pos = y_pred == 1
    if pred_pos.sum() == 0:
        return np.nan
    return np.average(y_true[pred_pos], weights=w[pred_pos])


def _fast_weighted_pr_auc(y_true, y_prob, w):
    """Weighted average precision (PR-AUC).

    Uses sklearn for correctness but only called per-group per-replicate.
    """
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return average_precision_score(y_true, y_prob, sample_weight=w)


# Map metric names to fast implementations and whether they use binary/prob
FAST_METRICS = {
    "tpr": (_fast_weighted_tpr, "binary"),
    "fpr": (_fast_weighted_fpr, "binary"),
    "fnr": (_fast_weighted_fnr, "binary"),
    "precision": (_fast_weighted_precision, "binary"),
    "pr_auc": (_fast_weighted_pr_auc, "prob"),
}


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    metric_fn: Callable,
    n_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for a single metric on a single group."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci_level

    point = float(metric_fn(y_true, y_pred, sample_weight=weights))

    boot_values = np.empty(n_replicates)
    valid = 0
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        w = weights[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        val = metric_fn(yt, yp, sample_weight=w)
        if np.isfinite(val):
            boot_values[valid] = val
            valid += 1

    if valid < 10:
        return {"point_estimate": point, "ci_lower": None, "ci_upper": None, "ci_width": None}

    boot_values = boot_values[:valid]
    ci_lower = float(np.percentile(boot_values, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_values, (1 - alpha / 2) * 100))
    return {"point_estimate": point, "ci_lower": ci_lower, "ci_upper": ci_upper, "ci_width": ci_upper - ci_lower}


def bootstrap_subgroup_cis(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    weights: np.ndarray,
    group_labels: np.ndarray,
    metric_fns: dict[str, tuple[Callable, str]],
    n_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CIs for all metrics across all subgroups.

    Optimized with fast numpy metrics and pre-computed group encoding.
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci_level

    # Encode groups as integers for fast comparison
    unique_groups = np.unique(group_labels)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    g_encoded = np.array([group_to_int[g] for g in group_labels], dtype=np.int32)
    n_groups = len(unique_groups)
    n_metrics = len(FAST_METRICS)

    # Pre-allocate: [n_replicates, n_groups, n_metrics]
    boot_store = np.full((n_replicates, n_groups, n_metrics), np.nan)
    metric_names = list(FAST_METRICS.keys())

    # Generate all bootstrap indices at once
    all_indices = rng.choice(n, size=(n_replicates, n), replace=True)

    for rep in range(n_replicates):
        idx = all_indices[rep]
        yt = y_true[idx]
        yp_bin = y_pred_binary[idx]
        yp_prob = y_pred_prob[idx]
        w = weights[idx]
        gl = g_encoded[idx]

        for gi in range(n_groups):
            g_mask = gl == gi
            g_n = g_mask.sum()
            if g_n < 2:
                continue
            g_yt = yt[g_mask]
            if g_yt.sum() == 0 or g_yt.sum() == g_n:
                continue
            g_w = w[g_mask]
            g_yp_bin = yp_bin[g_mask]
            g_yp_prob = yp_prob[g_mask]

            for mi, mname in enumerate(metric_names):
                fn, pred_type = FAST_METRICS[mname]
                yp = g_yp_bin if pred_type == "binary" else g_yp_prob
                try:
                    boot_store[rep, gi, mi] = fn(g_yt, yp, g_w)
                except Exception:
                    pass

    # Compute percentile CIs
    result: dict[str, dict[str, dict]] = {}
    for gi, g in enumerate(unique_groups):
        g_str = str(g)
        result[g_str] = {}
        for mi, mname in enumerate(metric_names):
            vals = boot_store[:, gi, mi]
            valid = vals[np.isfinite(vals)]
            if len(valid) < 10:
                result[g_str][mname] = {"ci_lower": None, "ci_upper": None, "ci_width": None}
            else:
                ci_l = float(np.percentile(valid, alpha / 2 * 100))
                ci_u = float(np.percentile(valid, (1 - alpha / 2) * 100))
                result[g_str][mname] = {"ci_lower": ci_l, "ci_upper": ci_u, "ci_width": ci_u - ci_l}

    return result
