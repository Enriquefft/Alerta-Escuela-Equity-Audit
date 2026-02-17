"""Permutation-based hypothesis tests for fairness metric disparities.

Tests H0: metric(group_A) == metric(group_B) by shuffling group labels
between A and B while preserving survey weights, then computing an
empirical p-value from the permutation distribution.

Optimized with fast numpy-only metric implementations.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from fairness.bootstrap import FAST_METRICS


def permutation_test_disparity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    group_mask_a: np.ndarray,
    group_mask_b: np.ndarray,
    metric_fn: Callable,
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict:
    """Test H0: metric(group_A) == metric(group_B).

    Permutes group labels between A and B while preserving weights.
    Two-sided test: p = proportion of |permuted_gap| >= |observed_gap|.
    """
    rng = np.random.default_rng(seed)

    # Extract combined A + B subset
    combined_mask = group_mask_a | group_mask_b
    yt = y_true[combined_mask]
    yp = y_pred[combined_mask]
    w = weights[combined_mask]
    local_a = group_mask_a[combined_mask]
    n_combined = len(yt)
    n_a = int(local_a.sum())

    def _compute(mask):
        if mask.sum() < 2:
            return np.nan
        m_yt = yt[mask]
        if m_yt.sum() == 0 or m_yt.sum() == mask.sum():
            return np.nan
        return metric_fn(m_yt, yp[mask], w[mask])

    obs_a = _compute(local_a)
    obs_b = _compute(~local_a)

    if not np.isfinite(obs_a) or not np.isfinite(obs_b):
        return {"observed_gap": None, "p_value": None, "n_permutations": n_permutations}

    observed_gap = obs_a - obs_b

    # Vectorized permutation: generate all shuffled indices at once
    count_extreme = 0
    for _ in range(n_permutations):
        perm_idx = rng.permutation(n_combined)
        perm_a = np.zeros(n_combined, dtype=bool)
        perm_a[perm_idx[:n_a]] = True

        perm_val_a = _compute(perm_a)
        perm_val_b = _compute(~perm_a)
        if not np.isfinite(perm_val_a) or not np.isfinite(perm_val_b):
            continue
        if abs(perm_val_a - perm_val_b) >= abs(observed_gap):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "observed_gap": round(float(observed_gap), 6),
        "p_value": round(float(p_value), 6),
        "n_permutations": n_permutations,
    }


def test_all_disparities(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    weights: np.ndarray,
    group_labels: np.ndarray,
    reference_group: str,
    metric_fns: dict[str, tuple[Callable, str]],
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict:
    """Test all groups against reference for all metrics.

    Returns {group: {metric: {observed_gap, p_value, n_permutations}}}.
    Reference group gets null p-values.
    """
    unique_groups = np.unique(group_labels)
    ref_mask = group_labels == reference_group

    result: dict[str, dict[str, dict]] = {}

    for g in unique_groups:
        g_str = str(g)
        result[g_str] = {}

        if g_str == str(reference_group):
            for metric_name in metric_fns:
                result[g_str][metric_name] = {
                    "observed_gap": None,
                    "p_value": None,
                    "n_permutations": n_permutations,
                }
            continue

        g_mask = group_labels == g

        for metric_name, (_, pred_type) in metric_fns.items():
            # Use fast metric from bootstrap module
            fast_fn, _ = FAST_METRICS[metric_name]
            y_pred = y_pred_binary if pred_type == "binary" else y_pred_prob
            test_result = permutation_test_disparity(
                y_true=y_true,
                y_pred=y_pred,
                weights=weights,
                group_mask_a=g_mask,
                group_mask_b=ref_mask,
                metric_fn=fast_fn,
                n_permutations=n_permutations,
                seed=seed,
            )
            result[g_str][metric_name] = test_result

    return result
