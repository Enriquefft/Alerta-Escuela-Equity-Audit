"""Gate test 3.1: Subgroup fairness metrics validation.

Validates fairness_metrics.json structure, completeness, and correctness.
Prints FNR by language group, calibration table, and intersectional highlights
for human review.

Usage::

    uv run pytest tests/gates/test_gate_3_1.py -v -s

Use ``-s`` flag to see the human-review print tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import find_project_root

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = find_project_root()
FAIRNESS_PATH = ROOT / "data" / "exports" / "fairness_metrics.json"

REQUIRED_DIMENSIONS = [
    "language",
    "language_disaggregated",
    "sex",
    "geography",
    "region",
    "poverty",
    "nationality",
]

REQUIRED_GROUP_METRICS = [
    "n_unweighted",
    "n_weighted",
    "tpr",
    "fpr",
    "fnr",
    "precision",
    "pr_auc",
]

REQUIRED_GAP_KEYS = [
    "equalized_odds_tpr",
    "equalized_odds_fpr",
    "predictive_parity",
    "max_fnr_gap",
    "max_fnr_groups",
]


@pytest.fixture(scope="module")
def fairness_data():
    """Load and parse fairness_metrics.json."""
    assert FAIRNESS_PATH.exists(), (
        f"fairness_metrics.json not found at {FAIRNESS_PATH}. "
        "Run: uv run python src/fairness/metrics.py"
    )
    with open(FAIRNESS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_json_exists_and_valid(fairness_data):
    """Assert fairness_metrics.json exists, is valid JSON, has required top-level keys."""
    required_keys = {"generated_at", "model", "threshold", "dimensions", "intersections"}
    assert required_keys.issubset(fairness_data.keys()), (
        f"Missing top-level keys: {required_keys - fairness_data.keys()}"
    )


def test_all_dimensions_present(fairness_data):
    """Assert dimensions dict has all 7 required keys."""
    dims = fairness_data["dimensions"]
    for dim in REQUIRED_DIMENSIONS:
        assert dim in dims, f"Missing dimension: {dim}"
    assert len(dims) == 7, f"Expected 7 dimensions, got {len(dims)}"


def test_dimension_groups_have_required_metrics(fairness_data):
    """For each dimension, each group must have TPR/FPR/FNR/precision/PR-AUC + calibration."""
    for dim_name, dim_data in fairness_data["dimensions"].items():
        assert "groups" in dim_data, f"Dimension {dim_name} missing 'groups'"
        for group_name, group_data in dim_data["groups"].items():
            for metric in REQUIRED_GROUP_METRICS:
                assert metric in group_data, (
                    f"{dim_name}/{group_name} missing metric: {metric}"
                )
            assert "calibration_high_risk" in group_data, (
                f"{dim_name}/{group_name} missing calibration_high_risk"
            )
            cal = group_data["calibration_high_risk"]
            assert "n_predicted_high" in cal, (
                f"{dim_name}/{group_name} missing n_predicted_high"
            )


def test_primary_dimension_sample_sizes(fairness_data):
    """Primary dimensions: non-flagged groups have n_unweighted >= 100.
    Nationality: es_peruano=0 group must be flagged."""
    primary_dims = ["language", "sex", "geography", "region", "poverty"]
    for dim_name in primary_dims:
        dim_data = fairness_data["dimensions"][dim_name]
        for group_name, group_data in dim_data["groups"].items():
            if not group_data.get("flagged_small_sample", False):
                assert group_data["n_unweighted"] >= 100, (
                    f"{dim_name}/{group_name}: n_unweighted={group_data['n_unweighted']} "
                    f"but not flagged"
                )

    # Nationality: non_peruvian must be flagged
    nat_dim = fairness_data["dimensions"]["nationality"]
    assert "non_peruvian" in nat_dim["groups"], "Missing non_peruvian group"
    assert nat_dim["groups"]["non_peruvian"].get("flagged_small_sample", False), (
        "non_peruvian group (n=27) should be flagged as small sample"
    )


def test_disaggregated_language_threshold(fairness_data):
    """Disaggregated language uses min_sample=50 threshold."""
    dim = fairness_data["dimensions"]["language_disaggregated"]
    assert dim["min_sample"] == 50, (
        f"Expected min_sample=50, got {dim['min_sample']}"
    )
    # Any group with n_unweighted < 50 must be flagged
    for group_name, group_data in dim["groups"].items():
        if group_data["n_unweighted"] < 50:
            assert group_data.get("flagged_small_sample", False), (
                f"language_disaggregated/{group_name}: n={group_data['n_unweighted']} < 50 "
                f"but not flagged"
            )


def test_weighted_differs_from_unweighted(fairness_data):
    """Confirm FACTOR07 weights are actually applied by comparing weighted vs unweighted TPR for sex."""
    import polars as pl

    pred = pl.read_parquet(ROOT / "data" / "processed" / "predictions_lgbm_calibrated.parquet")
    feat = pl.read_parquet(ROOT / "data" / "processed" / "enaho_with_features.parquet")

    test_pred = pred.filter(pl.col("split") == "test_2023")
    join_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"]
    merged = test_pred.join(
        feat.select(join_keys + ["es_mujer"]),
        on=join_keys,
        how="left",
    )

    # Load threshold
    with open(ROOT / "data" / "exports" / "model_results.json") as f:
        threshold = json.load(f)["test_2023_calibrated"]["metadata"]["optimal_threshold"]

    y_true = merged["dropout"].cast(pl.Int8).to_numpy()
    y_prob = merged["prob_dropout"].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)

    # Compute unweighted TPR for female
    female_mask = merged["es_mujer"].to_numpy() == 1
    unweighted_tpr_female = recall_score(
        y_true[female_mask], y_pred[female_mask], zero_division=np.nan
    )

    # Get weighted TPR from JSON
    weighted_tpr_female = fairness_data["dimensions"]["sex"]["groups"]["female"]["tpr"]

    diff = abs(weighted_tpr_female - unweighted_tpr_female)
    assert diff > 0.001, (
        f"Weighted TPR ({weighted_tpr_female:.6f}) and unweighted TPR "
        f"({unweighted_tpr_female:.6f}) are too similar (diff={diff:.6f}). "
        f"FACTOR07 may not be applied."
    )


def test_gaps_present(fairness_data):
    """Each dimension must have gaps dict with required keys."""
    for dim_name, dim_data in fairness_data["dimensions"].items():
        assert "gaps" in dim_data, f"Dimension {dim_name} missing 'gaps'"
        gaps = dim_data["gaps"]
        for key in REQUIRED_GAP_KEYS:
            assert key in gaps, f"{dim_name}/gaps missing key: {key}"


def test_intersections_present(fairness_data):
    """Assert 3 intersections present with at least 3 groups each."""
    required_ints = ["language_x_rural", "sex_x_poverty", "language_x_region"]
    ints = fairness_data["intersections"]
    for int_name in required_ints:
        assert int_name in ints, f"Missing intersection: {int_name}"
        assert "groups" in ints[int_name], f"Intersection {int_name} missing 'groups'"
        n_groups = len(ints[int_name]["groups"])
        assert n_groups >= 3, (
            f"Intersection {int_name} has only {n_groups} groups (expected >= 3)"
        )


def test_intersection_small_sample_flagging(fairness_data):
    """At least one intersection group must be flagged as small sample."""
    any_flagged = False
    for int_name, int_data in fairness_data["intersections"].items():
        for group_name, group_data in int_data["groups"].items():
            if group_data.get("flagged_small_sample", False):
                any_flagged = True
                break
        if any_flagged:
            break
    assert any_flagged, "No intersection groups flagged as small sample"


def test_fnr_consistency(fairness_data):
    """For each dimension and group, FNR + TPR should approximately equal 1.0."""
    for dim_name, dim_data in fairness_data["dimensions"].items():
        for group_name, group_data in dim_data["groups"].items():
            tpr = group_data["tpr"]
            fnr = group_data["fnr"]
            # Skip NaN entries (e.g., groups with no positives)
            if tpr is None or fnr is None:
                continue
            # Handle NaN strings from JSON serialization
            if isinstance(tpr, str) or isinstance(fnr, str):
                continue
            if np.isnan(tpr) or np.isnan(fnr):
                continue
            total = tpr + fnr
            assert abs(total - 1.0) < 0.001, (
                f"{dim_name}/{group_name}: TPR={tpr} + FNR={fnr} = {total} != 1.0"
            )


def test_model_and_threshold(fairness_data):
    """Assert model is lightgbm and threshold is approximately 0.165716."""
    assert fairness_data["model"] == "lightgbm", (
        f"Expected model='lightgbm', got '{fairness_data['model']}'"
    )
    assert abs(fairness_data["threshold"] - 0.165716) < 0.001, (
        f"Expected threshold ~0.165716, got {fairness_data['threshold']}"
    )


def test_print_human_review(fairness_data):
    """Print fairness findings tables for human review. Always passes.

    Run with ``-s`` flag to see output: ``uv run pytest tests/gates/test_gate_3_1.py -v -s``
    """
    print()
    print()
    print("=" * 70)
    print("=== FNR BY LANGUAGE GROUP (HARMONIZED) ===")
    print("=" * 70)
    lang = fairness_data["dimensions"]["language"]
    sorted_groups = sorted(
        lang["groups"].items(), key=lambda x: x[1]["fnr"] if not _is_nan(x[1]["fnr"]) else -1, reverse=True
    )
    print(f"{'Group':<25s} {'FNR':>10s} {'TPR':>10s} {'n_unweighted':>14s} {'Flagged':>10s}")
    print("-" * 70)
    for gname, gdata in sorted_groups:
        fnr_str = f"{gdata['fnr']:.4f}" if not _is_nan(gdata["fnr"]) else "NaN"
        tpr_str = f"{gdata['tpr']:.4f}" if not _is_nan(gdata["tpr"]) else "NaN"
        flag = "YES" if gdata.get("flagged_small_sample") else ""
        print(f"{gname:<25s} {fnr_str:>10s} {tpr_str:>10s} {gdata['n_unweighted']:>14d} {flag:>10s}")

    print()
    print("=" * 70)
    print("=== FNR BY GEOGRAPHY ===")
    print("=" * 70)
    geo = fairness_data["dimensions"]["geography"]
    for gname, gdata in sorted(
        geo["groups"].items(), key=lambda x: x[1]["fnr"], reverse=True
    ):
        print(f"  {gname:<15s}  FNR={gdata['fnr']:.4f}  n={gdata['n_unweighted']}")

    print()
    print("=" * 70)
    print("=== CALIBRATION BY GROUP (HIGH RISK >0.7 UNCALIBRATED) ===")
    print("=" * 70)
    for dim_name in ["language", "sex", "geography", "region"]:
        dim = fairness_data["dimensions"][dim_name]
        print(f"\n  {dim_name}:")
        for gname, gdata in dim["groups"].items():
            cal = gdata["calibration_high_risk"]
            rate = cal["actual_dropout_rate"]
            rate_str = f"{rate:.4f}" if rate is not None else "N/A (n<30)"
            print(f"    {gname:<25s}  n_high={cal['n_predicted_high']:>4d}  actual_rate={rate_str}")

    print()
    print("=" * 70)
    print("=== MAX FNR GAPS BY DIMENSION ===")
    print("=" * 70)
    for dim_name, dim_data in fairness_data["dimensions"].items():
        gaps = dim_data["gaps"]
        groups_str = " vs ".join(gaps["max_fnr_groups"]) if gaps["max_fnr_groups"] else "N/A"
        print(f"  {dim_name:<30s}  gap={gaps['max_fnr_gap']:.4f}  ({groups_str})")

    print()
    print("=" * 70)
    print("=== FLAGGED SMALL SAMPLE GROUPS ===")
    print("=" * 70)
    for dim_name, dim_data in fairness_data["dimensions"].items():
        for gname, gdata in dim_data["groups"].items():
            if gdata.get("flagged_small_sample"):
                print(f"  {dim_name}/{gname}: n={gdata['n_unweighted']}")
    for int_name, int_data in fairness_data["intersections"].items():
        for gname, gdata in int_data["groups"].items():
            if gdata.get("flagged_small_sample"):
                print(f"  {int_name}/{gname}: n={gdata['n_unweighted']}")

    print()
    print("=" * 70)
    print("=== INTERSECTIONAL HIGHLIGHTS: LANGUAGE x RURAL ===")
    print("=" * 70)
    int_data = fairness_data["intersections"]["language_x_rural"]
    sorted_groups = sorted(
        int_data["groups"].items(),
        key=lambda x: x[1]["fnr"] if not _is_nan(x[1]["fnr"]) else 999,
        reverse=True,
    )
    print(f"{'Group':<30s} {'FNR':>10s} {'n_unweighted':>14s} {'Flagged':>10s}")
    print("-" * 70)
    for gname, gdata in sorted_groups:
        fnr_str = f"{gdata['fnr']:.4f}" if not _is_nan(gdata["fnr"]) else "NaN"
        flag = "YES" if gdata.get("flagged_small_sample") else ""
        print(f"{gname:<30s} {fnr_str:>10s} {gdata['n_unweighted']:>14d} {flag:>10s}")

    print()
    # Always pass
    assert True


def _is_nan(val) -> bool:
    """Check if value is NaN (handles None, str, float)."""
    if val is None:
        return True
    if isinstance(val, str):
        return val.lower() == "nan"
    try:
        return np.isnan(float(val))
    except (ValueError, TypeError):
        return False
