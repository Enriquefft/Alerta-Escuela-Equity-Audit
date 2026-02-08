"""Gate test 3.3: District-level cross-validation with admin dropout rates.

Validates choropleth.json structure, district coverage, positive correlation,
significant p-value, and stratified error analysis by indigenous language
prevalence. Prints a human-review summary with extreme-error districts.

Usage::

    uv run pytest tests/gates/test_gate_3_3.py -v -s

Use ``-s`` flag to see the human-review print tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import find_project_root

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = find_project_root()
CHOROPLETH_PATH = ROOT / "data" / "exports" / "choropleth.json"


@pytest.fixture(scope="module")
def choropleth_data():
    """Load and parse choropleth.json."""
    assert CHOROPLETH_PATH.exists(), (
        "choropleth.json not found. "
        "Run: uv run python src/fairness/cross_validation.py"
    )
    with open(CHOROPLETH_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_json_exists_and_valid(choropleth_data):
    """Assert choropleth.json exists, is valid JSON, has required top-level keys."""
    required_keys = {
        "generated_at",
        "n_districts",
        "n_with_predictions",
        "correlation",
        "error_by_indigenous_group",
        "districts",
    }
    actual_keys = set(choropleth_data.keys())
    missing = required_keys - actual_keys
    assert not missing, f"Missing top-level keys: {missing}"


def test_district_count_exceeds_1500(choropleth_data):
    """Assert n_districts > 1500."""
    n = choropleth_data["n_districts"]
    print(f"  District count: {n}")
    assert n > 1500, f"Expected > 1500 districts, got {n}"


def test_district_required_fields(choropleth_data):
    """Assert first 20 districts each have the 6 required keys."""
    required_fields = {
        "ubigeo",
        "predicted_dropout_rate",
        "admin_dropout_rate",
        "model_error",
        "indigenous_language_pct",
        "poverty_index",
    }
    for i, district in enumerate(choropleth_data["districts"][:20]):
        actual = set(district.keys())
        missing = required_fields - actual
        assert not missing, (
            f"District {i} (ubigeo={district.get('ubigeo')}) "
            f"missing fields: {missing}"
        )


def test_positive_correlation(choropleth_data):
    """Assert Pearson r is positive."""
    r = choropleth_data["correlation"]["pearson_r"]
    print(f"  Pearson r: {r:.6f}")
    assert r > 0, f"Pearson r = {r} is not positive"


def test_correlation_significant(choropleth_data):
    """Assert p-value < 0.05."""
    p = choropleth_data["correlation"]["p_value"]
    print(f"  p-value: {p:.6e}")
    assert p < 0.05, f"p-value = {p} is not < 0.05"


def test_error_by_indigenous_group_exists(choropleth_data):
    """Assert both high and low indigenous groups exist with valid data."""
    error_groups = choropleth_data["error_by_indigenous_group"]

    for group_key in ["high_indigenous_gt50", "low_indigenous_lt10"]:
        assert group_key in error_groups, f"Missing group: {group_key}"
        group = error_groups[group_key]
        assert group["n_districts"] > 0, (
            f"{group_key}: n_districts must be > 0, got {group['n_districts']}"
        )
        assert group["mean_absolute_error"] >= 0, (
            f"{group_key}: MAE must be >= 0, got {group['mean_absolute_error']}"
        )


def test_predictions_coverage(choropleth_data):
    """Assert n_with_predictions >= 1500."""
    n = choropleth_data["n_with_predictions"]
    print(f"  Districts with predictions: {n}")
    assert n >= 1500, f"Expected >= 1500 districts with predictions, got {n}"


def test_ubigeo_format(choropleth_data):
    """Assert ubigeo values are 6-character strings (sample 50 districts)."""
    sample = choropleth_data["districts"][:50]
    for i, district in enumerate(sample):
        ubigeo = district["ubigeo"]
        assert isinstance(ubigeo, str), (
            f"District {i}: ubigeo is {type(ubigeo).__name__}, expected str"
        )
        assert len(ubigeo) == 6, (
            f"District {i}: ubigeo '{ubigeo}' has length {len(ubigeo)}, expected 6"
        )


def test_no_nan_in_json():
    """Assert raw JSON text does not contain NaN (invalid JSON)."""
    with open(CHOROPLETH_PATH) as f:
        raw_text = f.read()
    assert "NaN" not in raw_text, (
        "choropleth.json contains 'NaN' -- this is invalid JSON. "
        "Use null instead of NaN."
    )
    assert "Infinity" not in raw_text, (
        "choropleth.json contains 'Infinity' -- this is invalid JSON."
    )


# ---------------------------------------------------------------------------
# Human review summary (always passes)
# ---------------------------------------------------------------------------


def test_human_review_summary(choropleth_data):
    """Print formatted cross-validation summary for human review. Always passes.

    Run with ``-s`` flag to see output:
    ``uv run pytest tests/gates/test_gate_3_3.py -v -s``
    """
    print()
    print()
    print("=" * 70)
    print("GATE 3.3: CROSS-VALIDATION WITH ADMIN DATA -- HUMAN REVIEW")
    print("=" * 70)

    # --- Correlation ---
    corr = choropleth_data["correlation"]
    print(f"\n=== CORRELATION (test 2023, out-of-sample) ===")
    print(f"  Pearson r:          {corr['pearson_r']:.6f}")
    print(f"  p-value:            {corr['p_value']:.6e}")
    print(f"  N districts:        {corr['n_districts_correlated']:,}")
    if "caveat" in corr:
        print(f"  Caveat:             {corr['caveat']}")

    # --- Error by indigenous group ---
    err = choropleth_data["error_by_indigenous_group"]
    print(f"\n=== PREDICTION ERROR BY INDIGENOUS GROUP ===")
    high = err["high_indigenous_gt50"]
    low = err["low_indigenous_lt10"]
    print(f"  {'Group':<30s} {'N districts':>12s} {'MAE (pp)':>10s}")
    print(f"  {'-' * 54}")
    print(f"  {'High-indigenous (>50%)':<30s} {high['n_districts']:>12d} {high['mean_absolute_error']:>10.4f}")
    print(f"  {'Low-indigenous (<10%)':<30s} {low['n_districts']:>12d} {low['mean_absolute_error']:>10.4f}")
    print(f"  {'MAE difference':<30s} {'':>12s} {abs(high['mean_absolute_error'] - low['mean_absolute_error']):>10.4f}")

    # --- Choropleth coverage ---
    n_total = choropleth_data["n_districts"]
    n_pred = choropleth_data["n_with_predictions"]
    print(f"\n=== CHOROPLETH COVERAGE ===")
    print(f"  Total districts:        {n_total:,}")
    print(f"  With predictions:       {n_pred:,}")
    print(f"  Without predictions:    {n_total - n_pred:,}")

    # --- Top 5 highest model_error (overprediction) ---
    districts = choropleth_data["districts"]
    with_error = [
        d for d in districts
        if d.get("model_error") is not None
    ]
    sorted_over = sorted(with_error, key=lambda x: x["model_error"], reverse=True)
    sorted_under = sorted(with_error, key=lambda x: x["model_error"])

    print(f"\n=== TOP 5 OVERPREDICTED DISTRICTS (highest model_error) ===")
    print(f"  {'UBIGEO':<10s} {'Predicted':>10s} {'Admin':>10s} {'Error':>10s} {'Indig%':>10s}")
    print(f"  {'-' * 52}")
    for d in sorted_over[:5]:
        pred = d["predicted_dropout_rate"]
        admin = d["admin_dropout_rate"]
        err_val = d["model_error"]
        indig = d.get("indigenous_language_pct")
        pred_str = f"{pred:.2f}" if pred is not None else "N/A"
        admin_str = f"{admin:.2f}" if admin is not None else "N/A"
        err_str = f"{err_val:.2f}" if err_val is not None else "N/A"
        indig_str = f"{indig:.1f}" if indig is not None else "N/A"
        print(f"  {d['ubigeo']:<10s} {pred_str:>10s} {admin_str:>10s} {err_str:>10s} {indig_str:>10s}")

    print(f"\n=== TOP 5 UNDERPREDICTED DISTRICTS (lowest model_error) ===")
    print(f"  {'UBIGEO':<10s} {'Predicted':>10s} {'Admin':>10s} {'Error':>10s} {'Indig%':>10s}")
    print(f"  {'-' * 52}")
    for d in sorted_under[:5]:
        pred = d["predicted_dropout_rate"]
        admin = d["admin_dropout_rate"]
        err_val = d["model_error"]
        indig = d.get("indigenous_language_pct")
        pred_str = f"{pred:.2f}" if pred is not None else "N/A"
        admin_str = f"{admin:.2f}" if admin is not None else "N/A"
        err_str = f"{err_val:.2f}" if err_val is not None else "N/A"
        indig_str = f"{indig:.1f}" if indig is not None else "N/A"
        print(f"  {d['ubigeo']:<10s} {pred_str:>10s} {admin_str:>10s} {err_str:>10s} {indig_str:>10s}")

    # --- Admin feature leakage caveat ---
    print(f"\n=== CAVEATS ===")
    print(f"  1. The model uses district_dropout_rate_admin_z as a feature")
    print(f"     (SHAP importance ~0.03). Correlation is partially mechanical.")
    print(f"  2. Admin data is SYNTHETIC (generated in Phase 3).")
    print(f"     Actual admin data was unavailable (datosabiertos.gob.pe 404).")
    print(f"  3. Model predictions are 6-year averages (2018-2023 all-years");
    print(f"     aggregation) while admin rates are for 2023 only.")

    print()
    print("  HUMAN REVIEW: Do indigenous-majority districts show")
    print("  higher prediction error? Is the spatial pattern consistent")
    print("  with known equity gaps in Peru?")
    print()
    print("=" * 70)

    # Always pass
    assert True
