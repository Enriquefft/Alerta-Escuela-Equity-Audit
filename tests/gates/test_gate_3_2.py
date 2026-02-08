"""Gate test 3.2: SHAP interpretability analysis validation.

Validates shap_values.json structure, completeness, and correctness.
Prints top-5 SHAP features, LR overlap, ES_PERUANO/ES_MUJER magnitudes,
regional differences, and profile summaries for human review.

Usage::

    uv run pytest tests/gates/test_gate_3_2.py -v -s

Use ``-s`` flag to see the human-review print tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import find_project_root

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = find_project_root()
SHAP_PATH = ROOT / "data" / "exports" / "shap_values.json"


@pytest.fixture(scope="module")
def shap_data():
    """Load and parse shap_values.json."""
    assert SHAP_PATH.exists(), (
        "shap_values.json not found. Run: uv run python src/fairness/shap_analysis.py"
    )
    with open(SHAP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_json_exists_and_valid(shap_data):
    """Assert shap_values.json exists, is valid JSON, has required top-level keys."""
    required_keys = {
        "generated_at", "model", "computed_on", "n_test",
        "feature_names", "feature_labels_es",
        "global_importance", "regional", "profiles",
        "es_peruano", "es_mujer", "interactions",
    }
    actual_keys = set(shap_data.keys())
    missing = required_keys - actual_keys
    assert not missing, f"Missing top-level keys: {missing}"


def test_global_importance_complete(shap_data):
    """Assert global_importance has exactly 25 features with non-negative values."""
    gi = shap_data["global_importance"]
    assert len(gi) == 25, f"Expected 25 features, got {len(gi)}"
    for feat, val in gi.items():
        assert isinstance(val, (int, float)), f"{feat} value is not numeric: {val}"
        assert val >= 0, f"{feat} has negative importance: {val}"
    total = sum(gi.values())
    assert total > 0, "All global importance values are zero"


def test_feature_names_match(shap_data):
    """Assert feature_names matches MODEL_FEATURES from data.features."""
    from data.features import MODEL_FEATURES
    assert shap_data["feature_names"] == list(MODEL_FEATURES), (
        "feature_names in JSON does not match MODEL_FEATURES constant"
    )


def test_feature_labels_es_complete(shap_data):
    """Assert feature_labels_es has exactly 25 non-empty strings."""
    labels = shap_data["feature_labels_es"]
    assert len(labels) == 25, f"Expected 25 labels, got {len(labels)}"
    for i, label in enumerate(labels):
        assert isinstance(label, str), f"Label {i} is not a string: {label}"
        assert len(label) > 0, f"Label {i} is empty"


def test_regional_shap_complete(shap_data):
    """Assert regional dict has costa, sierra, selva with 25 features each."""
    regional = shap_data["regional"]
    for region in ["costa", "sierra", "selva"]:
        assert region in regional, f"Missing region: {region}"
        region_data = regional[region]
        assert len(region_data) == 25, (
            f"Region {region} has {len(region_data)} features, expected 25"
        )
        for feat, val in region_data.items():
            assert val >= 0, f"{region}/{feat} has negative value: {val}"


def test_lr_overlap_documented(shap_data):
    """Assert top-5 SHAP and LR lists documented with overlap count.

    NOTE: If overlap < 3, overlap_note must explain why.
    The gate does NOT fail on low overlap -- it documents the finding.
    """
    assert "top_5_shap" in shap_data
    assert "top_5_lr" in shap_data
    assert "overlap_count" in shap_data

    assert len(shap_data["top_5_shap"]) == 5, (
        f"Expected 5 SHAP features, got {len(shap_data['top_5_shap'])}"
    )
    assert len(shap_data["top_5_lr"]) == 5, (
        f"Expected 5 LR features, got {len(shap_data['top_5_lr'])}"
    )
    assert isinstance(shap_data["overlap_count"], int)
    assert shap_data["overlap_count"] >= 0

    # If overlap < 3, require an explanation note
    if shap_data["overlap_count"] < 3:
        assert "overlap_note" in shap_data, (
            f"Overlap is {shap_data['overlap_count']}/5 but no overlap_note provided"
        )
        assert len(shap_data["overlap_note"]) > 0, (
            "overlap_note is empty despite overlap < 3"
        )


def test_es_peruano_es_mujer_quantified(shap_data):
    """Assert es_peruano and es_mujer sections have required fields."""
    for section_name in ["es_peruano", "es_mujer"]:
        section = shap_data[section_name]
        assert "mean_abs_shap" in section, (
            f"{section_name} missing mean_abs_shap"
        )
        assert "mean_signed_shap" in section, (
            f"{section_name} missing mean_signed_shap"
        )
        assert "rank" in section, f"{section_name} missing rank"
        assert section["mean_abs_shap"] >= 0, (
            f"{section_name} mean_abs_shap is negative"
        )
        assert 1 <= section["rank"] <= 25, (
            f"{section_name} rank {section['rank']} out of range 1-25"
        )


def test_profiles_complete(shap_data):
    """Assert profiles has exactly 10 entries with required fields."""
    profiles = shap_data["profiles"]
    assert len(profiles) == 10, f"Expected 10 profiles, got {len(profiles)}"

    for p in profiles:
        assert "profile_id" in p, "Profile missing profile_id"
        assert "feature_values" in p, f"{p.get('profile_id')} missing feature_values"
        assert "shap_values" in p, f"{p.get('profile_id')} missing shap_values"
        assert "predicted_probability" in p, (
            f"{p.get('profile_id')} missing predicted_probability"
        )
        assert "base_value" in p, f"{p.get('profile_id')} missing base_value"
        assert "n_in_group" in p, f"{p.get('profile_id')} missing n_in_group"

        # Check counts
        assert len(p["feature_values"]) == 25, (
            f"{p['profile_id']} has {len(p['feature_values'])} feature_values, expected 25"
        )
        assert len(p["shap_values"]) == 25, (
            f"{p['profile_id']} has {len(p['shap_values'])} shap_values, expected 25"
        )

        # Check value ranges
        prob = p["predicted_probability"]
        assert 0 <= prob <= 1, (
            f"{p['profile_id']} predicted_probability {prob} out of [0, 1] range"
        )
        assert p["n_in_group"] > 0, (
            f"{p['profile_id']} has n_in_group={p['n_in_group']}"
        )


def test_interactions_present(shap_data):
    """Assert interactions section has subsample_n (1000) and key_pairs."""
    interactions = shap_data["interactions"]
    assert interactions["subsample_n"] == 1000, (
        f"Expected subsample_n=1000, got {interactions['subsample_n']}"
    )
    assert "key_pairs" in interactions
    assert len(interactions["key_pairs"]) >= 2, (
        f"Expected at least 2 key_pairs, got {len(interactions['key_pairs'])}"
    )
    for pair in interactions["key_pairs"]:
        assert "feature_a" in pair
        assert "feature_b" in pair
        assert "mean_abs_interaction" in pair


def test_figures_exist():
    """Assert all 5 SHAP PNG figures exist and are non-empty."""
    fig_dir = ROOT / "data" / "exports" / "figures"
    expected_figures = [
        "shap_beeswarm_global.png",
        "shap_bar_top10.png",
        "shap_regional_comparison.png",
        "shap_force_es_peruano.png",
        "shap_force_es_mujer.png",
    ]
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 0, (
            f"Figure is empty: {fig_path}"
        )


def test_print_human_review(shap_data):
    """Print formatted SHAP analysis tables for human review. Always passes.

    Run with ``-s`` flag to see output:
    ``uv run pytest tests/gates/test_gate_3_2.py -v -s``
    """
    print()
    print()
    print("=" * 70)
    print("GATE 3.2: SHAP INTERPRETABILITY ANALYSIS -- HUMAN REVIEW")
    print("=" * 70)

    # --- Top-10 global SHAP features ---
    print("\n=== TOP-10 GLOBAL SHAP FEATURES (mean |SHAP|) ===")
    gi = shap_data["global_importance"]
    sorted_gi = sorted(gi.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Rank':<6s} {'Feature':<40s} {'Mean |SHAP|':>12s}")
    print("-" * 60)
    for i, (feat, val) in enumerate(sorted_gi[:10], 1):
        print(f"{i:<6d} {feat:<40s} {val:>12.6f}")

    # --- LR vs SHAP top-5 overlap ---
    print(f"\n=== LR vs SHAP TOP-5 OVERLAP: {shap_data['overlap_count']}/5 ===")
    print(f"  SHAP top-5: {shap_data['top_5_shap']}")
    print(f"  LR   top-5: {shap_data['top_5_lr']}")
    print(f"  Overlap features: {shap_data.get('overlap_features', [])}")
    if shap_data.get("overlap_note"):
        print(f"  Note: {shap_data['overlap_note']}")

    # --- ES_PERUANO ---
    ep = shap_data["es_peruano"]
    print(f"\n=== ES_PERUANO ===")
    print(f"  Mean |SHAP|:  {ep['mean_abs_shap']:.6f}")
    print(f"  Mean signed:  {ep['mean_signed_shap']:.6f}")
    print(f"  Rank:         {ep['rank']}/25")
    if "conditional" in ep:
        cond = ep["conditional"]
        print(f"  Peruvian (n={cond.get('n_peruano', 'N/A')}): "
              f"mean SHAP = {cond.get('peruano_mean_shap', 'N/A')}")
        print(f"  Foreign  (n={cond.get('n_foreign', 'N/A')}): "
              f"mean SHAP = {cond.get('foreign_mean_shap', 'N/A')}")

    # --- ES_MUJER ---
    em = shap_data["es_mujer"]
    print(f"\n=== ES_MUJER ===")
    print(f"  Mean |SHAP|:  {em['mean_abs_shap']:.6f}")
    print(f"  Mean signed:  {em['mean_signed_shap']:.6f}")
    print(f"  Rank:         {em['rank']}/25")
    if "conditional" in em:
        cond = em["conditional"]
        print(f"  Female (n={cond.get('n_female', 'N/A')}): "
              f"mean SHAP = {cond.get('female_mean_shap', 'N/A')}")
        print(f"  Male   (n={cond.get('n_male', 'N/A')}): "
              f"mean SHAP = {cond.get('male_mean_shap', 'N/A')}")

    # --- Regional top-3 ---
    print(f"\n=== REGIONAL TOP-3 FEATURES ===")
    for region_name in ["costa", "sierra", "selva"]:
        region_data = shap_data["regional"][region_name]
        sorted_region = sorted(region_data.items(), key=lambda x: x[1], reverse=True)
        top3 = [f"{f}={v:.4f}" for f, v in sorted_region[:3]]
        print(f"  {region_name:<8s}: {', '.join(top3)}")

    # --- Interaction pairs ---
    print(f"\n=== KEY INTERACTION PAIRS ===")
    for pair in shap_data["interactions"]["key_pairs"]:
        print(
            f"  {pair['feature_a']} x {pair['feature_b']}: "
            f"{pair['mean_abs_interaction']:.6f}"
        )
    if "focused_pairs" in shap_data["interactions"]:
        fp = shap_data["interactions"]["focused_pairs"]
        print(f"\n  Focused pairs (spec):")
        for pair_name, val in fp.items():
            print(f"    {pair_name}: {val:.6f}")

    # --- Profile summaries ---
    print(f"\n=== PROFILES ({len(shap_data['profiles'])}) ===")
    print(f"{'Profile ID':<32s} {'n':>6s} {'Prob':>8s} {'Flag':>8s}")
    print("-" * 56)
    for p in shap_data["profiles"]:
        flag = "SMALL" if p.get("flagged_small_sample") else ""
        print(
            f"{p['profile_id']:<32s} {p['n_in_group']:>6d} "
            f"{p['predicted_probability']:>8.4f} {flag:>8s}"
        )

    # --- Figures ---
    print(f"\n=== FIGURES ===")
    fig_dir = ROOT / "data" / "exports" / "figures"
    for fig_name in [
        "shap_beeswarm_global.png",
        "shap_bar_top10.png",
        "shap_regional_comparison.png",
        "shap_force_es_peruano.png",
        "shap_force_es_mujer.png",
    ]:
        fig_path = fig_dir / fig_name
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            print(f"  [OK] {fig_name} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {fig_name}")

    print()
    print("=" * 70)

    # Always pass
    assert True
