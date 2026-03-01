"""Unit tests for ENAHO panel linkage assessment.

Tests cover both feasible and infeasible linkage outcomes, decision logic,
report structure, and trajectory feature computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from data.panel_linkage import (
    JOIN_KEYS,
    _PROCEED_THRESHOLD,
    _SKIP_THRESHOLD,
    _assess_year_pair,
    assess_panel_linkage,
    build_trajectory_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mod200(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Create a synthetic Module 200 DataFrame for testing."""
    import random
    random.seed(seed)
    return pl.DataFrame({
        "CONGLOME": [f"C{i:04d}" for i in range(n)],
        "VIVIENDA": [f"V{i:04d}" for i in range(n)],
        "HOGAR": [f"H{i:02d}" for i in range(n)],
        "CODPERSO": [f"P{i:03d}" for i in range(n)],
        "UBIGEO": [f"{random.randint(10101, 250901):06d}" for _ in range(n)],
        "P207": [random.choice([1, 2]) for _ in range(n)],
        "P208A": [random.randint(6, 17) for _ in range(n)],
        "DOMINIO": [random.randint(1, 8) for _ in range(n)],
        "ESTRATO": [random.randint(1, 5) for _ in range(n)],
    })


# ---------------------------------------------------------------------------
# Test: assess_panel_linkage returns correct structure
# ---------------------------------------------------------------------------


class TestAssessPanelLinkage:
    """Tests for the full assessment pipeline."""

    def test_returns_dict_with_required_keys(self):
        """assess_panel_linkage returns dict with all required top-level keys."""
        report = assess_panel_linkage()
        required_keys = {
            "assessment_date",
            "year_pairs",
            "overall",
            "decision",
            "reason",
            "trajectory_features_built",
            "publishable_finding",
        }
        assert required_keys.issubset(set(report.keys())), (
            f"Missing keys: {required_keys - set(report.keys())}"
        )

    def test_decision_is_valid(self):
        """Decision must be one of skip, marginal, proceed."""
        report = assess_panel_linkage()
        assert report["decision"] in ("skip", "marginal", "proceed")

    def test_year_pairs_structure(self):
        """Each year-pair entry has required fields."""
        report = assess_panel_linkage()
        for pair in report["year_pairs"]:
            assert "years" in pair
            assert "n_matched" in pair
            assert "linkage_rate" in pair
            assert "quality_age_consistent" in pair
            assert "quality_sex_consistent" in pair
            assert len(pair["years"]) == 2

    def test_at_least_four_year_pairs(self):
        """Should assess at least 4 consecutive year-pairs (2018-2023)."""
        report = assess_panel_linkage()
        assert len(report["year_pairs"]) >= 4

    def test_overall_statistics(self):
        """Overall section has linkage rate, quality, and effective rate."""
        report = assess_panel_linkage()
        overall = report["overall"]
        assert "mean_linkage_rate" in overall
        assert "quality_rate" in overall
        assert "effective_rate" in overall
        assert 0.0 <= overall["mean_linkage_rate"] <= 1.0
        assert 0.0 <= overall["effective_rate"] <= 1.0

    def test_report_json_exists(self):
        """Report JSON should be saved to disk."""
        from utils import find_project_root
        report_path = find_project_root() / "data" / "exports" / "panel_linkage_report.json"
        assert report_path.exists(), f"Report not found at {report_path}"
        data = json.loads(report_path.read_text())
        assert "decision" in data

    def test_publishable_finding_is_substantive(self):
        """Publishable finding should be a non-empty string."""
        report = assess_panel_linkage()
        assert isinstance(report["publishable_finding"], str)
        assert len(report["publishable_finding"]) > 50


# ---------------------------------------------------------------------------
# Test: Decision logic with mocked data
# ---------------------------------------------------------------------------


class TestDecisionLogic:
    """Test go/no-go decision boundaries."""

    def test_assess_year_pair_no_overlap(self):
        """Year pair with no common keys should produce 0% linkage."""
        df_t = _make_mod200(50, seed=1)
        # Create df_t1 with completely different keys by adding offset
        df_t1 = _make_mod200(50, seed=999).with_columns([
            (pl.lit("X") + pl.col("CONGLOME")).alias("CONGLOME"),
            (pl.lit("X") + pl.col("VIVIENDA")).alias("VIVIENDA"),
            (pl.lit("X") + pl.col("CODPERSO")).alias("CODPERSO"),
        ])
        result = _assess_year_pair(df_t, df_t1, 2020, 2021)
        assert result["linkage_rate"] == 0.0
        assert result["n_matched"] == 0

    def test_assess_year_pair_full_overlap(self):
        """Year pair with identical keys should have 100% linkage."""
        df = _make_mod200(50, seed=42)
        # Same data, age incremented by 1
        df_next = df.with_columns(
            (pl.col("P208A") + 1).alias("P208A")
        )
        result = _assess_year_pair(df, df_next, 2020, 2021)
        assert result["linkage_rate"] == 1.0
        assert result["n_matched"] == 50
        # All ages should be consistent (diff = 1)
        assert result["quality_age_consistent"] == 1.0
        # All sexes should be consistent
        assert result["quality_sex_consistent"] == 1.0

    def test_assess_year_pair_partial_overlap(self):
        """Partial key overlap produces intermediate linkage rate."""
        df_t = _make_mod200(100, seed=42)
        # Take first 30 from df_t plus 70 with completely different keys
        shared = df_t.head(30).with_columns(
            (pl.col("P208A") + 1).alias("P208A")
        )
        new = _make_mod200(70, seed=999).with_columns([
            (pl.lit("X") + pl.col("CONGLOME")).alias("CONGLOME"),
            (pl.lit("X") + pl.col("VIVIENDA")).alias("VIVIENDA"),
            (pl.lit("X") + pl.col("CODPERSO")).alias("CODPERSO"),
        ])
        df_t1 = pl.concat([shared, new])
        result = _assess_year_pair(df_t, df_t1, 2020, 2021)
        assert result["linkage_rate"] == pytest.approx(0.30, abs=0.01)
        assert result["n_matched"] == 30


# ---------------------------------------------------------------------------
# Test: build_trajectory_features
# ---------------------------------------------------------------------------


class TestBuildTrajectoryFeatures:
    """Test trajectory feature computation."""

    def test_empty_when_no_data(self):
        """Returns empty DataFrame with correct schema when data is None."""
        result = build_trajectory_features(None, None)
        assert result.height == 0
        assert "income_change" in result.columns
        assert "sibling_dropout" in result.columns
        assert "work_transition" in result.columns
        assert "year" in result.columns

    def test_empty_when_empty_dict(self):
        """Returns empty DataFrame when year_data dict is empty."""
        result = build_trajectory_features({}, [])
        assert result.height == 0

    def test_schema_correct_types(self):
        """Output schema has correct column types."""
        result = build_trajectory_features(None, None)
        assert result.schema["income_change"] == pl.Float64
        assert result.schema["sibling_dropout"] == pl.Int64
        assert result.schema["work_transition"] == pl.Int64
        assert result.schema["year"] == pl.Int64
        for key in JOIN_KEYS:
            assert result.schema[key] == pl.Utf8
