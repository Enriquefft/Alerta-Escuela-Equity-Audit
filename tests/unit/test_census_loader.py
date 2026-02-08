"""Unit tests for Census 2017 district-level indicators loader.

Tests CensusResult dataclass, UBIGEO validation, coverage calculations,
missing file behavior, and value range checks using synthetic data.
"""

import sys

sys.path.insert(0, "src")

import polars as pl
import pytest
from unittest.mock import patch

from data.census import CensusResult, load_census_2017


# ---------------------------------------------------------------------------
# CensusResult dataclass
# ---------------------------------------------------------------------------


class TestCensusResult:
    """Tests for the CensusResult container."""

    def test_defaults(self):
        """CensusResult has sensible defaults."""
        df = pl.DataFrame({"UBIGEO": ["010101"]})
        result = CensusResult(df=df)
        assert result.districts_count == 0
        assert result.coverage_stats == {}
        assert result.warnings == []

    def test_with_all_fields(self):
        """CensusResult stores all provided fields."""
        df = pl.DataFrame({"UBIGEO": ["010101"]})
        result = CensusResult(
            df=df,
            districts_count=1890,
            coverage_stats={"poverty_rate": 0.99},
            warnings=["test"],
        )
        assert result.districts_count == 1890
        assert result.coverage_stats["poverty_rate"] == 0.99


# ---------------------------------------------------------------------------
# Missing file behavior
# ---------------------------------------------------------------------------


class TestCensusMissingFile:
    """Tests for placeholder behavior when census file is missing."""

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing file returns empty DataFrame with warning."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            result = load_census_2017()
            assert result.districts_count == 0
            assert result.df.height == 0
            assert len(result.warnings) > 0
            assert "not found" in result.warnings[0].lower()

    def test_missing_file_has_correct_schema(self, tmp_path):
        """Empty DataFrame has correct column schema."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            result = load_census_2017()
            assert "UBIGEO" in result.df.columns
            # Check prefixed columns exist
            expected_cols = [
                "census_poverty_rate",
                "census_indigenous_lang_pct",
                "census_water_access_pct",
                "census_electricity_pct",
                "census_literacy_rate",
            ]
            for col in expected_cols:
                assert col in result.df.columns


# ---------------------------------------------------------------------------
# Successful loading
# ---------------------------------------------------------------------------


class TestCensusLoading:
    """Tests for successful census data loading."""

    def test_load_success(self, tmp_path):
        """Successfully loads census data with correct columns."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            census_dir = tmp_path / "data" / "raw" / "census"
            census_dir.mkdir(parents=True)
            (census_dir / "census_2017_districts.csv").write_text(
                "UBIGEO,poverty_rate,indigenous_lang_pct,water_access_pct,"
                "electricity_pct,literacy_rate\n"
                "010101,45.0,50.0,55.0,65.0,78.0\n"
                "150101,12.0,8.0,90.0,97.0,96.0\n"
            )

            result = load_census_2017()
            assert result.districts_count == 2
            assert "census_poverty_rate" in result.df.columns
            assert "census_literacy_rate" in result.df.columns

    def test_ubigeo_padding(self, tmp_path):
        """UBIGEO values are zero-padded to 6 characters."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            census_dir = tmp_path / "data" / "raw" / "census"
            census_dir.mkdir(parents=True)
            (census_dir / "census_2017_districts.csv").write_text(
                "UBIGEO,poverty_rate\n10101,45.0\n150101,12.0\n"
            )

            result = load_census_2017()
            ubigeos = result.df["UBIGEO"].to_list()
            assert all(len(u) == 6 for u in ubigeos)
            assert "010101" in ubigeos


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestCensusValidation:
    """Tests for census data validation."""

    def test_duplicate_ubigeo_raises(self, tmp_path):
        """Duplicate UBIGEO values raise ValueError."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            census_dir = tmp_path / "data" / "raw" / "census"
            census_dir.mkdir(parents=True)
            (census_dir / "census_2017_districts.csv").write_text(
                "UBIGEO,poverty_rate\n010101,45.0\n010101,12.0\n"
            )

            with pytest.raises(ValueError, match="duplicate UBIGEO"):
                load_census_2017()

    def test_bad_ubigeo_length_raises(self, tmp_path):
        """UBIGEO values with wrong length raise ValueError."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            census_dir = tmp_path / "data" / "raw" / "census"
            census_dir.mkdir(parents=True)
            # 7-digit UBIGEO after padding (original is 7 digits, won't change)
            (census_dir / "census_2017_districts.csv").write_text(
                "UBIGEO,poverty_rate\n0101010,45.0\n"
            )

            with pytest.raises(ValueError, match="UBIGEO.*length != 6"):
                load_census_2017()


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------


class TestCoverageStats:
    """Tests for coverage rate calculations."""

    def test_full_coverage(self, tmp_path):
        """All non-null values produce 100% coverage."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            census_dir = tmp_path / "data" / "raw" / "census"
            census_dir.mkdir(parents=True)
            (census_dir / "census_2017_districts.csv").write_text(
                "UBIGEO,poverty_rate,indigenous_lang_pct\n"
                "010101,45.0,50.0\n150101,12.0,8.0\n"
            )

            result = load_census_2017()
            assert result.coverage_stats["poverty_rate"] == 1.0
            assert result.coverage_stats["indigenous_lang_pct"] == 1.0

    def test_partial_coverage(self, tmp_path):
        """Null values reduce coverage rate."""
        with patch("data.census.find_project_root", return_value=tmp_path):
            census_dir = tmp_path / "data" / "raw" / "census"
            census_dir.mkdir(parents=True)
            (census_dir / "census_2017_districts.csv").write_text(
                "UBIGEO,poverty_rate\n010101,45.0\n150101,\n"
            )

            result = load_census_2017()
            assert result.coverage_stats["poverty_rate"] == 0.5
