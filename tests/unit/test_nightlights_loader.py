"""Unit tests for VIIRS nighttime lights district-level loader.

Tests NightlightsResult dataclass, UBIGEO validation, negative value
detection, missing file behavior, and coverage calculation using
synthetic data.
"""

import sys

sys.path.insert(0, "src")

import polars as pl
import pytest
from unittest.mock import patch

from data.nightlights import NightlightsResult, load_viirs_nightlights


# ---------------------------------------------------------------------------
# NightlightsResult dataclass
# ---------------------------------------------------------------------------


class TestNightlightsResult:
    """Tests for the NightlightsResult container."""

    def test_defaults(self):
        """NightlightsResult has sensible defaults."""
        df = pl.DataFrame({"UBIGEO": ["010101"]})
        result = NightlightsResult(df=df)
        assert result.districts_count == 0
        assert result.coverage_rate == 0.0
        assert result.stats == {}
        assert result.warnings == []

    def test_with_all_fields(self):
        """NightlightsResult stores all provided fields."""
        df = pl.DataFrame({"UBIGEO": ["010101"]})
        result = NightlightsResult(
            df=df,
            districts_count=1839,
            coverage_rate=0.95,
            stats={"mean": 14.0, "median": 10.0},
            warnings=["test"],
        )
        assert result.districts_count == 1839
        assert result.coverage_rate == 0.95
        assert result.stats["mean"] == 14.0


# ---------------------------------------------------------------------------
# Missing file behavior
# ---------------------------------------------------------------------------


class TestNightlightsMissingFile:
    """Tests for placeholder behavior when nightlights file is missing."""

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing file returns empty DataFrame with warning."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            result = load_viirs_nightlights()
            assert result.districts_count == 0
            assert result.coverage_rate == 0.0
            assert result.df.height == 0
            assert len(result.warnings) > 0
            assert "not found" in result.warnings[0].lower()

    def test_missing_file_has_correct_schema(self, tmp_path):
        """Empty DataFrame has correct column schema."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            result = load_viirs_nightlights()
            assert "UBIGEO" in result.df.columns
            assert "nightlights_mean_radiance" in result.df.columns


# ---------------------------------------------------------------------------
# Successful loading
# ---------------------------------------------------------------------------


class TestNightlightsLoading:
    """Tests for successful nightlights data loading."""

    def test_load_success(self, tmp_path):
        """Successfully loads nightlights data with correct columns."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            (nl_dir / "viirs_districts.csv").write_text(
                "UBIGEO,mean_radiance\n010101,3.5\n150101,45.2\n"
            )

            result = load_viirs_nightlights()
            assert result.districts_count == 2
            assert "nightlights_mean_radiance" in result.df.columns
            assert result.stats["mean"] == pytest.approx(24.35, abs=0.01)

    def test_ubigeo_padding(self, tmp_path):
        """UBIGEO values are zero-padded to 6 characters."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            (nl_dir / "viirs_districts.csv").write_text(
                "UBIGEO,mean_radiance\n10101,3.5\n150101,45.2\n"
            )

            result = load_viirs_nightlights()
            ubigeos = result.df["UBIGEO"].to_list()
            assert all(len(u) == 6 for u in ubigeos)
            assert "010101" in ubigeos


# ---------------------------------------------------------------------------
# Negative value detection
# ---------------------------------------------------------------------------


class TestNegativeValues:
    """Tests for validation rejecting negative nightlight values."""

    def test_negative_radiance_raises(self, tmp_path):
        """Negative radiance values raise ValueError."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            (nl_dir / "viirs_districts.csv").write_text(
                "UBIGEO,mean_radiance\n010101,-0.5\n150101,3.0\n"
            )

            with pytest.raises(ValueError, match="negative radiance"):
                load_viirs_nightlights()

    def test_zero_radiance_accepted(self, tmp_path):
        """Zero radiance value is accepted (not negative)."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            (nl_dir / "viirs_districts.csv").write_text(
                "UBIGEO,mean_radiance\n010101,0.0\n150101,3.0\n"
            )

            result = load_viirs_nightlights()
            assert result.stats["min"] == 0.0


# ---------------------------------------------------------------------------
# Coverage calculation
# ---------------------------------------------------------------------------


class TestCoverageCalculation:
    """Tests for district coverage rate calculation."""

    def test_full_coverage(self, tmp_path):
        """1839 districts produces 100% coverage."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            # Create 1839 rows
            lines = ["UBIGEO,mean_radiance"]
            for i in range(1839):
                ubigeo = f"{i:06d}"
                lines.append(f"{ubigeo},{i * 0.01}")
            (nl_dir / "viirs_districts.csv").write_text("\n".join(lines))

            result = load_viirs_nightlights()
            assert result.coverage_rate == pytest.approx(1.0, abs=0.001)

    def test_low_coverage_warning(self, tmp_path):
        """Coverage below 85% produces a warning."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            (nl_dir / "viirs_districts.csv").write_text(
                "UBIGEO,mean_radiance\n010101,3.5\n"
            )

            result = load_viirs_nightlights()
            assert result.coverage_rate < 0.85
            assert any("coverage below 85%" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Duplicate UBIGEO detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """Tests for duplicate UBIGEO detection."""

    def test_duplicate_ubigeo_raises(self, tmp_path):
        """Duplicate UBIGEO values raise ValueError."""
        with patch("data.nightlights.find_project_root", return_value=tmp_path):
            nl_dir = tmp_path / "data" / "raw" / "nightlights"
            nl_dir.mkdir(parents=True)
            (nl_dir / "viirs_districts.csv").write_text(
                "UBIGEO,mean_radiance\n010101,3.5\n010101,4.0\n"
            )

            with pytest.raises(ValueError, match="duplicate UBIGEO"):
                load_viirs_nightlights()
