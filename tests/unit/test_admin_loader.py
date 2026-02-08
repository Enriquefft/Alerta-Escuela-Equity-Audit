"""Unit tests for administrative dropout rate loader.

Tests AdminResult dataclass, UBIGEO validation, rate validation,
and error handling using synthetic data (no real CSV files required).
"""

import sys

sys.path.insert(0, "src")

import polars as pl
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from data.admin import AdminResult, load_admin_dropout_rates, _load_admin_csv


# ---------------------------------------------------------------------------
# AdminResult dataclass
# ---------------------------------------------------------------------------


class TestAdminResult:
    """Tests for the AdminResult container."""

    def test_defaults(self):
        """AdminResult has sensible defaults."""
        df = pl.DataFrame({"UBIGEO": ["010101"], "admin_primaria_rate": [0.5]})
        result = AdminResult(df=df)
        assert result.primaria_rate == 0.0
        assert result.secundaria_rate == 0.0
        assert result.districts_count == 0
        assert result.warnings == []

    def test_with_all_fields(self):
        """AdminResult stores all provided fields."""
        df = pl.DataFrame({"UBIGEO": ["010101"]})
        result = AdminResult(
            df=df,
            primaria_rate=0.93,
            secundaria_rate=2.05,
            districts_count=1890,
            warnings=["test warning"],
        )
        assert result.primaria_rate == 0.93
        assert result.secundaria_rate == 2.05
        assert result.districts_count == 1890
        assert result.warnings == ["test warning"]


# ---------------------------------------------------------------------------
# UBIGEO validation
# ---------------------------------------------------------------------------


class TestUbigeoValidation:
    """Tests for UBIGEO zero-padding and duplicate detection."""

    def test_ubigeo_zero_padding(self, tmp_path):
        """UBIGEO values are zero-padded to 6 characters."""
        csv_content = "UBIGEO,tasa_desercion\n10101,0.5\n150101,0.3\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = _load_admin_csv(csv_file, "primaria")
        ubigeo_vals = df["UBIGEO"].to_list()
        assert all(len(u) == 6 for u in ubigeo_vals)
        assert "010101" in ubigeo_vals

    def test_duplicate_ubigeo_raises(self, tmp_path):
        """Duplicate UBIGEO values raise ValueError."""
        csv_content = "UBIGEO,tasa_desercion\n010101,0.5\n010101,0.3\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="duplicate UBIGEO"):
            _load_admin_csv(csv_file, "primaria")


# ---------------------------------------------------------------------------
# Rate validation
# ---------------------------------------------------------------------------


class TestRateValidation:
    """Tests for admin dropout rate reasonableness checks."""

    def test_valid_rates(self, tmp_path):
        """Valid rates within 0-100 pass validation."""
        csv_content = "UBIGEO,tasa_desercion\n010101,0.5\n150101,1.2\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = _load_admin_csv(csv_file, "primaria")
        assert df.height == 2

    def test_negative_rate_raises(self, tmp_path):
        """Negative rates raise ValueError."""
        csv_content = "UBIGEO,tasa_desercion\n010101,-0.5\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="outside \\[0, 100\\]"):
            _load_admin_csv(csv_file, "primaria")

    def test_rate_above_100_raises(self, tmp_path):
        """Rates above 100% raise ValueError."""
        csv_content = "UBIGEO,tasa_desercion\n010101,105.0\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="outside \\[0, 100\\]"):
            _load_admin_csv(csv_file, "primaria")


# ---------------------------------------------------------------------------
# Load admin dropout rates (integration with files)
# ---------------------------------------------------------------------------


class TestLoadAdminDropoutRates:
    """Tests for the load_admin_dropout_rates function."""

    def test_missing_primaria_file_raises(self, tmp_path):
        """Missing primaria file raises FileNotFoundError."""
        with patch("data.admin.find_project_root", return_value=tmp_path):
            admin_dir = tmp_path / "data" / "raw" / "admin"
            admin_dir.mkdir(parents=True)
            # Only create secundaria
            (admin_dir / "secundaria_2023.csv").write_text(
                "UBIGEO,tasa_desercion\n010101,1.0\n"
            )
            with pytest.raises(FileNotFoundError, match="primaria"):
                load_admin_dropout_rates()

    def test_missing_secundaria_file_raises(self, tmp_path):
        """Missing secundaria file raises FileNotFoundError."""
        with patch("data.admin.find_project_root", return_value=tmp_path):
            admin_dir = tmp_path / "data" / "raw" / "admin"
            admin_dir.mkdir(parents=True)
            (admin_dir / "primaria_2023.csv").write_text(
                "UBIGEO,tasa_desercion\n010101,0.5\n"
            )
            with pytest.raises(FileNotFoundError, match="secundaria"):
                load_admin_dropout_rates()

    def test_successful_load(self, tmp_path):
        """Successful load returns AdminResult with correct stats."""
        with patch("data.admin.find_project_root", return_value=tmp_path):
            admin_dir = tmp_path / "data" / "raw" / "admin"
            admin_dir.mkdir(parents=True)
            (admin_dir / "primaria_2023.csv").write_text(
                "UBIGEO,tasa_desercion\n010101,0.5\n150101,0.3\n"
            )
            (admin_dir / "secundaria_2023.csv").write_text(
                "UBIGEO,tasa_desercion\n010101,1.0\n150101,0.8\n"
            )

            result = load_admin_dropout_rates()
            assert result.districts_count == 2
            assert result.primaria_rate == pytest.approx(0.4, abs=0.01)
            assert result.secundaria_rate == pytest.approx(0.9, abs=0.01)
            assert "admin_primaria_rate" in result.df.columns
            assert "admin_secundaria_rate" in result.df.columns

    def test_column_rename(self, tmp_path):
        """Rate columns are renamed with admin_ prefix."""
        with patch("data.admin.find_project_root", return_value=tmp_path):
            admin_dir = tmp_path / "data" / "raw" / "admin"
            admin_dir.mkdir(parents=True)
            (admin_dir / "primaria_2023.csv").write_text(
                "UBIGEO,tasa_desercion\n010101,0.5\n"
            )
            (admin_dir / "secundaria_2023.csv").write_text(
                "UBIGEO,tasa_desercion\n010101,1.0\n"
            )

            result = load_admin_dropout_rates()
            assert "admin_primaria_rate" in result.df.columns
            assert "admin_secundaria_rate" in result.df.columns
            assert "tasa_desercion" not in result.df.columns
