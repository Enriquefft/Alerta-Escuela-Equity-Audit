"""Unit tests for the spatial merge pipeline.

Tests merge_spatial_data function, row count preservation, merge rate
calculations, null reporting, and duplicate detection using mocked
data loaders to avoid I/O dependencies.
"""

import sys

sys.path.insert(0, "src")

import polars as pl
import pytest
from unittest.mock import patch

from data.admin import AdminResult
from data.census import CensusResult
from data.nightlights import NightlightsResult
from data.merge import (
    MergeResult,
    merge_spatial_data,
    validate_merge_pipeline,
    _calculate_merge_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enaho_df():
    """Synthetic ENAHO DataFrame for testing merges."""
    return pl.DataFrame({
        "UBIGEO": ["010101", "010101", "150101", "150101", "150102"],
        "P208A": [10, 12, 8, 15, 11],
        "dropout": [False, True, False, True, False],
        "FACTOR07": [100.0, 200.0, 150.0, 180.0, 120.0],
        "year": [2023, 2023, 2023, 2023, 2023],
    })


@pytest.fixture
def admin_result():
    """Synthetic admin result with matching districts."""
    return AdminResult(
        df=pl.DataFrame({
            "UBIGEO": ["010101", "150101", "150102"],
            "admin_primaria_rate": [1.5, 0.3, 0.4],
            "admin_secundaria_rate": [3.0, 0.8, 0.9],
        }),
        primaria_rate=0.73,
        secundaria_rate=1.57,
        districts_count=3,
        warnings=[],
    )


@pytest.fixture
def census_result():
    """Synthetic census result with matching districts."""
    return CensusResult(
        df=pl.DataFrame({
            "UBIGEO": ["010101", "150101", "150102"],
            "census_poverty_rate": [45.0, 12.0, 15.0],
            "census_indigenous_lang_pct": [50.0, 8.0, 10.0],
        }),
        districts_count=3,
        coverage_stats={"poverty_rate": 1.0, "indigenous_lang_pct": 1.0},
        warnings=[],
    )


@pytest.fixture
def nightlights_result():
    """Synthetic nightlights result with partial coverage."""
    return NightlightsResult(
        df=pl.DataFrame({
            "UBIGEO": ["010101", "150101"],  # 150102 missing
            "nightlights_mean_radiance": [3.5, 45.0],
        }),
        districts_count=2,
        coverage_rate=0.67,
        stats={"mean": 24.25, "median": 24.25, "min": 3.5, "max": 45.0},
        warnings=[],
    )


# ---------------------------------------------------------------------------
# MergeResult dataclass
# ---------------------------------------------------------------------------


class TestMergeResult:
    """Tests for the MergeResult container."""

    def test_defaults(self):
        """MergeResult has sensible defaults."""
        df = pl.DataFrame({"x": [1]})
        result = MergeResult(df=df)
        assert result.initial_rows == 0
        assert result.final_rows == 0
        assert result.merge_rates == {}
        assert result.null_report == {}
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Merge rate calculation
# ---------------------------------------------------------------------------


class TestMergeRateCalculation:
    """Tests for merge rate calculations."""

    def test_full_match(self):
        """All rows matched produces 100% merge rate."""
        df = pl.DataFrame({"col": [1.0, 2.0, 3.0]})
        rate = _calculate_merge_rate(df, "col")
        assert rate == 1.0

    def test_partial_match(self):
        """Some nulls produce partial merge rate."""
        df = pl.DataFrame({"col": [1.0, None, 3.0]})
        rate = _calculate_merge_rate(df, "col")
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_no_match(self):
        """All nulls produce 0% merge rate."""
        df = pl.DataFrame({"col": [None, None, None]}, schema={"col": pl.Float64})
        rate = _calculate_merge_rate(df, "col")
        assert rate == 0.0

    def test_missing_column(self):
        """Missing column returns 0%."""
        df = pl.DataFrame({"other_col": [1.0]})
        rate = _calculate_merge_rate(df, "nonexistent")
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Merge pipeline with mocked loaders
# ---------------------------------------------------------------------------


class TestMergeSpatialData:
    """Tests for the complete merge pipeline."""

    def test_row_count_preservation(
        self, enaho_df, admin_result, census_result, nightlights_result
    ):
        """LEFT JOIN preserves ENAHO row count."""
        with (
            patch("data.merge.load_admin_dropout_rates", return_value=admin_result),
            patch("data.merge.load_census_2017", return_value=census_result),
            patch(
                "data.merge.load_viirs_nightlights",
                return_value=nightlights_result,
            ),
        ):
            result = merge_spatial_data(enaho_df)
            assert result.initial_rows == 5
            assert result.final_rows == 5
            assert result.initial_rows == result.final_rows

    def test_all_sources_merged(
        self, enaho_df, admin_result, census_result, nightlights_result
    ):
        """All three data sources are merged into the output."""
        with (
            patch("data.merge.load_admin_dropout_rates", return_value=admin_result),
            patch("data.merge.load_census_2017", return_value=census_result),
            patch(
                "data.merge.load_viirs_nightlights",
                return_value=nightlights_result,
            ),
        ):
            result = merge_spatial_data(enaho_df)
            assert "admin_primaria_rate" in result.df.columns
            assert "census_poverty_rate" in result.df.columns
            assert "nightlights_mean_radiance" in result.df.columns

    def test_merge_rates_calculated(
        self, enaho_df, admin_result, census_result, nightlights_result
    ):
        """Merge rates are calculated for each source."""
        with (
            patch("data.merge.load_admin_dropout_rates", return_value=admin_result),
            patch("data.merge.load_census_2017", return_value=census_result),
            patch(
                "data.merge.load_viirs_nightlights",
                return_value=nightlights_result,
            ),
        ):
            result = merge_spatial_data(enaho_df)
            assert "admin" in result.merge_rates
            assert "census" in result.merge_rates
            assert "nightlights" in result.merge_rates
            # Admin and census should be 100% (all UBIGEOs match)
            assert result.merge_rates["admin"] == 1.0
            assert result.merge_rates["census"] == 1.0
            # Nightlights is missing 150102 -> 1 row unmatched out of 5
            assert result.merge_rates["nightlights"] == pytest.approx(4 / 5, abs=0.01)

    def test_missing_ubigeo_raises(self):
        """DataFrame without UBIGEO raises ValueError."""
        df = pl.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="UBIGEO"):
            merge_spatial_data(df)


# ---------------------------------------------------------------------------
# Null reporting
# ---------------------------------------------------------------------------


class TestNullReporting:
    """Tests for identification of >10% null columns."""

    def test_high_null_columns_reported(
        self, enaho_df, admin_result, census_result, nightlights_result
    ):
        """Columns with >10% nulls are in the null report."""
        with (
            patch("data.merge.load_admin_dropout_rates", return_value=admin_result),
            patch("data.merge.load_census_2017", return_value=census_result),
            patch(
                "data.merge.load_viirs_nightlights",
                return_value=nightlights_result,
            ),
        ):
            result = merge_spatial_data(enaho_df)
            # nightlights is missing 150102, so 1/5 = 20% nulls
            assert "nightlights_mean_radiance" in result.null_report
            assert result.null_report["nightlights_mean_radiance"] == pytest.approx(
                0.2, abs=0.01
            )


# ---------------------------------------------------------------------------
# Validate merge pipeline
# ---------------------------------------------------------------------------


class TestValidateMergePipeline:
    """Tests for the validate_merge_pipeline function."""

    def test_rows_preserved_check(self, enaho_df):
        """Validates row count preservation."""
        merged = enaho_df.with_columns(pl.lit(1.0).alias("admin_primaria_rate"))
        result = validate_merge_pipeline(
            enaho_df, merged, ["admin_primaria_rate"], [], []
        )
        assert result["rows_preserved"] is True

    def test_rows_not_preserved(self, enaho_df):
        """Detects row count changes."""
        smaller = enaho_df.head(3)
        result = validate_merge_pipeline(
            enaho_df, smaller, [], [], []
        )
        assert result["rows_preserved"] is False

    def test_duplicate_check(self, enaho_df):
        """Detects duplicate rows."""
        # enaho_df has 2 rows for 010101 and 2 for 150101 (distinct by P208A)
        result = validate_merge_pipeline(enaho_df, enaho_df, [], [], [])
        # Rows are distinct since P208A etc. differ
        assert result["has_duplicates"] is False
