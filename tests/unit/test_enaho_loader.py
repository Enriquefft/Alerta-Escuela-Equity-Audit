"""Unit tests for ENAHO loader using synthetic data.

Tests ENAHOResult dataclass, dropout construction, school-age filtering,
UBIGEO validation, and null detection without requiring real ENAHO CSVs.
"""

import polars as pl
import pytest

from data.enaho import ENAHOResult, _validate_critical_nulls, _validate_ubigeo_length


# ---------------------------------------------------------------------------
# ENAHOResult dataclass
# ---------------------------------------------------------------------------


class TestENAHOResult:
    """Tests for the ENAHOResult container."""

    def test_enahoresult_defaults(self):
        """ENAHOResult has sensible defaults for stats and warnings."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = ENAHOResult(df=df)
        assert isinstance(result.df, pl.DataFrame)
        assert result.stats == {}
        assert result.warnings == []

    def test_enahoresult_with_stats(self):
        """ENAHOResult stores provided stats and warnings."""
        df = pl.DataFrame({"x": [1]})
        stats = {"year": 2023, "total_rows": 100}
        warnings = ["test warning"]
        result = ENAHOResult(df=df, stats=stats, warnings=warnings)
        assert result.stats["year"] == 2023
        assert result.warnings == ["test warning"]


# ---------------------------------------------------------------------------
# Dropout target construction
# ---------------------------------------------------------------------------


class TestDropoutConstruction:
    """Tests for the dropout boolean column logic."""

    def _build_df_with_dropout(self, p303_vals: list, p306_vals: list) -> pl.DataFrame:
        """Helper: build a DataFrame and apply dropout logic inline."""
        df = pl.DataFrame({
            "P303": p303_vals,
            "P306": p306_vals,
        }).with_columns(
            pl.col("P303").cast(pl.Int64),
            pl.col("P306").cast(pl.Int64),
        )
        return df.with_columns(
            ((pl.col("P303") == 1) & (pl.col("P306") == 2)).alias("dropout")
        )

    def test_dropout_true_enrolled_then_dropped(self):
        """P303=1 (enrolled last year) and P306=2 (not enrolled now) -> dropout=True."""
        df = self._build_df_with_dropout([1], [2])
        assert df["dropout"][0] is True

    def test_dropout_false_still_enrolled(self):
        """P303=1 (enrolled last year) and P306=1 (still enrolled) -> dropout=False."""
        df = self._build_df_with_dropout([1], [1])
        assert df["dropout"][0] is False

    def test_dropout_false_never_enrolled(self):
        """P303=2 (not enrolled last year) and P306=2 (not enrolled) -> dropout=False."""
        df = self._build_df_with_dropout([2], [2])
        assert df["dropout"][0] is False

    def test_dropout_false_newly_enrolled(self):
        """P303=2 (not enrolled last year) and P306=1 (now enrolled) -> dropout=False."""
        df = self._build_df_with_dropout([2], [1])
        assert df["dropout"][0] is False

    def test_dropout_batch(self):
        """Batch test with all four P303/P306 combinations."""
        df = self._build_df_with_dropout(
            [1, 1, 2, 2],
            [1, 2, 1, 2],
        )
        expected = [False, True, False, False]
        assert df["dropout"].to_list() == expected


# ---------------------------------------------------------------------------
# School-age filter
# ---------------------------------------------------------------------------


class TestSchoolAgeFilter:
    """Tests for the 6-17 age filter logic."""

    def test_filter_excludes_young(self):
        """Ages below 6 are excluded."""
        df = pl.DataFrame({"P208A": [3, 5, 6, 10, 17, 18, 25]})
        filtered = df.filter((pl.col("P208A") >= 6) & (pl.col("P208A") <= 17))
        assert filtered["P208A"].to_list() == [6, 10, 17]

    def test_filter_includes_boundary(self):
        """Boundary ages 6 and 17 are included."""
        df = pl.DataFrame({"P208A": [5, 6, 17, 18]})
        filtered = df.filter((pl.col("P208A") >= 6) & (pl.col("P208A") <= 17))
        assert 6 in filtered["P208A"].to_list()
        assert 17 in filtered["P208A"].to_list()
        assert 5 not in filtered["P208A"].to_list()
        assert 18 not in filtered["P208A"].to_list()


# ---------------------------------------------------------------------------
# UBIGEO validation
# ---------------------------------------------------------------------------


class TestUbigeoValidation:
    """Tests for UBIGEO length validation."""

    def test_valid_ubigeo_passes(self):
        """All 6-character UBIGEO values pass validation."""
        df = pl.DataFrame({"UBIGEO": ["010101", "150101", "999999"]})
        # Should not raise
        _validate_ubigeo_length(df)

    def test_short_ubigeo_raises(self):
        """UBIGEO values shorter than 6 characters raise ValueError."""
        df = pl.DataFrame({"UBIGEO": ["010101", "1234", "150101"]})
        with pytest.raises(ValueError, match="UBIGEO length validation failed"):
            _validate_ubigeo_length(df)

    def test_long_ubigeo_raises(self):
        """UBIGEO values longer than 6 characters raise ValueError."""
        df = pl.DataFrame({"UBIGEO": ["0101010", "150101"]})
        with pytest.raises(ValueError, match="UBIGEO length validation failed"):
            _validate_ubigeo_length(df)


# ---------------------------------------------------------------------------
# Critical column null detection
# ---------------------------------------------------------------------------


class TestCriticalNullDetection:
    """Tests for strict null validation on critical columns."""

    def test_no_nulls_passes(self):
        """All critical columns without nulls pass validation."""
        df = pl.DataFrame({
            "UBIGEO": ["010101", "150101"],
            "P208A": [10, 12],
            "P303": [1, 2],
            "P306": [1, 2],
            "FACTOR07": [100.0, 200.0],
        })
        # Should not raise
        _validate_critical_nulls(df)

    def test_null_factor07_raises(self):
        """Nulls in FACTOR07 raise ValueError."""
        df = pl.DataFrame({
            "UBIGEO": ["010101", "150101"],
            "P208A": [10, 12],
            "P303": [1, 2],
            "P306": [1, 2],
            "FACTOR07": [100.0, None],
        })
        with pytest.raises(ValueError, match="Critical columns have nulls"):
            _validate_critical_nulls(df)

    def test_null_ubigeo_raises(self):
        """Nulls in UBIGEO raise ValueError."""
        df = pl.DataFrame({
            "UBIGEO": ["010101", None],
            "P208A": [10, 12],
            "P303": [1, 2],
            "P306": [1, 2],
            "FACTOR07": [100.0, 200.0],
        })
        with pytest.raises(ValueError, match="Critical columns have nulls"):
            _validate_critical_nulls(df)

    def test_null_p208a_raises(self):
        """Nulls in P208A raise ValueError."""
        df = pl.DataFrame({
            "UBIGEO": ["010101", "150101"],
            "P208A": [10, None],
            "P303": [1, 2],
            "P306": [1, 2],
            "FACTOR07": [100.0, 200.0],
        })
        with pytest.raises(ValueError, match="Critical columns have nulls"):
            _validate_critical_nulls(df)

    def test_multiple_null_columns_reported(self):
        """Multiple columns with nulls are all reported in the error."""
        df = pl.DataFrame({
            "UBIGEO": [None, "150101"],
            "P208A": [10, None],
            "P303": [1, 2],
            "P306": [1, 2],
            "FACTOR07": [100.0, 200.0],
        })
        with pytest.raises(ValueError, match="UBIGEO"):
            _validate_critical_nulls(df)
