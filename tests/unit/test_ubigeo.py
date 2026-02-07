"""Unit tests for pad_ubigeo and sniff_delimiter utilities."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from utils import pad_ubigeo, sniff_delimiter


# ---------------------------------------------------------------------------
# pad_ubigeo tests
# ---------------------------------------------------------------------------


class TestPadUbigeo:
    """Tests for the pad_ubigeo polars expression helper."""

    def test_pad_ubigeo_short_numeric(self):
        """Short numeric string '1234' is zero-padded to '001234'."""
        df = pl.DataFrame({"ubigeo": ["1234"]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        assert result["ubigeo"][0] == "001234"

    def test_pad_ubigeo_already_six(self):
        """A 6-character string '010101' is unchanged."""
        df = pl.DataFrame({"ubigeo": ["010101"]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        assert result["ubigeo"][0] == "010101"

    def test_pad_ubigeo_with_whitespace(self):
        """Whitespace is stripped before padding: ' 1234 ' -> '001234'."""
        df = pl.DataFrame({"ubigeo": [" 1234 "]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        assert result["ubigeo"][0] == "001234"

    def test_pad_ubigeo_from_integer(self):
        """Integer column cast to Utf8 then padded: 1234 -> '001234'."""
        df = pl.DataFrame({"ubigeo": [1234]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        assert result["ubigeo"][0] == "001234"

    def test_pad_ubigeo_preserves_leading_zeros(self):
        """String '010101' keeps its leading zero after padding."""
        df = pl.DataFrame({"ubigeo": ["010101"]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        assert result["ubigeo"][0] == "010101"
        assert len(result["ubigeo"][0]) == 6

    def test_pad_ubigeo_multiple_rows(self):
        """Batch of mixed values all become 6 characters."""
        df = pl.DataFrame({"ubigeo": ["1", "12", "123", "1234", "12345", "123456"]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        for val in result["ubigeo"].to_list():
            assert len(val) == 6

    def test_pad_ubigeo_five_digit(self):
        """Five-digit UBIGEO '10101' is padded to '010101'."""
        df = pl.DataFrame({"ubigeo": ["10101"]})
        result = df.with_columns(pad_ubigeo(pl.col("ubigeo")).alias("ubigeo"))
        assert result["ubigeo"][0] == "010101"


# ---------------------------------------------------------------------------
# sniff_delimiter tests
# ---------------------------------------------------------------------------


class TestSniffDelimiter:
    """Tests for CSV delimiter detection."""

    def test_sniff_delimiter_comma(self, tmp_path: Path):
        """Detects comma delimiter in a standard CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="latin-1")
        assert sniff_delimiter(csv_file) == ","

    def test_sniff_delimiter_pipe(self, tmp_path: Path):
        """Detects pipe delimiter (used by pre-2020 ENAHO files)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a|b|c\n1|2|3\n4|5|6\n", encoding="latin-1")
        assert sniff_delimiter(csv_file) == "|"

    def test_sniff_delimiter_tab(self, tmp_path: Path):
        """Detects tab delimiter."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a\tb\tc\n1\t2\t3\n", encoding="latin-1")
        assert sniff_delimiter(csv_file) == "\t"

    def test_sniff_delimiter_semicolon(self, tmp_path: Path):
        """Detects semicolon delimiter."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="latin-1")
        assert sniff_delimiter(csv_file) == ";"
