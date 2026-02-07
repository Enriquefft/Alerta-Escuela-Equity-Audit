"""Shared utility functions for the Alerta Escuela Equity Audit.

Provides project root discovery, UBIGEO zero-padding, and CSV delimiter
detection used across data loaders and tests.
"""

import csv
from collections import Counter
from pathlib import Path

import polars as pl


def find_project_root() -> Path:
    """Walk up from the caller's file to find the project root.

    The project root is identified as the first ancestor directory that
    contains a ``pyproject.toml`` file.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Raises
    ------
    RuntimeError
        If no ``pyproject.toml`` is found before reaching the filesystem root.
    """
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "pyproject.toml").exists():
            return path
        path = path.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def pad_ubigeo(col: pl.Expr) -> pl.Expr:
    """Return a polars expression that zero-pads UBIGEO codes to 6 characters.

    Casts to Utf8, strips leading/trailing whitespace, then left-pads with
    zeros so that every value is exactly 6 characters.  This prevents
    leading-zero loss when UBIGEO has been parsed as an integer.

    Parameters
    ----------
    col : pl.Expr
        A polars column expression containing UBIGEO values (string or numeric).

    Returns
    -------
    pl.Expr
        Expression producing 6-character zero-padded UBIGEO strings.
    """
    return col.cast(pl.Utf8).str.strip_chars().str.pad_start(6, "0")


def sniff_delimiter(filepath: Path, n_bytes: int = 8192) -> str:
    """Detect the delimiter used in a CSV file.

    Reads the first *n_bytes* of the file using ``latin-1`` encoding (ENAHO
    files use Latin-1) and attempts detection via :class:`csv.Sniffer`.  If
    the sniffer fails, falls back to counting candidate delimiter frequencies
    in the first line.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.
    n_bytes : int, optional
        Number of bytes to read for sniffing (default 8192).

    Returns
    -------
    str
        The detected single-character delimiter (e.g. ``","`` or ``"|"``).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If no delimiter can be detected.
    """
    with open(filepath, encoding="latin-1") as fh:
        sample = fh.read(n_bytes)

    # Primary: csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        return dialect.delimiter
    except csv.Error:
        pass

    # Fallback: count candidate delimiters in the first line
    first_line = sample.split("\n", maxsplit=1)[0]
    counts = Counter(ch for ch in first_line if ch in ",|\t;")
    if counts:
        return counts.most_common(1)[0][0]

    raise ValueError(f"Could not detect delimiter in {filepath}")
