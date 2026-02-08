"""Administrative dropout rate loader for district-level MINEDU data.

Reads primaria and secundaria interannual dropout rate CSVs from
data/raw/admin/, validates UBIGEO formatting, and merges into a single
district-level DataFrame with both rates.

The admin data comes from MINEDU's ESCALE/datosabiertos.gob.pe portal and
contains district-level dropout rates for EBR (Educacion Basica Regular).

Usage::

    from data.admin import load_admin_dropout_rates

    result = load_admin_dropout_rates()
    print(result.districts_count, result.primaria_rate, result.secundaria_rate)
    print(result.df.head())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from utils import find_project_root, pad_ubigeo

logger = logging.getLogger(__name__)

# Expected file names in data/raw/admin/
_PRIMARIA_FILE = "primaria_2023.csv"
_SECUNDARIA_FILE = "secundaria_2023.csv"

# Column name for dropout rate in admin CSVs
_RATE_COL = "tasa_desercion"


@dataclass
class AdminResult:
    """Container for loaded administrative dropout rate data.

    Attributes
    ----------
    df : pl.DataFrame
        District-level DataFrame with UBIGEO, admin_primaria_rate,
        admin_secundaria_rate columns.
    primaria_rate : float
        Mean primaria dropout rate across all districts.
    secundaria_rate : float
        Mean secundaria dropout rate across all districts.
    districts_count : int
        Number of unique districts in the merged result.
    warnings : list[str]
        Non-fatal issues encountered during loading.
    """

    df: pl.DataFrame
    primaria_rate: float = 0.0
    secundaria_rate: float = 0.0
    districts_count: int = 0
    warnings: list[str] = field(default_factory=list)


def _load_admin_csv(filepath: Path, level: str) -> pl.DataFrame:
    """Load and validate a single admin dropout rate CSV.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.
    level : str
        Education level label (``"primaria"`` or ``"secundaria"``).

    Returns
    -------
    pl.DataFrame
        DataFrame with ``UBIGEO`` (6-char string) and
        ``admin_{level}_rate`` (Float64) columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If UBIGEO validation fails or rates are out of range.
    """
    logger.info("Loading admin %s data: %s", level, filepath.name)

    df = pl.read_csv(
        filepath,
        schema_overrides={"UBIGEO": pl.Utf8},
        infer_schema_length=10_000,
    )

    # Zero-pad UBIGEO
    df = df.with_columns(pad_ubigeo(pl.col("UBIGEO")).alias("UBIGEO"))

    # Validate UBIGEO length
    bad_ubigeo = df.filter(pl.col("UBIGEO").str.len_chars() != 6)
    if bad_ubigeo.height > 0:
        examples = bad_ubigeo["UBIGEO"].head(5).to_list()
        raise ValueError(
            f"Admin {level}: {bad_ubigeo.height} UBIGEO values have length != 6. "
            f"Examples: {examples}"
        )

    # Validate no duplicate UBIGEO
    n_unique = df["UBIGEO"].n_unique()
    if n_unique != df.height:
        n_dupes = df.height - n_unique
        raise ValueError(
            f"Admin {level}: {n_dupes} duplicate UBIGEO values found. "
            f"Expected {df.height} unique, got {n_unique}."
        )

    # Validate rate column exists and is in valid range
    if _RATE_COL not in df.columns:
        raise ValueError(
            f"Admin {level}: expected column '{_RATE_COL}' not found. "
            f"Available columns: {df.columns}"
        )

    # Cast rate to float if needed
    df = df.with_columns(pl.col(_RATE_COL).cast(pl.Float64))

    # Validate rates within 0-100%
    out_of_range = df.filter(
        (pl.col(_RATE_COL) < 0) | (pl.col(_RATE_COL) > 100)
    )
    if out_of_range.height > 0:
        raise ValueError(
            f"Admin {level}: {out_of_range.height} rows have rates outside [0, 100]. "
            f"Min: {df[_RATE_COL].min()}, Max: {df[_RATE_COL].max()}"
        )

    # Rename rate column to level-specific name
    rate_col_name = f"admin_{level}_rate"
    df = df.select([
        pl.col("UBIGEO"),
        pl.col(_RATE_COL).alias(rate_col_name),
    ])

    logger.info(
        "Admin %s: %d districts, mean rate %.4f%%",
        level,
        df.height,
        df[rate_col_name].mean(),
    )

    return df


def load_admin_dropout_rates() -> AdminResult:
    """Load and merge primaria and secundaria district dropout rates.

    Reads CSV files from ``data/raw/admin/``, validates UBIGEO formatting,
    and performs a full outer join to combine both education levels into a
    single district-level DataFrame.

    Returns
    -------
    AdminResult
        Merged admin data with primaria and secundaria dropout rates.

    Raises
    ------
    FileNotFoundError
        If either admin CSV file is missing.
    ValueError
        If UBIGEO or rate validation fails.
    """
    warnings: list[str] = []

    root = find_project_root()
    admin_dir = root / "data" / "raw" / "admin"

    primaria_path = admin_dir / _PRIMARIA_FILE
    secundaria_path = admin_dir / _SECUNDARIA_FILE

    # Check files exist
    if not primaria_path.exists():
        raise FileNotFoundError(
            f"Admin primaria data not found at {primaria_path}. "
            "Run 'uv run python src/data/download.py' to download admin data."
        )
    if not secundaria_path.exists():
        raise FileNotFoundError(
            f"Admin secundaria data not found at {secundaria_path}. "
            "Run 'uv run python src/data/download.py' to download admin data."
        )

    # Load both levels
    primaria_df = _load_admin_csv(primaria_path, "primaria")
    secundaria_df = _load_admin_csv(secundaria_path, "secundaria")

    # Merge on UBIGEO (outer join to keep all districts from both sources)
    merged = primaria_df.join(
        secundaria_df,
        on="UBIGEO",
        how="full",
        coalesce=True,
    )

    # Check for districts in one but not the other
    primaria_only = merged.filter(pl.col("admin_secundaria_rate").is_null()).height
    secundaria_only = merged.filter(pl.col("admin_primaria_rate").is_null()).height
    if primaria_only > 0:
        warnings.append(
            f"{primaria_only} districts have primaria data but no secundaria"
        )
    if secundaria_only > 0:
        warnings.append(
            f"{secundaria_only} districts have secundaria data but no primaria"
        )

    # Compute statistics
    primaria_mean = merged["admin_primaria_rate"].drop_nulls().mean() or 0.0
    secundaria_mean = merged["admin_secundaria_rate"].drop_nulls().mean() or 0.0
    districts_count = merged["UBIGEO"].n_unique()

    logger.info(
        "Admin merged: %d districts, primaria mean=%.4f%%, secundaria mean=%.4f%%",
        districts_count,
        primaria_mean,
        secundaria_mean,
    )

    if warnings:
        for w in warnings:
            logger.warning("Admin: %s", w)

    return AdminResult(
        df=merged,
        primaria_rate=primaria_mean,
        secundaria_rate=secundaria_mean,
        districts_count=districts_count,
        warnings=warnings,
    )
