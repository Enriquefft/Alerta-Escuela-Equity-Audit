"""Census 2017 district-level indicators loader.

Reads Census 2017 district-level data from data/raw/census/ to provide
contextual variables for the equity audit: poverty rates, indigenous
language prevalence, access to services, and literacy rates.

If the census data file is not available, returns a placeholder result
with an empty DataFrame and appropriate warnings.

Usage::

    from data.census import load_census_2017

    result = load_census_2017()
    print(result.districts_count, result.coverage_stats)
    print(result.df.head())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from utils import find_project_root, pad_ubigeo

logger = logging.getLogger(__name__)

# Expected file location
_CENSUS_FILE = "census_2017_districts.csv"

# Key indicator columns expected in census data
_INDICATOR_COLUMNS = [
    "poverty_rate",
    "indigenous_lang_pct",
    "water_access_pct",
    "electricity_pct",
    "literacy_rate",
]


@dataclass
class CensusResult:
    """Container for loaded Census 2017 district-level data.

    Attributes
    ----------
    df : pl.DataFrame
        District-level DataFrame with UBIGEO and indicator columns,
        prefixed with ``census_``.
    districts_count : int
        Number of unique districts loaded.
    coverage_stats : dict[str, float]
        Non-null rates for each indicator column (0.0 to 1.0).
    warnings : list[str]
        Non-fatal issues encountered during loading.
    """

    df: pl.DataFrame
    districts_count: int = 0
    coverage_stats: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_census_2017() -> CensusResult:
    """Load Census 2017 district-level indicators.

    Reads the census CSV from ``data/raw/census/census_2017_districts.csv``,
    validates UBIGEO formatting, and computes coverage statistics for each
    indicator variable.

    If the file does not exist, returns a ``CensusResult`` with an empty
    DataFrame and a warning message. This allows the merge pipeline to
    proceed gracefully when census data is not yet available.

    Returns
    -------
    CensusResult
        Census district data with coverage statistics.
    """
    warnings: list[str] = []

    root = find_project_root()
    census_path = root / "data" / "raw" / "census" / _CENSUS_FILE

    if not census_path.exists():
        logger.warning("Census 2017 data not found at %s", census_path)
        warnings.append(
            f"Census 2017 file not found at {census_path}. "
            "Download from INEI (https://censos2017.inei.gob.pe/redatam/) "
            "and place in data/raw/census/."
        )
        empty_df = pl.DataFrame(
            schema={"UBIGEO": pl.Utf8}
            | {f"census_{col}": pl.Float64 for col in _INDICATOR_COLUMNS}
        )
        return CensusResult(
            df=empty_df,
            districts_count=0,
            coverage_stats={col: 0.0 for col in _INDICATOR_COLUMNS},
            warnings=warnings,
        )

    logger.info("Loading Census 2017: %s", census_path.name)

    df = pl.read_csv(
        census_path,
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
            f"Census 2017: {bad_ubigeo.height} UBIGEO values have length != 6. "
            f"Examples: {examples}"
        )

    # Validate no duplicate UBIGEO
    n_unique = df["UBIGEO"].n_unique()
    if n_unique != df.height:
        n_dupes = df.height - n_unique
        raise ValueError(
            f"Census 2017: {n_dupes} duplicate UBIGEO values found. "
            f"Expected {df.height} unique, got {n_unique}."
        )

    # Check which indicator columns are present
    available_indicators = [col for col in _INDICATOR_COLUMNS if col in df.columns]
    missing_indicators = [col for col in _INDICATOR_COLUMNS if col not in df.columns]

    if missing_indicators:
        warnings.append(
            f"Missing census indicator columns: {missing_indicators}. "
            f"Available: {available_indicators}."
        )

    # Validate value ranges (0-100 for percentages)
    for col in available_indicators:
        df = df.with_columns(pl.col(col).cast(pl.Float64))
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min is not None and col_min < 0:
            warnings.append(
                f"Census column {col} has negative values (min={col_min:.2f})"
            )
        if col_max is not None and col_max > 100:
            warnings.append(
                f"Census column {col} has values > 100 (max={col_max:.2f})"
            )

    # Calculate coverage statistics (non-null rate for each indicator)
    coverage_stats = {}
    total_rows = df.height
    for col in available_indicators:
        non_null = total_rows - df[col].null_count()
        coverage_stats[col] = non_null / total_rows if total_rows > 0 else 0.0

    # Rename indicator columns with census_ prefix
    rename_map = {col: f"census_{col}" for col in available_indicators}
    select_cols = [pl.col("UBIGEO")] + [
        pl.col(col).alias(f"census_{col}") for col in available_indicators
    ]
    df = df.select(select_cols)

    districts_count = df["UBIGEO"].n_unique()

    logger.info(
        "Census 2017: %d districts, %d indicators, coverage: %s",
        districts_count,
        len(available_indicators),
        {k: f"{v:.2%}" for k, v in coverage_stats.items()},
    )

    if warnings:
        for w in warnings:
            logger.warning("Census: %s", w)

    return CensusResult(
        df=df,
        districts_count=districts_count,
        coverage_stats=coverage_stats,
        warnings=warnings,
    )
