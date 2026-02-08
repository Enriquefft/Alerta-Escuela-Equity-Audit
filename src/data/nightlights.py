"""VIIRS nighttime lights district-level loader.

Reads pre-aggregated VIIRS (Visible Infrared Imaging Radiometer Suite)
nighttime lights data at district level as an economic proxy.  Higher
nightlight intensity correlates with economic activity, urbanization, and
access to infrastructure.

If the nightlights data file is not available, returns a placeholder result
with an empty DataFrame and appropriate warnings.

Usage::

    from data.nightlights import load_viirs_nightlights

    result = load_viirs_nightlights()
    print(result.districts_count, result.coverage_rate, result.stats)
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
_NIGHTLIGHTS_FILE = "viirs_districts.csv"

# Expected number of districts in Peru (~1,839 with nightlight coverage)
_EXPECTED_DISTRICTS = 1839

# Column name for radiance intensity in the CSV
_RADIANCE_COL = "mean_radiance"


@dataclass
class NightlightsResult:
    """Container for loaded VIIRS nighttime lights data.

    Attributes
    ----------
    df : pl.DataFrame
        District-level DataFrame with UBIGEO and nightlights_mean_radiance.
    districts_count : int
        Number of districts with nightlight data.
    coverage_rate : float
        Fraction of expected districts covered (0.0 to 1.0).
    stats : dict[str, float]
        Summary statistics (mean, median, min, max) for radiance values.
    warnings : list[str]
        Non-fatal issues encountered during loading.
    """

    df: pl.DataFrame
    districts_count: int = 0
    coverage_rate: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_viirs_nightlights() -> NightlightsResult:
    """Load VIIRS district-level nighttime light intensity.

    Reads the nightlights CSV from
    ``data/raw/nightlights/viirs_districts.csv``, validates UBIGEO
    formatting and non-negative radiance values, and computes summary
    statistics.

    If the file does not exist, returns a ``NightlightsResult`` with an
    empty DataFrame and a warning message. This allows the merge pipeline
    to proceed gracefully when nightlights data is not yet available.

    Returns
    -------
    NightlightsResult
        Nightlights district data with coverage and statistics.

    Raises
    ------
    ValueError
        If UBIGEO validation fails, duplicate UBIGEO found, or negative
        radiance values detected.
    """
    warnings: list[str] = []

    root = find_project_root()
    nightlights_path = root / "data" / "raw" / "nightlights" / _NIGHTLIGHTS_FILE

    if not nightlights_path.exists():
        logger.warning("VIIRS nightlights data not found at %s", nightlights_path)
        warnings.append(
            f"VIIRS nightlights file not found at {nightlights_path}. "
            "Download pre-aggregated district data from Google Earth Engine "
            "or Jiaxiong Yao's research site."
        )
        empty_df = pl.DataFrame(
            schema={
                "UBIGEO": pl.Utf8,
                "nightlights_mean_radiance": pl.Float64,
            }
        )
        return NightlightsResult(
            df=empty_df,
            districts_count=0,
            coverage_rate=0.0,
            stats={"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            warnings=warnings,
        )

    logger.info("Loading VIIRS nightlights: %s", nightlights_path.name)

    df = pl.read_csv(
        nightlights_path,
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
            f"Nightlights: {bad_ubigeo.height} UBIGEO values have length != 6. "
            f"Examples: {examples}"
        )

    # Validate no duplicate UBIGEO
    n_unique = df["UBIGEO"].n_unique()
    if n_unique != df.height:
        n_dupes = df.height - n_unique
        raise ValueError(
            f"Nightlights: {n_dupes} duplicate UBIGEO values found. "
            f"Expected {df.height} unique, got {n_unique}."
        )

    # Validate radiance column exists
    if _RADIANCE_COL not in df.columns:
        raise ValueError(
            f"Nightlights: expected column '{_RADIANCE_COL}' not found. "
            f"Available columns: {df.columns}"
        )

    # Cast radiance to float
    df = df.with_columns(pl.col(_RADIANCE_COL).cast(pl.Float64))

    # Validate no negative values
    negative_count = df.filter(pl.col(_RADIANCE_COL) < 0).height
    if negative_count > 0:
        raise ValueError(
            f"Nightlights: {negative_count} rows have negative radiance values. "
            f"Min value: {df[_RADIANCE_COL].min()}"
        )

    # Compute statistics
    stats = {
        "mean": float(df[_RADIANCE_COL].mean() or 0.0),
        "median": float(df[_RADIANCE_COL].median() or 0.0),
        "min": float(df[_RADIANCE_COL].min() or 0.0),
        "max": float(df[_RADIANCE_COL].max() or 0.0),
    }

    # Calculate coverage rate
    districts_count = df.height
    coverage_rate = districts_count / _EXPECTED_DISTRICTS

    if coverage_rate < 0.85:
        warnings.append(
            f"Nightlights coverage below 85%: {coverage_rate:.2%} "
            f"({districts_count}/{_EXPECTED_DISTRICTS} districts)"
        )

    # Rename column with prefix
    df = df.select([
        pl.col("UBIGEO"),
        pl.col(_RADIANCE_COL).alias("nightlights_mean_radiance"),
    ])

    logger.info(
        "VIIRS nightlights: %d districts, coverage=%.2f%%, mean=%.2f, median=%.2f",
        districts_count,
        coverage_rate * 100,
        stats["mean"],
        stats["median"],
    )

    if warnings:
        for w in warnings:
            logger.warning("Nightlights: %s", w)

    return NightlightsResult(
        df=df,
        districts_count=districts_count,
        coverage_rate=coverage_rate,
        stats=stats,
        warnings=warnings,
    )
