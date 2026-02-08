#!/usr/bin/env python3
"""Build the full merged dataset for the Alerta Escuela Equity Audit.

Loads pooled ENAHO data across all years, merges with admin dropout
rates, Census 2017 indicators, and VIIRS nightlights, then saves
the result as a Parquet file for downstream feature engineering.

Usage::

    uv run python src/data/build_dataset.py

Output::

    data/processed/full_dataset.parquet
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

import polars as pl

from data.enaho import load_all_years
from data.merge import merge_spatial_data
from utils import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Build and save the full merged dataset."""
    root = find_project_root()
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full_dataset.parquet"

    logger.info("=" * 60)
    logger.info("Building full merged dataset")
    logger.info("=" * 60)

    # Step 1: Load pooled ENAHO data
    logger.info("Loading pooled ENAHO data (2018-2023)...")
    enaho_result = load_all_years()
    enaho_df = enaho_result.df
    logger.info("ENAHO: %d rows, %d columns", enaho_df.height, enaho_df.width)

    # Step 2: Run spatial merge pipeline
    logger.info("Running spatial merge pipeline...")
    merge_result = merge_spatial_data(enaho_df)
    final_df = merge_result.df

    # Step 3: Validate before saving
    logger.info("Validating merged dataset...")

    # Row count preservation
    assert merge_result.initial_rows == merge_result.final_rows, (
        f"Row count mismatch: {merge_result.initial_rows} != {merge_result.final_rows}"
    )

    # UBIGEO integrity
    bad_ubigeo = final_df.filter(pl.col("UBIGEO").str.len_chars() != 6)
    assert bad_ubigeo.height == 0, (
        f"{bad_ubigeo.height} UBIGEO values have length != 6"
    )

    # Critical columns present
    for col in ["UBIGEO", "P208A", "P303", "P306", "FACTOR07", "dropout", "year"]:
        assert col in final_df.columns, f"Missing critical column: {col}"

    # No critical nulls
    for col in ["UBIGEO", "P208A", "P303", "P306", "FACTOR07", "dropout"]:
        null_count = final_df[col].null_count()
        assert null_count == 0, f"Critical column {col} has {null_count} nulls"

    logger.info("Validation passed")

    # Step 4: Save to Parquet
    logger.info("Saving to %s...", output_path)
    final_df.write_parquet(output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved: %.1f MB", file_size_mb)

    # Step 5: Report summary
    logger.info("=" * 60)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("  Output: %s", output_path)
    logger.info("  Rows: %d", final_df.height)
    logger.info("  Columns: %d", final_df.width)
    logger.info("  File size: %.1f MB", file_size_mb)
    logger.info("  UBIGEO unique: %d", final_df["UBIGEO"].n_unique())
    logger.info("  Years: %s", sorted(final_df["year"].unique().to_list()))
    logger.info("  Merge rates:")
    for source, rate in merge_result.merge_rates.items():
        logger.info("    %s: %.2f%%", source, rate * 100)
    if merge_result.null_report:
        logger.info("  Columns with >10%% nulls:")
        for col, rate in merge_result.null_report.items():
            logger.info("    %s: %.2f%%", col, rate * 100)
    else:
        logger.info("  No columns with >10%% nulls")
    if merge_result.warnings:
        logger.info("  Warnings:")
        for w in merge_result.warnings:
            logger.info("    %s", w)
    logger.info("  Timestamp: %s", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
