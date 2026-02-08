"""Spatial data merge pipeline.

Implements sequential LEFT JOINs to enrich ENAHO microdata with
district-level administrative dropout rates, Census 2017 indicators,
and VIIRS nighttime lights data.

All merges use LEFT JOIN on UBIGEO to preserve the original ENAHO
row count. Each merge step is validated independently for row count
preservation, merge rate, and data quality.

Usage::

    from data.enaho import load_all_years
    from data.merge import merge_spatial_data

    enaho_df = load_all_years().df
    result = merge_spatial_data(enaho_df)
    print(result.merge_rates)
    print(result.null_report)
    result.df.write_parquet("data/processed/full_dataset.parquet")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from data.admin import load_admin_dropout_rates
from data.census import load_census_2017
from data.nightlights import load_viirs_nightlights

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Container for the spatial merge pipeline output.

    Attributes
    ----------
    df : pl.DataFrame
        Final merged DataFrame with all spatial enrichments.
    initial_rows : int
        Row count of the input ENAHO DataFrame.
    final_rows : int
        Row count after all merges (should equal ``initial_rows``).
    merge_rates : dict[str, float]
        Coverage rates for each data source (0.0 to 1.0).
    null_report : dict[str, float]
        Null rates for new columns with >10% nulls.
    warnings : list[str]
        Non-fatal issues encountered during merging.
    """

    df: pl.DataFrame
    initial_rows: int = 0
    final_rows: int = 0
    merge_rates: dict[str, float] = field(default_factory=dict)
    null_report: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _calculate_merge_rate(
    df: pl.DataFrame,
    indicator_col: str,
) -> float:
    """Calculate the fraction of rows where an indicator column is non-null.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame after a LEFT JOIN.
    indicator_col : str
        Column from the right side of the join to check for nulls.

    Returns
    -------
    float
        Merge rate (0.0 to 1.0).
    """
    if indicator_col not in df.columns:
        return 0.0
    non_null = df.height - df[indicator_col].null_count()
    return non_null / df.height if df.height > 0 else 0.0


def validate_merge_pipeline(
    original_df: pl.DataFrame,
    merged_df: pl.DataFrame,
    admin_cols: list[str],
    census_cols: list[str],
    nightlights_cols: list[str],
) -> dict:
    """Validate the complete merge pipeline output.

    Parameters
    ----------
    original_df : pl.DataFrame
        Original ENAHO DataFrame before merges.
    merged_df : pl.DataFrame
        Final merged DataFrame.
    admin_cols : list[str]
        Column names added by admin merge.
    census_cols : list[str]
        Column names added by census merge.
    nightlights_cols : list[str]
        Column names added by nightlights merge.

    Returns
    -------
    dict
        Validation results including row counts, merge rates, null report,
        and duplicate check.
    """
    validation = {}

    # Row count check
    validation["initial_rows"] = original_df.height
    validation["final_rows"] = merged_df.height
    validation["rows_preserved"] = original_df.height == merged_df.height

    # Merge rates for each source
    merge_rates = {}
    if admin_cols:
        merge_rates["admin"] = _calculate_merge_rate(merged_df, admin_cols[0])
    if census_cols:
        merge_rates["census"] = _calculate_merge_rate(merged_df, census_cols[0])
    if nightlights_cols:
        merge_rates["nightlights"] = _calculate_merge_rate(
            merged_df, nightlights_cols[0]
        )
    validation["merge_rates"] = merge_rates

    # Null report for columns with >10% nulls
    null_report = {}
    new_cols = admin_cols + census_cols + nightlights_cols
    for col in new_cols:
        if col in merged_df.columns:
            null_rate = merged_df[col].null_count() / merged_df.height
            if null_rate > 0.10:
                null_report[col] = null_rate
    validation["null_report"] = null_report

    # Duplicate check
    validation["has_duplicates"] = merged_df.is_duplicated().any()

    return validation


def merge_spatial_data(enaho_df: pl.DataFrame) -> MergeResult:
    """Merge ENAHO microdata with spatial and supplementary data sources.

    Performs sequential LEFT JOINs on UBIGEO to enrich ENAHO data with:

    1. Administrative dropout rates (primaria and secundaria)
    2. Census 2017 district indicators (poverty, indigenous language, etc.)
    3. VIIRS nighttime lights (economic proxy)

    Each merge preserves the original ENAHO row count via LEFT JOIN.
    Intermediate validation ensures no data loss or unexpected row growth.

    Parameters
    ----------
    enaho_df : pl.DataFrame
        ENAHO microdata from ``load_all_years().df``. Must contain a
        ``UBIGEO`` column with 6-character zero-padded district codes.

    Returns
    -------
    MergeResult
        Enriched DataFrame with comprehensive merge statistics.

    Raises
    ------
    ValueError
        If UBIGEO column is missing or row count changes during any merge.
    """
    warnings: list[str] = []
    initial_rows = enaho_df.height
    admin_cols: list[str] = []
    census_cols: list[str] = []
    nightlights_cols: list[str] = []

    if "UBIGEO" not in enaho_df.columns:
        raise ValueError("ENAHO DataFrame must contain 'UBIGEO' column")

    logger.info("Starting spatial merge pipeline with %d ENAHO rows", initial_rows)

    result_df = enaho_df

    # --- Step 1: Admin dropout rates ---
    try:
        admin_result = load_admin_dropout_rates()
        admin_df = admin_result.df

        if admin_df.height > 0:
            pre_join = result_df.height
            result_df = result_df.join(
                admin_df,
                on="UBIGEO",
                how="left",
                validate="m:1",
                coalesce=True,
            )

            if result_df.height != pre_join:
                raise ValueError(
                    f"Admin merge changed row count: {pre_join} -> {result_df.height}"
                )

            admin_cols = [c for c in admin_df.columns if c != "UBIGEO"]
            admin_rate = _calculate_merge_rate(result_df, admin_cols[0])
            logger.info(
                "Admin merge: %d admin districts, merge rate=%.2f%%",
                admin_df.height,
                admin_rate * 100,
            )
            warnings.extend(admin_result.warnings)
        else:
            warnings.append("Admin data empty -- skipped merge")
            logger.warning("Admin data empty, skipping merge step")

    except FileNotFoundError as e:
        warnings.append(f"Admin data unavailable: {e}")
        logger.warning("Admin data unavailable: %s", e)

    # --- Step 2: Census 2017 indicators ---
    try:
        census_result = load_census_2017()
        census_df = census_result.df

        if census_df.height > 0:
            pre_join = result_df.height
            result_df = result_df.join(
                census_df,
                on="UBIGEO",
                how="left",
                validate="m:1",
                coalesce=True,
            )

            if result_df.height != pre_join:
                raise ValueError(
                    f"Census merge changed row count: {pre_join} -> {result_df.height}"
                )

            census_cols = [c for c in census_df.columns if c != "UBIGEO"]
            census_rate = _calculate_merge_rate(result_df, census_cols[0])
            logger.info(
                "Census merge: %d census districts, merge rate=%.2f%%",
                census_df.height,
                census_rate * 100,
            )
            warnings.extend(census_result.warnings)
        else:
            warnings.append("Census data empty -- skipped merge")
            logger.warning("Census data empty, skipping merge step")

    except Exception as e:
        warnings.append(f"Census merge failed: {e}")
        logger.warning("Census merge failed: %s", e)

    # --- Step 3: VIIRS nightlights ---
    try:
        nightlights_result = load_viirs_nightlights()
        nightlights_df = nightlights_result.df

        if nightlights_df.height > 0:
            pre_join = result_df.height
            result_df = result_df.join(
                nightlights_df,
                on="UBIGEO",
                how="left",
                validate="m:1",
                coalesce=True,
            )

            if result_df.height != pre_join:
                raise ValueError(
                    f"Nightlights merge changed row count: "
                    f"{pre_join} -> {result_df.height}"
                )

            nightlights_cols = [c for c in nightlights_df.columns if c != "UBIGEO"]
            nl_rate = _calculate_merge_rate(result_df, nightlights_cols[0])
            logger.info(
                "Nightlights merge: %d NL districts, merge rate=%.2f%%",
                nightlights_df.height,
                nl_rate * 100,
            )
            warnings.extend(nightlights_result.warnings)
        else:
            warnings.append("Nightlights data empty -- skipped merge")
            logger.warning("Nightlights data empty, skipping merge step")

    except Exception as e:
        warnings.append(f"Nightlights merge failed: {e}")
        logger.warning("Nightlights merge failed: %s", e)

    # --- Validation ---
    validation = validate_merge_pipeline(
        enaho_df, result_df, admin_cols, census_cols, nightlights_cols
    )

    final_rows = result_df.height

    if not validation["rows_preserved"]:
        raise ValueError(
            f"Row count changed during merge pipeline: "
            f"{initial_rows} -> {final_rows}"
        )

    logger.info(
        "Merge pipeline complete: %d rows, merge rates=%s, "
        "columns with >10%% nulls=%d",
        final_rows,
        {k: f"{v:.2%}" for k, v in validation["merge_rates"].items()},
        len(validation["null_report"]),
    )

    return MergeResult(
        df=result_df,
        initial_rows=initial_rows,
        final_rows=final_rows,
        merge_rates=validation["merge_rates"],
        null_report=validation["null_report"],
        warnings=warnings,
    )
