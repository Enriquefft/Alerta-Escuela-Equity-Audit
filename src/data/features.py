"""Feature engineering pipeline for the Alerta Escuela Equity Audit.

Transforms the merged ``full_dataset.parquet`` into a complete feature matrix
with all 19+ model features per spec Section 5.  Each model feature is stored
as a new lowercase column.  Supplementary raw ENAHO modules (200, 500, 700,
sumaria) are loaded on-the-fly to construct features not present in the
base dataset.

Usage::

    import polars as pl
    from data.features import build_features

    df = pl.read_parquet("data/processed/full_dataset.parquet")
    result = build_features(df)
    print(result.stats)
    result.df.write_parquet("data/processed/enaho_with_features.parquet")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from data.enaho import (
    _find_module_file,
    _read_data_file,
    _KEY_OVERRIDES,
    JOIN_KEYS,
)
from utils import find_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_FEATURES: list[str] = [
    "age",
    "is_secundaria_age",
    "es_mujer",
    "lang_castellano",
    "lang_quechua",
    "lang_aimara",
    "lang_other_indigenous",
    "lang_foreign",
    "rural",
    "is_sierra",
    "is_selva",
    "district_dropout_rate_admin_z",
    "nightlight_intensity_z",
    "poverty_index_z",
    "poverty_quintile",
    "es_peruano",
    "has_disability",
    "is_working",
    "juntos_participant",
    "log_income",
    "parent_education_years",
    "census_indigenous_lang_pct_z",
    "census_literacy_rate_z",
    "census_electricity_pct_z",
    "census_water_access_pct_z",
]

META_COLUMNS: list[str] = [
    "UBIGEO",
    "FACTOR07",
    "year",
    "dropout",
    "p300a_original",
    "p300a_harmonized",
    "region_natural",
    "department",
    "poverty_index",
    "nightlight_intensity",
    "district_dropout_rate_admin",
]

# Education level (P301A) to approximate years of schooling mapping.
# Source: INEI ENAHO codebook.
#  1 = Sin nivel (0 years)
#  2 = Educacion Inicial (0 years)
#  3 = Primaria incompleta (~3 years, midpoint)
#  4 = Primaria completa (6 years)
#  5 = Secundaria incompleta (~9 years, midpoint)
#  6 = Secundaria completa (11 years)
#  7 = Superior no universitaria incompleta (~12 years)
#  8 = Superior no universitaria completa (14 years)
#  9 = Superior universitaria incompleta (~14 years)
# 10 = Superior universitaria completa (16 years)
# 11 = Maestria/Doctorado (18 years)
# 12 = Basica especial (6 years, treated as primary)
_P301A_TO_YEARS: dict[int, int] = {
    1: 0,
    2: 0,
    3: 3,
    4: 6,
    5: 9,
    6: 11,
    7: 12,
    8: 14,
    9: 14,
    10: 16,
    11: 18,
    12: 6,
}

# Household-level join keys (no CODPERSO).
_HH_KEYS = ["CONGLOME", "VIVIENDA", "HOGAR"]

# District-level columns to z-score standardize.
_SPATIAL_COLS_FOR_ZSCORE: list[tuple[str, str]] = [
    ("district_dropout_rate_admin", "district_dropout_rate_admin_z"),
    ("nightlight_intensity", "nightlight_intensity_z"),
    ("poverty_index", "poverty_index_z"),
    ("census_indigenous_lang_pct", "census_indigenous_lang_pct_z"),
    ("census_literacy_rate", "census_literacy_rate_z"),
    ("census_electricity_pct", "census_electricity_pct_z"),
    ("census_water_access_pct", "census_water_access_pct_z"),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureResult:
    """Container for the feature engineering result.

    Attributes
    ----------
    df : pl.DataFrame
        Full DataFrame with original + engineered columns.
    stats : dict
        Feature counts, null rates, quintile balance, binary validation.
    warnings : list[str]
        Non-fatal issues (e.g. missing ESCALE data, imputed values).
    model_features : list[str]
        List of lowercase model feature column names.
    """

    df: pl.DataFrame
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    model_features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zscore(df: pl.DataFrame, src_col: str, dst_col: str, impute_null: bool = False) -> pl.DataFrame:
    """Add a z-score standardized column.

    Parameters
    ----------
    df : pl.DataFrame
    src_col : str
        Source column name.
    dst_col : str
        Destination column name (``_z`` suffix).
    impute_null : bool
        If True, fill null z-scores with 0.0 (mean of distribution).

    Returns
    -------
    pl.DataFrame
        DataFrame with added z-score column.
    """
    non_null = df[src_col].drop_nulls()
    mean_val = non_null.mean()
    std_val = non_null.std()

    if std_val is None or std_val == 0:
        logger.warning("Column %s has zero std; z-score set to 0.0", src_col)
        df = df.with_columns(pl.lit(0.0).alias(dst_col))
    else:
        expr = ((pl.col(src_col) - mean_val) / std_val).alias(dst_col)
        df = df.with_columns(expr)

    if impute_null:
        null_count = df[dst_col].null_count()
        if null_count > 0:
            logger.info(
                "Imputing %d null z-scores in %s with 0.0 (distribution mean)",
                null_count, dst_col,
            )
            df = df.with_columns(pl.col(dst_col).fill_null(0.0))

    return df


def _load_module_for_year(
    year_dir: Path,
    prefix: str,
    module: str,
    columns: list[str],
    key_cols: list[str],
    year: int,
) -> pl.DataFrame | None:
    """Load specific columns from an ENAHO module file.

    Returns None if the module file is not found (graceful degradation).
    """
    try:
        filepath = _find_module_file(year_dir, prefix, module)
    except FileNotFoundError:
        logger.warning("Module %s not found for year %d in %s", module, year, year_dir)
        return None

    logger.info("Loading Module %s [%d]: %s", module, year, filepath.name)
    df = _read_data_file(filepath, _KEY_OVERRIDES)

    # Select only needed columns (those present in the file)
    available = [c for c in key_cols + columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("Module %s [%d]: missing columns %s", module, year, missing)

    if not any(c in df.columns for c in columns):
        logger.warning("Module %s [%d]: none of the target columns found", module, year)
        return None

    df = df.select(available)
    return df


def _find_sumaria_file(year_dir: Path, year: int) -> Path | None:
    """Find the sumaria DTA file for a given year directory."""
    # Try common naming patterns
    candidates = [
        year_dir / f"sumaria-{year}.dta",
        year_dir / f"Sumaria-{year}.dta",
        year_dir / f"SUMARIA-{year}.dta",
        year_dir / f"sumaria-{year}-12g.dta",  # skip this variant, it's a different file
    ]

    for path in candidates[:3]:  # Only first 3 patterns (skip 12g)
        if path.exists():
            return path

    # Glob fallback
    matches = list(year_dir.glob("*umaria*.dta"))
    # Filter out 12g variants
    matches = [m for m in matches if "12g" not in m.name.lower()]
    if matches:
        return matches[0]

    return None


def _load_supplementary_features(years: list[int]) -> pl.DataFrame:
    """Load supplementary features from raw ENAHO modules for all years.

    Loads columns from Modules 200 (P209 birthplace, P210 disability, P203
    relationship), 500 (P501 employment), 700 (P710_04 JUNTOS), and Sumaria
    (INGHOG1D income).

    Returns a DataFrame keyed on (CONGLOME, VIVIENDA, HOGAR, CODPERSO, year)
    with supplementary feature columns.
    """
    root = find_project_root()
    all_person_frames: list[pl.DataFrame] = []

    for year in years:
        year_dir = root / "data" / "raw" / "enaho" / str(year)
        if not year_dir.exists():
            logger.warning("Year directory not found: %s", year_dir)
            continue

        person_key = JOIN_KEYS + ["year"]

        # --- Module 200: P209 (birthplace), P210 (disability), P203 (relationship) ---
        mod200 = _load_module_for_year(
            year_dir, "Enaho01", "200",
            columns=["P209", "P210", "P203", "P301A"],
            key_cols=JOIN_KEYS,
            year=year,
        )

        # --- Module 300: P301A for parent education (may already be loaded) ---
        # P301A is in module 300. Load it separately in case mod200 does not have it.
        mod300 = _load_module_for_year(
            year_dir, "Enaho01a", "300",
            columns=["P301A"],
            key_cols=JOIN_KEYS,
            year=year,
        )

        # --- Module 500: P501 (employment) ---
        mod500 = _load_module_for_year(
            year_dir, "Enaho01a", "500",
            columns=["P501"],
            key_cols=JOIN_KEYS,
            year=year,
        )

        # --- Module 700: P710_04 (JUNTOS participation) ---
        mod700 = _load_module_for_year(
            year_dir, "Enaho01", "700",
            columns=["P710_04", "P710_4"],
            key_cols=_HH_KEYS,
            year=year,
        )

        # --- Sumaria: INGHOG1D (household income) ---
        sumaria_path = _find_sumaria_file(year_dir, year)
        sumaria_df = None
        if sumaria_path is not None:
            logger.info("Loading Sumaria [%d]: %s", year, sumaria_path.name)
            sumaria_raw = _read_data_file(sumaria_path, _KEY_OVERRIDES)
            sum_cols = [c for c in _HH_KEYS + ["INGHOG1D"] if c in sumaria_raw.columns]
            if "INGHOG1D" in sumaria_raw.columns:
                sumaria_df = sumaria_raw.select(sum_cols)
            else:
                logger.warning("Sumaria [%d]: INGHOG1D column not found", year)
        else:
            logger.warning("Sumaria file not found for year %d", year)

        # --- Build person-level supplementary frame ---
        # Start with mod200 as the person-level base
        if mod200 is not None:
            person_df = mod200.with_columns(pl.lit(year).cast(pl.Int32).alias("year"))
        else:
            logger.warning("Module 200 not loaded for year %d; skipping supplementary", year)
            continue

        # Merge P301A from module 300 if not already in mod200
        if "P301A" not in person_df.columns and mod300 is not None:
            person_df = person_df.join(
                mod300.select(JOIN_KEYS + ["P301A"]),
                on=JOIN_KEYS,
                how="left",
                suffix="_m300",
            )

        # Merge P501 from module 500 (person-level)
        if mod500 is not None:
            person_df = person_df.join(
                mod500.select(JOIN_KEYS + ["P501"]),
                on=JOIN_KEYS,
                how="left",
                suffix="_m500",
            )
        else:
            person_df = person_df.with_columns(pl.lit(None).cast(pl.Float64).alias("P501"))

        # Merge P710_04 from module 700 (household-level)
        if mod700 is not None:
            # Normalize column name: P710_4 -> P710_04
            if "P710_4" in mod700.columns and "P710_04" not in mod700.columns:
                mod700 = mod700.rename({"P710_4": "P710_04"})
            if "P710_04" in mod700.columns:
                person_df = person_df.join(
                    mod700.select(_HH_KEYS + ["P710_04"]),
                    on=_HH_KEYS,
                    how="left",
                    suffix="_m700",
                )
            else:
                person_df = person_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("P710_04")
                )
        else:
            person_df = person_df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("P710_04")
            )

        # Merge INGHOG1D from sumaria (household-level)
        if sumaria_df is not None:
            person_df = person_df.join(
                sumaria_df,
                on=_HH_KEYS,
                how="left",
                suffix="_sum",
            )
        else:
            person_df = person_df.with_columns(
                pl.lit(None).cast(pl.Float64).alias("INGHOG1D")
            )

        # Select final columns for this year
        final_cols = person_key + [
            c for c in ["P209", "P210", "P203", "P301A", "P501", "P710_04", "INGHOG1D"]
            if c in person_df.columns
        ]
        person_df = person_df.select(final_cols)
        all_person_frames.append(person_df)

    if not all_person_frames:
        logger.warning("No supplementary data loaded from any year")
        return pl.DataFrame()

    result = pl.concat(all_person_frames, how="vertical_relaxed")
    logger.info(
        "Supplementary data loaded: %d rows, columns: %s",
        result.height, result.columns,
    )
    return result


def _compute_parent_education(
    supp_df: pl.DataFrame,
    years: list[int],
) -> pl.DataFrame:
    """Extract parent education years per household and join back to children.

    Finds the household head (P203==1) or spouse (P203==2) for each household,
    maps their P301A education level to approximate years of schooling, and
    returns a DataFrame keyed on (CONGLOME, VIVIENDA, HOGAR, year) with a
    ``parent_education_years`` column.
    """
    if "P203" not in supp_df.columns or "P301A" not in supp_df.columns:
        logger.warning("P203 or P301A missing from supplementary data; cannot compute parent education")
        return pl.DataFrame()

    # Filter to heads (P203==1) and spouses (P203==2)
    parents = supp_df.filter(
        pl.col("P203").is_in([1.0, 2.0])
    )

    # Map P301A to education years
    ed_map_expr = pl.col("P301A").cast(pl.Int64)
    for code, yrs in _P301A_TO_YEARS.items():
        ed_map_expr = pl.when(pl.col("P301A") == code).then(pl.lit(yrs)).otherwise(ed_map_expr)

    # Build the mapping expression properly
    mapping_expr = pl.lit(None).cast(pl.Int64)
    for code, yrs in _P301A_TO_YEARS.items():
        mapping_expr = (
            pl.when(pl.col("P301A") == float(code))
            .then(pl.lit(yrs))
            .otherwise(mapping_expr)
        )

    parents = parents.with_columns(
        mapping_expr.alias("_ed_years")
    )

    # Take the max education years per household (head or spouse, whichever higher)
    hh_education = (
        parents
        .group_by(_HH_KEYS + ["year"])
        .agg(pl.col("_ed_years").max().alias("parent_education_years"))
    )

    return hh_education


def _compute_weighted_poverty_quintile(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Assign weighted poverty quintiles (1-5) based on census_poverty_rate.

    Sorts by census_poverty_rate, computes cumulative FACTOR07 weight,
    and assigns quintile: Q1 = least poor (0-20%), Q5 = most poor (80-100%).
    """
    total_weight = df["FACTOR07"].sum()

    # Sort by poverty rate and compute cumulative weight
    sorted_df = df.with_row_index("_sort_idx").sort("census_poverty_rate")

    cum_weight = sorted_df["FACTOR07"].cum_sum()
    # Normalize to [0, 1]
    cum_pct = cum_weight / total_weight

    # Assign quintile based on cumulative weight percentage
    quintile_expr = (
        pl.when(cum_pct <= 0.2).then(pl.lit(1))
        .when(cum_pct <= 0.4).then(pl.lit(2))
        .when(cum_pct <= 0.6).then(pl.lit(3))
        .when(cum_pct <= 0.8).then(pl.lit(4))
        .otherwise(pl.lit(5))
    )

    sorted_df = sorted_df.with_columns(
        cum_pct.alias("_cum_pct"),
        quintile_expr.alias("poverty_quintile"),
    )

    # Restore original row order
    sorted_df = sorted_df.sort("_sort_idx").drop("_sort_idx", "_cum_pct")

    return sorted_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(df: pl.DataFrame) -> FeatureResult:
    """Build the complete feature matrix from the merged dataset.

    Takes the ``full_dataset.parquet`` DataFrame and returns a
    :class:`FeatureResult` with all engineered features as new lowercase
    columns.

    Parameters
    ----------
    df : pl.DataFrame
        Merged dataset from ``full_dataset.parquet`` (150,135 rows x 27 cols).

    Returns
    -------
    FeatureResult
        Feature matrix with model features, meta columns, and diagnostics.
    """
    warnings: list[str] = []
    initial_rows = df.height
    logger.info("Starting feature engineering on %d rows", initial_rows)

    # -----------------------------------------------------------------------
    # 1. Direct mappings (no additional data needed)
    # -----------------------------------------------------------------------
    logger.info("Step 1: Direct mappings from existing columns")

    df = df.with_columns([
        # Age: continuous (already 6-17)
        pl.col("P208A").cast(pl.Int64).alias("age"),

        # Secundaria-age binary (>= 12)
        pl.when(pl.col("P208A") >= 12)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("is_secundaria_age"),

        # Sex binary: female = 1
        pl.when(pl.col("P207") == 2.0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("es_mujer"),

        # Mother tongue dummies
        pl.when(pl.col("P300A") == 4.0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("lang_castellano"),

        pl.when(pl.col("P300A") == 1.0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("lang_quechua"),

        pl.when(pl.col("P300A") == 2.0)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("lang_aimara"),

        pl.when(pl.col("p300a_harmonized") == 3)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("lang_other_indigenous"),

        pl.when(pl.col("P300A").is_in([6.0, 7.0]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("lang_foreign"),

        # Rural: ESTRATO >= 6
        pl.when(pl.col("ESTRATO") >= 6)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("rural"),

        # Region dummies (costa as reference)
        pl.when(pl.col("DOMINIO").is_in([4, 5, 6]))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("is_sierra"),

        pl.when(pl.col("DOMINIO") == 7)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("is_selva"),

        # Region natural (categorical, for analysis / meta)
        pl.when(pl.col("DOMINIO").is_in([1, 2, 3, 8]))
        .then(pl.lit("costa"))
        .when(pl.col("DOMINIO").is_in([4, 5, 6]))
        .then(pl.lit("sierra"))
        .when(pl.col("DOMINIO") == 7)
        .then(pl.lit("selva"))
        .otherwise(pl.lit("unknown"))
        .alias("region_natural"),

        # Department code (first 2 digits of UBIGEO)
        pl.col("UBIGEO").str.slice(0, 2).alias("department"),

        # Nightlight intensity (rename)
        pl.col("nightlights_mean_radiance").alias("nightlight_intensity"),

        # Poverty index (rename)
        pl.col("census_poverty_rate").alias("poverty_index"),
    ])

    # District admin dropout rate: primaria for ages 6-11, secundaria for 12-17
    # Fallback to primaria when secundaria is null
    df = df.with_columns(
        pl.when(pl.col("age") >= 12)
        .then(
            pl.when(pl.col("admin_secundaria_rate").is_not_null())
            .then(pl.col("admin_secundaria_rate"))
            .otherwise(pl.col("admin_primaria_rate"))
        )
        .otherwise(pl.col("admin_primaria_rate"))
        .alias("district_dropout_rate_admin")
    )

    logger.info("Direct mappings complete: %d features added", 16)

    # -----------------------------------------------------------------------
    # 2. Weighted poverty quintile
    # -----------------------------------------------------------------------
    logger.info("Step 2: Computing weighted poverty quintiles")
    df = _compute_weighted_poverty_quintile(df)

    # Validate quintile balance
    quintile_weighted = {}
    total_weight = df["FACTOR07"].sum()
    for q in range(1, 6):
        q_weight = df.filter(pl.col("poverty_quintile") == q)["FACTOR07"].sum()
        quintile_weighted[q] = round(float(q_weight / total_weight), 4)
    logger.info("Quintile weighted shares: %s", quintile_weighted)

    # -----------------------------------------------------------------------
    # 3. Supplementary features from raw ENAHO modules
    # -----------------------------------------------------------------------
    logger.info("Step 3: Loading supplementary features from raw ENAHO modules")
    years = sorted(df["year"].unique().to_list())
    supp_df = _load_supplementary_features(years)

    supplementary_load_summary: dict[str, int] = {}

    if supp_df.height > 0:
        # Join supplementary data to main DataFrame on person keys + year
        supp_join_keys = JOIN_KEYS + ["year"]
        # Deduplicate supplementary data (take first occurrence)
        supp_df = supp_df.unique(subset=supp_join_keys, keep="first")

        pre_join = df.height
        df = df.join(
            supp_df,
            on=supp_join_keys,
            how="left",
            suffix="_supp",
        )
        assert df.height == pre_join, (
            f"Supplementary join changed row count: {pre_join} -> {df.height}"
        )

        # Track match rates per module column
        for col in ["P209", "P210", "P501", "P710_04", "INGHOG1D"]:
            if col in df.columns:
                matched = df.height - df[col].null_count()
                supplementary_load_summary[col] = matched
                logger.info("  %s: %d/%d matched (%.1f%%)",
                            col, matched, df.height, 100 * matched / df.height)

        # --- es_peruano: 1 = Peruvian, 0 = foreign-born ---
        if "P209" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("P209") == 5.0)
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("es_peruano")
            )
        else:
            df = df.with_columns(pl.lit(1).alias("es_peruano"))
            warnings.append("P209 not available; es_peruano set to 1 for all rows")

        # --- has_disability: 1 if P210 == 1 ---
        if "P210" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("P210") == 1.0)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("has_disability")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("has_disability"))
            warnings.append("P210 not available; has_disability set to 0 for all rows")

        # --- is_working: 1 if P501 == 1 ---
        if "P501" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("P501") == 1.0)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("is_working")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("is_working"))
            warnings.append("P501 not available; is_working set to 0 for all rows")

        # --- juntos_participant: 1 if P710_04 == 1 ---
        if "P710_04" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("P710_04") == 1.0)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("juntos_participant")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("juntos_participant"))
            warnings.append("P710_04 not available; juntos_participant set to 0 for all rows")

        # --- log_income: log(INGHOG1D + 1) ---
        if "INGHOG1D" in df.columns:
            # Median imputation for missing household income
            median_income = df["INGHOG1D"].drop_nulls().median()
            n_income_null = df["INGHOG1D"].null_count()
            if n_income_null > 0:
                logger.info(
                    "Imputing %d null INGHOG1D values with median %.2f",
                    n_income_null, median_income,
                )
                warnings.append(
                    f"Imputed {n_income_null} null INGHOG1D with median {median_income:.2f}"
                )
                df = df.with_columns(
                    pl.col("INGHOG1D").fill_null(median_income)
                )
            df = df.with_columns(
                (pl.col("INGHOG1D") + 1.0).log().alias("log_income")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("log_income"))
            warnings.append("INGHOG1D not available; log_income set to 0.0")

        # --- parent_education_years ---
        hh_education = _compute_parent_education(supp_df, years)
        if hh_education.height > 0:
            df = df.join(
                hh_education,
                on=_HH_KEYS + ["year"],
                how="left",
                suffix="_ped",
            )
            # Median imputation for children without parent education info
            if "parent_education_years" in df.columns:
                median_ed = df["parent_education_years"].drop_nulls().median()
                n_ed_null = df["parent_education_years"].null_count()
                if n_ed_null > 0:
                    logger.info(
                        "Imputing %d null parent_education_years with median %.1f",
                        n_ed_null, median_ed,
                    )
                    warnings.append(
                        f"Imputed {n_ed_null} null parent_education_years with median {median_ed}"
                    )
                    df = df.with_columns(
                        pl.col("parent_education_years").fill_null(median_ed).cast(pl.Float64)
                    )
            else:
                df = df.with_columns(pl.lit(0.0).alias("parent_education_years"))
                warnings.append("parent_education_years column not created; set to 0.0")
        else:
            df = df.with_columns(pl.lit(0.0).alias("parent_education_years"))
            warnings.append("No parent education data; parent_education_years set to 0.0")
    else:
        # No supplementary data at all -- set defaults
        warnings.append("No supplementary data loaded; all supplementary features set to defaults")
        df = df.with_columns([
            pl.lit(1).alias("es_peruano"),
            pl.lit(0).alias("has_disability"),
            pl.lit(0).alias("is_working"),
            pl.lit(0).alias("juntos_participant"),
            pl.lit(0.0).alias("log_income"),
            pl.lit(0.0).alias("parent_education_years"),
        ])

    logger.info("Supplementary features complete")

    # -----------------------------------------------------------------------
    # 4. Z-score standardization of spatial features
    # -----------------------------------------------------------------------
    logger.info("Step 4: Z-score standardization of district-level features")
    for src_col, dst_col in _SPATIAL_COLS_FOR_ZSCORE:
        if src_col in df.columns:
            # Impute nulls with 0 (mean of z-distribution) for nightlights
            impute = (src_col == "nightlight_intensity")
            df = _zscore(df, src_col, dst_col, impute_null=impute)
            logger.info("  z-scored %s -> %s (impute_null=%s)", src_col, dst_col, impute)
        else:
            logger.warning("  Column %s not found; skipping z-score", src_col)
            df = df.with_columns(pl.lit(0.0).alias(dst_col))
            warnings.append(f"{src_col} not found; {dst_col} set to 0.0")

    # -----------------------------------------------------------------------
    # 5. school_student_teacher_ratio: ESCALE not available
    # -----------------------------------------------------------------------
    logger.warning(
        "school_student_teacher_ratio: ESCALE data not available. "
        "Feature set to null. Model runs with %d features.",
        len(MODEL_FEATURES),
    )
    warnings.append(
        "school_student_teacher_ratio: ESCALE data not available. "
        f"Feature set to null. Model runs with {len(MODEL_FEATURES)} features."
    )
    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias("school_student_teacher_ratio")
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    logger.info("Validating feature matrix...")

    assert df.height == initial_rows, (
        f"Row count changed during feature engineering: {initial_rows} -> {df.height}"
    )

    # Check all model features exist
    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    # Binary feature validation
    binary_features = [
        "es_mujer", "lang_castellano", "lang_quechua", "lang_aimara",
        "lang_other_indigenous", "lang_foreign", "rural", "is_sierra",
        "is_selva", "es_peruano", "has_disability", "is_working",
        "juntos_participant", "is_secundaria_age",
    ]
    binary_validation: dict[str, list] = {}
    for col in binary_features:
        if col in df.columns:
            unique_vals = sorted(df[col].unique().drop_nulls().to_list())
            binary_validation[col] = unique_vals
            if not set(unique_vals).issubset({0, 1}):
                warnings.append(f"Binary feature {col} has unexpected values: {unique_vals}")

    # Null rate check (no feature should have >30% nulls)
    null_rates: dict[str, float] = {}
    for col in MODEL_FEATURES:
        n_null = df[col].null_count()
        rate = n_null / df.height
        null_rates[col] = round(rate, 4)
        if rate > 0.30:
            warnings.append(f"Feature {col} has {rate:.1%} nulls (exceeds 30% threshold)")

    # -----------------------------------------------------------------------
    # Build stats
    # -----------------------------------------------------------------------
    stats = {
        "total_rows": df.height,
        "total_features": len(MODEL_FEATURES),
        "model_feature_count": len(MODEL_FEATURES),
        "null_rates": null_rates,
        "quintile_balance": quintile_weighted,
        "binary_validation": binary_validation,
        "supplementary_load_summary": supplementary_load_summary,
    }

    logger.info(
        "Feature engineering complete: %d rows, %d model features, %d warnings",
        df.height, len(MODEL_FEATURES), len(warnings),
    )

    return FeatureResult(
        df=df,
        stats=stats,
        warnings=warnings,
        model_features=list(MODEL_FEATURES),
    )
