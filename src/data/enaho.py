"""ENAHO data loader (single-year and multi-year).

Reads ENAHO Module 200 (demographics) and Module 300 (education) CSVs for a
given year, joins them on household/person composite keys, filters to
school-age children (6-17), constructs the binary dropout target, and returns
a validated :class:`ENAHOResult`.

For multi-year analysis, :func:`load_all_years` pools multiple years into a
single :class:`PooledENAHOResult` with P300A mother tongue harmonization.

Usage::

    from data.enaho import load_enaho_year

    result = load_enaho_year(2023)
    print(result.stats)
    print(result.df.head())

    from data.enaho import load_all_years

    pooled = load_all_years()
    print(pooled.df.height, pooled.per_year_stats)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from utils import find_project_root, pad_ubigeo, sniff_delimiter

logger = logging.getLogger(__name__)

# Composite key columns used to join Module 200 and Module 300.
JOIN_KEYS = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]

# Columns that must have zero nulls in the final output.
CRITICAL_COLUMNS = ["UBIGEO", "P208A", "P303", "P306", "FACTOR07"]

# Schema overrides to prevent leading-zero loss and ensure consistent types
# for join key columns across modules.
_KEY_OVERRIDES: dict[str, pl.DataType] = {
    "UBIGEO": pl.Utf8,
    "CONGLOME": pl.Utf8,
    "VIVIENDA": pl.Utf8,
    "HOGAR": pl.Utf8,
    "CODPERSO": pl.Utf8,
}

# Columns selected for the pooled multi-year DataFrame.
# Only these ~20 columns are kept before vertical concatenation to ensure
# identical schemas across years (raw merged DataFrames have 512-548 columns).
POOLED_COLUMNS = [
    # Identifiers
    "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO",
    # Geographic
    "UBIGEO", "DOMINIO", "ESTRATO",
    # Demographics (Module 200)
    "P207",     # Sex (1=Male, 2=Female)
    "P208A",    # Age
    # Education (Module 300)
    "P300A",    # Mother tongue (original code)
    "P301A",    # Education level
    "P303",     # Enrolled last year
    "P306",     # Enrolled this year
    "P307",     # Currently attending
    # Survey weight
    "FACTOR07",
    # Constructed
    "dropout",  # Binary dropout target
    # Year identifier
    "year",     # Added by load_all_years
]


@dataclass
class ENAHOResult:
    """Container for a loaded ENAHO year.

    Attributes
    ----------
    df : pl.DataFrame
        Cleaned, school-age DataFrame with ``dropout`` boolean column.
    stats : dict
        Summary statistics (total_rows, dropout_count, weighted_dropout_rate, year).
    warnings : list[str]
        Non-fatal issues encountered during loading.
    """

    df: pl.DataFrame
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PooledENAHOResult:
    """Container for pooled multi-year ENAHO data.

    Attributes
    ----------
    df : pl.DataFrame
        Pooled DataFrame with year column and harmonized P300A.
    per_year_stats : list[dict]
        Stats dict from each load_enaho_year() call.
    warnings : list[str]
        All warnings, prefixed with [year].
    """

    df: pl.DataFrame
    per_year_stats: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_module_file(year_dir: Path, prefix: str, module: str) -> Path:
    """Locate an ENAHO module file (CSV or DTA) with case-insensitive search.

    Parameters
    ----------
    year_dir : Path
        Directory for a specific year, e.g. ``data/raw/enaho/2023``.
    prefix : str
        File prefix (``"Enaho01"`` for Module 200, ``"Enaho01a"`` for Module 300).
    module : str
        Module number, e.g. ``"200"`` or ``"300"``.

    Returns
    -------
    Path
        Path to the matching file (CSV preferred, DTA as fallback).

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    year = year_dir.name
    separators = ["-", "_"]
    case_variants = [prefix, prefix.lower(), prefix.upper()]

    # Prefer exact module match first (e.g., "300.dta" not "300a.dta"),
    # then fall back to wildcard (e.g., "300*.dta").
    for ext in ("csv", "dta"):
        for sep in separators:
            for variant in case_variants:
                # Exact match: prefix-year-module.ext
                exact = year_dir / f"{variant}{sep}{year}{sep}{module}.{ext}"
                if exact.exists():
                    return exact
        for sep in separators:
            for variant in case_variants:
                # Wildcard match: prefix-year-module*.ext (e.g., 300-extra.ext)
                pattern = f"{variant}{sep}{year}{sep}{module}*.{ext}"
                matches = list(year_dir.glob(pattern))
                if matches:
                    return matches[0]

    # Case-insensitive fallback on all data files.
    # Match module number at a word boundary (e.g., "300." not "300a").
    for ext in ("csv", "dta"):
        for data_file in year_dir.glob(f"*.{ext}"):
            name_lower = data_file.name.lower()
            if prefix.lower() in name_lower and f"{module}.{ext}" in name_lower:
                return data_file
        # Looser fallback: module anywhere in filename
        for data_file in year_dir.glob(f"*.{ext}"):
            name_lower = data_file.name.lower()
            if module in name_lower and prefix.lower() in name_lower:
                return data_file

    raise FileNotFoundError(
        f"Module {module} not found in {year_dir}. "
        "Run 'uv run python src/data/download.py' first."
    )


def _read_data_file(
    filepath: Path,
    schema_overrides: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Read an ENAHO data file (CSV or DTA) into a polars DataFrame.

    DTA (Stata) files are read via pandas and converted to polars.
    Column names are uppercased to normalize DTA lowercase names.

    Parameters
    ----------
    filepath : Path
        Path to the data file.
    schema_overrides : dict
        Column type overrides (applied after reading).

    Returns
    -------
    pl.DataFrame
    """
    suffix = filepath.suffix.lower()

    if suffix == ".dta":
        import pandas as pd

        pdf = pd.read_stata(filepath, convert_categoricals=False)
        # DTA files have lowercase column names — normalize to uppercase
        pdf.columns = [c.upper() for c in pdf.columns]
        # Rename AÑO -> ANIO if present (common in ENAHO DTA files)
        if "AÑO" in pdf.columns:
            pdf = pdf.rename(columns={"AÑO": "ANIO"})
        df = pl.from_pandas(pdf)
    else:
        # CSV path
        delimiter = sniff_delimiter(filepath)
        logger.info("Reading CSV: %s (delimiter=%r)", filepath.name, delimiter)
        df = pl.read_csv(
            filepath,
            separator=delimiter,
            encoding="utf8-lossy",
            schema_overrides=schema_overrides,
            infer_schema_length=10_000,
        )

    # Apply schema overrides for key columns (ensure string types for joins)
    for col_name, dtype in schema_overrides.items():
        if col_name in df.columns and df[col_name].dtype != dtype:
            df = df.with_columns(pl.col(col_name).cast(dtype))

    return df


def _validate_ubigeo_length(df: pl.DataFrame) -> None:
    """Assert all UBIGEO values have exactly 6 characters.

    Raises
    ------
    ValueError
        If any UBIGEO value does not have length 6.
    """
    bad = df.filter(pl.col("UBIGEO").str.len_chars() != 6)
    if len(bad) > 0:
        examples = bad.select("UBIGEO").head(10).to_series().to_list()
        raise ValueError(
            f"UBIGEO length validation failed: {len(bad)} rows have length != 6. "
            f"Examples: {examples}"
        )


def _validate_critical_nulls(df: pl.DataFrame) -> None:
    """Assert zero nulls on critical columns.

    Raises
    ------
    ValueError
        If any critical column has null values.
    """
    null_counts = {}
    for col in CRITICAL_COLUMNS:
        if col in df.columns:
            n_nulls = df[col].null_count()
            if n_nulls > 0:
                null_counts[col] = n_nulls

    if null_counts:
        raise ValueError(
            f"Critical columns have nulls (must be zero): {null_counts}"
        )


def _validate_schema(df: pl.DataFrame, warnings: list[str]) -> pl.DataFrame:
    """Validate and coerce column types where needed.

    Returns the DataFrame (potentially with coerced columns).
    """
    expected_types = {
        "UBIGEO": pl.Utf8,
        "P303": pl.Int64,
        "P306": pl.Int64,
        "FACTOR07": pl.Float64,
        "dropout": pl.Boolean,
    }

    for col_name, expected in expected_types.items():
        if col_name not in df.columns:
            continue
        actual = df[col_name].dtype
        if actual != expected:
            try:
                df = df.with_columns(pl.col(col_name).cast(expected))
                warnings.append(
                    f"Coerced {col_name} from {actual} to {expected}"
                )
            except Exception as exc:
                warnings.append(
                    f"Could not coerce {col_name} from {actual} to {expected}: {exc}"
                )

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_module_200(year: int) -> pl.DataFrame:
    """Load ENAHO Module 200 (demographics) for a given year.

    Parameters
    ----------
    year : int
        Survey year (e.g. 2023).

    Returns
    -------
    pl.DataFrame
        DataFrame with at minimum: CONGLOME, VIVIENDA, HOGAR, CODPERSO,
        UBIGEO, P207 (sex), P208A (age).
    """
    root = find_project_root()
    year_dir = root / "data" / "raw" / "enaho" / str(year)
    filepath = _find_module_file(year_dir, "Enaho01", "200")

    logger.info("Module 200 [%d]: %s", year, filepath.name)

    df = _read_data_file(filepath, _KEY_OVERRIDES)

    # Zero-pad UBIGEO
    df = df.with_columns(pad_ubigeo(pl.col("UBIGEO")).alias("UBIGEO"))

    return df


def load_module_300(year: int) -> pl.DataFrame:
    """Load ENAHO Module 300 (education) for a given year.

    Parameters
    ----------
    year : int
        Survey year (e.g. 2023).

    Returns
    -------
    pl.DataFrame
        DataFrame with at minimum: CONGLOME, VIVIENDA, HOGAR, CODPERSO,
        P300A, P303, P306, P307, FACTOR07, P301A.
    """
    root = find_project_root()
    year_dir = root / "data" / "raw" / "enaho" / str(year)
    filepath = _find_module_file(year_dir, "Enaho01a", "300")

    logger.info("Module 300 [%d]: %s", year, filepath.name)

    # Include join keys as Utf8 to match Module 200
    overrides = {**_KEY_OVERRIDES}

    df = _read_data_file(filepath, overrides)

    return df


def load_enaho_year(year: int) -> ENAHOResult:
    """Load and process ENAHO data for a single year.

    Joins Module 200 (demographics) and Module 300 (education), filters to
    school-age children (ages 6-17), constructs the binary dropout target,
    and validates data quality.

    Parameters
    ----------
    year : int
        Survey year (e.g. 2023).

    Returns
    -------
    ENAHOResult
        Validated result with ``.df``, ``.stats``, and ``.warnings``.

    Raises
    ------
    FileNotFoundError
        If module CSV files are missing.
    ValueError
        If critical columns have nulls or UBIGEO validation fails.
    """
    warnings: list[str] = []

    # 1. Load modules
    mod200 = load_module_200(year)
    mod300 = load_module_300(year)

    logger.info(
        "Loaded: Module 200 = %d rows, Module 300 = %d rows",
        len(mod200),
        len(mod300),
    )

    # 2. LEFT join: all people from Module 200, education variables from 300
    df = mod200.join(mod300, on=JOIN_KEYS, how="left", suffix="_300")

    logger.info("After join: %d rows", len(df))

    # 3. Filter to school-age (6-17)
    df = df.filter((pl.col("P208A") >= 6) & (pl.col("P208A") <= 17))

    logger.info("After school-age filter (6-17): %d rows", len(df))

    # 3b. Drop rows not matched in Module 300 (left-join mismatches).
    # These are children in Module 200 (demographics) who have no education
    # module record — all Module 300 columns are null, including FACTOR07.
    # They cannot contribute to dropout analysis (no enrollment status, no weight).
    pre_drop = len(df)
    df = df.filter(pl.col("FACTOR07").is_not_null())
    n_dropped = pre_drop - len(df)
    if n_dropped > 0:
        warnings.append(
            f"Dropped {n_dropped} school-age rows ({n_dropped/pre_drop:.2%}) "
            f"with no Module 300 match (FACTOR07 null)"
        )
        logger.info(
            "Dropped %d unmatched rows (no Module 300 data), %d remaining",
            n_dropped,
            len(df),
        )

    # 3c. Drop rows where P303 is null (COVID reduced questionnaire)
    # These rows cannot contribute to dropout analysis (no prior enrollment info).
    # Affects 2020 (~52.3% of school-age) and 2021 (~4.6%).
    p303_null_count = df["P303"].null_count()
    if p303_null_count > 0:
        pre_p303 = len(df)
        df = df.filter(pl.col("P303").is_not_null())
        n_p303_dropped = pre_p303 - len(df)
        frac_p303 = n_p303_dropped / pre_p303 if pre_p303 > 0 else 0
        warnings.append(
            f"Dropped {n_p303_dropped} rows ({frac_p303:.1%}) with P303 null "
            f"(COVID reduced questionnaire / phone interview)"
        )
        logger.info(
            "Dropped %d P303-null rows (%.1f%%), %d remaining",
            n_p303_dropped, frac_p303 * 100, len(df),
        )

    # 4. Handle nulls in P303/P306 among school-age rows
    for col_name in ("P303", "P306"):
        if col_name in df.columns:
            n_nulls = df[col_name].null_count()
            n_rows = len(df)
            if n_nulls > 0:
                frac = n_nulls / n_rows if n_rows > 0 else 0
                if frac < 0.005:
                    # Conservative fill: P303=2 (was not enrolled) -> dropout=False
                    df = df.with_columns(pl.col(col_name).fill_null(2))
                    warnings.append(
                        f"Filled {n_nulls} nulls ({frac:.4f}) in {col_name} "
                        f"with conservative value 2"
                    )
                else:
                    sample_rows = df.filter(pl.col(col_name).is_null()).head(5)
                    raise ValueError(
                        f"Too many nulls in {col_name}: {n_nulls}/{n_rows} "
                        f"({frac:.2%}). Sample rows:\n{sample_rows}"
                    )

    # 5. Cast P303, P306 to Int64 (may arrive as Utf8 or Float64 from some years)
    for col_name in ("P303", "P306"):
        if col_name in df.columns and df[col_name].dtype != pl.Int64:
            original_dtype = df[col_name].dtype
            df = df.with_columns(pl.col(col_name).cast(pl.Int64))
            warnings.append(f"Cast {col_name} from {original_dtype} to Int64")

    # 6. Construct dropout target: enrolled last year AND not enrolled this year
    df = df.with_columns(
        ((pl.col("P303") == 1) & (pl.col("P306") == 2)).alias("dropout")
    )

    # 7. Cast FACTOR07 to Float64 if needed
    if "FACTOR07" in df.columns and df["FACTOR07"].dtype != pl.Float64:
        df = df.with_columns(pl.col("FACTOR07").cast(pl.Float64))
        warnings.append(f"Cast FACTOR07 to Float64")

    # 8. Schema validation and coercion
    df = _validate_schema(df, warnings)

    # 9. Strict null validation on critical columns
    _validate_critical_nulls(df)

    # 10. UBIGEO length validation
    _validate_ubigeo_length(df)

    # 11. Compute statistics
    total_rows = len(df)
    dropout_count = df.filter(pl.col("dropout")).height
    total_weight = df["FACTOR07"].sum()
    dropout_weight = df.filter(pl.col("dropout"))["FACTOR07"].sum()
    weighted_dropout_rate = dropout_weight / total_weight if total_weight > 0 else 0.0

    stats = {
        "year": year,
        "total_rows": total_rows,
        "dropout_count": dropout_count,
        "weighted_dropout_rate": weighted_dropout_rate,
    }

    logger.info(
        "Year %d: %d school-age rows, %d dropouts, %.3f weighted rate",
        year,
        total_rows,
        dropout_count,
        weighted_dropout_rate,
    )

    return ENAHOResult(df=df, stats=stats, warnings=warnings)


# Disaggregated indigenous language codes introduced in ENAHO 2020+.
# These were previously aggregated under code 3 ("Otra lengua nativa").
_DISAGG_CODES = [10, 11, 12, 13, 14, 15]


def harmonize_p300a(df: pl.DataFrame) -> pl.DataFrame:
    """Add harmonized and original P300A mother tongue columns.

    - p300a_original: raw code from INEI (preserves 10-15 for 2020+ analysis)
    - p300a_harmonized: codes 10-15 collapsed to 3 for cross-year comparison

    P300A code reference (verified from Stata value labels):
        1 = Quechua
        2 = Aymara
        3 = Otra lengua nativa (aggregate pre-2020; residual 2020+)
        4 = Castellano (~80% of respondents)
        6 = Portugues
        7 = Otra lengua extranjera
        8 = No escucha/no habla
        9 = Lengua de senas peruanas
       10 = Ashaninka (2020+)
       11 = Awajun/Aguarun (2020+)
       12 = Shipibo-Konibo (2020+)
       13 = Shawi/Chayahuita (2020+)
       14 = Matsigenka/Machiguenga (2020+)
       15 = Achuar (2020+)
    """
    return df.with_columns([
        pl.col("P300A").alias("p300a_original"),
        pl.when(pl.col("P300A").is_in(_DISAGG_CODES))
        .then(pl.lit(3))
        .otherwise(pl.col("P300A"))
        .cast(pl.Int64)
        .alias("p300a_harmonized"),
    ])


def load_all_years(
    years: list[int] | None = None,
) -> PooledENAHOResult:
    """Load and pool ENAHO data across multiple years.

    Calls :func:`load_enaho_year` for each year, selects a fixed column set
    (POOLED_COLUMNS) for schema consistency, concatenates vertically, and
    applies P300A harmonization.

    Parameters
    ----------
    years : list[int] | None
        Years to load. Defaults to 2018-2023 (2024 not yet available).

    Returns
    -------
    PooledENAHOResult
        Pooled DataFrame with year column and p300a_harmonized/p300a_original.
    """
    if years is None:
        years = list(range(2018, 2024))  # 2018-2023

    frames: list[pl.DataFrame] = []
    all_stats: list[dict] = []
    all_warnings: list[str] = []

    for year in years:
        logger.info("Loading ENAHO year %d...", year)
        result = load_enaho_year(year)

        # Add year column
        df = result.df.with_columns(pl.lit(year).alias("year"))

        # Select only columns needed by downstream phases
        available = [c for c in POOLED_COLUMNS if c in df.columns]
        df = df.select(available)

        frames.append(df)
        all_stats.append(result.stats)
        all_warnings.extend(
            [f"[{year}] {w}" for w in result.warnings]
        )
        logger.info(
            "Year %d: %d rows selected (%d columns)",
            year, len(df), len(available),
        )

    # Vertical concat (enforces identical schemas)
    pooled = pl.concat(frames, how="vertical")
    logger.info("Pooled DataFrame: %d rows, %d columns", len(pooled), len(pooled.columns))

    # Apply P300A harmonization on the pooled data
    pooled = harmonize_p300a(pooled)

    logger.info(
        "Harmonization complete. Columns: %s",
        sorted(pooled.columns),
    )

    return PooledENAHOResult(
        df=pooled,
        per_year_stats=all_stats,
        warnings=all_warnings,
    )
