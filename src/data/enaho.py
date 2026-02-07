"""ENAHO single-year data loader.

Reads ENAHO Module 200 (demographics) and Module 300 (education) CSVs for a
given year, joins them on household/person composite keys, filters to
school-age children (6-17), constructs the binary dropout target, and returns
a validated :class:`ENAHOResult`.

Usage::

    from data.enaho import load_enaho_year

    result = load_enaho_year(2023)
    print(result.stats)
    print(result.df.head())
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_module_file(year_dir: Path, prefix: str, module: str) -> Path:
    """Locate an ENAHO module CSV with case- and separator-insensitive search.

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
        Path to the matching CSV file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    year = year_dir.name
    # Generate candidate patterns with hyphen and underscore separators,
    # across common case variations.
    separators = ["-", "_"]
    case_variants = [prefix, prefix.lower(), prefix.upper()]

    for sep in separators:
        for variant in case_variants:
            pattern = f"{variant}{sep}{year}{sep}{module}*.csv"
            matches = list(year_dir.glob(pattern))
            if matches:
                return matches[0]

    # Also try case-insensitive search on all CSV files as last resort
    for csv_file in year_dir.glob("*.csv"):
        name_lower = csv_file.name.lower()
        if module in name_lower and prefix.lower().replace("enaho", "enaho") in name_lower:
            return csv_file

    raise FileNotFoundError(
        f"Module {module} not found in {year_dir}. "
        "Run 'uv run python src/data/download.py' first."
    )


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

    delimiter = sniff_delimiter(filepath)
    logger.info("Module 200 [%d]: %s (delimiter=%r)", year, filepath.name, delimiter)

    df = pl.read_csv(
        filepath,
        separator=delimiter,
        encoding="utf8-lossy",
        schema_overrides=_KEY_OVERRIDES,
        infer_schema_length=10_000,
    )

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

    delimiter = sniff_delimiter(filepath)
    logger.info("Module 300 [%d]: %s (delimiter=%r)", year, filepath.name, delimiter)

    # Include join keys as Utf8 to match Module 200
    overrides = {**_KEY_OVERRIDES}

    df = pl.read_csv(
        filepath,
        separator=delimiter,
        encoding="utf8-lossy",
        schema_overrides=overrides,
        infer_schema_length=10_000,
    )

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

    # 5. Cast P303, P306 to Int64 (may arrive as Utf8 from some years)
    for col_name in ("P303", "P306"):
        if col_name in df.columns and df[col_name].dtype != pl.Int64:
            df = df.with_columns(pl.col(col_name).cast(pl.Int64))
            warnings.append(f"Cast {col_name} from {df[col_name].dtype} to Int64")

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
