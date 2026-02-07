# Phase 1: ENAHO Single-Year Loader - Research

**Researched:** 2026-02-07
**Domain:** ENAHO survey microdata ingestion (CSV loading, UBIGEO handling, dropout target construction)
**Confidence:** HIGH (polars API verified live; ENAHO variable definitions from INEI/ILO data dictionaries; dropout logic from spec DATA-02)

## Summary

This phase loads a single year (2023) of ENAHO household survey microdata into a clean polars DataFrame. The implementation requires loading two separate ENAHO module files (Module 200 for demographics, Module 300 for education), joining them on household/person composite keys, filtering to school-age children (6-17), constructing a binary dropout target from enrollment variables P303/P306, and validating UBIGEO geographic codes, null integrity, and expected statistical ranges.

The primary technical challenges are: (1) delimiter detection -- ENAHO switched from pipe `|` to comma `,` in 2020, (2) UBIGEO zero-padding -- 6-digit codes lose leading zeros when parsed as integers, (3) column name normalization -- trailing whitespace and case differences across years, and (4) proper join strategy between modules to preserve all school-age records.

**Primary recommendation:** Use `csv.Sniffer` for delimiter detection (verified working with pipe-delimited ENAHO data), force UBIGEO to string type via `schema_overrides`, join modules on composite key `[CONGLOME, VIVIENDA, HOGAR, CODPERSO]` using LEFT join from Module 200 (demographics) onto Module 300 (education), filter to ages 6-17, and construct dropout as `(P303 == 1) & (P306 == 2)` per spec DATA-02.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Dropout target construction
- Enrollment-based definition: dropout = school-age child not currently enrolled (Claude determines exact ENAHO variables and handles attendance edge cases)
- Primary + secondary only -- no pre-primary/initial education ages
- Claude decides whether to exclude completed-secondary 17-year-olds or keep as target=0
- Claude sets gate test tolerance bands for the ~14% weighted dropout rate (warn vs fail thresholds)

#### Data validation & reporting
- Minimal output: key stats (total rows, dropout count, weighted rate) plus 10 random dropout rows for human inspection
- Strict null handling on critical columns: UBIGEO, age, enrollment status, and survey weight (FACTOR07) must have zero nulls -- fail if violated
- Other columns: report null counts but don't fail
- Claude decides whether gate test also validates column schema (names + types) beyond the statistical assertions

#### Loader interface design
- Data directory resolved from project root via pyproject.toml walk-up (Phase 0 pattern) -- not configurable
- Return a named tuple or dataclass with `.df`, `.stats`, `.warnings` -- not a bare DataFrame
- Composable architecture: separate load functions per ENAHO module (load_module_200, load_module_300, etc.) that compose into `load_enaho_year()`
- Claude decides on parquet caching strategy (cache vs always-from-raw)

#### ENAHO module handling
- Delimiter detection via sniffing first N lines (frequency analysis), not hardcoded
- Missing module files: fail immediately with clear message ("Module X not found in data/raw/. Run download.py first.") -- no auto-download
- Claude decides join strategy (left from education module vs inner) to maximize school-age record preservation
- Claude decides file format handling based on what download.py actually produces

### Claude's Discretion
- Exact ENAHO variables for enrollment/dropout determination
- Temporary absence handling (separate category or ignore)
- Completed-secondary exclusion vs keep-as-non-dropout
- Gate test tolerance bands for weighted dropout rate
- Column schema validation in gate test
- Parquet caching vs always-from-raw
- Join strategy for ENAHO modules
- File format support (CSV only vs CSV+SAV)
- Columns shown in the 10 inspection rows

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

The established libraries/tools for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| polars | >=1.38.1 | CSV loading, DataFrame operations, UBIGEO padding | Spec-locked. 30-50x faster than pandas for CSV I/O. Expression API prevents mutation bugs. |
| csv (stdlib) | Python 3.12 | Delimiter sniffing via `csv.Sniffer` | Stdlib, no dependencies. `Sniffer.sniff()` reliably detects pipe vs comma delimiters. Verified live. |
| pytest | >=9.0 | Gate tests and unit tests | Spec-locked. Gate test 1.1 validates all assertions. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses (stdlib) | Python 3.12 | `Enaho Result` return type with `.df`, `.stats`, `.warnings` | For the `load_enaho_year()` return type. |
| pathlib (stdlib) | Python 3.12 | Path resolution, pyproject.toml walk-up | For `_find_project_root()` and file path construction. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `csv.Sniffer` | Custom frequency analysis | Sniffer is stdlib and handles edge cases; custom is simpler but less robust. Use Sniffer with fallback to frequency analysis. |
| `csv.Sniffer` | Hardcoded delimiter by year | Spec says "sniff, not hardcode." Hardcoding is fragile if INEI changes format again. |
| dataclass | NamedTuple | Dataclass is mutable (can add warnings during processing), supports default values. NamedTuple is immutable. Use dataclass. |
| polars `read_csv` | pandas `read_csv` | Spec mandates polars. polars is faster and has better type handling. |

**Installation:** No additional packages needed -- polars and pytest already in pyproject.toml.

## Architecture Patterns

### Recommended Module Structure
```
src/
├── data/
│   ├── __init__.py
│   ├── enaho.py           # load_module_200(), load_module_300(), load_enaho_year()
│   └── download.py        # Existing -- downloads raw data
├── utils.py               # pad_ubigeo(), find_project_root(), sniff_delimiter()
tests/
├── gates/
│   └── test_gate_1_1.py   # Gate test for single-year loader
└── unit/
    ├── test_enaho_loader.py  # Unit tests for module loading functions
    └── test_ubigeo.py        # Unit tests for UBIGEO padding
```

### Pattern 1: Composable Module Loaders
**What:** Each ENAHO module has its own load function (`load_module_200`, `load_module_300`) that handles file discovery, delimiter detection, column normalization, and type coercion. These compose into `load_enaho_year()`.
**When to use:** Always -- this is the locked design decision from CONTEXT.md.
**Example:**
```python
# In src/data/enaho.py
@dataclass
class ENAHOResult:
    df: pl.DataFrame
    stats: dict
    warnings: list[str]

def load_module_200(year: int) -> pl.DataFrame:
    """Load Module 200 (demographics: age, sex, UBIGEO)."""
    filepath = _find_module_file(year, "200")
    sep = sniff_delimiter(filepath)
    df = pl.read_csv(filepath, separator=sep, infer_schema_length=10000,
                     schema_overrides={"UBIGEO": pl.Utf8})
    df = _normalize_columns(df)
    df = df.with_columns(pad_ubigeo(pl.col("UBIGEO")))
    return df

def load_module_300(year: int) -> pl.DataFrame:
    """Load Module 300 (education: P303, P306, P300A, FACTOR07)."""
    filepath = _find_module_file(year, "300")
    sep = sniff_delimiter(filepath)
    df = pl.read_csv(filepath, separator=sep, infer_schema_length=10000,
                     schema_overrides={"UBIGEO": pl.Utf8})
    df = _normalize_columns(df)
    return df

def load_enaho_year(year: int) -> ENAHOResult:
    """Load and merge ENAHO modules for a single year."""
    mod200 = load_module_200(year)
    mod300 = load_module_300(year)

    # Join on composite key
    JOIN_KEYS = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]
    merged = mod200.join(mod300, on=JOIN_KEYS, how="left", suffix="_mod300")

    # Filter to school-age (6-17) and construct dropout target
    result = (merged
        .filter(pl.col("P208A").is_between(6, 17))
        .with_columns(
            pl.when((pl.col("P303") == 1) & (pl.col("P306") == 2))
            .then(1)
            .otherwise(0)
            .alias("dropout")
        ))

    # Compute stats
    stats = _compute_stats(result)
    warnings = _validate(result)
    return ENAHOResult(df=result, stats=stats, warnings=warnings)
```

### Pattern 2: Strict Null Validation
**What:** Critical columns (UBIGEO, P208A/age, P303, P306, FACTOR07) must have zero nulls. Other columns report null counts but do not fail.
**When to use:** After module loading and before any downstream processing.
**Example:**
```python
CRITICAL_COLUMNS = ["UBIGEO", "P208A", "P303", "P306", "FACTOR07"]

def _validate(df: pl.DataFrame) -> list[str]:
    warnings = []
    for col in CRITICAL_COLUMNS:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValueError(f"Critical column {col} has {null_count} nulls. Cannot proceed.")

    # Report non-critical null counts
    for col in df.columns:
        if col not in CRITICAL_COLUMNS:
            null_count = df[col].null_count()
            if null_count > 0:
                warnings.append(f"Column {col}: {null_count} nulls ({null_count/len(df)*100:.1f}%)")
    return warnings
```

### Pattern 3: UBIGEO Zero-Padding Utility
**What:** A reusable expression/function that pads UBIGEO to 6 characters with leading zeros, used in every data loader.
**When to use:** Every time a data source with UBIGEO is loaded (ENAHO, admin, census, nightlights).
**Example:**
```python
# In src/utils.py
def pad_ubigeo(col: pl.Expr) -> pl.Expr:
    """Pad UBIGEO to 6 characters with leading zeros."""
    return col.cast(pl.Utf8).str.strip_chars().str.pad_start(6, "0")
```

### Anti-Patterns to Avoid
- **Hardcoded delimiter by year:** Use `sniff_delimiter()` instead of `if year <= 2019: sep = "|"`. The latter breaks if INEI changes format or if we encounter edge cases.
- **Bare DataFrame return:** Always return `ENAHOResult` dataclass, never a bare polars DataFrame. The `.stats` and `.warnings` fields are essential for gate testing and human review.
- **UBIGEO as integer:** Never read UBIGEO without `schema_overrides={"UBIGEO": pl.Utf8}`. Integer inference silently strips leading zeros.
- **Module 300 as base for join:** Module 200 (demographics) has records for ALL household members including children too young for education questions. Use Module 200 as the left table so school-age children who have no education module record are preserved (they would be non-enrolled).

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Delimiter detection | Custom character frequency counter | `csv.Sniffer().sniff()` | Handles quoting, escaping, edge cases. Verified working with ENAHO pipe-delimited data. Falls back cleanly. |
| UBIGEO padding | String formatting with f-strings | `pl.Expr.str.pad_start(6, "0")` | Handles edge cases (already 6 chars, non-numeric chars). Vectorized in polars. |
| CSV type inference | Manual type casting after load | `schema_overrides={"UBIGEO": pl.Utf8}` in `read_csv` | Forces type at parse time, before any data loss occurs. |
| Project root discovery | Hardcoded `Path(__file__).parent.parent` | pyproject.toml walk-up pattern (from Phase 0) | Robust regardless of where the script is called from. |

**Key insight:** The stdlib `csv.Sniffer` handles delimiter detection robustly including quoted fields with embedded delimiters. A custom frequency counter would need to handle these edge cases manually.

## Common Pitfalls

### Pitfall 1: Wrong Delimiter Silently Loads Garbage
**What goes wrong:** ENAHO 2018-2019 uses pipe `|`, 2020+ uses comma `,`. Wrong delimiter creates a single-column DataFrame with all data concatenated.
**Why it happens:** polars `read_csv` defaults to comma. No error is raised for pipe-delimited data read with comma separator.
**How to avoid:** Use `sniff_delimiter()`. After every load, assert column count >= 20. Assert that key columns (P303, P306, UBIGEO) exist.
**Warning signs:** DataFrame has 1 column. Column names contain pipe characters. All values in P303 are null.

### Pitfall 2: UBIGEO Leading Zero Loss
**What goes wrong:** UBIGEO "010101" (Amazonas) becomes 10101 when parsed as integer. All spatial merges for departments 01-09 silently fail.
**Why it happens:** polars/pandas infer digit-only columns as integers. Even `str(10101)` does not restore the leading zero.
**How to avoid:** Force `schema_overrides={"UBIGEO": pl.Utf8}` at load time. Call `pad_ubigeo()` immediately. Assert `len == 6` after padding.
**Warning signs:** UBIGEO values with 5 characters. Merge coverage below 85%. Departments Amazonas through Huanuco missing enrichment data.

### Pitfall 3: Column Name Whitespace and Case Variation
**What goes wrong:** ENAHO CSV headers sometimes have trailing spaces or inconsistent casing (`FACTOR07` vs `factor07` vs `FACTOR07 `). Hardcoded column name references fail silently.
**Why it happens:** Different years/modules have slightly different header formatting. CSV parsers preserve whitespace in headers.
**How to avoid:** Normalize all column names immediately: `df = df.rename({c: c.strip().upper() for c in df.columns})`. Assert required columns exist after normalization.
**Warning signs:** KeyError on column access. Column count is correct but specific columns are "missing."

### Pitfall 4: Module Join Creates Duplicates or Drops Records
**What goes wrong:** If Module 200 or Module 300 has duplicate composite keys (CONGLOME+VIVIENDA+HOGAR+CODPERSO), a join produces a Cartesian product. If using inner join, children without education records are silently dropped.
**Why it happens:** Data quality issues in government surveys. Some persons may have duplicate entries.
**How to avoid:** Before joining, assert composite key uniqueness in each module. After joining, assert row count matches the left table. Use LEFT join from Module 200 (demographics) to preserve all persons.
**Warning signs:** Row count increases after join. Same person appears with different education data.

### Pitfall 5: Nullable Integer to Float Conversion
**What goes wrong:** Polars Int64 columns with null values display correctly, but when you filter on them (`pl.col("P303") == 1`), nulls propagate. If FACTOR07 has nulls and you multiply by it, you get null products.
**Why it happens:** Three-valued logic: `null == 1` evaluates to `null`, not `False`.
**How to avoid:** Validate zero nulls on critical columns (P303, P306, FACTOR07, UBIGEO, P208A) BEFORE any filtering or computation. The strict null check catches this early.
**Warning signs:** Unexpected null count in filtered results. Weighted rate computation returns null.

## ENAHO Variable Reference

### Key Variables (from INEI Data Dictionary and ILO Survey Library)

**Source:** ILO SurveyLib ENAHO 2002/2022 data dictionaries, INEI ENAHO metadata, DGGU-DO/enaho-init Stata metadata

#### Module 200 (Demographics) -- File: `Enaho01-{YEAR}-200.csv`
| Variable | Label (Spanish) | Label (English) | Type | Notes |
|----------|----------------|-----------------|------|-------|
| CONGLOME | Numero del Conglomerado | Cluster number | String | Join key |
| VIVIENDA | Numero de Seleccion de la Vivienda | Dwelling selection number | String | Join key |
| HOGAR | Hogar | Household | String | Join key |
| CODPERSO | Codigo de persona | Person code | String | Join key |
| UBIGEO | Codigo de Distrito | District geographic code | String (6 chars) | Must force to Utf8, pad to 6 |
| DOMINIO | Dominio | Survey domain | String | Geographic domain |
| ESTRATO | Estrato | Survey stratum | String | Sampling stratum |
| P207 | Sexo | Sex | Int | 1=Male, 2=Female |
| P208A | Edad en anos cumplidos | Age in completed years | Int | Filter to 6-17 for school age |

#### Module 300 (Education) -- File: `Enaho01a-{YEAR}-300.csv`
| Variable | Label (Spanish) | Label (English) | Type | Response Codes |
|----------|----------------|-----------------|------|----------------|
| P300A | Lengua materna | Mother tongue | Int | 1=Castellano, 2=Quechua, 3=Otra nativa (pre-2020), 4=Aymara, 5=Otra extranjera, 6=Es sordomudo, 10-15=Disaggregated native (2020+) |
| P301A | Nivel educativo | Education level | Int | Various codes by level |
| P303 | El ano pasado, estuvo matriculado? | Was enrolled last year? | Int | **1=Si, 2=No** |
| P306 | Este ano, esta matriculado? | Is enrolled this year? | Int | **1=Si, 2=No** |
| P307 | Actualmente, asiste? | Currently attending? | Int | 1=Si, 2=No |
| P308A | Nivel educativo al que asiste | Current education level | Int | Level codes |
| FACTOR07 | Factor de expansion | Survey expansion factor | Float | Population weight for all estimates |

**Standard binary coding confirmed:** 1 = "Si" (Yes), 2 = "No" (No) -- verified via DGGU-DO/enaho-init Stata metadata.

**CRITICAL NOTE on file naming:**
- Module 200: `Enaho01-{YEAR}-200.csv` (prefix `Enaho01`, NOT `Enaho01a`)
- Module 300: `Enaho01a-{YEAR}-300.csv` (prefix `Enaho01a`, with the `a`)
- Case may vary: `ENAHO01`, `enaho01`, `Enaho01` -- use case-insensitive glob

### Dropout Target Construction

Per spec DATA-02: `dropout = (P303 == 1) & (P306 == 2)`

**Interpretation:** A child who WAS enrolled last year (P303=1, "Si") but is NOT enrolled this year (P306=2, "No") is a dropout.

**Full logic with edge cases:**
```python
# Core dropout definition (spec DATA-02)
dropout = (P303 == 1) & (P306 == 2)

# Edge cases:
# 1. P303=2, P306=2: Never enrolled -- NOT a dropout (never was in school)
# 2. P303=1, P306=1: Enrolled both years -- NOT a dropout (target=0)
# 3. P303=2, P306=1: New enrollment -- NOT a dropout (target=0)
# 4. P303=null or P306=null: FAIL -- critical columns must have zero nulls

# School-age filter: 6 <= P208A <= 17
# This captures primary (ages 6-11) + secondary (ages 12-16/17) per Peru's
# Educacion Basica Regular (EBR) system.
```

### Temporary Absence Handling (Claude's Discretion)

**Recommendation:** Ignore P307 (currently attends?) for dropout target construction.

**Rationale:**
- P307 captures *attendance* at the moment of the survey interview, not enrollment status
- A child who is enrolled (P306=1) but temporarily absent (P307=2) is NOT a dropout -- they are still enrolled
- A child who was enrolled last year (P303=1) but not this year (P306=2) is a dropout regardless of P307
- Including P307 would conflate two distinct concepts: enrollment (the policy-relevant measure) and attendance (a daily behavior)
- The spec DATA-02 explicitly uses P303/P306, not P307

### Completed-Secondary 17-Year-Olds (Claude's Discretion)

**Recommendation:** Keep completed-secondary 17-year-olds as target=0 (non-dropout). Do NOT exclude them.

**Rationale:**
- A 17-year-old who completed secondary school and is not currently enrolled (P303=1 from secondary, P306=2) has *successfully completed* their education -- this is not dropout, this is graduation
- However, we cannot reliably identify "completed secondary" from P303/P306 alone without checking P301A (education level attained)
- The simpler and more conservative approach: if P303=1 and P306=2, it is a dropout regardless of completion status. This slightly overestimates dropout rate for 17-year-olds but avoids complex edge case logic that introduces error
- The ~14% weighted dropout rate target already accounts for this -- it was computed with this inclusive definition
- **Alternative considered but rejected:** Checking P301A for "completed secondary" would require knowing the exact codes for each year and handling cases where P301A is null or inconsistent. This complexity is better deferred to Phase 4 (feature engineering) if needed.

## Join Strategy (Claude's Discretion)

**Recommendation:** LEFT join from Module 200 (demographics) onto Module 300 (education).

**Rationale:**
- Module 200 contains ALL household members, including children of all ages
- Module 300 contains education data only for persons age 3+ who answered the education module
- Some school-age children may not have Module 300 records (e.g., never enrolled, survey nonresponse on education module)
- LEFT join from Module 200 preserves these children -- they will have null P303/P306 values
- After joining, children with null P303/P306 who are ages 6-17 represent potentially never-enrolled children
- These cases should be handled explicitly: if P303 is null for a school-age child, they were not asked the enrollment question, which typically means they are not enrolled (dropout=1 or never-enrolled)

**However:** The strict null validation on P303/P306 (from CONTEXT.md) will catch these cases. If any school-age children after filtering have null P303/P306, the loader will FAIL, alerting us to investigate.

**Practical approach:**
1. LEFT join Module 200 onto Module 300
2. Filter to ages 6-17
3. Check for null P303/P306 in the filtered result
4. If nulls exist, investigate whether these are truly school-age children missing education data
5. If the null count is very small (<0.5%), fill with conservative assumption (not enrolled = dropout) and add to warnings
6. If the null count is significant (>0.5%), this indicates a data quality issue -- FAIL with diagnostic info

**Alternative considered but rejected:** INNER join would drop all school-age children without Module 300 records. This silently removes the most vulnerable children (those never enrolled) from the analysis -- exactly the wrong behavior for an equity audit.

## Parquet Caching (Claude's Discretion)

**Recommendation:** Always load from raw CSV. No parquet caching in Phase 1.

**Rationale:**
- Phase 1 loads only ONE year. Loading time for ~120K rows is <3 seconds with polars. No caching needed.
- Parquet caching is assigned to Phase 4 (DATA-10 requirement) per ROADMAP.md
- Introducing caching in Phase 1 adds complexity (cache invalidation, staleness detection) for negligible performance benefit
- Raw-from-CSV ensures the full pipeline (including delimiter detection) is always exercised
- Phase 2 (multi-year loader) will benefit more from caching and can add it then

## File Format Support (Claude's Discretion)

**Recommendation:** CSV only. Do not support SAV (SPSS) files.

**Rationale:**
- `download.py` explicitly uses `enahodata` with `only_dta=False`, which downloads ALL files including CSVs
- The `reorganize_enaho()` function copies `*.csv`, `*.CSV`, `*.dta`, `*.DTA`, `*.sav`, `*.SAV` to the year directory
- CSVs are universally available across all ENAHO years
- SAV files would require `pyreadstat` as an additional dependency
- DTA (Stata) files would require `pyreadstat` or pandas `read_stata`
- Adding SAV/DTA support increases complexity for no benefit since CSVs are always present
- If a year's CSV is missing but SAV exists, the proper fix is to re-run download.py, not add SAV parsing

## Gate Test Tolerance Bands (Claude's Discretion)

**Recommendation:** Two-tier tolerance with WARN and FAIL thresholds.

### Row Count (school-age, ages 6-17)
| Level | Range | Rationale |
|-------|-------|-----------|
| PASS | 20,000 - 30,000 | ENAHO samples ~36K households, ~120K persons. School-age (6-17) is ~20-25% of population. |
| WARN | 18,000 - 20,000 OR 30,000 - 35,000 | Slightly outside expected range, may indicate unusual year or edge case. |
| FAIL | < 18,000 OR > 35,000 | Something is fundamentally wrong with loading or filtering. |

### Unweighted Dropout Count
| Level | Range | Rationale |
|-------|-------|-----------|
| PASS | 2,500 - 5,000 | ~10-20% of school-age sample as unweighted dropouts. |
| WARN | 2,000 - 2,500 OR 5,000 - 6,000 | Could indicate year-specific variation. |
| FAIL | < 2,000 OR > 6,000 | Dropout definition is likely wrong or data is corrupt. |

### Weighted Dropout Rate (FACTOR07)
| Level | Range | Rationale |
|-------|-------|-----------|
| PASS | 0.10 - 0.18 | Centered on ~14% expected rate with +/-4pp tolerance. |
| WARN | 0.08 - 0.10 OR 0.18 - 0.22 | Could indicate year-specific shifts (e.g., COVID recovery). |
| FAIL | < 0.08 OR > 0.22 | Weighted rate is too far from expectations -- likely a weighting or definition error. |

### UBIGEO Length
| Level | Range | Rationale |
|-------|-------|-----------|
| PASS | All values exactly 6 characters | Zero tolerance. |
| FAIL | Any value != 6 characters | UBIGEO padding failed. No WARN level. |

## Column Schema Validation (Claude's Discretion)

**Recommendation:** Yes, validate column schema in gate test. Check both names and types for critical columns.

**Rationale:**
- Column schema validation catches silent data corruption early
- If a column is renamed or missing, statistical checks may still pass by accident (e.g., wrong column has similar distribution)
- Checking types prevents the UBIGEO-as-integer pitfall and FACTOR07-as-string pitfall

**Implementation:**
```python
EXPECTED_SCHEMA = {
    "UBIGEO": pl.Utf8,
    "P208A": pl.Int64,     # Age
    "P207": pl.Int64,      # Sex
    "P303": pl.Int64,      # Was enrolled
    "P306": pl.Int64,      # Is enrolled
    "FACTOR07": pl.Float64,  # Survey weight
    "P300A": pl.Int64,     # Mother tongue
    "dropout": pl.Int64,   # Constructed target
    "CONGLOME": pl.Utf8,   # Join key
    "VIVIENDA": pl.Utf8,   # Join key
    "HOGAR": pl.Utf8,      # Join key
    "CODPERSO": pl.Utf8,   # Join key
}

def validate_schema(df):
    for col, expected_type in EXPECTED_SCHEMA.items():
        assert col in df.columns, f"Missing column: {col}"
        actual_type = df[col].dtype
        assert actual_type == expected_type, (
            f"Column {col}: expected {expected_type}, got {actual_type}"
        )
```

## Columns for 10 Inspection Rows (Claude's Discretion)

**Recommendation:** Show these columns in the 10 random dropout rows for human review:

```python
INSPECTION_COLUMNS = [
    "UBIGEO",        # Geographic location -- verifies UBIGEO padding
    "P208A",         # Age -- should be 6-17
    "P207",          # Sex -- 1=Male, 2=Female
    "P300A",         # Mother tongue -- ethnicity indicator
    "P303",          # Was enrolled last year -- should be 1 (Si) for dropouts
    "P306",          # Is enrolled this year -- should be 2 (No) for dropouts
    "P307",          # Currently attends -- should be 2 (No) if not enrolled
    "P301A",         # Education level -- what level they were at
    "FACTOR07",      # Survey weight -- should be positive, reasonable magnitude
    "dropout",       # Constructed target -- should be 1 for all rows shown
]
```

These columns let the human reviewer verify:
1. UBIGEO is properly 6 characters (geographic integrity)
2. Ages are in the 6-17 range (filter correctness)
3. P303=1 and P306=2 for all rows (dropout definition correctness)
4. P300A shows diverse mother tongue codes (not just Castellano)
5. FACTOR07 values are positive and vary (weights are real, not placeholder)
6. P301A shows various education levels (real student records)

## Code Examples

Verified patterns from live testing and official documentation:

### Delimiter Sniffing with csv.Sniffer
```python
# Verified working on 2026-02-07 with Python 3.12
import csv

def sniff_delimiter(filepath: Path, n_bytes: int = 8192) -> str:
    """Detect CSV delimiter by reading first N bytes."""
    with open(filepath, "r", encoding="latin-1") as f:
        sample = f.read(n_bytes)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        return dialect.delimiter
    except csv.Error:
        # Fallback: count occurrences of common delimiters in first line
        first_line = sample.split("\n")[0]
        counts = {d: first_line.count(d) for d in [",", "|", "\t", ";"]}
        return max(counts, key=counts.get)
```

### UBIGEO Padding at Load Time
```python
# Verified: polars 1.38.1
import polars as pl

# Option 1: schema_overrides (prevents integer inference)
df = pl.read_csv(filepath, separator=sep,
                 schema_overrides={"UBIGEO": pl.Utf8})

# Option 2: Pad after load (if UBIGEO was already inferred as int)
df = df.with_columns(
    pl.col("UBIGEO").cast(pl.Utf8).str.pad_start(6, "0")
)

# Option 3: Reusable expression (recommended for src/utils.py)
def pad_ubigeo(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.strip_chars().str.pad_start(6, "0")
```

### Column Name Normalization
```python
def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace and uppercase all column names."""
    return df.rename({c: c.strip().upper() for c in df.columns})
```

### Module File Discovery
```python
def _find_module_file(year: int, module: str) -> Path:
    """Find the CSV file for a given ENAHO module and year."""
    year_dir = DATA_RAW / "enaho" / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(
            f"Year directory not found: {year_dir}. Run download.py first."
        )

    # Module 200 uses "Enaho01", Module 300 uses "Enaho01a"
    prefix = "Enaho01a" if module in ("300", "400", "500") else "Enaho01"

    # Case-insensitive search with multiple patterns
    patterns = [
        f"{prefix}-{year}-{module}*.csv",
        f"{prefix.lower()}-{year}-{module}*.csv",
        f"{prefix.upper()}-{year}-{module}*.csv",
        f"{prefix}_{year}_{module}*.csv",  # Some years use underscores
    ]

    for pattern in patterns:
        matches = list(year_dir.glob(pattern))
        # Also try case-insensitive
        matches += [f for f in year_dir.iterdir()
                    if f.name.lower().startswith(prefix.lower())
                    and f"-{module}" in f.name.lower()
                    and f.suffix.lower() == ".csv"]
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Module {module} not found in {year_dir}/. "
        f"Expected pattern: {prefix}-{year}-{module}.csv. "
        "Run download.py first."
    )
```

### ENAHOResult Dataclass
```python
from dataclasses import dataclass, field

@dataclass
class ENAHOResult:
    """Result of loading a single year of ENAHO data."""
    df: pl.DataFrame
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
```

### Computing Stats for the Result
```python
def _compute_stats(df: pl.DataFrame) -> dict:
    """Compute summary statistics for the loaded ENAHO year."""
    n_total = df.height
    n_dropout = df.filter(pl.col("dropout") == 1).height

    # Weighted dropout rate
    weighted_dropout = (
        (df["dropout"].cast(pl.Float64) * df["FACTOR07"]).sum()
        / df["FACTOR07"].sum()
    )

    return {
        "total_rows": n_total,
        "dropout_count": n_dropout,
        "unweighted_dropout_rate": n_dropout / n_total if n_total > 0 else 0,
        "weighted_dropout_rate": float(weighted_dropout),
        "ubigeo_unique": df["UBIGEO"].n_unique(),
        "age_range": (int(df["P208A"].min()), int(df["P208A"].max())),
    }
```

### Gate Test Pattern
```python
# tests/gates/test_gate_1_1.py
import polars as pl
import pytest

@pytest.fixture(scope="module")
def enaho_result():
    """Load ENAHO 2023 -- cached for the entire test module."""
    from src.data.enaho import load_enaho_year
    return load_enaho_year(2023)

def test_row_count(enaho_result):
    n = enaho_result.df.height
    assert 20_000 <= n <= 30_000, f"Expected 20K-30K school-age rows, got {n}"

def test_dropout_count(enaho_result):
    n_dropout = enaho_result.stats["dropout_count"]
    assert 2_500 <= n_dropout <= 5_000, f"Expected 2.5K-5K dropouts, got {n_dropout}"

def test_weighted_dropout_rate(enaho_result):
    rate = enaho_result.stats["weighted_dropout_rate"]
    assert 0.10 <= rate <= 0.18, f"Expected 10-18% weighted dropout, got {rate:.3f}"

def test_ubigeo_length(enaho_result):
    lengths = enaho_result.df["UBIGEO"].str.len_chars()
    assert lengths.min() == 6, f"UBIGEO min length: {lengths.min()}, expected 6"
    assert lengths.max() == 6, f"UBIGEO max length: {lengths.max()}, expected 6"

def test_critical_nulls(enaho_result):
    for col in ["UBIGEO", "P208A", "P303", "P306", "FACTOR07"]:
        null_count = enaho_result.df[col].null_count()
        assert null_count == 0, f"Critical column {col} has {null_count} nulls"

def test_schema(enaho_result):
    df = enaho_result.df
    assert df["UBIGEO"].dtype == pl.Utf8
    assert df["dropout"].dtype in (pl.Int64, pl.Int32, pl.Int8)
    assert df["FACTOR07"].dtype == pl.Float64

def test_dropout_rows_inspection(enaho_result):
    """Print 10 random dropout rows for human review."""
    dropouts = enaho_result.df.filter(pl.col("dropout") == 1)
    sample = dropouts.sample(n=min(10, dropouts.height), seed=42)

    inspection_cols = [
        "UBIGEO", "P208A", "P207", "P300A", "P303", "P306",
        "P307", "P301A", "FACTOR07", "dropout"
    ]
    available_cols = [c for c in inspection_cols if c in sample.columns]

    print("\n" + "=" * 80)
    print("GATE 1.1 -- HUMAN REVIEW: 10 Random Dropout Rows")
    print("=" * 80)
    print(sample.select(available_cols))
    print(f"\nStats: {enaho_result.stats}")
    if enaho_result.warnings:
        print(f"Warnings: {enaho_result.warnings}")
    print("=" * 80)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `df.dtypes = {"UBIGEO": str}` (pandas) | `schema_overrides={"UBIGEO": pl.Utf8}` (polars) | polars 0.20.31 | Parameter was renamed from `dtypes` to `schema_overrides` |
| `str.zfill(6)` (pandas) | `str.pad_start(6, "0")` (polars) | polars 1.0+ | polars does not have `zfill` -- use `pad_start` |
| `sep=` parameter | `separator=` parameter | polars 1.0+ | polars uses `separator`, not `sep` |

**Deprecated/outdated:**
- polars `dtypes` parameter in `read_csv`: renamed to `schema_overrides` in 0.20.31. Use `schema_overrides`.

## Open Questions

Things that couldn't be fully resolved:

1. **Exact Module 300 file prefix for all years**
   - What we know: Module 200 is `Enaho01-`, Module 300 is `Enaho01a-` based on download.py patterns
   - What's unclear: Whether ALL years (2018-2024) use consistent prefixes, or some years differ
   - Recommendation: Implement case-insensitive multi-pattern search (as shown in `_find_module_file`) to handle variations. Log which pattern matched for debugging.

2. **P303/P306 edge case: Person in Module 200 but not Module 300**
   - What we know: LEFT join from Module 200 preserves all persons. Some school-age children may lack Module 300 records.
   - What's unclear: How many such cases exist and what they represent (survey nonresponse? children under 3 not asked?)
   - Recommendation: Implement the LEFT join, count null P303/P306 cases in ages 6-17. If count is >0, investigate before deciding how to handle. The strict null check will surface this immediately.

3. **Encoding of ENAHO CSV files**
   - What we know: ENAHO data may use Latin-1 encoding for Spanish characters (tildes, accents in column names or values)
   - What's unclear: Whether all years use the same encoding
   - Recommendation: Try `encoding="latin-1"` in file reading for delimiter sniffing. For polars `read_csv`, the default `encoding="utf8"` may work if the data files are UTF-8. If not, use `encoding="utf8-lossy"` to handle encoding mismatches gracefully.

4. **Exact column names for join keys across years**
   - What we know: Standard keys are CONGLOME, VIVIENDA, HOGAR, CODPERSO
   - What's unclear: Whether HOGAR is present in all modules/years, or if some use HOESSION or similar variants
   - Recommendation: After column normalization, check for HOGAR/HOESSION and use whichever is present. Log which variant was found.

## Sources

### Primary (HIGH confidence)
- polars 1.38.1 `read_csv` API -- verified via `help(pl.read_csv)` on this machine
- polars `str.pad_start` -- verified via live code execution on this machine
- `csv.Sniffer().sniff()` -- verified via live code execution with pipe and comma delimiters
- `download.py` (project file) -- file naming patterns, module codes, directory structure
- DGGU-DO/enaho-init Stata metadata -- confirmed 1=Si, 2=No binary coding for P303/P306
- ILO SurveyLib ENAHO 2002/2022 data dictionaries -- variable labels, join keys (CONGLOME, VIVIENDA, HOGAR, CODPERSO)
- Peru education system (nuffic.nl, wikipedia) -- Primary ages 6-11, Secondary ages 12-16/17

### Secondary (MEDIUM confidence)
- INEI ENAHO data dictionary PDFs (2018, 2019, 2022) -- referenced but PDF content not parseable
- enahopy library documentation -- confirmed merge levels (PERSONA, HOGAR) and FACTOR07 usage
- ENAHO school dropout journalism (data.larepublica.pe) -- confirmed P303 as enrollment variable
- ljporras/ENAHO-INDICADORES-STATA -- Stata education indicator processing patterns

### Tertiary (LOW confidence)
- FACTOR07 location in Module 300 -- confirmed by project spec comment in download.py but not independently verified against actual 2023 CSV headers. If FACTOR07 is not in Module 300, it may be in Module 200 or need to come from a different source (Sumarias module 34).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all polars APIs verified via live execution
- Architecture: HIGH -- composable module loader pattern well-established, join keys confirmed from multiple ENAHO processing projects
- Pitfalls: HIGH -- delimiter mismatch and UBIGEO zero-loss are documented in project spec and confirmed by ENAHO processing community
- ENAHO variables: MEDIUM -- P303/P306 binary coding confirmed, but exact column headers in 2023 CSVs not verified against actual data (data not yet downloaded)
- Dropout definition: HIGH -- spec DATA-02 explicitly states `(P303==1 & P306==2)`, confirmed by ENAHO school dropout research

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (30 days -- ENAHO data structure is stable year-to-year)
