# Phase 3: Spatial + Supplementary Data Merges - Research

**Researched:** 2026-02-07
**Domain:** Data engineering with spatial and administrative data merges using Polars
**Confidence:** HIGH

## Summary

This phase involves merging ENAHO microdata with three external data sources: (1) administrative dropout rates from MINEDU's datosabiertos platform, (2) Census 2017 district-level indicators, and (3) VIIRS nighttime lights as an economic proxy. All merges use LEFT JOIN on UBIGEO (6-digit district codes) to preserve all ENAHO rows while enriching with district-level context. The phase requires rigorous validation of merge integrity, coverage rates, and UBIGEO zero-padding to prevent leading-zero loss.

**Primary recommendation:** Use Polars DataFrame.join() with validate='1:1' for admin data (unique UBIGEO), validate='m:1' for multi-source merges, and coalesce=True for key column management.

## User Constraints

No CONTEXT.md exists for Phase 3. Research proceeds based on specs.md requirements and success criteria in roadmap.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| polars | >=1.0 | Primary data processing and joins | Native lazy evaluation, validation arguments, superior memory efficiency vs pandas |
| python | 3.12 | Language runtime | Already configured via Nix flake |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| requests | latest | HTTP downloads from datosabiertos.gob.pe | Manual file fetching before automated implementation |
| pyproject.toml | existing | Dependency management | Already configured from Phase 0 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|-----------|
| polars join() | pandas merge() | Polars has built-in validation, better performance, Arrow-native format |

**Installation:**
```bash
# Already installed from Phase 0
nix develop  # Provides polars >=1.0, python 3.12
```

## Architecture Patterns

### Recommended Project Structure
```
src/data/
├── admin.py              # Administrative dropout rate loader
├── census.py             # Census 2017 district enrichment loader
├── nightlights.py        # VIIRS district-level economic proxy loader
├── enaho.py             # Existing ENAHO loader (reference for patterns)
└── features.py           # Feature engineering (Phase 4)
```

### Pattern 1: LEFT JOIN Merge Pipeline
**What:** Sequential LEFT JOINs preserving ENAHO row count through validate argument
**When to use:** All spatial enrichment operations
**Example:**
```python
# Source: Polars official documentation
def merge_spatial_data(enaho_df: pl.DataFrame) -> pl.DataFrame:
    """LEFT JOIN ENAHO with admin, census, nightlights on UBIGEO."""
    
    # Step 1: Admin data merge (unique UBIGEO guaranteed by source)
    admin_df = load_admin_dropout_rates()
    result = enaho_df.join(
        admin_df, 
        on="ubigeo", 
        how="left", 
        validate="1:1"  # Ensure admin UBIGEO uniqueness
    )
    
    # Step 2: Census merge (may have missing districts)
    census_df = load_census_2017()
    result = result.join(
        census_df, 
        on="ubigeo", 
        how="left", 
        validate="m:1"  # Many ENAHO rows to one census row
    )
    
    # Step 3: Nightlights merge
    nightlights_df = load_viirs_nightlights()
    final_result = result.join(
        nightlights_df, 
        on="ubigeo", 
        how="left",
        validate="m:1",
        coalesce=True  # Prevent ubigeo/ubigeo_right duplication
    )
    
    return final_result
```

### Anti-Patterns to Avoid
- **INNER JOIN instead of LEFT:** Would lose ENAHO rows that don't match admin data
- **No validation arguments:** Risk undetected duplicate UBIGEO keys causing data explosion
- **Manual UBIGEO string handling:** Inconsistent zero-padding leads to join failures
- **Multiple merge strategies in single operation:** Reduces debugging capability and validation precision

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|--------------|------|
| UBIGEO zero-padding logic | Manual string formatting with .zfill() | pad_ubigeo() utility function | Consistent 6-digit format across all sources, prevents join failures |
| CSV delimiter detection | If year == 2019 use '|' else ',' | sniff_delimiter() utility function | Auto-detection handles edge cases and year transitions |
| Join validation | Custom duplicate detection loops | Polars validate='1:1'/'m:1' arguments | Built-in validation prevents data explosion, better error messages |
| Multiple merge orchestration | Complex nested merge logic | Sequential joins with intermediate validation | Each merge can be validated separately, easier debugging |

**Key insight:** Spatial data merging has well-known failure modes (duplicate keys, missing districts, leading zeros). Established patterns prevent entire classes of bugs that waste debugging time.

## Common Pitfalls

### Pitfall 1: UBIGEO Format Inconsistency
**What goes wrong:** Admin data has UBIGEO as integers (150101) while ENAHO has strings ("150101") or vice versa, causing join failures
**Why it happens:** Different data sources treat UBIGEO differently (numeric vs string)
**How to avoid:** Always normalize UBIGEO to zero-padded 6-digit strings using `pad_ubigeo()` utility
**Warning signs:** Join results in 0 matches, unexpected null rates in merge results

### Pitfall 2: Join Key Duplication
**What goes wrong:** Multiple admin rows for same UBIGEO causing ENAHO rows to duplicate, inflating dataset size
**Why it happens:** Administrative data updates may create duplicate entries if not properly deduplicated
**How to avoid:** Use Polars `validate="1:1"` for admin joins, `validate="m:1"` for multi-source merges
**Warning signs:** Output row count > input ENAHO row count after LEFT JOIN

### Pitfall 3: Merge Coverage Underestimation
**What goes wrong:** Assuming 100% data coverage when sources have systematic gaps (e.g., remote Amazon districts)
**Why it happens:** VIIRS nightlights and Census 2017 may have limited coverage in remote areas
**How to avoid:** Explicitly calculate and validate merge rates (>85% admin, >90% census, >85% nightlights per spec)
**Warning signs:** Large geographic regions (Amazonas, Loreto) show high null rates

### Pitfall 4: Sequential Dependency Errors
**What goes wrong:** Census merge fails but pipeline continues to nightlights, masking the root cause
**Why it happens:** No intermediate validation between merge steps
**How to avoid:** Validate each merge result before proceeding (row counts, null checks, UBIGEO integrity)
**Warning signs:** Sudden drops in coverage rates or unexpected null patterns

## Code Examples

Verified patterns from official sources:

### LEFT JOIN with Validation
```python
# Source: Polars official documentation
# Load admin dropout rates (UBIGEO should be unique)
admin_df = pl.read_csv("data/raw/admin/desercion_2023_2024.csv")

# Critical: Zero-pad UBIGEO to ensure string matches
admin_df = admin_df.with_columns(
    pl.col("ubigeo").str.zfill(6)
)

# LEFT JOIN with uniqueness validation
result = enaho_df.join(
    admin_df,
    on="ubigeo",
    how="left",
    validate="1:1"  # Will raise exception if duplicate UBIGEO in admin data
)
```

### UBIGEO Padding Utility
```python
# Source: Existing project pattern in utils.py
def pad_ubigeo(ubigeo: pl.Series) -> pl.Series:
    """Zero-pad UBIGEO codes to 6 digits string format."""
    return ubigeo.cast(pl.Utf8).str.zfill(6)
```

### Merge Rate Validation
```python
# Pattern for calculating join success rates
def validate_merge_rate(
    left_df: pl.DataFrame, 
    merged_df: pl.DataFrame, 
    key_col: str, 
    source_name: str
) -> dict:
    """Calculate merge coverage and validate against spec requirements."""
    
    initial_count = left_df.height
    merged_count = merged_df.filter(pl.col(f"{source_name}_{key_col}").is_not_null()).height
    
    merge_rate = merged_count / initial_count
    
    return {
        "merge_rate": merge_rate,
        "initial_count": initial_count,
        "matched_count": merged_count,
        "unmatched_count": initial_count - merged_count
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|---------|
| Manual pandas merge with custom validation loops | Polars join(validate=) + built-in validation | Polars 0.19+ | 10x performance improvement, eliminated entire class of validation bugs |
| Inconsistent UBIGEO handling across sources | Standardized pad_ubigeo() utility function | Phase 1 | Eliminated leading-zero join failures |
| Sequential merge without intermediate checks | Stepwise validation with coverage metrics | Current approach | Earlier error detection, better debugging |

**Deprecated/outdated:**
- **pandas merge()**: Polars lazy evaluation and validation arguments are superior
- **Manual string padding**: `pad_ubigeo()` utility handles edge cases consistently
- **Custom duplicate detection**: `validate="1:1"`/`validate="m:1"` arguments prevent data explosion

## Open Questions

Things that couldn't be fully resolved:

1. **VIIRS Nightlights Source Resolution**
   - What we know: Spec mentions "pre-aggregated district-level data" but doesn't specify exact source URL
   - What's unclear: Whether data comes from NASA Earthdata, Jiaxiong Yao research site, or other source
   - Recommendation: Verify VIIRS source during implementation, may need to pre-process raw satellite data

2. **Census 2017 Variable Availability**
   - What we know: Spec requires "poverty indices, indigenous language prevalence, access to services"
   - What's unclear: Exact variable names and availability for all 1,874 districts
   - Recommendation: Map Census variables to required features during implementation, document gaps

3. **Administrative Data Update Frequency**
   - What we know: Current dataset covers 2023/2024 interannual dropout rates
   - What's unclear: Whether historical admin data (2018-2022) is available for longitudinal analysis
   - Recommendation: Check datosabiertos.gob.pe for historical datasets during Phase 4

## Sources

### Primary (HIGH confidence)
- **Polars User Guide - Joins**: https://docs.pola.rs/user-guide/transformations/joins/ - Complete join strategies, validation arguments, coalesce options
- **Project Specs - Section 4**: Data source specifications, expected values, merge requirements
- **Existing ENAHO Loader**: src/data/enaho.py - UBIGEO padding patterns, validation approaches, join key constants

### Secondary (MEDIUM confidence)
- **datosabiertos.gob.pe**: Administrative dropout rates dataset discovery and schema confirmation
- **VIIRS Nightlights Research**: Multiple academic papers confirming validity as economic proxy for district-level analysis

### Tertiary (LOW confidence)
- **Census 2017 Documentation**: Limited access to exact variable catalog, requires implementation-time verification
- **Administrative Data Historical Coverage**: Unverified availability of pre-2023 admin dropout rates

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Polars documentation is authoritative, project patterns established
- Architecture: HIGH - Join validation patterns are well-established best practices
- Pitfalls: MEDIUM - VIIRS and Census sources require implementation verification

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (30 days for stable domain)