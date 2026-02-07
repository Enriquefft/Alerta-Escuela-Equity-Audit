# Phase 1: ENAHO Single-Year Loader - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Load a single year (2023) of ENAHO household survey microdata into a clean polars DataFrame with proper delimiter detection, UBIGEO geographic code padding, school-age filtering (ages 6-17), and binary dropout target construction. Gate test 1.1 validates row counts, dropout rate, UBIGEO integrity, and null checks. 10 random dropout rows printed for human inspection.

</domain>

<decisions>
## Implementation Decisions

### Dropout target construction
- Enrollment-based definition: dropout = school-age child not currently enrolled (Claude determines exact ENAHO variables and handles attendance edge cases)
- Primary + secondary only — no pre-primary/initial education ages
- Claude decides whether to exclude completed-secondary 17-year-olds or keep as target=0
- Claude sets gate test tolerance bands for the ~14% weighted dropout rate (warn vs fail thresholds)

### Data validation & reporting
- Minimal output: key stats (total rows, dropout count, weighted rate) plus 10 random dropout rows for human inspection
- Strict null handling on critical columns: UBIGEO, age, enrollment status, and survey weight (FACTOR07) must have zero nulls — fail if violated
- Other columns: report null counts but don't fail
- Claude decides whether gate test also validates column schema (names + types) beyond the statistical assertions

### Loader interface design
- Data directory resolved from project root via pyproject.toml walk-up (Phase 0 pattern) — not configurable
- Return a named tuple or dataclass with `.df`, `.stats`, `.warnings` — not a bare DataFrame
- Composable architecture: separate load functions per ENAHO module (load_module_200, load_module_300, etc.) that compose into `load_enaho_year()`
- Claude decides on parquet caching strategy (cache vs always-from-raw)

### ENAHO module handling
- Delimiter detection via sniffing first N lines (frequency analysis), not hardcoded
- Missing module files: fail immediately with clear message ("Module X not found in data/raw/. Run download.py first.") — no auto-download
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

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. User wants clean, minimal output and strict validation on critical columns. The composable module architecture is important for testability.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-enaho-single-year-loader*
*Context gathered: 2026-02-07*
