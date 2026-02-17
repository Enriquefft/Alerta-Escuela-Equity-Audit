# Phase 4: Feature Engineering + Descriptive Statistics -- Research

**Researched:** 2026-02-08
**Phase:** 04-feature-engineering-descriptive-statistics
**Plans:** 04-01 (Feature engineering pipeline), 04-02 (Descriptive statistics + export + gate test 1.5)

---

## 1. Current State of the Codebase

### 1.1 Input: full_dataset.parquet

**Location:** `data/processed/full_dataset.parquet`
**Shape:** 150,135 rows x 27 columns
**Built by:** `src/data/build_dataset.py` (calls `load_all_years()` then `merge_spatial_data()`)

Current schema (all uppercase column names):

| Column | Type | Null Count | Notes |
|--------|------|------------|-------|
| CONGLOME | String | 0 | Household cluster ID |
| VIVIENDA | String | 0 | Dwelling ID |
| HOGAR | String | 0 | Household ID |
| CODPERSO | String | 0 | Person ID |
| UBIGEO | String | 0 | 6-char district code |
| DOMINIO | Int8 | 0 | 1-8, natural region coding |
| ESTRATO | Int8 | 0 | 1-8, urban/rural stratum |
| P207 | Float64 | 0 | Sex: 1=Male, 2=Female |
| P208A | Float64 | 0 | Age: 6-17 |
| P300A | Float64 | 0 | Mother tongue (raw) |
| P301A | Float64 | 0 | Education level |
| P303 | Int64 | 0 | Enrolled last year (1=yes, 2=no) |
| P306 | Int64 | 0 | Enrolled this year (1=yes, 2=no) |
| P307 | Float64 | 28,343 (18.88%) | Currently attending |
| FACTOR07 | Float64 | 0 | Survey expansion weight |
| dropout | Boolean | 0 | Target: P303==1 & P306==2 |
| year | Int32 | 0 | 2018-2023 |
| p300a_original | Float64 | 0 | Raw P300A (preserves 10-15) |
| p300a_harmonized | Int64 | 0 | Codes 10-15 collapsed to 3 |
| admin_primaria_rate | Float64 | 0 | District admin primaria dropout % |
| admin_secundaria_rate | Float64 | 2,290 (1.53%) | District admin secundaria dropout % |
| census_poverty_rate | Float64 | 0 | District poverty rate 0-100 |
| census_indigenous_lang_pct | Float64 | 0 | District indigenous language % |
| census_literacy_rate | Float64 | 0 | District literacy rate % |
| census_electricity_pct | Float64 | 0 | District electricity access % |
| census_water_access_pct | Float64 | 0 | District water access % |
| nightlights_mean_radiance | Float64 | 6,150 (4.10%) | District mean radiance |

### 1.2 Source Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `src/data/enaho.py` | Single/multi-year ENAHO loader | `load_enaho_year()`, `load_all_years()`, `harmonize_p300a()` |
| `src/data/merge.py` | Spatial data merge pipeline | `merge_spatial_data()` |
| `src/data/build_dataset.py` | Build script for full_dataset.parquet | `main()` |
| `src/data/admin.py` | Admin dropout rate loader | `load_admin_dropout_rates()` |
| `src/data/census.py` | Census 2017 loader | `load_census_2017()` |
| `src/data/nightlights.py` | VIIRS nightlights loader | `load_viirs_nightlights()` |
| `src/utils.py` | Shared utilities | `find_project_root()`, `pad_ubigeo()`, `sniff_delimiter()` |
| `src/data/__init__.py` | Package re-exports | All loader functions |

### 1.3 New Files to Create

| File | Purpose |
|------|---------|
| `src/data/features.py` | Feature engineering pipeline (spec requires this exact path) |
| `tests/gates/test_gate_1_5.py` | Gate test for features + descriptive stats |
| `data/exports/descriptive_tables.json` | First JSON export (M4 schema) |
| `data/exports/figures/` | Directory for 7 matplotlib PNGs |
| `data/processed/enaho_with_features.parquet` | Feature matrix with all 19+ features |

### 1.4 Existing Patterns to Follow

- **Result dataclass pattern:** All loaders return `@dataclass` containers (df + stats + warnings). Feature engineering should follow this.
- **Uppercase column names:** Pipeline maintains uppercase throughout (UBIGEO, FACTOR07, P208A). The spec's feature names are lowercase (age, es_mujer). Both conventions must coexist: uppercase for raw ENAHO columns, lowercase for engineered features.
- **sys.path.insert(0, "src"):** All scripts and tests use this for imports.
- **Polars-first:** All data processing in polars. Only convert to pandas at sklearn boundary.
- **Logging:** All modules use `logging.getLogger(__name__)`.
- **Gate test pattern:** Print detailed results, assert on thresholds, include summary section.

---

## 2. Feature Engineering: Detailed Column-by-Column Plan

### 2.1 Features Available from Current Dataset (No Additional Data Needed)

These features can be derived directly from full_dataset.parquet columns:

| Spec Feature Name | Source Column | Construction | Type | Notes |
|-------------------|--------------|--------------|------|-------|
| `age` | P208A | Direct cast to Int64 | int | Already 6-17, zero nulls |
| `is_secundaria_age` | P208A | `1 if P208A >= 12 else 0` | binary | User-requested addition (primaria/secundaria split) |
| `es_mujer` | P207 | `1 if P207 == 2 else 0` | binary | P207: 1=Male, 2=Female |
| `lang_castellano` | P300A | `1 if P300A == 4 else 0` | binary | 132,825 rows (88.5%) |
| `lang_quechua` | P300A | `1 if P300A == 1 else 0` | binary | 11,230 rows (7.5%) |
| `lang_aimara` | P300A | `1 if P300A == 2 else 0` | binary | 518 rows (0.3%) |
| `lang_other_indigenous` | p300a_harmonized | `1 if harmonized == 3 else 0` | binary | 4,931 rows (3.3%) |
| `lang_foreign` | P300A | `1 if P300A in (6, 7) else 0` | binary | 301 rows (0.2%) |
| `rural` | ESTRATO | `1 if ESTRATO >= 6 else 0` | binary | INEI: 1-5=urban, 6-8=rural |
| `region_natural` | DOMINIO | Map: {1,2,3,8}=Costa, {4,5,6}=Sierra, {7}=Selva | categorical | Per-household, handles cross-region departments |
| `region_natural_encoded` | DOMINIO | One-hot or ordinal encoding | numeric | For model input |
| `department` | UBIGEO | `UBIGEO[:2]` | categorical (25 values) | Not a model feature, analysis key |
| `district_dropout_rate_admin` | admin_primaria_rate | Direct rename; or combine primaria+secundaria based on student age | float | 0 nulls for primaria, 1.53% for secundaria |
| `nightlight_intensity` | nightlights_mean_radiance | Direct rename | float | 4.10% nulls |
| `poverty_index` | census_poverty_rate | Direct rename | float | 0-100 scale, 0 nulls |
| `poverty_quintile` | census_poverty_rate + FACTOR07 | Weighted quintile breaks | int (1-5) | 5 groups with ~20% weighted population each |

### 2.2 Features Requiring Additional Module Loading

These features exist in ENAHO raw modules but are NOT in the current POOLED_COLUMNS:

| Spec Feature Name | Source Module | Column | Availability | Null Issues |
|-------------------|-------------|--------|--------------|-------------|
| `es_peruano` | Module 200 | P209 | Ages 12+ only; P209=5 means foreign-born | 100% null for ages 6-11; only 27 foreigners among school-age in 2023 |
| `has_disability` | Module 200 | P210 | All ages; 1=yes, 2=no | 0% null; ~11% have disability in 2023 |
| `is_working` | Module 500 | P501 | Only 33.6% of school-age match Module 500 | Employment module is primarily for 14+ |
| `juntos_participant` | Module 700 | P710_04 | Household-level; join on (CONGLOME, VIVIENDA, HOGAR) | 4,098 receiving households in 2023 |
| `log_income` | Sumaria | INGHOG1D | Household-level; available all years | Some very low values (min=15 soles/month) |
| `parent_education_years` | Module 200+300 | P203 (relationship) + P301A | Requires linking child to parent within household | Complex: need P203=1 or P203=2 for parent |
| `school_student_teacher_ratio` | ESCALE data | Not downloaded | Not available in current raw data | Spec lists "Escolar merge" but no ESCALE loader exists |

### 2.3 Strategy for Missing Features

**Approach: Expand POOLED_COLUMNS or Load Raw Modules in features.py**

The current pipeline pools only ~20 columns per year for schema consistency. To add the missing features, there are two options:

**Option A (Recommended): Load additional columns in features.py**
- `features.py` reads full_dataset.parquet AND loads additional raw module data per year
- Joins on (CONGLOME, VIVIENDA, HOGAR, CODPERSO) or (CONGLOME, VIVIENDA, HOGAR) for household-level
- Keeps the existing pipeline untouched

**Option B: Expand POOLED_COLUMNS in enaho.py**
- Add P209, P210 to POOLED_COLUMNS
- Requires re-running `load_all_years()` and `merge_spatial_data()`
- Cleaner but changes Phase 1-3 outputs

**Recommendation: Option A** -- load supplementary columns in features.py. This isolates feature engineering from the data loading pipeline and avoids re-running the expensive multi-year load.

For each missing feature:

1. **es_peruano**: Load P209 from Module 200 for each year. For ages 6-11 (P209 is null/not asked), assume Peruvian (only ~0.1% of 12-17 are foreign-born; for younger children it would be even lower). For ages 12+, set `es_peruano = 0 if P209 == 5 else 1`. Very few foreign-born children (27 in 2023); feature will have very low variance but is required by spec.

2. **has_disability**: Load P210 from Module 200. Map: `1 if P210 == 1 else 0`. Zero nulls. ~11% prevalence is reasonable (includes learning, hearing, visual disabilities per ENAHO definition).

3. **is_working**: Load P501 from Module 500. Only matches ~33.6% of school-age (primarily 14+). For unmatched children (ages 6-13), assume not working (`is_working = 0`). This is defensible: child labor under 14 exists but is not captured by Module 500's employment questions.

4. **juntos_participant**: Load P710_04 from Module 700. Household-level: join on (CONGLOME, VIVIENDA, HOGAR). If household not in Module 700, assume not receiving JUNTOS (= 0). Module 700 is household-level with complete coverage for sampled households.

5. **log_income**: Load INGHOG1D from sumaria module. Household-level: join on (CONGLOME, VIVIENDA, HOGAR). Compute `log(INGHOG1D + 1)`. Available all 6 years (sumaria exists for 2018-2023).

6. **parent_education_years**: Load P203 (relationship to head) from Module 200 and P301A (education level) from Module 300. For each child (P203 in {3=hijo, 5=nieto}), find the household head (P203=1) or spouse (P203=2) and get their P301A. Convert P301A to years: {1:0, 2:0, 3:6, 4:6, 5:11, 6:11, 7:14, 9:16, 12:18}. If no parent found, use median imputation.

7. **school_student_teacher_ratio**: ESCALE data not available. This feature will be null/missing. Document in code that it requires ESCALE data from MINEDU's school-level database. Set to null and flag in warnings. The model can run with 18 features instead of 19; this is documented as a known gap.

### 2.4 Z-Score Standardization (User Decision)

Per CONTEXT.md: "Standardize (z-score) all district-level features from the spatial merge."

Features to z-score standardize:
- `district_dropout_rate_admin`
- `nightlight_intensity`
- `poverty_index` (census_poverty_rate)
- census_indigenous_lang_pct (if kept as feature)
- census_literacy_rate (if kept as feature)
- census_electricity_pct (if kept as feature)
- census_water_access_pct (if kept as feature)

Method: `(x - mean) / std` using polars. Handle nulls by computing mean/std on non-null values, then standardizing only non-null entries.

Store both raw and standardized versions:
- Raw: for descriptive stats and interpretability
- Standardized: for model input (suffix `_z` or create a separate model feature matrix)

### 2.5 Feature Name Conventions

The spec uses lowercase feature names (age, es_mujer, rural) while the existing pipeline uses uppercase ENAHO columns (P208A, P207, ESTRATO). The feature engineering step bridges this:

```
Full dataset (uppercase ENAHO) --> features.py --> Feature matrix (lowercase spec names)
```

The output DataFrame should have BOTH:
- Uppercase columns preserved for traceability (UBIGEO, FACTOR07, year, p300a_original, p300a_harmonized)
- Lowercase engineered features added (age, es_mujer, rural, etc.)

This matches the spec's META_COLUMNS concept: `['ubigeo', 'year', 'factor07', 'p300a_original', 'department', 'dropout']`

---

## 3. Descriptive Statistics: Computation Details

### 3.1 Survey-Weighted Dropout Rate Computation

Standard pattern in polars:
```python
weighted_rate = (
    df.group_by("group_col").agg(
        weighted_rate=(pl.col("dropout").cast(pl.Float64) * pl.col("FACTOR07")).sum()
        / pl.col("FACTOR07").sum()
    )
)
```

### 3.2 Confidence Intervals

**Recommendation: Use statsmodels DescrStatsW (linearization)**

Tested and working. For each subgroup:
```python
from statsmodels.stats.weightstats import DescrStatsW
d = DescrStatsW(dropout_array, weights=weight_array)
mean = d.mean
se = d.std_mean  # standard error incorporating weights
ci_lower = max(0, mean - 1.96 * se)
ci_upper = min(1, mean + 1.96 * se)
```

This uses the Kish effective sample size approximation: `n_eff = (sum(w))^2 / sum(w^2)`.

**Tested baselines:**
- Awajun 2020+: rate=0.2047, 95% CI=[0.2018, 0.2076], n=738, n_eff=374
- Castellano all: rate=0.1526, 95% CI=[0.1525, 0.1527], n=132,825, n_eff=66,058

CIs are very tight because FACTOR07 weights are expansion factors (large), making the effective sample size large. This is expected for ENAHO.

### 3.3 Breakdowns Required

**DESC-01: Dropout by Mother Tongue**

Language groups per CONTEXT.md decision (Top 5 + Other):
| Group | Code | n (unweighted) | Weighted Rate |
|-------|------|----------------|---------------|
| Castellano | p300a_original == 4 | 132,825 | 0.1526 |
| Quechua | p300a_original == 1 | 11,230 | 0.2038 |
| Aymara | p300a_original == 2 | 518 | 0.1834 |
| Awajun | p300a_original == 11 | 738 | 0.2047 |
| Ashaninka | p300a_original == 10 | 576 | 0.1831 |
| Other indigenous | p300a_harmonized == 3, not Awajun/Ashaninka | 3,617 | 0.2313 |
| Foreign | P300A in (6, 7) | 301 | 0.1581 |

Also compute binary: indigenous (any non-4, non-6, non-7) vs Castellano (code 4) as headline stat.

Awajun by year (2020+) for success criterion check:
| Year | n | Weighted Rate |
|------|---|---------------|
| 2020 | 141 | 0.3386 |
| 2021 | 174 | 0.1740 |
| 2022 | 217 | 0.1235 |
| 2023 | 206 | 0.2054 |

Combined 2020+: 738 rows, 0.2047 weighted rate. **Exceeds 18% threshold.**

**DESC-02: Dropout by Sex x Education Level**

Groups: male_primaria (P207=1, age 6-11), female_primaria (P207=2, age 6-11), male_secundaria (P207=1, age 12-17), female_secundaria (P207=2, age 12-17).

**DESC-03: Dropout by Rural/Urban, Region, Poverty Quintile**

- Rural: ESTRATO >= 6 (61,388 rows, rate=0.1788) vs Urban: ESTRATO 1-5 (88,747 rows, rate=0.1495)
- Region: Costa/Sierra/Selva from DOMINIO mapping
  - Costa (DOMINIO 1,2,3,8): 56,341 rows, rate=0.1443
  - Sierra (DOMINIO 4,5,6): 52,956 rows, rate=0.1713
  - Selva (DOMINIO 7): 40,838 rows, rate=0.1666
- Poverty quintile: 5 groups from weighted quantile breaks on census_poverty_rate

**DESC-04: Heatmap Data (3 heatmaps per CONTEXT.md)**

1. Language x Rurality: 7 language groups x 2 (urban/rural) = 14 cells
2. Language x Poverty Quintile: 7 language groups x 5 quintiles = 35 cells
3. Language x Region: 7 language groups x 3 regions = 21 cells

Some cells may have very small n (e.g., Aymara x Selva). Flag cells with n < 50.

**DESC-05: Choropleth Prep (District-Level)**

Group by UBIGEO, compute per-district weighted dropout rate. Output columns: UBIGEO, weighted_dropout_rate, n_students, department.

**DESC-06: Temporal Trends**

Group by year (+ optionally year x language group), compute weighted rate per year.

Overall trend: 6 years (2018-2023). By language: 7 groups x 6 years = 42 data points.

### 3.4 Baseline Numbers for Gate Test 1.5

From data exploration:

| Metric | Observed Value | Gate Threshold |
|--------|---------------|----------------|
| Awajun 2020+ weighted rate | 0.2047 | > 0.18 |
| Castellano overall weighted rate | 0.1526 | 0.10 - 0.18 |
| Total rows | 150,135 | > 100,000 |
| Year coverage | 2018-2023 (6 years) | == 6 |
| Poverty quintile groups | 5 | == 5 |
| Quintile weight balance | ~20% each | within +/- 30% |

---

## 4. JSON Export Schema

### 4.1 descriptive_tables.json (from spec Section 11.6)

Per CONTEXT.md decisions, the schema extends the spec with:
- Confidence intervals (lower_ci, upper_ci) for each rate
- _metadata key with timestamp, source_rows, years_covered, sample sizes
- Numbers rounded to 4 decimal places
- Additional heatmaps (language x poverty, language x region)

```json
{
  "_metadata": {
    "generated_at": "2026-02-08T...",
    "source_rows": 150135,
    "years_covered": [2018, 2019, 2020, 2021, 2022, 2023],
    "pipeline_version": "0.1.0"
  },
  "language": [
    {
      "group": "castellano",
      "weighted_rate": 0.1526,
      "lower_ci": 0.1525,
      "upper_ci": 0.1527,
      "n_unweighted": 132825,
      "n_weighted": 35000000
    }
  ],
  "sex": [
    { "group": "male", "weighted_rate": 0.0, "lower_ci": 0.0, "upper_ci": 0.0, "n_unweighted": 0, "n_weighted": 0 },
    { "group": "female", "weighted_rate": 0.0, "lower_ci": 0.0, "upper_ci": 0.0, "n_unweighted": 0, "n_weighted": 0 }
  ],
  "sex_x_level": [
    { "group": "male_primaria", "weighted_rate": 0.0, "...": "..." },
    { "group": "female_primaria", "weighted_rate": 0.0, "...": "..." },
    { "group": "male_secundaria", "weighted_rate": 0.0, "...": "..." },
    { "group": "female_secundaria", "weighted_rate": 0.0, "...": "..." }
  ],
  "rural": [
    { "group": "urban", "weighted_rate": 0.0, "...": "..." },
    { "group": "rural", "weighted_rate": 0.0, "...": "..." }
  ],
  "region": [
    { "group": "costa", "weighted_rate": 0.0, "...": "..." },
    { "group": "sierra", "weighted_rate": 0.0, "...": "..." },
    { "group": "selva", "weighted_rate": 0.0, "...": "..." }
  ],
  "poverty": [
    { "group": "Q1_least_poor", "weighted_rate": 0.0, "...": "..." },
    { "group": "Q5_most_poor", "weighted_rate": 0.0, "...": "..." }
  ],
  "heatmap_language_x_rural": {
    "rows": ["castellano", "quechua", "aimara", "awajun", "ashaninka", "other_indigenous", "foreign"],
    "columns": ["urban", "rural"],
    "values": [],
    "ci_lower": [],
    "ci_upper": [],
    "n_unweighted": []
  },
  "heatmap_language_x_poverty": {
    "rows": ["..."],
    "columns": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "values": [],
    "ci_lower": [],
    "ci_upper": [],
    "n_unweighted": []
  },
  "heatmap_language_x_region": {
    "rows": ["..."],
    "columns": ["costa", "sierra", "selva"],
    "values": [],
    "ci_lower": [],
    "ci_upper": [],
    "n_unweighted": []
  },
  "temporal": {
    "years": [2018, 2019, 2020, 2021, 2022, 2023],
    "overall_rate": [],
    "by_language": {
      "castellano": [],
      "quechua": [],
      "...": "rate per year per language group"
    }
  }
}
```

### 4.2 Output Locations

| Output | Path | Git-tracked |
|--------|------|-------------|
| descriptive_tables.json | data/exports/descriptive_tables.json | Yes |
| enaho_with_features.parquet | data/processed/enaho_with_features.parquet | No (gitignored) |
| 7 visualizations | data/exports/figures/*.png | Yes (new directory) |

Note: data/exports/figures/ does not exist yet. Must create it. The CONTEXT.md decision says "data/exports/figures/" not "outputs/figures/". The outputs/ directory exists but figures should go to exports/ alongside the JSON.

---

## 5. Visualization Plan

### 5.1 Seven Visualizations

| # | Title | Type | X-axis | Y-axis | Notes |
|---|-------|------|--------|--------|-------|
| 1 | Dropout Rate by Mother Tongue | Horizontal bar | Language group | Weighted rate | Top 5 + Other; highlight Awajun |
| 2 | Dropout Rate by Sex x Education Level | Grouped bar | Primaria/Secundaria | Weighted rate | Male/Female side by side; show gender flip |
| 3 | Dropout Rate by Rural/Urban | Bar | Urban/Rural | Weighted rate | Simple 2-bar chart |
| 4 | Dropout Rate by Region | Bar | Costa/Sierra/Selva | Weighted rate | 3-bar chart |
| 5 | Dropout Rate by Poverty Quintile | Bar | Q1-Q5 | Weighted rate | Gradient showing poverty effect |
| 6 | Language x Rurality Heatmap | Heatmap (annotated) | Urban/Rural | Language group | 7x2 grid with rates as values |
| 7 | Temporal Trend by Language | Line chart | Year (2018-2023) | Weighted rate | Lines per language group; COVID spike visible |

### 5.2 Color Palette Recommendation

For an equity audit targeting both academic and journalistic audiences:

- **Primary palette:** Use a colorblind-safe qualitative palette. Recommend `matplotlib` tab10 or a custom muted palette.
- **Heatmap:** Sequential palette (e.g., Blues or YlOrRd) for dropout rates.
- **Bar charts:** Use consistent color per language group across all charts.
- **Highlight color:** Use a distinct red/orange to call out Awajun rates.
- **Figure size:** 10x6 or 8x6 inches for most charts; 10x8 for heatmaps.
- **Font size:** 12pt minimum for axes labels, 14pt for titles.
- **Error bars:** Include 95% CI whiskers on bar charts.

### 5.3 Console Output

Each visualization should also print a text table to stdout for gate review:

```
--- DROPOUT BY MOTHER TONGUE ---
  Castellano:       0.1526  [0.1525, 0.1527]  (n=132,825)
  Quechua:          0.2038  [0.2030, 0.2046]  (n=11,230)
  ...
  Awajun:           0.2047  [0.2018, 0.2076]  (n=738)  ***
```

---

## 6. Technical Implementation Details

### 6.1 Weighted Quintile Construction

The census_poverty_rate has 1,324 unique values (district-level). Weighted quintile assignment:

1. Sort by census_poverty_rate
2. Compute cumulative FACTOR07 weight
3. Assign quintile based on cumulative weight position (0-20% = Q1, 20-40% = Q2, etc.)
4. Handle ties at boundaries (individuals in same district get same quintile)

Tested: produces ~20% weighted population per quintile. Q1 (least poor) to Q5 (most poor).

### 6.2 Region Natural from DOMINIO

INEI DOMINIO coding:
| DOMINIO | Region | Label |
|---------|--------|-------|
| 1 | Costa | Costa norte |
| 2 | Costa | Costa centro |
| 3 | Costa | Costa sur |
| 4 | Sierra | Sierra norte |
| 5 | Sierra | Sierra centro |
| 6 | Sierra | Sierra sur |
| 7 | Selva | Selva |
| 8 | Costa | Lima Metropolitana |

This is preferable to UBIGEO[:2] department mapping because DOMINIO classifies at the household level. Departments like Amazonas (01) have households in both Sierra (DOMINIO 4, n=1,955) and Selva (DOMINIO 7, n=4,343). Cusco (08) has Sierra and Selva. DOMINIO handles this correctly.

For model encoding (region_natural_encoded):
- Option 1: Two binary dummies (sierra, selva) with costa as reference
- Option 2: Ordinal 0/1/2 (not recommended -- no ordinal relationship)

Recommendation: Two binary dummies (is_sierra, is_selva), dropping costa as reference category.

### 6.3 Rural/Urban from ESTRATO

INEI ESTRATO coding:
| ESTRATO | Size Class | Category |
|---------|-----------|----------|
| 1 | 500K+ inhabitants | Urban |
| 2 | 100K-500K | Urban |
| 3 | 50K-100K | Urban |
| 4 | 20K-50K | Urban |
| 5 | 2K-20K | Urban |
| 6 | 500-2K (rural) | Rural |
| 7 | <500 (rural) | Rural |
| 8 | Area de empadronamiento rural | Rural |

Binary: `rural = 1 if ESTRATO >= 6 else 0`

Distribution: 88,747 urban (rate=0.1495) vs 61,388 rural (rate=0.1788).

### 6.4 Admin Dropout Rate Feature

The spec has a single `district_dropout_rate_admin` feature. We have two: admin_primaria_rate and admin_secundaria_rate.

Options:
1. Use primaria rate for primaria-age (6-11), secundaria rate for secundaria-age (12-17)
2. Use the higher of the two (conservative)
3. Average of both (if both available)

Recommendation: Option 1 -- match admin rate to student's education level. This is more semantically correct and avoids mixing rates. For the 1.53% of rows with null secundaria rate, fall back to primaria rate.

### 6.5 Loading Additional Module Data

For features requiring raw module data (es_peruano, has_disability, is_working, juntos_participant, log_income, parent_education_years), the approach is:

```python
def _load_supplementary_columns(year: int) -> pl.DataFrame:
    """Load additional columns from ENAHO modules for feature engineering."""
    # Module 200: P209 (birthplace), P210 (disability), P203 (relationship to head)
    # Module 500: P501 (working)
    # Module 700: P710_04 (JUNTOS)
    # Sumaria: INGHOG1D (household income)
```

Each module has different join keys:
- Module 200, 500: (CONGLOME, VIVIENDA, HOGAR, CODPERSO) -- person level
- Module 700, sumaria: (CONGLOME, VIVIENDA, HOGAR) -- household level

The `_read_data_file()` and `_find_module_file()` helpers from enaho.py can be reused.

### 6.6 Handling P307 Nulls

P307 (currently attending) has 18.88% nulls. This column is NOT used for dropout target (that's P303 x P306), but may be useful for descriptive analysis. Do NOT impute P307 -- just exclude it from features.

### 6.7 admin_secundaria_rate Nulls (1.53%)

The 44 districts with primaria but no secundaria admin data (2,290 individual rows). For `district_dropout_rate_admin`, fall back to primaria rate for these students if they are secundaria-age.

### 6.8 nightlights_mean_radiance Nulls (4.10%)

6,150 rows have null nightlights (districts without VIIRS coverage). For z-score standardization, standardize on non-null values. For model input, impute with 0 (reasonable: unlit districts are likely very rural) or median. Document the choice.

---

## 7. Discretionary Decisions (Research Recommendations)

### 7.1 Mother Tongue Model Encoding

**Recommendation: Multi-category dummies (spec pattern)**

The spec defines 5 language dummies: lang_castellano, lang_quechua, lang_aimara, lang_other_indigenous, lang_foreign. This is the correct approach because:
- Binary (indigenous vs Castellano) loses the Quechua vs Awajun distinction
- The model needs to learn different coefficients per language group
- Descriptive analysis already uses disaggregated codes; model should too
- The spec's MODEL_FEATURES list explicitly includes all 5 dummies

Keep binary indigenous/Castellano as a DERIVED analysis variable, not a model feature.

### 7.2 Confidence Interval Method

**Recommendation: Linearization via statsmodels DescrStatsW**

- Faster than bootstrap (no resampling)
- Consistent with standard survey methodology
- Already tested and working
- Kish effective sample size is appropriate for expansion-factor weights
- Bootstrap would give similar results for proportions but take 10-100x longer

### 7.3 Additional Features Beyond Spec's 19

The user's CONTEXT.md allows additional features at Claude's discretion. Consider:

1. **is_secundaria_age** (DECIDED in CONTEXT.md): Binary for primaria (6-11) vs secundaria (12-17) age. Already requested.

2. **census_indigenous_lang_pct_z**: Z-scored district indigenous language prevalence. Already available from census merge. Captures whether the student lives in an indigenous-majority district (contextual effect beyond individual language).

3. **census_electricity_pct_z, census_water_access_pct_z, census_literacy_rate_z**: Additional district-level infrastructure indicators. Already in the dataset. Useful for capturing rural deprivation beyond the binary rural indicator.

Recommendation: Include census indicator z-scores as additional features. They are already in the dataset, cost nothing to add, and capture district-level deprivation that a binary rural/urban indicator cannot.

### 7.4 Color Palette

**Recommendation:** Use matplotlib's `tab10` base with a custom ordered mapping:

```python
PALETTE = {
    'castellano': '#1f77b4',      # Blue (reference group)
    'quechua': '#2ca02c',         # Green
    'aimara': '#9467bd',          # Purple
    'awajun': '#d62728',          # Red (highlight equity gap)
    'ashaninka': '#ff7f0e',       # Orange
    'other_indigenous': '#8c564b', # Brown
    'foreign': '#7f7f7f',         # Gray
}
```

Use red (#d62728) for Awajun to visually highlight the equity gap. Use blue for Castellano as the reference group. This is colorblind-accessible (red and blue are distinguishable in most color vision deficiency types).

---

## 8. Plan Split: 04-01 vs 04-02

### 04-01: Feature Engineering Pipeline

**Scope:**
1. Create `src/data/features.py` with `build_features(df) -> FeatureResult`
2. Load supplementary module data (P209, P210, P501, P710_04, INGHOG1D, P203/P301A for parents)
3. Construct all features per spec Section 5 (exact column names)
4. Weighted poverty quintile construction
5. Z-score standardization of district-level features
6. Save `data/processed/enaho_with_features.parquet`
7. Update `src/data/__init__.py` to export new functions
8. Unit tests for feature construction logic

**Key risks:**
- Loading 6 years of supplementary modules is slow (~300MB of DTA files per module x 6 years)
- Parent education linkage within household is complex (P203 relationship codes)
- school_student_teacher_ratio is unavailable (document as gap)

### 04-02: Descriptive Statistics + Export + Gate Test

**Scope:**
1. Compute all 6 descriptive breakdowns (DESC-01 through DESC-06)
2. Compute confidence intervals using statsmodels
3. Generate 7 matplotlib visualizations saved to data/exports/figures/
4. Print console tables for gate review
5. Export descriptive_tables.json to data/exports/
6. Write gate test 1.5 (tests/gates/test_gate_1_5.py)
7. Human gate: review rates, visualizations, Awajun gap

**Key risks:**
- Heatmap cells with very small n (< 50) need flagging
- JSON schema must match spec Section 11.6 while incorporating CONTEXT.md extensions
- 7 visualizations need consistent styling

---

## 9. Dependencies and Environment

### 9.1 Python Packages Already Available

All needed packages are in pyproject.toml:
- `polars` -- data processing
- `matplotlib` -- visualizations
- `seaborn` -- heatmaps (optional, can use matplotlib imshow)
- `statsmodels` -- DescrStatsW for confidence intervals
- `numpy` -- array operations
- `pyarrow` -- parquet I/O
- `pandas` -- DTA file reading (via pd.read_stata)

No new dependencies needed.

### 9.2 Test Execution

```bash
uv run pytest tests/gates/test_gate_1_5.py -v
uv run pytest tests/unit/ -v  # for any new unit tests
```

### 9.3 Data Dependencies

- `data/processed/full_dataset.parquet` -- primary input (exists)
- `data/raw/enaho/YYYY/` -- raw modules for supplementary features (exist for 2018-2023)
- `data/raw/enaho/YYYY/sumaria-YYYY.dta` -- household income (exist for all years)
- `data/exports/` -- output directory (exists, has .gitkeep)
- `data/exports/figures/` -- needs to be created

---

## 10. Gate Test 1.5 Design

### 10.1 Assertions

Based on spec and observed baselines:

```
Feature Validation:
- Feature matrix has >= 19 columns in MODEL_FEATURES
- All binary features {es_mujer, lang_*, rural, es_peruano, is_working, juntos_participant, has_disability} contain only values {0, 1}
- poverty_quintile has exactly 5 unique values {1,2,3,4,5}
- Each quintile represents 15-25% of weighted population (20% +/- 30% = 14-26%)
- age column has values 6-17 only
- No feature has > 30% null rate
- No pair of model features has Pearson |correlation| > 0.95

Dropout Rate Validation:
- Awajun (p300a_original == 11) weighted rate > 0.18 for years >= 2020
- Castellano (P300A == 4) weighted rate between 0.10 and 0.18
- Rural rate > Urban rate (directional check)
- Sierra rate > Costa rate (directional check)

Export Validation:
- data/exports/descriptive_tables.json exists
- JSON is valid (json.loads succeeds)
- Top-level keys include: language, sex, rural, region, poverty, heatmap_language_x_rural, temporal
- Each breakdown entry has: weighted_rate, lower_ci, upper_ci, n_unweighted
- _metadata key has: generated_at, source_rows, years_covered

Parquet Validation:
- data/processed/enaho_with_features.parquet exists
- Row count == full_dataset.parquet row count (150,135)
- Contains all MODEL_FEATURES columns
```

### 10.2 Human Gate Review Items

Print for human inspection:
1. Full dropout rate table by language group (with CIs and sample sizes)
2. Sex x education level table
3. Rural/urban comparison
4. Region comparison
5. Poverty quintile gradient
6. Awajun by-year breakdown (2020-2023)
7. Correlation matrix for top features
8. Paths to all 7 saved PNG figures

---

## 11. Known Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Loading 6 years of supplementary modules is slow | 5-10 min build time | Cache intermediate results; only load needed columns |
| school_student_teacher_ratio unavailable | 18 instead of 19 features | Document as known gap; model works with 18 features |
| Parent education linkage is complex | Some children may not have identifiable parent | Median imputation for missing; log warning |
| Awajun small sample size (141-217 per year) | Wide CIs, volatile rates | Report CIs prominently; flag small-n cells |
| 2020 COVID effect inflates Awajun rate (0.34) | May skew 2020+ combined rate | Show individual years; note COVID context |
| es_peruano has ~0.1% foreign prevalence | Near-zero variance feature | Keep per spec; document low variance |
| Heatmap cells with n < 50 | Unreliable rates | Flag with asterisk; exclude from CI computation |
| nightlights 4.1% null | Missing for some districts | Impute with 0 or median after z-scoring |

---

*Research completed: 2026-02-08*
*Phase: 04-feature-engineering-descriptive-statistics*
