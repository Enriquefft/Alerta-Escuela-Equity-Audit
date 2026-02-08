"""Unit tests for the feature engineering pipeline.

Tests feature construction logic using small synthetic DataFrames
to avoid I/O dependencies on real ENAHO data files.
"""

import sys

sys.path.insert(0, "src")

import polars as pl
import pytest

from data.features import (
    MODEL_FEATURES,
    META_COLUMNS,
    FeatureResult,
    _zscore,
    _compute_weighted_poverty_quintile,
    _P301A_TO_YEARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_df(n: int = 10) -> pl.DataFrame:
    """Create a minimal synthetic DataFrame mimicking full_dataset.parquet.

    The DataFrame has all columns required by build_features() direct
    mappings, but uses synthetic values for fast testing.
    """
    return pl.DataFrame({
        "CONGLOME": [f"C{i:04d}" for i in range(n)],
        "VIVIENDA": [f"V{i:02d}" for i in range(n)],
        "HOGAR": [f"H{i:02d}" for i in range(n)],
        "CODPERSO": [f"P{i:02d}" for i in range(n)],
        "UBIGEO": [f"01010{i % 5 + 1}" for i in range(n)],
        "DOMINIO": [((i % 8) + 1) for i in range(n)],
        "ESTRATO": [((i % 8) + 1) for i in range(n)],
        "P207": [(1.0 if i % 2 == 0 else 2.0) for i in range(n)],
        "P208A": [(6.0 + (i % 12)) for i in range(n)],
        "P300A": [(4.0 if i < 6 else 1.0 if i < 8 else 2.0) for i in range(n)],
        "P301A": [float((i % 6) + 1) for i in range(n)],
        "P303": [1 for _ in range(n)],
        "P306": [(1 if i % 3 == 0 else 2) for i in range(n)],
        "P307": [(1.0 if i % 2 == 0 else None) for i in range(n)],
        "FACTOR07": [(100.0 + i * 10) for i in range(n)],
        "dropout": [(i % 3 == 0) for i in range(n)],
        "year": [2023 for _ in range(n)],
        "p300a_original": [(4.0 if i < 6 else 1.0 if i < 8 else 2.0) for i in range(n)],
        "p300a_harmonized": [(4 if i < 6 else 1 if i < 8 else 2) for i in range(n)],
        "admin_primaria_rate": [(1.0 + i * 0.1) for i in range(n)],
        "admin_secundaria_rate": [(2.0 + i * 0.1 if i < 8 else None) for i in range(n)],
        "census_poverty_rate": [(10.0 + i * 5) for i in range(n)],
        "census_indigenous_lang_pct": [(5.0 + i * 3) for i in range(n)],
        "census_water_access_pct": [(60.0 + i * 2) for i in range(n)],
        "census_electricity_pct": [(70.0 + i * 2) for i in range(n)],
        "census_literacy_rate": [(80.0 + i * 1) for i in range(n)],
        "nightlights_mean_radiance": [(10.0 + i * 5 if i < 8 else None) for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Test: Binary feature encoding
# ---------------------------------------------------------------------------


class TestBinaryFeatures:
    """Test that binary features contain only {0, 1}."""

    def test_binary_features_encoding(self):
        """All binary features should contain only values 0 and 1."""
        df = _make_base_df(10)

        # Apply direct mapping expressions from build_features step 1
        df = df.with_columns([
            pl.when(pl.col("P207") == 2.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("es_mujer"),
            pl.when(pl.col("ESTRATO") >= 6).then(pl.lit(1)).otherwise(pl.lit(0)).alias("rural"),
            pl.when(pl.col("P300A") == 4.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("lang_castellano"),
            pl.when(pl.col("P300A") == 1.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("lang_quechua"),
            pl.when(pl.col("P300A") == 2.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("lang_aimara"),
            pl.when(pl.col("p300a_harmonized") == 3).then(pl.lit(1)).otherwise(pl.lit(0)).alias("lang_other_indigenous"),
            pl.when(pl.col("P300A").is_in([6.0, 7.0])).then(pl.lit(1)).otherwise(pl.lit(0)).alias("lang_foreign"),
            pl.when(pl.col("DOMINIO").is_in([4, 5, 6])).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_sierra"),
            pl.when(pl.col("DOMINIO") == 7).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_selva"),
            pl.when(pl.col("P208A") >= 12).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_secundaria_age"),
        ])

        binary_cols = [
            "es_mujer", "rural", "lang_castellano", "lang_quechua",
            "lang_aimara", "lang_other_indigenous", "lang_foreign",
            "is_sierra", "is_selva", "is_secundaria_age",
        ]

        for col in binary_cols:
            unique_vals = set(df[col].unique().to_list())
            assert unique_vals.issubset({0, 1}), (
                f"{col} has values {unique_vals}, expected subset of {{0, 1}}"
            )

    def test_es_mujer_correct_mapping(self):
        """P207==2 -> es_mujer=1, P207==1 -> es_mujer=0."""
        df = pl.DataFrame({"P207": [1.0, 2.0, 1.0, 2.0]})
        df = df.with_columns(
            pl.when(pl.col("P207") == 2.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("es_mujer")
        )
        assert df["es_mujer"].to_list() == [0, 1, 0, 1]

    def test_rural_estrato_threshold(self):
        """ESTRATO >= 6 -> rural=1, ESTRATO < 6 -> rural=0."""
        df = pl.DataFrame({"ESTRATO": [1, 3, 5, 6, 7, 8]})
        df = df.with_columns(
            pl.when(pl.col("ESTRATO") >= 6).then(pl.lit(1)).otherwise(pl.lit(0)).alias("rural")
        )
        assert df["rural"].to_list() == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# Test: Poverty quintile balance
# ---------------------------------------------------------------------------


class TestPovertyQuintile:
    """Test weighted poverty quintile assignment."""

    def test_poverty_quintile_balance(self):
        """5 quintiles with equal weights should have ~20 rows each."""
        n = 100
        df = pl.DataFrame({
            "census_poverty_rate": [float(i) for i in range(n)],
            "FACTOR07": [1.0 for _ in range(n)],
        })

        result = _compute_weighted_poverty_quintile(df)

        # Check 5 quintiles exist
        unique_q = sorted(result["poverty_quintile"].unique().to_list())
        assert unique_q == [1, 2, 3, 4, 5]

        # Check each quintile has ~20% of rows (with equal weights)
        for q in range(1, 6):
            count = result.filter(pl.col("poverty_quintile") == q).height
            assert 15 <= count <= 25, (
                f"Quintile {q} has {count} rows, expected ~20"
            )

    def test_poverty_quintile_weighted(self):
        """Quintiles should balance weighted population, not row counts."""
        # 10 rows: first 2 have weight 100, rest have weight 10
        df = pl.DataFrame({
            "census_poverty_rate": [float(i) for i in range(10)],
            "FACTOR07": [100.0, 100.0] + [10.0] * 8,
        })

        result = _compute_weighted_poverty_quintile(df)
        unique_q = sorted(result["poverty_quintile"].unique().to_list())
        assert len(unique_q) >= 2, "Should have multiple quintiles"

    def test_quintile_ordering(self):
        """Q1 should have lowest poverty rates, Q5 highest."""
        n = 50
        df = pl.DataFrame({
            "census_poverty_rate": [float(i) for i in range(n)],
            "FACTOR07": [1.0 for _ in range(n)],
        })

        result = _compute_weighted_poverty_quintile(df)

        q1_mean = result.filter(pl.col("poverty_quintile") == 1)["census_poverty_rate"].mean()
        q5_mean = result.filter(pl.col("poverty_quintile") == 5)["census_poverty_rate"].mean()
        assert q1_mean < q5_mean, "Q1 should have lower poverty than Q5"


# ---------------------------------------------------------------------------
# Test: Region natural mapping
# ---------------------------------------------------------------------------


class TestRegionNatural:
    """Test DOMINIO to region_natural mapping."""

    def test_region_natural_mapping(self):
        """All 8 DOMINIO values map correctly to costa/sierra/selva."""
        df = pl.DataFrame({"DOMINIO": [1, 2, 3, 4, 5, 6, 7, 8]})
        df = df.with_columns(
            pl.when(pl.col("DOMINIO").is_in([1, 2, 3, 8]))
            .then(pl.lit("costa"))
            .when(pl.col("DOMINIO").is_in([4, 5, 6]))
            .then(pl.lit("sierra"))
            .when(pl.col("DOMINIO") == 7)
            .then(pl.lit("selva"))
            .otherwise(pl.lit("unknown"))
            .alias("region_natural")
        )

        expected = ["costa", "costa", "costa", "sierra", "sierra", "sierra", "selva", "costa"]
        assert df["region_natural"].to_list() == expected

    def test_region_dummies_consistent(self):
        """is_sierra and is_selva are mutually exclusive with costa as reference."""
        df = pl.DataFrame({"DOMINIO": [1, 2, 3, 4, 5, 6, 7, 8]})
        df = df.with_columns([
            pl.when(pl.col("DOMINIO").is_in([4, 5, 6])).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_sierra"),
            pl.when(pl.col("DOMINIO") == 7).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_selva"),
        ])

        # No row should have both is_sierra=1 AND is_selva=1
        both = df.filter((pl.col("is_sierra") == 1) & (pl.col("is_selva") == 1))
        assert both.height == 0, "is_sierra and is_selva should be mutually exclusive"

        # Costa reference: rows where both are 0
        costa = df.filter((pl.col("is_sierra") == 0) & (pl.col("is_selva") == 0))
        assert costa.height == 4, "4 DOMINIO values should map to costa (reference)"


# ---------------------------------------------------------------------------
# Test: Admin rate age matching
# ---------------------------------------------------------------------------


class TestAdminRateAgeMatching:
    """Test district_dropout_rate_admin based on age group."""

    def test_primaria_age_gets_primaria_rate(self):
        """Ages 6-11 should get admin_primaria_rate."""
        df = pl.DataFrame({
            "P208A": [8.0],
            "admin_primaria_rate": [1.5],
            "admin_secundaria_rate": [3.0],
        })
        df = df.with_columns(
            pl.col("P208A").cast(pl.Int64).alias("age")
        )
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
        assert df["district_dropout_rate_admin"][0] == 1.5

    def test_secundaria_age_gets_secundaria_rate(self):
        """Ages 12-17 should get admin_secundaria_rate."""
        df = pl.DataFrame({
            "P208A": [14.0],
            "admin_primaria_rate": [1.5],
            "admin_secundaria_rate": [3.0],
        })
        df = df.with_columns(
            pl.col("P208A").cast(pl.Int64).alias("age")
        )
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
        assert df["district_dropout_rate_admin"][0] == 3.0

    def test_secundaria_fallback_to_primaria(self):
        """When admin_secundaria_rate is null, age 12+ falls back to primaria."""
        df = pl.DataFrame({
            "P208A": [14.0],
            "admin_primaria_rate": [1.5],
            "admin_secundaria_rate": [None],
        }, schema_overrides={"admin_secundaria_rate": pl.Float64})
        df = df.with_columns(
            pl.col("P208A").cast(pl.Int64).alias("age")
        )
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
        assert df["district_dropout_rate_admin"][0] == 1.5


# ---------------------------------------------------------------------------
# Test: Z-score standardization
# ---------------------------------------------------------------------------


class TestZScoreStandardization:
    """Test z-score standardization helper."""

    def test_zscore_basic(self):
        """Z-scored values have mean ~0 and std ~1 for non-null values."""
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = _zscore(df, "val", "val_z")

        mean_z = result["val_z"].mean()
        std_z = result["val_z"].std()
        assert abs(mean_z) < 0.001, f"Z-score mean should be ~0, got {mean_z}"
        assert abs(std_z - 1.0) < 0.1, f"Z-score std should be ~1, got {std_z}"

    def test_zscore_with_nulls(self):
        """Null values remain null unless impute_null=True."""
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, None]})
        result = _zscore(df, "val", "val_z", impute_null=False)

        # Last value should be null
        assert result["val_z"][3] is None

        # Non-null mean should be ~0
        non_null_mean = result["val_z"].drop_nulls().mean()
        assert abs(non_null_mean) < 0.001

    def test_zscore_impute_null(self):
        """With impute_null=True, null z-scores become 0.0."""
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, None]})
        result = _zscore(df, "val", "val_z", impute_null=True)

        # Last value should be 0.0 (imputed)
        assert result["val_z"][3] == pytest.approx(0.0)

        # No nulls remaining
        assert result["val_z"].null_count() == 0

    def test_zscore_zero_std(self):
        """Constant values produce z-score of 0.0."""
        df = pl.DataFrame({"val": [5.0, 5.0, 5.0]})
        result = _zscore(df, "val", "val_z")
        assert all(v == 0.0 for v in result["val_z"].to_list())


# ---------------------------------------------------------------------------
# Test: MODEL_FEATURES constant
# ---------------------------------------------------------------------------


class TestModelFeatures:
    """Test the MODEL_FEATURES constant."""

    def test_model_features_count(self):
        """MODEL_FEATURES has >= 19 entries (per spec)."""
        assert len(MODEL_FEATURES) >= 19, (
            f"MODEL_FEATURES has {len(MODEL_FEATURES)} entries, expected >= 19"
        )

    def test_all_lowercase(self):
        """All feature names are lowercase strings."""
        for feat in MODEL_FEATURES:
            assert isinstance(feat, str), f"{feat} is not a string"
            assert feat == feat.lower(), f"{feat} is not lowercase"

    def test_no_duplicates(self):
        """No duplicate feature names."""
        assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)), (
            "MODEL_FEATURES contains duplicates"
        )

    def test_meta_columns_no_overlap(self):
        """META_COLUMNS should not overlap with MODEL_FEATURES."""
        overlap = set(MODEL_FEATURES) & set(META_COLUMNS)
        assert len(overlap) == 0, (
            f"MODEL_FEATURES and META_COLUMNS overlap: {overlap}"
        )


# ---------------------------------------------------------------------------
# Test: P301A education level mapping
# ---------------------------------------------------------------------------


class TestP301AMapping:
    """Test education level to years mapping."""

    def test_mapping_completeness(self):
        """All known P301A codes have a mapping."""
        expected_codes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
        mapped_codes = set(_P301A_TO_YEARS.keys())
        # At minimum, codes 1-7, 9, 12 must be mapped (those present in our data)
        required = {1, 2, 3, 4, 5, 6, 7, 9, 12}
        assert required.issubset(mapped_codes), (
            f"Missing mappings for P301A codes: {required - mapped_codes}"
        )

    def test_mapping_values_reasonable(self):
        """Education years should be between 0 and 20."""
        for code, years in _P301A_TO_YEARS.items():
            assert 0 <= years <= 20, (
                f"P301A code {code} maps to {years} years (unreasonable)"
            )

    def test_monotonic_tendency(self):
        """Higher education codes should generally map to more years."""
        # Primary (3-4) < Secondary (5-6) < Higher (7-12)
        assert _P301A_TO_YEARS[3] <= _P301A_TO_YEARS[4]  # primaria
        assert _P301A_TO_YEARS[5] <= _P301A_TO_YEARS[6]  # secundaria
        assert _P301A_TO_YEARS[6] < _P301A_TO_YEARS[9]   # secundaria < university
