"""Gate Test 1.5: Feature Engineering + Descriptive Statistics Validation.

Validates:
- Feature matrix completeness and correctness
- Binary feature encoding
- Poverty quintile balance
- Survey-weighted dropout rate thresholds
- Correlation check for multicollinearity
- descriptive_tables.json export validation
- enaho_with_features.parquet validation
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.features import MODEL_FEATURES, META_COLUMNS
from utils import find_project_root


ROOT = find_project_root()
PARQUET_PATH = ROOT / "data" / "processed" / "enaho_with_features.parquet"
JSON_PATH = ROOT / "data" / "exports" / "descriptive_tables.json"
FIGURES_DIR = ROOT / "data" / "exports" / "figures"

BINARY_FEATURES = [
    "es_mujer",
    "lang_castellano",
    "lang_quechua",
    "lang_aimara",
    "lang_other_indigenous",
    "lang_foreign",
    "rural",
    "is_sierra",
    "is_selva",
    "es_peruano",
    "has_disability",
    "is_working",
    "juntos_participant",
    "is_secundaria_age",
]

EXPECTED_FIGURES = [
    "01_language_bars.png",
    "02_sex_education_bars.png",
    "03_rural_urban_bars.png",
    "04_region_bars.png",
    "05_poverty_quintile_bars.png",
    "06_language_rurality_heatmap.png",
    "07_temporal_trend_lines.png",
]


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """Load the feature matrix parquet once per test module."""
    return pl.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def json_data() -> dict:
    """Load the descriptive tables JSON once per test module."""
    with open(JSON_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature matrix tests
# ---------------------------------------------------------------------------


def test_feature_count(df: pl.DataFrame) -> None:
    """MODEL_FEATURES >= 19 and all present in parquet."""
    assert len(MODEL_FEATURES) >= 19, (
        f"Expected >= 19 model features, got {len(MODEL_FEATURES)}"
    )
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    assert not missing, f"Missing model features in parquet: {missing}"
    print(f"\n  Feature count: {len(MODEL_FEATURES)} model features -- PASS")


def test_binary_features_valid(df: pl.DataFrame) -> None:
    """14 binary features contain only {0, 1}."""
    print("\n  Binary feature validation:")
    for col in BINARY_FEATURES:
        assert col in df.columns, f"Binary feature {col} not in DataFrame"
        unique_vals = set(df[col].unique().drop_nulls().to_list())
        assert unique_vals.issubset({0, 1}), (
            f"Binary feature {col} has unexpected values: {unique_vals}"
        )
        counts = df[col].value_counts().sort("count", descending=True)
        count_str = ", ".join(
            f"{row[col]}={row['count']:,}"
            for row in counts.iter_rows(named=True)
        )
        print(f"    {col}: {count_str}")
    print(f"  Binary validation: PASS ({len(BINARY_FEATURES)} features)")


def test_age_range(df: pl.DataFrame) -> None:
    """Age 6-17, no nulls."""
    assert df["age"].null_count() == 0, "age has null values"
    assert df["age"].min() >= 6, f"age min={df['age'].min()} < 6"
    assert df["age"].max() <= 17, f"age max={df['age'].max()} > 17"
    print(f"\n  Age range: {df['age'].min()}-{df['age'].max()}, 0 nulls -- PASS")


def test_poverty_quintile_balance(df: pl.DataFrame) -> None:
    """5 groups, each 14-26% weighted share."""
    unique_q = sorted(df["poverty_quintile"].unique().to_list())
    assert unique_q == [1, 2, 3, 4, 5], (
        f"Expected quintiles [1,2,3,4,5], got {unique_q}"
    )

    total_weight = df["FACTOR07"].sum()
    print("\n  Poverty quintile balance:")
    for q in range(1, 6):
        q_weight = df.filter(pl.col("poverty_quintile") == q)["FACTOR07"].sum()
        share = q_weight / total_weight
        assert 0.14 <= share <= 0.26, (
            f"Quintile {q} weighted share {share:.4f} outside [0.14, 0.26]"
        )
        print(f"    Q{q}: {share:.4f} ({share * 100:.1f}%)")
    print("  Quintile balance: PASS (5 groups, 14-26% each)")


def test_no_high_null_features(df: pl.DataFrame) -> None:
    """No model feature > 30% nulls."""
    print("\n  Null rate check:")
    any_nonzero = False
    for col in MODEL_FEATURES:
        null_rate = df[col].null_count() / df.height
        if null_rate > 0:
            any_nonzero = True
            print(f"    {col}: {null_rate:.4f} ({null_rate * 100:.1f}%)")
        assert null_rate <= 0.30, (
            f"Feature {col} has {null_rate:.1%} nulls (exceeds 30%)"
        )
    if not any_nonzero:
        print("    All model features: 0% nulls")
    print("  Null check: PASS (no feature > 30%)")


def test_no_high_correlation(df: pl.DataFrame) -> None:
    """No |correlation| > 0.95 among numeric model features."""
    # Select only numeric model features present in the DataFrame
    numeric_cols = [
        col
        for col in MODEL_FEATURES
        if col in df.columns and df[col].dtype in (
            pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
            pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
        )
    ]

    # Convert to numpy for correlation
    matrix = df.select(numeric_cols).to_numpy()
    # Replace NaN with 0 for correlation computation
    matrix = np.nan_to_num(matrix, nan=0.0)
    corr = np.corrcoef(matrix, rowvar=False)

    # Find pairs with high correlation (excluding diagonal)
    n = len(numeric_cols)
    high_corr_pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr[i, j])
            if c > 0.95:
                high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr[i, j]))

    assert not high_corr_pairs, (
        f"High correlations (|r| > 0.95): {high_corr_pairs}"
    )

    # Print top 5 most correlated
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((numeric_cols[i], numeric_cols[j], abs(corr[i, j])))
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n  Top-5 correlated feature pairs:")
    for col1, col2, c in all_pairs[:5]:
        print(f"    {col1} x {col2}: r={c:.4f}")
    print("  Correlation check: PASS (no |r| > 0.95)")


# ---------------------------------------------------------------------------
# Dropout rate threshold tests
# ---------------------------------------------------------------------------


def test_awajun_dropout_rate(df: pl.DataFrame) -> None:
    """Awajun 2020+ weighted dropout rate > 0.18."""
    awajun = df.filter(
        (pl.col("p300a_original") == 11) & (pl.col("year") >= 2020)
    )
    assert awajun.height > 0, "No Awajun rows found for 2020+"

    d = awajun["dropout"].cast(pl.Float64).to_numpy()
    w = awajun["FACTOR07"].to_numpy()
    rate = float(np.average(d, weights=w))

    assert rate > 0.18, f"Awajun 2020+ rate {rate:.4f} not > 0.18"
    print(f"\n  Awajun 2020+ weighted dropout rate: {rate:.4f} (threshold: > 0.18) -- PASS")

    # By-year breakdown
    print("  Awajun by year:")
    for yr in sorted(awajun["year"].unique().to_list()):
        sub = awajun.filter(pl.col("year") == yr)
        d_yr = sub["dropout"].cast(pl.Float64).to_numpy()
        w_yr = sub["FACTOR07"].to_numpy()
        r_yr = float(np.average(d_yr, weights=w_yr))
        print(f"    {yr}: {r_yr:.4f} (n={sub.height})")


def test_castellano_dropout_rate(df: pl.DataFrame) -> None:
    """Castellano weighted dropout rate between 0.10 and 0.18."""
    castellano = df.filter(pl.col("lang_castellano") == 1)
    assert castellano.height > 0, "No Castellano rows found"

    d = castellano["dropout"].cast(pl.Float64).to_numpy()
    w = castellano["FACTOR07"].to_numpy()
    rate = float(np.average(d, weights=w))

    assert 0.10 < rate < 0.18, f"Castellano rate {rate:.4f} not in (0.10, 0.18)"
    print(
        f"\n  Castellano weighted dropout rate: {rate:.4f} "
        f"(threshold: 0.10-0.18) -- PASS"
    )


def test_directional_checks(df: pl.DataFrame) -> None:
    """Rural > Urban, Sierra > Costa."""
    # Rural vs Urban
    rural = df.filter(pl.col("rural") == 1)
    urban = df.filter(pl.col("rural") == 0)
    d_r = rural["dropout"].cast(pl.Float64).to_numpy()
    w_r = rural["FACTOR07"].to_numpy()
    d_u = urban["dropout"].cast(pl.Float64).to_numpy()
    w_u = urban["FACTOR07"].to_numpy()
    rate_rural = float(np.average(d_r, weights=w_r))
    rate_urban = float(np.average(d_u, weights=w_u))

    assert rate_rural > rate_urban, (
        f"Rural ({rate_rural:.4f}) not > Urban ({rate_urban:.4f})"
    )
    print(f"\n  Rural ({rate_rural:.4f}) > Urban ({rate_urban:.4f}) -- PASS")

    # Sierra vs Costa
    sierra = df.filter(pl.col("region_natural") == "sierra")
    costa = df.filter(pl.col("region_natural") == "costa")
    d_s = sierra["dropout"].cast(pl.Float64).to_numpy()
    w_s = sierra["FACTOR07"].to_numpy()
    d_c = costa["dropout"].cast(pl.Float64).to_numpy()
    w_c = costa["FACTOR07"].to_numpy()
    rate_sierra = float(np.average(d_s, weights=w_s))
    rate_costa = float(np.average(d_c, weights=w_c))

    assert rate_sierra > rate_costa, (
        f"Sierra ({rate_sierra:.4f}) not > Costa ({rate_costa:.4f})"
    )
    print(f"  Sierra ({rate_sierra:.4f}) > Costa ({rate_costa:.4f}) -- PASS")


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------


def test_descriptive_json_exists(json_data: dict) -> None:
    """descriptive_tables.json exists, valid JSON, all top-level keys."""
    assert JSON_PATH.exists(), f"JSON file not found: {JSON_PATH}"

    required_keys = [
        "_metadata",
        "language",
        "sex",
        "sex_x_level",
        "rural",
        "region",
        "poverty",
        "heatmap_language_x_rural",
        "temporal",
    ]
    for key in required_keys:
        assert key in json_data, f"Missing top-level key: {key}"

    meta = json_data["_metadata"]
    assert "generated_at" in meta, "Missing _metadata.generated_at"
    assert "source_rows" in meta, "Missing _metadata.source_rows"
    assert "years_covered" in meta, "Missing _metadata.years_covered"

    print(f"\n  JSON keys: {list(json_data.keys())}")
    print(f"  Metadata: {meta}")
    print("  JSON existence: PASS (valid, all keys present)")


def test_descriptive_json_schema(json_data: dict) -> None:
    """Each breakdown entry has required fields; CI ordering is valid."""
    breakdown_keys = ["language", "sex", "rural", "region", "poverty"]
    required_entry_fields = {
        "group",
        "weighted_rate",
        "lower_ci",
        "upper_ci",
        "n_unweighted",
    }

    print("\n  JSON schema validation:")
    for key in breakdown_keys:
        assert key in json_data, f"Missing breakdown key: {key}"
        entries = json_data[key]
        assert isinstance(entries, list), f"{key} is not a list"
        assert len(entries) > 0, f"{key} is empty"

        for entry in entries:
            missing_fields = required_entry_fields - set(entry.keys())
            assert not missing_fields, (
                f"{key} entry missing fields: {missing_fields}"
            )
            assert entry["lower_ci"] <= entry["weighted_rate"] <= entry["upper_ci"], (
                f"{key}/{entry['group']}: CI ordering violated: "
                f"{entry['lower_ci']} <= {entry['weighted_rate']} <= {entry['upper_ci']}"
            )
            assert entry["n_unweighted"] > 0, (
                f"{key}/{entry['group']}: n_unweighted must be > 0"
            )
        print(f"    {key}: {len(entries)} entries -- PASS")

    # Heatmap validation
    hm = json_data.get("heatmap_language_x_rural", {})
    for field in ["rows", "columns", "values", "ci_lower", "ci_upper", "n_unweighted"]:
        assert field in hm, f"heatmap_language_x_rural missing field: {field}"
    print(f"    heatmap_language_x_rural: {len(hm['rows'])}x{len(hm['columns'])} -- PASS")

    # Temporal validation
    temporal = json_data.get("temporal", {})
    for field in ["years", "overall_rate", "by_language"]:
        assert field in temporal, f"temporal missing field: {field}"
    print(f"    temporal: {len(temporal['years'])} years -- PASS")
    print("  JSON schema: PASS")


# ---------------------------------------------------------------------------
# Parquet and figure tests
# ---------------------------------------------------------------------------


def test_parquet_row_count(df: pl.DataFrame) -> None:
    """enaho_with_features.parquet has exactly 150,135 rows, all features."""
    assert df.height == 150_135, f"Expected 150,135 rows, got {df.height}"

    missing_model = [f for f in MODEL_FEATURES if f not in df.columns]
    assert not missing_model, f"Missing MODEL_FEATURES: {missing_model}"

    missing_meta = [c for c in META_COLUMNS if c not in df.columns]
    assert not missing_meta, f"Missing META_COLUMNS: {missing_meta}"

    print(f"\n  Parquet: {df.height:,} rows, {df.width} columns")
    print(f"  MODEL_FEATURES: {len(MODEL_FEATURES)} present")
    print(f"  META_COLUMNS: {len(META_COLUMNS)} present")
    print("  Parquet validation: PASS")


def test_figures_exist() -> None:
    """7 PNG files exist in data/exports/figures/."""
    assert FIGURES_DIR.exists(), f"Figures directory not found: {FIGURES_DIR}"

    print(f"\n  Figures directory: {FIGURES_DIR}")
    missing = []
    for fname in EXPECTED_FIGURES:
        fpath = FIGURES_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"    {fname}: {size_kb:.1f} KB")
        else:
            missing.append(fname)
            print(f"    {fname}: MISSING")

    assert not missing, f"Missing figure files: {missing}"
    print(f"  Figures: PASS ({len(EXPECTED_FIGURES)} PNGs)")


# ---------------------------------------------------------------------------
# Standalone summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_data = pl.read_parquet(PARQUET_PATH)
    with open(JSON_PATH) as f:
        jdata = json.load(f)

    print("=== GATE TEST 1.5 SUMMARY ===")
    print(f"Feature matrix: {len(MODEL_FEATURES)} model features, {df_data.height:,} rows")

    # Binary
    all_binary_ok = True
    for col in BINARY_FEATURES:
        vals = set(df_data[col].unique().drop_nulls().to_list())
        if not vals.issubset({0, 1}):
            all_binary_ok = False
    print(f"Binary validation: {'PASS' if all_binary_ok else 'FAIL'} ({len(BINARY_FEATURES)} features)")

    # Quintile
    total_w = df_data["FACTOR07"].sum()
    q_ok = True
    for q in range(1, 6):
        share = df_data.filter(pl.col("poverty_quintile") == q)["FACTOR07"].sum() / total_w
        if not (0.14 <= share <= 0.26):
            q_ok = False
    print(f"Quintile balance: {'PASS' if q_ok else 'FAIL'} (5 groups, 14-26% each)")

    # Awajun
    awajun = df_data.filter(
        (pl.col("p300a_original") == 11) & (pl.col("year") >= 2020)
    )
    d = awajun["dropout"].cast(pl.Float64).to_numpy()
    w = awajun["FACTOR07"].to_numpy()
    awajun_rate = float(np.average(d, weights=w))
    print(f"Awajun 2020+: {awajun_rate:.4f} > 0.18 {'PASS' if awajun_rate > 0.18 else 'FAIL'}")

    # Castellano
    cast = df_data.filter(pl.col("lang_castellano") == 1)
    d = cast["dropout"].cast(pl.Float64).to_numpy()
    w = cast["FACTOR07"].to_numpy()
    cast_rate = float(np.average(d, weights=w))
    print(f"Castellano: {cast_rate:.4f} in [0.10, 0.18] {'PASS' if 0.10 < cast_rate < 0.18 else 'FAIL'}")

    # Directional
    rural_r = float(np.average(
        df_data.filter(pl.col("rural") == 1)["dropout"].cast(pl.Float64).to_numpy(),
        weights=df_data.filter(pl.col("rural") == 1)["FACTOR07"].to_numpy(),
    ))
    urban_r = float(np.average(
        df_data.filter(pl.col("rural") == 0)["dropout"].cast(pl.Float64).to_numpy(),
        weights=df_data.filter(pl.col("rural") == 0)["FACTOR07"].to_numpy(),
    ))
    sierra_r = float(np.average(
        df_data.filter(pl.col("region_natural") == "sierra")["dropout"].cast(pl.Float64).to_numpy(),
        weights=df_data.filter(pl.col("region_natural") == "sierra")["FACTOR07"].to_numpy(),
    ))
    costa_r = float(np.average(
        df_data.filter(pl.col("region_natural") == "costa")["dropout"].cast(pl.Float64).to_numpy(),
        weights=df_data.filter(pl.col("region_natural") == "costa")["FACTOR07"].to_numpy(),
    ))
    dir_ok = rural_r > urban_r and sierra_r > costa_r
    print(f"Directional: rural > urban {'PASS' if rural_r > urban_r else 'FAIL'}, sierra > costa {'PASS' if sierra_r > costa_r else 'FAIL'}")

    # JSON
    required_keys = ["_metadata", "language", "sex", "rural", "region", "poverty", "temporal"]
    json_ok = all(k in jdata for k in required_keys)
    print(f"JSON export: {'PASS' if json_ok else 'FAIL'} (valid, all keys present)")

    # Figures
    figs_ok = all((FIGURES_DIR / f).exists() for f in EXPECTED_FIGURES)
    print(f"Figures: {'PASS' if figs_ok else 'FAIL'} ({len(EXPECTED_FIGURES)} PNGs)")

    # Parquet
    pq_ok = df_data.height == 150_135 and all(f in df_data.columns for f in MODEL_FEATURES)
    print(f"Parquet: {'PASS' if pq_ok else 'FAIL'} ({df_data.height:,} rows, all features)")
