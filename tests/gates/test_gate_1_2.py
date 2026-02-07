"""Gate Test 1.2: Multi-Year Loader + Harmonization Validation

Validates that load_all_years() produces a correctly pooled dataset:
- Pooled row count in expected range (~150K-160K)
- All 6 years present (2018-2023)
- Harmonization columns exist and are correct
- No disaggregated codes (10-15) in harmonized column
- Harmonization stability: code 3 proportion ratio < 2.0x
- Dropout count in expected range (~18K+)
- Column schema validation
- Per-year stats summary printed
"""

import sys
sys.path.insert(0, "src")

import polars as pl
from data.enaho import load_all_years


def test_gate_1_2():
    result = load_all_years()
    df = result.df

    # Print warnings if any
    if result.warnings:
        print("\n--- WARNINGS ---")
        for w in result.warnings:
            print(f"  WARN: {w}")

    # --- Pooled row count ---
    total = df.height
    assert 130_000 <= total <= 190_000, f"FAIL: pooled rows {total} outside [130K, 190K]"
    print(f"  PASS: pooled row count = {total:,}")

    # --- Year coverage ---
    years = sorted(df["year"].unique().to_list())
    assert years == [2018, 2019, 2020, 2021, 2022, 2023], (
        f"FAIL: expected years 2018-2023, got {years}"
    )
    print(f"  PASS: year coverage = {years}")

    # --- Per-year row counts ---
    print("\n--- PER-YEAR BREAKDOWN ---")
    for year in years:
        year_df = df.filter(pl.col("year") == year)
        n = year_df.height
        n_dropout = year_df.filter(pl.col("dropout")).height
        wt_total = year_df["FACTOR07"].sum()
        wt_dropout = year_df.filter(pl.col("dropout"))["FACTOR07"].sum()
        wt_rate = wt_dropout / wt_total if wt_total > 0 else 0.0
        print(f"  {year}: {n:>7,} rows, {n_dropout:>5,} dropouts, {wt_rate:.2%} weighted rate")

    # --- Harmonization columns exist ---
    assert "p300a_original" in df.columns, "FAIL: missing p300a_original"
    assert "p300a_harmonized" in df.columns, "FAIL: missing p300a_harmonized"
    print("  PASS: harmonization columns present")

    # --- No disaggregated codes in harmonized column ---
    harmonized_vals = df["p300a_harmonized"].drop_nulls().unique().to_list()
    for code in [10, 11, 12, 13, 14, 15]:
        assert code not in harmonized_vals, (
            f"FAIL: disaggregated code {code} found in p300a_harmonized"
        )
    print("  PASS: no disaggregated codes (10-15) in harmonized column")

    # --- Disaggregated codes present in original for 2020+ ---
    post2020 = df.filter(pl.col("year") >= 2020)
    original_vals = post2020["p300a_original"].drop_nulls().unique().to_list()
    disagg_present = [c for c in [10, 11, 12, 13, 14, 15] if c in original_vals]
    assert len(disagg_present) > 0, (
        "FAIL: no disaggregated codes found in p300a_original for 2020+"
    )
    print(f"  PASS: disaggregated codes in p300a_original for 2020+: {sorted(disagg_present)}")

    # --- Harmonization stability: code 3 proportion ---
    proportions = {}
    for year in years:
        year_df = df.filter(pl.col("year") == year)
        n_code3 = year_df.filter(pl.col("p300a_harmonized") == 3).height
        prop = n_code3 / year_df.height if year_df.height > 0 else 0
        proportions[year] = prop

    ratio = max(proportions.values()) / min(proportions.values()) if min(proportions.values()) > 0 else float("inf")
    assert ratio < 2.0, (
        f"FAIL: harmonized code 3 proportion ratio {ratio:.2f} exceeds 2.0x. "
        f"Proportions: {proportions}"
    )
    print(f"  PASS: harmonization stability ratio = {ratio:.2f} (< 2.0x)")
    for year, prop in sorted(proportions.items()):
        print(f"    {year}: {prop:.3%} code 3 (harmonized)")

    # --- Dropout count ---
    total_dropouts = df.filter(pl.col("dropout")).height
    assert total_dropouts >= 18_000, (
        f"FAIL: expected 18K+ dropouts, got {total_dropouts:,}"
    )
    print(f"  PASS: total dropouts = {total_dropouts:,}")

    # --- Column schema validation ---
    expected_cols = [
        "UBIGEO", "P208A", "P207", "P303", "P306",
        "FACTOR07", "dropout", "year",
        "p300a_original", "p300a_harmonized",
    ]
    for col in expected_cols:
        assert col in df.columns, f"FAIL: missing column {col}"
    print(f"  PASS: all expected columns present")

    # --- Per-year stats from loader ---
    assert len(result.per_year_stats) == 6, (
        f"FAIL: expected 6 per-year stats, got {len(result.per_year_stats)}"
    )
    print(f"  PASS: {len(result.per_year_stats)} per-year stats collected")

    # --- Summary ---
    print(f"\n--- GATE TEST 1.2 SUMMARY ---")
    print(f"  Pooled rows: {total:,}")
    print(f"  Years: {years}")
    print(f"  Total dropouts: {total_dropouts:,}")
    print(f"  Harmonization stability: {ratio:.2f}x")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"  Columns: {sorted(df.columns)}")
