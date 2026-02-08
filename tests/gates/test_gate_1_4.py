"""Gate Test 1.4: Census/Nightlights Merge and Full Dataset Validation

Validates the complete spatial merge pipeline:
- Census merge rate > 90%
- Nightlights: no negative values, coverage > 85%
- No duplicate rows in final dataset
- Final column count includes all sources
- Columns with >10% nulls reported for human review
- All three sources successfully merged
"""

import sys

sys.path.insert(0, "src")

import polars as pl
from data.enaho import load_all_years
from data.merge import merge_spatial_data


def test_gate_1_4():
    # Load ENAHO and run full merge pipeline
    enaho_result = load_all_years()
    enaho_df = enaho_result.df
    merge_result = merge_spatial_data(enaho_df)
    merged_df = merge_result.df

    print("\n--- MERGE PIPELINE RESULTS ---")
    print(f"  Initial ENAHO rows: {merge_result.initial_rows:,}")
    print(f"  Final merged rows:  {merge_result.final_rows:,}")
    print(f"  Total columns: {merged_df.width}")

    # --- Census merge rate > 90% ---
    census_rate = merge_result.merge_rates.get("census", 0.0)
    assert census_rate > 0.90, (
        f"FAIL: census merge rate {census_rate:.2%} below 90%"
    )
    print(f"  PASS: census merge rate = {census_rate:.2%}")

    # --- Nightlights: no negative values ---
    if "nightlights_mean_radiance" in merged_df.columns:
        non_null_nl = merged_df.filter(
            pl.col("nightlights_mean_radiance").is_not_null()
        )
        negative_count = non_null_nl.filter(
            pl.col("nightlights_mean_radiance") < 0
        ).height
        assert negative_count == 0, (
            f"FAIL: {negative_count} negative nightlights values"
        )
        print(f"  PASS: no negative nightlights values")
    else:
        print("  SKIP: nightlights column not present")

    # --- Nightlights coverage > 85% ---
    nl_rate = merge_result.merge_rates.get("nightlights", 0.0)
    assert nl_rate > 0.85, (
        f"FAIL: nightlights merge rate {nl_rate:.2%} below 85%"
    )
    print(f"  PASS: nightlights merge rate = {nl_rate:.2%}")

    # --- No duplicate rows ---
    has_dupes = merged_df.is_duplicated().any()
    assert not has_dupes, "FAIL: duplicate rows found in merged dataset"
    print("  PASS: no duplicate rows")

    # --- Row count preservation ---
    assert merge_result.initial_rows == merge_result.final_rows, (
        f"FAIL: row count changed {merge_result.initial_rows} -> {merge_result.final_rows}"
    )
    print(f"  PASS: row count preserved ({merge_result.final_rows:,} rows)")

    # --- Column count validation ---
    enaho_cols = set(enaho_df.columns)
    merged_cols = set(merged_df.columns)
    new_cols = merged_cols - enaho_cols
    print(f"\n  New columns added ({len(new_cols)}):")
    for col in sorted(new_cols):
        null_rate = merged_df[col].null_count() / merged_df.height
        print(f"    {col}: {null_rate:.2%} nulls")

    # --- Null reporting: columns with >10% nulls ---
    print(f"\n--- COLUMNS WITH >10% NULLS ---")
    high_null_cols = {}
    for col in sorted(new_cols):
        null_rate = merged_df[col].null_count() / merged_df.height
        if null_rate > 0.10:
            high_null_cols[col] = null_rate
            print(f"  {col}: {null_rate:.2%}")
    if not high_null_cols:
        print("  None -- all new columns have <10% nulls")

    # --- Merge completeness: all three sources present ---
    admin_present = any(c.startswith("admin_") for c in merged_df.columns)
    census_present = any(c.startswith("census_") for c in merged_df.columns)
    nightlights_present = any(
        c.startswith("nightlights_") for c in merged_df.columns
    )

    assert admin_present, "FAIL: admin columns not found in merged dataset"
    assert census_present, "FAIL: census columns not found in merged dataset"
    assert nightlights_present, (
        "FAIL: nightlights columns not found in merged dataset"
    )
    print(f"\n  PASS: all three sources merged successfully")
    print(f"    Admin columns: {[c for c in merged_df.columns if c.startswith('admin_')]}")
    print(f"    Census columns: {[c for c in merged_df.columns if c.startswith('census_')]}")
    print(f"    Nightlights columns: {[c for c in merged_df.columns if c.startswith('nightlights_')]}")

    # --- Admin merge rate > 85% ---
    admin_rate = merge_result.merge_rates.get("admin", 0.0)
    assert admin_rate > 0.85, (
        f"FAIL: admin merge rate {admin_rate:.2%} below 85%"
    )
    print(f"\n  PASS: admin merge rate = {admin_rate:.2%}")

    # --- Summary ---
    print(f"\n--- GATE TEST 1.4 SUMMARY ---")
    print(f"  ENAHO rows: {merge_result.initial_rows:,}")
    print(f"  Final rows: {merge_result.final_rows:,}")
    print(f"  Columns: {merged_df.width} ({len(new_cols)} new)")
    print(f"  Merge rates: admin={admin_rate:.2%}, census={census_rate:.2%}, nightlights={nl_rate:.2%}")
    print(f"  High-null columns: {len(high_null_cols)}")
    print(f"  Warnings: {len(merge_result.warnings)}")
    if merge_result.warnings:
        for w in merge_result.warnings:
            print(f"    WARN: {w}")
