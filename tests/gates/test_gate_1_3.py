"""Gate Test 1.3: Administrative Dropout Rate Merge Validation

Validates the admin data loading and merge with ENAHO:
- All UBIGEO exactly 6 characters
- Admin mean rates within 20% of expected (0.93% primaria, 2.05% secundaria)
- 25 unique departments represented
- ENAHO-to-admin merge rate > 85%
- Row count unchanged after admin merge
- No duplicate UBIGEO in admin data
- Spot-check: Lima low dropout, Amazonas high dropout
"""

import sys

sys.path.insert(0, "src")

import polars as pl
from data.admin import load_admin_dropout_rates
from data.enaho import load_all_years


def test_gate_1_3():
    # Load admin data
    admin_result = load_admin_dropout_rates()
    admin_df = admin_result.df

    # Load ENAHO for merge testing
    enaho_result = load_all_years()
    enaho_df = enaho_result.df

    # Print warnings
    if admin_result.warnings:
        print("\n--- ADMIN WARNINGS ---")
        for w in admin_result.warnings:
            print(f"  WARN: {w}")

    # --- UBIGEO format: all exactly 6 characters ---
    bad_ubigeo = admin_df.filter(pl.col("UBIGEO").str.len_chars() != 6)
    assert bad_ubigeo.height == 0, (
        f"FAIL: {bad_ubigeo.height} UBIGEO values have length != 6"
    )
    print("  PASS: all UBIGEO values are 6 characters")

    # --- Admin rate validation: within 20% of expected ---
    expected_primaria = 0.93
    expected_secundaria = 2.05
    primaria_rate = admin_result.primaria_rate
    secundaria_rate = admin_result.secundaria_rate

    assert abs(primaria_rate - expected_primaria) / expected_primaria < 0.20, (
        f"FAIL: primaria rate {primaria_rate:.4f}% outside 20% of {expected_primaria}%"
    )
    print(f"  PASS: primaria rate = {primaria_rate:.4f}% (expected ~{expected_primaria}%)")

    assert abs(secundaria_rate - expected_secundaria) / expected_secundaria < 0.20, (
        f"FAIL: secundaria rate {secundaria_rate:.4f}% outside 20% of {expected_secundaria}%"
    )
    print(f"  PASS: secundaria rate = {secundaria_rate:.4f}% (expected ~{expected_secundaria}%)")

    # --- Department count: 25 unique departments ---
    departments = admin_df.with_columns(
        pl.col("UBIGEO").str.slice(0, 2).alias("dept")
    )["dept"].unique()
    dept_count = departments.len()
    assert dept_count == 25, (
        f"FAIL: expected 25 departments, got {dept_count}"
    )
    print(f"  PASS: {dept_count} unique departments")

    # --- Admin merge rate > 85% ---
    enaho_ubigeos = enaho_df["UBIGEO"].unique()
    admin_ubigeos = set(admin_df["UBIGEO"].to_list())
    matched = sum(1 for u in enaho_ubigeos.to_list() if u in admin_ubigeos)
    merge_rate = matched / enaho_ubigeos.len()
    assert merge_rate > 0.85, (
        f"FAIL: admin merge rate {merge_rate:.2%} below 85%"
    )
    print(f"  PASS: admin merge rate = {merge_rate:.2%} ({matched}/{enaho_ubigeos.len()} districts)")

    # --- Row count preservation ---
    initial_rows = enaho_df.height
    merged = enaho_df.join(admin_df, on="UBIGEO", how="left", validate="m:1", coalesce=True)
    assert merged.height == initial_rows, (
        f"FAIL: row count changed {initial_rows} -> {merged.height}"
    )
    print(f"  PASS: row count preserved ({initial_rows:,} rows)")

    # --- No duplicate UBIGEO in admin data ---
    assert admin_df["UBIGEO"].n_unique() == admin_df.height, (
        f"FAIL: duplicate UBIGEO in admin data "
        f"({admin_df.height - admin_df['UBIGEO'].n_unique()} duplicates)"
    )
    print(f"  PASS: no duplicate UBIGEO ({admin_df.height} unique districts)")

    # --- Spot-check districts: Lima low, Amazonas high ---
    lima_rates = admin_df.filter(
        pl.col("UBIGEO").str.starts_with("15")
    )["admin_primaria_rate"].drop_nulls()
    amazonas_rates = admin_df.filter(
        pl.col("UBIGEO").str.starts_with("01")
    )["admin_primaria_rate"].drop_nulls()

    lima_mean = lima_rates.mean() if lima_rates.len() > 0 else 0.0
    amazonas_mean = amazonas_rates.mean() if amazonas_rates.len() > 0 else 0.0

    assert amazonas_mean > lima_mean, (
        f"FAIL: Amazonas ({amazonas_mean:.2f}%) should have higher dropout than "
        f"Lima ({lima_mean:.2f}%)"
    )
    print(f"  PASS: Amazonas primaria ({amazonas_mean:.2f}%) > Lima ({lima_mean:.2f}%) -- directionally correct")

    # --- Summary ---
    print(f"\n--- GATE TEST 1.3 SUMMARY ---")
    print(f"  Admin districts: {admin_result.districts_count}")
    print(f"  Primaria rate: {primaria_rate:.4f}%")
    print(f"  Secundaria rate: {secundaria_rate:.4f}%")
    print(f"  Admin merge rate: {merge_rate:.2%}")
    print(f"  Departments: {dept_count}")
    print(f"  Warnings: {len(admin_result.warnings)}")
