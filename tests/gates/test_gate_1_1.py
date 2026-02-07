"""Gate Test 1.1: ENAHO Single-Year Loader Validation

Validates that load_enaho_year(2023) produces correct output:
- Row count in expected range
- Dropout count in expected range
- Weighted dropout rate in expected range
- UBIGEO integrity (all 6 chars)
- Zero nulls on critical columns
- Column schema validation
- Prints 10 random dropout rows for human inspection
"""

import sys

sys.path.insert(0, "src")

import polars as pl
from data.enaho import load_enaho_year


def test_gate_1_1():
    result = load_enaho_year(2023)
    df = result.df
    stats = result.stats

    # Print warnings if any
    if result.warnings:
        print("\n--- WARNINGS ---")
        for w in result.warnings:
            print(f"  WARN: {w}")

    # --- Row count ---
    total = stats["total_rows"]
    assert 18_000 < total < 35_000, f"FAIL: row count {total} outside [18K, 35K]"
    if 20_000 <= total <= 30_000:
        print(f"  PASS: row count = {total}")
    else:
        print(
            f"  WARN: row count = {total} (outside ideal 20K-30K but within tolerance)"
        )

    # --- Dropout count ---
    dropouts = stats["dropout_count"]
    assert 2_000 < dropouts < 6_000, f"FAIL: dropout count {dropouts} outside [2K, 6K]"
    if 2_500 <= dropouts <= 5_000:
        print(f"  PASS: dropout count = {dropouts}")
    else:
        print(
            f"  WARN: dropout count = {dropouts} (outside ideal 2.5K-5K but within tolerance)"
        )

    # --- Weighted dropout rate ---
    rate = stats["weighted_dropout_rate"]
    assert 0.08 < rate < 0.22, f"FAIL: weighted rate {rate:.4f} outside [0.08, 0.22]"
    if 0.10 <= rate <= 0.18:
        print(f"  PASS: weighted dropout rate = {rate:.4f}")
    else:
        print(
            f"  WARN: weighted dropout rate = {rate:.4f} (outside ideal 0.10-0.18 but within tolerance)"
        )

    # --- UBIGEO integrity ---
    ubigeo_lengths = df.select(pl.col("UBIGEO").str.len_chars()).to_series()
    assert (ubigeo_lengths == 6).all(), "FAIL: not all UBIGEO values are 6 chars"
    print("  PASS: all UBIGEO values are 6 characters")

    # --- Null checks on critical columns ---
    critical_cols = ["UBIGEO", "P208A", "P303", "P306", "FACTOR07"]
    for col in critical_cols:
        null_count = df.select(pl.col(col).is_null().sum()).item()
        assert null_count == 0, f"FAIL: {col} has {null_count} nulls"
    print(f"  PASS: zero nulls on critical columns {critical_cols}")

    # --- Column schema validation ---
    expected_cols = [
        "UBIGEO",
        "P208A",
        "P207",
        "P303",
        "P306",
        "FACTOR07",
        "dropout",
    ]
    for col in expected_cols:
        assert col in df.columns, f"FAIL: missing column {col}"
    print("  PASS: all expected columns present")

    # --- Print 10 random dropout rows for human inspection ---
    inspection_cols = [
        "UBIGEO",
        "P208A",
        "P207",
        "P300A",
        "P303",
        "P306",
        "P307",
        "P301A",
        "FACTOR07",
        "dropout",
    ]
    available_cols = [c for c in inspection_cols if c in df.columns]
    dropout_rows = df.filter(pl.col("dropout"))
    sample = dropout_rows.select(available_cols).sample(
        n=min(10, len(dropout_rows)), seed=42
    )

    print("\n" + "=" * 80)
    print("HUMAN INSPECTION: 10 Random Dropout Rows")
    print("=" * 80)
    print(f"  Columns: {available_cols}")
    print(sample)
    print("=" * 80)

    # --- Summary ---
    print(f"\n--- GATE TEST 1.1 SUMMARY ---")
    print(f"  Total school-age rows: {total}")
    print(f"  Unweighted dropout count: {dropouts}")
    print(f"  Weighted dropout rate: {rate:.4f}")
    print(f"  Year: {stats['year']}")
    print(f"  Warnings: {len(result.warnings)}")
