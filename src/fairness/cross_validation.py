"""District-level cross-validation of model predictions against admin dropout rates.

Aggregates calibrated LightGBM predictions to district level, correlates with
MINEDU administrative dropout rates, quantifies prediction error by indigenous
language prevalence, and exports choropleth.json for the M4 scrollytelling site.

Usage::

    uv run python src/fairness/cross_validation.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.admin import load_admin_dropout_rates  # noqa: E402
from data.census import load_census_2017  # noqa: E402
from data.features import MODEL_FEATURES  # noqa: E402
from utils import find_project_root  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(v: object) -> object:
    """Convert NaN/None to None, round floats to 4 decimal places."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, float):
        return round(v, 4)
    return v


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_cross_validation_pipeline() -> dict:
    """Run district-level cross-validation and export choropleth.json.

    Returns
    -------
    dict
        The choropleth.json content.
    """
    root = find_project_root()
    export_dir = root / "data" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DISTRICT-LEVEL CROSS-VALIDATION")
    print("=" * 70)

    # ===================================================================
    # Step 1: Load model + full dataset
    # ===================================================================
    print("\nStep 1: Loading model and full dataset...")

    cal_model = joblib.load(root / "data" / "processed" / "model_lgbm_calibrated.joblib")
    print(f"  Calibrated model loaded: {type(cal_model).__name__}")

    feat = pl.read_parquet(root / "data" / "processed" / "enaho_with_features.parquet")
    print(f"  Feature matrix: {feat.height:,} rows x {feat.width} columns")
    print(f"  Unique districts (UBIGEO): {feat['UBIGEO'].n_unique():,}")
    print(f"  Years: {sorted(feat['year'].unique().to_list())}")

    # ===================================================================
    # Step 2: Score ALL 150,135 rows
    # ===================================================================
    print("\nStep 2: Scoring all rows through calibrated model...")

    X = feat.select(list(MODEL_FEATURES)).to_pandas().to_numpy()
    cal_probs = cal_model.predict_proba(X)[:, 1]

    scored = feat.with_columns(pl.Series("prob_dropout", cal_probs))
    print(f"  Scored: {scored.height:,} rows")
    print(f"  Prob range: [{cal_probs.min():.4f}, {cal_probs.max():.4f}]")

    # ===================================================================
    # Step 3: Aggregate to district level (FACTOR07-weighted)
    # ===================================================================
    print("\nStep 3: Aggregating to district level...")

    # All-years aggregation (for choropleth)
    district_pred = (
        scored
        .group_by("UBIGEO")
        .agg([
            pl.len().alias("n_students"),
            pl.col("FACTOR07").sum().alias("total_weight"),
            (pl.col("prob_dropout") * pl.col("FACTOR07")).sum().alias("_ws"),
            (pl.col("dropout").cast(pl.Float64) * pl.col("FACTOR07")).sum().alias("_wd"),
            pl.col("poverty_index").mean().alias("poverty_index"),
        ])
        .with_columns([
            (pl.col("_ws") / pl.col("total_weight") * 100).alias("predicted_dropout_rate"),
            (pl.col("_wd") / pl.col("total_weight") * 100).alias("actual_dropout_rate_enaho"),
        ])
        .drop("_ws", "_wd")
    )

    print(f"  All-years districts: {district_pred.height:,}")
    print(f"  Predicted dropout rate range: "
          f"[{district_pred['predicted_dropout_rate'].min():.2f}%, "
          f"{district_pred['predicted_dropout_rate'].max():.2f}%]")

    # Test-2023-only aggregation (for pure out-of-sample correlation)
    test_2023 = scored.filter(pl.col("year") == 2023)
    print(f"\n  Test 2023 rows: {test_2023.height:,}")

    district_test = (
        test_2023
        .group_by("UBIGEO")
        .agg([
            pl.len().alias("n_students_test"),
            pl.col("FACTOR07").sum().alias("total_weight_test"),
            (pl.col("prob_dropout") * pl.col("FACTOR07")).sum().alias("_ws"),
        ])
        .with_columns(
            (pl.col("_ws") / pl.col("total_weight_test") * 100).alias("predicted_dropout_rate_test")
        )
        .drop("_ws")
    )
    print(f"  Test 2023 districts: {district_test.height:,}")

    # ===================================================================
    # Step 4: Load admin + census data and join
    # ===================================================================
    print("\nStep 4: Loading admin and census data...")

    admin_result = load_admin_dropout_rates()
    admin_df = admin_result.df
    print(f"  Admin districts: {admin_df.height:,}")

    # Compute combined admin rate: average of primaria and secundaria
    # where both exist; whichever is available where one is null
    admin_df = admin_df.with_columns(
        pl.when(
            pl.col("admin_primaria_rate").is_not_null()
            & pl.col("admin_secundaria_rate").is_not_null()
        )
        .then((pl.col("admin_primaria_rate") + pl.col("admin_secundaria_rate")) / 2.0)
        .when(pl.col("admin_secundaria_rate").is_not_null())
        .then(pl.col("admin_secundaria_rate"))
        .otherwise(pl.col("admin_primaria_rate"))
        .alias("admin_dropout_rate")
    )

    census_result = load_census_2017()
    census_df = census_result.df.select([
        "UBIGEO",
        pl.col("census_indigenous_lang_pct").alias("indigenous_language_pct"),
        pl.col("census_poverty_rate").alias("census_poverty_rate"),
    ])
    print(f"  Census districts: {census_df.height:,}")

    # LEFT JOIN admin onto district predictions
    merged = district_pred.join(
        admin_df.select(["UBIGEO", "admin_dropout_rate"]),
        on="UBIGEO",
        how="left",
    )

    # LEFT JOIN census
    merged = merged.join(census_df, on="UBIGEO", how="left")

    print(f"  Merged (ENAHO districts): {merged.height:,}")

    # FULL OUTER JOIN to include admin-only districts for complete choropleth
    full_choropleth = district_pred.join(
        admin_df.select(["UBIGEO", "admin_dropout_rate"]),
        on="UBIGEO",
        how="full",
        coalesce=True,
    )
    full_choropleth = full_choropleth.join(census_df, on="UBIGEO", how="left")

    n_admin_only = full_choropleth.filter(
        pl.col("predicted_dropout_rate").is_null()
    ).height
    print(f"  Full choropleth districts: {full_choropleth.height:,}")
    print(f"  Admin-only (no ENAHO): {n_admin_only:,}")

    # ===================================================================
    # Step 5: Pearson correlation (test_2023 only for purity)
    # ===================================================================
    print("\nStep 5: Computing Pearson correlation (test 2023 only)...")

    # Join test predictions with admin data
    corr_df = district_test.join(
        admin_df.select(["UBIGEO", "admin_dropout_rate"]),
        on="UBIGEO",
        how="inner",
    )

    # Filter to districts with both predicted and admin rates
    valid_corr = corr_df.filter(
        pl.col("predicted_dropout_rate_test").is_not_null()
        & pl.col("admin_dropout_rate").is_not_null()
    )

    pred_rates = valid_corr["predicted_dropout_rate_test"].to_numpy()
    admin_rates = valid_corr["admin_dropout_rate"].to_numpy()

    r_val, p_val = pearsonr(pred_rates, admin_rates)
    n_corr = len(pred_rates)

    print(f"  Pearson r = {r_val:.6f}")
    print(f"  p-value   = {p_val:.6e}")
    print(f"  N districts correlated = {n_corr:,}")
    print(f"  Caveat: Model uses district_dropout_rate_admin_z as feature "
          f"(SHAP importance ~0.03)")

    # ===================================================================
    # Step 6: Stratified error analysis
    # ===================================================================
    print("\nStep 6: Stratified error analysis...")

    # Add model_error to full choropleth (only where both rates exist)
    full_choropleth = full_choropleth.with_columns(
        (pl.col("predicted_dropout_rate") - pl.col("admin_dropout_rate")).alias("model_error")
    )

    # Filter to districts with BOTH predicted and admin rates + indigenous data
    error_df = full_choropleth.filter(
        pl.col("predicted_dropout_rate").is_not_null()
        & pl.col("admin_dropout_rate").is_not_null()
        & pl.col("indigenous_language_pct").is_not_null()
    )

    high_indig = error_df.filter(pl.col("indigenous_language_pct") > 50)
    low_indig = error_df.filter(pl.col("indigenous_language_pct") < 10)

    high_mae = float(high_indig["model_error"].abs().mean()) if high_indig.height > 0 else 0.0
    low_mae = float(low_indig["model_error"].abs().mean()) if low_indig.height > 0 else 0.0

    print(f"  High-indigenous (>50%) districts: {high_indig.height:,}")
    print(f"  High-indigenous MAE: {high_mae:.4f} pp")
    print(f"  Low-indigenous  (<10%) districts: {low_indig.height:,}")
    print(f"  Low-indigenous  MAE: {low_mae:.4f} pp")
    print(f"  MAE difference: {abs(high_mae - low_mae):.4f} pp")

    # ===================================================================
    # Step 7: Export choropleth.json
    # ===================================================================
    print("\nStep 7: Exporting choropleth.json...")

    districts_list = []
    for row in full_choropleth.iter_rows(named=True):
        record = {
            "ubigeo": row["UBIGEO"],
            "predicted_dropout_rate": _safe(row.get("predicted_dropout_rate")),
            "admin_dropout_rate": _safe(row.get("admin_dropout_rate")),
            "model_error": _safe(row.get("model_error")),
            "indigenous_language_pct": _safe(row.get("indigenous_language_pct")),
            "poverty_index": _safe(row.get("poverty_index")),
        }
        districts_list.append(record)

    n_with_predictions = sum(
        1 for d in districts_list if d["predicted_dropout_rate"] is not None
    )

    choropleth_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_districts": len(districts_list),
        "n_with_predictions": n_with_predictions,
        "correlation": {
            "pearson_r": round(float(r_val), 6),
            "p_value": float(p_val),
            "n_districts_correlated": n_corr,
            "caveat": (
                "Model uses district_dropout_rate_admin_z as feature "
                "(SHAP importance ~0.03). Correlation computed on test_2023 "
                "districts only (out-of-sample)."
            ),
        },
        "error_by_indigenous_group": {
            "high_indigenous_gt50": {
                "n_districts": high_indig.height,
                "mean_absolute_error": round(high_mae, 4),
            },
            "low_indigenous_lt10": {
                "n_districts": low_indig.height,
                "mean_absolute_error": round(low_mae, 4),
            },
        },
        "districts": districts_list,
    }

    output_path = export_dir / "choropleth.json"
    with open(output_path, "w") as f:
        json.dump(choropleth_data, f, indent=2, ensure_ascii=False)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({file_size_kb:.1f} KB)")

    # Verify no NaN in JSON
    with open(output_path) as f:
        raw_text = f.read()
    assert "NaN" not in raw_text, "JSON contains NaN -- invalid JSON"
    assert "Infinity" not in raw_text, "JSON contains Infinity -- invalid JSON"
    print("  JSON validation: no NaN or Infinity values")

    # ===================================================================
    # Step 8: Console summary for human review
    # ===================================================================
    print()
    print("=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n--- CORRELATION (test 2023, out-of-sample) ---")
    print(f"  Pearson r:     {r_val:.6f}")
    print(f"  p-value:       {p_val:.6e}")
    print(f"  N districts:   {n_corr:,}")

    print(f"\n--- PREDICTION ERROR BY INDIGENOUS GROUP ---")
    print(f"  High-indigenous (>50%): N={high_indig.height:,}, MAE={high_mae:.4f} pp")
    print(f"  Low-indigenous  (<10%): N={low_indig.height:,}, MAE={low_mae:.4f} pp")

    print(f"\n--- CHOROPLETH COVERAGE ---")
    print(f"  Total districts:        {len(districts_list):,}")
    print(f"  With predictions:       {n_with_predictions:,}")
    print(f"  Without predictions:    {len(districts_list) - n_with_predictions:,}")

    print(f"\n--- ADMIN DATA NOTE ---")
    print(f"  Admin data is synthetic (generated in Phase 3).")
    print(f"  Findings should be interpreted accordingly.")

    print(f"\n  JSON file: {output_path} ({file_size_kb:.1f} KB)")
    print("=" * 70)

    return choropleth_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_cross_validation_pipeline()
