"""Pooled validation+test CI for urban other-indigenous subgroup."""
import json
import numpy as np
import polars as pl
from pathlib import Path
from fairness.bootstrap import _fast_weighted_fnr

ROOT = Path(__file__).resolve().parents[2]


def compute_pooled_ci(
    n_replicates: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Pool val+test urban other-indigenous, compute FNR with weighted bootstrap CI."""
    # 1. Load predictions + features
    preds = pl.read_parquet(ROOT / "data/processed/predictions_lgbm_calibrated.parquet")
    features = pl.read_parquet(ROOT / "data/processed/enaho_with_features.parquet")

    # 2. Filter to urban other-indigenous (BOTH splits)
    # Join to get rurality and language
    features_slim = features.select(["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year",
                                      "rural", "lang_other_indigenous"])
    merged = preds.join(features_slim, on=["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"])

    urban_oi = merged.filter(
        (pl.col("rural") == 0) & (pl.col("lang_other_indigenous") == 1)
    )

    # 3. Extract numpy arrays
    y_true = urban_oi["dropout"].cast(pl.Int8).to_numpy()
    y_pred = urban_oi["pred_dropout"].to_numpy()
    weights = urban_oi["FACTOR07"].to_numpy()
    n = len(y_true)
    n_dropouts = int(y_true.sum())

    # 4. Point estimate
    fnr_point = float(_fast_weighted_fnr(y_true, y_pred, weights))

    # 5. Simple weighted bootstrap (NOT PSU -- pooling breaks survey design)
    rng = np.random.default_rng(seed)
    boot_fnr = np.empty(n_replicates)
    valid = 0
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        val = _fast_weighted_fnr(y_true[idx], y_pred[idx], weights[idx])
        if np.isfinite(val):
            boot_fnr[valid] = val
            valid += 1
    boot_fnr = boot_fnr[:valid]
    ci_lower = float(np.percentile(boot_fnr, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_fnr, (1 - alpha / 2) * 100))

    # 6. Credibility assessment
    credible = ci_lower > 0.50  # "model misses majority" holds

    result = {
        "subgroup": "other_indigenous_urban",
        "pooled_splits": ["validate_2022", "test_2023"],
        "n_unweighted": n,
        "n_dropouts_unweighted": n_dropouts,
        "fnr_point": fnr_point,
        "fnr_ci_lower": ci_lower,
        "fnr_ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "credible": credible,
        "credibility_criterion": "ci_lower > 0.50",
        "bootstrap_method": "simple_weighted",
        "bootstrap_note": "PSU bootstrap inappropriate -- pooling two survey years breaks single-design assumption",
        "n_replicates": n_replicates,
        "includes_non_test_data": True,
    }

    # 7. Save
    output_path = ROOT / "data/exports/pooled_ci_urban_indigenous.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Pooled n={n} (dropouts={n_dropouts}), FNR={fnr_point:.3f}, "
          f"CI=[{ci_lower:.3f}, {ci_upper:.3f}], credible={credible}")
    return result


if __name__ == "__main__":
    compute_pooled_ci()
