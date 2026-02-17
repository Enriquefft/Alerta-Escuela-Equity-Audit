"""SHAP interpretability analysis for the Alerta Escuela Equity Audit.

Computes global, regional, and interaction SHAP values for the LightGBM
dropout model on the 2023 test set.  Quantifies ES_PERUANO and ES_MUJER
contributions.  Selects 10 representative student profiles.  Generates 5
publication-quality figures.  Exports M4-schema-compliant shap_values.json.

Usage::

    uv run python src/fairness/shap_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import shap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.features import MODEL_FEATURES  # noqa: E402
from utils import find_project_root  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spanish feature labels (for M4 scrollytelling site)
# ---------------------------------------------------------------------------

FEATURE_LABELS_ES: list[str] = [
    "Edad",                                    # age
    "Edad de secundaria (12+)",                # is_secundaria_age
    "Sexo femenino",                           # es_mujer
    "Lengua castellana",                       # lang_castellano
    "Lengua quechua",                          # lang_quechua
    "Lengua aimara",                           # lang_aimara
    "Otra lengua indigena",                    # lang_other_indigenous
    "Lengua extranjera",                       # lang_foreign
    "Zona rural",                              # rural
    "Region Sierra",                           # is_sierra
    "Region Selva",                            # is_selva
    "Tasa de desercion distrital (admin, z)",  # district_dropout_rate_admin_z
    "Intensidad de luces nocturnas (z)",       # nightlight_intensity_z
    "Indice de pobreza (z)",                   # poverty_index_z
    "Quintil de pobreza",                      # poverty_quintile
    "Nacionalidad peruana",                    # es_peruano
    "Tiene discapacidad",                      # has_disability
    "Trabaja",                                 # is_working
    "Beneficiario JUNTOS",                     # juntos_participant
    "Ingreso del hogar (log)",                 # log_income
    "Educacion de los padres (anos)",          # parent_education_years
    "Poblacion indigena del distrito (z)",     # census_indigenous_lang_pct_z
    "Tasa de alfabetismo del distrito (z)",    # census_literacy_rate_z
    "Acceso a electricidad del distrito (z)",  # census_electricity_pct_z
    "Acceso a agua del distrito (z)",          # census_water_access_pct_z
]

# LR top-5 by |coefficient| (from model_results.json)
LR_TOP_5: list[str] = [
    "lang_other_indigenous",
    "lang_foreign",
    "lang_quechua",
    "is_secundaria_age",
    "lang_aimara",
]

# Profile type definitions: (profile_id, description_es, filter_function)
PROFILE_DEFINITIONS: list[tuple[str, str, dict]] = [
    (
        "lima_urban_castellano_male",
        "Estudiante masculino, castellano, zona urbana de Lima",
        {"department": "15", "rural": 0, "lang_castellano": 1, "es_mujer": 0},
    ),
    (
        "lima_urban_foreign",
        "Estudiante extranjero en zona urbana de Lima",
        {"department": "15", "rural": 0, "es_peruano": 0},
    ),
    (
        "sierra_rural_quechua",
        "Estudiante quechuahablante rural de la Sierra",
        {"region_natural": "sierra", "rural": 1, "lang_quechua": 1},
    ),
    (
        "sierra_rural_castellano",
        "Estudiante castellanohablante rural de la Sierra",
        {"region_natural": "sierra", "rural": 1, "lang_castellano": 1},
    ),
    (
        "selva_rural_indigenous",
        "Estudiante de lengua indigena rural de la Selva",
        {"region_natural": "selva", "rural": 1, "lang_other_indigenous": 1},
    ),
    (
        "selva_rural_castellano",
        "Estudiante castellanohablante rural de la Selva",
        {"region_natural": "selva", "rural": 1, "lang_castellano": 1},
    ),
    (
        "female_secundaria_urban",
        "Estudiante femenina de secundaria, zona urbana",
        {"es_mujer": 1, "is_secundaria_age": 1, "rural": 0},
    ),
    (
        "female_secundaria_rural",
        "Estudiante femenina de secundaria, zona rural",
        {"es_mujer": 1, "is_secundaria_age": 1, "rural": 1},
    ),
    (
        "male_secundaria_urban",
        "Estudiante masculino de secundaria, zona urbana",
        {"es_mujer": 0, "is_secundaria_age": 1, "rural": 0},
    ),
    (
        "male_secundaria_rural",
        "Estudiante masculino de secundaria, zona rural",
        {"es_mujer": 0, "is_secundaria_age": 1, "rural": 1},
    ),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _select_representative(probas: np.ndarray, mask: np.ndarray) -> int:
    """Select the row whose predicted probability is closest to the group median.

    Parameters
    ----------
    probas : np.ndarray
        Calibrated predicted probabilities for all test rows.
    mask : np.ndarray
        Boolean mask selecting group members.

    Returns
    -------
    int
        Index into the full test array of the representative row.
    """
    group_proba = probas[mask]
    median_prob = np.median(group_proba)
    group_indices = np.where(mask)[0]
    closest_idx = group_indices[np.argmin(np.abs(group_proba - median_prob))]
    return int(closest_idx)


def _build_profile_mask(
    merged_df: pl.DataFrame, filters: dict
) -> np.ndarray:
    """Build a boolean mask for profile selection from filter criteria."""
    mask = np.ones(merged_df.height, dtype=bool)
    for col, val in filters.items():
        col_values = merged_df[col].to_numpy()
        if isinstance(val, str):
            mask = mask & (col_values == val)
        else:
            mask = mask & (col_values == val)
    return mask


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_shap_pipeline() -> dict:
    """Run the full SHAP interpretability analysis pipeline.

    Returns
    -------
    dict
        The shap_values.json content.
    """
    root = find_project_root()
    fig_dir = root / "data" / "exports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Step 0: Load data
    # ===================================================================
    print("=" * 70)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    print("\nStep 0: Loading model and data...")

    # Load raw LightGBM model (NOT calibrated)
    lgbm = joblib.load(root / "data" / "processed" / "model_lgbm.joblib")
    print(f"  Model: LightGBM ({lgbm.n_estimators_} trees)")

    # Load predictions and filter to test_2023
    pred = pl.read_parquet(
        root / "data" / "processed" / "predictions_lgbm_calibrated.parquet"
    )
    test_pred = pred.filter(pl.col("split") == "test_2023")
    assert test_pred.height == 25635, (
        f"Expected 25,635 test rows, got {test_pred.height}"
    )
    print(f"  Test set: {test_pred.height:,} rows")

    # Load features and join to get MODEL_FEATURES + meta columns
    feat = pl.read_parquet(
        root / "data" / "processed" / "enaho_with_features.parquet"
    )
    join_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"]
    meta_cols = list(MODEL_FEATURES) + [
        "region_natural",
        "department",
        "FACTOR07",
    ]
    # Deduplicate meta_cols (FACTOR07 might already be in predictions)
    feat_select_cols = join_keys + [
        c for c in meta_cols if c not in join_keys
    ]
    # Remove duplicate cols that are already in test_pred
    feat_only_cols = [
        c for c in feat_select_cols
        if c not in test_pred.columns or c in join_keys
    ]

    merged = test_pred.join(
        feat.select(feat_only_cols),
        on=join_keys,
        how="left",
    )
    assert merged.height == test_pred.height, (
        f"Join changed row count: {test_pred.height} -> {merged.height}"
    )
    print(f"  Merged: {merged.height:,} rows, {merged.width} columns")

    # Extract numpy arrays
    X_test = merged.select(list(MODEL_FEATURES)).to_numpy().astype(np.float64)
    probas = merged["prob_dropout"].to_numpy()
    print(f"  X_test shape: {X_test.shape}")

    # Load LR coefficients for overlap check
    with open(root / "data" / "exports" / "model_results.json") as f:
        model_results = json.load(f)
    lr_coefs = model_results["logistic_regression"]["coefficients"]
    lr_sorted = sorted(
        [c for c in lr_coefs if c["feature"] != "intercept"],
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )
    lr_top_5 = [c["feature"] for c in lr_sorted[:5]]
    print(f"  LR top-5 by |coef|: {lr_top_5}")

    # ===================================================================
    # Step 1: Global SHAP
    # ===================================================================
    print("\nStep 1: Computing global SHAP values...")

    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(X_test)

    # Verify shape: single 2D ndarray for binary classification
    assert isinstance(sv, np.ndarray), (
        f"Expected ndarray, got {type(sv)}. "
        "shap 0.50.0 returns single 2D array for LightGBM binary."
    )
    assert sv.shape == (X_test.shape[0], len(MODEL_FEATURES)), (
        f"Expected shape ({X_test.shape[0]}, {len(MODEL_FEATURES)}), got {sv.shape}"
    )
    print(f"  SHAP values shape: {sv.shape}")
    print(f"  Base value (log-odds): {explainer.expected_value:.6f}")

    # Global importance: mean |SHAP| per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    sorted_features = sorted(
        zip(MODEL_FEATURES, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n  TOP-10 GLOBAL SHAP FEATURES (mean |SHAP|):")
    for i, (feat_name, val) in enumerate(sorted_features[:10], 1):
        print(f"    {i:>2d}. {feat_name:<35s} {val:.6f}")

    # Top-5 SHAP features
    shap_top_5 = [f for f, _ in sorted_features[:5]]
    overlap_set = set(shap_top_5) & set(lr_top_5)
    overlap_count = len(overlap_set)
    overlap_list = sorted(overlap_set)

    print(f"\n  SHAP top-5: {shap_top_5}")
    print(f"  LR   top-5: {lr_top_5}")
    print(f"  Overlap: {overlap_count}/5 -- {overlap_list}")

    overlap_note = ""
    if overlap_count < 3:
        overlap_note = (
            "Overlap < 3/5 because LR is a linear model where language dummies "
            "(categorical) dominate via large coefficients, while LightGBM SHAP "
            "captures nonlinear effects of continuous features (age, poverty, "
            "nightlights) that drive tree splits. Both identify equity-relevant "
            "features; the ranking difference reflects model paradigm, not "
            "contradictory findings."
        )
        print(f"  Note: {overlap_note}")

    # ===================================================================
    # Step 2: Beeswarm plot (global)
    # ===================================================================
    print("\nStep 2: Generating beeswarm plot...")
    explanation = explainer(X_test)
    explanation.feature_names = list(MODEL_FEATURES)
    shap.plots.beeswarm(explanation, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(
        fig_dir / "shap_beeswarm_global.png", bbox_inches="tight", dpi=150
    )
    plt.close("all")
    print("  Saved: shap_beeswarm_global.png")

    # ===================================================================
    # Step 3: Top-10 bar plot
    # ===================================================================
    print("\nStep 3: Generating top-10 bar plot...")
    shap.plots.bar(explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(
        fig_dir / "shap_bar_top10.png", bbox_inches="tight", dpi=150
    )
    plt.close("all")
    print("  Saved: shap_bar_top10.png")

    # ===================================================================
    # Step 4: Regional SHAP
    # ===================================================================
    print("\nStep 4: Computing regional SHAP...")

    regions = merged["region_natural"].to_numpy()
    region_names = ["costa", "sierra", "selva"]

    # Per-region mean |SHAP|
    regional_shap: dict[str, dict] = {}
    for region_name in region_names:
        mask = regions == region_name
        region_mean = np.abs(sv[mask]).mean(axis=0)
        regional_shap[region_name] = {
            feat_name: round(float(val), 6)
            for feat_name, val in zip(MODEL_FEATURES, region_mean)
        }

    # Regional comparison plot
    shap.plots.bar(
        explanation.cohorts(regions).abs.mean(0),
        max_display=10,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        fig_dir / "shap_regional_comparison.png", bbox_inches="tight", dpi=150
    )
    plt.close("all")
    print("  Saved: shap_regional_comparison.png")

    # Print regional top-3
    print("\n  REGIONAL TOP-3 FEATURES:")
    for region_name in region_names:
        region_sorted = sorted(
            regional_shap[region_name].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top3 = [f"{f}={v:.4f}" for f, v in region_sorted[:3]]
        n_region = int((regions == region_name).sum())
        print(f"    {region_name:<8s} (n={n_region:,}): {', '.join(top3)}")

    # ===================================================================
    # Step 5: Interaction values
    # ===================================================================
    print("\nStep 5: Computing interaction values (1000-row subsample)...")

    rng = np.random.default_rng(42)
    sub_idx = rng.choice(X_test.shape[0], 1000, replace=False)
    X_sub = X_test[sub_idx]

    interaction_values = explainer.shap_interaction_values(X_sub)
    assert interaction_values.shape == (1000, len(MODEL_FEATURES), len(MODEL_FEATURES)), (
        f"Expected (1000, 25, 25), got {interaction_values.shape}"
    )
    print(f"  Interaction values shape: {interaction_values.shape}")

    # Compute mean absolute interaction strength for all pairs
    n_feat = len(MODEL_FEATURES)
    interaction_pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            strength = float(np.abs(interaction_values[:, i, j]).mean())
            interaction_pairs.append({
                "feature_a": MODEL_FEATURES[i],
                "feature_b": MODEL_FEATURES[j],
                "mean_abs_interaction": round(strength, 6),
            })
    interaction_pairs.sort(key=lambda x: x["mean_abs_interaction"], reverse=True)

    # Key pairs from spec
    poverty_idx = MODEL_FEATURES.index("poverty_index_z")
    lang_indig_idx = MODEL_FEATURES.index("lang_other_indigenous")
    rural_idx = MODEL_FEATURES.index("rural")
    es_mujer_idx = MODEL_FEATURES.index("es_mujer")

    key_poverty_lang = float(
        np.abs(interaction_values[:, poverty_idx, lang_indig_idx]).mean()
    )
    key_rural_gender = float(
        np.abs(interaction_values[:, rural_idx, es_mujer_idx]).mean()
    )

    print(f"\n  KEY INTERACTION PAIRS:")
    print(f"    poverty_index_z x lang_other_indigenous: {key_poverty_lang:.6f}")
    print(f"    rural x es_mujer:                       {key_rural_gender:.6f}")
    print(f"\n  TOP-5 INTERACTION PAIRS:")
    for pair in interaction_pairs[:5]:
        print(
            f"    {pair['feature_a']} x {pair['feature_b']}: "
            f"{pair['mean_abs_interaction']:.6f}"
        )

    # ===================================================================
    # Step 6: ES_PERUANO and ES_MUJER quantification
    # ===================================================================
    print("\nStep 6: Quantifying ES_PERUANO and ES_MUJER...")

    es_peruano_idx = MODEL_FEATURES.index("es_peruano")
    es_mujer_idx_feat = MODEL_FEATURES.index("es_mujer")

    # Mean absolute and signed SHAP
    ep_mean_abs = float(np.abs(sv[:, es_peruano_idx]).mean())
    ep_mean_signed = float(sv[:, es_peruano_idx].mean())
    em_mean_abs = float(np.abs(sv[:, es_mujer_idx_feat]).mean())
    em_mean_signed = float(sv[:, es_mujer_idx_feat].mean())

    # Rank (1-indexed)
    ep_rank = [f for f, _ in sorted_features].index("es_peruano") + 1
    em_rank = [f for f, _ in sorted_features].index("es_mujer") + 1

    # Conditional means
    ep_vals = X_test[:, es_peruano_idx]
    ep_shap_peruano = float(sv[ep_vals == 1, es_peruano_idx].mean())
    ep_shap_foreign = float(sv[ep_vals == 0, es_peruano_idx].mean())
    n_peruano = int((ep_vals == 1).sum())
    n_foreign = int((ep_vals == 0).sum())

    em_vals = X_test[:, es_mujer_idx_feat]
    em_shap_female = float(sv[em_vals == 1, es_mujer_idx_feat].mean())
    em_shap_male = float(sv[em_vals == 0, es_mujer_idx_feat].mean())
    n_female = int((em_vals == 1).sum())
    n_male = int((em_vals == 0).sum())

    print(f"\n  ES_PERUANO:")
    print(f"    Mean |SHAP|:  {ep_mean_abs:.6f}  (rank {ep_rank}/25)")
    print(f"    Mean signed:  {ep_mean_signed:.6f}")
    print(f"    Peruvian (n={n_peruano:,}):     mean SHAP = {ep_shap_peruano:.6f}")
    print(f"    Foreign  (n={n_foreign}):       mean SHAP = {ep_shap_foreign:.6f}")

    print(f"\n  ES_MUJER:")
    print(f"    Mean |SHAP|:  {em_mean_abs:.6f}  (rank {em_rank}/25)")
    print(f"    Mean signed:  {em_mean_signed:.6f}")
    print(f"    Female (n={n_female:,}):   mean SHAP = {em_shap_female:.6f}")
    print(f"    Male   (n={n_male:,}):   mean SHAP = {em_shap_male:.6f}")

    # ===================================================================
    # Step 7: 10 representative student profiles
    # ===================================================================
    print("\nStep 7: Selecting 10 representative student profiles...")

    profiles_list = []
    force_plot_indices: dict[str, int] = {}

    for profile_id, desc_es, filters in PROFILE_DEFINITIONS:
        mask = _build_profile_mask(merged, filters)
        n_in_group = int(mask.sum())

        if n_in_group == 0:
            print(f"  WARNING: No students match '{profile_id}' -- skipping")
            continue

        idx = _select_representative(probas, mask)

        profile = {
            "profile_id": profile_id,
            "description_es": desc_es,
            "feature_values": {
                feat_name: round(float(X_test[idx, fi]), 6)
                for fi, feat_name in enumerate(MODEL_FEATURES)
            },
            "shap_values": {
                feat_name: round(float(sv[idx, fi]), 6)
                for fi, feat_name in enumerate(MODEL_FEATURES)
            },
            "predicted_probability": round(float(probas[idx]), 6),
            "base_value": round(float(explainer.expected_value), 6),
            "raw_prediction": round(
                float(sv[idx].sum() + explainer.expected_value), 6
            ),
            "n_in_group": n_in_group,
            "flagged_small_sample": n_in_group < 30,
        }
        profiles_list.append(profile)
        force_plot_indices[profile_id] = idx

        flag = " [SMALL SAMPLE]" if profile["flagged_small_sample"] else ""
        print(
            f"  {profile_id:<30s}  n={n_in_group:>5d}  "
            f"p={profile['predicted_probability']:.4f}{flag}"
        )

    print(f"  Total profiles: {len(profiles_list)}")

    # ===================================================================
    # Step 8: Force plots for ES_PERUANO and ES_MUJER profiles
    # ===================================================================
    print("\nStep 8: Generating force plots...")

    # ES_PERUANO focus: lima_urban_foreign profile
    if "lima_urban_foreign" in force_plot_indices:
        idx_ep = force_plot_indices["lima_urban_foreign"]
        shap.force_plot(
            explainer.expected_value,
            sv[idx_ep : idx_ep + 1],
            X_test[idx_ep : idx_ep + 1],
            feature_names=list(MODEL_FEATURES),
            matplotlib=True,
            show=False,
        )
        plt.savefig(
            fig_dir / "shap_force_es_peruano.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close("all")
        print("  Saved: shap_force_es_peruano.png")

    # ES_MUJER focus: female_secundaria_rural profile
    if "female_secundaria_rural" in force_plot_indices:
        idx_em = force_plot_indices["female_secundaria_rural"]
        shap.force_plot(
            explainer.expected_value,
            sv[idx_em : idx_em + 1],
            X_test[idx_em : idx_em + 1],
            feature_names=list(MODEL_FEATURES),
            matplotlib=True,
            show=False,
        )
        plt.savefig(
            fig_dir / "shap_force_es_mujer.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close("all")
        print("  Saved: shap_force_es_mujer.png")

    # ===================================================================
    # Step 9: Build and export shap_values.json
    # ===================================================================
    print("\nStep 9: Exporting shap_values.json...")

    shap_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm",
        "computed_on": "test_2023",
        "n_test": int(X_test.shape[0]),
        "shap_space": "log_odds",
        "base_value": round(float(explainer.expected_value), 6),
        "feature_names": list(MODEL_FEATURES),
        "feature_labels_es": FEATURE_LABELS_ES,
        "global_importance": {
            feat_name: round(float(val), 6)
            for feat_name, val in zip(MODEL_FEATURES, mean_abs_shap)
        },
        "top_5_shap": shap_top_5,
        "top_5_lr": lr_top_5,
        "overlap_count": overlap_count,
        "overlap_features": overlap_list,
        "overlap_note": overlap_note,
        "regional": regional_shap,
        "es_peruano": {
            "mean_abs_shap": round(ep_mean_abs, 6),
            "mean_signed_shap": round(ep_mean_signed, 6),
            "rank": ep_rank,
            "conditional": {
                "peruano_mean_shap": round(ep_shap_peruano, 6),
                "foreign_mean_shap": round(ep_shap_foreign, 6),
                "n_peruano": n_peruano,
                "n_foreign": n_foreign,
            },
        },
        "es_mujer": {
            "mean_abs_shap": round(em_mean_abs, 6),
            "mean_signed_shap": round(em_mean_signed, 6),
            "rank": em_rank,
            "conditional": {
                "female_mean_shap": round(em_shap_female, 6),
                "male_mean_shap": round(em_shap_male, 6),
                "n_female": n_female,
                "n_male": n_male,
            },
        },
        "interactions": {
            "subsample_n": 1000,
            "key_pairs": interaction_pairs[:5],
            "focused_pairs": {
                "poverty_index_z_x_lang_other_indigenous": round(key_poverty_lang, 6),
                "rural_x_es_mujer": round(key_rural_gender, 6),
            },
        },
        "profiles": profiles_list,
    }

    output_path = root / "data" / "exports" / "shap_values.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(shap_json, f, indent=2, default=str)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({file_size_kb:.1f} KB)")

    # ===================================================================
    # Step 10: Console summary for human review
    # ===================================================================
    print()
    print("=" * 70)
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nModel: LightGBM ({lgbm.n_estimators_} trees)")
    print(f"Test set: {X_test.shape[0]:,} rows x {X_test.shape[1]} features")
    print(f"SHAP space: log-odds")
    print(f"Base value: {explainer.expected_value:.6f}")

    print("\n--- TOP-10 GLOBAL SHAP FEATURES ---")
    for i, (feat_name, val) in enumerate(sorted_features[:10], 1):
        print(f"  {i:>2d}. {feat_name:<35s} {val:.6f}")

    print(f"\n--- LR vs SHAP TOP-5 OVERLAP: {overlap_count}/5 ---")
    print(f"  SHAP: {shap_top_5}")
    print(f"  LR:   {lr_top_5}")
    print(f"  Overlap: {overlap_list}")
    if overlap_note:
        print(f"  Note: {overlap_note[:100]}...")

    print(f"\n--- ES_PERUANO ---")
    print(f"  Mean |SHAP|: {ep_mean_abs:.6f} (rank {ep_rank}/25)")
    print(f"  Mean signed: {ep_mean_signed:.6f}")
    print(f"  Peruvian: {ep_shap_peruano:.6f}, Foreign: {ep_shap_foreign:.6f}")

    print(f"\n--- ES_MUJER ---")
    print(f"  Mean |SHAP|: {em_mean_abs:.6f} (rank {em_rank}/25)")
    print(f"  Mean signed: {em_mean_signed:.6f}")
    print(f"  Female: {em_shap_female:.6f}, Male: {em_shap_male:.6f}")

    print(f"\n--- REGIONAL TOP-3 ---")
    for region_name in region_names:
        region_sorted = sorted(
            regional_shap[region_name].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top3 = [f"{f}={v:.4f}" for f, v in region_sorted[:3]]
        print(f"  {region_name:<8s}: {', '.join(top3)}")

    print(f"\n--- KEY INTERACTIONS ---")
    print(f"  poverty_index_z x lang_other_indigenous: {key_poverty_lang:.6f}")
    print(f"  rural x es_mujer: {key_rural_gender:.6f}")

    print(f"\n--- PROFILES: {len(profiles_list)} ---")
    for p in profiles_list:
        flag = " [SMALL]" if p["flagged_small_sample"] else ""
        print(
            f"  {p['profile_id']:<30s}  n={p['n_in_group']:>5d}  "
            f"p={p['predicted_probability']:.4f}{flag}"
        )

    print(f"\n--- FIGURES: 5 PNGs ---")
    for fig_name in [
        "shap_beeswarm_global.png",
        "shap_bar_top10.png",
        "shap_regional_comparison.png",
        "shap_force_es_peruano.png",
        "shap_force_es_mujer.png",
    ]:
        fig_path = fig_dir / fig_name
        status = "OK" if fig_path.exists() else "MISSING"
        size_kb = fig_path.stat().st_size / 1024 if fig_path.exists() else 0
        print(f"  [{status}] {fig_name} ({size_kb:.1f} KB)")

    print(f"\n--- JSON EXPORT ---")
    print(f"  {output_path} ({file_size_kb:.1f} KB)")
    print("=" * 70)

    return shap_json


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
    )
    run_shap_pipeline()
