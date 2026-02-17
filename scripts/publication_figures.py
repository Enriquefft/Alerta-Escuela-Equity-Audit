"""Generate all publication-quality figures for the Alerta Escuela paper.

Reads from existing v1.0 exports (JSON + parquet) and produces 7 figures
in dual format (PNG 300dpi + PDF vector) in paper/figures/.

Usage::

    uv run python scripts/publication_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.calibration import CalibrationDisplay  # noqa: E402

from plotting import (  # noqa: E402
    FEATURE_TO_LABEL_ES,
    LANGUAGE_LABELS_ES,
    MODEL_COLORS,
    MODEL_FEATURES,
    MODEL_LABELS,
    PALETTE,
    PUB_STYLE,
    save_dual_format,
    setup_pub_style,
    small_sample_annotation,
)
from utils import find_project_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = find_project_root()
EXPORTS = ROOT / "data" / "exports"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "paper" / "figures"


# ===================================================================
# FIG-01: Combined PR Curves
# ===================================================================
def fig01_pr_curves() -> None:
    """Combined precision-recall curves for all 3 models on test_2023."""
    logger.info("FIG-01: PR curves")

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_key, parquet_name in [
        ("lr", "predictions_lr.parquet"),
        ("lgbm", "predictions_lgbm.parquet"),
        ("xgb", "predictions_xgb.parquet"),
    ]:
        df = pl.read_parquet(PROCESSED / parquet_name)
        df = df.filter(pl.col("split") == "test_2023")

        y_true = df["dropout"].cast(pl.Int8).to_numpy()
        y_prob = df["prob_dropout"].to_numpy()
        weights = df["FACTOR07"].to_numpy()

        precision, recall, _ = precision_recall_curve(
            y_true, y_prob, sample_weight=weights
        )
        pr_auc = average_precision_score(y_true, y_prob, sample_weight=weights)

        label = f"{MODEL_LABELS[model_key]} (PR-AUC={pr_auc:.4f})"
        ax.plot(recall, precision, color=MODEL_COLORS[model_key], label=label, lw=1.5)

    # Prevalence baseline
    df_test = pl.read_parquet(PROCESSED / "predictions_lr.parquet").filter(
        pl.col("split") == "test_2023"
    )
    prevalence = (
        (df_test["dropout"].cast(pl.Float64) * df_test["FACTOR07"]).sum()
        / df_test["FACTOR07"].sum()
    )
    ax.axhline(prevalence, color="gray", ls="--", lw=0.8, label=f"Prevalencia ({prevalence:.3f})")

    ax.set_xlabel("Recall (Sensitividad)")
    ax.set_ylabel("Precision (Precision)")
    ax.set_title("Curvas Precision-Recall — Test 2023")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_dual_format(fig, FIGURES / "fig01_pr_curves")


# ===================================================================
# FIG-02: Calibration Curve
# ===================================================================
def fig02_calibration() -> None:
    """Calibration curves for LightGBM (before and after Platt scaling)."""
    logger.info("FIG-02: Calibration curve")

    df = pl.read_parquet(PROCESSED / "predictions_lgbm_calibrated.parquet")
    df = df.filter(pl.col("split") == "test_2023")

    y_true = df["dropout"].cast(pl.Int8).to_numpy()
    prob_uncal = df["prob_dropout_uncalibrated"].to_numpy()
    prob_cal = df["prob_dropout"].to_numpy()

    # Brier scores
    brier_uncal = np.mean((prob_uncal - y_true) ** 2)
    brier_cal = np.mean((prob_cal - y_true) ** 2)

    fig, ax = plt.subplots(figsize=(6, 5))

    CalibrationDisplay.from_predictions(
        y_true,
        prob_uncal,
        n_bins=10,
        strategy="uniform",
        name=f"Sin calibrar (Brier={brier_uncal:.4f})",
        ax=ax,
        color="#d62728",
    )
    CalibrationDisplay.from_predictions(
        y_true,
        prob_cal,
        n_bins=10,
        strategy="uniform",
        name=f"Calibrado (Brier={brier_cal:.4f})",
        ax=ax,
        color="#2ca02c",
    )

    ax.set_title("Curva de Calibracion — LightGBM")
    ax.set_xlabel("Probabilidad predicha (promedio por bin)")
    ax.set_ylabel("Fraccion de positivos observados")
    ax.legend(loc="lower right")

    save_dual_format(fig, FIGURES / "fig02_calibration")


# ===================================================================
# FIG-03: FNR/FPR Grouped Bar by Language (headline figure)
# ===================================================================
def fig03_fnr_fpr_language() -> None:
    """FNR vs FPR grouped bar chart by language group — the 'money figure'."""
    logger.info("FIG-03: FNR/FPR by language")

    with open(EXPORTS / "fairness_metrics.json") as f:
        fm = json.load(f)

    groups_data = fm["dimensions"]["language"]["groups"]

    # Order: castellano first (reference), then indigenous, then foreign
    # Drop "unknown" (flagged as unreliable)
    order = ["castellano", "quechua", "aimara", "other_indigenous", "foreign"]

    fnrs = [groups_data[g]["fnr"] for g in order]
    fprs = [groups_data[g]["fpr"] for g in order]
    ns = [groups_data[g]["n_unweighted"] for g in order]

    labels = []
    for g, n in zip(order, ns):
        label = LANGUAGE_LABELS_ES.get(g, g)
        ann = small_sample_annotation(n, threshold=100)
        labels.append(f"{label}{ann}")

    x = np.arange(len(order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_fnr = ax.bar(x - width / 2, fnrs, width, label="FNR (falsos negativos)", color="#d62728", alpha=0.85)
    bars_fpr = ax.bar(x + width / 2, fprs, width, label="FPR (falsos positivos)", color="#1f77b4", alpha=0.85)

    # Value labels on bars
    for bar in bars_fnr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars_fpr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)

    # Overall reference lines (weighted average across all groups)
    total_w = sum(groups_data[g]["n_unweighted"] for g in order)
    overall_fnr = sum(groups_data[g]["fnr"] * groups_data[g]["n_unweighted"] for g in order) / total_w
    overall_fpr = sum(groups_data[g]["fpr"] * groups_data[g]["n_unweighted"] for g in order) / total_w
    ax.axhline(overall_fnr, color="#d62728", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(overall_fpr, color="#1f77b4", ls="--", lw=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tasa (0–1)")
    ax.set_title("Tasa de Falsos Negativos y Falsos Positivos por Grupo Linguistico")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(max(fnrs), max(fprs)) + 0.1)

    # Explanatory text box
    textstr = ("FNR alto = el modelo no detecta desercion\n"
               "FPR alto = falsas alarmas excesivas")
    props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8, edgecolor="gray")
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    fig.tight_layout()
    save_dual_format(fig, FIGURES / "fig03_fnr_fpr_language")


# ===================================================================
# FIG-04: Dropout Rate Heatmap (language × rurality)
# ===================================================================
def fig04_dropout_heatmap() -> None:
    """Language × rurality dropout rate heatmap."""
    logger.info("FIG-04: Dropout rate heatmap")

    with open(EXPORTS / "descriptive_tables.json") as f:
        dt = json.load(f)

    hm = dt["heatmap_language_x_rural"]
    rows = hm["rows"]
    cols = hm["columns"]
    values = np.array(hm["values"])
    n_unweighted = hm["n_unweighted"]

    # Spanish labels
    row_labels_map = {
        "awajun": "Awajun",
        "ashaninka": "Ashaninka",
        "other_indigenous": "Otros indigenas",
        "quechua": "Quechua",
        "aimara": "Aimara",
        "castellano": "Castellano",
        "foreign": "Extranjero",
    }
    col_labels_map = {"urban": "Urbano", "rural": "Rural"}

    row_labels = [row_labels_map.get(r, r) for r in rows]
    col_labels = [col_labels_map.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Cell annotations: rate + asterisk for small n
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = values[i, j]
            n = n_unweighted[i][j]
            ann = small_sample_annotation(n)
            text = f"{val:.1%}{ann}"
            color = "white" if val > 0.18 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    ax.set_title("Tasa de Desercion: Grupo Linguistico x Ruralidad")
    fig.colorbar(im, ax=ax, label="Tasa de desercion", shrink=0.8)
    fig.tight_layout()

    save_dual_format(fig, FIGURES / "fig04_dropout_heatmap")


# ===================================================================
# FIG-05: FNR Heatmap Language × Rurality (NEW)
# ===================================================================
def fig05_fnr_heatmap() -> None:
    """FNR heatmap by language × rurality — shows urban indigenous blind spot."""
    logger.info("FIG-05: FNR heatmap")

    with open(EXPORTS / "fairness_metrics.json") as f:
        fm = json.load(f)

    inter = fm["intersections"]["language_x_rural"]["groups"]

    # Build matrix: rows=language groups, cols=[urban, rural]
    lang_order = ["castellano", "quechua", "aimara", "other_indigenous"]
    col_order = ["urban", "rural"]
    col_labels = ["Urbano", "Rural"]
    row_labels_map = {
        "castellano": "Castellano",
        "quechua": "Quechua",
        "aimara": "Aimara",
        "other_indigenous": "Otros indigenas",
    }

    matrix = np.full((len(lang_order), len(col_order)), np.nan)
    n_matrix = np.full((len(lang_order), len(col_order)), 0)

    for i, lang in enumerate(lang_order):
        for j, loc in enumerate(col_order):
            key = f"{lang}_{loc}"
            if key in inter:
                matrix[i, j] = inter[key]["fnr"]
                n_matrix[i, j] = inter[key]["n_unweighted"]

    row_labels = [row_labels_map.get(r, r) for r in lang_order]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Cell annotations
    for i in range(len(lang_order)):
        for j in range(len(col_order)):
            val = matrix[i, j]
            n = n_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
                continue
            ann = small_sample_annotation(n, threshold=100)
            text = f"{val:.1%}{ann}"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, fontweight="bold" if val > 0.7 else "normal",
                    color=color)

    # Highlight other_indigenous_urban cell (FNR=75.3%)
    oi_idx = lang_order.index("other_indigenous")
    urban_idx = col_order.index("urban")
    rect = plt.Rectangle(
        (urban_idx - 0.5, oi_idx - 0.5), 1, 1,
        linewidth=3, edgecolor="black", facecolor="none"
    )
    ax.add_patch(rect)

    ax.set_title("Tasa de Falsos Negativos: Grupo Linguistico x Ruralidad")
    ax.text(0.5, -0.12, "FNR alto = el modelo falla en detectar desercion",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    fig.colorbar(im, ax=ax, label="FNR", shrink=0.8)
    fig.tight_layout()

    save_dual_format(fig, FIGURES / "fig05_fnr_heatmap")


# ===================================================================
# FIG-06: SHAP Bar Top 10
# ===================================================================
def fig06_shap_bar() -> None:
    """SHAP global importance bar chart (top 10 features)."""
    logger.info("FIG-06: SHAP bar")

    with open(EXPORTS / "shap_values.json") as f:
        sv = json.load(f)

    gi = sv["global_importance"]
    sorted_feats = sorted(gi.items(), key=lambda x: x[1], reverse=True)[:10]

    features = [f for f, _ in sorted_feats]
    importances = [v for _, v in sorted_feats]

    # Spanish labels
    labels = [FEATURE_TO_LABEL_ES.get(f, f) for f in features]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, importances, color="#2ca02c", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Media |SHAP| (escala log-odds)")
    ax.set_title("Importancia Global de Variables — Top 10 (SHAP)")

    fig.tight_layout()
    save_dual_format(fig, FIGURES / "fig06_shap_bar")


# ===================================================================
# FIG-07: SHAP Beeswarm
# ===================================================================
def fig07_shap_beeswarm() -> None:
    """SHAP beeswarm plot with Spanish labels using TreeExplainer."""
    logger.info("FIG-07: SHAP beeswarm (computing SHAP values...)")

    # Load model and feature-engineered data
    model = joblib.load(PROCESSED / "model_lgbm.joblib")
    dataset = pl.read_parquet(PROCESSED / "enaho_with_features.parquet")

    # Get test_2023 rows via predictions parquet (has split column)
    preds = pl.read_parquet(PROCESSED / "predictions_lgbm.parquet")
    test_ids = preds.filter(pl.col("split") == "test_2023").select(
        "CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"
    )
    test_data = dataset.join(test_ids, on=["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"])

    X_test = test_data.select(MODEL_FEATURES).to_pandas()

    # Use 1000-row subsample for beeswarm (fast enough)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    X_sub = X_test.iloc[idx]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sub)

    # Apply Spanish feature labels
    explanation.feature_names = [FEATURE_TO_LABEL_ES.get(f, f) for f in MODEL_FEATURES]

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    ax.set_title("SHAP Beeswarm — LightGBM (Test 2023)")
    ax.set_xlabel("Valor SHAP (impacto en log-odds)")

    fig = plt.gcf()
    fig.tight_layout()
    save_dual_format(fig, FIGURES / "fig07_shap_beeswarm")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    setup_pub_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig01_pr_curves()
    fig02_calibration()
    fig03_fnr_fpr_language()
    fig04_dropout_heatmap()
    fig05_fnr_heatmap()
    fig06_shap_bar()
    fig07_shap_beeswarm()

    # Count outputs
    pngs = list(FIGURES.glob("*.png"))
    pdfs = list(FIGURES.glob("*.pdf"))
    logger.info("Done: %d PNGs + %d PDFs in %s", len(pngs), len(pdfs), FIGURES)


if __name__ == "__main__":
    main()
