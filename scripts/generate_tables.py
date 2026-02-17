#!/usr/bin/env python3
"""Generate LaTeX table fragments from JSON exports.

Reads data/exports/*.json and writes 8 tabular-only .tex files
to paper/tables/ for \\input{} inclusion in the main paper.

All numeric values are read from JSON — no hardcoded metric values.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "data" / "exports"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load JSON data
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    with open(EXPORTS / name) as f:
        return json.load(f)

desc = _load("descriptive_tables.json")
model = _load("model_results.json")
fair = _load("fairness_metrics.json")
shap = _load("shap_values.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPANISH_NAMES: dict[str, str] = dict(
    zip(shap["feature_names"], shap["feature_labels_es"])
)

LANGUAGE_DISPLAY = {
    "awajun": "Awaj\\'{u}n",
    "ashaninka": "Ash\\'{a}ninka",
    "other_indigenous": "Otros ind\\'{i}genas",
    "quechua": "Quechua",
    "aimara": "Aimara",
    "castellano": "Castellano",
    "foreign": "Extranjero",
}

REGION_DISPLAY = {"costa": "Costa", "sierra": "Sierra", "selva": "Selva"}
SEX_DISPLAY = {"male": "Masculino", "female": "Femenino"}
RURAL_DISPLAY = {"urban": "Urbano", "rural": "Rural"}

def fmt_rate(v: float | None, places: int = 3) -> str:
    """Format a rate/metric value. Handle None/NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{v:.{places}f}"

def fmt_int(v: int | float) -> str:
    return f"{int(v):,}"

def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"

def bold(s: str) -> str:
    return f"\\textbf{{{s}}}"

def small_sample_mark(n: int, threshold: int = 100) -> str:
    return "*" if n < threshold else ""

def _write(name: str, content: str) -> None:
    (OUT / name).write_text(content)
    print(f"  wrote {name}")


# ===========================================================================
# T1 — Sample Description
# ===========================================================================
def table_01():
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dimension & Category & $n$ (unwtd) & $n$ (wtd) & Dropout Rate \\\\",
        "\\midrule",
    ]

    # Overall
    meta = desc["_metadata"]
    # Compute overall from language (sum all groups)
    total_n = sum(g["n_unweighted"] for g in desc["language"])
    total_nw = sum(g["n_weighted"] for g in desc["language"])
    # Weighted overall dropout rate: sum(rate*nw)/sum(nw)
    total_rate = sum(g["weighted_rate"] * g["n_weighted"] for g in desc["language"]) / total_nw
    lines.append(f"Overall & --- & {fmt_int(total_n)} & {fmt_int(total_nw)} & {fmt_rate(total_rate)} \\\\")
    lines.append("\\midrule")

    # Language
    for g in desc["language"]:
        name = LANGUAGE_DISPLAY.get(g["group"], g["group"])
        lines.append(
            f"Language & {name} & {fmt_int(g['n_unweighted'])} & {fmt_int(g['n_weighted'])} & {fmt_rate(g['weighted_rate'])} \\\\"
        )
    lines.append("\\midrule")

    # Sex
    for g in desc["sex"]:
        name = SEX_DISPLAY.get(g["group"], g["group"])
        lines.append(
            f"Sex & {name} & {fmt_int(g['n_unweighted'])} & {fmt_int(g['n_weighted'])} & {fmt_rate(g['weighted_rate'])} \\\\"
        )
    lines.append("\\midrule")

    # Geography (rural/urban)
    for g in desc["rural"]:
        name = RURAL_DISPLAY.get(g["group"], g["group"])
        lines.append(
            f"Geography & {name} & {fmt_int(g['n_unweighted'])} & {fmt_int(g['n_weighted'])} & {fmt_rate(g['weighted_rate'])} \\\\"
        )
    lines.append("\\midrule")

    # Region
    for g in desc["region"]:
        name = REGION_DISPLAY.get(g["group"], g["group"])
        lines.append(
            f"Region & {name} & {fmt_int(g['n_unweighted'])} & {fmt_int(g['n_weighted'])} & {fmt_rate(g['weighted_rate'])} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write("table_01_sample.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T2 — Dropout by Language Group
# ===========================================================================
def table_02():
    groups = sorted(desc["language"], key=lambda g: g["weighted_rate"], reverse=True)
    max_rate = groups[0]["weighted_rate"]

    lines = [
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Language Group & Weighted Rate & 95\\% CI & $n$ (unwtd) \\\\",
        "\\midrule",
    ]
    for g in groups:
        name = LANGUAGE_DISPLAY.get(g["group"], g["group"])
        rate_str = fmt_rate(g["weighted_rate"])
        if g["weighted_rate"] == max_rate:
            rate_str = bold(rate_str)
        ci = fmt_ci(g["lower_ci"], g["upper_ci"])
        lines.append(f"{name} & {rate_str} & {ci} & {fmt_int(g['n_unweighted'])} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write("table_02_language.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T3 — Dropout by Region and Poverty
# ===========================================================================
def table_03():
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Category & Weighted Rate & 95\\% CI \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Panel A: Region}} \\\\",
    ]
    for g in desc["region"]:
        name = REGION_DISPLAY.get(g["group"], g["group"])
        ci = fmt_ci(g["lower_ci"], g["upper_ci"])
        lines.append(f"{name} & {fmt_rate(g['weighted_rate'])} & {ci} \\\\")

    lines.append("\\midrule")
    lines.append("\\multicolumn{3}{l}{\\textit{Panel B: Poverty Quintile}} \\\\")

    poverty_display = {
        "Q1_least_poor": "Q1 (least poor)",
        "Q2": "Q2",
        "Q3": "Q3",
        "Q4": "Q4",
        "Q5_most_poor": "Q5 (most poor)",
    }
    for g in desc["poverty"]:
        name = poverty_display.get(g["group"], g["group"])
        ci = fmt_ci(g["lower_ci"], g["upper_ci"])
        lines.append(f"{name} & {fmt_rate(g['weighted_rate'])} & {ci} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write("table_03_region_poverty.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T4 — Model Performance Comparison
# ===========================================================================
def table_04():
    # For LR and XGBoost: val = validate_2022.weighted, test = test_2023.weighted
    # For LightGBM: val = lightgbm.metrics.validate_2022.weighted,
    #               test = test_2023_calibrated.metrics.test_2023.weighted
    lr_val = model["logistic_regression"]["metrics"]["validate_2022"]["weighted"]
    lr_test = model["logistic_regression"]["metrics"]["test_2023"]["weighted"]

    lgb_val = model["lightgbm"]["metrics"]["validate_2022"]["weighted"]
    lgb_test = model["test_2023_calibrated"]["metrics"]["test_2023"]["weighted"]

    xgb_val = model["xgboost"]["metrics"]["validate_2022"]["weighted"]
    xgb_test = model["xgboost"]["metrics"]["test_2023"]["weighted"]

    rows = [
        ("PR-AUC (val)", lr_val["pr_auc"], lgb_val["pr_auc"], xgb_val["pr_auc"]),
        ("PR-AUC (test)", lr_test["pr_auc"], lgb_test["pr_auc"], xgb_test["pr_auc"]),
        ("ROC-AUC (val)", lr_val["roc_auc"], lgb_val["roc_auc"], xgb_val["roc_auc"]),
        ("ROC-AUC (test)", lr_test["roc_auc"], lgb_test["roc_auc"], xgb_test["roc_auc"]),
        ("F1 (test)", lr_test["f1"], lgb_test["f1"], xgb_test["f1"]),
        ("Precision (test)", lr_test["precision"], lgb_test["precision"], xgb_test["precision"]),
        ("Recall (test)", lr_test["recall"], lgb_test["recall"], xgb_test["recall"]),
    ]

    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Metric & Logistic Regression & LightGBM & XGBoost \\\\",
        "\\midrule",
    ]
    for label, lr, lgb, xgb in rows:
        vals = [lr, lgb, xgb]
        best = max(vals)
        strs = []
        for v in vals:
            s = fmt_rate(v)
            if v == best:
                s = bold(s)
            strs.append(s)
        lines.append(f"{label} & {strs[0]} & {strs[1]} & {strs[2]} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write("table_04_models.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T5 — Logistic Regression Coefficients
# ===========================================================================
def table_05():
    coeffs = [
        c for c in model["logistic_regression"]["coefficients"]
        if c["feature"] != "intercept"
    ]
    coeffs.sort(key=lambda c: abs(c["coefficient"]), reverse=True)

    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Feature & Coefficient & Odds Ratio & Dir. \\\\",
        "\\midrule",
    ]
    for c in coeffs:
        feat = c["feature"]
        spanish = SPANISH_NAMES.get(feat, feat)
        # Escape underscores and special chars for LaTeX
        spanish_safe = spanish  # Already clean from shap_values.json
        direction = "\\up" if c["coefficient"] > 0 else "\\down"
        lines.append(
            f"{spanish_safe} & {c['coefficient']:.4f} & {c['odds_ratio']:.3f} & {direction} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write("table_05_lr_coefficients.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T6 — Fairness Metrics by Language
# ===========================================================================
def table_06():
    groups = fair["dimensions"]["language"]["groups"]
    gaps = fair["dimensions"]["language"]["gaps"]

    # Drop "unknown"
    filtered = {k: v for k, v in groups.items() if k != "unknown"}

    # Sort by FNR ascending
    sorted_groups = sorted(filtered.items(), key=lambda kv: kv[1].get("fnr", 999))

    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Language Group & $n$ & FNR & FPR & Precision & PR-AUC \\\\",
        "\\midrule",
    ]
    for name, g in sorted_groups:
        display = LANGUAGE_DISPLAY.get(name, name)
        n = g["n_unweighted"]
        mark = small_sample_mark(n)
        fnr_val = g.get("fnr")
        fnr_str = fmt_rate(fnr_val)
        if fnr_val is not None and not math.isnan(fnr_val) and fnr_val > 0.5:
            fnr_str = bold(fnr_str)
        lines.append(
            f"{display}{mark} & {fmt_int(n)} & {fnr_str} & {fmt_rate(g.get('fpr'))} & "
            f"{fmt_rate(g.get('precision'))} & {fmt_rate(g.get('pr_auc'))} \\\\"
        )

    # Max gap row
    lines.append("\\midrule")
    lines.append(
        f"Max FNR Gap & --- & {fmt_rate(gaps['max_fnr_gap'])} & --- & --- & --- \\\\"
    )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write("table_06_fairness_language.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T7 — Intersection: Language x Rurality
# ===========================================================================
def table_07():
    groups = fair["intersections"]["language_x_rural"]["groups"]

    # Build language list (unique prefixes)
    lang_order = ["other_indigenous", "aimara", "quechua", "castellano"]

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Language Group & Urban FNR & Rural FNR & Urban $n$ & Rural $n$ \\\\",
        "\\midrule",
    ]
    for lang in lang_order:
        urban_key = f"{lang}_urban"
        rural_key = f"{lang}_rural"
        urban = groups.get(urban_key, {})
        rural = groups.get(rural_key, {})

        u_fnr = urban.get("fnr")
        r_fnr = rural.get("fnr")
        u_n = urban.get("n_unweighted", 0)
        r_n = rural.get("n_unweighted", 0)

        u_mark = small_sample_mark(u_n)
        r_mark = small_sample_mark(r_n)

        u_fnr_str = fmt_rate(u_fnr)
        r_fnr_str = fmt_rate(r_fnr)

        # Highlight other_indigenous_urban (the key finding)
        if lang == "other_indigenous" and u_fnr is not None and not (isinstance(u_fnr, float) and math.isnan(u_fnr)):
            u_fnr_str = bold(u_fnr_str)

        display = LANGUAGE_DISPLAY.get(lang, lang)
        lines.append(
            f"{display} & {u_fnr_str}{u_mark} & {r_fnr_str}{r_mark} & "
            f"{fmt_int(u_n)} & {fmt_int(r_n)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\multicolumn{5}{l}{\\footnotesize * $n < 100$; interpret with caution} \\\\")
    lines.append("\\end{tabular}")
    _write("table_07_intersection.tex", "\n".join(lines) + "\n")


# ===========================================================================
# T8 — SHAP Feature Importance (Top 15)
# ===========================================================================
def table_08():
    importance = shap["global_importance"]

    # Sort by importance descending
    sorted_feats = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:15]

    # Build LR rank lookup (by |coefficient|, excluding intercept)
    lr_coeffs = [
        c for c in model["logistic_regression"]["coefficients"]
        if c["feature"] != "intercept"
    ]
    lr_coeffs.sort(key=lambda c: abs(c["coefficient"]), reverse=True)
    lr_rank = {c["feature"]: i + 1 for i, c in enumerate(lr_coeffs)}

    lines = [
        "\\begin{tabular}{rlrr}",
        "\\toprule",
        "Rank & Feature & Mean $|$SHAP$|$ & LR Rank \\\\",
        "\\midrule",
    ]
    for rank, (feat, imp) in enumerate(sorted_feats, 1):
        spanish = SPANISH_NAMES.get(feat, feat)
        lr_r = lr_rank.get(feat, "---")
        lines.append(f"{rank} & {spanish} & {fmt_rate(imp, 4)} & {lr_r} \\\\")

    lines.append("\\bottomrule")
    lines.append(
        "\\multicolumn{4}{l}{\\footnotesize SHAP computed on uncalibrated LightGBM; values in log-odds space} \\\\"
    )
    lines.append("\\end{tabular}")
    _write("table_08_shap.tex", "\n".join(lines) + "\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating LaTeX tables from JSON exports...")
    table_01()
    table_02()
    table_03()
    table_04()
    table_05()
    table_06()
    table_07()
    table_08()
    print(f"Done. {len(list(OUT.glob('*.tex')))} table files in {OUT}")

if __name__ == "__main__":
    main()
