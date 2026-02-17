# Phase 13 Context: LaTeX Template + Table Generation

## Goal
ACM-format LaTeX template with full paper structure and 8 auto-generated tables pulling data directly from v1.0 JSON exports.

## Key Decisions

- **ACM acmart class** (`acmsmall` format) — standard for ACM journals, provides two-column layout with abstract, keywords, CCS concepts
- **Python script generates .tex table files** — `scripts/generate_tables.py` reads JSON exports, outputs standalone `\input`-able `.tex` files to `paper/tables/`
- **booktabs for tables** — `\toprule`, `\midrule`, `\bottomrule` for professional formatting
- **siunitx for number formatting** — consistent decimal alignment and thousand separators
- **8 tables total** covering descriptive stats, model performance, fairness metrics, and SHAP importance
- **All tables bilingual** — Spanish column headers/captions (paper language), English feature names where needed
- **No LaTeX in system PATH** — generate .tex files; user compiles locally after installing texlive
- **Figures already exist** in `paper/figures/` (7 figures × PNG+PDF from Phase 12)

## Data Sources → Tables

| Table | Source File | JSON Path | Content |
|-------|-----------|-----------|---------|
| T1: Sample description | descriptive_tables.json | `_metadata`, `language`, `rural`, `region` | N by year, dimension breakdown |
| T2: Dropout by language | descriptive_tables.json | `language` | 7 groups, weighted rate, 95% CI, n |
| T3: Dropout by region & poverty | descriptive_tables.json | `region`, `poverty` | 3 regions + 5 quintiles, rate, CI |
| T4: Model comparison | model_results.json | `logistic_regression.metrics`, `lightgbm.metrics`, `xgboost.metrics` | PR-AUC, ROC-AUC, F1 for val+test |
| T5: LR coefficients | model_results.json | `logistic_regression.coefficients` | Top features, coeff, OR, p-value |
| T6: Fairness by language | fairness_metrics.json | `dimensions.language.groups` | FNR, FPR, TPR, precision per group |
| T7: Intersection analysis | fairness_metrics.json | `intersections.language_x_rural.groups` | Language × rurality FNR matrix |
| T8: SHAP importance | shap_values.json | `global_importance`, `top_5_shap`, `top_5_lr` | Top 15 features, |SHAP|, LR rank |

## Paper Structure (8 sections + appendix)

1. Introduction
2. Related Work (Alerta Escuela, equity in ML, Peru education)
3. Data (ENAHO, supplementary sources, feature engineering)
4. Methods (LR, LightGBM, XGBoost, calibration, fairness framework)
5. Results (model performance, descriptive statistics)
6. Fairness Analysis (subgroup metrics, intersections, SHAP)
7. Discussion (implications, limitations, recommendations)
8. Conclusion
A. Appendix (full feature list, additional tables)

## Environment Notes

- Tables generated with: `uv run python scripts/generate_tables.py`
- LaTeX compilation: `cd paper && latexmk -pdf main.tex` (requires texlive — not in flake.nix yet)
- All generated .tex table files committed to version control for reproducibility
