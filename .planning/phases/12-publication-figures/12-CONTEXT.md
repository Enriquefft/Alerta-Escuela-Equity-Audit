# Phase 12 Context: Publication Figures

## Goal
Create 2 new headline figures and polish all existing figures to publication quality with consistent styling, proper feature names, and dual-format export (PNG 300dpi + PDF vector).

## Key Decisions

- **Single script** `scripts/publication_figures.py` generates all publication figures from existing exports — no model re-training needed
- **PR curves from predictions parquets** — all 3 models have prob_dropout + dropout in `data/processed/predictions_*.parquet` with test_2023 split
- **FNR/FPR data from fairness_metrics.json** — already computed, validated in Phase 8
- **FNR heatmap data from fairness_metrics.json** — intersection `language_x_rural` already computed
- **Existing figures regenerated** at 300 DPI + PDF — not manually polished (reproducible)
- **Output directory:** `paper/figures/` (new)
- **seaborn available** but stick with matplotlib for consistency with existing codebase
- **FEATURE_LABELS_ES** reused from shap_analysis.py for Spanish feature names
- **PALETTE** reused from descriptive.py for language group colors

## Data Sources (all existing)

| Figure | Source File | Key Fields |
|--------|-----------|-----------|
| FIG-01 PR curves | predictions_{lr,lgbm,xgb}.parquet (test_2023) | prob_dropout, dropout, FACTOR07 |
| FIG-02 Calibration | predictions_lgbm_calibrated.parquet | prob_dropout, prob_dropout_uncalibrated, dropout |
| FIG-03 FNR/FPR bar | fairness_metrics.json → language.groups | fnr, fpr, n_unweighted |
| FIG-04 Dropout heatmap | descriptive_tables.json → heatmap_language_x_rural | rates, n_unweighted |
| FIG-05 FNR heatmap | fairness_metrics.json → intersections.language_x_rural | fnr, n_unweighted |
