---
phase: 22
plan: 1
subsystem: analysis-code
tags: [fairness, ablation, power-analysis, figures, flagging]
dependency-graph:
  requires: [model_results.json, fairness_metrics.json, predictions_lgbm_calibrated.parquet]
  provides: [pooled_ci_urban_indigenous.json, feature_ablation.json, power_analysis.json]
  affects: [fairness_metrics.json, paper/figures/*.png, paper/figures/*.pdf]
tech-stack:
  added: []
  patterns: [per-variant-threshold, ci-width-flagging]
key-files:
  created:
    - src/fairness/pooled_ci.py
    - src/models/feature_ablation.py
    - src/fairness/power_analysis.py
    - data/exports/pooled_ci_urban_indigenous.json
    - data/exports/feature_ablation.json
    - data/exports/power_analysis.json
  modified:
    - scripts/publication_figures.py
    - src/fairness/metrics.py
    - src/fairness/shap_analysis.py
    - src/plotting.py
    - tests/gates/test_gate_3_1.py
    - data/exports/fairness_metrics.json
    - paper/figures/fig01_pr_curves.png
    - paper/figures/fig02_calibration.png
    - paper/figures/fig03_fnr_fpr_language.png
    - paper/figures/fig04_dropout_heatmap.png
    - paper/figures/fig05_fnr_heatmap.png
    - paper/figures/fig06_shap_bar.png
    - paper/figures/fig07_shap_beeswarm.png
decisions:
  - Pooled CI credible=false (CI lower 0.392 < 0.50) -- castellano invisibility becomes headline
  - Per-variant thresholds for ablation (max weighted F1 on val set) instead of shared calibrated threshold
  - Ablation interpretation is "mixed" -- castellano consistently highest FNR but second-highest varies
  - CI-width > 0.5 flagging for intersections (Option B from plan)
metrics:
  duration: ~36 min
  completed: 2026-02-18
---

# Phase 22 Plan 1: Analysis Code Summary

Pooled CI for urban other-indigenous shows credible=false; feature ablation confirms castellano invisibility persists across model variants; power analysis quantifies ENAHO data ceiling; all figures translated to English; CI-width flagging added.

## Key Results

### Task 1: Pooled CI (pooled_ci_urban_indigenous.json)
- Pooled val+test: n=167, dropouts=13
- FNR point estimate: 0.688
- 95% CI: [0.392, 0.979], width=0.587
- **credible=false** (CI lower bound 0.392 does not exceed 0.50)
- Implication: urban other-indigenous FNR=0.753 cannot be confirmed as "model misses majority"
- Castellano invisibility (FNR=0.633, n=22K+) becomes the headline finding

### Task 2: Feature Ablation (feature_ablation.json)
- Individual-only (18 features): val PR-AUC=0.2502, best_iter=24
- Spatial-only (7 features): val PR-AUC=0.1760, best_iter=24
- **Castellano has highest FNR in ALL three variants** (full: 0.633, individual: 0.649, spatial: 0.317)
- Interpretation: "mixed" -- highest-FNR group consistent but second-highest varies
- Spatial model alone produces lowest FNRs overall (more aggressive positive predictions)

### Task 3: Power Analysis (power_analysis.json)
- Detecting FNR > 0.50: need 46 dropout observations (~8 ENAHO years)
- Detecting FNR gap vs castellano: need 192 dropout observations (~32 ENAHO years)
- Conclusion: survey-based intersectional auditing cannot produce significant results for groups with ~6 dropouts/year

### Task 4: English Figure Labels
- All 7 figures (14 files: PNG + PDF) regenerated with English labels
- Translated: titles, axis labels, legends, text boxes, heatmap labels
- SHAP feature labels translated in shap_analysis.py (source of truth)

### Task 5: CI-Width Flagging
- Added secondary flag: FNR CI width > 0.5 triggers `flagged_small_sample=true`
- other_indigenous_urban (n=89, CI width=0.789) now correctly flagged
- All 117 gate tests pass

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Per-variant thresholds for feature ablation**
- **Found during:** Task 2
- **Issue:** Plan used the full model's calibrated threshold (0.167) for ablation variants. With `scale_pos_weight=4.8`, ablation model probabilities are shifted upward (min prob=0.206), making threshold 0.167 classify ALL observations as positive (FNR=0.000 for all groups).
- **Fix:** Compute optimal threshold per variant using max weighted F1 on validation set (individual: 0.400, spatial: 0.380).
- **Files modified:** src/models/feature_ablation.py
- **Commit:** 96006fe

**2. [Rule 1 - Bug] Gate test accepted only 'lightgbm' model name**
- **Found during:** Task 5
- **Issue:** test_gate_3_1.py::test_model_and_threshold asserted `model == 'lightgbm'` but pipeline writes `model == 'lgbm_calibrated'` (correct since Phase 17 multi-model refactor).
- **Fix:** Updated test to accept both 'lightgbm' and 'lgbm_calibrated'.
- **Files modified:** tests/gates/test_gate_3_1.py
- **Commit:** 96006fe

## Verification

- All 117 gate tests pass (uv run pytest tests/gates/ -v)
- All 3 new export JSONs exist and contain expected fields
- All 7 figures regenerated with English labels (confirmed visually on fig03)
- other_indigenous_urban has flagged_small_sample=true in fairness_metrics.json

## Self-Check: PASSED
