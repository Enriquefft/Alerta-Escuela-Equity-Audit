#!/usr/bin/env bash
# Re-run the full pipeline (Phases 3-10) after replacing synthetic data.
#
# Usage:
#   bash scripts/rerun_pipeline.sh
#
# This script runs each phase sequentially. If any step fails, it stops.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  Alerta Escuela Audit -- Full Pipeline Re-run"
echo "  (After replacing synthetic data with real Census + VIIRS)"
echo "============================================================"
echo ""

# Phase 3: Build merged dataset (ENAHO + admin + census + nightlights)
echo "[Phase 3] Building merged dataset..."
uv run python src/data/build_dataset.py
echo ""

# Phase 4: Feature engineering + descriptive statistics
echo "[Phase 4] Feature engineering..."
uv run python src/data/features.py
echo "[Phase 4] Descriptive statistics..."
uv run python src/data/descriptive.py
echo ""

# Phase 5: Baseline model (logistic regression)
echo "[Phase 5] Baseline model..."
uv run python src/models/baseline.py
echo ""

# Phase 6: LightGBM + XGBoost
echo "[Phase 6] LightGBM + XGBoost..."
uv run python src/models/lightgbm_xgboost.py
echo ""

# Phase 7: Calibration + ONNX export
echo "[Phase 7] Calibration + ONNX export..."
uv run python src/models/calibration.py
echo ""

# Phase 8: Subgroup fairness metrics
echo "[Phase 8] Fairness metrics..."
uv run python src/fairness/metrics.py
echo ""

# Phase 9: SHAP interpretability
echo "[Phase 9] SHAP analysis..."
uv run python src/fairness/shap_analysis.py
echo ""

# Phase 10: Cross-validation with admin data
echo "[Phase 10] Cross-validation..."
uv run python src/fairness/cross_validation.py
echo ""

echo "============================================================"
echo "  Pipeline complete! Running all gate tests..."
echo "============================================================"
echo ""

# Run all gate tests
uv run pytest tests/gates/ -v --tb=short 2>&1 || {
    echo ""
    echo "WARNING: Some gate tests failed. Review output above."
    echo "Gate test thresholds may need adjustment for real data."
}

echo ""
echo "Done!"
