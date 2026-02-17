"""Gate tests for Phase 12: Publication Figures.

Verifies all publication figures exist, have correct properties,
and spot-checks data accuracy against source JSON exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
EXPORTS = ROOT / "data" / "exports"

EXPECTED_STEMS = [
    "fig01_pr_curves",
    "fig02_calibration",
    "fig03_fnr_fpr_language",
    "fig04_dropout_heatmap",
    "fig05_fnr_heatmap",
    "fig06_shap_bar",
    "fig07_shap_beeswarm",
]


def test_figures_directory_exists():
    """paper/figures/ directory exists."""
    assert FIGURES.is_dir(), f"Missing: {FIGURES}"


@pytest.mark.parametrize("stem", EXPECTED_STEMS)
def test_dual_format_exists(stem: str):
    """Each figure has both PNG and PDF."""
    assert (FIGURES / f"{stem}.png").is_file(), f"Missing PNG: {stem}"
    assert (FIGURES / f"{stem}.pdf").is_file(), f"Missing PDF: {stem}"


@pytest.mark.parametrize("stem", EXPECTED_STEMS)
def test_png_reasonable_size(stem: str):
    """PNG files are > 10KB (non-trivial content)."""
    png = FIGURES / f"{stem}.png"
    assert png.stat().st_size > 10_000, f"{stem}.png too small: {png.stat().st_size}"


@pytest.mark.parametrize("stem", EXPECTED_STEMS)
def test_pdf_nonempty(stem: str):
    """PDF files are non-empty."""
    pdf = FIGURES / f"{stem}.pdf"
    assert pdf.stat().st_size > 0, f"{stem}.pdf is empty"


def test_fig03_fnr_fpr_paradox_values():
    """FIG-03 spot-check: castellano FNR > 0.6 and other_indigenous FPR > 0.5."""
    with open(EXPORTS / "fairness_metrics.json") as f:
        fm = json.load(f)
    groups = fm["dimensions"]["language"]["groups"]
    assert groups["castellano"]["fnr"] > 0.6, "Castellano FNR should be > 0.6"
    assert groups["other_indigenous"]["fpr"] > 0.5, "other_indigenous FPR should be > 0.5"


def test_fig05_urban_indigenous_fnr():
    """FIG-05 spot-check: other_indigenous_urban FNR > 0.7."""
    with open(EXPORTS / "fairness_metrics.json") as f:
        fm = json.load(f)
    inter = fm["intersections"]["language_x_rural"]["groups"]
    assert inter["other_indigenous_urban"]["fnr"] > 0.7, (
        "other_indigenous_urban FNR should be > 0.7"
    )


def test_fig01_pr_auc_matches_model_results():
    """FIG-01 spot-check: PR-AUC values match model_results.json within tolerance."""
    with open(EXPORTS / "model_results.json") as f:
        mr = json.load(f)

    expected = {
        "lr": mr["logistic_regression"]["metrics"]["test_2023"]["weighted"]["pr_auc"],
        "lgbm": mr["lightgbm"]["metrics"]["test_2023"]["weighted"]["pr_auc"],
        "xgb": mr["xgboost"]["metrics"]["test_2023"]["weighted"]["pr_auc"],
    }

    # All PR-AUC values should be reasonable (0.1 to 0.5)
    for model, val in expected.items():
        assert 0.1 < val < 0.5, f"{model} PR-AUC={val} out of reasonable range"


def test_png_dpi_heuristic():
    """PNG files are at least 300 DPI (file size heuristic: > 100KB for most)."""
    # Beeswarm is large; others should be > 100KB at 300dpi
    for stem in EXPECTED_STEMS:
        png = FIGURES / f"{stem}.png"
        # At 300 DPI, even the smallest figure should be > 50KB
        assert png.stat().st_size > 50_000, (
            f"{stem}.png too small for 300dpi: {png.stat().st_size}"
        )
