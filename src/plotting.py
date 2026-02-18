"""Shared publication styling constants and helpers.

Provides consistent figure styling across all publication figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Re-export from existing modules for convenience
from data.descriptive import PALETTE  # noqa: F401
from fairness.shap_analysis import FEATURE_LABELS_ES  # noqa: F401
from data.features import MODEL_FEATURES  # noqa: F401

# ---------------------------------------------------------------------------
# Publication style defaults
# ---------------------------------------------------------------------------

PUB_STYLE: dict = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
}

# Model colors for PR curves
MODEL_COLORS: dict[str, str] = {
    "lr": "#1f77b4",
    "lgbm": "#2ca02c",
    "xgb": "#d62728",
}

MODEL_LABELS: dict[str, str] = {
    "lr": "Logistic Regression",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
}

# Language group labels (English)
LANGUAGE_LABELS_EN: dict[str, str] = {
    "castellano": "Castellano",
    "quechua": "Quechua",
    "aimara": "Aimara",
    "other_indigenous": "Other indigenous",
    "foreign": "Foreign",
}

# Keep old name as alias for backwards compatibility
LANGUAGE_LABELS_ES = LANGUAGE_LABELS_EN

# Feature name → Spanish label mapping (for SHAP figures)
FEATURE_TO_LABEL_ES: dict[str, str] = dict(zip(MODEL_FEATURES, FEATURE_LABELS_ES))


def setup_pub_style() -> None:
    """Apply publication-quality matplotlib defaults."""
    matplotlib.use("Agg")
    plt.rcParams.update(PUB_STYLE)


def save_dual_format(fig: plt.Figure, path_stem: str | Path) -> None:
    """Save figure as both PNG (300dpi) and PDF (vector)."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def small_sample_annotation(n: int, threshold: int = 50) -> str:
    """Return '*' for small samples, '' otherwise."""
    return "*" if n < threshold else ""
