# Who Gets Missed? A Proxy Equity Audit of Survey-Derived Dropout Risk in Perú

First independent equity audit of Peru's **Alerta Escuela**, a LightGBM-based student dropout prediction system deployed by MINEDU (Ministry of Education) to 90,000+ educators since October 2020.

## Key Findings

- **Two-sided detection gap:** Indigenous-language students face 52% false positive rates ("surveillance bias") while Spanish-speaking students have 63% false negative rates ("invisibility bias")
- **Place, not identity:** SHAP analysis shows the model predicts through spatial/structural features (nightlights, district literacy rates, poverty index) rather than identity features, with zero overlap between top-5 predictors from logistic regression and gradient boosting
- **Urban indigenous blind spot:** Urban indigenous students have 75.3% false negative rates (n=89) — the most systematically missed subgroup
- **Algorithm-independent:** LightGBM and XGBoost produce near-identical fairness metrics (ratio = 1.0006), confirming findings are structural, not algorithmic artifacts

## Repository Structure

```
├── src/
│   ├── data/           # ENAHO loading, spatial merges, feature engineering
│   ├── models/         # Logistic regression, LightGBM, XGBoost, calibration
│   ├── fairness/       # Subgroup metrics, SHAP analysis, findings distillation
│   └── plotting.py     # Shared plotting utilities
├── scripts/
│   ├── rerun_pipeline.sh       # End-to-end pipeline runner
│   ├── publication_figures.py  # Publication-quality figures
│   └── generate_tables.py     # Auto-generate LaTeX tables from JSON
├── tests/
│   ├── gates/          # Gate tests verifying each phase's success criteria
│   └── unit/           # Unit tests
├── data/
│   ├── raw/            # ENAHO microdata (not tracked — see Data section)
│   ├── processed/      # Intermediate parquet files, trained models
│   └── exports/        # Final JSON exports, ONNX model, figures
├── paper/
│   ├── main.tex        # ACM-format LaTeX paper
│   ├── figures/        # Publication figures (PDF + PNG)
│   └── tables/         # Auto-generated LaTeX tables
└── outputs/
    └── figures/        # Additional analysis figures
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/enriqueflores/Alerta-Escuela-Equity-Audit.git
cd Alerta-Escuela-Equity-Audit
uv sync
```

For Nix users with direnv:

```bash
direnv allow
```

## Data

This analysis uses **ENAHO** (Encuesta Nacional de Hogares) microdata from Peru's national statistics institute (INEI) for years 2018-2023.

**ENAHO download (required):** ENAHO microdata must be downloaded separately from [INEI's data portal](http://iinei.inei.gob.pe/microdatos/). The pipeline expects Modules 200 and 300 for each year in `data/raw/`.

To download ENAHO data:

```bash
uv run python src/data/download.py --years 2018 2019 2020 2021 2022 2023
```

**Supplementary data:** Three supplementary sources (MINEDU administrative data, INEI census indicators, VIIRS nightlight imagery) are included as synthetic placeholders. The pipeline is designed to accept real data seamlessly when available — replace files in `data/raw/` and re-run.

## Reproducing the Analysis

After downloading ENAHO data:

```bash
bash scripts/rerun_pipeline.sh
```

This runs the full pipeline (Phases 3-13) sequentially: dataset merging, feature engineering, model training (LR + LightGBM + XGBoost), calibration, fairness analysis, SHAP interpretability, findings distillation, and publication outputs.

To run gate tests independently:

```bash
uv run pytest tests/gates/ -v
```

## Paper

The `paper/` directory contains an ACM-format LaTeX paper with auto-generated tables. To compile:

```bash
cd paper && latexmk -pdf main.tex
```

A pre-compiled PDF is included at `paper/main.pdf`.

## Outputs

The `data/exports/` directory contains all analysis outputs:

| File | Description |
|------|-------------|
| `descriptive_tables.json` | Weighted dropout rates by demographic dimensions |
| `model_results.json` | Model metrics, coefficients, feature importances |
| `fairness_metrics.json` | Subgroup fairness metrics across 7 dimensions + 3 intersections |
| `shap_values.json` | SHAP feature importance and interaction values |
| `findings.json` | 7 distilled findings with metric sources |
| `choropleth.json` | Regional dropout rates for map visualization |
| `onnx/lightgbm_dropout.onnx` | Exported LightGBM model (0.10 MB) |

## Citation

If you use this work, please cite:

```bibtex
@software{flores2026alertaaudit,
  author = {Flores, Enrique},
  title = {Who Gets Missed? A Proxy Equity Audit of Survey-Derived Dropout Risk in Perú},
  year = {2026},
  url = {https://github.com/enriqueflores/Alerta-Escuela-Equity-Audit}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

[MIT](LICENSE)
