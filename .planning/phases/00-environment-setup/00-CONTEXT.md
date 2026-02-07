# Phase 0: Environment Setup - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Developer can clone the repo, run `nix develop`, then `uv sync`, and have a fully working Python 3.12 environment with all ML dependencies (polars, lightgbm, xgboost, fairlearn, shap, optuna, onnxmltools) resolved. Directory structure matches spec Section 3 exactly. Download script fetches ENAHO and admin data into `data/raw/`. Data directories are gitignored; `data/exports/` is tracked.

</domain>

<decisions>
## Implementation Decisions

### Nix flake
- Python 3.12 managed via Nix flakes
- System dependencies: OpenMP (for LightGBM), cmake
- Include `uv` in the flake so `uv sync` is available immediately after `nix develop`
- Include `ruff` in the Nix shell (already mentioned in spec Section 2)
- direnv integration: use `.envrc` with `use flake` so entering the directory activates the environment (per user's CLAUDE.md preference for `direnv reload`)

### pyproject.toml
- All Python packages from spec Section 2: polars>=1.0, scikit-learn>=1.5, lightgbm>=4.0, xgboost>=2.0, fairlearn>=0.11, shap>=0.45, statsmodels, matplotlib, seaborn, plotly, onnx, onnxmltools, onnxruntime, pytest, jupyterlab, optuna
- Also include: `enahodata` and `requests` and `tqdm` (needed by download.py)
- Use compatible version ranges (>=X.Y) as specified in the tech stack, not exact pins — uv lock file handles reproducibility

### Directory structure
- Exact match to spec Section 3 — no deviations
- All `__init__.py` files created (src/, src/data/, src/models/, src/fairness/)
- Notebooks directory with placeholder `.ipynb` files (01-04)
- `outputs/figures/` directory created

### .gitignore
- Exact match to spec Section 3: `data/raw/`, `data/processed/`, `*.pyc`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, `*.egg-info/`, `dist/`, `build/`, `.env`
- `data/exports/` is NOT gitignored — it's tracked

### Download script
- Existing `download.py` in repo root will be moved to `src/data/download.py` per spec directory structure
- Script uses `enahodata` library for ENAHO (modules 02, 03, 05, 34, 37 for years 2018-2024)
- Admin dropout rates downloaded from datosabiertos.gob.pe
- Interactive prompts are acceptable (this is a manual developer step)
- Census, nightlights, and escolar data remain manual downloads (noted in script output)

### Claude's Discretion
- Exact nixpkgs revision to pin (use a recent stable revision)
- Whether to include a Makefile/justfile for convenience commands
- Shell hook message content (if any)
- Exact notebook placeholder content

</decisions>

<specifics>
## Specific Ideas

- specs.md is the single source of truth — all implementation details come from the spec (per roadmap decision)
- User prefers `direnv reload` workflow (from CLAUDE.md)
- download.py already exists and works — relocate it, don't rewrite from scratch

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 00-environment-setup*
*Context gathered: 2026-02-07*
