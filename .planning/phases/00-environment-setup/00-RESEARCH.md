# Phase 0: Environment Setup - Research

**Researched:** 2026-02-07
**Domain:** Nix flakes, uv, Python 3.12, ML dependency stack
**Confidence:** HIGH

## Summary

This phase sets up a reproducible development environment using Nix flakes for system dependencies (Python 3.12, OpenMP, cmake, uv, ruff) and uv for Python package management (polars, lightgbm, xgboost, fairlearn, shap, optuna, onnxmltools). The user is running NixOS 25.11 (unstable) with Nix 2.31.2, uv 0.9.28, and direnv 2.37.1 already available system-wide.

The standard approach is: Nix flake provides a `devShell` with Python 3.12 + system libraries + `LD_LIBRARY_PATH` configuration, then `uv sync` installs the full Python ML stack into a `.venv`. The critical NixOS-specific pitfall is that LightGBM's manylinux wheel dynamically links to `libgomp.so.1` at runtime, requiring `libgcc.lib` in `LD_LIBRARY_PATH`. direnv with `use flake` automates shell entry.

**Primary recommendation:** Use `nixos-25.05` as the pinned nixpkgs input (latest stable), provide Python 3.12 + system libs via `mkShell`, set `LD_LIBRARY_PATH` to include `stdenv.cc.cc.lib` (libstdc++) and `libgcc.lib` (libgomp), and let uv handle all Python packages via `pyproject.toml`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Nix flake
- Python 3.12 managed via Nix flakes
- System dependencies: OpenMP (for LightGBM), cmake
- Include `uv` in the flake so `uv sync` is available immediately after `nix develop`
- Include `ruff` in the Nix shell (already mentioned in spec Section 2)
- direnv integration: use `.envrc` with `use flake` so entering the directory activates the environment (per user's CLAUDE.md preference for `direnv reload`)

#### pyproject.toml
- All Python packages from spec Section 2: polars>=1.0, scikit-learn>=1.5, lightgbm>=4.0, xgboost>=2.0, fairlearn>=0.11, shap>=0.45, statsmodels, matplotlib, seaborn, plotly, onnx, onnxmltools, onnxruntime, pytest, jupyterlab, optuna
- Also include: `enahodata` and `requests` and `tqdm` (needed by download.py)
- Use compatible version ranges (>=X.Y) as specified in the tech stack, not exact pins -- uv lock file handles reproducibility

#### Directory structure
- Exact match to spec Section 3 -- no deviations
- All `__init__.py` files created (src/, src/data/, src/models/, src/fairness/)
- Notebooks directory with placeholder `.ipynb` files (01-04)
- `outputs/figures/` directory created

#### .gitignore
- Exact match to spec Section 3: `data/raw/`, `data/processed/`, `*.pyc`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, `*.egg-info/`, `dist/`, `build/`, `.env`
- `data/exports/` is NOT gitignored -- it's tracked

#### Download script
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

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core: Nix Flake Components

| Package | Nixpkgs Attr | Version (verified) | Purpose |
|---------|-------------|-------------------|---------|
| Python 3.12 | `python312` | 3.12.12 | Python interpreter |
| uv | `uv` | 0.9.28 | Python package/project manager |
| ruff | `ruff` | 0.14.6 | Python linter/formatter |
| cmake | `cmake` | 4.1.2 | Build system (LightGBM/XGBoost source builds) |
| OpenMP | `llvmPackages.openmp` | 21.1.7 | Parallel computation for LightGBM |
| libgcc (lib) | `libgcc.lib` | gcc-14.3.0 | Provides libgomp.so.1 at runtime |
| zlib | `zlib` | 1.3.1 | Compression library (numpy, polars wheels) |
| stdenv.cc.cc.lib | (builtin) | gcc-14.3.0 | Provides libstdc++.so.6 for compiled wheels |

**Nixpkgs input:** Pin to `nixos-25.05` (NixOS 25.05 "Warbler", released May 2025, EOL 2025-12-31). This is the latest stable channel. The user's system runs nixos-unstable (25.11), but the flake should pin to stable for reproducibility.

### Core: Python Packages (via pyproject.toml + uv)

| Package | Version Constraint | Latest Verified | Purpose |
|---------|-------------------|-----------------|---------|
| polars | >=1.0 | (check PyPI) | DataFrame library |
| scikit-learn | >=1.5 | (check PyPI) | ML algorithms, preprocessing |
| lightgbm | >=4.0 | 4.6.0 (Feb 2025) | Gradient boosting (matches Alerta Escuela) |
| xgboost | >=2.0 | (check PyPI) | Gradient boosting comparison |
| fairlearn | >=0.11 | 0.13.0 (Oct 2025) | Fairness metrics (MetricFrame) |
| shap | >=0.45 | 0.50.0 (Nov 2025) | SHAP interpretability |
| statsmodels | (any) | (check PyPI) | Statistical models |
| matplotlib | (any) | (check PyPI) | Plotting |
| seaborn | (any) | (check PyPI) | Statistical plotting |
| plotly | (any) | (check PyPI) | Interactive plots |
| onnx | (any) | (check PyPI) | ONNX model format |
| onnxmltools | (any) | 1.16.0 | Convert LightGBM to ONNX |
| onnxruntime | (any) | 1.24.1 | Run ONNX models |
| pytest | (any) | (check PyPI) | Testing framework |
| jupyterlab | (any) | (check PyPI) | Notebook interface |
| optuna | (any) | (check PyPI) | Hyperparameter tuning |
| enahodata | (any) | 0.0.3 (Jan 2025) | ENAHO data download from INEI |
| requests | (any) | (check PyPI) | HTTP client (download.py) |
| tqdm | (any) | (check PyPI) | Progress bars (download.py) |

### Alternatives Considered

| Instead of | Could Use | Why NOT |
|------------|-----------|--------|
| uv | poetry/pip | User decided uv; it's faster, handles lockfiles |
| Nix flake | conda | User decided Nix; better reproducibility |
| nixos-25.05 | nixos-unstable | Stable is more reproducible for flake pins |

## Architecture Patterns

### Recommended Flake Structure

```nix
# flake.nix
{
  description = "Alerta Escuela Equity Audit";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Libraries needed by Python wheels at runtime
      libPath = with pkgs; lib.makeLibraryPath [
        stdenv.cc.cc.lib   # libstdc++.so.6
        libgcc.lib         # libgomp.so.1 (OpenMP runtime for LightGBM)
        zlib               # libz.so (numpy, polars)
        llvmPackages.openmp # libomp.so
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          ruff
          cmake
          llvmPackages.openmp
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
          # Shell welcome message here
        '';
      };
    };
}
```

**Key pattern:** Use `lib.makeLibraryPath` to construct `LD_LIBRARY_PATH` with all libraries that manylinux wheels need at runtime. This is the standard NixOS pattern for Python development with pip/uv-installed packages.

### .envrc Pattern

```bash
# .envrc
use flake
```

This is the minimal direnv configuration. When a developer enters the directory, direnv automatically runs `nix develop` and exports the shell environment. The user can run `direnv reload` to refresh after flake changes.

**IMPORTANT:** The current `.gitignore` ignores `.envrc` (line 139). This must be removed so `.envrc` is tracked in git.

### pyproject.toml Pattern

```toml
[project]
name = "alerta-escuela-audit"
version = "0.1.0"
description = "Independent equity audit of Peru's Alerta Escuela dropout prediction system"
requires-python = ">=3.12"

dependencies = [
    "polars>=1.0",
    "scikit-learn>=1.5",
    "lightgbm>=4.0",
    # ... etc
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"
```

Use `uv init` or create manually. The `>=X.Y` version ranges match the spec; the `uv.lock` file (auto-generated by `uv sync`) handles exact reproducibility.

### Directory Structure (from spec Section 3)

```
alerta-escuela-audit/
├── flake.nix
├── flake.lock
├── .envrc
├── pyproject.toml
├── uv.lock
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── download.py
│   ├── models/
│   │   └── __init__.py
│   └── fairness/
│       └── __init__.py
├── tests/
│   ├── gates/
│   └── unit/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_fairness_analysis.ipynb
│   └── 04_findings_visualization.ipynb
├── data/
│   ├── raw/          # gitignored
│   ├── processed/    # gitignored
│   └── exports/      # TRACKED
└── outputs/
    └── figures/
```

### download.py Relocation Pattern

The existing `download.py` at repo root must be moved to `src/data/download.py`. When relocated, the `PROJECT_ROOT` calculation on line 44 must be updated:

**Current (repo root):** `Path(__file__).resolve().parent.parent` -- goes up 2 levels (assumes `scripts/` subdirectory, which was the original intent per the docstring "python scripts/download_data.py")

**After move to src/data/:** `Path(__file__).resolve().parent.parent.parent` -- needs 3 levels up to reach project root

**Better pattern:** Use a more robust approach that doesn't depend on nesting depth:
```python
# Find project root by looking for pyproject.toml
def _find_project_root():
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "pyproject.toml").exists():
            return path
        path = path.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")

PROJECT_ROOT = _find_project_root()
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Python version management | Custom install scripts | Nix `python312` package | Exact version, reproducible |
| Package dependency resolution | Manual pip install | `uv sync` with pyproject.toml | Handles conflicts, lockfile |
| OpenMP runtime path | Manual `export LD_LIBRARY_PATH` | `lib.makeLibraryPath` in flake | Correct nix store paths |
| Environment activation | Shell scripts | direnv `use flake` | Automatic, well-tested |
| ENAHO data download | Custom HTTP scraping | `enahodata` library | Handles INEI's download portal |

**Key insight:** The entire point of this phase is to make the "works on my machine" problem disappear. Every path, version, and dependency must be declared, not assumed.

## Common Pitfalls

### Pitfall 1: LightGBM libgomp.so.1 Missing on NixOS

**What goes wrong:** `import lightgbm` fails with `OSError: libgomp.so.1: cannot open shared object file: No such file or directory`
**Why it happens:** LightGBM's manylinux wheel (`lightgbm-4.6.0-py3-none-manylinux_2_28_x86_64.whl`) bundles its own `lib_lightgbm.so` but dynamically links to `libgomp.so.1` (GCC's OpenMP runtime) from the system. On NixOS, system libraries are not in `/usr/lib/`.
**How to avoid:** Include `libgcc.lib` in the `LD_LIBRARY_PATH` via the Nix flake's `lib.makeLibraryPath`. Verified: `libgcc.lib` exists at `/nix/store/gnpwfj9gpk8ll7dhf65a6r5gjbs4qbap-gcc-14.3.0-lib` on this system.
**Warning signs:** Any `OSError` about missing `.so` files after `uv sync`.
**Confidence:** HIGH -- verified on this exact NixOS system.

### Pitfall 2: .envrc Gitignored by Default

**What goes wrong:** The `.envrc` file needed for direnv is in `.gitignore` (line 139 of the current file).
**Why it happens:** The current `.gitignore` is a standard GitHub Python template that treats `.envrc` as an environment file to exclude.
**How to avoid:** When rewriting `.gitignore` to match the spec, ensure `.envrc` is NOT listed. The spec's gitignore list does not include `.envrc`.
**Warning signs:** New clone doesn't auto-activate the Nix shell because `.envrc` wasn't committed.
**Confidence:** HIGH -- directly verified in the current repo.

### Pitfall 3: download.py PROJECT_ROOT Path After Relocation

**What goes wrong:** After moving `download.py` from repo root to `src/data/download.py`, the `PROJECT_ROOT = Path(__file__).resolve().parent.parent` calculation points to `src/` instead of the actual project root.
**Why it happens:** The script was written assuming a `scripts/` location (1 level deep) but is being moved to `src/data/` (2 levels deep).
**How to avoid:** Update the `parent` chain to `.parent.parent.parent`, or use a root-finding function that searches upward for `pyproject.toml`.
**Warning signs:** Data downloads go to wrong directory; `data/raw/` created inside `src/` instead of project root.
**Confidence:** HIGH -- directly verified by reading download.py line 44.

### Pitfall 4: SHAP Requires Python >=3.11

**What goes wrong:** If someone tried to use Python 3.10 or earlier, `shap>=0.45` (specifically 0.50.0) would fail to install.
**Why it happens:** SHAP 0.50.0 dropped support for Python <3.11.
**How to avoid:** Python 3.12 (as specified) satisfies this constraint. Just be aware this is a floor.
**Warning signs:** `uv sync` resolution errors mentioning Python version.
**Confidence:** HIGH -- verified on PyPI.

### Pitfall 5: fairlearn Only Supports Up to Python 3.12

**What goes wrong:** If someone used Python 3.13, fairlearn 0.13.0 might not work.
**Why it happens:** fairlearn 0.13.0 classifiers only list Python 3.9-3.12.
**How to avoid:** Python 3.12 (as specified) is the sweet spot -- newest version supported by both shap and fairlearn.
**Warning signs:** `uv sync` picks an older fairlearn version or fails.
**Confidence:** HIGH -- verified on PyPI.

### Pitfall 6: uv.lock Should Be Committed

**What goes wrong:** Different developers get different package versions.
**Why it happens:** `pyproject.toml` has version ranges, not exact pins. Without `uv.lock`, each `uv sync` resolves independently.
**How to avoid:** Commit `uv.lock` to git. The current `.gitignore` template has `#uv.lock` (commented out), which means it IS tracked by default. Keep it that way.
**Warning signs:** "Works on my machine" issues between developers.
**Confidence:** HIGH -- standard uv practice.

### Pitfall 7: Nix Flake Requires flake.lock Commit

**What goes wrong:** `nix develop` resolves nixpkgs to a different revision for different users.
**Why it happens:** Without `flake.lock`, the nixpkgs input is resolved at build time.
**How to avoid:** Run `nix flake lock` after creating `flake.nix`, then commit `flake.lock`.
**Warning signs:** Different developers get different Python/cmake versions.
**Confidence:** HIGH -- standard Nix flake practice.

## Code Examples

### Complete flake.nix (Recommended)

```nix
# Source: Synthesized from NixOS wiki + ML flake patterns + system verification
{
  description = "Alerta Escuela Equity Audit - independent equity audit of Peru's dropout prediction system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Runtime libraries for Python manylinux wheels
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib      # libstdc++.so.6
        libgcc.lib            # libgomp.so.1 (OpenMP for LightGBM)
        zlib                  # libz.so (various wheels)
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          ruff
          cmake
        ];

        env = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        };

        shellHook = ''
          echo "Alerta Escuela Equity Audit"
          echo "Python: $(python3 --version)"
          echo "uv: $(uv --version)"
          echo ""
          echo "Run 'uv sync' to install Python dependencies"
        '';
      };
    };
}
```

**Note:** Using `env.LD_LIBRARY_PATH` instead of exporting in `shellHook` is cleaner and the modern `mkShell` approach.

### Complete .envrc

```bash
use flake
```

### Complete pyproject.toml Structure

```toml
[project]
name = "alerta-escuela-audit"
version = "0.1.0"
description = "Independent equity audit of Peru's Alerta Escuela dropout prediction system"
requires-python = ">=3.12"
license = "MIT"

dependencies = [
    "polars>=1.0",
    "scikit-learn>=1.5",
    "lightgbm>=4.0",
    "xgboost>=2.0",
    "fairlearn>=0.11",
    "shap>=0.45",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "plotly",
    "onnx",
    "onnxmltools",
    "onnxruntime",
    "pytest",
    "jupyterlab",
    "optuna",
    "enahodata",
    "requests",
    "tqdm",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"
```

### Minimal Notebook Placeholder (JSON)

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook Title\n",
    "\n",
    "Placeholder -- to be populated in later phases."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

### .gitignore (Spec-Compliant)

```gitignore
# Data directories
data/raw/
data/processed/
# NOTE: data/exports/ is intentionally NOT gitignored -- it is tracked

# Python
*.pyc
__pycache__/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# Environment
.env

# Nix
result
```

## Claude's Discretion Recommendations

### Nixpkgs Revision: Use `nixos-25.05`

**Recommendation:** Pin to `github:NixOS/nixpkgs/nixos-25.05`

**Rationale:** NixOS 25.05 "Warbler" (released May 2025) is the latest stable release. It will receive security updates until 2025-12-31. The user's system runs nixos-unstable (25.11), but flakes should pin to stable for reproducibility. The verified package versions on the current system are all compatible (Python 3.12.12, cmake 4.1.2, ruff 0.14.6).

**Confidence:** HIGH -- NixOS release verified via `nixos-version` and search results.

### Justfile: Yes, Include One

**Recommendation:** Include a `justfile` with convenience commands. `just` is available in nixpkgs (version 1.43.1). Add it to the flake's packages.

**Rationale:** A justfile provides discoverability of common operations without requiring developers to remember exact commands. It is lightweight (single file, no build system) and `just --list` serves as documentation.

**Suggested commands:**
```just
# List available commands
default:
    @just --list

# Install Python dependencies
sync:
    uv sync

# Run tests
test:
    uv run pytest

# Run gate tests only
gates:
    uv run pytest tests/gates/

# Download data
download:
    uv run python src/data/download.py

# Lint
lint:
    ruff check src/ tests/

# Format
fmt:
    ruff format src/ tests/
```

**Confidence:** HIGH -- `just` is well-established, trivially simple.

### Shell Hook: Brief Welcome Message

**Recommendation:** Display project name, Python version, and uv version. Include a reminder to run `uv sync`. Keep it under 5 lines.

**Rationale:** Confirms the environment is active without being noisy. The "run uv sync" reminder is useful for fresh clones.

**Confidence:** HIGH -- standard practice.

### Notebook Placeholders: Minimal Markdown Cell

**Recommendation:** Each notebook gets a single markdown cell with the notebook title and "Placeholder -- to be populated in later phases." Use the minimal valid `.ipynb` JSON structure (shown in Code Examples above).

**Notebook titles:**
1. `01_data_exploration.ipynb` -- "Data Exploration"
2. `02_model_training.ipynb` -- "Model Training"
3. `03_fairness_analysis.ipynb` -- "Fairness Analysis"
4. `04_findings_visualization.ipynb` -- "Findings Visualization"

**Confidence:** HIGH -- minimal valid notebooks; content is not important at this phase.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pip + requirements.txt | uv + pyproject.toml + uv.lock | 2024 | 10-100x faster installs, proper lockfile |
| conda for ML | Nix + uv | 2024-2025 | Better system-level reproducibility |
| virtualenv + activate | direnv `use flake` | 2023+ | Automatic environment on directory entry |
| pip-tools / pip-compile | uv lock | 2024 | Faster resolution, single tool |
| shap <0.50 (Python 3.8+) | shap 0.50+ (Python >=3.11) | Nov 2025 | Python 3.12 minimum practically required |

## Open Questions

1. **enahodata reliability for 2024 data**
   - What we know: enahodata 0.0.3 supports modules 02, 03, 05, 34, 37 and years. The existing download.py uses it successfully (per the fact that it exists and was written).
   - What's unclear: Whether INEI has changed their download URLs for 2024 data since the script was written.
   - Recommendation: The download script already has good error handling and manual download fallback messages. Accept this risk; fix URLs if needed during execution.

2. **datosabiertos.gob.pe URL stability**
   - What we know: The admin dropout CSV URLs in download.py are hardcoded and "may change" (per the script's own comment on line 51).
   - What's unclear: Whether the 2023 URLs are still valid.
   - Recommendation: Include a comment with the base search URL. The script already handles 404s gracefully.

## Sources

### Primary (HIGH confidence)
- NixOS system verification: `nix eval nixpkgs#python312.version` = "3.12.12", `nix eval nixpkgs#llvmPackages.openmp.version` = "21.1.7", `nix eval nixpkgs#cmake.version` = "4.1.2", `nix eval nixpkgs#ruff.version` = "0.14.6", `nix eval nixpkgs#uv.version` = "0.9.28"
- `nixos-version` = 25.11.20260203.e576e3c (Xantusia) -- confirms NixOS unstable
- `nix eval nixpkgs#libgcc.lib.outPath` = `/nix/store/gnpwfj9gpk8ll7dhf65a6r5gjbs4qbap-gcc-14.3.0-lib` -- confirms libgomp availability
- [NixOS/nixpkgs lightgbm derivation](https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/python-modules/lightgbm/default.nix) -- system dependencies: cmake, ninja, llvmPackages.openmp
- [LightGBM PyPI](https://pypi.org/project/lightgbm/) -- 4.6.0, manylinux_2_28 wheel confirmed
- [fairlearn PyPI](https://pypi.org/project/fairlearn/) -- 0.13.0, Python 3.9-3.12
- [SHAP PyPI](https://pypi.org/project/shap/) -- 0.50.0, Python >=3.11
- [enahodata PyPI](https://pypi.org/project/enahodata/) -- 0.0.3, deps: requests, tqdm
- [enahodata GitHub](https://github.com/MaykolMedrano/enahodata_py) -- API confirmed: enahodata(modulos, anios, descomprimir, only_dta, overwrite, output_dir, panel)
- Existing download.py at `/home/hybridz/Projects/Alerta-Escuela-Equity-Audit/download.py` -- verified line 44 path issue

### Secondary (MEDIUM confidence)
- [NixOS 25.05 release announcement](https://nixos.org/blog/announcements/2025/nixos-2505/) -- "Warbler", released May 2025
- [NixOS Discourse: libgomp on NixOS](https://discourse.nixos.org/t/which-package-provides-libgomp1/35515) -- `libgcc.lib` provides libgomp.so.1
- [ML Nix Flake guide](https://romaingrx.com/notes/hassle-free-ml-environment-with-nix-flakes/) -- flake pattern with uv + LD_LIBRARY_PATH
- [NixOS Wiki: Python](https://wiki.nixos.org/wiki/Python) -- uv recommended, lib.makeLibraryPath pattern
- [LightGBM Installation Guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) -- OpenMP required for default build

### Tertiary (LOW confidence)
- None -- all findings verified with primary or secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions verified directly on user's NixOS system via `nix eval`
- Architecture: HIGH -- flake pattern synthesized from multiple verified sources and NixOS wiki
- Pitfalls: HIGH -- all pitfalls verified by direct inspection of repo files and system state
- Claude's discretion recommendations: HIGH -- all based on verified package availability

**Research date:** 2026-02-07
**Valid until:** 2026-03-09 (30 days -- stable domain, pinned versions)
