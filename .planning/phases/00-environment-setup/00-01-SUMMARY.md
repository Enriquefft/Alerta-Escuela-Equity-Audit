---
phase: 00-environment-setup
plan: 01
subsystem: infra
tags: [nix, flake, python312, uv, lightgbm, nixos, devshell, direnv]

# Dependency graph
requires: []
provides:
  - "Nix flake devShell with Python 3.12, uv, ruff, cmake"
  - "LD_LIBRARY_PATH for manylinux wheel compatibility on NixOS"
  - "pyproject.toml with 19 Python dependencies (ML, audit, viz)"
  - "uv.lock pinning 141 packages for reproducibility"
  - "direnv integration via .envrc"
affects:
  - 00-environment-setup  # Plan 02 (gitignore, tooling config)
  - 01-data-acquisition   # All subsequent phases depend on working env

# Tech tracking
tech-stack:
  added:
    - "Python 3.12.12 (via Nix)"
    - "uv 0.7.22 (Python package manager)"
    - "ruff 0.11.13 (linter/formatter)"
    - "cmake 3.31.6 (build system)"
    - "polars 1.38.1"
    - "lightgbm 4.6.0"
    - "xgboost 3.1.3"
    - "fairlearn 0.13.0"
    - "shap 0.50.0"
    - "scikit-learn 1.8.0"
    - "optuna 4.7.0"
    - "onnx 1.20.1"
    - "onnxmltools 1.16.0"
    - "onnxruntime 1.24.1"
    - "statsmodels 0.14.6"
    - "matplotlib 3.10.8"
    - "seaborn 0.13.2"
    - "plotly 6.5.2"
    - "pytest 9.0.2"
    - "jupyterlab 4.5.3"
    - "enahodata 0.0.3"
    - "hatchling (build backend)"
  patterns:
    - "Nix flake for system deps, uv for Python deps"
    - "LD_LIBRARY_PATH via lib.makeLibraryPath for NixOS wheel compat"
    - "src layout with hatchling build backend"
    - "direnv reload workflow for environment activation"

key-files:
  created:
    - "flake.nix"
    - "flake.lock"
    - ".envrc"
    - "pyproject.toml"
    - "uv.lock"
    - "src/alerta_escuela_audit/__init__.py"
  modified: []

key-decisions:
  - "nixos-25.05 pinned as nixpkgs input (latest stable)"
  - "hatchling build backend with src layout (not flat layout)"
  - ">=X.Y version ranges; uv.lock handles reproducibility"
  - "libgcc.lib + stdenv.cc.cc.lib + zlib in LD_LIBRARY_PATH for manylinux wheels"
  - "uv sync --python python3.12 to avoid uv selecting newer system Python"

patterns-established:
  - "Nix provides system deps + Python interpreter; uv manages Python packages"
  - "LD_LIBRARY_PATH = lib.makeLibraryPath runtimeLibs in mkShell env attribute"
  - "direnv reload to activate environment (per user preference)"

# Metrics
duration: 12min
completed: 2026-02-07
---

# Phase 0 Plan 1: Nix Flake + pyproject.toml Summary

**Nix flake devShell with Python 3.12, uv, ruff, cmake + pyproject.toml declaring 19 ML/audit dependencies with NixOS LD_LIBRARY_PATH for LightGBM/XGBoost wheel compatibility**

## Performance

- **Duration:** 12 min
- **Started:** 2026-02-07T20:42:55Z
- **Completed:** 2026-02-07T20:54:55Z
- **Tasks:** 2
- **Files created:** 6

## Accomplishments

- Nix flake providing Python 3.12.12, uv 0.7.22, ruff 0.11.13, cmake 3.31.6
- LD_LIBRARY_PATH with libstdc++.so.6, libgomp.so.1, libz.so for NixOS manylinux wheel compatibility
- pyproject.toml with all 19 Python dependencies resolving to 141 packages via uv
- LightGBM 4.6.0 imports without libgomp.so.1 errors on NixOS
- All critical ML imports verified: polars, lightgbm, xgboost, fairlearn, shap, optuna, onnxmltools, onnxruntime
- direnv integration via .envrc with `use flake`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create flake.nix with Python 3.12 devShell** - `d571c49` (chore)
2. **Task 2: Create pyproject.toml with all Python dependencies** - `2c9e6c4` (feat)

## Files Created/Modified

- `flake.nix` - Nix flake with nixos-25.05, Python 3.12 devShell, LD_LIBRARY_PATH
- `flake.lock` - Nix lock file pinning nixpkgs to ac62194c (2026-01-02)
- `.envrc` - direnv integration (`use flake`)
- `pyproject.toml` - Project metadata + 19 Python dependencies + hatchling build
- `uv.lock` - Lock file pinning 141 Python packages
- `src/alerta_escuela_audit/__init__.py` - Package stub for hatchling build

## Decisions Made

- **nixos-25.05 as nixpkgs input:** Latest stable NixOS channel, provides Python 3.12.12
- **hatchling with src layout:** Added `src/alerta_escuela_audit/__init__.py` and `[tool.hatch.build.targets.wheel] packages` config because hatchling requires a discoverable package directory for editable installs
- **>=X.Y version ranges:** Followed spec guidance; uv.lock handles exact reproducibility
- **libgcc.lib for libgomp.so.1:** On NixOS 25.05, `libgcc.lib` and `stdenv.cc.cc.lib` resolve to same gcc-14.3.0-lib store path, providing both libstdc++ and libgomp

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed hatchling build backend path**
- **Found during:** Task 2 (pyproject.toml creation)
- **Issue:** Plan specified `build-backend = "hatchling.backends"` which is incorrect; the correct module path is `hatchling.build`
- **Fix:** Changed to `build-backend = "hatchling.build"`
- **Files modified:** pyproject.toml
- **Verification:** `uv sync` succeeds with correct build backend
- **Committed in:** 2c9e6c4 (Task 2 commit)

**2. [Rule 3 - Blocking] Created package stub for hatchling editable install**
- **Found during:** Task 2 (uv sync)
- **Issue:** hatchling could not find package directory matching project name `alerta_escuela_audit`; errored with "Unable to determine which files to ship inside the wheel"
- **Fix:** Created `src/alerta_escuela_audit/__init__.py` and added `[tool.hatch.build.targets.wheel] packages = ["src/alerta_escuela_audit"]` to pyproject.toml
- **Files modified:** pyproject.toml, src/alerta_escuela_audit/__init__.py (new)
- **Verification:** `uv sync` completes successfully, package installed in editable mode
- **Committed in:** 2c9e6c4 (Task 2 commit)

**3. [Rule 3 - Blocking] Forced Python 3.12 for uv sync**
- **Found during:** Task 2 (uv sync)
- **Issue:** uv selected Python 3.14.2 (from user's uv-managed Python installs) instead of Nix-provided Python 3.12, because `requires-python = ">=3.12"` allows any version
- **Fix:** Used `uv sync --python python3.12` to force the Nix-provided interpreter
- **Files modified:** None (runtime behavior only)
- **Verification:** `.venv` created with Python 3.12.12, all packages install correctly
- **Committed in:** 2c9e6c4 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All auto-fixes necessary for correct operation. No scope creep. The plan's recommended flake.nix content had a typo in the build backend path, and hatchling's editable install requirements were not accounted for.

## Issues Encountered

- **uv Python version selection:** uv defaults to highest available Python when `requires-python` allows it. The user has Python 3.14.2 installed via uv's managed Pythons. Fixed by explicitly specifying `--python python3.12`. Future sessions using `direnv reload` within the Nix shell will have the correct Python in PATH.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Environment fully operational: `nix develop` provides all system tools, `uv sync` installs all Python packages
- Ready for Plan 00-02 (gitignore, ruff config, pytest config, directory structure)
- Ready for Phase 1 data acquisition once environment setup phase completes
- Note: `.envrc` is now tracked in git (current .gitignore does not exclude it)

## Self-Check: PASSED

---
*Phase: 00-environment-setup*
*Completed: 2026-02-07*
