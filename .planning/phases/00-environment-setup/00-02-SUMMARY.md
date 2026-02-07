---
phase: 00-environment-setup
plan: 02
subsystem: infra
tags: [directory-structure, gitignore, jupyter, justfile, download-script, enahodata]

# Dependency graph
requires:
  - phase: 00-environment-setup
    provides: "flake.nix, pyproject.toml, .envrc from plan 00-01"
provides:
  - "Complete directory tree per spec Section 3"
  - "src/data/download.py with root-finding function"
  - "Python package markers (__init__.py) for src/, src/data/, src/models/, src/fairness/"
  - "Placeholder Jupyter notebooks (4)"
  - "justfile with convenience commands"
  - "Project-specific .gitignore"
affects: [01-data-pipeline, 02-model-training, 03-fairness-analysis]

# Tech tracking
tech-stack:
  added: [just, jupyter]
  patterns:
    - "Root-finding via pyproject.toml walk-up pattern"
    - "gitkeep files for empty tracked directories"
    - "justfile as task runner (uv-based commands)"

key-files:
  created:
    - src/__init__.py
    - src/data/__init__.py
    - src/data/download.py
    - src/models/__init__.py
    - src/fairness/__init__.py
    - tests/gates/.gitkeep
    - tests/unit/.gitkeep
    - data/exports/.gitkeep
    - outputs/figures/.gitkeep
    - notebooks/01_data_exploration.ipynb
    - notebooks/02_model_training.ipynb
    - notebooks/03_fairness_analysis.ipynb
    - notebooks/04_findings_visualization.ipynb
    - justfile
  modified:
    - .gitignore
    - download.py (deleted from root, relocated to src/data/download.py)

key-decisions:
  - "PROJECT_ROOT uses pyproject.toml walk-up instead of hardcoded parent.parent"
  - "Replaced GitHub Python template .gitignore with focused project-specific rules"
  - ".envrc tracked in git (not gitignored) for direnv team integration"
  - "data/exports/ tracked (not gitignored) for M4 site export artifacts"

patterns-established:
  - "Root discovery: _find_project_root() walks up to pyproject.toml"
  - "Directory markers: .gitkeep for empty dirs, __init__.py for Python packages"
  - "Task runner: justfile with uv-prefixed commands"

# Metrics
duration: 8min
completed: 2026-02-07
---

# Phase 0 Plan 2: Directory Scaffolding Summary

**Complete project directory tree with relocated download script, placeholder notebooks, justfile, and project-specific gitignore**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-07T20:45:01Z
- **Completed:** 2026-02-07T20:53:13Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments

- Created all directories from spec Section 3 (src/data/, src/models/, src/fairness/, tests/gates/, tests/unit/, data/raw/, data/processed/, data/exports/, outputs/figures/, notebooks/)
- Relocated download.py from project root to src/data/download.py with pyproject.toml-based root discovery
- Replaced bloated GitHub Python template .gitignore with focused project-specific rules (data/raw/ and data/processed/ ignored; data/exports/, .envrc, uv.lock tracked)
- Created 4 placeholder Jupyter notebooks and justfile with 8 convenience commands

## Task Commits

Each task was committed atomically:

1. **Task 1: Create directory structure + __init__.py files + .gitignore** - `300b79b` (chore)
2. **Task 2: Relocate download.py + create notebooks + justfile** - `459af15` (feat)

## Files Created/Modified

- `src/__init__.py` - Python package marker for src/
- `src/data/__init__.py` - Python package marker for src/data/
- `src/data/download.py` - ENAHO + admin data download script (relocated from root)
- `src/models/__init__.py` - Python package marker for src/models/
- `src/fairness/__init__.py` - Python package marker for src/fairness/
- `tests/gates/.gitkeep` - Directory marker for gate tests
- `tests/unit/.gitkeep` - Directory marker for unit tests
- `data/exports/.gitkeep` - Directory marker for tracked exports
- `outputs/figures/.gitkeep` - Directory marker for figure outputs
- `notebooks/01_data_exploration.ipynb` - Placeholder notebook
- `notebooks/02_model_training.ipynb` - Placeholder notebook
- `notebooks/03_fairness_analysis.ipynb` - Placeholder notebook
- `notebooks/04_findings_visualization.ipynb` - Placeholder notebook
- `justfile` - Task runner with uv sync, test, gates, download, lint, fmt, lab commands
- `.gitignore` - Replaced GitHub template with project-specific rules

## Decisions Made

- **Root discovery pattern:** Used pyproject.toml walk-up function instead of hardcoded `Path(__file__).resolve().parent.parent` since the script moved from root to src/data/ (2 levels deeper). This pattern is robust against future refactors.
- **.envrc tracked:** Removed .envrc from .gitignore (GitHub Python template included it). Required for direnv integration -- team members need this file when cloning.
- **data/exports/ tracked:** Not gitignored because it holds M4 site export JSON files that should be version-controlled.
- **Minimal .gitignore:** Replaced 208-line GitHub template with 27-line focused file covering only project-relevant patterns.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

- `python3` is not on PATH outside the Nix devShell; used `direnv exec` to access Python for notebook validation. This is expected behavior with the Nix flake setup.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- All directories exist for Phase 1 (data pipeline) to place files in correct locations
- download.py is ready to fetch ENAHO data (requires enahodata, requests packages)
- Blocker: ENAHO raw data must be downloaded before Phase 1 can process it
- justfile provides `just download` convenience command for data fetching

---
*Phase: 00-environment-setup*
*Completed: 2026-02-07*

## Self-Check: PASSED
