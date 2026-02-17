"""Gate test 3.4: Findings distillation and final export validation.

Validates findings.json structure, bilingual content, metric_source path
resolution against actual export files, and completeness of the export
README. Prints a human-review summary with all finding headlines.

Usage::

    uv run pytest tests/gates/test_gate_3_4.py -v -s

Use ``-s`` flag to see the human-review print tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import find_project_root

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = find_project_root()
FINDINGS_PATH = ROOT / "data" / "exports" / "findings.json"
EXPORTS_DIR = ROOT / "data" / "exports"

REQUIRED_FIELDS = {
    "id",
    "stat",
    "headline_es",
    "headline_en",
    "explanation_es",
    "explanation_en",
    "metric_source",
    "visualization_type",
    "data_key",
    "severity",
}

VALID_SEVERITIES = {"critical", "high", "medium", "low"}

ALL_EXPORT_FILES = [
    "findings.json",
    "fairness_metrics.json",
    "shap_values.json",
    "choropleth.json",
    "model_results.json",
    "descriptive_tables.json",
    "onnx/lightgbm_dropout.onnx",
]


@pytest.fixture(scope="module")
def findings_data():
    """Load and parse findings.json."""
    assert FINDINGS_PATH.exists(), (
        "findings.json not found. "
        "Run: uv run python src/fairness/findings.py"
    )
    with open(FINDINGS_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_exports():
    """Load all JSON exports into a dict keyed by filename."""
    exports = {}
    for fname in ALL_EXPORT_FILES:
        fpath = EXPORTS_DIR / fname
        if fpath.exists() and fpath.suffix == ".json":
            with open(fpath) as f:
                exports[fname] = json.load(f)
    return exports


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_findings_json_exists_and_valid(findings_data):
    """Assert findings.json exists, is valid JSON, and has a findings list."""
    assert isinstance(findings_data, dict), "findings.json root must be a dict"
    assert "findings" in findings_data, "findings.json must have 'findings' key"
    assert isinstance(findings_data["findings"], list), "'findings' must be a list"


def test_findings_count(findings_data):
    """Assert 5-10 findings are present."""
    n = len(findings_data["findings"])
    print(f"  Finding count: {n}")
    assert 5 <= n <= 10, f"Expected 5-10 findings, got {n}"


def test_findings_required_fields(findings_data):
    """Assert each finding has all 10 required fields."""
    for i, finding in enumerate(findings_data["findings"]):
        actual = set(finding.keys())
        missing = REQUIRED_FIELDS - actual
        assert not missing, (
            f"Finding {i} (id={finding.get('id')}) missing fields: {missing}"
        )


def test_findings_ids_unique(findings_data):
    """Assert no duplicate finding IDs."""
    ids = [f["id"] for f in findings_data["findings"]]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_findings_severity_values(findings_data):
    """Assert severity values are valid."""
    for finding in findings_data["findings"]:
        assert finding["severity"] in VALID_SEVERITIES, (
            f"Finding {finding['id']}: invalid severity '{finding['severity']}'. "
            f"Must be one of {VALID_SEVERITIES}"
        )


def test_metric_source_paths_resolve(findings_data, all_exports):
    """KEY TEST: Each metric_source path resolves to a non-null value."""
    for finding in findings_data["findings"]:
        ms = finding["metric_source"]
        path_str = ms["path"]

        # Parse format: filename.json#dot.separated.path
        assert "#" in path_str, (
            f"Finding {finding['id']}: metric_source path '{path_str}' "
            f"must contain '#' separator"
        )
        filename, json_path = path_str.split("#", 1)

        assert filename in all_exports, (
            f"Finding {finding['id']}: export file '{filename}' not loaded. "
            f"Available: {list(all_exports.keys())}"
        )

        # Navigate the dot-separated path
        data = all_exports[filename]
        parts = json_path.split(".")
        for j, part in enumerate(parts):
            traversed = ".".join(parts[: j + 1])
            assert isinstance(data, dict), (
                f"Finding {finding['id']}: at '{traversed}' in {filename}, "
                f"expected dict but got {type(data).__name__}"
            )
            assert part in data, (
                f"Finding {finding['id']}: key '{part}' not found at "
                f"'{traversed}' in {filename}. Available: {list(data.keys())}"
            )
            data = data[part]

        # Value must be non-null (lists must be non-empty)
        if isinstance(data, list):
            assert len(data) > 0, (
                f"Finding {finding['id']}: resolved path '{path_str}' "
                f"is an empty list"
            )
        else:
            assert data is not None, (
                f"Finding {finding['id']}: resolved path '{path_str}' "
                f"is null"
            )

        print(f"  {finding['id']}: {path_str} -> {repr(data)[:80]}")


def test_all_seven_exports_present():
    """Assert all 7 export files exist in data/exports/."""
    missing = []
    for fname in ALL_EXPORT_FILES:
        fpath = EXPORTS_DIR / fname
        if not fpath.exists():
            missing.append(fname)
    assert not missing, f"Missing export files: {missing}"


def test_readme_exists():
    """Assert data/exports/README.md exists."""
    readme = EXPORTS_DIR / "README.md"
    assert readme.exists(), "data/exports/README.md not found"


def test_readme_documents_all_files():
    """Assert README mentions all 7 export filenames."""
    readme = EXPORTS_DIR / "README.md"
    assert readme.exists(), "README.md not found"
    content = readme.read_text()

    for fname in ALL_EXPORT_FILES:
        # Use just the basename for matching
        basename = Path(fname).name
        assert basename in content, (
            f"README.md does not mention '{basename}'"
        )


def test_headlines_not_empty(findings_data):
    """Assert no empty headline strings."""
    for finding in findings_data["findings"]:
        assert finding["headline_es"].strip(), (
            f"Finding {finding['id']}: headline_es is empty"
        )
        assert finding["headline_en"].strip(), (
            f"Finding {finding['id']}: headline_en is empty"
        )


def test_explanations_length(findings_data):
    """Assert explanations are 2-3 sentences (50-500 chars)."""
    for finding in findings_data["findings"]:
        for lang in ("explanation_es", "explanation_en"):
            text = finding[lang]
            length = len(text)
            assert 50 <= length <= 500, (
                f"Finding {finding['id']}: {lang} length={length}, "
                f"expected 50-500 chars. Text: {text[:80]}..."
            )


# ---------------------------------------------------------------------------
# Human review (always passes)
# ---------------------------------------------------------------------------


def test_human_review_findings(findings_data):
    """Print all finding headlines for human review. Always passes.

    Run with ``-s`` flag to see output:
    ``uv run pytest tests/gates/test_gate_3_4.py -v -s``
    """
    print()
    print()
    print("=" * 70)
    print("GATE 3.4: FINDINGS DISTILLATION -- HUMAN REVIEW")
    print("=" * 70)

    for i, finding in enumerate(findings_data["findings"], 1):
        print(f"\n--- Finding {i}: {finding['id']} ---")
        print(f"  Severity:     {finding['severity']}")
        print(f"  Viz type:     {finding['visualization_type']}")
        print(f"  Stat:         {finding['stat']}")
        print(f"  Headline ES:  {finding['headline_es']}")
        print(f"  Headline EN:  {finding['headline_en']}")
        print(f"  Explain ES:   {finding['explanation_es'][:120]}...")
        print(f"  Explain EN:   {finding['explanation_en'][:120]}...")
        print(f"  Source:       {finding['metric_source']['path']}")
        print(f"  Data key:     {finding['data_key']}")

    print()
    print("  HUMAN REVIEW: Are headlines stat-forward and media-ready?")
    print("  Do Spanish headlines use natural Peruvian context?")
    print("  Do English headlines add geographic/institutional context?")
    print()
    print("=" * 70)

    # Always pass
    assert True
