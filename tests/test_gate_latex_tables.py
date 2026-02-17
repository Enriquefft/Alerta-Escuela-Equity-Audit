"""Gate tests for Phase 13: LaTeX Template + Table Generation."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
EXPORTS = ROOT / "data" / "exports"

TABLE_FILES = [
    "table_01_sample.tex",
    "table_02_language.tex",
    "table_03_region_poverty.tex",
    "table_04_models.tex",
    "table_05_lr_coefficients.tex",
    "table_06_fairness_language.tex",
    "table_07_intersection.tex",
    "table_08_shap.tex",
]


def test_01_main_tex_exists():
    """paper/main.tex exists and contains \\documentclass."""
    tex = PAPER / "main.tex"
    assert tex.exists(), "paper/main.tex not found"
    content = tex.read_text()
    assert r"\documentclass" in content


def test_02_references_bib_exists():
    """paper/references.bib exists and contains at least 10 @ entries."""
    bib = PAPER / "references.bib"
    assert bib.exists(), "paper/references.bib not found"
    content = bib.read_text()
    entries = re.findall(r"^@\w+\{", content, re.MULTILINE)
    assert len(entries) >= 10, f"Only {len(entries)} bib entries, need >= 10"


def test_03_all_table_files_exist():
    """All 8 table files exist in paper/tables/."""
    for name in TABLE_FILES:
        assert (TABLES / name).exists(), f"Missing {name}"


def test_04_tables_have_tabular():
    """Each table file contains \\begin{tabular} and \\end{tabular}."""
    for name in TABLE_FILES:
        content = (TABLES / name).read_text()
        assert r"\begin{tabular}" in content, f"{name} missing \\begin{{tabular}}"
        assert r"\end{tabular}" in content, f"{name} missing \\end{{tabular}}"


def test_05_t2_castellano_rate():
    """T2 spot-check: castellano weighted_rate matches source within 0.001."""
    with open(EXPORTS / "descriptive_tables.json") as f:
        data = json.load(f)
    castellano = [g for g in data["language"] if g["group"] == "castellano"][0]
    expected = castellano["weighted_rate"]

    content = (TABLES / "table_02_language.tex").read_text()
    # Find castellano row and extract rate
    for line in content.splitlines():
        if "Castellano" in line or "castellano" in line.lower():
            nums = re.findall(r"(\d+\.\d+)", line)
            # First decimal should be the rate
            if nums:
                actual = float(nums[0])
                assert abs(actual - expected) < 0.001, (
                    f"Castellano rate {actual} != {expected}"
                )
                return
    pytest.fail("Castellano row not found in table_02_language.tex")


def test_06_t4_lgbm_val_prauc():
    """T4 spot-check: LightGBM val PR-AUC matches source within 0.001."""
    with open(EXPORTS / "model_results.json") as f:
        data = json.load(f)
    expected = data["lightgbm"]["metrics"]["validate_2022"]["weighted"]["pr_auc"]

    content = (TABLES / "table_04_models.tex").read_text()
    for line in content.splitlines():
        if "PR-AUC" in line and ("val" in line.lower() or "Val" in line):
            nums = re.findall(r"(\d+\.\d+)", line)
            # LightGBM column value
            for n in nums:
                if abs(float(n) - expected) < 0.001:
                    return
    # Fallback: just check the value appears anywhere in the table
    nums = re.findall(r"(\d+\.\d+)", content)
    matches = [n for n in nums if abs(float(n) - expected) < 0.001]
    assert matches, f"LightGBM val PR-AUC {expected:.4f} not found in table_04"


def test_07_t6_castellano_fnr():
    """T6 spot-check: castellano FNR matches source within 0.001."""
    with open(EXPORTS / "fairness_metrics.json") as f:
        data = json.load(f)
    expected = data["dimensions"]["language"]["groups"]["castellano"]["fnr"]

    content = (TABLES / "table_06_fairness_language.tex").read_text()
    nums = re.findall(r"(\d+\.\d+)", content)
    matches = [n for n in nums if abs(float(n) - expected) < 0.001]
    assert matches, f"Castellano FNR {expected:.3f} not found in table_06"


def test_08_t8_top_shap_feature():
    """T8 spot-check: top SHAP feature is 'age' and value matches within 0.001."""
    with open(EXPORTS / "shap_values.json") as f:
        data = json.load(f)
    gi = data["global_importance"]
    top_feature = max(gi, key=gi.get)
    assert top_feature == "age", f"Top SHAP feature is {top_feature}, expected age"

    expected = gi["age"]
    content = (TABLES / "table_08_shap.tex").read_text()
    # Check age appears as rank 1 and value matches
    nums = re.findall(r"(\d+\.\d+)", content)
    matches = [n for n in nums if abs(float(n) - expected) < 0.001]
    assert matches, f"Age SHAP value {expected:.4f} not found in table_08"


def test_09_t7_other_indigenous_urban_fnr():
    """T7 spot-check: other_indigenous_urban FNR > 0.7."""
    with open(EXPORTS / "fairness_metrics.json") as f:
        data = json.load(f)
    fnr = data["intersections"]["language_x_rural"]["groups"][
        "other_indigenous_urban"
    ]["fnr"]
    assert fnr > 0.7, f"other_indigenous_urban FNR={fnr}, expected > 0.7"

    content = (TABLES / "table_07_intersection.tex").read_text()
    nums = re.findall(r"(\d+\.\d+)", content)
    matches = [n for n in nums if abs(float(n) - fnr) < 0.01]
    assert matches, f"FNR {fnr:.3f} not found in table_07"


def test_10_figure_references_have_files():
    """All figure references in main.tex have corresponding files."""
    content = (PAPER / "main.tex").read_text()
    refs = re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", content)
    for ref in refs:
        fig_path = PAPER / ref
        assert fig_path.exists(), f"Referenced figure missing: {ref}"


def test_11_generate_tables_runs():
    """scripts/generate_tables.py runs without errors."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/generate_tables.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"


def test_12_idempotent_generation():
    """Re-running generate_tables.py produces identical output."""
    # Read current tables
    before = {}
    for name in TABLE_FILES:
        before[name] = (TABLES / name).read_text()

    # Re-run
    subprocess.run(
        ["uv", "run", "python", "scripts/generate_tables.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Compare
    for name in TABLE_FILES:
        after = (TABLES / name).read_text()
        assert after == before[name], f"{name} changed on re-run (not idempotent)"
