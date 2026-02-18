"""Power analysis: survey data ceiling for intersectional fairness auditing."""
import json
import math
from pathlib import Path
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower

ROOT = Path(__file__).resolve().parents[2]


def compute_power_analysis() -> dict:
    """
    Calculate minimum n for detecting FNR = 0.75 vs 0.50 at 80% power.
    Framed as: what's the data ceiling for survey-based intersectional auditing?
    """
    # Parameters
    fnr_observed = 0.75      # Point estimate for urban other-indigenous
    fnr_null = 0.50          # Credibility threshold ("model misses majority")
    alpha = 0.05
    power_target = 0.80

    # Effect size (Cohen's h for proportions)
    effect_size = proportion_effectsize(fnr_observed, fnr_null)

    # Required sample size (one-sample test: is FNR > 0.50?)
    analyzer = NormalIndPower()
    required_n = math.ceil(analyzer.solve_power(
        effect_size=effect_size,
        nobs1=None,
        alpha=alpha,
        power=power_target,
        alternative="larger",  # One-sided: FNR > 0.50
    ))

    # Also compute for gap detection: FNR = 0.75 vs 0.63 (castellano)
    effect_size_gap = proportion_effectsize(0.75, 0.633)
    required_n_gap = math.ceil(analyzer.solve_power(
        effect_size=effect_size_gap,
        nobs1=None,
        alpha=alpha,
        power=power_target,
        alternative="larger",
    ))

    # Translation to ENAHO years
    # Urban other-indigenous per ENAHO year: ~30 students, ~6 dropouts
    students_per_year = 30  # Approximate from 6 years x ~180 total
    dropouts_per_year = 6   # ~20% dropout rate x 30

    # Required n is for the positive class (dropouts) since FNR is conditional on y=1
    years_majority_miss = math.ceil(required_n / dropouts_per_year)
    years_gap_detect = math.ceil(required_n_gap / dropouts_per_year)

    result = {
        "metadata": {
            "purpose": "Minimum sample size for credible intersectional fairness auditing with survey data",
            "framing": "methodological_limitation",
        },
        "majority_miss_test": {
            "description": "Can we confirm FNR > 0.50 (model misses majority)?",
            "fnr_observed": fnr_observed,
            "fnr_null": fnr_null,
            "effect_size_h": round(effect_size, 4),
            "alpha": alpha,
            "power": power_target,
            "required_n_dropouts": required_n,
            "enaho_years_required": years_majority_miss,
            "students_per_year_approx": students_per_year,
            "dropouts_per_year_approx": dropouts_per_year,
        },
        "gap_detection_test": {
            "description": "Can we confirm FNR gap vs castellano (0.75 vs 0.633)?",
            "fnr_target": 0.75,
            "fnr_castellano": 0.633,
            "effect_size_h": round(effect_size_gap, 4),
            "required_n_dropouts": required_n_gap,
            "enaho_years_required": years_gap_detect,
        },
        "conclusion": (
            f"Detecting FNR > 0.50 at 80% power requires {required_n} dropout observations "
            f"(~{years_majority_miss} ENAHO years). Detecting the gap vs castellano requires "
            f"{required_n_gap} (~{years_gap_detect} years). Survey-based intersectional "
            f"fairness auditing cannot produce significant results for groups contributing "
            f"~{dropouts_per_year} positive observations per survey year. "
            f"This demonstrates why administrative data (SIAGIE) must be opened for "
            f"meaningful intersectional auditing."
        ),
    }

    output_path = ROOT / "data/exports/power_analysis.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Majority-miss: need {required_n} dropouts (~{years_majority_miss} ENAHO years)")
    print(f"Gap detection: need {required_n_gap} dropouts (~{years_gap_detect} ENAHO years)")
    return result


if __name__ == "__main__":
    compute_power_analysis()
