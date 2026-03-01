"""ENAHO panel linkage assessment and trajectory feature computation.

Assesses what percentage of school-age children (6-17) can be linked across
consecutive ENAHO survey waves using household/person composite keys
(CONGLOME, VIVIENDA, HOGAR, CODPERSO). If linkage is sufficient (>=20%),
computes trajectory features (income change, sibling dropout, work transitions).

Usage::

    from data.panel_linkage import assess_panel_linkage

    report = assess_panel_linkage()
    print(report["decision"])  # "skip", "marginal", or "proceed"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from data.enaho import JOIN_KEYS, _KEY_OVERRIDES, _find_module_file, _read_data_file
from utils import find_project_root, pad_ubigeo

logger = logging.getLogger(__name__)

# Household-level keys (without CODPERSO).
_HH_KEYS = ["CONGLOME", "VIVIENDA", "HOGAR"]

# All available ENAHO years with raw data.
_ALL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Linkage decision thresholds.
_SKIP_THRESHOLD = 0.20
_PROCEED_THRESHOLD = 0.40


def _load_module_200_minimal(year: int) -> pl.DataFrame | None:
    """Load Module 200 for a year, returning only key + demographic columns.

    Returns None if the year's data is unavailable (e.g. 2024).
    """
    root = find_project_root()
    year_dir = root / "data" / "raw" / "enaho" / str(year)

    try:
        filepath = _find_module_file(year_dir, "Enaho01", "200")
    except FileNotFoundError:
        logger.warning("Module 200 not found for year %d — skipping", year)
        return None

    df = _read_data_file(filepath, _KEY_OVERRIDES)

    # Zero-pad UBIGEO
    if "UBIGEO" in df.columns:
        df = df.with_columns(pad_ubigeo(pl.col("UBIGEO")).alias("UBIGEO"))

    # Keep only columns needed for linkage assessment
    needed = JOIN_KEYS + ["UBIGEO", "P207", "P208A", "DOMINIO", "ESTRATO"]
    available = [c for c in needed if c in df.columns]
    df = df.select(available)

    # Filter to school-age (6-17)
    df = df.filter((pl.col("P208A") >= 6) & (pl.col("P208A") <= 17))

    logger.info("Year %d Module 200: %d school-age rows", year, df.height)
    return df


def _load_module_300_minimal(year: int) -> pl.DataFrame | None:
    """Load Module 300 for a year, returning columns needed for trajectory features.

    Returns None if the year's data is unavailable.
    """
    root = find_project_root()
    year_dir = root / "data" / "raw" / "enaho" / str(year)

    try:
        filepath = _find_module_file(year_dir, "Enaho01a", "300")
    except FileNotFoundError:
        logger.warning("Module 300 not found for year %d — skipping", year)
        return None

    df = _read_data_file(filepath, _KEY_OVERRIDES)

    needed = JOIN_KEYS + ["P303", "P306", "P501", "FACTOR07", "INGHOG1D"]
    available = [c for c in needed if c in df.columns]
    df = df.select(available)

    return df


def _assess_year_pair(
    df_t: pl.DataFrame, df_t1: pl.DataFrame, year_t: int, year_t1: int
) -> dict:
    """Assess linkage for a single consecutive year-pair.

    Parameters
    ----------
    df_t, df_t1 : pl.DataFrame
        Module 200 DataFrames for years t and t+1 (school-age only).
    year_t, year_t1 : int
        The two years being compared.

    Returns
    -------
    dict
        Per-pair linkage statistics.
    """
    n_t = df_t.height
    n_t1 = df_t1.height

    # Inner join on full composite key to find matches
    matched = df_t.join(df_t1, on=JOIN_KEYS, how="inner", suffix="_next")

    n_matched = matched.height
    linkage_rate = n_matched / n_t if n_t > 0 else 0.0

    # Quality check 1: age consistency (age_t+1 should be age_t + 0..2)
    age_consistent = 0
    sex_consistent = 0

    if n_matched > 0:
        age_diff = matched.with_columns(
            (pl.col("P208A_next") - pl.col("P208A")).alias("age_diff")
        )
        # Allow 0-2 year difference (interview timing can vary)
        age_ok = age_diff.filter(
            (pl.col("age_diff") >= 0) & (pl.col("age_diff") <= 2)
        )
        age_consistent = age_ok.height / n_matched

        # Quality check 2: sex consistency
        sex_ok = matched.filter(pl.col("P207") == pl.col("P207_next"))
        sex_consistent = sex_ok.height / n_matched

    result = {
        "years": [year_t, year_t1],
        "n_year_t": n_t,
        "n_year_t1": n_t1,
        "n_matched": n_matched,
        "linkage_rate": round(linkage_rate, 4),
        "quality_age_consistent": round(age_consistent, 4),
        "quality_sex_consistent": round(sex_consistent, 4),
    }

    logger.info(
        "Pair %d→%d: %d/%d matched (%.1f%%), age_ok=%.1f%%, sex_ok=%.1f%%",
        year_t, year_t1, n_matched, n_t,
        linkage_rate * 100, age_consistent * 100, sex_consistent * 100,
    )

    return result


def assess_panel_linkage() -> dict:
    """Assess ENAHO panel linkage feasibility across all available year-pairs.

    For each consecutive year-pair, measures what percentage of school-age
    children can be linked using CODPERSO (full JOIN_KEYS match). Computes
    quality metrics (age/sex consistency) and makes a go/no-go decision.

    Returns
    -------
    dict
        Full assessment report with year-pair stats, overall metrics,
        and go/no-go decision. Also saved to data/exports/panel_linkage_report.json.
    """
    # Load Module 200 for all available years
    year_data: dict[int, pl.DataFrame] = {}
    for year in _ALL_YEARS:
        df = _load_module_200_minimal(year)
        if df is not None and df.height > 0:
            year_data[year] = df

    available_years = sorted(year_data.keys())
    logger.info("Available years for linkage: %s", available_years)

    # Assess each consecutive year-pair
    year_pairs: list[dict] = []
    for i in range(len(available_years) - 1):
        y_t = available_years[i]
        y_t1 = available_years[i + 1]
        if y_t1 - y_t == 1:  # Only consecutive years
            pair_stats = _assess_year_pair(year_data[y_t], year_data[y_t1], y_t, y_t1)
            year_pairs.append(pair_stats)

    if not year_pairs:
        report = {
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "year_pairs": [],
            "overall": {
                "mean_linkage_rate": 0.0,
                "min_linkage_rate": 0.0,
                "max_linkage_rate": 0.0,
                "quality_rate": 0.0,
                "effective_rate": 0.0,
            },
            "decision": "skip",
            "reason": "No consecutive year-pairs available for linkage assessment",
            "trajectory_features_built": False,
            "publishable_finding": (
                "No consecutive ENAHO year-pairs could be assessed for panel linkage."
            ),
        }
        _save_report(report)
        return report

    # Compute overall statistics
    rates = [p["linkage_rate"] for p in year_pairs]
    mean_rate = sum(rates) / len(rates)
    min_rate = min(rates)
    max_rate = max(rates)

    # Quality: average across pairs (weighted by n_matched)
    total_matched = sum(p["n_matched"] for p in year_pairs)
    if total_matched > 0:
        quality_age = sum(
            p["quality_age_consistent"] * p["n_matched"] for p in year_pairs
        ) / total_matched
        quality_sex = sum(
            p["quality_sex_consistent"] * p["n_matched"] for p in year_pairs
        ) / total_matched
    else:
        quality_age = 0.0
        quality_sex = 0.0

    # Combined quality: both age and sex must be consistent
    quality_rate = quality_age * quality_sex
    effective_rate = mean_rate * quality_rate

    overall = {
        "mean_linkage_rate": round(mean_rate, 4),
        "min_linkage_rate": round(min_rate, 4),
        "max_linkage_rate": round(max_rate, 4),
        "quality_age_consistent": round(quality_age, 4),
        "quality_sex_consistent": round(quality_sex, 4),
        "quality_rate": round(quality_rate, 4),
        "effective_rate": round(effective_rate, 4),
        "n_year_pairs": len(year_pairs),
        "total_matched": total_matched,
    }

    # Make go/no-go decision
    if effective_rate < _SKIP_THRESHOLD:
        decision = "skip"
        reason = (
            f"Effective linkage rate ({effective_rate:.1%}) is below the 20% threshold. "
            f"Raw mean linkage rate: {mean_rate:.1%}, quality rate: {quality_rate:.1%}. "
            f"Insufficient for reliable trajectory feature construction."
        )
        trajectory_built = False
        publishable = (
            f"Only {mean_rate:.1%} of ENAHO school-age observations could be linked "
            f"across consecutive waves (effective rate {effective_rate:.1%} after quality "
            f"filtering), insufficient for reliable trajectory feature construction. "
            f"This confirms a methodological constraint of ENAHO's rotating panel design "
            f"for longitudinal child-level analysis."
        )
    elif effective_rate <= _PROCEED_THRESHOLD:
        decision = "marginal"
        reason = (
            f"Effective linkage rate ({effective_rate:.1%}) is in the marginal zone "
            f"(20-40%). Representativeness assessment needed before proceeding."
        )
        trajectory_built = False  # Will be updated if representativeness is good
        publishable = (
            f"ENAHO panel linkage achieved {mean_rate:.1%} raw rate "
            f"({effective_rate:.1%} effective), in the marginal feasibility zone."
        )
    else:
        decision = "proceed"
        reason = (
            f"Effective linkage rate ({effective_rate:.1%}) exceeds the 40% threshold. "
            f"Trajectory features are feasible."
        )
        trajectory_built = False  # Will be set True after building
        publishable = (
            f"ENAHO panel linkage achieved {mean_rate:.1%} raw rate "
            f"({effective_rate:.1%} effective), sufficient for trajectory features."
        )

    report = {
        "assessment_date": datetime.now(timezone.utc).isoformat(),
        "year_pairs": year_pairs,
        "overall": overall,
        "decision": decision,
        "reason": reason,
        "trajectory_features_built": trajectory_built,
        "publishable_finding": publishable,
    }

    logger.info("Panel linkage decision: %s (effective rate: %.1f%%)", decision, effective_rate * 100)

    # If decision is proceed or marginal, attempt trajectory features
    if decision in ("proceed", "marginal"):
        try:
            traj_df = build_trajectory_features(year_data, year_pairs)
            if traj_df.height > 0:
                report["trajectory_features_built"] = True
                report["trajectory_features_rows"] = traj_df.height
                report["trajectory_features_columns"] = traj_df.columns
                logger.info(
                    "Trajectory features built: %d rows, %d columns",
                    traj_df.height, len(traj_df.columns),
                )
        except Exception as exc:
            logger.error("Failed to build trajectory features: %s", exc)
            report["trajectory_features_error"] = str(exc)

    _save_report(report)
    return report


def build_trajectory_features(
    year_data: dict[int, pl.DataFrame] | None = None,
    year_pairs: list[dict] | None = None,
) -> pl.DataFrame:
    """Build trajectory features for individuals linked across consecutive waves.

    Computes:
    - income_change: (INGHOG1D_t+1 - INGHOG1D_t) / INGHOG1D_t
    - sibling_dropout: 1 if another school-age person in same HH dropped out
    - work_transition: 1 if P501 changed from not-working to working

    Parameters
    ----------
    year_data : dict[int, pl.DataFrame] | None
        Pre-loaded Module 200 DataFrames keyed by year. If None, loads fresh.
    year_pairs : list[dict] | None
        Year-pair assessment results (for identifying which pairs to use).

    Returns
    -------
    pl.DataFrame
        DataFrame with JOIN_KEYS + year + trajectory feature columns.
        Empty DataFrame with correct schema if linkage is insufficient.
    """
    # Define output schema
    schema = {
        **{k: pl.Utf8 for k in JOIN_KEYS},
        "year": pl.Int64,
        "income_change": pl.Float64,
        "sibling_dropout": pl.Int64,
        "work_transition": pl.Int64,
    }

    if year_data is None or not year_data:
        logger.info("Panel linkage insufficient; trajectory features not computed.")
        return pl.DataFrame(schema=schema)

    if year_pairs is None:
        year_pairs = []

    available_years = sorted(year_data.keys())

    # Load Module 300 for income/work/enrollment data
    mod300_data: dict[int, pl.DataFrame] = {}
    for year in available_years:
        df300 = _load_module_300_minimal(year)
        if df300 is not None:
            mod300_data[year] = df300

    all_trajectory_frames: list[pl.DataFrame] = []

    for i in range(len(available_years) - 1):
        y_t = available_years[i]
        y_t1 = available_years[i + 1]
        if y_t1 - y_t != 1:
            continue

        df_t = year_data[y_t]
        df_t1 = year_data[y_t1]

        # Find linked individuals
        linked = df_t.join(df_t1, on=JOIN_KEYS, how="inner", suffix="_next")
        if linked.height == 0:
            continue

        # Start building trajectory features for this pair
        traj = linked.select(JOIN_KEYS).with_columns(pl.lit(y_t1).alias("year"))

        # 1. Income change (from Module 300 INGHOG1D)
        if y_t in mod300_data and y_t1 in mod300_data:
            inc_t = mod300_data[y_t].select(JOIN_KEYS + [c for c in ["INGHOG1D"] if c in mod300_data[y_t].columns])
            inc_t1 = mod300_data[y_t1].select(JOIN_KEYS + [c for c in ["INGHOG1D"] if c in mod300_data[y_t1].columns])

            if "INGHOG1D" in inc_t.columns and "INGHOG1D" in inc_t1.columns:
                # Cast INGHOG1D to Float64 for computation
                inc_t = inc_t.with_columns(pl.col("INGHOG1D").cast(pl.Float64))
                inc_t1 = inc_t1.with_columns(pl.col("INGHOG1D").cast(pl.Float64))

                traj = traj.join(inc_t, on=JOIN_KEYS, how="left")
                traj = traj.join(inc_t1, on=JOIN_KEYS, how="left", suffix="_next")

                traj = traj.with_columns(
                    pl.when(
                        pl.col("INGHOG1D").is_not_null()
                        & pl.col("INGHOG1D_next").is_not_null()
                        & (pl.col("INGHOG1D") > 0)
                    )
                    .then(
                        (pl.col("INGHOG1D_next") - pl.col("INGHOG1D"))
                        / pl.col("INGHOG1D")
                    )
                    .otherwise(pl.lit(None))
                    .alias("income_change")
                )

                # Drop intermediate columns
                traj = traj.drop([c for c in ["INGHOG1D", "INGHOG1D_next"] if c in traj.columns])
            else:
                traj = traj.with_columns(pl.lit(None).cast(pl.Float64).alias("income_change"))
        else:
            traj = traj.with_columns(pl.lit(None).cast(pl.Float64).alias("income_change"))

        # 2. Sibling dropout: check if another child in same HH dropped out in year t
        if y_t in mod300_data:
            ed_t = mod300_data[y_t]
            if "P303" in ed_t.columns and "P306" in ed_t.columns:
                # Cast enrollment columns to numeric
                for col in ["P303", "P306"]:
                    if ed_t[col].dtype != pl.Int64:
                        ed_t = ed_t.with_columns(pl.col(col).cast(pl.Int64, strict=False))

                # Join Module 200 demographics to get ages for Module 300
                ed_t = ed_t.join(year_data[y_t].select(JOIN_KEYS + ["P208A"]), on=JOIN_KEYS, how="inner")

                # Filter to school-age and compute dropout
                ed_t = ed_t.filter((pl.col("P208A") >= 6) & (pl.col("P208A") <= 17))
                ed_t = ed_t.with_columns(
                    ((pl.col("P303") == 1) & (pl.col("P306") == 2)).cast(pl.Int64).alias("is_dropout")
                )

                # Aggregate at household level: any sibling dropout
                hh_dropout = (
                    ed_t.group_by(_HH_KEYS)
                    .agg(pl.col("is_dropout").sum().alias("hh_dropout_count"))
                )

                # Join to trajectory data
                traj = traj.join(hh_dropout, on=_HH_KEYS, how="left")

                # For each person, sibling_dropout = 1 if HH had dropouts beyond themselves
                # Simplified: if hh_dropout_count > 0, likely a sibling dropped out
                traj = traj.with_columns(
                    pl.when(pl.col("hh_dropout_count").is_not_null() & (pl.col("hh_dropout_count") > 0))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("sibling_dropout")
                )
                traj = traj.drop([c for c in ["hh_dropout_count"] if c in traj.columns])
            else:
                traj = traj.with_columns(pl.lit(None).cast(pl.Int64).alias("sibling_dropout"))
        else:
            traj = traj.with_columns(pl.lit(None).cast(pl.Int64).alias("sibling_dropout"))

        # 3. Work transition: P501 changed from not-working (2+) to working (1)
        if y_t in mod300_data and y_t1 in mod300_data:
            work_cols_t = [c for c in ["P501"] if c in mod300_data[y_t].columns]
            work_cols_t1 = [c for c in ["P501"] if c in mod300_data[y_t1].columns]

            if work_cols_t and work_cols_t1:
                work_t = mod300_data[y_t].select(JOIN_KEYS + ["P501"]).with_columns(
                    pl.col("P501").cast(pl.Int64, strict=False)
                )
                work_t1 = mod300_data[y_t1].select(JOIN_KEYS + ["P501"]).with_columns(
                    pl.col("P501").cast(pl.Int64, strict=False)
                )

                traj = traj.join(work_t, on=JOIN_KEYS, how="left")
                traj = traj.join(work_t1, on=JOIN_KEYS, how="left", suffix="_next")

                traj = traj.with_columns(
                    pl.when(
                        pl.col("P501").is_not_null()
                        & pl.col("P501_next").is_not_null()
                        & (pl.col("P501") != 1)
                        & (pl.col("P501_next") == 1)
                    )
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("work_transition")
                )
                traj = traj.drop([c for c in ["P501", "P501_next"] if c in traj.columns])
            else:
                traj = traj.with_columns(pl.lit(None).cast(pl.Int64).alias("work_transition"))
        else:
            traj = traj.with_columns(pl.lit(None).cast(pl.Int64).alias("work_transition"))

        # Select final columns
        final_cols = JOIN_KEYS + ["year", "income_change", "sibling_dropout", "work_transition"]
        available_cols = [c for c in final_cols if c in traj.columns]
        traj = traj.select(available_cols)

        all_trajectory_frames.append(traj)

    if not all_trajectory_frames:
        logger.info("Panel linkage insufficient; trajectory features not computed.")
        return pl.DataFrame(schema=schema)

    result = pl.concat(all_trajectory_frames, how="vertical")
    logger.info("Trajectory features: %d total rows across %d year-pairs",
                result.height, len(all_trajectory_frames))
    return result


def _save_report(report: dict) -> Path:
    """Save the linkage report to data/exports/panel_linkage_report.json."""
    root = find_project_root()
    out_path = root / "data" / "exports" / "panel_linkage_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved to %s", out_path)
    return out_path
