"""Descriptive statistics pipeline for the Alerta Escuela Equity Audit.

Computes survey-weighted dropout rates across 6 fairness dimensions (language,
sex, rural/urban, region, poverty, temporal), generates 7 matplotlib
visualizations, and exports ``descriptive_tables.json`` matching the M4
scrollytelling schema.

Usage::

    uv run python src/data/descriptive.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
from statsmodels.stats.weightstats import DescrStatsW  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import find_project_root  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (per research recommendation)
# ---------------------------------------------------------------------------

PALETTE = {
    "castellano": "#1f77b4",  # Blue (reference group)
    "quechua": "#2ca02c",  # Green
    "aimara": "#9467bd",  # Purple
    "awajun": "#d62728",  # Red (highlight equity gap)
    "ashaninka": "#ff7f0e",  # Orange
    "other_indigenous": "#8c564b",  # Brown
    "foreign": "#7f7f7f",  # Gray
}

# Ordered for consistent rendering (highest expected rate first)
LANGUAGE_ORDER = [
    "awajun",
    "ashaninka",
    "other_indigenous",
    "quechua",
    "aimara",
    "castellano",
    "foreign",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _assign_language_group(df: pl.DataFrame) -> pl.DataFrame:
    """Add a ``language_group`` string column based on p300a_original.

    Mapping:
        4  -> castellano
        1  -> quechua
        2  -> aimara
        11 -> awajun
        10 -> ashaninka
        6, 7 -> foreign
        p300a_harmonized == 3 AND p300a_original not in (10, 11) -> other_indigenous
        everything else -> other_indigenous (fallback)
    """
    expr = (
        pl.when(pl.col("p300a_original") == 4)
        .then(pl.lit("castellano"))
        .when(pl.col("p300a_original") == 1)
        .then(pl.lit("quechua"))
        .when(pl.col("p300a_original") == 2)
        .then(pl.lit("aimara"))
        .when(pl.col("p300a_original") == 11)
        .then(pl.lit("awajun"))
        .when(pl.col("p300a_original") == 10)
        .then(pl.lit("ashaninka"))
        .when(pl.col("p300a_original").is_in([6, 7]))
        .then(pl.lit("foreign"))
        .when(
            (pl.col("p300a_harmonized") == 3)
            & (~pl.col("p300a_original").is_in([10, 11]))
        )
        .then(pl.lit("other_indigenous"))
        .otherwise(pl.lit("other_indigenous"))
    )
    return df.with_columns(expr.alias("language_group"))


def _weighted_rate_with_ci(
    dropout_array: np.ndarray, weight_array: np.ndarray
) -> dict:
    """Compute survey-weighted dropout rate with 95% confidence interval.

    Parameters
    ----------
    dropout_array : np.ndarray
        Binary dropout indicator (0/1).
    weight_array : np.ndarray
        FACTOR07 survey weights.

    Returns
    -------
    dict
        Keys: weighted_rate, lower_ci, upper_ci, n_unweighted, n_weighted.
    """
    n = len(dropout_array)
    if n < 2:
        logger.warning("Group has n=%d < 2; returning rate=0, ci=[0,0]", n)
        return {
            "weighted_rate": 0.0,
            "lower_ci": 0.0,
            "upper_ci": 0.0,
            "n_unweighted": n,
            "n_weighted": round(float(weight_array.sum()), 0) if n > 0 else 0.0,
        }

    wstats = DescrStatsW(dropout_array, weights=weight_array)
    mean = wstats.mean
    se = wstats.std_mean

    return {
        "weighted_rate": round(float(mean), 4),
        "lower_ci": round(float(max(0, mean - 1.96 * se)), 4),
        "upper_ci": round(float(min(1, mean + 1.96 * se)), 4),
        "n_unweighted": int(n),
        "n_weighted": round(float(weight_array.sum()), 0),
    }


def _get_dropout_and_weights(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract dropout (float) and weight arrays from a DataFrame."""
    d = df["dropout"].cast(pl.Float64).to_numpy()
    w = df["FACTOR07"].to_numpy()
    return d, w


def _print_table(title: str, rows: list[dict]) -> None:
    """Print a formatted table of weighted dropout rates to stdout."""
    print(f"\n--- {title} ---")
    for row in rows:
        group = row["group"]
        rate = row["weighted_rate"]
        lo = row["lower_ci"]
        hi = row["upper_ci"]
        n = row["n_unweighted"]
        flag = "  ***" if group == "awajun" else ""
        print(f"  {group:25s} {rate:.4f}  [{lo:.4f}, {hi:.4f}]  (n={n:,}){flag}")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_descriptive_stats(df: pl.DataFrame) -> dict:
    """Compute survey-weighted descriptive statistics across all breakdowns.

    Parameters
    ----------
    df : pl.DataFrame
        The ``enaho_with_features.parquet`` DataFrame.

    Returns
    -------
    dict
        Full statistics dict matching the M4 JSON export schema.
    """
    logger.info("Computing descriptive statistics on %d rows", df.height)

    # Assign language groups
    df = _assign_language_group(df)

    # -------------------------------------------------------------------
    # DESC-01: Language breakdown
    # -------------------------------------------------------------------
    language_stats: list[dict] = []
    for lang in LANGUAGE_ORDER:
        sub = df.filter(pl.col("language_group") == lang)
        if sub.height == 0:
            continue
        d, w = _get_dropout_and_weights(sub)
        stats = _weighted_rate_with_ci(d, w)
        stats["group"] = lang
        language_stats.append(stats)

    # Sort by rate descending
    language_stats.sort(key=lambda x: x["weighted_rate"], reverse=True)
    _print_table("DROPOUT BY MOTHER TONGUE", language_stats)

    # Binary: indigenous vs castellano
    indigenous = df.filter(pl.col("language_group") != "castellano")
    indigenous = indigenous.filter(pl.col("language_group") != "foreign")
    d_ind, w_ind = _get_dropout_and_weights(indigenous)
    ind_stats = _weighted_rate_with_ci(d_ind, w_ind)
    ind_stats["group"] = "indigenous"

    cast_sub = df.filter(pl.col("language_group") == "castellano")
    d_cast, w_cast = _get_dropout_and_weights(cast_sub)
    cast_stats = _weighted_rate_with_ci(d_cast, w_cast)
    cast_stats["group"] = "castellano"

    language_binary = [ind_stats, cast_stats]
    _print_table("INDIGENOUS vs CASTELLANO", language_binary)

    # -------------------------------------------------------------------
    # DESC-02: Sex x Education Level
    # -------------------------------------------------------------------
    sex_stats: list[dict] = []
    for sex_val, sex_name in [(1.0, "male"), (2.0, "female")]:
        sub = df.filter(pl.col("P207") == sex_val)
        d, w = _get_dropout_and_weights(sub)
        stats = _weighted_rate_with_ci(d, w)
        stats["group"] = sex_name
        sex_stats.append(stats)
    _print_table("DROPOUT BY SEX", sex_stats)

    sex_x_level: list[dict] = []
    for sex_val, sex_name in [(1.0, "male"), (2.0, "female")]:
        for level_name, age_lo, age_hi in [("primaria", 6, 11), ("secundaria", 12, 17)]:
            sub = df.filter(
                (pl.col("P207") == sex_val)
                & (pl.col("age") >= age_lo)
                & (pl.col("age") <= age_hi)
            )
            d, w = _get_dropout_and_weights(sub)
            stats = _weighted_rate_with_ci(d, w)
            stats["group"] = f"{sex_name}_{level_name}"
            sex_x_level.append(stats)
    _print_table("DROPOUT BY SEX x EDUCATION LEVEL", sex_x_level)

    # -------------------------------------------------------------------
    # DESC-03: Rural/Urban, Region, Poverty Quintile
    # -------------------------------------------------------------------
    # Rural/Urban
    rural_stats: list[dict] = []
    for val, name in [(0, "urban"), (1, "rural")]:
        sub = df.filter(pl.col("rural") == val)
        d, w = _get_dropout_and_weights(sub)
        stats = _weighted_rate_with_ci(d, w)
        stats["group"] = name
        rural_stats.append(stats)
    _print_table("DROPOUT: URBAN vs RURAL", rural_stats)

    # Region
    region_stats: list[dict] = []
    for region in ["costa", "sierra", "selva"]:
        sub = df.filter(pl.col("region_natural") == region)
        d, w = _get_dropout_and_weights(sub)
        stats = _weighted_rate_with_ci(d, w)
        stats["group"] = region
        region_stats.append(stats)
    _print_table("DROPOUT BY REGION", region_stats)

    # Poverty quintile
    poverty_stats: list[dict] = []
    quintile_labels = {
        1: "Q1_least_poor",
        2: "Q2",
        3: "Q3",
        4: "Q4",
        5: "Q5_most_poor",
    }
    for q in range(1, 6):
        sub = df.filter(pl.col("poverty_quintile") == q)
        d, w = _get_dropout_and_weights(sub)
        stats = _weighted_rate_with_ci(d, w)
        stats["group"] = quintile_labels[q]
        poverty_stats.append(stats)
    _print_table("DROPOUT BY POVERTY QUINTILE", poverty_stats)

    # -------------------------------------------------------------------
    # DESC-04: Heatmap Data
    # -------------------------------------------------------------------
    heatmap_warnings: list[str] = []

    def _compute_heatmap(
        df_with_lang: pl.DataFrame,
        row_col: str,
        row_values: list,
        col_col: str,
        col_values: list,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
    ) -> dict:
        """Compute heatmap data: rows x columns grid of weighted rates."""
        r_labels = row_labels if row_labels else [str(v) for v in row_values]
        c_labels = col_labels if col_labels else [str(v) for v in col_values]

        values = []
        ci_lower = []
        ci_upper = []
        n_unweighted = []

        for rv in row_values:
            row_vals = []
            row_lo = []
            row_hi = []
            row_n = []
            for cv in col_values:
                sub = df_with_lang.filter(
                    (pl.col(row_col) == rv) & (pl.col(col_col) == cv)
                )
                if sub.height == 0:
                    row_vals.append(None)
                    row_lo.append(None)
                    row_hi.append(None)
                    row_n.append(0)
                else:
                    d, w = _get_dropout_and_weights(sub)
                    stats = _weighted_rate_with_ci(d, w)
                    row_vals.append(stats["weighted_rate"])
                    row_lo.append(stats["lower_ci"])
                    row_hi.append(stats["upper_ci"])
                    row_n.append(stats["n_unweighted"])
                    if sub.height < 50:
                        heatmap_warnings.append(
                            f"Cell ({rv}, {cv}): n={sub.height} < 50"
                        )
            values.append(row_vals)
            ci_lower.append(row_lo)
            ci_upper.append(row_hi)
            n_unweighted.append(row_n)

        return {
            "rows": r_labels,
            "columns": c_labels,
            "values": values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_unweighted": n_unweighted,
        }

    # Add rural_label column for heatmap
    df = df.with_columns(
        pl.when(pl.col("rural") == 1)
        .then(pl.lit("rural"))
        .otherwise(pl.lit("urban"))
        .alias("rural_label")
    )

    # Heatmap 1: Language x Rurality
    heatmap_lang_rural = _compute_heatmap(
        df,
        row_col="language_group",
        row_values=LANGUAGE_ORDER,
        col_col="rural_label",
        col_values=["urban", "rural"],
    )
    print("\n--- HEATMAP: LANGUAGE x RURALITY ---")
    for i, lang in enumerate(heatmap_lang_rural["rows"]):
        vals = heatmap_lang_rural["values"][i]
        ns = heatmap_lang_rural["n_unweighted"][i]
        parts = []
        for j, col in enumerate(heatmap_lang_rural["columns"]):
            v = vals[j] if vals[j] is not None else "N/A"
            n = ns[j]
            flag = "*" if n < 50 else ""
            parts.append(f"{col}={v}{flag}(n={n})")
        print(f"  {lang:25s} {', '.join(parts)}")

    # Heatmap 2: Language x Poverty Quintile
    heatmap_lang_poverty = _compute_heatmap(
        df,
        row_col="language_group",
        row_values=LANGUAGE_ORDER,
        col_col="poverty_quintile",
        col_values=[1, 2, 3, 4, 5],
        col_labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    # Heatmap 3: Language x Region
    heatmap_lang_region = _compute_heatmap(
        df,
        row_col="language_group",
        row_values=LANGUAGE_ORDER,
        col_col="region_natural",
        col_values=["costa", "sierra", "selva"],
    )

    if heatmap_warnings:
        print(f"\n  Heatmap warnings (n < 50): {len(heatmap_warnings)} cells")
        for w in heatmap_warnings[:10]:
            print(f"    {w}")

    # -------------------------------------------------------------------
    # DESC-05: Choropleth prep
    # -------------------------------------------------------------------
    choropleth_data: list[dict] = []
    for ubigeo_row in (
        df.group_by("UBIGEO", "department")
        .agg(
            [
                pl.col("dropout").cast(pl.Float64).alias("dropout_vals"),
                pl.col("FACTOR07").alias("weight_vals"),
                pl.len().alias("n_students"),
            ]
        )
        .iter_rows(named=True)
    ):
        d = np.array(ubigeo_row["dropout_vals"])
        w = np.array(ubigeo_row["weight_vals"])
        if len(d) < 2:
            rate = float(np.average(d, weights=w)) if len(d) == 1 else 0.0
            choropleth_data.append(
                {
                    "ubigeo": ubigeo_row["UBIGEO"],
                    "department": ubigeo_row["department"],
                    "weighted_rate": round(rate, 4),
                    "n_students": ubigeo_row["n_students"],
                }
            )
        else:
            stats = _weighted_rate_with_ci(d, w)
            choropleth_data.append(
                {
                    "ubigeo": ubigeo_row["UBIGEO"],
                    "department": ubigeo_row["department"],
                    "weighted_rate": stats["weighted_rate"],
                    "n_students": ubigeo_row["n_students"],
                }
            )

    logger.info("Choropleth prep: %d districts", len(choropleth_data))
    print(f"\n--- CHOROPLETH PREP ---")
    print(f"  Districts: {len(choropleth_data)}")

    # -------------------------------------------------------------------
    # DESC-06: Temporal trends
    # -------------------------------------------------------------------
    years = sorted(df["year"].unique().to_list())
    overall_rates: list[float] = []
    for yr in years:
        sub = df.filter(pl.col("year") == yr)
        d, w = _get_dropout_and_weights(sub)
        rate = round(float(np.average(d, weights=w)), 4)
        overall_rates.append(rate)

    by_language: dict[str, list[float | None]] = {}
    for lang in LANGUAGE_ORDER:
        lang_rates: list[float | None] = []
        for yr in years:
            sub = df.filter(
                (pl.col("language_group") == lang) & (pl.col("year") == yr)
            )
            if sub.height < 2:
                lang_rates.append(None)
            else:
                d, w = _get_dropout_and_weights(sub)
                rate = round(float(np.average(d, weights=w)), 4)
                lang_rates.append(rate)
        by_language[lang] = lang_rates

    temporal_stats = {
        "years": years,
        "overall_rate": overall_rates,
        "by_language": by_language,
    }

    print(f"\n--- TEMPORAL TRENDS ---")
    print(f"  Years: {years}")
    print(f"  Overall: {overall_rates}")
    print(f"\n  Awajun by year:")
    for i, yr in enumerate(years):
        awajun_rate = by_language.get("awajun", [None] * len(years))[i]
        if awajun_rate is not None:
            print(f"    {yr}: {awajun_rate:.4f}")
        else:
            print(f"    {yr}: N/A (no data)")

    # -------------------------------------------------------------------
    # Build export dict
    # -------------------------------------------------------------------
    export = {
        "_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_rows": df.height,
            "years_covered": years,
            "pipeline_version": "0.4.0",
        },
        "language": language_stats,
        "language_binary": language_binary,
        "sex": sex_stats,
        "sex_x_level": sex_x_level,
        "rural": rural_stats,
        "region": region_stats,
        "poverty": poverty_stats,
        "heatmap_language_x_rural": heatmap_lang_rural,
        "heatmap_language_x_poverty": heatmap_lang_poverty,
        "heatmap_language_x_region": heatmap_lang_region,
        "choropleth_prep": choropleth_data,
        "temporal": temporal_stats,
    }

    logger.info("Descriptive statistics complete")
    return export


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def generate_visualizations(stats: dict, output_dir: Path) -> list[Path]:
    """Generate 7 matplotlib PNG visualizations from the stats dict.

    Parameters
    ----------
    stats : dict
        Output from :func:`compute_descriptive_stats`.
    output_dir : Path
        Directory to save PNG files (created if missing).

    Returns
    -------
    list[Path]
        Paths to all generated PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # ---- Viz 1: Language Bars ----
    fig, ax = plt.subplots(figsize=(10, 6))
    lang_data = stats["language"]
    # Sort by rate ascending for horizontal bar (highest at top)
    lang_data_sorted = sorted(lang_data, key=lambda x: x["weighted_rate"])
    groups = [d["group"] for d in lang_data_sorted]
    rates = [d["weighted_rate"] for d in lang_data_sorted]
    colors = [PALETTE.get(g, "#333333") for g in groups]
    ci_lo = [d["weighted_rate"] - d["lower_ci"] for d in lang_data_sorted]
    ci_hi = [d["upper_ci"] - d["weighted_rate"] for d in lang_data_sorted]

    ax.barh(
        groups,
        rates,
        color=colors,
        xerr=[ci_lo, ci_hi],
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    # Overall rate reference line
    overall_d = np.concatenate(
        [np.full(d["n_unweighted"], d["weighted_rate"]) for d in lang_data]
    )
    overall_rate = np.mean(rates)
    # Use the actual weighted overall from temporal if available
    if stats.get("temporal", {}).get("overall_rate"):
        # Use average of all years as overall
        overall_rate = np.mean(stats["temporal"]["overall_rate"])
    ax.axvline(
        x=overall_rate,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"Overall rate ({overall_rate:.3f})",
    )
    ax.set_xlabel("Weighted Dropout Rate", fontsize=12)
    ax.set_title(
        "Survey-Weighted Dropout Rate by Mother Tongue (ENAHO 2018-2023)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    p = output_dir / "01_language_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # ---- Viz 2: Sex x Education Level ----
    fig, ax = plt.subplots(figsize=(10, 6))
    sxl = {d["group"]: d for d in stats["sex_x_level"]}
    levels = ["primaria", "secundaria"]
    x = np.arange(len(levels))
    width = 0.35

    male_rates = [sxl[f"male_{l}"]["weighted_rate"] for l in levels]
    female_rates = [sxl[f"female_{l}"]["weighted_rate"] for l in levels]
    male_err = [
        [sxl[f"male_{l}"]["weighted_rate"] - sxl[f"male_{l}"]["lower_ci"] for l in levels],
        [sxl[f"male_{l}"]["upper_ci"] - sxl[f"male_{l}"]["weighted_rate"] for l in levels],
    ]
    female_err = [
        [sxl[f"female_{l}"]["weighted_rate"] - sxl[f"female_{l}"]["lower_ci"] for l in levels],
        [sxl[f"female_{l}"]["upper_ci"] - sxl[f"female_{l}"]["weighted_rate"] for l in levels],
    ]

    bars_m = ax.bar(
        x - width / 2,
        male_rates,
        width,
        label="Male",
        color="#4393c3",
        yerr=male_err,
        capsize=4,
    )
    bars_f = ax.bar(
        x + width / 2,
        female_rates,
        width,
        label="Female",
        color="#d6604d",
        yerr=female_err,
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Primaria (6-11)", "Secundaria (12-17)"], fontsize=12)
    ax.set_ylabel("Weighted Dropout Rate", fontsize=12)
    ax.set_title("Dropout Rate by Sex and Education Level", fontsize=14)
    ax.legend(fontsize=11)

    # Annotate gender flip if male_secundaria > female_secundaria
    if sxl["male_secundaria"]["weighted_rate"] > sxl["female_secundaria"]["weighted_rate"]:
        ax.annotate(
            "Male > Female\n(gender reversal)",
            xy=(1 - width / 2, sxl["male_secundaria"]["weighted_rate"]),
            xytext=(0.5, sxl["male_secundaria"]["weighted_rate"] + 0.02),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
            ha="center",
        )

    ax.tick_params(labelsize=12)
    plt.tight_layout()
    p = output_dir / "02_sex_education_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # ---- Viz 3: Rural/Urban ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ru_data = {d["group"]: d for d in stats["rural"]}
    groups_ru = ["urban", "rural"]
    rates_ru = [ru_data[g]["weighted_rate"] for g in groups_ru]
    err_ru = [
        [ru_data[g]["weighted_rate"] - ru_data[g]["lower_ci"] for g in groups_ru],
        [ru_data[g]["upper_ci"] - ru_data[g]["weighted_rate"] for g in groups_ru],
    ]
    colors_ru = ["#4393c3", "#d6604d"]

    ax.bar(groups_ru, rates_ru, color=colors_ru, yerr=err_ru, capsize=5, width=0.5)
    ax.set_ylabel("Weighted Dropout Rate", fontsize=12)
    ax.set_title("Dropout Rate: Urban vs Rural", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    p = output_dir / "03_rural_urban_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # ---- Viz 4: Region ----
    fig, ax = plt.subplots(figsize=(10, 6))
    reg_data = {d["group"]: d for d in stats["region"]}
    regions = ["costa", "sierra", "selva"]
    rates_reg = [reg_data[r]["weighted_rate"] for r in regions]
    err_reg = [
        [reg_data[r]["weighted_rate"] - reg_data[r]["lower_ci"] for r in regions],
        [reg_data[r]["upper_ci"] - reg_data[r]["weighted_rate"] for r in regions],
    ]
    colors_reg = ["#66c2a5", "#fc8d62", "#8da0cb"]

    ax.bar(regions, rates_reg, color=colors_reg, yerr=err_reg, capsize=5, width=0.5)
    ax.set_ylabel("Weighted Dropout Rate", fontsize=12)
    ax.set_title("Dropout Rate by Natural Region", fontsize=14)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(["Costa", "Sierra", "Selva"], fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    p = output_dir / "04_region_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # ---- Viz 5: Poverty Quintile ----
    fig, ax = plt.subplots(figsize=(10, 6))
    pov_data = stats["poverty"]
    pov_groups = [d["group"] for d in pov_data]
    pov_rates = [d["weighted_rate"] for d in pov_data]
    pov_err = [
        [d["weighted_rate"] - d["lower_ci"] for d in pov_data],
        [d["upper_ci"] - d["weighted_rate"] for d in pov_data],
    ]
    # Gradient from light to dark
    from matplotlib.colors import LinearSegmentedColormap

    n_q = len(pov_groups)
    cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
    pov_colors = [cmap(0.2 + 0.7 * i / (n_q - 1)) for i in range(n_q)]

    ax.bar(
        pov_groups,
        pov_rates,
        color=pov_colors,
        yerr=pov_err,
        capsize=5,
        width=0.6,
    )
    ax.set_ylabel("Weighted Dropout Rate", fontsize=12)
    ax.set_title("Dropout Rate by Poverty Quintile", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    p = output_dir / "05_poverty_quintile_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # ---- Viz 6: Language x Rurality Heatmap ----
    fig, ax = plt.subplots(figsize=(10, 8))
    hm = stats["heatmap_language_x_rural"]
    row_labels = hm["rows"]
    col_labels = hm["columns"]
    vals = np.array(
        [
            [v if v is not None else np.nan for v in row]
            for row in hm["values"]
        ]
    )
    ns = hm["n_unweighted"]

    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([c.title() for c in col_labels], fontsize=12)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = vals[i, j]
            n = ns[i][j]
            if np.isnan(v):
                text = "N/A"
            else:
                text = f"{v:.4f}"
                if n < 50:
                    text += "*"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="black" if v < np.nanmedian(vals) else "white",
            )

    fig.colorbar(im, ax=ax, label="Weighted Dropout Rate", shrink=0.8)
    ax.set_title("Dropout Rate: Language x Rurality Interaction", fontsize=14)
    plt.tight_layout()
    p = output_dir / "06_language_rurality_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # ---- Viz 7: Temporal Trends ----
    fig, ax = plt.subplots(figsize=(10, 6))
    temporal = stats["temporal"]
    years = temporal["years"]

    for lang in LANGUAGE_ORDER:
        lang_rates = temporal["by_language"].get(lang, [])
        if not lang_rates:
            continue
        # Filter out None values for plotting
        valid = [
            (yr, r)
            for yr, r in zip(years, lang_rates)
            if r is not None
        ]
        if not valid:
            continue
        yrs_plot, rates_plot = zip(*valid)
        ax.plot(
            yrs_plot,
            rates_plot,
            marker="o",
            label=lang,
            color=PALETTE.get(lang, "#333333"),
            linewidth=2,
            markersize=5,
        )

    # COVID vertical line
    ax.axvline(
        x=2020,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="COVID-19 (2020)",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Weighted Dropout Rate", fontsize=12)
    ax.set_title(
        "Dropout Rate Trends by Mother Tongue (2018-2023)", fontsize=14
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_xticks(years)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    p = output_dir / "07_temporal_trend_lines.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    logger.info("Generated %d visualizations in %s", len(paths), output_dir)
    return paths


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class _RoundingEncoder(json.JSONEncoder):
    """JSON encoder that rounds floats to 4 decimal places."""

    def default(self, obj: object) -> object:
        if isinstance(obj, float):
            return round(obj, 4)
        if isinstance(obj, np.floating):
            return round(float(obj), 4)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj: object) -> str:
        return super().encode(self._round_floats(obj))

    def _round_floats(self, obj: object) -> object:
        if isinstance(obj, float):
            return round(obj, 4)
        if isinstance(obj, np.floating):
            return round(float(obj), 4)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: self._round_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._round_floats(v) for v in obj]
        return obj


def export_descriptive_json(stats: dict, output_path: Path) -> Path:
    """Serialize the stats dict to JSON.

    Parameters
    ----------
    stats : dict
        Output from :func:`compute_descriptive_stats`.
    output_path : Path
        Destination file path.

    Returns
    -------
    Path
        The output path (same as input for chaining).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, cls=_RoundingEncoder)

    logger.info("Exported descriptive stats to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the descriptive statistics pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = find_project_root()
    parquet_path = root / "data" / "processed" / "enaho_with_features.parquet"
    df = pl.read_parquet(parquet_path)

    print(f"Loaded {df.height} rows, {df.width} columns from {parquet_path.name}")

    stats = compute_descriptive_stats(df)

    figures_dir = root / "data" / "exports" / "figures"
    paths = generate_visualizations(stats, figures_dir)

    json_path = export_descriptive_json(
        stats, root / "data" / "exports" / "descriptive_tables.json"
    )

    print(f"\nJSON exported to: {json_path}")
    print(f"Figures saved to: {figures_dir}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
