"""Threshold sweep analysis: verify FNR rank-order invariance across language groups.

Sweeps classification thresholds from 0.05 to 0.95 and computes weighted FNR
per language group at each threshold, for both v1 (25-feature calibration) and
v2 (31-feature calibration) probability scales.

Outputs:
  - data/exports/threshold_sweep.json
  - paper/tables/table_13_threshold_sweep.tex

Usage::

    uv run python src/fairness/threshold_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import find_project_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Platt scaling parameters (from model_results.json / MEMORY)
PLATT_V1 = {"A": -6.236085, "B": 4.442308}  # 25-feature model
PLATT_V2 = {"A": -8.156711, "B": 5.069181}  # 31-feature model

THRESHOLDS = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95

# Optimal thresholds to include in the sweep
V1_OPTIMAL = 0.167268
V2_OPTIMAL = 0.185024

# Language groups to include (exclude foreign n=43)
INCLUDE_GROUPS = {"castellano", "quechua", "aimara", "other_indigenous"}

HARMONIZED_LANGUAGE_MAP = {
    "lang_castellano": "castellano",
    "lang_quechua": "quechua",
    "lang_aimara": "aimara",
    "lang_other_indigenous": "other_indigenous",
    "lang_foreign": "foreign",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def platt_calibrate(raw_probs: np.ndarray, A: float, B: float) -> np.ndarray:
    """Apply Platt sigmoid calibration: p = 1 / (1 + exp(A*f + B))."""
    return 1.0 / (1.0 + np.exp(A * raw_probs + B))


def weighted_fnr(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """Compute survey-weighted False Negative Rate."""
    positives = y_true == 1
    if positives.sum() == 0:
        return float("nan")
    missed = positives & (y_pred == 0)
    return float(
        np.sum(weights[missed]) / np.sum(weights[positives])
    )


def build_language_groups(merged: pl.DataFrame) -> np.ndarray:
    """Build harmonized language group labels from language dummies."""
    lang_cols = [
        ("lang_castellano", "castellano"),
        ("lang_quechua", "quechua"),
        ("lang_aimara", "aimara"),
        ("lang_other_indigenous", "other_indigenous"),
        ("lang_foreign", "foreign"),
    ]
    labels = []
    for i in range(merged.height):
        found = False
        for col, label in lang_cols:
            if merged[col][i] == 1:
                labels.append(label)
                found = True
                break
        if not found:
            labels.append("unknown")
    return np.array(labels)


def sweep_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    thresholds: list[float],
) -> dict:
    """Sweep thresholds and compute FNR per group at each threshold.

    Returns dict: {threshold_str: {group: {fnr, rank, n_dropouts}}}
    """
    results = {}
    for t in thresholds:
        t_str = f"{t}" if t != round(t, 2) else f"{t:.2f}"
        y_pred = (probs >= t).astype(int)
        group_fnrs = {}
        for g in sorted(INCLUDE_GROUPS):
            mask = groups == g
            fnr = weighted_fnr(y_true[mask], y_pred[mask], weights[mask])
            n_dropouts = int(y_true[mask].sum())
            group_fnrs[g] = {"fnr": round(fnr, 6), "n_dropouts": n_dropouts}

        # Assign ranks (1 = highest FNR = most missed)
        sorted_groups = sorted(
            group_fnrs.items(), key=lambda x: x[1]["fnr"], reverse=True
        )
        for rank, (g, data) in enumerate(sorted_groups, 1):
            data["rank"] = rank

        results[t_str] = group_fnrs
    return results


def check_rank_invariance(
    sweep_results: dict, max_calibrated_prob: float
) -> dict:
    """Check FNR rank-order invariance across thresholds.

    Returns dict with:
      - full_invariant: bool (entire rank order identical at all valid thresholds)
      - castellano_rank1_invariant: bool (castellano has highest FNR at all valid thresholds)
      - castellano_rank1_range: str (threshold range where castellano is rank 1)
      - valid_thresholds: list (thresholds where FNR values are distinguishable)
      - description: str
    """
    rank_orders = []
    valid_thresholds = []
    castellano_rank1 = []

    for t_str, groups in sorted(sweep_results.items(), key=lambda x: float(x[0])):
        t_val = float(t_str)
        # Skip thresholds above max calibrated prob (all FNR = 1.0)
        if t_val > max_calibrated_prob:
            continue
        fnrs = [d["fnr"] for _, d in groups.items()]
        # Skip if all FNR identical (degenerate)
        if len(set(round(f, 4) for f in fnrs)) <= 1:
            continue

        ranked = sorted(groups.items(), key=lambda x: x[1]["rank"])
        order = tuple(g for g, _ in ranked)
        rank_orders.append(order)
        valid_thresholds.append(t_str)
        castellano_rank1.append(groups["castellano"]["rank"] == 1)

    if not rank_orders:
        return {
            "full_invariant": False,
            "castellano_rank1_invariant": False,
            "castellano_rank1_range": "none",
            "valid_thresholds": [],
            "description": "No thresholds with distinguishable FNR values",
        }

    first_order = rank_orders[0]
    full_inv = all(order == first_order for order in rank_orders)
    cast_inv = all(castellano_rank1)

    # Find range where castellano is rank 1
    cast_range = [t for t, r1 in zip(valid_thresholds, castellano_rank1) if r1]
    cast_range_str = (
        f"{cast_range[0]}--{cast_range[-1]}" if cast_range else "none"
    )

    if full_inv:
        desc = (
            f"Full rank order invariant across {valid_thresholds[0]}--"
            f"{valid_thresholds[-1]} ({len(valid_thresholds)} thresholds)"
        )
    elif cast_inv:
        desc = (
            f"Castellano has highest FNR at all {len(valid_thresholds)} "
            f"operationally meaningful thresholds ({valid_thresholds[0]}--"
            f"{valid_thresholds[-1]}); minor reordering among indigenous groups "
            f"at higher thresholds"
        )
    else:
        desc = (
            f"Castellano has highest FNR at {len(cast_range)}/"
            f"{len(valid_thresholds)} thresholds ({cast_range_str}); "
            f"at higher thresholds above {cast_range[-1] if cast_range else '?'}, "
            f"other_indigenous overtakes castellano as probability ceiling "
            f"compresses discrimination range"
        )

    return {
        "full_invariant": full_inv,
        "castellano_rank1_invariant": cast_inv,
        "castellano_rank1_range": cast_range_str,
        "valid_thresholds": valid_thresholds,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def generate_latex_table(
    sweep_v2: dict,
    optimal_threshold: float,
    max_calibrated_prob: float,
) -> str:
    """Generate LaTeX table for appendix (Table 13).

    Shows thresholds from 0.05 to the max calibrated probability range,
    plus the optimal threshold if not already included.
    """
    # Display thresholds: every 0.05 up to ceiling, skipping degenerate range
    # Stop before all groups hit FNR=1.0
    max_display = max_calibrated_prob - 0.02  # Leave margin before ceiling
    display_thresholds = [
        f"{t:.2f}" for t in
        [round(i * 0.05, 2) for i in range(1, 20)]
        if t <= max_display
    ]
    # Format optimal threshold to match sweep key precision
    opt_str = f"{optimal_threshold}"
    # Find the matching key in sweep_v2
    opt_key = None
    for k in sweep_v2:
        if abs(float(k) - optimal_threshold) < 0.001:
            opt_key = k
            break

    # Column order
    groups_order = ["castellano", "quechua", "other_indigenous", "aimara"]
    group_headers = ["Castellano", "Quechua", "Otros Ind.", "Aimara"]

    # Build ordered list of all thresholds to display (regular + optimal)
    all_display = []
    for t_str in display_thresholds:
        if t_str in sweep_v2:
            all_display.append((float(t_str), t_str, False))
    if opt_key and opt_key not in display_thresholds:
        all_display.append((optimal_threshold, opt_key, True))
    all_display.sort(key=lambda x: x[0])

    # Filter out degenerate rows (all FNR >= 0.99)
    filtered_display = []
    for t_val, t_key, is_opt in all_display:
        data = sweep_v2[t_key]
        fnrs = [data[g]["fnr"] for g in groups_order]
        if min(fnrs) >= 0.99 and not is_opt:
            continue
        filtered_display.append((t_val, t_key, is_opt))

    lines = []
    lines.append(r"\begin{tabular}{l" + "S[table-format=1.3]" * len(groups_order) + "}")
    lines.append(r"\toprule")
    header = r"Threshold & " + " & ".join(f"{{{h}}}" for h in group_headers) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for t_val, t_key, is_opt in filtered_display:
        data = sweep_v2[t_key]
        if is_opt:
            prefix = r"\textbf{" + f"{optimal_threshold:.3f}" + r"}\textsuperscript{*}"
        else:
            prefix = t_key

        cells = []
        for g in groups_order:
            fnr = data[g]["fnr"]
            rank = data[g]["rank"]
            cells.append(f"{fnr:.3f}\\textsuperscript{{{rank}}}")
        line = f"{prefix} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(
        r"{\footnotesize Superscripts indicate FNR rank (1\,=\,highest, most missed). "
        r"* = optimal threshold (max weighted F1). "
        f"Thresholds above {max_calibrated_prob:.2f} (max calibrated probability) produce "
        r"FNR\,=\,1.0 for all groups and are omitted. "
        r"Aimara ($n=76$) shows rank instability at extreme thresholds due to small sample.}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    root = find_project_root()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading predictions and features...")

    # Raw (uncalibrated) LightGBM predictions
    raw_pred = pl.read_parquet(root / "data" / "processed" / "predictions_lgbm.parquet")
    raw_test = raw_pred.filter(pl.col("split") == "test_2023")
    assert raw_test.height == 25635

    # Features for language groups
    feat = pl.read_parquet(root / "data" / "processed" / "enaho_with_features.parquet")
    join_keys = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO", "year"]
    meta_cols = [
        "lang_castellano", "lang_quechua", "lang_aimara",
        "lang_other_indigenous", "lang_foreign",
    ]

    merged = raw_test.join(
        feat.select(join_keys + meta_cols), on=join_keys, how="left"
    )
    assert merged.height == raw_test.height, f"Join error: {raw_test.height} -> {merged.height}"

    # Extract arrays
    y_true = merged["dropout"].cast(pl.Int8).to_numpy()
    raw_probs = merged["prob_dropout"].to_numpy()
    weights = merged["FACTOR07"].to_numpy()
    groups = build_language_groups(merged)

    print(f"  Test set: {merged.height:,} rows, {int(y_true.sum()):,} dropouts")

    # -----------------------------------------------------------------------
    # Calibrate with both parameter sets
    # -----------------------------------------------------------------------
    probs_v1 = platt_calibrate(raw_probs, PLATT_V1["A"], PLATT_V1["B"])
    probs_v2 = platt_calibrate(raw_probs, PLATT_V2["A"], PLATT_V2["B"])

    print(f"  v1 calibrated prob range: {probs_v1.min():.4f} - {probs_v1.max():.4f}")
    print(f"  v2 calibrated prob range: {probs_v2.min():.4f} - {probs_v2.max():.4f}")

    # -----------------------------------------------------------------------
    # Sweep thresholds
    # -----------------------------------------------------------------------
    # Include optimal thresholds in the sweep
    all_thresholds_v1 = sorted(set(THRESHOLDS + [V1_OPTIMAL]))
    all_thresholds_v2 = sorted(set(THRESHOLDS + [V2_OPTIMAL]))

    print("\nSweeping thresholds (v1, 25-feature calibration)...")
    sweep_v1 = sweep_thresholds(y_true, probs_v1, weights, groups, all_thresholds_v1)

    print("Sweeping thresholds (v2, 31-feature calibration)...")
    sweep_v2 = sweep_thresholds(y_true, probs_v2, weights, groups, all_thresholds_v2)

    # -----------------------------------------------------------------------
    # Check rank invariance
    # -----------------------------------------------------------------------
    inv_v1 = check_rank_invariance(sweep_v1, max_calibrated_prob=float(probs_v1.max()))
    inv_v2 = check_rank_invariance(sweep_v2, max_calibrated_prob=float(probs_v2.max()))

    print(f"\n  v1: {inv_v1['description']}")
    print(f"  v2: {inv_v2['description']}")

    # Print rank order at a few key thresholds
    for label, sweep in [("v1", sweep_v1), ("v2", sweep_v2)]:
        print(f"\n  === {label} FNR by threshold (selected) ===")
        for t in ["0.05", "0.10", "0.20", "0.30", "0.50", "0.70", "0.90"]:
            if t in sweep:
                fnrs = {g: d["fnr"] for g, d in sweep[t].items()}
                ranked = sorted(fnrs.items(), key=lambda x: x[1], reverse=True)
                order_str = " > ".join(f"{g}({f:.3f})" for g, f in ranked)
                print(f"    t={t}: {order_str}")

    # -----------------------------------------------------------------------
    # Export JSON
    # -----------------------------------------------------------------------
    export = {
        "metadata": {
            "thresholds": THRESHOLDS,
            "models": ["v1_25f", "v2_31f"],
            "platt_v1": PLATT_V1,
            "platt_v2": PLATT_V2,
            "max_calibrated_prob_v1": round(float(probs_v1.max()), 4),
            "max_calibrated_prob_v2": round(float(probs_v2.max()), 4),
            "note": (
                "Both v1 and v2 use the same v2 (31-feature) raw LightGBM probabilities, "
                "calibrated with different Platt parameters. This tests whether the FNR "
                "rank order is invariant to calibration parameterization. Thresholds above "
                "max_calibrated_prob produce degenerate FNR=1.0 for all groups."
            ),
            "n_test": int(merged.height),
            "n_dropouts": int(y_true.sum()),
            "groups_included": sorted(INCLUDE_GROUPS),
            "groups_excluded": ["foreign (n=43)", "unknown"],
        },
        "v1_25f": sweep_v1,
        "v2_31f": sweep_v2,
        "rank_invariant": inv_v2["castellano_rank1_invariant"],
        "castellano_rank1_range_v1": inv_v1["castellano_rank1_range"],
        "castellano_rank1_range_v2": inv_v2["castellano_rank1_range"],
        "invariant_range": f"v1: {inv_v1['description']}; v2: {inv_v2['description']}",
    }

    out_json = root / "data" / "exports" / "threshold_sweep.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Saved: {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

    # -----------------------------------------------------------------------
    # Generate LaTeX table (v2 model, primary)
    # -----------------------------------------------------------------------
    v2_optimal = 0.185024
    latex = generate_latex_table(sweep_v2, v2_optimal, float(probs_v2.max()))

    out_tex = root / "paper" / "tables" / "table_13_threshold_sweep.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"  Saved: {out_tex}")

    print("\nDone.")


if __name__ == "__main__":
    main()
