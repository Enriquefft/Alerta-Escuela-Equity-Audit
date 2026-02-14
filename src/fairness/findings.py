"""Findings distillation: assembles bilingual media-ready findings from exports.

Loads existing JSON exports from ``data/exports/``, extracts runtime statistics
via metric_source paths, and writes ``data/exports/findings.json`` with 7
equity-focused findings ordered by narrative arc.

Usage::

    uv run python src/fairness/findings.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import find_project_root

ROOT = find_project_root()
EXPORTS_DIR = ROOT / "data" / "exports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_export(filename: str) -> dict:
    """Load a JSON export file from data/exports/."""
    path = EXPORTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Export not found: {path}")
    with open(path) as f:
        return json.load(f)


def _resolve_path(data: dict, dot_path: str):
    """Navigate a dot-separated path in a nested dict.

    Returns the value at the path. Raises KeyError if any segment is missing.
    """
    parts = dot_path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(
                f"Path segment '{part}' not found at "
                f"'{'.'.join(parts[:parts.index(part)])}'. "
                f"Available keys: {list(current.keys()) if isinstance(current, dict) else 'N/A'}"
            )
        current = current[part]
    return current


def _format_pct(value: float) -> str:
    """Format a proportion as a percentage string (e.g. 0.632 -> '63.3%')."""
    return f"{value * 100:.1f}%"


def _format_ratio(value: float, total: int = 10) -> str:
    """Format a proportion as 'X de cada Y' ratio."""
    numerator = round(value * total)
    return f"{numerator} de cada {total}"


# ---------------------------------------------------------------------------
# Finding definitions
# ---------------------------------------------------------------------------


def _build_findings(exports: dict[str, dict]) -> list[dict]:
    """Build the 7 findings with runtime-extracted stats."""
    fm = exports["fairness_metrics.json"]
    sv = exports["shap_values.json"]
    ch = exports["choropleth.json"]

    # Extract runtime values
    castellano_fnr = _resolve_path(fm, "dimensions.language.groups.castellano.fnr")
    indigenous_fpr = _resolve_path(fm, "dimensions.language.groups.other_indigenous.fpr")
    indigenous_fnr = _resolve_path(fm, "dimensions.language.groups.other_indigenous.fnr")
    castellano_fpr = _resolve_path(fm, "dimensions.language.groups.castellano.fpr")
    urban_indig_fnr = _resolve_path(
        fm, "intersections.language_x_rural.groups.other_indigenous_urban.fnr"
    )
    urban_indig_n = _resolve_path(
        fm, "intersections.language_x_rural.groups.other_indigenous_urban.n_unweighted"
    )
    top_5_shap = _resolve_path(sv, "top_5_shap")
    selva_fnr = _resolve_path(fm, "dimensions.region.groups.selva.fnr")
    pearson_r = _resolve_path(ch, "correlation.pearson_r")
    sex_fnr_gap = _resolve_path(fm, "dimensions.sex.gaps.max_fnr_gap")

    # Derived values for headlines
    castellano_fnr_ratio = round(castellano_fnr * 10)
    urban_indig_fnr_ratio = round(urban_indig_fnr * 4)  # 3 of 4

    findings = [
        # --- Finding 1: FNR overall ---
        {
            "id": "fnr_overall",
            "stat": f"FNR = {_format_pct(castellano_fnr)} for Spanish-speaking students",
            "headline_es": (
                f"{castellano_fnr_ratio} de cada 10 estudiantes en riesgo "
                f"no son detectados por Alerta Escuela"
            ),
            "headline_en": (
                f"{castellano_fnr_ratio} in 10 at-risk students are missed by "
                f"Peru's Alerta Escuela early warning system"
            ),
            "explanation_es": (
                f"El sistema de alerta temprana falla en detectar al "
                f"{_format_pct(castellano_fnr)} de los estudiantes hispanohablantes "
                f"que efectivamente abandonan la escuela. Esta tasa de falsos "
                f"negativos significa que la mayoria de los estudiantes en riesgo "
                f"nunca reciben intervencion."
            ),
            "explanation_en": (
                f"The early warning system fails to flag "
                f"{_format_pct(castellano_fnr)} of Spanish-speaking students who "
                f"actually drop out. This false negative rate means the majority "
                f"of at-risk students never receive intervention."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.language.groups.castellano.fnr",
                "label": "FNR for Castellano (Spanish) language group",
            },
            "visualization_type": "bar_chart",
            "data_key": "dimensions.language",
            "severity": "critical",
        },
        # --- Finding 2: Surveillance bias ---
        {
            "id": "surveillance_bias",
            "stat": (
                f"FPR = {_format_pct(indigenous_fpr)} (indigenous) vs "
                f"{_format_pct(castellano_fpr)} (Spanish); "
                f"FNR = {_format_pct(indigenous_fnr)} vs {_format_pct(castellano_fnr)}"
            ),
            "headline_es": (
                f"Sesgo de vigilancia: estudiantes indigenas son "
                f"sobremarcados ({_format_pct(indigenous_fpr)} FPR) mientras "
                f"hispanohablantes son invisibilizados ({_format_pct(castellano_fnr)} FNR)"
            ),
            "headline_en": (
                f"Surveillance bias: indigenous students are over-flagged "
                f"({_format_pct(indigenous_fpr)} FPR) while Spanish-speakers are "
                f"invisible ({_format_pct(castellano_fnr)} FNR)"
            ),
            "explanation_es": (
                f"El modelo detecta mejor a estudiantes de lenguas indigenas "
                f"(FNR={_format_pct(indigenous_fnr)}) pero los sobremarca como "
                f"en riesgo cuando no lo estan (FPR={_format_pct(indigenous_fpr)}). "
                f"Para hispanohablantes ocurre lo inverso: pasan desapercibidos "
                f"con un FNR del {_format_pct(castellano_fnr)}."
            ),
            "explanation_en": (
                f"The model catches more indigenous-language students "
                f"(FNR={_format_pct(indigenous_fnr)}) but over-flags them when they "
                f"are not at risk (FPR={_format_pct(indigenous_fpr)}). For Spanish-"
                f"speakers, the pattern reverses: they go undetected at a "
                f"{_format_pct(castellano_fnr)} false negative rate."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.language.groups.other_indigenous.fpr",
                "label": "FPR for Other Indigenous language group",
            },
            "visualization_type": "grouped_bar",
            "data_key": "dimensions.language",
            "severity": "critical",
        },
        # --- Finding 3: Urban indigenous invisible ---
        {
            "id": "urban_indigenous_invisible",
            "stat": (
                f"FNR = {_format_pct(urban_indig_fnr)} for urban indigenous "
                f"students (n={urban_indig_n})"
            ),
            "headline_es": (
                f"{urban_indig_fnr_ratio} de cada 4 estudiantes indigenas "
                f"urbanos en riesgo no son detectados por el sistema"
            ),
            "headline_en": (
                f"{urban_indig_fnr_ratio} in 4 urban indigenous students at risk "
                f"are completely missed by the early warning system"
            ),
            "explanation_es": (
                f"Los estudiantes indigenas en zonas urbanas son los mas "
                f"invisibles para el sistema, con un FNR del "
                f"{_format_pct(urban_indig_fnr)}. Aunque son una muestra pequena "
                f"(n={urban_indig_n}), el patron sugiere que el sistema asocia "
                f"riesgo con ruralidad, no con vulnerabilidad linguistica urbana."
            ),
            "explanation_en": (
                f"Urban indigenous students are the most invisible to the system, "
                f"with a {_format_pct(urban_indig_fnr)} false negative rate. "
                f"While the sample is small (n={urban_indig_n}), the pattern "
                f"suggests the model associates risk with rurality, missing "
                f"linguistic vulnerability in cities."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#intersections.language_x_rural.groups.other_indigenous_urban.fnr",
                "label": "FNR for urban indigenous-language students",
            },
            "visualization_type": "heatmap",
            "data_key": "intersections.language_x_rural",
            "severity": "high",
        },
        # --- Finding 4: Model sees poverty not identity ---
        {
            "id": "model_sees_poverty",
            "stat": (
                f"Top-5 SHAP features: {', '.join(top_5_shap)}; "
                f"0/5 overlap with logistic regression identity features"
            ),
            "headline_es": (
                f"El modelo predice desercion a traves de la pobreza y la "
                f"geografia, no de la identidad del estudiante"
            ),
            "headline_en": (
                f"The model predicts dropout through poverty and geography, "
                f"not student identity"
            ),
            "explanation_es": (
                f"Los 5 factores mas influyentes segun SHAP son estructurales: "
                f"edad, luminosidad nocturna (proxy de urbanizacion), situacion "
                f"laboral, porcentaje de lengua indigena del distrito y "
                f"alfabetizacion censal. Ninguno coincide con las variables "
                f"identitarias que dominan el modelo lineal."
            ),
            "explanation_en": (
                f"The 5 most influential factors according to SHAP are "
                f"structural: age, nightlight intensity (urbanization proxy), "
                f"employment status, district indigenous language prevalence, and "
                f"census literacy rate. None overlap with the identity features "
                f"that dominate the linear model."
            ),
            "metric_source": {
                "path": "shap_values.json#top_5_shap",
                "label": "Top 5 SHAP features (LightGBM)",
            },
            "visualization_type": "bar_chart",
            "data_key": "global_importance",
            "severity": "medium",
        },
        # --- Finding 5: Selva FNR crisis ---
        {
            "id": "selva_fnr_crisis",
            "stat": f"FNR = {_format_pct(selva_fnr)} in the Selva region",
            "headline_es": (
                f"Alerta Escuela no detecta a {round(selva_fnr * 10)} de cada "
                f"10 estudiantes en riesgo en la region Selva"
            ),
            "headline_en": (
                f"Alerta Escuela misses {round(selva_fnr * 10)} in 10 at-risk "
                f"students in the Selva region (Amazon basin)"
            ),
            "explanation_es": (
                f"En la Selva, el sistema falla en detectar al "
                f"{_format_pct(selva_fnr)} de los estudiantes que abandonan la "
                f"escuela. La region amazonica presenta las condiciones mas "
                f"dificiles para la prediccion: alta dispersion geografica, "
                f"menor conectividad y poblaciones indigenas diversas."
            ),
            "explanation_en": (
                f"In the Selva (Amazon basin), the system fails to flag "
                f"{_format_pct(selva_fnr)} of students who drop out. The Amazon "
                f"region presents the hardest prediction conditions: high "
                f"geographic dispersion, lower connectivity, and diverse "
                f"indigenous populations."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.region.groups.selva.fnr",
                "label": "FNR for Selva (Amazon) region",
            },
            "visualization_type": "choropleth",
            "data_key": "dimensions.region",
            "severity": "high",
        },
        # --- Finding 6: District mismatch ---
        {
            "id": "district_mismatch",
            "stat": f"Pearson r = {pearson_r:.3f} between predictions and admin rates",
            "headline_es": (
                f"Las predicciones distritales no coinciden con las tasas "
                f"oficiales de desercion (r = {pearson_r:.3f})"
            ),
            "headline_en": (
                f"District-level predictions show near-zero correlation with "
                f"official dropout rates (r = {pearson_r:.3f})"
            ),
            "explanation_es": (
                f"La correlacion de Pearson entre las predicciones agregadas "
                f"del modelo y las tasas de desercion del MINEDU es practicamente "
                f"nula (r = {pearson_r:.3f}). Esto sugiere que el modelo de "
                f"alerta individual no se traduce en patrones distritales "
                f"coherentes con las estadisticas administrativas."
            ),
            "explanation_en": (
                f"The Pearson correlation between aggregated model predictions "
                f"and Peru's Ministry of Education (MINEDU) dropout rates is "
                f"near zero (r = {pearson_r:.3f}). This suggests the individual "
                f"alert model does not translate into district-level patterns "
                f"consistent with administrative statistics."
            ),
            "metric_source": {
                "path": "choropleth.json#correlation.pearson_r",
                "label": "Pearson r between model predictions and admin dropout rates",
            },
            "visualization_type": "choropleth",
            "data_key": "correlation",
            "severity": "medium",
        },
        # --- Finding 7: Sex equity minimal ---
        {
            "id": "sex_equity_minimal",
            "stat": f"FNR gap = {_format_pct(sex_fnr_gap)} between sexes",
            "headline_es": (
                f"La brecha de equidad por sexo es minima: solo "
                f"{_format_pct(sex_fnr_gap)} de diferencia en FNR"
            ),
            "headline_en": (
                f"The gender equity gap is minimal: only "
                f"{_format_pct(sex_fnr_gap)} FNR difference between boys and girls"
            ),
            "explanation_es": (
                f"A diferencia de la lengua y la geografia, el modelo trata "
                f"de manera casi equitativa a ninos y ninas, con una brecha "
                f"de FNR de solo {_format_pct(sex_fnr_gap)}. Este hallazgo "
                f"positivo contrasta con las disparidades criticas encontradas "
                f"en otras dimensiones."
            ),
            "explanation_en": (
                f"Unlike language and geography, the model treats boys and "
                f"girls nearly equally, with an FNR gap of just "
                f"{_format_pct(sex_fnr_gap)}. This positive finding contrasts "
                f"sharply with the critical disparities found across linguistic "
                f"and regional dimensions."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.sex.gaps.max_fnr_gap",
                "label": "Maximum FNR gap between sex groups",
            },
            "visualization_type": "bar_chart",
            "data_key": "dimensions.sex",
            "severity": "low",
        },
    ]

    return findings


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_metric_sources(findings: list[dict], exports: dict[str, dict]) -> None:
    """Validate that every metric_source path resolves to a non-null value."""
    for finding in findings:
        path_str = finding["metric_source"]["path"]
        filename, json_path = path_str.split("#", 1)

        if filename not in exports:
            raise ValueError(
                f"Finding '{finding['id']}': export file '{filename}' not available"
            )

        value = _resolve_path(exports[filename], json_path)

        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(
                    f"Finding '{finding['id']}': path '{path_str}' resolved to empty list"
                )
        elif value is None:
            raise ValueError(
                f"Finding '{finding['id']}': path '{path_str}' resolved to null"
            )

        print(f"  VALIDATED: {finding['id']} -> {path_str} = {repr(value)[:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load exports, build findings, validate, and write findings.json."""
    print("Loading exports...")
    export_files = [
        "fairness_metrics.json",
        "shap_values.json",
        "choropleth.json",
        "model_results.json",
        "descriptive_tables.json",
    ]
    exports = {}
    for fname in export_files:
        exports[fname] = _load_export(fname)
        print(f"  Loaded {fname}")

    print("\nBuilding findings...")
    findings = _build_findings(exports)
    print(f"  Built {len(findings)} findings")

    print("\nValidating metric_source paths...")
    _validate_metric_sources(findings, exports)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_findings": len(findings),
        "findings": findings,
    }

    output_path = EXPORTS_DIR / "findings.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"Findings: {len(findings)}")
    for finding in findings:
        print(f"  [{finding['severity']}] {finding['id']}: {finding['headline_en'][:70]}")


if __name__ == "__main__":
    main()
